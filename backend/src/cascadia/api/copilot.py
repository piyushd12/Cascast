"""Operator Copilot: a Claude agent whose tools call the real simulator and GNN.

Streams Server-Sent Events: {"type": "text" | "tool" | "tool_result" | "error" | "done", ...}.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import anthropic

from cascadia.api import ops
from cascadia.config import settings
from cascadia.grid.model import load_grid

MAX_TURNS = 8

SYSTEM = """You are the Operator Copilot for Cascadia, a real-time digital twin of the IEEE \
118-bus transmission grid (118 buses, 186 branches, DC power-flow physics).

A graph neural network screens every single-branch (N-1) contingency each second and \
predicts the probability that each branch trips in the resulting cascade and the expected \
load shed. You help \
grid operators understand risk and decide what to do.

Ground every claim in tool results; never invent line numbers or values. Branches are \
identified by integer id (0-185). "Shed" is the fraction of total demand that would be lost. \
Protection is stochastic (relay thresholds vary; exposed relays can misoperate), so speak \
in probabilities and expected load shed. When useful, compare the GNN with the Monte Carlo \
physics estimate returned by simulate_contingency. When recommending actions (e.g. demand \
response), quantify the effect with evaluate_load_relief before recommending it.

Be concise and operational: lead with the answer, use short bullet points, cite line ids and \
percentages."""


def _tool(name: str, description: str, props: dict, required: list[str]) -> dict:
    return {"name": name, "description": description, "strict": True,
            "eager_input_streaming": True,
            "input_schema": {"type": "object", "properties": props, "required": required,
                             "additionalProperties": False}}


TOOLS = [
    _tool("get_grid_status",
          "Current operating point and N-1 risk summary: clock, weather, demand, served load, "
          "out-of-service lines, most heavily loaded lines, and the riskiest contingencies.",
          {}, []),
    _tool("simulate_contingency",
          "Trip one or more branches on the current operating point. Protection is "
          "stochastic, so outcomes are probabilities: returns the GNN's predicted failure "
          "probabilities and expected load shed, a 100-run Monte Carlo physics estimate, and "
          "one random realisation of the cascade.",
          {"lines": {"type": "array", "items": {"type": "integer"},
                     "description": "Branch ids to trip (0-185)."},
           "load_scale": {"type": "number",
                          "description": "Optional demand multiplier, e.g. 1.1 for +10%. "
                                         "Use 1.0 for the current demand."}},
          ["lines", "load_scale"]),
    _tool("inspect_line",
          "Details of one branch: endpoints, rating, flow, loading, status, vulnerability.",
          {"line": {"type": "integer", "description": "Branch id (0-185)."}}, ["line"]),
    _tool("evaluate_load_relief",
          "Re-run the full N-1 screen with demand reduced by `percent` (demand response). "
          "Returns worst-case shed and number of critical contingencies before and after.",
          {"percent": {"type": "number", "description": "Demand reduction in percent, 0-30."}},
          ["percent"]),
]


def _status(live: ops.LiveState) -> dict:
    g, s, r = load_grid(), live.state, live.risk
    loading = s["loading"]
    top = sorted(range(g.n_branch), key=lambda e: -loading[e])[:8]
    out = {
        "clock": s["clock"], "weather": s["weather"], "demand_mw": s["demand_mw"],
        "served_mw": s["served_mw"], "islands": s["n_islands"],
        "out_of_service": [e for e, a in enumerate(s["alive"]) if not a],
        "most_loaded": [{"line": e, "loading_pct": round(100 * loading[e], 1)} for e in top],
        "live_feed": live.live,
    }
    risk = r or ops.screen(live)
    out["risk"] = {k: risk[k] for k in ("status", "worst_shed_pct", "n_critical", "n_screened")}
    out["risk"]["top_contingencies"] = risk["top"][:8]
    return out


def _top(prob: list[float], k: int = 8) -> list[dict]:
    order = sorted(range(len(prob)), key=lambda e: -prob[e])[:k]
    return [{"line": e, "prob": prob[e]} for e in order if prob[e] >= 0.05]


def _brief_whatif(res: dict) -> dict:
    """Summarise: top failure probabilities instead of 186-long arrays."""
    gnn, mc, sample = res["gnn"], res["monte_carlo"], res["sample"]
    return {
        "lines": res["lines"], "load_scale": res["load_scale"],
        "gnn_prediction": {"expected_shed_pct": gnn["shed_pct"], "top_risks": _top(gnn["prob"]),
                           "latency_ms": gnn["latency_ms"]},
        "monte_carlo_physics": {k: mc[k] for k in ("runs", "expected_shed_pct", "shed_p90_pct",
                                                   "cascade_prob", "latency_ms")}
        | {"top_risks": _top(mc["prob"])},
        "one_random_realisation": {"failed": sample["failed"], "relay_misoperations":
                                   sample["hidden"], "shed_pct": sample["shed_pct"],
                                   "stages": [s["tripped"] for s in sample["stages"]]},
        "agreement": res["agreement"],
    }


def run_tool(live: ops.LiveState, name: str, args: dict) -> dict:
    n = load_grid().n_branch
    match name:
        case "get_grid_status":
            return _status(live)
        case "simulate_contingency":
            lines = [int(e) for e in args["lines"] if 0 <= int(e) < n]
            if not lines:
                raise ValueError("no valid branch ids (0-185)")
            scale = float(args.get("load_scale") or 1.0)
            return _brief_whatif(ops.whatif(live, lines, min(max(scale, 0.5), 1.6)))
        case "inspect_line":
            line = int(args["line"])
            if not 0 <= line < n:
                raise ValueError("branch id must be 0-185")
            return ops.line_info(live, line)
        case "evaluate_load_relief":
            pct = min(max(float(args["percent"]), 0.0), 30.0)
            before, after = ops.screen(live), ops.screen(live, 1 - pct / 100)
            keep = ("status", "worst_shed_pct", "n_critical")
            return {"percent": pct, "before": {k: before[k] for k in keep},
                    "after": {k: after[k] for k in keep}}
    raise ValueError(f"unknown tool {name}")


def _valid(name: str, args) -> bool:
    if not isinstance(args, dict):
        return False
    schema = next(t for t in TOOLS if t["name"] == name)["input_schema"]
    return all(k in args for k in schema["required"])


async def stream(live: ops.LiveState, history: list[dict]) -> AsyncIterator[dict]:
    if not settings.anthropic_api_key:
        yield {"type": "error", "message": "Copilot is disabled: set ANTHROPIC_API_KEY."}
        return
    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    messages: list = [{"role": m["role"], "content": m["content"]} for m in history]
    try:
        for _ in range(MAX_TURNS):
            async with client.beta.messages.stream(
                model=settings.copilot_model,
                max_tokens=16000,
                system=SYSTEM,
                tools=TOOLS,
                messages=messages,
                thinking={"type": "adaptive"},
                output_config={"effort": "medium"},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            ) as s:
                async for event in s:
                    if event.type == "text":
                        yield {"type": "text", "text": event.text}
                response = await s.get_final_message()

            messages.append({"role": "assistant", "content": response.content})
            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if response.stop_reason == "refusal":
                yield {"type": "error", "message": "The request was declined."}
                break
            if response.stop_reason == "pause_turn":
                continue
            if not tool_uses or response.stop_reason == "max_tokens":
                break

            results = []
            for b in tool_uses:
                yield {"type": "tool", "name": b.name, "input": b.input}
                try:
                    if not _valid(b.name, b.input):
                        raise ValueError(f"invalid input: {json.dumps(b.input)}")
                    out = run_tool(live, b.name, b.input)
                    results.append({"type": "tool_result", "tool_use_id": b.id,
                                    "content": json.dumps(out)})
                    yield {"type": "tool_result", "name": b.name, "ok": True}
                except Exception as e:  # report tool errors to the model, don't crash the turn
                    results.append({"type": "tool_result", "tool_use_id": b.id,
                                    "is_error": True, "content": str(e)})
                    yield {"type": "tool_result", "name": b.name, "ok": False, "error": str(e)}
            messages.append({"role": "user", "content": results})
    except ValueError:
        yield {"type": "error", "message": "Malformed tool call from the model; please retry."}
    except anthropic.RateLimitError:
        yield {"type": "error", "message": "Rate limited by the Claude API; try again shortly."}
    except anthropic.APIStatusError as e:
        yield {"type": "error", "message": f"Claude API error {e.status_code}."}
    except anthropic.APIConnectionError:
        yield {"type": "error", "message": "Could not reach the Claude API."}
    yield {"type": "done"}
