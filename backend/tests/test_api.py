import json

import pytest
from fastapi.testclient import TestClient

from cascadia.api.main import app
from cascadia.config import settings


@pytest.fixture(scope="module")
def client():
    # Point at unreachable services: the API must degrade gracefully.
    settings.redis_url = "redis://127.0.0.1:1/0"
    settings.database_url = "postgresql+asyncpg://x:x@127.0.0.1:1/x"
    with TestClient(app, base_url="http://localhost") as c:
        yield c


def test_health(client):
    body = client.get("/api/health").json()
    assert body["ok"] and not body["live"] and not body["redis"]


def test_grid(client):
    g = client.get("/api/grid").json()
    assert len(g["buses"]) == 118 and len(g["branches"]) == 186


def test_state_falls_back_to_nominal(client):
    body = client.get("/api/state").json()
    assert body["live"] is False
    assert body["risk"]["status"] in {"secure", "alert", "emergency"}


def test_whatif(client):
    r = client.post("/api/whatif", json={"lines": [7, 8], "load_scale": 1.25}).json()
    assert r["lines"] == [7, 8]
    assert len(r["gnn"]["prob"]) == len(r["monte_carlo"]["prob"]) == 186
    assert r["monte_carlo"]["runs"] == 100
    assert r["sample"]["stages"][0]["tripped"] == [7, 8]
    assert 0 <= r["agreement"]["likely_jaccard"] <= 1


def test_whatif_validation(client):
    assert client.post("/api/whatif", json={"lines": [999]}).status_code == 422
    assert client.post("/api/whatif", json={"lines": []}).status_code == 422


def test_commands_need_live_sim(client):
    assert client.post("/api/commands", json={"cmd": "restore_all"}).status_code == 503


def test_copilot_disabled_without_key(client, monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", None)
    r = client.post("/api/copilot", json={"messages": [{"role": "user", "content": "hi"}]})
    events = [json.loads(line[6:]) for line in r.text.splitlines() if line.startswith("data: ")]
    assert events[0]["type"] == "error"


def test_rejects_out_of_range_command_lines(client):
    """Unvalidated ids used to crash the simulator process (IndexError in GridTwin.trip)."""
    for bad in ([999], [-1]):
        assert client.post("/api/commands", json={"cmd": "trip", "lines": bad}).status_code == 422


def test_rejects_negative_limits(client):
    assert client.get("/api/incidents?limit=-1").status_code == 422
    assert client.get("/api/timeline?limit=0").status_code == 422


def test_no_wildcard_cors(client):
    r = client.options("/api/commands", headers={"Origin": "https://evil.example",
                                                 "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in r.headers


def test_simple_cross_site_posts_are_not_parsed(client):
    """CSRF: browsers send text/plain and form posts without a preflight; they must not act."""
    for ctype in ("text/plain", "application/x-www-form-urlencoded"):
        r = client.post("/api/commands", content='{"cmd": "restore_all"}',
                        headers={"Content-Type": ctype})
        assert r.status_code == 422


def test_rejects_foreign_host_header(client):
    """DNS rebinding: a page on evil.example resolving to 127.0.0.1 must not reach the API."""
    assert client.get("/api/state", headers={"Host": "evil.example"}).status_code == 400
