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
    with TestClient(app) as c:
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
