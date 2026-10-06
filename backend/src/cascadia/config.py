from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CASCADIA_", env_file=".env", extra="ignore")

    artifacts_dir: Path = BACKEND_DIR / "artifacts"
    data_dir: Path = BACKEND_DIR / "data"
    redis_url: str = "redis://localhost:6379/0"
    database_url: str = "postgresql+asyncpg://cascadia:cascadia@localhost:5432/cascadia"
    # Host headers the API answers to (blocks DNS-rebinding attacks from other web pages).
    # For a LAN demo add the machine's IP: CASCADIA_ALLOWED_HOSTS='["localhost","192.168.1.20"]'
    allowed_hosts: list[str] = ["localhost", "127.0.0.1"]

    # Simulator: each tick advances `sim_minutes_per_tick` simulated minutes,
    # once every `tick_seconds` real seconds.
    tick_seconds: float = 1.0
    sim_minutes_per_tick: float = 2.0
    fault_rate_per_hour: float = 0.6  # random line faults per simulated hour (x5 during storms)

    # Risk engine
    verify_every: int = 10  # cross-check GNN against physics every N ticks
    risk_group: str = "risk-engine"

    # Copilot (optional)
    anthropic_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("ANTHROPIC_API_KEY", "CASCADIA_ANTHROPIC_API_KEY"),
    )
    copilot_model: str = "claude-opus-5-5"


settings = Settings()

# Redis stream names
STATE_STREAM = "grid.state"
RISK_STREAM = "grid.risk"
EVENT_STREAM = "grid.events"
COMMAND_STREAM = "grid.commands"
