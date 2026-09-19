"""Runtime configuration. Everything is env-driven so the container needs no code edits."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent
DEFAULT_FIXTURES = PROJECT_DIR / "fixtures"

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_DIR / ".env")
except ImportError:
    pass


@dataclass
class Settings:
    # --- AI model auth -----------------------------------------------------
    api_key: str | None = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY") or None)
    model: str = field(default_factory=lambda: os.getenv("SOP_MODEL", "claude-opus-5"))
    # Perception is a cheap structured-extraction call; the speaking model carries the persona.
    perceive_effort: str = field(default_factory=lambda: os.getenv("SOP_PERCEIVE_EFFORT", "low"))
    speak_effort: str = field(default_factory=lambda: os.getenv("SOP_SPEAK_EFFORT", "medium"))
    max_tokens: int = field(default_factory=lambda: int(os.getenv("SOP_MAX_TOKENS", "2000")))

    # --- Harness behaviour -------------------------------------------------
    fixtures_dir: Path = field(
        default_factory=lambda: Path(os.getenv("SOP_FIXTURES_DIR", str(DEFAULT_FIXTURES)))
    )
    # "Today" for date math (deadlines). Defaults to a date consistent with the sample fixtures.
    as_of_date: str = field(default_factory=lambda: os.getenv("SOP_AS_OF_DATE", "2026-03-05"))
    consent_scenario: str = field(default_factory=lambda: os.getenv("SOP_CONSENT_SCENARIO", "default"))

    required_id_matches: int = field(default_factory=lambda: int(os.getenv("SOP_REQUIRED_ID_MATCHES", "3")))
    max_offtopic_strikes: int = field(default_factory=lambda: int(os.getenv("SOP_MAX_OFFTOPIC", "3")))
    max_persuasion_attempts: int = field(default_factory=lambda: int(os.getenv("SOP_MAX_PERSUASION", "3")))

    def has_credentials(self) -> bool:
        return bool(self.api_key)

    def require_credentials(self) -> None:
        if not self.has_credentials():
            raise RuntimeError(
                "ANTHROPIC_API_KEY is required. Create insurance_claims/.env from .env.example "
                "or set ANTHROPIC_API_KEY in the server environment."
            )


settings = Settings()
