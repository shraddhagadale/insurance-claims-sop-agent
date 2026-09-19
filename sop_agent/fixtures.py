"""Read-only access to the sample back-office data (the 'systems of record')."""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .config import settings


@dataclass(frozen=True)
class Fixtures:
    policyholders: list[dict[str, Any]]
    claims: list[dict[str, Any]]
    claim_schema: dict[str, Any]
    doc_guidelines: dict[str, Any]
    representatives: list[dict[str, Any]]
    consent_scenarios: dict[str, Any]

    def claims_for(self, party_id: str) -> list[dict[str, Any]]:
        return [c for c in self.claims if c["party_id"] == party_id]

    def claim(self, case_id: str) -> dict[str, Any] | None:
        return next((c for c in self.claims if c["case_id"].upper() == case_id.upper()), None)

    def policyholder(self, party_id: str) -> dict[str, Any] | None:
        return next((p for p in self.policyholders if p["party_id"] == party_id), None)


def _load(path: Path) -> Any:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=4)
def load_fixtures(directory: str | None = None) -> Fixtures:
    root = Path(directory) if directory else settings.fixtures_dir
    return Fixtures(
        policyholders=_load(root / "policyholders.json"),
        claims=_load(root / "claims.json"),
        claim_schema=_load(root / "claim_schema.json"),
        doc_guidelines=_load(root / "required_document_guideline.json"),
        representatives=_load(root / "representatives.json"),
        consent_scenarios=_load(root / "consent_scenarios.json"),
    )
