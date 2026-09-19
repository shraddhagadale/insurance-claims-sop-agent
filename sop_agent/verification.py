"""Deterministic identity verification.

The LLM extracts candidate values from free text; this module - not the model - decides
whether they match a record. The result never leaks *which* field mismatched to the caller.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .fixtures import Fixtures
from .state import IDENTITY_FIELDS

_DATE_FORMATS = (
    "%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%m-%d-%Y", "%m/%d/%y", "%B %d, %Y", "%B %d %Y",
    "%b %d, %Y", "%b %d %Y", "%d %B %Y", "%d %b %Y", "%Y%m%d", "%m.%d.%Y",
)


def normalize_name(value: str | None) -> str:
    if not value:
        return ""
    value = re.sub(r"[^a-z\s]", " ", value.lower())
    return " ".join(value.split())


def normalize_phone(value: str | None) -> str:
    digits = re.sub(r"\D", "", value or "")
    return digits[-10:] if len(digits) >= 10 else digits


def normalize_email(value: str | None) -> str:
    return (value or "").strip().lower()


def normalize_last4(value: str | None) -> str:
    digits = re.sub(r"\D", "", value or "")
    return digits[-4:] if len(digits) >= 4 else ""


def normalize_dob(value: str | None) -> str:
    if not value:
        return ""
    raw = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", value.strip(), flags=re.I)
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return ""


def normalize_policy(value: str | None) -> str:
    return re.sub(r"[^A-Z0-9]", "", (value or "").upper())


def _name_matches(given: str, record: dict[str, Any]) -> bool:
    g = normalize_name(given)
    if not g:
        return False
    names = [record.get("name", "")] + list(record.get("name_aliases", []))
    for n in names:
        rn = normalize_name(n)
        # exact, or same tokens in a different order ("Chen Margaret")
        if g == rn or sorted(g.split()) == sorted(rn.split()):
            return True
    return False


def _field_matches(field_name: str, given: str, record: dict[str, Any]) -> bool:
    if field_name == "full_name":
        return _name_matches(given, record)
    if field_name == "dob":
        return bool(normalize_dob(given)) and normalize_dob(given) == record.get("dob")
    if field_name == "phone":
        on_file = [record.get("phone")] + list(record.get("phone_aliases", []))
        g = normalize_phone(given)
        return len(g) == 10 and g in {normalize_phone(p) for p in on_file if p}
    if field_name == "email":
        on_file = [record.get("email")] + list(record.get("email_aliases", []))
        return normalize_email(given) in {normalize_email(e) for e in on_file if e}
    if field_name == "id_last4":
        g = normalize_last4(given)
        return bool(g) and g == record.get("id_last4")
    return False


@dataclass
class VerificationResult:
    verified: bool
    party_id: str | None
    matched: list[str] = field(default_factory=list)
    mismatched: list[str] = field(default_factory=list)
    provided: list[str] = field(default_factory=list)
    needed: int = 0

    @property
    def remaining(self) -> int:
        return max(0, self.needed - len(self.matched))


def verify_identity(claims: dict[str, str], fixtures: Fixtures, required: int = 3) -> VerificationResult:
    """claims: identity values the caller has asserted so far (from cross-phase memory)."""
    provided = [f for f in IDENTITY_FIELDS if claims.get(f)]
    policy = normalize_policy(claims.get("policy_number"))

    best: tuple[int, int, dict[str, Any] | None, list[str], list[str]] = (-1, 0, None, [], [])
    for record in fixtures.policyholders:
        matched = [f for f in provided if _field_matches(f, claims[f], record)]
        mismatched = [f for f in provided if f not in matched]
        policy_hit = 1 if policy and normalize_policy(record.get("policy_number")) == policy else 0
        key = (len(matched), policy_hit)
        if key > best[:2]:
            best = (len(matched), policy_hit, record, matched, mismatched)

    n_matched, _, record, matched, mismatched = best
    if record is None or n_matched <= 0:
        return VerificationResult(False, None, [], list(provided), provided, required)
    return VerificationResult(
        verified=n_matched >= required,
        party_id=record["party_id"],
        matched=matched,
        mismatched=mismatched,
        provided=provided,
        needed=required,
    )


def find_representative(rep_name: str | None, party_id: str, fixtures: Fixtures) -> dict[str, Any] | None:
    if not rep_name:
        return None
    n = normalize_name(rep_name)
    for rep in fixtures.representatives:
        if rep["buyer_party_id"] == party_id and normalize_name(rep["rep_name"]) == n:
            return rep
    return None
