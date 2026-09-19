"""Thin Anthropic client wrapper used by the perceiver and the speaker."""
from __future__ import annotations

import json
import logging
from typing import Any

from .config import Settings

log = logging.getLogger("sop_agent.llm")

REFUSAL_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMUnavailable(Exception):
    pass


class AnthropicLLM:
    def __init__(self, settings: Settings, api_key: str | None = None):
        import anthropic  # imported lazily so deterministic tests do not need SDK setup

        self._anthropic = anthropic
        self.settings = settings
        self.client = anthropic.Anthropic(api_key=api_key or settings.api_key, max_retries=2, timeout=90.0)
        self.model = settings.model
        self._use_fallbacks = True

    # effort is not accepted on Haiku 4.5
    def _output_config(self, effort: str, fmt: dict[str, Any] | None = None) -> dict[str, Any]:
        cfg: dict[str, Any] = {}
        if "haiku" not in self.model:
            cfg["effort"] = effort
        if fmt:
            cfg["format"] = fmt
        return cfg

    def _create(self, **kwargs: Any):
        extra: dict[str, Any] = {}
        if self._use_fallbacks and self.model.startswith(("claude-opus-5", "claude-fable-5")):
            extra = {"extra_headers": {"anthropic-beta": REFUSAL_FALLBACK_BETA},
                     "extra_body": {"fallbacks": "default"}}
        try:
            return self.client.messages.create(**kwargs, **extra)
        except self._anthropic.BadRequestError as e:
            if extra and "fallback" in str(e).lower():
                log.warning("server-side fallbacks not available for this key; continuing without")
                self._use_fallbacks = False
                return self.client.messages.create(**kwargs)
            raise

    def _text(self, resp) -> str:
        if resp.stop_reason == "refusal":
            raise LLMUnavailable("model declined")
        return "".join(b.text for b in resp.content if b.type == "text").strip()

    def structured(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        try:
            resp = self._create(
                model=self.model,
                max_tokens=self.settings.max_tokens,
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": user}],
                output_config=self._output_config(
                    self.settings.perceive_effort, {"type": "json_schema", "schema": schema}),
            )
            return json.loads(self._text(resp))
        except (self._anthropic.APIError, json.JSONDecodeError) as e:
            raise LLMUnavailable(str(e)) from e

    def text(self, system: str, user: str) -> str:
        try:
            resp = self._create(
                model=self.model,
                max_tokens=self.settings.max_tokens,
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": user}],
                output_config=self._output_config(self.settings.speak_effort),
            )
            out = self._text(resp)
            if not out:
                raise LLMUnavailable("empty response")
            return out
        except self._anthropic.APIError as e:
            raise LLMUnavailable(str(e)) from e
