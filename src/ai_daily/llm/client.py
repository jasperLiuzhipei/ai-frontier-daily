"""OpenAI-compatible LLM client with automatic fallback and JSON extraction."""

from __future__ import annotations

import json
import re
import time
from typing import Any, Callable

from ai_daily.http_client import HTTPClient, HTTPErrorDetail
from ai_daily.models import LLMResponse

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def extract_json_object(text: str) -> dict[str, Any]:
    """Robustly parse a JSON object from raw LLM output (with or without markdown fences)."""
    raw = text.strip()
    if not raw:
        raise ValueError("Empty LLM response")

    # 1. Direct parse
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # 2. Markdown code fence
    fence_match = _JSON_FENCE_RE.search(raw)
    if fence_match:
        try:
            obj = json.loads(fence_match.group(1))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    # 3. Balanced outer braces extraction
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = raw[start : end + 1]
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj

    raise ValueError(f"Could not extract valid JSON object from LLM output: {raw[:240]}")


class OpenAICompatClient:
    """Calls any OpenAI-compatible Chat Completions endpoint with model fallback & auto-discovery."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        primary_model: str,
        fallback_models: list[str] | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        extra_body: dict[str, Any] | None = None,
        timeout_seconds: int = 90,
        max_retries: int = 2,
        auto_discover_on_error: bool = True,
        http: HTTPClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.primary_model = primary_model
        self.fallback_models = list(fallback_models or [])
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_body = dict(extra_body or {})
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.auto_discover_on_error = auto_discover_on_error
        self.http = http or HTTPClient(timeout=timeout_seconds, max_retries=max_retries)

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.base_url and self.primary_model)

    def _auth_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json; charset=utf-8",
        }

    def list_models(self) -> list[str]:
        if not self.api_key:
            raise RuntimeError("Missing LLM API key (set DEEPSEEK_API_KEY or LLM_API_KEY)")
        url = f"{self.base_url}/models"
        resp = self.http.get(url, headers=self._auth_headers(), timeout=30)
        data = resp.json()
        models: list[str] = []
        items = data.get("data") if isinstance(data, dict) else data
        if isinstance(items, list):
            for entry in items:
                if isinstance(entry, dict) and entry.get("id"):
                    models.append(str(entry["id"]))
                elif isinstance(entry, str):
                    models.append(entry)
        return models

    def _single_completion(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        extra_body_override: dict[str, Any] | None = None,
    ) -> LLMResponse:
        url = f"{self.base_url}/chat/completions"
        eff_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
        }
        if eff_max_tokens is not None:
            payload["max_tokens"] = eff_max_tokens
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        extra = extra_body_override if extra_body_override is not None else self.extra_body
        if extra:
            payload.update(extra)

        t0 = time.monotonic()
        try:
            resp = self.http.post_json(
                url,
                payload,
                headers=self._auth_headers(),
                timeout=self.timeout_seconds,
                max_retries=self.max_retries,
            )
        except HTTPErrorDetail as exc:
            # Downgrade optional vendor fields (response_format / extra_body) on 400
            if exc.status == 400 and (json_mode or extra):
                retry_payload = {
                    "model": model,
                    "messages": messages,
                    "temperature": self.temperature if temperature is None else temperature,
                }
                if eff_max_tokens is not None:
                    retry_payload["max_tokens"] = eff_max_tokens
                resp = self.http.post_json(
                    url,
                    retry_payload,
                    headers=self._auth_headers(),
                    timeout=self.timeout_seconds,
                    max_retries=self.max_retries,
                )
            else:
                raise

        latency_ms = int((time.monotonic() - t0) * 1000)
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"LLM returned empty choices for model '{model}': {data}")
        message = choices[0].get("message") or {}
        content = message.get("content") or ""
        reasoning = message.get("reasoning_content") or ""
        # DeepSeek V4 thinking models can exhaust max_tokens in reasoning_content leaving content empty;
        # automatically retry once with thinking disabled so structured JSON always completes.
        if not content.strip() and reasoning and extra_body_override is None:
            return self._single_completion(
                model,
                messages,
                temperature=temperature,
                max_tokens=max(eff_max_tokens or 8192, 8192),
                json_mode=json_mode,
                extra_body_override={"thinking": {"type": "disabled"}},
            )
        usage = data.get("usage") or {}
        return LLMResponse(
            content=content,
            model=str(data.get("model") or model),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            latency_ms=latency_ms,
            raw=data,
        )

    def _iter_candidates(self, model: str | None) -> list[str]:
        candidates: list[str] = []
        if model:
            candidates.append(model)
        else:
            candidates.append(self.primary_model)
            for fb in self.fallback_models:
                if fb and fb not in candidates:
                    candidates.append(fb)
        return candidates

    def _run_with_fallback(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        parse_fn: Callable[[LLMResponse], Any] | None = None,
    ) -> tuple[Any, LLMResponse]:
        if not self.api_key:
            raise RuntimeError("Missing LLM API key (set DEEPSEEK_API_KEY or LLM_API_KEY in .env)")

        candidates = self._iter_candidates(model)
        errors: list[str] = []
        saw_model_missing = False

        def _attempt(candidate_model: str) -> tuple[Any, LLMResponse]:
            resp = self._single_completion(
                candidate_model,
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=json_mode,
            )
            parsed = parse_fn(resp) if parse_fn is not None else None
            return parsed, resp

        for candidate in candidates:
            try:
                return _attempt(candidate)
            except HTTPErrorDetail as exc:
                errors.append(f"{candidate}: HTTP {exc.status} ({exc.body[:200]})")
                if exc.status in (400, 404) and (
                    "model" in exc.body.lower() or "not found" in exc.body.lower()
                ):
                    saw_model_missing = True
                # Fatal auth error: no point retrying other model names
                if exc.status in (401, 403):
                    raise RuntimeError(
                        f"LLM authentication failed (HTTP {exc.status}). Check your API key and base_url."
                    ) from exc
            except Exception as exc:
                errors.append(f"{candidate}: {exc}")

        # Auto-discover from /models if configured models rotated/deprecated
        if self.auto_discover_on_error and saw_model_missing and not model:
            try:
                remote = self.list_models()
                for discovered in remote:
                    if discovered in candidates:
                        continue
                    low = discovered.lower()
                    if any(bad in low for bad in ("embed", "rerank", "tts", "whisper")):
                        continue
                    if "deepseek" in low or len(remote) <= 5:
                        try:
                            return _attempt(discovered)
                        except Exception as exc:
                            errors.append(f"auto-discovered {discovered}: {exc}")
            except Exception as exc:
                errors.append(f"/models discovery failed: {exc}")

        raise RuntimeError("All LLM model attempts failed:\n  - " + "\n  - ".join(errors))

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        _, resp = self._run_with_fallback(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
        )
        return resp

    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        validator: Callable[[dict[str, Any]], bool] | None = None,
    ) -> tuple[dict[str, Any], LLMResponse]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        def _parse_and_validate(resp: LLMResponse) -> dict[str, Any]:
            obj = extract_json_object(resp.content)
            if validator is not None and not validator(obj):
                raise ValueError(
                    f"JSON object failed schema validation for model '{resp.model}': keys={sorted(obj.keys())}"
                )
            return obj

        parsed, resp = self._run_with_fallback(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
            parse_fn=_parse_and_validate,
        )
        return parsed, resp
