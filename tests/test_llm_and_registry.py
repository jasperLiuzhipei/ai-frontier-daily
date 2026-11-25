"""Tests for OpenAI-compatible LLM client, fallback logic, and self-iterating model registry."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai_daily.config import ConfigManager
from ai_daily.http_client import HTTPErrorDetail, HTTPResponse
from ai_daily.llm.client import OpenAICompatClient, extract_json_object
from ai_daily.llm.registry import ModelRegistry, filter_and_rank_models, parse_model_version
from ai_daily.storage import StateStore


class MockLLMHTTP:
    def __init__(self, models: list[str], failing_models: set[str] | None = None) -> None:
        self.models = models
        self.failing_models = failing_models or set()
        self.calls: list[tuple[str, str]] = []

    def get(self, url: str, **kwargs) -> HTTPResponse:
        self.calls.append(("GET", url))
        if url.endswith("/models"):
            payload = {"data": [{"id": m} for m in self.models]}
            return HTTPResponse(
                status=200,
                headers={"content-type": "application/json"},
                body=json.dumps(payload).encode("utf-8"),
                url=url,
            )
        raise AssertionError(f"Unexpected GET: {url}")

    def post_json(self, url: str, payload: dict, **kwargs) -> HTTPResponse:
        model = payload["model"]
        self.calls.append(("POST", model))
        if model in self.failing_models:
            raise HTTPErrorDetail(404, "Not Found", f"model '{model}' not found")
        resp_body = {
            "model": model,
            "choices": [{"message": {"content": '{"status": "ok"}'}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 6},
        }
        return HTTPResponse(
            status=200,
            headers={"content-type": "application/json"},
            body=json.dumps(resp_body).encode("utf-8"),
            url=url,
        )


class TestLLMAndRegistry(unittest.TestCase):
    def test_extract_json_variants(self) -> None:
        self.assertEqual(extract_json_object('{"a": 1}'), {"a": 1})
        self.assertEqual(
            extract_json_object('Here you go:\n```json\n{"a": 2}\n```'),
            {"a": 2},
        )
        self.assertEqual(extract_json_object('prefix {"a": 3} suffix'), {"a": 3})

    def test_parse_and_rank_model_versions(self) -> None:
        models = [
            "deepseek-chat",
            "deepseek-v4.1-flash1017",
            "deepseek-v4.1-flash1118",
            "deepseek-v4.2-flash1101",
            "deepseek-embedding-v2",
        ]
        ranked = filter_and_rank_models(
            models,
            family_patterns=[r"deepseek.*flash", r"deepseek-chat"],
            exclude_patterns=[r"embed"],
        )
        self.assertEqual(
            ranked,
            [
                "deepseek-v4.2-flash1101",
                "deepseek-v4.1-flash1118",
                "deepseek-v4.1-flash1017",
                "deepseek-chat",
            ],
        )
        self.assertGreater(
            parse_model_version("deepseek-v4.2-flash1101"),
            parse_model_version("deepseek-v4.1-flash1017"),
        )

    def test_client_fallback_and_autodiscover(self) -> None:
        mock_http = MockLLMHTTP(
            models=["deepseek-v4.2-flash1120", "deepseek-chat"],
            failing_models={"deepseek-v4.1-flash1017", "deepseek-chat"},
        )
        client = OpenAICompatClient(
            base_url="https://api.deepseek.com/v1",
            api_key="sk-test-key",
            primary_model="deepseek-v4.1-flash1017",
            fallback_models=["deepseek-chat"],
            auto_discover_on_error=True,
            http=mock_http,
        )
        data, resp = client.chat_json("sys", "user")
        self.assertEqual(data, {"status": "ok"})
        self.assertEqual(resp.model, "deepseek-v4.2-flash1120")

    def test_thinking_exhaustion_auto_retries_non_thinking(self) -> None:
        class ThinkingHTTP:
            def __init__(self) -> None:
                self.payloads: list[dict] = []

            def post_json(self, url: str, payload: dict, **kwargs) -> HTTPResponse:
                self.payloads.append(dict(payload))
                if len(self.payloads) == 1:
                    body = {
                        "model": payload["model"],
                        "choices": [{"message": {"content": "", "reasoning_content": "thinking..."}}],
                        "usage": {"prompt_tokens": 100, "completion_tokens": 4096},
                    }
                else:
                    body = {
                        "model": payload["model"],
                        "choices": [{"message": {"content": '{"ok": true}'}}],
                        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
                    }
                return HTTPResponse(
                    status=200,
                    headers={"content-type": "application/json"},
                    body=json.dumps(body).encode("utf-8"),
                    url=url,
                )

        t_http = ThinkingHTTP()
        client = OpenAICompatClient(
            base_url="https://api.deepseek.com/v1",
            api_key="sk-test",
            primary_model="deepseek-flash",
            extra_body={"reasoning_effort": "low"},
            http=t_http,
        )
        data, _ = client.chat_json("sys", "usr")
        self.assertEqual(data, {"ok": True})
        self.assertEqual(len(t_http.payloads), 2)
        self.assertEqual(t_http.payloads[1]["thinking"], {"type": "disabled"})

    def test_registry_sync_upgrade_and_rollback(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            shutil.copytree(repo_root / "config", tmp_root / "config")
            cfg = ConfigManager(root=tmp_root)
            store = StateStore(tmp_root / "data" / "state.db")

            mock_http = MockLLMHTTP(
                models=["deepseek-flash", "deepseek-v4.2-flash1120", "deepseek-v4-pro"]
            )
            client = OpenAICompatClient(
                base_url="https://api.deepseek.com/v1",
                api_key="sk-test-key",
                primary_model="deepseek-flash",
                fallback_models=["deepseek-v4-flash", "deepseek-v4-pro"],
                http=mock_http,
            )
            registry = ModelRegistry(cfg, client, store=store)
            report = registry.sync(apply=True, probe=True)

            self.assertTrue(report["upgraded"])
            self.assertEqual(report["recommended_upgrade"], "deepseek-v4.2-flash1120")

            updated_cfg = cfg.load("llm")
            self.assertEqual(updated_cfg["primary_model"], "deepseek-v4.2-flash1120")
            self.assertIn("deepseek-flash", updated_cfg["fallback_models"])

            # Rollback restores previous primary
            cfg.rollback("llm")
            restored_cfg = cfg.load("llm")
            self.assertEqual(restored_cfg["primary_model"], "deepseek-flash")


if __name__ == "__main__":
    unittest.main()
