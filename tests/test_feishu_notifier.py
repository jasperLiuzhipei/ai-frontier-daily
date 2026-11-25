"""Tests for Feishu Interactive Card notifier, HMAC-SHA256 signing, size budget, and channel routing."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai_daily.http_client import HTTPResponse, _redact_url
from ai_daily.models import Article, CuratedItem, CurateResult, EmergingSignal
from ai_daily.notifiers import FeishuNotifier, build_digest_card, generate_feishu_sign, resolve_notifiers
from ai_daily.notifiers.feishu import FEISHU_MAX_CARD_BYTES, _trim_card_to_budget


class MockFeishuHTTP:
    def __init__(self) -> None:
        self.sent: list[tuple[str, dict]] = []

    def post_json(self, url: str, payload: dict, **kwargs) -> HTTPResponse:
        self.sent.append((url, payload))
        return HTTPResponse(
            status=200,
            headers={"content-type": "application/json"},
            body=json.dumps({"code": 0, "msg": "success"}).encode("utf-8"),
            url=url,
        )


class TestFeishuNotifier(unittest.TestCase):
    def test_hmac_sha256_signature_matches_feishu_spec(self) -> None:
        ts = 1795200000
        secret = "test-secret-key"
        expected = base64.b64encode(
            hmac.new(f"{ts}\n{secret}".encode("utf-8"), b"", hashlib.sha256).digest()
        ).decode("utf-8")
        self.assertEqual(generate_feishu_sign(secret, ts), expected)

    def test_build_structured_digest_card_and_send_with_sign(self) -> None:
        art = Article(
            source_id="openai_codex_releases",
            source_name="OpenAI Codex Releases",
            category="harness",
            title="0.171.0",
            url="https://github.com/openai/codex/releases/tag/rust-v0.171.0",
        )
        res = CurateResult(
            headline_summary=["Codex 0.171.0 默认开启 TUI 工作区"],
            selected=[
                CuratedItem(
                    candidate_id=1,
                    article=art,
                    related_articles=[],
                    track="harness",
                    score=9.2,
                    zh_title="Codex 0.171.0：TUI 工作区转正",
                    one_liner="默认启用 TUI 并支持子智能体轨迹预览。",
                    why_it_matters="终端 Harness 交互标杆更新。",
                    tags=["Codex", "Harness"],
                )
            ],
            emerging_signals=[EmergingSignal("TUI Workspace", "harness", "多智能体终端工作区")],
            model_used="deepseek-flash",
            prompt_tokens=1000,
            completion_tokens=400,
        )
        card = build_digest_card(
            res,
            run_date="2026-11-21",
            raw_count=100,
            sources_ok=17,
            sources_total=17,
        )
        self.assertEqual(card["header"]["template"], "indigo")
        self.assertIn("2026-11-21", card["header"]["title"]["content"])
        card_json = json.dumps(card, ensure_ascii=False)
        self.assertIn("https://github.com/openai/codex/releases/tag/rust-v0.171.0", card_json)
        self.assertIn("TUI Workspace", card_json)

        mock_http = MockFeishuHTTP()
        notifier = FeishuNotifier(
            webhook_url="https://open.feishu.cn/open-apis/bot/v2/hook/abc123token",
            secret="my-sign-secret",
            http=mock_http,
        )
        notifier.send_digest(
            res,
            run_date="2026-11-21",
            raw_count=100,
            sources_ok=17,
            sources_total=17,
        )
        self.assertEqual(len(mock_http.sent), 1)
        _, payload = mock_http.sent[0]
        self.assertEqual(payload["msg_type"], "interactive")
        self.assertIn("timestamp", payload)
        self.assertIn("sign", payload)

    def test_trim_card_budget_and_url_redaction(self) -> None:
        huge_card = {
            "config": {"wide_screen_mode": True},
            "header": {"title": {"tag": "plain_text", "content": "Huge"}, "template": "indigo"},
            "elements": [
                {"tag": "markdown", "content": f"Section {i}: " + ("前沿内容" * 300)}
                for i in range(20)
            ],
        }
        trimmed = _trim_card_to_budget(huge_card, max_bytes=8000)
        self.assertLessEqual(
            len(json.dumps(trimmed, ensure_ascii=False).encode("utf-8")), 8000
        )
        self.assertIn("30KB", json.dumps(trimmed, ensure_ascii=False))

        redacted = _redact_url("https://open.feishu.cn/open-apis/bot/v2/hook/secret-token-999")
        self.assertNotIn("secret-token-999", redacted)
        self.assertIn("***REDACTED***", redacted)

    def test_resolve_notifiers_prefers_feishu_app_in_auto(self) -> None:
        old_env = {
            k: os.environ.get(k)
            for k in (
                "NOTIFY_CHANNEL",
                "FEISHU_APP_ID",
                "FEISHU_APP_SECRET",
                "FEISHU_CHAT_ID",
                "FEISHU_WEBHOOK_URL",
            )
        }
        try:
            os.environ["NOTIFY_CHANNEL"] = "auto"
            os.environ["FEISHU_APP_ID"] = "cli_test_app"
            os.environ["FEISHU_APP_SECRET"] = "sec_test_app"
            os.environ["FEISHU_CHAT_ID"] = "oc_test_chat"
            os.environ["FEISHU_WEBHOOK_URL"] = "https://open.feishu.cn/open-apis/bot/v2/hook/live123"

            auto_targets = resolve_notifiers()
            self.assertEqual([t.delivery_mode for t in auto_targets], ["app_api"])

            hook_targets = resolve_notifiers(channel="feishu-webhook")
            self.assertEqual([t.delivery_mode for t in hook_targets], ["webhook"])

            app_targets = resolve_notifiers(channel="feishu-app")
            self.assertEqual([t.delivery_mode for t in app_targets], ["app_api"])
        finally:
            for k, v in old_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


if __name__ == "__main__":
    unittest.main()
