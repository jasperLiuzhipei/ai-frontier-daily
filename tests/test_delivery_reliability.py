"""Fault-injection & reliability tests for persistent drafts, Feishu App API uuid, Strategy B LLM degradation, and state-branch sync."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai_daily.cli import main as cli_main
from ai_daily.config import ConfigManager
from ai_daily.delivery import DeliveryStore
from ai_daily.http_client import HTTPErrorDetail, HTTPResponse, HTTPUnknownOutcomeError
from ai_daily.llm.client import OpenAICompatClient
from ai_daily.monitor import report_healthcheck
from ai_daily.notifiers import (
    DeliveryUnknownError,
    FeishuAppNotifier,
    FeishuNotifier,
)
from ai_daily.pipeline.runner import run_daily_digest


def _sample_llm_json() -> str:
    return json.dumps(
        {
            "headline_summary": [
                "Antigravity Harness 调度延迟大幅优化",
                "Seedance 2.0 发布，运动一致性显著提升",
            ],
            "selected": [
                {
                    "id": 1,
                    "related_ids": [],
                    "track": "harness",
                    "score": 9.3,
                    "zh_title": "Google Antigravity Harness 延迟优化",
                    "one_liner": "Agent Harness 调度速度显著提升。",
                    "why_it_matters": "一线 Coding Agent Harness 工程演进。",
                    "tags": ["Antigravity", "Harness"],
                }
            ],
            "emerging_signals": [],
        },
        ensure_ascii=False,
    )


class ScriptedReliabilityHTTP:
    """Deterministic HTTP mock supporting source outages, LLM bad JSON, and Feishu API/Webhook faults."""

    def __init__(self) -> None:
        self.source_get_calls = 0
        self.llm_calls: list[str] = []
        self.feishu_app_messages: list[dict] = []
        self.feishu_webhook_messages: list[dict] = []
        self.token_calls = 0

        self.all_sources_fail = False
        self.llm_behavior: dict[str, str] = {}  # model -> "ok" | "bad_json" | "bad_schema" | "http_500"
        self.feishu_app_mode = "ok"  # "ok" | "http_429" | "http_400" | "timeout" | "token_expired_once" | "timeout_once"
        self.feishu_webhook_mode = "ok"  # "ok" | "http_429" | "timeout" | "code_err"
        self._feishu_token_expired_fired = False
        self._feishu_timeout_once_fired = False

    def get(self, url: str, **kwargs) -> HTTPResponse:
        if "/open-apis/im/v1/chats" in url:
            body = {
                "code": 0,
                "data": {
                    "has_more": False,
                    "items": [
                        {"chat_id": "oc_test_group_001", "name": "AI 前沿日报群", "description": "", "chat_mode": "group"}
                    ],
                },
            }
            return HTTPResponse(200, {"content-type": "application/json"}, json.dumps(body).encode("utf-8"), url)

        self.source_get_calls += 1
        if self.all_sources_fail:
            raise HTTPErrorDetail(503, "Service Unavailable", "upstream outage")

        now_rfc = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")
        if "aihot.news" in url or "aihot.virxact.com" in url:
            payload = {
                "items": [
                    {
                        "title": "Google Antigravity Harness Latency Optimization",
                        "summary": "Major agent harness scheduling speedup and inference benchmarks.",
                        "score": 9.1,
                        "links": {"original": "https://deepmind.google/blog/antigravity-harness-opt"},
                        "source": {"name": "Google DeepMind"},
                        "publishedAt": datetime.now(timezone.utc).isoformat(),
                    }
                ]
            }
            return HTTPResponse(200, {"content-type": "application/json"}, json.dumps(payload).encode("utf-8"), url)
        if "deepmind" in url or "smol.ai" in url:
            xml = f"""<?xml version="1.0" encoding="UTF-8"?>
            <rss version="2.0"><channel><title>Feed</title>
              <item>
                <title>Google Antigravity Harness Latency Optimization</title>
                <link>https://deepmind.google/blog/antigravity-harness-opt</link>
                <description>Major agent harness scheduling speedup.</description>
                <pubDate>{now_rfc}</pubDate>
              </item>
            </channel></rss>"""
            return HTTPResponse(200, {"content-type": "application/xml"}, xml.encode("utf-8"), url)
        if "huggingface.co/api/daily_papers" in url or "huggingface.co/api/spaces" in url:
            return HTTPResponse(200, {"content-type": "application/json"}, b"[]", url)
        if "hn.algolia.com" in url:
            return HTTPResponse(200, {"content-type": "application/json"}, b'{"hits":[]}', url)
        if "api.github.com" in url:
            return HTTPResponse(200, {"content-type": "application/json"}, b'{"items":[]}', url)
        return HTTPResponse(200, {"content-type": "text/html"}, b"<html></html>", url)

    def post_json(self, url: str, payload: dict, **kwargs) -> HTTPResponse:
        if "/open-apis/auth/v3/tenant_access_token/internal" in url:
            self.token_calls += 1
            body = {"code": 0, "tenant_access_token": f"t-fake-token-{self.token_calls}", "expire": 7200}
            return HTTPResponse(200, {"content-type": "application/json"}, json.dumps(body).encode("utf-8"), url)

        if "/open-apis/im/v1/messages" in url:
            self.feishu_app_messages.append(dict(payload))
            if self.feishu_app_mode == "token_expired_once" and not self._feishu_token_expired_fired:
                self._feishu_token_expired_fired = True
                return HTTPResponse(
                    200,
                    {"content-type": "application/json"},
                    json.dumps({"code": 99991663, "msg": "token expired"}).encode("utf-8"),
                    url,
                )
            if self.feishu_app_mode == "timeout_once" and not self._feishu_timeout_once_fired:
                self._feishu_timeout_once_fired = True
                raise HTTPUnknownOutcomeError("read timed out")
            if self.feishu_app_mode == "timeout":
                raise HTTPUnknownOutcomeError("read timed out")
            if self.feishu_app_mode == "http_429":
                raise HTTPErrorDetail(429, "Too Many Requests", '{"code": 99991400, "msg": "rate limit"}')
            if self.feishu_app_mode == "http_400":
                raise HTTPErrorDetail(400, "Bad Request", '{"code": 230002, "msg": "bot not in chat"}')
            body = {"code": 0, "msg": "success", "data": {"message_id": f"om_msg_{len(self.feishu_app_messages)}"}}
            return HTTPResponse(200, {"content-type": "application/json"}, json.dumps(body).encode("utf-8"), url)

        if "/open-apis/bot/v2/hook/" in url:
            self.feishu_webhook_messages.append(dict(payload))
            if self.feishu_webhook_mode == "timeout":
                raise HTTPUnknownOutcomeError("webhook read timed out")
            if self.feishu_webhook_mode == "http_429":
                raise HTTPErrorDetail(429, "Too Many Requests", '{"code": 11232, "msg": "frequency limit"}')
            if self.feishu_webhook_mode == "code_err":
                return HTTPResponse(200, {"content-type": "application/json"}, json.dumps({"code": 19021, "msg": "sign mismatch"}).encode("utf-8"), url)
            return HTTPResponse(200, {"content-type": "application/json"}, json.dumps({"code": 0, "msg": "success"}).encode("utf-8"), url)

        if url.endswith("/chat/completions"):
            model = str(payload.get("model") or "")
            self.llm_calls.append(model)
            mode = self.llm_behavior.get(model, "ok")
            if mode == "http_500":
                raise HTTPErrorDetail(500, "Internal Server Error", "model overloaded")
            if mode == "bad_json":
                content = '{"selected": [{"id": 1, "zh_title": "truncated...'
            elif mode == "bad_schema":
                content = '{"unexpected": "no selected list"}'
            else:
                content = _sample_llm_json()
            resp_obj = {
                "model": model,
                "choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": 300, "completion_tokens": 120},
            }
            return HTTPResponse(200, {"content-type": "application/json"}, json.dumps(resp_obj).encode("utf-8"), url)

        raise AssertionError(f"Unexpected POST URL: {url}")


class TestDeliveryReliability(unittest.TestCase):
    def _setup_workspace(self, tmp: str) -> ConfigManager:
        repo_root = Path(__file__).resolve().parents[1]
        tmp_root = Path(tmp)
        shutil.copytree(repo_root / "config", tmp_root / "config")
        # Shrink sources.json to 2 fast mocked sources for crisp tests
        src_path = tmp_root / "config" / "sources.json"
        sources_data = json.loads(src_path.read_text(encoding="utf-8"))
        sources_data["sources"] = [
            s for s in sources_data["sources"] if s["id"] in ("aihot_news", "google_deepmind_blog")
        ]
        src_path.write_text(json.dumps(sources_data, ensure_ascii=False, indent=2), encoding="utf-8")
        return ConfigManager(root=tmp_root)

    def test_first_push_fails_then_second_run_reuses_draft_without_refetch_or_llm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._setup_workspace(tmp)
            http = ScriptedReliabilityHTTP()
            http.feishu_app_mode = "http_429"

            llm = OpenAICompatClient(
                base_url="https://api.deepseek.com/v1",
                api_key="sk-fake",
                primary_model="deepseek-flash",
                fallback_models=["deepseek-v4-pro"],
                http=http,
            )
            notifier = FeishuAppNotifier(
                app_id="cli_test",
                app_secret="sec_test",
                receive_id="oc_test_group_001",
                http=http,
                max_delivery_retries=1,
                retry_backoff_seconds=0.0,
            )
            state_dir = cfg.root / "state-branch"

            # Round 1 (10:23): Feishu rate-limited -> fails cleanly, NO output/, NO sent_history.json!
            r1 = run_daily_digest(
                cfg,
                http=http,
                llm_client=llm,
                notifier=notifier,
                state_dir=state_dir,
                run_date_override="2026-11-22",
            )
            self.assertFalse(r1["ok"])
            self.assertFalse(r1["pushed"])
            self.assertEqual(r1["delivery_status"], "failed")
            self.assertFalse((cfg.output_dir / "2026-11-22.md").exists())
            self.assertFalse((cfg.data_dir / "sent_history.json").exists())
            self.assertTrue((state_dir / "2026-11-22" / "draft.json").is_file())

            first_uuid = http.feishu_app_messages[0]["uuid"]
            self.assertLessEqual(len(first_uuid), 50)
            calls_after_r1 = (http.source_get_calls, len(http.llm_calls))

            # Round 2 (10:47 fallback): Feishu recovers -> reuses draft, ZERO source fetch / LLM calls, SAME uuid!
            http.feishu_app_mode = "ok"
            r2 = run_daily_digest(
                cfg,
                http=http,
                llm_client=llm,
                notifier=notifier,
                state_dir=state_dir,
                run_date_override="2026-11-22",
            )
            self.assertTrue(r2["ok"])
            self.assertTrue(r2["pushed"])
            self.assertTrue(r2["reused_draft"])
            self.assertEqual(r2["delivery_status"], "sent")
            self.assertEqual((http.source_get_calls, len(http.llm_calls)), calls_after_r1)
            self.assertEqual(http.feishu_app_messages[-1]["uuid"], first_uuid)
            self.assertTrue((cfg.output_dir / "2026-11-22.md").is_file())
            self.assertTrue((cfg.data_dir / "sent_history.json").is_file())

            # Round 3 (extra trigger): receipt confirms sent -> idempotently skipped, zero extra sends!
            sent_before_r3 = len(http.feishu_app_messages)
            r3 = run_daily_digest(
                cfg,
                http=http,
                llm_client=llm,
                notifier=notifier,
                state_dir=state_dir,
                run_date_override="2026-11-22",
            )
            self.assertTrue(r3["ok"])
            self.assertTrue(r3["skipped"])
            self.assertTrue(r3["already_sent"])
            self.assertEqual(len(http.feishu_app_messages), sent_before_r3)

    def test_feishu_app_api_token_refresh_chat_list_and_timeout_idempotent_uuid(self) -> None:
        http = ScriptedReliabilityHTTP()
        notifier = FeishuAppNotifier(
            app_id="cli_test",
            app_secret="sec_test",
            receive_id="oc_test_group_001",
            http=http,
            max_delivery_retries=1,
            max_unknown_retries=1,
            retry_backoff_seconds=0.0,
        )

        # 1. list_chats works with im:chat:readonly
        chats = notifier.list_chats()
        self.assertEqual(len(chats), 1)
        self.assertEqual(chats[0]["chat_id"], "oc_test_group_001")

        # 2. Token expired mid-call -> automatically refreshes token and succeeds
        http.feishu_app_mode = "token_expired_once"
        res = notifier.send("测试标题", "**测试内容**", idempotency_key="fixed-uuid-001")
        self.assertEqual(res["code"], 0)
        self.assertTrue(res["message_id"].startswith("om_msg_"))
        self.assertGreaterEqual(http.token_calls, 2)

        # 3. Single timeout with idempotent uuid -> safe retry with EXACT SAME uuid succeeds
        http.feishu_app_messages.clear()
        http.feishu_app_mode = "timeout_once"
        res2 = notifier.send("测试标题", "**测试内容**", idempotency_key="fixed-uuid-002")
        self.assertEqual(res2["code"], 0)
        self.assertEqual(len(http.feishu_app_messages), 2)
        self.assertEqual(http.feishu_app_messages[0]["uuid"], "fixed-uuid-002")
        self.assertEqual(http.feishu_app_messages[1]["uuid"], "fixed-uuid-002")
        # Content must be a JSON-encoded string per Feishu im/v1/messages spec
        self.assertIsInstance(http.feishu_app_messages[0]["content"], str)
        parsed_card = json.loads(http.feishu_app_messages[0]["content"])
        self.assertIn("elements", parsed_card)

        # 4. Persistent timeout -> raises DeliveryUnknownError
        http.feishu_app_mode = "timeout"
        with self.assertRaises(DeliveryUnknownError):
            notifier.send("测试标题", "**测试内容**", idempotency_key="fixed-uuid-003")

    def test_webhook_timeout_no_blind_same_run_retry_and_unknown_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._setup_workspace(tmp)
            http = ScriptedReliabilityHTTP()
            http.feishu_webhook_mode = "timeout"

            llm = OpenAICompatClient(
                base_url="https://api.deepseek.com/v1",
                api_key="sk-fake",
                primary_model="deepseek-flash",
                http=http,
            )
            hook = FeishuNotifier(
                webhook_url="https://open.feishu.cn/open-apis/bot/v2/hook/secret123",
                http=http,
                max_delivery_retries=2,
                retry_backoff_seconds=0.0,
            )
            state_dir = cfg.root / "state"

            # Round 1: timeout -> MUST NOT blindly retry within same run (exactly 1 POST), status=unknown
            r1 = run_daily_digest(
                cfg,
                http=http,
                llm_client=llm,
                notifier=hook,
                state_dir=state_dir,
                run_date_override="2026-11-22",
            )
            self.assertFalse(r1["ok"])
            self.assertEqual(r1["delivery_status"], "unknown")
            self.assertEqual(len(http.feishu_webhook_messages), 1)

            # Strict at-most-once mode (--no-retry-unknown): blocks resend
            r2_blocked = run_daily_digest(
                cfg,
                http=http,
                llm_client=llm,
                notifier=hook,
                state_dir=state_dir,
                retry_unknown=False,
                run_date_override="2026-11-22",
            )
            self.assertFalse(r2_blocked["ok"])
            self.assertEqual(r2_blocked["delivery_status"], "unknown")
            self.assertEqual(len(http.feishu_webhook_messages), 1)

            # User-chosen at-least-once mode (retry_unknown=True): allows 1 fallback recovery send
            http.feishu_webhook_mode = "ok"
            r2_ok = run_daily_digest(
                cfg,
                http=http,
                llm_client=llm,
                notifier=hook,
                state_dir=state_dir,
                retry_unknown=True,
                run_date_override="2026-11-22",
            )
            self.assertTrue(r2_ok["ok"])
            self.assertEqual(r2_ok["delivery_status"], "sent")
            self.assertEqual(len(http.feishu_webhook_messages), 2)

    def test_partial_channel_failure_only_retries_failed_channel(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._setup_workspace(tmp)
            http = ScriptedReliabilityHTTP()
            http.feishu_app_mode = "ok"
            http.feishu_webhook_mode = "code_err"

            llm = OpenAICompatClient(
                base_url="https://api.deepseek.com/v1",
                api_key="sk-fake",
                primary_model="deepseek-flash",
                http=http,
            )
            fs = FeishuAppNotifier(
                app_id="cli_test",
                app_secret="sec_test",
                receive_id="oc_test_group_001",
                http=http,
                retry_backoff_seconds=0.0,
            )

            class SecondaryFeishuWebhook(FeishuNotifier):
                name = "feishu_backup"

            backup_hook = SecondaryFeishuWebhook(
                webhook_url="https://open.feishu.cn/open-apis/bot/v2/hook/backup123",
                http=http,
                max_delivery_retries=0,
                retry_backoff_seconds=0.0,
            )
            state_dir = cfg.root / "state"

            # Round 1: Primary Feishu App API succeeds, secondary Feishu webhook fails
            r1 = run_daily_digest(
                cfg,
                http=http,
                llm_client=llm,
                notifiers=[fs, backup_hook],
                state_dir=state_dir,
                run_date_override="2026-11-22",
            )
            self.assertFalse(r1["ok"])
            self.assertEqual(r1["channel_statuses"], {"feishu": "sent", "feishu_backup": "failed"})
            self.assertEqual(len(http.feishu_app_messages), 1)
            self.assertEqual(len(http.feishu_webhook_messages), 1)

            # Round 2: Secondary webhook recovers -> ONLY failed channel is retried; primary Feishu is NOT sent again!
            http.feishu_webhook_mode = "ok"
            r2 = run_daily_digest(
                cfg,
                http=http,
                llm_client=llm,
                notifiers=[fs, backup_hook],
                state_dir=state_dir,
                run_date_override="2026-11-22",
            )
            self.assertTrue(r2["ok"])
            self.assertTrue(r2["reused_draft"])
            self.assertEqual(len(http.feishu_app_messages), 1)  # zero primary duplicate!
            self.assertEqual(len(http.feishu_webhook_messages), 2)

    def test_primary_bad_json_switches_to_fallback_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._setup_workspace(tmp)
            http = ScriptedReliabilityHTTP()
            http.llm_behavior = {
                "deepseek-flash": "bad_json",
                "deepseek-v4-pro": "ok",
            }
            llm = OpenAICompatClient(
                base_url="https://api.deepseek.com/v1",
                api_key="sk-fake",
                primary_model="deepseek-flash",
                fallback_models=["deepseek-v4-pro"],
                http=http,
            )
            fs = FeishuAppNotifier(
                app_id="cli_test",
                app_secret="sec_test",
                receive_id="oc_test_group_001",
                http=http,
                retry_backoff_seconds=0.0,
            )

            res = run_daily_digest(
                cfg,
                http=http,
                llm_client=llm,
                notifier=fs,
                run_date_override="2026-11-22",
            )
            self.assertTrue(res["ok"])
            self.assertFalse(res["degraded"])
            self.assertEqual(res["model_used"], "deepseek-v4-pro")
            self.assertEqual(http.llm_calls, ["deepseek-flash", "deepseek-v4-pro"])

            # Also verify schema-invalid JSON triggers fallback
            http2 = ScriptedReliabilityHTTP()
            http2.llm_behavior = {
                "deepseek-flash": "bad_schema",
                "deepseek-v4-pro": "ok",
            }
            llm2 = OpenAICompatClient(
                base_url="https://api.deepseek.com/v1",
                api_key="sk-fake",
                primary_model="deepseek-flash",
                fallback_models=["deepseek-v4-pro"],
                http=http2,
            )
            parsed, llm_resp = llm2.chat_json(
                "sys", "user", validator=lambda obj: isinstance(obj.get("selected"), list)
            )
            self.assertEqual(llm_resp.model, "deepseek-v4-pro")
            self.assertIsInstance(parsed["selected"], list)

    def test_strategy_b_all_models_fail_primary_fails_and_fallback_degrades(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._setup_workspace(tmp)
            http = ScriptedReliabilityHTTP()
            http.llm_behavior = {
                "deepseek-flash": "bad_json",
                "deepseek-v4-pro": "http_500",
            }
            llm = OpenAICompatClient(
                base_url="https://api.deepseek.com/v1",
                api_key="sk-fake",
                primary_model="deepseek-flash",
                fallback_models=["deepseek-v4-pro"],
                auto_discover_on_error=False,
                http=http,
            )
            fs = FeishuAppNotifier(
                app_id="cli_test",
                app_secret="sec_test",
                receive_id="oc_test_group_001",
                http=http,
                retry_backoff_seconds=0.0,
            )
            state_dir = cfg.root / "state"

            # 10:23 primary run (degrade_on_llm_error=False): fails cleanly, no degraded draft locked!
            r1 = run_daily_digest(
                cfg,
                http=http,
                llm_client=llm,
                notifier=fs,
                state_dir=state_dir,
                degrade_on_llm_error=False,
                run_date_override="2026-11-22",
            )
            self.assertFalse(r1["ok"])
            self.assertEqual(r1["delivery_status"], "llm_failed")
            self.assertFalse((state_dir / "2026-11-22" / "draft.json").exists())
            self.assertFalse((cfg.output_dir / "2026-11-22.md").exists())
            self.assertEqual(len(http.feishu_app_messages), 0)

            # 10:47 fallback run (degrade_on_llm_error=True): still failing -> sends clearly badged degraded short digest
            r2 = run_daily_digest(
                cfg,
                http=http,
                llm_client=llm,
                notifier=fs,
                state_dir=state_dir,
                degrade_on_llm_error=True,
                run_date_override="2026-11-22",
            )
            self.assertTrue(r2["ok"])
            self.assertTrue(r2["degraded"])
            self.assertEqual(r2["delivery_status"], "sent")
            self.assertEqual(len(http.feishu_app_messages), 1)
            card = json.loads(http.feishu_app_messages[0]["content"])
            self.assertEqual(card["header"]["template"], "orange")
            self.assertIn("降级短报", card["header"]["title"]["content"])
            md = (cfg.output_dir / "2026-11-22.md").read_text(encoding="utf-8")
            self.assertIn("降级短报", md)

    def test_all_sources_fail_preserves_failure_state_and_next_run_recovers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._setup_workspace(tmp)
            http = ScriptedReliabilityHTTP()
            http.all_sources_fail = True
            llm = OpenAICompatClient(
                base_url="https://api.deepseek.com/v1",
                api_key="sk-fake",
                primary_model="deepseek-flash",
                http=http,
            )
            fs = FeishuAppNotifier(
                app_id="cli_test",
                app_secret="sec_test",
                receive_id="oc_test_group_001",
                http=http,
                retry_backoff_seconds=0.0,
            )
            state_dir = cfg.root / "state"

            r1 = run_daily_digest(
                cfg,
                http=http,
                llm_client=llm,
                notifier=fs,
                state_dir=state_dir,
                run_date_override="2026-11-22",
            )
            self.assertFalse(r1["ok"])
            self.assertEqual(r1["delivery_status"], "fetch_failed")
            self.assertFalse((cfg.output_dir / "2026-11-22.md").exists())
            self.assertFalse((state_dir / "2026-11-22" / "draft.json").exists())
            self.assertEqual(len(http.llm_calls), 0)

            # Sources recover on next run -> full run succeeds
            http.all_sources_fail = False
            r2 = run_daily_digest(
                cfg,
                http=http,
                llm_client=llm,
                notifier=fs,
                state_dir=state_dir,
                run_date_override="2026-11-22",
            )
            self.assertTrue(r2["ok"])
            self.assertEqual(r2["delivery_status"], "sent")
            self.assertTrue((cfg.output_dir / "2026-11-22.md").is_file())

    def test_stage_split_and_state_branch_sync_with_receipt_push_failure(self) -> None:
        """Simulate GitHub Actions staged execution + post-send receipt push failure + 10:47 recovery."""
        repo_root = Path(__file__).resolve().parents[1]
        sync_script = repo_root / "scripts" / "sync_state_branch.sh"

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            remote_bare = tmp_path / "remote.git"
            work_repo = tmp_path / "work"
            subprocess.run(["git", "init", "--bare", str(remote_bare)], check=True, capture_output=True)
            subprocess.run(["git", "init", "-b", "main", str(work_repo)], check=True, capture_output=True)
            subprocess.run(["git", "-C", str(work_repo), "config", "user.name", "tester"], check=True)
            subprocess.run(["git", "-C", str(work_repo), "config", "user.email", "t@example.com"], check=True)
            subprocess.run(["git", "-C", str(work_repo), "remote", "add", "origin", str(remote_bare)], check=True)
            shutil.copytree(repo_root / "config", work_repo / "config")
            src_path = work_repo / "config" / "sources.json"
            sources_data = json.loads(src_path.read_text(encoding="utf-8"))
            sources_data["sources"] = [
                s for s in sources_data["sources"] if s["id"] in ("aihot_news", "google_deepmind_blog")
            ]
            src_path.write_text(json.dumps(sources_data, ensure_ascii=False, indent=2), encoding="utf-8")
            (work_repo / ".gitignore").write_text(".delivery-state/\ndata/state.db\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(work_repo), "add", "."], check=True)
            subprocess.run(["git", "-C", str(work_repo), "commit", "-m", "init"], check=True, capture_output=True)
            subprocess.run(["git", "-C", str(work_repo), "push", "-u", "origin", "main"], check=True, capture_output=True)

            # Step 1: Pull/init orphan delivery-state branch into .delivery-state
            subprocess.run(
                ["bash", str(sync_script), "pull", "delivery-state", ".delivery-state"],
                cwd=str(work_repo),
                check=True,
                capture_output=True,
            )

            cfg = ConfigManager(root=work_repo)
            state_dir = work_repo / ".delivery-state"
            http = ScriptedReliabilityHTTP()
            llm = OpenAICompatClient(
                base_url="https://api.deepseek.com/v1",
                api_key="sk-fake",
                primary_model="deepseek-flash",
                http=http,
            )
            fs = FeishuAppNotifier(
                app_id="cli_test",
                app_secret="sec_test",
                receive_id="oc_test_group_001",
                http=http,
                retry_backoff_seconds=0.0,
            )

            # Stage A: prepare -> push pending draft to delivery-state
            prep = run_daily_digest(
                cfg,
                stage="prepare",
                state_dir=state_dir,
                http=http,
                llm_client=llm,
                notifier=fs,
                run_date_override="2026-11-22",
            )
            self.assertTrue(prep["ok"])
            self.assertEqual(prep["delivery_status"], "pending")
            subprocess.run(
                ["bash", str(sync_script), "push", "delivery-state", ".delivery-state", "prepare 2026-11-22"],
                cwd=str(work_repo),
                check=True,
                capture_output=True,
            )

            # Stage B: deliver succeeds on Feishu, but SIMULATE post-send git push failure (don't push receipt!)
            deliv1 = run_daily_digest(
                cfg,
                stage="deliver",
                state_dir=state_dir,
                http=http,
                llm_client=llm,
                notifier=fs,
                run_date_override="2026-11-22",
            )
            self.assertTrue(deliv1["ok"])
            uuid_round1 = http.feishu_app_messages[0]["uuid"]

            # Next runner (10:47) checks out fresh remote delivery-state (which only has the pushed draft + pending)
            subprocess.run(
                ["bash", str(sync_script), "pull", "delivery-state", ".delivery-state"],
                cwd=str(work_repo),
                check=True,
                capture_output=True,
            )
            prep2 = run_daily_digest(
                cfg,
                stage="prepare",
                state_dir=state_dir,
                http=http,
                llm_client=llm,
                notifier=fs,
                run_date_override="2026-11-22",
            )
            self.assertTrue(prep2["reused_draft"])
            self.assertFalse(prep2["already_sent"])

            deliv2 = run_daily_digest(
                cfg,
                stage="deliver",
                state_dir=state_dir,
                http=http,
                llm_client=llm,
                notifier=fs,
                run_date_override="2026-11-22",
            )
            self.assertTrue(deliv2["ok"])
            # Exact same uuid sent within 1-hour Feishu dedup window -> zero duplicate on Feishu!
            self.assertEqual(http.feishu_app_messages[1]["uuid"], uuid_round1)

            # Push confirmed receipt cleanly
            subprocess.run(
                ["bash", str(sync_script), "push", "delivery-state", ".delivery-state", "receipt sent"],
                cwd=str(work_repo),
                check=True,
                capture_output=True,
            )

            # Verify metrics summary & CLI status command
            metrics = DeliveryStore(state_dir).summarize_metrics(days=14)
            self.assertEqual(metrics["delivered"], 1)
            self.assertEqual(metrics["failed_or_pending"], 0)
            rc = cli_main(["--root", str(work_repo), "status", "--days", "14", "--state-dir", str(state_dir)])
            self.assertEqual(rc, 0)

    def test_healthcheck_monitor_ok_fail_and_redaction(self) -> None:
        calls: list[tuple[str, str, bytes]] = []

        class FakeMonitorHTTP:
            def request(self, method: str, url: str, **kwargs) -> HTTPResponse:
                calls.append((method, url, kwargs.get("data") or b""))
                return HTTPResponse(200, {}, b"OK", url)

        ok1, msg1 = report_healthcheck(
            "https://hc-ping.com/secret-uuid-1234",
            status="ok",
            message="date=2026-11-22 status=sent",
            http=FakeMonitorHTTP(),
        )
        self.assertTrue(ok1)
        self.assertNotIn("secret-uuid-1234", msg1)
        self.assertIn("***REDACTED***", msg1)
        self.assertEqual(calls[0][1], "https://hc-ping.com/secret-uuid-1234")

        ok2, _ = report_healthcheck(
            "https://hc-ping.com/secret-uuid-1234",
            status="fail",
            message="deliver failed",
            http=FakeMonitorHTTP(),
        )
        self.assertTrue(ok2)
        self.assertEqual(calls[1][1], "https://hc-ping.com/secret-uuid-1234/fail")

        ok3, msg3 = report_healthcheck("", status="ok", http=FakeMonitorHTTP())
        self.assertFalse(ok3)
        self.assertIn("未配置", msg3)

    def test_resolve_notifiers_prefers_feishu_app_over_webhook_and_cli_exit_codes(self) -> None:
        import os
        from ai_daily.notifiers import resolve_notifiers

        keys = (
            "NOTIFY_CHANNEL",
            "FEISHU_APP_ID",
            "FEISHU_APP_SECRET",
            "FEISHU_CHAT_ID",
            "FEISHU_WEBHOOK_URL",
        )
        old = {k: os.environ.get(k) for k in keys}
        try:
            os.environ["NOTIFY_CHANNEL"] = "feishu"
            os.environ["FEISHU_APP_ID"] = "cli_test_app"
            os.environ["FEISHU_APP_SECRET"] = "sec_test_app"
            os.environ["FEISHU_CHAT_ID"] = "oc_test_chat"
            os.environ["FEISHU_WEBHOOK_URL"] = "https://open.feishu.cn/open-apis/bot/v2/hook/abc"

            chosen = resolve_notifiers()
            self.assertEqual(len(chosen), 1)
            self.assertIsInstance(chosen[0], FeishuAppNotifier)
            self.assertTrue(chosen[0].supports_idempotent_retry)

            # Removing FEISHU_CHAT_ID falls back gracefully to Webhook
            os.environ.pop("FEISHU_CHAT_ID", None)
            chosen_fb = resolve_notifiers()
            self.assertIsInstance(chosen_fb[0], FeishuNotifier)
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

        # CLI returns non-zero when delivery cannot proceed (missing draft in deliver stage)
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._setup_workspace(tmp)
            rc_missing = cli_main(
                [
                    "--root",
                    str(cfg.root),
                    "run",
                    "--stage",
                    "deliver",
                    "--date",
                    "2026-11-22",
                    "--state-dir",
                    str(cfg.root / "empty-state"),
                ]
            )
            self.assertEqual(rc_missing, 1)


if __name__ == "__main__":
    unittest.main()
