"""End-to-end tests for filtering, curation, Feishu push, feedback, and self-evolution."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai_daily.config import ConfigManager
from ai_daily.evolution.engine import EvolutionEngine
from ai_daily.http_client import HTTPResponse
from ai_daily.llm.client import OpenAICompatClient
from ai_daily.models import Article, SourceFetchResult
from ai_daily.notifiers import FeishuAppNotifier
from ai_daily.pipeline.filter import canonicalize_url, prepare_candidates
from ai_daily.pipeline.generator import curate_candidates, render_digest_markdown
from ai_daily.pipeline.runner import run_daily_digest
from ai_daily.storage import StateStore


class FakeEndToEndHTTP:
    def __init__(self) -> None:
        self.feishu_payloads: list[dict] = []

    def get(self, url: str, **kwargs) -> HTTPResponse:
        now_rfc = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")
        if "aihot.news" in url or "aihot.virxact.com" in url:
            payload = {
                "items": [
                    {
                        "title": "ByteDance Seedance 2.0 Video Model Technical Update",
                        "summary": "字节跳动 Seed 团队发布 Seedance 新版本，运动一致性显著提升。",
                        "score": 8.9,
                        "links": {"original": "https://seed.bytedance.com/seedance-2"},
                        "source": {"name": "ByteDance Seed"},
                        "publishedAt": datetime.now(timezone.utc).isoformat(),
                    }
                ]
            }
            return HTTPResponse(
                status=200,
                headers={"content-type": "application/json"},
                body=json.dumps(payload).encode("utf-8"),
                url=url,
            )
        if "smol.ai" in url or "deepmind" in url or "news.google.com" in url or "openai/codex" in url:
            xml = f"""<?xml version="1.0" encoding="UTF-8"?>
            <rss version="2.0"><channel><title>Feed</title>
              <item>
                <title>Google Antigravity &amp; Codex Harness Latency Optimization</title>
                <link>https://deepmind.google/blog/antigravity-harness-update?utm_source=x</link>
                <description>Major agent harness scheduling speedup and DeepSeek inference optimization benchmarks.</description>
                <pubDate>{now_rfc}</pubDate>
              </item>
            </channel></rss>"""
            return HTTPResponse(status=200, headers={"content-type": "application/xml"}, body=xml.encode("utf-8"), url=url)
        if "huggingface.co/api/daily_papers" in url:
            return HTTPResponse(status=200, headers={"content-type": "application/json"}, body=b"[]", url=url)
        if "hn.algolia.com" in url:
            return HTTPResponse(status=200, headers={"content-type": "application/json"}, body=b'{"hits":[]}', url=url)
        if "api.github.com" in url:
            return HTTPResponse(status=200, headers={"content-type": "application/json"}, body=b'{"items":[]}', url=url)
        return HTTPResponse(status=200, headers={"content-type": "text/html"}, body=b"<html></html>", url=url)

    def post_json(self, url: str, payload: dict, **kwargs) -> HTTPResponse:
        if "/open-apis/auth/v3/tenant_access_token/internal" in url:
            return HTTPResponse(
                status=200,
                headers={"content-type": "application/json"},
                body=json.dumps({"code": 0, "tenant_access_token": "t-fake-token", "expire": 7200}).encode("utf-8"),
                url=url,
            )
        if "/open-apis/im/v1/messages" in url:
            self.feishu_payloads.append(payload)
            return HTTPResponse(
                status=200,
                headers={"content-type": "application/json"},
                body=json.dumps({"code": 0, "msg": "success", "data": {"message_id": "om_e2e_1"}}).encode("utf-8"),
                url=url,
            )
        if url.endswith("/chat/completions"):
            content = json.dumps(
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
                            "score": 9.4,
                            "zh_title": "Google Antigravity Harness 延迟优化与 Gemini 更新",
                            "one_liner": "Agent Harness 调度速度显著提升并同步公布 Gemini 新评测结果。",
                            "why_it_matters": "直接反映一线 Coding Agent Harness 的工程演进方向。",
                            "tags": ["Antigravity", "Harness", "Gemini"],
                        },
                        {
                            "id": 2,
                            "related_ids": [],
                            "track": "models",
                            "score": 8.9,
                            "zh_title": "字节 Seedance 2.0 视频生成模型更新",
                            "one_liner": "字节 Seed 团队更新 Seedance 模型，大幅提升复杂镜头运动一致性。",
                            "why_it_matters": "国产多模态视频模型关键迭代。",
                            "tags": ["Seedance", "ByteDance Seed"],
                        },
                    ],
                    "emerging_signals": [
                        {
                            "keyword": "Agentic Harness",
                            "track": "harness",
                            "reason": "近期工程博客中高频出现的新架构术语",
                        }
                    ],
                },
                ensure_ascii=False,
            )
            resp_obj = {
                "model": payload["model"],
                "choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": 500, "completion_tokens": 220},
            }
            return HTTPResponse(
                status=200,
                headers={"content-type": "application/json"},
                body=json.dumps(resp_obj).encode("utf-8"),
                url=url,
            )
        raise AssertionError(f"Unexpected POST URL: {url}")


class TestPipelineAndEvolution(unittest.TestCase):
    def test_canonical_url(self) -> None:
        u1 = canonicalize_url("https://www.example.com/a/b/?utm_source=x&id=1#top")
        self.assertEqual(u1, "https://example.com/a/b?id=1")
        u2 = canonicalize_url(
            "https://antigravity.google/docs/changelog#v2-4", keep_fragment=True
        )
        self.assertTrue(u2.endswith("#v2-4"))

    def test_dedup_and_prescore_excludes_noise(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        cfg = ConfigManager(root=repo_root)
        profile = cfg.load_profile()
        sources = cfg.load_sources()["sources"]
        now = datetime.now(timezone.utc)
        articles = [
            Article(
                source_id="google_deepmind_blog",
                source_name="DeepMind",
                category="models",
                title="Gemini & Antigravity Harness Deep Dive",
                url="https://deepmind.google/post-a?utm_medium=rss",
                published_at=now - timedelta(hours=2),
            ),
            Article(
                source_id="smol_ai_news",
                source_name="Smol",
                category="portals",
                title="Gemini & Antigravity Harness Deep Dive",
                url="https://deepmind.google/post-a",
                published_at=now - timedelta(hours=1),
            ),
            Article(
                source_id="jiqizhixin",
                source_name="机器之心",
                category="portals",
                title="Sponsored Webinar: Buy Crypto AI Coins",
                url="https://jiqizhixin.com/spam",
                published_at=now - timedelta(hours=1),
            ),
        ]
        cands, _ = prepare_candidates(
            articles,
            profile=profile,
            sources=sources,
            pushed_hashes=set(),
            pushed_norm_titles=set(),
        )
        self.assertEqual(len(cands), 1)
        self.assertIn("antigravity", [k.lower() for k in cands[0].matched_keywords])

    def test_end_to_end_run_and_evolution(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            shutil.copytree(repo_root / "config", tmp_root / "config")
            cfg = ConfigManager(root=tmp_root)
            fake_http = FakeEndToEndHTTP()
            llm = OpenAICompatClient(
                base_url="https://api.deepseek.com/v1",
                api_key="sk-fake",
                primary_model="deepseek-v4.1-flash1017",
                http=fake_http,
            )
            notifier = FeishuAppNotifier(
                app_id="cli_test",
                app_secret="sec_test",
                receive_id="oc_test_group",
                http=fake_http,
                retry_backoff_seconds=0.0,
            )

            summary = run_daily_digest(
                cfg,
                http=fake_http,
                llm_client=llm,
                notifier=notifier,
            )
            self.assertTrue(summary["pushed"])
            self.assertEqual(summary["selected_count"], 2)
            self.assertEqual(len(fake_http.feishu_payloads), 1)
            payload = fake_http.feishu_payloads[0]
            self.assertEqual(payload["msg_type"], "interactive")
            self.assertIn("https://deepmind.google/blog/antigravity-harness-update", payload["content"])
            self.assertIn("Seedance 2.0", payload["content"])

            # Second run immediately after should deduplicate already-pushed stories
            summary2 = run_daily_digest(
                cfg,
                http=fake_http,
                llm_client=llm,
                notifier=notifier,
                no_push=True,
            )
            self.assertEqual(summary2["candidate_count"], 0)

            # Record feedback + repeated failures, then run rule-based EvolutionEngine
            store = StateStore(tmp_root / "data" / "state.db")
            store.add_feedback("topic", "Context Compaction", "非常关注")
            store.add_feedback("dislike", "纯融资新闻")
            for idx in range(5):
                store.record_source_runs(
                    f"fail-run-{idx}",
                    [SourceFetchResult("smol_ai_news", "Smol.ai", ok=False, error="HTTP 503")],
                    {},
                    [],
                )

            engine = EvolutionEngine(cfg, store, llm_client=None)
            evo_res = engine.run_cycle(apply=True, use_llm=False)
            self.assertTrue(evo_res["apply_result"]["applied"])

            updated_profile = cfg.load_profile()
            self.assertIn("Context Compaction", updated_profile["boost_keywords"])
            self.assertIn("纯融资新闻", updated_profile["exclude_keywords"])

            updated_sources = {s["id"]: s for s in cfg.load_sources()["sources"]}
            self.assertFalse(updated_sources["smol_ai_news"]["enabled"])


if __name__ == "__main__":
    unittest.main()
