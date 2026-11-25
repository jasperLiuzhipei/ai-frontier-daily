"""Tests for v2 quality upgrades: freshness/snapshot-date guard, enrichment, quotas, and metadata cleanup."""

from __future__ import annotations

import json
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai_daily.fetchers.aihot import clean_source_name
from ai_daily.fetchers.hf_spaces import fetch_hf_spaces_source
from ai_daily.history import SentHistoryStore, article_matches_history
from ai_daily.http_client import HTTPResponse
from ai_daily.models import Article, CuratedItem, CurateResult, format_related_links
from ai_daily.notifiers.feishu import build_digest_card
from ai_daily.pipeline.enrich import extract_article_body_and_anchors, extract_meta_published_date
from ai_daily.pipeline.filter import infer_embedded_release_date, infer_origin_date, prepare_candidates
from ai_daily.pipeline.generator import compute_or_gate_score, render_digest_markdown


class TestQualityUpgrades(unittest.TestCase):
    def test_snapshot_date_filters_month_old_release(self) -> None:
        now = datetime(2026, 11, 21, 10, 0, tzinfo=timezone.utc)
        inferred = infer_embedded_release_date(
            "[DeepSeek API 官方更新日志] DeepSeek-V4.1-Flash-1017 Update",
            now,
        )
        self.assertIsNotNone(inferred)
        self.assertEqual((inferred.year, inferred.month, inferred.day), (2026, 10, 17))

        # Should be rejected by prepare_candidates against 72h hard cutoff
        old_art = Article(
            source_id="deepseek_changelog",
            source_name="DeepSeek API 官方更新日志",
            category="models",
            title="[DeepSeek API 官方更新日志] DeepSeek-V4.1-Flash-1017 Update",
            url="https://api-docs.deepseek.com/updates#deepseek-v41-flash-1017-update",
            summary="DeepSeek V4.1 Flash upgraded to 1017",
            published_at=None,
        )
        cands, _ = prepare_candidates(
            [old_art],
            profile={"window_hours": 36, "hard_max_age_hours": 72, "tracks": [], "exclude_keywords": []},
            sources=[{"id": "deepseek_changelog", "type": "web_watch", "mode": "changelog", "weight": 1.5}],
            pushed_hashes=set(),
            pushed_norm_titles=set(),
        )
        self.assertEqual(len(cands), 0)

    def test_clean_source_name_and_dedup_related_links(self) -> None:
        self.assertEqual(
            clean_source_name("Thinking Machines Lab：News（网页）"),
            "Thinking Machines Lab",
        )
        self.assertEqual(
            clean_source_name("Google Research：Blog（RSS）"),
            "Google Research",
        )
        primary = Article(
            source_id="antigravity_changelog",
            source_name="Google Antigravity Changelog",
            category="harness",
            title="[Google Antigravity Changelog] Gemini 4 Neon, MCP Apps",
            url="https://antigravity.google/docs/changelog#a1",
        )
        rel1 = Article(
            source_id="antigravity_changelog",
            source_name="Google Antigravity Changelog",
            category="harness",
            title="[Google Antigravity Changelog] Dynamic Workflows",
            url="https://antigravity.google/docs/changelog#b2",
        )
        rel_dup = Article(
            source_id="antigravity_changelog",
            source_name="Google Antigravity Changelog",
            category="harness",
            title="[Google Antigravity Changelog] Dynamic Workflows",
            url="https://antigravity.google/docs/changelog#b2",
        )
        pairs = format_related_links(primary, [rel1, rel_dup])
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0][0], "Dynamic Workflows")

    def test_extract_article_body_and_geek_anchors(self) -> None:
        html_doc = """
        <html><body>
          <nav>Home About Pricing</nav>
          <article>
            <h1>Introducing Inkling 2 Mini</h1>
            <p>We train layers 1-20 natively in NVFP4 and layers 21-24 in MXFP8 with Attention Residuals and 128-expert MoE (top-8).</p>
            <p>Post-training uses 2.2T tokens of RL plus on-policy distillation. Chinese output may occasionally truncate.</p>
            <pre><code>uv tool install browser-use</code></pre>
            <a href="https://huggingface.co/thinkingmachines/inkling-2-mini">Weights</a>
          </article>
          <footer>Copyright 2026</footer>
        </body></html>"""
        body, anchors = extract_article_body_and_anchors(html_doc, max_chars=2000)
        self.assertIn("NVFP4", body)
        self.assertIn("MXFP8", body)
        self.assertNotIn("Copyright 2026", body)
        self.assertTrue(any("uv tool install browser-use" in a for a in anchors))
        self.assertTrue(any("huggingface.co/thinkingmachines/inkling-2-mini" in a for a in anchors))

    def test_skip_if_today_exists_is_idempotent(self) -> None:
        import shutil
        import tempfile
        from ai_daily.config import ConfigManager
        from ai_daily.pipeline.runner import run_daily_digest

        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            shutil.copytree(repo_root / "config", tmp_root / "config")
            today = datetime.now().astimezone().strftime("%Y-%m-%d")
            out_dir = tmp_root / "output"
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{today}.md").write_text("# already sent\n", encoding="utf-8")

            res = run_daily_digest(ConfigManager(root=tmp_root), skip_if_today_exists=True)
            self.assertTrue(res.get("skipped"))
            self.assertFalse(res.get("pushed"))

    def test_cross_day_history_and_origin_date_penetration(self) -> None:
        import tempfile

        now = datetime(2026, 11, 22, 10, 0, tzinfo=timezone.utc)
        # 1. URL path date from August 2026 -> detected as old
        dt_url, tag_url = infer_origin_date(
            "https://example.com/2026/08/13/deepseek-harness-preview",
            "DeepSeek Harness preview",
            now,
        )
        self.assertIsNotNone(dt_url)
        self.assertEqual(tag_url, "url:2026-08-13")

        # 2. Old arXiv paper reshared in Nov -> detected as 2026-10
        dt_ax, tag_ax = infer_origin_date(
            "https://x.com/rohanpaul/status/123",
            "Great paper arXiv:2610.06783 on Claude 3SUM/APSP",
            now,
        )
        self.assertIsNotNone(dt_ax)
        self.assertEqual(tag_ax, "arxiv:2026-10")

        # 3. Meta published_time extraction from HTML
        pub = extract_meta_published_date(
            '<html><head><meta property="article:published_time" content="2026-08-13T10:00:00Z"></head></html>'
        )
        self.assertEqual(pub, "2026-08-13")

        # 4. 7-day history hard drops exact (product, version) and GitHub repo root
        with tempfile.TemporaryDirectory() as tmp:
            hs = SentHistoryStore(Path(tmp) / "sent_history.json")
            art1 = Article(
                source_id="claude_code_releases",
                source_name="Claude Code",
                category="harness",
                title="v2.1.330",
                url="https://github.com/anthropics/claude-code/releases/tag/v2.1.330",
            )
            art2 = Article(
                source_id="hn",
                source_name="HN",
                category="geek",
                title="OpenComputer",
                url="https://github.com/diggerhq/opencomputer",
            )
            hs.record_items(
                "2026-11-21",
                [
                    CuratedItem(1, art1, [], "harness", 8.6, "Claude Code 2.1.330", "Enterprise login", "PM"),
                    CuratedItem(2, art2, [], "geek", 8.8, "OpenComputer", "Devin alt", "PM"),
                ],
            )
            urls, repos, evs, notes = hs.build_lookup_sets(within_days=7)

            # Different blog covering same Claude Code 2.1.330 -> hard drop via (entity, version)
            repost_cc = Article(
                source_id="gnews",
                source_name="Meteora",
                category="harness",
                title="Claude Code 2.1.330 Adds Enterprise Sign-In",
                url="https://meteora.example/cc-21330",
            )
            drop_cc, _ = article_matches_history(
                repost_cc, hist_urls=urls, hist_repos=repos, hist_entity_versions=evs, repo_notes=notes
            )
            self.assertTrue(drop_cc)

            # Secondary blog mentioning same OpenComputer repo -> soft flag for LLM
            blog_oc = Article(
                source_id="blog",
                source_name="TechBlog",
                category="geek",
                title="Deep dive into OpenComputer",
                url="https://techblog.example/opencomputer-review",
                summary="Check out https://github.com/diggerhq/opencomputer for details",
            )
            drop_oc, soft_note = article_matches_history(
                blog_oc, hist_urls=urls, hist_repos=repos, hist_entity_versions=evs, repo_notes=notes
            )
            self.assertFalse(drop_oc)
            self.assertIn("opencomputer", soft_note.lower())

    def test_or_gate_scoring_dynamic_sections_and_related_dedup(self) -> None:
        # Wildly fun / technical item with modest PM impact still scores high via OR-gate
        self.assertEqual(compute_or_gate_score(9.0, 9.5, 4.0), 8.3)
        self.assertGreater(compute_or_gate_score(9.5, 4.0, 4.0), 7.0)

        # Duplicate secondary source names (e.g. 2 Simon Willison posts) get distinct labels
        primary = Article("aihot", "AI Hot · The Decoder", "models", "Riemann proof", "https://the-decoder.com/1")
        r1 = Article("simon", "Simon Willison's Weblog（极客）", "geek", "Quoting Levent Alpöge", "https://simonwillison.net/1")
        r2 = Article("simon", "Simon Willison's Weblog（极客）", "geek", "Quoting Ravi Vakil", "https://simonwillison.net/2")
        pairs = format_related_links(primary, [r1, r2])
        self.assertEqual(len(pairs), 2)
        self.assertNotEqual(pairs[0][0], pairs[1][0])
        self.assertIn("Levent", pairs[0][0])
        self.assertIn("Ravi", pairs[1][0])

        # Dynamic section grouping renders in both Markdown and Feishu card
        item = CuratedItem(
            candidate_id=1,
            article=primary,
            related_articles=[r1, r2],
            track="models",
            score=8.9,
            zh_title="黎曼猜想 Lean 证明与竞争草稿",
            one_liner="两份形式化/纯人类证明相继公布",
            why_it_matters="可验证推理里程碑",
            section="🧮 AI 科研与形式化突破",
            tech_score=9.2,
            fun_score=8.8,
            impact_score=9.0,
        )
        res = CurateResult(["黎曼猜想证明引爆数学界"], [item], [], model_used="deepseek-flash")
        _, _, md = render_digest_markdown(res, run_date="2026-11-22", raw_count=50, sources_ok=20, sources_total=20)
        self.assertIn("## 🧮 AI 科研与形式化突破", md)
        card = build_digest_card(res, run_date="2026-11-22", raw_count=50, sources_ok=20, sources_total=20)
        card_str = json.dumps(card, ensure_ascii=False)
        self.assertIn("🧮 AI 科研与形式化突破", card_str)
        self.assertIn("Levent", card_str)
        self.assertIn("Ravi", card_str)

    def test_hf_spaces_fetcher(self) -> None:
        class FakeSpacesHTTP:
            def get(self, url: str, **kwargs) -> HTTPResponse:
                body = json.dumps([
                    {
                        "id": "cool-org/live-video-agent",
                        "likes": 120,
                        "trendingScore": 45,
                        "sdk": "gradio",
                        "lastModified": "2026-11-22T01:00:00Z",
                        "cardData": {"title": "Live Video Agent", "short_description": "Real-time vision demo"},
                    }
                ]).encode("utf-8")
                return HTTPResponse(200, {"content-type": "application/json"}, body, url)

        items = fetch_hf_spaces_source({"id": "hf_spaces", "name": "HF Spaces", "min_likes": 30}, FakeSpacesHTTP())
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].url, "https://huggingface.co/spaces/cool-org/live-video-agent")


if __name__ == "__main__":
    unittest.main()
