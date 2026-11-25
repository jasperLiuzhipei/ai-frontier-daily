"""Tests for RSS/Atom, Hacker News, HF Daily Papers, GitHub, and WebWatch fetchers."""

from __future__ import annotations

import json
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai_daily.fetchers.aihot import fetch_aihot_source, parse_aihot_markdown
from ai_daily.fetchers.github_trending import fetch_github_trending_source
from ai_daily.fetchers.hackernews import fetch_hackernews_source
from ai_daily.fetchers.hf_papers import fetch_hf_papers_source
from ai_daily.fetchers.rss import parse_feed_xml, strip_html
from ai_daily.fetchers.web_watch import parse_web_watch_html
from ai_daily.http_client import HTTPResponse


class FakeHTTP:
    def __init__(self, responses: dict[str, bytes | str | dict | list]) -> None:
        self.responses = responses

    def get(self, url: str, **kwargs) -> HTTPResponse:
        for prefix, val in self.responses.items():
            if url.startswith(prefix):
                if isinstance(val, (dict, list)):
                    body = json.dumps(val).encode("utf-8")
                    ctype = "application/json; charset=utf-8"
                elif isinstance(val, str):
                    body = val.encode("utf-8")
                    ctype = "text/xml; charset=utf-8"
                else:
                    body = val
                    ctype = "text/xml; charset=utf-8"
                return HTTPResponse(status=200, headers={"content-type": ctype}, body=body, url=url)
        raise AssertionError(f"Unexpected GET URL: {url}")


class TestFetchers(unittest.TestCase):
    def test_parse_rss_with_bare_ampersand_and_html(self) -> None:
        rss_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
          <channel>
            <title>Test Portal</title>
            <item>
              <title>Gemini & Antigravity Harness Update</title>
              <link>https://example.com/post-1?utm_source=rss</link>
              <description><![CDATA[<p>Big <b>harness</b> update &amp; eval speedups.</p>]]></description>
              <pubDate>Fri, 20 Nov 2026 10:00:00 GMT</pubDate>
            </item>
          </channel>
        </rss>"""
        src = {"id": "portal", "name": "Test Portal", "category": "portals", "tags": ["portal"]}
        articles = parse_feed_xml(rss_xml, src)
        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0].title, "Gemini & Antigravity Harness Update")
        self.assertIn("harness", articles[0].summary)
        self.assertNotIn("<b>", articles[0].summary)
        self.assertIsNotNone(articles[0].published_at)
        self.assertEqual(articles[0].published_at.year, 2026)

    def test_parse_atom_feed(self) -> None:
        atom_xml = """<?xml version="1.0" encoding="utf-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
          <title>Simon Willison</title>
          <entry>
            <title>Designing reliable coding agent harnesses</title>
            <link rel="alternate" href="https://simonwillison.net/2026/Nov/20/harness/"/>
            <updated>2026-11-20T12:30:00Z</updated>
            <summary type="html">&lt;p&gt;Context engineering tips.&lt;/p&gt;</summary>
          </entry>
        </feed>"""
        src = {"id": "simon", "name": "Simon", "category": "tech_share", "tags": ["engineering"]}
        articles = parse_feed_xml(atom_xml, src)
        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0].url, "https://simonwillison.net/2026/Nov/20/harness/")
        self.assertEqual(articles[0].summary, "Context engineering tips.")

    def test_web_watch_changelog_and_link_list(self) -> None:
        changelog_html = """
        <html><body>
          <main>
            <h2 id="v2-4">2026-11-20 Antigravity Harness v2.4</h2>
            <p>Improved tool latency by 35% and added subagent context compaction.</p>
            <h3>2026-11-12 Antigravity v2.3</h3>
            <ul><li>New browser sandbox workflow</li></ul>
          </main>
        </body></html>"""
        src = {
            "id": "antigravity_changelog",
            "name": "Google Antigravity Changelog",
            "url": "https://antigravity.google/docs/changelog",
            "mode": "changelog",
            "category": "harness",
            "tags": ["antigravity"],
        }
        items = parse_web_watch_html(changelog_html, src)
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0].url, "https://antigravity.google/docs/changelog#v2-4")
        self.assertIn("tool latency", items[0].summary)
        self.assertEqual(items[0].published_at, datetime(2026, 11, 20, tzinfo=timezone.utc))

        portal_html = """
        <html><body>
          <a href="/zh/news/10234">字节跳动 Seedance 视频生成模型迎来重大升级</a>
          <a href="/about">关于我们</a>
        </body></html>"""
        src2 = {
            "id": "aibase",
            "name": "AIbase",
            "url": "https://www.aibase.com/zh/daily",
            "mode": "link_list",
            "link_pattern": r"/zh/news/\d+",
            "category": "portals",
        }
        items2 = parse_web_watch_html(portal_html, src2)
        self.assertEqual(len(items2), 1)
        self.assertEqual(items2[0].url, "https://www.aibase.com/zh/news/10234")
        self.assertIn("Seedance", items2[0].title)

    def test_hf_hn_github_fetchers(self) -> None:
        fake = FakeHTTP(
            {
                "https://huggingface.co/api/daily_papers": [
                    {
                        "title": "HarnessBench: Evaluating Long-Horizon Coding Agents",
                        "publishedAt": "2026-11-20T08:00:00Z",
                        "paper": {
                            "id": "2611.12345",
                            "upvotes": 64,
                            "summary": "A new benchmark for coding agent harnesses.",
                            "authors": [{"name": "Alice"}],
                        },
                    }
                ],
                "https://hn.algolia.com/api/v1/search_by_date": {
                    "hits": [
                        {
                            "objectID": "999001",
                            "title": "Lessons from building an agent harness in production",
                            "url": "https://example.org/agent-harness",
                            "points": 210,
                            "num_comments": 88,
                            "created_at": "2026-11-20T14:00:00Z",
                            "author": "pg",
                        }
                    ]
                },
                "https://api.github.com/search/repositories": {
                    "items": [
                        {
                            "full_name": "acme/agent-harness-kit",
                            "html_url": "https://github.com/acme/agent-harness-kit",
                            "stargazers_count": 820,
                            "language": "Python",
                            "description": "Modular evaluation and execution harness for LLM agents",
                            "topics": ["agent", "harness"],
                            "pushed_at": "2026-11-20T15:00:00Z",
                        }
                    ]
                },
            }
        )
        hf_items = fetch_hf_papers_source(
            {"id": "hf", "name": "HF", "min_upvotes": 10}, fake
        )
        self.assertEqual(len(hf_items), 1)
        self.assertEqual(hf_items[0].url, "https://huggingface.co/papers/2611.12345")

        hn_items = fetch_hackernews_source(
            {"id": "hn", "name": "HN", "queries": ["agent harness"], "min_points": 20}, fake
        )
        self.assertEqual(len(hn_items), 1)
        self.assertEqual(hn_items[0].metrics["points"], 210)

        gh_items = fetch_github_trending_source(
            {"id": "gh", "name": "GH", "queries": ["agent harness"], "min_stars": 50}, fake
        )
        self.assertEqual(len(gh_items), 1)
        self.assertIn("agent-harness-kit", gh_items[0].title)

    def test_strip_html_truncation(self) -> None:
        text = strip_html("<script>alert(1)</script><p>Hello <b>world</b></p>", max_chars=20)
        self.assertEqual(text, "Hello world")

    def test_aihot_json_and_fallback_domain_and_markdown(self) -> None:
        class FailPrimaryHTTP(FakeHTTP):
            def get(self, url: str, **kwargs) -> HTTPResponse:
                if "aihot.news" in url:
                    raise RuntimeError("primary temporarily down")
                return super().get(url, **kwargs)

        fake = FailPrimaryHTTP(
            {
                "https://aihot.virxact.com/api/v1/items": {
                    "items": [
                        {
                            "title": "DeepSeek 发布应用全家桶 API 与推理优化",
                            "originalTitle": "DeepSeek Full Suite Application API",
                            "summary": "内置联网搜索、URL 阅读和文件解析，支持 deepseek-flash。",
                            "score": 9.1,
                            "category": "models",
                            "links": {
                                "original": "https://api-docs.deepseek.com/updates",
                                "aihot": "https://aihot.news/item/123",
                            },
                            "source": {"name": "DeepSeek Official"},
                            "publishedAt": "2026-11-20T09:00:00Z",
                        }
                    ]
                }
            }
        )
        src = {
            "id": "aihot_news",
            "name": "AI Hot",
            "url": "https://aihot.news/api/v1/items",
            "fallback_urls": ["https://aihot.virxact.com/api/v1/items"],
            "category": "portals",
            "tags": ["portal", "aihot"],
        }
        items = fetch_aihot_source(src, fake)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].url, "https://api-docs.deepseek.com/updates")
        self.assertEqual(items[0].source_name, "AI Hot · DeepSeek Official")
        self.assertEqual(items[0].metrics["aihot_score"], 9.1)

        md_items = parse_aihot_markdown(
            "- [OpenAI Codex CLI v0.9 并发沙箱提速 40%](https://github.com/openai/codex/releases/tag/v0.9)\n",
            src,
        )
        self.assertEqual(len(md_items), 1)
        self.assertIn("Codex CLI", md_items[0].title)


if __name__ == "__main__":
    unittest.main()
