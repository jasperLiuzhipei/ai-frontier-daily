"""External dead-man's-switch heartbeat monitoring (e.g., Healthchecks.io)."""

from __future__ import annotations

import os

from ai_daily.config import sanitize_secret
from ai_daily.http_client import HTTPClient, _redact_url


def report_healthcheck(
    ping_url: str | None = None,
    *,
    status: str = "ok",
    message: str = "",
    http: HTTPClient | None = None,
) -> tuple[bool, str]:
    """Send a success/fail/start heartbeat to an external monitor (outside GitHub Actions).

    Compatible with Healthchecks.io (`<url>`, `<url>/fail`, `<url>/start`) and similar URL monitors.
    Returns (sent_ok, detail_message). Never raises on network failure.
    """
    raw_url = sanitize_secret(ping_url or os.environ.get("HEALTHCHECK_PING_URL", ""))
    if not raw_url or not raw_url.startswith(("http://", "https://")) or "your-check-uuid" in raw_url:
        return False, "HEALTHCHECK_PING_URL 未配置，跳过外部监控上报"

    base = raw_url.rstrip("/")
    st = (status or "ok").strip().lower()
    if st == "fail":
        target = f"{base}/fail"
    elif st == "start":
        target = f"{base}/start"
    else:
        target = base

    client = http or HTTPClient(timeout=15, max_retries=2)
    body_bytes = (message or f"ai-daily status={st}").encode("utf-8")[:4000]
    try:
        resp = client.request(
            "POST",
            target,
            headers={"Content-Type": "text/plain; charset=utf-8"},
            data=body_bytes,
            timeout=15,
            max_retries=2,
            retry_on_unknown=True,
        )
        return True, f"外部监控上报成功 ({st}, HTTP {resp.status}, {_redact_url(target)})"
    except Exception as exc:
        return False, f"外部监控上报失败 ({st}, {_redact_url(target)}): {exc}"
