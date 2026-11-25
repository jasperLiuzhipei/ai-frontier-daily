"""Lightweight standard-library HTTP client with retries, gzip, and charset handling."""

from __future__ import annotations

import gzip
import http.client
import json
import os
import re
import socket
import ssl
import time
import urllib.error
import urllib.request
import zlib
from dataclasses import dataclass
from typing import Any

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "AI-Daily/0.1 Chrome/131.0.0.0 Safari/537.36"
)


class HTTPErrorDetail(RuntimeError):
    def __init__(self, status: int, message: str, body: str = "") -> None:
        super().__init__(f"HTTP {status}: {message}")
        self.status = status
        self.body = body


class HTTPConnectError(RuntimeError):
    """Raised when a connection failed before sending the HTTP request (safe to retry)."""


class HTTPUnknownOutcomeError(RuntimeError):
    """Raised when a timeout or mid-flight disconnect leaves server-side delivery outcome unknown."""


@dataclass
class HTTPResponse:
    status: int
    headers: dict[str, str]
    body: bytes
    url: str

    def text(self, default_encoding: str = "utf-8") -> str:
        charset = _extract_charset(self.headers.get("content-type", ""))
        candidates = [c for c in [charset, default_encoding, "utf-8", "gb18030"] if c]
        for enc in candidates:
            try:
                return self.body.decode(enc)
            except (UnicodeDecodeError, LookupError):
                continue
        return self.body.decode("utf-8", errors="replace")

    def json(self) -> Any:
        return json.loads(self.text())


_FEISHU_HOOK_RE = re.compile(
    r"(https?://open\.(?:feishu\.cn|larksuite\.com)/open-apis/bot/v2/hook/)([^/?#\s]+)",
    re.IGNORECASE,
)
_HC_PING_RE = re.compile(
    r"(https?://(?:hc-ping\.com|healthchecks\.io/ping)/)([^/?#\s]+)",
    re.IGNORECASE,
)


def _redact_url(url: str) -> str:
    redacted = _FEISHU_HOOK_RE.sub(r"\1***REDACTED***", url)
    redacted = _HC_PING_RE.sub(r"\1***REDACTED***", redacted)
    return redacted


def _classify_network_error(exc: Exception, url: str) -> RuntimeError:
    safe_url = _redact_url(url)
    if isinstance(exc, urllib.error.URLError):
        reason = exc.reason
        if isinstance(reason, (socket.gaierror, ConnectionRefusedError, ssl.SSLError)):
            return HTTPConnectError(f"Connection error for {safe_url}: {reason}")
        return HTTPUnknownOutcomeError(f"Uncertain network error for {safe_url}: {reason}")
    if isinstance(exc, (socket.gaierror, ConnectionRefusedError, ssl.SSLError)):
        return HTTPConnectError(f"Connection error for {safe_url}: {exc}")
    if isinstance(
        exc,
        (TimeoutError, socket.timeout, http.client.HTTPException, ConnectionResetError, BrokenPipeError),
    ):
        return HTTPUnknownOutcomeError(f"Timeout or interrupted response for {safe_url}: {exc}")
    return HTTPUnknownOutcomeError(f"Network error for {safe_url}: {exc}")


def _extract_charset(content_type: str) -> str:
    match = re.search(r"charset=([^\s;]+)", content_type, flags=re.IGNORECASE)
    if not match:
        return ""
    return match.group(1).strip("\"'")


def _decompress(raw: bytes, encoding: str) -> bytes:
    enc = encoding.lower().strip()
    if enc == "gzip":
        return gzip.decompress(raw)
    if enc == "deflate":
        try:
            return zlib.decompress(raw)
        except zlib.error:
            return zlib.decompress(raw, -zlib.MAX_WBITS)
    return raw


class HTTPClient:
    """Zero-dependency HTTP client suitable for RSS, JSON APIs, LLM endpoints, and webhooks."""

    def __init__(
        self,
        timeout: int = 20,
        max_retries: int = 2,
        user_agent: str = DEFAULT_USER_AGENT,
    ) -> None:
        self.timeout = timeout
        self.max_retries = max_retries
        self.user_agent = user_agent
        self._ssl_ctx = ssl.create_default_context()

    def _build_opener(self) -> urllib.request.OpenerDirector:
        handlers: list[Any] = [urllib.request.HTTPSHandler(context=self._ssl_ctx)]
        proxies: dict[str, str] = {}
        for scheme, env_keys in [
            ("http", ("HTTP_PROXY", "http_proxy")),
            ("https", ("HTTPS_PROXY", "https_proxy")),
        ]:
            for key in env_keys:
                val = os.environ.get(key, "").strip()
                if val:
                    proxies[scheme] = val
                    break
        if proxies:
            handlers.append(urllib.request.ProxyHandler(proxies))
        return urllib.request.build_opener(*handlers)

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        json_body: Any | None = None,
        data: bytes | None = None,
        timeout: int | None = None,
        max_retries: int | None = None,
        retry_on_unknown: bool = True,
    ) -> HTTPResponse:
        req_headers = {
            "User-Agent": self.user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Accept": "application/json, application/xml, application/rss+xml, application/atom+xml, text/html, */*;q=0.8",
        }
        if headers:
            req_headers.update(headers)

        payload = data
        if json_body is not None:
            payload = json.dumps(json_body, ensure_ascii=False).encode("utf-8")
            req_headers.setdefault("Content-Type", "application/json; charset=utf-8")

        retries = self.max_retries if max_retries is None else max_retries
        req_timeout = self.timeout if timeout is None else timeout
        opener = self._build_opener()

        last_err: Exception | None = None
        for attempt in range(retries + 1):
            req = urllib.request.Request(
                url,
                data=payload,
                headers=req_headers,
                method=method.upper(),
            )
            try:
                with opener.open(req, timeout=req_timeout) as resp:
                    raw = resp.read()
                    resp_headers = {k.lower(): v for k, v in resp.headers.items()}
                    raw = _decompress(raw, resp_headers.get("content-encoding", ""))
                    return HTTPResponse(
                        status=getattr(resp, "status", 200),
                        headers=resp_headers,
                        body=raw,
                        url=resp.geturl(),
                    )
            except urllib.error.HTTPError as exc:
                err_body = ""
                try:
                    raw_err = exc.read()
                    err_headers = {k.lower(): v for k, v in exc.headers.items()}
                    raw_err = _decompress(raw_err, err_headers.get("content-encoding", ""))
                    err_body = raw_err.decode("utf-8", errors="replace")[:1500]
                except Exception:
                    pass
                detail = HTTPErrorDetail(exc.code, exc.reason or "HTTP error", err_body)
                retryable_statuses = (429, 500, 502, 503, 504) if retry_on_unknown else (429, 503)
                if exc.code in retryable_statuses and attempt < retries:
                    time.sleep(1.2 * (attempt + 1))
                    last_err = detail
                    continue
                raise detail from exc
            except (urllib.error.URLError, TimeoutError, http.client.HTTPException, OSError) as exc:
                classified = _classify_network_error(exc, url)
                last_err = classified
                can_retry = isinstance(classified, HTTPConnectError) or retry_on_unknown
                if can_retry and attempt < retries:
                    time.sleep(1.0 * (attempt + 1))
                    continue
                raise classified from exc

        assert last_err is not None
        raise last_err

    def get(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> HTTPResponse:
        return self.request("GET", url, headers=headers, timeout=timeout)

    def post_json(
        self,
        url: str,
        payload: Any,
        *,
        headers: dict[str, str] | None = None,
        timeout: int | None = None,
        max_retries: int | None = None,
        retry_on_unknown: bool = True,
    ) -> HTTPResponse:
        return self.request(
            "POST",
            url,
            headers=headers,
            json_body=payload,
            timeout=timeout,
            max_retries=max_retries,
            retry_on_unknown=retry_on_unknown,
        )
