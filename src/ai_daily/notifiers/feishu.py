"""Feishu (Lark) notifiers: Custom App Message API (idempotent uuid) and Custom Bot Webhook."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import time
import urllib.parse
import uuid
from typing import Any

from ai_daily.config import sanitize_secret
from ai_daily.http_client import (
    HTTPClient,
    HTTPConnectError,
    HTTPErrorDetail,
    HTTPUnknownOutcomeError,
)
from ai_daily.models import TRACK_ORDER, CuratedItem, CurateResult, format_related_links
from ai_daily.notifiers.errors import (
    DeliveryPermanentError,
    DeliveryRetryableError,
    DeliveryUnknownError,
)

FEISHU_MAX_CARD_BYTES = 28_500  # Feishu interactive card payload limit is 30KB
FEISHU_DEFAULT_API_BASE = "https://open.feishu.cn"
TRACK_HEADING_MAP = dict(TRACK_ORDER)

# Feishu Open API & Webhook error codes that explicitly indicate transient throttling/retryability
_FEISHU_RETRYABLE_CODES = {
    11232,     # Webhook rate limit
    99991400,  # App API rate limit exceeded
    99991403,  # App API busy / try later
    2200,      # Internal server error (explicitly rejected)
}
_FEISHU_TOKEN_EXPIRED_CODES = {
    99991661,  # Missing/expired tenant_access_token
    99991663,  # Invalid/expired tenant_access_token
    99991664,  # Invalid/expired app_access_token
}


def make_delivery_uuid(
    run_date: str, channel: str, draft_sha256: str, *, nonce: str = ""
) -> str:
    """Build a deterministic 36-char UUIDv5 (<= Feishu's 50-char uuid limit) for 1-hour deduplication."""
    seed = f"ai-daily:{run_date}:{channel}:{draft_sha256}"
    if nonce:
        seed = f"{seed}:{nonce}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))


def generate_feishu_sign(secret: str, timestamp: int) -> str:
    """Compute Feishu custom bot webhook signature: Base64(HMAC-SHA256(f'{ts}\\n{secret}', ''))."""
    string_to_sign = f"{timestamp}\n{secret}"
    hmac_code = hmac.new(
        string_to_sign.encode("utf-8"),
        msg=b"",
        digestmod=hashlib.sha256,
    ).digest()
    return base64.b64encode(hmac_code).decode("utf-8")


def _parse_feishu_error_body(body: str) -> tuple[int | None, str]:
    if not body:
        return None, ""
    try:
        parsed = json.loads(body)
        if isinstance(parsed, dict):
            raw_code = parsed.get("code")
            if raw_code is None:
                raw_code = parsed.get("StatusCode")
            code = int(raw_code) if raw_code is not None else None
            msg = str(parsed.get("msg") or parsed.get("StatusMessage") or "")
            return code, msg
    except Exception:
        pass
    return None, body[:240]


def _classify_http_exception(exc: Exception, *, channel: str = "feishu") -> Exception:
    if isinstance(exc, (DeliveryRetryableError, DeliveryPermanentError, DeliveryUnknownError)):
        return exc
    if isinstance(exc, HTTPErrorDetail):
        api_code, api_msg = _parse_feishu_error_body(exc.body)
        detail = f"HTTP {exc.status}" + (f" (code={api_code}): {api_msg}" if api_code is not None else f": {api_msg or exc}")
        if api_code in _FEISHU_RETRYABLE_CODES or exc.status in (429, 503):
            return DeliveryRetryableError(f"飞书限流或服务繁忙，可重试：{detail}", channel=channel, code=api_code or exc.status)
        if exc.status in (502, 504):
            return DeliveryUnknownError(f"飞书网关超时/异常，送达结果未知：{detail}", channel=channel, code=api_code or exc.status)
        if exc.status >= 500:
            if api_code is not None:
                return DeliveryRetryableError(f"飞书服务端拒绝，可重试：{detail}", channel=channel, code=api_code)
            return DeliveryUnknownError(f"飞书服务端 5xx 异常，送达结果未知：{detail}", channel=channel, code=exc.status)
        return DeliveryPermanentError(f"飞书拒绝请求（不可重试）：{detail}", channel=channel, code=api_code or exc.status)
    if isinstance(exc, HTTPConnectError):
        return DeliveryRetryableError(f"飞书连接失败（请求未发出，可重试）：{exc}", channel=channel)
    if isinstance(exc, (HTTPUnknownOutcomeError, TimeoutError, ConnectionResetError, BrokenPipeError)):
        return DeliveryUnknownError(f"飞书请求超时或连接中断，送达结果未知：{exc}", channel=channel)
    if "timed out" in str(exc).lower() or "timeout" in str(exc).lower():
        return DeliveryUnknownError(f"飞书请求超时，送达结果未知：{exc}", channel=channel)
    return DeliveryRetryableError(f"飞书网络异常：{exc}", channel=channel)


def _convert_markdown_for_feishu(markdown: str) -> list[dict[str, Any]]:
    """Convert standard Markdown digest text into clean Feishu card elements (zero raw # headings)."""
    elements: list[dict[str, Any]] = []
    blocks: list[str] = []
    current_lines: list[str] = []

    for raw_line in markdown.splitlines():
        line = raw_line.rstrip()
        if line.strip() == "---":
            if current_lines:
                blocks.append("\n".join(current_lines).strip())
                current_lines = []
            blocks.append("---HR---")
            continue
        if line.startswith("# "):
            # Skip top-level H1 because it's already in the card header
            continue
        if line.startswith("## "):
            if current_lines:
                blocks.append("\n".join(current_lines).strip())
                current_lines = []
                blocks.append("---HR---")
            heading = line[3:].strip()
            current_lines.append(f"**{heading}**")
            continue
        if line.startswith("### "):
            heading = line[4:].strip()
            current_lines.append(f"\n**{heading}**")
            continue
        if line.startswith("> "):
            current_lines.append(f"<font color='grey'>{line[2:].strip()}</font>")
            continue
        current_lines.append(line)

    if current_lines:
        blocks.append("\n".join(current_lines).strip())

    for blk in blocks:
        if not blk:
            continue
        if blk == "---HR---":
            if elements and elements[-1].get("tag") != "hr":
                elements.append({"tag": "hr"})
        else:
            elements.append({"tag": "markdown", "content": blk})

    # Avoid leading/trailing hr
    while elements and elements[0].get("tag") == "hr":
        elements.pop(0)
    while elements and elements[-1].get("tag") == "hr":
        elements.pop()
    return elements


def build_digest_card(
    result: CurateResult,
    *,
    run_date: str,
    raw_count: int,
    sources_ok: int,
    sources_total: int,
    evolution_notes: list[str] | None = None,
) -> dict[str, Any]:
    """Build a crisp, zero-click-nesting Feishu Interactive Card directly from CurateResult."""
    elements: list[dict[str, Any]] = []
    is_degraded = bool(result.used_fallback_rule_mode and result.degrade_reason)

    if is_degraded:
        elements.append(
            {
                "tag": "markdown",
                "content": (
                    f"<font color='orange'>**⚠️ LLM 降级短报**：主备模型调用均失败，"
                    f"以下为已通过时效与 7 天去重检查的规则精选候选。</font>\n"
                    f"<font color='grey'>失败摘要：{result.degrade_reason[:180]}</font>"
                ),
            }
        )

    meta_line = (
        f"<font color='grey'>扫描信源 {sources_ok}/{sources_total} 个 · "
        f"原始候选 {raw_count} 条 · 精选 {len(result.selected)} 条 · "
        f"模型 `{result.model_used or 'N/A'}`</font>"
    )
    elements.append({"tag": "markdown", "content": meta_line})

    if result.headline_summary:
        summary_lines = ["**⚡ 今日必看速览**"]
        for h in result.headline_summary:
            summary_lines.append(f"• {h}")
        elements.append({"tag": "markdown", "content": "\n".join(summary_lines)})

    if not result.selected:
        elements.append({"tag": "hr"})
        elements.append(
            {"tag": "markdown", "content": "今日时间窗口内暂无满足阈值的新增前沿动态。"}
        )
    else:
        grouped: dict[str, list[CuratedItem]] = {}
        section_order: list[str] = []
        for item in result.selected:
            sec = item.section.strip() if item.section else TRACK_HEADING_MAP.get(item.track, "🔥 今日精选")
            if sec not in grouped:
                grouped[sec] = []
                section_order.append(sec)
            grouped[sec].append(item)

        for section_heading in section_order:
            items = grouped[section_heading]
            elements.append({"tag": "hr"})
            sec_lines = [f"**{section_heading}**"]
            for idx, item in enumerate(items, start=1):
                score_color = "red" if item.score >= 8.8 else "blue"
                dim_badge = ""
                if item.tech_score or item.fun_score or item.impact_score:
                    dim_badge = f" <font color='grey'>(技{item.tech_score:.0f}/趣{item.fun_score:.0f}/影{item.impact_score:.0f})</font>"
                sec_lines.append(
                    f"\n**{idx}.** [{item.zh_title}]({item.article.url}) "
                    f"<font color='{score_color}'>{item.score:.1f}分</font>{dim_badge}"
                )
                sec_lines.append(f"• **要点**：{item.one_liner}")
                if item.tech_depth:
                    sec_lines.append(f"• **🔬 技术/架构**：{item.tech_depth}")
                if item.why_it_matters:
                    sec_lines.append(f"• **💡 启发**：{item.why_it_matters}")
                if item.geek_action:
                    sec_lines.append(f"• **⚡ 极客入口**：{item.geek_action}")
                tag_str = " ".join(f"`{t}`" for t in item.tags[:4])
                source_meta = f"来源：{item.article.source_name}"
                if tag_str:
                    source_meta += f" ｜ {tag_str}"
                sec_lines.append(f"<font color='grey'>{source_meta}</font>")
                rel_pairs = format_related_links(item.article, item.related_articles)
                if rel_pairs:
                    rel_parts = [f"[{lbl}]({u})" for lbl, u in rel_pairs]
                    sec_lines.append(f"<font color='grey'>关联报道：</font>{'、'.join(rel_parts)}")
            elements.append({"tag": "markdown", "content": "\n".join(sec_lines)})

    footer_notes = list(evolution_notes or [])
    if result.emerging_signals:
        sig_str = "、".join(f"`{s.keyword}`" for s in result.emerging_signals[:5])
        footer_notes.append(f"新发现信号：{sig_str}")
    if result.prompt_tokens or result.completion_tokens:
        footer_notes.append(
            f"Token 消耗：输入 {result.prompt_tokens} / 输出 {result.completion_tokens}"
        )

    if footer_notes:
        elements.append({"tag": "hr"})
        note_lines = ["**🧬 系统自迭代与运行状态**"]
        for note in footer_notes:
            note_lines.append(f"• <font color='grey'>{note}</font>")
        elements.append({"tag": "markdown", "content": "\n".join(note_lines)})

    header_title = f"AI 前沿日报 · {run_date}（{len(result.selected)} 条精选）"
    if is_degraded:
        header_title = f"AI 前沿日报 · {run_date}（降级短报 · {len(result.selected)} 条）"

    return {
        "config": {"wide_screen_mode": True, "enable_forward": True},
        "header": {
            "title": {
                "tag": "plain_text",
                "content": header_title,
            },
            "template": "orange" if is_degraded else "indigo",
        },
        "elements": elements,
    }


def _trim_card_to_budget(card: dict[str, Any], max_bytes: int = FEISHU_MAX_CARD_BYTES) -> dict[str, Any]:
    """Ensure the serialized card payload stays safely under Feishu's 30KB card limit."""
    encoded = json.dumps(card, ensure_ascii=False).encode("utf-8")
    if len(encoded) <= max_bytes:
        return card

    trimmed = dict(card)
    elements = list(card.get("elements") or [])
    notice = {
        "tag": "markdown",
        "content": "<font color='grey'>⚠️ 内容超出飞书单卡片 30KB 上限，末尾已精简，完整版请查看本地/仓库 output/ 归档。</font>",
    }
    while len(elements) > 2:
        elements.pop()
        candidate = {**trimmed, "elements": [*elements, {"tag": "hr"}, notice]}
        if len(json.dumps(candidate, ensure_ascii=False).encode("utf-8")) <= max_bytes:
            return candidate
    return {**trimmed, "elements": [*elements, notice]}


def _build_markdown_card(title: str, markdown: str, *, template: str = "indigo") -> dict[str, Any]:
    clean_title = re.sub(r"^#+\s*", "", title).strip() or "AI 前沿日报"
    elements = _convert_markdown_for_feishu(markdown)
    if not elements:
        elements = [{"tag": "markdown", "content": markdown.strip() or clean_title}]
    return {
        "config": {"wide_screen_mode": True, "enable_forward": True},
        "header": {
            "title": {"tag": "plain_text", "content": clean_title[:100]},
            "template": template,
        },
        "elements": elements,
    }


class FeishuNotifier:
    """Delivers rich Interactive Cards into a Feishu/Lark group chat via Custom Bot Webhook."""

    name = "feishu"
    delivery_mode = "webhook"
    supports_idempotent_retry = False

    def __init__(
        self,
        webhook_url: str | None = None,
        secret: str | None = None,
        http: HTTPClient | None = None,
        *,
        max_delivery_retries: int = 2,
        retry_backoff_seconds: float = 1.5,
    ) -> None:
        self.webhook_url = sanitize_secret(
            webhook_url or os.environ.get("FEISHU_WEBHOOK_URL", "")
        )
        self.secret = sanitize_secret(
            secret or os.environ.get("FEISHU_WEBHOOK_SECRET", "")
        )
        self.http = http or HTTPClient(timeout=25, max_retries=0)
        self.max_delivery_retries = max(0, int(max_delivery_retries))
        self.retry_backoff_seconds = max(0.0, float(retry_backoff_seconds))

    @property
    def configured(self) -> bool:
        return bool(
            self.webhook_url
            and self.webhook_url.startswith(("http://", "https://"))
            and "your-webhook" not in self.webhook_url.lower()
        )

    def _build_payload(self, card: dict[str, Any]) -> dict[str, Any]:
        safe_card = _trim_card_to_budget(card)
        payload: dict[str, Any] = {
            "msg_type": "interactive",
            "card": safe_card,
        }
        if self.secret:
            ts = int(time.time())
            payload["timestamp"] = str(ts)
            payload["sign"] = generate_feishu_sign(self.secret, ts)
        return payload

    def send_card(
        self, card: dict[str, Any], *, idempotency_key: str | None = None
    ) -> dict[str, Any]:
        del idempotency_key  # Webhook v2 does not support client uuid deduplication
        if not self.configured:
            raise DeliveryPermanentError(
                "FEISHU_WEBHOOK_URL 未配置。请配置飞书自建应用凭证（FEISHU_APP_ID/SECRET/CHAT_ID）或自定义机器人 Webhook。",
                channel=self.name,
            )

        last_err: Exception | None = None
        for attempt in range(self.max_delivery_retries + 1):
            payload = self._build_payload(card)
            try:
                resp = self.http.post_json(
                    self.webhook_url,
                    payload,
                    max_retries=0,
                    retry_on_unknown=False,
                )
            except Exception as exc:
                classified = _classify_http_exception(exc, channel=self.name)
                last_err = classified
                if isinstance(classified, DeliveryRetryableError) and attempt < self.max_delivery_retries:
                    if self.retry_backoff_seconds > 0:
                        time.sleep(self.retry_backoff_seconds * (attempt + 1))
                    continue
                raise classified from exc

            try:
                data = resp.json()
            except Exception as exc:
                raise DeliveryUnknownError(
                    f"飞书 Webhook 返回非 JSON 响应，送达结果未知：{exc}",
                    channel=self.name,
                ) from exc

            code = data.get("code")
            status_code = data.get("StatusCode")
            eff_code = code if code is not None else status_code
            ok = (code == 0) or (code is None and status_code == 0)
            if ok:
                return data

            msg = data.get("msg") or data.get("StatusMessage") or str(data)
            if eff_code in _FEISHU_RETRYABLE_CODES:
                last_err = DeliveryRetryableError(
                    f"飞书推送限流/临时拒绝 (code={eff_code}): {msg}",
                    channel=self.name,
                    code=eff_code,
                )
                if attempt < self.max_delivery_retries:
                    if self.retry_backoff_seconds > 0:
                        time.sleep(self.retry_backoff_seconds * (attempt + 1))
                    continue
                raise last_err

            raise DeliveryPermanentError(
                f"飞书推送失败 (code={eff_code}): {msg}",
                channel=self.name,
                code=eff_code,
            )

        assert last_err is not None
        raise last_err

    def send_digest(
        self,
        result: CurateResult,
        *,
        run_date: str,
        raw_count: int,
        sources_ok: int,
        sources_total: int,
        evolution_notes: list[str] | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        card = build_digest_card(
            result,
            run_date=run_date,
            raw_count=raw_count,
            sources_ok=sources_ok,
            sources_total=sources_total,
            evolution_notes=evolution_notes,
        )
        return self.send_card(card, idempotency_key=idempotency_key)

    def send(
        self,
        title: str,
        markdown: str,
        *,
        template: str = "indigo",
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """Send arbitrary Markdown text wrapped in a clean Feishu Interactive Card."""
        card = _build_markdown_card(title, markdown, template=template)
        return self.send_card(card, idempotency_key=idempotency_key)


class FeishuAppNotifier:
    """Delivers Interactive Cards via Feishu Open API (im/v1/messages) with 1-hour uuid idempotency."""

    name = "feishu"
    delivery_mode = "app_api"
    supports_idempotent_retry = True

    def __init__(
        self,
        app_id: str | None = None,
        app_secret: str | None = None,
        receive_id: str | None = None,
        *,
        receive_id_type: str | None = None,
        api_base: str | None = None,
        http: HTTPClient | None = None,
        max_delivery_retries: int = 2,
        max_unknown_retries: int = 1,
        retry_backoff_seconds: float = 1.5,
    ) -> None:
        self.app_id = sanitize_secret(app_id or os.environ.get("FEISHU_APP_ID", ""))
        self.app_secret = sanitize_secret(app_secret or os.environ.get("FEISHU_APP_SECRET", ""))
        self.receive_id = sanitize_secret(
            receive_id
            or os.environ.get("FEISHU_CHAT_ID", "")
            or os.environ.get("FEISHU_RECEIVE_ID", "")
        )
        self.receive_id_type = (
            sanitize_secret(receive_id_type or os.environ.get("FEISHU_RECEIVE_ID_TYPE", ""))
            or "chat_id"
        )
        self.api_base = (
            sanitize_secret(api_base or os.environ.get("FEISHU_API_BASE", ""))
            or FEISHU_DEFAULT_API_BASE
        ).rstrip("/")
        self.http = http or HTTPClient(timeout=25, max_retries=0)
        self.max_delivery_retries = max(0, int(max_delivery_retries))
        self.max_unknown_retries = max(0, int(max_unknown_retries))
        self.retry_backoff_seconds = max(0.0, float(retry_backoff_seconds))
        self._cached_token: str = ""
        self._token_expires_at: float = 0.0

    @property
    def auth_configured(self) -> bool:
        return bool(
            self.app_id
            and self.app_secret
            and not self.app_id.startswith("cli_your-")
            and not self.app_secret.startswith("your-")
        )

    @property
    def configured(self) -> bool:
        return bool(
            self.auth_configured
            and self.receive_id
            and not self.receive_id.startswith("oc_your-")
        )

    def get_tenant_access_token(self, *, force_refresh: bool = False) -> str:
        if not self.auth_configured:
            raise DeliveryPermanentError(
                "FEISHU_APP_ID / FEISHU_APP_SECRET 未配置。请在飞书开放平台「凭证与基础信息」获取并填入 .env 或 GitHub Secrets。",
                channel=self.name,
            )
        now = time.monotonic()
        if not force_refresh and self._cached_token and now < self._token_expires_at:
            return self._cached_token

        url = f"{self.api_base}/open-apis/auth/v3/tenant_access_token/internal"
        payload = {"app_id": self.app_id, "app_secret": self.app_secret}
        try:
            resp = self.http.post_json(
                url,
                payload,
                timeout=20,
                max_retries=1,
                retry_on_unknown=True,
            )
        except Exception as exc:
            classified = _classify_http_exception(exc, channel=self.name)
            if isinstance(classified, DeliveryUnknownError):
                raise DeliveryRetryableError(
                    f"获取飞书 tenant_access_token 超时（未发出消息，可重试）：{exc}",
                    channel=self.name,
                ) from exc
            raise classified from exc

        data = resp.json()
        code = data.get("code")
        if code != 0 or not data.get("tenant_access_token"):
            msg = data.get("msg") or str(data)
            if code in _FEISHU_RETRYABLE_CODES:
                raise DeliveryRetryableError(
                    f"获取飞书 tenant_access_token 临时失败 (code={code}): {msg}",
                    channel=self.name,
                    code=code,
                )
            raise DeliveryPermanentError(
                f"获取飞书 tenant_access_token 失败 (code={code}): {msg}",
                channel=self.name,
                code=code,
            )

        token = str(data["tenant_access_token"]).strip()
        expire = int(data.get("expire") or 7200)
        self._cached_token = token
        self._token_expires_at = now + max(60, expire - 120)
        return token

    def list_chats(self, *, page_size: int = 50) -> list[dict[str, str]]:
        """List Feishu chats where this app bot is a member (requires im:chat:readonly)."""
        token = self.get_tenant_access_token()
        headers = {"Authorization": f"Bearer {token}"}
        chats: list[dict[str, str]] = []
        page_token = ""

        for _ in range(10):
            params: dict[str, str] = {"page_size": str(max(1, min(page_size, 100)))}
            if page_token:
                params["page_token"] = page_token
            qs = urllib.parse.urlencode(params)
            url = f"{self.api_base}/open-apis/im/v1/chats?{qs}"
            resp = self.http.get(url, headers=headers, timeout=20)
            data = resp.json()
            if data.get("code") != 0:
                raise DeliveryPermanentError(
                    f"获取飞书群列表失败 (code={data.get('code')}): {data.get('msg')}",
                    channel=self.name,
                    code=data.get("code"),
                )
            body = data.get("data") or {}
            for item in body.get("items") or []:
                if not isinstance(item, dict) or not item.get("chat_id"):
                    continue
                chats.append(
                    {
                        "chat_id": str(item.get("chat_id") or ""),
                        "name": str(item.get("name") or "(未命名群聊)"),
                        "description": str(item.get("description") or ""),
                        "chat_mode": str(item.get("chat_mode") or ""),
                    }
                )
            if not body.get("has_more") or not body.get("page_token"):
                break
            page_token = str(body["page_token"])

        return chats

    def send_card(
        self, card: dict[str, Any], *, idempotency_key: str | None = None
    ) -> dict[str, Any]:
        if not self.configured:
            raise DeliveryPermanentError(
                "飞书应用消息 API 未完整配置：需要 FEISHU_APP_ID、FEISHU_APP_SECRET 与 FEISHU_CHAT_ID（可运行 `python3 run.py feishu-chats` 自动查询群 ID）。",
                channel=self.name,
            )

        safe_card = _trim_card_to_budget(card)
        payload: dict[str, Any] = {
            "receive_id": self.receive_id,
            "msg_type": "interactive",
            "content": json.dumps(safe_card, ensure_ascii=False),
        }
        if idempotency_key:
            payload["uuid"] = str(idempotency_key).strip()[:50]

        qs = urllib.parse.urlencode({"receive_id_type": self.receive_id_type})
        url = f"{self.api_base}/open-apis/im/v1/messages?{qs}"

        retryable_used = 0
        unknown_used = 0
        token_refreshed = False

        while True:
            token = self.get_tenant_access_token(force_refresh=token_refreshed)
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json; charset=utf-8",
            }
            try:
                resp = self.http.post_json(
                    url,
                    payload,
                    headers=headers,
                    timeout=25,
                    max_retries=0,
                    retry_on_unknown=False,
                )
            except Exception as exc:
                classified = _classify_http_exception(exc, channel=self.name)
                if (
                    isinstance(exc, HTTPErrorDetail)
                    and not token_refreshed
                    and (
                        exc.status == 401
                        or _parse_feishu_error_body(exc.body)[0] in _FEISHU_TOKEN_EXPIRED_CODES
                    )
                ):
                    token_refreshed = True
                    continue
                if isinstance(classified, DeliveryRetryableError) and retryable_used < self.max_delivery_retries:
                    retryable_used += 1
                    if self.retry_backoff_seconds > 0:
                        time.sleep(self.retry_backoff_seconds * retryable_used)
                    continue
                if (
                    isinstance(classified, DeliveryUnknownError)
                    and payload.get("uuid")
                    and unknown_used < self.max_unknown_retries
                ):
                    unknown_used += 1
                    if self.retry_backoff_seconds > 0:
                        time.sleep(self.retry_backoff_seconds * unknown_used)
                    continue
                raise classified from exc

            try:
                data = resp.json()
            except Exception as exc:
                err = DeliveryUnknownError(
                    f"飞书应用 API 返回非 JSON 响应，送达结果未知：{exc}",
                    channel=self.name,
                )
                if payload.get("uuid") and unknown_used < self.max_unknown_retries:
                    unknown_used += 1
                    if self.retry_backoff_seconds > 0:
                        time.sleep(self.retry_backoff_seconds * unknown_used)
                    continue
                raise err from exc

            code = data.get("code")
            if code == 0:
                msg_data = data.get("data") or {}
                return {
                    "code": 0,
                    "message_id": str(msg_data.get("message_id") or ""),
                    "uuid": payload.get("uuid", ""),
                    "data": msg_data,
                }

            msg = str(data.get("msg") or data)
            if code in _FEISHU_TOKEN_EXPIRED_CODES and not token_refreshed:
                token_refreshed = True
                continue
            if code in _FEISHU_RETRYABLE_CODES:
                err = DeliveryRetryableError(
                    f"飞书应用 API 限流/临时拒绝 (code={code}): {msg}",
                    channel=self.name,
                    code=code,
                )
                if retryable_used < self.max_delivery_retries:
                    retryable_used += 1
                    if self.retry_backoff_seconds > 0:
                        time.sleep(self.retry_backoff_seconds * retryable_used)
                    continue
                raise err
            raise DeliveryPermanentError(
                f"飞书应用 API 发送失败 (code={code}): {msg}",
                channel=self.name,
                code=code,
            )

    def send_digest(
        self,
        result: CurateResult,
        *,
        run_date: str,
        raw_count: int,
        sources_ok: int,
        sources_total: int,
        evolution_notes: list[str] | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        card = build_digest_card(
            result,
            run_date=run_date,
            raw_count=raw_count,
            sources_ok=sources_ok,
            sources_total=sources_total,
            evolution_notes=evolution_notes,
        )
        return self.send_card(card, idempotency_key=idempotency_key)

    def send(
        self,
        title: str,
        markdown: str,
        *,
        template: str = "indigo",
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """Send arbitrary Markdown text wrapped in a clean Feishu Interactive Card."""
        card = _build_markdown_card(title, markdown, template=template)
        return self.send_card(card, idempotency_key=idempotency_key)
