"""Notification backends (Feishu Custom App API primary, Custom Bot Webhook fallback)."""

from __future__ import annotations

import os
from typing import Any

from ai_daily.http_client import HTTPClient
from ai_daily.notifiers.errors import (
    DeliveryError,
    DeliveryPermanentError,
    DeliveryRetryableError,
    DeliveryUnknownError,
)
from ai_daily.notifiers.feishu import (
    FeishuAppNotifier,
    FeishuNotifier,
    build_digest_card,
    generate_feishu_sign,
    make_delivery_uuid,
)


def resolve_feishu_notifier(http: HTTPClient | None = None) -> FeishuAppNotifier | FeishuNotifier:
    """Prefer Feishu Custom App Message API (idempotent uuid) when configured, else Webhook."""
    app_notifier = FeishuAppNotifier(http=http)
    if app_notifier.configured:
        return app_notifier
    return FeishuNotifier(http=http)


def resolve_notifiers(
    channel: str | None = None,
    http: HTTPClient | None = None,
) -> list[Any]:
    """Select active Feishu notification backend based on NOTIFY_CHANNEL or explicit CLI flag.

    Priority in `auto` / `feishu` mode:
      Feishu App Message API (im/v1/messages + uuid) > Feishu Custom Bot Webhook.
    """
    mode = (channel or os.environ.get("NOTIFY_CHANNEL") or "auto").strip().lower()
    feishu_app = FeishuAppNotifier(http=http)
    feishu_hook = FeishuNotifier(http=http)
    feishu = feishu_app if feishu_app.configured else feishu_hook

    if mode == "feishu-app":
        return [feishu_app]
    if mode == "feishu-webhook":
        return [feishu_hook]
    return [feishu]


__all__ = [
    "DeliveryError",
    "DeliveryPermanentError",
    "DeliveryRetryableError",
    "DeliveryUnknownError",
    "FeishuAppNotifier",
    "FeishuNotifier",
    "build_digest_card",
    "generate_feishu_sign",
    "make_delivery_uuid",
    "resolve_feishu_notifier",
    "resolve_notifiers",
]
