"""Structured delivery error classes for retryability and unknown-outcome classification."""

from __future__ import annotations


class DeliveryError(RuntimeError):
    """Base error raised when sending a notification fails."""

    def __init__(
        self,
        message: str,
        *,
        channel: str = "",
        retryable: bool = False,
        outcome_unknown: bool = False,
        code: int | str | None = None,
    ) -> None:
        super().__init__(message)
        self.channel = channel
        self.retryable = retryable
        self.outcome_unknown = outcome_unknown
        self.code = code


class DeliveryRetryableError(DeliveryError):
    """Explicitly rejected before delivery; safe to retry with backoff."""

    def __init__(
        self,
        message: str,
        *,
        channel: str = "",
        code: int | str | None = None,
    ) -> None:
        super().__init__(
            message,
            channel=channel,
            retryable=True,
            outcome_unknown=False,
            code=code,
        )


class DeliveryPermanentError(DeliveryError):
    """Explicitly rejected with a non-retryable configuration/auth/payload error."""

    def __init__(
        self,
        message: str,
        *,
        channel: str = "",
        code: int | str | None = None,
    ) -> None:
        super().__init__(
            message,
            channel=channel,
            retryable=False,
            outcome_unknown=False,
            code=code,
        )


class DeliveryUnknownError(DeliveryError):
    """Timeout or mid-flight network error where the receiver may already have accepted the message."""

    def __init__(
        self,
        message: str,
        *,
        channel: str = "",
        code: int | str | None = None,
    ) -> None:
        super().__init__(
            message,
            channel=channel,
            retryable=False,
            outcome_unknown=True,
            code=code,
        )
