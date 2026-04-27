from __future__ import annotations

from utils.runner.integrations import (
    DEFAULT_MAX_OUTBOX_ITEMS,
    DEFAULT_SNAPSHOT_INTERVAL_SECONDS,
    DEFAULT_STOP_FLUSH_SECONDS,
    DISCORD_SECRET_HEADER,
    CommandHandler,
    HttpRunnerIntegrationBridge,
    SnapshotProvider,
)


class DiscordBridge(HttpRunnerIntegrationBridge):
    """Compatibility wrapper for the local Discord bot service bridge."""

    def __init__(
        self,
        *,
        service_url: str,
        session_id: str,
        enabled: bool,
        source_dir: str = "",
        shared_secret: str = "",
        max_outbox_items: int = DEFAULT_MAX_OUTBOX_ITEMS,
        snapshot_interval_seconds: float = DEFAULT_SNAPSHOT_INTERVAL_SECONDS,
        stop_flush_seconds: float = DEFAULT_STOP_FLUSH_SECONDS,
    ) -> None:
        super().__init__(
            service_url=service_url,
            session_id=session_id,
            enabled=enabled,
            source_dir=source_dir,
            shared_secret=shared_secret,
            secret_header=DISCORD_SECRET_HEADER,
            name="discord",
            max_outbox_items=max_outbox_items,
            snapshot_interval_seconds=snapshot_interval_seconds,
            stop_flush_seconds=stop_flush_seconds,
        )


__all__ = [
    "CommandHandler",
    "DEFAULT_MAX_OUTBOX_ITEMS",
    "DEFAULT_SNAPSHOT_INTERVAL_SECONDS",
    "DEFAULT_STOP_FLUSH_SECONDS",
    "DiscordBridge",
    "SnapshotProvider",
]

