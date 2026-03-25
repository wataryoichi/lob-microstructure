"""Notification system for trade events and system alerts.

Supports:
- Discord Webhooks
- Telegram Bot API
- Console fallback (always active)

Events:
- Trade completion (with PnL)
- Risk Manager halt
- System start/stop
- Daily summary
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import requests

logger = logging.getLogger(__name__)


def _load_env(env_path: str = ".env") -> dict[str, str]:
    env = {}
    p = Path(env_path)
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


class Notifier:
    """Multi-channel notification sender."""

    def __init__(
        self,
        discord_webhook_url: str = "",
        telegram_bot_token: str = "",
        telegram_chat_id: str = "",
        env_path: str = ".env",
    ):
        env = _load_env(env_path)

        self.discord_url = (discord_webhook_url
                            or env.get("DISCORD_WEBHOOK_URL", "")
                            or os.environ.get("DISCORD_WEBHOOK_URL", ""))
        self.tg_token = (telegram_bot_token
                         or env.get("TELEGRAM_BOT_TOKEN", "")
                         or os.environ.get("TELEGRAM_BOT_TOKEN", ""))
        self.tg_chat_id = (telegram_chat_id
                           or env.get("TELEGRAM_CHAT_ID", "")
                           or os.environ.get("TELEGRAM_CHAT_ID", ""))

        self._channels: list[str] = ["console"]
        if self.discord_url:
            self._channels.append("discord")
        if self.tg_token and self.tg_chat_id:
            self._channels.append("telegram")

        logger.info(f"Notifier channels: {self._channels}")

    def send(self, message: str, level: str = "info") -> None:
        """Send notification to all configured channels."""
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
        icon = {"info": "ℹ️", "trade": "💰", "warning": "⚠️", "error": "🚨"}.get(level, "📋")
        formatted = f"{icon} [{ts}] {message}"

        # Console (always)
        logger.info(f"[NOTIFY] {formatted}")

        # Discord
        if "discord" in self._channels:
            self._send_discord(formatted)

        # Telegram
        if "telegram" in self._channels:
            self._send_telegram(formatted)

    def _send_discord(self, message: str) -> None:
        try:
            resp = requests.post(
                self.discord_url,
                json={"content": message},
                timeout=5,
            )
            if resp.status_code not in (200, 204):
                logger.warning(f"Discord webhook failed: {resp.status_code}")
        except Exception as e:
            logger.warning(f"Discord send error: {e}")

    def _send_telegram(self, message: str) -> None:
        try:
            url = f"https://api.telegram.org/bot{self.tg_token}/sendMessage"
            resp = requests.post(
                url,
                json={"chat_id": self.tg_chat_id, "text": message, "parse_mode": "HTML"},
                timeout=5,
            )
            if resp.status_code != 200:
                logger.warning(f"Telegram send failed: {resp.status_code}")
        except Exception as e:
            logger.warning(f"Telegram send error: {e}")

    # ------------------------------------------------------------------
    # Convenience methods for common events
    # ------------------------------------------------------------------

    def notify_trade(self, config_name: str, side: str, symbol: str,
                     entry_price: float, exit_price: float,
                     gross_bps: float, net_bps: float,
                     total_trades: int, total_net_bps: float) -> None:
        """Notify on trade completion."""
        result = "WIN" if net_bps > 0 else "LOSS"
        msg = (f"TRADE [{config_name}] {symbol} {side.upper()}\n"
               f"Entry: {entry_price:.2f} → Exit: {exit_price:.2f}\n"
               f"Gross: {gross_bps:+.2f} bps | Net: {net_bps:+.2f} bps ({result})\n"
               f"Total: {total_trades} trades, {total_net_bps:+.1f} bps cumul")
        self.send(msg, level="trade")

    def notify_risk_halt(self, reason: str, cooldown_s: float) -> None:
        """Notify on Risk Manager halt."""
        msg = (f"RISK HALT: {reason}\n"
               f"Trading paused for {cooldown_s/60:.0f} minutes")
        self.send(msg, level="warning")

    def notify_system_event(self, event: str, details: str = "") -> None:
        """Notify on system start/stop/error."""
        msg = f"SYSTEM: {event}"
        if details:
            msg += f"\n{details}"
        level = "error" if "error" in event.lower() else "info"
        self.send(msg, level=level)

    def notify_daily_summary(self, config_name: str, n_trades: int,
                             net_bps: float, win_rate: float,
                             max_dd_bps: float) -> None:
        """Daily performance summary."""
        msg = (f"DAILY SUMMARY [{config_name}]\n"
               f"Trades: {n_trades} | Net: {net_bps:+.1f} bps\n"
               f"Win rate: {win_rate:.1%} | Max DD: {max_dd_bps:.1f} bps")
        self.send(msg, level="info")
