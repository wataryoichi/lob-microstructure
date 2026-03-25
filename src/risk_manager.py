"""Risk Manager for live trading safety.

Hard limits that block new orders when breached:
- Max daily drawdown
- Max open positions
- Consecutive loss limit
- Max daily trade count

This module is a gatekeeper: it does NOT place orders.
It only answers "is this trade allowed?" based on current state.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    """Hard risk limits. Any breach blocks new orders."""
    max_daily_drawdown_bps: float = 100.0
    max_open_positions: int = 1
    consecutive_losses_limit: int = 5
    max_daily_trades: int = 200
    max_position_size: float = 1.0  # in units (e.g., 1 BTC contract)
    cooldown_after_halt_s: float = 3600.0  # 1 hour halt after breach


@dataclass
class RiskState:
    """Current risk state, updated after each trade."""
    # Daily tracking (resets at UTC midnight)
    current_day: str = ""
    daily_pnl_bps: float = 0.0
    daily_peak_pnl_bps: float = 0.0
    daily_drawdown_bps: float = 0.0
    daily_trade_count: int = 0

    # Consecutive losses
    consecutive_losses: int = 0

    # Open positions
    open_position_count: int = 0

    # Halt state
    halted: bool = False
    halt_reason: str = ""
    halt_until: float = 0.0  # monotonic time


class RiskManager:
    """Gatekeeper that blocks trades when risk limits are breached.

    Usage:
        rm = RiskManager(limits)
        if rm.allow_new_trade(config_name):
            # place order
        rm.on_trade_open(config_name)
        rm.on_trade_close(config_name, net_bps)
    """

    def __init__(self, limits: RiskLimits | None = None):
        self.limits = limits or RiskLimits()
        self.state = RiskState()
        self._config_states: dict[str, RiskState] = {}

    def allow_new_trade(self, config_name: str = "") -> tuple[bool, str]:
        """Check if a new trade is allowed. Returns (allowed, reason)."""
        self._maybe_reset_day()

        # Check halt
        if self.state.halted:
            if time.monotonic() < self.state.halt_until:
                return False, f"HALTED: {self.state.halt_reason}"
            else:
                # Halt expired
                self.state.halted = False
                self.state.halt_reason = ""
                logger.info("Risk halt expired, resuming trading")

        # Check daily drawdown
        if self.state.daily_drawdown_bps >= self.limits.max_daily_drawdown_bps:
            self._halt(f"daily_drawdown={self.state.daily_drawdown_bps:.1f} >= {self.limits.max_daily_drawdown_bps}")
            return False, self.state.halt_reason

        # Check open positions
        if self.state.open_position_count >= self.limits.max_open_positions:
            return False, f"max_open_positions={self.limits.max_open_positions}"

        # Check consecutive losses
        if self.state.consecutive_losses >= self.limits.consecutive_losses_limit:
            self._halt(f"consecutive_losses={self.state.consecutive_losses} >= {self.limits.consecutive_losses_limit}")
            return False, self.state.halt_reason

        # Check daily trade count
        if self.state.daily_trade_count >= self.limits.max_daily_trades:
            return False, f"max_daily_trades={self.limits.max_daily_trades}"

        return True, "ok"

    def on_trade_open(self, config_name: str = "") -> None:
        """Called when a position is opened."""
        self.state.open_position_count += 1

    def on_trade_close(self, config_name: str = "", net_bps: float = 0.0) -> None:
        """Called when a position is closed. Updates all risk state."""
        self._maybe_reset_day()

        self.state.open_position_count = max(0, self.state.open_position_count - 1)
        self.state.daily_trade_count += 1
        self.state.daily_pnl_bps += net_bps

        # Update peak and drawdown
        if self.state.daily_pnl_bps > self.state.daily_peak_pnl_bps:
            self.state.daily_peak_pnl_bps = self.state.daily_pnl_bps
        self.state.daily_drawdown_bps = self.state.daily_peak_pnl_bps - self.state.daily_pnl_bps

        # Consecutive losses
        if net_bps < 0:
            self.state.consecutive_losses += 1
        else:
            self.state.consecutive_losses = 0

        logger.info(
            f"Risk update: daily_pnl={self.state.daily_pnl_bps:+.1f}, "
            f"dd={self.state.daily_drawdown_bps:.1f}, "
            f"consec_loss={self.state.consecutive_losses}, "
            f"trades_today={self.state.daily_trade_count}"
        )

    def get_status(self) -> dict:
        """Return current risk state as dict."""
        return {
            "day": self.state.current_day,
            "daily_pnl_bps": self.state.daily_pnl_bps,
            "daily_drawdown_bps": self.state.daily_drawdown_bps,
            "daily_trades": self.state.daily_trade_count,
            "consecutive_losses": self.state.consecutive_losses,
            "open_positions": self.state.open_position_count,
            "halted": self.state.halted,
            "halt_reason": self.state.halt_reason,
        }

    def _halt(self, reason: str) -> None:
        """Halt trading."""
        self.state.halted = True
        self.state.halt_reason = reason
        self.state.halt_until = time.monotonic() + self.limits.cooldown_after_halt_s
        logger.warning(f"RISK HALT: {reason} (cooldown {self.limits.cooldown_after_halt_s}s)")

    def _maybe_reset_day(self) -> None:
        """Reset daily counters at UTC midnight."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self.state.current_day:
            if self.state.current_day:
                logger.info(
                    f"Day reset: {self.state.current_day} -> {today} "
                    f"(prev day: pnl={self.state.daily_pnl_bps:+.1f}, "
                    f"trades={self.state.daily_trade_count})"
                )
            self.state.current_day = today
            self.state.daily_pnl_bps = 0.0
            self.state.daily_peak_pnl_bps = 0.0
            self.state.daily_drawdown_bps = 0.0
            self.state.daily_trade_count = 0
            # Don't reset consecutive_losses across days (carry over)
            # Don't reset halt (must expire naturally)
