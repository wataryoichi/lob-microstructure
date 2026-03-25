"""Bybit V5 API wrapper for Linear/USDT Perpetual.

Handles authentication, order management, and balance queries.
API keys loaded from .env file (never hardcoded).

Supports both Mainnet and Testnet via `testnet` flag.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

MAINNET_REST = "https://api.bybit.com"
TESTNET_REST = "https://api-testnet.bybit.com"

RECV_WINDOW = "5000"


@dataclass
class OrderResult:
    """Standardized order result."""
    success: bool
    order_id: str = ""
    order_link_id: str = ""
    error_code: str = ""
    error_msg: str = ""
    raw: dict | None = None


def _load_env(env_path: str = ".env") -> dict[str, str]:
    """Load key=value pairs from .env file."""
    env = {}
    p = Path(env_path)
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


class BybitClient:
    """Bybit V5 REST API client."""

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        testnet: bool = True,
        env_path: str = ".env",
    ):
        # Load from env if not provided
        if not api_key or not api_secret:
            env = _load_env(env_path)
            api_key = api_key or env.get("BYBIT_API_KEY", "") or os.environ.get("BYBIT_API_KEY", "")
            api_secret = api_secret or env.get("BYBIT_API_SECRET", "") or os.environ.get("BYBIT_API_SECRET", "")

        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = TESTNET_REST if testnet else MAINNET_REST
        self.testnet = testnet

        if not self.api_key:
            logger.warning("No API key configured. Set BYBIT_API_KEY in .env or environment.")

    def _sign(self, timestamp: str, params_str: str) -> str:
        """Generate HMAC-SHA256 signature."""
        sign_payload = f"{timestamp}{self.api_key}{RECV_WINDOW}{params_str}"
        return hmac.new(
            self.api_secret.encode("utf-8"),
            sign_payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def _headers(self, timestamp: str, sign: str) -> dict:
        return {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-SIGN": sign,
            "X-BAPI-SIGN-TYPE": "2",
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": RECV_WINDOW,
            "Content-Type": "application/json",
        }

    def _get(self, endpoint: str, params: dict | None = None) -> dict:
        """Authenticated GET request."""
        ts = str(int(time.time() * 1000))
        qs = "&".join(f"{k}={v}" for k, v in sorted((params or {}).items()))
        sign = self._sign(ts, qs)
        url = f"{self.base_url}{endpoint}"
        if qs:
            url += f"?{qs}"
        resp = requests.get(url, headers=self._headers(ts, sign), timeout=10)
        return resp.json()

    def _post(self, endpoint: str, body: dict) -> dict:
        """Authenticated POST request."""
        ts = str(int(time.time() * 1000))
        body_str = json.dumps(body)
        sign = self._sign(ts, body_str)
        resp = requests.post(
            f"{self.base_url}{endpoint}",
            headers=self._headers(ts, sign),
            data=body_str,
            timeout=10,
        )
        return resp.json()

    # ------------------------------------------------------------------
    # Public API (no auth needed)
    # ------------------------------------------------------------------

    def get_server_time(self) -> dict:
        """Check API connectivity."""
        resp = requests.get(f"{self.base_url}/v5/market/time", timeout=5)
        return resp.json()

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    def get_wallet_balance(self, account_type: str = "UNIFIED") -> dict:
        """Get wallet balance."""
        return self._get("/v5/account/wallet-balance", {"accountType": account_type})

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def place_order(
        self,
        symbol: str,
        side: str,
        qty: str,
        price: str,
        order_type: str = "Limit",
        time_in_force: str = "PostOnly",
        order_link_id: str = "",
        reduce_only: bool = False,
    ) -> OrderResult:
        """Place a Limit Post-Only order.

        Args:
            symbol: e.g. "BTCUSDT"
            side: "Buy" or "Sell"
            qty: Order quantity as string
            price: Limit price as string
            order_type: "Limit" (forced)
            time_in_force: "PostOnly" (forced for Maker)
            order_link_id: Custom order ID
            reduce_only: True for closing positions

        Returns:
            OrderResult with order_id and status
        """
        body = {
            "category": "linear",
            "symbol": symbol,
            "side": side,
            "orderType": order_type,
            "qty": qty,
            "price": price,
            "timeInForce": time_in_force,
            "reduceOnly": reduce_only,
        }
        if order_link_id:
            body["orderLinkId"] = order_link_id

        try:
            resp = self._post("/v5/order/create", body)
            ret_code = resp.get("retCode", -1)
            if ret_code == 0:
                result = resp.get("result", {})
                return OrderResult(
                    success=True,
                    order_id=result.get("orderId", ""),
                    order_link_id=result.get("orderLinkId", ""),
                    raw=resp,
                )
            else:
                return OrderResult(
                    success=False,
                    error_code=str(ret_code),
                    error_msg=resp.get("retMsg", ""),
                    raw=resp,
                )
        except Exception as e:
            return OrderResult(success=False, error_msg=str(e))

    def cancel_order(self, symbol: str, order_id: str = "", order_link_id: str = "") -> OrderResult:
        """Cancel a single order."""
        body: dict = {"category": "linear", "symbol": symbol}
        if order_id:
            body["orderId"] = order_id
        elif order_link_id:
            body["orderLinkId"] = order_link_id
        else:
            return OrderResult(success=False, error_msg="Must provide orderId or orderLinkId")

        try:
            resp = self._post("/v5/order/cancel", body)
            if resp.get("retCode") == 0:
                return OrderResult(success=True, order_id=order_id, raw=resp)
            return OrderResult(success=False, error_code=str(resp.get("retCode")),
                               error_msg=resp.get("retMsg", ""), raw=resp)
        except Exception as e:
            return OrderResult(success=False, error_msg=str(e))

    def cancel_all_orders(self, symbol: str) -> OrderResult:
        """Cancel all open orders for a symbol."""
        body = {"category": "linear", "symbol": symbol}
        try:
            resp = self._post("/v5/order/cancel-all", body)
            if resp.get("retCode") == 0:
                return OrderResult(success=True, raw=resp)
            return OrderResult(success=False, error_code=str(resp.get("retCode")),
                               error_msg=resp.get("retMsg", ""), raw=resp)
        except Exception as e:
            return OrderResult(success=False, error_msg=str(e))

    def get_open_orders(self, symbol: str) -> dict:
        """Get open orders."""
        return self._get("/v5/order/realtime", {"category": "linear", "symbol": symbol})

    def get_positions(self, symbol: str) -> dict:
        """Get current positions."""
        return self._get("/v5/position/list", {"category": "linear", "symbol": symbol})
