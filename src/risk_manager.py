"""
Risk Manager
--------------
Calculates position size based on:
  - Account balance
  - Risk % per trade
  - Distance from entry to stop loss
  - Leverage
"""

from __future__ import annotations


class RiskManager:
    def __init__(self, cfg: dict):
        r = cfg["risk"]
        self.risk_pct = r["risk_per_trade_pct"] / 100.0
        self.leverage = r["leverage"]
        self.max_open = r["max_open_positions"]

    def position_size(
        self,
        balance: float,
        entry_price: float,
        stop_loss: float,
        contract_value: float = 1.0,
    ) -> float:
        """
        Calculate how many contracts/units to buy.

        Formula:
          risk_amount  = balance * risk_pct
          sl_distance  = |entry - stop_loss|
          size         = risk_amount / sl_distance

        If contract_value != 1, divide by contract_value for inverse contracts.
        """
        if entry_price <= 0 or stop_loss <= 0:
            return 0.0

        risk_amount = balance * self.risk_pct
        sl_distance = abs(entry_price - stop_loss)
        if sl_distance == 0:
            return 0.0

        size = risk_amount / sl_distance
        return round(size, 6)

    def margin_required(self, entry_price: float, size: float) -> float:
        """Notional / leverage = margin required."""
        return (entry_price * size) / self.leverage

    def can_open_trade(self, open_positions: int, balance: float, margin_needed: float) -> tuple[bool, str]:
        if open_positions >= self.max_open:
            return False, f"Max open positions reached ({self.max_open})"
        if margin_needed > balance * 0.95:
            return False, f"Insufficient balance (need {margin_needed:.2f}, have {balance:.2f})"
        return True, "OK"
