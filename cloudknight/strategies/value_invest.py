"""
价值投资策略 - 寻找低估优质公司，长期持有 [AKQuant]
核心：估值分位 + PE/PB + 分批建仓
"""

import numpy as np
from .base import CloudKnightStrategy


class ValueInvestStrategy(CloudKnightStrategy):
    name = "value_invest"
    description = "价值投资 - 寻找低估优质公司"

    def __init__(self, params: dict = None):
        default = {
            "max_position_pct": 0.25, "stop_loss_pct": 0.15, "stop_profit_pct": 0.50,
            "pe_max": 20, "pb_max": 3, "roe_min": 15, "dividend_yield_min": 2,
            "market_cap_min": 10e9, "debt_ratio_max": 60,
            "batch_count": 3, "batch_interval_pct": 0.08,
        }
        default.update(params or {})
        super().__init__(default)

    @property
    def warmup_period(self) -> int:
        return 300

    def on_bar(self, bar):
        symbol = bar.symbol
        closes = self.get_history_closes(symbol, 300)

        if len(closes) < 250:
            return

        # 估值评分：基于 250 日价格分位
        lookback = closes[-250:]
        percentile = (lookback < self.close).sum() / 250 * 100
        val_score = 50.0
        if percentile < 20:
            val_score += 25
        elif percentile < 40:
            val_score += 15
        elif percentile > 80:
            val_score -= 20
        elif percentile > 60:
            val_score -= 10
        val_score = max(0, min(100, val_score))

        # RSI
        rsi = self._calc_rsi(closes, 14)

        # MA250
        ma250 = self._calc_ma(closes, 250)

        has_pos = self.has_position(symbol)
        max_pos_pct = self._params["max_position_pct"]
        batch_count = self._params["batch_count"]
        batch_pct = max_pos_pct / batch_count

        if not has_pos:
            # 建仓：估值评分 >= 70 且 RSI < 45
            if val_score >= 70 and rsi < 45:
                self.enter_long(symbol, batch_pct,
                                f"价值建仓(1/{batch_count}): 估值{val_score:.0f}")
        else:
            pnl_pct = self.position_pnl_pct(symbol)

            # 止损
            if pnl_pct <= -self._params["stop_loss_pct"] * 100:
                self.exit_position(symbol, f"价值止损: 亏损{pnl_pct:.1f}%")
                return

            # 止盈
            if pnl_pct >= self._params["stop_profit_pct"] * 100:
                self.exit_position(symbol, f"价值止盈: +{pnl_pct:.1f}%")
                return

            # 分批加仓：每跌 batch_interval_pct 加一次
            current_batches = self._entry_count.get(symbol, 1)
            interval = self._params["batch_interval_pct"]
            if current_batches < batch_count and pnl_pct < -interval * 100 * current_batches:
                if val_score >= 70:
                    self.add_position(symbol, batch_pct,
                                      f"价值加仓({current_batches + 1}/{batch_count})")

            # 估值偏高减持
            if val_score < 30 and pnl_pct > 20:
                self.reduce_position(symbol, 0.5, f"价值减持: 估值偏高(val={val_score:.0f})")

            # 严重高估清仓
            if val_score < 15:
                self.exit_position(symbol, f"价值清仓: 严重高估(val={val_score:.0f})")
                return

            # 跌破 250 日线减持
            if not np.isnan(ma250) and self.close < ma250 * 0.9 and pnl_pct > 0:
                self.reduce_position(symbol, 0.3, "价值减持: 跌破250日线")
