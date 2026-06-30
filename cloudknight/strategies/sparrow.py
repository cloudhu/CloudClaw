"""
麻雀战法策略 - 短线高频，积小胜为大胜 [AKQuant]
核心：回调支撑 + KDJ金叉 + 严格止盈止损
"""

import numpy as np
from .base import CloudKnightStrategy


class SparrowStrategy(CloudKnightStrategy):
    name = "sparrow"
    description = "麻雀战法 - 短线高频，积小胜为大胜"

    def __init__(self, params: dict = None):
        default = {
            "max_position_pct": 0.25, "stop_loss_pct": 0.03, "stop_profit_pct": 0.05,
            "half_profit_pct": 0.03, "ma_short": 5, "ma_mid": 10, "ma_long": 20,
            "rsi_oversold": 35, "rsi_overbought": 70,
        }
        default.update(params or {})
        super().__init__(default)

    @property
    def warmup_period(self) -> int:
        return 80

    def on_bar(self, bar):
        symbol = bar.symbol
        closes = self.get_history_closes(symbol, 80)
        highs = self.get_history_highs(symbol, 40)
        lows = self.get_history_lows(symbol, 40)
        volumes = self.get_history_volumes(symbol, 21)

        if len(closes) < 40:
            return

        # 指标计算
        ma5 = self._calc_ma(closes, 5)
        ma10 = self._calc_ma(closes, 10)
        ma20 = self._calc_ma(closes, 20)
        rsi = self._calc_rsi(closes, 14)
        k, d, j = self._calc_kdj(highs, lows, closes, n=9, m1=3, m2=3)

        # 前一根 KDJ
        prev_closes = closes[:-1]
        prev_highs = highs[:-1] if len(highs) > 0 else highs
        prev_lows = lows[:-1] if len(lows) > 0 else lows
        prev_k, prev_d, _ = self._calc_kdj(prev_highs, prev_lows, prev_closes, n=9, m1=3, m2=3)

        kdj_golden = (prev_k <= prev_d) and (k > d)
        near_support = ma20 * 0.98 <= self.close <= ma20 * 1.02
        has_pos = self.has_position(symbol)

        if not has_pos:
            # 建仓：MA20 支撑 + KDJ金叉 + RSI温和
            if near_support and kdj_golden and rsi < 55:
                self.enter_long(symbol, self._params["max_position_pct"],
                                f"麻雀建仓: MA20支撑 KDJ金叉 RSI={rsi:.1f}")
        else:
            pnl_pct = self.position_pnl_pct(symbol)

            # 止损
            if pnl_pct <= -self._params["stop_loss_pct"] * 100:
                self.exit_position(symbol, f"麻雀止损: 亏损{pnl_pct:.1f}%")
                return

            # 半仓止盈
            if pnl_pct >= self._params["half_profit_pct"] * 100:
                self.reduce_position(symbol, 0.5, f"麻雀半仓止盈: +{pnl_pct:.1f}%")

            # 全仓止盈
            if pnl_pct >= self._params["stop_profit_pct"] * 100:
                self.exit_position(symbol, f"麻雀止盈: +{pnl_pct:.1f}%")
                return

            # 跌破 MA20 减持
            if self.close < ma20:
                self.reduce_position(symbol, 0.5, "麻雀减持: 跌破MA20")

            # 有效跌破 MA20 清仓
            if self.close < ma20 * 0.97:
                self.exit_position(symbol, "麻雀清仓: 有效跌破MA20")
                return

            # RSI 超买减持
            if rsi > 75:
                self.reduce_position(symbol, 0.3, f"麻雀减持: RSI={rsi:.1f}超买")
