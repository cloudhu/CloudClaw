"""
海龟战法策略 - 趋势跟踪，让利润奔跑 [AKQuant]
核心：唐奇安通道突破 + ATR仓位管理 + 金字塔加仓
"""

import numpy as np
from .base import CloudKnightStrategy


class TurtleStrategy(CloudKnightStrategy):
    name = "turtle"
    description = "海龟战法 - 趋势跟踪，让利润奔跑"

    def __init__(self, params: dict = None):
        default = {
            "max_position_pct": 0.25, "max_units": 4, "unit_pct": 0.06,
            "stop_loss_atr_mult": 2, "entry_period": 20, "exit_period": 10,
            "atr_period": 20, "filter_ma_period": 200, "add_unit_atr": 0.5,
        }
        default.update(params or {})
        super().__init__(default)

    @property
    def warmup_period(self) -> int:
        return 250

    def on_bar(self, bar):
        symbol = bar.symbol
        closes = self.get_history_closes(symbol, 250)
        highs = self.get_history_highs(symbol, 250)
        lows = self.get_history_lows(symbol, 250)

        if len(closes) < 100:
            return

        # MA200 过滤
        ma200 = self._calc_ma(closes, 200)
        if np.isnan(ma200) or self.close < ma200:
            return  # 不在 MA200 上方则跳过

        # ATR
        atr = self._calc_atr(highs, lows, closes, self._params["atr_period"])
        if atr <= 0:
            return

        # 唐奇安通道
        entry_upper, _ = self._calc_donchian(highs, lows, self._params["entry_period"])
        _, exit_lower = self._calc_donchian(highs, lows, self._params["exit_period"])

        has_pos = self.has_position(symbol)
        max_units = self._params["max_units"]
        unit_pct = self._params["unit_pct"]

        if not has_pos:
            # 建仓：突破 entry_period 日高点
            if not np.isnan(entry_upper) and self.close >= entry_upper:
                self.enter_long(symbol, unit_pct,
                                f"海龟建仓: 突破{self._params['entry_period']}日高点 {entry_upper:.2f}")
        else:
            pnl_pct = self.position_pnl_pct(symbol)

            # ATR 止损
            stop_loss_price = self._entry_price.get(symbol, self.close) - \
                              self._params["stop_loss_atr_mult"] * atr
            if self.close <= stop_loss_price:
                self.exit_position(symbol, f"海龟止损: 跌破{self._params['stop_loss_atr_mult']}倍ATR")
                return

            # 退出：跌破 exit_period 日低点
            if not np.isnan(exit_lower) and self.close <= exit_lower:
                self.exit_position(symbol, f"海龟退出: 跌破{self._params['exit_period']}日低点")
                return

            # 金字塔加仓
            current_units = self._entry_count.get(symbol, 1)
            if current_units < max_units:
                entry_price = self._entry_price.get(symbol, self.close)
                add_price = entry_price + self._params["add_unit_atr"] * atr
                if self.close >= add_price and pnl_pct > 0:
                    self.add_position(symbol, unit_pct, f"海龟加仓: 第{current_units + 1}单位")

            # 跌破 20 日均线减持
            ma20 = self._calc_ma(closes, 20)
            if not np.isnan(ma20) and self.close < ma20 and pnl_pct > 10:
                self.reduce_position(symbol, 0.5, "海龟减持: 跌破20日均线")
