"""
龙头战法策略 - 追踪市场最强龙头股 [AKQuant]
核心：涨停连板 + 高换手 + 量比 + 封板强度
"""

import numpy as np
from .base import CloudKnightStrategy


class DragonHeadStrategy(CloudKnightStrategy):
    name = "dragon_head"
    description = "龙头战法 - 追踪市场最强龙头股"

    def __init__(self, params: dict = None):
        default = {
            "max_position_pct": 0.30, "stop_loss_pct": 0.07, "stop_profit_pct": 0.20,
            "min_limit_up_days": 2, "volume_ratio_min": 1.5,
            "turnover_rate_min": 5, "turnover_rate_max": 30,
        }
        default.update(params or {})
        super().__init__(default)

    @property
    def warmup_period(self) -> int:
        return 40

    def on_bar(self, bar):
        symbol = bar.symbol
        closes = self.get_history_closes(symbol, 40)
        if len(closes) < 10:
            return

        # 计算涨跌幅序列（近似，实际由数据提供更准确）
        pct_changes = self.get_pct_changes(symbol, 20)
        volumes = self.get_history_volumes(symbol, 21)
        highs = self.get_history_highs(symbol, 5)
        lows = self.get_history_lows(symbol, 5)

        # 连板天数
        limit_days = 0
        for i in range(len(pct_changes) - 1, -1, -1):
            if pct_changes[i] >= 9.5:
                limit_days += 1
            else:
                break

        # 量比：当日量 / 5日均量
        if len(volumes) >= 6:
            avg_vol_5 = np.mean(volumes[-6:-1])
            vol_ratio = volumes[-1] / avg_vol_5 if avg_vol_5 > 0 else 1
        else:
            vol_ratio = 1

        pct_today = pct_changes[-1] if len(pct_changes) > 0 else 0
        is_limit_up = pct_today >= 9.5

        has_pos = self.has_position(symbol)

        if not has_pos:
            # 建仓条件：涨停且连板 >= min_limit_up_days
            if is_limit_up and limit_days >= self._params["min_limit_up_days"]:
                if vol_ratio >= self._params["volume_ratio_min"]:
                    self.enter_long(symbol, self._params["max_position_pct"],
                                    f"龙头建仓: 连板{limit_days}天 量比{vol_ratio:.1f}")
        else:
            pnl_pct = self.position_pnl_pct(symbol)

            # 止损
            if pnl_pct <= -self._params["stop_loss_pct"] * 100:
                self.exit_position(symbol, f"龙头止损: 亏损{pnl_pct:.1f}%")
                return

            # 止盈
            if pnl_pct >= self._params["stop_profit_pct"] * 100:
                self.exit_position(symbol, f"龙头止盈: 盈利{pnl_pct:.1f}%")
                return

            # 炸板/冲高回落 减持
            high_today = self.high
            low_today = self.low
            if high_today > self.close * 1.05 and self.close < self.open:
                self.reduce_position(symbol, 0.5, "龙头减持: 炸板/冲高回落")

            # 量比异常放大 减持
            if vol_ratio > 3 and pnl_pct > 5:
                self.reduce_position(symbol, 0.3, f"龙头减持: 量比{vol_ratio:.1f}异常放大")

            # 断板清仓
            prev_is_limit = pct_changes[-2] >= 9.5 if len(pct_changes) >= 2 else False
            if prev_is_limit and not is_limit_up and limit_days == 0:
                self.exit_position(symbol, "龙头清仓: 断板")

            # 缩量加速加仓
            if is_limit_up and vol_ratio < 0.7 and limit_days <= 5:
                self.add_position(symbol, 0.15, "龙头加仓: 缩量加速")
