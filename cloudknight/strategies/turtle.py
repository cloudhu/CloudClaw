"""
海龟战法策略 - 趋势跟踪，让利润奔跑 [AKQuant]
核心：唐奇安通道突破 + ATR仓位管理 + 金字塔加仓
"""

import numpy as np

from .base import CloudKnightStrategy


class TurtleStrategy(CloudKnightStrategy):
    name = "turtle"
    description = "海龟战法 - 趋势跟踪，让利润奔跑"

    def __init__(self, params: dict | None = None):
        default = {
            "max_position_pct": 0.25,
            "max_units": 4,
            "unit_pct": 0.06,
            "stop_loss_atr_mult": 2,
            "entry_period": 20,
            "exit_period": 10,
            "atr_period": 20,
            "filter_ma_period": 200,
            "add_unit_atr": 0.5,
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
                self.enter_long(
                    symbol, unit_pct, f"海龟建仓: 突破{self._params['entry_period']}日高点 {entry_upper:.2f}"
                )
        else:
            pnl_pct = self.position_pnl_pct(symbol)

            # ATR 止损
            stop_loss_price = self._entry_price.get(symbol, self.close) - self._params["stop_loss_atr_mult"] * atr
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

    # ── 诊股接口 ──────────────────────────────────────
    @staticmethod
    def diagnose(closes, highs, lows, volumes, opens, indicators, fund=None, cap=None) -> dict:
        """海龟战法诊股评分：MA200过滤、唐奇安突破、ATR波动"""
        reasons, warnings = [], []
        score = 30

        cur = closes[-1]
        ma200 = indicators.get("ma200", 0)

        # MA200 过滤
        if ma200 > 0 and cur >= ma200:
            score += 20
            reasons.append(f"站上MA200(>={ma200:.2f})")
        elif ma200 > 0:
            score -= 15
            warnings.append(f"低于MA200({cur:.2f}<{ma200:.2f})")
        else:
            warnings.append("MA200无数据(需250日K线)")

        # 唐奇安通道
        dh20 = indicators.get("donchian_h20", 0)
        if dh20 > 0:
            dist_to_h = (dh20 - cur) / dh20 * 100
            if dist_to_h <= 2:
                score += 25
                reasons.append(f"接近突破({dist_to_h:.1f}%到20日高点)")
            elif dist_to_h <= 5:
                score += 15
                reasons.append(f"靠近通道上沿({dist_to_h:.1f}%)")
            elif dist_to_h <= 10:
                score += 8
            else:
                score -= 5
                warnings.append(f"远离突破({dist_to_h:.1f}%)")
        else:
            warnings.append("唐奇安通道无数据")

        # ATR 波动率
        atr = indicators.get("atr14", 0)
        if atr > 0 and cur > 0:
            atr_pct = atr / cur * 100
            if 2 <= atr_pct <= 5:
                score += 15
                reasons.append(f"ATR={atr_pct:.1f}%适度波动")
            elif atr_pct < 2:
                score += 5
                warnings.append(f"ATR={atr_pct:.1f}%低波动")
            else:
                score += 8
                reasons.append(f"ATR={atr_pct:.1f}%高波动")
        else:
            score += 5

        # 趋势强度
        ma20 = indicators.get("ma20", 0)
        ma60 = indicators.get("ma60", 0)
        if ma20 > 0 and ma60 > 0:
            if cur > ma20 > ma60:
                score += 15
                reasons.append("均线多头排列")
            elif cur > ma20:
                score += 8
            elif cur < ma60:
                score -= 10
                warnings.append("均线空头排列")

        score = max(0, min(100, score))
        match = len(reasons)
        total = 4

        if score >= 70:
            signal, rating = "buy", "建议建仓"
        elif score >= 50:
            signal, rating = "hold", "等待突破"
        else:
            signal, rating = "sell", "不适用"

        return {
            "name": "海龟战法",
            "key": "turtle",
            "score": score,
            "signal": signal,
            "rating": rating,
            "match_count": match,
            "total_conditions": total,
            "reasons": reasons,
            "warnings": warnings,
            "auto_add_threshold": 65,
        }
