"""
均线交叉策略 - 双均线金叉死叉 + EMA趋势过滤 [AKQuant]
核心：快线上穿慢线做多，慢线下穿快线平仓，辅以EMA过滤假突破
"""

import numpy as np

from .base import CloudKnightStrategy


class MACrossoverStrategy(CloudKnightStrategy):
    name = "ma_cross"
    description = "均线交叉 - 双均线金叉死叉 + EMA趋势过滤"

    def __init__(self, params: dict | None = None):
        default = {
            "max_position_pct": 0.25,
            "ma_fast": 10,
            "ma_slow": 30,
            "ema_trend": 60,  # 趋势过滤EMA
            "stop_loss_atr_mult": 2.5,
            "atr_period": 14,
            "stop_profit_atr_mult": 5,  # 止盈ATR倍数
            "rsi_min": 30,  # 建仓最低RSI
            "rsi_max": 70,  # 建仓最高RSI
            "volume_confirm": 1.2,  # 量能确认倍数
            "consecutive_confirm": 1,  # 连续确认K线数
        }
        default.update(params or {})
        super().__init__(default)
        self._cross_confirmed: dict[str, int] = {}  # symbol -> 连续确认计数

    @property
    def warmup_period(self) -> int:
        return 150

    def on_bar(self, bar):
        symbol = bar.symbol
        closes = self.get_history_closes(symbol, 150)
        highs = self.get_history_highs(symbol, 60)
        lows = self.get_history_lows(symbol, 60)
        volumes = self.get_history_volumes(symbol, 6)

        if len(closes) < 80:
            return

        # 计算指标
        ma_fast = self._calc_ma(closes, self._params["ma_fast"])
        ma_slow = self._calc_ma(closes, self._params["ma_slow"])
        ema_trend = self._calc_ema(closes, self._params["ema_trend"])[-1]
        rsi = self._calc_rsi(closes, 14)
        atr = self._calc_atr(highs, lows, closes, self._params["atr_period"])

        # 计算前一K线的均线值（判断交叉）
        prev_closes = closes[:-1]
        prev_fast = self._calc_ma(prev_closes, self._params["ma_fast"])
        prev_slow = self._calc_ma(prev_closes, self._params["ma_slow"])

        # 交叉信号
        golden_cross = prev_fast <= prev_slow and ma_fast > ma_slow  # 金叉
        death_cross = prev_fast >= prev_slow and ma_fast < ma_slow  # 死叉

        # 趋势过滤
        above_ema_trend = self.close > ema_trend
        strong_uptrend = ma_fast > ma_slow > ema_trend

        # 量能确认
        vol_ratio = volumes[-1] / np.mean(volumes[:-1]) if len(volumes) >= 2 and np.mean(volumes[:-1]) > 0 else 1.0

        has_pos = self.has_position(symbol)

        if not has_pos:
            # 重置确认计数
            self._cross_confirmed.pop(symbol, None)

            # 建仓条件：金叉 + EMA趋势上方 + RSI合理 + 量能确认
            if (
                golden_cross
                and above_ema_trend
                and self._params["rsi_min"] < rsi < self._params["rsi_max"]
                and vol_ratio >= self._params["volume_confirm"]
            ):
                self.enter_long(
                    symbol,
                    self._params["max_position_pct"],
                    f"均线金叉: MA{self._params['ma_fast']}↑MA{self._params['ma_slow']} RSI={rsi:.1f} 量比{vol_ratio:.1f}",
                )
        else:
            pnl_pct = self.position_pnl_pct(symbol)
            entry_price = self._entry_price.get(symbol, self.close)

            # ATR止损
            if atr > 0 and self.close <= entry_price - self._params["stop_loss_atr_mult"] * atr:
                self.exit_position(symbol, f"均线止损: -{abs(pnl_pct):.1f}%")
                return

            # ATR止盈
            if atr > 0 and pnl_pct > 0 and self.close >= entry_price + self._params["stop_profit_atr_mult"] * atr:
                self.exit_position(symbol, f"均线止盈: +{pnl_pct:.1f}%")
                return

            # 死叉清仓
            if death_cross:
                self.exit_position(symbol, f"均线死叉: MA{self._params['ma_fast']}↓MA{self._params['ma_slow']}")
                return

            # 跌破趋势EMA减持
            if not above_ema_trend and pnl_pct > 0:
                self.reduce_position(symbol, 0.5, "均线减持: 跌破趋势EMA")

            # 跌破EMA且亏损 清仓
            if not above_ema_trend and pnl_pct < 0:
                self.exit_position(symbol, "均线清仓: 跌破EMA+亏损")
                return

            # 趋势加速加仓
            if strong_uptrend and rsi < 60 and pnl_pct > 3:
                self.add_position(symbol, 0.10, "均线加仓: 趋势加速")

    # ── 诊股接口 ──────────────────────────────────────
    @staticmethod
    def diagnose(closes, highs, lows, volumes, opens, indicators, fund=None, cap=None) -> dict:
        """均线交叉诊股评分：MA交叉信号、趋势EMA位置、RSI"""
        reasons, warnings = [], []
        score = 30

        cur = closes[-1]
        ma10 = indicators.get("ma10", 0)
        ma30 = sum(closes[-30:]) / 30 if len(closes) >= 30 else 0  # 慢线
        ma60 = indicators.get("ma60", 0)  # 趋势EMA替代
        rsi = indicators.get("rsi", 50)
        vol_ratio = indicators.get("vol_ratio", 1)

        # MA交叉信号
        if ma10 > 0 and ma30 > 0:
            if ma10 > ma30:
                score += 25
                reasons.append("MA10>MA30，多头排列")
            elif ma10 < ma30:
                score -= 10
                warnings.append("MA10<MA30，空头排列")

            # 金叉/死叉状态
            prev_ma10 = sum(closes[-11:-1]) / 10 if len(closes) >= 11 else ma10
            prev_ma30_val = sum(closes[-31:-1]) / 30 if len(closes) >= 31 else ma30
            if prev_ma10 <= prev_ma30_val and ma10 > ma30:
                score += 15
                reasons.append("刚发生金叉")
            elif prev_ma10 >= prev_ma30_val and ma10 < ma30:
                score -= 10
                warnings.append("刚发生死叉")
        else:
            warnings.append("均线数据不足")

        # 趋势EMA过滤
        if ma60 > 0:
            if cur > ma60:
                score += 15
                reasons.append(f"站上MA60趋势均线")
            else:
                score -= 10
                warnings.append(f"低于MA60趋势线")
        else:
            score += 3

        # RSI
        if 40 <= rsi <= 65:
            score += 15
            reasons.append(f"RSI={rsi:.0f}合理区间")
        elif rsi > 75:
            score -= 5
            warnings.append(f"RSI={rsi:.0f}超买")
        elif rsi < 30:
            score -= 5
            warnings.append(f"RSI={rsi:.0f}超卖")

        # 量能
        if vol_ratio >= 1.2:
            score += 10
            reasons.append(f"量比{vol_ratio:.1f}放量确认")
        elif vol_ratio < 0.7:
            warnings.append(f"量比{vol_ratio:.1f}缩量")

        score = max(0, min(100, score))
        match = len(reasons)
        total = 4

        if score >= 70:
            signal, rating = "buy", "建议关注金叉"
        elif score >= 55:
            signal, rating = "buy", "关注"
        elif score >= 40:
            signal, rating = "hold", "观望"
        else:
            signal, rating = "sell", "回避"

        return {
            "name": "均线交叉",
            "key": "ma_cross",
            "score": score,
            "signal": signal,
            "rating": rating,
            "match_count": match,
            "total_conditions": total,
            "reasons": reasons,
            "warnings": warnings,
            "auto_add_threshold": 65,
        }
