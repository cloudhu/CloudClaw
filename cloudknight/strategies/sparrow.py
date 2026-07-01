"""
麻雀战法策略 - 短线高频，积小胜为大胜 [AKQuant]
核心：回调支撑 + KDJ金叉 + 严格止盈止损
"""

from .base import CloudKnightStrategy


class SparrowStrategy(CloudKnightStrategy):
    name = "sparrow"
    description = "麻雀战法 - 短线高频，积小胜为大胜"

    def __init__(self, params: dict | None = None):
        default = {
            "max_position_pct": 0.25,
            "stop_loss_pct": 0.03,
            "stop_profit_pct": 0.05,
            "half_profit_pct": 0.03,
            "ma_short": 5,
            "ma_mid": 10,
            "ma_long": 20,
            "rsi_oversold": 35,
            "rsi_overbought": 70,
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
        self.get_history_volumes(symbol, 21)

        if len(closes) < 40:
            return

        # 指标计算
        self._calc_ma(closes, 5)
        self._calc_ma(closes, 10)
        ma20 = self._calc_ma(closes, 20)
        rsi = self._calc_rsi(closes, 14)
        k, d, _j = self._calc_kdj(highs, lows, closes, n=9, m1=3, m2=3)

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
                self.enter_long(symbol, self._params["max_position_pct"], f"麻雀建仓: MA20支撑 KDJ金叉 RSI={rsi:.1f}")
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

    # ── 诊股接口 ──────────────────────────────────────
    @staticmethod
    def diagnose(closes, highs, lows, volumes, opens, indicators, fund=None, cap=None) -> dict:
        """麻雀战法诊股评分：MA20支撑、KDJ金叉、RSI"""
        reasons, warnings = [], []
        score = 30

        cur = closes[-1]
        ma20 = indicators.get("ma20", 0)

        # MA20 支撑
        if ma20 > 0:
            dist = (cur - ma20) / ma20 * 100
            if -2 <= dist <= 2:
                score += 25
                reasons.append(f"紧贴MA20支撑(dist={dist:+.1f}%)")
            elif -5 <= dist <= 5:
                score += 15
                reasons.append(f"靠近MA20(dist={dist:+.1f}%)")
            elif dist > 10:
                score += 5
                warnings.append(f"远离MA20上方({dist:+.1f}%)")
            elif dist < -5:
                score -= 10
                warnings.append(f"跌破MA20({dist:+.1f}%)")
        else:
            warnings.append("MA20无数据")

        # KDJ 金叉
        k, d = indicators.get("k", 50), indicators.get("d", 50)
        kp, dp = indicators.get("k_prev", 50), indicators.get("d_prev", 50)
        if kp <= dp and k > d:
            score += 25
            reasons.append(f"KDJ金叉(K={k:.0f},D={d:.0f})")
        elif k > d:
            score += 10
            reasons.append(f"KDJ多头(K={k:.0f}>D={d:.0f})")
        elif k < d:
            score -= 5
            warnings.append(f"KDJ空头(K={k:.0f}<D={d:.0f})")

        # RSI
        rsi = indicators.get("rsi", 50)
        if 30 <= rsi <= 55:
            score += 20
            reasons.append(f"RSI={rsi:.0f}温和")
        elif 55 < rsi <= 70:
            score += 8
            reasons.append(f"RSI={rsi:.0f}偏强")
        elif rsi > 70:
            score -= 5
            warnings.append(f"RSI={rsi:.0f}超买")
        elif rsi < 30:
            score -= 10
            warnings.append(f"RSI={rsi:.0f}超卖")

        score = max(0, min(100, score))
        match = len(reasons)
        total = 3

        if score >= 70:
            signal, rating = "buy", "建议建仓"
        elif score >= 55:
            signal, rating = "buy", "关注"
        elif score >= 35:
            signal, rating = "hold", "观望"
        else:
            signal, rating = "sell", "回避"

        return {
            "name": "麻雀战法",
            "key": "sparrow",
            "score": score,
            "signal": signal,
            "rating": rating,
            "match_count": match,
            "total_conditions": total,
            "reasons": reasons,
            "warnings": warnings,
            "auto_add_threshold": 65,
        }
