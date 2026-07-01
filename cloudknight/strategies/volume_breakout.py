"""
量价突破策略 - 放量突破前高，捕获主升浪 [AKQuant]
核心：成交量激增 + 价格突破关键阻力 + 强度确认
"""

import numpy as np

from .base import CloudKnightStrategy


class VolumeBreakoutStrategy(CloudKnightStrategy):
    name = "volume_breakout"
    description = "量价突破 - 放量突破前高，捕捉启动点"

    def __init__(self, params: dict | None = None):
        default = {
            "max_position_pct": 0.25,
            "breakout_period": 20,  # 突破周期（N日高点）
            "vol_mult": 2.0,  # 放量倍数
            "vol_avg_period": 20,  # 均量周期
            "atr_period": 14,
            "stop_loss_atr_mult": 2,
            "stop_profit_pct": 0.15,
            "trailing_stop_atr": 3,  # 移动止损ATR
            "rsi_strength_min": 55,  # 建仓最低RSI（需有动量）
            "streak_confirm": 1,  # 连续突破确认天数
        }
        default.update(params or {})
        super().__init__(default)
        self._breakout_streak: dict[str, int] = {}
        self._highest_since_entry: dict[str, float] = {}

    @property
    def warmup_period(self) -> int:
        return 120

    def on_bar(self, bar):
        symbol = bar.symbol
        closes = self.get_history_closes(symbol, 120)
        highs = self.get_history_highs(symbol, 120)
        lows = self.get_history_lows(symbol, 120)
        volumes = self.get_history_volumes(symbol, 21)

        if len(closes) < 40:
            return

        bp = self._params["breakout_period"]
        vol_mult = self._params["vol_mult"]

        # 突破价格（N日最高价）
        breakout_price = float(np.max(highs[-(bp + 1) : -1])) if len(highs) >= bp + 1 else float(np.max(highs))

        # 均量
        avg_vol = (
            float(np.mean(volumes[-(self._params["vol_avg_period"] + 1) : -1]))
            if len(volumes) > self._params["vol_avg_period"]
            else 0
        )
        current_vol = volumes[-1] if len(volumes) > 0 else 0
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0

        # ATR / RSI
        atr = self._calc_atr(highs, lows, closes, self._params["atr_period"])
        rsi = self._calc_rsi(closes, 14)
        ma20 = self._calc_ma(closes, 20)

        # 突破确认
        price_breakout = self.close > breakout_price and self.close > ma20
        volume_surge = vol_ratio >= vol_mult
        strength_ok = rsi >= self._params["rsi_strength_min"]

        has_pos = self.has_position(symbol)

        if not has_pos:
            self._breakout_streak.pop(symbol, None)

            # 建仓：放量突破 + RSI 强势
            if price_breakout and volume_surge and strength_ok and atr > 0:
                streak = self._breakout_streak.get(symbol, 0) + 1
                self._breakout_streak[symbol] = streak
                if streak >= self._params["streak_confirm"]:
                    self.enter_long(
                        symbol,
                        self._params["max_position_pct"],
                        f"量价突破: 突破{bp}日高 量比{vol_ratio:.1f} RSI={rsi:.1f}",
                    )
                    self._highest_since_entry[symbol] = self.close
            else:
                self._breakout_streak.pop(symbol, None)
        else:
            pnl_pct = self.position_pnl_pct(symbol)
            entry_price = self._entry_price.get(symbol, self.close)

            # 更新持仓期间最高价
            prev_high = self._highest_since_entry.get(symbol, self.close)
            self._highest_since_entry[symbol] = max(prev_high, self.close)

            # ATR 初始止损
            if atr > 0 and self.close <= entry_price - self._params["stop_loss_atr_mult"] * atr:
                self.exit_position(symbol, f"量价止损: -{abs(pnl_pct):.1f}%")
                self._highest_since_entry.pop(symbol, None)
                return

            # 移动止损：从最高点回撤 N*ATR
            highest = self._highest_since_entry.get(symbol, self.close)
            if atr > 0 and self.close <= highest - self._params["trailing_stop_atr"] * atr:
                self.exit_position(symbol, f"量价移动止盈: +{pnl_pct:.1f}%(最高回撤)")
                self._highest_since_entry.pop(symbol, None)
                return

            # 固定止盈
            if pnl_pct >= self._params["stop_profit_pct"] * 100:
                self.exit_position(symbol, f"量价止盈: +{pnl_pct:.1f}%")
                self._highest_since_entry.pop(symbol, None)
                return

            # 缩量回落 减持
            if vol_ratio < 0.5 and self.close < self.open and pnl_pct > 3:
                self.reduce_position(symbol, 0.4, f"量价减持: 缩量回落 量比{vol_ratio:.1f}")

            # 跌破MA20 + 盈利回吐 减持
            if self.close < ma20 and pnl_pct > 5:
                self.reduce_position(symbol, 0.5, "量价减持: 跌破MA20")

            # 放量加仓：突破后续再次放量上攻
            if price_breakout and volume_surge and rsi < 70 and pnl_pct > 5:
                self.add_position(symbol, 0.10, "量价加仓: 再次放量突破")

    # ── 诊股接口 ──────────────────────────────────────
    @staticmethod
    def diagnose(closes, highs, lows, volumes, opens, indicators, fund=None, cap=None) -> dict:
        """量价突破诊股评分：突破强度、量比、RSI动量"""
        reasons, warnings = [], []
        score = 30

        cur = closes[-1]
        rsi = indicators.get("rsi", 50)
        vol_ratio = indicators.get("vol_ratio", 1)
        ma20 = indicators.get("ma20", 0)

        # 突破强度（20日高点距离）
        dh20 = indicators.get("donchian_h20", 0)
        if dh20 > 0:
            breakout_pct = (cur - dh20) / dh20 * 100
            if breakout_pct >= 0:
                score += 30
                reasons.append(f"已突破20日高点(+{breakout_pct:+.1f}%)")
            elif breakout_pct >= -2:
                score += 20
                reasons.append(f"接近突破(距高点{abs(breakout_pct):.1f}%)")
            elif breakout_pct >= -5:
                score += 10
                reasons.append(f"靠近高点(距{abs(breakout_pct):.1f}%)")
            else:
                warnings.append(f"远离高点({abs(breakout_pct):.1f}%)")
        else:
            warnings.append("无突破参考数据")

        # 量比
        if vol_ratio >= 2.0:
            score += 25
            reasons.append(f"量比{vol_ratio:.1f}大幅放量")
        elif vol_ratio >= 1.5:
            score += 15
            reasons.append(f"量比{vol_ratio:.1f}温和放量")
        elif vol_ratio >= 1.0:
            score += 5
        else:
            warnings.append(f"量比{vol_ratio:.1f}缩量")

        # RSI动量
        if 55 <= rsi <= 70:
            score += 15
            reasons.append(f"RSI={rsi:.0f}强势区间")
        elif rsi > 70:
            score += 5
            warnings.append(f"RSI={rsi:.0f}超买")
        elif rsi < 40:
            score -= 10
            warnings.append(f"RSI={rsi:.0f}弱势")

        # MA20确认
        if ma20 > 0 and cur > ma20:
            score += 10
            reasons.append("站上MA20")
        elif ma20 > 0:
            score -= 5
            warnings.append("低于MA20")

        score = max(0, min(100, score))
        match = len(reasons)
        total = 4

        if score >= 70:
            signal, rating = "buy", "建议关注突破"
        elif score >= 55:
            signal, rating = "buy", "关注"
        elif score >= 40:
            signal, rating = "hold", "观望"
        else:
            signal, rating = "sell", "回避"

        return {
            "name": "量价突破",
            "key": "volume_breakout",
            "score": score,
            "signal": signal,
            "rating": rating,
            "match_count": match,
            "total_conditions": total,
            "reasons": reasons,
            "warnings": warnings,
            "auto_add_threshold": 65,
        }
