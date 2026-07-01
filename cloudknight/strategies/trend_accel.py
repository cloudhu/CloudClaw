"""
趋势加速策略 - EMA多周期排列 + 斜率加速 + 回调买入 [AKQuant]
核心：EMA多头排列确认趋势，斜率扩张确认加速，回调支撑位入场
"""

import numpy as np

from .base import CloudKnightStrategy


class TrendAccelerationStrategy(CloudKnightStrategy):
    name = "trend_accel"
    description = "趋势加速 - EMA多周期排列 + 加速度识别 + 回调买入"

    def __init__(self, params: dict | None = None):
        default = {
            "max_position_pct": 0.25,
            "ema_fast": 8,
            "ema_mid": 21,
            "ema_slow": 55,
            "ma_filter": 120,  # 长期趋势过滤
            "atr_period": 14,
            "stop_loss_atr_mult": 2,
            "trailing_stop_atr": 3,  # 移动止损ATR
            "accel_lookback": 10,  # 加速度计算回溯
            "rsi_pullback_min": 35,  # 回调RSI下限
            "rsi_pullback_max": 55,  # 回调RSI上限
            "volume_confirm": 0.8,  # 回调缩量确认
        }
        default.update(params or {})
        super().__init__(default)

    @property
    def warmup_period(self) -> int:
        return 200

    def on_bar(self, bar):
        symbol = bar.symbol
        closes = self.get_history_closes(symbol, 200)
        highs = self.get_history_highs(symbol, 60)
        lows = self.get_history_lows(symbol, 60)
        volumes = self.get_history_volumes(symbol, 6)

        if len(closes) < 140:
            return

        # EMA 序列
        ema_fast = self._calc_ema(closes, self._params["ema_fast"])
        ema_mid = self._calc_ema(closes, self._params["ema_mid"])
        ema_slow = self._calc_ema(closes, self._params["ema_slow"])
        ma_filter = self._calc_ma(closes, self._params["ma_filter"])

        # 多头排列检查
        fast_val = ema_fast[-1]
        mid_val = ema_mid[-1]
        slow_val = ema_slow[-1]
        aligned = fast_val > mid_val > slow_val
        above_filter = self.close > ma_filter

        # 加速度：EMA间距变化率
        lookback = self._params["accel_lookback"]
        if len(ema_fast) > lookback and len(ema_mid) > lookback and len(ema_slow) > lookback:
            prev_spread = ema_fast[-lookback] - ema_slow[-lookback]
            curr_spread = ema_fast[-1] - ema_slow[-1]
            acceleration = (curr_spread - prev_spread) / abs(prev_spread) if prev_spread > 0 else 0
        else:
            acceleration = 0

        # ATR / RSI
        atr = self._calc_atr(highs, lows, closes, self._params["atr_period"])
        rsi = self._calc_rsi(closes, 14)

        # 量能
        avg_vol = np.mean(volumes[:-1]) if len(volumes) >= 2 else volumes[-1] if len(volumes) > 0 else 0
        vol_ratio = volumes[-1] / avg_vol if avg_vol > 0 else 1.0

        has_pos = self.has_position(symbol)

        # 回调检测：价格回调至 mid EMA 附近
        near_mid_ema = mid_val > 0 and abs(self.close / mid_val - 1) < 0.03
        pullback_rsi_ok = self._params["rsi_pullback_min"] < rsi < self._params["rsi_pullback_max"]
        volume_shrink = vol_ratio < self._params["volume_confirm"]

        if not has_pos:
            if not (aligned and above_filter):
                return

            # 建仓：多头排列 + 回调至中轨 + 缩量 + RSI低位
            if near_mid_ema and pullback_rsi_ok and volume_shrink:
                self.enter_long(
                    symbol,
                    self._params["max_position_pct"],
                    f"趋势加速建仓: EMA排列 回调mid RSI={rsi:.1f} {'加速' if acceleration > 0.1 else '维持'}",
                )
        else:
            pnl_pct = self.position_pnl_pct(symbol)
            entry_price = self._entry_price.get(symbol, self.close)

            # ATR止损
            if atr > 0 and self.close <= entry_price - self._params["stop_loss_atr_mult"] * atr:
                self.exit_position(symbol, f"趋势加速止损: -{abs(pnl_pct):.1f}%")
                return

            # EMA死叉清仓（快线下穿中线）
            prev_fast = ema_fast[-2] if len(ema_fast) >= 2 else fast_val
            prev_mid = ema_mid[-2] if len(ema_mid) >= 2 else mid_val
            ema_dead_cross = prev_fast >= prev_mid and fast_val < mid_val
            if ema_dead_cross:
                self.exit_position(symbol, "趋势加速清仓: EMA死叉")
                return

            # 跌破慢线清仓
            if self.close < slow_val:
                self.exit_position(symbol, "趋势加速清仓: 跌破慢EMA")
                return

            # RSI极度超买减持
            if rsi > 80 and pnl_pct > 5:
                self.reduce_position(symbol, 0.3, f"趋势加速减持: RSI={rsi:.1f}极度超买")

            # 加速度衰竭减持
            if acceleration < -0.15 and pnl_pct > 3:
                self.reduce_position(symbol, 0.4, "趋势加速减持: 加速度衰竭")

            # 回调至慢线加仓
            near_slow_ema = slow_val > 0 and abs(self.close / slow_val - 1) < 0.02
            if near_slow_ema and rsi < 45 and acceleration > 0 and pnl_pct > 0:
                self.add_position(symbol, 0.10, "趋势加速加仓: 回调慢线支撑")

    # ── 诊股接口 ──────────────────────────────────────
    @staticmethod
    def diagnose(closes, highs, lows, volumes, opens, indicators, fund=None, cap=None) -> dict:
        """趋势加速诊股评分：EMA排列、趋势强度、回调位置"""
        reasons, warnings = [], []
        score = 30

        cur = closes[-1]
        ma5 = indicators.get("ma5", 0)
        ma10 = indicators.get("ma10", 0)
        ma20 = indicators.get("ma20", 0)
        ma60 = indicators.get("ma60", 0)
        ma120 = indicators.get("ma120", 0)
        rsi = indicators.get("rsi", 50)
        vol_ratio = indicators.get("vol_ratio", 1)
        atr = indicators.get("atr14", 0)

        # EMA多头排列检查（用MA近似）
        mas = [(5, ma5), (10, ma10), (20, ma20), (60, ma60)]
        valid_mas = [(p, v) for p, v in mas if v > 0]
        aligned = len(valid_mas) >= 2 and all(
            valid_mas[i][1] >= valid_mas[i + 1][1] for i in range(len(valid_mas) - 1)
        )

        if aligned:
            score += 20
            reasons.append("均线多头排列")
            # 发散程度
            first_val = valid_mas[0][1]
            last_val = valid_mas[-1][1]
            spread = (first_val - last_val) / last_val * 100 if last_val > 0 else 0
            if spread > 5:
                score += 10
                reasons.append(f"均线发散({spread:.1f}%)趋势强劲")
            elif spread > 2:
                score += 5
        else:
            score -= 5
            warnings.append("均线未形成多头排列")

        # 长期趋势过滤
        if ma120 > 0:
            if cur > ma120:
                score += 15
                reasons.append("站上MA120长期趋势线")
            else:
                score -= 10
                warnings.append("低于MA120")
        else:
            score += 3

        # 回调位置（距MA20距离）
        if ma20 > 0:
            dist_ma20 = (cur - ma20) / ma20 * 100
            if 0 <= dist_ma20 <= 3:
                score += 20
                reasons.append(f"回调至MA20附近({dist_ma20:+.1f}%)")
            elif -2 <= dist_ma20 < 0:
                score += 15
                reasons.append(f"轻微跌破MA20({dist_ma20:+.1f}%)")
            elif 3 < dist_ma20 <= 8:
                score += 5
                reasons.append(f"MA20上方{dist_ma20:.1f}%，待回调")
            elif dist_ma20 > 15:
                score -= 5
                warnings.append(f"远离MA20上方({dist_ma20:.1f}%)")

        # RSI回调
        if 35 <= rsi <= 55:
            score += 15
            reasons.append(f"RSI={rsi:.0f}回调到位")
        elif rsi < 35:
            score += 5
            reasons.append(f"RSI={rsi:.0f}超卖")
        elif rsi > 70:
            score -= 5
            warnings.append(f"RSI={rsi:.0f}超买")

        score = max(0, min(100, score))
        match = len(reasons)
        total = 4

        if score >= 70:
            signal, rating = "buy", "趋势加速中"
        elif score >= 55:
            signal, rating = "buy", "关注回调"
        elif score >= 40:
            signal, rating = "hold", "观望"
        else:
            signal, rating = "sell", "回避"

        return {
            "name": "趋势加速",
            "key": "trend_accel",
            "score": score,
            "signal": signal,
            "rating": rating,
            "match_count": match,
            "total_conditions": total,
            "reasons": reasons,
            "warnings": warnings,
            "auto_add_threshold": 65,
        }
