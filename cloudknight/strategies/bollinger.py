"""
布林带策略 - 均值回归 + 波动率收缩扩张 [AKQuant]
核心：触碰下轨做多回归中轨，触碰上轨减持，配合带宽识别趋势/震荡
"""

import numpy as np

from .base import CloudKnightStrategy


class BollingerBandStrategy(CloudKnightStrategy):
    name = "bollinger"
    description = "布林带回归 - 均值回归反弹 + 带宽波动率轮动"

    def __init__(self, params: dict | None = None):
        default = {
            "max_position_pct": 0.25,
            "bb_period": 20,
            "bb_std": 2.0,
            "rsi_oversold": 30,
            "rsi_overbought": 75,
            "stop_loss_pct": 0.05,
            "stop_profit_pct": 0.10,
            "bandwidth_min": 5,  # 最小带宽百分比（避免死股）
            "bandwidth_max": 40,  # 最大带宽百分比（避免极端波动）
            "squeeze_lookback": 20,  # 挤压检测回溯期
        }
        default.update(params or {})
        super().__init__(default)

    @property
    def warmup_period(self) -> int:
        return 80

    def on_bar(self, bar):
        symbol = bar.symbol
        closes = self.get_history_closes(symbol, 80)

        if len(closes) < 30:
            return

        # 布林带计算
        period = self._params["bb_period"]
        std_mult = self._params["bb_std"]
        mid_band = self._calc_ma(closes, period)
        std = float(np.std(closes[-period:])) if len(closes) >= period else 0
        upper_band = mid_band + std_mult * std if std > 0 else self.close * 1.1
        lower_band = mid_band - std_mult * std if std > 0 else self.close * 0.9

        # %B 指标：价格在布林带中的位置 (0=下轨, 1=上轨)
        band_range = upper_band - lower_band
        if band_range > 0:
            percent_b = (self.close - lower_band) / band_range
        else:
            percent_b = 0.5

        # 带宽检测
        if mid_band > 0:
            bandwidth = (upper_band - lower_band) / mid_band * 100
        else:
            bandwidth = 0

        # RSI
        rsi = self._calc_rsi(closes, 14)

        # 量能对比
        volumes = self.get_history_volumes(symbol, 6)
        avg_vol = np.mean(volumes[:-1]) if len(volumes) >= 2 else volumes[-1] if len(volumes) > 0 else 0
        vol_ratio = volumes[-1] / avg_vol if avg_vol > 0 else 1.0

        has_pos = self.has_position(symbol)
        bw_min = self._params["bandwidth_min"]
        bw_max = self._params["bandwidth_max"]

        # 带宽过滤：太窄没波动空间，太宽风险过大
        bandwidth_ok = bw_min <= bandwidth <= bw_max

        if not has_pos:
            if not bandwidth_ok:
                return

            # 建仓：价格触及下轨 + RSI超卖 + 放量反弹确认
            touch_lower = percent_b <= 0.05
            rsi_oversold = rsi < self._params["rsi_oversold"]
            bounce_confirm = self.close > self.open and vol_ratio > 0.8

            if touch_lower and rsi_oversold and bounce_confirm:
                self.enter_long(
                    symbol,
                    self._params["max_position_pct"],
                    f"布林建仓: 触及下轨 %B={percent_b:.2f} RSI={rsi:.1f} 带宽{bw_min:.1f}%",
                )
        else:
            pnl_pct = self.position_pnl_pct(symbol)

            # 止损
            if pnl_pct <= -self._params["stop_loss_pct"] * 100:
                self.exit_position(symbol, f"布林止损: -{abs(pnl_pct):.1f}%")
                return

            # 止盈（回归中轨以上）
            if pnl_pct >= self._params["stop_profit_pct"] * 100:
                self.exit_position(symbol, f"布林止盈: +{pnl_pct:.1f}%")
                return

            # 触及上轨减持
            if percent_b >= 0.90 and pnl_pct > 3:
                self.reduce_position(symbol, 0.5, f"布林减持: 触及上轨 %B={percent_b:.2f}")

            # RSI超买 + 触及上轨 清仓
            if percent_b >= 0.95 and rsi > self._params["rsi_overbought"]:
                self.exit_position(symbol, f"布林清仓: 上轨超买 RSI={rsi:.1f}")
                return

            # 再次触及下轨 加仓（分批）
            if percent_b <= 0.10 and rsi < 40 and pnl_pct < 0:
                current_batches = self._entry_count.get(symbol, 1)
                if current_batches < 3:
                    self.add_position(symbol, 0.08, f"布林加仓: 二次探底 %B={percent_b:.2f}")

    # ── 诊股接口 ──────────────────────────────────────
    @staticmethod
    def diagnose(closes, highs, lows, volumes, opens, indicators, fund=None, cap=None) -> dict:
        """布林带回归诊股评分：%B位置、带宽、RSI"""
        reasons, warnings = [], []
        score = 30

        cur = closes[-1]
        rsi = indicators.get("rsi", 50)

        # %B 位置（价格在布林带中的位置，0=下轨, 1=上轨）
        # 从 indicators 中获取或从 MA20 + 标准差估算
        ma20 = indicators.get("ma20", 0)
        if ma20 > 0 and len(closes) >= 20:
            import numpy as np
            std = float(np.std(closes[-20:]))
            if std > 0:
                upper = ma20 + 2 * std
                lower = ma20 - 2 * std
                band_range = upper - lower
                percent_b = (cur - lower) / band_range if band_range > 0 else 0.5
                # 带宽
                bandwidth = band_range / ma20 * 100

                if percent_b <= 0.1:
                    score += 30
                    reasons.append(f"触及下轨(%B={percent_b:.2f})")
                elif percent_b <= 0.3:
                    score += 15
                    reasons.append(f"靠近下轨(%B={percent_b:.2f})")
                elif percent_b >= 0.9:
                    score -= 15
                    warnings.append(f"触及上轨(%B={percent_b:.2f})")
                elif percent_b >= 0.7:
                    score -= 5
                    warnings.append(f"偏向上轨(%B={percent_b:.2f})")

                if 5 <= bandwidth <= 40:
                    score += 5
                    reasons.append(f"带宽{bandwidth:.1f}%适中")
                elif bandwidth < 5:
                    warnings.append(f"带宽{bandwidth:.1f}%偏窄")
                else:
                    warnings.append(f"带宽{bandwidth:.1f}%偏大")
            else:
                warnings.append("波动率为零")
        else:
            warnings.append("MA20无数据(需20日K线)")

        # RSI
        if rsi < 35:
            score += 20
            reasons.append(f"RSI={rsi:.0f}超卖区")
        elif rsi < 50:
            score += 10
            reasons.append(f"RSI={rsi:.0f}偏低位")
        elif rsi > 70:
            score -= 10
            warnings.append(f"RSI={rsi:.0f}超买区")

        # 量价
        vol_ratio = indicators.get("vol_ratio", 1)
        if vol_ratio > 1.2:
            score += 5
            reasons.append(f"量比{vol_ratio:.1f}放量")

        score = max(0, min(100, score))
        match = len(reasons)
        total = 3

        if score >= 70:
            signal, rating = "buy", "建议回归做多"
        elif score >= 55:
            signal, rating = "buy", "关注"
        elif score >= 40:
            signal, rating = "hold", "观望"
        else:
            signal, rating = "sell", "回避"

        return {
            "name": "布林带回归",
            "key": "bollinger",
            "score": score,
            "signal": signal,
            "rating": rating,
            "match_count": match,
            "total_conditions": total,
            "reasons": reasons,
            "warnings": warnings,
            "auto_add_threshold": 65,
        }
