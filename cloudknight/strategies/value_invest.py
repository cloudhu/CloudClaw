"""
价值投资策略 - 寻找低估优质公司，长期持有 [AKQuant]
核心：估值分位 + PE/PB + 分批建仓
"""

import numpy as np

from .base import CloudKnightStrategy


class ValueInvestStrategy(CloudKnightStrategy):
    name = "value_invest"
    description = "价值投资 - 寻找低估优质公司"

    def __init__(self, params: dict | None = None):
        default = {
            "max_position_pct": 0.25,
            "stop_loss_pct": 0.15,
            "stop_profit_pct": 0.50,
            "pe_max": 20,
            "pb_max": 3,
            "roe_min": 15,
            "dividend_yield_min": 2,
            "market_cap_min": 10e9,
            "debt_ratio_max": 60,
            "batch_count": 3,
            "batch_interval_pct": 0.08,
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
                self.enter_long(symbol, batch_pct, f"价值建仓(1/{batch_count}): 估值{val_score:.0f}")
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
            if current_batches < batch_count and pnl_pct < -interval * 100 * current_batches and val_score >= 70:
                self.add_position(symbol, batch_pct, f"价值加仓({current_batches + 1}/{batch_count})")

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

    # ── 诊股接口 ──────────────────────────────────────
    @staticmethod
    def diagnose(closes, highs, lows, volumes, opens, indicators, fund=None, cap=None) -> dict:
        """价值投资诊股评分：价格分位、估值、RSI"""
        reasons, warnings = [], []
        score = 30

        # 250日价格分位
        pct = indicators.get("price_percentile", 50)
        if pct <= 20:
            score += 30
            reasons.append(f"极度低估(250日分位{pct:.0f}%)")
        elif pct <= 40:
            score += 20
            reasons.append(f"低估(250日分位{pct:.0f}%)")
        elif pct <= 60:
            score += 5
            reasons.append(f"估值合理(250日分位{pct:.0f}%)")
        elif pct <= 80:
            score -= 10
            warnings.append(f"偏高(250日分位{pct:.0f}%)")
        else:
            score -= 20
            warnings.append(f"严重高估(250日分位{pct:.0f}%)")

        # RSI 入场时机
        rsi = indicators.get("rsi", 50)
        if rsi < 45:
            score += 20
            reasons.append(f"RSI={rsi:.0f}入场时机好")
        elif rsi < 60:
            score += 10
        else:
            score -= 5
            warnings.append(f"RSI={rsi:.0f}偏高")

        # MA250 位置
        ma250 = indicators.get("ma250", 0)
        cur = closes[-1]
        if ma250 > 0:
            dist250 = (cur - ma250) / ma250 * 100
            if dist250 < -10:
                score += 10
                reasons.append("跌破年线，超跌信号")
            elif -10 <= dist250 <= 10:
                score += 5
                reasons.append("靠近年线")
            elif dist250 > 30:
                score -= 5
                warnings.append(f"远离年线上方({dist250:.0f}%)")
        else:
            score += 3

        # 如果基本面数据可用
        data_available = False
        if fund is not None:
            data_available = fund.data_available if hasattr(fund, 'data_available') else bool(fund)
            if data_available:
                pe = getattr(fund, 'pe', None)
                pb = getattr(fund, 'pb', None)
                roe = getattr(fund, 'roe', None)
                div_yield = getattr(fund, 'dividend_yield', None)
                if pe is not None and pe < 15:
                    score += 10
                    reasons.append(f"PE={pe:.1f}低估")
                elif pe is not None and pe > 50:
                    score -= 10
                    warnings.append(f"PE={pe:.1f}高估")
                if pb is not None and pb < 1.5:
                    score += 8
                    reasons.append(f"PB={pb:.2f}破净")
                if roe is not None and roe > 15:
                    score += 7
                    reasons.append(f"ROE={roe:.1f}%优秀")
                if div_yield is not None and div_yield > 3:
                    score += 5
                    reasons.append(f"股息率{div_yield:.1f}%")
            else:
                warnings.append("无财务数据(仅技术面估值)")

        score = max(0, min(100, score))
        match = len(reasons)
        total = 4 if data_available else 3

        if score >= 70:
            signal, rating = "buy", "建议长线布局"
        elif score >= 55:
            signal, rating = "buy", "逢低关注"
        elif score >= 40:
            signal, rating = "hold", "观望"
        else:
            signal, rating = "sell", "回避"

        return {
            "name": "价值投资",
            "key": "value_invest",
            "score": score,
            "signal": signal,
            "rating": rating,
            "match_count": match,
            "total_conditions": total,
            "reasons": reasons,
            "warnings": warnings,
            "auto_add_threshold": 65,
        }
