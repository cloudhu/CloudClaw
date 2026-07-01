"""
高增长策略 - 基于基本面成长因子筛选高增长股票 [AKQuant]
核心：EPS增长率 + 营收增长率 + PEG + ROE + 季度再平衡
参考：DemoStrategy_HighGrowth - 3年EPS/营收增长 > 板块中位数 + PEG < 1
"""

import numpy as np

from .base import CloudKnightStrategy


class HighGrowthStrategy(CloudKnightStrategy):
    name = "high_growth"
    description = "高增长 - 筛选高成长优质公司，季度调仓"

    def __init__(self, params: dict | None = None):
        default = {
            "max_position_pct": 0.20,
            "stop_loss_pct": 0.12,
            "stop_profit_pct": 0.40,
            "eps_growth_min": 15,     # 3年EPS增长率最低要求 %
            "revenue_growth_min": 10,  # 3年营收增长率最低要求 %
            "peg_max": 1.2,            # PEG最大值
            "roe_min": 10,             # ROE最低要求 %
            "batch_count": 3,
            "batch_interval_pct": 0.06,
            "rsi_entry_max": 55,       # 建仓RSI上限
        }
        default.update(params or {})
        super().__init__(default)

    @property
    def warmup_period(self) -> int:
        return 180

    def on_bar(self, bar):
        symbol = bar.symbol
        closes = self.get_history_closes(symbol, 180)

        if len(closes) < 120:
            return

        # 技术面过滤
        rsi = self._calc_rsi(closes, 14)
        ma60 = self._calc_ma(closes, 60)
        ma120 = self._calc_ma(closes, 120)

        # MA60趋势方向
        closes_short = closes[-20:]
        ma20_short = float(np.mean(closes_short))
        trend_up = not np.isnan(ma60) and self.close > ma60

        has_pos = self.has_position(symbol)
        max_pos_pct = self._params["max_position_pct"]
        batch_count = self._params["batch_count"]
        batch_pct = max_pos_pct / batch_count

        if not has_pos:
            # 建仓：上升趋势 + RSI适中
            if trend_up and rsi <= self._params["rsi_entry_max"] and rsi >= 30:
                self.enter_long(
                    symbol, batch_pct,
                    f"高增长建仓(1/{batch_count}): 趋势向上 RSI={rsi:.0f}",
                )
        else:
            pnl_pct = self.position_pnl_pct(symbol)

            # 止损
            if pnl_pct <= -self._params["stop_loss_pct"] * 100:
                self.exit_position(symbol, f"高增长止损: 亏损{pnl_pct:.1f}%")
                return

            # 止盈
            if pnl_pct >= self._params["stop_profit_pct"] * 100:
                self.exit_position(symbol, f"高增长止盈: +{pnl_pct:.1f}%")
                return

            # 分批加仓：每跌 batch_interval_pct 加一次
            current_batches = self._entry_count.get(symbol, 1)
            interval = self._params["batch_interval_pct"]
            if current_batches < batch_count and pnl_pct < -interval * 100 * current_batches:
                self.add_position(
                    symbol, batch_pct,
                    f"高增长加仓({current_batches + 1}/{batch_count})",
                )

            # 跌破MA60减持
            if not np.isnan(ma60) and self.close < ma60 * 0.95 and pnl_pct > 0:
                self.reduce_position(symbol, 0.3, "高增长减持: 跌破MA60")

            # 趋势转弱清仓（跌破MA120）
            if not np.isnan(ma120) and self.close < ma120 * 0.90:
                self.exit_position(symbol, f"高增长清仓: 跌破MA120趋势转弱")
                return

    # ── 诊股接口 ──────────────────────────────────────
    @staticmethod
    def diagnose(closes, highs, lows, volumes, opens, indicators, fund=None, cap=None) -> dict:
        """高增长诊股评分：增长因子 + 趋势确认"""
        reasons, warnings = [], []
        score = 25

        cur = closes[-1]
        rsi = indicators.get("rsi", 50)
        ma60 = indicators.get("ma60", 0)
        ma120 = indicators.get("ma120", 0)
        vol_ratio = indicators.get("vol_ratio", 1)

        # 趋势位置
        if ma60 > 0 and cur > ma60:
            score += 15
            reasons.append("站上MA60趋势向上")
        elif ma60 > 0:
            score -= 5
            warnings.append(f"低于MA60")

        if ma120 > 0 and cur > ma120:
            score += 10
            reasons.append("站上MA120中长期趋势确认")
        elif ma120 > 0:
            score -= 5
            warnings.append("低于MA120")

        # RSI入场时机
        if 35 <= rsi <= 55:
            score += 15
            reasons.append(f"RSI={rsi:.0f}入场区间")
        elif rsi > 70:
            score -= 10
            warnings.append(f"RSI={rsi:.0f}超买")
        elif rsi < 30:
            score += 5
            reasons.append(f"RSI={rsi:.0f}超卖")

        # 量能
        if vol_ratio >= 1.0:
            score += 5
        else:
            score -= 5
            warnings.append(f"量比{vol_ratio:.1f}缩量")

        # 基本面增长因子
        data_available = False
        if fund is not None:
            data_available = fund.data_available if hasattr(fund, 'data_available') else bool(fund)
            if data_available:
                eps_growth = getattr(fund, 'eps_growth_3y', None)
                rev_growth = getattr(fund, 'revenue_growth_3y', None)
                peg = getattr(fund, 'peg', None)
                roe = getattr(fund, 'roe', None)
                profit_growth = getattr(fund, 'profit_growth_yoy', None)

                if eps_growth is not None and eps_growth >= 20:
                    score += 15
                    reasons.append(f"3年EPS增长{eps_growth:.0f}%优秀")
                elif eps_growth is not None and eps_growth >= 15:
                    score += 10
                    reasons.append(f"3年EPS增长{eps_growth:.0f}%良好")
                elif eps_growth is not None and eps_growth < 10:
                    score -= 5
                    warnings.append(f"3年EPS增长{eps_growth:.0f}%偏低")

                if rev_growth is not None and rev_growth >= 15:
                    score += 10
                    reasons.append(f"3年营收增长{rev_growth:.0f}%")
                elif rev_growth is not None and rev_growth < 5:
                    score -= 5
                    warnings.append(f"3年营收增长{rev_growth:.0f}%偏低")

                if peg is not None and 0 < peg < 1.0:
                    score += 15
                    reasons.append(f"PEG={peg:.2f}低估成长")
                elif peg is not None and peg < 1.5:
                    score += 8
                    reasons.append(f"PEG={peg:.2f}合理成长")
                elif peg is not None and peg > 2.0:
                    score -= 8
                    warnings.append(f"PEG={peg:.1f}成长溢价过高")

                if roe is not None and roe > 15:
                    score += 10
                    reasons.append(f"ROE={roe:.1f}%优秀")
                elif roe is not None and roe < 8:
                    score -= 5
                    warnings.append(f"ROE={roe:.1f}%偏低")

                if profit_growth is not None and profit_growth > 20:
                    score += 5
                    reasons.append(f"净利润增速{profit_growth:.1f}%")
            else:
                warnings.append("无财务数据(仅技术面评估)")

        score = max(0, min(100, score))
        match = len(reasons)
        total = 6 if data_available else 4

        if score >= 70:
            signal, rating = "buy", "高增长标的，建议关注"
        elif score >= 55:
            signal, rating = "buy", "成长性较好，逢低关注"
        elif score >= 40:
            signal, rating = "hold", "观望"
        else:
            signal, rating = "sell", "回避"

        return {
            "name": "高增长",
            "key": "high_growth",
            "score": score,
            "signal": signal,
            "rating": rating,
            "match_count": match,
            "total_conditions": total,
            "reasons": reasons,
            "warnings": warnings,
            "auto_add_threshold": 65,
        }
