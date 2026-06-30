"""
价值投资策略 - 寻找低估优质公司，长期持有
核心：低PE/PB + 历史分位估值 + 分批建仓
"""

from typing import List, Dict, Optional
import pandas as pd
import numpy as np

from .base import BaseStrategy, StrategySignal, SignalType, Position
from ..indicators import calc_ma, calc_rsi


class ValueInvestStrategy(BaseStrategy):
    name = "value_invest"
    description = "价值投资 - 寻找低估优质公司"

    def __init__(self, params: Dict = None):
        default_params = {"max_position_pct": 0.25, "stop_loss_pct": 0.15, "stop_profit_pct": 0.50,
                          "pe_max": 20, "pb_max": 3, "roe_min": 15, "dividend_yield_min": 2,
                          "market_cap_min": 10e9, "debt_ratio_max": 60, "batch_count": 3, "batch_interval_pct": 0.08}
        default_params.update(params or {})
        super().__init__(default_params)

    def select_stocks(self, df: pd.DataFrame, market_data: Dict[str, pd.DataFrame]) -> List[str]:
        selected = []
        if df is None or df.empty:
            return selected
        for _, row in df.iterrows():
            code = str(row.get("代码", row.get("code", "")))
            pe = row.get("市盈率-动态", row.get("pe", 999))
            pb = row.get("市净率", row.get("pb", 999))
            market_cap = row.get("总市值", row.get("total_mv", 0))
            if pe is None or pb is None or pe <= 0 or pe > self.params["pe_max"]:
                continue
            if pb <= 0 or pb > self.params["pb_max"]:
                continue
            if market_cap < self.params["market_cap_min"]:
                continue
            selected.append(code)
        return selected[:20]

    def _calc_valuation_score(self, df: pd.DataFrame) -> float:
        score = 50.0
        if df.empty:
            return score
        if "close" in df.columns and len(df) > 250:
            close = df["close"].iloc[-1]
            close_series = df["close"]
            percentile = (close_series < close).sum() / len(close_series) * 100
            if percentile < 20:
                score += 25
            elif percentile < 40:
                score += 15
            elif percentile > 80:
                score -= 20
            elif percentile > 60:
                score -= 10
        return max(0, min(100, score))

    def generate_signals(self, code: str, df: pd.DataFrame,
                         position: Optional[Position] = None) -> List[StrategySignal]:
        signals = []
        if df.empty or len(df) < 60:
            return signals
        df = calc_ma(df, [60, 120, 250])
        df = calc_rsi(df, 14)
        latest = df.iloc[-1]
        close = latest["close"]
        rsi = latest.get("RSI", 50)
        val_score = self._calc_valuation_score(df)

        if position is None or position.volume == 0:
            if val_score >= 70 and rsi < 45:
                stop_loss = close * (1 - self.params["stop_loss_pct"])
                stop_profit = close * (1 + self.params["stop_profit_pct"])
                first_batch_pct = self.params["max_position_pct"] / self.params["batch_count"]
                signals.append(StrategySignal(signal_type=SignalType.BUY, code=code, price=close,
                    volume_pct=first_batch_pct, stop_loss_price=stop_loss, stop_profit_price=stop_profit,
                    reason=f"价值建仓({1}/{self.params['batch_count']}): 估值{val_score:.0f}", confidence=65))
            return signals

        profit_pct = position.profit_pct
        if profit_pct <= -self.params["stop_loss_pct"] * 100:
            signals.append(StrategySignal(signal_type=SignalType.STOP_LOSS, code=code, price=close,
                volume_pct=1.0, reason=f"价值止损: 亏损{profit_pct:.1f}%", confidence=85))
            return signals
        if profit_pct >= self.params["stop_profit_pct"] * 100:
            signals.append(StrategySignal(signal_type=SignalType.STOP_PROFIT, code=code, price=close,
                volume_pct=1.0, reason=f"价值止盈: 盈利{profit_pct:.1f}%", confidence=80))
            return signals

        batch_interval = self.params["batch_interval_pct"]
        current_batches = getattr(position, "batches", 1)
        if current_batches < self.params["batch_count"] and profit_pct < -batch_interval * 100 * current_batches:
            if val_score >= 70:
                batch_pct = self.params["max_position_pct"] / self.params["batch_count"]
                signals.append(StrategySignal(signal_type=SignalType.ADD, code=code, price=close,
                    volume_pct=batch_pct, reason=f"价值加仓({current_batches+1}/{self.params['batch_count']})", confidence=70))
        if val_score < 30 and profit_pct > 20:
            signals.append(StrategySignal(signal_type=SignalType.REDUCE, code=code, price=close,
                volume_pct=0.5, reason=f"价值减持: 估值偏高", confidence=70))
        if val_score < 15:
            signals.append(StrategySignal(signal_type=SignalType.SELL, code=code, price=close,
                volume_pct=1.0, reason="价值清仓: 严重高估", confidence=75))
        ma250 = latest.get("MA250", 0)
        if ma250 > 0 and close < ma250 * 0.9 and profit_pct > 0:
            signals.append(StrategySignal(signal_type=SignalType.REDUCE, code=code, price=close,
                volume_pct=0.3, reason="价值减持: 跌破250日线", confidence=65))
        if not signals:
            signals.append(StrategySignal(signal_type=SignalType.HOLD, code=code, price=close,
                reason=f"价值持有: 盈利{profit_pct:.1f}%", confidence=65))
        return signals
