"""
麻雀战法策略 - 积小胜为大胜，短线高频操作
核心：回调支撑 + KDJ金叉 + 严格止盈止损
"""

from typing import List, Dict, Optional
import pandas as pd
import numpy as np

from .base import BaseStrategy, StrategySignal, SignalType, Position
from ..indicators import calc_ma, calc_macd, calc_kdj, calc_rsi


class SparrowStrategy(BaseStrategy):
    name = "sparrow"
    description = "麻雀战法 - 短线高频，积小胜为大胜"

    def __init__(self, params: Dict = None):
        default_params = {"max_position_pct": 0.25, "stop_loss_pct": 0.03, "stop_profit_pct": 0.05,
                          "half_profit_pct": 0.03, "ma_short": 5, "ma_mid": 10, "ma_long": 20,
                          "rsi_oversold": 35, "rsi_overbought": 70, "min_volume_ratio": 0.8}
        default_params.update(params or {})
        super().__init__(default_params)

    def select_stocks(self, df: pd.DataFrame, market_data: Dict[str, pd.DataFrame]) -> List[str]:
        selected = []
        if df is None or df.empty:
            return selected
        for _, row in df.iterrows():
            code = str(row.get("代码", row.get("code", "")))
            if not code:
                continue
            stock_df = market_data.get(code)
            if stock_df is None or len(stock_df) < 60:
                continue
            stock_df = calc_ma(stock_df, [5, 10, 20])
            stock_df = calc_rsi(stock_df)
            latest = stock_df.iloc[-1]
            ma5, ma10, ma20 = latest.get("MA5", 0), latest.get("MA10", 0), latest.get("MA20", 0)
            rsi = latest.get("RSI", 50)
            if not (ma5 > ma10 > ma20):
                continue
            close = latest["close"]
            if not (ma10 <= close <= ma5 * 1.02):
                continue
            if not (35 <= rsi <= 65):
                continue
            selected.append(code)
        return selected[:20]

    def generate_signals(self, code: str, df: pd.DataFrame,
                         position: Optional[Position] = None) -> List[StrategySignal]:
        signals = []
        if df.empty or len(df) < 20:
            return signals
        df = calc_ma(df, [5, 10, 20])
        df = calc_macd(df)
        df = calc_kdj(df)
        df = calc_rsi(df)
        latest = df.iloc[-1]
        close, ma5, ma10, ma20 = latest["close"], latest["MA5"], latest["MA10"], latest["MA20"]
        rsi = latest["RSI"]

        if position is None or position.volume == 0:
            near_support = ma20 * 0.98 <= close <= ma20 * 1.02
            prev = df.iloc[-2]
            kdj_golden = (prev["K"] <= prev["D"]) and (latest["K"] > latest["D"])
            if near_support and kdj_golden and rsi < 55:
                stop_loss = close * (1 - self.params["stop_loss_pct"])
                stop_profit = close * (1 + self.params["stop_profit_pct"])
                signals.append(StrategySignal(signal_type=SignalType.BUY, code=code, price=close,
                    volume_pct=self.params["max_position_pct"], stop_loss_price=stop_loss,
                    stop_profit_price=stop_profit, reason=f"麻雀建仓: MA20支撑, KDJ金叉, RSI={rsi:.1f}", confidence=70))
            return signals

        profit_pct = position.profit_pct
        if profit_pct <= -self.params["stop_loss_pct"] * 100:
            signals.append(StrategySignal(signal_type=SignalType.STOP_LOSS, code=code, price=close,
                volume_pct=1.0, reason=f"麻雀止损: 亏损{profit_pct:.1f}%", confidence=99))
            return signals
        if profit_pct >= self.params["half_profit_pct"] * 100:
            signals.append(StrategySignal(signal_type=SignalType.REDUCE, code=code, price=close,
                volume_pct=0.5, reason=f"麻雀半仓止盈: 盈利{profit_pct:.1f}%", confidence=80))
        if profit_pct >= self.params["stop_profit_pct"] * 100:
            signals.append(StrategySignal(signal_type=SignalType.STOP_PROFIT, code=code, price=close,
                volume_pct=1.0, reason=f"麻雀止盈: 盈利{profit_pct:.1f}%", confidence=95))
            return signals
        if close < ma20:
            signals.append(StrategySignal(signal_type=SignalType.REDUCE, code=code, price=close,
                volume_pct=0.5, reason="麻雀减持: 跌破MA20", confidence=70))
        if close < ma20 * 0.97:
            signals.append(StrategySignal(signal_type=SignalType.SELL, code=code, price=close,
                volume_pct=1.0, reason="麻雀清仓: 有效跌破MA20", confidence=85))
        if rsi > 75:
            signals.append(StrategySignal(signal_type=SignalType.REDUCE, code=code, price=close,
                volume_pct=0.3, reason=f"麻雀减持: RSI={rsi:.1f}超买", confidence=70))
        if not signals:
            signals.append(StrategySignal(signal_type=SignalType.HOLD, code=code, price=close,
                reason=f"麻雀持有: 盈利{profit_pct:.1f}%", confidence=60))
        return signals
