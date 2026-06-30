"""
海龟战法策略 - 趋势跟踪，让利润奔跑
核心：唐奇安通道突破 + ATR仓位管理 + 金字塔加仓
"""

from typing import List, Dict, Optional
import pandas as pd
import numpy as np

from .base import BaseStrategy, StrategySignal, SignalType, Position
from ..indicators import calc_atr, calc_breakout


class TurtleStrategy(BaseStrategy):
    name = "turtle"
    description = "海龟战法 - 趋势跟踪，让利润奔跑"

    def __init__(self, params: Dict = None):
        default_params = {"max_position_pct": 0.25, "max_units": 4, "unit_pct": 0.06,
                          "stop_loss_atr_mult": 2, "entry_period": 20, "exit_period": 10,
                          "atr_period": 20, "filter_ma_period": 200, "add_unit_atr": 0.5}
        default_params.update(params or {})
        super().__init__(default_params)

    def select_stocks(self, df: pd.DataFrame, market_data: Dict[str, pd.DataFrame]) -> List[str]:
        selected = []
        if df is None or df.empty:
            return selected
        for _, row in df.iterrows():
            code = str(row.get("代码", row.get("code", "")))
            amount = row.get("成交额", row.get("amount", 0))
            if amount < 1e8:
                continue
            stock_df = market_data.get(code)
            if stock_df is None or len(stock_df) < 200:
                continue
            stock_df["MA200"] = stock_df["close"].rolling(200).mean()
            latest = stock_df.iloc[-1]
            if pd.isna(latest.get("MA200")) or latest["close"] < latest["MA200"]:
                continue
            upper, lower = calc_breakout(stock_df, self.params["entry_period"])
            if latest["close"] >= upper.iloc[-1]:
                selected.append(code)
        return selected[:15]

    def generate_signals(self, code: str, df: pd.DataFrame,
                         position: Optional[Position] = None) -> List[StrategySignal]:
        signals = []
        if df.empty or len(df) < 100:
            return signals
        atr = calc_atr(df, self.params["atr_period"])
        entry_upper, entry_lower = calc_breakout(df, self.params["entry_period"])
        exit_upper, exit_lower = calc_breakout(df, self.params["exit_period"])
        latest = df.iloc[-1]
        close = latest["close"]
        current_atr = atr.iloc[-1]
        current_entry_upper = entry_upper.iloc[-1]
        current_exit_lower = exit_lower.iloc[-1]

        if position is None or position.volume == 0:
            if close >= current_entry_upper and current_atr > 0:
                stop_loss = close - self.params["stop_loss_atr_mult"] * current_atr
                stop_profit = close + 5 * current_atr
                signals.append(StrategySignal(signal_type=SignalType.BUY, code=code, price=close,
                    volume_pct=self.params["unit_pct"], stop_loss_price=max(stop_loss, close * 0.93),
                    stop_profit_price=stop_profit, reason=f"海龟建仓: 突破{self.params['entry_period']}日高点", confidence=75))
            return signals

        profit_pct = position.profit_pct
        entry_price = position.cost
        current_units = min(position.market_value / (self.capital * self.params["unit_pct"]) if self.capital > 0 else 1,
                            self.params["max_units"])
        stop_loss_atr = entry_price - self.params["stop_loss_atr_mult"] * current_atr
        if close <= stop_loss_atr:
            signals.append(StrategySignal(signal_type=SignalType.STOP_LOSS, code=code, price=close,
                volume_pct=1.0, reason=f"海龟止损: 跌破{self.params['stop_loss_atr_mult']}倍ATR", confidence=90))
            return signals
        if close <= current_exit_lower:
            signals.append(StrategySignal(signal_type=SignalType.SELL, code=code, price=close,
                volume_pct=1.0, reason=f"海龟退出: 跌破{self.params['exit_period']}日低点", confidence=85))
            return signals
        if current_units < self.params["max_units"]:
            add_price = entry_price + self.params["add_unit_atr"] * current_atr
            if close >= add_price and profit_pct > 0:
                signals.append(StrategySignal(signal_type=SignalType.ADD, code=code, price=close,
                    volume_pct=self.params["unit_pct"], reason=f"海龟加仓: 第{int(current_units+1)}单位", confidence=75))
        df["MA20"] = df["close"].rolling(20).mean()
        if close < df["MA20"].iloc[-1] and profit_pct > 10:
            signals.append(StrategySignal(signal_type=SignalType.REDUCE, code=code, price=close,
                volume_pct=0.5, reason="海龟减持: 跌破20日均线", confidence=65))
        if not signals:
            signals.append(StrategySignal(signal_type=SignalType.HOLD, code=code, price=close,
                reason=f"海龟持有: 盈利{profit_pct:.1f}%", confidence=70))
        return signals
