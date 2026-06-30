"""
龙头战法策略 - 追踪市场最强龙头股
核心：涨停连板 + 高换手 + 大资金流入
"""

from typing import List, Dict, Optional
import pandas as pd
import numpy as np

from .base import BaseStrategy, StrategySignal, SignalType, Position


class DragonHeadStrategy(BaseStrategy):
    name = "dragon_head"
    description = "龙头战法 - 追踪市场最强龙头股"

    def __init__(self, params: Dict = None):
        default_params = {"max_position_pct": 0.30, "stop_loss_pct": 0.07, "stop_profit_pct": 0.20,
                          "min_limit_up_days": 2, "max_chasing_pct": 0.05,
                          "volume_ratio_min": 1.5, "turnover_rate_min": 5, "turnover_rate_max": 30,
                          "market_cap_min": 2e9, "market_cap_max": 50e9}
        default_params.update(params or {})
        super().__init__(default_params)

    def select_stocks(self, df: pd.DataFrame, market_data: Dict[str, pd.DataFrame]) -> List[str]:
        selected = []
        if df is None or df.empty:
            return selected
        for _, row in df.iterrows():
            code = str(row.get("代码", row.get("code", "")))
            pct = row.get("涨跌幅", row.get("pct_change", 0))
            turnover = row.get("换手率", row.get("turnover", 0))
            if pct is None or turnover is None:
                continue
            if pct < 9.5:
                continue
            if not (self.params["turnover_rate_min"] <= turnover <= self.params["turnover_rate_max"]):
                continue
            selected.append(code)
        return selected

    def generate_signals(self, code: str, df: pd.DataFrame,
                         position: Optional[Position] = None) -> List[StrategySignal]:
        signals = []
        if df.empty or len(df) < 5:
            return signals
        latest = df.iloc[-1]
        close = latest.get("close", latest.get("收盘", 0))
        open_p = latest.get("open", latest.get("开盘", 0))
        high = latest.get("high", latest.get("最高", 0))
        volume = latest.get("volume", latest.get("成交量", 0))
        pct = latest.get("pct_change", latest.get("涨跌幅", 0))
        is_limit_up = pct >= 9.5
        avg_vol_5 = df["volume" if "volume" in df.columns else "成交量"].tail(5).mean()
        vol_ratio = volume / avg_vol_5 if avg_vol_5 > 0 else 1
        limit_up_days = 0
        for i in range(len(df) - 1, -1, -1):
            if df.iloc[i].get("pct_change", df.iloc[i].get("涨跌幅", 0)) >= 9.5:
                limit_up_days += 1
            else:
                break
        prev_pct = df.iloc[-2].get("pct_change", df.iloc[-2].get("涨跌幅", 0)) if len(df) >= 2 else 0
        is_break_board = not is_limit_up and prev_pct >= 9.5

        if position is None or position.volume == 0:
            if is_limit_up and limit_up_days >= self.params["min_limit_up_days"]:
                stop_loss = close * (1 - self.params["stop_loss_pct"])
                stop_profit = close * (1 + self.params["stop_profit_pct"])
                signals.append(StrategySignal(signal_type=SignalType.BUY, code=code, price=close,
                    volume_pct=self.params["max_position_pct"], stop_loss_price=stop_loss, stop_profit_price=stop_profit,
                    reason=f"龙头建仓: 连板{limit_up_days}天, 量比{vol_ratio:.1f}", confidence=min(90, 60 + limit_up_days * 5)))
            return signals

        profit_pct = position.profit_pct
        hold_days = position.hold_days

        if profit_pct <= -self.params["stop_loss_pct"] * 100:
            signals.append(StrategySignal(signal_type=SignalType.STOP_LOSS, code=code, price=close, volume_pct=1.0,
                reason=f"龙头止损: 亏损{profit_pct:.1f}%", confidence=95))
            return signals
        if profit_pct >= self.params["stop_profit_pct"] * 100:
            signals.append(StrategySignal(signal_type=SignalType.STOP_PROFIT, code=code, price=close, volume_pct=1.0,
                reason=f"龙头止盈: 盈利{profit_pct:.1f}%", confidence=90))
            return signals
        if is_break_board or (high > close * 1.05 and close < open_p):
            signals.append(StrategySignal(signal_type=SignalType.REDUCE, code=code, price=close,
                volume_pct=0.5, reason="龙头减持: 炸板/冲高回落", confidence=80))
        if vol_ratio > 3 and profit_pct > 5:
            signals.append(StrategySignal(signal_type=SignalType.REDUCE, code=code, price=close,
                volume_pct=0.3, reason=f"龙头减持: 量比{vol_ratio:.1f}异常放大", confidence=75))
        if is_break_board and hold_days >= 2:
            two_day_ago = df.iloc[-3].get("pct_change", df.iloc[-3].get("涨跌幅", 0)) if len(df) >= 3 else 0
            if two_day_ago < 9.5:
                signals.append(StrategySignal(signal_type=SignalType.SELL, code=code, price=close,
                    volume_pct=1.0, reason="龙头清仓: 连续断板", confidence=85))
        if is_limit_up and vol_ratio < 0.7 and limit_up_days <= 5:
            signals.append(StrategySignal(signal_type=SignalType.ADD, code=code, price=close,
                volume_pct=0.15, reason="龙头加仓: 缩量加速", confidence=80))
        if not signals:
            signals.append(StrategySignal(signal_type=SignalType.HOLD, code=code, price=close,
                reason=f"龙头持有: 连板{limit_up_days}天, 盈利{profit_pct:.1f}%", confidence=60))
        return signals
