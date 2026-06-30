"""策略基类 - 所有策略的抽象基类"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any
from datetime import datetime

import pandas as pd
import numpy as np


class SignalType(Enum):
    BUY = "buy"
    ADD = "add"
    REDUCE = "reduce"
    SELL = "sell"
    STOP_LOSS = "stop_loss"
    STOP_PROFIT = "stop_profit"
    HOLD = "hold"


@dataclass
class StrategySignal:
    signal_type: SignalType
    code: str
    name: str = ""
    price: float = 0.0
    volume_pct: float = 0.0
    stop_loss_price: float = 0.0
    stop_profit_price: float = 0.0
    reason: str = ""
    confidence: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Position:
    code: str
    name: str
    cost: float
    current_price: float
    volume: int
    market_value: float
    profit_pct: float
    hold_days: int


class BaseStrategy(ABC):
    name: str = "base"
    description: str = "策略基类"

    def __init__(self, params: Dict[str, Any] = None):
        self.params = params or {}
        self.positions: Dict[str, Position] = {}
        self.capital: float = 0.0
        self.available_cash: float = 0.0

    def set_capital(self, capital: float):
        self.capital = capital
        self.available_cash = capital

    def update_position(self, code: str, position: Position):
        self.positions[code] = position

    def remove_position(self, code: str):
        self.positions.pop(code, None)

    @abstractmethod
    def select_stocks(self, df: pd.DataFrame, market_data: Dict[str, pd.DataFrame]) -> List[str]:
        pass

    @abstractmethod
    def generate_signals(self, code: str, df: pd.DataFrame,
                         position: Optional[Position] = None) -> List[StrategySignal]:
        pass

    def get_stop_loss_price(self, code: str, entry_price: float, df: pd.DataFrame) -> float:
        return entry_price * 0.93

    def get_stop_profit_price(self, code: str, entry_price: float, df: pd.DataFrame) -> float:
        return entry_price * 1.15

    def risk_check(self, signal: StrategySignal, positions: Dict[str, Position], total_capital: float) -> bool:
        if signal.signal_type in (SignalType.BUY, SignalType.ADD):
            total_position_value = sum(p.market_value for p in positions.values())
            if signal.price > 0 and total_position_value + signal.volume_pct * total_capital > total_capital * 0.9:
                return False
            if signal.volume_pct > 0.3:
                return False
        return True

    def get_params_info(self) -> Dict[str, Any]:
        return self.params
