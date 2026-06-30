"""回测引擎 - 模拟交易与历史数据回测"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

from .config import (DEFAULT_CAPITAL, DEFAULT_COMMISSION, DEFAULT_STAMP_TAX,
                     DEFAULT_SLIPPAGE, BACKTEST_START_DATE, BACKTEST_END_DATE)
from .strategies.base import BaseStrategy, StrategySignal, SignalType, Position
from .data_manager import DataFetcher

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    date: str; code: str; name: str; action: str; price: float
    volume: int; amount: float; commission: float; stamp_tax: float
    reason: str; pnl: float = 0.0


@dataclass
class BacktestPosition:
    code: str; name: str; cost: float; volume: int; buy_date: str
    last_add_price: float = 0.0; batches: int = 1


@dataclass
class BacktestResult:
    strategy_name: str; start_date: str; end_date: str
    initial_capital: float; final_capital: float
    total_return: float; annual_return: float; max_drawdown: float
    sharpe_ratio: float; win_rate: float; total_trades: int
    profit_trades: int; avg_profit: float; avg_loss: float; profit_factor: float
    daily_returns: List[float] = field(default_factory=list)
    trades: List[Trade] = field(default_factory=list)
    equity_curve: pd.DataFrame = None


class BacktestEngine:
    def __init__(self, capital: float = None, commission: float = None,
                 stamp_tax: float = None, slippage: float = None):
        self.initial_capital = capital or DEFAULT_CAPITAL
        self.commission = commission or DEFAULT_COMMISSION
        self.stamp_tax = stamp_tax or DEFAULT_STAMP_TAX
        self.slippage = slippage or DEFAULT_SLIPPAGE
        self.capital = self.initial_capital
        self.available_cash = self.initial_capital
        self.positions: Dict[str, BacktestPosition] = {}
        self.trades: List[Trade] = []
        self.daily_values: List[Dict] = []
        self.fetcher = DataFetcher()

    def reset(self):
        self.capital = self.initial_capital
        self.available_cash = self.initial_capital
        self.positions = {}
        self.trades = []
        self.daily_values = []

    def _calc_cost(self, price: float, volume: int, is_buy: bool) -> Tuple[float, float, float]:
        amount = price * volume
        commission = max(5, amount * self.commission)
        stamp_tax = amount * self.stamp_tax if not is_buy else 0
        slippage_cost = amount * self.slippage
        return commission + stamp_tax + slippage_cost, commission, stamp_tax

    def _get_market_price(self, code: str, date: str) -> Optional[float]:
        df = self.fetcher.fetch_daily_kline(code)
        if df.empty:
            return None
        row = df[df["date"] == date] if "date" in df.columns else df[df["日期"] == date]
        if row.empty:
            return None
        return row.iloc[-1].get("close", row.iloc[-1].get("收盘", None))

    def execute_signal(self, signal: StrategySignal, date: str, stock_name: str = "") -> Optional[Trade]:
        code = signal.code
        price = signal.price
        if signal.signal_type in (SignalType.BUY, SignalType.ADD):
            max_amount = min(self.capital * signal.volume_pct, self.available_cash)
            if max_amount < price * 100:
                return None
            volume = int(max_amount / price / 100) * 100
            if volume < 100:
                return None
            actual_price = price * (1 + self.slippage)
            amount = actual_price * volume
            total_cost, commission, stamp_tax = self._calc_cost(price, volume, True)
            if amount + total_cost > self.available_cash:
                volume = int((self.available_cash - total_cost) / actual_price / 100) * 100
                if volume < 100:
                    return None
                amount = actual_price * volume
                total_cost, commission, stamp_tax = self._calc_cost(price, volume, True)
            self.available_cash -= (amount + total_cost)
            if code in self.positions:
                pos = self.positions[code]
                pos.cost = (pos.cost * pos.volume + amount) / (pos.volume + volume)
                pos.volume += volume
                pos.last_add_price = price
                pos.batches += 1
            else:
                self.positions[code] = BacktestPosition(code=code, name=stock_name, cost=price,
                    volume=volume, buy_date=date, last_add_price=price)
            trade = Trade(date=date, code=code, name=stock_name, action="BUY", price=price,
                          volume=volume, amount=amount, commission=commission, stamp_tax=stamp_tax, reason=signal.reason)
            self.trades.append(trade)
            return trade
        elif signal.signal_type in (SignalType.SELL, SignalType.STOP_LOSS, SignalType.STOP_PROFIT, SignalType.REDUCE):
            if code not in self.positions:
                return None
            pos = self.positions[code]
            sell_volume = int(pos.volume * signal.volume_pct / 100) * 100
            if sell_volume < 100 and signal.volume_pct >= 1.0:
                sell_volume = pos.volume
            if sell_volume < 100:
                return None
            actual_price = price * (1 - self.slippage)
            amount = actual_price * sell_volume
            total_cost, commission, stamp_tax = self._calc_cost(price, sell_volume, False)
            self.available_cash += (amount - total_cost)
            pnl = (actual_price - pos.cost) * sell_volume - total_cost
            if sell_volume >= pos.volume:
                del self.positions[code]
            else:
                pos.volume -= sell_volume
            trade = Trade(date=date, code=code, name=stock_name, action="SELL", price=price,
                          volume=sell_volume, amount=amount, commission=commission, stamp_tax=stamp_tax,
                          reason=signal.reason, pnl=pnl)
            self.trades.append(trade)
            return trade
        return None

    def _calc_total_value(self, date: str) -> float:
        position_value = 0
        for code, pos in list(self.positions.items()):
            mp = self._get_market_price(code, date)
            position_value += (mp if mp else pos.cost) * pos.volume
        return self.available_cash + position_value

    def run_backtest(self, strategy: BaseStrategy, start_date: str = None, end_date: str = None,
                     stock_pool: List[str] = None, verbose: bool = False) -> BacktestResult:
        self.reset()
        strategy.set_capital(self.initial_capital)
        start_date = start_date or BACKTEST_START_DATE
        end_date = end_date or BACKTEST_END_DATE
        index_df = self.fetcher.fetch_index_daily("000001", start_date, end_date)
        if index_df.empty:
            return None
        trading_dates = index_df["date"].tolist() if "date" in index_df.columns else sorted(index_df.index.tolist())
        if verbose:
            logger.info(f"回测期间: {start_date} ~ {end_date}, 共 {len(trading_dates)} 个交易日")
        for i, date in enumerate(trading_dates):
            for code in list(self.positions.keys()):
                mp = self._get_market_price(code, date)
                if mp:
                    self.positions[code].current_price = mp
            for code in list(self.positions.keys()):
                df = self.fetcher.fetch_daily_kline(code, start_date, date)
                if df.empty or len(df) < 20:
                    continue
                pos = self.positions.get(code)
                cp = getattr(pos, "current_price", pos.cost) if pos else 0
                position_obj = Position(code=code, name=pos.name, cost=pos.cost, current_price=cp,
                    volume=pos.volume, market_value=cp * pos.volume,
                    profit_pct=(cp / pos.cost - 1) * 100 if pos.cost > 0 else 0,
                    hold_days=(datetime.strptime(date, "%Y%m%d") - datetime.strptime(pos.buy_date, "%Y%m%d")).days) if pos else None
                signals = strategy.generate_signals(code, df, position_obj)
                for signal in signals:
                    if signal.signal_type != SignalType.HOLD:
                        self.execute_signal(signal, date, pos.name if pos else code)
            if i % 5 == 0 and stock_pool:
                for code in stock_pool:
                    if code in self.positions:
                        continue
                    df = self.fetcher.fetch_daily_kline(code, start_date, date)
                    if df.empty or len(df) < 20:
                        continue
                    signals = strategy.generate_signals(code, df, None)
                    for signal in signals:
                        if signal.signal_type != SignalType.HOLD:
                            self.execute_signal(signal, date, code)
            total_value = self._calc_total_value(date)
            self.daily_values.append({"date": date, "value": total_value,
                                       "return": (total_value / self.initial_capital - 1) * 100})
        for code in list(self.positions.keys()):
            pos = self.positions[code]
            mp = self._get_market_price(code, trading_dates[-1])
            if mp:
                signal = StrategySignal(signal_type=SignalType.SELL, code=code, price=mp,
                                        volume_pct=1.0, reason="回测结束强制平仓")
                self.execute_signal(signal, trading_dates[-1], pos.name)
        final_value = self.daily_values[-1]["value"] if self.daily_values else self.initial_capital
        return self._calc_performance(strategy.name, start_date, end_date, final_value)

    def _calc_performance(self, strategy_name: str, start_date: str, end_date: str, final_value: float) -> BacktestResult:
        total_return = (final_value / self.initial_capital - 1) * 100
        try:
            days = (datetime.strptime(end_date, "%Y%m%d") - datetime.strptime(start_date, "%Y%m%d")).days
            annual_return = ((final_value / self.initial_capital) ** (1 / max(days / 365, 0.01)) - 1) * 100
        except Exception:
            annual_return = total_return
        daily_returns = []
        equity_curve = pd.DataFrame(self.daily_values)
        if len(equity_curve) > 1:
            equity_curve["daily_return"] = equity_curve["value"].pct_change()
            daily_returns = equity_curve["daily_return"].dropna().tolist()
        max_drawdown = 0
        if len(equity_curve) > 0:
            cummax = equity_curve["value"].cummax()
            drawdown = (equity_curve["value"] - cummax) / cummax * 100
            max_drawdown = abs(drawdown.min()) if not drawdown.empty else 0
        sharpe = 0
        if daily_returns and len(daily_returns) > 1:
            avg_daily = np.mean(daily_returns)
            std_daily = np.std(daily_returns, ddof=1)
            if std_daily > 0:
                sharpe = (avg_daily - 0.02 / 252) / std_daily * np.sqrt(252)
        sell_trades = [t for t in self.trades if t.action == "SELL"]
        total_trades = len(sell_trades)
        profit_trades = [t for t in sell_trades if t.pnl > 0]
        loss_trades = [t for t in sell_trades if t.pnl < 0]
        win_rate = len(profit_trades) / total_trades * 100 if total_trades > 0 else 0
        avg_profit = np.mean([t.pnl for t in profit_trades]) if profit_trades else 0
        avg_loss = abs(np.mean([t.pnl for t in loss_trades])) if loss_trades else 0
        total_profit = sum(t.pnl for t in profit_trades)
        total_loss = abs(sum(t.pnl for t in loss_trades))
        profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")
        return BacktestResult(strategy_name=strategy_name, start_date=start_date, end_date=end_date,
            initial_capital=self.initial_capital, final_capital=final_value,
            total_return=round(total_return, 2), annual_return=round(annual_return, 2),
            max_drawdown=round(max_drawdown, 2), sharpe_ratio=round(sharpe, 2),
            win_rate=round(win_rate, 2), total_trades=total_trades, profit_trades=len(profit_trades),
            avg_profit=round(avg_profit, 2), avg_loss=round(avg_loss, 2),
            profit_factor=round(profit_factor, 2) if profit_factor != float("inf") else 999)

    def compare_strategies(self, strategies: List[BaseStrategy], start_date: str = None,
                           end_date: str = None, stock_pool: List[str] = None,
                           verbose: bool = False) -> List[BacktestResult]:
        results = []
        for strategy in strategies:
            result = self.run_backtest(strategy, start_date, end_date, stock_pool, verbose)
            if result:
                results.append(result)
        return results
