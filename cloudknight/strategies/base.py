"""
策略基类 - 基于 AKQuant 的云侠策略抽象
所有策略继承自 akquant.Strategy，实现 on_bar() 回调
"""

from typing import Any

import numpy as np

try:
    from akquant import Strategy as _AKStrategy
except ImportError:
    _AKStrategy = object


class CloudKnightStrategy(_AKStrategy):
    """云侠策略基类 - 继承 AKQuant Strategy

    子类必须实现:
        - __init__(params): 初始化策略参数
        - on_bar(bar): 每根 K 线回调，发出买卖信号

    子类可选覆盖:
        - warmup_period: 预热期 Bar 数（默认 60）
        - on_start(): 策略启动时调用
        - get_params_info(): 返回参数说明
    """

    name: str = "base"
    description: str = "策略基类"

    def __init__(self, params: dict[str, Any] | None = None):
        super().__init__()
        self._params = params or {}
        self._entry_price: dict[str, float] = {}  # symbol -> entry price
        self._entry_count: dict[str, int] = {}  # symbol -> entry count (for batch/pyramid)

    @property
    def warmup_period(self) -> int:
        return 60

    def get_params_info(self) -> dict[str, Any]:
        """返回策略参数"""
        return dict(self._params)

    def get_history_closes(self, symbol: str, count: int) -> np.ndarray:
        """获取收盘价历史（numpy array）"""
        return self.get_history(count=count, symbol=symbol, field="close")

    def get_history_highs(self, symbol: str, count: int) -> np.ndarray:
        return self.get_history(count=count, symbol=symbol, field="high")

    def get_history_lows(self, symbol: str, count: int) -> np.ndarray:
        return self.get_history(count=count, symbol=symbol, field="low")

    def get_history_volumes(self, symbol: str, count: int) -> np.ndarray:
        return self.get_history(count=count, symbol=symbol, field="volume")

    def get_history_opens(self, symbol: str, count: int) -> np.ndarray:
        return self.get_history(count=count, symbol=symbol, field="open")

    def get_pct_changes(self, symbol: str, count: int) -> np.ndarray:
        """计算涨跌幅序列（百分比）"""
        closes = self.get_history_closes(symbol, count + 1)
        if len(closes) < 2:
            return np.array([])
        return (closes[1:] / closes[:-1] - 1) * 100

    # ─── 技术指标（在 on_bar 中调用） ───────────────────

    @staticmethod
    def _calc_ma(closes: np.ndarray, period: int) -> float:
        if len(closes) < period:
            return float("nan")
        return float(np.mean(closes[-period:]))

    @staticmethod
    def _calc_ema(closes: np.ndarray, period: int) -> np.ndarray:
        """返回 EMA 序列"""
        alpha = 2 / (period + 1)
        result = np.zeros_like(closes)
        result[0] = closes[0]
        for i in range(1, len(closes)):
            result[i] = alpha * closes[i] + (1 - alpha) * result[i - 1]
        return result

    @staticmethod
    def _calc_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
        n = min(len(highs), len(lows), len(closes))
        if n < 2:
            return 0.0
        tr = np.zeros(n)
        tr[0] = highs[0] - lows[0]
        for i in range(1, n):
            tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
        alpha = 2 / (period + 1)
        atr = np.zeros(n)
        atr[0] = tr[0]
        for i in range(1, n):
            atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]
        return float(atr[-1])

    @staticmethod
    def _calc_kdj(
        highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, n: int = 9, m1: int = 3, m2: int = 3
    ) -> tuple:
        """返回 (K, D, J) 最新值"""
        length = min(len(highs), len(lows), len(closes))
        if length < n:
            return 50.0, 50.0, 50.0

        k_vals = np.full(length, 50.0)
        d_vals = np.full(length, 50.0)
        alpha_k = 2 / (m1 + 1) if m1 > 0 else 1.0
        alpha_d = 2 / (m2 + 1) if m2 > 0 else 1.0

        for i in range(n - 1, length):
            hh = np.max(highs[i - n + 1 : i + 1])
            ll = np.min(lows[i - n + 1 : i + 1])
            rsv = (closes[i] - ll) / (hh - ll) * 100 if hh > ll else 50.0
            if i == n - 1:
                k_vals[i] = 50 * (1 - alpha_k) + rsv * alpha_k
                d_vals[i] = 50 * (1 - alpha_d) + k_vals[i] * alpha_d
            else:
                k_vals[i] = k_vals[i - 1] * (1 - alpha_k) + rsv * alpha_k
                d_vals[i] = d_vals[i - 1] * (1 - alpha_d) + k_vals[i] * alpha_d

        k = float(k_vals[-1])
        d = float(d_vals[-1])
        j = 3 * k - 2 * d
        return k, d, j

    @staticmethod
    def _calc_rsi(closes: np.ndarray, period: int = 14) -> float:
        if len(closes) < period + 1:
            return 50.0
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        alpha = 1 / period
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        for i in range(period, len(gains)):
            avg_gain = alpha * gains[i] + (1 - alpha) * avg_gain
            avg_loss = alpha * losses[i] + (1 - alpha) * avg_loss
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return float(100 - 100 / (1 + rs))

    @staticmethod
    def _calc_donchian(highs: np.ndarray, lows: np.ndarray, period: int) -> tuple:
        """返回 (上轨, 下轨)"""
        if len(highs) < period:
            return float("nan"), float("nan")
        return float(np.max(highs[-period:])), float(np.min(lows[-period:]))

    # ─── 下单封装 ─────────────────────────────────────

    def enter_long(self, symbol: str, pct: float, reason: str = ""):
        """按账户总权益百分比开仓"""
        cash = self.get_cash()
        price = self.close
        target_value = cash * pct
        qty = int(target_value / price / 100) * 100
        if qty >= 100:
            self.buy(symbol=symbol, quantity=qty)
            self._entry_price[symbol] = price
            self._entry_count[symbol] = self._entry_count.get(symbol, 0) + 1
            self.log(f"BUY {symbol} x{qty} @ {price:.2f} [{reason}]", level="info")

    def add_position(self, symbol: str, pct: float, reason: str = ""):
        """加仓"""
        price = self.close
        pos = self.get_position(symbol)
        if pos is None or pos.size <= 0:
            return self.enter_long(symbol, pct, reason)
        current_value = pos.size * price
        equity = self.get_cash() + current_value
        cash = self.get_cash()
        target_value = equity * pct
        qty = int(target_value / price / 100) * 100
        if qty >= 100 and qty * price <= cash:
            self.buy(symbol=symbol, quantity=qty)
            self._entry_count[symbol] = self._entry_count.get(symbol, 0) + 1
            self.log(f"ADD {symbol} x{qty} @ {price:.2f} [{reason}]", level="info")

    def reduce_position(self, symbol: str, pct: float, reason: str = ""):
        """减仓（按持仓百分比）"""
        pos = self.get_position(symbol)
        if pos is None or pos.size <= 0:
            return
        qty = int(pos.size * pct / 100) * 100
        if qty >= 100:
            self.sell(symbol=symbol, quantity=qty)
            self.log(f"REDUCE {symbol} x{qty} [{reason}]", level="info")

    def exit_position(self, symbol: str, reason: str = ""):
        """清仓"""
        self.close_position(symbol=symbol)
        self.log(f"SELL {symbol} all [{reason}]", level="info")
        self._entry_price.pop(symbol, None)
        self._entry_count.pop(symbol, None)

    def has_position(self, symbol: str) -> bool:
        pos = self.get_position(symbol)
        return pos is not None and pos.size > 0

    def position_pnl_pct(self, symbol: str) -> float:
        """持仓盈亏百分比"""
        pos = self.get_position(symbol)
        if pos is None or pos.size <= 0 or pos.entry_price is None or pos.entry_price <= 0:
            return 0.0
        return (self.close / pos.entry_price - 1) * 100
