"""
策略基类 - 基于 AKQuant 的云侠策略抽象
所有策略继承自 CloudKnightStrategy，实现 on_bar() 回调

AKQuant 完整回调生命周期（教材 §5）:
  on_start() → pre_open() → on_bar()/on_tick() → on_order() → on_trade() → on_stop()

A股微观结构处理（教材 §6）:
  - T+1 制度确保当日买入次日可卖
  - 涨跌停板不可交易检查
  - 滑点模拟与最小报价单位
"""

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

try:
    from akquant import Strategy as _AKStrategy
except ImportError:
    _AKStrategy = object

# 导入内置指标库（教材 §16），替换原手写的 _calc_* 方法
from ..indicators import calc_ma as _ind_calc_ma
from ..indicators import calc_atr as _ind_calc_atr
from ..indicators import calc_kdj as _ind_calc_kdj
from ..indicators import calc_rsi as _ind_calc_rsi
from ..indicators import calc_donchian as _ind_calc_donchian

# A股最小报价单位（教材 §6.3）
TICK_SIZE = 0.01  # 大部分 A 股 0.01 元
LIMIT_UP_THRESHOLD = 9.5  # 涨停判定阈值 (%)
LIMIT_DOWN_THRESHOLD = -9.5  # 跌停判定阈值 (%)


class CloudKnightStrategy(_AKStrategy):
    """云侠策略基类 - 继承 AKQuant Strategy

    子类必须实现:
        - __init__(params): 初始化策略参数
        - on_bar(bar): 每根 K 线回调，发出买卖信号

    子类可选覆盖（教材 §5 - 完整 on_xxx 回调地图）:
        - warmup_period: 预热期 Bar 数（默认 60）
        - on_start(): 策略启动时调用（一次性初始化）
        - pre_open(): 盘前准备（加载数据、计算指标）
        - on_tick(tick): Tick 级行情回调（高频策略）
        - on_order(order): 订单状态变更回调
        - on_trade(trade): 成交回报回调
        - on_stop(): 策略停止时调用（清理资源）
        - get_params_info(): 返回参数说明
    """

    name: str = "base"
    description: str = "策略基类"

    def __init__(self, params: dict[str, Any] | None = None):
        super().__init__()
        self._params = params or {}
        self._entry_price: dict[str, float] = {}  # symbol -> entry price
        self._entry_count: dict[str, int] = {}  # symbol -> entry count (for batch/pyramid)
        self._pre_close: dict[str, float] = {}  # symbol -> 前收盘价（涨跌停判断）
        self._daily_trades: int = 0  # 当日交易计数
        self._last_trade_date: str = ""  # 上次交易日期

    # ─── AKQuant 生命周期回调（教材 §5） ─────────────

    @property
    def warmup_period(self) -> int:
        """预热期 Bar 数，子类覆盖"""
        return 60

    def on_start(self):
        """策略启动时调用一次。可用于初始化外部连接、加载模型等。

        教材参考: §5.2 策略生命周期 - on_start
        """
        self.log(f"[{self.name}] 策略启动 | 参数: {self._params}", level="info")

    def pre_open(self):
        """盘前准备回调。在每日开盘前调用，用于数据预加载、指标预处理。

        教材参考: §5.4 进阶 - 盘前开盘语义与"双阶段次日执行"模式
        """
        self._daily_trades = 0

    def on_stop(self):
        """策略停止时调用。用于清理资源、保存状态。

        教材参考: §5.2 策略生命周期 - on_stop
        """
        self.log(f"[{self.name}] 策略停止", level="info")

    def get_params_info(self) -> dict[str, Any]:
        """返回策略参数"""
        return dict(self._params)

    # ─── A股微观结构处理（教材 §6） ──────────────────

    @staticmethod
    def is_limit_up(close: float, pre_close: float) -> bool:
        """判断是否涨停（A股 ±10% 规则，含 0.5% 容差）

        教材参考: §6.2 涨跌停场景建模
        """
        if pre_close <= 0:
            return False
        return (close / pre_close - 1) * 100 >= LIMIT_UP_THRESHOLD

    @staticmethod
    def is_limit_down(close: float, pre_close: float) -> bool:
        """判断是否跌停"""
        if pre_close <= 0:
            return False
        return (close / pre_close - 1) * 100 <= LIMIT_DOWN_THRESHOLD

    def can_execute_trade(self, symbol: str, side: str, close: float, pre_close: float) -> tuple[bool, str]:
        """A股微观结构合规检查 - 在发出买卖指令前调用

        教材参考: §6.1 T+1 交易制度的工程实现 + §6.2 涨跌停场景建模

        Args:
            symbol: 股票代码
            side: 'buy' 或 'sell'
            close: 当前收盘价
            pre_close: 前收盘价

        Returns:
            (是否可执行, 原因说明)
        """
        # 涨跌停不可交易检查
        if side == "buy" and self.is_limit_up(close, pre_close):
            return False, f"{symbol} 涨停封板，无法买入"
        if side == "sell" and self.is_limit_down(close, pre_close):
            return False, f"{symbol} 跌停封板，无法卖出"

        # T+1 制度检查：当日买入不可当日卖出（AKQuant 引擎自动处理，此处仅记录）
        # 教材注: AKQuant 的 t_plus_one=True 配置会自动阻止当日买入后卖出

        return True, "合规"

    def apply_slippage(self, price: float, side: str, slippage_pct: float = 0.001) -> float:
        """应用滑点模型

        教材参考: §6.3 滑点模拟
        - 买入: 实际成交价 = 价格 * (1 + slippage_pct)
        - 卖出: 实际成交价 = 价格 * (1 - slippage_pct)

        Args:
            price: 信号价格
            side: 'buy' 或 'sell'
            slippage_pct: 滑点比例（默认 0.1%）

        Returns:
            含滑点的盈亏后价格
        """
        if side == "buy":
            return price * (1 + slippage_pct)
        else:
            return price * (1 - slippage_pct)

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
    #  全部委托给内置指标库 indicators.py（教材 §16），消除 ~80 行冗余手写逻辑

    @staticmethod
    def _to_df(closes, highs=None, lows=None, opens=None, volumes=None):
        """快速构造 DataFrame 供 indicators.py 使用"""
        data = {"close": closes}
        if highs is not None:
            data["high"] = highs
        if lows is not None:
            data["low"] = lows
        if opens is not None:
            data["open"] = opens
        if volumes is not None:
            data["volume"] = volumes
        return pd.DataFrame(data)

    @staticmethod
    def _calc_ma(closes: np.ndarray, period: int) -> float:
        if len(closes) < period:
            return float("nan")
        df = pd.DataFrame({"close": closes})
        result = _ind_calc_ma(df, [period])
        col = f"MA{period}"
        if col in result.columns:
            return float(result[col].iloc[-1])
        return float("nan")

    @staticmethod
    def _calc_ema(closes: np.ndarray, period: int) -> np.ndarray:
        """返回 EMA 序列"""
        alpha = 2 / (period + 1)
        result = np.zeros_like(closes, dtype=float)
        result[0] = float(closes[0])
        for i in range(1, len(closes)):
            result[i] = alpha * float(closes[i]) + (1 - alpha) * result[i - 1]
        return result

    @staticmethod
    def _calc_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
        df = CloudKnightStrategy._to_df(closes, highs=highs, lows=lows)
        try:
            result = _ind_calc_atr(df, period=period)
            return float(result.iloc[-1])
        except Exception:
            return 0.0

    @staticmethod
    def _calc_kdj(
        highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, n: int = 9, m1: int = 3, m2: int = 3
    ) -> tuple:
        """返回 (K, D, J) 最新值 — 委托给 indicators.py"""
        df = CloudKnightStrategy._to_df(closes, highs=highs, lows=lows)
        try:
            k_series, d_series, j_series = _ind_calc_kdj(df, n=n, m1=m1, m2=m2)
            return float(k_series.iloc[-1]), float(d_series.iloc[-1]), float(j_series.iloc[-1])
        except Exception:
            return 50.0, 50.0, 50.0

    @staticmethod
    def _calc_rsi(closes: np.ndarray, period: int = 14) -> float:
        df = pd.DataFrame({"close": closes})
        try:
            result = _ind_calc_rsi(df, period=period)
            return float(result.iloc[-1])
        except Exception:
            return 50.0

    @staticmethod
    def _calc_donchian(highs: np.ndarray, lows: np.ndarray, period: int) -> tuple:
        """返回 (上轨, 下轨) — 委托给 indicators.py"""
        df = CloudKnightStrategy._to_df(np.zeros_like(highs), highs=highs, lows=lows)
        try:
            hh, ll = _ind_calc_donchian(df, period)
            return float(hh.iloc[-1]), float(ll.iloc[-1])
        except Exception:
            if len(highs) < period:
                return float("nan"), float("nan")
            return float(np.max(highs[-period:])), float(np.min(lows[-period:]))

    # ─── 下单封装 ─────────────────────────────────────

    def enter_long(self, symbol: str, pct: float, reason: str = "", check_micro: bool = True):
        """按账户总权益百分比开仓

        Args:
            symbol: 股票代码
            pct: 仓位百分比
            reason: 交易原因
            check_micro: 是否进行 A 股微观结构检查（教材 §6）
        """
        cash = self.get_cash()
        price = self.close

        # A股微观结构检查：涨跌停不可买
        if check_micro:
            pre_close = self._pre_close.get(symbol, price)
            can_exec, msg = self.can_execute_trade(symbol, "buy", price, pre_close)
            if not can_exec:
                self.log(f"SKIP BUY {symbol}: {msg}", level="warn")
                return

        # 应用滑点模型
        exec_price = self.apply_slippage(price, "buy")
        target_value = cash * pct
        qty = int(target_value / exec_price / 100) * 100
        if qty >= 100:
            self.buy(symbol=symbol, quantity=qty)
            self._entry_price[symbol] = exec_price
            self._entry_count[symbol] = self._entry_count.get(symbol, 0) + 1
            self._pre_close[symbol] = price  # 记录前收
            self._daily_trades += 1
            self.log(f"BUY {symbol} x{qty} @ {exec_price:.2f} [{reason}]", level="info")

    def add_position(self, symbol: str, pct: float, reason: str = "", check_micro: bool = True):
        """加仓"""
        price = self.close
        pos = self.get_position(symbol)
        if pos is None or pos.size <= 0:
            return self.enter_long(symbol, pct, reason, check_micro=check_micro)

        if check_micro:
            pre_close = self._pre_close.get(symbol, price)
            can_exec, msg = self.can_execute_trade(symbol, "buy", price, pre_close)
            if not can_exec:
                self.log(f"SKIP ADD {symbol}: {msg}", level="warn")
                return

        exec_price = self.apply_slippage(price, "buy")
        current_value = pos.size * price
        equity = self.get_cash() + current_value
        cash = self.get_cash()
        target_value = equity * pct
        qty = int(target_value / exec_price / 100) * 100
        if qty >= 100 and qty * exec_price <= cash:
            self.buy(symbol=symbol, quantity=qty)
            self._entry_count[symbol] = self._entry_count.get(symbol, 0) + 1
            self._daily_trades += 1
            self.log(f"ADD {symbol} x{qty} @ {exec_price:.2f} [{reason}]", level="info")

    def reduce_position(self, symbol: str, pct: float, reason: str = "", check_micro: bool = True):
        """减仓（按持仓百分比）"""
        pos = self.get_position(symbol)
        if pos is None or pos.size <= 0:
            return

        price = self.close
        if check_micro:
            pre_close = self._pre_close.get(symbol, price)
            can_exec, msg = self.can_execute_trade(symbol, "sell", price, pre_close)
            if not can_exec:
                self.log(f"SKIP REDUCE {symbol}: {msg}", level="warn")
                return

        exec_price = self.apply_slippage(price, "sell")
        qty = int(pos.size * pct / 100) * 100
        if qty >= 100:
            self.sell(symbol=symbol, quantity=qty)
            self._daily_trades += 1
            self.log(f"REDUCE {symbol} x{qty} [{reason}]", level="warn")

    def exit_position(self, symbol: str, reason: str = "", check_micro: bool = True):
        """清仓"""
        price = self.close
        if check_micro:
            pre_close = self._pre_close.get(symbol, price)
            can_exec, msg = self.can_execute_trade(symbol, "sell", price, pre_close)
            if not can_exec:
                self.log(f"SKIP EXIT {symbol}: {msg}", level="warn")
                return

        self.close_position(symbol=symbol)
        self._daily_trades += 1
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

    # ─── 信号生成（策略模式，教材 §5.3） ────────────────

    @staticmethod
    def generate_signal(code: str, name: str, df, analysis: dict, trend: dict) -> dict | None:
        """从技术指标数据生成交易信号（静态方法，供 SignalHunter 降级路径调用）

        教材参考: §5.3 进阶 - 策略信号生成模式

        子类可覆盖此方法以实现特定策略的信号逻辑。
        默认返回 None（无信号）。

        Args:
            code: 股票代码
            name: 股票名称
            df: 标准化后的 DataFrame (含 open/high/low/close/volume 列)
            analysis: comprehensive_analysis() 返回的指标结果
            trend: trend_score() 返回的趋势评分

        Returns:
            {"signal_type": "buy"/"sell", "confidence": "high"/"medium"/"low",
             "price": float, "reason": str, "stop_loss": float, "take_profit": float} | None
        """
        return None
