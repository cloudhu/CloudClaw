"""
风控熔断管理器 - 基于 AKQuant 教材第15章实盘风控标准

提供系统级风控能力：
- 日内亏损熔断（当日累计亏损超阈值 → 停止交易）
- 连续亏损熔断（连续 N 笔亏损 → 暂停）
- 最大回撤熔断（累计回撤超阈值 → 强制清仓）
- 市场异常波动熔断（大盘暴涨暴跌 → 暂停买入）
- 流动性风险控制（最小成交量/成交额过滤）
- 黑名单机制（涨停不可买、跌停不可卖）
- 交易频率限制（单日最大交易次数）

教材参考: AKQuant Textbook §15 - 实盘交易系统与运维
"""

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """熔断状态"""
    NORMAL = "normal"           # 正常交易
    WARNING = "warning"         # 预警（可交易但提示）
    PAUSED = "paused"           # 暂停开仓（平仓允许）
    STOPPED = "stopped"         # 完全停止（强制平仓）


class TriggerReason(Enum):
    """熔断触发原因"""
    DAILY_LOSS = "daily_loss"               # 日内亏损
    CONSECUTIVE_LOSS = "consecutive_loss"   # 连续亏损
    MAX_DRAWDOWN = "max_drawdown"           # 最大回撤
    MARKET_PANIC = "market_panic"           # 市场恐慌（大盘暴跌）
    MARKET_FRENZY = "market_frenzy"         # 市场狂热（大盘暴涨）
    TRADE_LIMIT = "trade_limit"             # 交易频率超限
    LIQUIDITY = "liquidity"                 # 流动性不足


@dataclass
class RiskEvent:
    """风控事件记录"""
    timestamp: datetime
    reason: TriggerReason
    detail: str
    state_before: CircuitState
    state_after: CircuitState


@dataclass
class TradeRecord:
    """单笔交易记录"""
    symbol: str
    timestamp: datetime
    side: str          # buy / sell
    price: float
    quantity: int
    pnl: float = 0.0   # 平仓盈亏
    is_close: bool = False


class CircuitBreaker:
    """熔断器 - 单维度熔断控制

    教材参考: §15 风控与熔断机制
    """

    def __init__(
        self,
        name: str,
        reason: TriggerReason,
        cooldown_seconds: int = 1800,  # 默认冷却 30 分钟
        auto_recover: bool = True,
    ):
        self.name = name
        self.reason = reason
        self.cooldown_seconds = cooldown_seconds
        self.auto_recover = auto_recover
        self._triggered_at: datetime | None = None
        self._state = CircuitState.NORMAL
        self._trigger_count = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            # 检查是否应该自动恢复
            if self._state != CircuitState.NORMAL and self.auto_recover and self._triggered_at:
                elapsed = (datetime.now() - self._triggered_at).total_seconds()
                if elapsed >= self.cooldown_seconds:
                    self._state = CircuitState.NORMAL
                    logger.info(f"[风控熔断] {self.name} 冷却结束，自动恢复")
            return self._state

    def trip(self, detail: str = "") -> RiskEvent:
        """触发熔断"""
        with self._lock:
            old_state = self._state
            if self._state == CircuitState.NORMAL:
                self._state = CircuitState.PAUSED
            else:
                # 重复触发，可能升级
                self._state = CircuitState.STOPPED
            self._triggered_at = datetime.now()
            self._trigger_count += 1
            logger.warning(f"[风控熔断] {self.name} 触发! 原因: {detail} | 状态: {old_state.value} → {self._state.value}")
            return RiskEvent(
                timestamp=self._triggered_at,
                reason=self.reason,
                detail=detail,
                state_before=old_state,
                state_after=self._state,
            )

    def reset(self):
        """手动重置"""
        with self._lock:
            self._state = CircuitState.NORMAL
            self._triggered_at = None
            logger.info(f"[风控熔断] {self.name} 手动重置")

    def get_status(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state.value,
            "triggered_at": self._triggered_at.isoformat() if self._triggered_at else None,
            "trigger_count": self._trigger_count,
            "cooldown_seconds": self.cooldown_seconds,
        }


class RiskManager:
    """统一风控管理器

    管理多个熔断器，提供统一的交易许可判断。
    所有策略在发出买卖指令前都应通过 RiskManager 检查。

    用法:
        rm = RiskManager(initial_capital=1_000_000)
        if rm.can_open():
            strategy.enter_long(...)
        rm.record_trade(TradeRecord(...))
        rm.update_equity(current_equity)
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000,
        daily_loss_limit_pct: float = 5.0,          # 日内亏损熔断 5%
        consecutive_loss_limit: int = 5,             # 连续亏损 5 笔熔断
        max_drawdown_limit_pct: float = 15.0,        # 最大回撤 15% 熔断
        market_panic_threshold: float = -5.0,        # 大盘跌 5% 恐慌熔断
        market_frenzy_threshold: float = 3.0,        # 大盘涨 3% 狂热熔断（避免追高）
        max_daily_trades: int = 20,                  # 单日最大交易次数
        single_stock_max_pct: float = 30.0,          # 单票最大仓位
        min_daily_volume: int = 1_000_000,           # 最小日成交量（万股，约100万股）
        min_daily_amount: float = 10_000_000,        # 最小日成交额（千万元）
    ):
        self.initial_capital = initial_capital
        self._peak_equity = initial_capital

        # 创建各维度熔断器
        self.breakers: dict[TriggerReason, CircuitBreaker] = {
            TriggerReason.DAILY_LOSS: CircuitBreaker(
                "日内亏损熔断", TriggerReason.DAILY_LOSS, cooldown_seconds=86400  # 次日恢复
            ),
            TriggerReason.CONSECUTIVE_LOSS: CircuitBreaker(
                "连续亏损熔断", TriggerReason.CONSECUTIVE_LOSS, cooldown_seconds=3600  # 1小时
            ),
            TriggerReason.MAX_DRAWDOWN: CircuitBreaker(
                "最大回撤熔断", TriggerReason.MAX_DRAWDOWN, cooldown_seconds=0, auto_recover=False  # 手动恢复
            ),
            TriggerReason.MARKET_PANIC: CircuitBreaker(
                "市场恐慌熔断", TriggerReason.MARKET_PANIC, cooldown_seconds=1800  # 30分钟
            ),
            TriggerReason.MARKET_FRENZY: CircuitBreaker(
                "市场狂热熔断", TriggerReason.MARKET_FRENZY, cooldown_seconds=1800
            ),
            TriggerReason.TRADE_LIMIT: CircuitBreaker(
                "交易频率熔断", TriggerReason.TRADE_LIMIT, cooldown_seconds=3600
            ),
            TriggerReason.LIQUIDITY: CircuitBreaker(
                "流动性不足", TriggerReason.LIQUIDITY, cooldown_seconds=300  # 5分钟，单票检查
            ),
        }

        # 风控参数
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.consecutive_loss_limit = consecutive_loss_limit
        self.max_drawdown_limit_pct = max_drawdown_limit_pct
        self.market_panic_threshold = market_panic_threshold
        self.market_frenzy_threshold = market_frenzy_threshold
        self.max_daily_trades = max_daily_trades
        self.single_stock_max_pct = single_stock_max_pct
        self.min_daily_volume = min_daily_volume
        self.min_daily_amount = min_daily_amount

        # 交易记录
        self._trades: list[TradeRecord] = []
        self._events: list[RiskEvent] = []
        self._lock = threading.Lock()

        # 状态
        self._current_equity = initial_capital
        self._day_start_equity = initial_capital
        self._last_reset_date = datetime.now().date()
        self._consecutive_losses = 0

    # ─── 交易许可检查 ──────────────────────────────

    def can_open(self) -> tuple[bool, str]:
        """是否可以开仓

        Returns:
            (允许, 原因说明)
        """
        with self._lock:
            self._daily_reset_if_needed()

            # 检查所有熔断器状态
            for reason, breaker in self.breakers.items():
                if breaker.state in (CircuitState.PAUSED, CircuitState.STOPPED):
                    # 部分熔断只暂停开仓（允许平仓）
                    if reason in (
                        TriggerReason.DAILY_LOSS,
                        TriggerReason.CONSECUTIVE_LOSS,
                        TriggerReason.MARKET_PANIC,
                        TriggerReason.MARKET_FRENZY,
                        TriggerReason.TRADE_LIMIT,
                    ):
                        return False, f"熔断中: {breaker.name}"

            # 仅 MAX_DRAWDOWN 和 LIQUIDITY 允许手动覆盖
            if self.breakers[TriggerReason.MAX_DRAWDOWN].state == CircuitState.STOPPED:
                return False, "最大回撤熔断，请手动恢复"

            return True, "正常"

    def can_close(self) -> tuple[bool, str]:
        """是否可以平仓（通常始终允许，除非完全熔断）"""
        with self._lock:
            if self.breakers[TriggerReason.MAX_DRAWDOWN].state == CircuitState.STOPPED:
                return False, "最大回撤强制熔断，不允许操作"
            return True, "正常"

    def can_trade(self, symbol: str, price: float, volume: int) -> tuple[bool, str]:
        """单笔交易前检查（含流动性过滤）

        Args:
            symbol: 股票代码
            price: 价格
            volume: 成交量
        """
        # 检查开仓许可
        ok, reason = self.can_open()
        if not ok:
            return ok, reason

        with self._lock:
            self._daily_reset_if_needed()

            # 交易频率检查
            today_trades = sum(
                1 for t in self._trades if t.timestamp.date() == datetime.now().date()
            )
            if today_trades >= self.max_daily_trades:
                self.breakers[TriggerReason.TRADE_LIMIT].trip(
                    f"当日已完成 {today_trades} 笔交易，超过上限 {self.max_daily_trades}"
                )
                return False, f"当日交易次数已满 ({self.max_daily_trades})"

            # 流动性检查
            trade_amount = price * volume
            if trade_amount < self.min_daily_amount and volume < self.min_daily_volume:
                return False, f"流动性不足: 成交额 {trade_amount:.0f} < {self.min_daily_amount:.0f}"

            return True, "正常"

    # ─── 状态更新 ──────────────────────────────────

    def update_equity(self, current_equity: float):
        """更新当前权益（每次 on_bar / 盘中定时调用）"""
        with self._lock:
            self._daily_reset_if_needed()
            self._current_equity = current_equity

            # 更新峰值权益
            if current_equity > self._peak_equity:
                self._peak_equity = current_equity

            # 检查日内亏损
            daily_pnl_pct = (current_equity - self._day_start_equity) / self._day_start_equity * 100
            if daily_pnl_pct <= -self.daily_loss_limit_pct:
                self.breakers[TriggerReason.DAILY_LOSS].trip(
                    f"日内亏损 {daily_pnl_pct:.2f}% >= {self.daily_loss_limit_pct}%"
                )

            # 检查最大回撤
            dd_pct = (self._peak_equity - current_equity) / self._peak_equity * 100
            if dd_pct >= self.max_drawdown_limit_pct:
                self.breakers[TriggerReason.MAX_DRAWDOWN].trip(
                    f"回撤 {dd_pct:.2f}% >= {self.max_drawdown_limit_pct}%"
                )

    def update_market(self, index_change_pct: float):
        """更新市场状态（大盘涨跌幅）

        教材参考: §6 A股市场微观结构 - 大盘异常波动应对
        """
        with self._lock:
            if index_change_pct <= self.market_panic_threshold:
                self.breakers[TriggerReason.MARKET_PANIC].trip(
                    f"大盘暴跌 {index_change_pct:.2f}% <= {self.market_panic_threshold}%"
                )
            elif index_change_pct >= self.market_frenzy_threshold:
                self.breakers[TriggerReason.MARKET_FRENZY].trip(
                    f"大盘急涨 {index_change_pct:.2f}% >= {self.market_frenzy_threshold}%"
                )

    def record_trade(self, trade: TradeRecord):
        """记录一笔交易"""
        with self._lock:
            self._trades.append(trade)

            # 更新连续亏损计数
            if trade.is_close:
                if trade.pnl < 0:
                    self._consecutive_losses += 1
                else:
                    self._consecutive_losses = 0

                if self._consecutive_losses >= self.consecutive_loss_limit:
                    self.breakers[TriggerReason.CONSECUTIVE_LOSS].trip(
                        f"连续亏损 {self._consecutive_losses} 笔 >= {self.consecutive_loss_limit}"
                    )

    def record_event(self, event: RiskEvent):
        self._events.append(event)

    # ─── 查询接口 ──────────────────────────────────

    def get_overall_state(self) -> CircuitState:
        """获取全局风控状态"""
        states = [b.state for b in self.breakers.values()]
        if CircuitState.STOPPED in states:
            return CircuitState.STOPPED
        if CircuitState.PAUSED in states:
            return CircuitState.PAUSED
        return CircuitState.NORMAL

    def get_daily_stats(self) -> dict[str, Any]:
        """获取当日风控统计"""
        with self._lock:
            self._daily_reset_if_needed()
            today = datetime.now().date()
            today_trades = [t for t in self._trades if t.timestamp.date() == today]
            daily_pnl_pct = (self._current_equity - self._day_start_equity) / self._day_start_equity * 100
            dd_pct = (self._peak_equity - self._current_equity) / self._peak_equity * 100

            return {
                "date": today.isoformat(),
                "current_equity": round(self._current_equity, 2),
                "peak_equity": round(self._peak_equity, 2),
                "daily_pnl_pct": round(daily_pnl_pct, 2),
                "max_drawdown_pct": round(dd_pct, 2),
                "consecutive_losses": self._consecutive_losses,
                "today_trade_count": len(today_trades),
                "total_trade_count": len(self._trades),
                "overall_state": self.get_overall_state().value,
                "breakers": {r.value: b.get_status() for r, b in self.breakers.items()},
            }

    def reset_daily(self):
        """重置日内状态（新交易日开始时调用）"""
        with self._lock:
            self._day_start_equity = self._current_equity
            self._last_reset_date = datetime.now().date()
            self._consecutive_losses = 0

            # 重置日内相关熔断器
            for reason in (
                TriggerReason.DAILY_LOSS,
                TriggerReason.TRADE_LIMIT,
                TriggerReason.MARKET_PANIC,
                TriggerReason.MARKET_FRENZY,
            ):
                self.breakers[reason].reset()

            logger.info(f"[风控] 新交易日开始，重置日内统计。起始权益: {self._current_equity:.2f}")

    def reset_all(self):
        """完全重置风控系统"""
        with self._lock:
            self._peak_equity = self._current_equity
            self._day_start_equity = self._current_equity
            self._consecutive_losses = 0
            for breaker in self.breakers.values():
                breaker.reset()
            self._trades.clear()
            self._events.clear()
            logger.info("[风控] 完全重置")

    # ─── A股微观结构处理 ───────────────────────────

    @staticmethod
    def is_limit_up(close: float, pre_close: float) -> bool:
        """判断是否涨停（A股 ±10% 规则，ST 为 ±5%）

        教材参考: §6 A股市场微观结构 - 涨跌停场景建模
        """
        if pre_close <= 0:
            return False
        change_pct = (close / pre_close - 1) * 100
        return change_pct >= 9.5  # 留 0.5% 容差

    @staticmethod
    def is_limit_down(close: float, pre_close: float) -> bool:
        """判断是否跌停"""
        if pre_close <= 0:
            return False
        change_pct = (close / pre_close - 1) * 100
        return change_pct <= -9.5

    @staticmethod
    def can_buy_at_limit(close: float, pre_close: float) -> bool:
        """涨停板是否可以买入（通常涨停封死不可买）"""
        return not RiskManager.is_limit_up(close, pre_close)

    @staticmethod
    def can_sell_at_limit(close: float, pre_close: float) -> bool:
        """跌停板是否可以卖出（通常跌停封死不可卖）"""
        return not RiskManager.is_limit_down(close, pre_close)

    # ─── 内部方法 ──────────────────────────────────

    def _daily_reset_if_needed(self):
        """检查是否需要日内重置"""
        today = datetime.now().date()
        if today > self._last_reset_date:
            self.reset_daily()
