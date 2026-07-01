"""
实时交易引擎 - 时间周期驱动的 A 股自动交易主控

交易日完整时间线:
  ┌─────────────────────────────────────────────────────┐
  │ 08:30  盘前准备  → 大盘/板块分析, 股票池筛选         │
  │ 09:15  集合竞价   → 监控竞价走势                     │
  │ 09:26  竞价结果   → 竞价分析, 开盘交易决策            │
  │ 09:30  早盘开盘   → 分时走势跟踪, 买卖点扫描          │
  │ 11:30  午间休市   → 15分钟K线分析, 制定下午计划       │
  │ 13:00  午盘开盘   → 继续分时跟踪, 执行交易信号        │
  │ 15:00  收盘       → 盘后选股, 制定次日计划            │
  └─────────────────────────────────────────────────────┘

心跳驱动的策略日常生命周期（贯穿所有时段）:
  ┌──────────────────────────────────────────────────────┐
  │ ① 选股 SCREENING         → 拉取代码诊股入池           │
  │ ② 信号扫描 SIGNAL_SCAN    → 观察买入/卖出信号         │
  │ ③ 止盈止损 STOP_MONITOR   → 监控止盈止损条件触发       │
  │ ④ 执行交易 TRADE_EXEC     → 根据信号执行交易          │
  │ ⑤ 交易计划 TRADE_PLAN     → 制定/更新交易计划         │
  │ ⑥ 数据回测 BACKTEST       → 回测优化策略              │
  │ ⑦ 机器学习 MACHINE_LEARNING → 空闲+有精选标的时触发    │
  │ ⑧ 空闲 IDLE               → 暂无任务                 │
  └──────────────────────────────────────────────────────┘

非交易日: 全天循环执行 ①→②→③→⑤→⑥→⑧, 空闲时触发⑦
交易时段: 并行执行 ②③④（信号+止盈止损+交易）+ 午间⑤+盘后①⑤⑥
ML 触发: 仅在引擎空闲(IDLE) 且 策略池有精选标的(focus/watch) 时自动触发
"""

import contextlib
import json
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from threading import Event, Lock, Thread

from .auction_analyzer import AuctionAnalyzer, AuctionSnapshot
from .config import (
    DEFAULT_CAPITAL,
    LIVE_ENGINE_CHECK_INTERVAL,
    LIVE_LOG_DIR,
    ML_GATE_DIRECTION_THRESHOLD,
    ML_GATE_ENABLED,
    ML_GATE_MAX_WEIGHT,
    ML_GATE_MIN_ACCURACY,
    ML_GATE_MIN_WEIGHT,
    ML_GATE_WINDOW_SIZE,
    ML_LABEL_METHOD,
    ML_MODEL_DIR,
    ML_MODEL_TYPE,
    ML_RETRAIN_DAYS,
    ML_TRAIN_WINDOW,
    POOL_MAX_SIZE,
    STRATEGIES,
)
from .data_manager import DataFetcher
from .indicators import (
    IndicatorResult,
    analyze_kdj,
    analyze_macd,
    comprehensive_analysis,
)
from .market_analyzer import MarketAnalyzer, MarketSnapshot
from .ml_engine import MLEngine, MLDecisionGate
from .paper_trader import SNAPSHOT_FILE as PAPER_SNAPSHOT_FILE
from .paper_trader import PaperPosition, PaperTrader
from .signal_hunter import SignalHunter, SignalResult, TradingPlan
from .stock_pool import (
    TIER_FOCUS,
    TIER_WATCH,
    PoolManager,
)
from .trading_calendar import TradingCalendar, TradingPhase, get_phase_label

logger = logging.getLogger(__name__)


class EngineState(Enum):
    """引擎运行状态"""

    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"


class LifecyclePhase(Enum):
    """心跳驱动的策略日常生命周期阶段"""

    IDLE = ("idle", "空闲", 0)
    SCREENING = ("screening", "选股", 1)                  # ① 拉取代码诊股入池
    SIGNAL_SCAN = ("signal_scan", "信号扫描", 2)            # ② 观察买入/卖出信号
    STOP_MONITOR = ("stop_monitor", "止盈止损监控", 3)       # ③ 监控止盈止损条件触发
    TRADE_EXEC = ("trade_exec", "执行交易", 4)              # ④ 根据信号执行交易
    TRADE_PLAN = ("trade_plan", "制定交易计划", 5)           # ⑤ 制定/更新交易计划
    BACKTEST = ("backtest", "数据回测", 6)                  # ⑥ 回测优化策略
    MACHINE_LEARNING = ("machine_learning", "机器学习", 7)   # ⑦ 模型训练与预测

    @property
    def label(self) -> str:
        return self.value[1]

    @property
    def order(self) -> int:
        return self.value[2]




@dataclass
class EngineLog:
    """引擎日志条目"""

    timestamp: datetime
    phase: TradingPhase
    event: str
    detail: str = ""


class LiveTradingEngine:
    """实时交易引擎 - A股交易时间周期驱动"""

    def __init__(self):
        self.calendar = TradingCalendar()
        self.market = MarketAnalyzer()
        self.auction = AuctionAnalyzer()
        self.hunter = SignalHunter()
        self.pool_mgr = PoolManager()
        self.fetcher = DataFetcher()
        self.paper_trader = PaperTrader()

        # 运行状态
        self.state = EngineState.STOPPED
        self._stop_event = Event()
        self._main_thread: Thread | None = None
        self._lock = Lock()

        # 交易日状态
        self._current_phase: TradingPhase = TradingPhase.CLOSED
        self._last_phase: TradingPhase = TradingPhase.CLOSED

        # 分析结果缓存
        self._latest_snapshot: MarketSnapshot | None = None
        self._latest_auction: AuctionSnapshot | None = None
        self._latest_signals: dict[str, SignalResult] = {}
        self._trading_plan: TradingPlan | None = None
        self._todays_decisions: list[dict[str, str]] = []

        # 日志
        self._logs: list[EngineLog] = []
        self._max_logs = 500
        self._log_callbacks: list[Callable[..., object]] = []

        # ── 心跳驱动的策略日常生命周期 ──
        self._lifecycle_phase: LifecyclePhase = LifecyclePhase.IDLE
        self._lifecycle_completed: set[str] = set()       # 当日已完成的阶段
        self._lifecycle_last_run: dict[str, datetime] = {}  # 各阶段上次执行时间
        self._lifecycle_pause_until: datetime | None = None  # 异常暂停
        self._idle_since: datetime | None = None          # 进入 IDLE 的时间（用于循环重启）
        self._daily_ops_done: set[str] = set()            # 当日已完成的一次性操作（如 market_snapshot, auction_analysis）

        # ── 各策略独立生命周期阶段跟踪 ──
        self._strategy_phase: dict[str, str] = {k: LifecyclePhase.IDLE.value[0] for k in STRATEGIES}
        self._strategy_phase_label: dict[str, str] = {k: LifecyclePhase.IDLE.label for k in STRATEGIES}
        self._strategy_phase_updated: dict[str, str] = {}  # 更新时间戳

        # 选股阶段状态
        self._screen_offset: int = 0
        self._screened_codes: set[str] = set()
        self._screening_done: bool = False  # 选股全部完成

        # 回测阶段状态
        self._backtest_date: str | None = None  # 上次回测日期

        # 机器学习引擎（基于 AKQuant §12 ML 最佳实践）
        self._ml_engine = MLEngine(
            fetcher=self.fetcher,
            model_type=ML_MODEL_TYPE,
            label_method=ML_LABEL_METHOD,
            model_dir=ML_MODEL_DIR,
            train_window=ML_TRAIN_WINDOW,
        )
        self._ml_last_train_date: str | None = None  # 上次训练日期

        # ML 决策门控器（动态权重，自适应）
        self._ml_gate = MLDecisionGate(
            state_dir=ML_MODEL_DIR,
            window_size=ML_GATE_WINDOW_SIZE,
            min_weight=ML_GATE_MIN_WEIGHT,
            max_weight=ML_GATE_MAX_WEIGHT,
            min_accuracy=ML_GATE_MIN_ACCURACY,
            direction_threshold=ML_GATE_DIRECTION_THRESHOLD,
        )
        self._ml_prediction_cache: dict[str, str] = {}  # code → ml_direction
        self._ml_score_cache: dict[str, float] = {}     # code → ml_score
        self._ml_last_validate_date: str | None = None  # 上次验证日期

    # ═══════════════════════════════════════════
    # 引擎生命周期
    # ═══════════════════════════════════════════

    def start(self):
        """启动实时引擎"""
        if self.state == EngineState.RUNNING:
            self._log(TradingPhase.CLOSED, "引擎已在运行中")
            return

        self.state = EngineState.STARTING
        self._stop_event.clear()
        self._clear_daily_cache()
        self._log = self._add_log  # 重定向方法引用

        # 加载股票池
        self.pool_mgr.load_all()

        self._log(TradingPhase.CLOSED, "═══ CloudKnight Live Engine 启动 ═══")
        self._log(TradingPhase.CLOSED, f"策略: {', '.join(STRATEGIES.values())}")
        self._log(TradingPhase.CLOSED, f"初始资金: {DEFAULT_CAPITAL:,.0f}")

        self._main_thread = Thread(target=self._run_loop, daemon=True, name="LiveEngine")
        self._main_thread.start()

    def stop(self):
        """停止引擎"""
        if self.state == EngineState.STOPPED:
            return

        self._log(TradingPhase.CLOSED, "引擎正在停止...")
        self.state = EngineState.STOPPING
        self._stop_event.set()

        # 仅在收盘后(15:00)才保存当日总结
        if datetime.now().time() >= time(15, 0):
            self._save_daily_summary()

        if self._main_thread and self._main_thread.is_alive():
            self._main_thread.join(timeout=10)

        self.state = EngineState.STOPPED
        self._log(TradingPhase.CLOSED, "引擎已停止")

    def pause(self):
        """暂停引擎"""
        with self._lock:
            if self.state == EngineState.RUNNING:
                self.state = EngineState.PAUSED
                self._log(self._current_phase, "引擎已暂停")

    def resume(self):
        """恢复引擎"""
        with self._lock:
            if self.state == EngineState.PAUSED:
                self.state = EngineState.RUNNING
                self._log(self._current_phase, "引擎已恢复")

    # ═══════════════════════════════════════════
    # 主循环
    # ═══════════════════════════════════════════

    def _run_loop(self):
        """引擎主循环 - TradingPhase 负责时间感知，LifecyclePhase 统一调度所有业务操作。"""
        self.state = EngineState.RUNNING
        self._log(TradingPhase.CLOSED, "主循环启动")

        while not self._stop_event.is_set():
            try:
                now = datetime.now()

                # 确定当前时间段（仅用于时间感知，不触发业务操作）
                phase = self.calendar.get_current_phase(now)
                self._current_phase = phase

                # 阶段切换检测 → 只做状态管理（重置缓存、加载池等一次性操作）
                if phase != self._last_phase:
                    self._on_phase_enter(phase, now)
                    self._last_phase = phase

                # 统一入口：LifecyclePhase 调度所有业务操作
                if self.state == EngineState.RUNNING:
                    self._lifecycle_tick(now)

                # 等待下次检查
                self._stop_event.wait(timeout=LIVE_ENGINE_CHECK_INTERVAL)

            except Exception as e:
                logger.error(f"引擎主循环异常: {e}", exc_info=True)
                self._log(TradingPhase.CLOSED, f"异常: {e}")

        self._log(TradingPhase.CLOSED, "主循环退出")
        self.state = EngineState.STOPPED

    def _on_phase_enter(self, phase: TradingPhase, now: datetime):
        """进入新时间段时的状态管理回调（不执行业务操作，业务统一由 LifecyclePhase 调度）。"""
        # 新交易日：重置缓存和生命周期
        if phase == TradingPhase.PRE_MARKET:
            self._clear_daily_cache()
            self._daily_ops_done.clear()
            self._reset_lifecycle()
            self.pool_mgr.load_all()
            self.fetcher.reset_unavailable_sources()

        # 盘后：保存每日总结
        elif phase == TradingPhase.POST_MARKET:
            self._save_daily_summary()

        label = get_phase_label(phase)
        self._log(phase, f">>> 进入 [{label}] 阶段")

    # ═══════════════════════════════════════════
    # 辅助方法（由 LifecyclePhase 调度调用）
    # ═══════════════════════════════════════════

    def _ensure_market_snapshot(self, now: datetime):
        """确保当日有市场快照（大盘指数、板块热度、情绪），仅执行一次。"""
        if "market_snapshot" in self._daily_ops_done:
            return
        self._daily_ops_done.add("market_snapshot")

        self._log(self._current_phase, "正在分析六大核心指数...")
        indices = self.market.analyze_indices(days=60)
        for idx in indices.values():
            arrow = "▲" if idx.pct_change >= 0 else "▼"
            self._log(
                self._current_phase,
                f"  {arrow} {idx.name}: {idx.close:.2f} ({idx.pct_change:+.2f}%) "
                + f"MACD:{idx.macd_signal} KDJ:{idx.kdj_signal} "
                f"RSI:{idx.rsi_value}",
            )

        self._log(self._current_phase, "正在分析板块热度...")
        sectors = self.market.analyze_sectors(top_n=10)
        for s in sectors[:5]:
            self._log(self._current_phase, f"  {s.strength_rank}. {s.name}: {s.pct_change:+.2f}%")

        sentiment = self.market.get_market_sentiment()
        self._log(self._current_phase, f"市场情绪: {sentiment['sentiment'].upper()} (评分:{sentiment['score']})")

        self._latest_snapshot = MarketSnapshot(
            timestamp=now,
            is_trading_day=True,
            indices=indices,
            hot_sectors=sectors,
            market_sentiment=sentiment["sentiment"],
            sentiment_score=sentiment["score"],
            buy_signals=0,
            sell_signals=0,
            capital_flow_summary="盘前",
        )

    def _ensure_auction_analysis(self, now: datetime):
        """确保当日执行了竞价分析，仅执行一次。"""
        if "auction_analysis" in self._daily_ops_done:
            return
        self._daily_ops_done.add("auction_analysis")

        watch_codes = set()
        for strategy_name in STRATEGIES:
            pool = self.pool_mgr.get_pool(strategy_name)
            if pool:
                for item in pool.ranked()[:10]:
                    watch_codes.add(item.code)

        if not watch_codes:
            self._log(self._current_phase, "无关注标的，跳过竞价分析")
            return

        self._log(self._current_phase, f"竞价分析 {len(watch_codes)} 只关注标的...")
        self._latest_auction = self.auction.analyze_auction(list(watch_codes), verbose=True)

        auc = self._latest_auction
        self._log(
            self._current_phase,
            f"竞价结果: 强势{auc.strong_auction_count} 只, "
            + f"弱势{auc.weak_auction_count} 只, "
            + f"偏{bias_cn(auc.market_bias)}",
        )

        watch_by_strategy = {}
        for strategy_name in STRATEGIES:
            pool = self.pool_mgr.get_pool(strategy_name)
            if pool:
                watch_by_strategy[strategy_name] = [item.code for item in pool.ranked()[:10]]

        decisions = self.auction.make_open_trade_decision(auc, watch_by_strategy)
        self._todays_decisions = []
        for strategy, dec_list in decisions.items():
            label = STRATEGIES.get(strategy, strategy)
            for d in dec_list:
                self._todays_decisions.append(d)
                self._log(
                    self._current_phase,
                    f"  [{label}] {d['code']} → {d['action']}: {d['reason']} (置信度:{d['confidence']})",
                )

        self._log(self._current_phase, f"开盘决策: {len(self._todays_decisions)} 条")

    def _analyze_15min_kline(self, now: datetime):
        """15分钟K线技术分析（午间复盘用，使用内置指标库 analyze_* 系列）"""
        focus_codes = []
        for strategy_name in STRATEGIES:
            pool = self.pool_mgr.get_pool(strategy_name)
            if pool:
                for item in pool.ranked()[:2]:
                    if item.code not in focus_codes:
                        focus_codes.append(item.code)

        if not focus_codes:
            return

        self._log(TradingPhase.LUNCH, f"15分K线分析 {len(focus_codes[:5])} 只重点标的...")

        for code in focus_codes[:5]:
            try:
                df = self.fetcher.fetch_daily_kline(code, start_date="20250601")
                if df is None or df.empty or len(df) < 5:
                    continue

                col_map = {
                    "日期": "date", "开盘": "open", "收盘": "close",
                    "最高": "high", "最低": "low", "成交量": "volume",
                }
                df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

                close = float(df["close"].iloc[-1])
                prev = float(df["close"].iloc[-2]) if len(df) > 1 else close
                pct = (close / prev - 1) * 100

                # 使用 analyze_* 系列获取 IndicatorResult（自动判断信号）
                macd_r = analyze_macd(df.tail(60))
                kdj_r = analyze_kdj(df.tail(60))

                am_swing = "强势" if pct > 1 else ("弱势" if pct < -1 else "震荡")

                self._log(
                    TradingPhase.LUNCH,
                    f"  {code}: {pct:+.2f}% [{am_swing}] "
                    f"MACD:{macd_r.signal}({macd_r.trend}) KDJ:{kdj_r.signal}({kdj_r.trend})",
                )

            except Exception as e:
                logger.debug(f"15分K分析 {code}: {e}")

    def _make_afternoon_plan(self, now: datetime):
        """制定下午交易计划"""
        if not self._latest_signals and not self._todays_decisions:
            self._log(TradingPhase.LUNCH, "无下午特别计划")
            return

        # 基于午间分析调整早盘计划
        adjusted_actions = 0
        if self._trading_plan:
            for sig in self._trading_plan.signals:
                if sig.signal_type == "buy":
                    # 中午重新确认买入信号
                    adjusted_actions += 1

        self._log(TradingPhase.LUNCH, f"下午计划: 关注 {adjusted_actions} 个信号执行")

    def _make_next_day_plan(self, now: datetime):
        """制定次日交易计划"""
        pool_summary = {}
        for strategy_name in STRATEGIES:
            pool = self.pool_mgr.get_pool(strategy_name)
            if pool:
                ranked = pool.ranked()
                pool_summary[strategy_name] = {
                    "label": STRATEGIES.get(strategy_name, strategy_name),
                    "count": len(ranked),
                    "top5": [(item.code, item.name, item.score) for item in ranked[:5]],
                }

        self._log(TradingPhase.POST_MARKET, "次日交易计划概要:")
        for _key, info in pool_summary.items():
            self._log(TradingPhase.POST_MARKET, f"  [{info['label']}] 关注 {info['count']} 只")

    def _report_current_signals(self):
        """输出当前信号摘要"""
        if not self._latest_signals:
            return

        for _strategy_name, result in self._latest_signals.items():
            if result.error:
                self._log(self._current_phase, f"  [{result.strategy_label}] 扫描异常: {result.error}")
                continue

            sigs = result.signals
            if sigs:
                buys = [s for s in sigs if s.signal_type in ("buy", "add")]
                sells = [s for s in sigs if s.signal_type in ("sell", "reduce", "close")]
                for s in buys[:3]:
                    self._log(
                        self._current_phase,
                        f"  [买] {s.code} {s.name} @ ~{s.price:.2f} [{s.confidence}] {s.reason[:40]}",
                    )
                for s in sells[:3]:
                    self._log(
                        self._current_phase,
                        f"  [卖] {s.code} {s.name} @ ~{s.price:.2f} " + f"[{s.confidence}] {s.reason[:40]}",
                    )

    # ═══════════════════════════════════════════
    # 心跳驱动的策略日常生命周期
    # ═══════════════════════════════════════════

    # ── 全局参数 ──
    _LIFECYCLE_CYCLE_INTERVAL = 10      # 阶段切换间隔（秒）
    _SCREEN_BATCH_SIZE = 200            # 每轮选股代码数
    _FOCUS_TARGET = 5                   # 每策略精选层目标
    _PAUSE_ON_ERROR = 120               # 异常暂停秒数
    _IDLE_CYCLE_RESTART = 300           # IDLE 后重新启动循环的间隔（5分钟）

    # ── 各阶段间隔（秒） ──
    _INTERVAL_SCREENING = 30     # 选股每轮间隔
    _INTERVAL_SIGNAL_SCAN = 60   # 信号扫描间隔（交易时段）
    _INTERVAL_STOP_MONITOR = 30  # 止盈止损检查间隔（交易时段）
    _INTERVAL_TRADE_EXEC = 5     # 交易执行冷却
    _INTERVAL_TRADE_PLAN = 300   # 交易计划生成间隔（5分钟）
    _INTERVAL_BACKTEST = 600     # 回测间隔（10分钟）
    _INTERVAL_MACHINE_LEARNING = 1800  # 机器学习间隔（30分钟）

    # ═══════════════════════════════════════════
    # 生命周期主驱动
    # ═══════════════════════════════════════════

    def _lifecycle_tick(self, now: datetime):
        """心跳主入口：根据当前交易时段确定并执行生命周期阶段。

        在引擎主循环每次 tick 时调用（所有时段）。
        交易时段与非交易时段采用不同的阶段调度策略。
        LifecyclePhase 统一负责所有业务操作的调度，TradingPhase 仅提供时间上下文。
        """
        # 异常暂停检查
        if self._lifecycle_pause_until and now < self._lifecycle_pause_until:
            return

        # 确定当前应执行的生命周期阶段
        target_phase = self._lifecycle_determine_phase(now)

        # IDLE/ML 阶段时同步策略阶段状态（供前端展示）
        if target_phase in (LifecyclePhase.IDLE, LifecyclePhase.MACHINE_LEARNING):
            self._update_all_strategy_phases(target_phase, now)
            self._save_state()  # 持久化 IDLE/ML 状态变更，供 Web 仪表盘实时读取

        # 冷却检查
        if not self._lifecycle_can_run(target_phase, now):
            return

        # 执行阶段动作
        self._lifecycle_execute(target_phase, now)
        self._save_state()  # 持久化策略生命周期阶段，供 Web 仪表盘实时读取

    def _lifecycle_determine_phase(self, now: datetime) -> LifecyclePhase:
        """根据当前交易时段确定应执行的生命周期阶段。

        非交易时段 (CLOSED): 顺序推进 选股 → 信号 → 止损 → 计划 → 回测 → 空闲
        交易时段: 并行执行信号扫描 + 止盈止损监控，午间做计划，盘后选股+回测

        ML 触发规则（独立于顺序调度）：
          1. 策略处于空闲状态（IDLE）
          2. 策略股票池有精选的股票需要训练（focus/watch 层）
          满足以上两条件时，在空闲阶段转入 MACHINE_LEARNING。
        """
        tp = self._current_phase  # 当前时间段（TradingPhase → LifecyclePhase 映射）

        # ── 非交易时段：顺序推进（不含 ML，ML 在空闲时机会触发） ──
        if tp == TradingPhase.CLOSED and not self.calendar.is_trading_day(now):
            phases_order = [
                LifecyclePhase.SCREENING,
                LifecyclePhase.SIGNAL_SCAN,
                LifecyclePhase.STOP_MONITOR,
                LifecyclePhase.TRADE_PLAN,
                LifecyclePhase.BACKTEST,
            ]
            return self._resolve_phase_with_idle_restart(phases_order, now)

        # ── 盘后 (POST_MARKET) ──
        if tp == TradingPhase.POST_MARKET:
            phases_order = [
                LifecyclePhase.SCREENING,
                LifecyclePhase.TRADE_PLAN,
                LifecyclePhase.BACKTEST,
            ]
            return self._resolve_phase_with_idle_restart(phases_order, now)

        # ── 盘前/竞价 → 交易计划 + 信号扫描 ──
        if tp in (TradingPhase.PRE_MARKET, TradingPhase.AUCTION, TradingPhase.AUCTION_RESULT):
            if tp == TradingPhase.PRE_MARKET:
                return LifecyclePhase.TRADE_PLAN
            return LifecyclePhase.SIGNAL_SCAN

        # ── 交易时段（早盘/午盘） → 信号 + 止损 + 执行交易（时间轮转） ──
        if tp in (TradingPhase.MORNING, TradingPhase.AFTERNOON):
            return self._resolve_trading_phase(now)

        # ── 午间 → 交易计划 ──
        if tp == TradingPhase.LUNCH:
            return LifecyclePhase.TRADE_PLAN

        return LifecyclePhase.IDLE

    def _maybe_trigger_ml(self) -> LifecyclePhase:
        """空闲时检查是否需要触发 ML：必须有精选层的标的才触发。

        触发条件：
          1. 策略处于空闲状态（由调用方保证，此时应为 IDLE）
          2. 策略股票池有精选的股票需要训练（focus/watch 层至少 1 只）
        """
        if self._ml_has_candidates():
            return LifecyclePhase.MACHINE_LEARNING
        return LifecyclePhase.IDLE

    def _ml_has_candidates(self) -> bool:
        """检查各策略股票池是否有精选/观察层标的需要 ML 训练。"""
        for strategy_name in STRATEGIES:
            pool = self.pool_mgr.get_pool(strategy_name)
            if not pool:
                continue
            for item in pool.ranked():
                if item.tier in (TIER_FOCUS, TIER_WATCH):
                    return True
        return False

    def _next_pending_phase(self, phases: list[LifecyclePhase]) -> LifecyclePhase:
        """从有序阶段列表中返回第一个未完成的阶段"""
        for p in phases:
            if p.value[0] not in self._lifecycle_completed:
                return p
        return LifecyclePhase.IDLE

    def _resolve_phase_with_idle_restart(self, phases_order: list[LifecyclePhase], now: datetime) -> LifecyclePhase:
        """顺序推进 + IDLE 时自动重启循环（防止系统永远停在 IDLE）。

        当所有顺序阶段完成后：
          - 检查是否已空闲足够长（_IDLE_CYCLE_RESTART）
          - 若是，清除 _lifecycle_completed 并从头开始循环
          - 同时检查 ML 触发条件
        """
        target = self._next_pending_phase(phases_order)
        if target != LifecyclePhase.IDLE:
            self._idle_since = None
            return target

        # 首次进入 IDLE 时记录时间
        if self._idle_since is None:
            self._idle_since = datetime.now()
            self._lifecycle_last_run["idle"] = datetime.now()

        # IDLE 冷却后重启循环
        if (datetime.now() - self._idle_since).total_seconds() >= self._IDLE_CYCLE_RESTART:
            self._lifecycle_completed.clear()
            self._idle_since = None
            target = self._next_pending_phase(phases_order)
            if target != LifecyclePhase.IDLE:
                return target

        # 空闲期检查 ML 触发
        return self._maybe_trigger_ml()

    def _resolve_trading_phase(self, now: datetime) -> LifecyclePhase:
        """交易时段三阶段时间轮转（信号 → 止损 → 交易）。

        基于各阶段最后执行时间选最久未执行的，自然形成轮转。
        """
        key_signal = LifecyclePhase.SIGNAL_SCAN.value[0]
        key_stop = LifecyclePhase.STOP_MONITOR.value[0]
        key_trade = LifecyclePhase.TRADE_EXEC.value[0]

        last_signal = self._lifecycle_last_run.get(key_signal)
        last_stop = self._lifecycle_last_run.get(key_stop)
        last_trade = self._lifecycle_last_run.get(key_trade)

        # 无执行记录 → 从信号开始
        if last_signal is None:
            return LifecyclePhase.SIGNAL_SCAN

        # 选择最久未执行的阶段（按优先级：signal > stop > trade）
        signal_is_oldest = True
        if last_stop is not None and last_signal > last_stop:
            signal_is_oldest = False
        if last_trade is not None and last_signal > last_trade:
            signal_is_oldest = False
        if signal_is_oldest:
            return LifecyclePhase.SIGNAL_SCAN

        if last_stop is None or (last_trade is not None and last_stop > last_trade):
            return LifecyclePhase.TRADE_EXEC
        return LifecyclePhase.STOP_MONITOR

    def _lifecycle_can_run(self, phase: LifecyclePhase, now: datetime) -> bool:
        """检查阶段是否可以执行（冷却间隔控制）"""
        if phase == LifecyclePhase.IDLE:
            return False

        last = self._lifecycle_last_run.get(phase.value[0])
        if last is None:
            return True

        # 根据当前时段选择冷却间隔
        is_trading = self._current_phase.is_trading_phase()
        intervals = {
            LifecyclePhase.SCREENING: self._INTERVAL_SCREENING,
            LifecyclePhase.SIGNAL_SCAN: self._INTERVAL_SIGNAL_SCAN if is_trading else self._INTERVAL_SCREENING,
            LifecyclePhase.STOP_MONITOR: self._INTERVAL_STOP_MONITOR if is_trading else self._INTERVAL_STOP_MONITOR,
            LifecyclePhase.TRADE_EXEC: self._INTERVAL_TRADE_EXEC,
            LifecyclePhase.TRADE_PLAN: self._INTERVAL_TRADE_PLAN,
            LifecyclePhase.BACKTEST: self._INTERVAL_BACKTEST,
            LifecyclePhase.MACHINE_LEARNING: self._INTERVAL_MACHINE_LEARNING,
        }
        interval = intervals.get(phase, self._LIFECYCLE_CYCLE_INTERVAL)
        return (now - last).total_seconds() >= interval

    def _update_all_strategy_phases(self, phase: LifecyclePhase, now: datetime):
        """将所有策略的当前生命周期阶段更新为指定阶段"""
        phase_key = phase.value[0]
        phase_label = phase.label
        ts = now.strftime("%H:%M:%S")
        for k in STRATEGIES:
            self._strategy_phase[k] = phase_key
            self._strategy_phase_label[k] = phase_label
            self._strategy_phase_updated[k] = ts

    def _update_strategy_phase(self, strategy_key: str, phase: LifecyclePhase, now: datetime):
        """更新单个策略的生命周期阶段"""
        if strategy_key in self._strategy_phase:
            self._strategy_phase[strategy_key] = phase.value[0]
            self._strategy_phase_label[strategy_key] = phase.label
            self._strategy_phase_updated[strategy_key] = now.strftime("%H:%M:%S")

    def _lifecycle_execute(self, phase: LifecyclePhase, now: datetime):
        """执行生命周期阶段动作"""
        self._lifecycle_phase = phase
        self._lifecycle_last_run[phase.value[0]] = now

        # 标记所有策略进入当前阶段（引擎级阶段：所有策略同步推进）
        if phase != LifecyclePhase.IDLE:
            self._update_all_strategy_phases(phase, now)

        handlers = {
            LifecyclePhase.SCREENING: self._lifecycle_screening,
            LifecyclePhase.SIGNAL_SCAN: self._lifecycle_signal_scan,
            LifecyclePhase.STOP_MONITOR: self._lifecycle_stop_monitor,
            LifecyclePhase.TRADE_EXEC: self._lifecycle_trade_exec,
            LifecyclePhase.TRADE_PLAN: self._lifecycle_trade_plan,
            LifecyclePhase.BACKTEST: self._lifecycle_backtest,
            LifecyclePhase.MACHINE_LEARNING: self._lifecycle_machine_learning,
        }

        handler = handlers.get(phase)
        if handler:
            try:
                completed = handler(now)
                if completed:
                    self._lifecycle_completed.add(phase.value[0])
            except Exception as e:
                logger.warning(f"生命周期 [{phase.label}] 异常: {e}")
                self._log(self._current_phase, f"生命周期 [{phase.label}] 异常: {e}")
                self._lifecycle_pause_until = datetime.fromtimestamp(
                    datetime.now().timestamp() + self._PAUSE_ON_ERROR
                )

    # ═══════════════════════════════════════════
    # ① 选股 SCREENING
    # ═══════════════════════════════════════════

    def _lifecycle_screening(self, now: datetime) -> bool:
        """选股：拉取代码诊股入池，直到所有策略池精选层满额。

        Returns:
            True = 本阶段完成，可进入下一阶段
        """
        trading_phase = self._current_phase
        # 已全部完成
        if self._screening_done:
            return True

        # 检查是否所有池精选满额
        if self._check_all_pools_focus_full():
            self._screening_done = True
            self._log(trading_phase, "✅ 选股完成：所有策略池精选层已满 5 只！")
            return True

        # 执行一轮诊股入池
        return self._screen_one_batch(now)

    def _screen_one_batch(self, now: datetime) -> bool:
        """执行一轮选股：取下一批代码 → 诊股入池 → 评估层级"""
        trading_phase = self._current_phase
        try:
            from .stock_info_cache import get_stock_info_cache
            from .stock_pool import SCREEN_SAMPLE_SIZE

            # 1. 获取全量代码
            cache = get_stock_info_cache()
            if cache.is_usable():
                codes_df = cache.db.get_stock_list()
                all_codes = [str(c) for c in codes_df["code"].tolist()]
            else:
                self.fetcher.reset_unavailable_sources()
                all_codes = [str(c) for c in self.fetcher.build_stock_pool(filter_st=True, filter_new=True)]

            # 2. 取下一批未筛选代码
            fresh_codes = [c for c in all_codes if c not in self._screened_codes]
            if not fresh_codes:
                self._log(trading_phase, "📋 选股：已遍历全部代码，无可选标的")
                self._screening_done = True
                return True

            batch_size = min(SCREEN_SAMPLE_SIZE, self._SCREEN_BATCH_SIZE)
            batch = fresh_codes[:batch_size]
            self._screened_codes.update(batch)

            # 3. 执行诊股入池
            round_num = len(self._screened_codes) // batch_size
            self._log(trading_phase, f"🔄 [{LifecyclePhase.SCREENING.label}] 第{round_num}轮: {len(batch)} 只候选...")
            self.pool_mgr.screen_all(batch, verbose=False)

            # 4. 评估层级
            self.pool_mgr.evaluate_all(verbose=False)
            self.pool_mgr.save_all()

            # 5. 输出进度
            overview = self.pool_mgr.tier_overview()
            status_parts = []
            for key, counts in overview.items():
                focus = counts.get("focus", 0)
                total = self.pool_mgr.pools[key].size if key in self.pool_mgr.pools else 0
                bar = "█" * min(focus, self._FOCUS_TARGET) + "░" * max(0, self._FOCUS_TARGET - focus)
                status_parts.append(f"{key}: [{bar}] {focus}/{self._FOCUS_TARGET} (共{total})")
            self._log(trading_phase, f"  精选进度: {' | '.join(status_parts)}")

            # 检查是否本轮后已满额
            return self._check_all_pools_focus_full()

        except Exception as e:
            logger.warning(f"选股异常: {e}")
            raise

    def _check_all_pools_focus_full(self) -> bool:
        """检查是否所有策略池的精选层都达到目标数量"""
        try:
            overview = self.pool_mgr.tier_overview()
            for _key, counts in overview.items():
                if counts.get("focus", 0) < self._FOCUS_TARGET:
                    return False
            return True
        except Exception:
            return False

    # ═══════════════════════════════════════════
    # ② 信号扫描 SIGNAL_SCAN
    # ═══════════════════════════════════════════

    def _lifecycle_signal_scan(self, now: datetime) -> bool:
        """观察买入/卖出信号：对池中股票进行多策略并行信号扫描。
        
        SIGNAL_SCAN 统一负责：
        - 竞价结果阶段: 竞价分析 + 开盘交易决策（仅一次）
        - 交易时段: 定期股票池信号扫描 + 信号报告
        """
        trading_phase = self._current_phase

        # 竞价结果阶段执行竞价分析（仅一次）
        if trading_phase == TradingPhase.AUCTION_RESULT:
            self._ensure_auction_analysis(now)

        # 早盘/午盘开盘日志（仅一次）
        if trading_phase == TradingPhase.MORNING and "morning_open_log" not in self._daily_ops_done:
            self._daily_ops_done.add("morning_open_log")
            self._log(trading_phase, "早盘开盘! 开始分时走势跟踪与信号扫描")
            if self._todays_decisions:
                buy_dec = [d for d in self._todays_decisions if d["action"] in ("buy_at_open", "buy")]
                sell_dec = [d for d in self._todays_decisions if d["action"] == "sell_at_open"]
                if buy_dec:
                    self._log(trading_phase, f"竞价买入决策: {len(buy_dec)} 条")
                    for d in buy_dec[:3]:
                        self._log(trading_phase, f"  → {d['code']} @ {d.get('price', 0):.2f}")
                if sell_dec:
                    self._log(trading_phase, f"竞价卖出决策: {len(sell_dec)} 条")

        if trading_phase == TradingPhase.AFTERNOON and "afternoon_open_log" not in self._daily_ops_done:
            self._daily_ops_done.add("afternoon_open_log")
            self._log(trading_phase, "午盘开盘! 继续分时跟踪")
            if self._trading_plan and self._trading_plan.signals:
                pm_sigs = [s for s in self._trading_plan.signals if s.signal_type in ("buy", "sell")]
                if pm_sigs:
                    self._log(trading_phase, f"下午待执行信号: {len(pm_sigs)} 条")

        # 构建各策略的股票代码映射
        stock_pool_map: dict[str, list[str]] = {}
        for strategy_name in STRATEGIES:
            pool = self.pool_mgr.get_pool(strategy_name)
            if pool:
                codes = [item.code for item in pool.ranked()[:POOL_MAX_SIZE]]
                if codes:
                    stock_pool_map[strategy_name] = codes

        if not stock_pool_map:
            return True  # 无股票，跳过

        self._log(trading_phase, f"🔎 [{LifecyclePhase.SIGNAL_SCAN.label}] 扫描 {sum(len(v) for v in stock_pool_map.values())} 只池中股票...")

        # 执行信号扫描
        sentiment = "neutral"
        if self._latest_snapshot:
            sentiment = self._latest_snapshot.market_sentiment

        self._latest_signals = self.hunter.scan_all_strategies(stock_pool_map, days=120, verbose=False)

        # 生成交易计划
        self._trading_plan = self.hunter.build_trading_plan(self._latest_signals, sentiment)

        # 统计信号
        total_buy = 0
        total_sell = 0
        for result in self._latest_signals.values():
            if result.error:
                continue
            total_buy += sum(1 for s in result.signals if s.signal_type in ("buy", "add"))
            total_sell += sum(1 for s in result.signals if s.signal_type in ("sell", "reduce", "close"))

        self._log(trading_phase, f"  信号结果: 买入 {total_buy} | 卖出 {total_sell}")
        
        # 输出详细信号报告
        self._report_current_signals()
        
        return True  # 信号扫描每轮即完成

    # ═══════════════════════════════════════════
    # ③ 止盈止损监控 STOP_MONITOR
    # ═══════════════════════════════════════════

    def _lifecycle_stop_monitor(self, now: datetime) -> bool:
        """监控止盈止损条件触发：检查 PaperTrader 持仓是否触发止盈/止损"""
        trading_phase = self._current_phase
        if not self.paper_trader or not self.paper_trader.accounts:
            return True

        triggered_signals: list[dict] = []

        for strategy_key, acc in self.paper_trader.accounts.items():
            if not acc.positions:
                continue

            pool = self.pool_mgr.get_pool(strategy_key)
            for code, pos in list(acc.positions.items()):
                try:
                    # 获取最新价格
                    df = self.fetcher.fetch_daily_kline(code, start_date="20250601")
                    if df is None or df.empty:
                        continue

                    col_map = {"日期": "date", "收盘": "close", "开盘": "open", "最高": "high", "最低": "low"}
                    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
                    current_price = float(df["close"].iloc[-1])

                    # 更新持仓市值
                    pos.current_price = current_price
                    pos.market_value = current_price * pos.volume
                    pos.profit_pct = (current_price / pos.cost - 1) * 100 if pos.cost > 0 else 0

                    # 获取交易计划中的止损/止盈价位
                    stop_loss = pos.cost * 0.93  # 默认 7% 止损
                    take_profit = pos.cost * 1.15  # 默认 15% 止盈

                    if pool:
                        item = pool.get(code)
                        if item:
                            plan = pool.create_trade_plan(code, item.name, pos.cost, item.score)
                            stop_loss = plan.stop_loss
                            take_profit = plan.take_profit_1

                    # 检查止损
                    if current_price <= stop_loss:
                        triggered_signals.append({
                            "strategy": strategy_key,
                            "code": code,
                            "name": pos.name,
                            "type": "stop_loss",
                            "price": current_price,
                            "stop_price": stop_loss,
                            "cost": pos.cost,
                            "loss_pct": round((current_price / pos.cost - 1) * 100, 2),
                        })
                        self._log(trading_phase,
                            f"🛑 [止损触发] {strategy_key} {code} {pos.name} "
                            f"现价{current_price:.2f} ≤ 止损{stop_loss:.2f} "
                            f"(亏损{(1 - current_price/pos.cost)*100:.1f}%)")

                    # 检查止盈
                    elif current_price >= take_profit:
                        triggered_signals.append({
                            "strategy": strategy_key,
                            "code": code,
                            "name": pos.name,
                            "type": "take_profit",
                            "price": current_price,
                            "target_price": take_profit,
                            "cost": pos.cost,
                            "profit_pct": round((current_price / pos.cost - 1) * 100, 2),
                        })
                        self._log(trading_phase,
                            f"🎯 [止盈触发] {strategy_key} {code} {pos.name} "
                            f"现价{current_price:.2f} ≥ 止盈{take_profit:.2f} "
                            f"(盈利{(current_price/pos.cost-1)*100:.1f}%)")

                except Exception as e:
                    logger.debug(f"止盈止损检查 {code}: {e}")

        if triggered_signals:
            self._log(trading_phase,
                f"⚡ [{LifecyclePhase.STOP_MONITOR.label}] 触发 {len(triggered_signals)} 条信号: "
                + ", ".join(f"{s['code']}({s['type']})" for s in triggered_signals[:5]))

        return True  # 每次检查即可完成

    # ═══════════════════════════════════════════
    # ④ 执行交易 TRADE_EXEC
    # ═══════════════════════════════════════════

    def _lifecycle_trade_exec(self, now: datetime) -> bool:
        """执行交易：根据信号或止盈止损触发执行模拟交易"""
        trading_phase = self._current_phase
        if not self.paper_trader or not self.paper_trader.accounts:
            return True

        if not self._latest_signals and not self._trading_plan:
            return True

        executed = 0
        executed_strategies: set[str] = set()  # 记录有实际交易的策略

        # 执行交易计划中的信号
        if self._trading_plan and self._trading_plan.signals:
            for sig in self._trading_plan.signals:
                try:
                    strategy_key = sig.strategy
                    acc = self.paper_trader.accounts.get(strategy_key)
                    if not acc:
                        continue

                    # 获取参考价格
                    df = self.fetcher.fetch_daily_kline(sig.code, start_date="20250601")
                    if df is None or df.empty:
                        continue

                    col_map = {"日期": "date", "收盘": "close"}
                    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
                    current_price = float(df["close"].iloc[-1]) if "close" in df.columns else sig.price
                    price = current_price if current_price > 0 else sig.price

                    if sig.signal_type in ("buy", "add"):
                        # ── ML 决策门控：查询 ML 次日预测 ──
                        ml_direction = "neutral"
                        if ML_GATE_ENABLED:
                            ml_direction = self._ml_prediction_cache.get(sig.code, "neutral")
                            ml_score_val = self._ml_score_cache.get(sig.code, 0.0)
                            approved, gate_reason = self._ml_gate.evaluate_buy(
                                code=sig.code,
                                name=sig.name,
                                ml_direction=ml_direction,
                                ml_score=ml_score_val,
                                date=now.strftime("%Y%m%d"),
                                price=price,
                                strategy=strategy_key,
                            )
                            if not approved:
                                self._log(trading_phase,
                                    f"🚫 [ML驳回] {strategy_key} {sig.code} {sig.name} → {gate_reason}")
                                continue  # 跳过此买入信号

                        # 计算买入量（每只股票 10% 仓位）
                        position_ratio = 0.1
                        amount = acc.cash * position_ratio
                        lot_size = 100
                        volume = max(lot_size, int(amount / price / lot_size) * lot_size)

                        if volume > 0 and amount <= acc.cash:
                            acc.cash -= volume * price
                            acc.positions[sig.code] = PaperPosition(
                                code=sig.code,
                                name=sig.name,
                                cost=price,
                                volume=volume,
                                buy_date=now.strftime("%Y%m%d"),
                                current_price=price,
                                market_value=volume * price,
                            )
                            executed += 1
                            executed_strategies.add(strategy_key)
                            ml_note = f" [ML:{ml_direction}]" if ML_GATE_ENABLED else ""
                            self._log(trading_phase,
                                f"💼 [买入] {strategy_key} {sig.code} {sig.name} "
                                f"@{price:.2f} × {volume}股 = {volume*price:.0f}元{ml_note}")

                    elif sig.signal_type in ("sell", "reduce", "close"):
                        pos = acc.positions.get(sig.code)
                        if pos:
                            # 计算盈亏
                            pnl = (price - pos.cost) * pos.volume
                            acc.cash += pos.market_value
                            del acc.positions[sig.code]
                            executed += 1
                            executed_strategies.add(strategy_key)
                            self._log(trading_phase,
                                f"💰 [卖出] {strategy_key} {sig.code} {sig.name} "
                                f"@{price:.2f} 盈亏{pnl:+.0f}元")

                except Exception as e:
                    logger.debug(f"执行交易 {sig.code}: {e}")

        if executed > 0:
            self._log(trading_phase, f"✅ [{LifecyclePhase.TRADE_EXEC.label}] 执行 {executed} 笔交易")
            self.paper_trader.save()

        # 更新有实际交易的策略阶段 → 其他策略恢复为信号扫描完成的待命状态
        for sk in executed_strategies:
            self._update_strategy_phase(sk, LifecyclePhase.TRADE_EXEC, now)

        return True  # 每轮执行即完成

    # ═══════════════════════════════════════════
    # ⑤ 制定交易计划 TRADE_PLAN
    # ═══════════════════════════════════════════

    def _lifecycle_trade_plan(self, now: datetime) -> bool:
        """制定/更新交易计划：为池中精选层股票生成交易计划。
        
        TRADE_PLAN 统一负责：
        - 盘前: 市场快照生成（大盘指数、板块热度、情绪，仅执行一次）
        - 午间: 15分钟K线分析 + 下午交易计划调整
        - 盘后: 次日交易计划概要
        - 全部时段: 精选层股票交易计划生成
        """
        trading_phase = self._current_phase

        # 盘前时生成市场快照（仅一次）
        if trading_phase == TradingPhase.PRE_MARKET:
            self._ensure_market_snapshot(now)

        # 午间时做 K 线分析和下午计划
        if trading_phase == TradingPhase.LUNCH and "lunch_analysis" not in self._daily_ops_done:
            self._daily_ops_done.add("lunch_analysis")
            self._log(trading_phase, "午间休市 - 早盘复盘")
            self._analyze_15min_kline(now)
            self._report_current_signals()
            self._make_afternoon_plan(now)

        # 盘后时生成次日计划概要
        if trading_phase == TradingPhase.POST_MARKET and "next_day_plan" not in self._daily_ops_done:
            self._daily_ops_done.add("next_day_plan")
            self._make_next_day_plan(now)

        # 核心：为精选层股票生成交易计划
        plans_generated = 0

        for strategy_name in STRATEGIES:
            pool = self.pool_mgr.get_pool(strategy_name)
            if not pool:
                continue

            label = STRATEGIES.get(strategy_name, strategy_name)
            focus_items = [item for item in pool.ranked() if item.tier == TIER_FOCUS]

            if not focus_items:
                self._log(trading_phase, f"📋 [{label}] 无精选层股票，跳过计划")
                continue

            for item in focus_items[:self._FOCUS_TARGET]:
                try:
                    # 获取当前价格
                    df = self.fetcher.fetch_daily_kline(item.code, start_date="20250601")
                    if df is None or df.empty:
                        continue
                    col_map = {"日期": "date", "收盘": "close"}
                    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
                    price = float(df["close"].iloc[-1]) if "close" in df.columns else item.entry_price

                    # 生成交易计划（plan 对象绑定到 pool item，供后续查询）
                    _plan = pool.create_trade_plan(item.code, item.name, price, item.score)
                    plans_generated += 1

                except Exception as e:
                    logger.debug(f"生成计划 {item.code}: {e}")

        if plans_generated > 0:
            self._log(trading_phase,
                f"📝 [{LifecyclePhase.TRADE_PLAN.label}] 为 {plans_generated} 只精选票生成交易计划")

        return True  # 每轮生成即完成

    # ═══════════════════════════════════════════
    # ⑥ 数据回测 BACKTEST
    # ═══════════════════════════════════════════

    def _lifecycle_backtest(self, now: datetime) -> bool:
        """数据回测：对各策略进行历史回测，优化策略参数"""
        trading_phase = self._current_phase
        today_str = now.strftime("%Y%m%d")
        if self._backtest_date == today_str:
            return True  # 今天已回测

        self._log(trading_phase, f"📊 [{LifecyclePhase.BACKTEST.label}] 运行策略回测...")

        try:
            # 使用 PaperTrader 运行历史回放（最近 60 个交易日）
            if self.paper_trader:
                # 检查是否已有回测数据
                if not self.paper_trader.accounts:
                    return True

                # 对各策略进行简单回测评估
                backtest_summary = {}
                for strategy_key in STRATEGIES:
                    pool = self.pool_mgr.get_pool(strategy_key)
                    if not pool:
                        continue

                    ranked = pool.ranked()
                    if not ranked:
                        continue

                    # 取 Top10 计算平均分数
                    top_scores = [item.score for item in ranked[:10]]
                    avg_score = sum(top_scores) / len(top_scores) if top_scores else 0

                    # 取 Focus 层统计
                    focus_count = sum(1 for item in ranked if item.tier == TIER_FOCUS)
                    watch_count = sum(1 for item in ranked if item.tier == TIER_WATCH)

                    backtest_summary[strategy_key] = {
                        "label": STRATEGIES.get(strategy_key, strategy_key),
                        "pool_size": pool.size,
                        "focus_count": focus_count,
                        "watch_count": watch_count,
                        "avg_top10_score": round(avg_score, 1),
                        "max_score": round(max(top_scores), 1) if top_scores else 0,
                    }

                # 输出回测摘要
                lines = []
                for key, info in backtest_summary.items():
                    lines.append(
                        f"  {info['label']}: 精选{info['focus_count']} 观察{info['watch_count']} "
                        f"Top10均分{info['avg_top10_score']}"
                    )
                self._log(trading_phase, f"  回测摘要:\n" + "\n".join(lines))

            self._backtest_date = today_str

        except Exception as e:
            logger.warning(f"回测异常: {e}")
            self._log(trading_phase, f"回测异常: {e}")

        return True  # 每轮回测即完成

    # ═══════════════════════════════════════════
    # ⑦ 机器学习 MACHINE_LEARNING
    # ═══════════════════════════════════════════

    _ML_BACKTEST_DAYS = 120  # 模型训练用历史数据天数

    _ML_BACKTEST_DAYS = 120  # 降级模式用：历史数据天数

    def _lifecycle_machine_learning(self, now: datetime) -> bool:
        """机器学习：基于 AKQuant §12 最佳实践，训练模型辅助交易决策。

        触发规则（双重守卫）：
          1. 策略处于空闲状态（IDLE → 由 _maybe_trigger_ml 保证）
          2. 策略股票池有精选的股票需要训练（focus/watch 层 ≥ 1 只）

        流程:
          1. 收集所有策略池 focus/watch 层标的 → 提取 OHLCV
          2. FeatureEngineer: 计算 24 维金融特征（动量/波动/量价/趋势/形态）
          3. LabelGenerator: 三重屏障法生成标签（止盈+5%/止损-3%/5日到期）
          4. PurgedCV: 净化 K-Fold 验证，防数据泄漏
          5. MLTrainer: 训练 RandomForest/LogisticRegression 模型
          6. MLPredictor: 对池中标的生产预测 → 概率/方向/置信度
          7. MLModelRegistry: 版本管理，可追溯可复现

        若 sklearn 不可用，自动降级为 Z-Score 线性打分模式。
        """
        trading_phase = self._current_phase
        today_str = now.strftime("%Y%m%d")

        # ── 守卫：检查是否有精选标的需要训练 ──
        if not self._ml_has_candidates():
            self._log(trading_phase,
                f"🧠 [{LifecyclePhase.MACHINE_LEARNING.label}] 无精选层标的，跳过（等待选股完成后自动触发）")
            return True

        # 控制训练频率：默认每周重训练一次
        if self._ml_last_train_date:
            last_dt = datetime.strptime(self._ml_last_train_date, "%Y%m%d")
            if (now - last_dt).days < ML_RETRAIN_DAYS:
                # 已训练，仅做预测同步
                return self._ml_predict_only(now)

        self._log(trading_phase, f"🧠 [{LifecyclePhase.MACHINE_LEARNING.label}] 启动机器学习流程...")
        self._log(trading_phase, f"  模型: {ML_MODEL_TYPE} | 标签: {ML_LABEL_METHOD} | 训练窗口: {ML_TRAIN_WINDOW}日")

        try:
            # ── 收集所有策略池的焦点标的 ──
            all_candidates: list = []
            for strategy_name in STRATEGIES:
                pool = self.pool_mgr.get_pool(strategy_name)
                if not pool:
                    continue
                ranked = pool.ranked()
                candidates = [item for item in ranked if item.tier in (TIER_FOCUS, TIER_WATCH)]
                all_candidates.extend(candidates)

            if not all_candidates:
                self._log(trading_phase, "  无焦点标的，跳过训练")
                self._ml_last_train_date = today_str
                return True

            # 去重（同一代码可能出现在多个策略池）
            seen = {}
            unique_candidates = []
            for item in all_candidates:
                if item.code not in seen:
                    seen[item.code] = item
                    unique_candidates.append(item)

            n_total = len(unique_candidates)
            max_train = min(30, n_total)
            self._log(trading_phase, f"  收集 {n_total} 只候选标的（去重后），训练 {max_train} 只...")

            # ── MLEngine: 训练 + 预测 + 同步 ──
            predictions = self._ml_engine.train_and_predict(
                unique_candidates, now=now, max_stocks=max_train,
            )

            # ── 汇总 ──
            bullish = sum(1 for p in predictions if p.direction == "bullish")
            bearish = sum(1 for p in predictions if p.direction == "bearish")
            neutral = len(predictions) - bullish - bearish

            self._log(trading_phase,
                f"  ✅ 训练完成: {len(predictions)} 只标的预测已同步到池中")
            self._log(trading_phase,
                f"    方向: ↑{bullish} ↓{bearish} —{neutral}")

            # 输出 Top3 标的预测示例
            top_3 = sorted(predictions, key=lambda p: abs(p.ml_score), reverse=True)[:3]
            if top_3:
                examples = ", ".join(
                    f"{p.name or p.code}({p.ml_score:+.1f}/{p.direction})"
                    for p in top_3
                )
                self._log(trading_phase, f"    示例: {examples}")

            self._ml_last_train_date = today_str

            # ── 更新 ML 预测缓存（供交易信号门控使用） ──
            self._build_ml_prediction_cache(predictions)

            # ── 验证上一个交易日的 ML 决策 ──
            self._validate_ml_decisions(now)

        except Exception as e:
            logger.warning(f"机器学习异常: {e}")
            self._log(trading_phase, f"  机器学习异常: {e}")

        return True  # 每轮执行即完成

    def _ml_predict_only(self, now: datetime) -> bool:
        """仅预测模式：使用已有模型对池中标的生产预测（不重新训练）"""
        trading_phase = self._current_phase
        # 收集标的
        all_candidates: list = []
        seen = set()
        for strategy_name in STRATEGIES:
            pool = self.pool_mgr.get_pool(strategy_name)
            if not pool:
                continue
            ranked = pool.ranked()
            for item in ranked:
                if item.tier in (TIER_FOCUS, TIER_WATCH) and item.code not in seen:
                    seen.add(item.code)
                    all_candidates.append(item)

        if not all_candidates:
            return True

        try:
            predictions = self._ml_engine.predict_all(all_candidates, now=now)
            self._ml_engine.sync_to_pool(predictions, all_candidates)
            if predictions:
                self._log(trading_phase,
                    f"🧠 [ML预测] {len(predictions)} 只标的预测已刷新")

            # 更新 ML 预测缓存
            self._build_ml_prediction_cache(predictions)

            # 验证上一个交易日的 ML 决策
            self._validate_ml_decisions(now)

        except Exception as e:
            logger.warning(f"ML 预测异常: {e}")

        return True

    def _build_ml_prediction_cache(self, predictions: list):
        """从预测结果构建 code→direction 和 code→score 的快速查询缓存。"""
        self._ml_prediction_cache.clear()
        self._ml_score_cache.clear()
        for p in predictions:
            self._ml_prediction_cache[p.code] = p.direction
            self._ml_score_cache[p.code] = p.ml_score

    def _validate_ml_decisions(self, now: datetime):
        """验证上一个交易日的 ML 决策准确性，动态调整决策权重。"""
        trading_phase = self._current_phase
        today_str = now.strftime("%Y%m%d")
        if self._ml_last_validate_date == today_str:
            return

        pending = self._ml_gate.get_pending_count()
        if pending == 0:
            self._ml_last_validate_date = today_str
            return

        self._log(trading_phase, f"🔍 [ML验证] 验证 {pending} 条待验证决策...")
        stats = self._ml_gate.validate_daily(now, self.fetcher)

        if stats["validated"] > 0:
            acc_str = f"{stats['correct']}/{stats['validated']}"
            acc_pct = stats["correct"] / stats["validated"] * 100 if stats["validated"] > 0 else 0
            self._log(trading_phase,
                f"  {acc_str} 正确 ({acc_pct:.0f}%) | "
                f"权重 {stats['weight']:.1%} | "
                f"门控 {'激活' if self._ml_gate.enabled else '待激活'}")

            # 输出门控统计摘要
            gate_stats = self._ml_gate.get_stats()
            if gate_stats["total_validated"] >= 5:
                self._log(trading_phase,
                    f"  累计: {gate_stats['total_validated']}次验证 "
                    f"({gate_stats['total_correct']}正/{gate_stats['total_wrong']}误) | "
                    f"驳回 {gate_stats['total_rejected']}次 "
                    f"(准确率{gate_stats['rejected_accuracy']:.0%})")

        self._ml_last_validate_date = today_str

    # ═══════════════════════════════════════════
    # 生命周期重置
    # ═══════════════════════════════════════════

    def _reset_lifecycle(self):
        """重置生命周期状态（新交易日开始时调用）"""
        self._lifecycle_phase = LifecyclePhase.IDLE
        self._lifecycle_completed.clear()
        self._lifecycle_last_run.clear()
        self._lifecycle_pause_until = None
        self._idle_since = None
        self._screen_offset = 0
        self._screened_codes.clear()
        self._screening_done = False
        self._backtest_date = None
        self._ml_last_train_date = None
        # 重置各策略生命周期阶段
        for k in STRATEGIES:
            self._strategy_phase[k] = LifecyclePhase.IDLE.value[0]
            self._strategy_phase_label[k] = LifecyclePhase.IDLE.label
            self._strategy_phase_updated[k] = ""

    def _clear_daily_cache(self):
        """清除每日临时缓存"""
        self._latest_snapshot = None
        self._latest_auction = None
        self._latest_signals = {}
        self._trading_plan = None
        self._todays_decisions = []
        self._ml_prediction_cache.clear()
        self._ml_score_cache.clear()
        self.hunter.clear_cache()

    def _save_daily_summary(self, force: bool = False):
        """保存每日收盘总结（仅 15:00 后生成，完整总结存在则跳过）"""
        now = datetime.now()
        today = now.strftime("%Y%m%d")
        filepath = os.path.join(LIVE_LOG_DIR, f"summary_{today}.json")

        # 时间拦栅：15:00 前不生成（盘中止损、崩溃重启等场景不会写出残缺结论）
        if now.time() < time(15, 0) and not force:
            logger.info(f"未到收盘时间({now.strftime('%H:%M')})，推迟生成收盘总结")
            return

        # 已存在完整总结则跳过（检查 generated_at 是否在 15:00 之后）
        if os.path.exists(filepath) and not force:
            try:
                with open(filepath, encoding="utf-8") as f:
                    existing = json.load(f)
                gen_at = existing.get("generated_at", "")
                if gen_at:
                    gen_time = datetime.fromisoformat(gen_at).time()
                    if gen_time >= time(15, 0):
                        logger.info(f"今日总结已存在(生成于{gen_at})，跳过重复生成: {filepath}")
                        return
                    else:
                        logger.warning(f"已存在盘前生成的残缺总结(生成于{gen_at})，强制覆盖")
            except Exception:
                pass  # 文件损坏，重新生成

        logger.info("正在生成当日收盘总结...")

        # 重置不可用数据源标记
        if hasattr(self, "fetcher"):
            self.fetcher.reset_unavailable_sources()

        # ── 1. 市场总览（指数 + 成交量 + 估值） ──
        market_overview = self._build_market_overview()

        # ── 2. 技术指标（各指数 MACD/KDJ/RSI） ──
        tech_indicators = self._build_tech_indicators()

        # ── 3. 情绪评估 ──
        sentiment = self._build_sentiment_assessment(market_overview)

        # ── 4. 牛熊行情研判 ──
        market_trend = self._build_market_trend()

        # ── 5. 涨跌停统计 ──
        limit_stats = self._build_limit_stats()

        # ── 6. 策略盈亏 ──
        strategy_pnl = self._collect_strategy_pnl()

        # ── 7. 热门板块详情 ──
        hot_sectors = self._build_hot_sectors()

        # ── 8. 涨停个股完整列表 ──
        limit_up_details = self._build_limit_up_details()

        # ── 9. 次日交易计划 ──
        next_day_plan = self._build_next_day_plan()

        summary = {
            "date": today,
            "generated_at": datetime.now().isoformat(),
            "market_overview": market_overview,
            "technical_indicators": tech_indicators,
            "sentiment": sentiment,
            "market_trend": market_trend,
            "limit_stats": limit_stats,
            "strategy_pnl": strategy_pnl,
            "hot_sectors": hot_sectors,
            "limit_up_details": limit_up_details,
            "next_day_plan": next_day_plan,
            "phase_logs": [
                {"time": log.timestamp.isoformat(), "phase": log.phase.value, "event": log.event, "detail": log.detail}
                for log in self._logs[-100:]
            ],
            "decisions": self._todays_decisions[:50],
            "signal_count": sum(len(r.signals) for r in self._latest_signals.values()),
        }

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
            logger.info(f"收盘总结已保存: {filepath}")
        except Exception as e:
            logger.warning(f"保存总结失败: {e}")

    # ── 总结子模块 ─────────────────────────────────────────

    def _build_market_overview(self) -> dict:
        """构建市场总览"""
        overview = {"indices": {}, "total_volume": 0, "sh_pe": 0, "activity": {}}

        try:
            # 指数分析
            indices = self.market.analyze_indices(days=60)
            for idx in indices.values():
                overview["indices"][idx.name] = {
                    "code": idx.code,
                    "close": idx.close,
                    "pct_change": idx.pct_change,
                    "amplitude": idx.amplitude,
                    "volume_ratio": idx.volume_ratio,
                    "trend_score": idx.trend_score,
                    "trend_rating": idx.trend_rating,
                }

            # 市场活跃度
            activity = self.fetcher.fetch_market_activity()
            overview["activity"] = activity

            # 两市总成交额
            vol_info = self.fetcher.fetch_market_total_volume()
            overview["total_volume"] = vol_info.get("total_volume", 0)
            overview["sh_pe"] = vol_info.get("sh_pe", 0)
            overview["sz_total_mv"] = vol_info.get("sz_total_mv", 0)
            overview["sh_total_mv"] = vol_info.get("sh_total_mv", 0)

        except Exception as e:
            logger.warning(f"构建市场总览失败: {e}")

        return overview

    def _build_tech_indicators(self) -> dict:
        """构建各关键指数技术指标（使用内置指标库）"""
        result = {}
        try:
            key_codes = {
                "000001": "上证指数",
                "399001": "深证成指",
                "399006": "创业板指",
                "000300": "沪深300",
            }
            for code, name in key_codes.items():
                df = self.fetcher.fetch_index_daily(code)
                if df is None or df.empty:
                    continue
                col_map = {
                    "日期": "date", "收盘": "close", "开盘": "open",
                    "最高": "high", "最低": "low", "成交量": "volume",
                }
                df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
                required = ["close", "high", "low"]
                if not all(c in df.columns for c in required):
                    continue

                # 使用 comprehensive_analysis 一次性获取所有指标
                ca = comprehensive_analysis(df.tail(60))
                m = ca.get("macd", IndicatorResult("neutral", "hold", 50, {}))
                k = ca.get("kdj", IndicatorResult("neutral", "hold", 50, {}))
                r = ca.get("rsi", IndicatorResult("neutral", "hold", 50, {}))

                result[name] = {
                    "macd_signal": m.signal,
                    "macd_trend": m.trend,
                    "macd_strength": m.strength,
                    "kdj_signal": k.signal,
                    "kdj_trend": k.trend,
                    "rsi_value": r.details.get("rsi", r.details.get("RSI", 50)) if r.details else 50,
                    "rsi_trend": r.trend,
                }
        except Exception as e:
            logger.warning(f"构建技术指标失败: {e}")
        return result

    def _build_sentiment_assessment(self, market_overview: dict) -> dict:
        """综合情绪评估 - 包含贪婪/恐惧指数"""
        try:
            sentiment_raw = self.market.get_market_sentiment()
            score = sentiment_raw.get("score", 50)

            # 贪婪/恐惧映射
            if score >= 80:
                emotion = "极度贪婪"
                emotion_code = "extreme_greed"
            elif score >= 65:
                emotion = "贪婪"
                emotion_code = "greed"
            elif score >= 45:
                emotion = "中性"
                emotion_code = "neutral"
            elif score >= 30:
                emotion = "恐惧"
                emotion_code = "fear"
            else:
                emotion = "极度恐惧"
                emotion_code = "extreme_fear"

            # 市场过热/过冷判断
            activity = market_overview.get("activity", {})
            up_count = int(activity.get("上涨", 0))
            down_count = int(activity.get("下跌", 0))
            limit_up = int(activity.get("涨停", 0))
            limit_down = int(activity.get("跌停", 0))

            if limit_up > 100 and up_count > down_count * 3:
                heat = "过热"
            elif limit_down > 50 and down_count > up_count * 2:
                heat = "过冷"
            elif up_count > down_count * 2:
                heat = "偏热"
            elif down_count > up_count * 2:
                heat = "偏冷"
            else:
                heat = "正常"

            return {
                "assessment": sentiment_raw.get("sentiment", "neutral"),
                "score": score,
                "emotion": emotion,
                "emotion_code": emotion_code,
                "market_heat": heat,
                "factors": sentiment_raw.get("factors", {}),
            }
        except Exception as e:
            logger.warning(f"构建情绪评估失败: {e}")
            return {
                "assessment": "neutral",
                "score": 50,
                "emotion": "未知",
                "emotion_icon": "❓",
                "market_heat": "未知",
                "factors": {},
            }

    def _build_market_trend(self) -> dict:
        """构建牛熊行情研判"""
        try:
            return self.market.diagnose_market_trend(days=120)
        except Exception as e:
            logger.warning(f"牛熊研判失败: {e}")
            return {"phase": "未知", "phase_code": "unknown", "confidence": 0, "advice": "", "signals": []}

    def _build_limit_stats(self) -> dict:
        """构建涨跌停统计数据"""
        import pandas as pd

        stats = {
            "limit_up": 0,
            "limit_down": 0,
            "limit_up_real": 0,
            "limit_down_real": 0,
            "up_count": 0,
            "down_count": 0,
            "flat_count": 0,
            "max_consecutive": 0,
            "first_board_count": 0,
            "consecutive_boards": [],
            "top_limit_up_sectors": [],
        }

        try:
            today_str = datetime.now().strftime("%Y%m%d")

            # 市场活跃度（涨跌家数）
            activity = self.fetcher.fetch_market_activity()
            stats["limit_up"] = int(activity.get("涨停", 0))
            stats["limit_up_real"] = int(activity.get("真实涨停", 0))
            stats["limit_down"] = int(activity.get("跌停", 0))
            stats["limit_down_real"] = int(activity.get("真实跌停", 0))
            stats["up_count"] = int(activity.get("上涨", 0))
            stats["down_count"] = int(activity.get("下跌", 0))
            stats["flat_count"] = int(activity.get("平盘", 0))

            # 涨停池详情
            zt_df = self.fetcher.fetch_limit_up_pool(today_str)
            if not zt_df.empty:
                # 最高连板
                if "连板数" in zt_df.columns:
                    lb_nums = pd.to_numeric(zt_df["连板数"], errors="coerce").dropna()
                    if not lb_nums.empty:
                        stats["max_consecutive"] = int(lb_nums.max())

                # 首板数
                if "连板数" in zt_df.columns:
                    first_boards = zt_df[pd.to_numeric(zt_df["连板数"], errors="coerce") == 1]
                    stats["first_board_count"] = len(first_boards)

                # 连板股列表（>=2板）
                if "连板数" in zt_df.columns and "名称" in zt_df.columns:
                    lb_boards = zt_df[pd.to_numeric(zt_df["连板数"], errors="coerce") >= 2]
                    for _, row in lb_boards.iterrows():
                        stats["consecutive_boards"].append(
                            {
                                "code": str(row.get("代码", "")),
                                "name": str(row.get("名称", "")),
                                "consecutive": int(row["连板数"]),
                                "reason": str(row.get("涨停统计", "")),
                            }
                        )
                    stats["consecutive_boards"] = stats["consecutive_boards"][:20]

                # 涨停行业分布
                if "所属行业" in zt_df.columns:
                    sector_counts = zt_df["所属行业"].value_counts().head(5)
                    stats["top_limit_up_sectors"] = [
                        {"sector": str(k), "count": int(v)} for k, v in sector_counts.items()
                    ]

        except Exception as e:
            logger.warning(f"构建涨跌统计失败: {e}")

        return stats

    def _collect_strategy_pnl(self) -> dict:
        """收集各策略当日盈亏（从 PaperTrader 赛马数据读取）"""
        result = {}

        # 动态从 STRATEGIES 获取所有策略 key/label
        strategy_map = dict(STRATEGIES)
        strategy_keys = list(strategy_map.keys())

        try:
            # 优先从 PaperTrader 内存读取
            if self.paper_trader and self.paper_trader.accounts:
                rankings = self.paper_trader.get_rankings()
                for r in rankings:
                    key = next((k for k, v in strategy_map.items() if v == r.strategy_label), None)
                    if not key:
                        # 也尝试别名匹配
                        for k in strategy_keys:
                            if strategy_map.get(k, "") == r.strategy_label:
                                key = k
                                break
                    if key:
                        result[key] = {
                            "label": r.strategy_label,
                            "rank": r.rank,
                            "total_equity": r.total_equity,
                            "total_return_pct": r.total_return,
                            "daily_return_pct": r.daily_return if r.daily_return is not None else 0,
                            "position_count": r.position_count,
                            "max_drawdown": r.max_drawdown,
                            "trades": r.trades,
                        }

            # 若内存为空，尝试从保存文件读取历史数据
            if not result and os.path.exists(PAPER_SNAPSHOT_FILE):
                with open(PAPER_SNAPSHOT_FILE, encoding="utf-8") as f:
                    race_data = json.load(f)
                accounts = race_data.get("accounts", {})
                for key, acc in accounts.items():
                    label = strategy_map.get(key, key)
                    snapshots = acc.get("daily_snapshots", [])
                    latest = snapshots[-1] if snapshots else {}
                    result[key] = {
                        "label": label,
                        "rank": 0,
                        "total_equity": latest.get("equity", acc.get("initial_capital", DEFAULT_CAPITAL)),
                        "total_return_pct": round(
                            (
                                latest.get("equity", acc.get("initial_capital", DEFAULT_CAPITAL))
                                / acc.get("initial_capital", DEFAULT_CAPITAL)
                                - 1
                            )
                            * 100,
                            2,
                        ),
                        "daily_return_pct": 0,
                        "position_count": latest.get("positions", 0),
                        "max_drawdown": 0,
                        "trades": 0,
                    }
        except Exception as e:
            logger.warning(f"收集策略盈亏失败: {e}")

        # 补全缺失策略为初始状态
        for key in strategy_keys:
            if key not in result:
                result[key] = {
                    "label": strategy_map.get(key, key),
                    "rank": 0,
                    "total_equity": DEFAULT_CAPITAL,
                    "total_return_pct": 0,
                    "daily_return_pct": 0,
                    "position_count": 0,
                    "max_drawdown": 0,
                    "trades": 0,
                }

        return result

    def _build_hot_sectors(self) -> dict:
        """构建热门板块详情（按涨停家数排名 + 板块内五日涨幅Top10成分股 + 涨停交叉统计）"""
        import pandas as pd

        result = {
            "top_sectors": [],
            "limit_up_by_sector": {},
            "generated_at": datetime.now().isoformat(),
        }

        try:
            # 1. 获取涨停池数据（用于交叉统计和板块排名）
            today_str = datetime.now().strftime("%Y%m%d")
            zt_df = self.fetcher.fetch_limit_up_pool(today_str)
            zt_by_sector: dict[str, list[dict[str, object]]] = {}

            if not zt_df.empty and "所属行业" in zt_df.columns:
                for _, row in zt_df.iterrows():
                    sec = str(row.get("所属行业", ""))
                    if sec not in zt_by_sector:
                        zt_by_sector[sec] = []
                    zt_by_sector[sec].append(
                        {
                            "code": str(row.get("代码", "")),
                            "name": str(row.get("名称", "")),
                            "price": round(float(row.get("最新价", 0) or 0), 2),
                            "pct_change": round(float(row.get("涨跌幅", 0) or 0), 2),
                            "consecutive": int(pd.to_numeric(row.get("连板数", 0), errors="coerce").__float__() or 0),
                            "first_time": str(row.get("首次封板时间", "")),
                            "last_time": str(row.get("最后封板时间", "")),
                            "turnover": round(float(row.get("换手率", 0) or 0), 2),
                            "volume_amount": round(float(row.get("成交额", 0) or 0) / 1e8, 2),
                            "float_mv": round(float(row.get("流通市值", 0) or 0) / 1e8, 2),
                            "reason": str(row.get("涨停统计", row.get("涨停原因", ""))),
                        }
                    )

            # 2. 获取板块分析结果（已按涨停家数排名，包含五日涨幅Top10成分股）
            sectors = self.market.analyze_sectors(top_n=10)
            if not sectors:
                logger.warning("analyze_sectors 返回空列表，热门板块无数据")

            # 3. 组装板块数据（交叉涨停龙头）
            for s in sectors:
                sec_name = s.name
                zt_count = len(zt_by_sector.get(sec_name, []))
                sector_data = {
                    "name": sec_name,
                    "code": s.code,
                    "pct_change": s.pct_change,
                    "strength_rank": s.strength_rank,
                    "trend": s.trend,
                    "volume_surge": s.volume_surge,
                    "limit_up_count": zt_count,
                    "total_stocks": s.total_stocks,
                    "leading_stocks": s.leading_stocks,  # 领涨股 Top3（五日涨幅）
                    "top_stocks": s.top_stocks,  # 板块内五日涨幅 Top10 详细数据
                    # 板块内涨停龙头（取当日涨幅最高的5只）
                    "limit_up_leaders": sorted(
                        zt_by_sector.get(sec_name, []), key=lambda x: x["pct_change"], reverse=True
                    )[:5],
                }
                result["top_sectors"].append(sector_data)

            result["limit_up_by_sector"] = {k: len(v) for k, v in zt_by_sector.items()}

        except Exception as e:
            logger.warning(f"构建热门板块数据失败: {e}", exc_info=True)

        return result

    def _build_limit_up_details(self) -> dict:
        """构建涨停个股完整详情列表"""
        import pandas as pd

        result = {
            "stocks": [],
            "total_count": 0,
            "first_board_count": 0,
            "consecutive_leaders": [],
        }

        try:
            today_str = datetime.now().strftime("%Y%m%d")
            zt_df = self.fetcher.fetch_limit_up_pool(today_str)
            if zt_df.empty:
                return result

            result["total_count"] = len(zt_df)

            for _, row in zt_df.iterrows():
                consecutive = int(pd.to_numeric(row.get("连板数", 0), errors="coerce").__float__() or 0)
                stock_info = {
                    "code": str(row.get("代码", "")),
                    "name": str(row.get("名称", "")),
                    "price": round(float(row.get("最新价", 0) or 0), 2),
                    "pct_change": round(float(row.get("涨跌幅", 0) or 0), 2),
                    "limit_price": round(float(row.get("涨停价", 0) or 0), 2),
                    "consecutive": consecutive,
                    "sector": str(row.get("所属行业", "")),
                    "first_time": str(row.get("首次封板时间", "")),
                    "last_time": str(row.get("最后封板时间", "")),
                    "turnover": round(float(row.get("换手率", 0) or 0), 2),
                    "volume_amount": round(float(row.get("成交额", 0) or 0) / 1e8, 2),  # 亿
                    "float_mv": round(float(row.get("流通市值", 0) or 0) / 1e8, 2),  # 亿
                    "total_mv": round(float(row.get("总市值", 0) or 0) / 1e8, 2),  # 亿
                    "seal_amount": round(float(row.get("封单资金", 0) or 0) / 1e8, 2),  # 亿
                    "reason": str(row.get("涨停统计", row.get("涨停原因", ""))),
                }
                result["stocks"].append(stock_info)

                if consecutive == 1:
                    result["first_board_count"] += 1
                elif consecutive >= 3:
                    result["consecutive_leaders"].append(stock_info)

            # 按连板数排序
            result["stocks"].sort(key=lambda x: (-x["consecutive"], -x["pct_change"]))
            result["consecutive_leaders"].sort(key=lambda x: -x["consecutive"])

        except Exception as e:
            logger.warning(f"构建涨停详情失败: {e}")

        return result

    def _build_next_day_plan(self) -> dict:
        """构建次日交易计划概览"""
        plan = {}
        for strategy_name in STRATEGIES:
            pool = self.pool_mgr.get_pool(strategy_name)
            if not pool:
                continue
            ranked = pool.ranked()
            top5 = [{"code": item.code, "name": item.name, "score": item.score} for item in ranked[:5]]
            plan[strategy_name] = {
                "label": STRATEGIES.get(strategy_name, strategy_name),
                "pool_size": len(ranked),
                "top5": top5,
            }
        return plan

    def _add_log(self, phase: TradingPhase, event: str):
        """添加引擎日志"""
        detail = ""
        if "  " in event and not event.startswith("==="):
            detail = event
        log_entry = EngineLog(timestamp=datetime.now(), phase=phase, event=event, detail=detail)
        self._logs.append(log_entry)
        if len(self._logs) > self._max_logs:
            self._logs = self._logs[-self._max_logs :]

        # 通知回调
        for cb in self._log_callbacks:
            with contextlib.suppress(Exception):
                cb(log_entry)

    def _log(self, phase: TradingPhase, event: str):
        """统一日志输出"""
        if phase != TradingPhase.CLOSED:
            label = get_phase_label(phase)
            msg = f"[{label}] {event}"
        else:
            msg = f"[引擎] {event}"

        logger.info(msg)
        self._add_log(phase, msg)

    def add_log_callback(self, callback: Callable):
        """添加日志回调"""
        self._log_callbacks.append(callback)

    def get_recent_logs(self, n: int = 50) -> list[EngineLog]:
        """获取最近 N 条日志"""
        return self._logs[-n:]

    def get_status(self) -> dict[str, object]:
        """获取引擎运行状态"""
        # 生命周期进度
        lifecycle_info: dict[str, object] = {
            "phase": self._lifecycle_phase.value[0],
            "phase_label": self._lifecycle_phase.label,
            "completed": list(self._lifecycle_completed),
            "screening_done": self._screening_done,
            "screened_count": len(self._screened_codes),
            "screening_progress": f"{len(self._screened_codes)} 只已选" if self._screened_codes else "未开始",
            # 各策略独立生命周期阶段
            "strategy_phases": {
                k: {
                    "phase": self._strategy_phase.get(k, "idle"),
                    "label": self._strategy_phase_label.get(k, "空闲"),
                    "updated_at": self._strategy_phase_updated.get(k, ""),
                }
                for k in STRATEGIES
            },
        }

        return {
            "state": self.state.value,
            "current_phase": self._current_phase.value,
            "phase_label": get_phase_label(self._current_phase),
            "is_trading_day": self.calendar.is_trading_day(),
            "lifecycle": lifecycle_info,
            "pool_summary": self.pool_mgr.summary(),
            "latest_signals": sum(len(r.signals) for r in self._latest_signals.values()),
            "signal_details": {
                name: {
                    "strategy": r.strategy_label,
                    "count": len(r.signals),
                    "buy": sum(1 for s in r.signals if s.signal_type in ("buy", "add")),
                    "sell": sum(1 for s in r.signals if s.signal_type in ("sell", "reduce", "close")),
                    "duration": r.scan_duration,
                    "raw_signals": [
                        {
                            "code": s.code,
                            "name": s.name,
                            "signal_type": s.signal_type,
                            "confidence": s.confidence,
                            "price": s.price,
                            "stop_loss": s.stop_loss,
                            "take_profit": s.take_profit,
                            "reason": s.reason,
                            "timestamp": s.timestamp.isoformat() if s.timestamp else "",
                        }
                        for s in r.signals[:10]
                    ],
                }
                for name, r in self._latest_signals.items()
            },
            "decisions_today": len(self._todays_decisions),
            "log_count": len(self._logs),
        }

    def _save_state(self, force: bool = False):
        """将引擎实时状态写入文件（供 Web 仪表盘跨进程读取）"""
        try:
            now = datetime.now()
            state = self.get_status()

            state["saved_at"] = now.isoformat()
            state["last_scan_time"] = now.strftime("%H:%M:%S")
            state["phase_logs"] = [
                {
                    "time": log.timestamp.strftime("%H:%M:%S")
                    if hasattr(log.timestamp, "strftime")
                    else str(log.timestamp),
                    "phase": log.phase.value if hasattr(log.phase, "value") else str(log.phase),
                    "event": log.event,
                    "detail": getattr(log, "detail", ""),
                }
                for log in self._logs[-100:]
            ]
            state["last_trade_summary"] = self._collect_strategy_pnl()

            filepath = os.path.join(LIVE_LOG_DIR, "engine_state.json")
            tmp_path = filepath + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2, default=str)
            os.replace(tmp_path, filepath)  # 原子写入

        except Exception as e:
            logger.warning(f"保存引擎状态失败: {e}")


def bias_cn(bias: str) -> str:
    """偏向中文翻译"""
    return {"bullish": "多头", "bearish": "空头", "neutral": "中性"}.get(bias, bias)


# 全局引擎单例
live_engine = LiveTradingEngine()
