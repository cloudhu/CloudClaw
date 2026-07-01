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

非交易日: 全天 = 选股模式（所有策略筛选股票入池）
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
    LIVE_ENGINE_SCAN_INTERVAL,
    LIVE_LOG_DIR,
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
from .paper_trader import SNAPSHOT_FILE as PAPER_SNAPSHOT_FILE
from .paper_trader import PaperTrader
from .signal_hunter import SignalHunter, SignalResult, TradingPlan
from .stock_pool import PoolManager
from .trading_calendar import TradingCalendar, TradingPhase, get_phase_label

logger = logging.getLogger(__name__)


class EngineState(Enum):
    """引擎运行状态"""

    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"


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
        self._phase_completed: set[str] = set()

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
        self._phase_completed.clear()
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
        """引擎主循环 - 时间周期驱动"""
        self.state = EngineState.RUNNING
        self._log(TradingPhase.CLOSED, "主循环启动")

        while not self._stop_event.is_set():
            try:
                now = datetime.now()

                # 确定当前阶段
                phase = self.calendar.get_current_phase(now)
                self._current_phase = phase

                # 阶段切换检测
                if phase != self._last_phase:
                    self._on_phase_enter(phase, now)
                    self._last_phase = phase

                # 根据当前阶段执行任务
                if self.state == EngineState.RUNNING:
                    self._dispatch_phase_action(phase, now)

                # 等待下次检查
                self._stop_event.wait(timeout=LIVE_ENGINE_CHECK_INTERVAL)

            except Exception as e:
                logger.error(f"引擎主循环异常: {e}", exc_info=True)
                self._log(TradingPhase.CLOSED, f"异常: {e}")

        self._log(TradingPhase.CLOSED, "主循环退出")
        self.state = EngineState.STOPPED

    def _on_phase_enter(self, phase: TradingPhase, now: datetime):
        """进入新阶段时的回调"""
        # 重置阶段完成标记（新的一轮）
        if phase == TradingPhase.PRE_MARKET:
            self._phase_completed.clear()
            self._clear_daily_cache()

        label = get_phase_label(phase)
        self._log(phase, f">>> 进入 [{label}] 阶段")

    def _dispatch_phase_action(self, phase: TradingPhase, now: datetime):
        """根据当前阶段分发执行动作"""
        # 每个阶段只执行一次（避免重复执行）
        phase_key = f"{phase.value}_{now.strftime('%Y%m%d')}"
        if phase_key in self._phase_completed:
            return

        if phase == TradingPhase.CLOSED:
            # 非交易日 → 选股模式
            if not self.calendar.is_trading_day(now):
                self._action_non_trading_day(now)

        elif phase == TradingPhase.PRE_MARKET:
            self._action_pre_market(now)

        elif phase == TradingPhase.AUCTION:
            self._action_auction(now)

        elif phase == TradingPhase.AUCTION_RESULT:
            self._action_auction_result(now)

        elif phase == TradingPhase.MORNING:
            self._action_morning(now)

        elif phase == TradingPhase.LUNCH:
            self._action_lunch(now)

        elif phase == TradingPhase.AFTERNOON:
            self._action_afternoon(now)

        elif phase == TradingPhase.POST_MARKET:
            self._action_post_market(now)

        self._phase_completed.add(phase_key)

        # 保存引擎状态供 Web 仪表盘读取
        self._save_state()

    # ═══════════════════════════════════════════
    # 各阶段行为实现
    # ═══════════════════════════════════════════

    def _action_non_trading_day(self, now: datetime):
        """非交易日：全策略选股入池"""
        self._log(TradingPhase.CLOSED, "非交易日 → 执行选股任务")
        # 每个交易日开始前重置数据源状态，允许恢复重试
        self.fetcher.reset_unavailable_sources()
        self._log(TradingPhase.CLOSED, "获取全市场股票基础池...")

        try:
            codes = self.fetcher.build_stock_pool(filter_st=True, filter_new=True)
            self._log(TradingPhase.CLOSED, f"基础池: {len(codes)} 只")

            # 为四种策略并行筛选
            results = self.pool_mgr.screen_all(codes, verbose=False)
            for key, items in results.items():
                label = STRATEGIES.get(key, key)
                top_score = items[0].score if items else 0
                self._log(TradingPhase.CLOSED, f"  [{label}] 筛选 {len(items)} 只 → 最高评分: {top_score}")

            # 保存股票池
            self.pool_mgr.save_all()
            self._log(TradingPhase.CLOSED, "选股完成，股票池已保存")

        except Exception as e:
            self._log(TradingPhase.CLOSED, f"选股异常: {e}")

    def _action_pre_market(self, now: datetime):
        """盘前准备 08:30-09:15"""
        self._log(TradingPhase.PRE_MARKET, "盘前准备开始...")

        # 重置数据源状态，允许之前不可用的数据源恢复尝试
        self.fetcher.reset_unavailable_sources()

        # 1. 判断是否为交易日
        if not self.calendar.is_trading_day(now):
            self._log(TradingPhase.PRE_MARKET, "今日非交易日，跳过盘前分析")
            return

        # 2. 加载股票池
        self.pool_mgr.load_all()
        pool_summary = self.pool_mgr.summary()
        self._log(TradingPhase.PRE_MARKET, f"已加载股票池: {pool_summary}")

        # 3. 大盘指数分析
        self._log(TradingPhase.PRE_MARKET, "正在分析六大核心指数...")
        indices = self.market.analyze_indices(days=60)
        for idx in indices.values():
            arrow = "▲" if idx.pct_change >= 0 else "▼"
            self._log(
                TradingPhase.PRE_MARKET,
                f"  {arrow} {idx.name}: {idx.close:.2f} ({idx.pct_change:+.2f}%) "
                + f"MACD:{idx.macd_signal} KDJ:{idx.kdj_signal} "
                f"RSI:{idx.rsi_value}",
            )

        # 4. 板块热度分析
        self._log(TradingPhase.PRE_MARKET, "正在分析板块热度...")
        sectors = self.market.analyze_sectors(top_n=10)
        for s in sectors[:5]:
            self._log(TradingPhase.PRE_MARKET, f"  {s.strength_rank}. {s.name}: {s.pct_change:+.2f}%")

        # 5. 市场情绪评估
        sentiment = self.market.get_market_sentiment()
        self._log(TradingPhase.PRE_MARKET, f"市场情绪: {sentiment['sentiment'].upper()} (评分:{sentiment['score']})")

        # 6. 生成市场快照
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

        # 7. 盘前股票池信号扫描
        self._log(TradingPhase.PRE_MARKET, "盘前信号扫描...")
        self._scan_pool_signals(now, sentiment["sentiment"])

        self._log(TradingPhase.PRE_MARKET, "盘前准备完成，等待集合竞价")

    def _action_auction(self, now: datetime):
        """集合竞价监控 09:15-09:25"""
        self._log(TradingPhase.AUCTION, "集合竞价进行中...")
        # 实际系统中此处可订阅实时竞价数据流
        # 当前通过 akshare 的延时数据做近似监控

    def _action_auction_result(self, now: datetime):
        """竞价结果分析 09:26-09:29"""
        self._log(TradingPhase.AUCTION_RESULT, "分析竞价结果...")

        # 获取所有策略关注的股票代码
        watch_codes = set()
        for strategy_name in STRATEGIES:
            pool = self.pool_mgr.get_pool(strategy_name)
            if pool:
                for item in pool.ranked()[:10]:  # 每策略取前10
                    watch_codes.add(item.code)

        if not watch_codes:
            self._log(TradingPhase.AUCTION_RESULT, "无关注标的，跳过竞价分析")
            return

        self._log(TradingPhase.AUCTION_RESULT, f"竞价分析 {len(watch_codes)} 只关注标的...")

        # 竞价分析
        self._latest_auction = self.auction.analyze_auction(list(watch_codes), verbose=True)

        auc = self._latest_auction
        self._log(
            TradingPhase.AUCTION_RESULT,
            f"竞价结果: 强势{auc.strong_auction_count} 只, " + f"弱势{auc.weak_auction_count} 只, "
            f"偏{bias_cn(auc.market_bias)}",
        )

        # 开盘交易决策
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
                    TradingPhase.AUCTION_RESULT,
                    f"  [{label}] {d['code']} → {d['action']}: " + f"{d['reason']} (置信度:{d['confidence']})",
                )

        self._log(TradingPhase.AUCTION_RESULT, f"开盘决策: {len(self._todays_decisions)} 条")

    def _action_morning(self, now: datetime):
        """早盘交易 09:30-11:30"""
        if not self._phase_completed_full("morning"):
            self._log(TradingPhase.MORNING, "早盘开盘! 开始分时走势跟踪与信号扫描")

            # 输出开盘决策摘要
            if self._todays_decisions:
                buy_dec = [d for d in self._todays_decisions if d["action"] in ("buy_at_open", "buy")]
                sell_dec = [d for d in self._todays_decisions if d["action"] == "sell_at_open"]
                if buy_dec:
                    self._log(TradingPhase.MORNING, f"竞价买入决策: {len(buy_dec)} 条")
                    for d in buy_dec[:3]:
                        self._log(TradingPhase.MORNING, f"  → {d['code']} @ {d.get('price', 0):.2f}")
                if sell_dec:
                    self._log(TradingPhase.MORNING, f"竞价卖出决策: {len(sell_dec)} 条")

        # 定期信号扫描（每扫描间隔执行一次）
        if self._should_scan(now, "morning_scan"):
            self._scan_pool_signals(now)
            self._log(TradingPhase.MORNING, "早盘信号扫描完成")
            self._report_current_signals()

    def _action_lunch(self, now: datetime):
        """午间休市 11:30-13:00"""
        self._log(TradingPhase.LUNCH, "午间休市 - 早盘复盘")

        # 1. 15分钟K线分析
        self._log(TradingPhase.LUNCH, "15分钟K线技术分析 (早盘)...")
        self._analyze_15min_kline(now)

        # 2. 早盘信号回顾
        self._log(TradingPhase.LUNCH, "早盘信号回顾:")
        self._report_current_signals()

        # 3. 制定下午交易计划
        self._log(TradingPhase.LUNCH, "制定下午交易计划...")
        self._make_afternoon_plan(now)

        self._log(TradingPhase.LUNCH, "午间分析完成，等待下午开盘")

    def _action_afternoon(self, now: datetime):
        """午盘交易 13:00-15:00"""
        if not self._phase_completed_full("afternoon"):
            self._log(TradingPhase.AFTERNOON, "午盘开盘! 继续分时跟踪")

            if self._trading_plan and self._trading_plan.signals:
                pm_sigs = [s for s in self._trading_plan.signals if s.signal_type in ("buy", "sell")]
                if pm_sigs:
                    self._log(TradingPhase.AFTERNOON, f"下午待执行信号: {len(pm_sigs)} 条")

        # 定期信号扫描
        if self._should_scan(now, "afternoon_scan"):
            self._scan_pool_signals(now)
            self._log(TradingPhase.AFTERNOON, "午盘信号扫描完成")
            self._report_current_signals()

    def _action_post_market(self, now: datetime):
        """盘后选股 15:00-次日"""
        # 双保险：仅在 15:00 后执行（防止跨天/时区等边缘场景）
        if now.time() < time(15, 0):
            self._log(TradingPhase.POST_MARKET, "未到收盘时间，暂不执行盘后作业")
            return

        self._log(TradingPhase.POST_MARKET, "收盘! 开始盘后选股")

        # 1. 保存当日日志
        self._save_daily_summary()

        # 2. 获取全市场股票列表
        codes = self.fetcher.build_stock_pool(filter_st=True, filter_new=True)
        self._log(TradingPhase.POST_MARKET, f"基础池: {len(codes)} 只")

        # 3. 为四种策略筛选股票
        self._log(TradingPhase.POST_MARKET, "策略独立选股中...")
        results = self.pool_mgr.screen_all(codes, verbose=False)

        for key, items in results.items():
            label = STRATEGIES.get(key, key)
            top_items = items[:5]
            top_str = ", ".join(f"{i.code}({i.score})" for i in top_items)
            self._log(TradingPhase.POST_MARKET, f"  [{label}] {len(items)} 只 → Top5: {top_str}")

        # 4. 保存股票池
        self.pool_mgr.save_all()

        # 5. 制定次日交易计划
        self._log(TradingPhase.POST_MARKET, "制定次日交易计划...")
        self._make_next_day_plan(now)

        self._log(TradingPhase.POST_MARKET, "盘后选股完成，等待次日")

    # ═══════════════════════════════════════════
    # 辅助方法
    # ═══════════════════════════════════════════

    def _scan_pool_signals(self, now: datetime, sentiment: str = "neutral"):
        """扫描各策略股票池的交易信号"""
        stock_pool_map = {}
        for strategy_name in STRATEGIES:
            pool = self.pool_mgr.get_pool(strategy_name)
            if pool:
                codes = [item.code for item in pool.ranked()[:POOL_MAX_SIZE]]
                if codes:
                    stock_pool_map[strategy_name] = codes

        if not stock_pool_map:
            self._log(self._current_phase, "股票池为空，跳过信号扫描")
            return

        self._latest_signals = self.hunter.scan_all_strategies(stock_pool_map, days=120, verbose=False)

        # 生成交易计划
        self._trading_plan = self.hunter.build_trading_plan(self._latest_signals, sentiment)

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

    def _should_scan(self, now: datetime, key: str) -> bool:
        """判断是否应该执行扫描"""
        attr = f"_last_{key}"
        last = getattr(self, attr, None)
        if last is None:
            setattr(self, attr, now)
            return True
        if (now - last).total_seconds() >= LIVE_ENGINE_SCAN_INTERVAL:
            setattr(self, attr, now)
            return True
        return False

    def _phase_completed_full(self, prefix: str) -> bool:
        """检查阶段是否已完成全部初始化"""
        today = datetime.now().strftime("%Y%m%d")
        full_key = f"morning_init_{today}" if prefix == "morning" else f"afternoon_init_{today}"
        if full_key in self._phase_completed:
            return True
        self._phase_completed.add(full_key)
        return False

    def _clear_daily_cache(self):
        """清除每日临时缓存"""
        self._latest_snapshot = None
        self._latest_auction = None
        self._latest_signals = {}
        self._trading_plan = None
        self._todays_decisions = []
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
        return {
            "state": self.state.value,
            "current_phase": self._current_phase.value,
            "phase_label": get_phase_label(self._current_phase),
            "is_trading_day": self.calendar.is_trading_day(),
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
