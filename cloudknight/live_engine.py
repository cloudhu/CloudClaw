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

import logging
import time
import json
import os
from datetime import datetime, date, timedelta
from threading import Thread, Event, Lock
from typing import Dict, List, Optional, Any, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum

from .config import (STRATEGIES, DEFAULT_CAPITAL, LIVE_ENGINE_CHECK_INTERVAL,
                     LIVE_ENGINE_SCAN_INTERVAL, LIVE_LOG_DIR, LIVE_TRADE_DIR,
                     INDEX_CODES, POOL_MAX_SIZE, POOL_SCREEN_SAMPLE)
from .trading_calendar import (TradingCalendar, TradingPhase,
                               trading_calendar, get_phase_label)
from .market_analyzer import (MarketAnalyzer, MarketSnapshot, IndexAnalysis,
                              SectorAnalysis)
from .auction_analyzer import (AuctionAnalyzer, AuctionSnapshot,
                               AuctionStockResult)
from .signal_hunter import (SignalHunter, TradeSignal, SignalResult, TradingPlan)
from .stock_pool import PoolManager
from .data_manager import DataFetcher

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

        # 运行状态
        self.state = EngineState.STOPPED
        self._stop_event = Event()
        self._main_thread: Optional[Thread] = None
        self._lock = Lock()

        # 交易日状态
        self._current_phase: TradingPhase = TradingPhase.CLOSED
        self._last_phase: TradingPhase = TradingPhase.CLOSED
        self._phase_completed: set = set()

        # 分析结果缓存
        self._latest_snapshot: Optional[MarketSnapshot] = None
        self._latest_auction: Optional[AuctionSnapshot] = None
        self._latest_signals: Dict[str, SignalResult] = {}
        self._trading_plan: Optional[TradingPlan] = None
        self._todays_decisions: List[Dict] = []

        # 日志
        self._logs: List[EngineLog] = []
        self._max_logs = 500
        self._log_callbacks: List[Callable] = []

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

    # ═══════════════════════════════════════════
    # 各阶段行为实现
    # ═══════════════════════════════════════════

    def _action_non_trading_day(self, now: datetime):
        """非交易日：全策略选股入池"""
        self._log(TradingPhase.CLOSED, "非交易日 → 执行选股任务")
        self._log(TradingPhase.CLOSED, "获取全市场股票基础池...")

        try:
            codes = self.fetcher.build_stock_pool(filter_st=True, filter_new=True)
            self._log(TradingPhase.CLOSED, f"基础池: {len(codes)} 只")

            # 为四种策略并行筛选
            results = self.pool_mgr.screen_all(codes, verbose=False)
            for key, items in results.items():
                label = STRATEGIES.get(key, key)
                top_score = items[0].score if items else 0
                self._log(TradingPhase.CLOSED,
                          f"  [{label}] 筛选 {len(items)} 只 → "
                          f"最高评分: {top_score}")

            # 保存股票池
            self.pool_mgr.save_all()
            self._log(TradingPhase.CLOSED, "选股完成，股票池已保存")

        except Exception as e:
            self._log(TradingPhase.CLOSED, f"选股异常: {e}")

    def _action_pre_market(self, now: datetime):
        """盘前准备 08:30-09:15"""
        self._log(TradingPhase.PRE_MARKET, "盘前准备开始...")

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
            self._log(TradingPhase.PRE_MARKET,
                      f"  {arrow} {idx.name}: {idx.close:.2f} "
                      f"({idx.pct_change:+.2f}%) "
                      f"MACD:{idx.macd_signal} KDJ:{idx.kdj_signal} "
                      f"RSI:{idx.rsi_value}")

        # 4. 板块热度分析
        self._log(TradingPhase.PRE_MARKET, "正在分析板块热度...")
        sectors = self.market.analyze_sectors(top_n=10)
        for s in sectors[:5]:
            self._log(TradingPhase.PRE_MARKET,
                      f"  {s.strength_rank}. {s.name}: {s.pct_change:+.2f}%")

        # 5. 市场情绪评估
        sentiment = self.market.get_market_sentiment()
        self._log(TradingPhase.PRE_MARKET,
                  f"市场情绪: {sentiment['sentiment'].upper()} "
                  f"(评分:{sentiment['score']})")

        # 6. 生成市场快照
        self._latest_snapshot = MarketSnapshot(
            timestamp=now, is_trading_day=True,
            indices=indices, hot_sectors=sectors,
            market_sentiment=sentiment["sentiment"],
            sentiment_score=sentiment["score"],
            buy_signals=0, sell_signals=0,
            capital_flow_summary="盘前"
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

        self._log(TradingPhase.AUCTION_RESULT,
                  f"竞价分析 {len(watch_codes)} 只关注标的...")

        # 竞价分析
        self._latest_auction = self.auction.analyze_auction(
            list(watch_codes), verbose=True
        )

        auc = self._latest_auction
        self._log(TradingPhase.AUCTION_RESULT,
                  f"竞价结果: 强势{auc.strong_auction_count} 只, "
                  f"弱势{auc.weak_auction_count} 只, "
                  f"偏{bias_cn(auc.market_bias)}")

        # 开盘交易决策
        watch_by_strategy = {}
        for strategy_name in STRATEGIES:
            pool = self.pool_mgr.get_pool(strategy_name)
            if pool:
                watch_by_strategy[strategy_name] = [
                    item.code for item in pool.ranked()[:10]
                ]

        decisions = self.auction.make_open_trade_decision(auc, watch_by_strategy)

        self._todays_decisions = []
        for strategy, dec_list in decisions.items():
            label = STRATEGIES.get(strategy, strategy)
            for d in dec_list:
                self._todays_decisions.append(d)
                self._log(TradingPhase.AUCTION_RESULT,
                          f"  [{label}] {d['code']} → {d['action']}: "
                          f"{d['reason']} (置信度:{d['confidence']})")

        self._log(TradingPhase.AUCTION_RESULT,
                  f"开盘决策: {len(self._todays_decisions)} 条")

    def _action_morning(self, now: datetime):
        """早盘交易 09:30-11:30"""
        if not self._phase_completed_full("morning"):
            self._log(TradingPhase.MORNING,
                      "早盘开盘! 开始分时走势跟踪与信号扫描")

            # 输出开盘决策摘要
            if self._todays_decisions:
                buy_dec = [d for d in self._todays_decisions
                           if d["action"] in ("buy_at_open", "buy")]
                sell_dec = [d for d in self._todays_decisions
                            if d["action"] == "sell_at_open"]
                if buy_dec:
                    self._log(TradingPhase.MORNING,
                              f"竞价买入决策: {len(buy_dec)} 条")
                    for d in buy_dec[:3]:
                        self._log(TradingPhase.MORNING,
                                  f"  → {d['code']} @ {d.get('price', 0):.2f}")
                if sell_dec:
                    self._log(TradingPhase.MORNING,
                              f"竞价卖出决策: {len(sell_dec)} 条")

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
                pm_sigs = [
                    s for s in self._trading_plan.signals
                    if s.signal_type in ("buy", "sell")
                ]
                if pm_sigs:
                    self._log(TradingPhase.AFTERNOON,
                              f"下午待执行信号: {len(pm_sigs)} 条")

        # 定期信号扫描
        if self._should_scan(now, "afternoon_scan"):
            self._scan_pool_signals(now)
            self._log(TradingPhase.AFTERNOON, "午盘信号扫描完成")
            self._report_current_signals()

    def _action_post_market(self, now: datetime):
        """盘后选股 15:00-次日"""
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
            top_str = ", ".join(
                f"{i.code}({i.score})" for i in top_items
            )
            self._log(TradingPhase.POST_MARKET,
                      f"  [{label}] {len(items)} 只 "
                      f"→ Top5: {top_str}")

        # 4. 保存股票池
        self.pool_mgr.save_all()

        # 5. 制定次日交易计划
        self._log(TradingPhase.POST_MARKET, "制定次日交易计划...")
        self._make_next_day_plan(now)

        self._log(TradingPhase.POST_MARKET, "盘后选股完成，等待次日")

    # ═══════════════════════════════════════════
    # 辅助方法
    # ═══════════════════════════════════════════

    def _scan_pool_signals(self, now: datetime,
                           sentiment: str = "neutral"):
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

        self._latest_signals = self.hunter.scan_all_strategies(
            stock_pool_map, days=120, verbose=False
        )

        # 生成交易计划
        self._trading_plan = self.hunter.build_trading_plan(
            self._latest_signals, sentiment
        )

    def _analyze_15min_kline(self, now: datetime):
        """15分钟K线技术分析（午间复盘用）"""
        # 选取池中前5只重点股进行15分钟K线分析
        focus_codes = []
        for strategy_name in STRATEGIES:
            pool = self.pool_mgr.get_pool(strategy_name)
            if pool:
                for item in pool.ranked()[:2]:
                    if item.code not in focus_codes:
                        focus_codes.append(item.code)

        if not focus_codes:
            return

        self._log(TradingPhase.LUNCH,
                  f"15分K线分析 {len(focus_codes[:5])} 只重点标的...")

        from .indicators import calc_macd, calc_kdj, calc_rsi

        for code in focus_codes[:5]:
            try:
                # 获取日K线（近期数据用于15分钟级别参考）
                df = self.fetcher.fetch_daily_kline(code, start_date="20250601")
                if df is None or df.empty or len(df) < 5:
                    continue

                # 标准15分K不可直接获取，用日线MACD/KDJ做近似
                col_map = {"日期": "date", "开盘": "open", "收盘": "close",
                           "最高": "high", "最低": "low", "成交量": "volume"}
                df = df.rename(columns={k: v for k, v in col_map.items()
                                        if k in df.columns})

                close = df["close"].iloc[-1]
                prev = df["close"].iloc[-2] if len(df) > 1 else close
                pct = (close / prev - 1) * 100

                # 日K的MACD+KDJ信号
                macd_result = calc_macd(df)
                kdj_result = calc_kdj(df)

                k_val = kdj_result[0].iloc[-1] if not kdj_result[0].empty else 50
                d_val = kdj_result[1].iloc[-1] if not kdj_result[1].empty else 50
                _ = kdj_result[2].iloc[-1] if len(kdj_result) > 2 and not kdj_result[2].empty else 50

                dif = macd_result[0].iloc[-1] if not macd_result[0].empty else 0
                dea = macd_result[1].iloc[-1] if not macd_result[1].empty else 0

                macd_sig = "金叉" if dif > dea else "死叉"
                kdj_sig = "金叉" if k_val > d_val else "死叉"

                am_swing = "强势" if pct > 1 else ("弱势" if pct < -1 else "震荡")

                self._log(TradingPhase.LUNCH,
                          f"  {code}: {pct:+.2f}% [{am_swing}] "
                          f"MACD:{macd_sig} KDJ:{kdj_sig}")

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

        self._log(TradingPhase.LUNCH,
                  f"下午计划: 关注 {adjusted_actions} 个信号执行")

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
                    "top5": [(item.code, item.name, item.score)
                             for item in ranked[:5]]
                }

        self._log(TradingPhase.POST_MARKET, "次日交易计划概要:")
        for key, info in pool_summary.items():
            self._log(TradingPhase.POST_MARKET,
                      f"  [{info['label']}] "
                      f"关注 {info['count']} 只")

    def _report_current_signals(self):
        """输出当前信号摘要"""
        if not self._latest_signals:
            return

        for strategy_name, result in self._latest_signals.items():
            if result.error:
                self._log(self._current_phase,
                          f"  [{result.strategy_label}] 扫描异常: {result.error}")
                continue

            sigs = result.signals
            if sigs:
                buys = [s for s in sigs if s.signal_type in ("buy", "add")]
                sells = [s for s in sigs if s.signal_type in ("sell", "reduce", "close")]
                for s in buys[:3]:
                    self._log(self._current_phase,
                              f"  [买] {s.code} {s.name} @ ~{s.price:.2f} "
                              f"[{s.confidence}] {s.reason[:40]}")
                for s in sells[:3]:
                    self._log(self._current_phase,
                              f"  [卖] {s.code} {s.name} @ ~{s.price:.2f} "
                              f"[{s.confidence}] {s.reason[:40]}")

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

    def _save_daily_summary(self):
        """保存每日交易总结"""
        today = datetime.now().strftime("%Y%m%d")
        filepath = os.path.join(LIVE_LOG_DIR, f"summary_{today}.json")

        summary = {
            "date": today,
            "phase_logs": [
                {"time": l.timestamp.isoformat(),
                 "phase": l.phase.value,
                 "event": l.event,
                 "detail": l.detail}
                for l in self._logs[-100:]
            ],
            "decisions": self._todays_decisions[:50],
            "signal_count": sum(
                len(r.signals) for r in self._latest_signals.values()
            ),
        }

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
            logger.info(f"每日总结已保存: {filepath}")
        except Exception as e:
            logger.warning(f"保存总结失败: {e}")

    def _add_log(self, phase: TradingPhase, event: str):
        """添加引擎日志"""
        detail = ""
        if "  " in event and not event.startswith("==="):
            detail = event
        log_entry = EngineLog(
            timestamp=datetime.now(), phase=phase,
            event=event, detail=detail
        )
        self._logs.append(log_entry)
        if len(self._logs) > self._max_logs:
            self._logs = self._logs[-self._max_logs:]

        # 通知回调
        for cb in self._log_callbacks:
            try:
                cb(log_entry)
            except Exception:
                pass

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

    def get_recent_logs(self, n: int = 50) -> List[EngineLog]:
        """获取最近 N 条日志"""
        return self._logs[-n:]

    def get_status(self) -> Dict[str, Any]:
        """获取引擎运行状态"""
        return {
            "state": self.state.value,
            "current_phase": self._current_phase.value,
            "phase_label": get_phase_label(self._current_phase),
            "is_trading_day": self.calendar.is_trading_day(),
            "pool_summary": self.pool_mgr.summary(),
            "latest_signals": sum(
                len(r.signals) for r in self._latest_signals.values()
            ),
            "signal_details": {
                name: {
                    "strategy": r.strategy_label,
                    "count": len(r.signals),
                    "buy": sum(1 for s in r.signals if s.signal_type in ("buy", "add")),
                    "sell": sum(1 for s in r.signals if s.signal_type in ("sell", "reduce", "close")),
                    "duration": r.scan_duration,
                }
                for name, r in self._latest_signals.items()
            },
            "decisions_today": len(self._todays_decisions),
            "log_count": len(self._logs),
        }


def bias_cn(bias: str) -> str:
    """偏向中文翻译"""
    return {"bullish": "多头", "bearish": "空头", "neutral": "中性"}.get(bias, bias)


# 全局引擎单例
live_engine = LiveTradingEngine()
