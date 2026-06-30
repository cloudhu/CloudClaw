"""
信号猎人 - 多线程并行策略扫描买卖点

基于 AKQuant 引擎，在盘中/盘后对股票池进行多策略并行扫描:
  - 每个策略独立线程运行 AKQuant 回测计算最近 N 日信号
  - 提取当日买卖点信号
  - 汇总为交易计划
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import pandas as pd
import numpy as np

from .config import (STRATEGIES, DEFAULT_CAPITAL, DEFAULT_COMMISSION,
                     DEFAULT_STAMP_TAX, DEFAULT_SLIPPAGE,
                     LIVE_ENGINE_THREAD_POOL_SIZE, BACKTEST_LOT_SIZE)
from .strategies import (DragonHeadStrategy, SparrowStrategy,
                         TurtleStrategy, ValueInvestStrategy, STRATEGY_CLASSES)
from .data_manager import DataFetcher
from .indicators import (comprehensive_analysis, trend_score,
                         calc_macd, calc_kdj, calc_rsi, analyze_volume_price)

logger = logging.getLogger(__name__)

# 策略类到策略名映射（按注册名）
STRATEGY_HANDLERS = {
    "dragon_head": DragonHeadStrategy,
    "sparrow": SparrowStrategy,
    "turtle": TurtleStrategy,
    "value_invest": ValueInvestStrategy,
}


@dataclass
class TradeSignal:
    """交易信号"""
    code: str
    name: str
    strategy: str               # 策略标识
    signal_type: str            # buy / sell / add / reduce / close
    confidence: str             # high / medium / low
    price: float                # 参考价格
    stop_loss: float = 0.0      # 止损价
    take_profit: float = 0.0    # 止盈价
    reason: str = ""
    indicators: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SignalResult:
    """策略信号扫描结果"""
    strategy: str
    strategy_label: str
    signals: List[TradeSignal]
    scan_duration: float        # 扫描耗时(秒)
    error: Optional[str] = None


@dataclass
class TradingPlan:
    """交易计划"""
    date: str
    created_at: datetime
    signals: List[TradeSignal]
    market_sentiment: str
    plan_summary: str = ""
    position_changes: Dict[str, Dict[str, int]] = field(default_factory=dict)
    # { strategy: { buy: N, sell: N, hold: N } }


class SignalHunter:
    """信号猎人 - 多线程并行策略信号扫描引擎"""

    def __init__(self):
        self.fetcher = DataFetcher()
        self._data_cache: Dict[str, pd.DataFrame] = {}
        self._cache_lock = Lock()

    def scan_all_strategies(self, stock_pool: Dict[str, List[str]],
                            days: int = 120, verbose: bool = False
                            ) -> Dict[str, SignalResult]:
        """多线程并行扫描所有策略

        每个策略在自己的线程中运行，对其独立的股票池进行信号扫描。

        Args:
            stock_pool: {策略名: [股票代码列表]}
            days: 数据回溯天数
            verbose: 是否输出日志

        Returns:
            {策略名: SignalResult}
        """
        if verbose:
            logger.info(f"[SignalHunter] 启动 {len(stock_pool)} 路策略并行扫描...")

        results: Dict[str, SignalResult] = {}

        with ThreadPoolExecutor(max_workers=LIVE_ENGINE_THREAD_POOL_SIZE) as pool:
            futures = {}
            for strategy_name, codes in stock_pool.items():
                if not codes:
                    if verbose:
                        logger.info(f"  [{STRATEGIES.get(strategy_name, strategy_name)}] "
                                    f"跳过（无股票）")
                    continue
                cls = STRATEGY_HANDLERS.get(strategy_name)
                if cls is None:
                    continue
                label = STRATEGIES.get(strategy_name, strategy_name)
                future = pool.submit(
                    self._scan_strategy, strategy_name, label, cls, codes, days, verbose
                )
                futures[future] = strategy_name

            for future in as_completed(futures):
                strategy_name = futures[future]
                try:
                    result = future.result()
                    results[strategy_name] = result
                except Exception as e:
                    logger.error(f"[{strategy_name}] 扫描异常: {e}")
                    results[strategy_name] = SignalResult(
                        strategy=strategy_name,
                        strategy_label=STRATEGIES.get(strategy_name, strategy_name),
                        signals=[], scan_duration=0, error=str(e)
                    )

        return results

    def _scan_strategy(self, strategy_name: str, label: str,
                       strategy_cls, codes: List[str], days: int,
                       verbose: bool) -> SignalResult:
        """单策略信号扫描（在线程中运行）"""
        import time
        t0 = time.time()

        if verbose:
            logger.info(f"  [{label}] 扫描 {len(codes)} 只股票...")

        signals: List[TradeSignal] = []
        today = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")

        # 使用 AKQuant 引擎扫描 (如果可用)
        try:
            import akquant
            akq_signals = self._scan_with_akquant(
                strategy_name, strategy_cls, codes, start, today, verbose
            )
            signals.extend(akq_signals)
        except ImportError:
            if verbose:
                logger.info(f"  [{label}] AKQuant 未安装，使用指标分析降级扫描")
            signals = self._scan_with_indicators(
                strategy_name, label, codes, start, verbose
            )
        except Exception as e:
            logger.warning(f"  [{label}] AKQuant 扫描异常: {e}，降级到指标扫描")
            signals = self._scan_with_indicators(
                strategy_name, label, codes, start, verbose
            )

        elapsed = time.time() - t0
        if verbose:
            buy_sigs = sum(1 for s in signals if s.signal_type == "buy")
            sell_sigs = sum(1 for s in signals if s.signal_type in ("sell", "reduce", "close"))
            logger.info(f"  [{label}] 完成: {len(signals)} 信号 "
                        f"(买:{buy_sigs} 卖:{sell_sigs}) "
                        f"耗时 {elapsed:.1f}s")

        return SignalResult(
            strategy=strategy_name, strategy_label=label,
            signals=signals, scan_duration=round(elapsed, 2)
        )

    def _scan_with_akquant(self, strategy_name: str, strategy_cls,
                           codes: List[str], start: str, end: str,
                           verbose: bool) -> List[TradeSignal]:
        """使用 AKQuant 引擎扫描信号"""
        signals: List[TradeSignal] = []

        try:
            import akquant

            # 准备 AKQuant 格式数据
            data = self.fetcher.prepare_akquant_data(codes, start_date=start, end_date=end)
            if not data:
                return signals

            # 创建策略实例
            inst = strategy_cls()

            # 配置回测参数
            config = akquant.BacktestConfig(
                initial_capital=DEFAULT_CAPITAL,
                commission_rate=DEFAULT_COMMISSION,
                stamp_tax_rate=DEFAULT_STAMP_TAX,
                slippage=DEFAULT_SLIPPAGE,
                lot_size=BACKTEST_LOT_SIZE,
                t_plus_one=True,
            )

            # 运行回测
            result = akquant.run_backtest(
                strategy=inst,
                data=data,
                config=config,
            )

            if result is None:
                return signals

            # 提取交易信号
            signals = self._extract_signals_from_akquant_result(
                result, strategy_name, codes
            )

        except Exception as e:
            logger.debug(f"AKQuant 扫描 {strategy_name}: {e}")

        return signals

    def _extract_signals_from_akquant_result(self, result,
                                             strategy_name: str,
                                             codes: List[str]
                                             ) -> List[TradeSignal]:
        """从 AKQuant 回测结果中提取最新交易信号"""
        signals: List[TradeSignal] = []

        try:
            # 获取交易记录
            trades = getattr(result, "trades", None)
            if trades is None:
                trades = getattr(result, "trades_df", None)
                if trades is not None and isinstance(trades, pd.DataFrame):
                    trades = trades.to_dict("records")

            if not trades:
                return signals

            # 取最近一日的交易
            today = datetime.now().date()
            trade_records = trades if isinstance(trades, list) else []

            for t in trade_records:
                trade_date = t.get("date", t.get("trade_date", ""))
                if isinstance(trade_date, pd.Timestamp):
                    trade_date = trade_date.date()
                elif isinstance(trade_date, str):
                    try:
                        trade_date = datetime.strptime(
                            trade_date[:10], "%Y-%m-%d"
                        ).date()
                    except Exception:
                        continue

                if trade_date != today:
                    continue

                code = str(t.get("code", t.get("symbol", "")))
                if code not in codes:
                    continue

                action = str(t.get("action", t.get("side", ""))).lower()
                signal_type = "buy" if "buy" in action or "long" in action else "sell"
                price = float(t.get("price", t.get("avg_price", 0)))
                name = self._get_stock_name(code)

                signals.append(TradeSignal(
                    code=code, name=name, strategy=strategy_name,
                    signal_type=signal_type,
                    confidence="medium" if "close" not in action else "high",
                    price=price,
                    reason=f"AKQuant引擎信号: {action}",
                    timestamp=datetime.now(),
                ))
        except Exception as e:
            logger.debug(f"提取AKQ信号异常: {e}")

        return signals

    def _scan_with_indicators(self, strategy_name: str, label: str,
                              codes: List[str], start: str,
                              verbose: bool) -> List[TradeSignal]:
        """降级扫描 - 使用技术指标分析（不依赖 AKQuant）"""
        signals: List[TradeSignal] = []

        for code in codes:
            try:
                df = self._get_cached_data(code, start)
                if df is None or df.empty or len(df) < 40:
                    continue

                df_std = self._normalize_df(df)

                # 综合分析
                analysis = comprehensive_analysis(df_std)
                trend = trend_score(df_std)

                # 根据策略类型提取买卖点
                sig = self._generate_signal_from_indicators(
                    code, strategy_name, df_std, analysis, trend
                )
                if sig:
                    signals.append(sig)

            except Exception as e:
                logger.debug(f"指标扫描 {code}: {e}")

        # 打分排序，只返回高置信度信号
        signals.sort(key=lambda s: self._signal_score(s), reverse=True)
        # 每种信号类型最多返回前N个
        max_signals = {"buy": 10, "sell": 5}
        filtered = []
        type_counts: Dict[str, int] = {}
        for s in signals:
            limit = max_signals.get(s.signal_type, 5)
            if type_counts.get(s.signal_type, 0) < limit:
                filtered.append(s)
                type_counts[s.signal_type] = type_counts.get(s.signal_type, 0) + 1

        return filtered

    def _generate_signal_from_indicators(self, code: str, strategy: str,
                                         df: pd.DataFrame,
                                         analysis: Dict[str, Any],
                                         trend: Dict[str, Any]
                                         ) -> Optional[TradeSignal]:
        """根据技术指标生成买卖信号"""
        close = df["close"].iloc[-1]
        name = self._get_stock_name(code)

        # 检查 MACD 金叉/死叉
        macd_result = analysis.get("macd")
        kdj_result = analysis.get("kdj")
        rsi_result = analysis.get("rsi")

        rsi_val = 50
        if rsi_result and rsi_result.details:
            rsi_val = float(rsi_result.details.get("rsi", 50))

        # 策略特定信号
        if strategy == "dragon_head":
            if "金叉" in str(macd_result.signal if macd_result else ""):
                return TradeSignal(
                    code=code, name=name, strategy=strategy,
                    signal_type="buy", confidence="medium",
                    price=close, reason=f"MACD金叉+趋势评分{trend.get('score', 50)}"
                )

        elif strategy == "sparrow":
            if kdj_result and kdj_result.signal == "金叉" and rsi_val < 65:
                return TradeSignal(
                    code=code, name=name, strategy=strategy,
                    signal_type="buy", confidence="medium",
                    price=close, reason=f"KDJ金叉 RSI={rsi_val:.1f}"
                )
            if kdj_result and kdj_result.signal == "死叉":
                return TradeSignal(
                    code=code, name=name, strategy=strategy,
                    signal_type="sell", confidence="medium",
                    price=close, reason=f"KDJ死叉 RSI={rsi_val:.1f}"
                )

        elif strategy == "turtle":
            if trend.get("rating", "") in ("强烈看多", "看多"):
                return TradeSignal(
                    code=code, name=name, strategy=strategy,
                    signal_type="buy", confidence="low",
                    price=close, reason=f"趋势跟踪:{trend.get('rating')}"
                )

        elif strategy == "value_invest":
            if "金叉" in str(macd_result.signal if macd_result else "") and rsi_val < 45:
                return TradeSignal(
                    code=code, name=name, strategy=strategy,
                    signal_type="buy", confidence="low",
                    price=close, reason=f"低估区域 MACD金叉"
                )

        return None

    def _signal_score(self, signal: TradeSignal) -> int:
        """信号评分排序"""
        conf_map = {"high": 3, "medium": 2, "low": 1}
        action_map = {"buy": 5, "add": 3, "reduce": 2, "sell": 1, "close": 1}
        return conf_map.get(signal.confidence, 1) * 10 + action_map.get(signal.signal_type, 0)

    def _get_cached_data(self, code: str, start: str) -> Optional[pd.DataFrame]:
        """获取数据（带缓存）"""
        with self._cache_lock:
            if code in self._data_cache:
                return self._data_cache[code]

        try:
            df = self.fetcher.fetch_daily_kline(code, start_date=start)
            if df is not None and not df.empty:
                with self._cache_lock:
                    self._data_cache[code] = df
                return df
        except Exception:
            pass
        return None

    def _normalize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        col_map = {"日期": "date", "开盘": "open", "收盘": "close",
                   "最高": "high", "最低": "low", "成交量": "volume",
                   "成交额": "amount", "涨跌幅": "pct_change", "换手率": "turnover"}
        return df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    def _get_stock_name(self, code: str) -> str:
        try:
            df = self.fetcher.fetch_stock_list()
            row = df[df["股票代码"].astype(str) == str(code)]
            if not row.empty:
                return str(row.iloc[0]["股票简称"])
        except Exception:
            pass
        return code

    def clear_cache(self):
        """清除数据缓存"""
        with self._cache_lock:
            self._data_cache.clear()

    def build_trading_plan(self, scan_results: Dict[str, SignalResult],
                           market_sentiment: str = "neutral"
                           ) -> TradingPlan:
        """根据信号扫描结果生成交易计划"""
        all_signals: List[TradeSignal] = []
        position_changes: Dict[str, Dict[str, int]] = {}

        for strategy_name, result in scan_results.items():
            label = result.strategy_label
            buys = [s for s in result.signals if s.signal_type in ("buy", "add")]
            sells = [s for s in result.signals if s.signal_type in ("sell", "reduce", "close")]
            position_changes[label] = {
                "buy": len(buys), "sell": len(sells), "hold": 0
            }
            all_signals.extend(result.signals)

        # 根据市场情绪调整信号优先级
        if market_sentiment == "bearish":
            # 空头市场，降低买入信号权重，提高卖出
            all_signals.sort(key=lambda s: (
                -1 if s.signal_type in ("sell", "reduce", "close") else 1,
                self._signal_score(s)
            ), reverse=False)
        else:
            all_signals.sort(key=self._signal_score, reverse=True)

        summary_parts = []
        for label, changes in position_changes.items():
            if changes["buy"] or changes["sell"]:
                summary_parts.append(
                    f"{label}: 买{changes['buy']} 卖{changes['sell']}"
                )

        return TradingPlan(
            date=datetime.now().strftime("%Y-%m-%d"),
            created_at=datetime.now(),
            signals=all_signals,
            market_sentiment=market_sentiment,
            plan_summary="; ".join(summary_parts) if summary_parts else "今日无交易信号",
            position_changes=position_changes,
        )
