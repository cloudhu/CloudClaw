"""
回测引擎 - 基于 AKQuant 高性能事件驱动回测

使用 akquant.run_backtest() 替代自定义向量化回测，
享受 Rust 内核带来的 20x 性能提升。

集成高级回测指标（教材 §10）：
  - Sortino 比率、Calmar 比率、Alpha/Beta
  - VaR/CVaR、信息比率
  - 月度/年度收益统计、连续胜/负期
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .backtest_metrics import AdvancedMetrics, compute_advanced_metrics
from .config import (
    BACKTEST_END_DATE,
    BACKTEST_LOT_SIZE,
    BACKTEST_MIN_COMMISSION,
    BACKTEST_START_DATE,
    BACKTEST_T_PLUS_ONE,
    DEFAULT_CAPITAL,
    DEFAULT_COMMISSION,
    DEFAULT_SLIPPAGE,
    DEFAULT_STAMP_TAX,
)
from .data_manager import DataFetcher
from .strategies.base import CloudKnightStrategy

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """统一回测结果（兼容原有接口 + 高级指标）"""

    strategy_name: str
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float
    total_return: float  # 总收益率 (%)
    annual_return: float  # 年化收益率 (%)
    max_drawdown: float  # 最大回撤 (%)
    sharpe_ratio: float  # 夏普比率
    win_rate: float  # 胜率 (%)
    total_trades: int  # 总交易次数
    profit_trades: int  # 盈利交易次数
    avg_profit: float  # 平均盈利
    avg_loss: float  # 平均亏损
    profit_factor: float  # 盈亏比
    daily_returns: list[float] = field(default_factory=list)
    trades: list[dict] = field(default_factory=list)
    equity_curve: pd.DataFrame | None = None
    raw_result: object = None  # AKQuant 原始 BacktestResult 对象

    # 高级指标（教材 §10）
    advanced: AdvancedMetrics | None = None


class BacktestEngine:
    """AKQuant 回测包装器"""

    def __init__(
        self,
        capital: float | None = None,
        commission: float | None = None,
        stamp_tax: float | None = None,
        slippage: float | None = None,
    ):
        self.initial_capital = capital or DEFAULT_CAPITAL
        self.commission = commission or DEFAULT_COMMISSION
        self.stamp_tax = stamp_tax or DEFAULT_STAMP_TAX
        self.slippage = slippage or DEFAULT_SLIPPAGE
        self.fetcher = DataFetcher()

    def run_backtest(
        self,
        strategy_cls: type[CloudKnightStrategy],
        start_date: str | None = None,
        end_date: str | None = None,
        stock_pool: list[str] | None = None,
        verbose: bool = False,
    ) -> BacktestResult | None:
        """使用 AKQuant 执行单策略回测

        Args:
            strategy_cls: 策略类（继承 CloudKnightStrategy）
            start_date: 开始日期 YYYYMMDD
            end_date: 结束日期 YYYYMMDD
            stock_pool: 股票池代码列表
            verbose: 是否显示进度
        """
        start_date = start_date or BACKTEST_START_DATE
        end_date = end_date or BACKTEST_END_DATE
        stock_pool = (stock_pool or [])[:50]  # 最多 50 只

        # 准备 AKQuant 格式数据
        data = self._prepare_data(stock_pool, start_date, end_date)
        if not data:
            logger.error("无法获取回测数据")
            return None

        if verbose:
            symbols = (
                list(data.keys())
                if isinstance(data, dict)
                else data["symbol"].nunique()
                if isinstance(data, pd.DataFrame)
                else 0
            )
            logger.info(f"回测: {start_date} ~ {end_date}, {len(stock_pool) if stock_pool else symbols} 只股票")

        try:
            result = self._run_akquant_backtest(
                data=data,
                strategy_cls=strategy_cls,
                start_date=start_date,
                end_date=end_date,
            )
            if result is None:
                return None
            return self._parse_akquant_result(result, strategy_cls.name, start_date, end_date)
        except Exception as e:
            logger.error(f"AKQuant 回测失败: {e}")
            return None

    def compare_strategies(
        self,
        strategy_classes: list[type[CloudKnightStrategy]],
        start_date: str | None = None,
        end_date: str | None = None,
        stock_pool: list[str] | None = None,
        verbose: bool = False,
    ) -> list[BacktestResult]:
        """多策略对比回测（逐个运行）"""
        results = []
        for cls in strategy_classes:
            try:
                r = self.run_backtest(cls, start_date, end_date, stock_pool, verbose)
                if r:
                    results.append(r)
            except Exception as e:
                logger.error(f"策略 {cls.name} 回测异常: {e}")
        return results

    # ─── 内部方法 ───────────────────────────────────────

    def _prepare_data(self, stock_pool: list[str], start_date: str, end_date: str) -> dict[str, pd.DataFrame] | None:
        """获取并格式化数据为 AKQuant 兼容格式"""
        data = {}
        for code in stock_pool:
            try:
                df = self.fetcher.fetch_daily_kline(str(code), start_date=start_date, end_date=end_date)
                if df is None or df.empty or len(df) < 20:
                    continue

                # 标准化列名
                col_map = {
                    "日期": "date",
                    "开盘": "open",
                    "收盘": "close",
                    "最高": "high",
                    "最低": "low",
                    "成交量": "volume",
                    "成交额": "amount",
                }
                df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

                # 确保必要列存在
                required = ["date", "open", "high", "low", "close", "volume"]
                if not all(c in df.columns for c in required):
                    continue

                df = df[[*required, "amount"] if "amount" in df.columns else required].copy()
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df = df.dropna(subset=["date"])
                df = df.sort_values("date")
                df["symbol"] = str(code)

                if len(df) >= 20:
                    data[str(code)] = df[["date", "open", "high", "low", "close", "volume", "symbol"]]
            except Exception as e:
                logger.debug(f"获取 {code} 数据失败: {e}")
                continue

        return data if data else None

    def _run_akquant_backtest(
        self,
        data: dict[str, pd.DataFrame],
        strategy_cls: type[CloudKnightStrategy],
        start_date: str | None = None,
        end_date: str | None = None,
    ):
        """调用 AKQuant 回测引擎"""
        try:
            from akquant import run_backtest
        except ImportError:
            logger.error("未安装 akquant, 请执行: pip install akquant")
            return None

        kwargs = {
            "data": data,
            "strategy": strategy_cls,
            "initial_cash": self.initial_capital,
            "commission_policy": {"type": "percent", "value": self.commission},
            "stamp_tax_rate": self.stamp_tax,
            "slippage": self.slippage,
            "t_plus_one": BACKTEST_T_PLUS_ONE,
            "lot_size": BACKTEST_LOT_SIZE,
            "min_commission": BACKTEST_MIN_COMMISSION,
        }

        if start_date:
            kwargs["start_time"] = pd.Timestamp(start_date[:4] + "-" + start_date[4:6] + "-" + start_date[6:8])
        if end_date:
            kwargs["end_time"] = pd.Timestamp(end_date[:4] + "-" + end_date[4:6] + "-" + end_date[6:8])

        strategy_inst = strategy_cls()
        warmup = getattr(strategy_inst, "warmup_period", 60)
        if warmup:
            kwargs["warmup_period"] = warmup

        return run_backtest(**kwargs)

    def _parse_akquant_result(self, raw_result, strategy_name: str, start_date: str, end_date: str) -> BacktestResult:
        """解析 AKQuant BacktestResult 为统一格式"""
        initial_cap = self.initial_capital
        final_cap = initial_cap
        total_return = 0.0
        annual_return = 0.0
        max_dd = 0.0
        sharpe = 0.0
        win_rate = 0.0
        total_trades = 0
        profit_trades = 0
        avg_profit = 0.0
        avg_loss = 0.0
        profit_factor = 0.0
        daily_returns = []
        trades_list = []
        equity_df = None

        try:
            # 解析 metrics_df
            metrics = raw_result.metrics_df if hasattr(raw_result, "metrics_df") else None
            if metrics is not None and not metrics.empty:
                cols_lower = {c.lower(): c for c in metrics.columns}
                final_cap = self._safe_metric(
                    metrics, cols_lower, ["final_value", "final_capital", "ending_value", "total_equity"]
                )
                if final_cap == 0:
                    final_cap = initial_cap
                total_return = (final_cap / initial_cap - 1) * 100
                max_dd = (
                    self._safe_metric(metrics, cols_lower, ["max_drawdown", "max_drawdown_pct"]) * 100
                    if metrics is not None
                    else 0
                )
                sharpe = self._safe_metric(metrics, cols_lower, ["sharpe_ratio", "sharpe"])
                win_rate = self._safe_metric(metrics, cols_lower, ["win_rate", "win_ratio"]) * 100
                total_trades = int(self._safe_metric(metrics, cols_lower, ["total_trades", "trade_count"]))

                try:
                    start_dt = pd.Timestamp(start_date[:4] + "-" + start_date[4:6] + "-" + start_date[6:8])
                    end_dt = pd.Timestamp(end_date[:4] + "-" + end_date[4:6] + "-" + end_date[6:8])
                    days = max((end_dt - start_dt).days, 1)
                    annual_return = ((final_cap / initial_cap) ** (365 / days) - 1) * 100
                except Exception:
                    annual_return = total_return

            # 解析交易记录
            trades_df = raw_result.trades_df if hasattr(raw_result, "trades_df") else None
            if trades_df is not None and not trades_df.empty:
                total_trades = max(total_trades, len(trades_df))
                for _, t in trades_df.iterrows():
                    try:
                        pnl = float(t.get("pnl", t.get("profit", 0)))
                        trades_list.append(
                            {
                                "date": str(t.get("date", t.get("exit_time", ""))),
                                "code": str(t.get("symbol", "")),
                                "action": str(t.get("side", "")),
                                "price": float(t.get("exit_price", t.get("price", 0))),
                                "volume": int(t.get("quantity", t.get("qty", 0))),
                                "pnl": pnl,
                            }
                        )
                        if pnl > 0:
                            profit_trades += 1
                        elif pnl < 0:
                            pass  # loss trades counted implicitly
                    except Exception:
                        continue

                total_trades - profit_trades
                profits = [t["pnl"] for t in trades_list if t["pnl"] > 0]
                losses = [t["pnl"] for t in trades_list if t["pnl"] < 0]
                avg_profit = float(np.mean(profits)) if profits else 0
                avg_loss = float(abs(np.mean(losses))) if losses else 0
                total_profit = sum(profits)
                total_loss = abs(sum(losses))
                profit_factor = round(total_profit / total_loss, 2) if total_loss > 0 else 999

            # 解析权益曲线
            equity_df = raw_result.equity_curve if hasattr(raw_result, "equity_curve") else None
            if equity_df is not None and not equity_df.empty:
                daily_returns = (
                    equity_df["equity"].pct_change().dropna().tolist() if "equity" in equity_df.columns else []
                )

        except Exception as e:
            logger.warning(f"解析 AKQuant 结果异常: {e}")

        # 计算高级指标（教材 §10）
        advanced = None
        if equity_df is not None and not equity_df.empty and "equity" in equity_df.columns and len(daily_returns) > 20:
            try:
                # 将 daily_returns 构造成带日期索引的 Series
                dr_series = pd.Series(daily_returns)
                advanced = compute_advanced_metrics(
                    equity_curve=equity_df,
                    daily_returns=dr_series,
                    trades_df=(
                        pd.DataFrame(trades_list) if trades_list else None
                    ),
                    initial_capital=initial_cap,
                )
            except Exception as e:
                logger.debug(f"高级指标计算失败: {e}")

        return BacktestResult(
            strategy_name=strategy_name,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_cap,
            final_capital=round(final_cap, 2),
            total_return=round(total_return, 2),
            annual_return=round(annual_return, 2),
            max_drawdown=round(max_dd, 2),
            sharpe_ratio=round(sharpe, 2),
            win_rate=round(win_rate, 2) if total_trades > 0 else 0,
            total_trades=total_trades,
            profit_trades=profit_trades,
            avg_profit=round(avg_profit, 2),
            avg_loss=round(avg_loss, 2),
            profit_factor=profit_factor,
            daily_returns=daily_returns,
            trades=trades_list[-20:],
            equity_curve=equity_df,
            raw_result=raw_result,
            advanced=advanced,
        )

    @staticmethod
    def _safe_metric(metrics, cols_lower, keys, default=0.0):
        for k in keys:
            if k in cols_lower:
                v = metrics.iloc[0][cols_lower[k]]
                try:
                    return float(v)
                except (ValueError, TypeError):
                    pass
        return float(default)
