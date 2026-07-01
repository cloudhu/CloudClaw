"""
参数优化框架 - 基于 AKQuant 教材第11章

提供网格搜索和 Walk-Forward Optimization (WFO) 能力。

教材参考: AKQuant Textbook §11 - 参数优化与稳健性检验

用法:
    from .param_optimizer import ParamGrid, optimize_grid_search

    grid = ParamGrid(
        param_ranges={"ma_fast": [5, 10, 20], "ma_slow": [20, 30, 60]},
        metric="sharpe_ratio",
    )
    results = optimize_grid_search(strategy_cls, grid, stock_pool, start, end)
"""

import itertools
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .backtest_engine import BacktestEngine, BacktestResult
from .strategies.base import CloudKnightStrategy

logger = logging.getLogger(__name__)


@dataclass
class ParamGrid:
    """参数网格定义

    教材参考: §11.1 网格搜索
    """

    param_ranges: dict[str, list[Any]]       # 参数名 -> 可取值列表
    metric: str = "sharpe_ratio"             # 优化目标指标
    maximize: bool = True                     # 是否最大化

    def __iter__(self):
        """生成所有参数组合"""
        keys = list(self.param_ranges.keys())
        values = list(self.param_ranges.values())
        for combo in itertools.product(*values):
            yield dict(zip(keys, combo))

    def __len__(self) -> int:
        """参数组合总数"""
        n = 1
        for v in self.param_ranges.values():
            n *= len(v)
        return n


@dataclass
class ParamResult:
    """单组参数的优化结果"""

    params: dict[str, Any]
    metric_name: str
    metric_value: float
    backtest_result: BacktestResult | None = None

    # 从 BacktestResult 提取的关键指标
    total_return: float = 0.0
    annual_return: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    calmar_ratio: float = 0.0
    sortino_ratio: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0


@dataclass
class OptimizationReport:
    """参数优化报告"""

    strategy_name: str
    param_grid: ParamGrid
    start_date: str
    end_date: str
    total_combinations: int
    elapsed_seconds: float
    best_params: dict[str, Any]
    best_metric: str
    best_metric_value: float
    top_n: list[ParamResult] = field(default_factory=list)
    all_results: list[ParamResult] = field(default_factory=list)
    param_sensitivity: dict[str, list[dict]] = field(default_factory=dict)

    def summary(self) -> str:
        """生成优化报告摘要"""
        lines = [
            f"═══════════════════════════════════════",
            f"  参数优化报告: {self.strategy_name}",
            f"═══════════════════════════════════════",
            f"回测区间: {self.start_date} ~ {self.end_date}",
            f"参数组合: {self.total_combinations}",
            f"优化指标: {self.best_metric}",
            f"耗时:     {self.elapsed_seconds:.1f}s",
            f"───────────────────────────────────────",
            f"  🏆 最优参数:",
        ]
        for k, v in self.best_params.items():
            lines.append(f"     {k} = {v}")
        lines.append(f"───────────────────────────────────────")
        lines.append(f"  最优 {self.best_metric}: {self.best_metric_value:+.4f}")
        lines.append(f"───────────────────────────────────────")
        if self.top_n:
            lines.append(f"  Top {min(5, len(self.top_n))} 参数组合:")
            for i, r in enumerate(self.top_n[:5]):
                params_str = ", ".join(f"{k}={v}" for k, v in r.params.items())
                lines.append(
                    f"  {i+1}. [{r.metric_name}={r.metric_value:+.4f}] "
                    f"年化={r.annual_return:+.1f}% 回撤={r.max_drawdown:.1f}% "
                    f"胜率={r.win_rate:.0f}% | {params_str}"
                )
        lines.append(f"═══════════════════════════════════════")
        return "\n".join(lines)


def optimize_grid_search(
    strategy_cls: type[CloudKnightStrategy],
    param_grid: ParamGrid,
    stock_pool: list[str],
    start_date: str,
    end_date: str,
    metric_fn: Callable[[BacktestResult], float] | None = None,
    verbose: bool = False,
    top_n: int = 10,
) -> OptimizationReport:
    """网格搜索参数优化

    教材参考: §11.1 网格搜索 + §11.2 滚动回测(WFO)

    Args:
        strategy_cls: 策略类
        param_grid: 参数网格
        stock_pool: 股票池
        start_date: 开始日期
        end_date: 结束日期
        metric_fn: 自定义指标计算函数 (BacktestResult -> float)
        verbose: 是否输出进度
        top_n: 保留最优前 N 组

    Returns:
        OptimizationReport
    """
    t0 = time.time()
    engine = BacktestEngine()
    results: list[ParamResult] = []

    total = len(param_grid)
    if verbose:
        logger.info(f"[参数优化] {strategy_cls.name} | 网格: {total} 组合 | 区间: {start_date}~{end_date}")

    for i, params in enumerate(param_grid):
        if verbose and (i == 0 or (i + 1) % max(1, total // 10) == 0):
            logger.info(f"  进度: {i+1}/{total} ({100*(i+1)//total}%)")

        try:
            # 用参数创建策略实例
            strategy_inst = strategy_cls(params=params)

            # 运行回测
            bt_result = engine.run_backtest(
                strategy_cls=strategy_cls,
                start_date=start_date,
                end_date=end_date,
                stock_pool=stock_pool,
                verbose=False,
            )

            if bt_result is None:
                continue

            # 计算指标值
            if metric_fn:
                metric_value = metric_fn(bt_result)
            else:
                metric_value = _extract_metric(bt_result, param_grid.metric)

            # 提取关联指标
            calmar = 0.0
            sortino = 0.0
            if bt_result.advanced:
                calmar = bt_result.advanced.calmar_ratio
                sortino = bt_result.advanced.sortino_ratio

            pr = ParamResult(
                params=dict(params),
                metric_name=param_grid.metric,
                metric_value=round(float(metric_value), 4),
                backtest_result=bt_result,
                total_return=bt_result.total_return,
                annual_return=bt_result.annual_return,
                max_drawdown=bt_result.max_drawdown,
                sharpe_ratio=bt_result.sharpe_ratio,
                calmar_ratio=calmar,
                sortino_ratio=sortino,
                win_rate=bt_result.win_rate,
                total_trades=bt_result.total_trades,
            )
            results.append(pr)

        except Exception as e:
            logger.debug(f"参数组合 {params} 回测失败: {e}")
            continue

    # 排序
    reverse = param_grid.maximize
    results.sort(key=lambda r: r.metric_value, reverse=reverse)

    # 最优参数
    best = results[0] if results else None
    elapsed = time.time() - t0

    # 参数敏感性分析
    sensitivity = _analyze_sensitivity(results, param_grid)

    report = OptimizationReport(
        strategy_name=strategy_cls.name,
        param_grid=param_grid,
        start_date=start_date,
        end_date=end_date,
        total_combinations=total,
        elapsed_seconds=round(elapsed, 1),
        best_params=best.params if best else {},
        best_metric=param_grid.metric,
        best_metric_value=best.metric_value if best else 0,
        top_n=results[:max(top_n, len(results))],
        all_results=results,
        param_sensitivity=sensitivity,
    )

    if verbose:
        logger.info(report.summary())

    return report


def optimize_walk_forward(
    strategy_cls: type[CloudKnightStrategy],
    param_grid: ParamGrid,
    stock_pool: list[str],
    start_date: str,
    end_date: str,
    train_months: int = 12,
    test_months: int = 3,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """Walk-Forward Optimization 滚动窗口优化

    教材参考: §11.2 滚动回测 (WFO)

    Args:
        strategy_cls: 策略类
        param_grid: 参数网格
        stock_pool: 股票池
        start_date: 整体开始日期
        end_date: 整体结束日期
        train_months: 训练窗口月数
        test_months: 测试窗口月数
        verbose: 是否输出进度

    Returns:
        [{window, train_start, train_end, test_start, test_end, best_params, test_result, ...}]
    """
    import pandas as pd

    window_results = []
    train_start = pd.Timestamp(start_date[:4] + "-" + start_date[4:6] + "-" + start_date[6:8])
    final_end = pd.Timestamp(end_date[:4] + "-" + end_date[4:6] + "-" + end_date[6:8])

    window_idx = 0
    while True:
        # 训练窗口
        train_end = train_start + pd.DateOffset(months=train_months)
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=test_months)

        if test_end > final_end:
            break

        train_s = train_start.strftime("%Y%m%d")
        train_e = train_end.strftime("%Y%m%d")
        test_s = test_start.strftime("%Y%m%d")
        test_e = test_end.strftime("%Y%m%d")

        if verbose:
            logger.info(f"[WFO] 窗口 {window_idx+1}: 训练 {train_s}~{train_e} | 测试 {test_s}~{test_e}")

        # 在训练窗口内网格搜索最优参数
        train_report = optimize_grid_search(
            strategy_cls=strategy_cls,
            param_grid=param_grid,
            stock_pool=stock_pool,
            start_date=train_s,
            end_date=train_e,
            verbose=False,
            top_n=1,
        )

        best_params = train_report.best_params

        # 在测试窗口验证最优参数
        engine = BacktestEngine()
        test_result = engine.run_backtest(
            strategy_cls=strategy_cls,
            start_date=test_s,
            end_date=test_e,
            stock_pool=stock_pool,
            verbose=False,
        )

        window_results.append({
            "window": window_idx + 1,
            "train_start": train_s,
            "train_end": train_e,
            "test_start": test_s,
            "test_end": test_e,
            "best_params": best_params,
            "train_best_metric": train_report.best_metric_value,
            "test_total_return": test_result.total_return if test_result else 0,
            "test_annual_return": test_result.annual_return if test_result else 0,
            "test_max_drawdown": test_result.max_drawdown if test_result else 0,
            "test_sharpe": test_result.sharpe_ratio if test_result else 0,
            "test_win_rate": test_result.win_rate if test_result else 0,
        })

        # 推进窗口
        train_start = train_start + pd.DateOffset(months=test_months)
        window_idx += 1

    if window_results and verbose:
        avg_annual = np.mean([w["test_annual_return"] for w in window_results])
        avg_sharpe = np.mean([w["test_sharpe"] for w in window_results])
        logger.info(
            f"[WFO] 完成 {len(window_results)} 窗口 | "
            f"平均年化: {avg_annual:+.2f}% | 平均夏普: {avg_sharpe:.2f}"
        )

    return window_results


def _extract_metric(result: BacktestResult, metric: str) -> float:
    """从 BacktestResult 提取指标值"""
    metric_map = {
        "total_return": result.total_return,
        "annual_return": result.annual_return,
        "max_drawdown": result.max_drawdown,
        "sharpe_ratio": result.sharpe_ratio,
        "win_rate": result.win_rate,
        "profit_factor": result.profit_factor,
    }

    # 高级指标
    if result.advanced:
        metric_map.update({
            "sortino_ratio": result.advanced.sortino_ratio,
            "calmar_ratio": result.advanced.calmar_ratio,
            "annual_volatility": result.advanced.annual_volatility,
        })

    return metric_map.get(metric, 0.0)


def _analyze_sensitivity(
    results: list[ParamResult], param_grid: ParamGrid
) -> dict[str, list[dict]]:
    """参数敏感性分析

    对每个参数，计算在不同取值下的平均指标表现。
    """
    if not results:
        return {}

    sensitivity = {}
    for param_name in param_grid.param_ranges:
        value_metrics: dict[Any, list[float]] = {}
        for pr in results:
            val = pr.params.get(param_name)
            if val not in value_metrics:
                value_metrics[val] = []
            value_metrics[val].append(pr.metric_value)

        items = [
            {"value": k, "avg_metric": round(float(np.mean(v)), 4), "count": len(v)}
            for k, v in value_metrics.items()
        ]
        items.sort(key=lambda x: x["avg_metric"], reverse=param_grid.maximize)
        sensitivity[param_name] = items

    return sensitivity
