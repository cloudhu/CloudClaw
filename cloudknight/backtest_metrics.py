"""
高级回测指标 - 基于 AKQuant 教材第10章评价体系

在基础指标（收益率、夏普、最大回撤、胜率、盈亏比）之上补充：
- Sortino Ratio（下行风险调整收益）
- Calmar Ratio（年化收益/最大回撤）
- Alpha / Beta 系数
- Information Ratio（信息比率）
- VaR / CVaR（在险价值）
- 月度/年度收益率统计
- 最大连续亏损/盈利期
- 换手率统计

教材参考: AKQuant Textbook §10 - 策略评价体系与风险指标
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 无风险利率（中国十年期国债收益率年化约 2.5%）
RISK_FREE_RATE = 0.025


@dataclass
class AdvancedMetrics:
    """高级回测指标"""

    # ─── 收益率指标 ───
    total_return: float = 0.0          # 总收益率 (%)
    annual_return: float = 0.0         # 年化收益率 (%)
    monthly_returns: list[dict] = field(default_factory=list)  # 月度收益表
    yearly_returns: list[dict] = field(default_factory=list)   # 年度收益表

    # ─── 风险指标 ───
    max_drawdown: float = 0.0          # 最大回撤 (%)
    max_drawdown_duration: int = 0     # 最大回撤持续天数
    annual_volatility: float = 0.0     # 年化波动率 (%)
    downside_volatility: float = 0.0   # 下行波动率 (%)

    # ─── 风险调整收益 ───
    sharpe_ratio: float = 0.0          # 夏普比率
    sortino_ratio: float = 0.0         # Sortino 比率
    calmar_ratio: float = 0.0          # Calmar 比率
    information_ratio: float = 0.0     # 信息比率

    # ─── 市场相关 ───
    alpha: float = 0.0                 # Alpha（年化 %）
    beta: float = 0.0                  # Beta 系数
    r_squared: float = 0.0             # R²（拟合优度）

    # ─── 风控指标 ───
    var_95: float = 0.0                # 95% VaR（%）
    cvar_95: float = 0.0               # 95% CVaR（%）
    var_99: float = 0.0                # 99% VaR（%）

    # ─── 交易统计 ───
    win_rate: float = 0.0              # 胜率 (%)
    profit_factor: float = 0.0         # 盈亏比
    avg_profit: float = 0.0            # 平均盈利
    avg_loss: float = 0.0              # 平均亏损
    max_consecutive_wins: int = 0      # 最大连续盈利
    max_consecutive_losses: int = 0    # 最大连续亏损
    total_trades: int = 0              # 总交易次数

    # ─── 组合指标 ───
    avg_holding_days: float = 0.0      # 平均持仓天数
    turnover_rate: float = 0.0         # 换手率（年化 %）
    daily_win_rate: float = 0.0        # 日胜率 (%)

    def to_dict(self) -> dict[str, Any]:
        """转为字典供 JSON 序列化"""
        return {
            "total_return": round(self.total_return, 2),
            "annual_return": round(self.annual_return, 2),
            "max_drawdown": round(self.max_drawdown, 2),
            "annual_volatility": round(self.annual_volatility, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 2),
            "sortino_ratio": round(self.sortino_ratio, 2),
            "calmar_ratio": round(self.calmar_ratio, 2),
            "alpha": round(self.alpha, 2),
            "beta": round(self.beta, 2),
            "var_95": round(self.var_95, 2),
            "cvar_95": round(self.cvar_95, 2),
            "win_rate": round(self.win_rate, 2),
            "profit_factor": round(self.profit_factor, 2),
            "total_trades": self.total_trades,
            "max_consecutive_wins": self.max_consecutive_wins,
            "max_consecutive_losses": self.max_consecutive_losses,
            "avg_holding_days": round(self.avg_holding_days, 1),
            "turnover_rate": round(self.turnover_rate, 2),
            "monthly_returns": self.monthly_returns,
            "yearly_returns": self.yearly_returns,
        }


def compute_advanced_metrics(
    equity_curve: pd.DataFrame,
    daily_returns: pd.Series | None = None,
    trades_df: pd.DataFrame | None = None,
    benchmark_returns: pd.Series | None = None,
    initial_capital: float = 1_000_000,
    risk_free_rate: float = RISK_FREE_RATE,
) -> AdvancedMetrics:
    """从回测结果计算完整的高级指标

    Args:
        equity_curve: 权益曲线 DataFrame，需含 'equity' 列和日期索引
        daily_returns: 日收益率序列（若为 None 则从 equity_curve 计算）
        trades_df: 交易记录 DataFrame
        benchmark_returns: 基准日收益率序列（用于 Alpha/Beta 计算）
        initial_capital: 初始资金
        risk_free_rate: 无风险利率（年化）

    Returns:
        AdvancedMetrics 对象
    """
    m = AdvancedMetrics()

    if equity_curve is None or equity_curve.empty:
        return m

    # ─── 收益率分析 ────────────────────────────────
    equity = equity_curve["equity"].values if "equity" in equity_curve.columns else None
    if equity is None:
        return m

    # 日收益率
    if daily_returns is None:
        daily_returns = pd.Series(equity).pct_change().dropna()

    rets = daily_returns.values if isinstance(daily_returns, pd.Series) else daily_returns
    if len(rets) == 0:
        return m

    # 总收益 & 年化收益
    final_equity = float(equity[-1])
    m.total_return = (final_equity / initial_capital - 1) * 100
    n_days = len(rets)
    m.annual_return = ((final_equity / initial_capital) ** (252 / max(n_days, 1)) - 1) * 100

    # 年化波动率
    daily_vol = float(np.std(rets))
    m.annual_volatility = daily_vol * np.sqrt(252) * 100

    # 下行波动率
    downside_rets = rets[rets < 0] if len(rets[rets < 0]) > 0 else np.array([0])
    m.downside_volatility = float(np.std(downside_rets)) * np.sqrt(252) * 100

    # ─── 最大回撤分析 ──────────────────────────────
    peak = np.maximum.accumulate(equity)
    drawdowns = (peak - equity) / peak * 100
    m.max_drawdown = float(np.max(drawdowns))

    # 最大回撤持续天数
    dd_start = 0
    max_dd_days = 0
    in_dd = False
    current_dd_days = 0
    for i in range(len(drawdowns)):
        if drawdowns[i] > 0 and not in_dd:
            dd_start = i
            in_dd = True
            current_dd_days = 1
        elif drawdowns[i] > 0 and in_dd:
            current_dd_days += 1
        elif drawdowns[i] == 0 and in_dd:
            max_dd_days = max(max_dd_days, current_dd_days)
            in_dd = False
    m.max_drawdown_duration = max_dd_days

    # ─── 夏普比率 ──────────────────────────────────
    daily_rf = risk_free_rate / 252
    excess_rets = rets - daily_rf
    if daily_vol > 0:
        m.sharpe_ratio = float(np.mean(excess_rets) / daily_vol * np.sqrt(252))

    # ─── Sortino 比率 ──────────────────────────────
    if len(downside_rets) > 0:
        down_stdev = float(np.std(downside_rets)) * np.sqrt(252)
        if down_stdev > 0:
            m.sortino_ratio = float((np.mean(rets) * 252 - risk_free_rate) / down_stdev)

    # ─── Calmar 比率 ───────────────────────────────
    if m.max_drawdown > 0.01:
        m.calmar_ratio = m.annual_return / m.max_drawdown

    # ─── VaR / CVaR ────────────────────────────────
    if len(rets) > 20:
        m.var_95 = float(np.percentile(rets, 5)) * 100   # 日 VaR 百分比
        m.var_99 = float(np.percentile(rets, 1)) * 100
        cvar_95_vals = rets[rets <= np.percentile(rets, 5)]
        if len(cvar_95_vals) > 0:
            m.cvar_95 = float(np.mean(cvar_95_vals)) * 100

    # ─── Alpha / Beta ──────────────────────────────
    if benchmark_returns is not None and len(benchmark_returns) == len(rets):
        try:
            bench = benchmark_returns.values if isinstance(benchmark_returns, pd.Series) else benchmark_returns
            cov_matrix = np.cov(rets, bench)
            bench_var = np.var(bench)
            if bench_var > 0:
                m.beta = float(cov_matrix[0, 1] / bench_var)
                m.alpha = float((np.mean(rets) - daily_rf - m.beta * (np.mean(bench) - daily_rf)) * 252 * 100)

                # R²
                corr = np.corrcoef(rets, bench)[0, 1]
                m.r_squared = float(corr ** 2)
        except Exception as e:
            logger.debug(f"Alpha/Beta 计算异常: {e}")

    # ─── 信息比率 ──────────────────────────────────
    if benchmark_returns is not None and len(benchmark_returns) == len(rets):
        try:
            bench = benchmark_returns.values if isinstance(benchmark_returns, pd.Series) else benchmark_returns
            tracking_error = np.std(rets - bench) * np.sqrt(252)
            if tracking_error > 0:
                m.information_ratio = float((np.mean(rets) - np.mean(bench)) * 252 / tracking_error)
        except Exception:
            pass

    # ─── 月度/年度收益 ──────────────────────────────
    if isinstance(daily_returns, pd.Series) and isinstance(daily_returns.index, pd.DatetimeIndex):
        try:
            monthly = daily_returns.resample("ME").apply(lambda x: (1 + x).prod() - 1) * 100
            m.monthly_returns = [
                {"month": d.strftime("%Y-%m"), "return_pct": round(float(v), 2)}
                for d, v in monthly.items() if not pd.isna(v)
            ]
            yearly = daily_returns.resample("YE").apply(lambda x: (1 + x).prod() - 1) * 100
            m.yearly_returns = [
                {"year": d.strftime("%Y"), "return_pct": round(float(v), 2)}
                for d, v in yearly.items() if not pd.isna(v)
            ]
        except Exception:
            pass

    # ─── 日胜率 ────────────────────────────────────
    if len(rets) > 0:
        m.daily_win_rate = float(np.sum(rets > 0) / len(rets) * 100)

    # ─── 交易统计分析 ──────────────────────────────
    if trades_df is not None and not trades_df.empty:
        pnl_col = None
        for col in ["pnl", "profit", "pnl_pct"]:
            if col in trades_df.columns:
                pnl_col = col
                break

        if pnl_col:
            pnls = trades_df[pnl_col].dropna().values
            m.total_trades = len(pnls)

            if m.total_trades > 0:
                profits = pnls[pnls > 0]
                losses = pnls[pnls < 0]
                m.win_rate = len(profits) / m.total_trades * 100
                m.avg_profit = float(np.mean(profits)) if len(profits) > 0 else 0
                m.avg_loss = float(abs(np.mean(losses))) if len(losses) > 0 else 0
                total_profit = float(np.sum(profits)) if len(profits) > 0 else 0
                total_loss = float(abs(np.sum(losses))) if len(losses) > 0 else 0
                m.profit_factor = round(total_profit / total_loss, 2) if total_loss > 0 else 999

                # 连续胜/负
                max_cw, max_cl = 0, 0
                cw, cl = 0, 0
                for p in pnls:
                    if p > 0:
                        cw += 1
                        cl = 0
                        max_cw = max(max_cw, cw)
                    elif p < 0:
                        cl += 1
                        cw = 0
                        max_cl = max(max_cl, cl)
                m.max_consecutive_wins = max_cw
                m.max_consecutive_losses = max_cl

    return m


def compute_benchmark_returns(benchmark_data: pd.DataFrame, col: str = "close") -> pd.Series:
    """从基准数据计算日收益率"""
    if benchmark_data is None or benchmark_data.empty:
        return pd.Series(dtype=float)
    if col in benchmark_data.columns:
        return benchmark_data[col].pct_change().dropna()
    return pd.Series(dtype=float)


def metrics_summary(metrics: AdvancedMetrics) -> str:
    """生成可读的指标摘要"""
    lines = [
        "═══════════════════════════════════════",
        "        高级回测指标摘要",
        "═══════════════════════════════════════",
        f"总收益率:    {metrics.total_return:>8.2f}%",
        f"年化收益率:  {metrics.annual_return:>8.2f}%",
        f"年化波动率:  {metrics.annual_volatility:>8.2f}%",
        f"最大回撤:    {metrics.max_drawdown:>8.2f}% (持续 {metrics.max_drawdown_duration} 天)",
        f"日胜率:      {metrics.daily_win_rate:>8.2f}%",
        "───────────────────────────────────────",
        f"夏普比率:    {metrics.sharpe_ratio:>8.2f}",
        f"Sortino:     {metrics.sortino_ratio:>8.2f}",
        f"Calmar:      {metrics.calmar_ratio:>8.2f}",
        f"Alpha:       {metrics.alpha:>8.2f}%",
        f"Beta:        {metrics.beta:>8.2f}",
        f"信息比率:    {metrics.information_ratio:>8.2f}",
        "───────────────────────────────────────",
        f"95% VaR:     {metrics.var_95:>8.2f}%",
        f"95% CVaR:    {metrics.cvar_95:>8.2f}%",
        "───────────────────────────────────────",
        f"胜率:        {metrics.win_rate:>8.2f}%",
        f"盈亏比:      {metrics.profit_factor:>8.2f}",
        f"最大连胜:    {metrics.max_consecutive_wins:>8} 笔",
        f"最大连亏:    {metrics.max_consecutive_losses:>8} 笔",
        f"总交易次数:  {metrics.total_trades:>8}",
        "═══════════════════════════════════════",
    ]
    return "\n".join(lines)
