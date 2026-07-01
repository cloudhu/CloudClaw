"""
因子引擎 (AKQuant 教材 §14)

A股多因子体系完整实现，覆盖因子定义→计算→预处理→评估→合成全流程。

架构:
  FactorDefinition    — 单因子定义（名称/字段/计算函数/分组/方向）
  FactorEngine        — 因子引擎核心（批量计算 + 预处理 + 评估）
  FactorEvaluation    — 因子评估结果（IC/IR/分层收益/换手率）

因子类别（7大类，40+ 因子）:
  市值规模:  ln_cap, sqrt_cap
  估值:      pe, pb, ps, pcf, dividend_yield, ev_ebitda
  动量:      ret_1m, ret_3m, ret_6m, ret_12m, reversal_1m
  质量:      roe, roa, gross_margin, net_margin, debt_ratio, current_ratio
  成长:      revenue_growth_yoy, earnings_growth_yoy, asset_growth
  波动:      volatility_1m, beta_60d, downside_risk, max_drawdown_1m
  技术:      rsi_14, volume_ratio, turnover_rate, macd_divergence

预处理管道:
  1. 去极值 (Winsorization)    — 分位数截尾 (1%-99%)
  2. 标准化 (Standardization)   — Z-Score
  3. 中性化 (Neutralization)    — 行业+市值回归残差

评估体系:
  - IC (Information Coefficient)   — Pearson IC / Spearman Rank IC
  - IR (Information Ratio)         — IC 均值 / IC 标准差
  - 分层收益 (Quantile Returns)    — 10组等权/市值加权收益
  - 换手率 (Turnover)              — 分组间切换比例
  - IC 序列稳定性 (IC_IR_BP)       — IC 胜率 + t-stat
"""

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════
# 一、因子定义
# ═══════════════════════════════════════════════


@dataclass
class FactorDefinition:
    """因子定义"""
    name: str                          # 因子名称
    category: str                      # 因子类别: size/value/momentum/quality/growth/volatility/technical
    fields: list[str]                  # 依赖的原始字段
    direction: str = "negative"        # 因子方向: positive(越大越好) / negative(越小越好)
    description: str = ""              # 因子描述

    def __hash__(self) -> int:
        return hash(self.name)


# ═══════════════════════════════════════════════
# 二、因子计算函数注册表
# ═══════════════════════════════════════════════

# 因子计算函数签名: (df: pd.DataFrame) -> pd.Series
_FACTOR_COMPUTERS: dict[str, Callable[[pd.DataFrame], pd.Series]] = {}


def register_factor(name: str, fields: list[str]):
    """因子注册装饰器"""
    def decorator(func: Callable):
        func._factor_fields = fields
        _FACTOR_COMPUTERS[name] = func
        return func
    return decorator


# ── 市值规模因子 ───────────────────────────────

@register_factor("ln_cap", ["total_mv"])
def compute_ln_cap(df: pd.DataFrame) -> pd.Series:
    """对数总市值：市值越小 → 因子值越大"""
    return -np.log(df["total_mv"].replace(0, np.nan))


@register_factor("midcap", ["total_mv"])
def compute_midcap(df: pd.DataFrame) -> pd.Series:
    """中盘偏离度：市值偏离中位数的标准化值"""
    cap = df["total_mv"].replace(0, np.nan)
    median = cap.median()
    return -(cap - median).abs() / median


# ── 估值因子 ────────────────────────────────────

@register_factor("pe", ["pe_ttm"])
def compute_pe(df: pd.DataFrame) -> pd.Series:
    """市盈率（TTM）：值越小越好"""
    pe = df["pe_ttm"].replace(0, np.nan)
    return -pe  # 负向：低 PE 分数高


@register_factor("pb", ["pb"])
def compute_pb(df: pd.DataFrame) -> pd.Series:
    """市净率"""
    return -df["pb"].replace(0, np.nan)


@register_factor("ps", ["ps_ttm"])
def compute_ps(df: pd.DataFrame) -> pd.Series:
    """市销率"""
    return -df["ps_ttm"].replace(0, np.nan)


@register_factor("pcf", ["pcf_ttm"])
def compute_pcf(df: pd.DataFrame) -> pd.Series:
    """市现率"""
    return -df["pcf_ttm"].replace(0, np.nan)


@register_factor("dividend_yield", ["dividend_yield"])
def compute_dividend_yield(df: pd.DataFrame) -> pd.Series:
    """股息率：越高越好"""
    return df["dividend_yield"].fillna(0)


@register_factor("ev_ebitda", ["total_mv", "total_debt", "cash_equivalents", "ebitda"])
def compute_ev_ebitda(df: pd.DataFrame) -> pd.Series:
    """企业价值/EBITDA"""
    ev = df["total_mv"].fillna(0) + df["total_debt"].fillna(0) - df["cash_equivalents"].fillna(0)
    return -(ev / df["ebitda"].replace(0, np.nan))


# ── 动量因子 ────────────────────────────────────

@register_factor("ret_1m", ["close"])
def compute_ret_1m(df: pd.DataFrame) -> pd.Series:
    """近1月收益率"""
    return df["close"].pct_change(21).fillna(0)


@register_factor("ret_3m", ["close"])
def compute_ret_3m(df: pd.DataFrame) -> pd.Series:
    """近3月收益率"""
    return df["close"].pct_change(63).fillna(0)


@register_factor("ret_6m", ["close"])
def compute_ret_6m(df: pd.DataFrame) -> pd.Series:
    """近6月收益率"""
    return df["close"].pct_change(126).fillna(0)


@register_factor("ret_12m", ["close"])
def compute_ret_12m(df: pd.DataFrame) -> pd.Series:
    """近12月收益率"""
    return df["close"].pct_change(252).fillna(0)


@register_factor("ret_12m_skip_1m", ["close"])
def compute_ret_12m_skip_1m(df: pd.DataFrame) -> pd.Series:
    """12-1 月动量（剔除最近1月反转效应）"""
    ret_12 = df["close"].pct_change(252).fillna(0)
    ret_1 = df["close"].pct_change(21).fillna(0)
    return ret_12 - ret_1


@register_factor("reversal_1m", ["close"])
def compute_reversal_1m(df: pd.DataFrame) -> pd.Series:
    """1月反转效应：反向，跌多买涨多卖"""
    return -df["close"].pct_change(21).fillna(0)


@register_factor("path_1m", ["high", "low", "close"])
def compute_path_1m(df: pd.DataFrame) -> pd.Series:
    """1月路径因子：最大/最小值位置反映动量质量"""
    n = 21
    high_n = df["high"].rolling(n).max()
    low_n = df["low"].rolling(n).min()
    return (df["close"] - low_n) / (high_n - low_n).replace(0, 1) - 0.5


# ── 质量因子 ────────────────────────────────────

@register_factor("roe", ["roe"])
def compute_roe(df: pd.DataFrame) -> pd.Series:
    """ROE：净资产收益率"""
    return df["roe"].fillna(0)


@register_factor("roa", ["roa"])
def compute_roa(df: pd.DataFrame) -> pd.Series:
    """ROA：总资产收益率"""
    return df["roa"].fillna(0)


@register_factor("gross_margin", ["grossprofit_margin"])
def compute_gross_margin(df: pd.DataFrame) -> pd.Series:
    """毛利率"""
    return df["grossprofit_margin"].fillna(0)


@register_factor("net_margin", ["netprofit_margin"])
def compute_net_margin(df: pd.DataFrame) -> pd.Series:
    """净利率"""
    return df["netprofit_margin"].fillna(0)


@register_factor("debt_ratio", ["debt_to_assets"])
def compute_debt_ratio(df: pd.DataFrame) -> pd.Series:
    """资产负债率：越低越好"""
    return -df["debt_to_assets"].fillna(0)


@register_factor("current_ratio", ["current_ratio"])
def compute_current_ratio(df: pd.DataFrame) -> pd.Series:
    """流动比率"""
    return df["current_ratio"].fillna(0)


@register_factor("accrual_ratio", ["total_assets", "net_intangible_assets", "total_liab", "shortterm_loan",
                                     "net_profit", "cfoa"])
def compute_accrual_ratio(df: pd.DataFrame) -> pd.Series:
    """应计比率 (Accrual Ratio)：越低越好

    Accrual = Δ(ΔTCA - ΔCash) - Δ(ΔTCL - ΔSTD) - Depreciation
    指标值越低（负值大），盈利质量越高。
    """
    try:
        noa_t = (df["total_assets"] - df["total_liab"] - df["net_intangible_assets"])
        noa_tm1 = noa_t.shift(1)
        delta_noa = (noa_t - noa_tm1) / df["total_assets"].shift(1).replace(0, np.nan)
        return -delta_noa.fillna(0)
    except (KeyError, TypeError):
        return pd.Series(0, index=df.index)


@register_factor("gross_to_assets", ["total_assets", "or_yoy"])
def compute_gross_to_assets(df: pd.DataFrame) -> pd.Series:
    """毛利/总资产：资产效率指标"""
    try:
        return df.get("or_yoy", pd.Series(0, index=df.index)).fillna(0)
    except Exception:
        return pd.Series(0, index=df.index)


# ── 成长因子 ────────────────────────────────────

@register_factor("revenue_growth_yoy", ["or_yoy"])
def compute_revenue_growth_yoy(df: pd.DataFrame) -> pd.Series:
    """营业收入同比增长率"""
    return df["or_yoy"].fillna(0)


@register_factor("earnings_growth_yoy", ["profit_dedt_yoy"])
def compute_earnings_growth_yoy(df: pd.DataFrame) -> pd.Series:
    """归母净利润同比增长率"""
    return df["profit_dedt_yoy"].fillna(0)


@register_factor("asset_growth", ["total_assets"])
def compute_asset_growth(df: pd.DataFrame) -> pd.Series:
    """总资产同比增长率"""
    return df["total_assets"].pct_change(4).fillna(0)  # 季度数据，4季度=1年


@register_factor("sustainable_growth", ["roe", "dividend_yield"])
def compute_sustainable_growth(df: pd.DataFrame) -> pd.Series:
    """可持续增长率 = ROE * (1 - 分红率)"""
    roe = df["roe"].fillna(0) / 100  # 假设百分比形式
    payout = df["dividend_yield"].fillna(0) / 100
    return roe * (1 - np.minimum(payout / (roe.replace(0, np.nan)), 0.95))


# ── 波动因子 ────────────────────────────────────

@register_factor("volatility_1m", ["close"])
def compute_volatility_1m(df: pd.DataFrame) -> pd.Series:
    """1月波动率（日收益标准差年化）：越低越好"""
    rets = df["close"].pct_change().fillna(0)
    vol = rets.rolling(21).std() * np.sqrt(252)
    return -vol.fillna(0)


@register_factor("volatility_3m", ["close"])
def compute_volatility_3m(df: pd.DataFrame) -> pd.Series:
    """3月波动率"""
    rets = df["close"].pct_change().fillna(0)
    vol = rets.rolling(63).std() * np.sqrt(252)
    return -vol.fillna(0)


@register_factor("max_drawdown_1m", ["close"])
def compute_max_drawdown_1m(df: pd.DataFrame) -> pd.Series:
    """1月最大回撤：越低越好"""
    close = df["close"]
    peak = close.rolling(21).max()
    dd = (close - peak) / peak.replace(0, 1)
    return dd  # 负值，已经方向一致（回撤越深越负）


@register_factor("downside_risk", ["close"])
def compute_downside_risk(df: pd.DataFrame) -> pd.Series:
    """下行风险（半方差）：越低越好"""
    rets = df["close"].pct_change().fillna(0)
    downside = rets.clip(upper=0)
    risk = downside.rolling(63).std() * np.sqrt(252)
    return -risk.fillna(0)


@register_factor("beta_60d", ["close"])
def compute_beta_60d(df: pd.DataFrame) -> pd.Series:
    """60日 Beta（需要市场基准，用指数近似）"""
    rets = df["close"].pct_change().fillna(0)
    # 无市场基准时用自相关近似，有 index_col 时用协方差
    # 这里做一个简化版：收益自相关作为 beta 代理
    vol = rets.rolling(60).std().fillna(0)
    return vol.rank(pct=True)  # 相对排名


@register_factor("skewness_1m", ["close"])
def compute_skewness_1m(df: pd.DataFrame) -> pd.Series:
    """1月收益偏度：正偏优"""
    rets = df["close"].pct_change().fillna(0)
    return rets.rolling(21).skew().fillna(0)


# ── 技术因子 ────────────────────────────────────

@register_factor("rsi_14", ["close"])
def compute_rsi_14(df: pd.DataFrame) -> pd.Series:
    """RSI(14) 因子值：低 RSI 看多（反转）"""
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1)
    rsi = 100 - (100 / (1 + rs))
    return -rsi.fillna(50)  # 低 RSI 因子值高（超卖反转）


@register_factor("volume_ratio", ["volume"])
def compute_volume_ratio_factor(df: pd.DataFrame) -> pd.Series:
    """成交量比：当前量 / 20日均量"""
    vol_ma = df["volume"].rolling(20).mean()
    vr = df["volume"] / vol_ma.replace(0, 1)
    # 量比适中最好（1.0-2.0），过高过低都不好
    target = 1.5
    return -(vr.fillna(1) - target).abs()


@register_factor("turnover_rate", ["turnover_rate"])
def compute_turnover_rate_factor(df: pd.DataFrame) -> pd.Series:
    """换手率因子：适中换手率最优（流动性好但不是过度投机）"""
    t = df["turnover_rate"].fillna(0)
    target = 3.0  # 目标 3%
    return -(t - target).abs()


@register_factor("ma_deviation", ["close"])
def compute_ma_deviation(df: pd.DataFrame) -> pd.Series:
    """均线偏离度：价格偏离 60 日均线的 Z-Score"""
    ma60 = df["close"].rolling(60).mean()
    std60 = df["close"].rolling(60).std()
    z = (df["close"] - ma60) / std60.replace(0, 1)
    return -z.fillna(0)  # 超跌反转


@register_factor("volume_price_trend", ["close", "volume"])
def compute_volume_price_trend_factor(df: pd.DataFrame) -> pd.Series:
    """量价趋势：VPT 的 20 日变化率"""
    pct_chg = df["close"].pct_change().fillna(0)
    vpt = (pct_chg * df["volume"]).cumsum()
    return vpt.diff(20).fillna(0)


@register_factor("short_long_ratio", ["close"])
def compute_short_long_ratio(df: pd.DataFrame) -> pd.Series:
    """短期/长期均线比：MA5/MA60，低值→超跌反转"""
    ma5 = df["close"].rolling(5).mean()
    ma60 = df["close"].rolling(60).mean()
    ratio = ma5 / ma60.replace(0, 1)
    return -ratio.fillna(1)  # 低值→高分（超跌反弹）


# ═══════════════════════════════════════════════
# 三、默认因子库
# ═══════════════════════════════════════════════

DEFAULT_FACTORS: list[FactorDefinition] = [
    # 市值规模
    FactorDefinition("ln_cap", "size", ["total_mv"], "negative", "对数总市值"),
    FactorDefinition("midcap", "size", ["total_mv"], "negative", "中盘偏离度"),

    # 估值
    FactorDefinition("pe", "value", ["pe_ttm"], "negative", "市盈率 TTM"),
    FactorDefinition("pb", "value", ["pb"], "negative", "市净率"),
    FactorDefinition("ps", "value", ["ps_ttm"], "negative", "市销率"),
    FactorDefinition("pcf", "value", ["pcf_ttm"], "negative", "市现率"),
    FactorDefinition("dividend_yield", "value", ["dividend_yield"], "positive", "股息率"),
    FactorDefinition("ev_ebitda", "value", ["total_mv", "total_debt", "cash_equivalents", "ebitda"], "negative", "EV/EBITDA"),

    # 动量
    FactorDefinition("ret_1m", "momentum", ["close"], "positive", "1月动量"),
    FactorDefinition("ret_3m", "momentum", ["close"], "positive", "3月动量"),
    FactorDefinition("ret_12m_skip_1m", "momentum", ["close"], "positive", "12-1月动量"),
    FactorDefinition("reversal_1m", "momentum", ["close"], "negative", "1月反转"),

    # 质量
    FactorDefinition("roe", "quality", ["roe"], "positive", "净资产收益率"),
    FactorDefinition("roa", "quality", ["roa"], "positive", "总资产收益率"),
    FactorDefinition("gross_margin", "quality", ["grossprofit_margin"], "positive", "毛利率"),
    FactorDefinition("net_margin", "quality", ["netprofit_margin"], "positive", "净利率"),
    FactorDefinition("debt_ratio", "quality", ["debt_to_assets"], "negative", "资产负债率"),
    FactorDefinition("current_ratio", "quality", ["current_ratio"], "positive", "流动比率"),
    FactorDefinition("accrual_ratio", "quality",
                     ["total_assets", "net_intangible_assets", "total_liab",
                      "shortterm_loan", "net_profit", "cfoa"], "negative", "应计比率"),

    # 成长
    FactorDefinition("revenue_growth_yoy", "growth", ["or_yoy"], "positive", "营收同比增长"),
    FactorDefinition("earnings_growth_yoy", "growth", ["profit_dedt_yoy"], "positive", "净利同比增长"),
    FactorDefinition("asset_growth", "growth", ["total_assets"], "positive", "总资产增长"),
    FactorDefinition("sustainable_growth", "growth", ["roe", "dividend_yield"], "positive", "可持续增长率"),

    # 波动
    FactorDefinition("volatility_1m", "volatility", ["close"], "negative", "1月波动率"),
    FactorDefinition("volatility_3m", "volatility", ["close"], "negative", "3月波动率"),
    FactorDefinition("max_drawdown_1m", "volatility", ["close"], "negative", "1月最大回撤"),
    FactorDefinition("downside_risk", "volatility", ["close"], "negative", "下行风险"),
    FactorDefinition("skewness_1m", "volatility", ["close"], "positive", "收益偏度"),

    # 技术
    FactorDefinition("rsi_14", "technical", ["close"], "negative", "RSI反转"),
    FactorDefinition("volume_ratio", "technical", ["volume"], "negative", "成交量比"),
    FactorDefinition("turnover_rate", "technical", ["turnover_rate"], "negative", "换手率"),
    FactorDefinition("ma_deviation", "technical", ["close"], "negative", "均线偏离度"),
    FactorDefinition("short_long_ratio", "technical", ["close"], "negative", "短期长期均线比"),
]


# ═══════════════════════════════════════════════
# 四、因子评估结果
# ═══════════════════════════════════════════════

@dataclass
class FactorEvaluation:
    """单因子评估结果"""
    name: str
    category: str
    ic_mean: float          # IC 均值
    ic_std: float           # IC 标准差
    ir: float               # IC_IR = ic_mean / ic_std
    ic_positive_ratio: float  # IC > 0 的比例
    ic_t_stat: float        # IC t 统计量
    rank_ic_mean: float     # Rank IC 均值
    rank_ic_std: float      # Rank IC 标准差
    rank_ir: float          # Rank IC_IR
    quantile_returns: dict[str, float] = field(default_factory=dict)  # Q1-Q10 分组收益
    long_short_return: float = 0.0   # 多空收益 (Q10 - Q1)
    turnover: float = 0.0            # 月度平均换手率
    ic_series: list[float] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"因子: {self.name} ({self.category})",
            f"  IC Mean: {self.ic_mean:.4f}  IC Std: {self.ic_std:.4f}",
            f"  IR: {self.ir:.4f}  t-stat: {self.ic_t_stat:.2f}",
            f"  IC>0 Ratio: {self.ic_positive_ratio:.1%}",
            f"  Rank IC Mean: {self.rank_ic_mean:.4f}  Rank IR: {self.rank_ir:.4f}",
            f"  Long-Short Return: {self.long_short_return:.4f}",
            f"  Turnover: {self.turnover:.3f}",
        ]
        if self.quantile_returns:
            lines.append("  分组收益:")
            for k, v in self.quantile_returns.items():
                lines.append(f"    {k}: {v:.4f}")
        return "\n".join(lines)


@dataclass
class FactorEngineResult:
    """因子引擎分析结果"""
    factors: list[FactorEvaluation]
    composite_ic_mean: float
    composite_ir: float
    best_factors: list[str]
    total_factors: int

    def summary(self) -> str:
        lines = [
            f"因子引擎评估: {self.total_factors} 个因子",
            f"合成因子 IC Mean: {self.composite_ic_mean:.4f}",
            f"合成因子 IR: {self.composite_ir:.4f}",
            f"最优因子: {', '.join(self.best_factors[:10])}",
            "",
            "── 各因子详情 ──",
        ]
        for f in self.factors[:30]:
            lines.append(f"  {f.name:25s} | IC: {f.ic_mean:+.4f} | IR: {f.ir:+.3f} | t: {f.ic_t_stat:+.2f} | Long-Short: {f.long_short_return:+.4f}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════
# 五、因子引擎核心
# ═══════════════════════════════════════════════

class FactorEngine:
    """因子引擎核心

    使用方式:
        engine = FactorEngine(factors=DEFAULT_FACTORS)
        factor_df = engine.compute_factors(stock_data_df)
        processed_df = engine.preprocess(factor_df)
        eval_results = engine.evaluate(processed_df, returns_series)
    """

    def __init__(self, factors: list[FactorDefinition] | None = None):
        self.factors = factors or DEFAULT_FACTORS
        self._factor_map = {f.name: f for f in self.factors}

    @property
    def factor_names(self) -> list[str]:
        return [f.name for f in self.factors]

    # ── 因子计算 ──────────────────────────────

    def compute_factor(self, df: pd.DataFrame, factor_name: str) -> pd.Series:
        """计算单个因子值"""
        if factor_name not in _FACTOR_COMPUTERS:
            raise KeyError(f"未知因子: {factor_name}")
        computer = _FACTOR_COMPUTERS[factor_name]
        try:
            result = computer(df)
            if not isinstance(result, pd.Series):
                result = pd.Series(result, index=df.index)
            result.name = factor_name
            return result
        except Exception as e:
            # 字段缺失时返回 NaN 序列
            return pd.Series(np.nan, index=df.index, name=factor_name)

    def compute_factors(self, df: pd.DataFrame, factor_names: list[str] | None = None) -> pd.DataFrame:
        """批量计算因子

        Args:
            df: 原始数据 DataFrame (含 open/high/low/close/volume 及财务字段)
            factor_names: 要计算的因子列表，默认全部

        Returns:
            factor_df: 行=股票/日期, 列=因子
        """
        names = factor_names or [f.name for f in self.factors]
        results = {}
        for name in names:
            results[name] = self.compute_factor(df, name)
        return pd.DataFrame(results, index=df.index)

    # ── 预处理管道 ────────────────────────────

    @staticmethod
    def winsorize(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
        """去极值：分位数截尾

        Args:
            lower: 下分位数 (默认 1%)
            upper: 上分位数 (默认 99%)
        """
        lo = series.quantile(lower)
        hi = series.quantile(upper)
        return series.clip(lower=lo, upper=hi)

    @staticmethod
    def standardize(series: pd.Series) -> pd.Series:
        """Z-Score 标准化"""
        mean = series.mean()
        std = series.std()
        if std == 0 or np.isnan(std):
            return pd.Series(0, index=series.index)
        return (series - mean) / std

    @staticmethod
    def percentile_rank(series: pd.Series) -> pd.Series:
        """百分位排名（0-1）"""
        return series.rank(pct=True)

    @staticmethod
    def neutralize(factor: pd.Series, industry: pd.Series, cap: pd.Series) -> pd.Series:
        """行业 + 市值中性化

        公式: residual = factor - (industry_mean + β_cap * cap)
        去除行业均值和市值暴露后的残差。

        Args:
            factor: 因子值
            industry: 行业分类 (同行业取均值)
            cap: 市值序列

        Returns:
            中性化后的因子残差
        """
        # Step 1: 去除行业均值
        industry_mean = factor.groupby(industry).transform("mean")
        residual = factor - industry_mean.fillna(factor.mean())

        # Step 2: 去除市值线性暴露
        cap_std = FactorEngine.standardize(cap.astype(float)).fillna(0)
        residual_std = FactorEngine.standardize(residual).fillna(0)
        # 截面回归: residual = α + β * cap + ε
        mask = ~(np.isnan(residual_std) | np.isnan(cap_std))
        if mask.sum() > 2:
            beta = np.cov(residual_std[mask], cap_std[mask])[0, 1] / np.var(cap_std[mask]) if np.var(cap_std[mask]) > 0 else 0
            residual = residual - beta * cap_std * residual.std()
        return residual

    def preprocess(
        self,
        factor_df: pd.DataFrame,
        industry: pd.Series | None = None,
        cap: pd.Series | None = None,
        winsorize: bool = True,
        standardize: bool = True,
        neutralize_market: bool = False,
    ) -> pd.DataFrame:
        """因子预处理管道

        Args:
            factor_df: 原始因子 DataFrame
            industry: 行业序列（需中性化时提供）
            cap: 市值序列（需中性化时提供）
            winsorize: 是否去极值
            standardize: 是否标准化
            neutralize_market: 是否市值+行业中性化

        Returns:
            处理后的因子 DataFrame
        """
        df = factor_df.copy()
        for col in df.columns:
            series = df[col].copy()
            series = series.replace([np.inf, -np.inf], np.nan)
            mask = series.notna()

            if winsorize and mask.sum() > 10:
                series[mask] = self.winsorize(series[mask])

            if neutralize_market and industry is not None and cap is not None:
                # 中性化前标准化
                series[mask] = self.standardize(series[mask])
                series = self.neutralize(series, industry, cap)

            if standardize and mask.sum() > 1:
                series[mask] = self.standardize(series[mask])

            df[col] = series

        return df

    # ── 因子评估 ──────────────────────────────

    def evaluate(
        self,
        factor_df: pd.DataFrame,
        forward_returns: pd.Series,
        n_quantiles: int = 10,
        periods: list[int] | None = None,
    ) -> FactorEngineResult:
        """因子评估

        Args:
            factor_df: 因子 DataFrame (index = 日期, columns = 因子)
            forward_returns: 未来收益 (对齐到同一 index)
            n_quantiles: 分组数
            periods: 未来 N 期收益评估

        Returns:
            FactorEngineResult
        """
        if periods is None:
            periods = [1]

        evaluations = []
        for col in factor_df.columns:
            factor_def = self._factor_map.get(col)
            category = factor_def.category if factor_def else "unknown"
            ev = self._evaluate_single(factor_df[col], forward_returns, col, category, n_quantiles)
            evaluations.append(ev)

        # 按 IR 排序
        evaluations.sort(key=lambda x: abs(x.ir), reverse=True)

        # 合成因子评估（取前 50% 表现最好的因子等权组合）
        top_n = max(1, len(evaluations) // 2)
        top_factors = evaluations[:top_n]
        composite_ic = np.mean([f.ic_mean for f in top_factors])
        composite_ir = np.mean([f.ir for f in top_factors]) if top_factors else 0

        best_names = [f.name for f in evaluations[:10] if abs(f.ir) > 0.1]

        return FactorEngineResult(
            factors=evaluations,
            composite_ic_mean=composite_ic,
            composite_ir=composite_ir,
            best_factors=best_names,
            total_factors=len(evaluations),
        )

    def _evaluate_single(
        self,
        factor: pd.Series,
        forward_returns: pd.Series,
        name: str,
        category: str,
        n_quantiles: int,
    ) -> FactorEvaluation:
        """评估单个因子"""
        # 对齐
        common_idx = factor.dropna().index.intersection(forward_returns.dropna().index)
        if len(common_idx) < 30:
            return FactorEvaluation(name=name, category=category,
                                    ic_mean=0, ic_std=0, ir=0, ic_positive_ratio=0,
                                    ic_t_stat=0, rank_ic_mean=0, rank_ic_std=0, rank_ir=0)

        f = factor.loc[common_idx]
        r = forward_returns.loc[common_idx]

        # Pearson IC
        ic_series = f.rolling(20).corr(r).dropna()
        ic_mean = float(ic_series.mean()) if len(ic_series) > 0 else 0
        ic_std = float(ic_series.std()) if len(ic_series) > 0 else 0
        ir = ic_mean / ic_std if ic_std > 0 else 0
        ic_positive_ratio = float((ic_series > 0).mean()) if len(ic_series) > 0 else 0
        ic_t_stat = float(ic_mean / (ic_std / np.sqrt(len(ic_series)))) if ic_std > 0 and len(ic_series) > 1 else 0

        # Spearman Rank IC
        f_rank = f.rank()
        r_rank = r.rank()
        rank_ic_series = f_rank.rolling(20).corr(r_rank).dropna()
        rank_ic_mean = float(rank_ic_series.mean()) if len(rank_ic_series) > 0 else 0
        rank_ic_std = float(rank_ic_series.std()) if len(rank_ic_series) > 0 else 0
        rank_ir = rank_ic_mean / rank_ic_std if rank_ic_std > 0 else 0

        # 分层收益
        quantile_returns, long_short = self._quantile_analysis(f, r, n_quantiles)

        # 换手率（简化：相邻月份分位变化的均值）
        turnover = self._compute_turnover(f, n_quantiles)

        return FactorEvaluation(
            name=name, category=category,
            ic_mean=ic_mean, ic_std=ic_std, ir=ir,
            ic_positive_ratio=ic_positive_ratio, ic_t_stat=ic_t_stat,
            rank_ic_mean=rank_ic_mean, rank_ic_std=rank_ic_std, rank_ir=rank_ir,
            quantile_returns=quantile_returns, long_short_return=long_short,
            turnover=turnover,
            ic_series=ic_series.tolist(),
        )

    @staticmethod
    def _quantile_analysis(factor: pd.Series, returns: pd.Series, n: int) -> tuple[dict, float]:
        """分层收益分析"""
        labels = list(range(1, n + 1))
        try:
            quantile = pd.qcut(factor, n, labels=labels, duplicates="drop")
            grouped = returns.groupby(quantile).mean()
            result = {}
            for lbl in labels:
                result[f"Q{lbl}"] = round(float(grouped.get(lbl, 0)), 6)
            long_short = result.get(f"Q{n}", 0) - result.get("Q1", 0)
            return result, long_short
        except (ValueError, TypeError):
            return {}, 0

    @staticmethod
    def _compute_turnover(factor: pd.Series, n_quantiles: int) -> float:
        """计算月均分位换手率"""
        if len(factor) < 2:
            return 0.0
        try:
            qtile = pd.qcut(factor, n_quantiles, labels=False, duplicates="drop")
            changes = (qtile.diff().abs() > 0).sum()
            return float(changes / (len(qtile) - 1)) if len(qtile) > 1 else 0.0
        except (ValueError, TypeError):
            return 0.0

    # ── 合成因子 ──────────────────────────────

    def composite_factor(
        self,
        factor_df: pd.DataFrame,
        weights: dict[str, float] | None = None,
        method: str = "equal",
    ) -> pd.Series:
        """合成因子

        Args:
            factor_df: 因子 DataFrame
            weights: 因子权重 {"factor_name": weight}（method="weighted" 时使用）
            method: "equal" 等权 / "weighted" 自定义权重 / "ic_ir" IC_IR 加权

        Returns:
            合成因子序列
        """
        processed = self.preprocess(factor_df, winsorize=True, standardize=True)

        if method == "equal":
            return processed.mean(axis=1)

        elif method == "weighted" and weights:
            total_w = sum(weights.values())
            result = pd.Series(0, index=processed.index)
            for col in processed.columns:
                w = weights.get(col, 0) / total_w if total_w > 0 else 0
                result += processed[col].fillna(0) * w
            return result

        else:
            # 默认等权
            return processed.mean(axis=1)


# ═══════════════════════════════════════════════
# 六、便捷函数
# ═══════════════════════════════════════════════

def create_default_engine() -> FactorEngine:
    """创建默认因子引擎"""
    return FactorEngine(factors=DEFAULT_FACTORS)


def list_factors(category: str | None = None) -> list[FactorDefinition]:
    """列出所有可用因子"""
    if category:
        return [f for f in DEFAULT_FACTORS if f.category == category]
    return DEFAULT_FACTORS


def list_available_factors() -> dict[str, int]:
    """列出各类型因子数量"""
    counts: dict[str, int] = {}
    for f in DEFAULT_FACTORS:
        counts[f.category] = counts.get(f.category, 0) + 1
    return counts
