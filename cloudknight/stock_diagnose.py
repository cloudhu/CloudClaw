"""
诊股模块 - 个股多维度深度诊断

对 A 股单只股票进行：
  1. 技术面诊断 — 趋势/均线/MACD/KDJ/RSI/布林带/量价/支撑压力位
  2. 基本面诊断 — PE/PB/ROE/增长率/市值/股息率
  3. 资金面诊断 — 主力资金流向/换手率
  4. 市场面诊断 — 所属板块热度/市场情绪背景
  5. 综合评分 — 0-100 加权评分 + 操作建议

所有网络请求有超时保护，VPN 环境友好。
"""

import logging
import time as _time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from .config import MA_PERIODS
from .indicators import (
    IndicatorResult,
    calc_atr,
    calc_breakout,
    calc_ma,
    comprehensive_analysis,
    trend_score,
)

logger = logging.getLogger(__name__)

# ─── 诊股超时配置 ─────────────────────────────────────────
_DIAGNOSE_KLINE_TIMEOUT = 10  # K线获取超时（秒）
_DIAGNOSE_FUNDAMENTAL_TIMEOUT = 8  # 财务数据获取超时（秒）
_DIAGNOSE_MONEYFLOW_TIMEOUT = 8  # 资金流向获取超时（秒）
_DIAGNOSE_TOTAL_TIMEOUT = 25  # 全流程总超时（秒）


# ─── 数据模型 ─────────────────────────────────────────────


@dataclass
class TechDiagnosis:
    """技术面诊断结果"""

    score: int  # 技术面评分 0-100
    rating: str  # 评级
    current_price: float  # 当前价格
    pct_change_5d: float | None  # 5日涨跌幅 %
    pct_change_20d: float | None  # 20日涨跌幅 %
    amplitude_5d: float | None  # 5日振幅 %
    ma_alignment: str  # 均线排列: 多头/空头/交叉/缠绕
    ma_positions: dict[str, float]  # 各均线当前值 {MA5: x, MA10: x, ...}
    macd: IndicatorResult
    kdj: IndicatorResult
    rsi: IndicatorResult
    bollinger: IndicatorResult
    volume_price: IndicatorResult
    atr: float | None  # 平均真实波幅
    support: float  # 支撑位
    resistance: float  # 阻力位
    stock_name: str = ""  # 股票名称


@dataclass
class FundamentalDiagnosis:
    """基本面诊断结果"""

    score: int = 50  # 基本面评分 0-100
    rating: str = "未评估"  # 评级
    pe: float | None = None  # 市盈率
    pb: float | None = None  # 市净率
    roe: float | None = None  # ROE %
    revenue_growth: float | None = None  # 营收增长率 %
    profit_growth: float | None = None  # 净利润增长率 %
    market_cap: float | None = None  # 总市值（亿）
    dividend_yield: float | None = None  # 股息率 %
    debt_ratio: float | None = None  # 资产负债率 %
    gross_margin: float | None = None  # 毛利率 %
    net_margin: float | None = None  # 净利率 %
    industry_pe: float | None = None  # 行业平均PE
    data_available: bool = False  # 是否获取到财务数据


@dataclass
class CapitalDiagnosis:
    """资金面诊断结果"""

    score: int = 50  # 资金面评分 0-100
    rating: str = "未评估"  # 评级
    main_force_direction: str = "unknown"  # 主力动向: inflow/outflow/neutral/unknown
    main_force_5d_net: float | None = None  # 近5日主力净流入（亿）
    turnover_rate: float | None = None  # 换手率 %
    volume_ratio: float | None = None  # 量比
    data_available: bool = False


@dataclass
class MarketContext:
    """市场面诊断结果"""

    score: int = 50  # 市场面评分 0-100
    rating: str = "未评估"  # 评级
    sector_name: str = "未知板块"  # 所属板块
    sector_rank: int | None = None  # 板块强度排名
    sector_pct: float | None = None  # 板块涨跌幅 %
    market_trend: str = "未知"  # 大盘趋势
    market_sentiment: str = "未知"  # 市场情绪
    limit_status: str = "正常"  # 涨跌停状态


@dataclass
class StrategyDiagnosis:
    """单策略诊断结果"""

    name: str  # 策略名称
    key: str  # 策略 key
    score: int  # 策略评分 0-100
    signal: str  # 信号: buy/hold/sell
    rating: str  # 评级文字
    match_count: int  # 符合条件的项数
    total_conditions: int  # 总条件数
    reasons: list[str] = field(default_factory=list)  # 匹配原因
    warnings: list[str] = field(default_factory=list)  # 风险警示


@dataclass
class StockDiagnosis:
    """个股综合诊断报告"""

    code: str  # 股票代码
    name: str  # 股票名称
    timestamp: str  # 诊断时间
    technical: TechDiagnosis
    fundamental: FundamentalDiagnosis
    capital: CapitalDiagnosis
    market: MarketContext
    composite_score: int  # 综合评分 0-100
    composite_rating: str  # 综合评级
    recommendation: str  # 操作建议
    risk_level: str  # 风险等级: 低/中/高/极高
    summary: str  # 一句话总结
    detail_scores: dict[str, int] = field(default_factory=dict)  # 各维度得分明细
    strategy_diagnoses: list[StrategyDiagnosis] = field(default_factory=list)  # 各策略诊断
    data_quality: str = "完整"  # 数据质量: 完整/部分缺失/严重缺失
    pool_status: dict[str, dict] = field(default_factory=dict)  # 各策略池中的状态 {key: {in_pool, tier, score, entry_price}}


# ─── 诊股引擎 ─────────────────────────────────────────────


class StockDiagnoser:
    """个股诊断器

    使用现有 DataFetcher 获取数据，复用 indicators 模块做技术分析，
    每项数据获取有独立超时保护。
    """

    def __init__(self):
        from .data_manager import DataFetcher

        self._fetcher = DataFetcher()

    # ── 公开接口 ──────────────────────────────────────────

    def diagnose(self, code: str) -> StockDiagnosis:
        """对指定股票进行全面诊断

        Args:
            code: 6位股票代码，如 '000001' 或 '600519'

        Returns:
            StockDiagnosis 综合诊断报告（含自动入池结果）
        """
        code = str(code).zfill(6)
        deadline = _time.time() + _DIAGNOSE_TOTAL_TIMEOUT

        # 并行获取 K 线 + 财务数据
        df_kline = self._fetch_kline_safe(code)
        fund = self._diagnose_fundamental_safe(code, deadline)

        # 技术面依赖 K 线数据
        tech = self._diagnose_technical(code, df_kline)

        # 资金面
        cap = self._diagnose_capital_safe(code, deadline)

        # 市场面
        market = self._diagnose_market_context(code, tech.current_price, deadline)

        # 策略诊断（动态遍历策略池）
        strategy_diags = self._diagnose_strategies(df_kline, tech, fund, cap)

        # 自动入池：评分达标且未在池中的股票自动加入对应策略池
        auto_add_results = self._auto_add_to_pools(code, tech.stock_name, tech.current_price, strategy_diags, df_kline)

        # 数据质量评估
        data_quality = self._assess_data_quality(tech, fund, cap, market)

        # 综合评分（加入策略一致性）
        composite, rating, rec, risk, summary = self._compute_composite(tech, fund, cap, market, strategy_diags)

        name = tech.stock_name or self._resolve_name(code)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pool_status = self.check_pool_status(code)

        result = StockDiagnosis(
            code=code,
            name=name,
            timestamp=ts,
            technical=tech,
            fundamental=fund,
            capital=cap,
            market=market,
            strategy_diagnoses=strategy_diags,
            composite_score=composite,
            composite_rating=rating,
            recommendation=rec,
            risk_level=risk,
            summary=summary,
            detail_scores={
                "技术面": tech.score,
                "基本面": fund.score,
                "资金面": cap.score,
                "市场面": market.score,
            },
            data_quality=data_quality,
            pool_status=pool_status,
        )
        # 注入自动入池结果（非标准字段，仅供前端展示）
        result.__dict__["auto_add_results"] = auto_add_results
        return result

    def quick_diagnose(self, code: str) -> StockDiagnosis:
        """快速诊断 — 跳过低优先级数据，仅技术面+市场面"""
        code = str(code).zfill(6)
        df_kline = self._fetch_kline_safe(code)
        tech = self._diagnose_technical(code, df_kline)
        market = self._diagnose_market_context(code, tech.current_price, deadline=0)
        fund = FundamentalDiagnosis(score=50, rating="未评估", data_available=False)
        cap = CapitalDiagnosis(score=50, rating="未评估", main_force_direction="unknown", data_available=False)
        strategy_diags = self._diagnose_strategies(df_kline, tech, fund, cap)

        # 自动入池
        auto_add_results = self._auto_add_to_pools(code, tech.stock_name, tech.current_price, strategy_diags, df_kline)

        composite, rating, rec, risk, summary = self._compute_composite(tech, fund, cap, market, strategy_diags)
        name = tech.stock_name or self._resolve_name(code)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data_quality = "部分缺失"  # 快速诊断必然缺失基本面和资金面
        pool_status = self.check_pool_status(code)
        result = StockDiagnosis(
            code=code,
            name=name,
            timestamp=ts,
            technical=tech,
            fundamental=fund,
            capital=cap,
            market=market,
            strategy_diagnoses=strategy_diags,
            composite_score=composite,
            composite_rating=rating,
            recommendation=rec,
            risk_level=risk,
            summary=summary,
            detail_scores={"技术面": tech.score, "基本面": fund.score, "资金面": cap.score, "市场面": market.score},
            data_quality=data_quality,
            pool_status=pool_status,
        )
        result.__dict__["auto_add_results"] = auto_add_results
        return result

    # ── 安全数据获取（带超时） ─────────────────────────────

    def _fetch_kline_safe(self, code: str) -> pd.DataFrame:
        """获取日K线，超时返回空 DataFrame"""
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(self._fetcher.fetch_daily_kline, code, start_date="20230101")
            try:
                df = fut.result(timeout=_DIAGNOSE_KLINE_TIMEOUT)
                return df if df is not None and not df.empty else pd.DataFrame()
            except FutureTimeoutError:
                logger.warning(f"诊股: {code} K线获取超时")
                return pd.DataFrame()
            except Exception as e:
                logger.warning(f"诊股: {code} K线获取异常: {e}")
                return pd.DataFrame()

    def _diagnose_fundamental_safe(self, code: str, deadline: float) -> FundamentalDiagnosis:
        """基本面诊断（带超时）"""
        if _time.time() > deadline:
            return FundamentalDiagnosis(score=50, rating="超时跳过", data_available=False)
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(self._diagnose_fundamental, code)
            try:
                remaining = max(1, deadline - _time.time())
                return fut.result(timeout=min(_DIAGNOSE_FUNDAMENTAL_TIMEOUT, remaining))
            except FutureTimeoutError:
                logger.warning(f"诊股: {code} 基本面获取超时")
                return FundamentalDiagnosis(score=50, rating="超时跳过", data_available=False)
            except Exception as e:
                logger.warning(f"诊股: {code} 基本面异常: {e}")
                return FundamentalDiagnosis(score=50, rating="获取失败", data_available=False)

    def _diagnose_capital_safe(self, code: str, deadline: float) -> CapitalDiagnosis:
        """资金面诊断（带超时）"""
        if _time.time() > deadline:
            return CapitalDiagnosis(score=50, rating="超时跳过", main_force_direction="unknown", data_available=False)
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(self._diagnose_capital, code)
            try:
                remaining = max(1, deadline - _time.time())
                return fut.result(timeout=min(_DIAGNOSE_MONEYFLOW_TIMEOUT, remaining))
            except FutureTimeoutError:
                return CapitalDiagnosis(
                    score=50, rating="超时跳过", main_force_direction="unknown", data_available=False
                )
            except Exception as e:
                logger.warning(f"诊股: {code} 资金面异常: {e}")
                return CapitalDiagnosis(
                    score=50, rating="获取失败", main_force_direction="unknown", data_available=False
                )

    # ── 技术面诊断 ────────────────────────────────────────

    def _diagnose_technical(self, code: str, df: pd.DataFrame) -> TechDiagnosis:
        """技术面全面诊断"""
        if df is None or df.empty or len(df) < 30:
            return self._empty_tech()

        # 标准化列名
        col_map = {
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        # 确保必要的列
        required = ["open", "close", "high", "low", "volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            logger.warning(f"诊股: {code} K线数据缺少列: {missing}")
            return self._empty_tech()

        latest = df.iloc[-1]
        current_price = float(latest["close"])

        # 提取股票名称
        stock_name = ""
        if "name" in df.columns or "名称" in df.columns:
            name_col = "name" if "name" in df.columns else "名称"
            stock_name = str(df.iloc[-1][name_col])

        # 5/20日涨跌幅
        pct_5d = None
        if len(df) >= 6:
            pct_5d = round((current_price / float(df.iloc[-6]["close"]) - 1) * 100, 2)
        pct_20d = None
        if len(df) >= 21:
            pct_20d = round((current_price / float(df.iloc[-21]["close"]) - 1) * 100, 2)

        # 5日振幅
        amp_5d = None
        if len(df) >= 5:
            recent_5 = df.tail(5)
            amp_5d = round((float(recent_5["high"].max()) / float(recent_5["low"].min()) - 1) * 100, 2)

        # 均线计算与排列判断（使用 comprehensive_analysis 中的 MA 数据）
        ma_values = {}
        ma_alignment = "未知"
        df_ma = calc_ma(df, MA_PERIODS)
        for p in MA_PERIODS:
            col = f"MA{p}"
            if col in df_ma.columns:
                val = df_ma[col].iloc[-1]
                if pd.notna(val):
                    ma_values[col] = round(float(val), 2)

        # 判断均线排列
        ideal_multi = [f"MA{p}" for p in [5, 10, 20, 60, 120, 250] if f"MA{p}" in ma_values]
        is_bullish = all(
            ma_values.get(ideal_multi[i], 0) >= ma_values.get(ideal_multi[i + 1], 0)
            for i in range(len(ideal_multi) - 1)
        )
        is_bearish = all(
            ma_values.get(ideal_multi[i], 0) <= ma_values.get(ideal_multi[i + 1], 0)
            for i in range(len(ideal_multi) - 1)
        )
        if is_bullish and len(ideal_multi) >= 4:
            ma_alignment = "多头排列"
        elif is_bearish and len(ideal_multi) >= 4:
            ma_alignment = "空头排列"
        elif len(ma_values) >= 2:
            first = next(iter(ma_values.keys()))
            last = list(ma_values.keys())[-1]
            if ma_values[first] > ma_values[last]:
                ma_alignment = "偏多头"
            else:
                ma_alignment = "偏空头"

        # ── 使用 comprehensive_analysis() 一次性获取全部技术指标（教材 §16） ──
        ca = comprehensive_analysis(df)
        macd = ca.get("macd", IndicatorResult("neutral", "hold", 50, {}))
        kdj = ca.get("kdj", IndicatorResult("neutral", "hold", 50, {}))
        rsi = ca.get("rsi", IndicatorResult("neutral", "hold", 50, {}))
        boll = ca.get("bollinger", IndicatorResult("neutral", "hold", 50, {}))
        vol_pr = ca.get("volume_price", IndicatorResult("neutral", "hold", 50, {}))

        # ATR
        atr_val = None
        try:
            atr_series = calc_atr(df)
            atr_val = round(float(atr_series.iloc[-1]), 2)
        except Exception:
            pass

        # 支撑位/阻力位（利用 comprehensive_analysis 中已计算的指标）
        support, resistance = self._calc_sr(df, current_price, ca)

        # 技术面评分
        ts = trend_score(df)
        tech_score = int(ts.get("score", 50))
        tech_rating = ts.get("rating", "震荡")

        return TechDiagnosis(
            score=tech_score,
            rating=tech_rating,
            current_price=round(current_price, 2),
            pct_change_5d=pct_5d,
            pct_change_20d=pct_20d,
            amplitude_5d=amp_5d,
            ma_alignment=ma_alignment,
            ma_positions=ma_values,
            macd=macd,
            kdj=kdj,
            rsi=rsi,
            bollinger=boll,
            volume_price=vol_pr,
            atr=atr_val,
            support=support,
            resistance=resistance,
            stock_name=stock_name,
        )

    def _calc_sr(self, df: pd.DataFrame, price: float, ca: dict | None = None) -> tuple[float, float]:
        """计算支撑位和阻力位（利用 comprehensive_analysis 结果）"""
        support, resistance = price * 0.95, price * 1.05  # 默认 ±5%
        try:
            if len(df) < 60:
                return round(support, 2), round(resistance, 2)

            # 布林带上下轨
            if ca and "bollinger" in ca:
                boll_det = ca["bollinger"].details
                bb_low = float(boll_det.get("lower", 0))
                bb_high = float(boll_det.get("upper", 0))
                if bb_low > 0 and bb_low < price:
                    support = max(support, bb_low)
                if bb_high > price:
                    resistance = min(resistance, bb_high)
            else:
                # 降级：手动计算
                from .indicators import calc_bollinger
                df_bb = calc_bollinger(df)
                if "BOLL_DN" in df_bb.columns and "BOLL_UP" in df_bb.columns:
                    bb_low = float(df_bb["BOLL_DN"].iloc[-1])
                    bb_high = float(df_bb["BOLL_UP"].iloc[-1])
                    if pd.notna(bb_low) and pd.notna(bb_high):
                        support = min(support, bb_low)
                        resistance = max(resistance, bb_high)

            # Donchian 突破
            hh, ll = calc_breakout(df, 20)
            if not hh.empty and not ll.empty:
                hh_val, ll_val = float(hh.iloc[-1]), float(ll.iloc[-1])
                if pd.notna(hh_val) and pd.notna(ll_val):
                    resistance = max(resistance, hh_val)
                    support = min(support, ll_val)

            # MA20/MA60 支撑阻力
            df_ma = calc_ma(df, [20, 60])
            for p in [20, 60]:
                col = f"MA{p}"
                if col in df_ma.columns:
                    ma_val = float(df_ma[col].iloc[-1])
                    if pd.notna(ma_val):
                        if ma_val < price:
                            support = max(support, ma_val)
                        else:
                            resistance = min(resistance, ma_val)
        except Exception:
            pass
        return round(support, 2), round(resistance, 2)

    def _empty_tech(self) -> TechDiagnosis:
        return TechDiagnosis(
            score=50,
            rating="无数据",
            current_price=0,
            pct_change_5d=None,
            pct_change_20d=None,
            amplitude_5d=None,
            ma_alignment="无数据",
            ma_positions={},
            macd=IndicatorResult("neutral", "hold", 50, {}),
            kdj=IndicatorResult("neutral", "hold", 50, {}),
            rsi=IndicatorResult("neutral", "hold", 50, {}),
            bollinger=IndicatorResult("neutral", "hold", 50, {}),
            volume_price=IndicatorResult("neutral", "hold", 50, {}),
            atr=None,
            support=0,
            resistance=0,
        )

    # ── 基本面诊断 ────────────────────────────────────────

    def _diagnose_fundamental(self, code: str) -> FundamentalDiagnosis:
        """基本面分析（带衍生估算）"""
        try:
            df = self._fetcher.fetch_financial_data(code)
        except Exception:
            # 进阶级降级：尝试直接从 K 线数据获取时，标记为估算模式
            return self._derived_fundamental(code, "数据源异常")

        if df is None or df.empty:
            return self._derived_fundamental(code, "暂无财务报告")

        return self._parse_fundamental_data(code, df)

    def _parse_fundamental_data(self, code: str, df: pd.DataFrame) -> FundamentalDiagnosis:
        """解析已获取的财务数据"""
        fund = FundamentalDiagnosis(score=50, rating="待分析", data_available=True)
        try:
            latest = df.iloc[-1] if len(df) > 0 else None
            if latest is None:
                return fund

            fund.pe = self._safe_float(latest.get("市盈率") or latest.get("PE"))
            fund.pb = self._safe_float(latest.get("市净率") or latest.get("PB"))
            fund.roe = self._safe_float(latest.get("净资产收益率") or latest.get("ROE"))
            fund.revenue_growth = self._safe_float(latest.get("营业收入增长率") or latest.get("营收增长率"))
            fund.profit_growth = self._safe_float(latest.get("净利润增长率") or latest.get("利润增长率"))
            fund.market_cap = self._safe_float(latest.get("总市值") or latest.get("市值"))
            fund.dividend_yield = self._safe_float(latest.get("股息率"))
            fund.debt_ratio = self._safe_float(latest.get("资产负债率") or latest.get("负债率"))
            fund.gross_margin = self._safe_float(latest.get("毛利率"))
            fund.net_margin = self._safe_float(latest.get("净利率"))

            fund.score = self._score_fundamental(fund)
            fund.rating = self._score_to_rating(fund.score)

        except Exception as e:
            logger.warning(f"诊股: {code} 基本面计算异常: {e}")
            fund.score = 50
            fund.rating = "计算异常"

        return fund

    def _derived_fundamental(self, code: str, reason: str) -> FundamentalDiagnosis:
        """无财务数据时，从K线数据推导替代估值指标"""
        try:
            df = self._fetcher.fetch_daily_kline(code, start_date="20230101")
            if df is None or df.empty or len(df) < 60:
                return FundamentalDiagnosis(score=50, rating=reason, data_available=False)

            # 标准化列名
            col_map = {
                "日期": "date",
                "开盘": "open",
                "收盘": "close",
                "最高": "high",
                "最低": "low",
                "成交量": "volume",
            }
            df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
            if "close" not in df.columns:
                return FundamentalDiagnosis(score=50, rating=reason, data_available=False)

            closes = df["close"].values.astype(float)
            cur = closes[-1]
            fund = FundamentalDiagnosis(score=50, rating="技术面估算", data_available=False)

            # 推导250日价格分位（替代PE估值）
            if len(closes) >= 250:
                lookback = closes[-250:]
                percentile = float((lookback < cur).sum() / 250 * 100)
                fund.pe = None  # 无真实PE
                # 基于分位估算基本面得分
                if percentile < 15:
                    fund.score = 75
                    fund.rating = "技术面低估"
                elif percentile < 35:
                    fund.score = 65
                    fund.rating = "技术面偏低估"
                elif percentile < 65:
                    fund.score = 50
                    fund.rating = "技术面中性"
                elif percentile < 85:
                    fund.score = 35
                    fund.rating = "技术面偏高估"
                else:
                    fund.score = 20
                    fund.rating = "技术面高估"
            else:
                fund.score = 45
                fund.rating = "数据不足"

            # 推导股息率替代：价格稳定性
            if len(closes) >= 60:
                returns = np.diff(closes[-60:]) / closes[-61:-1] * 100
                volatility = float(np.std(returns))
                if volatility < 1.5:
                    fund.score = min(100, fund.score + 8)
                elif volatility > 4:
                    fund.score = max(0, fund.score - 5)
                fund.gross_margin = None  # 无真实毛利率

            # 基于成交量推导"活跃度"（替代利润增长率）
            if "volume" in df.columns:
                vols = df["volume"].values.astype(float)
                if len(vols) >= 20:
                    vol_trend = (
                        float(np.mean(vols[-5:]) / np.mean(vols[-20:-5]) - 1) if np.mean(vols[-20:-5]) > 0 else 0
                    )
                    if vol_trend > 0.3:
                        fund.score = min(100, fund.score + 5)
                    elif vol_trend < -0.3:
                        fund.score = max(0, fund.score - 5)

            fund.rating = self._score_to_rating(fund.score)
            fund.data_available = False  # 仍标记为非真实数据
            return fund

        except Exception:
            return FundamentalDiagnosis(score=50, rating=reason, data_available=False)

    def _score_fundamental(self, f: FundamentalDiagnosis) -> int:
        """基本面多维度加权评分"""
        score = 50
        items = 0

        # PE 评分：15-30 正常偏高，<15 低估，>50 高估
        if f.pe is not None and f.pe > 0:
            items += 1
            if f.pe < 10:
                score += 15
            elif f.pe < 15:
                score += 10
            elif f.pe < 25:
                score += 5
            elif f.pe < 40:
                score -= 5
            elif f.pe < 60:
                score -= 10
            else:
                score -= 15

        # PB 评分
        if f.pb is not None and f.pb > 0:
            items += 1
            if f.pb < 1:
                score += 10
            elif f.pb < 2:
                score += 5
            elif f.pb < 4:
                score += 0
            elif f.pb < 6:
                score -= 5
            else:
                score -= 10

        # ROE 评分
        if f.roe is not None:
            items += 1
            if f.roe > 20:
                score += 15
            elif f.roe > 15:
                score += 10
            elif f.roe > 10:
                score += 5
            elif f.roe > 5:
                score += 0
            elif f.roe > 0:
                score -= 5
            else:
                score -= 15

        # 利润增长率
        if f.profit_growth is not None:
            items += 1
            if f.profit_growth > 30:
                score += 10
            elif f.profit_growth > 15:
                score += 5
            elif f.profit_growth > 0:
                score += 0
            elif f.profit_growth > -10:
                score -= 5
            else:
                score -= 10

        # 负债率评分：<40% 健康
        if f.debt_ratio is not None:
            items += 1
            if f.debt_ratio < 30:
                score += 5
            elif f.debt_ratio < 50:
                score += 0
            elif f.debt_ratio < 70:
                score -= 5
            else:
                score -= 10

        return max(0, min(100, score)) if items > 0 else 50

    # ── 资金面诊断 ────────────────────────────────────────

    def _diagnose_capital(self, code: str) -> CapitalDiagnosis:
        """资金面分析"""
        cap = CapitalDiagnosis(score=50, rating="待分析", main_force_direction="unknown", data_available=False)
        try:
            df = self._fetcher.fetch_money_flow(code)
        except Exception:
            return cap

        if df is None or df.empty:
            return cap

        cap.data_available = True
        try:
            # 分析近5日主力资金
            if len(df) >= 5:
                recent = df.tail(5)
                net_cols = ["主力净流入", "main_net", "主力净流入-净额"]
                for c in net_cols:
                    if c in recent.columns:
                        cap.main_force_5d_net = round(float(recent[c].sum()) / 1e8, 2)
                        break

            # 换手率
            turnover_cols = ["换手率", "turnover_rate"]
            for c in turnover_cols:
                if c in df.columns:
                    cap.turnover_rate = round(float(df[c].iloc[-1]), 2)
                    break

            # 量比
            vol_cols = ["量比", "volume_ratio"]
            for c in vol_cols:
                if c in df.columns:
                    cap.volume_ratio = round(float(df[c].iloc[-1]), 2)
                    break

            # 主力方向判断
            if cap.main_force_5d_net is not None:
                if cap.main_force_5d_net > 0.5:
                    cap.main_force_direction = "inflow"
                elif cap.main_force_5d_net < -0.5:
                    cap.main_force_direction = "outflow"
                else:
                    cap.main_force_direction = "neutral"

            # 评分
            cap.score = self._score_capital(cap)
            cap.rating = self._score_to_rating(cap.score)

        except Exception as e:
            logger.warning(f"诊股: {code} 资金面计算异常: {e}")

        return cap

    def _score_capital(self, c: CapitalDiagnosis) -> int:
        """资金面评分"""
        score = 50
        # 主力资金
        if c.main_force_direction == "inflow":
            score += 15
            if c.main_force_5d_net and c.main_force_5d_net > 2:
                score += 10
        elif c.main_force_direction == "outflow":
            score -= 15
            if c.main_force_5d_net and c.main_force_5d_net < -2:
                score -= 10
        # 换手率：适度换手 2%-8% 最佳
        if c.turnover_rate is not None:
            if 2 <= c.turnover_rate <= 8:
                score += 5
            elif c.turnover_rate > 15:
                score -= 5
        return max(0, min(100, score))

    # ── 市场面诊断 ────────────────────────────────────────

    def _diagnose_market_context(self, code: str, price: float, deadline: float = 0) -> MarketContext:
        """市场环境分析"""
        from .market_analyzer import MarketAnalyzer

        analyzer = MarketAnalyzer()
        sector_name = self._resolve_sector(code)
        sector_rank = None
        sector_pct = None
        market_trend = "未知"
        market_sentiment = "未知"
        limit_status = "正常"

        try:
            trend_result = analyzer.diagnose_market_trend()
            market_trend = trend_result.get("trend_desc", "未知") if isinstance(trend_result, dict) else "未知"
        except Exception:
            pass

        try:
            sentiment = analyzer.get_market_sentiment()
            market_sentiment = sentiment.get("sentiment", "未知") if isinstance(sentiment, dict) else "未知"
        except Exception:
            pass

        # 市场面评分
        trend_score_map = {
            "强势牛市": 90,
            "偏多震荡": 65,
            "震荡市": 50,
            "偏空震荡": 35,
            "弱势熊市": 15,
            "未知": 50,
        }
        sentiment_score_map = {
            "极度贪婪": 85,
            "贪婪": 70,
            "中性": 50,
            "恐惧": 30,
            "极度恐惧": 15,
            "未知": 50,
        }

        m_score = int(trend_score_map.get(market_trend, 50) * 0.6 + sentiment_score_map.get(market_sentiment, 50) * 0.4)

        return MarketContext(
            score=m_score,
            rating=self._score_to_rating(m_score),
            sector_name=sector_name,
            sector_rank=sector_rank,
            sector_pct=sector_pct,
            market_trend=market_trend,
            market_sentiment=market_sentiment,
            limit_status=limit_status,
        )

    # ── 综合评分 ──────────────────────────────────────────

    def _compute_composite(
        self,
        tech: TechDiagnosis,
        fund: FundamentalDiagnosis,
        cap: CapitalDiagnosis,
        market: MarketContext,
        strategy_diags: list[StrategyDiagnosis] | None = None,
    ) -> tuple[int, str, str, str, str]:
        """计算综合评分、评级、建议、风险"""
        # 基础四维权重
        weights = {"tech": 0.35, "fund": 0.20, "capital": 0.15, "market": 0.10, "strategy": 0.20}

        # 如果基本面无数据，重新分配权重
        if not fund.data_available:
            weights = {"tech": 0.40, "fund": 0.08, "capital": 0.17, "market": 0.12, "strategy": 0.23}
        if not cap.data_available:
            if not fund.data_available:
                weights = {"tech": 0.42, "fund": 0.08, "capital": 0.07, "market": 0.15, "strategy": 0.28}
            else:
                weights = {"tech": 0.38, "fund": 0.22, "capital": 0.07, "market": 0.13, "strategy": 0.20}

        # 策略一致性评分
        strategy_score = self._strategy_consensus_score(strategy_diags or [])

        composite = int(
            tech.score * weights["tech"]
            + fund.score * weights["fund"]
            + cap.score * weights["capital"]
            + market.score * weights["market"]
            + strategy_score * weights["strategy"]
        )

        # 评级
        if composite >= 80:
            rating = "强烈推荐"
        elif composite >= 65:
            rating = "推荐关注"
        elif composite >= 50:
            rating = "中性观望"
        elif composite >= 35:
            rating = "谨慎回避"
        else:
            rating = "强烈回避"

        # 操作建议（加入策略信息）
        rec = self._generate_recommendation(tech, fund, cap, composite, strategy_diags)

        # 风险等级
        risk = "低" if composite >= 75 else ("中" if composite >= 50 else ("高" if composite >= 30 else "极高"))

        # 一句话总结
        parts = [f"技术面{tech.rating}"]
        if fund.data_available:
            parts.append(f"基本面{fund.rating}")
        else:
            parts.append("基本面无数据")
        if cap.data_available:
            parts.append(f"资金面{cap.rating}")
        if strategy_diags:
            buy_count = sum(1 for s in strategy_diags if s.signal == "buy")
            total_count = len(strategy_diags)
            if buy_count >= total_count * 0.7:
                parts.append(f"策略一致看多({buy_count}/{total_count})")
            elif buy_count >= total_count * 0.4:
                parts.append(f"策略分歧({buy_count}/{total_count}看多)")
            else:
                parts.append(f"策略偏空({total_count - buy_count}/{total_count}看空)")
        parts.append(f"综合{rating}")
        summary = f"{tech.stock_name or '该股'} — " + "，".join(parts)

        return composite, rating, rec, risk, summary

    def _strategy_consensus_score(self, diags: list[StrategyDiagnosis]) -> int:
        """策略一致性评分：取策略评分的加权平均"""
        if not diags:
            return 50
        # 收益率权重：有 buy 信号的策略给更高权重
        total_weight = 0
        weighted = 0
        for s in diags:
            w = 1.5 if s.signal == "buy" else (0.8 if s.signal == "sell" else 1.0)
            weighted += s.score * w
            total_weight += w
        return int(weighted / total_weight) if total_weight > 0 else 50

    def _generate_recommendation(
        self,
        tech: TechDiagnosis,
        fund: FundamentalDiagnosis,
        cap: CapitalDiagnosis,
        composite: int,
        strategy_diags: list[StrategyDiagnosis] | None = None,
    ) -> str:
        """生成操作建议"""
        recs = []
        # 技术面信号
        if tech.macd.signal == "buy":
            recs.append("MACD金叉买入信号")
        elif tech.macd.signal == "sell":
            recs.append("MACD死叉卖出信号")
        if tech.kdj.signal == "buy":
            recs.append("KDJ低位金叉")
        elif tech.kdj.signal == "sell" and tech.kdj.trend == "overbought":
            recs.append("KDJ高位超买")
        if tech.rsi.trend == "oversold":
            recs.append("RSI超卖，关注反弹")
        elif tech.rsi.trend == "overbought":
            recs.append("RSI超买，注意回调风险")

        # 策略信号
        if strategy_diags:
            buy_strats = [s.name for s in strategy_diags if s.signal == "buy"]
            sell_strats = [s.name for s in strategy_diags if s.signal == "sell"]
            if len(buy_strats) >= 3:
                recs.append(f"多策略一致看多({','.join(buy_strats)})")
            elif len(buy_strats) >= 2:
                recs.append(f"部分策略看多({','.join(buy_strats)})")
            elif len(sell_strats) >= 3:
                recs.append(f"多策略一致看空({','.join(sell_strats)})")

        # 基本面
        if fund.data_available:
            if fund.score >= 75:
                recs.append("基本面优秀，适合长线布局")
            elif fund.score <= 30:
                recs.append("基本面较差，长线需谨慎")
        else:
            recs.append("基本面无数据(需财报季更新)")

        # 资金面
        if cap.data_available:
            if cap.main_force_direction == "inflow":
                recs.append("主力资金流入，短期看多")
            elif cap.main_force_direction == "outflow":
                recs.append("主力资金流出，短期偏空")

        if not recs:
            if composite >= 60:
                recs.append("各策略指标中性偏多，可适量关注")
            elif composite >= 40:
                recs.append("各项指标中性偏空，建议观望等待信号")
            else:
                recs.append("综合评分偏低，建议规避")

        return "；".join(recs) if recs else "暂无明确建议"

    # ── 策略诊断（动态策略池） ──────────────────────────

    def _diagnose_strategies(
        self,
        df: pd.DataFrame,
        tech: TechDiagnosis,
        fund: FundamentalDiagnosis,
        cap: CapitalDiagnosis,
    ) -> list[StrategyDiagnosis]:
        """从策略池动态遍历所有支持诊股的策略进行评分"""
        from .strategies import get_diagnose_strategies

        if df is None or df.empty or len(df) < 40:
            return self._empty_strategy_results()

        # 标准化列名（兼容中英文列名）
        col_map = {
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        required = ["open", "close", "high", "low", "volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            return self._empty_strategy_results()

        closes = df["close"].values.astype(float)
        highs = df["high"].values.astype(float)
        lows = df["low"].values.astype(float)
        volumes = df["volume"].values.astype(float)
        opens = df["open"].values.astype(float)

        # 计算策略所需的基础指标
        indicators = self._calc_strategy_indicators(closes, highs, lows, volumes, opens)

        # 动态遍历所有注册的诊股策略
        results = []
        strategy_registry = get_diagnose_strategies()
        for key, meta in strategy_registry.items():
            strategy_cls = meta["class"]
            try:
                if hasattr(strategy_cls, 'diagnose'):
                    raw = strategy_cls.diagnose(closes, highs, lows, volumes, opens, indicators, fund, cap)
                    diag = StrategyDiagnosis(
                        name=raw["name"],
                        key=raw["key"],
                        score=raw["score"],
                        signal=raw["signal"],
                        rating=raw["rating"],
                        match_count=raw["match_count"],
                        total_conditions=raw["total_conditions"],
                        reasons=raw.get("reasons", []),
                        warnings=raw.get("warnings", []),
                    )
                    results.append(diag)
            except Exception as e:
                logger.warning(f"策略 {key} 诊股异常: {e}")
                results.append(StrategyDiagnosis(
                    name=meta.get("label", key),
                    key=key,
                    score=50,
                    signal="hold",
                    rating="评估异常",
                    match_count=0,
                    total_conditions=0,
                    warnings=[f"评估异常: {str(e)}"],
                ))

        return results

    def _calc_strategy_indicators(self, closes, highs, lows, volumes, opens):
        """计算各策略共用的技术指标（纯 numpy）"""
        ind = {}

        # 均线
        for p in [5, 10, 20, 60, 120, 200, 250]:
            if len(closes) >= p:
                ind[f"ma{p}"] = float(np.mean(closes[-p:]))
        # 唐奇安通道
        for p in [10, 20]:
            if len(highs) >= p and len(lows) >= p:
                ind[f"donchian_h{p}"] = float(np.max(highs[-p:]))
                ind[f"donchian_l{p}"] = float(np.min(lows[-p:]))
        # ATR
        if len(highs) >= 15:
            tr = np.zeros(len(closes))
            tr[0] = highs[0] - lows[0]
            for i in range(1, len(closes)):
                tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
            ind["atr14"] = float(np.mean(tr[-14:]))
        # RSI
        if len(closes) >= 15:
            deltas = np.diff(closes[-15:])
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)
            rs = avg_gain / avg_loss if avg_loss > 0 else 100
            ind["rsi"] = float(100 - 100 / (1 + rs))
        # KDJ
        if len(highs) >= 10:
            h9, l9 = np.max(highs[-9:]), np.min(lows[-9:])
            cur = closes[-1]
            rsv = (cur - l9) / (h9 - l9) * 100 if h9 > l9 else 50
            # 用前一根历史K/D做近似
            h9p, l9p = np.max(highs[-10:-1]), np.min(lows[-10:-1])
            prev = closes[-2]
            rsv_prev = (prev - l9p) / (h9p - l9p) * 100 if h9p > l9p else 50
            k_prev = 50 * 2 / 3 + rsv_prev * 1 / 3
            d_prev = 50 * 2 / 3 + k_prev * 1 / 3
            k = k_prev * 2 / 3 + rsv * 1 / 3
            d = d_prev * 2 / 3 + k * 1 / 3
            ind["k"] = float(k)
            ind["d"] = float(d)
            ind["j"] = float(3 * k - 2 * d)
            ind["k_prev"] = float(k_prev)
            ind["d_prev"] = float(d_prev)
        # 量比（当日/5日均量）
        if len(volumes) >= 6:
            ind["vol_ratio"] = float(volumes[-1] / np.mean(volumes[-6:-1])) if np.mean(volumes[-6:-1]) > 0 else 1.0
        # 涨跌幅序列
        if len(closes) >= 21:
            ind["pct_today"] = float((closes[-1] / closes[-2] - 1) * 100) if closes[-2] > 0 else 0
            pcts = []
            for i in range(1, min(21, len(closes))):
                pcts.append(float((closes[-i] / closes[-i - 1] - 1) * 100))
            ind["pcts"] = pcts
        # 250日价格分位
        if len(closes) >= 250:
            lookback = closes[-250:]
            ind["price_percentile"] = float((lookback < closes[-1]).sum() / 250 * 100)

        return ind

    # ── 策略评估方法已移至各策略类的 diagnose() 静态方法 ──
    # 参见 strategies/dragon_head.py, sparrow.py, turtle.py, value_invest.py 等

    def _empty_strategy_results(self) -> list[StrategyDiagnosis]:
        """K线数据不足时返回空策略结果（动态遍历策略池）"""
        from .strategies import get_diagnose_strategies

        results = []
        for key, meta in get_diagnose_strategies().items():
            results.append(
                StrategyDiagnosis(
                    name=meta.get("label", key),
                    key=key,
                    score=50,
                    signal="hold",
                    rating="数据不足",
                    match_count=0,
                    total_conditions=0,
                    warnings=["K线数据不足，无法评估"],
                )
            )
        return results

    # ── 策略池集成 ────────────────────────────────────

    def _get_pool_manager(self):
        """延迟加载 PoolManager 单例"""
        try:
            from .stock_pool import PoolManager

            return PoolManager()
        except Exception:
            return None

    def _compute_pool_components(self, code: str, strategy_key: str, df: pd.DataFrame,
                                  extra: dict | None = None) -> dict[str, float]:
        """使用策略专属评分器计算各维度得分，返回 components dict。

        Args:
            code: 股票代码
            strategy_key: 策略 key (dragon_head/sparrow/turtle/value_invest/grid/...)
            df: K线 DataFrame
            extra: 额外数据（如基本面数据）
        """
        try:
            from .stock_pool import (
                DragonHeadScorer, SparrowScorer, TurtleScorer,
                ValueInvestScorer, GridScorer, BollingerBandScorer,
                MACrossoverScorer,
            )

            scorer_map = {
                "dragon_head": DragonHeadScorer,
                "sparrow": SparrowScorer,
                "turtle": TurtleScorer,
                "value_invest": ValueInvestScorer,
                "grid": GridScorer,
                "bollinger": BollingerBandScorer,
                "ma_cross": MACrossoverScorer,
            }

            scorer_cls = scorer_map.get(strategy_key)
            if scorer_cls is None:
                return {"综合评分": 100.0}

            scorer = scorer_cls()
            item = scorer.score(code, "", df, extra)
            if item.components:
                return item.components
            return {"综合评分": item.score}
        except Exception as e:
            logger.warning(f"计算 {strategy_key} 评分维度失败: {e}")
            return {}

    def _auto_add_to_pools(
        self, code: str, name: str, price: float, strategy_diags: list[StrategyDiagnosis],
        df_kline: pd.DataFrame | None = None,
    ) -> dict[str, dict]:
        """诊股后自动将高评分股票加入对应策略池

        根据 DIAGNOSE_STRATEGIES 中定义的 auto_add_threshold，评分达标的策略
        自动将该股票加入其股票池（如果尚未在池中）。

        Returns:
            {strategy_key: {added: bool, already_in: bool, score: int, threshold: int, plan: dict|null, error: str|null}}
        """
        from .strategies import get_diagnose_strategies

        registry = get_diagnose_strategies()
        results = {}

        for diag in strategy_diags:
            meta = registry.get(diag.key, {})
            threshold = meta.get("auto_add_threshold", 70)

            result = {
                "added": False,
                "already_in": False,
                "pool_full": False,
                "score": diag.score,
                "threshold": threshold,
                "plan": None,
                "error": None,
                "eliminated": None,
            }

            # 评分未达自动入池阈值，跳过
            if diag.score < threshold:
                results[diag.key] = result
                continue

            # 检查是否已在池中
            dm = self._get_pool_manager()
            if dm is None:
                result["error"] = "PoolManager 不可用"
                results[diag.key] = result
                continue

            pool = dm.get_pool(diag.key)
            if pool is None:
                result["error"] = f"策略 {diag.key} 无对应池"
                results[diag.key] = result
                continue

            existing = pool.get(code)
            if existing and existing.status == "active":
                result["already_in"] = True
                results[diag.key] = result
                continue

            # 执行自动入池（含池满末位淘汰）
            try:
                from .stock_pool import MAX_POOL_SIZE, StockPoolItem

                # 确保名称不为空（优先缓存，其次 _resolve_name，最后回退）
                resolved_name = name
                if not resolved_name:
                    resolved_name = cls._resolve_name(code)
                    # 如果 _resolve_name 仍返回占位格式，尝试再查一次缓存
                    if resolved_name.startswith("股票"):
                        try:
                            from .stock_info_cache import get_stock_info_cache
                            cached = get_stock_info_cache().get_name(code)
                            if cached:
                                resolved_name = cached
                        except Exception:
                            pass

                # 池满时的末位淘汰：找到活跃股票中评分最低的
                if pool.size >= MAX_POOL_SIZE:
                    active_items = [(c, it) for c, it in pool.items.items() if it.status == "active"]
                    if active_items:
                        weakest_code, weakest_item = min(active_items, key=lambda x: x[1].score)
                        if float(diag.score) > weakest_item.score:
                            # 新票评分更高：淘汰最低分，纳入新票
                            weakest_item.status = "eliminated"
                            result["eliminated"] = {"code": weakest_code, "name": weakest_item.name, "score": weakest_item.score}
                            logger.info(
                                f"末位淘汰 [{diag.key}]: {weakest_code} {weakest_item.name} "
                                f"(评分{weakest_item.score:.0f}) → 让位 {code} (评分{diag.score})"
                            )
                        else:
                            # 新票评分不敌池内最低分，拒绝入池
                            result["pool_full"] = True
                            result["required_min_score"] = weakest_item.score
                            results[diag.key] = result
                            continue

                # 计算策略专属各维度评分
                comps = self._compute_pool_components(code, diag.key, df_kline) if df_kline is not None and not df_kline.empty else {}
                if not comps:
                    comps = {"诊断推荐": float(diag.score)}

                item = StockPoolItem(
                    code=code,
                    name=resolved_name,
                    score=float(diag.score),
                    components=comps,
                    screened_at=datetime.now().strftime("%Y%m%d"),
                    entry_price=round(price, 2),
                    max_price=round(price, 2),
                )
                pool.items[code] = item
                pool.save()

                # 生成交易计划
                plan_obj = pool.create_trade_plan(code, resolved_name, price, float(diag.score))

                result["added"] = True
                result["plan"] = {
                    "entry_price": plan_obj.entry_price,
                    "entry_type": plan_obj.entry_type,
                    "stop_loss": plan_obj.stop_loss,
                    "stop_loss_pct": plan_obj.stop_loss_pct,
                    "take_profit_1": plan_obj.take_profit_1,
                    "take_profit_1_pct": plan_obj.take_profit_1_pct,
                    "take_profit_2": plan_obj.take_profit_2,
                    "take_profit_2_pct": plan_obj.take_profit_2_pct,
                    "position_pct": plan_obj.position_pct,
                    "risk_reward_ratio": plan_obj.risk_reward_ratio,
                    "hold_days": plan_obj.hold_days,
                    "reasons": plan_obj.reasons,
                    "warnings": plan_obj.warnings,
                    "created_at": plan_obj.created_at,
                }
                logger.info(f"自动入池 [{diag.key}]: {code} {name} 评分{diag.score}≥{threshold}")

            except Exception as e:
                result["error"] = str(e)
                logger.warning(f"自动入池失败 [{diag.key}]: {code} {e}")

            results[diag.key] = result

        return results

    def check_pool_status(self, code: str) -> dict[str, dict]:
        """查询该股票在各策略池中的状态

        返回: {strategy_key: {in_pool: bool, tier: str, score: float, entry_price: float}}
        """
        dm = self._get_pool_manager()
        if dm is None:
            return {}
        status = {}
        for key, pool in dm.pools.items():
            item = pool.get(code)
            if item and item.status == "active":
                status[key] = {
                    "in_pool": True,
                    "tier": item.tier or "broad",
                    "score": round(item.score, 1),
                    "entry_price": round(item.entry_price, 2) if item.entry_price > 0 else 0,
                }
            else:
                status[key] = {
                    "in_pool": False,
                    "tier": "",
                    "score": 0,
                    "entry_price": 0,
                }
        return status

    def add_diagnosed_to_pool(self, code: str, name: str, strategy_key: str, score: int, price: float) -> dict | None:
        """将诊股中高评分的股票加入对应策略池，并生成交易计划

        Args:
            code: 股票代码
            name: 股票名称
            strategy_key: 策略 key (dragon_head/sparrow/turtle/value_invest)
            score: 诊股时该策略的评分
            price: 当前价格

        Returns:
            {success, item, plan} 或 None (策略不存在)
        """
        dm = self._get_pool_manager()
        if dm is None:
            return {"success": False, "error": "PoolManager 初始化失败"}

        pool = dm.get_pool(strategy_key)
        if pool is None:
            return {"success": False, "error": f"未知策略: {strategy_key}"}

        # 检查是否已在池中
        existing = pool.get(code)
        if existing and existing.status == "active":
            return {
                "success": False,
                "error": f"已在 {pool.strategy_label} 池中（{existing.tier or '待评估'}层，评分{existing.score:.0f}）",
            }

        # 加入池（含池满末位淘汰）
        try:
            from .stock_pool import MAX_POOL_SIZE, StockPoolItem

            # 池满时的末位淘汰
            eliminated = None
            if pool.size >= MAX_POOL_SIZE:
                active_items = [(c, it) for c, it in pool.items.items() if it.status == "active"]
                if active_items:
                    weakest_code, weakest_item = min(active_items, key=lambda x: x[1].score)
                    if float(score) > weakest_item.score:
                        weakest_item.status = "eliminated"
                        eliminated = {"code": weakest_code, "name": weakest_item.name, "score": weakest_item.score}
                        logger.info(
                            f"末位淘汰 [{strategy_key}]: {weakest_code} {weakest_item.name} "
                            f"(评分{weakest_item.score:.0f}) → 让位 {code} (评分{score})"
                        )
                    else:
                        return {
                            "success": False,
                            "pool_full": True,
                            "error": f"池已满({pool.size}/{MAX_POOL_SIZE})，评分{score}低于池内最低分{weakest_item.score:.0f}",
                        }

            # 获取 K 线数据用于计算策略专属各维度评分
            df_kline = self._fetch_kline_safe(code)
            comps = self._compute_pool_components(code, strategy_key, df_kline) if not df_kline.empty else {}
            if not comps:
                comps = {"诊断推荐": float(score)}

            item = StockPoolItem(
                code=code,
                name=name,
                score=float(score),
                components=comps,
                screened_at=datetime.now().strftime("%Y%m%d"),
                entry_price=round(price, 2),
                max_price=round(price, 2),
            )
            pool.items[code] = item
            pool.save()

            # 生成交易计划
            plan = pool.create_trade_plan(code, name, price, float(score))

            return {
                "success": True,
                "item": item.to_dict(),
                "plan": {
                    "entry_price": plan.entry_price,
                    "entry_type": plan.entry_type,
                    "stop_loss": plan.stop_loss,
                    "stop_loss_pct": plan.stop_loss_pct,
                    "take_profit_1": plan.take_profit_1,
                    "take_profit_1_pct": plan.take_profit_1_pct,
                    "take_profit_2": plan.take_profit_2,
                    "take_profit_2_pct": plan.take_profit_2_pct,
                    "position_pct": plan.position_pct,
                    "risk_reward_ratio": plan.risk_reward_ratio,
                    "hold_days": plan.hold_days,
                    "reasons": plan.reasons,
                    "warnings": plan.warnings,
                    "created_at": plan.created_at,
                },
            }
        except Exception as e:
            logger.error(f"加入策略池失败: {e}")
            return {"success": False, "error": str(e)}

    # ── 数据质量评估 ───────────────────────────────────

    def _assess_data_quality(
        self,
        tech: TechDiagnosis,
        fund: FundamentalDiagnosis,
        cap: CapitalDiagnosis,
        market: MarketContext,
    ) -> str:
        """评估整体数据质量"""
        missing = []
        if tech.score == 50 and tech.rating == "无数据":
            missing.append("技术面")
        if not fund.data_available:
            missing.append("基本面")
        if not cap.data_available:
            missing.append("资金面")
        if market.market_trend == "未知" and market.market_sentiment == "未知":
            missing.append("市场面")

        if not missing:
            return "完整"
        elif len(missing) <= 1:
            return "部分缺失"
        elif len(missing) <= 3:
            return "较多缺失"
        else:
            return "严重缺失"

    @staticmethod
    def _resolve_name(code: str) -> str:
        """解析股票名称（优先本地缓存，避免网络请求）"""
        # 1. 优先从本地 stock_info 缓存查询（最快，网络无关）
        try:
            from .stock_info_cache import get_stock_info_cache

            cache = get_stock_info_cache()
            name = cache.get_name(code)
            if name:
                return name
        except Exception:
            pass

        # 2. 回退：从数据源拉取（可能触发网络请求）
        try:
            from .data_manager import DataFetcher

            fetcher = DataFetcher()
            df = fetcher.fetch_stock_list()
            if df is not None and not df.empty:
                mask = None
                for c in ["代码", "code", "symbol"]:
                    if c in df.columns:
                        mask = df[c].astype(str).str.zfill(6) == str(code).zfill(6)
                        break
                if mask is not None and mask.any():
                    for c in ["名称", "name", "股票简称"]:
                        if c in df.columns:
                            return str(df.loc[mask, c].iloc[0])
        except Exception:
            pass
        return f"股票{code}"

    @staticmethod
    def _resolve_sector(code: str) -> str:
        """解析所属板块"""
        return "未知板块"

    @staticmethod
    def _safe_float(val) -> float | None:
        """安全转换为 float"""
        if val is None:
            return None
        try:
            v = float(val)
            return v if np.isfinite(v) else None
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _score_to_rating(score: int) -> str:
        if score >= 80:
            return "优秀"
        elif score >= 65:
            return "良好"
        elif score >= 50:
            return "中等"
        elif score >= 35:
            return "较差"
        else:
            return "差"


# ─── 模块级单例 ────────────────────────────────────────────
_diagnoser: StockDiagnoser | None = None


def get_diagnoser() -> StockDiagnoser:
    """获取诊断器单例"""
    global _diagnoser
    if _diagnoser is None:
        _diagnoser = StockDiagnoser()
    return _diagnoser
