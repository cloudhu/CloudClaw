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
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

import pandas as pd
import numpy as np

from .indicators import (
    IndicatorResult, comprehensive_analysis, trend_score,
    calc_ma, calc_bollinger, calc_atr, calc_breakout,
)
from .config import MA_PERIODS, INDEX_CODES

logger = logging.getLogger(__name__)

# ─── 诊股超时配置 ─────────────────────────────────────────
_DIAGNOSE_KLINE_TIMEOUT = 10       # K线获取超时（秒）
_DIAGNOSE_FUNDAMENTAL_TIMEOUT = 8  # 财务数据获取超时（秒）
_DIAGNOSE_MONEYFLOW_TIMEOUT = 8    # 资金流向获取超时（秒）
_DIAGNOSE_TOTAL_TIMEOUT = 25       # 全流程总超时（秒）


# ─── 数据模型 ─────────────────────────────────────────────

@dataclass
class TechDiagnosis:
    """技术面诊断结果"""
    score: int                          # 技术面评分 0-100
    rating: str                         # 评级
    current_price: float                # 当前价格
    pct_change_5d: Optional[float]      # 5日涨跌幅 %
    pct_change_20d: Optional[float]     # 20日涨跌幅 %
    amplitude_5d: Optional[float]       # 5日振幅 %
    ma_alignment: str                   # 均线排列: 多头/空头/交叉/缠绕
    ma_positions: Dict[str, float]      # 各均线当前值 {MA5: x, MA10: x, ...}
    macd: IndicatorResult
    kdj: IndicatorResult
    rsi: IndicatorResult
    bollinger: IndicatorResult
    volume_price: IndicatorResult
    atr: Optional[float]                # 平均真实波幅
    support: float                      # 支撑位
    resistance: float                   # 阻力位
    stock_name: str = ""                # 股票名称


@dataclass
class FundamentalDiagnosis:
    """基本面诊断结果"""
    score: int = 50                        # 基本面评分 0-100
    rating: str = "未评估"                   # 评级
    pe: Optional[float] = None             # 市盈率
    pb: Optional[float] = None             # 市净率
    roe: Optional[float] = None            # ROE %
    revenue_growth: Optional[float] = None # 营收增长率 %
    profit_growth: Optional[float] = None  # 净利润增长率 %
    market_cap: Optional[float] = None     # 总市值（亿）
    dividend_yield: Optional[float] = None # 股息率 %
    debt_ratio: Optional[float] = None     # 资产负债率 %
    gross_margin: Optional[float] = None   # 毛利率 %
    net_margin: Optional[float] = None     # 净利率 %
    industry_pe: Optional[float] = None    # 行业平均PE
    data_available: bool = False           # 是否获取到财务数据


@dataclass
class CapitalDiagnosis:
    """资金面诊断结果"""
    score: int = 50                          # 资金面评分 0-100
    rating: str = "未评估"                    # 评级
    main_force_direction: str = "unknown"    # 主力动向: inflow/outflow/neutral/unknown
    main_force_5d_net: Optional[float] = None  # 近5日主力净流入（亿）
    turnover_rate: Optional[float] = None     # 换手率 %
    volume_ratio: Optional[float] = None      # 量比
    data_available: bool = False


@dataclass
class MarketContext:
    """市场面诊断结果"""
    score: int = 50                          # 市场面评分 0-100
    rating: str = "未评估"                    # 评级
    sector_name: str = "未知板块"             # 所属板块
    sector_rank: Optional[int] = None        # 板块强度排名
    sector_pct: Optional[float] = None       # 板块涨跌幅 %
    market_trend: str = "未知"               # 大盘趋势
    market_sentiment: str = "未知"            # 市场情绪
    limit_status: str = "正常"               # 涨跌停状态


@dataclass
class StrategyDiagnosis:
    """单策略诊断结果"""
    name: str                           # 策略名称
    key: str                            # 策略 key
    score: int                          # 策略评分 0-100
    signal: str                         # 信号: buy/hold/sell
    rating: str                         # 评级文字
    match_count: int                    # 符合条件的项数
    total_conditions: int               # 总条件数
    reasons: List[str] = field(default_factory=list)  # 匹配原因
    warnings: List[str] = field(default_factory=list) # 风险警示


@dataclass
class StockDiagnosis:
    """个股综合诊断报告"""
    code: str                           # 股票代码
    name: str                           # 股票名称
    timestamp: str                      # 诊断时间
    technical: TechDiagnosis
    fundamental: FundamentalDiagnosis
    capital: CapitalDiagnosis
    market: MarketContext
    composite_score: int                # 综合评分 0-100
    composite_rating: str               # 综合评级
    recommendation: str                 # 操作建议
    risk_level: str                     # 风险等级: 低/中/高/极高
    summary: str                        # 一句话总结
    detail_scores: Dict[str, int] = field(default_factory=dict)  # 各维度得分明细
    strategy_diagnoses: List[StrategyDiagnosis] = field(default_factory=list)  # 各策略诊断
    data_quality: str = "完整"           # 数据质量: 完整/部分缺失/严重缺失


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
            StockDiagnosis 综合诊断报告
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

        # 策略诊断
        strategy_diags = self._diagnose_strategies(df_kline, tech, fund, cap)

        # 数据质量评估
        data_quality = self._assess_data_quality(tech, fund, cap, market)

        # 综合评分（加入策略一致性）
        composite, rating, rec, risk, summary = self._compute_composite(
            tech, fund, cap, market, strategy_diags
        )

        name = tech.stock_name or self._resolve_name(code)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return StockDiagnosis(
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
        )

    def quick_diagnose(self, code: str) -> StockDiagnosis:
        """快速诊断 — 跳过低优先级数据，仅技术面+市场面"""
        code = str(code).zfill(6)
        df_kline = self._fetch_kline_safe(code)
        tech = self._diagnose_technical(code, df_kline)
        market = self._diagnose_market_context(code, tech.current_price, deadline=0)
        fund = FundamentalDiagnosis(score=50, rating="未评估", data_available=False)
        cap = CapitalDiagnosis(score=50, rating="未评估", main_force_direction="unknown",
                               data_available=False)
        strategy_diags = self._diagnose_strategies(df_kline, tech, fund, cap)
        composite, rating, rec, risk, summary = self._compute_composite(
            tech, fund, cap, market, strategy_diags
        )
        name = tech.stock_name or self._resolve_name(code)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data_quality = "部分缺失"  # 快速诊断必然缺失基本面和资金面
        return StockDiagnosis(
            code=code, name=name, timestamp=ts,
            technical=tech, fundamental=fund, capital=cap, market=market,
            strategy_diagnoses=strategy_diags,
            composite_score=composite, composite_rating=rating,
            recommendation=rec, risk_level=risk, summary=summary,
            detail_scores={"技术面": tech.score, "基本面": fund.score,
                           "资金面": cap.score, "市场面": market.score},
            data_quality=data_quality,
        )

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
            return CapitalDiagnosis(score=50, rating="超时跳过", main_force_direction="unknown",
                                    data_available=False)
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(self._diagnose_capital, code)
            try:
                remaining = max(1, deadline - _time.time())
                return fut.result(timeout=min(_DIAGNOSE_MONEYFLOW_TIMEOUT, remaining))
            except FutureTimeoutError:
                return CapitalDiagnosis(score=50, rating="超时跳过", main_force_direction="unknown",
                                        data_available=False)
            except Exception as e:
                logger.warning(f"诊股: {code} 资金面异常: {e}")
                return CapitalDiagnosis(score=50, rating="获取失败", main_force_direction="unknown",
                                        data_available=False)

    # ── 技术面诊断 ────────────────────────────────────────

    def _diagnose_technical(self, code: str, df: pd.DataFrame) -> TechDiagnosis:
        """技术面全面诊断"""
        if df is None or df.empty or len(df) < 30:
            return self._empty_tech()

        # 标准化列名
        col_map = {
            "日期": "date", "开盘": "open", "收盘": "close",
            "最高": "high", "最低": "low", "成交量": "volume",
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
            amp_5d = round(
                (float(recent_5["high"].max()) / float(recent_5["low"].min()) - 1) * 100, 2
            )

        # 均线计算与排列判断
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
        sorted_mas = sorted(ma_values.items(), key=lambda x: x[1], reverse=True)
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
            # 检查是否有交叉
            first = list(ma_values.keys())[0]
            last = list(ma_values.keys())[-1]
            if ma_values[first] > ma_values[last]:
                ma_alignment = "偏多头"
            else:
                ma_alignment = "偏空头"

        # 各项技术指标
        macd = self._safe_indicator(df, "analyze_macd")
        kdj = self._safe_indicator(df, "analyze_kdj")
        rsi = self._safe_indicator(df, "analyze_rsi")
        boll = self._safe_indicator(df, "analyze_bollinger")
        vol_pr = self._safe_indicator(df, "analyze_volume_price")

        # ATR
        atr_val = None
        try:
            atr_series = calc_atr(df)
            atr_val = round(float(atr_series.iloc[-1]), 2)
        except Exception:
            pass

        # 支撑位/阻力位
        support, resistance = self._calc_sr(df, current_price)

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

    def _safe_indicator(self, df, func_name: str) -> IndicatorResult:
        """安全计算技术指标"""
        try:
            from . import indicators as ind
            fn = getattr(ind, func_name)
            return fn(df)
        except Exception:
            return IndicatorResult("neutral", "hold", 50, {})

    def _calc_sr(self, df: pd.DataFrame, price: float) -> Tuple[float, float]:
        """计算支撑位和阻力位"""
        support, resistance = price * 0.95, price * 1.05  # 默认 ±5%
        try:
            df = df.copy()
            if len(df) < 60:
                return round(support, 2), round(resistance, 2)
            # 布林带
            df_bb = calc_bollinger(df)
            if "BOLL_DN" in df_bb.columns and "BOLL_UP" in df_bb.columns:
                bb_low = df_bb["BOLL_DN"].iloc[-1]
                bb_high = df_bb["BOLL_UP"].iloc[-1]
                if pd.notna(bb_low) and pd.notna(bb_high):
                    support = min(support, float(bb_low))
                    resistance = max(resistance, float(bb_high))
            # Donchian 突破
            hh, ll = calc_breakout(df, 20)
            if not hh.empty and not ll.empty:
                hh_val, ll_val = float(hh.iloc[-1]), float(ll.iloc[-1])
                if pd.notna(hh_val) and pd.notna(ll_val):
                    resistance = max(resistance, hh_val)
                    support = min(support, ll_val)
            # MA20/MA60
            df_ma = calc_ma(df, [20, 60])
            for p in [20, 60]:
                col = f"MA{p}"
                if col in df_ma.columns:
                    ma_val = df_ma[col].iloc[-1]
                    if pd.notna(ma_val):
                        ma_val = float(ma_val)
                        if ma_val < price:
                            support = max(support, ma_val)
                        else:
                            resistance = min(resistance, ma_val)
        except Exception:
            pass
        return round(support, 2), round(resistance, 2)

    def _empty_tech(self) -> TechDiagnosis:
        return TechDiagnosis(
            score=50, rating="无数据", current_price=0,
            pct_change_5d=None, pct_change_20d=None, amplitude_5d=None,
            ma_alignment="无数据", ma_positions={},
            macd=IndicatorResult("neutral", "hold", 50, {}),
            kdj=IndicatorResult("neutral", "hold", 50, {}),
            rsi=IndicatorResult("neutral", "hold", 50, {}),
            bollinger=IndicatorResult("neutral", "hold", 50, {}),
            volume_price=IndicatorResult("neutral", "hold", 50, {}),
            atr=None, support=0, resistance=0,
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
            fund.revenue_growth = self._safe_float(
                latest.get("营业收入增长率") or latest.get("营收增长率")
            )
            fund.profit_growth = self._safe_float(
                latest.get("净利润增长率") or latest.get("利润增长率")
            )
            fund.market_cap = self._safe_float(latest.get("总市值") or latest.get("市值"))
            fund.dividend_yield = self._safe_float(latest.get("股息率"))
            fund.debt_ratio = self._safe_float(
                latest.get("资产负债率") or latest.get("负债率")
            )
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
            col_map = {"日期": "date", "开盘": "open", "收盘": "close", "最高": "high", "最低": "low", "成交量": "volume"}
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
                    vol_trend = float(np.mean(vols[-5:]) / np.mean(vols[-20:-5]) - 1) if np.mean(vols[-20:-5]) > 0 else 0
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
        cap = CapitalDiagnosis(
            score=50, rating="待分析", main_force_direction="unknown",
            data_available=False
        )
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

    def _diagnose_market_context(self, code: str, price: float,
                                  deadline: float = 0) -> MarketContext:
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
            "强势牛市": 90, "偏多震荡": 65, "震荡市": 50,
            "偏空震荡": 35, "弱势熊市": 15, "未知": 50,
        }
        sentiment_score_map = {
            "极度贪婪": 85, "贪婪": 70, "中性": 50,
            "恐惧": 30, "极度恐惧": 15, "未知": 50,
        }

        m_score = int(
            trend_score_map.get(market_trend, 50) * 0.6 +
            sentiment_score_map.get(market_sentiment, 50) * 0.4
        )

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
        strategy_diags: List[StrategyDiagnosis] = None,
    ) -> Tuple[int, str, str, str, str]:
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
            if buy_count >= 3:
                parts.append(f"策略一致看多({buy_count}/4)")
            elif buy_count >= 2:
                parts.append(f"策略分歧({buy_count}/4看多)")
            else:
                parts.append(f"策略偏空({4-buy_count}/4看空)")
        parts.append(f"综合{rating}")
        summary = f"{tech.stock_name or '该股'} — " + "，".join(parts)

        return composite, rating, rec, risk, summary

    def _strategy_consensus_score(self, diags: List[StrategyDiagnosis]) -> int:
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
        self, tech: TechDiagnosis, fund: FundamentalDiagnosis,
        cap: CapitalDiagnosis, composite: int,
        strategy_diags: List[StrategyDiagnosis] = None,
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
                recs.append(("各策略指标中性偏多，可适量关注"))
            elif composite >= 40:
                recs.append("各项指标中性偏空，建议观望等待信号")
            else:
                recs.append("综合评分偏低，建议规避")

        return "；".join(recs) if recs else "暂无明确建议"

    # ── 策略诊断 ──────────────────────────────────────────

    def _diagnose_strategies(
        self, df: pd.DataFrame,
        tech: TechDiagnosis,
        fund: FundamentalDiagnosis,
        cap: CapitalDiagnosis,
    ) -> List[StrategyDiagnosis]:
        """用四种交易策略分别评估该股票"""
        results = []
        if df is None or df.empty or len(df) < 40:
            return self._empty_strategy_results()

        # 标准化列名（兼容中英文列名）
        col_map = {
            "日期": "date", "开盘": "open", "收盘": "close",
            "最高": "high", "最低": "low", "成交量": "volume",
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

        results.append(self._eval_dragon_head(closes, highs, lows, volumes, opens, indicators, cap))
        results.append(self._eval_sparrow(closes, highs, lows, indicators))
        results.append(self._eval_turtle(closes, highs, lows, indicators))
        results.append(self._eval_value_invest(closes, indicators, fund))

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
                tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
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
            k_prev = 50 * 2/3 + rsv_prev * 1/3
            d_prev = 50 * 2/3 + k_prev * 1/3
            k = k_prev * 2/3 + rsv * 1/3
            d = d_prev * 2/3 + k * 1/3
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
                pcts.append(float((closes[-i] / closes[-i-1] - 1) * 100))
            ind["pcts"] = pcts
        # 250日价格分位
        if len(closes) >= 250:
            lookback = closes[-250:]
            ind["price_percentile"] = float((lookback < closes[-1]).sum() / 250 * 100)

        return ind

    # ── 龙头战法评估 ──

    def _eval_dragon_head(self, closes, highs, lows, volumes, opens, ind, cap):
        """龙头战法评分：连板、量比、换手率"""
        reasons, warnings = [], []
        score = 30  # 基础分

        pcts = ind.get("pcts", [])
        vol_ratio = ind.get("vol_ratio", 1)

        # 连板天数
        limit_days = 0
        for p in pcts:
            if p >= 9.5:
                limit_days += 1
            else:
                break
        if limit_days > 0:
            score += min(limit_days * 10, 30)
            reasons.append(f"已{limit_days}连板")
        else:
            warnings.append("无涨停连板")

        # 涨停强度
        pct_today = ind.get("pct_today", 0)
        if pct_today >= 9.5:
            score += 15
            reasons.append("今日涨停")
        elif pct_today >= 5:
            score += 5
            reasons.append("今日大幅上涨")
        elif pct_today <= -9.5:
            score -= 20
            warnings.append("今日跌停")

        # 量比
        if 1.5 <= vol_ratio <= 3:
            score += 15
            reasons.append(f"量比{vol_ratio:.1f}适中")
        elif vol_ratio > 3:
            score += 8
            reasons.append(f"量比{vol_ratio:.1f}偏大")
        elif vol_ratio >= 1:
            score += 3
        else:
            score -= 5
            warnings.append(f"量比{vol_ratio:.1f}偏小")

        # 换手率（资金面数据优先，否则从成交量估算）
        turnover = cap.turnover_rate
        if turnover is not None:
            if 5 <= turnover <= 30:
                score += 10
                reasons.append(f"换手率{turnover:.1f}%活跃")
            elif 2 <= turnover < 5:
                score += 5
                reasons.append(f"换手率{turnover:.1f}%")
            elif turnover > 30:
                score -= 5
                warnings.append(f"换手率{turnover:.1f}%过高")
            else:
                warnings.append(f"换手率{turnover:.1f}%偏低")

        score = max(0, min(100, score))
        match = len(reasons)
        total = 4

        if score >= 70:
            signal, rating = "buy", "强烈推荐"
        elif score >= 55:
            signal, rating = "buy", "关注"
        elif score >= 40:
            signal, rating = "hold", "观望"
        else:
            signal, rating = "sell", "不适用"

        return StrategyDiagnosis(
            name="龙头战法", key="dragon_head",
            score=score, signal=signal, rating=rating,
            match_count=match, total_conditions=total,
            reasons=reasons, warnings=warnings,
        )

    # ── 麻雀战法评估 ──

    def _eval_sparrow(self, closes, highs, lows, ind):
        """麻雀战法评分：MA20支撑、KDJ金叉、RSI"""
        reasons, warnings = [], []
        score = 30

        cur = closes[-1]
        ma20 = ind.get("ma20", 0)

        # MA20 支撑
        if ma20 > 0:
            dist = (cur - ma20) / ma20 * 100
            if -2 <= dist <= 2:
                score += 25
                reasons.append(f"紧贴MA20支撑(dist={dist:+.1f}%)")
            elif -5 <= dist <= 5:
                score += 15
                reasons.append(f"靠近MA20(dist={dist:+.1f}%)")
            elif dist > 10:
                score += 5
                warnings.append(f"远离MA20上方({dist:+.1f}%)")
            elif dist < -5:
                score -= 10
                warnings.append(f"跌破MA20({dist:+.1f}%)")
        else:
            warnings.append("MA20无数据")

        # KDJ 金叉
        k, d = ind.get("k", 50), ind.get("d", 50)
        kp, dp = ind.get("k_prev", 50), ind.get("d_prev", 50)
        if kp <= dp and k > d:
            score += 25
            reasons.append(f"KDJ金叉(K={k:.0f},D={d:.0f})")
        elif k > d:
            score += 10
            reasons.append(f"KDJ多头(K={k:.0f}>D={d:.0f})")
        elif k < d:
            score -= 5
            warnings.append(f"KDJ空头(K={k:.0f}<D={d:.0f})")

        # RSI
        rsi = ind.get("rsi", 50)
        if 30 <= rsi <= 55:
            score += 20
            reasons.append(f"RSI={rsi:.0f}温和")
        elif 55 < rsi <= 70:
            score += 8
            reasons.append(f"RSI={rsi:.0f}偏强")
        elif rsi > 70:
            score -= 5
            warnings.append(f"RSI={rsi:.0f}超买")
        elif rsi < 30:
            score -= 10
            warnings.append(f"RSI={rsi:.0f}超卖")

        score = max(0, min(100, score))
        match = len(reasons)
        total = 3

        if score >= 70:
            signal, rating = "buy", "建议建仓"
        elif score >= 55:
            signal, rating = "buy", "关注"
        elif score >= 35:
            signal, rating = "hold", "观望"
        else:
            signal, rating = "sell", "回避"

        return StrategyDiagnosis(
            name="麻雀战法", key="sparrow",
            score=score, signal=signal, rating=rating,
            match_count=match, total_conditions=total,
            reasons=reasons, warnings=warnings,
        )

    # ── 海龟战法评估 ──

    def _eval_turtle(self, closes, highs, lows, ind):
        """海龟战法评分：MA200过滤、唐奇安突破、ATR波动"""
        reasons, warnings = [], []
        score = 30

        cur = closes[-1]
        ma200 = ind.get("ma200", 0)

        # MA200 过滤
        if ma200 > 0 and cur >= ma200:
            score += 20
            reasons.append(f"站上MA200(>={ma200:.2f})")
        elif ma200 > 0:
            score -= 15
            warnings.append(f"低于MA200({cur:.2f}<{ma200:.2f})")
        else:
            warnings.append("MA200无数据(需250日K线)")

        # 唐奇安通道
        dh20 = ind.get("donchian_h20", 0)
        dl20 = ind.get("donchian_l20", 0)
        if dh20 > 0:
            dist_to_h = (dh20 - cur) / dh20 * 100
            if dist_to_h <= 2:
                score += 25
                reasons.append(f"接近突破({dist_to_h:.1f}%到20日高点)")
            elif dist_to_h <= 5:
                score += 15
                reasons.append(f"靠近通道上沿({dist_to_h:.1f}%)")
            elif dist_to_h <= 10:
                score += 8
            else:
                score -= 5
                warnings.append(f"远离突破({dist_to_h:.1f}%)")
        else:
            warnings.append("唐奇安通道无数据")

        # ATR 波动率
        atr = ind.get("atr14", 0)
        if atr > 0 and cur > 0:
            atr_pct = atr / cur * 100
            if 2 <= atr_pct <= 5:
                score += 15
                reasons.append(f"ATR={atr_pct:.1f}%适度波动")
            elif atr_pct < 2:
                score += 5
                warnings.append(f"ATR={atr_pct:.1f}%低波动")
            else:
                score += 8
                reasons.append(f"ATR={atr_pct:.1f}%高波动")
        else:
            score += 5  # 默认中性

        # 趋势强度（价格与多条均线的关系）
        ma20 = ind.get("ma20", 0)
        ma60 = ind.get("ma60", 0)
        if ma20 > 0 and ma60 > 0:
            if cur > ma20 > ma60:
                score += 15
                reasons.append("均线多头排列")
            elif cur > ma20:
                score += 8
            elif cur < ma60:
                score -= 10
                warnings.append("均线空头排列")

        score = max(0, min(100, score))
        match = len(reasons)
        total = 4

        if score >= 70:
            signal, rating = "buy", "建议建仓"
        elif score >= 50:
            signal, rating = "hold", "等待突破"
        else:
            signal, rating = "sell", "不适用"

        return StrategyDiagnosis(
            name="海龟战法", key="turtle",
            score=score, signal=signal, rating=rating,
            match_count=match, total_conditions=total,
            reasons=reasons, warnings=warnings,
        )

    # ── 价值投资评估 ──

    def _eval_value_invest(self, closes, ind, fund):
        """价值投资评分：价格分位、估值、RSI"""
        reasons, warnings = [], []
        score = 30

        # 250日价格分位
        pct = ind.get("price_percentile", 50)
        if pct <= 20:
            score += 30
            reasons.append(f"极度低估(250日分位{pct:.0f}%)")
        elif pct <= 40:
            score += 20
            reasons.append(f"低估(250日分位{pct:.0f}%)")
        elif pct <= 60:
            score += 5
            reasons.append(f"估值合理(250日分位{pct:.0f}%)")
        elif pct <= 80:
            score -= 10
            warnings.append(f"偏高(250日分位{pct:.0f}%)")
        else:
            score -= 20
            warnings.append(f"严重高估(250日分位{pct:.0f}%)")

        # RSI 入场时机
        rsi = ind.get("rsi", 50)
        if rsi < 45:
            score += 20
            reasons.append(f"RSI={rsi:.0f}入场时机好")
        elif rsi < 60:
            score += 10
        else:
            score -= 5
            warnings.append(f"RSI={rsi:.0f}偏高")

        # MA250 位置
        ma250 = ind.get("ma250", 0)
        cur = closes[-1]
        if ma250 > 0:
            dist250 = (cur - ma250) / ma250 * 100
            if dist250 < -10:
                score += 10
                reasons.append("跌破年线，超跌信号")
            elif -10 <= dist250 <= 10:
                score += 5
                reasons.append("靠近年线")
            elif dist250 > 30:
                score -= 5
                warnings.append(f"远离年线上方({dist250:.0f}%)")
        else:
            score += 3  # 数据不足，中性

        # 如果基本面数据可用
        if fund.data_available:
            if fund.pe is not None and fund.pe < 15:
                score += 10
                reasons.append(f"PE={fund.pe:.1f}低估")
            elif fund.pe is not None and fund.pe > 50:
                score -= 10
                warnings.append(f"PE={fund.pe:.1f}高估")
            if fund.pb is not None and fund.pb < 1.5:
                score += 8
                reasons.append(f"PB={fund.pb:.2f}破净")
            if fund.roe is not None and fund.roe > 15:
                score += 7
                reasons.append(f"ROE={fund.roe:.1f}%优秀")
            if fund.dividend_yield is not None and fund.dividend_yield > 3:
                score += 5
                reasons.append(f"股息率{fund.dividend_yield:.1f}%")
        else:
            warnings.append("无财务数据(仅技术面估值)")

        score = max(0, min(100, score))
        match = len(reasons)
        total = 4 if fund.data_available else 3

        if score >= 70:
            signal, rating = "buy", "建议长线布局"
        elif score >= 55:
            signal, rating = "buy", "逢低关注"
        elif score >= 40:
            signal, rating = "hold", "观望"
        else:
            signal, rating = "sell", "回避"

        return StrategyDiagnosis(
            name="价值投资", key="value_invest",
            score=score, signal=signal, rating=rating,
            match_count=match, total_conditions=total,
            reasons=reasons, warnings=warnings,
        )

    def _empty_strategy_results(self) -> List[StrategyDiagnosis]:
        """K线数据不足时返回空策略结果"""
        results = []
        for key, name in [("dragon_head", "龙头战法"), ("sparrow", "麻雀战法"),
                           ("turtle", "海龟战法"), ("value_invest", "价值投资")]:
            results.append(StrategyDiagnosis(
                name=name, key=key, score=50, signal="hold", rating="数据不足",
                match_count=0, total_conditions=0,
                warnings=["K线数据不足，无法评估"],
            ))
        return results

    # ── 数据质量评估 ───────────────────────────────────

    def _assess_data_quality(
        self, tech: TechDiagnosis, fund: FundamentalDiagnosis,
        cap: CapitalDiagnosis, market: MarketContext,
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
        """解析股票名称"""
        try:
            from .data_manager import DataFetcher
            fetcher = DataFetcher()
            df = fetcher.fetch_stock_list()
            if df is not None and not df.empty:
                cols = df.columns
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
    def _safe_float(val) -> Optional[float]:
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
_diagnoser: Optional[StockDiagnoser] = None


def get_diagnoser() -> StockDiagnoser:
    """获取诊断器单例"""
    global _diagnoser
    if _diagnoser is None:
        _diagnoser = StockDiagnoser()
    return _diagnoser
