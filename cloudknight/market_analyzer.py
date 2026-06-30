"""
市场分析器 - 大盘指数、板块指数、主力资金流向分析

盘前/盘中/盘后对 A 股市场进行多维度技术分析：
  - 6 大核心指数走势分析（上证/深证/创业板/科创50/沪深300/中证500）
  - 行业板块热度排名
  - 主力资金流向（北上资金、主力净流入）
  - 量价关系中寻找买卖点信号
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np

from .config import INDEX_CODES, MACD_FAST, MACD_SLOW, MACD_SIGNAL
from .indicators import (calc_macd, calc_kdj, calc_rsi, calc_ma,
                         analyze_macd, analyze_kdj, analyze_rsi,
                         trend_score, IndicatorResult)
from .data_manager import DataFetcher

logger = logging.getLogger(__name__)


@dataclass
class IndexAnalysis:
    """指数分析结果"""
    code: str
    name: str
    close: float
    pct_change: float          # 涨跌幅 %
    amplitude: float            # 振幅 %
    volume_ratio: float         # 量比
    trend_score: int            # 趋势评分 0-100
    trend_rating: str           # 趋势评级
    macd_signal: str            # MACD 信号
    kdj_signal: str             # KDJ 信号
    rsi_value: float            # RSI 值
    support: float = 0.0        # 支撑位
    resistance: float = 0.0     # 阻力位
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SectorAnalysis:
    """板块分析结果"""
    name: str
    code: str
    pct_change: float
    strength_rank: int          # 板块强度排名
    leading_stocks: List[str]   # 领涨股
    volume_surge: bool          # 是否放量
    trend: str                  # 趋势方向


@dataclass
class MarketSnapshot:
    """市场快照 - 汇总所有分析结果"""
    timestamp: datetime
    is_trading_day: bool
    indices: Dict[str, IndexAnalysis]
    hot_sectors: List[SectorAnalysis]
    market_sentiment: str       # bullish / bearish / neutral
    sentiment_score: int        # 0-100 市场情绪评分
    buy_signals: int            # 买入信号数
    sell_signals: int           # 卖出信号数
    capital_flow_summary: str   # 资金流向概要


class MarketAnalyzer:
    """市场分析器 - 多维度技术分析"""

    def __init__(self):
        self.fetcher = DataFetcher()

    def analyze_indices(self, days: int = 60) -> Dict[str, IndexAnalysis]:
        """分析六大核心指数

        Args:
            days: 分析用历史数据天数
        """
        results = {}
        today = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")

        def _analyze_one(code: str, name: str) -> Optional[IndexAnalysis]:
            try:
                clean_code = code[2:]  # 去掉 sh/sz 前缀
                df = self.fetcher.fetch_index_daily(clean_code, start_date=start)
                if df is None or df.empty or len(df) < 20:
                    return None

                # 标准化列名
                col_map = {"日期": "date", "开盘": "open", "收盘": "close",
                           "最高": "high", "最低": "low", "成交量": "volume"}
                df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

                required = ["close", "high", "low", "volume"]
                if not all(c in df.columns for c in required):
                    return None

                df = df.copy()
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df = df.dropna(subset=["date"]).sort_values("date").tail(days)

                close = df["close"].iloc[-1]
                prev_close = df["close"].iloc[-2] if len(df) > 1 else close
                pct = (close / prev_close - 1) * 100

                high = df["high"].iloc[-1]
                low = df["low"].iloc[-1]
                amplitude = (high - low) / prev_close * 100 if prev_close else 0

                vol_now = df["volume"].iloc[-1]
                vol_avg = df["volume"].iloc[-6:-1].mean() if len(df) >= 6 else vol_now
                vol_ratio = vol_now / vol_avg if vol_avg > 0 else 1.0

                score_result = trend_score(df)
                ts = score_result.get("score", 50)
                tr = score_result.get("rating", "中性")

                macd_result = analyze_macd(df)
                kdj_result = analyze_kdj(df)
                rsi_result = analyze_rsi(df)

                rsi_val = float(rsi_result.details.get("rsi", 50)) if rsi_result.details else 50.0

                # 支撑位/阻力位
                df_calc = calc_ma(df, [20, 60])
                support = min(
                    df_calc["MA20"].iloc[-1] if "MA20" in df_calc.columns else close,
                    df["low"].tail(20).min()
                )
                resistance = max(
                    df["high"].tail(20).max(),
                    df_calc["MA60"].iloc[-1] if "MA60" in df_calc.columns else close
                )

                return IndexAnalysis(
                    code=code, name=name, close=close,
                    pct_change=round(pct, 2), amplitude=round(amplitude, 2),
                    volume_ratio=round(vol_ratio, 2),
                    trend_score=ts, trend_rating=tr,
                    macd_signal=macd_result.signal,
                    kdj_signal=kdj_result.signal,
                    rsi_value=round(rsi_val, 1),
                    support=round(support, 2),
                    resistance=round(resistance, 2),
                    details={
                        "macd_trend": macd_result.trend,
                        "macd_strength": macd_result.strength,
                        "kdj_trend": kdj_result.trend,
                        "rsi_trend": rsi_result.trend,
                    }
                )
            except Exception as e:
                logger.debug(f"分析指数 {name}({code}) 失败: {e}")
                return None

        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = {pool.submit(_analyze_one, code, name): code
                       for code, name in INDEX_CODES.items()}
            for f in as_completed(futures):
                result = f.result()
                if result:
                    results[result.code] = result

        return results

    def analyze_sectors(self, top_n: int = 10) -> List[SectorAnalysis]:
        """分析行业板块热度排名

        Returns:
            板块分析结果列表，按强度排序
        """
        results = []
        try:
            import akshare as ak
            # 获取行业板块行情
            df = ak.stock_board_industry_name_em()
            if df is None or df.empty:
                return results

            # 按涨跌幅排序
            if "涨跌幅" in df.columns:
                df = df.sort_values("涨跌幅", ascending=False)

            for i, (_, row) in enumerate(df.head(top_n).iterrows()):
                name = str(row.get("板块名称", row.get("板块", "")))
                code = str(row.get("板块代码", row.get("代码", "")))
                pct = float(row.get("涨跌幅", 0))

                # 获取领涨股
                leaders = []
                try:
                    detail = ak.stock_board_concept_cons_em(symbol=name)
                    if detail is not None and not detail.empty and "涨跌幅" in detail.columns:
                        top3 = detail.nlargest(3, "涨跌幅")
                        leaders = [f"{r.get('代码', '')}({r.get('涨跌幅', 0):.1f}%)"
                                   for _, r in top3.iterrows()]
                except Exception:
                    pass

                results.append(SectorAnalysis(
                    name=name, code=code,
                    pct_change=round(pct, 2),
                    strength_rank=i + 1,
                    leading_stocks=leaders,
                    volume_surge=pct > 2.0,
                    trend="up" if pct > 0 else ("down" if pct < 0 else "flat")
                ))
        except Exception as e:
            logger.warning(f"获取板块数据失败: {e}")

        return results

    def get_market_sentiment(self) -> Dict[str, Any]:
        """综合市场情绪评估

        Returns:
            sentiment: bullish/bearish/neutral
            score: 0-100
            factors: 各项因子评分
        """
        indices = self.analyze_indices(days=60)

        if not indices:
            return {"sentiment": "neutral", "score": 50, "factors": {}}

        # 上涨指数占比
        up_count = sum(1 for i in indices.values() if i.pct_change > 0)
        up_ratio = up_count / len(indices) if indices else 0

        # 平均趋势评分
        avg_trend = np.mean([i.trend_score for i in indices.values()])

        # MACD 金叉指数数
        macd_bull = sum(1 for i in indices.values() if "金叉" in i.macd_signal.lower())

        # RSI 均值
        avg_rsi = np.mean([i.rsi_value for i in indices.values()])

        # 综合评分
        factor_scores = {
            "涨跌比": int(up_ratio * 100),
            "趋势评分": int(avg_trend),
            "MACD_金叉数": min(int(macd_bull / max(len(indices), 1) * 100), 100),
            "RSI_状态": max(0, min(100, int(100 - abs(avg_rsi - 50)))),
        }
        score = int(np.mean(list(factor_scores.values())))

        if score >= 65:
            sentiment = "bullish"
        elif score <= 35:
            sentiment = "bearish"
        else:
            sentiment = "neutral"

        return {
            "sentiment": sentiment,
            "score": score,
            "factors": factor_scores,
            "indices": {i.code: {
                "name": i.name, "pct": i.pct_change, "trend": i.trend_score,
                "macd": i.macd_signal, "kdj": i.kdj_signal
            } for i in indices.values()}
        }

    def get_market_snapshot(self) -> MarketSnapshot:
        """获取市场全景快照

        Returns:
            MarketSnapshot: 包含指数、板块、情绪、资金流等完整数据
        """
        indices = self.analyze_indices(days=60)
        sectors = self.analyze_sectors(top_n=10)
        sentiment = self.get_market_sentiment()

        # 统计买卖信号数
        buy_count = sum(1 for i in indices.values()
                        if "金叉" in i.macd_signal.lower() or "金叉" in i.kdj_signal.lower())
        sell_count = sum(1 for i in indices.values()
                         if "死叉" in i.macd_signal.lower() or "死叉" in i.kdj_signal.lower())

        return MarketSnapshot(
            timestamp=datetime.now(),
            is_trading_day=True,  # caller 会设置
            indices=indices,
            hot_sectors=sectors,
            market_sentiment=sentiment["sentiment"],
            sentiment_score=sentiment["score"],
            buy_signals=buy_count,
            sell_signals=sell_count,
            capital_flow_summary="待获取"  # 实盘中填充
        )

    def format_snapshot_report(self, snapshot: MarketSnapshot) -> str:
        """格式化市场快照为可读报告"""
        lines = []
        lines.append("╔══════════════════════════════════════════╗")
        lines.append("║        A 股市场全景分析报告              ║")
        lines.append(f"║  时间: {snapshot.timestamp.strftime('%Y-%m-%d %H:%M')}                     ║")
        sentiment_icon = {"bullish": "🐂 多头", "bearish": "🐻 空头", "neutral": "→ 中性"}
        lines.append(f"║  情绪: {sentiment_icon.get(snapshot.market_sentiment, '中立')} "
                     f"({snapshot.sentiment_score}/100)        ║")
        lines.append("╠══════════════════════════════════════════╣")

        # 指数
        lines.append("║ 核心指数:                                  ║")
        for idx in snapshot.indices.values():
            arrow = "▲" if idx.pct_change >= 0 else "▼"
            lines.append(f"║  {arrow} {idx.name:<10s} {idx.close:>8.2f} "
                         f"{idx.pct_change:+.2f}%  "
                         f"MACD:{idx.macd_signal:<4s} RSI:{idx.rsi_value:<5.1f} ║")

        # 板块
        if snapshot.hot_sectors:
            lines.append("╠══════════════════════════════════════════╣")
            lines.append("║ 热门板块 (Top 5):                          ║")
            for s in snapshot.hot_sectors[:5]:
                lines.append(f"║  {s.strength_rank}. {s.name:<12s} {s.pct_change:+.2f}%      ║")

        # 信号汇总
        lines.append("╠══════════════════════════════════════════╣")
        lines.append(f"║  买入信号: {snapshot.buy_signals}  |  卖出信号: {snapshot.sell_signals}        ║")
        lines.append("╚══════════════════════════════════════════╝")
        return "\n".join(lines)
