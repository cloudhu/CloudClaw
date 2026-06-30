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
    leading_stocks: List[str]   # 领涨股（前3名，简短字符串）
    volume_surge: bool          # 是否放量
    trend: str                  # 趋势方向
    top_stocks: List[Dict] = field(default_factory=list)  # 板块内涨幅Top10详细数据
    total_stocks: int = 0       # 板块成分股总数


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
        """分析热门板块 - 按涨停家数排名 Top N，板块内按五日涨幅选 Top10 成分股

        不依赖板块指数接口（避免网络不可用时数据为空），直接使用涨停池数据：
          1. 从涨停池统计各行业涨停家数 → 热门板块排名
          2. 获取板块成分股，计算五日涨幅 → 板块内 Top10
          3. 若成分股接口不可用，降级使用涨停池内该行业股票作为成分股

        Returns:
            板块分析结果列表，按涨停家数降序排列
        """
        import pandas as pd

        results = []
        try:
            today_str = datetime.now().strftime("%Y%m%d")

            # 1. 获取涨停池数据
            zt_df = self.fetcher.fetch_limit_up_pool(today_str)
            if zt_df is None or zt_df.empty:
                logger.warning("涨停池数据为空，无法分析热门板块")
                return results

            if "所属行业" not in zt_df.columns:
                logger.warning("涨停池缺少「所属行业」列，无法统计板块涨停家数")
                return results

            # 2. 按行业统计涨停家数，降序取 Top N 作为热门板块
            sector_zt_counts = zt_df["所属行业"].value_counts().head(top_n)

            for rank_idx, (sec_name, zt_count) in enumerate(sector_zt_counts.items()):
                # 该行业涨停股的平均涨跌幅作为板块参考涨跌幅
                sec_zt = zt_df[zt_df["所属行业"] == sec_name]
                avg_pct = float(sec_zt["涨跌幅"].mean()) if "涨跌幅" in sec_zt.columns else 0.0

                leaders = []
                top_stocks = []
                total_stocks = 0
                detail = None

                # 3. 尝试获取板块全部成分股（含多级备用链路）
                try:
                    detail = self.fetcher.fetch_industry_stocks(industry=sec_name, board_code="")
                except Exception as e:
                    logger.debug(f"获取板块 {sec_name} 成分股失败: {e}")

                # 4. 降级：若成分股接口不可用，用涨停池内该行业股票作为成分股
                use_zt_fallback = False
                if detail is None or detail.empty or "代码" not in detail.columns:
                    logger.info(f"板块 {sec_name} 成分股接口不可用，降级使用涨停池数据")
                    # 从涨停池中提取该行业所有股票构造 detail
                    detail = sec_zt.copy()
                    # 统一列名
                    col_rename = {}
                    for src, dst in [
                        ("代码", "代码"), ("名称", "名称"), ("最新价", "最新价"),
                        ("涨跌幅", "涨跌幅"), ("成交量", "成交量"), ("成交额", "成交额"),
                        ("换手率", "换手率"), ("流通市值", "总市值"),
                    ]:
                        if src in detail.columns:
                            col_rename[src] = dst
                    if col_rename:
                        detail = detail.rename(columns=col_rename)
                    use_zt_fallback = True

                if detail is not None and not detail.empty and "代码" in detail.columns:
                    total_stocks = len(detail)

                    codes = [
                        str(r.get("代码", ""))
                        for _, r in detail.iterrows()
                        if str(r.get("代码", ""))
                    ]

                    # 5. 批量计算五日涨幅（仅对涨停池降级时的少量股票计算）
                    gains_5d: Dict[str, float] = {}
                    if codes:
                        gains_5d = self.fetcher.fetch_batch_5day_gains(codes)

                    # 6. 组装个股数据，按五日涨幅降序取 Top10
                    stock_list = []
                    for _, row in detail.iterrows():
                        code = str(row.get("代码", ""))
                        if not code:
                            continue
                        stock_list.append({
                            "code": code,
                            "name": str(row.get("名称", "")),
                            "price": round(float(row.get("最新价", 0) or 0), 2),
                            "pct_change": round(float(row.get("涨跌幅", 0) or 0), 2),
                            "pct_5d": round(gains_5d.get(code, 0.0), 2),
                            "volume": round(float(row.get("成交量", 0) or 0) / 1e4, 1),
                            "amount": round(float(row.get("成交额", 0) or 0) / 1e8, 2),
                            "turnover": round(float(row.get("换手率", 0) or 0), 2),
                            "pe": round(float(row.get("市盈率-动态", 0) or 0), 1),
                        })

                    # 按五日涨幅降序
                    stock_list.sort(key=lambda x: x["pct_5d"], reverse=True)
                    top_stocks = stock_list[:10]

                    # 领涨股 Top3（按五日涨幅）
                    top3 = stock_list[:3]
                    leaders = [
                        f"{s['code']}({s['pct_5d']:.1f}%)"
                        for s in top3
                    ]

                results.append(SectorAnalysis(
                    name=sec_name,
                    code="",
                    pct_change=round(avg_pct, 2),
                    strength_rank=rank_idx + 1,
                    leading_stocks=leaders,
                    volume_surge=zt_count >= 3,
                    trend="up",
                    top_stocks=top_stocks,
                    total_stocks=total_stocks,
                ))

        except Exception as e:
            logger.warning(f"获取板块数据失败: {e}", exc_info=True)

        logger.info(f"板块分析完成（按涨停家数）, 获取 {len(results)} 个热门板块")
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

    def diagnose_market_trend(self, days: int = 120) -> Dict[str, Any]:
        """牛熊行情研判 - 基于多指数长期趋势分析

        判断依据:
          - MA20/MA60/MA120/MA250 排列形态（多头/空头排列）
          - 当前价在年线上方/下方
          - MACD 周线级别趋势
          - 涨跌比与成交量趋势
        """
        result: Dict[str, Any] = {
            "phase": "震荡",
            "phase_code": "neutral",
            "confidence": 50,
            "bull_bear_score": 50,
            "signals": [],
            "advice": "",
        }

        try:
            start = (datetime.now() - timedelta(days=days + 60)).strftime("%Y%m%d")
            fetcher = DataFetcher()

            # 分析关键指数趋势
            key_indices = {
                "sh000001": "上证指数", "sz399001": "深证成指",
                "sh000300": "沪深300", "sh000905": "中证500",
            }

            bull_signals = 0
            bear_signals = 0
            total_checks = 0

            for code, name in key_indices.items():
                clean = code[2:]
                df = fetcher.fetch_index_daily(clean, start_date=start)
                if df is None or df.empty or len(df) < 60:
                    continue

                col_map = {"日期": "date", "收盘": "close", "成交量": "volume"}
                df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
                if "close" not in df.columns:
                    continue

                df = df.tail(days)
                close = df["close"]

                # 均线排列
                ma20 = close.rolling(20).mean().iloc[-1]
                ma60 = close.rolling(60).mean().iloc[-1]
                ma120 = close.rolling(120).mean().iloc[-1] if len(df) >= 120 else ma60
                latest = close.iloc[-1]

                # 多头排列: 价格 > MA20 > MA60 > MA120
                if latest > ma20 > ma60 > ma120:
                    bull_signals += 1
                # 空头排列: 价格 < MA20 < MA60 < MA120
                elif latest < ma20 < ma60 < ma120:
                    bear_signals += 1
                elif latest > ma20 > ma60:
                    bull_signals += 0.5
                elif latest < ma20 < ma60:
                    bear_signals += 0.5

                # 年线位置
                if len(df) >= 250:
                    ma250 = close.rolling(250).mean().iloc[-1]
                    if latest > ma250:
                        bull_signals += 0.5
                    else:
                        bear_signals += 0.5

                total_checks += 1

            if total_checks == 0:
                return result

            max_signal = total_checks * 2  # 每指数最多 2 个信号
            score = (bull_signals / max_signal) * 100 if max_signal > 0 else 50
            score = max(0, min(100, score))
            result["bull_bear_score"] = round(score, 1)

            # 牛熊判定
            if score >= 75:
                result["phase"] = "强势牛市"
                result["phase_code"] = "strong_bull"
                result["confidence"] = round(score)
                result["advice"] = "多头格局确立，趋势跟踪策略优先，持仓可适当激进"
            elif score >= 60:
                result["phase"] = "偏多震荡"
                result["phase_code"] = "weak_bull"
                result["confidence"] = round(score)
                result["advice"] = "短期偏强但趋势未完全确立，控制仓位分批建仓"
            elif score >= 40:
                result["phase"] = "震荡市"
                result["phase_code"] = "neutral"
                result["confidence"] = 50
                result["advice"] = "方向不明，轻仓参与高确定性机会，严格控制止损"
            elif score >= 25:
                result["phase"] = "偏空震荡"
                result["phase_code"] = "weak_bear"
                result["confidence"] = round(100 - score)
                result["advice"] = "市场偏弱，以防守为主，减少新开仓，持有头寸收紧止损"
            else:
                result["phase"] = "弱势熊市"
                result["phase_code"] = "strong_bear"
                result["confidence"] = round(100 - score)
                result["advice"] = "空头趋势明确，建议空仓或极轻仓防守，等待右侧信号"

            result["signals"].append(f"多头信号: {bull_signals}/{max_signal}, 空头信号: {bear_signals}/{max_signal}")

        except Exception as e:
            logger.warning(f"牛熊研判失败: {e}")

        return result

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
