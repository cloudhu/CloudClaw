"""
集合竞价分析器 - 9:26 竞价结果分析

A股集合竞价时间段: 9:15-9:25
- 9:15-9:20  可以挂单也可以撤单
- 9:20-9:25  只能挂单不能撤单（虚拟成交价逐渐收敛）
- 9:25       产生开盘价
- 9:26       竞价结果发布

分析维度:
  1. 竞价量比 - 成交量放大程度
  2. 竞价涨幅 - 价格变动方向与力度
  3. 竞价末段走势 - 最后1分钟的挂单强度
  4. 标的股池竞价表现
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .config import AUCTION_PRICE_CHANGE_ALERT, AUCTION_VOLUME_RATIO_MIN
from .data_manager import DataFetcher

logger = logging.getLogger(__name__)


@dataclass
class AuctionStockResult:
    """个股竞价分析结果"""

    code: str
    name: str
    prev_close: float  # 昨日收盘
    open_price: float  # 今日开盘价
    auction_pct: float  # 竞价涨幅 %
    auction_volume: int  # 竞价成交量（手）
    volume_ratio: float  # 竞价量比
    strength: str  # 竞价强弱: strong/weak/neutral
    signal: str  # 信号: 抢筹/出货/中性
    notes: str = ""


@dataclass
class AuctionSnapshot:
    """竞价结果快照"""

    timestamp: datetime
    total_stocks_analyzed: int
    strong_auction_count: int  # 强者恒强
    weak_auction_count: int  # 弱势低开
    results: list[AuctionStockResult]
    market_bias: str  # 市场偏向: bullish/bearish/neutral


class AuctionAnalyzer:
    """集合竞价分析器"""

    def __init__(self):
        self.fetcher = DataFetcher()

    def analyze_auction(self, watchlist: list[str], verbose: bool = False) -> AuctionSnapshot:
        """对关注列表进行竞价分析

        Args:
            watchlist: 关注的股票代码列表
            verbose: 是否输出详细日志

        Returns:
            AuctionSnapshot: 竞价分析快照
        """
        results = []
        datetime.now().strftime("%Y%m%d")

        for code in watchlist:
            try:
                result = self._analyze_one(code)
                if result:
                    if verbose:
                        logger.info(
                            f"  {code} {result.name}: {result.auction_pct:+.2f}% [{result.strength}] {result.signal}"
                        )
                    results.append(result)
            except Exception as e:
                logger.debug(f"分析 {code} 竞价异常: {e}")

        strong = sum(1 for r in results if r.strength == "strong")
        weak = sum(1 for r in results if r.strength == "weak")

        # 市场偏向
        if strong > weak * 1.5:
            bias = "bullish"
        elif weak > strong * 1.5:
            bias = "bearish"
        else:
            bias = "neutral"

        return AuctionSnapshot(
            timestamp=datetime.now(),
            total_stocks_analyzed=len(watchlist),
            strong_auction_count=strong,
            weak_auction_count=weak,
            results=results,
            market_bias=bias,
        )

    def _analyze_one(self, code: str) -> AuctionStockResult | None:
        """分析单只股票的竞价表现

        依赖 AKShare 获取前一日数据和开盘价
        """
        # 获取近期日K线（含昨日收盘和今日开盘）
        try:
            df = self.fetcher.fetch_daily_kline(code, start_date=(datetime.now().replace(day=1).strftime("%Y%m%d")))
            if df is None or df.empty or len(df) < 2:
                return None

            # 列名映射
            cols = df.columns.tolist()
            next((c for c in cols if "日期" in c or "date" in c.lower()), None)
            open_col = next((c for c in cols if "开盘" in c or "open" in c.lower()), None)
            close_col = next((c for c in cols if "收盘" in c or "close" in c.lower()), None)
            vol_col = next((c for c in cols if "成交量" in c or "volume" in c.lower()), None)

            if not all([open_col, close_col, vol_col]):
                return None

            latest = df.iloc[-1]
            prev = df.iloc[-2]

            prev_close = float(prev[close_col])
            open_price = float(latest[open_col])
            today_vol = float(latest[vol_col])
            # 取前5日均量
            avg_vol = float(df[vol_col].iloc[-6:-1].mean()) if len(df) >= 6 else today_vol

            if prev_close <= 0:
                return None

            auction_pct = (open_price / prev_close - 1) * 100
            vol_ratio = today_vol / avg_vol if avg_vol > 0 else 1.0

            # 竞价强势判断
            if auction_pct > 3 and vol_ratio > AUCTION_VOLUME_RATIO_MIN:
                strength = "strong"
                signal = "抢筹信号"
            elif auction_pct < -3 and vol_ratio > AUCTION_VOLUME_RATIO_MIN:
                strength = "weak"
                signal = "出货信号"
            elif abs(auction_pct) < AUCTION_PRICE_CHANGE_ALERT * 100:
                strength = "neutral"
                signal = "平开中性"
            elif auction_pct > 0:
                strength = "weak"
                signal = "温和高开"
            else:
                strength = "neutral"
                signal = "低开观望"

            name = self._get_stock_name(code)

            return AuctionStockResult(
                code=code,
                name=name,
                prev_close=prev_close,
                open_price=open_price,
                auction_pct=round(auction_pct, 2),
                auction_volume=int(today_vol),
                volume_ratio=round(vol_ratio, 2),
                strength=strength,
                signal=signal,
            )
        except Exception as e:
            logger.debug(f"竞价分析 {code}: {e}")
            return None

    def _get_stock_name(self, code: str) -> str:
        """获取股票名称"""
        try:
            df = self.fetcher.fetch_stock_list()
            row = df[df["股票代码"].astype(str) == str(code)]
            if not row.empty:
                return str(row.iloc[0]["股票简称"])
        except Exception:
            pass
        return code

    def filter_strong_auction(self, snapshot: AuctionSnapshot, min_pct: float = 2.0) -> list[AuctionStockResult]:
        """筛出竞价强势的个股"""
        return [r for r in snapshot.results if r.strength == "strong" and r.auction_pct >= min_pct]

    def make_open_trade_decision(
        self, snapshot: AuctionSnapshot, watch_codes: dict[str, list[str]]
    ) -> dict[str, list[dict[str, Any]]]:
        """根据竞价结果制定开盘交易决策

        Args:
            snapshot: 竞价分析快照
            watch_codes: {策略名: [代码列表]}

        Returns:
            {策略名: [决策dict]}
        """
        decisions: dict[str, list[dict]] = {}

        # 建立竞价结果索引
        auction_map = {r.code: r for r in snapshot.results}

        for strategy, codes in watch_codes.items():
            strategy_decisions = []
            for code in codes:
                auc = auction_map.get(code)
                if not auc:
                    continue

                if auc.strength == "strong" and auc.auction_pct >= 3.0:
                    strategy_decisions.append(
                        {
                            "code": code,
                            "name": auc.name,
                            "action": "buy_at_open",
                            "reason": f"竞价强势 {auc.auction_pct:+.2f}% 量比{auc.volume_ratio}",
                            "price": auc.open_price,
                            "confidence": "high" if auc.volume_ratio > 3 else "medium",
                        }
                    )
                elif auc.strength == "strong" and auc.auction_pct >= 2.0:
                    strategy_decisions.append(
                        {
                            "code": code,
                            "name": auc.name,
                            "action": "watch",
                            "reason": f"竞价偏强 {auc.auction_pct:+.2f}%，开盘跟踪",
                            "price": auc.open_price,
                            "confidence": "low",
                        }
                    )
                elif auc.strength == "weak":
                    strategy_decisions.append(
                        {
                            "code": code,
                            "name": auc.name,
                            "action": "sell_at_open" if auc.auction_pct < -3 else "caution",
                            "reason": f"竞价弱势 {auc.auction_pct:+.2f}%",
                            "price": auc.open_price,
                            "confidence": "high" if auc.auction_pct < -5 else "medium",
                        }
                    )

            if strategy_decisions:
                decisions[strategy] = strategy_decisions

        return decisions
