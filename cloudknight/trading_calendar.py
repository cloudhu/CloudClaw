"""
交易日历与交易时间阶段管理

A股交易时间周期:
  - 08:30-09:15  盘前准备 (PreMarket)
  - 09:15-09:25  集合竞价 (Auction)
  - 09:26-09:29  竞价结果 (AuctionResult)
  - 09:30-11:30  早盘交易 (Morning)
  - 11:30-13:00  午间休市 (Lunch)
  - 13:00-15:00  午盘交易 (Afternoon)
  - 15:00-次日   盘后选股 (PostMarket)
"""

import logging
from datetime import date, datetime, time, timedelta
from enum import Enum

import pandas as pd

try:
    import akshare as ak
except ImportError:
    ak = None

logger = logging.getLogger(__name__)


class TradingPhase(Enum):
    """交易时间段枚举"""

    CLOSED = "closed"  # 非交易日 / 休市
    PRE_MARKET = "pre_market"  # 盘前 08:30-09:15
    AUCTION = "auction"  # 集合竞价 09:15-09:25
    AUCTION_RESULT = "auction_result"  # 竞价结果 09:26-09:29
    MORNING = "morning"  # 早盘 09:30-11:30
    LUNCH = "lunch"  # 午休 11:30-13:00
    AFTERNOON = "afternoon"  # 午盘 13:00-15:00
    POST_MARKET = "post_market"  # 盘后 15:00-次日

    def is_trading_phase(self) -> bool:
        """是否处于可交易时间段"""
        return self in (
            TradingPhase.AUCTION_RESULT,
            TradingPhase.MORNING,
            TradingPhase.AFTERNOON,
        )

    def is_active_phase(self) -> bool:
        """是否处于活跃监控时间段"""
        return self in (
            TradingPhase.PRE_MARKET,
            TradingPhase.AUCTION,
            TradingPhase.AUCTION_RESULT,
            TradingPhase.MORNING,
            TradingPhase.LUNCH,
            TradingPhase.AFTERNOON,
            TradingPhase.POST_MARKET,
        )


class TradingCalendar:
    """A股交易日历管理器"""

    def __init__(self):
        self._trading_dates: list[date] = []
        self._cache_year: int | None = None

    def is_trading_day(self, dt: datetime | None = None) -> bool:
        """判断是否为交易日

        Args:
            dt: 日期时间，默认当前时间
        """
        if dt is None:
            dt = datetime.now()

        # 周末一定不是交易日
        if dt.weekday() >= 5:
            return False

        d = dt.date()
        trading_dates = self._get_trading_dates(d.year)

        # 直接查表
        if d in trading_dates:
            return True

        # 如果在缓存中未找到，尝试联网获取更多数据
        if ak is not None:
            try:
                df = ak.tool_trade_date_hist_sina()
                if df is not None and not df.empty:
                    dates_col = df.columns[0]
                    all_dates = {d.date() if hasattr(d, "date") else d for d in pd.to_datetime(df[dates_col]).tolist()}
                    self._trading_dates = sorted(all_dates)
                    return d in all_dates
            except Exception as e:
                logger.debug(f"获取交易日历异常: {e}")

        return False

    def _get_trading_dates(self, year: int) -> list[date]:
        """获取指定年份的交易日列表（缓存）"""
        if self._cache_year == year and self._trading_dates:
            return self._trading_dates

        dates = []
        if ak is not None:
            try:
                import pandas as pd

                df = ak.tool_trade_date_hist_sina()
                if df is not None and not df.empty:
                    dates_col = df.columns[0]
                    all_dates = sorted(
                        d.date() if hasattr(d, "date") else d for d in pd.to_datetime(df[dates_col]).tolist()
                    )
                    dates = [d for d in all_dates if d.year == year]
                    self._trading_dates = sorted(all_dates)
                    self._cache_year = year
            except Exception as e:
                logger.warning(f"获取交易日历失败: {e}")

        # 降级：排除周末
        if not dates:
            d = date(year, 1, 1)
            end = date(year, 12, 31)
            while d <= end:
                if d.weekday() < 5:
                    dates.append(d)
                d += timedelta(days=1)

        return dates

    def get_next_trading_day(self, dt: datetime | None = None) -> date:
        """获取下一个交易日"""
        if dt is None:
            dt = datetime.now()
        d = dt.date() + timedelta(days=1)
        while not self.is_trading_day(datetime.combine(d, time.min)):
            d += timedelta(days=1)
        return d

    def get_previous_trading_day(self, dt: datetime | None = None) -> date:
        """获取上一个交易日"""
        if dt is None:
            dt = datetime.now()
        d = dt.date() - timedelta(days=1)
        while not self.is_trading_day(datetime.combine(d, time.min)):
            d -= timedelta(days=1)
        return d

    def get_current_phase(self, dt: datetime | None = None) -> TradingPhase:
        """获取当前所处的交易时间段"""
        if dt is None:
            dt = datetime.now()

        if not self.is_trading_day(dt):
            return TradingPhase.CLOSED

        now = dt.time()

        if time(8, 30) <= now < time(9, 15):
            return TradingPhase.PRE_MARKET
        elif time(9, 15) <= now < time(9, 26):
            return TradingPhase.AUCTION
        elif time(9, 26) <= now < time(9, 30):
            return TradingPhase.AUCTION_RESULT
        elif time(9, 30) <= now < time(11, 30):
            return TradingPhase.MORNING
        elif time(11, 30) <= now < time(13, 0):
            return TradingPhase.LUNCH
        elif time(13, 0) <= now < time(15, 0):
            return TradingPhase.AFTERNOON
        elif now >= time(15, 0):
            return TradingPhase.POST_MARKET

        # 00:00-08:29，交易日但交易所未开盘，返回休市
        return TradingPhase.CLOSED

    def time_until_next_phase(self, dt: datetime | None = None) -> tuple[TradingPhase, int] | None:
        """计算距离下一阶段还有多少秒

        Returns:
            (下一阶段, 秒数)，如果已收盘则返回 None
        """
        if dt is None:
            dt = datetime.now()

        current_phase = self.get_current_phase(dt)
        if current_phase == TradingPhase.CLOSED:
            return None

        today = dt.date()
        phase_boundaries = [
            (TradingPhase.PRE_MARKET, time(8, 30)),
            (TradingPhase.AUCTION, time(9, 15)),
            (TradingPhase.AUCTION_RESULT, time(9, 26)),
            (TradingPhase.MORNING, time(9, 30)),
            (TradingPhase.LUNCH, time(11, 30)),
            (TradingPhase.AFTERNOON, time(13, 0)),
            (TradingPhase.POST_MARKET, time(15, 0)),
        ]

        for phase, t in phase_boundaries:
            target = datetime.combine(today, t)
            if target > dt:
                seconds = int((target - dt).total_seconds())
                if seconds > 0:
                    return phase, seconds

        # 已收盘
        next_day = self.get_next_trading_day(dt)
        if next_day == today:
            return None
        target = datetime.combine(next_day, time(8, 30))
        seconds = int((target - dt).total_seconds())
        return TradingPhase.PRE_MARKET, seconds

    def is_within_window(self, start_time: time, end_time: time, dt: datetime | None = None) -> bool:
        """判断当前是否在指定时间窗口内"""
        if dt is None:
            dt = datetime.now()
        now = dt.time()
        return start_time <= now < end_time


# 全局单例
trading_calendar = TradingCalendar()


def get_phase_label(phase: TradingPhase) -> str:
    """获取阶段中文标签"""
    labels = {
        TradingPhase.CLOSED: "休市",
        TradingPhase.PRE_MARKET: "盘前准备",
        TradingPhase.AUCTION: "集合竞价",
        TradingPhase.AUCTION_RESULT: "竞价结果分析",
        TradingPhase.MORNING: "早盘交易",
        TradingPhase.LUNCH: "午间休市",
        TradingPhase.AFTERNOON: "午盘交易",
        TradingPhase.POST_MARKET: "盘后选股",
    }
    return labels.get(phase, "未知")
