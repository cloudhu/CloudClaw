"""
股票池系统 - 每种策略独立筛选、评分、排序

核心架构：
  StockPoolItem        - 池中单只股票（含评分明细）
  StrategyStockPool    - 单策略股票池（筛选/评分/排序/持久化）
  PoolManager          - 全局池管理器（统一调度）

四种策略的评分器：
  DragonHeadScorer     - 龙头战法：涨停连板 + 换手 + 量比 + 封板 + 资金
  SparrowScorer        - 麻雀战法：均线多头 + KDJ金叉 + 回调 + RSI
  TurtleScorer         - 海龟战法：通道突破 + ATR + 趋势 + 流动性
  ValueInvestScorer    - 价值投资：PE/PB分位 + ROE + 股息率 + 负债率
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from .config import DATA_DIR, STRATEGIES
from .indicators import calc_atr, calc_breakout, calc_kdj, calc_ma, calc_rsi

logger = logging.getLogger(__name__)

POOL_DIR = os.path.join(DATA_DIR, "pools")
os.makedirs(POOL_DIR, exist_ok=True)

MAX_POOL_SIZE = 30  # 每种策略池最多保留 30 只
SCREEN_SAMPLE_SIZE = 200  # 一次筛选最多遍历 200 只

# 层级定义
TIER_FOCUS = "focus"  # 精选层：评分 ≥ 80，可直接候选开仓
TIER_WATCH = "watch"  # 观察层：评分 60-80，持续跟踪等待信号
TIER_BROAD = "broad"  # 备选层：评分 40-60，潜力储备
TIER_ELIMINATED = "eliminated"  # 淘汰层：已剔除

TIER_THRESHOLDS = {
    TIER_FOCUS: 80,
    TIER_WATCH: 60,
    TIER_BROAD: 40,
}  # 评分 ≥ threshold 才有资格进入该层

TIER_LABELS = {
    TIER_FOCUS: "精选",
    TIER_WATCH: "观察",
    TIER_BROAD: "备选",
    TIER_ELIMINATED: "淘汰",
}

# 淘汰条件阈值
ELIMINATE_DRAWDOWN = -8.0  # 累计跌幅超过 8% 淘汰
ELIMINATE_SCORE_FLOOR = 30  # 评分低于 30 分淘汰
ELIMINATE_DAYS_STALE = 20  # 入池超过 N 天仍未晋级则淘汰

# 维护频率（按策略）
MAINTENANCE_INTERVALS = {
    "dragon_head": 1,  # 龙头：每日
    "sparrow": 2,  # 麻雀：每2天
    "turtle": 5,  # 海龟：每周(5个交易日)
    "value_invest": 10,  # 价值：每2周
    "bollinger": 3,  # 布林带：每3天
    "grid": 3,  # 网格：每3天
    "ma_cross": 3,  # 均线：每3天
    "volume_breakout": 2,  # 量价：每2天
    "trend_accel": 3,  # 趋势加速：每3天
    "high_growth": 10,  # 高增长：每2周（财报驱动）
}


# ─── 交易计划 ──────────────────────────────────────────


@dataclass
class TradePlan:
    """策略股票池中个股的交易计划"""

    strategy_key: str
    strategy_name: str
    code: str
    name: str
    entry_price: float  # 建议入场价
    entry_type: str  # 入场方式: 现价/限价/突破
    stop_loss: float  # 止损价
    stop_loss_pct: float  # 止损幅度 %
    take_profit_1: float  # 第一止盈价
    take_profit_1_pct: float  # 第一止盈幅度 %
    take_profit_2: float  # 第二止盈价
    take_profit_2_pct: float  # 第二止盈幅度 %
    position_pct: float  # 建议仓位占比（总资金的 %）
    risk_reward_ratio: float  # 风险收益比
    hold_days: int  # 建议持仓天数
    reasons: list[str] = field(default_factory=list)  # 交易理由
    warnings: list[str] = field(default_factory=list)  # 风险提示
    created_at: str = ""


# ─── 数据结构 ─────────────────────────────────────────────


@dataclass
class StockPoolItem:
    """股票池中的单只股票"""

    code: str
    name: str
    score: float  # 综合评分 0-100
    components: dict[str, float] = field(default_factory=dict)  # 各维度得分
    factors: dict[str, float] = field(default_factory=dict)  # 各维度原始因子数据
    screened_at: str = ""  # 入池日期
    status: str = "active"  # active | traded | removed
    entry_price: float = 0.0  # 入池时的收盘价
    max_price: float = 0.0  # 入池后最高收盘价
    tier: str = ""  # 层级: focus/watch/broad/eliminated
    evaluated_at: str = ""  # 上次评估日期

    def to_dict(self) -> dict:
        d = {
            "code": self.code,
            "name": self.name,
            "score": round(self.score, 1),
            "components": {k: round(v, 1) for k, v in self.components.items()},
            "screened_at": self.screened_at,
            "status": self.status,
        }
        if self.factors:
            d["factors"] = {
                k: round(v, 2) if isinstance(v, float) else v
                for k, v in self.factors.items()
            }
        if self.tier:
            d["tier"] = self.tier
        if self.evaluated_at:
            d["evaluated_at"] = self.evaluated_at
        if self.entry_price > 0:
            d["entry_price"] = round(self.entry_price, 2)
            d["max_price"] = round(max(self.max_price, self.entry_price), 2)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "StockPoolItem":
        return cls(
            code=d["code"],
            name=d.get("name", ""),
            score=d.get("score", 0),
            components=d.get("components", {}),
            factors=d.get("factors", {}),
            screened_at=d.get("screened_at", ""),
            status=d.get("status", "active"),
            entry_price=d.get("entry_price", 0.0),
            max_price=d.get("max_price", 0.0),
            tier=d.get("tier", ""),
            evaluated_at=d.get("evaluated_at", ""),
        )

    def compute_tier(self, cumulative_gain: float = 0) -> str:
        """根据评分和累计涨幅计算当前应属层级"""
        s = self.score

        # 硬淘汰条件
        if self.status != "active":
            return TIER_ELIMINATED
        if s < ELIMINATE_SCORE_FLOOR:
            return TIER_ELIMINATED
        if cumulative_gain <= ELIMINATE_DRAWDOWN:
            return TIER_ELIMINATED

        # 评分阈值判定
        if s >= TIER_THRESHOLDS[TIER_FOCUS]:
            return TIER_FOCUS
        elif s >= TIER_THRESHOLDS[TIER_WATCH]:
            return TIER_WATCH
        elif s >= TIER_THRESHOLDS[TIER_BROAD]:
            return TIER_BROAD
        else:
            return TIER_ELIMINATED


# ─── 评分器基类 ──────────────────────────────────────────


class BaseScorer:
    """评分器基类"""

    name: str = "base"

    def score(self, code: str, name: str, df: pd.DataFrame, extra: dict | None = None) -> StockPoolItem:
        """返回 StockPoolItem，含综合分和各维度明细"""
        raise NotImplementedError

    def _safe_val(self, row, *keys, default=0.0):
        for k in keys:
            v = row.get(k)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                return float(v)
        return float(default)

    def _safe_iloc(self, df, col, idx, default=0.0):
        try:
            v = df[col].iloc[idx]
            if pd.notna(v):
                return float(v)
        except (IndexError, KeyError):
            pass
        return float(default)

    def _linear_score(self, val: float, low: float, high: float, floor: float = 0, ceil: float = 100) -> float:
        """线性映射 val ∈ [low, high] → [0, 1]，再拉伸到 [floor, ceil]"""
        if high <= low:
            return floor
        ratio = max(0, min(1, (val - low) / (high - low)))
        return round(floor + ratio * (ceil - floor), 1)

    def _bell_score(self, val: float, low: float, mid: float, high: float, max_pts: float = 100) -> float:
        """钟形得分：在 mid 附近最高，两端递减"""
        if val < low or val > high:
            return 0
        if val <= mid:
            return round(self._linear_score(val, low, mid, 0, max_pts), 1)
        else:
            return round(self._linear_score(val, mid, high, max_pts, 0), 1)


# ─── 龙头战法评分器 ──────────────────────────────────────


class DragonHeadScorer(BaseScorer):
    """龙头战法评分：涨停连板 + 换手 + 量比 + 封板强度 + 资金流向"""

    name = "dragon_head"

    def score(self, code: str, name: str, df: pd.DataFrame, extra: dict | None = None) -> StockPoolItem:
        if df.empty or len(df) < 5:
            return StockPoolItem(code=code, name=name, score=0, components={})

        latest = df.iloc[-1]
        close = self._safe_val(latest, "close", "收盘")
        self._safe_val(latest, "open", "开盘")
        high = self._safe_val(latest, "high", "最高")
        low = self._safe_val(latest, "low", "最低")
        volume = self._safe_val(latest, "volume", "成交量")
        pct = self._safe_val(latest, "pct_change", "涨跌幅")
        turnover = self._safe_val(latest, "turnover", "换手率")

        # 1. 连板得分 (25分)
        limit_days = self._count_limit_days(df)
        lu_score = self._linear_score(limit_days, 1, 5, 0, 25)

        # 2. 量比得分 (20分)：当日量 / 5日均量
        avg_vol = df["volume"].tail(6).head(5).mean() if "volume" in df.columns else 1
        vol_ratio = volume / avg_vol if avg_vol > 0 else 1
        vr_score = self._bell_score(vol_ratio, 0.5, 2.0, 5.0, 20)

        # 3. 换手率得分 (20分)：5%~30%区间内最佳
        if turnover:
            to_score = self._bell_score(turnover, 3, 15, 35, 20)
        else:
            to_score = 10

        # 4. 封板强度 (20分)：振幅越小越好
        amplitude = (high - low) / close * 100 if close > 0 else 100
        amp_score = self._linear_score(amplitude, 0.5, 12, 20, 0)  # 振幅越小分越高

        # 5. 趋势加速 (15分)：量缩反映筹码锁定
        if limit_days >= 2 and vol_ratio < 1.0:
            accel_score = 15
        elif limit_days >= 2 and vol_ratio < 1.5:
            accel_score = 10
        elif pct >= 9.5:
            accel_score = 8
        else:
            accel_score = 0

        comp = {
            "连板天数": round(lu_score, 1),
            "量比": round(vr_score, 1),
            "换手率": round(to_score, 1),
            "封板强度": round(amp_score, 1),
            "趋势加速": round(accel_score, 1),
        }
        factors = {
            "连板天数": limit_days,
            "量比": round(vol_ratio, 2),
            "换手率%": round(turnover, 2) if turnover else 0,
            "振幅%": round(amplitude, 2),
            "趋势强化": 1 if (limit_days >= 2 and vol_ratio < 1.0) else (0.5 if (limit_days >= 2 and vol_ratio < 1.5) else 0),
        }
        total = sum(comp.values())
        return StockPoolItem(
            code=code,
            name=name,
            score=round(total, 1),
            components=comp,
            factors=factors,
            screened_at=datetime.now().strftime("%Y%m%d"),
            entry_price=round(close, 2),
            max_price=round(close, 2),
        )

    def _count_limit_days(self, df: pd.DataFrame) -> int:
        days = 0
        for i in range(len(df) - 1, -1, -1):
            pct = df.iloc[i].get("pct_change", df.iloc[i].get("涨跌幅", 0))
            if pct and pct >= 9.5:
                days += 1
            else:
                break
        return days


# ─── 麻雀战法评分器 ──────────────────────────────────────


class SparrowScorer(BaseScorer):
    """麻雀战法评分：均线多头 + KDJ金叉 + 回调支撑 + RSI"""

    name = "sparrow"

    def score(self, code: str, name: str, df: pd.DataFrame, extra: dict | None = None) -> StockPoolItem:
        if df.empty or len(df) < 60:
            return StockPoolItem(code=code, name=name, score=0, components={})

        df = calc_ma(df, [5, 10, 20])
        df = calc_kdj(df)
        df = calc_rsi(df, 14)

        latest = df.iloc[-1]
        prev = df.iloc[-2]
        close = latest["close"]
        ma5 = latest.get("MA5", 0)
        ma10 = latest.get("MA10", 0)
        ma20 = latest.get("MA20", 0)
        rsi = latest.get("RSI", 50)
        k, d, _j = latest["K"], latest["D"], latest["J"]

        # 1. 均线多头排列 (25分)
        if ma5 > ma10 > ma20:
            spread = (ma5 - ma20) / ma20 * 100 if ma20 > 0 else 0
            ma_score = self._linear_score(spread, 0, 5, 10, 25)
        elif ma5 > ma10:
            ma_score = 8
        else:
            ma_score = 0

        # 2. KDJ 金叉质量 (25分)
        prev_kdj = (prev["K"] <= prev["D"]) and (latest["K"] > latest["D"])
        if prev_kdj:  # 刚金叉
            if k < 30:
                kdj_score = 25  # 低位金叉最佳
            elif k < 50:
                kdj_score = 20
            else:
                kdj_score = 12
        elif k > d:
            kdj_score = 8  # 已处于多头但未金叉
        else:
            kdj_score = 0

        # 3. 回调深度 (20分)：股价在 MA10~MA20 附近最佳
        if ma20 > 0:
            dist_to_ma20 = (close - ma20) / ma20 * 100
            cb_score = self._bell_score(dist_to_ma20, -1, 0.5, 3, 20)
        else:
            cb_score = 10

        # 4. RSI 位置 (15分)：35-55 区间最佳
        rsi_score = self._bell_score(rsi, 25, 45, 70, 15)

        # 5. 量价配合 (15分)：近3日量价配合度
        vol_price_score = self._volume_price_score(df, 3)

        comp = {
            "均线多头": round(ma_score, 1),
            "KDJ金叉": round(kdj_score, 1),
            "回调深度": round(cb_score, 1),
            "RSI": round(rsi_score, 1),
            "量价配合": round(vol_price_score, 1),
        }
        factors = {
            "均线价差%": round((ma5 - ma20) / ma20 * 100, 2) if ma20 > 0 else 0,
            "K": round(k, 1),
            "D": round(d, 1),
            "回调%": round((close - ma20) / ma20 * 100, 2) if ma20 > 0 else 0,
            "RSI": round(rsi, 1),
            "量价配合": round(vol_price_score, 1),
        }
        total = sum(comp.values())
        return StockPoolItem(
            code=code,
            name=name,
            score=round(total, 1),
            components=comp,
            factors=factors,
            screened_at=datetime.now().strftime("%Y%m%d"),
            entry_price=round(close, 2),
            max_price=round(close, 2),
        )

    def _volume_price_score(self, df: pd.DataFrame, window: int = 3) -> float:
        """量价配合度得分：近N日量增价增得高分"""
        tail = df.tail(window)
        if len(tail) < 2:
            return 7.5
        try:
            pct_changes = tail["close"].pct_change().dropna()
            vol_changes = tail["volume"].pct_change().dropna()
            if len(pct_changes) == 0:
                return 7.5
            align = 0
            for pc, vc in zip(pct_changes, vol_changes, strict=False):
                if pc > 0 and vc > 0:
                    align += 1  # 价量同向：好
                elif pc < 0 and vc < 0:
                    align += 0.5  # 缩量下跌：中性偏多
                elif pc > 0 and vc < 0:
                    align += 0.3  # 缩量上涨：一般
                # 价跌量增：不加分
            return round(align / len(pct_changes) * 15, 1)
        except Exception:
            return 7.5


# ─── 海龟战法评分器 ──────────────────────────────────────


class TurtleScorer(BaseScorer):
    """海龟战法评分：通道突破 + ATR波动率 + 趋势 + 流动性"""

    name = "turtle"

    def score(self, code: str, name: str, df: pd.DataFrame, extra: dict | None = None) -> StockPoolItem:
        if df.empty or len(df) < 100:
            return StockPoolItem(code=code, name=name, score=0, components={})

        latest = df.iloc[-1]
        close = latest["close"]
        entry_period = 20

        # ATR
        atr_series = calc_atr(df, 20)
        current_atr = atr_series.iloc[-1]

        # 通道
        upper, _lower = calc_breakout(df, entry_period)
        entry_upper = upper.iloc[-1]

        # MA200
        df["MA200"] = df["close"].rolling(200).mean()
        ma200 = df["MA200"].iloc[-1]

        # 1. 突破强度 (30分)：收盘价突破通道上轨的程度
        if close >= entry_upper:
            breakout_pct = (close - entry_upper) / entry_upper * 100 if entry_upper > 0 else 0
            bs_score = self._linear_score(breakout_pct, 0, 5, 15, 30)
        elif close >= entry_upper * 0.98:
            bs_score = 10  # 接近突破
        else:
            bs_score = 0

        # 2. 趋势强度 (25分)：股价在 MA200 上方
        if pd.notna(ma200) and ma200 > 0:
            trend_pct = (close - ma200) / ma200 * 100
            tr_score = self._linear_score(trend_pct, 0, 30, 5, 25)
        else:
            tr_score = 10

        # 3. ATR 波动率 (15分)：适中的波动率
        if current_atr > 0 and close > 0:
            atr_pct = current_atr / close * 100
            atr_score = self._bell_score(atr_pct, 1, 3, 8, 15)
        else:
            atr_score = 7

        # 4. 流动性 (15分)：近20日均成交额
        amount = latest.get("amount", latest.get("成交额", 0))
        if amount and amount > 0:
            liq_score = self._linear_score(amount, 5e7, 5e8, 3, 15)
        else:
            vol = latest.get("volume", latest.get("成交量", 0))
            liq_score = self._linear_score(vol, 1e6, 1e7, 3, 15) if vol > 0 else 5

        # 5. 加仓空间 (15分)：距离上次突破位置
        if current_atr > 0 and close > entry_upper:
            room = (close - entry_upper) / current_atr
            room_score = self._linear_score(room, 0, 1.5, 3, 15)
        else:
            room_score = 0

        comp = {
            "突破强度": round(bs_score, 1),
            "趋势强度": round(tr_score, 1),
            "波动率": round(atr_score, 1),
            "流动性": round(liq_score, 1),
            "加仓空间": round(room_score, 1),
        }
        # 计算原始取值
        pct_close = close
        breakthrough_pct = (pct_close - entry_upper) / entry_upper * 100 if entry_upper > 0 else 0
        trend_pct = (pct_close - ma200) / ma200 * 100 if (pd.notna(ma200) and ma200 > 0) else 0
        atr_pct = current_atr / pct_close * 100 if current_atr > 0 and pct_close > 0 else 0
        room = (pct_close - entry_upper) / current_atr if (current_atr > 0 and pct_close > entry_upper) else 0
        amount_val = latest.get("amount", latest.get("成交额", 0))
        factors = {
            "突破%": round(breakthrough_pct, 2),
            "趋势%": round(trend_pct, 2),
            "ATR%": round(atr_pct, 2),
            "成交额万": round(amount_val / 1e4, 0) if amount_val else 0,
            "AT倍数": round(room, 1),
        }
        total = sum(comp.values())
        return StockPoolItem(
            code=code,
            name=name,
            score=round(total, 1),
            components=comp,
            factors=factors,
            screened_at=datetime.now().strftime("%Y%m%d"),
            entry_price=round(close, 2),
            max_price=round(close, 2),
        )


# ─── 价值投资评分器 ──────────────────────────────────────


class ValueInvestScorer(BaseScorer):
    """价值投资评分：PE/PB分位 + ROE + 股息率 + 负债率"""

    name = "value_invest"

    def score(self, code: str, name: str, df: pd.DataFrame, extra: dict | None = None) -> StockPoolItem:
        extra = extra or {}
        if df.empty or len(df) < 100:
            return StockPoolItem(code=code, name=name, score=0, components={})

        latest = df.iloc[-1]
        close = latest["close"]

        pe = extra.get("pe", 999)
        pb = extra.get("pb", 999)
        roe = extra.get("roe", 0)
        div_yield = extra.get("dividend_yield", 0)
        market_cap = extra.get("total_mv", extra.get("market_cap", 0))
        debt_ratio = extra.get("debt_ratio", 50)

        # 1. 估值分位 (25分)：基于250日收盘价历史分位
        if len(df) >= 250:
            close_series = df["close"]
            percentile = (close_series < close).sum() / len(close_series) * 100
            val_score = self._linear_score(percentile, 10, 90, 25, 0)  # 分位越低越好
        else:
            val_score = 12.5

        # 2. PE 得分 (15分)
        if pe and pe > 0 and pe < 999:
            pe_score = self._linear_score(pe, 5, 25, 15, 0)
        else:
            pe_score = 5

        # 3. PB 得分 (10分)
        if pb and pb > 0 and pb < 999:
            pb_score = self._linear_score(pb, 0.5, 3, 10, 0)
        else:
            pb_score = 3

        # 4. ROE 得分 (15分)
        if roe and roe > 0:
            roe_score = self._linear_score(roe, 5, 25, 3, 15)
        else:
            roe_score = 5

        # 5. 股息率得分 (15分)
        if div_yield and div_yield > 0:
            dy_score = self._linear_score(div_yield, 0.5, 5, 3, 15)
        else:
            dy_score = 3

        # 6. 市值规模 (10分)
        if market_cap and market_cap > 0:
            mc_score = self._bell_score(market_cap, 5e9, 50e9, 500e9, 10)
        else:
            mc_score = 5

        # 7. 负债率 (10分)：越低越好
        if debt_ratio is not None:
            dr_score = self._linear_score(debt_ratio, 80, 10, 0, 10)  # 注意反向
        else:
            dr_score = 5

        comp = {
            "估值分位": round(val_score, 1),
            "PE": round(pe_score, 1),
            "PB": round(pb_score, 1),
            "ROE": round(roe_score, 1),
            "股息率": round(dy_score, 1),
            "市值规模": round(mc_score, 1),
            "负债率": round(dr_score, 1),
        }
        factors = {
            "价格分位%": round(percentile, 1) if len(df) >= 250 else 0,
            "PE": round(pe, 2) if (pe and pe < 999) else 0,
            "PB": round(pb, 2) if (pb and pb < 999) else 0,
            "ROE%": round(roe, 2) if (roe and roe > 0) else 0,
            "股息率%": round(div_yield, 2) if (div_yield and div_yield > 0) else 0,
            "市值亿": round(market_cap / 1e8, 1) if (market_cap and market_cap > 0) else 0,
            "负债率%": round(debt_ratio, 1) if debt_ratio is not None else 0,
        }
        total = sum(comp.values())
        return StockPoolItem(
            code=code,
            name=name,
            score=round(total, 1),
            components=comp,
            factors=factors,
            screened_at=datetime.now().strftime("%Y%m%d"),
            entry_price=round(close, 2),
            max_price=round(close, 2),
        )


# ─── 布林带回归评分器 ────────────────────────────────────


class BollingerBandScorer(BaseScorer):
    """布林带回归评分：%B位置 + 带宽 + RSI"""

    name = "bollinger"

    def score(self, code: str, name: str, df: pd.DataFrame, extra: dict | None = None) -> StockPoolItem:
        if df.empty or len(df) < 25:
            return StockPoolItem(code=code, name=name, score=0, components={})

        latest = df.iloc[-1]
        close = latest["close"]
        closes = df["close"].values.astype(float)
        volumes = df["volume"].values.astype(float) if "volume" in df.columns else np.array([0])

        # 布林带计算
        period = 20
        mid = np.mean(closes[-period:]) if len(closes) >= period else close
        std = np.std(closes[-period:]) if len(closes) >= period else 0
        upper = mid + 2 * std
        lower = mid - 2 * std
        band_range = upper - lower
        percent_b = (close - lower) / band_range if band_range > 0 else 0.5
        bandwidth = band_range / mid * 100 if mid > 0 else 0

        # RSI
        rsi = self._safe_val(latest, "RSI", "rsi", default=50.0)

        # 1. %B位置 (35分): 越低越好
        bb_score = self._linear_score(percent_b, 1.0, 0.0, 0, 35) if percent_b <= 0.5 else self._linear_score(percent_b, 1.0, 0.5, 35, 0)

        # 2. 带宽 (25分): 适中
        bw_score = self._bell_score(bandwidth, 3, 10, 30, 25)

        # 3. RSI超卖 (20分): 越低越好
        rsi_score = self._linear_score(rsi, 50, 25, 0, 20)

        # 4. 量能 (20分)
        vol = volumes[-1] if len(volumes) > 0 else 0
        avg_vol = np.mean(volumes[-6:-1]) if len(volumes) >= 6 else vol
        vol_ratio = vol / avg_vol if avg_vol > 0 else 1
        vol_score = self._bell_score(vol_ratio, 0.5, 1.5, 3.0, 20)

        comp = {
            "%B低位": round(bb_score, 1),
            "带宽适中": round(bw_score, 1),
            "RSI超卖": round(rsi_score, 1),
            "量能": round(vol_score, 1),
        }
        factors = {
            "%B": round(percent_b, 2),
            "带宽%": round(bandwidth, 1),
            "RSI": round(rsi, 1),
            "量比": round(vol_ratio, 2),
        }
        total = sum(comp.values())
        return StockPoolItem(
            code=code, name=name, score=round(total, 1), components=comp,
            factors=factors,
            screened_at=datetime.now().strftime("%Y%m%d"),
            entry_price=round(close, 2), max_price=round(close, 2),
        )


# ─── 网格交易评分器 ──────────────────────────────────────


class GridScorer(BaseScorer):
    """网格交易评分：价格位置 + 波动率 + RSI"""

    name = "grid"

    def score(self, code: str, name: str, df: pd.DataFrame, extra: dict | None = None) -> StockPoolItem:
        if df.empty or len(df) < 60:
            return StockPoolItem(code=code, name=name, score=0, components={})

        df = calc_ma(df, [60])
        latest = df.iloc[-1]
        close = latest["close"]
        ma60 = latest.get("MA60", close)
        rsi = self._safe_val(latest, "RSI", "rsi", default=50.0)

        # ATR
        closes = df["close"].values.astype(float)
        highs = df["high"].values.astype(float)
        lows = df["low"].values.astype(float)
        atr_series = calc_atr(df, 20)
        atr = float(atr_series.iloc[-1]) if len(atr_series) > 0 else 0

        # 1. MA60偏离 (30分): 偏离适中最佳
        dist_ma60 = (close - ma60) / ma60 * 100 if ma60 > 0 else 0
        grid_score = self._bell_score(abs(dist_ma60), 2, 8, 20, 30)

        # 2. 波动率 (25分): 适度波动
        if close > 0 and atr > 0:
            atr_pct = atr / close * 100
            atr_score = self._bell_score(atr_pct, 0.8, 3, 7, 25)
        else:
            atr_score = 12

        # 3. RSI超卖 (25分): 低吸判断
        rsi_score = self._linear_score(rsi, 50, 25, 0, 25)

        # 4. 成交量 (20分)
        volumes = df["volume"].values.astype(float)
        avg_vol = np.mean(volumes[-20:]) if len(volumes) >= 20 else volumes[-1]
        vol_stable = volumes[-1] / avg_vol if avg_vol > 0 else 1
        stable_score = self._bell_score(vol_stable, 0.5, 1.0, 2.0, 20)

        comp = {
            "MA60偏离": round(grid_score, 1),
            "波动适中": round(atr_score, 1),
            "RSI低位": round(rsi_score, 1),
            "量能稳定": round(stable_score, 1),
        }
        factors = {
            "偏离%": round(dist_ma60, 1),
            "ATR%": round(atr_pct, 2) if close > 0 and atr > 0 else 0,
            "RSI": round(rsi, 1),
            "量比": round(vol_stable, 2),
        }
        total = sum(comp.values())
        return StockPoolItem(
            code=code, name=name, score=round(total, 1), components=comp,
            factors=factors,
            screened_at=datetime.now().strftime("%Y%m%d"),
            entry_price=round(close, 2), max_price=round(close, 2),
        )


# ─── 均线交叉评分器 ──────────────────────────────────────


class MACrossoverScorer(BaseScorer):
    """均线交叉评分：金叉信号 + 趋势 + RSI"""

    name = "ma_cross"

    def score(self, code: str, name: str, df: pd.DataFrame, extra: dict | None = None) -> StockPoolItem:
        if df.empty or len(df) < 60:
            return StockPoolItem(code=code, name=name, score=0, components={})

        df = calc_ma(df, [10, 30, 60])
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        close = latest["close"]
        ma10 = latest.get("MA10", 0)
        ma30 = latest.get("MA30", 0)
        ma60 = latest.get("MA60", 0)
        rsi = self._safe_val(latest, "RSI", "rsi", default=50.0)

        # 1. 金叉信号 (30分)
        prev_ma10 = prev.get("MA10", ma10)
        prev_ma30 = prev.get("MA30", ma30)
        golden = prev_ma10 <= prev_ma30 and ma10 > ma30
        if golden:
            cross_score = 30
        elif ma10 > ma30:
            cross_score = 15
        else:
            cross_score = 0

        # 2. 趋势强度 (25分)
        if ma60 > 0 and close > ma60:
            trend_score = 25
        elif ma60 > 0:
            trend_score = 5
        else:
            trend_score = 10

        # 3. RSI合理 (25分)
        rsi_score = self._bell_score(rsi, 30, 50, 75, 25)

        # 4. 量能确认 (20分)
        volumes = df["volume"].values.astype(float)
        avg_vol = np.mean(volumes[-6:-1]) if len(volumes) >= 6 else volumes[-1]
        vol_ratio = volumes[-1] / avg_vol if avg_vol > 0 else 1
        vol_score = self._bell_score(vol_ratio, 0.5, 1.5, 3.0, 20)

        comp = {
            "金叉信号": round(cross_score, 1),
            "趋势强度": round(trend_score, 1),
            "RSI": round(rsi_score, 1),
            "量能确认": round(vol_score, 1),
        }
        factors = {
            "金叉": 1 if golden else 0,
            "MA60上方": 1 if (ma60 > 0 and close > ma60) else 0,
            "RSI": round(rsi, 1),
            "量比": round(vol_ratio, 2),
        }
        total = sum(comp.values())
        return StockPoolItem(
            code=code, name=name, score=round(total, 1), components=comp,
            factors=factors,
            screened_at=datetime.now().strftime("%Y%m%d"),
            entry_price=round(close, 2), max_price=round(close, 2),
        )


# ─── 量价突破评分器 ──────────────────────────────────────


class VolumeBreakoutScorer(BaseScorer):
    """量价突破评分：突破强度 + 量比 + RSI + MA确认"""

    name = "volume_breakout"

    def score(self, code: str, name: str, df: pd.DataFrame, extra: dict | None = None) -> StockPoolItem:
        if df.empty or len(df) < 30:
            return StockPoolItem(code=code, name=name, score=0, components={})

        df = calc_ma(df, [20])
        latest = df.iloc[-1]
        close = latest["close"]
        high = latest["high"]
        ma20 = latest.get("MA20", close)
        rsi = self._safe_val(latest, "RSI", "rsi", default=50.0)

        # 突破距离
        highs = df["high"].values.astype(float)
        recent_high = float(np.max(highs[-21:-1])) if len(highs) >= 21 else high
        breakout_pct = (close - recent_high) / recent_high * 100 if recent_high > 0 else 0

        # 量比
        volumes = df["volume"].values.astype(float)
        avg_vol = np.mean(volumes[-21:-1]) if len(volumes) >= 21 else volumes[-1]
        vol_ratio = volumes[-1] / avg_vol if avg_vol > 0 else 1

        # 1. 突破强度 (30分)
        break_score = self._linear_score(breakout_pct, -3, 3, 0, 30)

        # 2. 量比 (30分)
        vol_score = self._bell_score(vol_ratio, 0.5, 2.0, 5.0, 30)

        # 3. RSI动量 (20分)
        rsi_score = self._bell_score(rsi, 40, 60, 80, 20)

        # 4. MA20确认 (20分)
        ma_score = 20 if close > ma20 and ma20 > 0 else (10 if ma20 > 0 else 5)

        comp = {
            "突破强度": round(break_score, 1),
            "量比放大": round(vol_score, 1),
            "RSI动量": round(rsi_score, 1),
            "均线站位": round(ma_score, 1),
        }
        factors = {
            "突破%": round(breakout_pct, 2),
            "量比": round(vol_ratio, 2),
            "RSI": round(rsi, 1),
            "MA20上方": 1 if (close > ma20 and ma20 > 0) else 0,
        }
        total = sum(comp.values())
        return StockPoolItem(
            code=code, name=name, score=round(total, 1), components=comp,
            factors=factors,
            screened_at=datetime.now().strftime("%Y%m%d"),
            entry_price=round(close, 2), max_price=round(close, 2),
        )


# ─── 趋势加速评分器 ──────────────────────────────────────


class TrendAccelerationScorer(BaseScorer):
    """趋势加速评分：EMA排列 + 加速度 + 回调位置 + RSI"""

    name = "trend_accel"

    def score(self, code: str, name: str, df: pd.DataFrame, extra: dict | None = None) -> StockPoolItem:
        if df.empty or len(df) < 80:
            return StockPoolItem(code=code, name=name, score=0, components={})

        df = calc_ma(df, [5, 10, 20, 60, 120])
        latest = df.iloc[-1]
        close = latest["close"]
        ma5 = latest.get("MA5", 0)
        ma10 = latest.get("MA10", 0)
        ma20 = latest.get("MA20", 0)
        ma60 = latest.get("MA60", 0)
        ma120 = latest.get("MA120", 0)
        rsi = self._safe_val(latest, "RSI", "rsi", default=50.0)

        # 1. EMA多头排列 (30分): 多周期MA排列
        mas = [(5, ma5), (10, ma10), (20, ma20), (60, ma60)]
        aligned = sum(1 for i in range(len(mas) - 1) if mas[i][1] > 0 and mas[i + 1][1] > 0 and mas[i][1] > mas[i + 1][1])
        align_score = self._linear_score(aligned, 0, 3, 0, 30)

        # 2. 趋势位置 (25分): vs MA120
        if ma120 > 0:
            trend_score = self._linear_score((close - ma120) / ma120 * 100, 0, 30, 5, 25)
        else:
            trend_score = 10

        # 3. 回调深度 (25分): 距MA20
        if ma20 > 0:
            pullback = abs(close / ma20 - 1) * 100
            pb_score = self._bell_score(pullback, 0.5, 3, 10, 25)
        else:
            pb_score = 12

        # 4. RSI位置 (20分)
        rsi_score = self._bell_score(rsi, 35, 50, 70, 20)

        comp = {
            "EMA排列": round(align_score, 1),
            "趋势位置": round(trend_score, 1),
            "回调深度": round(pb_score, 1),
            "RSI": round(rsi_score, 1),
        }
        factors = {
            "排列数": aligned,
            "趋势%": round((close - ma120) / ma120 * 100, 2) if ma120 > 0 else 0,
            "回调%": round(abs(close / ma20 - 1) * 100, 2) if ma20 > 0 else 0,
            "RSI": round(rsi, 1),
        }
        total = sum(comp.values())
        return StockPoolItem(
            code=code, name=name, score=round(total, 1), components=comp,
            factors=factors,
            screened_at=datetime.now().strftime("%Y%m%d"),
            entry_price=round(close, 2), max_price=round(close, 2),
        )


# ─── 高增长 Scorer ──────────────────────────────────────


class HighGrowthScorer(BaseScorer):
    """高增长评分器 - 基于基本面成长因子

    评分维度 (总分 100):
      - 3年EPS增长率 (20分)
      - 3年营收增长率 (15分)
      - PEG (25分)
      - ROE (15分)
      - 净利润增速 (15分)
      - 净资产增速 (10分)
    """

    name = "growth"

    def score(self, code, name, df, extra=None):
        close = float(df["close"].iloc[-1])
        rsi = calc_rsi(df) if "close" in df.columns else 50
        volumes = df["volume"].values.astype(float) if "volume" in df.columns else np.array([])
        avg_vol = np.mean(volumes[-21:-1]) if len(volumes) >= 21 else (volumes[-1] if len(volumes) > 0 else 0)
        vol_ratio = volumes[-1] / avg_vol if avg_vol > 0 else 1

        extra = extra or {}
        eps_g3 = extra.get("eps_growth_3y", 0)
        rev_g3 = extra.get("revenue_growth_3y", 0)
        peg = extra.get("peg", 999)
        roe = extra.get("roe", 0)
        profit_g = extra.get("profit_growth_yoy", 0)
        equity_g = extra.get("equity_growth_yoy", 0)

        # 1. 3年EPS增长率 (20分): 5%~40% 线性
        eps_score = self._linear_score(eps_g3, 5, 40, 0, 20)

        # 2. 3年营收增长率 (15分): 5%~30% 线性
        rev_score = self._linear_score(rev_g3, 5, 30, 0, 15)

        # 3. PEG (25分): 0~0.5=满分, 0.5~1.5=递减, >2=0分
        if peg >= 0 and peg <= 0.5:
            peg_score = 25
        elif peg > 0.5 and peg <= 1.5:
            peg_score = 25 - (peg - 0.5) / 1.0 * 20
        elif peg > 1.5 and peg <= 2.0:
            peg_score = max(0, 5 - (peg - 1.5) / 0.5 * 5)
        else:
            peg_score = 0

        # 4. ROE (15分): 8%~30% 线性
        roe_score = self._linear_score(roe, 8, 30, 0, 15)

        # 5. 净利润增速 (15分): 10%~50% 线性
        profit_score = self._linear_score(profit_g, 10, 50, 0, 15)

        # 6. 净资产增速 (10分): 5%~30% 线性
        equity_score = self._linear_score(equity_g, 5, 30, 0, 10)

        comp = {
            "EPS增长率": round(eps_score, 1),
            "营收增长率": round(rev_score, 1),
            "PEG": round(peg_score, 1),
            "ROE": round(roe_score, 1),
            "净利润增速": round(profit_score, 1),
            "净资产增速": round(equity_score, 1),
        }
        factors = {
            "EPS增长3Y%": round(eps_g3, 1) if eps_g3 else 0,
            "营收增长3Y%": round(rev_g3, 1) if rev_g3 else 0,
            "PEG": round(peg, 2) if peg and peg < 999 else 0,
            "ROE%": round(roe, 2) if roe else 0,
            "利润增速%": round(profit_g, 2) if profit_g else 0,
            "净资产增速%": round(equity_g, 2) if equity_g else 0,
        }
        total = sum(comp.values())
        return StockPoolItem(
            code=code, name=name, score=round(total, 1), components=comp,
            factors=factors,
            screened_at=datetime.now().strftime("%Y%m%d"),
            entry_price=round(close, 2), max_price=round(close, 2),
        )


# ─── 策略股票池 ──────────────────────────────────────────


class StrategyStockPool:
    """单个策略的股票池"""

    def __init__(self, strategy_key: str, strategy_label: str, scorer: BaseScorer):
        self.strategy_key = strategy_key
        self.strategy_label = strategy_label
        self.scorer = scorer
        self.items: dict[str, StockPoolItem] = {}  # code → item
        self.last_screened: str | None = None
        self._file = os.path.join(POOL_DIR, f"{strategy_key}.json")

    def _get_fetcher(self):
        from .data_manager import DataFetcher

        return DataFetcher()

    def screen(
        self, stock_pool: list[str] | None = None, stock_info: pd.DataFrame = None, verbose: bool = False
    ) -> list[StockPoolItem]:
        """
        从股票池中筛选并评分的个股，加入策略池。

        流程：
        1. 获取全市场股票列表（或使用传入的 stock_pool）
        2. 按策略条件逐只取K线、计算评分
        3. 保留评分最高的 MAX_POOL_SIZE 只
        """
        fetcher = self._get_fetcher()

        # 获取候选列表
        if stock_pool is None:
            stock_pool = fetcher.build_stock_pool(filter_st=True, filter_new=True)
        sample = stock_pool[:SCREEN_SAMPLE_SIZE]

        # 股票信息：code → name
        if stock_info is None or stock_info.empty:
            stock_info = fetcher.fetch_stock_list()

        name_map = {}
        if not stock_info.empty:
            name_map = dict(zip(stock_info["股票代码"].astype(str), stock_info["股票简称"].astype(str), strict=False))

        if verbose:
            logger.info(f"  [{self.strategy_label}] 开始筛选，候选 {len(sample)} 只...")

        candidates: list[StockPoolItem] = []
        screened = 0

        for code in sample:
            try:
                code = str(code)
                df = fetcher.fetch_daily_kline(code, start_date="20240101")
                if df.empty or len(df) < 20:
                    continue

                df_std = self._normalize_df(df)
                name = name_map.get(code, code)

                # 基本面策略需要财务数据
                extra = {}
                if self.strategy_key == "value_invest":
                    extra = self._get_finance_extra(fetcher, code)
                elif self.strategy_key == "high_growth":
                    extra = self._get_growth_finance_extra(fetcher, code)

                item = self.scorer.score(code, name, df_std, extra)
                if item.score > 0:
                    candidates.append(item)
                    screened += 1
            except Exception as e:
                logger.debug(f"评分 {code} 异常: {e}")
                continue

        # 按得分降序，保留 Top N
        candidates.sort(key=lambda x: x.score, reverse=True)
        kept = candidates[:MAX_POOL_SIZE]
        for item in candidates[MAX_POOL_SIZE:]:
            item.status = "overflow"

        # 更新池
        self.items.clear()
        for item in kept:
            self.items[item.code] = item
        self.last_screened = datetime.now().strftime("%Y%m%d")

        if verbose:
            logger.info(f"  [{self.strategy_label}] 筛选完成: {screened} 只达标, 保留 Top {len(self.items)}")

        # 自动分配初始层级
        for item in kept:
            item.tier = item.compute_tier(0)
            item.evaluated_at = self.last_screened
        # 自动保存
        self.save()
        return self.ranked()

    def _normalize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        col_map = {
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "涨跌幅": "pct_change",
            "换手率": "turnover",
        }
        return df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    def _get_finance_extra(self, fetcher, code: str) -> dict:
        """获取价值投资评分需要的财务数据"""
        try:
            fin = fetcher.fetch_financial_data(code)
            if fin is not None and not fin.empty:
                latest_row = fin.iloc[-1]
                return {
                    "pe": float(latest_row.get("市盈率-动态", latest_row.get("pe", 999)) or 999),
                    "pb": float(latest_row.get("市净率", latest_row.get("pb", 999)) or 999),
                    "roe": float(latest_row.get("净资产收益率", latest_row.get("roe", 0)) or 0),
                    "dividend_yield": float(latest_row.get("股息率", latest_row.get("dividend_yield", 0)) or 0),
                    "market_cap": float(latest_row.get("总市值", latest_row.get("market_cap", 0)) or 0),
                    "debt_ratio": float(latest_row.get("资产负债率", latest_row.get("debt_ratio", 50)) or 50),
                }
        except Exception:
            pass
        # fallback: 从 stock_info 表获取
        try:
            info = fetcher.db.get_stock_list()
            row = info[info["code"] == code]
            if not row.empty:
                r = row.iloc[0]
                return {
                    "pe": float(r.get("pe", 999) or 999),
                    "pb": float(r.get("pb", 999) or 999),
                    "market_cap": float(r.get("total_mv", 0) or 0),
                }
        except Exception:
            pass
        return {}

    def _get_growth_finance_extra(self, fetcher, code: str) -> dict:
        """获取高增长策略需要的财务增长数据"""
        try:
            fin = fetcher.fetch_financial_data(code)
            if fin is not None and not fin.empty:
                # 尝试从多期财务数据中计算3年复合增长率
                try:
                    eps_vals = []
                    rev_vals = []
                    equity_vals = []
                    for col_pat, vals in [
                        ("基本每股收益", eps_vals),
                        ("营业收入", rev_vals),
                        ("净资产", equity_vals),
                    ]:
                        for c in fin.columns:
                            if col_pat in str(c):
                                try:
                                    v = float(fin[c].dropna().iloc[-1])
                                    vals.append(v)
                                except Exception:
                                    pass
                                break  # 只取第一个匹配列

                    eps_g3 = 0
                    rev_g3 = 0
                    equity_g3 = 0
                    if len(eps_vals) >= 4:
                        old_eps = eps_vals[0]
                        new_eps = eps_vals[-1]
                        if old_eps > 0:
                            eps_g3 = ((new_eps / old_eps) ** (1 / 3) - 1) * 100
                    if len(rev_vals) >= 4:
                        old_rev = rev_vals[0] if rev_vals[0] > 0 else 1
                        new_rev = rev_vals[-1]
                        rev_g3 = ((new_rev / old_rev) ** (1 / 3) - 1) * 100
                    if len(equity_vals) >= 4:
                        old_eq = equity_vals[0] if equity_vals[0] > 0 else 1
                        new_eq = equity_vals[-1]
                        equity_g3 = ((new_eq / old_eq) ** (1 / 3) - 1) * 100
                except Exception:
                    eps_g3 = 0
                    rev_g3 = 0
                    equity_g3 = 0

                latest_row = fin.iloc[-1]
                return {
                    "eps_growth_3y": round(eps_g3 or 0, 1),
                    "revenue_growth_3y": round(rev_g3 or 0, 1),
                    "peg": float(latest_row.get("PEG", latest_row.get("peg", 999)) or 999),
                    "roe": float(latest_row.get("净资产收益率", latest_row.get("roe", 0)) or 0),
                    "profit_growth_yoy": float(
                        latest_row.get("净利润增长率", latest_row.get("profit_growth_yoy", 0)) or 0
                    ),
                    "equity_growth_yoy": float(
                        latest_row.get("净资产同比增长率", latest_row.get("equity_growth_yoy", 0)) or 0
                    ),
                }
        except Exception:
            pass
        # fallback: 从东方财富实时行情获取基础数据
        try:
            # 尝试通过东财直连接口获取增长相关数据
            from .data_manager import DataFetcher
            fetcher2 = DataFetcher()
            df = fetcher2._fetch_financial_eastmoney_direct(code)
            if df is not None and not df.empty:
                r = df.iloc[0]
                return {
                    "eps_growth_3y": 0,
                    "revenue_growth_3y": 0,
                    "peg": 0,
                    "roe": float(r.get("roe", 0) or 0),
                    "profit_growth_yoy": float(r.get("净利润增长率", 0) or 0),
                    "equity_growth_yoy": 0,
                }
        except Exception:
            pass
        return {}

    def ranked(self, min_score: float = 0) -> list[StockPoolItem]:
        """返回按评分降序的股票列表"""
        items = [it for it in self.items.values() if it.status == "active" and it.score >= min_score]
        items.sort(key=lambda x: x.score, reverse=True)
        return items

    def top_codes(self, n: int = 10, min_score: float = 30) -> list[str]:
        """返回 Top N 股票代码"""
        return [it.code for it in self.ranked(min_score)[:n]]

    def get(self, code: str) -> StockPoolItem | None:
        return self.items.get(code)

    def add(self, code: str, name: str, score: float = 0, components: dict | None = None):
        """手动添加"""
        self.items[code] = StockPoolItem(
            code=code,
            name=name,
            score=score,
            components=components or {},
            screened_at=datetime.now().strftime("%Y%m%d"),
        )

    def remove(self, code: str) -> bool:
        if code in self.items:
            self.items[code].status = "removed"
            return True
        return False

    def clear(self):
        self.items.clear()
        self.last_screened = None

    def mark_traded(self, code: str):
        """标记某只已成交（赛马中建仓后标记）"""
        if code in self.items:
            self.items[code].status = "traded"

    @property
    def size(self) -> int:
        return sum(1 for it in self.items.values() if it.status == "active")

    def summary(self) -> dict:
        top = self.ranked()[:5]
        return {
            "strategy": self.strategy_label,
            "total": self.size,
            "last_screened": self.last_screened,
            "avg_score": round(np.mean([it.score for it in self.ranked()]), 1) if self.items else 0,
            "top5": [{"code": it.code, "name": it.name, "score": it.score} for it in top],
        }

    # ─── 持久化 ──────────────────────────────────────

    def save(self):
        try:
            data = {
                "strategy_key": self.strategy_key,
                "strategy_label": self.strategy_label,
                "last_screened": self.last_screened,
                "items": [it.to_dict() for it in self.items.values()],
            }
            with open(self._file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"保存池 {self.strategy_key} 失败: {e}")

    def load(self) -> bool:
        if not os.path.exists(self._file):
            return False
        try:
            with open(self._file, encoding="utf-8") as f:
                data = json.load(f)
            self.last_screened = data.get("last_screened")
            self.items = {}
            for d in data.get("items", []):
                item = StockPoolItem.from_dict(d)
                self.items[item.code] = item
            return True
        except Exception as e:
            logger.warning(f"加载池 {self.strategy_key} 失败: {e}")
            return False

    def refresh_gains(self, verbose: bool = False) -> dict[str, dict]:
        """刷新池内所有股票的涨幅数据（累计/当日/最大）

        对于没有 entry_price 的旧数据，自动从入池日K线回填。
        返回: {code: {cumulative_gain_pct, daily_gain_pct, max_gain_pct}}
        """
        fetcher = self._get_fetcher()
        gains = {}

        active_items = [it for it in self.items.values() if it.status == "active"]
        if not active_items:
            return gains

        if verbose:
            logger.info(f"  [{self.strategy_label}] 刷新涨幅，共 {len(active_items)} 只...")

        need_save = False

        for item in active_items:
            try:
                df = fetcher.fetch_daily_kline(item.code, start_date="20240101")
                if df.empty or len(df) < 1:
                    continue

                df_std = self._normalize_df(df)
                latest = df_std.iloc[-1]
                current_close = float(latest["close"])
                daily_pct = float(latest.get("pct_change", 0) or 0)

                # 回填 entry_price（旧数据缺失时，从入池日K线获取收盘价）
                if item.entry_price <= 0 and item.screened_at and "date" in df_std.columns:
                    df_std["_date_str"] = df_std["date"].astype(str)
                    entry_rows = df_std[df_std["_date_str"] >= str(item.screened_at)]
                    if not entry_rows.empty:
                        item.entry_price = round(float(entry_rows.iloc[0]["close"]), 2)
                        item.max_price = item.entry_price
                        need_save = True

                entry = item.entry_price if item.entry_price > 0 else current_close

                # 当日涨幅
                daily_gain = round(daily_pct, 2)

                # 累计涨幅
                cumulative = round((current_close / entry - 1) * 100, 2) if entry > 0 else 0

                # 最大涨幅：从入池日后的K线中找最高收盘价
                max_close = current_close
                if item.screened_at and "date" in df_std.columns and len(df_std) > 1:
                    df_std["_date_str"] = df_std["date"].astype(str)
                    mask = df_std["_date_str"] >= str(item.screened_at)
                    if mask.any():
                        max_close = float(df_std.loc[mask, "close"].max())
                max_gain = round((max_close / entry - 1) * 100, 2) if entry > 0 else 0

                gains[item.code] = {
                    "cumulative_gain_pct": cumulative,
                    "daily_gain_pct": daily_gain,
                    "max_gain_pct": max_gain,
                    "current_price": round(current_close, 2),
                }

                # 更新记录的 max_price
                if max_close > item.max_price:
                    item.max_price = round(max_close, 2)
                    need_save = True

            except Exception as e:
                logger.debug(f"刷新涨幅 {item.code} 异常: {e}")
                continue

        # 保存更新后的数据
        if need_save:
            self.save()

        if verbose:
            updated = len(gains)
            logger.info(f"  [{self.strategy_label}] 涨幅刷新完成: {updated}/{len(active_items)} 只")

        return gains

    # ─── 层级评估与池维护 ────────────────────────────

    def evaluate_pool(self, verbose: bool = False) -> dict[str, list[str]]:
        """评估池内所有标的，分配层级（focus/watch/broad/eliminated）

        流程：
        1. 刷新涨幅数据
        2. 基于当前评分 + 累计涨幅重新分配层级
        3. 自动剔除不达标标的

        返回: {focus: [...], watch: [...], broad: [...], eliminated: [...]}
        """
        # 先刷新涨幅
        gains = self.refresh_gains(verbose=False)

        today = datetime.now().strftime("%Y%m%d")
        result: dict[str, list[str]] = {TIER_FOCUS: [], TIER_WATCH: [], TIER_BROAD: [], TIER_ELIMINATED: []}

        need_save = False
        elim_count = 0

        for item in self.items.values():
            if item.status != "active":
                continue

            gain_info = gains.get(item.code, {})
            cum_gain = gain_info.get("cumulative_gain_pct", 0) or 0

            # 计算应属层级
            new_tier = item.compute_tier(cum_gain)
            item.evaluated_at = today

            # 检查是否需要淘汰
            if new_tier == TIER_ELIMINATED:
                item.tier = TIER_ELIMINATED
                item.status = "removed"
                elim_count += 1
                result[TIER_ELIMINATED].append(item.code)
                need_save = True
                if verbose:
                    reason = (
                        f"评分{item.score:.0f}" if item.score < ELIMINATE_SCORE_FLOOR else f"累计跌幅{cum_gain:.1f}%"
                    )
                    logger.info(f"  [{self.strategy_label}] 淘汰 {item.code} {item.name}: {reason}")
            else:
                if item.tier != new_tier:
                    if verbose and item.tier:
                        logger.info(
                            f"  [{self.strategy_label}] {item.code} {item.name}: "
                            f"{TIER_LABELS.get(item.tier, '?')} → {TIER_LABELS[new_tier]}"
                        )
                    item.tier = new_tier
                    need_save = True
                result[new_tier].append(item.code)

        if need_save:
            self.save()

        if verbose:
            focus_n = len(result[TIER_FOCUS])
            watch_n = len(result[TIER_WATCH])
            broad_n = len(result[TIER_BROAD])
            logger.info(
                f"  [{self.strategy_label}] 评估完成: 精选{focus_n} | 观察{watch_n} | 备选{broad_n} | 淘汰{elim_count}"
            )

        return result

    def create_trade_plan(
        self, code: str, name: str, current_price: float, score: float, reasons: list[str] | None = None
    ) -> TradePlan:
        """根据策略类型生成个股交易计划

        不同策略有不同的入场/止损/止盈/仓位逻辑：
        - 龙头战法: 短线激进，涨停价入场，-5% 严格止损，+15%/+30% 分批止盈，仓位20%
        - 麻雀战法: 中线波段，现价入场，MA20-3%止损，+8%/+15%止盈，仓位15%
        - 海龟战法: 趋势跟踪，突破入场，-2ATR 止损，+3ATR/+6ATR 止盈，仓位10%
        - 价值投资: 长线布局，现价入场，-10%宽松止损，+20%/+40%止盈，仓位25%
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if self.strategy_key == "dragon_head":
            return TradePlan(
                strategy_key=self.strategy_key,
                strategy_name=self.strategy_label,
                code=code,
                name=name,
                entry_price=round(current_price, 2),
                entry_type="涨停价入场" if score >= 70 else "现价入场",
                stop_loss=round(current_price * 0.95, 2),
                stop_loss_pct=-5.0,
                take_profit_1=round(current_price * 1.15, 2),
                take_profit_1_pct=15.0,
                take_profit_2=round(current_price * 1.30, 2),
                take_profit_2_pct=30.0,
                position_pct=20.0,
                risk_reward_ratio=3.0,
                hold_days=5,
                reasons=reasons or [f"龙头战法评分{score:.0f}分", "短线强势追涨"],
                warnings=["涨停板买入需排队，可能买不到", "严格止损-5%，不可扛单", "不适合重仓"],
                created_at=now,
            )

        elif self.strategy_key == "sparrow":
            return TradePlan(
                strategy_key=self.strategy_key,
                strategy_name=self.strategy_label,
                code=code,
                name=name,
                entry_price=round(current_price, 2),
                entry_type="回调至MA20附近入场",
                stop_loss=round(current_price * 0.93, 2),
                stop_loss_pct=-7.0,
                take_profit_1=round(current_price * 1.08, 2),
                take_profit_1_pct=8.0,
                take_profit_2=round(current_price * 1.15, 2),
                take_profit_2_pct=15.0,
                position_pct=15.0,
                risk_reward_ratio=2.1,
                hold_days=10,
                reasons=reasons or [f"麻雀战法评分{score:.0f}分", "均线支撑+KDJ金叉信号"],
                warnings=["等待回调至MA20附近再入场", "若跌破MA20立即止损", "分批建仓降低风险"],
                created_at=now,
            )

        elif self.strategy_key == "turtle":
            return TradePlan(
                strategy_key=self.strategy_key,
                strategy_name=self.strategy_label,
                code=code,
                name=name,
                entry_price=round(current_price, 2),
                entry_type="突破20日高点确认后入场",
                stop_loss=round(current_price * 0.90, 2),
                stop_loss_pct=-10.0,
                take_profit_1=round(current_price * 1.15, 2),
                take_profit_1_pct=15.0,
                take_profit_2=round(current_price * 1.30, 2),
                take_profit_2_pct=30.0,
                position_pct=10.0,
                risk_reward_ratio=3.0,
                hold_days=20,
                reasons=reasons or [f"海龟战法评分{score:.0f}分", "趋势突破信号"],
                warnings=["趋势跟踪需要较大止损空间", "可能出现假突破", "建议分批加仓（金字塔）"],
                created_at=now,
            )

        elif self.strategy_key == "value_invest":
            return TradePlan(
                strategy_key=self.strategy_key,
                strategy_name=self.strategy_label,
                code=code,
                name=name,
                entry_price=round(current_price, 2),
                entry_type="现价分批入场",
                stop_loss=round(current_price * 0.90, 2),
                stop_loss_pct=-10.0,
                take_profit_1=round(current_price * 1.20, 2),
                take_profit_1_pct=20.0,
                take_profit_2=round(current_price * 1.40, 2),
                take_profit_2_pct=40.0,
                position_pct=25.0,
                risk_reward_ratio=4.0,
                hold_days=60,
                reasons=reasons or [f"价值投资评分{score:.0f}分", "估值合理，适合长线布局"],
                warnings=["长线持有需要耐心", "关注基本面变化", "建议分3-5批建仓"],
                created_at=now,
            )

        # 兜底：通用交易计划
        return TradePlan(
            strategy_key=self.strategy_key,
            strategy_name=self.strategy_label,
            code=code,
            name=name,
            entry_price=round(current_price, 2),
            entry_type="现价入场",
            stop_loss=round(current_price * 0.92, 2),
            stop_loss_pct=-8.0,
            take_profit_1=round(current_price * 1.12, 2),
            take_profit_1_pct=12.0,
            take_profit_2=round(current_price * 1.22, 2),
            take_profit_2_pct=22.0,
            position_pct=15.0,
            risk_reward_ratio=2.5,
            hold_days=15,
            reasons=reasons or [f"评分{score:.0f}分"],
            warnings=["请根据个人风险偏好调整"],
            created_at=now,
        )

    def maintenance(self, verbose: bool = False) -> dict:
        """执行一次完整的池维护（评估 + 淘汰 + 补入）

        1. 评估现有标的层级
        2. 淘汰不达标标的
        3. 如果池中数量不足，从市场补充筛选
        4. 更新最后维护日期

        返回: {evaluated, eliminated, replenished, tier_counts, error}
        """
        result = {"evaluated": 0, "eliminated": 0, "replenished": 0, "tier_counts": {}, "error": None}

        try:
            # 1. 评估层级
            tier_result = self.evaluate_pool(verbose=verbose)
            active = sum(1 for it in self.items.values() if it.status == "active")
            result["tier_counts"] = {tier: len(codes) for tier, codes in tier_result.items() if tier != TIER_ELIMINATED}
            result["evaluated"] = active + len(tier_result[TIER_ELIMINATED])
            result["eliminated"] = len(tier_result[TIER_ELIMINATED])

            # 2. 补入：如果 active 总数 < MAX_POOL_SIZE，从市场筛选补充
            if active < MAX_POOL_SIZE:
                need = MAX_POOL_SIZE - active
                fetcher = self._get_fetcher()
                stock_pool = fetcher.build_stock_pool(filter_st=True, filter_new=True)

                # 排除已在池中的代码
                existing = set(self.items.keys())
                fresh_pool = [c for c in stock_pool if c not in existing]

                if fresh_pool:
                    candidates: list[StockPoolItem] = []
                    for code in fresh_pool[:SCREEN_SAMPLE_SIZE]:
                        try:
                            df = fetcher.fetch_daily_kline(str(code), start_date="20240101")
                            if df.empty or len(df) < 20:
                                continue
                            df_std = self._normalize_df(df)
                            name = str(code)
                            extra = {}
                            if self.strategy_key == "value_invest":
                                extra = self._get_finance_extra(fetcher, code)
                            item = self.scorer.score(str(code), name, df_std, extra)
                            if item.score >= TIER_THRESHOLDS[TIER_BROAD]:  # 至少达到备选
                                item.tier = item.compute_tier(0)
                                item.evaluated_at = datetime.now().strftime("%Y%m%d")
                                candidates.append(item)
                        except Exception:
                            continue

                    # 按评分降序取前 need 只
                    candidates.sort(key=lambda x: x.score, reverse=True)
                    for item in candidates[:need]:
                        self.items[item.code] = item
                        result["replenished"] += 1
                        if verbose:
                            logger.info(
                                f"  [{self.strategy_label}] 补入 {item.code} {item.name} "
                                f"评分{item.score:.0f} → {TIER_LABELS.get(item.tier, '?')}"
                            )

            self.save()

        except Exception as e:
            result["error"] = str(e)
            logger.error(f"[{self.strategy_label}] 维护异常: {e}")

        return result

    def tier_summary(self) -> dict[str, int]:
        """返回各层级数量统计"""
        counts = {TIER_FOCUS: 0, TIER_WATCH: 0, TIER_BROAD: 0, TIER_ELIMINATED: 0}
        for item in self.items.values():
            if item.tier and item.status == "active":
                counts[item.tier] = counts.get(item.tier, 0) + 1
        return counts

    def get_by_tier(self, tier: str) -> list[StockPoolItem]:
        """获取指定层级的所有标的（按评分降序）"""
        items = [it for it in self.items.values() if it.tier == tier and it.status == "active"]
        items.sort(key=lambda x: x.score, reverse=True)
        return items

    def should_maintain(self) -> bool:
        """检查是否需要进行维护（基于策略的维护频率）"""
        interval = MAINTENANCE_INTERVALS.get(self.strategy_key, 5)
        if not self.last_screened:
            return True
        try:
            last_date = datetime.strptime(self.last_screened, "%Y%m%d")
            days_passed = (datetime.now() - last_date).days
            return days_passed >= interval
        except ValueError:
            return True


# ─── 全局池管理器 ────────────────────────────────────────

SCORER_MAP: dict[str, BaseScorer] = {
    "dragon_head": DragonHeadScorer(),
    "sparrow": SparrowScorer(),
    "turtle": TurtleScorer(),
    "value_invest": ValueInvestScorer(),
    "bollinger": BollingerBandScorer(),
    "grid": GridScorer(),
    "ma_cross": MACrossoverScorer(),
    "volume_breakout": VolumeBreakoutScorer(),
    "trend_accel": TrendAccelerationScorer(),
    "high_growth": HighGrowthScorer(),
}

POOL_LABEL_MAP = {
    "dragon_head": "龙头战法",
    "sparrow": "麻雀战法",
    "turtle": "海龟战法",
    "value_invest": "价值投资",
    "bollinger": "布林带回归",
    "grid": "网格交易",
    "ma_cross": "均线交叉",
    "volume_breakout": "量价突破",
    "trend_accel": "趋势加速",
    "high_growth": "高增长",
}


class PoolManager:
    """全局股票池管理器 - 统一管理四种策略的独立股票池"""

    def __init__(self):
        self.pools: dict[str, StrategyStockPool] = {}
        self._init_pools()

    def _init_pools(self):
        for key in STRATEGIES:
            if key not in SCORER_MAP:
                continue
            label = STRATEGIES[key]
            scorer = SCORER_MAP[key]
            pool = StrategyStockPool(key, label, scorer)
            # 尝试加载已保存的池
            pool.load()
            self.pools[key] = pool

    # ─── 统一操作 ──────────────────────────────────────

    def screen_all(self, stock_pool: list[str] | None = None, verbose: bool = False) -> dict[str, list[StockPoolItem]]:
        """为所有策略执行筛选"""
        results = {}
        if stock_pool is None:
            from .data_manager import DataFetcher

            stock_pool = DataFetcher().build_stock_pool(filter_st=True, filter_new=True)
        for key, pool in self.pools.items():
            results[key] = pool.screen(stock_pool, verbose=verbose)
        return results

    def screen_one(
        self, strategy_key: str, stock_pool: list[str] | None = None, verbose: bool = False
    ) -> list[StockPoolItem]:
        """为指定策略执行筛选"""
        pool = self.pools.get(strategy_key)
        if pool is None:
            logger.warning(f"未知策略: {strategy_key}")
            return []
        return pool.screen(stock_pool, verbose=verbose)

    def get_pool(self, strategy_key: str) -> StrategyStockPool | None:
        return self.pools.get(strategy_key)

    def get_ranked_codes(self, strategy_key: str, n: int = 10) -> list[str]:
        """获取某策略 Top N 股票代码（赛马引擎直接调用）"""
        pool = self.pools.get(strategy_key)
        return pool.top_codes(n, min_score=30) if pool else []

    def overview(self) -> list[dict]:
        """返回所有池概览"""
        return [p.summary() for p in self.pools.values()]

    def summary(self) -> dict[str, int]:
        """返回各策略池股票数量的简单汇总"""
        return {key: pool.size for key, pool in self.pools.items()}

    def save_all(self):
        for pool in self.pools.values():
            pool.save()

    def load_all(self):
        for pool in self.pools.values():
            pool.load()

    def refresh_all_gains(self, verbose: bool = False) -> dict[str, dict[str, dict]]:
        """刷新所有策略池的涨幅数据"""
        all_gains = {}
        for key, pool in self.pools.items():
            all_gains[key] = pool.refresh_gains(verbose=verbose)
        return all_gains

    def evaluate_all(self, verbose: bool = False) -> dict[str, dict[str, list[str]]]:
        """评估所有策略池的层级"""
        results = {}
        for key, pool in self.pools.items():
            if verbose:
                logger.info(f"评估 {pool.strategy_label}...")
            results[key] = pool.evaluate_pool(verbose=verbose)
        return results

    def maintenance_all(self, verbose: bool = False) -> dict[str, dict]:
        """对所有策略池执行维护（评估 + 淘汰 + 补入）"""
        results = {}
        for key, pool in self.pools.items():
            if pool.should_maintain():
                if verbose:
                    logger.info(f"\n维护 {pool.strategy_label}（频率: {MAINTENANCE_INTERVALS.get(key, 5)}天）...")
                results[key] = pool.maintenance(verbose=verbose)
            else:
                results[key] = {"skipped": True, "reason": f"距上次维护不足{MAINTENANCE_INTERVALS.get(key, 5)}天"}
        return results

    def maintenance_one(self, strategy_key: str, verbose: bool = False) -> dict:
        """对指定策略池执行维护"""
        pool = self.pools.get(strategy_key)
        if pool is None:
            return {"error": f"未知策略: {strategy_key}"}
        return pool.maintenance(verbose=verbose)

    def tier_overview(self) -> dict[str, dict[str, int]]:
        """返回所有池的层级统计"""
        return {key: pool.tier_summary() for key, pool in self.pools.items()}
