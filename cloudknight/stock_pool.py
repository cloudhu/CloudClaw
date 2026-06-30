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
import os
import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable

import pandas as pd
import numpy as np

from .config import DATA_DIR, STRATEGIES
from .indicators import calc_ma, calc_macd, calc_kdj, calc_rsi, calc_atr, calc_breakout

logger = logging.getLogger(__name__)

POOL_DIR = os.path.join(DATA_DIR, "pools")
os.makedirs(POOL_DIR, exist_ok=True)

MAX_POOL_SIZE = 30         # 每种策略池最多保留 30 只
SCREEN_SAMPLE_SIZE = 200   # 一次筛选最多遍历 200 只


# ─── 数据结构 ─────────────────────────────────────────────

@dataclass
class StockPoolItem:
    """股票池中的单只股票"""
    code: str
    name: str
    score: float                           # 综合评分 0-100
    components: Dict[str, float] = field(default_factory=dict)  # 各维度得分
    screened_at: str = ""                  # 入池日期
    status: str = "active"                 # active | traded | removed

    def to_dict(self) -> Dict:
        return {
            "code": self.code,
            "name": self.name,
            "score": round(self.score, 1),
            "components": {k: round(v, 1) for k, v in self.components.items()},
            "screened_at": self.screened_at,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "StockPoolItem":
        return cls(
            code=d["code"], name=d.get("name", ""), score=d.get("score", 0),
            components=d.get("components", {}), screened_at=d.get("screened_at", ""),
            status=d.get("status", "active"),
        )


# ─── 评分器基类 ──────────────────────────────────────────

class BaseScorer:
    """评分器基类"""
    name: str = "base"

    def score(self, code: str, name: str, df: pd.DataFrame,
              extra: Dict = None) -> StockPoolItem:
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

    def _linear_score(self, val: float, low: float, high: float,
                      floor: float = 0, ceil: float = 100) -> float:
        """线性映射 val ∈ [low, high] → [0, 1]，再拉伸到 [floor, ceil]"""
        if high <= low:
            return floor
        ratio = max(0, min(1, (val - low) / (high - low)))
        return round(floor + ratio * (ceil - floor), 1)

    def _bell_score(self, val: float, low: float, mid: float, high: float,
                    max_pts: float = 100) -> float:
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

    def score(self, code: str, name: str, df: pd.DataFrame,
              extra: Dict = None) -> StockPoolItem:
        if df.empty or len(df) < 5:
            return StockPoolItem(code=code, name=name, score=0, components={})

        latest = df.iloc[-1]
        close = self._safe_val(latest, "close", "收盘")
        open_p = self._safe_val(latest, "open", "开盘")
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
        elif is_limit_up := (pct >= 9.5):
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
        total = sum(comp.values())
        return StockPoolItem(
            code=code, name=name, score=round(total, 1),
            components=comp, screened_at=datetime.now().strftime("%Y%m%d"),
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

    def score(self, code: str, name: str, df: pd.DataFrame,
              extra: Dict = None) -> StockPoolItem:
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
        k, d, j = latest["K"], latest["D"], latest["J"]

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
            kdj_score = 8   # 已处于多头但未金叉
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
        total = sum(comp.values())
        return StockPoolItem(
            code=code, name=name, score=round(total, 1),
            components=comp, screened_at=datetime.now().strftime("%Y%m%d"),
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
            for pc, vc in zip(pct_changes, vol_changes):
                if pc > 0 and vc > 0:
                    align += 1     # 价量同向：好
                elif pc < 0 and vc < 0:
                    align += 0.5   # 缩量下跌：中性偏多
                elif pc > 0 and vc < 0:
                    align += 0.3   # 缩量上涨：一般
                # 价跌量增：不加分
            return round(align / len(pct_changes) * 15, 1)
        except Exception:
            return 7.5


# ─── 海龟战法评分器 ──────────────────────────────────────

class TurtleScorer(BaseScorer):
    """海龟战法评分：通道突破 + ATR波动率 + 趋势 + 流动性"""
    name = "turtle"

    def score(self, code: str, name: str, df: pd.DataFrame,
              extra: Dict = None) -> StockPoolItem:
        if df.empty or len(df) < 100:
            return StockPoolItem(code=code, name=name, score=0, components={})

        latest = df.iloc[-1]
        close = latest["close"]
        entry_period = 20

        # ATR
        atr_series = calc_atr(df, 20)
        current_atr = atr_series.iloc[-1]

        # 通道
        upper, lower = calc_breakout(df, entry_period)
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
        total = sum(comp.values())
        return StockPoolItem(
            code=code, name=name, score=round(total, 1),
            components=comp, screened_at=datetime.now().strftime("%Y%m%d"),
        )


# ─── 价值投资评分器 ──────────────────────────────────────

class ValueInvestScorer(BaseScorer):
    """价值投资评分：PE/PB分位 + ROE + 股息率 + 负债率"""
    name = "value_invest"

    def score(self, code: str, name: str, df: pd.DataFrame,
              extra: Dict = None) -> StockPoolItem:
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
        total = sum(comp.values())
        return StockPoolItem(
            code=code, name=name, score=round(total, 1),
            components=comp, screened_at=datetime.now().strftime("%Y%m%d"),
        )


# ─── 策略股票池 ──────────────────────────────────────────

class StrategyStockPool:
    """单个策略的股票池"""

    def __init__(self, strategy_key: str, strategy_label: str, scorer: BaseScorer):
        self.strategy_key = strategy_key
        self.strategy_label = strategy_label
        self.scorer = scorer
        self.items: Dict[str, StockPoolItem] = {}   # code → item
        self.last_screened: Optional[str] = None
        self._file = os.path.join(POOL_DIR, f"{strategy_key}.json")

    def _get_fetcher(self):
        from .data_manager import DataFetcher
        return DataFetcher()

    def screen(self, stock_pool: List[str] = None,
               stock_info: pd.DataFrame = None,
               verbose: bool = False) -> List[StockPoolItem]:
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
            name_map = dict(zip(
                stock_info["股票代码"].astype(str),
                stock_info["股票简称"].astype(str)
            ))

        if verbose:
            logger.info(f"  [{self.strategy_label}] 开始筛选，候选 {len(sample)} 只...")

        candidates: List[StockPoolItem] = []
        screened = 0

        for code in sample:
            try:
                code = str(code)
                df = fetcher.fetch_daily_kline(code, start_date="20240101")
                if df.empty or len(df) < 20:
                    continue

                df_std = self._normalize_df(df)
                name = name_map.get(code, code)

                # 价值投资需要财务数据
                extra = {}
                if self.strategy_key == "value_invest":
                    extra = self._get_finance_extra(fetcher, code)

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
            logger.info(f"  [{self.strategy_label}] 筛选完成: {screened} 只达标, "
                        f"保留 Top {len(self.items)}")

        # 自动保存
        self.save()
        return self.ranked()

    def _normalize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        col_map = {"日期": "date", "开盘": "open", "收盘": "close", "最高": "high",
                   "最低": "low", "成交量": "volume", "成交额": "amount",
                   "涨跌幅": "pct_change", "换手率": "turnover"}
        return df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    def _get_finance_extra(self, fetcher, code: str) -> Dict:
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

    # ─── 池操作 ──────────────────────────────────────

    def ranked(self, min_score: float = 0) -> List[StockPoolItem]:
        """返回按评分降序的股票列表"""
        items = [it for it in self.items.values() if it.status == "active" and it.score >= min_score]
        items.sort(key=lambda x: x.score, reverse=True)
        return items

    def top_codes(self, n: int = 10, min_score: float = 30) -> List[str]:
        """返回 Top N 股票代码"""
        return [it.code for it in self.ranked(min_score)[:n]]

    def get(self, code: str) -> Optional[StockPoolItem]:
        return self.items.get(code)

    def add(self, code: str, name: str, score: float = 0,
            components: Dict = None):
        """手动添加"""
        self.items[code] = StockPoolItem(
            code=code, name=name, score=score,
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

    def summary(self) -> Dict:
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
            with open(self._file, "r", encoding="utf-8") as f:
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


# ─── 全局池管理器 ────────────────────────────────────────

SCORER_MAP: Dict[str, BaseScorer] = {
    "dragon_head": DragonHeadScorer(),
    "sparrow": SparrowScorer(),
    "turtle": TurtleScorer(),
    "value_invest": ValueInvestScorer(),
}

POOL_LABEL_MAP = {
    "dragon_head": "龙头战法",
    "sparrow": "麻雀战法",
    "turtle": "海龟战法",
    "value_invest": "价值投资",
}


class PoolManager:
    """全局股票池管理器 - 统一管理四种策略的独立股票池"""

    def __init__(self):
        self.pools: Dict[str, StrategyStockPool] = {}
        self._init_pools()

    def _init_pools(self):
        for key in STRATEGIES:
            label = STRATEGIES[key]
            scorer = SCORER_MAP[key]
            pool = StrategyStockPool(key, label, scorer)
            # 尝试加载已保存的池
            pool.load()
            self.pools[key] = pool

    # ─── 统一操作 ──────────────────────────────────────

    def screen_all(self, stock_pool: List[str] = None, verbose: bool = False) -> Dict[str, List[StockPoolItem]]:
        """为所有策略执行筛选"""
        results = {}
        if stock_pool is None:
            from .data_manager import DataFetcher
            stock_pool = DataFetcher().build_stock_pool(filter_st=True, filter_new=True)
        for key, pool in self.pools.items():
            results[key] = pool.screen(stock_pool, verbose=verbose)
        return results

    def screen_one(self, strategy_key: str, stock_pool: List[str] = None,
                   verbose: bool = False) -> List[StockPoolItem]:
        """为指定策略执行筛选"""
        pool = self.pools.get(strategy_key)
        if pool is None:
            logger.warning(f"未知策略: {strategy_key}")
            return []
        return pool.screen(stock_pool, verbose=verbose)

    def get_pool(self, strategy_key: str) -> Optional[StrategyStockPool]:
        return self.pools.get(strategy_key)

    def get_ranked_codes(self, strategy_key: str, n: int = 10) -> List[str]:
        """获取某策略 Top N 股票代码（赛马引擎直接调用）"""
        pool = self.pools.get(strategy_key)
        return pool.top_codes(n, min_score=30) if pool else []

    def overview(self) -> List[Dict]:
        """返回所有池概览"""
        return [p.summary() for p in self.pools.values()]

    def summary(self) -> Dict[str, int]:
        """返回各策略池股票数量的简单汇总"""
        return {key: pool.size for key, pool in self.pools.items()}

    def save_all(self):
        for pool in self.pools.values():
            pool.save()

    def load_all(self):
        for pool in self.pools.values():
            pool.load()
