"""
模拟盘引擎 - 多策略赛马系统 [AKQuant]

每个策略配置独立 100 万账户，通过 AKQuant 回测引擎并行评估，
支持每日赛马、收益率排名、持仓跟踪。
"""

import json
import logging
import os
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

from .config import (
    DEFAULT_CAPITAL, DEFAULT_COMMISSION, DEFAULT_STAMP_TAX,
    DEFAULT_SLIPPAGE, DATA_DIR, BACKTEST_LOT_SIZE, BACKTEST_T_PLUS_ONE,
    BACKTEST_MIN_COMMISSION,
)
from .data_manager import DataFetcher
from .stock_pool import PoolManager

logger = logging.getLogger(__name__)

RACE_CAPITAL = 1_000_000  # 每个策略 100 万
SNAPSHOT_FILE = os.path.join(DATA_DIR, "paper_race.json")


# ─── 数据结构 ─────────────────────────────────────────────

@dataclass
class PaperPosition:
    """模拟持仓"""
    code: str
    name: str
    cost: float
    volume: int
    buy_date: str
    current_price: float = 0.0
    market_value: float = 0.0
    profit_pct: float = 0.0
    hold_days: int = 0


@dataclass
class PaperTrade:
    """模拟成交记录"""
    date: str
    code: str
    name: str
    action: str
    price: float
    volume: int
    amount: float
    reason: str
    pnl: float = 0.0


@dataclass
class PaperAccount:
    """模拟账户"""
    strategy_name: str
    strategy_label: str
    initial_capital: float = RACE_CAPITAL
    cash: float = RACE_CAPITAL
    positions: Dict[str, PaperPosition] = field(default_factory=dict)
    trades: List[PaperTrade] = field(default_factory=list)
    daily_snapshots: List[Dict] = field(default_factory=list)
    status: str = "idle"

    @property
    def total_equity(self) -> float:
        pos_value = sum(p.market_value for p in self.positions.values())
        return self.cash + pos_value

    @property
    def total_return_pct(self) -> float:
        return (self.total_equity / self.initial_capital - 1) * 100

    @property
    def position_count(self) -> int:
        return len(self.positions)

    @property
    def daily_return(self) -> Optional[float]:
        if len(self.daily_snapshots) >= 2:
            prev = self.daily_snapshots[-2]["equity"]
            curr = self.daily_snapshots[-1]["equity"]
            return (curr / prev - 1) * 100 if prev > 0 else 0
        return None

    def to_dict(self) -> Dict:
        return {
            "strategy_name": self.strategy_name,
            "strategy_label": self.strategy_label,
            "initial_capital": self.initial_capital,
            "cash": self.cash,
            "total_equity": self.total_equity,
            "total_return_pct": round(self.total_return_pct, 2),
            "position_count": self.position_count,
            "status": self.status,
        }


@dataclass
class RaceRanking:
    """赛马排名条目"""
    rank: int
    strategy_label: str
    total_return: float
    total_equity: float
    daily_return: Optional[float]
    position_count: int
    max_drawdown: float
    trades: int


# ─── 模拟盘引擎 ───────────────────────────────────────────

class PaperTrader:
    """本地模拟盘 - 基于 AKQuant 的多策略赛马"""

    STRATEGY_KEYS = ["dragon", "sparrow", "turtle", "value"]
    STRATEGY_LABELS = {
        "dragon": "龙头战法", "sparrow": "麻雀战法",
        "turtle": "海龟战法", "value": "价值投资",
    }

    def __init__(self, commission: float = None, stamp_tax: float = None, slippage: float = None):
        self.commission = commission or DEFAULT_COMMISSION
        self.stamp_tax = stamp_tax or DEFAULT_STAMP_TAX
        self.slippage = slippage or DEFAULT_SLIPPAGE
        self.fetcher = DataFetcher()
        self.accounts: Dict[str, PaperAccount] = {}
        self.stock_pool: List[str] = []
        self.pool_mgr = PoolManager()
        self.current_date: Optional[str] = None
        self._init_accounts()

    def _init_accounts(self):
        for key in self.STRATEGY_KEYS:
            label = self.STRATEGY_LABELS[key]
            self.accounts[key] = PaperAccount(
                strategy_name=key, strategy_label=label,
                initial_capital=RACE_CAPITAL, cash=RACE_CAPITAL,
            )

    def set_stock_pool(self, codes: List[str]):
        self.stock_pool = codes

    # ─── 每日赛马 ──────────────────────────────────────

    def run_daily(self, date: str = None, verbose: bool = False) -> Dict[str, Dict]:
        """
        执行一个交易日的赛马：
        1. 获取各策略独立股票池
        2. 通过 AKQuant 在单日数据上运行回测
        3. 记录快照并排名
        """
        self.current_date = date or datetime.now().strftime("%Y%m%d")
        results = {}

        for key in self.STRATEGY_KEYS:
            acc = self.accounts[key]
            try:
                # 获取策略独立股票池
                pool_codes = self.pool_mgr.get_ranked_codes(key, 20)
                target_pool = pool_codes if pool_codes else self.stock_pool[:30]

                # 单日 AKQuant 回测
                prev_snapshots = len(acc.daily_snapshots)
                self._run_strategy_backtest(key, acc, target_pool, self.current_date)

                # 记录快照
                equity = acc.total_equity
                if len(acc.daily_snapshots) > prev_snapshots:
                    pass  # 回测内部已记录
                else:
                    acc.daily_snapshots.append({
                        "date": self.current_date,
                        "equity": round(equity, 2),
                        "cash": round(acc.cash, 2),
                        "positions": acc.position_count,
                    })

                # 重新估值
                self._mark_to_market(key)

                results[key] = acc.to_dict()
                if verbose:
                    logger.info(f"  {acc.strategy_label}: 权益 {equity:,.0f} "
                                f"({acc.total_return_pct:+.2f}%), 持仓 {acc.position_count}")

            except Exception as e:
                logger.error(f"账户 {key} 日跑异常: {e}")

        self.save()
        return results

    def _run_strategy_backtest(self, key: str, acc: PaperAccount,
                               pool_codes: List[str], date: str):
        """通过 AKQuant 运行策略在指定日期池上的回测"""
        from .strategies import STRATEGY_CLASSES
        strategy_cls = STRATEGY_CLASSES.get(key)
        if strategy_cls is None:
            return

        try:
            from akquant import run_backtest

            # 确定回测区间：从赛马起始日到当前日期
            start_dt = "20240101"
            if acc.daily_snapshots:
                start_dt = acc.daily_snapshots[0]["date"]
            date_str = date[:4] + "-" + date[4:6] + "-" + date[6:8]
            start_str = start_dt[:4] + "-" + start_dt[4:6] + "-" + start_dt[6:8]

            # 准备数据
            data = {}
            for code in pool_codes[:20]:
                df = self.fetcher.fetch_daily_kline(
                    str(code), start_date=start_dt, end_date=date
                )
                if df is None or df.empty or len(df) < 20:
                    continue
                col_map = {"日期": "date", "开盘": "open", "收盘": "close",
                          "最高": "high", "最低": "low", "成交量": "volume",
                          "成交额": "amount"}
                df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
                required = ["date", "open", "high", "low", "close", "volume"]
                if not all(c in df.columns for c in required):
                    continue
                df = df[required].copy()
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df = df.dropna(subset=["date"]).sort_values("date")
                df["symbol"] = str(code)
                if len(df) >= 20:
                    data[str(code)] = df[["date", "open", "high", "low", "close", "volume", "symbol"]]

            if not data:
                return

            strategy_inst = strategy_cls()
            warmup = getattr(strategy_inst, "warmup_period", 60)

            result = run_backtest(
                data=data,
                strategy=strategy_cls,
                initial_cash=acc.initial_capital,
                commission_policy={"type": "percent", "value": self.commission},
                stamp_tax_rate=self.stamp_tax,
                slippage=self.slippage,
                t_plus_one=BACKTEST_T_PLUS_ONE,
                lot_size=BACKTEST_LOT_SIZE,
                min_commission=BACKTEST_MIN_COMMISSION,
                start_time=pd.Timestamp(start_str),
                end_time=pd.Timestamp(date_str),
                warmup_period=warmup,
            )

            # 解析结果更新账户
            self._merge_akquant_result(key, acc, result, date)

        except ImportError:
            logger.warning("AKQuant 未安装，使用备用逻辑")
            self._run_fallback_logic(key, acc, pool_codes, date)
        except Exception as e:
            logger.debug(f"策略 {key} AKQ 日跑异常: {e}")
            self._run_fallback_logic(key, acc, pool_codes, date)

    def _merge_akquant_result(self, key: str, acc: PaperAccount, result, date: str):
        """将 AKQuant 回测结果合并到模拟账户"""
        try:
            equity_df = result.equity_curve if hasattr(result, "equity_curve") else None
            if equity_df is not None and not equity_df.empty:
                # 取最新权益
                final_equity = float(equity_df["equity"].iloc[-1]) if "equity" in equity_df.columns else acc.initial_capital
                acc.cash = 0
                for s in acc.daily_snapshots:
                    if s["date"] == date:
                        s.update({"equity": round(final_equity, 2), "cash": 0, "positions": 0})
                        return
                acc.daily_snapshots.append({
                    "date": date, "equity": round(final_equity, 2),
                    "cash": 0, "positions": 0,
                })
            else:
                # 无权益曲线时，使用 metrics
                metrics = result.metrics_df if hasattr(result, "metrics_df") else None
                if metrics is not None and not metrics.empty:
                    cols_lower = {c.lower(): c for c in metrics.columns}
                    for k in ["final_value", "final_capital", "ending_value", "total_equity"]:
                        if k in cols_lower:
                            final_equity = float(metrics.iloc[0][cols_lower[k]])
                            acc.daily_snapshots.append({
                                "date": date, "equity": round(final_equity, 2),
                                "cash": 0, "positions": 0,
                            })
                            return
        except Exception as e:
            logger.debug(f"合并 AKQ 结果异常: {e}")

    def _run_fallback_logic(self, key: str, acc: PaperAccount,
                            pool_codes: List[str], date: str):
        """备用逻辑：简单记录快照（不交易）"""
        prev = acc.cash
        for pos in acc.positions.values():
            mp = self._get_market_price(pos.code, date)
            if mp:
                pos.current_price = mp
                pos.market_value = mp * pos.volume
                pos.profit_pct = (mp / pos.cost - 1) * 100 if pos.cost > 0 else 0
        equity = acc.total_equity
        exists = False
        for s in acc.daily_snapshots:
            if s["date"] == date:
                s.update({"equity": round(equity, 2), "cash": round(acc.cash, 2),
                         "positions": acc.position_count})
                exists = True
                break
        if not exists:
            acc.daily_snapshots.append({
                "date": date, "equity": round(equity, 2),
                "cash": round(acc.cash, 2), "positions": acc.position_count,
            })

    # ─── 持仓估值 ──────────────────────────────────────

    def _mark_to_market(self, key: str):
        acc = self.accounts[key]
        for code, pos in list(acc.positions.items()):
            mp = self._get_market_price(code, self.current_date)
            if mp is not None:
                pos.current_price = mp
                pos.market_value = mp * pos.volume
                pos.profit_pct = (mp / pos.cost - 1) * 100 if pos.cost > 0 else 0

    def _get_market_price(self, code: str, date: str) -> Optional[float]:
        df = self.fetcher.fetch_daily_kline(code)
        if df.empty:
            return None
        for col in ("date", "日期"):
            if col in df.columns:
                row = df[df[col] == date]
                if not row.empty:
                    return row.iloc[-1].get("close", row.iloc[-1].get("收盘", None))
        return None

    # ─── 赛马排名 ──────────────────────────────────────

    def get_rankings(self) -> List[RaceRanking]:
        rankings = []
        for key, acc in self.accounts.items():
            md = self._calc_max_drawdown(key)
            rankings.append(RaceRanking(
                rank=0,
                strategy_label=acc.strategy_label,
                total_return=round(acc.total_return_pct, 2),
                total_equity=round(acc.total_equity, 2),
                daily_return=round(acc.daily_return, 2) if acc.daily_return is not None else None,
                position_count=acc.position_count,
                max_drawdown=round(md, 2),
                trades=len(acc.trades),
            ))
        rankings.sort(key=lambda r: r.total_return, reverse=True)
        for i, r in enumerate(rankings):
            r.rank = i + 1
        return rankings

    def _calc_max_drawdown(self, key: str) -> float:
        acc = self.accounts[key]
        snapshots = acc.daily_snapshots
        if len(snapshots) < 2:
            return 0
        values = [s["equity"] for s in snapshots]
        cummax = np.maximum.accumulate(values)
        drawdowns = (np.array(values) - cummax) / cummax * 100
        return abs(float(drawdowns.min()))

    # ─── 历史回放 ──────────────────────────────────────

    def run_history(self, start_date: str, end_date: str = None,
                    stock_pool: List[str] = None, verbose: bool = False) -> List[RaceRanking]:
        """从历史数据回放赛马"""
        self.reset()
        if stock_pool:
            self.set_stock_pool(stock_pool)

        end_date = end_date or datetime.now().strftime("%Y%m%d")
        index_df = self.fetcher.fetch_index_daily("000001", start_date, end_date)
        if index_df.empty:
            logger.error("无法获取交易日历")
            return []

        trading_dates = index_df["date"].tolist() if "date" in index_df.columns else sorted(index_df.index.tolist())
        if verbose:
            logger.info(f"赛马回放: {start_date} ~ {end_date}, 共 {len(trading_dates)} 个交易日")

        for i, date in enumerate(trading_dates):
            self.run_daily(date, verbose=(verbose and i % 20 == 0))
            if verbose and i % 50 == 0:
                ranks = self.get_rankings()
                top = ranks[0] if ranks else None
                if top:
                    logger.info(f"  Day {i}: {top.strategy_label} ({top.total_return:+.2f}%)")

        return self.get_rankings()

    # ─── 持久化 ──────────────────────────────────────

    def save(self):
        data = {
            "current_date": self.current_date,
            "stock_pool": self.stock_pool[:200],
            "accounts": {},
        }
        for key, acc in self.accounts.items():
            data["accounts"][key] = {
                "strategy_name": acc.strategy_name,
                "strategy_label": acc.strategy_label,
                "initial_capital": acc.initial_capital,
                "cash": acc.cash,
                "daily_snapshots": acc.daily_snapshots,
                "status": acc.status,
            }
        try:
            with open(SNAPSHOT_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"保存赛马状态失败: {e}")

    def load(self) -> bool:
        if not os.path.exists(SNAPSHOT_FILE):
            return False
        try:
            with open(SNAPSHOT_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.current_date = data.get("current_date")
            self.stock_pool = data.get("stock_pool", [])
            for key, a in data.get("accounts", {}).items():
                if key in self.accounts:
                    acc = self.accounts[key]
                    acc.cash = a.get("cash", RACE_CAPITAL)
                    acc.daily_snapshots = a.get("daily_snapshots", [])
                    acc.status = a.get("status", "idle")
            return True
        except Exception as e:
            logger.warning(f"加载赛马状态失败: {e}")
            return False

    def reset(self):
        self._init_accounts()
        self.current_date = None
        if os.path.exists(SNAPSHOT_FILE):
            try:
                os.remove(SNAPSHOT_FILE)
            except Exception:
                pass
