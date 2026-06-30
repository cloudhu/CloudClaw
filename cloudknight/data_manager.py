"""
数据管理模块 - 负责数据获取、存储和缓存
使用 akshare 作为免费数据源，SQLite 作为本地存储
"""

import os
import pickle
import sqlite3
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

import pandas as pd
import numpy as np

from .config import DATA_DIR, DB_PATH

logger = logging.getLogger(__name__)

AK_AVAILABLE = False


def _ensure_akshare():
    global AK_AVAILABLE
    if not AK_AVAILABLE:
        try:
            import akshare as ak
            AK_AVAILABLE = True
            return ak
        except ImportError:
            raise ImportError("请安装 akshare: pip install akshare")
    import akshare as ak
    return ak


class CacheManager:
    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir or os.path.join(DATA_DIR, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)

    def _cache_key(self, prefix: str, **kwargs) -> str:
        raw = f"{prefix}_{sorted(kwargs.items())}"
        return hashlib.md5(raw.encode()).hexdigest() + ".pkl"

    def get(self, prefix: str, max_age_hours: int = 24, **kwargs) -> Optional[pd.DataFrame]:
        key = self._cache_key(prefix, **kwargs)
        path = os.path.join(self.cache_dir, key)
        if os.path.exists(path):
            age = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(path))).total_seconds() / 3600
            if age < max_age_hours:
                try:
                    with open(path, "rb") as f:
                        return pickle.load(f)
                except Exception:
                    pass
        return None

    def set(self, prefix: str, data: pd.DataFrame, **kwargs):
        key = self._cache_key(prefix, **kwargs)
        path = os.path.join(self.cache_dir, key)
        with open(path, "wb") as f:
            pickle.dump(data, f)


class Database:
    def __init__(self, db_path: str = None):
        self.db_path = db_path or DB_PATH
        self._init_db()

    def _get_conn(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._get_conn() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS stock_daily (
                code TEXT NOT NULL, date TEXT NOT NULL, open REAL, high REAL,
                low REAL, close REAL, volume REAL, amount REAL,
                amplitude REAL, pct_change REAL, turnover REAL,
                PRIMARY KEY (code, date))""")
            conn.execute("""CREATE TABLE IF NOT EXISTS stock_info (
                code TEXT PRIMARY KEY, name TEXT, industry TEXT, market TEXT,
                list_date TEXT, total_mv REAL, circ_mv REAL, pe REAL, pb REAL)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS trade_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT, strategy TEXT, code TEXT,
                name TEXT, action TEXT, price REAL, volume INTEGER, amount REAL,
                trade_time TEXT, reason TEXT)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS backtest_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT, strategy TEXT,
                start_date TEXT, end_date TEXT, initial_capital REAL,
                final_capital REAL, total_return REAL, annual_return REAL,
                max_drawdown REAL, sharpe_ratio REAL, win_rate REAL,
                total_trades INTEGER, params TEXT,
                created_at TEXT DEFAULT (datetime('now')))""")
            conn.commit()

    def save_daily(self, df: pd.DataFrame):
        if df.empty:
            return
        col_map = {"股票代码": "code", "日期": "date", "开盘": "open", "最高": "high",
                   "最低": "low", "收盘": "close", "成交量": "volume", "成交额": "amount",
                   "振幅": "amplitude", "涨跌幅": "pct_change", "换手率": "turnover"}
        df_db = df.rename(columns=col_map)
        expected_cols = ["code", "date", "open", "high", "low", "close", "volume", "amount"]
        cols = [c for c in expected_cols if c in df_db.columns]
        extra_cols = [c for c in ["amplitude", "pct_change", "turnover"] if c in df_db.columns]
        all_cols = cols + extra_cols
        with self._get_conn() as conn:
            for _, row in df_db[all_cols].iterrows():
                placeholders = ",".join(["?"] * len(all_cols))
                cols_str = ",".join(all_cols)
                sql = f"INSERT OR REPLACE INTO stock_daily ({cols_str}) VALUES ({placeholders})"
                conn.execute(sql, tuple(row[col] for col in all_cols))
            conn.commit()

    def load_daily(self, code: str, start: str = None, end: str = None) -> pd.DataFrame:
        sql = "SELECT * FROM stock_daily WHERE code = ?"
        params = [code]
        if start:
            sql += " AND date >= ?"
            params.append(start)
        if end:
            sql += " AND date <= ?"
            params.append(end)
        sql += " ORDER BY date ASC"
        with self._get_conn() as conn:
            return pd.read_sql_query(sql, conn, params=params)

    def save_stock_info(self, df: pd.DataFrame):
        col_map = {"股票代码": "code", "股票简称": "name", "行业": "industry",
                   "市场类型": "market", "上市时间": "list_date",
                   "总市值": "total_mv", "流通市值": "circ_mv", "市盈率-动态": "pe", "市净率": "pb"}
        df_db = df.rename(columns=col_map)
        cols = [c for c in ["code", "name", "industry", "market", "list_date",
                             "total_mv", "circ_mv", "pe", "pb"] if c in df_db.columns]
        with self._get_conn() as conn:
            for _, row in df_db[cols].iterrows():
                placeholders = ",".join(["?"] * len(cols))
                cols_str = ",".join(cols)
                sql = f"INSERT OR REPLACE INTO stock_info ({cols_str}) VALUES ({placeholders})"
                conn.execute(sql, tuple(row[col] for col in cols))
            conn.commit()

    def get_stock_list(self, industry: str = None) -> pd.DataFrame:
        sql = "SELECT * FROM stock_info"
        params = []
        if industry:
            sql += " WHERE industry = ?"
            params.append(industry)
        with self._get_conn() as conn:
            return pd.read_sql_query(sql, conn, params=params)


class DataFetcher:
    def __init__(self):
        self.cache = CacheManager()
        self.db = Database()

    def fetch_stock_list(self, force_refresh: bool = False) -> pd.DataFrame:
        if not force_refresh:
            cached = self.cache.get("stock_list", max_age_hours=24)
            if cached is not None:
                return cached
        ak = _ensure_akshare()
        try:
            df = ak.stock_info_a_code_name()
            df = df.rename(columns={"code": "股票代码", "name": "股票简称"})
            self.cache.set("stock_list", df)
            return df
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            return pd.DataFrame()

    def fetch_daily_kline(self, code: str, start_date: str = "20200101",
                          end_date: str = None, adjust: str = "qfq") -> pd.DataFrame:
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")
        cache_key = f"daily_{code}_{adjust}"
        cached = self.cache.get(cache_key, max_age_hours=6, start_date=start_date, end_date=end_date)
        if cached is not None:
            return cached
        ak = _ensure_akshare()
        try:
            df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date,
                                    end_date=end_date, adjust=adjust)
            if df is not None and not df.empty:
                self.db.save_daily(df)
                self.cache.set(cache_key, df, start_date=start_date, end_date=end_date)
            return df
        except Exception as e:
            logger.error(f"获取 {code} K线失败: {e}")
            db_data = self.db.load_daily(code, start_date, end_date)
            if not db_data.empty:
                return db_data
            return pd.DataFrame()

    def fetch_index_daily(self, index_code: str = "000001",
                          start_date: str = "20200101", end_date: str = None) -> pd.DataFrame:
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")
        cached = self.cache.get(f"index_{index_code}", max_age_hours=6,
                                start_date=start_date, end_date=end_date)
        if cached is not None:
            return cached
        ak = _ensure_akshare()
        try:
            # 自动判断交易所前缀：3开头=深市，其他=沪市
            prefix = "sz" if index_code.startswith("3") else "sh"
            symbol = f"{prefix}{index_code}"
            df = ak.stock_zh_index_daily(symbol=symbol)
            if df is not None and not df.empty:
                # 统一 date 列为字符串进行比较
                df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y%m%d")
                df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
                self.cache.set(f"index_{index_code}", df, start_date=start_date, end_date=end_date)
            return df
        except Exception as e:
            logger.error(f"获取指数 {index_code} 数据失败: {e}")
            return pd.DataFrame()

    def fetch_financial_data(self, code: str) -> pd.DataFrame:
        cached = self.cache.get(f"finance_{code}", max_age_hours=168)
        if cached is not None:
            return cached
        ak = _ensure_akshare()
        try:
            df = ak.stock_financial_analysis_indicator(symbol=code)
            if df is not None and not df.empty:
                self.cache.set(f"finance_{code}", df)
            return df
        except Exception as e:
            logger.error(f"获取 {code} 财务数据失败: {e}")
            return pd.DataFrame()

    def fetch_industry_stocks(self, industry: str) -> pd.DataFrame:
        ak = _ensure_akshare()
        try:
            df = ak.stock_board_industry_cons_em(symbol=industry)
            return df
        except Exception as e:
            logger.error(f"获取行业 {industry} 成分股失败: {e}")
            return pd.DataFrame()

    def fetch_industry_list(self) -> pd.DataFrame:
        ak = _ensure_akshare()
        try:
            df = ak.stock_board_industry_name_em()
            return df
        except Exception as e:
            logger.error(f"获取行业列表失败: {e}")
            return pd.DataFrame()

    def fetch_limit_up_pool(self, date: str = None) -> pd.DataFrame:
        if date is None:
            date = datetime.now().strftime("%Y%m%d")
        ak = _ensure_akshare()
        try:
            df = ak.stock_zt_pool_em(date=date)
            return df
        except Exception as e:
            logger.error(f"获取涨停板数据失败: {e}")
            return pd.DataFrame()

    def fetch_continuous_limit_up(self, date: str = None) -> pd.DataFrame:
        if date is None:
            date = datetime.now().strftime("%Y%m%d")
        ak = _ensure_akshare()
        try:
            df = ak.stock_zt_pool_strong_em(date=date)
            return df
        except Exception as e:
            logger.error(f"获取连板数据失败: {e}")
            return pd.DataFrame()

    def fetch_money_flow(self, code: str) -> pd.DataFrame:
        ak = _ensure_akshare()
        try:
            df = ak.stock_individual_fund_flow(stock=code, market="sh" if code.startswith("6") else "sz")
            return df
        except Exception as e:
            logger.error(f"获取 {code} 资金流向失败: {e}")
            return pd.DataFrame()

    def build_stock_pool(self, market: str = "全部", filter_st: bool = True,
                         filter_new: bool = True, min_days: int = 60) -> List[str]:
        stock_list = self.fetch_stock_list()
        if stock_list.empty:
            return []
        codes = stock_list["股票代码"].tolist()
        names = stock_list["股票简称"].tolist()
        filtered = []
        for code, name in zip(codes, names):
            if filter_st and ("ST" in str(name) or "退" in str(name)):
                continue
            filtered.append(code)
        return filtered

    def prepare_akquant_data(self, codes: List[str], start_date: str,
                             end_date: str = None) -> Dict[str, pd.DataFrame]:
        """准备 AKQuant 格式的多标的数据字典

        返回: {symbol: DataFrame[date, open, high, low, close, volume, symbol]}
        """
        import pandas as _pd
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")

        col_map = {"日期": "date", "开盘": "open", "收盘": "close",
                    "最高": "high", "最低": "low", "成交量": "volume", "成交额": "amount"}
        required = ["date", "open", "high", "low", "close", "volume"]
        data = {}

        for code in codes:
            try:
                df = self.fetch_daily_kline(str(code), start_date=start_date, end_date=end_date)
                if df is None or df.empty or len(df) < 20:
                    continue
                df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
                if not all(c in df.columns for c in required):
                    continue
                df = df[required + ["amount"] if "amount" in df.columns else required].copy()
                df["date"] = _pd.to_datetime(df["date"], errors="coerce")
                df = df.dropna(subset=["date"]).sort_values("date")
                df["symbol"] = str(code)
                if len(df) >= 20:
                    data[str(code)] = df[required + ["symbol"]]
            except Exception as e:
                logger.debug(f"准备 {code} AKQ数据失败: {e}")
                continue
        return data


data_fetcher = DataFetcher()
