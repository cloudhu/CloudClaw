"""
数据管理模块 - 负责数据获取、存储和缓存
使用 akshare 作为免费数据源，SQLite 作为本地存储
"""

import os
import json
import pickle
import sqlite3
import hashlib
import time
import random
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Callable
from functools import wraps

import pandas as pd
import numpy as np

from .config import DATA_DIR, DB_PATH

logger = logging.getLogger(__name__)


def retry_on_network_error(
    max_retries: int = 3,
    base_delay: float = 2.0,
    backoff: float = 2.0,
    jitter: float = 1.0,
):
    """网络请求重试装饰器，带指数退避和随机抖动
    
    处理以下异常:
    - ConnectionError / RemoteDisconnected / Timeout
    - HTTPError (429 限流等)
    """

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except ImportError:
                    raise
                except Exception as e:
                    last_exception = e
                    error_str = str(e).lower()
                    # 判断是否为可重试的网络错误
                    is_network_error = any(
                        kw in error_str
                        for kw in (
                            "connection abort",
                            "remote end closed",
                            "remote disconnected",
                            "connection refused",
                            "connection reset",
                            "timeout",
                            "too many requests",
                            "429",
                            "service unavailable",
                            "503",
                            "502",
                        )
                    )
                    if not is_network_error and attempt < max_retries:
                        # 不可重试的错误直接抛
                        raise
                    if attempt < max_retries:
                        delay = base_delay * (backoff ** attempt) + random.uniform(0, jitter)
                        logger.warning(
                            f"请求失败(第{attempt + 1}/{max_retries}次重试)，"
                            f"{delay:.1f}秒后重试: {e}"
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"请求重试{max_retries}次后仍然失败: {e}")
                        raise
            raise last_exception

        return wrapper

    return decorator

AK_AVAILABLE = False
_last_request_time = 0.0
_RATE_LIMIT_MIN_INTERVAL = 0.8  # 最小请求间隔（秒），避免触发远端限流
_http_session = None


def _get_http_session():
    """获取/创建带重试和浏览器 headers 的 requests Session"""
    global _http_session
    if _http_session is None:
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        _http_session = requests.Session()
        retry = Retry(total=2, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
        _http_session.mount("http://", adapter)
        _http_session.mount("https://", adapter)
        _http_session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        })
        _http_session.timeout = 15
    return _http_session


def _rate_limit():
    """全局限流：确保两次 HTTP 请求间隔不少于 _RATE_LIMIT_MIN_INTERVAL 秒"""
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < _RATE_LIMIT_MIN_INTERVAL:
        time.sleep(_RATE_LIMIT_MIN_INTERVAL - elapsed + random.uniform(0, 0.3))
    _last_request_time = time.time()


def _ensure_akshare():
    """获取 akshare 模块，同时配置健壮的 HTTP 会话"""
    global AK_AVAILABLE
    if not AK_AVAILABLE:
        try:
            import akshare as ak
            _configure_akshare_session(ak)
            AK_AVAILABLE = True
            return ak
        except ImportError:
            raise ImportError("请安装 akshare: pip install akshare")
    import akshare as ak
    return ak


def _configure_akshare_session(ak):
    """为 akshare 配置更健壮的 requests 会话，处理 SSL/连接问题"""
    try:
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        retry_strategy = Retry(
            total=2,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=10,
        )

        if hasattr(ak, "requests"):
            ak_session = ak.requests
        else:
            ak_session = requests.Session()

        ak_session.mount("http://", adapter)
        ak_session.mount("https://", adapter)
        ak_session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        })
        logger.debug("AKShare HTTP 会话已配置")
    except Exception as e:
        logger.debug(f"AKShare 会话配置跳过: {e}")


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
        # 过滤掉缺少 code/date 的行
        if "code" not in df_db.columns or "date" not in df_db.columns:
            logger.warning(f"save_daily: 缺少 code/date 列，可用列: {list(df_db.columns)}")
            return
        df_db = df_db.dropna(subset=["code", "date"])
        if df_db.empty:
            return
        expected_cols = ["code", "date", "open", "high", "low", "close", "volume", "amount"]
        cols = [c for c in expected_cols if c in df_db.columns]
        extra_cols = [c for c in ["amplitude", "pct_change", "turnover"] if c in df_db.columns]
        all_cols = cols + extra_cols
        with self._get_conn() as conn:
            for _, row in df_db[all_cols].iterrows():
                values = tuple(
                    row[col] if not pd.isna(row[col]) else None
                    for col in all_cols
                )
                placeholders = ",".join(["?"] * len(all_cols))
                cols_str = ",".join(all_cols)
                sql = f"INSERT OR REPLACE INTO stock_daily ({cols_str}) VALUES ({placeholders})"
                conn.execute(sql, values)
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
        self._unavailable_sources: set = set()  # 已知不可用的数据源，跳过重试浪费时间

    # ═══ 私有网络请求方法（异常传播到 retry 装饰器） ═══

    @retry_on_network_error(max_retries=3, base_delay=2.0)
    def _fetch_stock_list_raw(self):
        ak = _ensure_akshare()
        _rate_limit()
        return ak.stock_info_a_code_name()

    @retry_on_network_error(max_retries=2, base_delay=1.0)
    def _fetch_eastmoney_kline(self, code: str, start_date: str, end_date: str, adjust: str):
        """东方财富日K线（主数据源）"""
        ak = _ensure_akshare()
        _rate_limit()
        return ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date,
                                  end_date=end_date, adjust=adjust)

    @retry_on_network_error(max_retries=3, base_delay=2.0)
    def _fetch_sina_kline(self, code: str, start_date: str, end_date: str, adjust: str):
        """新浪日K线（备选数据源）"""
        ak = _ensure_akshare()
        prefix = "sh" if code.startswith(("6", "9")) else "sz"
        _rate_limit()
        df = ak.stock_zh_a_daily(symbol=f"{prefix}{code}", adjust=adjust)
        if df is not None and not df.empty:
            # 添加股票代码列（新浪数据不含此列）
            df["股票代码"] = code
            # 统一列名
            rename_map = {}
            for src, dst in [("date", "日期"), ("open", "开盘"), ("high", "最高"),
                              ("low", "最低"), ("close", "收盘"), ("volume", "成交量")]:
                if src in df.columns:
                    rename_map[src] = dst
            if rename_map:
                df = df.rename(columns=rename_map)
            # 统一日期格式，兼容 datetime.date / str / timestamp
            if "日期" in df.columns:
                df["日期"] = pd.to_datetime(df["日期"], errors="coerce").dt.strftime("%Y%m%d")
                mask = (df["日期"] >= start_date) & (df["日期"] <= end_date)
                df = df[mask]
        return df

    @retry_on_network_error(max_retries=3, base_delay=2.0)
    def _fetch_tencent_kline(self, code: str, start_date: str, end_date: str, adjust: str):
        """腾讯财经日K线（akshare 封装）"""
        ak = _ensure_akshare()
        prefix = "sh" if code.startswith(("6", "9")) else "sz"
        _rate_limit()
        df = ak.stock_zh_a_hist_tx(symbol=f"{prefix}{code}", start_date=start_date,
                                    end_date=end_date, adjust=adjust)
        if df is not None and not df.empty:
            df["股票代码"] = code
            col_map = {"date": "日期", "open": "开盘", "close": "收盘",
                        "high": "最高", "low": "最低", "amount": "成交额"}
            if "volume" in df.columns:
                col_map["volume"] = "成交量"
            df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        return df

    @retry_on_network_error(max_retries=3, base_delay=1.5)
    def _fetch_tencent_kline_direct(self, code: str, start_date: str, end_date: str, adjust: str):
        """腾讯财经日K线（直连 HTTP，不依赖 akshare）"""
        import json as _json
        _rate_limit()
        prefix = "sh" if code.startswith(("6", "9")) else "sz"
        symbol = f"{prefix}{code}"
        fqtype = "qfq" if adjust == "qfq" else ""
        url = f"http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={symbol},day,,,320,{fqtype}"

        session = _get_http_session()
        r = session.get(url, timeout=10)
        data = _json.loads(r.text)
        if data.get("code") != 0:
            raise RuntimeError(f"腾讯API返回错误: {data.get('msg', '未知')}")

        kline_key = f"{fqtype}day" if fqtype else "day"
        raw_klines = data.get("data", {}).get(symbol, {}).get(kline_key, [])
        if not raw_klines:
            return pd.DataFrame()

        rows = []
        for row in raw_klines:
            date_str = str(row[0])
            if date_str < start_date or date_str > end_date:
                continue
            rows.append({
                "日期": date_str, "开盘": float(row[1]), "收盘": float(row[2]),
                "最高": float(row[3]), "最低": float(row[4]), "成交量": float(row[5]),
                "股票代码": code,
            })
        return pd.DataFrame(rows)

    @retry_on_network_error(max_retries=3, base_delay=1.5)
    def _fetch_sina_kline_direct(self, code: str, start_date: str, end_date: str, adjust: str):
        """新浪财经日K线（直连 HTTP，备选 URL）"""
        import json as _json
        _rate_limit()
        prefix = "sh" if code.startswith(("6", "9")) else "sz"
        symbol = f"{prefix}{code}"

        session = _get_http_session()
        # 新浪 K 线 API（日线 scale=240）
        url = f"https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData?symbol={symbol}&scale=240&ma=no&datalen=1024"
        r = session.get(url, timeout=10)
        raw = _json.loads(r.text)
        if not raw or not isinstance(raw, list):
            return pd.DataFrame()

        rows = []
        for item in raw:
            date_str = item.get("day", "")
            if date_str < start_date or date_str > end_date:
                continue
            rows.append({
                "日期": date_str,
                "开盘": float(item["open"]), "收盘": float(item["close"]),
                "最高": float(item["high"]), "最低": float(item["low"]),
                "成交量": float(item["volume"]),
                "股票代码": code,
            })
        return pd.DataFrame(rows)

    @retry_on_network_error(max_retries=3, base_delay=2.0)
    def _fetch_index_daily_raw(self, symbol: str):
        ak = _ensure_akshare()
        _rate_limit()
        return ak.stock_zh_index_daily(symbol=symbol)

    @retry_on_network_error(max_retries=3, base_delay=2.0)
    def _fetch_index_daily_tx(self, symbol: str):
        """腾讯财经指数日线（akshare 封装）"""
        ak = _ensure_akshare()
        _rate_limit()
        return ak.stock_zh_index_daily_tx(symbol=symbol)

    @retry_on_network_error(max_retries=3, base_delay=1.5)
    def _fetch_index_daily_direct(self, symbol: str):
        """腾讯财经指数日线（直连 HTTP）"""
        import json as _json
        _rate_limit()
        session = _get_http_session()
        url = f"http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={symbol},day,,,10000,"
        r = session.get(url, timeout=10)
        data = _json.loads(r.text)
        raw = data.get("data", {}).get(symbol, {}).get("day", [])
        if not raw:
            return pd.DataFrame()
        rows = []
        for row in raw:
            rows.append({"date": str(row[0]), "open": float(row[1]),
                          "close": float(row[2]), "high": float(row[3]),
                          "low": float(row[4]), "volume": float(row[5])})
        return pd.DataFrame(rows)

    @retry_on_network_error(max_retries=3, base_delay=2.0)
    def _fetch_stock_list_direct(self):
        """东方财富股票列表（直连 HTTP，不依赖 akshare）"""
        import json as _json
        _rate_limit()
        session = _get_http_session()
        url = (
            "http://82.push2.eastmoney.com/api/qt/clist/get?"
            "pn=1&pz=6000&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281"
            "&fltt=2&invt=2&fid=f3"
            "&fs=m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23"
            "&fields=f2,f12,f14"
        )
        r = session.get(url, timeout=15)
        data = _json.loads(r.text)
        items = data.get("data", {}).get("diff", [])
        rows = [{"code": str(i["f12"]), "name": i["f14"]} for i in items]
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.rename(columns={"code": "股票代码", "name": "股票简称"})
        return df

    @retry_on_network_error(max_retries=2, base_delay=3.0)
    def _fetch_financial_raw(self, code: str):
        ak = _ensure_akshare()
        _rate_limit()
        return ak.stock_financial_analysis_indicator(symbol=code)

    @retry_on_network_error(max_retries=2, base_delay=2.0)
    def _fetch_industry_stocks_raw(self, industry: str):
        ak = _ensure_akshare()
        _rate_limit()
        return ak.stock_board_industry_cons_em(symbol=industry)

    @retry_on_network_error(max_retries=2, base_delay=2.0)
    def _fetch_industry_list_raw(self):
        ak = _ensure_akshare()
        _rate_limit()
        return ak.stock_board_industry_name_em()

    @retry_on_network_error(max_retries=2, base_delay=2.0)
    def _fetch_limit_up_raw(self, date: str):
        ak = _ensure_akshare()
        _rate_limit()
        return ak.stock_zt_pool_em(date=date)

    @retry_on_network_error(max_retries=2, base_delay=2.0)
    def _fetch_continuous_limit_raw(self, date: str):
        ak = _ensure_akshare()
        _rate_limit()
        return ak.stock_zt_pool_strong_em(date=date)

    @retry_on_network_error(max_retries=2, base_delay=1.5)
    def _fetch_money_flow_raw(self, code: str, market: str):
        ak = _ensure_akshare()
        _rate_limit()
        return ak.stock_individual_fund_flow(stock=code, market=market)

    @retry_on_network_error(max_retries=2, base_delay=2.0)
    def _fetch_limit_down_raw(self, date: str):
        ak = _ensure_akshare()
        _rate_limit()
        return ak.stock_zt_pool_dtgc_em(date=date)

    @retry_on_network_error(max_retries=2, base_delay=1.5)
    def _fetch_market_activity_raw(self):
        ak = _ensure_akshare()
        _rate_limit()
        return ak.stock_market_activity_legu()

    @retry_on_network_error(max_retries=2, base_delay=1.5)
    def _fetch_szse_summary_raw(self):
        ak = _ensure_akshare()
        _rate_limit()
        return ak.stock_szse_summary()

    @retry_on_network_error(max_retries=2, base_delay=1.5)
    def _fetch_sse_summary_raw(self):
        ak = _ensure_akshare()
        _rate_limit()
        return ak.stock_sse_summary()

    # ═══ 公共方法：缓存 + 降级 ═══

    def fetch_stock_list(self, force_refresh: bool = False) -> pd.DataFrame:
        """获取股票列表，回退链: akshare → 东财直连HTTP"""
        if not force_refresh:
            cached = self.cache.get("stock_list", max_age_hours=24)
            if cached is not None:
                return cached

        # 优先 akshare，失败则直连东财
        sources = [
            ("akshare", lambda: self._fetch_stock_list_raw()),
            ("东财直连", lambda: self._fetch_stock_list_direct()),
        ]

        for label, fetcher in sources:
            if label in self._unavailable_sources:
                continue
            try:
                df = fetcher()
                if df is not None and not df.empty:
                    if label == "akshare":
                        df = df.rename(columns={"code": "股票代码", "name": "股票简称"})
                    if label != sources[0][0]:
                        logger.info(f"  → 股票列表使用 {label} 数据源")
                    self.cache.set("stock_list", df)
                    return df
            except Exception as e:
                logger.warning(f"  股票列表 [{label}] 不可用({type(e).__name__})，标记跳过")
                self._unavailable_sources.add(label)

    def fetch_daily_kline(self, code: str, start_date: str = "20200101",
                          end_date: str = None, adjust: str = "qfq") -> pd.DataFrame:
        """获取日K线，5级回退链:
        东方财富(akshare) → 新浪(akshare) → 腾讯(akshare) → 腾讯(直连) → 新浪(直连) → 本地DB
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")
        cache_key = f"daily_{code}_{adjust}"
        cached = self.cache.get(cache_key, max_age_hours=6, start_date=start_date, end_date=end_date)
        if cached is not None:
            return cached

        # 尝试各数据源，成功后写缓存/DB 并返回
        sources = [
            ("东方财富", lambda: self._fetch_eastmoney_kline(code, start_date, end_date, adjust)),
            ("新浪", lambda: self._fetch_sina_kline(code, start_date, end_date, adjust)),
            ("腾讯", lambda: self._fetch_tencent_kline(code, start_date, end_date, adjust)),
            ("腾讯直连", lambda: self._fetch_tencent_kline_direct(code, start_date, end_date, adjust)),
            ("新浪直连", lambda: self._fetch_sina_kline_direct(code, start_date, end_date, adjust)),
        ]

        for label, fetcher in sources:
            if label in self._unavailable_sources:
                continue  # 已知失效，跳过
            try:
                df = fetcher()
                if df is not None and not df.empty:
                    if label != sources[0][0]:
                        logger.info(f"  → '{code}' 使用 {label} 数据源")
                    self.db.save_daily(df)
                    self.cache.set(cache_key, df, start_date=start_date, end_date=end_date)
                    return df
            except Exception as e:
                logger.warning(f"  {label} 数据源不可用({type(e).__name__})，标记跳过")
                self._unavailable_sources.add(label)

        # 所有远程源都失败 → 降级到本地 SQLite
        logger.error(f"获取 {code} K线失败（所有数据源均不可达）")
        db_data = self.db.load_daily(code, start_date, end_date)
        if not db_data.empty:
            logger.info(f"  → 从本地数据库回退，获取 {len(db_data)} 条历史记录")
            return db_data
        return pd.DataFrame()

    def fetch_index_daily(self, index_code: str = "000001",
                          start_date: str = "20200101", end_date: str = None) -> pd.DataFrame:
        """获取指数日线，3级回退链: 东方财富 → 腾讯(akshare) → 腾讯(直连)"""
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")
        cached = self.cache.get(f"index_{index_code}", max_age_hours=6,
                                start_date=start_date, end_date=end_date)
        if cached is not None:
            return cached

        prefix = "sz" if index_code.startswith("3") else "sh"
        symbol = f"{prefix}{index_code}"

        sources = [
            ("东方财富", lambda: self._fetch_index_daily_raw(symbol)),
            ("腾讯", lambda: self._fetch_index_daily_tx(symbol)),
            ("腾讯直连", lambda: self._fetch_index_daily_direct(symbol)),
        ]

        for label, fetcher in sources:
            if label in self._unavailable_sources:
                continue
            try:
                df = fetcher()
                if df is not None and not df.empty:
                    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y%m%d")
                    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
                    if df.empty:
                        continue
                    if label != sources[0][0]:
                        logger.info(f"  → 指数 {index_code} 使用 {label} 数据源")
                    self.cache.set(f"index_{index_code}", df, start_date=start_date, end_date=end_date)
                    return df
            except Exception as e:
                logger.warning(f"  指数 {index_code} [{label}] 不可用({type(e).__name__})，标记跳过")
                self._unavailable_sources.add(label)

        logger.error(f"获取指数 {index_code} 数据失败（所有数据源均不可达）")
        return pd.DataFrame()

    def fetch_financial_data(self, code: str) -> pd.DataFrame:
        cached = self.cache.get(f"finance_{code}", max_age_hours=168)
        if cached is not None:
            return cached
        try:
            df = self._fetch_financial_raw(code)
            if df is not None and not df.empty:
                self.cache.set(f"finance_{code}", df)
            return df
        except Exception as e:
            logger.error(f"获取 {code} 财务数据失败: {e}")
            return pd.DataFrame()

    def fetch_industry_stocks(self, industry: str, board_code: str = "") -> pd.DataFrame:
        from . import sector_fallback as _sf
        return _sf.fetch_stocks(self.cache, industry, board_code)

    def fetch_industry_list(self, force_refresh: bool = False) -> pd.DataFrame:
        from . import sector_fallback as _sf
        return _sf.fetch_list(self.cache, force=force_refresh)

    def fetch_limit_up_pool(self, date: str = None) -> pd.DataFrame:
        if date is None:
            date = datetime.now().strftime("%Y%m%d")
        try:
            return self._fetch_limit_up_raw(date)
        except Exception as e:
            logger.error(f"获取涨停板数据失败: {e}")
            return pd.DataFrame()

    def fetch_continuous_limit_up(self, date: str = None) -> pd.DataFrame:
        if date is None:
            date = datetime.now().strftime("%Y%m%d")
        try:
            return self._fetch_continuous_limit_raw(date)
        except Exception as e:
            logger.error(f"获取连板数据失败: {e}")
            return pd.DataFrame()

    def fetch_money_flow(self, code: str) -> pd.DataFrame:
        try:
            market = "sh" if code.startswith("6") else "sz"
            return self._fetch_money_flow_raw(code, market)
        except Exception as e:
            logger.error(f"获取 {code} 资金流向失败: {e}")
            return pd.DataFrame()

    def fetch_limit_down_pool(self, date: str = None) -> pd.DataFrame:
        """获取跌停板股票池"""
        if date is None:
            date = datetime.now().strftime("%Y%m%d")
        try:
            return self._fetch_limit_down_raw(date)
        except Exception as e:
            logger.error(f"获取跌停板数据失败: {e}")
            return pd.DataFrame()

    def fetch_market_activity(self) -> Dict[str, float]:
        """获取全市场活跃度概览（涨跌停数、涨跌家数、成交量等）

        Returns:
            dict: {上涨, 下跌, 平盘, 涨停, 真实涨停, 跌停, 真实跌停, 总市值, 成交量, ...}
        """
        try:
            df = self._fetch_market_activity_raw()
            if df is None or df.empty:
                return {}
            result = {}
            for _, row in df.iterrows():
                key = str(row["item"])
                val = row["value"]
                try:
                    result[key] = float(val)
                except (ValueError, TypeError):
                    result[key] = val
            return result
        except Exception as e:
            logger.error(f"获取市场活跃度失败: {e}")
            return {}

    def fetch_market_total_volume(self) -> Dict[str, Any]:
        """获取沪深两市总成交额与总市值

        Returns:
            dict: {sh_volume, sz_volume, total_volume, sh_pe, total_mv_sh, total_mv_sz, ...}
        """
        result = {}
        try:
            # 深交所成交额
            sz = self._fetch_szse_summary_raw()
            if sz is not None and not sz.empty:
                stock_row = sz[sz["证券类别"] == "股票"]
                if not stock_row.empty:
                    result["sz_volume"] = float(stock_row.iloc[0]["成交金额"])
                    result["sz_total_mv"] = float(stock_row.iloc[0]["总市值"])

            # 上交所总览（总市值、PE）
            sh = self._fetch_sse_summary_raw()
            if sh is not None and not sh.empty:
                items = dict(zip(sh["项目"], sh["股票"]))
                result["sh_total_mv"] = float(items.get("总市值", 0)) * 1e8  # 亿→元
                result["sh_pe"] = float(items.get("平均市盈率", 0))

            # 计算合计
            if "sz_volume" in result:
                result["total_volume"] = result["sz_volume"]
                # 用深市成交占全市场 ~45% 估算沪市
                sz_ratio = result.get("sz_ratio", 0.45)
                result["sh_volume_est"] = result["sz_volume"] * (1 - sz_ratio) / sz_ratio
                result["total_volume"] = result["sz_volume"] + result["sh_volume_est"]
        except Exception as e:
            logger.warning(f"获取两市总览部分失败: {e}")

        return result

    def fetch_batch_5day_gains(self, codes: List[str], max_workers: int = 3) -> Dict[str, float]:
        """批量获取多只股票近5个交易日涨幅（快速模式）。

        两级降级策略：
          1. 腾讯直连 K 线 API（快速，单次请求）
          2. akshare stock_zh_a_hist（据经验网络可达，仅一次尝试不重试）

        计算：(最新收盘 - 5交易日前收盘) / 5交易日前收盘 * 100

        Args:
            codes: 股票代码列表
            max_workers: 并行线程数（默认3）

        Returns:
            {code: gain_pct} 字典
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import json as _json

        result: Dict[str, float] = {}
        if not codes:
            return result

        today_str = datetime.now().strftime("%Y%m%d")
        start_str = (datetime.now() - timedelta(days=15)).strftime("%Y%m%d")

        def _calc_one(code: str):
            """快速计算单只股票五日涨幅"""
            try:
                # 检查缓存
                cached = self.cache.get(f"gain5d_{code}", max_age_hours=6)
                if cached is not None:
                    try:
                        if isinstance(cached, pd.DataFrame) and "gain_5d" in cached.columns:
                            return code, round(float(cached["gain_5d"].iloc[0]), 2)
                        return code, round(float(cached), 2)
                    except (TypeError, ValueError):
                        pass

                closes = None

                # 1. 腾讯直连 K 线（快）
                try:
                    prefix = "sh" if str(code).startswith(("6", "9")) else "sz"
                    symbol = f"{prefix}{code}"
                    session = _get_http_session()
                    url = f"http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={symbol},day,,,15,qfq"
                    r = session.get(url, timeout=8)
                    data = _json.loads(r.text)
                    if data.get("code") == 0:
                        raw = data.get("data", {}).get(symbol, {}).get("qfqday", [])
                        if not raw:
                            raw = data.get("data", {}).get(symbol, {}).get("day", [])
                        if raw and len(raw) >= 6:
                            valid = [row for row in raw if start_str <= str(row[0]) <= today_str]
                            if len(valid) >= 6:
                                closes = [float(row[2]) for row in valid]
                except Exception:
                    pass

                # 2. akshare stock_zh_a_hist（据经验网络可达，仅一层）
                if closes is None:
                    try:
                        ak = _ensure_akshare()
                        _rate_limit()
                        df = ak.stock_zh_a_hist(symbol=code, period="daily",
                                                start_date=start_str, end_date=today_str,
                                                adjust="qfq")
                        if df is not None and not df.empty and len(df) >= 6:
                            col_map = {"日期": "date", "收盘": "close"}
                            df2 = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
                            if "close" in df2.columns:
                                df2 = df2.sort_values("date")
                                closes = [float(v) for v in df2["close"].values[-6:]]
                    except Exception:
                        pass

                if closes is None or len(closes) < 6:
                    return code, 0.0

                latest = closes[-1]
                prev = closes[0]  # 6条数据：[... 5天前, 4天前, 3天前, 2天前, 昨天, 今天]，index 0 = 5天前
                if prev > 0:
                    gain = round((latest / prev - 1) * 100, 2)
                else:
                    gain = 0.0

                # 缓存
                cache_df = pd.DataFrame({"gain_5d": [gain]})
                self.cache.set(f"gain5d_{code}", cache_df)
                return code, gain

            except Exception:
                return code, 0.0

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_calc_one, code): code for code in codes}
            for f in as_completed(futures):
                try:
                    code, gain = f.result()
                    result[code] = gain
                except Exception:
                    pass

        return result

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
