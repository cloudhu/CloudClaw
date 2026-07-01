"""
数据管理模块 - 负责数据获取、存储和缓存
使用 akshare 作为免费数据源，SQLite 作为本地存储
"""

import hashlib
import logging
import os
import pickle
import random
import sqlite3
import time
from collections.abc import Callable
from datetime import datetime, timedelta
from functools import wraps

import pandas as pd

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
                        delay = base_delay * (backoff**attempt) + random.uniform(0, jitter)
                        logger.warning("请求失败(第%d/%d次重试)，%.1f秒后重试: %s", attempt + 1, max_retries, delay, e)
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
        _http_session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
            }
        )
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
    if not AK_AVAILABLE:
        try:
            import akshare as _ak

            _configure_akshare_session(_ak)
            # 首轮导入成功，设置全局可用标记
            globals()["AK_AVAILABLE"] = True
            return _ak
        except ImportError as err:
            raise ImportError("请安装 akshare: pip install akshare") from err
    import akshare as _ak

    return _ak


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
        ak_session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
            }
        )
        logger.debug("AKShare HTTP 会话已配置")
    except Exception as e:
        logger.debug(f"AKShare 会话配置跳过: {e}")


class CacheManager:
    def __init__(self, cache_dir: str | None = None):
        self.cache_dir = cache_dir or os.path.join(DATA_DIR, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)

    def _cache_key(self, prefix: str, **kwargs) -> str:
        raw = f"{prefix}_{sorted(kwargs.items())}"
        return hashlib.md5(raw.encode()).hexdigest() + ".pkl"

    def get(self, prefix: str, max_age_hours: int = 24, **kwargs) -> pd.DataFrame | None:
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
    def __init__(self, db_path: str | None = None):
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
        col_map = {
            "股票代码": "code",
            "日期": "date",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume",
            "成交额": "amount",
            "振幅": "amplitude",
            "涨跌幅": "pct_change",
            "换手率": "turnover",
        }
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
                values = tuple(row[col] if not pd.isna(row[col]) else None for col in all_cols)
                placeholders = ",".join(["?"] * len(all_cols))
                cols_str = ",".join(all_cols)
                sql = f"INSERT OR REPLACE INTO stock_daily ({cols_str}) VALUES ({placeholders})"
                conn.execute(sql, values)
            conn.commit()

    def load_daily(self, code: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
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
        col_map = {
            "股票代码": "code",
            "股票简称": "name",
            "行业": "industry",
            "市场类型": "market",
            "上市时间": "list_date",
            "总市值": "total_mv",
            "流通市值": "circ_mv",
            "市盈率-动态": "pe",
            "市净率": "pb",
        }
        df_db = df.rename(columns=col_map)
        cols = [
            c
            for c in ["code", "name", "industry", "market", "list_date", "total_mv", "circ_mv", "pe", "pb"]
            if c in df_db.columns
        ]
        with self._get_conn() as conn:
            for _, row in df_db[cols].iterrows():
                placeholders = ",".join(["?"] * len(cols))
                cols_str = ",".join(cols)
                sql = f"INSERT OR REPLACE INTO stock_info ({cols_str}) VALUES ({placeholders})"
                conn.execute(sql, tuple(row[col] for col in cols))
            conn.commit()

    def get_stock_list(self, industry: str | None = None) -> pd.DataFrame:
        sql = "SELECT * FROM stock_info"
        params = []
        if industry:
            sql += " WHERE industry = ?"
            params.append(industry)
        with self._get_conn() as conn:
            return pd.read_sql_query(sql, conn, params=params)


class DataFetcher:
    # 数据源不可用冷却时间（秒），超时后自动恢复重试
    SOURCE_COOLDOWN_SECONDS: float = 300.0

    def __init__(self):
        self.cache = CacheManager()
        self.db = Database()
        # 格式: {source_label: blocked_until_timestamp}
        self._unavailable_sources: dict[str, float] = {}
        # 收费数据源管理器（懒加载）
        self._ds_manager = None

    @property
    def ds_manager(self):
        """获取数据源管理器（懒加载，避免循环引用）"""
        if self._ds_manager is None:
            from .datasource.manager import DataSourceManager

            self._ds_manager = DataSourceManager()
            self._ds_manager.initialize()
        return self._ds_manager

    def _try_premium_datasource(self, method: str, *args, **kwargs) -> tuple[pd.DataFrame | None, str]:
        """尝试通过收费数据源获取数据

        返回: (DataFrame | None, 来源名称)
        若 DataFrame 为 None 表示收费源不可用/返回空，应降级到免费源
        """
        from .datasource.base import FetchResult

        try:
            mgr = self.ds_manager
            result: FetchResult = mgr._try_fetch(method, *args, **kwargs)
            if result.success and result.data is not None and not result.data.empty:
                logger.info(f"  ✓ 使用 {result.source_name} 获取数据")
                return result.data, result.source_name
            # 降级到免费源的信号
            if result.source_name == "free":
                return None, "降级到免费源"
        except Exception as e:
            logger.debug(f"收费数据源尝试失败: {e}")

        return None, ""

    def _is_source_available(self, label: str) -> bool:
        """检查数据源是否可用（冷却期自动过期）"""
        if label not in self._unavailable_sources:
            return True
        blocked_until = self._unavailable_sources[label]
        if time.time() > blocked_until:
            # 冷却期已过，自动恢复
            del self._unavailable_sources[label]
            logger.info(f"  ♻ 数据源 '{label}' 冷却期已过，恢复重试")
            return True
        remaining = int(blocked_until - time.time())
        logger.debug(f"  数据源 '{label}' 冷却中（剩余 {remaining} 秒）")
        return False

    def _mark_source_unavailable(self, label: str, error_type: str = ""):
        """标记数据源为暂时不可用，进入冷却期"""
        blocked_until = time.time() + self.SOURCE_COOLDOWN_SECONDS
        self._unavailable_sources[label] = blocked_until
        cooldown_min = self.SOURCE_COOLDOWN_SECONDS / 60
        logger.warning(f"  数据源 [{label}] 不可用({error_type})，冷却 {cooldown_min:.0f} 分钟")

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
        return ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, end_date=end_date, adjust=adjust)

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
            for src, dst in [
                ("date", "日期"),
                ("open", "开盘"),
                ("high", "最高"),
                ("low", "最低"),
                ("close", "收盘"),
                ("volume", "成交量"),
            ]:
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
        df = ak.stock_zh_a_hist_tx(symbol=f"{prefix}{code}", start_date=start_date, end_date=end_date, adjust=adjust)
        if df is not None and not df.empty:
            df["股票代码"] = code
            col_map = {
                "date": "日期",
                "open": "开盘",
                "close": "收盘",
                "high": "最高",
                "low": "最低",
                "amount": "成交额",
            }
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
            rows.append(
                {
                    "日期": date_str,
                    "开盘": float(row[1]),
                    "收盘": float(row[2]),
                    "最高": float(row[3]),
                    "最低": float(row[4]),
                    "成交量": float(row[5]),
                    "股票代码": code,
                }
            )
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
            rows.append(
                {
                    "日期": date_str,
                    "开盘": float(item["open"]),
                    "收盘": float(item["close"]),
                    "最高": float(item["high"]),
                    "最低": float(item["low"]),
                    "成交量": float(item["volume"]),
                    "股票代码": code,
                }
            )
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
            rows.append(
                {
                    "date": str(row[0]),
                    "open": float(row[1]),
                    "close": float(row[2]),
                    "high": float(row[3]),
                    "low": float(row[4]),
                    "volume": float(row[5]),
                }
            )
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

    @retry_on_network_error(max_retries=3, base_delay=2.0)
    def _fetch_stock_list_tencent_direct(self):
        """腾讯财经股票列表（直连 HTTP，第三级回退）"""
        import re

        _rate_limit()
        session = _get_http_session()
        url = "http://smartbox.gtimg.cn/s3/?q=stock&t=all&c=0&o=1&p=1&ps=6000"
        r = session.get(url, timeout=15)
        text = r.text
        # 格式: v_hint="...^股票代码~股票名称^..."
        rows = []
        try:
            # 提取 v_hint 中的股票列表
            hint_match = re.search(r'v_hint="([^"]*)"', text)
            if hint_match:
                items = hint_match.group(1).split("^")
                for item in items:
                    parts = item.strip().split("~")
                    if len(parts) >= 2:
                        code = parts[0].strip()
                        name = parts[1].strip()
                        if code.isdigit() and len(code) == 6:
                            rows.append({"code": code, "name": name})
        except Exception:
            pass

        if not rows:
            # 备选：尝试另一种腾讯接口格式
            try:
                url2 = "http://smartbox.gtimg.cn/s3/?q=stock&t=all&c=0&o=1&p=1&ps=6000&v=2"
                r2 = session.get(url2, timeout=15)
                text2 = r2.text
                hint_match2 = re.search(r'"(s[hz]\d{6}~[^~]+)"', text2)
                if hint_match2:
                    # 格式: sh600519~贵州茅台
                    for m in re.finditer(r"(s[hz]\d{6})~([^\^]+)", text2):
                        code = m.group(1)[2:]  # 去掉 sh/sz 前缀
                        name = m.group(2).strip()
                        if code.isdigit() and len(code) == 6:
                            rows.append({"code": code, "name": name})
            except Exception:
                pass

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
    def _fetch_financial_abstract_raw(self, code: str):
        """AKShare 备选财务接口 - stock_financial_abstract"""
        ak = _ensure_akshare()
        _rate_limit()
        return ak.stock_financial_abstract_ths(symbol=code)

    @retry_on_network_error(max_retries=3, base_delay=2.0)
    def _fetch_financial_eastmoney_direct(self, code: str):
        """东方财富 HTTP 直连获取财务指标（PE/PB/ROE/市值等）

        使用东方财富个股基本面数据接口，不依赖 akshare。
        """
        import json as _json

        _rate_limit()
        session = _get_http_session()
        prefix = "SH" if code.startswith(("6", "9")) else "SZ"
        # 深市 secid 为 0.{code}，沪市为 1.{code}
        secid = f"0.{code}" if prefix == "SZ" else f"1.{code}"

        # 东方财富个股基本面接口
        url = (
            f"https://emweb.securities.eastmoney.com/PC_HSF10/FinanceSummary/FinanceSummary"
            f"?code={prefix}{code}"
            f"&type=web"
        )
        try:
            r = session.get(url, timeout=10)
        except Exception:
            # 备选 URL 格式
            url = f"http://emweb.securities.eastmoney.com/ProfitForecast/Index?type=web&code={prefix}{code}"
            r = session.get(url, timeout=10)

        # 解析 HTML 中内嵌的 JSON 数据
        text = r.text
        result = {}

        # 尝试从页面中提取核心财务数据
        try:
            import re

            # 匹配 __NUXT__ 或 window.__INITIAL_STATE__ 等常见数据注入模式
            match = re.search(r"window\.__NUXT__\s*=\s*({.*?});\s*</script>", text, re.DOTALL)
            if not match:
                match = re.search(r"__NUXT__\s*=\s*({.*?});\s*</script>", text, re.DOTALL)
            if not match:
                match = re.search(r"window\.__INITIAL_STATE__\s*=\s*({.*?});\s*</script>", text, re.DOTALL)
            if match:
                data = _json.loads(match.group(1))

                # 递归查找财务数据
                def _find_fin_data(d, depth=0):
                    if depth > 8:
                        return
                    if isinstance(d, dict):
                        for key in ["finance", "financial", "f10", "quotation", "quote", "stock"]:
                            if key in d:
                                _find_fin_data(d[key], depth + 1)
                        if any(k in d for k in ["pe", "PB", "ROE", "总市值"]):
                            result.update(d)
                    elif isinstance(d, list) and d:
                        _find_fin_data(d[0], depth + 1)

                _find_fin_data(data)
        except Exception:
            pass

        # 如果 HTML 解析失败，使用备选 API 接口
        if not result:
            try:
                url2 = (
                    f"http://push2.eastmoney.com/api/qt/stock/get"
                    f"?secid={secid}"
                    f"&fields=f9,f20,f21,f22,f23,f24,f25,f37,f38,f39,f40,f41,f42,f43,f44,f45,f46,f48,f49,f50,f55,f57,f58,f59,f60,f115,f116,f117,f124,f128,f129,f130,f131,f132,f133,f148,f149,f152,f162,f168,f170,f171,f183,f184,f185,f186,f187,f188,f189,f190,f191,f192,f193"
                )
                r2 = session.get(url2, timeout=8)
                qt_data = _json.loads(r2.text).get("data", {})
                if qt_data:
                    result = {
                        "市盈率": qt_data.get("f9"),  # PE(TTM)
                        "市净率": qt_data.get("f23"),  # PB
                        "总市值": qt_data.get("f20"),  # 总市值
                        "流通市值": qt_data.get("f21"),  # 流通市值
                        "营业收入": qt_data.get("f44"),  # 总营收
                        "净利润": qt_data.get("f45"),  # 净利润
                        "roe": qt_data.get("f37"),  # ROE(%)
                        "毛利率": qt_data.get("f49"),  # 毛利率
                        "净利率": qt_data.get("f50"),  # 净利率
                        "资产负债率": qt_data.get("f46"),  # 负债率
                        "净利润增长率": qt_data.get("f43"),  # 净利润同比
                        "营业收入增长率": qt_data.get("f41"),  # 营收同比
                        "股息率": qt_data.get("f57"),  # 股息率
                    }
            except Exception:
                pass

        if not result:
            return pd.DataFrame()

        # 构造 DataFrame（与 AKShare 格式兼容）
        row = {k: v for k, v in result.items() if v is not None and v != "-" and v != ""}
        if not row:
            return pd.DataFrame()
        return pd.DataFrame([row])

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

    @retry_on_network_error(max_retries=3, base_delay=1.5)
    def _fetch_money_flow_direct(self, code: str):
        """东方财富 HTTP 直连获取资金流向数据

        使用东方财富个股资金流向接口，返回主力净流入等数据。
        """
        import json as _json

        _rate_limit()
        session = _get_http_session()
        prefix = "SH" if code.startswith(("6", "9")) else "SZ"
        secid = f"0.{code}" if prefix == "SZ" else f"1.{code}"

        # 东方财富资金流向接口
        url = (
            f"http://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get"
            f"?secid={secid}"
            f"&fields1=f1,f2,f3,f7"
            f"&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65"
            f"&lmt=30"
            f"&klt=101"  # 日线
        )
        r = session.get(url, timeout=8)
        data = _json.loads(r.text)

        klines = data.get("data", {}).get("klines", [])
        if not klines:
            return pd.DataFrame()

        rows = []
        for line in klines:
            parts = line.split(",")
            if len(parts) >= 12:
                rows.append(
                    {
                        "日期": parts[0],
                        "主力净流入": float(parts[1]) if parts[1] != "-" else 0,
                        "小单净流入": float(parts[2]) if parts[2] != "-" else 0,
                        "中单净流入": float(parts[3]) if parts[3] != "-" else 0,
                        "大单净流入": float(parts[4]) if parts[4] != "-" else 0,
                        "超大单净流入": float(parts[5]) if parts[5] != "-" else 0,
                    }
                )

        return pd.DataFrame(rows) if rows else pd.DataFrame()

    @retry_on_network_error(max_retries=2, base_delay=1.0)
    def _fetch_money_flow_sina_direct(self, code: str):
        """新浪财经 HTTP 直连获取资金流向（备选）"""
        import json as _json

        _rate_limit()
        session = _get_http_session()
        prefix = "sh" if code.startswith(("6", "9")) else "sz"
        symbol = f"{prefix}{code}"

        url = f"https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/MoneyFlow.ssl_qsfx_zijinliuliang?daima={symbol}"
        try:
            r = session.get(url, timeout=8)
            raw = _json.loads(r.text)
            if not raw or not isinstance(raw, list) or len(raw) < 2:
                return pd.DataFrame()

            # 取最近5天的数据
            rows = []
            for item in raw[-10:]:
                try:
                    date_str = item.get("opendate", item.get("date", ""))
                    rows.append(
                        {
                            "日期": date_str.replace("-", ""),
                            "主力净流入": float(item.get("superlarge_order", 0) or 0)
                            + float(item.get("large_order", 0) or 0),
                            "小单净流入": float(item.get("small_order", 0) or 0),
                            "中单净流入": float(item.get("medium_order", 0) or 0),
                            "大单净流入": float(item.get("large_order", 0) or 0),
                            "超大单净流入": float(item.get("superlarge_order", 0) or 0),
                        }
                    )
                except (ValueError, TypeError):
                    continue
            return pd.DataFrame(rows) if rows else pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    @retry_on_network_error(max_retries=3, base_delay=1.0)
    def _fetch_realtime_quote_sina(self, codes: list[str]):
        """新浪财经实时行情 API（批量获取）

        Args:
            codes: 股票代码列表, 如 ['sh600519', 'sz000001']

        Returns:
            DataFrame with columns: 代码, 名称, 最新价, 涨跌幅, 涨跌额, 成交量, 成交额, ...
        """
        _rate_limit()
        session = _get_http_session()

        # 新浪批量行情接口
        codes_str = ",".join(codes)
        url = f"https://hq.sinajs.cn/list={codes_str}"
        session.headers["Referer"] = "https://finance.sina.com.cn"

        r = session.get(url, timeout=8)
        text = r.text

        rows = []
        for line in text.strip().split("\n"):
            if not line.strip():
                continue
            try:
                # 格式: var hq_str_sh600519="贵州茅台,1850.00,1845.00,..."
                parts = line.split('"')
                if len(parts) < 2:
                    continue
                hq_str = parts[0].split("_str_")[-1] if "_str_" in parts[0] else parts[0].split("=")[0].strip()
                code = hq_str.replace("var hq_str_", "").strip()

                data_str = parts[1]
                fields = data_str.split(",")
                if len(fields) < 32:
                    continue

                rows.append(
                    {
                        "代码": code[2:],  # 去掉 sh/sz 前缀
                        "名称": fields[0],
                        "今开": float(fields[1]) if fields[1] else 0,
                        "昨收": float(fields[2]) if fields[2] else 0,
                        "最新价": float(fields[3]) if fields[3] else 0,
                        "最高": float(fields[4]) if fields[4] else 0,
                        "最低": float(fields[5]) if fields[5] else 0,
                        "成交量": float(fields[8]) if fields[8] else 0,
                        "成交额": float(fields[9]) if fields[9] else 0,
                        "涨跌额": float(fields[3]) - float(fields[2]) if fields[3] and fields[2] else 0,
                        "涨跌幅": round((float(fields[3]) - float(fields[2])) / float(fields[2]) * 100, 2)
                        if fields[3] and fields[2] and float(fields[2]) != 0
                        else 0,
                    }
                )
            except (IndexError, ValueError, ZeroDivisionError):
                continue

        return pd.DataFrame(rows) if rows else pd.DataFrame()

    @retry_on_network_error(max_retries=3, base_delay=1.0)
    def _fetch_realtime_quote_tencent(self, codes: list[str]):
        """腾讯财经实时行情 API（批量获取）

        Args:
            codes: 股票代码列表, 如 ['sh600519', 'sz000001']
        """
        _rate_limit()
        session = _get_http_session()

        codes_str = ",".join(codes)
        url = f"http://qt.gtimg.cn/q={codes_str}"
        session.headers["Referer"] = "https://gu.qq.com"

        r = session.get(url, timeout=8)
        text = r.text

        rows = []
        for line in text.strip().split("\n"):
            if not line.strip() or "=" not in line:
                continue
            try:
                # 格式: v_sh600519="1~贵州茅台~600519~1850.00~..."
                parts = line.split('"')
                if len(parts) < 2:
                    continue
                data_str = parts[1]
                fields = data_str.split("~")
                if len(fields) < 45:
                    continue

                rows.append(
                    {
                        "代码": fields[2] if len(fields) > 2 else "",
                        "名称": fields[1] if len(fields) > 1 else "",
                        "最新价": float(fields[3]) if fields[3] else 0,
                        "昨收": float(fields[4]) if fields[4] else 0,
                        "今开": float(fields[5]) if fields[5] else 0,
                        "成交量": float(fields[6]) if fields[6] else 0,  # 手
                        "成交额": float(fields[37]) if len(fields) > 37 and fields[37] else 0,  # 万
                        "最高": float(fields[33]) if len(fields) > 33 and fields[33] else 0,
                        "最低": float(fields[34]) if len(fields) > 34 and fields[34] else 0,
                        "涨跌额": float(fields[31]) if len(fields) > 31 and fields[31] else 0,
                        "涨跌幅": float(fields[32]) if len(fields) > 32 and fields[32] else 0,
                        "换手率": float(fields[38]) if len(fields) > 38 and fields[38] else 0,
                        "市盈率": float(fields[39]) if len(fields) > 39 and fields[39] else 0,
                        "总市值": float(fields[45]) if len(fields) > 45 and fields[45] else 0,  # 亿
                        "量比": float(fields[49]) if len(fields) > 49 and fields[49] else 0,
                    }
                )
            except (IndexError, ValueError):
                continue

        return pd.DataFrame(rows) if rows else pd.DataFrame()

    @retry_on_network_error(max_retries=3, base_delay=1.5)
    def _fetch_market_breadth_direct(self):
        """东方财富 HTTP 直连获取全市场涨跌家数等数据（备选）"""
        import json as _json

        _rate_limit()
        session = _get_http_session()

        url = (
            "http://push2.eastmoney.com/api/qt/clist/get"
            "?pn=1&pz=1&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281"
            "&fltt=2&invt=2&fid=f3"
            "&fs=m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23"
            "&fields=f2,f3,f4,f5,f6,f7,f8,f9,f12,f14,f15,f16,f17,f18,f20,f21"
        )
        r = session.get(url, timeout=10)
        total = _json.loads(r.text).get("data", {}).get("total", 0)

        if total > 0:
            # 获取全量数据用于统计涨跌
            url2 = (
                f"http://push2.eastmoney.com/api/qt/clist/get"
                f"?pn=1&pz={min(total, 100)}&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281"
                f"&fltt=2&invt=2&fid=f12"
                f"&fs=m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23"
                f"&fields=f2,f3,f12"
            )
            r2 = session.get(url2, timeout=10)
            items = _json.loads(r2.text).get("data", {}).get("diff", [])

            up_count = sum(1 for i in items if float(i.get("f3", 0) or 0) > 0)
            down_count = sum(1 for i in items if float(i.get("f3", 0) or 0) < 0)
            flat_count = len(items) - up_count - down_count

            return {
                "上涨": up_count,
                "下跌": down_count,
                "平盘": flat_count,
                "总数": len(items),
            }
        return {}

    @retry_on_network_error(max_retries=2, base_delay=2.0)
    def _fetch_limit_down_raw(self, date: str):
        ak = _ensure_akshare()
        _rate_limit()
        return ak.stock_zt_pool_dtgc_em(date=date)

    @retry_on_network_error(max_retries=3, base_delay=1.5)
    def _fetch_limit_up_direct(self, date: str):
        """东方财富 HTTP 直连获取涨停池数据（备选回退）"""
        import json as _json

        _rate_limit()
        session = _get_http_session()
        # 东方财富涨停板接口
        url = (
            "http://push2ex.eastmoney.com/getTopicZTPool"
            f"?ut=7eea3edcaed734bea9cbfce244eeed1e"
            f"&date={date}"
            f"&sort=fundsrate&sorttype=desc"
            f"&pagesize=200&pageindex=1"
        )
        try:
            r = session.get(url, timeout=10)
            data = _json.loads(r.text)
            items = data.get("data", {}).get("pool", [])
            if not items:
                return pd.DataFrame()

            rows = []
            for item in items:
                rows.append(
                    {
                        "代码": str(item.get("c", "")),
                        "名称": str(item.get("n", "")),
                        "最新价": float(item.get("p", 0) or 0),
                        "涨跌幅": float(item.get("zdf", 0) or 0),
                        "涨停价": float(item.get("zt", 0) or 0),
                        "成交额": float(item.get("amount", 0) or 0),
                        "流通市值": float(item.get("ltsz", 0) or 0),
                        "总市值": float(item.get("zsz", 0) or 0),
                        "换手率": float(item.get("lbl", 0) or 0),
                        "连板数": int(item.get("days", 0) or 0),
                        "封单资金": float(item.get("fba", 0) or 0),
                        "首次封板时间": str(item.get("ft", "")),
                        "最后封板时间": str(item.get("lt", "")),
                        "涨停统计": str(item.get("reason", item.get("hybk", ""))),
                        "所属行业": str(item.get("hybk", "")),
                    }
                )
            return pd.DataFrame(rows) if rows else pd.DataFrame()
        except Exception:
            # 备选 URL 格式（push2.eastmoney.com）
            try:
                url2 = (
                    "http://push2.eastmoney.com/api/qt/clist/get"
                    "?pn=1&pz=200&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281"
                    "&fltt=2&invt=2&fid=f3"
                    "&fs=b:KDJ%2Bb:BK0518"
                    "&fields=f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f14,f15,f16,f17,f18,f20,f21"
                )
                r2 = session.get(url2, timeout=10)
                data2 = _json.loads(r2.text)
                items2 = data2.get("data", {}).get("diff", [])
                if items2:
                    rows2 = []
                    for item in items2:
                        rows2.append(
                            {
                                "代码": str(item.get("f12", "")),
                                "名称": str(item.get("f14", "")),
                                "最新价": float(item.get("f2", 0) or 0),
                                "涨跌幅": float(item.get("f3", 0) or 0),
                                "涨跌额": float(item.get("f4", 0) or 0),
                                "成交量": float(item.get("f5", 0) or 0),
                                "成交额": float(item.get("f6", 0) or 0),
                                "换手率": float(item.get("f8", 0) or 0),
                                "市盈率-动态": float(item.get("f9", 0) or 0),
                                "总市值": float(item.get("f20", 0) or 0),
                                "流通市值": float(item.get("f21", 0) or 0),
                                "连板数": 0,
                                "涨停统计": "",
                                "所属行业": "",
                            }
                        )
                    return pd.DataFrame(rows2)
            except Exception:
                pass
        return pd.DataFrame()

    @retry_on_network_error(max_retries=3, base_delay=1.5)
    def _fetch_limit_down_direct(self, date: str):
        """东方财富 HTTP 直连获取跌停池数据（备选回退）"""
        import json as _json

        _rate_limit()
        session = _get_http_session()

        # 使用东方财富跌幅榜数据近似跌停池
        url = (
            "http://push2.eastmoney.com/api/qt/clist/get"
            "?pn=1&pz=200&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281"
            "&fltt=2&invt=2&fid=f3&fs=m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23"
            "&fields=f2,f3,f4,f5,f6,f8,f12,f14,f20,f21"
        )
        try:
            r = session.get(url, timeout=10)
            data = _json.loads(r.text)
            items = data.get("data", {}).get("diff", [])
            if not items:
                return pd.DataFrame()

            # 筛选接近跌停的股票（跌幅 < -9.5%）
            rows = []
            for item in items:
                zdf = float(item.get("f3", 0) or 0)
                if zdf <= -9.5:
                    rows.append(
                        {
                            "代码": str(item.get("f12", "")),
                            "名称": str(item.get("f14", "")),
                            "最新价": float(item.get("f2", 0) or 0),
                            "涨跌幅": zdf,
                            "成交额": float(item.get("f6", 0) or 0),
                            "换手率": float(item.get("f8", 0) or 0),
                            "总市值": float(item.get("f20", 0) or 0),
                            "流通市值": float(item.get("f21", 0) or 0),
                        }
                    )
            return pd.DataFrame(rows) if rows else pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    @retry_on_network_error(max_retries=3, base_delay=1.5)
    def _fetch_continuous_limit_direct(self, date: str):
        """东方财富 HTTP 直连获取连板池数据（备选回退）"""
        import json as _json

        _rate_limit()
        session = _get_http_session()

        # 使用涨停池数据，筛选连板 >= 2 的股票
        url = (
            "http://push2ex.eastmoney.com/getTopicZTPool"
            f"?ut=7eea3edcaed734bea9cbfce244eeed1e"
            f"&date={date}"
            f"&sort=days&sorttype=desc"
            f"&pagesize=100&pageindex=1"
        )
        try:
            r = session.get(url, timeout=10)
            data = _json.loads(r.text)
            items = data.get("data", {}).get("pool", [])
            if not items:
                return pd.DataFrame()

            rows = []
            for item in items:
                days = int(item.get("days", 0) or 0)
                if days >= 2:  # 只取连板股
                    rows.append(
                        {
                            "代码": str(item.get("c", "")),
                            "名称": str(item.get("n", "")),
                            "最新价": float(item.get("p", 0) or 0),
                            "涨跌幅": float(item.get("zdf", 0) or 0),
                            "涨停价": float(item.get("zt", 0) or 0),
                            "成交额": float(item.get("amount", 0) or 0),
                            "换手率": float(item.get("lbl", 0) or 0),
                            "流通市值": float(item.get("ltsz", 0) or 0),
                            "总市值": float(item.get("zsz", 0) or 0),
                            "连板数": days,
                            "封单资金": float(item.get("fba", 0) or 0),
                            "首次封板时间": str(item.get("ft", "")),
                            "最后封板时间": str(item.get("lt", "")),
                            "所属行业": str(item.get("hybk", "")),
                        }
                    )
            return pd.DataFrame(rows) if rows else pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    @retry_on_network_error(max_retries=3, base_delay=1.5)
    def _fetch_market_breadth_full(self):
        """东方财富 HTTP 直连获取全市场涨跌家数（带板块细分）"""
        import json as _json

        _rate_limit()
        session = _get_http_session()

        # 使用全市场统计接口
        url = (
            "http://push2.eastmoney.com/api/qt/clist/get"
            "?pn=1&pz=500&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281"
            "&fltt=2&invt=2&fid=f3"
            "&fs=m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23"
            "&fields=f2,f3,f4,f5,f6,f8,f12,f14,f20,f21"
        )
        try:
            r = session.get(url, timeout=12)
            data = _json.loads(r.text)
            items = data.get("data", {}).get("diff", [])
            if not items:
                return {}

            up_count = sum(1 for i in items if float(i.get("f3", 0) or 0) > 0)
            down_count = sum(1 for i in items if float(i.get("f3", 0) or 0) < 0)
            flat_count = len(items) - up_count - down_count

            # 统计涨停/跌停（涨跌幅 >= 9.5% 或 <= -9.5%）
            limit_up = sum(1 for i in items if float(i.get("f3", 0) or 0) >= 9.5)
            limit_down = sum(1 for i in items if float(i.get("f3", 0) or 0) <= -9.5)

            return {
                "上涨": up_count,
                "下跌": down_count,
                "平盘": flat_count,
                "涨停": limit_up,
                "跌停": limit_down,
                "总数": len(items),
            }
        except Exception:
            return {}
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
        """获取股票列表，优先级: 收费源 → 免费源: akshare → 东财直连HTTP → 腾讯直连HTTP"""
        if not force_refresh:
            cached = self.cache.get("stock_list", max_age_hours=24)
            if cached is not None:
                return cached

        # ── 优先尝试收费数据源 ──
        df_premium, _ = self._try_premium_datasource("fetch_stock_list")
        if df_premium is not None and not df_premium.empty:
            self.cache.set("stock_list", df_premium)
            return df_premium

        # 优先 akshare，失败则直连
        sources = [
            ("akshare", lambda: self._fetch_stock_list_raw()),
            ("东财直连", lambda: self._fetch_stock_list_direct()),
            ("腾讯直连", lambda: self._fetch_stock_list_tencent_direct()),
        ]

        for label, fetcher in sources:
            if not self._is_source_available(label):
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
                self._mark_source_unavailable(label, type(e).__name__)

        logger.error("获取股票列表失败（所有数据源均不可达）")
        return pd.DataFrame()

    def fetch_daily_kline(
        self, code: str, start_date: str = "20200101", end_date: str | None = None, adjust: str = "qfq"
    ) -> pd.DataFrame:
        """获取日K线，优先级: 收费源(TuShare > Choice > IBKR) → 免费源5级回退链:
        东方财富(akshare) → 新浪(akshare) → 腾讯(akshare) → 腾讯(直连) → 新浪(直连) → 本地DB
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")
        cache_key = f"daily_{code}_{adjust}"
        cached = self.cache.get(cache_key, max_age_hours=6, start_date=start_date, end_date=end_date)
        if cached is not None:
            return cached

        # ── 优先尝试收费数据源 ──
        df_premium, _ = self._try_premium_datasource("fetch_daily_kline", code, start_date, end_date, adjust)
        if df_premium is not None and not df_premium.empty:
            self.db.save_daily(df_premium)
            self.cache.set(cache_key, df_premium, start_date=start_date, end_date=end_date)
            return df_premium

        # ── 免费源降级 ──
        # 尝试各数据源，成功后写缓存/DB 并返回
        sources = [
            ("东方财富", lambda: self._fetch_eastmoney_kline(code, start_date, end_date, adjust)),
            ("新浪", lambda: self._fetch_sina_kline(code, start_date, end_date, adjust)),
            ("腾讯", lambda: self._fetch_tencent_kline(code, start_date, end_date, adjust)),
            ("腾讯直连", lambda: self._fetch_tencent_kline_direct(code, start_date, end_date, adjust)),
            ("新浪直连", lambda: self._fetch_sina_kline_direct(code, start_date, end_date, adjust)),
        ]

        for label, fetcher in sources:
            if not self._is_source_available(label):
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
                self._mark_source_unavailable(label, type(e).__name__)

        # 所有远程源都失败 → 降级到本地 SQLite
        logger.error(f"获取 {code} K线失败（所有数据源均不可达）")
        db_data = self.db.load_daily(code, start_date, end_date)
        if not db_data.empty:
            logger.info(f"  → 从本地数据库回退，获取 {len(db_data)} 条历史记录")
            return db_data
        return pd.DataFrame()

    def fetch_index_daily(
        self, index_code: str = "000001", start_date: str = "20200101", end_date: str | None = None
    ) -> pd.DataFrame:
        """获取指数日线，优先级: 收费源 → 免费源3级回退: 东方财富 → 腾讯(akshare) → 腾讯(直连)"""
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")
        cached = self.cache.get(f"index_{index_code}", max_age_hours=6, start_date=start_date, end_date=end_date)
        if cached is not None:
            return cached

        # ── 优先尝试收费数据源 ──
        df_premium, _ = self._try_premium_datasource("fetch_index_daily", index_code, start_date, end_date)
        if df_premium is not None and not df_premium.empty:
            self.cache.set(f"index_{index_code}", df_premium, start_date=start_date, end_date=end_date)
            return df_premium

        prefix = "sz" if index_code.startswith("3") else "sh"
        symbol = f"{prefix}{index_code}"

        sources = [
            ("东方财富", lambda: self._fetch_index_daily_raw(symbol)),
            ("腾讯", lambda: self._fetch_index_daily_tx(symbol)),
            ("腾讯直连", lambda: self._fetch_index_daily_direct(symbol)),
        ]

        for label, fetcher in sources:
            if not self._is_source_available(label):
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
                self._mark_source_unavailable(label, type(e).__name__)

        logger.error(f"获取指数 {index_code} 数据失败（所有数据源均不可达）")
        return pd.DataFrame()

    def fetch_financial_data(self, code: str) -> pd.DataFrame:
        """获取财务数据，优先级: 收费源(TuShare > Choice > IBKR) → 免费源三级回退链:
        AKShare(财务分析指标) → AKShare(财务摘要) → 东方财富HTTP直连
        """
        cached = self.cache.get(f"finance_{code}", max_age_hours=168)
        if cached is not None:
            return cached

        # ── 优先尝试收费数据源 ──
        df_premium, _ = self._try_premium_datasource("fetch_financial_data", code)
        if df_premium is not None and not df_premium.empty:
            self.cache.set(f"finance_{code}", df_premium)
            return df_premium

        sources = [
            ("AKShare财务指标", lambda: self._fetch_financial_raw(code)),
            ("AKShare财务摘要", lambda: self._fetch_financial_abstract_raw(code)),
            ("东方财富直连", lambda: self._fetch_financial_eastmoney_direct(code)),
        ]

        for label, fetcher in sources:
            if not self._is_source_available(label):
                continue
            try:
                df = fetcher()
                if df is not None and not df.empty:
                    if label != sources[0][0]:
                        logger.info(f"  → '{code}' 财务数据使用 {label} 数据源")
                    self.cache.set(f"finance_{code}", df)
                    return df
            except Exception as e:
                self._mark_source_unavailable(label, type(e).__name__)

        logger.error(f"获取 {code} 财务数据失败（所有数据源均不可达）")
        return pd.DataFrame()

    def fetch_industry_stocks(self, industry: str, board_code: str = "") -> pd.DataFrame:
        from . import sector_fallback as _sf

        return _sf.fetch_stocks(self.cache, industry, board_code)

    def fetch_industry_list(self, force_refresh: bool = False) -> pd.DataFrame:
        from . import sector_fallback as _sf

        return _sf.fetch_list(self.cache, force=force_refresh)

    def fetch_limit_up_pool(self, date: str | None = None) -> pd.DataFrame:
        """获取涨停板股票池，两级回退: AKShare → 东财HTTP直连"""
        if date is None:
            date = datetime.now().strftime("%Y%m%d")

        sources = [
            ("AKShare涨停池", lambda: self._fetch_limit_up_raw(date)),
            ("东方财富涨停池直连", lambda: self._fetch_limit_up_direct(date)),
        ]

        for label, fetcher in sources:
            if not self._is_source_available(label):
                continue
            try:
                df = fetcher()
                if df is not None and not df.empty:
                    if label != sources[0][0]:
                        logger.info(f"  → 涨停池使用 {label} 数据源")
                    return df
            except Exception as e:
                self._mark_source_unavailable(label, type(e).__name__)

        logger.error("获取涨停板数据失败（所有数据源均不可达）")
        return pd.DataFrame()

    def fetch_continuous_limit_up(self, date: str | None = None) -> pd.DataFrame:
        """获取连板股票池，两级回退: AKShare → 东财HTTP直连"""
        if date is None:
            date = datetime.now().strftime("%Y%m%d")

        sources = [
            ("AKShare连板池", lambda: self._fetch_continuous_limit_raw(date)),
            ("东方财富连板池直连", lambda: self._fetch_continuous_limit_direct(date)),
        ]

        for label, fetcher in sources:
            if not self._is_source_available(label):
                continue
            try:
                df = fetcher()
                if df is not None and not df.empty:
                    if label != sources[0][0]:
                        logger.info(f"  → 连板池使用 {label} 数据源")
                    return df
            except Exception as e:
                self._mark_source_unavailable(label, type(e).__name__)

        logger.error("获取连板数据失败（所有数据源均不可达）")
        return pd.DataFrame()

    def fetch_money_flow(self, code: str) -> pd.DataFrame:
        """获取资金流向，优先级: 收费源 → 免费源三级回退链:
        AKShare(个股资金流向) → 东方财富HTTP直连 → 新浪HTTP直连
        """
        # ── 优先尝试收费数据源 ──
        df_premium, _ = self._try_premium_datasource("fetch_money_flow", code)
        if df_premium is not None and not df_premium.empty:
            return df_premium

        sources = [
            ("AKShare资金流向", lambda: self._fetch_money_flow_raw(code, "sh" if code.startswith("6") else "sz")),
            ("东方财富资金流向直连", lambda: self._fetch_money_flow_direct(code)),
            ("新浪资金流向直连", lambda: self._fetch_money_flow_sina_direct(code)),
        ]

        for label, fetcher in sources:
            if not self._is_source_available(label):
                continue
            try:
                df = fetcher()
                if df is not None and not df.empty:
                    if label != sources[0][0]:
                        logger.info(f"  → '{code}' 资金流向使用 {label} 数据源")
                    return df
            except Exception as e:
                self._mark_source_unavailable(label, type(e).__name__)

        logger.error(f"获取 {code} 资金流向失败（所有数据源均不可达）")
        return pd.DataFrame()

    def fetch_limit_down_pool(self, date: str | None = None) -> pd.DataFrame:
        """获取跌停板股票池，两级回退: AKShare → 东财HTTP直连"""
        if date is None:
            date = datetime.now().strftime("%Y%m%d")

        sources = [
            ("AKShare跌停池", lambda: self._fetch_limit_down_raw(date)),
            ("东方财富跌停池直连", lambda: self._fetch_limit_down_direct(date)),
        ]

        for label, fetcher in sources:
            if not self._is_source_available(label):
                continue
            try:
                df = fetcher()
                if df is not None and not df.empty:
                    if label != sources[0][0]:
                        logger.info(f"  → 跌停池使用 {label} 数据源")
                    return df
            except Exception as e:
                self._mark_source_unavailable(label, type(e).__name__)

        logger.error("获取跌停板数据失败（所有数据源均不可达）")
        return pd.DataFrame()

    @retry_on_network_error(max_retries=2, base_delay=1.5)
    def _fetch_market_activity_raw(self):
        """AKShare 乐股市场活跃度数据（主数据源）"""
        ak = _ensure_akshare()
        _rate_limit()
        return ak.stock_market_activity_legu()

    def fetch_market_activity(self) -> dict[str, float]:
        """获取全市场活跃度概览（涨跌停数、涨跌家数、成交量等），三级回退

        Returns:
            dict: {上涨, 下跌, 平盘, 涨停, 真实涨停, 跌停, 真实跌停, 总市值, 成交量, ...}
        """
        # 一级: AKShare 乐股数据（最详细）
        if "AKShare乐股" not in self._unavailable_sources:
            try:
                df = self._fetch_market_activity_raw()
                if df is not None and not df.empty:
                    result = {}
                    for _, row in df.iterrows():
                        key = str(row["item"])
                        val = row["value"]
                        try:
                            result[key] = float(val)
                        except (ValueError, TypeError):
                            result[key] = val
                    self._unavailable_sources.pop("AKShare乐股", None)  # 恢复可用
                    return result
            except Exception as e:
                self._mark_source_unavailable("AKShare乐股", type(e).__name__)

        # 二级: 东方财富 HTTP 直连获取涨跌家数（采样100只）
        try:
            bread = self._fetch_market_breadth_direct()
            if bread:
                logger.info("  → 市场活跃度使用东方财富直连(采样)数据源")
                return {
                    "上涨": bread.get("上涨", 0),
                    "下跌": bread.get("下跌", 0),
                    "平盘": bread.get("平盘", 0),
                    "涨停": 0,
                    "跌停": 0,
                }
        except Exception as e:
            logger.warning(f"东方财富涨跌家数(采样)获取失败: {e}")

        # 三级: 东方财富全量统计（带涨停/跌停检测）
        try:
            bread_full = self._fetch_market_breadth_full()
            if bread_full:
                logger.info("  → 市场活跃度使用东方财富直连(全量)数据源")
                return bread_full
        except Exception as e:
            logger.error(f"获取市场活跃度失败: {e}")

        return {}

    # ═══ 北向资金 ═══

    @retry_on_network_error(max_retries=3, base_delay=1.5)
    def _fetch_northbound_flow_direct(self, days: int = 30):
        """东方财富 HTTP 直连获取北向资金（沪深港通）日流向

        使用东方财富北向资金 K 线接口，返回每日净流入数据。
        """
        import json as _json

        _rate_limit()
        session = _get_http_session()

        # 北向资金总计（沪股通+深股通）日线
        url = (
            "http://push2his.eastmoney.com/api/qt/kamt.kline/get"
            "?fields1=f1,f2,f3,f4"
            "&fields2=f51,f52,f53,f54,f55,f56"
            "&klt=101"  # 日线
            f"&lmt={min(days, 60)}"
            "&ut=b2884a393a59ad64002292a3e90d46a5"
        )
        try:
            r = session.get(url, timeout=10)
            data = _json.loads(r.text)
            klines = data.get("data", {}).get("klines", [])
            if not klines:
                return pd.DataFrame()

            rows = []
            for line in klines:
                parts = line.split(",")
                if len(parts) >= 4:
                    rows.append(
                        {
                            "日期": parts[0].replace("-", ""),
                            "当日净流入": round(float(parts[1]) / 1e8, 2) if parts[1] != "-" else 0,  # 亿
                            "沪股通净流入": round(float(parts[2]) / 1e8, 2) if parts[2] != "-" else 0,
                            "深股通净流入": round(float(parts[3]) / 1e8, 2) if parts[3] != "-" else 0,
                            "当日余额": round(float(parts[4]) / 1e8, 2) if len(parts) > 4 and parts[4] != "-" else 0,
                        }
                    )
            return pd.DataFrame(rows) if rows else pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    @retry_on_network_error(max_retries=3, base_delay=2.0)
    def _fetch_northbound_flow_akshare(self):
        """AKShare 北向资金数据（备选）"""
        ak = _ensure_akshare()
        _rate_limit()
        # stock_hsgt_hist_em 获取北向资金历史数据
        return ak.stock_hsgt_hist_em(symbol="北向资金")

    def fetch_northbound_flow(self, days: int = 30) -> pd.DataFrame:
        """获取北向资金（沪深港通）流向，两级回退: 东财直连 → AKShare

        Args:
            days: 获取最近多少天的数据

        Returns:
            DataFrame with columns: 日期, 当日净流入, 沪股通净流入, 深股通净流入, ...
        """
        cached = self.cache.get("northbound_flow", max_age_hours=6)
        if cached is not None:
            return cached

        sources = [
            ("东方财富北向资金直连", lambda: self._fetch_northbound_flow_direct(days)),
            ("AKShare北向资金", lambda: self._fetch_northbound_flow_akshare()),
        ]

        for label, fetcher in sources:
            if not self._is_source_available(label):
                continue
            try:
                df = fetcher()
                if df is not None and not df.empty:
                    if label != sources[0][0]:
                        logger.info(f"  → 北向资金使用 {label} 数据源")
                    self.cache.set("northbound_flow", df)
                    return df
            except Exception as e:
                self._mark_source_unavailable(label, type(e).__name__)

        logger.error("获取北向资金数据失败（所有数据源均不可达）")
        return pd.DataFrame()

    # ═══ 龙虎榜 ═══

    @retry_on_network_error(max_retries=3, base_delay=1.5)
    def _fetch_lhb_direct(self, date: str | None = None):
        """东方财富 HTTP 直连获取龙虎榜数据

        使用 push2.eastmoney.com 个股龙虎榜接口。
        """
        import json as _json

        _rate_limit()
        session = _get_http_session()

        if date is None:
            date = datetime.now().strftime("%Y%m%d")

        # 使用东方财富龙虎榜个股接口
        f"{date[:4]}-{date[4:6]}-{date[6:]}"
        url = (
            "http://push2.eastmoney.com/api/qt/clist/get"
            "?pn=1&pz=200&po=1&np=1"
            "&ut=bd1d9ddb04089700cf9c27f6f7426281"
            "&fltt=2&invt=2&fid=f3"
            "&fs=m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23"
            "&fields=f2,f3,f4,f5,f6,f7,f8,f12,f14,f15,f16,f17,f18,f20,f21,f184,f66,f69,f72"
        )
        try:
            r = session.get(url, timeout=10)
            data = _json.loads(r.text)
            items = data.get("data", {}).get("diff", [])
            if not items:
                return pd.DataFrame()

            # 筛选有龙虎榜数据的股票（f184 字段 = 龙虎榜标记）
            rows = []
            for item in items:
                # f184 可能是龙虎榜相关字段
                lhb_flag = item.get("f184", "")
                if lhb_flag:
                    rows.append(
                        {
                            "股票代码": str(item.get("f12", "")),
                            "股票名称": str(item.get("f14", "")),
                            "最新价": float(item.get("f2", 0) or 0),
                            "涨跌幅": float(item.get("f3", 0) or 0),
                            "成交额": round(float(item.get("f6", 0) or 0) / 1e8, 2),
                            "总市值": round(float(item.get("f20", 0) or 0) / 1e8, 2) if item.get("f20") else 0,
                            "流通市值": round(float(item.get("f21", 0) or 0) / 1e8, 2) if item.get("f21") else 0,
                            "上榜原因": str(item.get("f184", "")),
                        }
                    )
            # 如果 push2 方式拿不到，返回空（让 akshare 回退处理）
            return pd.DataFrame(rows) if rows else pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    @retry_on_network_error(max_retries=2, base_delay=2.0)
    def _fetch_lhb_akshare(self, date: str | None = None):
        """AKShare 龙虎榜数据（备选）- 多接口尝试"""
        import datetime as _dt

        ak = _ensure_akshare()
        _rate_limit()
        if date is None:
            date = _dt.datetime.now().strftime("%Y%m%d")
        # 尝试多种接口格式
        for fn_name, fn_args in [
            ("stock_lhb_detail_em", {"date": date}),
            ("stock_lhb_detail_daily_sina", {"date": date}),
            ("stock_lhb_stock_list_em", {"date": date}),
        ]:
            try:
                fn = getattr(ak, fn_name, None)
                if fn:
                    return fn(**fn_args)
            except Exception:
                continue
        raise AttributeError("AKShare 龙虎榜接口均不可用")

    def fetch_lhb_detail(self, date: str | None = None) -> pd.DataFrame:
        """获取龙虎榜详情，两级回退: 东财直连 → AKShare

        Returns:
            DataFrame with columns: 股票代码, 股票名称, 涨跌幅, 龙虎榜净买额, 上榜原因, ...
        """
        if date is None:
            date = datetime.now().strftime("%Y%m%d")

        cached = self.cache.get(f"lhb_{date}", max_age_hours=24)
        if cached is not None:
            return cached

        sources = [
            ("东方财富龙虎榜直连", lambda: self._fetch_lhb_direct(date)),
            ("AKShare龙虎榜", lambda: self._fetch_lhb_akshare(date)),
        ]

        for label, fetcher in sources:
            if not self._is_source_available(label):
                continue
            try:
                df = fetcher()
                if df is not None and not df.empty:
                    if label != sources[0][0]:
                        logger.info(f"  → 龙虎榜使用 {label} 数据源")
                    self.cache.set(f"lhb_{date}", df)
                    return df
            except Exception as e:
                self._mark_source_unavailable(label, type(e).__name__)

        logger.error("获取龙虎榜数据失败（所有数据源均不可达）")
        return pd.DataFrame()

    # ═══ 融资融券 ═══

    @retry_on_network_error(max_retries=3, base_delay=1.5)
    def _fetch_margin_trading_direct(self):
        """东方财富 HTTP 直连获取融资融券余额（全市场汇总）

        使用 push2.eastmoney.com 融资融券数据接口。
        """
        import json as _json

        _rate_limit()
        session = _get_http_session()

        # 沪深两市融资融券汇总数据
        url = (
            "http://push2.eastmoney.com/api/qt/clist/get"
            "?pn=1&pz=30&po=1&np=1"
            "&ut=bd1d9ddb04089700cf9c27f6f7426281"
            "&fltt=2&invt=2&fid=f62"
            "&fs=m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23"
            "&fields=f2,f3,f4,f12,f14,f62,f184,f66,f69"
        )
        try:
            r = session.get(url, timeout=12)
            data = _json.loads(r.text)
            items = data.get("data", {}).get("diff", [])
            if not items:
                return pd.DataFrame()

            # 汇总个股融资余额
            total_margin = 0.0
            total_sc = 0.0
            for item in items:
                margin_val = float(item.get("f62", 0) or 0)  # 融资余额近似
                sc_val = float(item.get("f66", 0) or 0)  # 融券余额近似
                total_margin += margin_val
                total_sc += sc_val

            rows = [
                {
                    "日期": datetime.now().strftime("%Y%m%d"),
                    "融资余额": round(total_margin / 1e8, 2),
                    "融券余额": round(total_sc / 1e8, 2),
                    "融资融券余额": round((total_margin + total_sc) / 1e8, 2),
                }
            ]
            return pd.DataFrame(rows) if rows else pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    @retry_on_network_error(max_retries=2, base_delay=2.0)
    def _fetch_margin_trading_akshare(self):
        """AKShare 融资融券数据（备选）- 多接口尝试"""
        ak = _ensure_akshare()
        _rate_limit()
        import datetime as _dt

        today = _dt.datetime.now()
        # 尝试多种接口
        for fn_name, fn_args in [
            ("stock_margin_detail_szse", {"date": today.strftime("%Y-%m-%d")}),
            ("stock_margin_detail_szse", {"date": today.strftime("%Y%m%d")}),
            (
                "stock_margin_sse",
                {
                    "start_date": (today - _dt.timedelta(days=30)).strftime("%Y%m%d"),
                    "end_date": today.strftime("%Y%m%d"),
                },
            ),
            ("stock_margin_sse", {}),
        ]:
            try:
                fn = getattr(ak, fn_name, None)
                if fn:
                    return fn(**fn_args)
            except Exception:
                continue
        raise AttributeError("AKShare 融资融券接口均不可用")

    def fetch_margin_trading(self) -> pd.DataFrame:
        """获取融资融券余额数据，两级回退: 东财直连 → AKShare

        Returns:
            DataFrame with columns: 日期, 融资余额, 融资买入额, 融券余额, 融资融券余额, ...
        """
        cached = self.cache.get("margin_trading", max_age_hours=6)
        if cached is not None:
            return cached

        sources = [
            ("东方财富融资融券直连", lambda: self._fetch_margin_trading_direct()),
            ("AKShare融资融券", lambda: self._fetch_margin_trading_akshare()),
        ]

        for label, fetcher in sources:
            if not self._is_source_available(label):
                continue
            try:
                df = fetcher()
                if df is not None and not df.empty:
                    if label != sources[0][0]:
                        logger.info(f"  → 融资融券使用 {label} 数据源")
                    self.cache.set("margin_trading", df)
                    return df
            except Exception as e:
                self._mark_source_unavailable(label, type(e).__name__)

        logger.error("获取融资融券数据失败（所有数据源均不可达）")
        return pd.DataFrame()

    def fetch_realtime_quote(self, codes: list[str]) -> pd.DataFrame:
        """获取实时行情（批量），两级回退: 新浪 → 腾讯

        Args:
            codes: 股票代码列表, 如 ['600519', '000001']（不含市场前缀）

        Returns:
            DataFrame with columns: 代码, 名称, 最新价, 涨跌幅, 成交量, 成交额, 换手率, 量比, 市盈率
        """
        if not codes:
            return pd.DataFrame()

        # 构造带市场前缀的代码
        sina_codes = [f"sh{code}" if code.startswith(("6", "9")) else f"sz{code}" for code in codes]

        sources = [
            ("新浪实时行情", lambda: self._fetch_realtime_quote_sina(sina_codes)),
            ("腾讯实时行情", lambda: self._fetch_realtime_quote_tencent(sina_codes)),
        ]

        for label, fetcher in sources:
            if not self._is_source_available(label):
                continue
            try:
                df = fetcher()
                if df is not None and not df.empty:
                    if label != sources[0][0]:
                        logger.info(f"  → 实时行情使用 {label} 数据源")
                    return df
            except Exception as e:
                self._mark_source_unavailable(label, type(e).__name__)

        return pd.DataFrame()

    def reset_unavailable_sources(self):
        """重置不可用数据源标记（用于定期恢复尝试）"""
        count = len(self._unavailable_sources)
        if count > 0:
            self._unavailable_sources.clear()
            logger.info(f"已重置 {count} 个不可用数据源标记（强制恢复重试）")

        # 同时重置 sector_fallback 模块的不可用标记
        try:
            from . import sector_fallback as _sf

            _sf.reset()
        except Exception:
            pass

    def get_data_source_health(self) -> dict[str, object]:
        """获取数据源健康状态（含收费源）"""
        all_sources = {
            "股票列表": ["akshare", "东财直连", "腾讯直连"],
            "K线": ["东方财富", "新浪", "腾讯", "腾讯直连", "新浪直连"],
            "指数": ["东方财富", "腾讯", "腾讯直连"],
            "财务": ["AKShare财务指标", "AKShare财务摘要", "东方财富直连"],
            "资金流向": ["AKShare资金流向", "东方财富资金流向直连", "新浪资金流向直连"],
            "实时行情": ["新浪实时行情", "腾讯实时行情"],
            "市场活跃度": ["AKShare乐股", "东方财富涨跌家数(采样)", "东方财富涨跌家数(全量)"],
            "涨停池": ["AKShare涨停池", "东方财富涨停池直连"],
            "连板池": ["AKShare连板池", "东方财富连板池直连"],
            "跌停池": ["AKShare跌停池", "东方财富跌停池直连"],
            "板块": ["AKShare行业", "AKShare概念", "东方财富HTTP行业", "东方财富HTTP概念"],
            "北向资金": ["东方财富北向资金直连", "AKShare北向资金"],
            "龙虎榜": ["东方财富龙虎榜直连", "AKShare龙虎榜"],
            "融资融券": ["东方财富融资融券直连", "AKShare融资融券"],
        }

        health = {}
        for category, sources in all_sources.items():
            available = [s for s in sources if s not in self._unavailable_sources]
            unavailable = [s for s in sources if s in self._unavailable_sources]
            health[category] = {
                "available": available,
                "unavailable": unavailable,
                "total": len(sources),
                "healthy": len(available) == len(sources),
            }

        # 添加收费数据源状态
        try:
            from .datasource.manager import _read_apikeys

            apikeys = _read_apikeys()
            premium_status = []
            premium_sources = {
                "tushare": "TuShare Pro",
                "choice": "Choice",
                "ibkr": "IBKR",
            }
            for src_id, src_name in premium_sources.items():
                cfg = apikeys.get(src_id, {})
                premium_status.append(
                    {
                        "id": src_id,
                        "name": src_name,
                        "configured": bool(cfg.get("api_key") if isinstance(cfg, dict) else cfg),
                        "status": "已配置" if (cfg.get("api_key") if isinstance(cfg, dict) else cfg) else "未配置",
                    }
                )
            health["_premium"] = premium_status
        except Exception:
            health["_premium"] = []

        return health

    def fetch_market_total_volume(self) -> dict[str, object]:
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
                items = dict(zip(sh["项目"], sh["股票"], strict=False))
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

    def fetch_batch_5day_gains(self, codes: list[str], max_workers: int = 3) -> dict[str, float]:
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
        import json as _json
        from concurrent.futures import ThreadPoolExecutor, as_completed

        result: dict[str, float] = {}
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
                        df = ak.stock_zh_a_hist(
                            symbol=code, period="daily", start_date=start_str, end_date=today_str, adjust="qfq"
                        )
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

    def build_stock_pool(
        self, market: str = "全部", filter_st: bool = True, filter_new: bool = True, min_days: int = 60
    ) -> list[str]:
        stock_list = self.fetch_stock_list()
        if stock_list.empty:
            return []
        codes = stock_list["股票代码"].tolist()
        names = stock_list["股票简称"].tolist()
        filtered = []
        for code, name in zip(codes, names, strict=False):
            if filter_st and ("ST" in str(name) or "退" in str(name)):
                continue
            filtered.append(code)
        return filtered

    def prepare_akquant_data(
        self, codes: list[str], start_date: str, end_date: str | None = None
    ) -> dict[str, pd.DataFrame]:
        """准备 AKQuant 格式的多标的数据字典

        返回: {symbol: DataFrame[date, open, high, low, close, volume, symbol]}
        """
        import pandas as _pd

        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")

        col_map = {
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
        }
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
                df = df[[*required, "amount"] if "amount" in df.columns else required].copy()
                df["date"] = _pd.to_datetime(df["date"], errors="coerce")
                df = df.dropna(subset=["date"]).sort_values("date")
                df["symbol"] = str(code)
                if len(df) >= 20:
                    data[str(code)] = df[[*required, "symbol"]]
            except Exception as e:
                logger.debug(f"准备 {code} AKQ数据失败: {e}")
                continue
        return data


data_fetcher = DataFetcher()
