"""
股票基本信息本地缓存模块

用于缓存股票代码、名称、行业、地区等长期不变的基本信息，
便于网络拥挤时查询、选股遍历等场景。

缓存内容（静态/准静态）：
- code: 股票代码（6位）
- name: 股票名称
- industry: 申万行业分类
- area: 地区
- market: 市场类型（主板/创业板/科创板）
- list_date: 上市日期
- exchange: 交易所（上交所/深交所）

不缓存的内容（高变化频率）：
- 股价、动态PE、市值等实时变动的估值数据

数据源策略：
1. TuShare stock_basic（含完整行业/地区/上市日期）→ 最高优先级
2. 东方财富 HTTP（可补充部分行业信息）
3. 本地 SQLite（降级查询）
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from .config import DB_PATH

logger = logging.getLogger(__name__)

# 股票基本信息缓存刷新间隔（天）
DEFAULT_REFRESH_DAYS = 7
# 最少可用的缓存记录数
MIN_CACHE_COUNT = 3000


class StockInfoCache:
    """股票基本信息本地缓存管理器。

    用法示例::

        cache = StockInfoCache()
        cache.ensure_cache()                    # 首次/定期填充
        name = cache.get_name("600519")         # 快速名称查询
        stocks = cache.get_by_industry("白酒")  # 按行业筛选
        industries = cache.get_industries()     # 行业列表
    """

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or DB_PATH
        self._db = None  # 懒加载

    @property
    def db(self):
        """懒加载 Database 实例（避免循环引用）"""
        if self._db is None:
            from .data_manager import Database

            self._db = Database(self.db_path)
        return self._db

    # ── 公开查询 API ──

    def get_name(self, code: str) -> str:
        """快速获取股票名称（网络不可用时降级查询）。

        Returns:
            股票名称，若未缓存返回空字符串
        """
        info = self.db.get_stock_info_by_code(code)
        return info["name"] if info else ""

    def get_info(self, code: str) -> dict | None:
        """获取单只股票的完整基本信息。

        Returns:
            dict(code/name/industry/area/market/list_date/exchange/updated_at)
            未缓存返回 None
        """
        return self.db.get_stock_info_by_code(code)

    def get_names_batch(self, codes: list[str]) -> dict[str, str]:
        """批量获取代码→名称映射"""
        return self.db.get_stock_names_by_codes(codes)

    def get_by_industry(self, industry: str) -> list[dict]:
        """按行业筛选股票列表"""
        return self.db.get_stocks_by_industry(industry)

    def get_industries(self) -> list[str]:
        """获取所有已缓存的行业分类列表"""
        return self.db.get_all_industries()

    def get_count(self) -> int:
        """已缓存的股票数量"""
        return self.db.get_stock_count()

    def is_fresh(self, max_age_days: int = DEFAULT_REFRESH_DAYS) -> bool:
        """缓存是否新鲜（最近 N 天内有更新）"""
        sample = self.db.get_stock_info_by_code("000001")
        if not sample or not sample.get("updated_at"):
            return False
        try:
            updated = datetime.strptime(sample["updated_at"], "%Y-%m-%d %H:%M:%S")
            return (datetime.now() - updated).days < max_age_days
        except (ValueError, KeyError):
            return False

    def is_usable(self) -> bool:
        """缓存是否达到可用标准（记录数足够）"""
        return self.db.get_stock_count() >= MIN_CACHE_COUNT

    # ── 缓存填充与同步 ──

    def ensure_cache(self, force_refresh: bool = False) -> dict:
        """确保本地缓存可用（填充或刷新）。

        优先级：
        1. TuShare 全量同步（包含行业/地区等完整信息）
        2. 东方财富补充行业信息
        3. 本地缓存已有数据（跳过刷新）

        Returns:
            dict: {source, count, status}
        """
        existing_count = self.db.get_stock_count()

        # 已有足够数据且无需强刷 → 跳过
        if not force_refresh and existing_count >= MIN_CACHE_COUNT and self.is_fresh():
            logger.info(f"stock_info 缓存已就绪: {existing_count} 条记录")
            return {"source": "cache", "count": existing_count, "status": "fresh"}

        # 尝试 TuShare 全量同步
        result = self._populate_from_tushare()
        if result["count"] >= MIN_CACHE_COUNT:
            return result

        # TuShare 不可用，尝试从东方财富补充
        result2 = self._populate_from_eastmoney()
        total = existing_count + result2.get("count", 0)
        if total >= MIN_CACHE_COUNT:
            status = "partial" if result2.get("count", 0) > 0 else "stale"
            return {"source": "eastmoney+cache", "count": total, "status": status}

        # 数据不足
        logger.warning(
            f"stock_info 缓存不足: 当前 {total} 条，最低要求 {MIN_CACHE_COUNT}"
        )
        return {"source": "cache", "count": existing_count, "status": "insufficient"}

    def _populate_from_tushare(self) -> dict:
        """通过 TuShare stock_basic 全量同步股票基本信息。

        利用 DataFetcher.fetch_stock_list() 内置的优先级链，
        当 TuShare 可用时自动获取含行业/地区等完整信息的数据，
        并自动写入 stock_info 缓存表。
        """
        try:
            from .data_manager import DataFetcher

            fetcher = DataFetcher()
            df = fetcher.fetch_stock_list(force_refresh=True)
            if df is None or df.empty:
                logger.info("fetch_stock_list 返回空数据")
                return {"source": "none", "count": 0, "status": "empty"}

            # DataFetcher._save_stock_list_to_cache 已自动写入 DB
            # 直接统计当前缓存数量
            count = self.db.get_stock_count()
            logger.info(f"stock_info 缓存同步完成: TuShare → 当前 {count} 条")
            return {"source": "tushare", "count": count, "status": "synced"}

        except ImportError:
            logger.debug("TuShare 未安装")
            return {"source": "none", "count": 0, "status": "no_module"}
        except Exception as e:
            logger.warning(f"TuShare stock_info 同步失败: {e}")
            return {"source": "none", "count": 0, "status": f"error: {e}"}

    def _populate_from_eastmoney(self) -> dict:
        """通过东方财富 API 补充行业信息（免费源回退）。

        东方财富 HTTP 接口 `fs:m:0+t:6,...` 可返回 f12(代码)+f14(名称)，
        但行业字段需通过个股详情接口获取。

        这里仅做名称补充：从东方财富获取全部代码+名称，
        与本地已有数据合并写入。
        """
        try:
            import json as _json

            import requests

            session = requests.Session()
            session.headers.update({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Referer": "http://quote.eastmoney.com/",
            })

            # 获取全市场股票列表（含板块细分 market fields）
            url = (
                "http://push2.eastmoney.com/api/qt/clist/get"
                "?pn=1&pz=6000&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281"
                "&fltt=2&invt=2&fid=f3"
                "&fs=m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23"
                "&fields=f12,f13,f14"
            )
            r = session.get(url, timeout=15)
            data = _json.loads(r.text)
            items = data.get("data", {}).get("diff", [])

            # f13: 市场代码 (0=深交所, 1=上交所)
            existing_codes = self.db.get_stock_names_by_codes(
                [str(it["f12"]) for it in items]
            ).keys()
            new_count = 0

            records = []
            for item in items:
                code = str(item["f12"])
                if code in existing_codes:
                    continue
                name = str(item.get("f14", ""))
                market_id = item.get("f13", -1)
                exchange = "上交所" if market_id == 1 else "深交所" if market_id == 0 else ""
                records.append({
                    "code": code,
                    "name": name,
                    "exchange": exchange,
                })
                new_count += 1

            if records:
                self.db.save_stock_info_batch(records)
                logger.info(f"stock_info 东方财富补充: {new_count} 条新记录")

            return {
                "source": "eastmoney",
                "count": new_count,
                "status": "synced",
            }

        except Exception as e:
            logger.debug(f"东方财富 stock_info 补充失败: {e}")
            return {"source": "none", "count": 0, "status": f"error: {e}"}

    def refresh_cache(self) -> dict:
        """强制刷新缓存（建议在 Web 服务启动时调用）"""
        logger.info("开始刷新 stock_info 缓存...")
        return self.ensure_cache(force_refresh=True)

    def get_stats(self) -> dict:
        """获取缓存统计信息"""
        count = self.db.get_stock_count()
        industries = self.db.get_all_industries()
        fresh = self.is_fresh()
        sample = self.db.get_stock_info_by_code("000001")
        updated = sample.get("updated_at", "") if sample else ""
        return {
            "total_stocks": count,
            "total_industries": len(industries),
            "is_fresh": fresh,
            "last_updated": updated,
            "industries_sample": industries[:10] if industries else [],
        }


# 模块级单例（懒初始化）
_cache_instance: Optional[StockInfoCache] = None


def get_stock_info_cache() -> StockInfoCache:
    """获取 StockInfoCache 单例"""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = StockInfoCache()
    return _cache_instance
