"""
多数据源管理器 - 负责优先级调度、额度管理、自动降级

优先级:
  1. 已配置的收费数据源优先使用
  2. 额度用完自动降级
  3. 连接失败自动跳过
  4. 所有收费源不可用时降级到免费源

默认优先级顺序: TuShare > Choice > IBKR > 免费源(DataFetcher)
"""

import json
import logging
import os
from datetime import datetime

import pandas as pd

from ..config import DATA_DIR
from .base import DataSourceAdapter, FetchResult
from .choice_source import ChoiceAdapter
from .ibkr_source import IBKRAdapter
from .tushare_source import TuShareAdapter

logger = logging.getLogger(__name__)

# API Key 存储路径
_APIKEY_FILE = os.path.join(DATA_DIR, "apikeys.json")
os.makedirs(DATA_DIR, exist_ok=True)


def _read_apikeys() -> dict:
    """读取 API Key 配置"""
    if not os.path.exists(_APIKEY_FILE):
        return {}
    try:
        with open(_APIKEY_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _write_apikeys(data: dict):
    """写入 API Key 配置"""
    with open(_APIKEY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


class DataSourceManager:
    """多数据源管理器

    管理所有第三方数据源的:
    - API Key 配置
    - 连接状态
    - 优先级排序
    - 额度跟踪
    - 自动降级

    使用方式:
        mgr = DataSourceManager()
        # 在面板中设置 API Key
        mgr.update_api_key("tushare", "your_token")
        # 获取 K 线时会自动按优先级尝试
        result = mgr.fetch_daily_kline("600519", "20240101", "20240601")
    """

    # 数据源优先级（数字越小越优先）
    DEFAULT_PRIORITY = {
        "tushare": 1,
        "choice": 2,
        "ibkr": 3,
        "free": 4,  # 免费源始终最低，不可调整
    }

    def __init__(self, api_keys: dict | None = None):
        self._api_keys = api_keys or _read_apikeys()
        self._adapters: dict[str, DataSourceAdapter] = {}
        self._initialized = False
        self._priority = dict(self.DEFAULT_PRIORITY)

    # ── 初始化 ───────────────────────────────────────────
    def initialize(self) -> bool:
        """初始化所有已配置的数据源"""
        if self._initialized:
            return True

        success_count = 0

        # TuShare
        if self.get_api_key("tushare"):
            try:
                adapter = TuShareAdapter(api_key=self.get_api_key("tushare"))
                if adapter.connect():
                    self._adapters["tushare"] = adapter
                    success_count += 1
            except Exception as e:
                logger.warning(f"TuShare 初始化失败: {e}")

        # Choice
        choice_config = self._api_keys.get("choice", {})
        choice_key = choice_config.get("api_key", "") if isinstance(choice_config, dict) else ""
        if choice_key or self._api_keys.get("choice_enabled"):
            try:
                adapter = ChoiceAdapter(
                    api_key=choice_key,
                    choice_url=choice_config.get("url", "http://127.0.0.1:8089")
                    if isinstance(choice_config, dict)
                    else "http://127.0.0.1:8089",
                    choice_use_cloud=choice_config.get("use_cloud", False)
                    if isinstance(choice_config, dict)
                    else False,
                )
                if adapter.connect():
                    self._adapters["choice"] = adapter
                    success_count += 1
            except Exception as e:
                logger.warning(f"Choice 初始化失败: {e}")

        # IBKR
        ibkr_config = self._api_keys.get("ibkr", {})
        if isinstance(ibkr_config, dict) and ibkr_config.get("enabled"):
            try:
                adapter = IBKRAdapter(
                    api_key="ibkr",  # 标记为已配置
                    ibkr_host=ibkr_config.get("host", "127.0.0.1"),
                    ibkr_port=ibkr_config.get("port", 7497),
                    ibkr_client_id=ibkr_config.get("client_id", 1),
                )
                if adapter.connect():
                    self._adapters["ibkr"] = adapter
                    success_count += 1
            except Exception as e:
                logger.warning(f"IBKR 初始化失败: {e}")

        self._initialized = True
        if success_count > 0:
            logger.info(f"数据源管理器初始化完成: {success_count} 个收费源已连接")
        return success_count > 0

    # ── API Key 管理 ─────────────────────────────────────
    def get_api_key(self, source: str) -> str:
        """获取指定数据源的 API Key"""
        config = self._api_keys.get(source, {})
        if isinstance(config, dict):
            return str(config.get("api_key", ""))
        return str(config) if config else ""

    def update_api_key(self, source: str, api_key: str, **extra) -> dict:
        """更新数据源 API Key 及额外配置

        返回: {"success": bool, "tested": bool, "error": str}
        """
        old_config = self._api_keys.get(source, {})
        if not isinstance(old_config, dict):
            old_config = {}

        new_config = {
            "api_key": api_key,
            "updated_at": datetime.now().isoformat(),
            **{k: v for k, v in extra.items() if v is not None},
        }

        self._api_keys[source] = new_config
        _write_apikeys(self._api_keys)

        # 断开旧连接
        if source in self._adapters:
            self._adapters[source].disconnect()
            del self._adapters[source]
            self._initialized = False

        # 测试新连接（如果提供了 key）
        if api_key:
            return self._test_connection(source)
        return {"success": True, "tested": False, "error": ""}

    def delete_api_key(self, source: str) -> dict:
        """删除数据源 API Key 配置"""
        if source in self._api_keys:
            del self._api_keys[source]
            _write_apikeys(self._api_keys)
        if source in self._adapters:
            self._adapters[source].disconnect()
            del self._adapters[source]
        return {"success": True}

    def _test_connection(self, source: str) -> dict:
        """测试数据源连接"""
        api_key = self.get_api_key(source)
        if not api_key:
            return {"success": False, "tested": False, "error": "未配置 API Key"}

        try:
            if source == "tushare":
                adapter = TuShareAdapter(api_key=api_key)
                ok = adapter.connect()
                if ok:
                    self._adapters[source] = adapter
                return {"success": True, "tested": ok, "error": "" if ok else "凭据验证失败"}
            elif source == "choice":
                cfg = self._api_keys.get("choice", {})
                adapter = ChoiceAdapter(
                    api_key=api_key,
                    choice_url=cfg.get("url", "http://127.0.0.1:8089")
                    if isinstance(cfg, dict)
                    else "http://127.0.0.1:8089",
                    choice_use_cloud=cfg.get("use_cloud", False) if isinstance(cfg, dict) else False,
                )
                ok = adapter.connect()
                if ok:
                    self._adapters[source] = adapter
                return {"success": True, "tested": ok, "error": "" if ok else "连接失败"}
            elif source == "ibkr":
                cfg = self._api_keys.get("ibkr", {})
                if not isinstance(cfg, dict) or not cfg.get("enabled"):
                    return {"success": False, "tested": False, "error": "IBKR 未启用"}
                adapter = IBKRAdapter(
                    api_key="ibkr",
                    ibkr_host=cfg.get("host", "127.0.0.1"),
                    ibkr_port=int(cfg.get("port", 7497)),
                    ibkr_client_id=int(cfg.get("client_id", 1)),
                )
                ok = adapter.connect()
                if ok:
                    self._adapters[source] = adapter
                return {"success": True, "tested": ok, "error": "" if ok else "连接失败"}
            else:
                return {"success": False, "tested": False, "error": f"未知数据源: {source}"}
        except ImportError as e:
            return {"success": False, "tested": False, "error": f"缺少依赖: {e}"}
        except Exception as e:
            return {"success": False, "tested": False, "error": str(e)}

    # ── 优先级管理 ───────────────────────────────────────
    def get_priority(self, source: str) -> int:
        return self._priority.get(source, 99)

    def update_priority(self, source: str, priority: int):
        """调整数据源优先级"""
        if source == "free":
            return
        self._priority[source] = priority

    def get_ordered_sources(self) -> list[str]:
        """获取按优先级排序的可用数据源列表"""
        available = []
        for source, adapter in self._adapters.items():
            if adapter.is_configured and adapter.is_connected and adapter.has_quota():
                available.append(source)
        available.sort(key=lambda s: self._priority.get(s, 99))
        # 免费源始终在最后
        available.append("free")
        return available

    # ── 状态查询 ─────────────────────────────────────────
    def get_all_status(self) -> list[dict]:
        """获取所有数据源的状态"""
        statuses = []

        # TuShare
        tushare_key = self.get_api_key("tushare")
        ts_adapter = self._adapters.get("tushare")
        statuses.append(
            {
                "id": "tushare",
                "name": "TuShare Pro",
                "icon": "📊",
                "configured": bool(tushare_key),
                "connected": ts_adapter is not None and ts_adapter.is_connected,
                "has_quota": ts_adapter.has_quota() if ts_adapter else True,
                "quota": ts_adapter.quota_info.__dict__ if ts_adapter else {"daily_limit": 2000, "daily_used": 0},
                "priority": self._priority.get("tushare", 1),
                "markets": ["A"],
                "setup_guide": "在 https://tushare.pro 注册获取 Token，填入下方。120积分/分钟。",
            }
        )

        # Choice
        choice_cfg = self._api_keys.get("choice", {})
        choice_adapter = self._adapters.get("choice")
        choice_configured = bool(choice_cfg.get("api_key")) if isinstance(choice_cfg, dict) else bool(choice_cfg)
        statuses.append(
            {
                "id": "choice",
                "name": "东方财富 Choice",
                "icon": "🏦",
                "configured": choice_configured,
                "connected": choice_adapter is not None and choice_adapter.is_connected,
                "has_quota": True,
                "quota": {"daily_limit": -1, "daily_used": 0},
                "priority": self._priority.get("choice", 2),
                "markets": ["A", "HK", "US"],
                "setup_guide": "需本地安装 Choice 终端，在终端设置中开启 HTTP 桥接。",
                "extra_config": {
                    "url": choice_cfg.get("url", "http://127.0.0.1:8089")
                    if isinstance(choice_cfg, dict)
                    else "http://127.0.0.1:8089",
                    "use_cloud": choice_cfg.get("use_cloud", False) if isinstance(choice_cfg, dict) else False,
                },
            }
        )

        # IBKR
        ibkr_cfg = self._api_keys.get("ibkr", {})
        ibkr_adapter = self._adapters.get("ibkr")
        ibkr_enabled = isinstance(ibkr_cfg, dict) and ibkr_cfg.get("enabled", False)
        statuses.append(
            {
                "id": "ibkr",
                "name": "Interactive Brokers",
                "icon": "🌍",
                "configured": ibkr_enabled,
                "connected": ibkr_adapter is not None and ibkr_adapter.is_connected,
                "has_quota": True,
                "quota": {"daily_limit": -1, "daily_used": 0},
                "priority": self._priority.get("ibkr", 3),
                "markets": ["A", "HK", "US"],
                "setup_guide": "需安装 TWS/IB Gateway 并登录，在 API 设置中开启连接。",
                "extra_config": {
                    "host": ibkr_cfg.get("host", "127.0.0.1") if isinstance(ibkr_cfg, dict) else "127.0.0.1",
                    "port": ibkr_cfg.get("port", 7497) if isinstance(ibkr_cfg, dict) else 7497,
                    "client_id": ibkr_cfg.get("client_id", 1) if isinstance(ibkr_cfg, dict) else 1,
                    "enabled": ibkr_enabled,
                },
            }
        )

        # 免费源
        statuses.append(
            {
                "id": "free",
                "name": "免费数据源 (东方财富/新浪/腾讯)",
                "icon": "🆓",
                "configured": True,
                "connected": True,
                "has_quota": True,
                "quota": {"daily_limit": -1, "daily_used": 0},
                "priority": self._priority.get("free", 4),
                "markets": ["A"],
                "setup_guide": "默认启用，无需配置。由 AKShare/东方财富/新浪/腾讯提供免费行情。",
            }
        )

        return statuses

    # ── 通用数据获取（优先级调度 + 自动降级） ──────────────
    def _try_fetch(
        self,
        method: str,
        *args,
        **kwargs,
    ) -> FetchResult:
        """通用获取: 按优先级依次尝试，首个成功即返回"""
        sources = self.get_ordered_sources()

        for source in sources:
            if source == "free":
                # 免费源由 DataFetcher 处理，这里返回 None 表示降级
                return FetchResult(success=False, source_name="free", data=None, error="降级到免费源")

            adapter = self._adapters.get(source)
            if adapter is None:
                continue

            try:
                fn = getattr(adapter, method, None)
                if fn is None:
                    continue

                result = fn(*args, **kwargs)
                if result.success and result.data is not None and not result.data.empty:
                    logger.debug(f"数据源 [{source}] 成功获取 {method}")
                    return result
                else:
                    logger.debug(f"数据源 [{source}] {method} 失败: {result.error}")
            except Exception as e:
                logger.warning(f"数据源 [{source}] 异常: {e}")
                continue

        return FetchResult(success=False, source_name="all", error="所有收费数据源均不可用")

    # ── 便利方法 ─────────────────────────────────────────
    def fetch_daily_kline(
        self,
        code: str,
        start_date: str = "20200101",
        end_date: str = "",
        adjust: str = "qfq",
    ) -> FetchResult:
        return self._try_fetch("fetch_daily_kline", code, start_date, end_date, adjust)

    def fetch_financial_data(self, code: str) -> FetchResult:
        return self._try_fetch("fetch_financial_data", code)

    def fetch_index_daily(
        self,
        index_code: str,
        start_date: str = "20200101",
        end_date: str = "",
    ) -> FetchResult:
        return self._try_fetch("fetch_index_daily", index_code, start_date, end_date)

    def fetch_money_flow(self, code: str) -> FetchResult:
        return self._try_fetch("fetch_money_flow", code)

    def fetch_stock_list(self) -> FetchResult:
        return self._try_fetch("fetch_stock_list")

    def fetch_realtime_quote(self, codes: list[str]) -> FetchResult:
        return self._try_fetch("fetch_realtime_quote", codes)

    def shutdown(self):
        """关闭所有数据源连接"""
        for source, adapter in self._adapters.items():
            try:
                adapter.disconnect()
            except Exception:
                pass
        self._adapters.clear()
        self._initialized = False
