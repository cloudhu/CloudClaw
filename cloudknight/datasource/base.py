"""
数据源适配器抽象基类

所有第三方数据源适配器必须实现此接口。
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FetchResult:
    """数据获取结果"""

    success: bool
    data: pd.DataFrame | None = None
    source_name: str = ""
    quota_used: int = 0
    quota_remaining: int = -1  # -1 = 无限制
    error: str = ""


@dataclass
class QuotaInfo:
    """数据源额度信息"""

    daily_limit: int = -1
    daily_used: int = 0
    monthly_limit: int = -1
    monthly_used: int = 0
    last_reset_date: str = ""  # YYYYMMDD


class DataSourceAdapter(ABC):
    """第三方数据源适配器抽象基类

    子类需要实现:
      - name: 数据源名称
      - requires_auth: 是否需要认证
      - _validate_credentials: 验证凭据
      - _do_connect: 建立连接
      - fetch_daily_kline: 获取日K线
      - fetch_financial_data: 获取财务数据
      - _fetch_money_flow: 获取资金流向
    """

    # 子类必须设置
    name: str = "unknown"
    requires_auth: bool = True
    supported_markets: list[str] = ["A"]  # A=A股, HK=港股, US=美股

    def __init__(self, api_key: str = "", api_secret: str = "", **kwargs):
        self._api_key = api_key
        self._api_secret = api_secret
        self._connected = False
        self._client = None
        self._quota = QuotaInfo()
        self._extra_config = kwargs

    # ── 凭据管理 ─────────────────────────────────────────
    @property
    def is_configured(self) -> bool:
        """是否已配置 API Key"""
        return bool(self._api_key)

    @property
    def is_connected(self) -> bool:
        return self._connected

    @abstractmethod
    def _validate_credentials(self) -> bool:
        """验证 API 凭据是否有效，返回 True/False"""
        ...

    @abstractmethod
    def _do_connect(self):
        """建立与数据源的连接"""
        ...

    def connect(self) -> bool:
        """连接数据源（含凭据验证）"""
        if not self.is_configured:
            logger.warning(f"数据源 [{self.name}] 未配置 API Key，跳过连接")
            return False
        try:
            self._do_connect()
            self._connected = True
            logger.info(f"数据源 [{self.name}] 连接成功")
            return True
        except Exception as e:
            logger.error(f"数据源 [{self.name}] 连接失败: {e}")
            self._connected = False
            return False

    def disconnect(self):
        """断开连接"""
        self._connected = False
        self._client = None

    # ── 额度管理 ─────────────────────────────────────────
    def has_quota(self) -> bool:
        """是否还有剩余额度"""
        q = self._quota
        # 每日额度检查
        if q.daily_limit > 0 and q.daily_used >= q.daily_limit:
            return False
        # 每月额度检查
        if q.monthly_limit > 0 and q.monthly_used >= q.monthly_limit:
            return False
        return True

    def _check_daily_reset(self):
        """检查是否需要重置每日计数器"""
        today = datetime.now().strftime("%Y%m%d")
        if self._quota.last_reset_date != today:
            self._quota.daily_used = 0
            self._quota.last_reset_date = today

    def _consume_quota(self, count: int = 1):
        """消耗额度"""
        self._check_daily_reset()
        self._quota.daily_used += count
        self._quota.monthly_used += count

    @property
    def quota_info(self) -> QuotaInfo:
        self._check_daily_reset()
        return self._quota

    # ── 数据获取接口（子类按需覆写） ──────────────────────

    @abstractmethod
    def fetch_daily_kline(
        self,
        code: str,
        start_date: str = "20200101",
        end_date: str = "",
        adjust: str = "qfq",
    ) -> FetchResult:
        """获取日K线数据"""
        ...

    @abstractmethod
    def fetch_financial_data(self, code: str) -> FetchResult:
        """获取财务/基本面数据"""
        ...

    def fetch_index_daily(
        self,
        index_code: str,
        start_date: str = "20200101",
        end_date: str = "",
    ) -> FetchResult:
        """获取指数日线（默认返回空，子类可覆写）"""
        return FetchResult(success=False, data=pd.DataFrame(), source_name=self.name)

    def fetch_money_flow(self, code: str) -> FetchResult:
        """获取资金流向（默认返回空，子类可覆写）"""
        return FetchResult(success=False, data=pd.DataFrame(), source_name=self.name)

    def fetch_stock_list(self) -> FetchResult:
        """获取股票列表（默认返回空，子类可覆写）"""
        return FetchResult(success=False, data=pd.DataFrame(), source_name=self.name)

    def fetch_realtime_quote(self, codes: list[str]) -> FetchResult:
        """获取实时行情（默认返回空，子类可覆写）"""
        return FetchResult(success=False, data=pd.DataFrame(), source_name=self.name)

    def get_status(self) -> dict:
        """获取数据源状态"""
        self._check_daily_reset()
        return {
            "name": self.name,
            "configured": self.is_configured,
            "connected": self._connected,
            "requires_auth": self.requires_auth,
            "quota": {
                "daily_limit": self._quota.daily_limit,
                "daily_used": self._quota.daily_used,
                "daily_remaining": max(0, self._quota.daily_limit - self._quota.daily_used)
                if self._quota.daily_limit > 0
                else -1,
                "monthly_limit": self._quota.monthly_limit,
                "monthly_used": self._quota.monthly_used,
            },
            "supported_markets": self.supported_markets,
        }
