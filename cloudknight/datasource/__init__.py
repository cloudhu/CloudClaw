"""
多数据源适配层 - 支持 TuShare / Choice / IBKR 等专业数据源

数据源优先级: TuShare > Choice > IBKR > 免费源(东方财富/新浪/腾讯/AKShare)
每个数据源按需绑定 API Key，额度用完自动降级到下一优先级。
"""

from .base import DataSourceAdapter, FetchResult
from .manager import DataSourceManager
from .tushare_source import TuShareAdapter
from .ibkr_source import IBKRAdapter
from .choice_source import ChoiceAdapter

__all__ = [
    "DataSourceAdapter",
    "DataSourceManager",
    "FetchResult",
    "TuShareAdapter",
    "IBKRAdapter",
    "ChoiceAdapter",
]

# 全局单例
_ds_manager: DataSourceManager | None = None


def get_manager() -> DataSourceManager:
    global _ds_manager
    if _ds_manager is None:
        _ds_manager = DataSourceManager()
    return _ds_manager
