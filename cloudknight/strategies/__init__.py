"""策略引擎模块 - 基于 AKQuant 高性能回测框架

提供：
  - STRATEGY_CLASSES: AKQuant 策略类注册表（回测/实盘用）
  - DIAGNOSE_STRATEGIES: 诊股策略注册表（个股诊断用，动态遍历）
  - get_diagnose_strategies(): 获取所有支持诊股的策略
"""

# AKQuant 策略类（继承 akquant.Strategy）
from .base import CloudKnightStrategy
from .bollinger import BollingerBandStrategy
from .dragon_head import DragonHeadStrategy
from .grid import GridStrategy
from .ma_cross import MACrossoverStrategy
from .sparrow import SparrowStrategy
from .trend_accel import TrendAccelerationStrategy
from .turtle import TurtleStrategy
from .value_invest import ValueInvestStrategy
from .volume_breakout import VolumeBreakoutStrategy

STRATEGY_CLASSES = {
    "dragon_head": DragonHeadStrategy,
    "dragon": DragonHeadStrategy,
    "sparrow": SparrowStrategy,
    "turtle": TurtleStrategy,
    "value_invest": ValueInvestStrategy,
    "value": ValueInvestStrategy,
    "grid": GridStrategy,
    "ma_cross": MACrossoverStrategy,
    "bollinger": BollingerBandStrategy,
    "volume_breakout": VolumeBreakoutStrategy,
    "trend_accel": TrendAccelerationStrategy,
}

# ─── 诊股策略注册表 ──────────────────────────────────────────
# 每个策略的元信息，用于个股诊断时动态遍历。
# 新增策略只需在此注册 + 实现 diagnose() 静态方法即可自动参与诊股。
DIAGNOSE_STRATEGIES = {
    "dragon_head": {
        "class": DragonHeadStrategy,
        "label": "龙头战法",
        "emoji": "🐲",
        "auto_add_threshold": 70,  # 评分 >= 此值自动加入策略池
    },
    "sparrow": {
        "class": SparrowStrategy,
        "label": "麻雀战法",
        "emoji": "🐦",
        "auto_add_threshold": 65,
    },
    "turtle": {
        "class": TurtleStrategy,
        "label": "海龟战法",
        "emoji": "🐢",
        "auto_add_threshold": 65,
    },
    "value_invest": {
        "class": ValueInvestStrategy,
        "label": "价值投资",
        "emoji": "💰",
        "auto_add_threshold": 65,
    },
    "bollinger": {
        "class": BollingerBandStrategy,
        "label": "布林带回归",
        "emoji": "📊",
        "auto_add_threshold": 65,
    },
    "grid": {
        "class": GridStrategy,
        "label": "网格交易",
        "emoji": "🔲",
        "auto_add_threshold": 65,
    },
    "ma_cross": {
        "class": MACrossoverStrategy,
        "label": "均线交叉",
        "emoji": "📈",
        "auto_add_threshold": 65,
    },
    "volume_breakout": {
        "class": VolumeBreakoutStrategy,
        "label": "量价突破",
        "emoji": "💥",
        "auto_add_threshold": 65,
    },
    "trend_accel": {
        "class": TrendAccelerationStrategy,
        "label": "趋势加速",
        "emoji": "🚀",
        "auto_add_threshold": 65,
    },
}


def get_diagnose_strategies() -> dict:
    """返回所有支持诊股的策略注册表（key → meta）

    新增策略只需：
    1. 在 strategies/ 目录下创建策略文件
    2. 实现 diagnose() 静态方法
    3. 在 DIAGNOSE_STRATEGIES 中注册
    """
    return dict(DIAGNOSE_STRATEGIES)


__all__ = [
    "STRATEGY_CLASSES",
    "DIAGNOSE_STRATEGIES",
    "get_diagnose_strategies",
    "BollingerBandStrategy",
    "CloudKnightStrategy",
    "DragonHeadStrategy",
    "GridStrategy",
    "MACrossoverStrategy",
    "SparrowStrategy",
    "TrendAccelerationStrategy",
    "TurtleStrategy",
    "ValueInvestStrategy",
    "VolumeBreakoutStrategy",
]
