"""策略引擎模块 - 基于 AKQuant 高性能回测框架"""

# AKQuant 策略类（继承 akquant.Strategy）
from .base import CloudKnightStrategy
from .dragon_head import DragonHeadStrategy
from .sparrow import SparrowStrategy
from .turtle import TurtleStrategy
from .value_invest import ValueInvestStrategy

STRATEGY_CLASSES = {
    "dragon_head": DragonHeadStrategy,
    "dragon": DragonHeadStrategy,
    "sparrow": SparrowStrategy,
    "turtle": TurtleStrategy,
    "value_invest": ValueInvestStrategy,
    "value": ValueInvestStrategy,
}

__all__ = ["CloudKnightStrategy",
           "DragonHeadStrategy", "SparrowStrategy",
           "TurtleStrategy", "ValueInvestStrategy",
           "STRATEGY_CLASSES"]
