"""
CloudKnight - 云侠量化交易系统 [AKQuant 高性能引擎]

基于 AKQuant (akfamily/akquant) 下一代混合回测框架:
  - Rust 核心引擎，20x 于 Backtrader 的回测性能
  - 事件驱动撮合 + 精确交易成本建模
  - AKShare 免费数据源，零成本量化投研
  - A股时间周期驱动的实时交易引擎
"""

from .version import SemVer as SemVer
from .version import get_semver as get_semver
from .version import get_version

__version__ = get_version()
__author__ = "CloudKnight"
__engine__ = "AKQuant"
