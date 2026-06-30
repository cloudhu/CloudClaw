"""
全局配置模块 - 基于 AKQuant 高性能回测引擎
"""

import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, "cloudknight.db")

STRATEGIES = {
    "dragon_head": "龙头战法",
    "sparrow": "麻雀战法",
    "turtle": "海龟战法",
    "value_invest": "价值投资",
}

DEFAULT_CAPITAL = 1000000.0
DEFAULT_COMMISSION = 0.0003
DEFAULT_STAMP_TAX = 0.001
DEFAULT_SLIPPAGE = 0.001

# AKQuant 回测配置
BACKTEST_LOT_SIZE = 100          # A股 1手 = 100股
BACKTEST_T_PLUS_ONE = True       # T+1 交易规则
BACKTEST_MIN_COMMISSION = 5.0    # 最低佣金 5元

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
KDJ_N = 9
KDJ_M1 = 3
KDJ_M2 = 3
RSI_PERIOD = 14
MA_PERIODS = [5, 10, 20, 60, 120, 250]

BACKTEST_START_DATE = "2020-01-01"
BACKTEST_END_DATE = "2025-12-31"

ST_POOL_FILTER = True
NEW_STOCK_FILTER_DAYS = 60

# 股票池配置
POOL_MAX_SIZE = 30           # 每种策略池最多保留
POOL_SCREEN_SAMPLE = 200     # 一次筛选采样数
POOL_MIN_SCORE = 30          # 最低入选评分
POOL_SCREEN_INTERVAL = 5     # 筛选间隔（交易日）

# AKQuant 引擎版本
ENGINE_BACKEND = "akquant"   # 回测引擎后端: "akquant"
