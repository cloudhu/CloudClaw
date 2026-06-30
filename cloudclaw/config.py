"""
全局配置模块 - 管理软件的所有配置参数
"""

import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, "cloudclaw.db")

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
