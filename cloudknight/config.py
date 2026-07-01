"""
全局配置模块 - 基于 AKQuant 高性能回测与实时交易引擎
"""

import os
from datetime import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, "cloudknight.db")

STRATEGIES = {
    "dragon_head": "龙头战法",
    "sparrow": "麻雀战法",
    "turtle": "海龟战法",
    "value_invest": "价值投资",
    "grid": "网格交易",
    "ma_cross": "均线交叉",
    "bollinger": "布林带回归",
    "volume_breakout": "量价突破",
    "trend_accel": "趋势加速",
}

DEFAULT_CAPITAL = 1000000.0
DEFAULT_COMMISSION = 0.0003
DEFAULT_STAMP_TAX = 0.001
DEFAULT_SLIPPAGE = 0.001

# AKQuant 回测配置
BACKTEST_LOT_SIZE = 100  # A股 1手 = 100股
BACKTEST_T_PLUS_ONE = True  # T+1 交易规则
BACKTEST_MIN_COMMISSION = 5.0  # 最低佣金 5元

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
POOL_MAX_SIZE = 30  # 每种策略池最多保留
POOL_SCREEN_SAMPLE = 200  # 一次筛选采样数
POOL_MIN_SCORE = 30  # 最低入选评分
POOL_SCREEN_INTERVAL = 5  # 筛选间隔（交易日）

# AKQuant 引擎版本
ENGINE_BACKEND = "akquant"  # 回测引擎后端: "akquant"

# ═══════════════════════════════════════════
# 实时交易：A股交易时间周期
# ═══════════════════════════════════════════
TRADING_PRE_MARKET_START = time(8, 30)
TRADING_AUCTION_START = time(9, 15)
TRADING_AUCTION_RESULT = time(9, 26)
TRADING_MORNING_START = time(9, 30)
TRADING_MORNING_END = time(11, 30)
TRADING_LUNCH_END = time(13, 0)
TRADING_AFTERNOON_END = time(15, 0)

# 实时引擎配置
LIVE_ENGINE_CHECK_INTERVAL = 5  # 主循环检查间隔（秒）
LIVE_ENGINE_SCAN_INTERVAL = 60  # 盘中扫描信号间隔（秒）
LIVE_ENGINE_THREAD_POOL_SIZE = 4  # 策略并行扫描线程数
LIVE_ENGINE_INTRADAY_INTERVAL = 30  # 分时图获取间隔（秒）

# 市场分析指标
INDEX_CODES = {
    "sh000001": "上证指数",
    "sz399001": "深证成指",
    "sz399006": "创业板指",
    "sh000688": "科创50",
    "sh000300": "沪深300",
    "sh000905": "中证500",
}

# 集合竞价分析阈值
AUCTION_VOLUME_RATIO_MIN = 1.5  # 竞价量比最低阈值
AUCTION_PRICE_CHANGE_ALERT = 0.03  # 竞价涨幅预警阈值 3%

# 分时走势分析参数
INTRADAY_MA_PERIOD = 5  # 分时均线周期（分钟）
INTRADAY_VOLUME_SURGE = 2.0  # 成交量突然放大倍数
INTRADAY_BREAKOUT_PCT = 0.02  # 盘中突破幅度 2%

# 日志与存储
LIVE_LOG_DIR = os.path.join(DATA_DIR, "live_logs")
LIVE_TRADE_DIR = os.path.join(DATA_DIR, "live_trades")
os.makedirs(LIVE_LOG_DIR, exist_ok=True)
os.makedirs(LIVE_TRADE_DIR, exist_ok=True)
