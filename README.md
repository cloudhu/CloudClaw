# CloudKnight - 云侠量化交易系统 v2

云侠量化交易系统，基于 **AKQuant** 高性能事件驱动回测引擎。
集成四大主流高胜率策略、技术分析和多策略赛马系统。

## 引擎升级

v2.0.0 重构核心回测引擎为 [AKQuant](https://github.com/akfamily/akquant)：
- **Rust 内核**：20x 于传统 Python 框架的回测性能
- **事件驱动**：精确模拟限价单、止损单、撮合成交
- **零拷贝数据**：`get_history()` 返回 numpy 数组引用
- **AKShare 无缝集成**：`akshare → akquant` 一站式投研闭环

## 版本管理

遵循 [SemVer 2.0.0](https://semver.org/lang/zh-CN/) 规范：`MAJOR.MINOR.PATCH`

```bash
python main.py --version      # 快速查看版本
python main.py version        # 查看详细版本信息
```

## 功能特性

### 四大交易策略
| 策略 | 类型 | 核心思想 | 适合行情 |
|------|------|---------|---------|
| 龙头战法 | 短线追涨 | 追踪涨停连板龙头股 | 强势市场 |
| 麻雀战法 | 高频短线 | 积小胜为大胜，严格止损 | 震荡市 |
| 海龟战法 | 趋势跟踪 | 突破入场，金字塔加仓 | 趋势市 |
| 价值投资 | 长线持有 | 寻找低估优质公司 | 熊市布局 |

每种策略均实现 `CloudKnightStrategy → akquant.Strategy`，覆盖完整交易流程：
选股 → 建仓 → 加仓 → 减持 → 清仓 → 止盈止损

### 自选股票池
每种策略独立筛选评分，支持全市场扫描排序：
- 龙头战法：连板天数 + 量比 + 换手率 + 封板 + 趋势加速
- 麻雀战法：均线多头 + KDJ金叉 + 回调 + RSI + 量价配合
- 海龟战法：突破强度 + 趋势 + ATR波动率 + 流动性 + 加仓空间
- 价值投资：估值分位 + PE/PB + ROE + 股息率 + 市值 + 负债率

### 回测与赛马
- `backtest` - AKQuant 单策略高性能回测
- `compare` - 多策略对比（夏普/收益/回撤/胜率）
- `race` - 多策略赛马模拟盘（每日追踪排名）

## 安装

```bash
pip install -r requirements.txt
python main.py
```

## 命令行

```bash
云侠> pool screen          # 全市场筛选股票池
云侠> strategy dragon      # 选择龙头战法
云侠> backtest             # AKQuant 回测
云侠> compare              # 四策略对比
云侠> race history         # 赛马历史回放
```

## 免责声明

本软件仅供学习和研究使用，不构成任何投资建议。股市有风险，投资需谨慎。
