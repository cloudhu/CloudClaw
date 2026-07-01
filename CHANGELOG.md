# Changelog

所有重要的版本变更记录，格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，版本遵循 [SemVer](https://semver.org/lang/zh-CN/)。

---

## [2.7.1] - 2026-07-01

### Changed
- **全项目内置指标库深度集成**：5 个核心模块完成优化，消除 ~260 行冗余手写代码
  - `strategies/base.py`：6 个手写 `_calc_*` 方法委托给 `indicators.py`，保持 numpy 接口兼容
  - `signal_hunter.py`：~130 行手写 if-elif 链改为读取 `analysis` 字典中的 `IndicatorResult.signal`
  - `stock_diagnose.py`：移除 `_safe_indicator()` 动态调用，5 次 `analyze_*` → 1 次 `comprehensive_analysis()`
  - `live_engine.py`：2 处动态导入 → 顶部导入，盘后总结使用 `analyze_*` 信号判断
  - `market_analyzer.py`：4 次函数调用 → 1 次 `comprehensive_analysis()`，减少 DataFrame 遍历
- 策略信号生成统一使用 `IndicatorResult.signal/.trend/.strength`，消除手工 K/D/%B 判断

### Fixed
- 修复 `live_engine.py` 和 `market_analyzer.py` 中 `IndicatorResult` 可能为 `None` 的问题

---

## [2.7.0] - 2026-07-01

### Added
- **内置指标库 (indicators.py 重构)**：参考教材 §16，从 5 种扩展到 6 大类 30+ 种指标
  - 趋势类 (7): SMA/EMA/WMA/MACD/TRIX/DMI(ADX+DI)/Parabolic SAR
  - 动量类 (8): RSI/KDJ/CCI/WR/MOM/ROC/MFI/Ultimate Oscillator
  - 波动类 (5): Bollinger/ATR/Keltner/Donchian/Historical Volatility
  - 量价类 (8): OBV/VWAP/VPT/Chaikin A_D/VR/Force Index/Volume MA/Volume Ratio
  - 统计类 (6): Beta/Correlation/Linear Regression(Z-Score/Mean Reversion)/Rolling Sharpe/Drawdown
  - K线形态 (8): Doji/Hammer/Shooting Star/Engulfing/Morning Star/Evening Star/Three Soldiers/Three Crows
  - `compute_all_indicators()` 批量计算所有指标到 DataFrame
  - `comprehensive_analysis()` 扩展至 8 维度分析
- **因子引擎 (factor_engine.py)**：参考教材 §14 多因子体系
  - 33 个默认因子覆盖 7 大类: size(2)/value(6)/momentum(4)/quality(7)/growth(4)/volatility(5)/technical(5)
  - `@register_factor` 装饰器注册模式，扩展新因子只需加装饰器
  - 预处理管道: Winsorization → Z-Score 标准化 → 行业+市值中性化
  - 评估体系: Pearson IC / Spearman Rank IC / IR / t-stat / 分层收益 / Long-Short / 换手率
  - 因子合成: 等权 / 自定义权重 / IC_IR 加权三种方式
  - 全部纯 numpy/pandas 实现，零外部依赖

## [2.6.0] - 2026-07-01

### Added
- **系统级风控熔断 (RiskManager)**：参考教材 §15 实盘风控标准
  - 日内亏损熔断（5%）、连续亏损熔断（5笔）、最大回撤熔断（15%）
  - 市场异常波动熔断（大盘暴跌5%/急涨3%暂停开仓）
  - 交易频率限制（单日20笔）、流动性过滤（最小成交量/成交额）
  - 涨停不可买、跌停不可卖的 A 股微观结构检查
  - `CircuitBreaker` 支持自动冷却恢复和状态记录
- **高级回测指标 (backtest_metrics.py)**：参考教材 §10 策略评价体系
  - Sortino 比率、Calmar 比率、Alpha/Beta 系数
  - 95%/99% VaR / CVaR 在险价值
  - 信息比率 (IR)、月度/年度收益率统计
  - 最大连续胜/负期、日胜率
- **参数优化框架 (param_optimizer.py)**：参考教材 §11 参数优化
  - 网格搜索 (Grid Search)：多参数组合遍历 + 指标最大化
  - Walk-Forward Optimization (WFO)：滚动训练窗口 + 测试验证
  - 参数敏感性分析：各参数取值对指标的边际影响
- **策略生命周期回调**：参考教材 §5 完整 on_xxx 回调地图
  - `on_start()`、`pre_open()`、`on_stop()` 生命周期回调
  - `generate_signal()` 静态方法支持策略模式信号生成
- **A 股微观结构处理**：参考教材 §6
  - 涨跌停判断（±9.5% 阈值含容差）
  - 交易合规检查（涨停不可买、跌停不可卖）自动集成到 `enter_long/add_position/exit_position`
  - T+1 制度工程处理、滑点模型 (`apply_slippage`)

### Changed
- `CloudKnightStrategy` 基类增强：生命周期回调 + 微观结构检查 + 滑点模型
- `BacktestEngine` 集成高级指标，`BacktestResult` 新增 `advanced` 字段
- `SignalHunter._generate_signal_from_indicators` 优先使用策略类自身的 `generate_signal()` 方法（策略模式）再回退 if-elif 链
- `config.py` 新增风控参数段 (`RISK_*`)

### Fixed
- `signal_hunter.py` 内部变量 `strategy` → `strategy_key` 避免与模块变量冲突

---

## [2.5.0] - 2026-07-01

### Added
- **高增长策略 (high_growth)**：基于基本面成长因子的选股策略（策略总数 9→10）
  - Scoring：3年EPS增长率 + 3年营收增长率 + PEG + ROE + 净利润增速 + 净资产增速
  - 使用 `_get_growth_finance_extra()` 从 AKShare 财务数据自动提取多期增长指标
  - 参考 DemoStrategy_HighGrowth 的 PEG<1 + 3年增长>中位数 筛选逻辑
- **兴登堡凶兆预警 (Hindenburg Omen)**：市场分化/背离检测
  - 新增 `MarketAnalyzer.check_hindenburg_omen()` 方法
  - 逐日计算个股与上证指数的30日滚动R²，3日变动>30%发出预警
  - 参考 DemoStrategy_HindenburgOmen 的 R² 时序变动检测逻辑

### Changed
- `_get_finance_extra()` 中的 docstring 更新为"基本面策略需要财务数据"
- `screen()` 方法扩展支持 `high_growth` 策略自动获取财务增长数据

---

## [2.4.0] - 2026-07-01

### Added
- **5 大新策略全面上线**：布林带回归、网格交易、均线交叉、量价突破、趋势加速（策略总数 4→9）
- **全策略信号挖掘引擎**：SignalHunter 扩展至全部 9 个策略，每个策略 2-3 个买卖点规则
- **策略活跃监控**：运维面板展示活跃/等待/离线状态、扫描频率、实时买卖信号
- **信号挖掘实时展示**：每个策略卡展示最新 3 条买卖信号，含置信度和理由
- **运维概览卡片**：活跃策略数、买入信号、卖出信号、信号总计一览

### Changed
- SignalHunter `STRATEGY_HANDLERS` 从 4 个策略扩展到 9 个
- LiveEngine `_collect_strategy_pnl()` 动态从 `STRATEGIES` 字典获取全部策略
- OpsMonitor 三个核心方法改为从 `POOL_LABEL_MAP` 动态发现策略，消除硬编码
- Dashboard 策略网格 4 列 → 3 列，图标/颜色映射覆盖全部 9 个策略
- 信号挖掘规则增强：布林带 %B 阈值、网格 MA60 偏离度、均线金叉+放量确认、量价突破多级判断、趋势加速多头排列

### Fixed
- 修复 OpsMonitor `es` 变量作用域导致潜在 UnboundLocalError
- 修复 PaperTrader 策略别名不匹配导致盈亏数据遗漏

---

## [2.3.0] - 2026-07-01

### Added
- **数据源扩展**：新增北向资金、龙虎榜、融资融券等多个数据源
- **多层回退机制**：AKShare → efinance → Tushare → 本地缓存，自动切换数据源
- **数据源健康检查**：`DataFetcher.get_data_source_health()` 实时监测各数据源状态
- **个股深度诊断模块** (`stock_diagnose.py`)：技术面/基本面/资金面/市场面四维诊断
- **诊股 Web 页面** (`diagnose.html`)：ECharts 可视化诊断结果
- **运维监控模块** (`ops_monitor.py`)：系统性能、策略运行、交易操作监控
- **运维面板 API**：`/api/ops/health`、`/api/ops/system` 等端点
- **启动脚本**：`start_all.bat`、`start_live.bat`、`start_web.bat`、`stop_web.bat`

### Changed
- 仪表盘集成 ECharts (`echarts.min.js`) 增强可视化
- Web Server API 端点扩展至 20+
- 代码全面类型现代化：清除所有 actionable lint 警告

### Fixed
- 修复基于 Pyright 的全部可修复类型错误
- 修复 `Dict`/`List`/`Optional` 等废弃类型注解
- 修复隐式字符串拼接 lint 警告

---

## [2.2.0] - 2026-06-30

### Added
- **评分驱动股票池层级系统**：基于评分阈值实现动态四层结构（精选/观察/备选/淘汰），池子大小随市场自适应
- **StockPoolItem 层级字段**：新增 `tier`、`evaluated_at` 字段，支持 `compute_tier()` 自动判定层级
- **淘汰机制**：评分 < 30、累计跌幅 > 8%、入池 20 日未晋级三条规则自动触发淘汰
- **池维护功能**：`StrategyStockPool.maintenance()` — 评估 → 淘汰 → 市场补入完整闭环
- **策略差异化维护频率**：龙头每日/麻雀每2天/海龟每周/价值每2周
- **PoolManager 全局方法**：`evaluate_all()` / `maintenance_all()` / `maintenance_one()` / `tier_overview()`
- **仪表盘层级展示**：层级统计卡片 + 三维筛选按钮 + 层级列 badge（🎯精选/👀观察/📋备选）
- **API 端点**：`POST /api/pool/{key}/evaluate`、`POST /api/pool/{key}/maintenance`
- **API 层级统计**：`/api/pools` 和 `/api/pool/{key}` 返回 `tier_counts`
- **CLI 命令**：`pool eval`、`pool maintain`、`pool tiers`

### Changed
- 股票池表格新增"层级"列，移除了"状态"列，按层级→评分降序排列
- `_inject_pool_gains()` 自动计算并注入层级信息
- README 重构为版本化结构，突出最新特性

---

## [2.1.0] - 2026-06-29

### Added
- **A股时间周期驱动的实时交易引擎**：覆盖全天8个关键时间节点（盘前→竞价→开盘→午休→收盘）
- 非交易日自动执行全策略股票池筛选
- 多线程并行信号扫描（四种策略独立线程）
- `live start/stop/status/market/signals` 命令
- 热点板块排行：按涨停数量统计板块，展示5日涨幅 Top10

---

## [2.0.0] - 2026-06-28

### Added
- 迁移至 [AKQuant](https://github.com/akfamily/akquant) 高性能回测引擎（Rust 内核）
- 四大交易策略：龙头战法、麻雀战法、海龟战法、价值投资
- 事件驱动撮合 + 精确交易成本建模
- 多策略赛马模拟盘（`race`）
- 策略对比分析（`compare`）

### Changed
- 框架从纯 Python 模拟升级为 AKQuant 引擎驱动的真实回测

---

## [1.0.0] - 2026-06-20

### Added
- 初始版本：数据管理、技术指标、回测引擎
- 交互式 CLI 终端（`云侠>` 命令行）
- 全市场股票筛选与评分系统
- SemVer 版本管理模块
