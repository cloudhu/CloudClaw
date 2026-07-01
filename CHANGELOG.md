# Changelog

所有重要的版本变更记录，格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，版本遵循 [SemVer](https://semver.org/lang/zh-CN/)。

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
