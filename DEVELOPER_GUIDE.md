# CloudKnight 开发者指南

## 交易系统生命周期架构

### 概述

CloudKnight 实时交易引擎基于**双层调度架构**，通过 5 秒心跳循环驱动全天候自动化运行：

```
┌──────────────────────────────────────────────┐
│                _run_loop (5s tick)            │
│                                              │
│  ① 获取当前 A 股交易阶段 (TradingPhase)       │
│  ② 阶段切换时执行一次性动作 (_dispatch_phase)  │
│  ③ 每 tick 调用生命周期驱动器 (_lifecycle_tick) │
└──────────────────┬───────────────────────────┘
                   │
     ┌─────────────┴──────────────┐
     │                            │
     ▼                            ▼
┌─────────────┐          ┌────────────────────┐
│ TradingPhase│          │  LifecyclePhase    │
│ (日程调度)   │          │  (策略生命周期)     │
│ 一次性执行   │          │  循环执行/冷却机制  │
└─────────────┘          └────────────────────┘
```

**第一层**：`TradingPhase` — 按 A 股日历划分的 8 个宏观阶段（盘前→竞价→早盘→午休→午盘→盘后→休市），每个阶段进入时执行一次性动作。

**第二层**：`LifecyclePhase` — 8 个策略微观生命周期阶段（选股→信号→止损→交易→计划→回测→ML→空闲），每个阶段有独立冷却间隔，按调度策略自动流转。

---

### 类关系图

```
live_engine.py
├── TradingCalendar (trading_calendar.py)
│   └── get_current_phase(now) → TradingPhase
├── LifecyclePhase (Enum)
│   ├── IDLE(0)  SCREENING(1)  SIGNAL_SCAN(2)  STOP_MONITOR(3)
│   ├── TRADE_EXEC(4)  TRADE_PLAN(5)  BACKTEST(6)  MACHINE_LEARNING(7)
│   └── 每个值: (key, label, order)
├── LiveTradingEngine
│   ├── _run_loop()          ← 主循环 (5s tick)
│   ├── _lifecycle_tick()    ← 生命周期入口
│   ├── _lifecycle_determine_phase()  ← 阶段决策
│   ├── _lifecycle_can_run() ← 冷却检查
│   ├── _lifecycle_execute() ← 执行动作 + 广播策略阶段
│   ├── _lifecycle_screening()  → PoolManager.screen_all()
│   ├── _lifecycle_signal_scan() → SignalHunter.scan_all_strategies()
│   ├── _lifecycle_stop_monitor() → PaperTrader 持仓检查
│   ├── _lifecycle_trade_exec()   → PaperTrader 模拟交易 + ML 门控
│   ├── _lifecycle_trade_plan()   → pool.create_trade_plan()
│   ├── _lifecycle_backtest()     → 分层统计
│   ├── _lifecycle_machine_learning() → MLEngine.train_and_predict()
│   ├── _update_all_strategy_phases()  ← 广播引擎级阶段
│   └── _update_strategy_phase()  ← 精确更新单个策略
```

---

### 生命周期阶段详解

#### ① 选股 (SCREENING)
```
触发条件: CLOSED / POST_MARKET
冷却间隔: 30 秒
批量大小: 200 只/批
目标数量: 每策略 focus 层 ≥ 5 只

执行流程:
  1. 检查 _screening_done → 是则跳过
  2. 从 StockInfoCache 取下一批 200 只代码
  3. 调用 PoolManager.screen_all() → diagnose() → 自动入池
  4. 检查各策略池 focus 层数量 ≥ 5 → 满则标记完成
  5. 全部代码遍历完仍未满额 → 标记完成（尽力模式）
```

#### ② 信号扫描 (SIGNAL_SCAN)
```
触发条件: 所有活跃时段
冷却间隔: 交易时段 60s / 非交易时段 30s

执行流程:
  1. 构建各策略股票代码映射 (POOL_MAX_SIZE=30)
  2. SignalHunter.scan_all_strategies() 并行扫描 (120 日数据)
  3. SignalHunter.build_trading_plan() 生成交易计划
  4. 统计 buy/sell 信号数量 → 更新 _latest_signals
```

#### ③ 止盈止损监控 (STOP_MONITOR)
```
触发条件: MORNING / AFTERNOON
冷却间隔: 30 秒

执行流程:
  1. 遍历 PaperTrader 各策略账户所有持仓
  2. 获取最新收盘价 → 计算浮动盈亏
  3. 默认止损 -7% / 止盈 +15%
  4. 若股票池有自定义计划则使用计划中的价位
  5. 触发时记录日志
```

#### ④ 执行交易 (TRADE_EXEC)
```
触发条件: MORNING / AFTERNOON (信号扫描 + 止损监控完成后)
冷却间隔: 5 秒

执行流程:
  1. 遍历 _trading_plan 中的 buy/sell 信号
  2. 买入: ML 决策门控 (MLDecisionGate.evaluate_buy) →
     - bullish → ✅ 确认买入
     - bearish → ❌ 驳回 (权重不足时 override)
     - neutral → 放行
  3. 买入仓位: 现金 × 10%，按 100 股整数手
  4. 卖出: 清仓持仓 → 计算盈亏 → 释放现金
  5. 精确更新 executed_strategies 的阶段状态
  6. PaperTrader.save() 持久化
```

#### ⑤ 制定交易计划 (TRADE_PLAN)
```
触发条件: PRE_MARKET / LUNCH / CLOSED / POST_MARKET
冷却间隔: 300 秒 (5 分钟)

执行流程:
  1. 为各策略池 focus 层股票调用 pool.create_trade_plan()
  2. 生成止盈止损价位
```

#### ⑥ 数据回测 (BACKTEST)
```
触发条件: CLOSED / POST_MARKET
冷却间隔: 600 秒 (10 分钟)
频率控制: 每日一次 (_backtest_date 标记)

执行流程:
  1. 统计各策略池分层数据
  2. focus/watch 数量、Top10 均分
  3. 输出回测摘要
```

#### ⑦ 机器学习 (MACHINE_LEARNING)
```
触发条件: IDLE 状态 + _ml_has_candidates() 返回 True
冷却间隔: 1800 秒 (30 分钟)
重训练: 每 7 天一次 (ML_RETRAIN_DAYS=7)

执行流程:
  1. 收集所有策略池 focus/watch 层标的 (去重, 最多 30 只)
  2. 检查是否到达重训练日:
     - 是 → MLEngine.train_and_predict()
     - 否 → _ml_predict_only()
  3. 同步预测到策略池 (_build_ml_prediction_cache)
  4. 验证上一个交易日 ML 决策 (_validate_ml_decisions)
  5. sklearn 不可用时自动降级为 Z-Score 模式
```

#### ⑧ 空闲 (IDLE)
```
说明: 所有顺序阶段完成后的等待状态
ML 触发: IDLE 状态下检查 _ml_has_candidates() → 有精选标的触发 ML
```

---

### 阶段调度决策树

```
_lifecycle_determine_phase(now, trading_phase) → target_phase

┌─────────────────┬──────────────────────────────────────────┐
│ trading_phase   │ 调度策略                                  │
├─────────────────┼──────────────────────────────────────────┤
│ CLOSED          │ 顺序: SCREENING → SIGNAL_SCAN →          │
│ (非交易日)       │       STOP_MONITOR → TRADE_PLAN →       │
│                 │       BACKTEST → IDLE → (有标?) → ML      │
├─────────────────┼──────────────────────────────────────────┤
│ POST_MARKET     │ 顺序: SCREENING → TRADE_PLAN →           │
│ (盘后)           │       BACKTEST → IDLE → (有标?) → ML    │
├─────────────────┼──────────────────────────────────────────┤
│ PRE_MARKET      │ 固定: TRADE_PLAN                         │
│ (盘前)           │                                          │
├─────────────────┼──────────────────────────────────────────┤
│ AUCTION /       │ 固定: SIGNAL_SCAN                        │
│ AUCTION_RESULT  │                                          │
│ (竞价)           │                                          │
├─────────────────┼──────────────────────────────────────────┤
│ MORNING /       │ 轮转: SIGNAL_SCAN → STOP_MONITOR →      │
│ AFTERNOON       │       TRADE_EXEC → IDLE → 循环           │
│ (交易时段)       │                                          │
├─────────────────┼──────────────────────────────────────────┤
│ LUNCH           │ 固定: TRADE_PLAN                         │
│ (午间)           │                                          │
└─────────────────┴──────────────────────────────────────────┘
```

---

### 配置参考

所有时间参数定义在 `live_engine.py` 中：

| 参数 | 值 | 说明 |
|------|-----|------|
| `_INTERVAL_SCREENING` | 30s | 选股每轮间隔 |
| `_INTERVAL_SIGNAL_SCAN` | 60s(交易)/30s(非交易) | 信号扫描间隔 |
| `_INTERVAL_STOP_MONITOR` | 30s | 止损检查间隔 |
| `_INTERVAL_TRADE_EXEC` | 5s | 交易执行冷却 |
| `_INTERVAL_TRADE_PLAN` | 300s | 交易计划生成间隔 |
| `_INTERVAL_BACKTEST` | 600s | 回测执行间隔 |
| `_INTERVAL_MACHINE_LEARNING` | 1800s | ML 训练预测间隔 |
| `_PAUSE_ON_ERROR` | 120s | 异常后暂停时间 |
| `_SCREEN_BATCH_SIZE` | 200 | 选股每批代码数 |
| `_FOCUS_TARGET` | 5 | 每策略精选层目标数 |
| `LIVE_ENGINE_CHECK_INTERVAL` | 5s | 主循环 tick 间隔 |
| `ML_RETRAIN_DAYS` | 7 | ML 重训练间隔(天) |

---

### 数据流

```
                    ┌──────────────────┐
                    │  TradingCalendar │
                    │  get_current_phase│
                    └────────┬─────────┘
                             │ TradingPhase
                             ▼
┌────────────┐    ┌──────────────────┐    ┌──────────────┐
│StockInfoCache│◄──│ LiveTradingEngine│──►│ OpsMonitor   │
│   (代码库)   │   │                  │   │ (监控采集)    │
└──────┬─────┘   └───┬──────┬───────┘   └──────┬───────┘
       │             │      │                   │
       ▼             ▼      ▼                   ▼
┌────────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│ PoolManager│  │SignalHunter│ │PaperTrader│  │Web Server│
│ (10 pool)  │  │ (信号挖掘) │ │ (模拟交易) │  │(仪表盘)  │
└─────┬──────┘  └─────┬────┘  └─────┬─────┘  └─────┬────┘
      │               │             │              │
      ▼               ▼             ▼              ▼
┌────────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│Diagnoser   │  │MLEngine  │  │MLDecision│  │Dashboard │
│ (多策略诊股)│  │(训练/预测) │  │Gate(门控) │  │(可视化)  │
└────────────┘  └──────────┘  └──────────┘  └──────────┘
```

---

### 各策略阶段追踪机制

```python
# 引擎级阶段 (所有策略同步推进)
self._update_all_strategy_phases(phase, now)
# 用于: SCREENING, SIGNAL_SCAN, STOP_MONITOR, TRADE_PLAN, BACKTEST, MACHINE_LEARNING

# 策略级阶段 (仅执行了交易的策略)
self._update_strategy_phase(strategy_key, LifecyclePhase.TRADE_EXEC, now)
# 用于: TRADE_EXEC 阶段后精确标记
```

数据传递路径：
```
live_engine.get_status()
  → lifecycle_info.strategy_phases = {strategy_key: {phase, label, updated_at}}
    → engine_state.json (磁盘)
      → OpsMonitor._collect_strategy_monitoring() 读取
        → Web API /api/ops/strategies
          → renderOpsStrategies() 渲染阶段 Badge
```

---

### 异常处理与自愈

```
_lifecycle_execute()
  │
  ├── try: handler(now, trading_phase)
  │
  └── except Exception:
        _lifecycle_pause_until = now + 120s  ← 暂停
        logger.warning(f"生命周期 [{phase.label}] 异常: {e}")
        ↓
_lifecycle_tick() 入口:
  if _lifecycle_pause_until and now < _lifecycle_pause_until:
      return  ← 跳过本次 tick
  else:
      _lifecycle_pause_until = None  ← 120 秒后自动恢复
```

每日重置：
```
_new_trading_day() → _reset_lifecycle()
  ├── _lifecycle_phase = IDLE
  ├── _lifecycle_completed.clear()
  ├── _screening_done = False
  ├── _backtest_date = None
  ├── _ml_last_train_date = None
  └── 所有策略阶段 → IDLE
```

---

### 扩展指南

#### 添加新的生命周期阶段

1. 在 `LifecyclePhase` 枚举中添加新值：
```python
class LifecyclePhase(Enum):
    # ... 现有阶段 ...
    MY_NEW_PHASE = ("my_new", "我的新阶段", 8)  # order 自增
```

2. 在 `_lifecycle_execute()` 的 `handlers` 字典中注册：
```python
handlers = {
    # ... 现有 handler ...
    LifecyclePhase.MY_NEW_PHASE: self._lifecycle_my_new_phase,
}
```

3. 实现阶段方法：
```python
def _lifecycle_my_new_phase(self, now: datetime, trading_phase: TradingPhase) -> bool:
    """我的新阶段"""
    # 实现阶段逻辑
    return True  # 返回 True 表示完成
```

4. 在 `_lifecycle_determine_phase()` 中添加触发逻辑：
```python
# 在合适的 trading_phase 分支中添加
elif trading_phase == TradingPhase.CLOSED:
    # 在现有阶段链中插入新阶段
    if not _lifecycle_completed("my_new"):
        return LifecyclePhase.MY_NEW_PHASE
```

5. 添加冷却间隔：
```python
_INTERVAL_MY_NEW_PHASE = 60  # 60 秒冷却
```

6. 更新前端 `phaseBadge`：
```javascript
const phaseBadge = {
    // ... 现有 badge ...
    my_new: { icon: '🆕', label: '新阶段', color: '#...', bg: '...' },
};
```

#### 添加新策略

1. 创建策略文件 `cloudknight/strategies/my_strategy.py`
2. 实现 `CloudKnightStrategy` 子类 + `diagnose()` 静态方法
3. 在 `strategies/__init__.py` 的 `DIAGNOSE_STRATEGIES` 注册
4. 在 `config.py` 的 `STRATEGIES` 和 `POOL_LABEL_MAP` 添加 key
5. 在 `stock_pool.py` 初始化对应池文件
6. 更新前端图标/颜色映射

---

### 调试与监控

#### 引擎状态检查
```bash
python main.py live status
# 或访问 http://127.0.0.1:8080/ 运维面板
```

#### 关键日志标记
| 日志前缀 | 含义 |
|---------|------|
| `🔍 [选股]` | 选股批次进度 |
| `📡 [信号扫描]` | 信号扫描结果统计 |
| `🛡 [止损监控]` | 触发止损信号 |
| `💼 [买入]` | 模拟买入成交 |
| `💰 [卖出]` | 模拟卖出 + 盈亏 |
| `🚫 [ML驳回]` | ML 门控拒绝买入 |
| `🐢 [生命周期] 异常` | 阶段执行异常 |

#### 常用排查点
- `data/engine_state.json` — 引擎实时状态（含 `lifecycle.strategy_phases`）
- `data/paper_race.json` — 模拟交易账户状态
- `data/ml_models/` — ML 模型版本和决策门控数据
- `data/pools/*.json` — 各策略股票池
