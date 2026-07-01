"""
运维监控模块 - 系统性能 + 策略运行 + 交易操作监控

提供:
- SystemMonitor: CPU / 内存 / 磁盘 / 进程信息
- OpsCollector: 聚合所有运维数据（系统 + 引擎 + 策略 + 交易 + 日志）
- EngineStateReader: 读取 live engine 写入的状态文件
"""

import json
import logging
import os
import sys
import time as time_module
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# psutil 是可选的，没有则降级
try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


# ─── 路径常量（延迟导入避免循环） ──────────────────────


def _get_data_paths():
    from .config import DATA_DIR, DEFAULT_CAPITAL, LIVE_LOG_DIR, LIVE_TRADE_DIR

    return DATA_DIR, LIVE_LOG_DIR, LIVE_TRADE_DIR, DEFAULT_CAPITAL


# ─── 系统监控器 ────────────────────────────────────────


@dataclass
class SystemHealth:
    """系统健康数据"""

    # CPU
    cpu_percent: float = 0.0
    cpu_count: int = 0
    # 内存
    memory_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_total_mb: float = 0.0
    # 磁盘
    disk_percent: float = 0.0
    disk_used_gb: float = 0.0
    disk_free_gb: float = 0.0
    # 进程
    process_pid: int = 0
    process_threads: int = 0
    process_memory_mb: float = 0.0
    process_cpu_percent: float = 0.0
    # 运行时间
    uptime_seconds: float = 0.0
    uptime_str: str = ""
    boot_time: str = ""
    # 环境
    python_version: str = ""

    def to_dict(self) -> dict:
        return {
            "cpu_percent": round(self.cpu_percent, 1),
            "cpu_count": self.cpu_count,
            "memory_percent": round(self.memory_percent, 1),
            "memory_used_mb": round(self.memory_used_mb, 1),
            "memory_total_mb": round(self.memory_total_mb, 1),
            "disk_percent": round(self.disk_percent, 1),
            "disk_used_gb": round(self.disk_used_gb, 1),
            "disk_free_gb": round(self.disk_free_gb, 1),
            "process_pid": self.process_pid,
            "process_threads": self.process_threads,
            "process_memory_mb": round(self.process_memory_mb, 1),
            "process_cpu_percent": round(self.process_cpu_percent, 1),
            "uptime_seconds": self.uptime_seconds,
            "uptime_str": self.uptime_str,
            "boot_time": self.boot_time,
            "python_version": self.python_version,
        }


class SystemMonitor:
    """系统性能监控"""

    def __init__(self):
        self._start_time = time_module.time()

    def collect(self) -> SystemHealth:
        """采集当前系统健康数据"""
        health = SystemHealth()
        health.python_version = sys.version.split()[0]

        if not HAS_PSUTIL:
            return health

        try:
            # CPU
            health.cpu_percent = psutil.cpu_percent(interval=0.1)
            health.cpu_count = psutil.cpu_count()

            # 内存
            mem = psutil.virtual_memory()
            health.memory_percent = mem.percent
            health.memory_used_mb = mem.used / (1024 * 1024)
            health.memory_total_mb = mem.total / (1024 * 1024)

            # 磁盘
            try:
                import os as _os

                cwd = _os.getcwd()
                disk = psutil.disk_usage(cwd)
                health.disk_percent = disk.percent
                health.disk_used_gb = disk.used / (1024**3)
                health.disk_free_gb = disk.free / (1024**3)
            except Exception:
                pass

            # 进程
            try:
                proc = psutil.Process(os.getpid())
                health.process_pid = proc.pid
                health.process_threads = proc.num_threads()
                mem_info = proc.memory_info()
                health.process_memory_mb = mem_info.rss / (1024 * 1024)
                health.process_cpu_percent = proc.cpu_percent(interval=0.1)
            except Exception:
                health.process_pid = os.getpid()

            # 运行时间
            boot = psutil.boot_time()
            health.uptime_seconds = time_module.time() - boot
            health.uptime_str = _fmt_uptime(health.uptime_seconds)
            health.boot_time = datetime.fromtimestamp(boot).strftime("%Y-%m-%d %H:%M:%S")

        except Exception as e:
            logger.warning(f"系统监控采集异常: {e}")

        return health


# ─── 引擎状态读取器 ────────────────────────────────────


@dataclass
class EngineStateSnapshot:
    """引擎状态快照"""

    available: bool = False
    state: str = "stopped"
    phase: str = "closed"
    phase_label: str = "离线"
    is_trading_day: bool = False
    scan_interval: int = 60
    last_scan_time: str = ""
    signal_count: int = 0
    decision_count: int = 0
    log_count: int = 0
    signal_details: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "available": self.available,
            "state": self.state,
            "phase": self.phase,
            "phase_label": self.phase_label,
            "is_trading_day": self.is_trading_day,
            "scan_interval": self.scan_interval,
            "last_scan_time": self.last_scan_time,
            "signal_count": self.signal_count,
            "decision_count": self.decision_count,
            "log_count": self.log_count,
            "signal_details": self.signal_details,
        }


class EngineStateReader:
    """读取 live engine 写入的状态文件"""

    STATE_FILE = "engine_state.json"

    @classmethod
    def read(cls, live_log_dir: str | None = None) -> EngineStateSnapshot:
        """读取引擎状态"""
        snapshot = EngineStateSnapshot()

        if live_log_dir is None:
            try:
                from .config import LIVE_LOG_DIR

                live_log_dir = LIVE_LOG_DIR
            except Exception:
                return snapshot

        state_path = os.path.join(live_log_dir, cls.STATE_FILE)
        if not os.path.exists(state_path):
            return snapshot

        try:
            with open(state_path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return snapshot

        # 检查数据新鲜度（超过 5 分钟视为不可用）
        saved_at = data.get("saved_at", "")
        if saved_at:
            try:
                saved_dt = datetime.fromisoformat(saved_at)
                if (datetime.now() - saved_dt).total_seconds() > 300:
                    snapshot.available = True
                    snapshot.state = data.get("state", "unknown")
                    snapshot.phase_label = "离线（数据过期）"
                    return snapshot
            except Exception:
                pass

        snapshot.available = True
        snapshot.state = data.get("state", "stopped")
        snapshot.phase = data.get("current_phase", "closed")
        snapshot.phase_label = data.get("phase_label", "")

        # 信号详情
        sig_details = data.get("signal_details", {})
        snapshot.signal_details = sig_details
        snapshot.signal_count = data.get("latest_signals", sum(d.get("count", 0) for d in sig_details.values()))
        snapshot.decision_count = data.get("decisions_today", 0)
        snapshot.log_count = data.get("log_count", 0)
        snapshot.is_trading_day = data.get("is_trading_day", False)
        snapshot.last_scan_time = data.get("last_scan_time", "")

        return snapshot


# ─── 运维数据聚合器 ────────────────────────────────────


class OpsCollector:
    """聚合所有运维数据"""

    def __init__(self):
        self.system = SystemMonitor()

    def collect_all(self, include_pool_signals: bool = True) -> dict[str, Any]:
        """采集完整的运维面板数据"""
        data_dir, live_log_dir, _live_trade_dir, default_capital = _get_data_paths()

        result = {
            "system": self.system.collect().to_dict(),
            "engine": EngineStateReader.read(live_log_dir).to_dict(),
            "strategies": self._collect_strategy_monitoring(
                data_dir, live_log_dir, default_capital, include_pool_signals
            ),
            "operations": self._collect_trade_operations(data_dir),
            "logs": self._collect_logs(live_log_dir),
            "pool_overview": self._collect_pool_overview(data_dir),
            "collected_at": datetime.now().isoformat(),
        }
        return result

    def _collect_strategy_monitoring(
        self, data_dir: str, live_log_dir: str, default_capital: float, include_signals: bool
    ) -> list[dict]:
        """收集各策略运行监控数据（动态发现全部注册策略）"""
        # 动态获取所有策略信息
        from .stock_pool import MAINTENANCE_INTERVALS, POOL_LABEL_MAP

        strategy_keys = list(POOL_LABEL_MAP.keys())
        strategies: list[dict] = []

        # 读取赛马数据获取各策略持仓/盈亏
        race_data = None
        snapshot_path = os.path.join(data_dir, "paper_race.json")
        if os.path.exists(snapshot_path):
            try:
                with open(snapshot_path, encoding="utf-8") as f:
                    race_data = json.load(f)
            except Exception:
                pass

        # 读取池数据
        pool_dir = os.path.join(data_dir, "pools")
        pool_data = {}
        for key in strategy_keys:
            pool_path = os.path.join(pool_dir, f"{key}.json")
            if os.path.exists(pool_path):
                try:
                    with open(pool_path, encoding="utf-8") as f:
                        pool_data[key] = json.load(f)
                except Exception:
                    pass

        # 读取引擎状态获取最新信号
        engine_signals = {}
        engine_state = {}
        engine_state_path = os.path.join(live_log_dir, "engine_state.json")
        if os.path.exists(engine_state_path):
            try:
                with open(engine_state_path, encoding="utf-8") as f:
                    engine_state = json.load(f)
                engine_signals = engine_state.get("signal_details", {})
            except Exception:
                pass

        # 引擎是否运行中
        engine_running = engine_state.get("state") == "running"
        last_scan_time = engine_state.get("last_scan_time", "")

        # emoji 映射
        emoji_map = {
            "dragon_head": "🐲", "sparrow": "🐦", "turtle": "🐢",
            "value_invest": "💰", "bollinger": "📊", "grid": "🔲",
            "ma_cross": "📈", "volume_breakout": "💥", "trend_accel": "🚀",
        }

        for key in strategy_keys:
            label = POOL_LABEL_MAP.get(key, key)
            interval = MAINTENANCE_INTERVALS.get(key, 5)

            strategy = {
                "key": key,
                "label": label,
                "emoji": emoji_map.get(key, "📊"),
                "is_active": engine_running,
                "signal_count": 0,
                "buy_count": 0,
                "sell_count": 0,
                "last_scan_duration": 0,
                "last_scan_time": last_scan_time,
                "pool_tiers": {"focus": 0, "watch": 0, "broad": 0},
                "equity": default_capital,
                "return_pct": 0,
                "position_count": 0,
                "max_drawdown": 0,
                "trade_count": 0,
                "maintenance_interval_days": interval,
                "scan_frequency_desc": "每日" if interval == 1 else f"每{interval}天" if interval <= 5 else f"每{interval}天(周级)",
                "latest_signals": [],
                "pool_size": 0,
            }

            # 池层级统计
            pd_item = pool_data.get(key, {})
            items = pd_item.get("items", [])
            active_items = [it for it in items if it.get("status") == "active"]
            strategy["pool_size"] = len(active_items)
            for it in active_items:
                tier = it.get("tier", "")
                if tier in strategy["pool_tiers"]:
                    strategy["pool_tiers"][tier] += 1

            # 引擎信号
            sig = engine_signals.get(label, {}) or engine_signals.get(key, {})
            if sig:
                strategy["signal_count"] = sig.get("count", 0)
                strategy["buy_count"] = sig.get("buy", 0)
                strategy["sell_count"] = sig.get("sell", 0)
                strategy["last_scan_duration"] = round(sig.get("duration", 0), 2)

            # 赛马数据（支持别名匹配）
            if race_data:
                accounts = race_data.get("accounts", {})
                # 尝试直接key匹配，也尝试别名
                acc = accounts.get(key)
                if not acc:
                    for ak, av in accounts.items():
                        if av.get("strategy_label", "") == label:
                            acc = av
                            break
                if acc:
                    snapshots = acc.get("daily_snapshots", [])
                    latest = snapshots[-1] if snapshots else {}
                    initial = acc.get("initial_capital", default_capital)
                    equity = latest.get("equity", initial)
                    strategy["equity"] = equity
                    strategy["return_pct"] = round((equity / initial - 1) * 100, 2) if initial > 0 else 0
                    strategy["position_count"] = latest.get("positions", len(acc.get("positions", {})))
                    strategy["trade_count"] = len(acc.get("trades", []))

                    # 最大回撤
                    if snapshots:
                        peak = initial
                        max_dd = 0
                        for s in snapshots:
                            eq = s.get("equity", initial)
                            if eq > peak:
                                peak = eq
                            dd = (peak - eq) / peak * 100 if peak > 0 else 0
                            if dd > max_dd:
                                max_dd = dd
                        strategy["max_drawdown"] = round(max_dd, 2)

            # 从信号详情提取 Top 信号
            if include_signals:
                latest_sigs = self._extract_recent_signals(key, engine_signals)
                strategy["latest_signals"] = latest_sigs

            strategies.append(strategy)

        return strategies

    def _extract_recent_signals(self, strategy_key: str, engine_signals: dict) -> list[dict]:
        """提取最近信号（从引擎状态和池数据），动态使用 POOL_LABEL_MAP"""
        signals = []

        # 动态获取策略标签
        from .stock_pool import POOL_LABEL_MAP

        label = POOL_LABEL_MAP.get(strategy_key, strategy_key)
        sig_data = engine_signals.get(label, {}) or engine_signals.get(strategy_key, {})
        raw_signals = sig_data.get("raw_signals", [])
        for s in raw_signals[:5]:
            signals.append(
                {
                    "code": s.get("code", ""),
                    "name": s.get("name", ""),
                    "type": s.get("signal_type", "buy"),
                    "confidence": s.get("confidence", "medium"),
                    "price": s.get("price", 0),
                    "reason": s.get("reason", ""),
                }
            )

        # 如果引擎无信号，检查池内精选标的
        if not signals:
            pool_path = os.path.join(_get_data_paths()[0], "pools", f"{strategy_key}.json")
            if os.path.exists(pool_path):
                try:
                    with open(pool_path, encoding="utf-8") as f:
                        pool = json.load(f)
                    focus_items = [
                        it for it in pool.get("items", []) if it.get("tier") == "focus" and it.get("status") == "active"
                    ]
                    focus_items.sort(key=lambda x: x.get("score", 0), reverse=True)
                    for it in focus_items[:3]:
                        signals.append(
                            {
                                "code": it.get("code", ""),
                                "name": it.get("name", ""),
                                "type": "watch",
                                "confidence": "high" if it.get("score", 0) >= 85 else "medium",
                                "price": it.get("entry_price", 0),
                                "reason": f"精选层 评分{it.get('score', 0):.0f}",
                            }
                        )
                except Exception:
                    pass

        return signals

    def _collect_trade_operations(self, data_dir: str) -> list[dict]:
        """收集最近的交易操作记录"""
        operations = []

        # 从赛马数据获取交易记录
        snapshot_path = os.path.join(data_dir, "paper_race.json")
        if not os.path.exists(snapshot_path):
            return operations

        try:
            with open(snapshot_path, encoding="utf-8") as f:
                race_data = json.load(f)

            accounts = race_data.get("accounts", {})
            all_trades = []

            for key, acc in accounts.items():
                strategy_label = acc.get("strategy_label", key)
                for t in acc.get("trades", []):
                    all_trades.append(
                        {
                            "date": t.get("date", ""),
                            "time": t.get("time", t.get("date", "")),
                            "strategy": strategy_label,
                            "strategy_key": key,
                            "action": t.get("action", ""),
                            "action_type": _action_type(t.get("action", "")),
                            "code": t.get("code", ""),
                            "name": t.get("name", ""),
                            "price": t.get("price", 0),
                            "volume": t.get("volume", 0),
                            "amount": t.get("amount", 0),
                            "pnl": t.get("pnl", 0),
                            "reason": t.get("reason", ""),
                        }
                    )

            # 按日期时间降序，取最近 50 条
            all_trades.sort(key=lambda x: str(x.get("date", "")), reverse=True)
            operations = all_trades[:50]

        except Exception as e:
            logger.warning(f"读取交易记录异常: {e}")

        return operations

    def _collect_logs(self, live_log_dir: str) -> list[dict]:
        """收集引擎日志"""
        logs = []

        # 从引擎状态文件读取
        state_path = os.path.join(live_log_dir, "engine_state.json")
        if os.path.exists(state_path):
            try:
                with open(state_path, encoding="utf-8") as f:
                    es = json.load(f)
                raw_logs = es.get("phase_logs", [])[-50:]
                for log in raw_logs:
                    logs.append(
                        {
                            "time": log.get("time", log.get("timestamp", "")),
                            "phase": log.get("phase", ""),
                            "event": log.get("event", ""),
                            "detail": log.get("detail", ""),
                        }
                    )
            except Exception:
                pass

        # 如果没有引擎状态日志，从当日总结获取
        if not logs:
            today = datetime.now().strftime("%Y%m%d")
            summary_path = os.path.join(live_log_dir, f"summary_{today}.json")
            if os.path.exists(summary_path):
                try:
                    with open(summary_path, encoding="utf-8") as f:
                        summary = json.load(f)
                    phase_logs = summary.get("phase_logs", [])[-50:]
                    for log in phase_logs:
                        logs.append(
                            {
                                "time": log.get("time", ""),
                                "phase": log.get("phase", ""),
                                "event": log.get("event", ""),
                                "detail": log.get("detail", ""),
                            }
                        )
                except Exception:
                    pass

        return logs

    def _collect_pool_overview(self, data_dir: str) -> dict[str, int]:
        """收集全策略池总览（动态发现所有注册的策略池）"""
        from .stock_pool import POOL_LABEL_MAP

        overview = {"focus": 0, "watch": 0, "broad": 0, "eliminated": 0, "total": 0}
        pool_dir = os.path.join(data_dir, "pools")

        for key in POOL_LABEL_MAP:
            pool_path = os.path.join(pool_dir, f"{key}.json")
            if not os.path.exists(pool_path):
                continue
            try:
                with open(pool_path, encoding="utf-8") as f:
                    pool = json.load(f)
                items = pool.get("items", [])
                for it in items:
                    tier = it.get("tier", "")
                    status = it.get("status", "active")
                    if tier in ("focus", "watch", "broad") and status == "active":
                        overview[tier] = overview.get(tier, 0) + 1
                        overview["total"] += 1
                    elif tier == "eliminated":
                        overview["eliminated"] = overview.get("eliminated", 0) + 1
            except Exception:
                pass

        return overview


def _action_type(action: str) -> str:
    """将 action 字段映射为操作类型"""
    action_lower = (action or "").lower().strip()
    mapping = {
        "buy": "buy",
        "建仓": "buy",
        "买入": "buy",
        "add": "add",
        "加仓": "add",
        "sell": "sell",
        "卖出": "sell",
        "reduce": "reduce",
        "减持": "reduce",
        "close": "close",
        "平仓": "close",
        "stop_loss": "stop_loss",
        "止损": "stop_loss",
        "take_profit": "take_profit",
        "止盈": "take_profit",
    }
    return mapping.get(action_lower, action_lower)


def _fmt_uptime(seconds: float) -> str:
    """格式化运行时间"""
    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    minutes = int((seconds % 3600) // 60)
    parts = []
    if days > 0:
        parts.append(f"{days}天")
    if hours > 0:
        parts.append(f"{hours}小时")
    if minutes > 0 or not parts:
        parts.append(f"{minutes}分钟")
    return "".join(parts)


# 全局单例
ops_collector = OpsCollector()
