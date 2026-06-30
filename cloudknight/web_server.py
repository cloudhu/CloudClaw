"""
CloudKnight 数据驱动仪表盘 - Web 服务

基于 FastAPI 提供 REST API，为前端仪表盘供应：
  系统状态、策略赛马、持仓明细、股票池、交割单、收盘总结
"""

import json
import os
import glob
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, Query, HTTPException
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse, JSONResponse
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


from .config import (
    DATA_DIR, LIVE_LOG_DIR, LIVE_TRADE_DIR,
    DEFAULT_CAPITAL, STRATEGIES,
)
from .stock_pool import POOL_DIR, TIER_FOCUS, TIER_WATCH, TIER_BROAD, TIER_LABELS, TIER_THRESHOLDS

# ─── 常量和路径 ─────────────────────────────────────────────

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
DASHBOARD_HTML = os.path.join(STATIC_DIR, "dashboard.html")

SNAPSHOT_FILE = os.path.join(DATA_DIR, "paper_race.json")

STRATEGY_KEYS = ["dragon", "sparrow", "turtle", "value"]
STRATEGY_LABELS = {
    "dragon": "龙头战法", "sparrow": "麻雀战法",
    "turtle": "海龟战法", "value": "价值投资",
}


# ─── 数据读取辅助 ───────────────────────────────────────────

def _read_json(path: str, default=None):
    """安全读取 JSON 文件"""
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _list_summary_files() -> List[str]:
    """列出所有收盘总结文件，按日期排序"""
    pattern = os.path.join(LIVE_LOG_DIR, "summary_*.json")
    files = glob.glob(pattern)
    dates = []
    for f in files:
        base = os.path.splitext(os.path.basename(f))[0]
        dates.append(base.replace("summary_", ""))
    dates.sort(reverse=True)
    return dates


def _inject_pool_gains(items: List[dict]) -> List[dict]:
    """为股票池数据注入累计涨幅、当日涨幅、最大涨幅、层级"""
    from .data_manager import DataFetcher
    if not items:
        return items

    fetcher = DataFetcher()
    col_map = {"日期": "date", "开盘": "open", "收盘": "close", "最高": "high",
               "最低": "low", "成交量": "volume", "涨跌幅": "pct_change"}

    for item in items:
        code = item.get("code", "")
        entry_price = item.get("entry_price", 0) or 0

        # 确保 tier 字段存在
        tier = item.get("tier", "")
        score = item.get("score", 0) or 0

        try:
            df = fetcher.fetch_daily_kline(str(code), start_date="20240101")
            if df.empty or len(df) < 1:
                _set_default_gains(item)
                item["tier"] = tier or _score_to_tier(score)
                continue

            df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
            latest = df.iloc[-1]
            current_close = float(latest.get("close", 0) or 0)
            daily_pct = float(latest.get("pct_change", 0) or 0)
            screened_at = item.get("screened_at", "")

            # 当日涨幅
            item["daily_gain_pct"] = round(daily_pct, 2)

            # 回填 entry_price
            if entry_price <= 0 and screened_at and "date" in df.columns:
                df["_date_str"] = df["date"].astype(str)
                entry_rows = df[df["_date_str"] >= str(screened_at)]
                if not entry_rows.empty:
                    entry_price = round(float(entry_rows.iloc[0]["close"]), 2)

            # 累计涨幅
            if entry_price > 0 and current_close > 0:
                cum_gain = round((current_close / entry_price - 1) * 100, 2)
                item["cumulative_gain_pct"] = cum_gain
            else:
                item["cumulative_gain_pct"] = None
                cum_gain = 0

            # 最大涨幅
            if entry_price > 0 and screened_at and "date" in df.columns and len(df) > 1:
                df["_date_str"] = df["date"].astype(str)
                mask = df["_date_str"] >= str(screened_at)
                if mask.any():
                    max_close = float(df.loc[mask, "close"].max())
                    item["max_gain_pct"] = round((max_close / entry_price - 1) * 100, 2)
                else:
                    item["max_gain_pct"] = round((current_close / entry_price - 1) * 100, 2)
            elif entry_price > 0 and current_close > 0:
                item["max_gain_pct"] = round((current_close / entry_price - 1) * 100, 2)
            else:
                item["max_gain_pct"] = None

            # 层级：优先用已持久化的，否则实时计算
            if tier:
                item["tier"] = tier
            else:
                item["tier"] = _score_to_tier(score, cum_gain)

        except Exception:
            _set_default_gains(item)
            item["tier"] = tier or _score_to_tier(score)

    return items


def _score_to_tier(score: float, cum_gain: float = 0) -> str:
    """根据评分和累计涨幅计算层级"""
    if score >= 80:
        return TIER_FOCUS
    elif score >= 60:
        return TIER_WATCH
    elif score >= 40:
        return TIER_BROAD
    else:
        return "eliminated"


def _set_default_gains(item: dict):
    item["cumulative_gain_pct"] = None
    item["daily_gain_pct"] = None
    item["max_gain_pct"] = None


def _list_trade_files() -> List[str]:
    """列出所有交割单文件"""
    pattern = os.path.join(LIVE_TRADE_DIR, "*.json")
    files = glob.glob(pattern)
    dates = []
    for f in files:
        base = os.path.splitext(os.path.basename(f))[0]
        dates.append(base)
    dates.sort(reverse=True)
    return dates


def _api_key_to_pool_key(api_key: str) -> str:
    """API 短键 → 池文件键映射"""
    mapping = {"dragon": "dragon_head", "sparrow": "sparrow",
               "turtle": "turtle", "value": "value_invest"}
    return mapping.get(api_key, api_key)


def _read_pool(strategy_key: str) -> Optional[dict]:
    """读取策略股票池"""
    pool_files = {
        "dragon": "dragon_head.json",
        "sparrow": "sparrow.json",
        "turtle": "turtle.json",
        "value": "value_invest.json",
    }
    filename = pool_files.get(strategy_key, f"{strategy_key}.json")
    path = os.path.join(POOL_DIR, filename)
    return _read_json(path)


def _read_trades(limit: int = 200) -> List[dict]:
    """读取交割单记录（优先从交易目录读取，其次从赛马快照）"""
    trades = []

    # 尝试从 LIVE_TRADE_DIR 读取
    dates = _list_trade_files()
    for date in dates[:30]:
        path = os.path.join(LIVE_TRADE_DIR, f"{date}.json")
        data = _read_json(path)
        if data:
            if isinstance(data, list):
                for t in data:
                    if isinstance(t, dict):
                        t.setdefault("_date", date)
                        trades.append(t)
            elif isinstance(data, dict):
                records = data.get("trades", data.get("records", []))
                for t in records:
                    t.setdefault("_date", date)
                    trades.append(t)

    if not trades:
        # 从赛马快照中尝试提取
        race = _read_json(SNAPSHOT_FILE, {})
        accounts = race.get("accounts", {})
        for key, acc in accounts.items():
            acc_trades = acc.get("trades", [])
            for t in acc_trades:
                t["_strategy"] = key
                t["_strategy_label"] = STRATEGY_LABELS.get(key, key)
                trades.append(t)

    trades.sort(key=lambda x: str(x.get("_date", x.get("date", ""))), reverse=True)
    return trades[:limit]


def _build_strategy_data(race_data: dict) -> List[dict]:
    """从赛马数据构建策略账户摘要"""
    strategies = []
    accounts = race_data.get("accounts", {})
    for key in STRATEGY_KEYS:
        label = STRATEGY_LABELS[key]
        acc = accounts.get(key, {})
        snapshots = acc.get("daily_snapshots", [])
        initial_cap = acc.get("initial_capital", DEFAULT_CAPITAL)

        # 当前权益
        latest = snapshots[-1] if snapshots else {}
        equity = latest.get("equity", initial_cap)

        # 收益率
        total_return = round((equity / initial_cap - 1) * 100, 2)

        # 日收益率
        daily_return = 0
        if len(snapshots) >= 2:
            prev_eq = snapshots[-2]["equity"]
            if prev_eq > 0:
                daily_return = round((equity / prev_eq - 1) * 100, 2)

        # 最大回撤
        max_dd = 0
        if len(snapshots) >= 2:
            values = [s["equity"] for s in snapshots]
            cummax = [values[0]]
            for v in values[1:]:
                cummax.append(max(cummax[-1], v))
            dd = max(abs((v - cm) / cm * 100) for v, cm in zip(values, cummax) if cm > 0)
            max_dd = round(dd, 2)

        # 权益曲线
        equity_curve = [
            {"date": s.get("date", ""), "equity": round(s.get("equity", 0), 2)}
            for s in snapshots
        ]

        strategies.append({
            "key": key,
            "label": label,
            "initial_capital": initial_cap,
            "cash": round(acc.get("cash", initial_cap), 2),
            "equity": round(equity, 2),
            "total_return_pct": total_return,
            "daily_return_pct": daily_return,
            "position_count": latest.get("positions", 0),
            "max_drawdown": max_dd,
            "status": acc.get("status", "idle"),
            "equity_curve": equity_curve,
            "trade_count": len(acc.get("trades", [])),
        })

    # 按收益率排序
    strategies.sort(key=lambda s: s["total_return_pct"], reverse=True)
    for i, s in enumerate(strategies):
        s["rank"] = i + 1

    return strategies


# ─── 应用工厂 ────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="CloudKnight Dashboard",
        description="云侠量化交易系统 - 数据驱动仪表盘",
        version="2.0",
    )

    # ── 静态文件 ──────────────────────────────────────────
    os.makedirs(STATIC_DIR, exist_ok=True)

    if os.path.exists(STATIC_DIR):
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    # ── 仪表盘首页 ────────────────────────────────────────
    @app.get("/")
    async def index():
        if os.path.exists(DASHBOARD_HTML):
            return FileResponse(DASHBOARD_HTML)
        raise HTTPException(404, "dashboard.html 未找到，请确认 static 目录已部署")

    # ── API: 系统状态 ─────────────────────────────────────
    @app.get("/api/status")
    async def get_status():
        """获取系统运行状态"""
        race = _read_json(SNAPSHOT_FILE, {})
        return {
            "timestamp": datetime.now().isoformat(),
            "current_date": race.get("current_date"),
            "stock_pool_size": len(race.get("stock_pool", [])),
            "account_count": len(race.get("accounts", {})),
        }

    # ── API: 策略赛马 ─────────────────────────────────────
    @app.get("/api/strategies")
    async def get_strategies():
        """获取所有策略账户摘要与排名"""
        race = _read_json(SNAPSHOT_FILE, {})
        return _build_strategy_data(race)

    @app.get("/api/strategy/{key}")
    async def get_strategy(key: str):
        """获取单个策略详情（含持仓）"""
        if key not in STRATEGY_KEYS:
            raise HTTPException(404, f"未知策略: {key}")

        race = _read_json(SNAPSHOT_FILE, {})
        strategies = _build_strategy_data(race)
        target = next((s for s in strategies if s["key"] == key), None)
        if not target:
            raise HTTPException(404, f"策略 {key} 无数据")

        # 附加持仓
        acc = race.get("accounts", {}).get(key, {})
        positions_list = []
        for code, pos in acc.get("positions", {}).items():
            positions_list.append({
                "code": code,
                "name": pos.get("name", ""),
                "cost": pos.get("cost", 0),
                "volume": pos.get("volume", 0),
                "current_price": pos.get("current_price", 0),
                "market_value": pos.get("market_value", 0),
                "profit_pct": round(pos.get("profit_pct", 0), 2),
                "hold_days": pos.get("hold_days", 0),
            })

        # 附加交易记录
        trades_list = []
        for t in acc.get("trades", []):
            trades_list.append({
                "date": t.get("date", ""),
                "code": t.get("code", ""),
                "name": t.get("name", ""),
                "action": t.get("action", ""),
                "price": t.get("price", 0),
                "volume": t.get("volume", 0),
                "amount": t.get("amount", 0),
                "reason": t.get("reason", ""),
                "pnl": t.get("pnl", 0),
            })

        target["positions"] = positions_list
        target["trades"] = trades_list

        return target

    # ── API: 持仓汇总 ─────────────────────────────────────
    @app.get("/api/positions")
    async def get_all_positions():
        """获取所有策略的持仓汇总"""
        race = _read_json(SNAPSHOT_FILE, {})
        all_positions = []
        for key in STRATEGY_KEYS:
            label = STRATEGY_LABELS[key]
            acc = race.get("accounts", {}).get(key, {})
            for code, pos in acc.get("positions", {}).items():
                all_positions.append({
                    "strategy": key,
                    "strategy_label": label,
                    "code": code,
                    "name": pos.get("name", ""),
                    "cost": pos.get("cost", 0),
                    "volume": pos.get("volume", 0),
                    "current_price": pos.get("current_price", 0),
                    "market_value": round(pos.get("market_value", 0), 2),
                    "profit_pct": round(pos.get("profit_pct", 0), 2),
                    "hold_days": pos.get("hold_days", 0),
                })
        return all_positions

    # ── API: 股票池 ──────────────────────────────────────
    @app.get("/api/pools")
    async def get_all_pools():
        """获取所有策略股票池（含涨幅数据和层级统计）"""
        pools = {}
        for key in STRATEGY_KEYS:
            label = STRATEGY_LABELS[key]
            data = _read_pool(key)
            items = data.get("items", []) if data else []
            # 注入涨幅数据和层级
            items = _inject_pool_gains(items)
            # 层级统计（只统计 active 的）
            active_items = [it for it in items if it.get("status") == "active"]
            tier_counts = {"focus": 0, "watch": 0, "broad": 0}
            for it in active_items:
                t = it.get("tier", "")
                if t in tier_counts:
                    tier_counts[t] += 1
            pools[key] = {
                "label": label,
                "last_screened": data.get("last_screened") if data else None,
                "total_items": len(active_items),
                "tier_counts": tier_counts,
                "items": items,
            }
        return pools

    @app.get("/api/pool/{key}")
    async def get_pool(key: str):
        """获取单个策略股票池（含涨幅数据和层级统计）"""
        if key not in STRATEGY_KEYS:
            raise HTTPException(404, f"未知策略: {key}")
        data = _read_pool(key)
        if not data:
            return {"label": STRATEGY_LABELS[key], "last_screened": None, "items": [],
                    "tier_counts": {"focus": 0, "watch": 0, "broad": 0}}
        items = _inject_pool_gains(data.get("items", []))
        active_items = [it for it in items if it.get("status") == "active"]
        tier_counts = {"focus": 0, "watch": 0, "broad": 0}
        for it in active_items:
            t = it.get("tier", "")
            if t in tier_counts:
                tier_counts[t] += 1
        return {
            "label": STRATEGY_LABELS[key],
            "last_screened": data.get("last_screened"),
            "total_items": len(active_items),
            "tier_counts": tier_counts,
            "items": items,
        }

    @app.post("/api/pool/{key}/evaluate")
    async def evaluate_pool(key: str):
        """触发指定策略池的层级评估"""
        if key not in STRATEGY_KEYS:
            raise HTTPException(404, f"未知策略: {key}")
        try:
            from .stock_pool import PoolManager
            pm = PoolManager()
            pool = pm.get_pool(_api_key_to_pool_key(key))
            if pool is None:
                raise HTTPException(404, f"策略池不存在: {key}")
            result = pool.evaluate_pool(verbose=False)
            return {"success": True, "tier_result": {k: len(v) for k, v in result.items()}}
        except Exception as e:
            return JSONResponse({"success": False, "error": str(e)}, status_code=500)

    @app.post("/api/pool/{key}/maintenance")
    async def maintenance_pool(key: str):
        """触发指定策略池的完整维护（评估 + 淘汰 + 补入）"""
        if key not in STRATEGY_KEYS:
            raise HTTPException(404, f"未知策略: {key}")
        try:
            from .stock_pool import PoolManager
            pm = PoolManager()
            result = pm.maintenance_one(_api_key_to_pool_key(key), verbose=False)
            return {"success": True, **result}
        except Exception as e:
            return JSONResponse({"success": False, "error": str(e)}, status_code=500)

    # ── API: 交易记录（交割单） ───────────────────────────
    @app.get("/api/trades")
    async def get_trades(
        strategy: Optional[str] = Query(None),
        limit: int = Query(200, ge=10, le=1000),
    ):
        """获取交割单记录，可按策略筛选"""
        trades = _read_trades(limit)
        if strategy and strategy in STRATEGY_KEYS:
            trades = [t for t in trades if t.get("_strategy") == strategy]
        return trades

    # ── API: 收盘总结 ────────────────────────────────────
    @app.get("/api/summaries")
    async def list_summaries():
        """列出所有可用的收盘总结日期"""
        dates = _list_summary_files()
        return {"dates": dates, "latest": dates[0] if dates else None}

    @app.get("/api/summary/{date}")
    async def get_summary(date: str):
        """获取指定日期的收盘总结"""
        path = os.path.join(LIVE_LOG_DIR, f"summary_{date}.json")
        data = _read_json(path)
        if not data:
            raise HTTPException(404, f"未找到 {date} 的收盘总结")
        return data

    # ── API: 市场总览（最新） ─────────────────────────────
    @app.get("/api/market/overview")
    async def get_market_overview():
        """获取最新市场总览（从最近收盘总结提取）"""
        dates = _list_summary_files()
        if not dates:
            return {"indices": {}, "activity": {}, "sentiment": {}}

        path = os.path.join(LIVE_LOG_DIR, f"summary_{dates[0]}.json")
        data = _read_json(path, {})
        return {
            "date": dates[0],
            "overview": data.get("market_overview", {}),
            "indicators": data.get("technical_indicators", {}),
            "sentiment": data.get("sentiment", {}),
            "trend": data.get("market_trend", {}),
            "limit_stats": data.get("limit_stats", {}),
        }

    return app


# ─── 启动入口 ────────────────────────────────────────────────

def start_dashboard(host: str = "0.0.0.0", port: int = 8080, reload: bool = False):
    """启动仪表盘 Web 服务"""
    if not HAS_FASTAPI:
        print("错误: 需要安装 fastapi 和 uvicorn")
        print("  pip install fastapi uvicorn")
        return

    # 确保 static 文件和 dashboard.html 存在
    os.makedirs(STATIC_DIR, exist_ok=True)
    _ensure_dashboard_html()

    print(f"""
╔══════════════════════════════════════════════╗
║    CloudKnight Dashboard                     ║
║    数据驱动仪表盘                              ║
╚══════════════════════════════════════════════╝

  仪表盘地址: http://{host}:{port}
  API 文档:   http://{host}:{port}/docs

  按 Ctrl+C 停止服务
""")

    app = create_app()
    uvicorn.run(app, host=host, port=port, log_level="info")


def _ensure_dashboard_html():
    """生成/更新 dashboard.html（每次启动都重新生成以保证最新）"""
    html = _build_dashboard_html()
    with open(DASHBOARD_HTML, "w", encoding="utf-8") as f:
        f.write(html)


def _build_dashboard_html() -> str:
    """生成仪表盘 HTML（内联，不依赖外部文件）"""
    return '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CloudKnight 仪表盘</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
<style>
:root {
  --bg: #0f1923;
  --panel: #1a2332;
  --border: #2a3a4a;
  --text: #c8d6e5;
  --text-dim: #6d8099;
  --accent: #3498db;
  --green: #27ae60;
  --red: #e74c3c;
  --orange: #f39c12;
  --gold: #f1c40f;
}
* { margin:0; padding:0; box-sizing:border-box; }
body {
  background: var(--bg);
  color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
  font-size: 13px;
  line-height: 1.5;
}
.header {
  background: var(--panel);
  border-bottom: 1px solid var(--border);
  padding: 12px 24px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  position: sticky;
  top: 0;
  z-index: 100;
}
.header h1 { font-size:20px; font-weight:700; color:#fff; letter-spacing:1px; }
.header .badge { font-size:12px; color:var(--accent); }
.header .time { color:var(--text-dim); font-size:12px; }

.tabs {
  display: flex;
  gap: 0;
  background: var(--panel);
  border-bottom: 1px solid var(--border);
  padding: 0 24px;
}
.tabs button {
  background: none;
  border: none;
  color: var(--text-dim);
  padding: 10px 20px;
  cursor: pointer;
  font-size: 13px;
  border-bottom: 2px solid transparent;
  transition: all .2s;
}
.tabs button:hover { color: var(--text); }
.tabs button.active {
  color: var(--accent);
  border-bottom-color: var(--accent);
}

.content { padding: 18px 24px; }
.tab-panel { display: none; }
.tab-panel.active { display: block; }

/* Cards */
.card-row { display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 16px; }
.card {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 14px 18px;
  flex: 1;
  min-width: 160px;
}
.card .card-label { color: var(--text-dim); font-size: 11px; text-transform: uppercase; letter-spacing: 1px; }
.card .card-value { font-size: 22px; font-weight: 700; color: #fff; margin-top: 4px; }
.card .card-sub { color: var(--text-dim); font-size: 11px; margin-top: 2px; }
.card.green .card-value { color: var(--green); }
.card.red .card-value { color: var(--red); }
.card.gold .card-value { color: var(--gold); }

/* Chart containers */
.chart-box {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 14px;
  margin-bottom: 14px;
}
.chart-box .chart-title {
  font-size: 14px;
  font-weight: 600;
  color: var(--text);
  margin-bottom: 10px;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--border);
}
.chart { width: 100%; }
.chart.h300 { height: 300px; }
.chart.h350 { height: 350px; }
.chart.h400 { height: 400px; }

/* Tables */
.table-wrap {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 8px;
  overflow: hidden;
  margin-bottom: 14px;
}
.table-wrap table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
}
.table-wrap th {
  background: #1e2d3d;
  color: var(--text-dim);
  text-align: left;
  padding: 8px 12px;
  font-weight: 500;
  border-bottom: 1px solid var(--border);
  white-space: nowrap;
}
.table-wrap td {
  padding: 7px 12px;
  border-bottom: 1px solid var(--border);
  white-space: nowrap;
}
.table-wrap tr:hover td { background: rgba(52,152,219,0.05); }
.pnl-pos { color: var(--red); }
.pnl-neg { color: var(--green); }
.tag {
  display: inline-block;
  padding: 1px 7px;
  border-radius: 3px;
  font-size: 10px;
  font-weight: 600;
}
.tag-buy { background: rgba(231,76,60,0.15); color: var(--red); }
.tag-sell { background: rgba(39,174,96,0.15); color: var(--green); }
.tag-active { background: rgba(52,152,219,0.15); color: var(--accent); }
.tag-removed { background: rgba(108,128,153,0.15); color: var(--text-dim); }
.tag-focus { background: rgba(231,76,60,0.18); color: #e74c3c; font-weight:700; }
.tag-watch { background: rgba(241,196,15,0.18); color: #f1c40f; }
.tag-broad { background: rgba(52,152,219,0.15); color: var(--accent); }
.tag-eliminated { background: rgba(108,128,153,0.12); color: var(--text-dim); text-decoration:line-through; }

/* Layout grid */
.grid-2 { display:grid; grid-template-columns: 1fr 1fr; gap:14px; }
.grid-3 { display:grid; grid-template-columns: 1fr 1fr 1fr; gap:14px; }
@media (max-width: 1100px) { .grid-2,.grid-3 { grid-template-columns: 1fr; } }

.sentiment-bar {
  height: 10px;
  border-radius: 5px;
  background: linear-gradient(to right,
    #2ecc71 0%, #2ecc71 20%,
    #95a5a6 50%,
    #e74c3c 80%, #e74c3c 100%);
  position: relative;
  margin-top: 4px;
}
.sentiment-indicator {
  position: absolute;
  top: -6px;
  width: 14px;
  height: 22px;
  background: #fff;
  border-radius: 3px;
  transform: translateX(-50%);
  box-shadow: 0 0 8px rgba(255,255,255,0.3);
}

.loading { text-align:center; padding:40px; color:var(--text-dim); }
.empty { text-align:center; padding:30px; color:var(--text-dim); font-size:13px; }
</style>
</head>
<body>

<div class="header">
  <div>
    <h1>CloudKnight<span class="badge"> 量化仪表盘</span></h1>
  </div>
  <div class="time" id="clock">--</div>
</div>

<div class="tabs">
  <button class="active" data-tab="overview">系统总览</button>
  <button data-tab="race">策略赛马</button>
  <button data-tab="positions">持仓明细</button>
  <button data-tab="pools">股票池</button>
  <button data-tab="trades">交割单</button>
  <button data-tab="summary">收盘总结</button>
</div>

<div class="content">
  <!-- 系统总览 -->
  <div class="tab-panel active" id="tab-overview">
    <div class="card-row" id="status-cards"></div>
    <div class="card-row" id="sentiment-card"></div>
    <div class="grid-2">
      <div class="chart-box"><div class="chart-title">四大策略权益曲线</div><div class="chart h350" id="chart-equity-overview"></div></div>
      <div class="chart-box"><div class="chart-title">涨跌分布</div><div class="chart h350" id="chart-advance-decline"></div></div>
    </div>
    <div class="grid-2">
      <div class="chart-box"><div class="chart-title">策略收益率对比</div><div class="chart h300" id="chart-return-bar"></div></div>
      <div class="chart-box"><div class="chart-title">策略风险指标</div><div class="chart h300" id="chart-risk-radar"></div></div>
    </div>
    <div class="chart-box"><div class="chart-title">系统运行日志（最近50条）</div><div class="table-wrap"><table id="table-logs"><thead><tr><th>时间</th><th>阶段</th><th>事件</th><th>详情</th></tr></thead><tbody></tbody></table></div></div>
  </div>

  <!-- 策略赛马 -->
  <div class="tab-panel" id="tab-race">
    <div class="card-row" id="race-cards"></div>
    <div class="chart-box"><div class="chart-title">各策略权益曲线对比</div><div class="chart h400" id="chart-race-equity"></div></div>
    <div class="grid-2">
      <div class="chart-box"><div class="chart-title">收益率走势</div><div class="chart h350" id="chart-race-returns"></div></div>
      <div class="chart-box"><div class="chart-title">最大回撤对比</div><div class="chart h350" id="chart-race-drawdown"></div></div>
    </div>
  </div>

  <!-- 持仓明细 -->
  <div class="tab-panel" id="tab-positions">
    <div class="card-row" id="pos-summary"></div>
    <div class="chart-box"><div class="chart-title">全部持仓清单</div>
      <div style="margin-bottom:10px;display:flex;gap:8px;">
        <select id="pos-filter" style="background:var(--panel);color:var(--text);border:1px solid var(--border);padding:4px 10px;border-radius:4px;">
          <option value="">全部策略</option>
          <option value="dragon">龙头战法</option>
          <option value="sparrow">麻雀战法</option>
          <option value="turtle">海龟战法</option>
          <option value="value">价值投资</option>
        </select>
        <input id="pos-search" placeholder="搜索代码/名称..." style="background:var(--panel);color:var(--text);border:1px solid var(--border);padding:4px 10px;border-radius:4px;width:200px;">
      </div>
      <div class="table-wrap"><table id="table-positions"><thead><tr><th>策略</th><th>代码</th><th>名称</th><th>成本</th><th>持有量</th><th>现价</th><th>市值</th><th>盈亏%</th><th>持有天数</th></tr></thead><tbody></tbody></table></div>
    </div>
  </div>

  <!-- 股票池 -->
  <div class="tab-panel" id="tab-pools">
    <div style="display:flex;gap:8px;margin-bottom:14px;" id="pool-tab-buttons"></div>
    <div class="card-row" id="pool-stats"></div>
    <div class="card-row" id="pool-tier-cards"></div>
    <div class="chart-box"><div class="chart-title">股票池列表 <span id="pool-name" style="color:var(--text-dim)"></span></div>
      <div style="margin-bottom:10px;display:flex;gap:6px;align-items:center;" id="pool-tier-filter">
        <span style="color:var(--text-dim);font-size:12px;">层级:</span>
        <button class="tier-filter-btn active" data-tier="" style="background:var(--accent);color:#fff;border:1px solid var(--border);padding:3px 12px;border-radius:4px;cursor:pointer;font-size:11px;">全部</button>
        <button class="tier-filter-btn" data-tier="focus" style="background:var(--panel);color:var(--text-dim);border:1px solid var(--border);padding:3px 12px;border-radius:4px;cursor:pointer;font-size:11px;">🎯 精选</button>
        <button class="tier-filter-btn" data-tier="watch" style="background:var(--panel);color:var(--text-dim);border:1px solid var(--border);padding:3px 12px;border-radius:4px;cursor:pointer;font-size:11px;">👀 观察</button>
        <button class="tier-filter-btn" data-tier="broad" style="background:var(--panel);color:var(--text-dim);border:1px solid var(--border);padding:3px 12px;border-radius:4px;cursor:pointer;font-size:11px;">📋 备选</button>
      </div>
      <div class="table-wrap"><table id="table-pool"><thead id="table-pool-head"></thead><tbody></tbody></table></div>
    </div>
  </div>

  <!-- 交割单 -->
  <div class="tab-panel" id="tab-trades">
    <div class="card-row" id="trade-stats"></div>
    <div style="margin-bottom:10px;display:flex;gap:8px;">
      <select id="trade-filter" style="background:var(--panel);color:var(--text);border:1px solid var(--border);padding:4px 10px;border-radius:4px;">
        <option value="">全部策略</option>
        <option value="dragon">龙头战法</option>
        <option value="sparrow">麻雀战法</option>
        <option value="turtle">海龟战法</option>
        <option value="value">价值投资</option>
      </select>
      <input id="trade-search" placeholder="搜索代码/名称..." style="background:var(--panel);color:var(--text);border:1px solid var(--border);padding:4px 10px;border-radius:4px;width:200px;">
    </div>
    <div class="chart-box"><div class="chart-title">交割单记录</div>
      <div class="table-wrap"><table id="table-trades"><thead><tr><th>日期</th><th>策略</th><th>代码</th><th>名称</th><th>方向</th><th>价格</th><th>数量</th><th>金额</th><th>盈亏</th><th>原因</th></tr></thead><tbody></tbody></table></div>
    </div>
  </div>

  <!-- 收盘总结 -->
  <div class="tab-panel" id="tab-summary">
    <div style="margin-bottom:10px;display:flex;gap:8px;align-items:center;">
      <label style="color:var(--text-dim)">选择日期:</label>
      <select id="summary-select" style="background:var(--panel);color:var(--text);border:1px solid var(--border);padding:4px 10px;border-radius:4px;min-width:150px;"></select>
      <button id="summary-load-btn" style="background:var(--accent);color:#fff;border:none;padding:4px 14px;border-radius:4px;cursor:pointer;">加载</button>
    </div>
    <div class="card-row" id="summary-overview-cards"></div>
    <div class="card-row" id="summary-sentiment-card"></div>

    <!-- 热门板块 -->
    <div class="chart-box"><div class="chart-title">🔥 当日热门板块 Top10</div>
      <div class="card-row" id="summary-hot-sectors" style="flex-wrap:wrap;"></div>
      <div class="grid-2" style="margin-top:10px;">
        <div class="chart h350" id="chart-sectors-bar"></div>
        <div class="chart h350" id="chart-sectors-zt-pie"></div>
      </div>
      <div class="table-wrap" style="margin-top:10px;" id="summary-sector-detail-wrap" hidden>
        <table id="table-sector-detail"><thead><tr><th>板块</th><th>涨幅</th><th>涨停家数</th><th>领涨龙头</th><th>操作</th></tr></thead><tbody></tbody></table>
      </div>
      <!-- 板块内Top10成分股展开面板 -->
      <div id="sector-top10-panel" style="margin-top:10px;display:none;background:var(--panel);border:1px solid var(--border);border-radius:6px;overflow:hidden;">
        <div style="display:flex;justify-content:space-between;align-items:center;padding:10px 14px;background:rgba(52,152,219,0.08);">
          <span style="font-weight:600;font-size:14px;" id="sector-top10-title">📈 板块涨幅Top10成分股</span>
          <button onclick="document.getElementById('sector-top10-panel').style.display='none'" style="background:none;border:none;color:var(--text-dim);cursor:pointer;font-size:18px;">&times;</button>
        </div>
        <div class="table-wrap">
          <table id="table-sector-top10"><thead><tr><th>排名</th><th>代码</th><th>名称</th><th>最新价</th><th>日涨幅</th><th>五日涨幅</th><th>成交量(万手)</th><th>成交额(亿)</th><th>换手率</th><th>市盈率</th></tr></thead><tbody></tbody></table>
        </div>
      </div>
    </div>

    <!-- 涨停个股概况 -->
    <div class="chart-box"><div class="chart-title">🚀 涨停个股概况 <span id="zt-total-label" style="color:var(--text-dim);font-size:12px;"></span></div>
      <div class="grid-3" style="margin-bottom:10px;" id="zt-top-leaders"></div>
      <div style="margin-bottom:10px;display:flex;gap:8px;">
        <input id="zt-search" placeholder="搜索涨停股代码/名称..." style="background:var(--panel);color:var(--text);border:1px solid var(--border);padding:4px 10px;border-radius:4px;width:200px;">
        <select id="zt-sector-filter" style="background:var(--panel);color:var(--text);border:1px solid var(--border);padding:4px 10px;border-radius:4px;min-width:130px;">
          <option value="">全部板块</option>
        </select>
        <select id="zt-board-filter" style="background:var(--panel);color:var(--text);border:1px solid var(--border);padding:4px 10px;border-radius:4px;">
          <option value="">全部连板</option><option value="1">首板</option><option value="2">2板+</option><option value="3">3板+</option><option value="5">5板+</option>
        </select>
        <button id="zt-search-btn" style="background:var(--accent);color:#fff;border:none;padding:4px 14px;border-radius:4px;cursor:pointer;">筛选</button>
      </div>
      <div class="table-wrap"><table id="table-zt"><thead><tr><th>代码</th><th>名称</th><th>最新价</th><th>涨幅</th><th>连板</th><th>所属板块</th><th>首封时间</th><th>换手率</th><th>成交额(亿)</th><th>流通市值(亿)</th><th>涨停原因</th></tr></thead><tbody></tbody></table></div>
    </div>

    <div class="grid-2">
      <div class="chart-box"><div class="chart-title">指数涨跌</div><div class="chart h300" id="chart-summary-indices"></div></div>
      <div class="chart-box"><div class="chart-title">涨跌停与情绪</div><div class="chart h300" id="chart-summary-limits"></div></div>
    </div>
    <div class="grid-2">
      <div class="chart-box"><div class="chart-title">策略当日盈亏</div><div class="chart h300" id="chart-summary-pnl"></div></div>
      <div class="chart-box"><div class="chart-title">牛熊研判 & 次日计划</div><div id="summary-advice" style="padding:8px 0;color:var(--text-dim)"></div></div>
    </div>
  </div>
</div>

<script>
// ═══ 工具函数 ═══════════════════════════════════════════════
const $ = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);
const API = (url) => fetch('/api' + url).then(r => r.json()).catch(e => null);
const fmtNum = (n, d=0) => (n||0).toLocaleString('zh-CN', {minimumFractionDigits:d,maximumFractionDigits:d});
const fmtPct = (n) => ((n||0)>=0?'+':'')+(n||0).toFixed(2)+'%';
const fmtVol = (v) => v>=1e12 ? (v/1e12).toFixed(2)+'万亿' : v>=1e8 ? (v/1e8).toFixed(2)+'亿' : v>=1e4 ? (v/1e4).toFixed(2)+'万' : fmtNum(v);
const pnlClass = (n) => (n||0)>0?'pnl-pos':(n||0)<0?'pnl-neg':'';
const dateLabel = (d) => d ? d.slice(0,4)+'-'+d.slice(4,6)+'-'+d.slice(6,8) : '';

// ═══ 时钟 ═══════════════════════════════════════════════════
function updateClock() {
  const now = new Date();
  $('#clock').textContent = now.toLocaleString('zh-CN', {hour12:false});
}
updateClock(); setInterval(updateClock, 1000);

// ═══ 标签切换 ═══════════════════════════════════════════════
$$('.tabs button').forEach(b => b.addEventListener('click', function() {
  $$('.tabs button').forEach(x => x.classList.remove('active'));
  this.classList.add('active');
  const tab = this.dataset.tab;
  $$('.tab-panel').forEach(p => p.classList.remove('active'));
  $('#tab-'+tab).classList.add('active');
  loadTab(tab);
  // 重绘图表
  setTimeout(resizeAllCharts, 200);
}));

function loadTab(name) {
  const loaders = {
    overview: loadOverview, race: loadRace,
    positions: loadPositions, pools: loadPools,
    trades: loadTrades, summary: loadSummary,
  };
  if (loaders[name]) loaders[name]();
}

// ═══ 图表管理 ═══════════════════════════════════════════════
const charts = {};
function initChart(id, opts) {
  const el = $('#'+id);
  if (!el) return null;
  if (charts[id]) charts[id].dispose();
  const c = echarts.init(el);
  if (opts) c.setOption(opts);
  charts[id] = c;
  return c;
}
function resizeAllCharts() {
  Object.values(charts).forEach(c => c.resize());
}
window.addEventListener('resize', resizeAllCharts);

// ═══ 通用主题 ═══════════════════════════════════════════════
const darkTheme = {
  textStyle: { color: '#c8d6e5' },
  legend: { textStyle: { color: '#6d8099' } },
};

// ═══ 1. 系统总览 ═══════════════════════════════════════════
async function loadOverview() {
  const market = await API('/market/overview');
  const strategies = await API('/strategies');

  // 状态卡片
  const ov = market.overview || {};
  const activity = ov.activity || {};
  const sent = market.sentiment || {};
  const trend = market.trend || {};
  const limits = market.limit_stats || {};

  $('#status-cards').innerHTML = `
    <div class="card"><div class="card-label">市场状态</div><div class="card-value">${sent.market_heat||'--'}</div><div class="card-sub">${sent.emotion||'--'}</div></div>
    <div class="card"><div class="card-label">两市成交额</div><div class="card-value">${fmtVol(ov.total_volume||0)}</div><div class="card-sub">上证PE ${(ov.sh_pe||0).toFixed(1)}</div></div>
    <div class="card"><div class="card-label">涨停 / 跌停</div><div class="card-value">${activity['涨停']||0} <span style="font-size:14px;color:var(--text-dim)">/</span> <span style="color:var(--green)">${activity['跌停']||0}</span></div><div class="card-sub">真实涨停 ${activity['真实涨停']||0} | 真实跌停 ${activity['真实跌停']||0}</div></div>
    <div class="card"><div class="card-label">上涨 / 下跌</div><div class="card-value">${activity['上涨']||0} <span style="font-size:14px;color:var(--text-dim)">/</span> <span style="color:var(--green)">${activity['下跌']||0}</span></div><div class="card-sub">平盘 ${activity['平盘']||0} | 停牌 ${activity['停牌']||0}</div></div>
    <div class="card"><div class="card-label">牛熊研判</div><div class="card-value" style="font-size:18px">${trend.phase||'--'}</div><div class="card-sub">置信度 ${trend.confidence||0}%</div></div>
    <div class="card"><div class="card-label">最高连板</div><div class="card-value">${limits.max_consecutive||0} <span style="font-size:14px;color:var(--text-dim)">板</span></div><div class="card-sub">首板 ${limits.first_board_count||0} 家</div></div>
  `;

  // 情绪条
  const score = sent.score || 50;
  const emotionMap = { extreme_greed: '🤑 极度贪婪', greed: '😀 贪婪', neutral: '😐 中性', fear: '😟 恐惧', extreme_fear: '😱 极度恐惧' };
  $('#sentiment-card').innerHTML = `
    <div class="card" style="flex:2">
      <div class="card-label">市场情绪指数</div>
      <div style="display:flex;align-items:center;gap:12px;">
        <span style="font-size:28px;font-weight:700;">${score}</span>
        <span style="color:var(--text-dim)">${emotionMap[sent.emotion_code] || sent.emotion || ''}</span>
      </div>
      <div class="sentiment-bar">
        <div class="sentiment-indicator" style="left:${score}%"></div>
      </div>
      <div style="display:flex;justify-content:space-between;font-size:10px;color:var(--text-dim);margin-top:4px;">
        <span>极度恐惧</span><span>恐惧</span><span>中性</span><span>贪婪</span><span>极度贪婪</span>
      </div>
    </div>
  `;

  // 权益曲线
  const eqSeries = strategies.map(s => ({
    name: s.label, type: 'line', smooth: true,
    symbol: 'none', lineStyle: { width: 2 },
    data: s.equity_curve.map(e => [e.date, e.equity]),
  }));
  initChart('chart-equity-overview', {
    tooltip: { trigger: 'axis' },
    legend: { data: strategies.map(s=>s.label), bottom: 0, ...darkTheme.legend },
    grid: { left: 70, right: 20, top: 10, bottom: 40 },
    xAxis: { type: 'category', axisLabel: { fontSize: 10 }, data: [] },
    yAxis: { type: 'value', axisLabel: { formatter: v => (v/10000).toFixed(0)+'万' } },
    series: eqSeries,
  });

  // 涨跌分布饼图
  initChart('chart-advance-decline', {
    tooltip: { trigger: 'item' },
    legend: { bottom: 0, ...darkTheme.legend },
    series: [{
      type: 'pie', radius: ['45%','72%'], center:['50%','45%'],
      label: { formatter: '{b}\\n{d}%' },
      data: [
        { value: activity['上涨']||0, name: '上涨', itemStyle: {color:'#e74c3c'} },
        { value: activity['下跌']||0, name: '下跌', itemStyle: {color:'#27ae60'} },
        { value: activity['平盘']||0, name: '平盘', itemStyle: {color:'#7f8c8d'} },
      ]
    }]
  });

  // 收益率柱状图
  initChart('chart-return-bar', {
    tooltip: { trigger: 'axis' },
    grid: { left: 70, right: 20, top: 10, bottom: 30 },
    xAxis: { type:'category', data: strategies.map(s=>s.label) },
    yAxis: { type:'value', axisLabel:{ formatter:'{value}%' } },
    series: [
      { name:'总收益率', type:'bar', data: strategies.map(s=>s.total_return_pct),
        itemStyle:{ color: p => p.value>=0?'#e74c3c':'#27ae60' } },
      { name:'日收益率', type:'bar', data: strategies.map(s=>s.daily_return_pct),
        itemStyle:{ color: p => p.value>=0?'#f39c12':'#2980b9' } },
    ],
  });

  // 风险雷达
  initChart('chart-risk-radar', {
    tooltip: {},
    legend: { data: strategies.map(s=>s.label), bottom: 0, ...darkTheme.legend },
    radar: {
      indicator: [
        { name:'收益率', max: Math.max(10, ...strategies.map(s=>Math.abs(s.total_return_pct))) },
        { name:'稳定性', max: 100 }, { name:'持仓', max: Math.max(1, ...strategies.map(s=>s.position_count)) },
        { name:'回撤', max: Math.max(5, ...strategies.map(s=>s.max_drawdown)) },
        { name:'活跃度', max: Math.max(1, ...strategies.map(s=>s.trade_count)) },
      ],
      center: ['50%','50%'], radius: '65%',
    },
    series: [{ type:'radar', data: strategies.map(s => ({
      name: s.label, value: [s.total_return_pct, 90-s.max_drawdown, s.position_count, s.max_drawdown, s.trade_count],
    })) }],
  });

  // 日志
  const summary = await API('/summary/'+market.date);
  if (summary && summary.phase_logs) {
    const logs = summary.phase_logs.slice(-50).reverse();
    $('#table-logs tbody').innerHTML = logs.map(l => `
      <tr><td>${l.time?.slice(11,19)||''}</td><td>${l.phase||''}</td><td>${l.event||''}</td><td style="max-width:300px;overflow:hidden;text-overflow:ellipsis;">${l.detail||''}</td></tr>
    `).join('');
  }
}

// ═══ 2. 策略赛马 ═══════════════════════════════════════════
async function loadRace() {
  const strategies = await API('/strategies');

  // 排名卡片
  const colorMap = ['#f1c40f','#95a5a6','#cd6133','#3498db'];
  $('#race-cards').innerHTML = strategies.map((s,i) => `
    <div class="card" style="border-left:3px solid ${colorMap[i]}">
      <div class="card-label">#${s.rank} ${s.label}</div>
      <div class="card-value">${fmtNum(s.equity)}</div>
      <div class="card-sub"><span class="${pnlClass(s.total_return_pct)}">${fmtPct(s.total_return_pct)}</span> | 持仓${s.position_count} | 回撤${s.max_drawdown.toFixed(1)}%</div>
    </div>
  `).join('');

  // 权益曲线对比
  initChart('chart-race-equity', {
    tooltip: { trigger: 'axis' },
    legend: { data: strategies.map(s=>s.label), bottom: 0, ...darkTheme.legend },
    grid: { left: 70, right: 20, top: 10, bottom: 40 },
    xAxis: { type: 'category' },
    yAxis: { type: 'value', axisLabel:{ formatter: v=>v>=1e4?(v/1e4).toFixed(0)+'万':v } },
    series: strategies.map((s,i) => ({
      name: s.label, type: 'line', smooth: true,
      symbol: 'none', lineStyle: { width: 2 },
      data: s.equity_curve.map(e => [e.date, e.equity]),
    })),
  });

  // 收益率走势
  initChart('chart-race-returns', {
    tooltip: { trigger: 'axis' },
    legend: { data: strategies.map(s=>s.label), bottom: 0, ...darkTheme.legend },
    grid: { left: 60, right: 15, top: 10, bottom: 40 },
    xAxis: { type: 'category' },
    yAxis: { type: 'value', axisLabel:{ formatter:'{value}%' } },
    series: strategies.map(s => {
      const cap = s.initial_capital || DEFAULT_CAPITAL;
      const data = s.equity_curve.map(e => [e.date, +((e.equity/cap-1)*100).toFixed(2)]);
      return { name: s.label, type: 'line', smooth: true, symbol:'none', lineStyle:{width:2}, data };
    }),
  });

  // 回撤柱状图
  initChart('chart-race-drawdown', {
    tooltip: { trigger: 'axis' },
    grid: { left: 60, right: 20, top: 10, bottom: 30 },
    xAxis: { type:'category', data: strategies.map(s=>s.label) },
    yAxis: { type: 'value', axisLabel:{ formatter:'{value}%' }, inverse: true },
    series: [{
      type: 'bar', name:'最大回撤', data: strategies.map(s=>s.max_drawdown),
      itemStyle: { color: '#e74c3c' }, label:{ show:true, position:'bottom', formatter:'{c}%' },
    }],
  });
}

// ═══ 3. 持仓明细 ═══════════════════════════════════════════
async function loadPositions() {
  const positions = await API('/positions');
  const strategies = await API('/strategies');

  $('#pos-summary').innerHTML = strategies.map(s => `
    <div class="card"><div class="card-label">${s.label}</div><div class="card-value">${s.position_count}</div><div class="card-sub">只持仓</div></div>
  `).join('');

  renderPositionsTable(positions);
  $('#pos-filter').onchange = () => renderPositionsTable(positions);
  $('#pos-search').oninput = () => renderPositionsTable(positions);
}

function renderPositionsTable(positions) {
  const filter = $('#pos-filter').value;
  const search = $('#pos-search').value.toLowerCase();
  let data = positions;
  if (filter) data = data.filter(p => p.strategy === filter);
  if (search) data = data.filter(p => p.code.includes(search) || p.name.toLowerCase().includes(search));

  if (!data.length) {
    $('#table-positions tbody').innerHTML = '<tr><td colspan="9" class="empty">暂无持仓数据</td></tr>';
    return;
  }
  const totalMv = data.reduce((s,p) => s+(p.market_value||0), 0);
  $('#table-positions tbody').innerHTML = data.map(p => `
    <tr>
      <td><span class="tag tag-active">${p.strategy_label}</span></td>
      <td>${p.code}</td><td>${p.name}</td>
      <td>${(p.cost||0).toFixed(2)}</td><td>${fmtNum(p.volume||0)}</td>
      <td>${(p.current_price||0).toFixed(2)}</td>
      <td>${fmtNum(p.market_value||0)}</td>
      <td class="${pnlClass(p.profit_pct)}">${fmtPct(p.profit_pct)}</td>
      <td>${p.hold_days||0}天</td>
    </tr>
  `).join('') + `<tr style="font-weight:600;background:rgba(52,152,219,0.08)"><td colspan="6" style="text-align:right">持仓总市值</td><td>${fmtNum(totalMv)}</td><td colspan="2"></td></tr>`;
}

// ═══ 4. 股票池 ═══════════════════════════════════════════
let allPools = {};
let currentPool = 'dragon';
let currentTier = '';  // '' means all
async function loadPools() {
  allPools = await API('/pools');
  // 池切换按钮
  $('#pool-tab-buttons').innerHTML = Object.entries(allPools).map(([k,v]) =>
    `<button class="pool-tab" data-key="${k}" style="background:${k===currentPool?'var(--accent)':'var(--panel)'};color:${k===currentPool?'#fff':'var(--text-dim)'};border:1px solid var(--border);padding:6px 16px;border-radius:4px;cursor:pointer;">${v.label} (${v.total_items})</button>`
  ).join('');
  $$('.pool-tab').forEach(b => b.addEventListener('click', function() {
    currentPool = this.dataset.key; currentTier = '';
    $$('.pool-tab').forEach(x => { x.style.background='var(--panel)'; x.style.color='var(--text-dim)'; });
    this.style.background='var(--accent)'; this.style.color='#fff';
    document.querySelectorAll('.tier-filter-btn').forEach(b => { b.classList.remove('active'); b.style.background='var(--panel)'; b.style.color='var(--text-dim)'; });
    document.querySelector('.tier-filter-btn[data-tier=""]').classList.add('active');
    document.querySelector('.tier-filter-btn[data-tier=""]').style.background='var(--accent)';
    document.querySelector('.tier-filter-btn[data-tier=""]').style.color='#fff';
    renderPool();
  }));
  // 层级筛选按钮绑定
  document.querySelectorAll('.tier-filter-btn').forEach(b => b.addEventListener('click', function() {
    currentTier = this.dataset.tier;
    document.querySelectorAll('.tier-filter-btn').forEach(b => { b.classList.remove('active'); b.style.background='var(--panel)'; b.style.color='var(--text-dim)'; });
    this.classList.add('active'); this.style.background='var(--accent)'; this.style.color='#fff';
    renderPool();
  }));
  renderPool();
}

function renderPool() {
  const pool = allPools[currentPool];
  if (!pool) return;
  const tc = pool.tier_counts || {};
  $('#pool-name').textContent = `${pool.label} | 筛选日期: ${dateLabel(pool.last_screened)}`;
  $('#pool-stats').innerHTML = `
    <div class="card"><div class="card-label">池内标的</div><div class="card-value">${pool.total_items}</div></div>
    <div class="card"><div class="card-label">上次筛选</div><div class="card-value" style="font-size:16px">${dateLabel(pool.last_screened)||'--'}</div></div>
  `;
  // 层级统计卡片
  const tierColors = { focus:'#e74c3c', watch:'#f1c40f', broad:'#3498db' };
  const tierIcons = { focus:'🎯', watch:'👀', broad:'📋' };
  $('#pool-tier-cards').innerHTML = ['focus','watch','broad'].map(t => `
    <div class="card" style="border-left:3px solid ${tierColors[t]};flex:1;min-width:100px;">
      <div class="card-label">${tierIcons[t]} ${t==='focus'?'精选层':t==='watch'?'观察层':'备选层'}</div>
      <div class="card-value" style="color:${tierColors[t]};font-size:20px;">${tc[t]||0}</div>
    </div>
  `).join('');

  let items = pool.items || [];
  // 只显示 active 且未被移除的
  items = items.filter(it => it.status === 'active');
  // 层级筛选
  if (currentTier) {
    items = items.filter(it => it.tier === currentTier);
  }
  // 排序：先按层级（focus→watch→broad），再按评分降序
  const tierOrder = { focus:0, watch:1, broad:2 };
  items.sort((a,b) => {
    const ta = tierOrder[a.tier] ?? 9;
    const tb = tierOrder[b.tier] ?? 9;
    if (ta !== tb) return ta - tb;
    return (b.score||0) - (a.score||0);
  });

  if (!items.length) {
    $('#table-pool-head').innerHTML = ''; $('#table-pool tbody').innerHTML = '<tr><td class="empty">该层级暂无标的</td></tr>';
    return;
  }
  // 动态表头：层级 + 基本列 + 涨幅列 + 评分维度
  const sample = items[0];
  const cols = ['tier','code','name','score','screened_at'];
  const gainCols = ['cumulative_gain_pct','daily_gain_pct','max_gain_pct'];
  const extraCols = sample.components ? Object.keys(sample.components) : [];
  const allCols = [...cols, ...gainCols, ...extraCols];
  const labels = { tier:'层级', code:'代码', name:'名称', score:'评分', screened_at:'入池日期',
    cumulative_gain_pct:'累计涨幅', daily_gain_pct:'当日涨幅', max_gain_pct:'最大涨幅' };
  $('#table-pool-head').innerHTML = '<tr>'+allCols.map(c => `<th>${labels[c]||c}</th>`).join('')+'</tr>';

  const fmtGain = (val) => val == null ? '<span style="color:var(--text-dim)">--</span>' : `<span class="${val>=0?'pnl-pos':'pnl-neg'}" style="font-weight:600">${fmtPct(val)}</span>`;
  const tierBadge = (t) => {
    if (t === 'focus') return '<span class="tag tag-focus">🎯 精选</span>';
    if (t === 'watch') return '<span class="tag tag-watch">👀 观察</span>';
    if (t === 'broad') return '<span class="tag tag-broad">📋 备选</span>';
    return '<span class="tag tag-removed">--</span>';
  };

  $('#table-pool tbody').innerHTML = items.map(item => `
    <tr>
      <td>${tierBadge(item.tier)}</td>
      <td>${item.code||''}</td><td>${item.name||''}</td>
      <td style="font-weight:600;color:${(item.score||0)>=80?'var(--red)':(item.score||0)>=60?'var(--orange)':'var(--text-dim)'}">${(item.score||0).toFixed(1)}</td>
      <td>${dateLabel(item.screened_at)}</td>
      <td>${fmtGain(item.cumulative_gain_pct)}</td>
      <td>${fmtGain(item.daily_gain_pct)}</td>
      <td>${fmtGain(item.max_gain_pct)}</td>
      ${extraCols.map(c => `<td>${item.components?.[c]?.toFixed?.(1) ?? item.components?.[c] ?? ''}</td>`).join('')}
    </tr>
  `).join('');
}

// ═══ 5. 交割单 ═══════════════════════════════════════════
let allTrades = [];
async function loadTrades() {
  allTrades = await API('/trades?limit=500');
  // 统计
  const buys = allTrades.filter(t => (t.action||'').includes('buy')||(t.action||'').includes('买'));
  const sells = allTrades.filter(t => (t.action||'').includes('sell')||(t.action||'').includes('卖'));
  const totalPnl = allTrades.reduce((s,t) => s+(t.pnl||0), 0);
  $('#trade-stats').innerHTML = `
    <div class="card"><div class="card-label">总交易</div><div class="card-value">${allTrades.length}</div></div>
    <div class="card"><div class="card-label">买入</div><div class="card-value">${buys.length}</div></div>
    <div class="card"><div class="card-label">卖出</div><div class="card-value">${sells.length}</div></div>
    <div class="card ${totalPnl>=0?'red':'green'}"><div class="card-label">盈亏合计</div><div class="card-value">${fmtPct(totalPnl)}</div></div>
  `;
  renderTradesTable();
  $('#trade-filter').onchange = () => renderTradesTable();
  $('#trade-search').oninput = () => renderTradesTable();
}

function renderTradesTable() {
  const filter = $('#trade-filter').value;
  const search = $('#trade-search').value.toLowerCase();
  let data = allTrades;
  if (filter) data = data.filter(t => t._strategy === filter);
  if (search) data = data.filter(t => (t.code||'').includes(search) || (t.name||'').toLowerCase().includes(search));

  if (!data.length) {
    $('#table-trades tbody').innerHTML = '<tr><td colspan="10" class="empty">暂无交割单记录</td></tr>';
    return;
  }
  $('#table-trades tbody').innerHTML = data.map(t => {
    const act = (t.action||'').includes('buy')?'buy':(t.action||'').includes('sell')?'sell':'other';
    return `
    <tr>
      <td>${dateLabel(t._date||t.date)}</td>
      <td>${t._strategy_label||''}</td><td>${t.code||''}</td><td>${t.name||''}</td>
      <td><span class="tag ${act==='buy'?'tag-buy':'tag-sell'}">${t.action||''}</span></td>
      <td>${(t.price||0).toFixed(2)}</td><td>${fmtNum(t.volume||0)}</td>
      <td>${fmtNum(t.amount||0)}</td>
      <td class="${pnlClass(t.pnl)}">${t.pnl?fmtPct(t.pnl):'-'}</td>
      <td style="max-width:180px;overflow:hidden;text-overflow:ellipsis;" title="${t.reason||''}">${t.reason||''}</td>
    </tr>`;
  }).join('');
}

// ═══ 6. 收盘总结 ═══════════════════════════════════════════
let summaryZtData = [];
async function loadSummary() {
  const res = await API('/summaries');
  const dates = res.dates || [];
  const sel = $('#summary-select');
  sel.innerHTML = dates.map(d => `<option value="${d}" ${d===res.latest?'selected':''}>${dateLabel(d)}</option>`).join('');
  if (res.latest) loadSummaryDetail(res.latest);

  sel.onchange = () => { if (sel.value) loadSummaryDetail(sel.value); };
  $('#summary-load-btn').onclick = () => { if (sel.value) loadSummaryDetail(sel.value); };
}

async function loadSummaryDetail(date) {
  const s = await API('/summary/'+date);
  if (!s) return;

  const ov = s.market_overview || {};
  const sent = s.sentiment || {};
  const trend = s.market_trend || {};
  const limits = s.limit_stats || {};
  const sp = s.strategy_pnl || {};
  const plan = s.next_day_plan || {};
  const indices = ov.indices || {};
  const tech = s.technical_indicators || {};
  const activity = ov.activity || {};
  const hotSectors = s.hot_sectors || {};
  const ztDetails = s.limit_up_details || {};

  $('#summary-overview-cards').innerHTML = `
    <div class="card"><div class="card-label">日期</div><div class="card-value" style="font-size:16px">${dateLabel(s.date)}</div></div>
    <div class="card"><div class="card-label">成交额</div><div class="card-value">${fmtVol(ov.total_volume||0)}</div><div class="card-sub">PE ${(ov.sh_pe||0).toFixed(1)}</div></div>
    <div class="card"><div class="card-label">涨/跌停</div><div class="card-value">${limits.limit_up||0} / ${limits.limit_down||0}</div><div class="card-sub">首板${ztDetails.first_board_count||limits.first_board_count||0} | 最高${limits.max_consecutive||0}板</div></div>
    <div class="card"><div class="card-label">牛熊研判</div><div class="card-value" style="font-size:16px">${trend.phase||'--'}</div><div class="card-sub">置信度 ${trend.confidence||0}%</div></div>
  `;

  $('#summary-sentiment-card').innerHTML = `
    <div class="card" style="flex:2">
      <div class="card-label">市场情绪: ${sent.emotion||'--'} | 冷热度: ${sent.market_heat||'--'}</div>
      <div style="margin-top:6px;color:var(--text-dim);font-size:12px;">${trend.advice||''}</div>
    </div>
  `;

  // ── 热门板块渲染 ──
  renderHotSectors(hotSectors);

  // ── 涨停个股渲染 ──
  renderLimitUpDetails(ztDetails);

  // 指数涨跌柱状图
  const idxNames = Object.keys(indices);
  initChart('chart-summary-indices', {
    tooltip: { trigger: 'axis' },
    grid: { left: 70, right: 25, top: 10, bottom: 30 },
    xAxis: { type:'category', data: idxNames, axisLabel:{ rotate: 15, fontSize: 10 } },
    yAxis: { type:'value', axisLabel:{ formatter:'{value}%' } },
    series: [{
      type: 'bar', name: '涨跌幅',
      data: idxNames.map(n => (indices[n].pct_change||0)),
      itemStyle: { color: p => p.value>=0?'#e74c3c':'#27ae60' },
      label: { show:true, position: 'top', formatter:'{c}%', fontSize:10 },
    }],
  });

  // 涨跌停与情绪
  initChart('chart-summary-limits', {
    tooltip: { trigger: 'axis' },
    grid: { left: 50, right: 20, top: 10, bottom: 30 },
    legend: { data: ['涨停','跌停','上涨','下跌'], bottom: 0, ...darkTheme.legend },
    xAxis: { type:'category', data: ['统计'] },
    yAxis: { type: 'value' },
    series: [
      { name:'涨停', type:'bar', data:[limits.limit_up||0], itemStyle:{color:'#e74c3c'} },
      { name:'跌停', type:'bar', data:[limits.limit_down||0], itemStyle:{color:'#27ae60'} },
      { name:'上涨', type:'bar', data:[activity['上涨']||0], itemStyle:{color:'#f39c12'} },
      { name:'下跌', type:'bar', data:[activity['下跌']||0], itemStyle:{color:'#3498db'} },
    ],
  });

  // 策略盈亏
  const spKeys = Object.keys(sp);
  initChart('chart-summary-pnl', {
    tooltip: { trigger: 'axis' },
    grid: { left: 70, right: 20, top: 10, bottom: 30 },
    xAxis: { type:'category', data: spKeys.map(k => sp[k].label||k) },
    yAxis: { type:'value', axisLabel:{ formatter:'{value}%' } },
    series: [
      { name:'总收益率', type:'bar', data: spKeys.map(k => sp[k].total_return_pct||0),
        itemStyle:{ color:p=>p.value>=0?'#e74c3c':'#27ae60' }, label:{show:true,position:'top',formatter:'{c}%',fontSize:10} },
      { name:'日收益率', type:'bar', data: spKeys.map(k => sp[k].daily_return_pct||0),
        itemStyle:{ color:p=>p.value>=0?'#f39c12':'#2980b9' } },
    ],
  });

  // 研判建议与次日计划
  let adviceHtml = `<div style="margin-bottom:12px;padding:10px;background:rgba(52,152,219,0.08);border-radius:4px;">
    <strong>${trend.phase||'--'}</strong> (置信度 ${trend.confidence||0}%)
    <div style="color:var(--text-dim);margin-top:4px;">${trend.advice||''}</div>
  </div>`;
  adviceHtml += '<div style="font-weight:600;margin-bottom:6px;">次日交易计划</div>';
  for (const [k,p] of Object.entries(plan)) {
    const top5 = (p.top5||[]).slice(0,3).map(x => x.name||x.code).join('、') || '无';
    adviceHtml += `<div style="margin-bottom:4px;font-size:12px;"><span class="tag tag-active">${p.label||k}</span> 池${p.pool_size||0}只 | 关注: ${top5}</div>`;
  }
  $('#summary-advice').innerHTML = adviceHtml;
}

// ── 热门板块渲染 ──
function renderHotSectors(hotSectors) {
  const sectors = hotSectors.top_sectors || [];
  if (!sectors.length) {
    $('#summary-hot-sectors').innerHTML = '<div class="empty">暂无板块数据（可能原因：非交易日、akshare API 超时或网络异常）</div>';
    // 清空关联图表和表格
    initChart('chart-sectors-bar', null);
    initChart('chart-sectors-zt-pie', null);
    $('#table-sector-detail tbody').innerHTML = '';
    $('#sector-top10-panel').style.display = 'none';
    return;
  }

  // 板块卡片（Top 6）
  const top6 = sectors.slice(0, 6);
  $('#summary-hot-sectors').innerHTML = top6.map(s => `
    <div class="card" style="border-left:3px solid ${s.pct_change>=0?'#e74c3c':'#27ae60'};min-width:170px;">
      <div class="card-label">#${s.strength_rank} ${s.name}</div>
      <div class="card-value ${s.pct_change>=0?'red':'green'}" style="font-size:18px;">${fmtPct(s.pct_change)}</div>
      <div class="card-sub">涨停${s.limit_up_count||0}只 | ${s.trend==='up'?'↑':s.trend==='down'?'↓':'→'}</div>
    </div>
  `).join('');

  // 板块涨幅柱状图
  const allSectors = sectors.slice(0, 10);
  initChart('chart-sectors-bar', {
    tooltip: { trigger:'axis', axisPointer:{type:'shadow'} },
    grid: { left: 80, right: 30, top: 10, bottom: 30 },
    xAxis: { type:'value', axisLabel:{ formatter:'{value}%' } },
    yAxis: { type:'category', data: allSectors.map(s=>s.name).reverse(), axisLabel:{ fontSize:10 } },
    series: [{
      type: 'bar', name:'涨跌幅',
      data: allSectors.map(s=>s.pct_change).reverse(),
      itemStyle:{ color: p=>p.value>=0?'#e74c3c':'#27ae60' },
      label:{ show:true, position:'right', formatter:'{c}%', fontSize:10 },
    }],
  });

  // 涨停板块分布饼图
  const pieData = sectors.filter(s => (s.limit_up_count||0) > 0).map(s => ({
    name: s.name, value: s.limit_up_count||0,
  }));
  initChart('chart-sectors-zt-pie', {
    tooltip: { trigger:'item', formatter:'{b}: {c}只 ({d}%)' },
    legend: { orient:'vertical', right:5, top:'middle', textStyle:{color:'#6d8099',fontSize:10} },
    series: [{
      type:'pie', radius:['35%','65%'], center:['38%','50%'],
      label:{ formatter:'{b}\\n{c}只', fontSize:10 },
      emphasis:{ label:{fontSize:14,fontWeight:'bold'} },
      data: pieData,
    }],
  });

  // 板块详情表格（可点击展开板块内 Top10 成分股）
  $('#table-sector-detail tbody').innerHTML = sectors.map((s, idx) => {
    const leaders = (s.limit_up_leaders || []).slice(0, 3).map(l =>
      `<span style="color:var(--accent)">${l.name||l.code}</span><span style="color:var(--text-dim);font-size:10px;">(${fmtPct(l.pct_change)})</span>`
    ).join(' ');
    const hasTopStocks = (s.top_stocks && s.top_stocks.length > 0);
    return `<tr class="sector-row" data-sector-idx="${idx}" style="cursor:pointer;" onclick="showSectorTop10(${idx})">
      <td>${s.name} <span style="font-size:10px;color:var(--text-dim);">${s.total_stocks||''}只</span></td>
      <td class="${pnlClass(s.pct_change)}">${fmtPct(s.pct_change)}</td>
      <td><strong>${s.limit_up_count||0}</strong></td>
      <td>${leaders||'-'}</td>
      <td>${hasTopStocks ? '<span class="tag tag-active" style="cursor:pointer;font-size:10px;">展开▼</span>' : '-'}</td>
    </tr>`;
  }).join('');
  $('#summary-sector-detail-wrap').hidden = false;

  // 存储板块数据供 drill-down 使用
  window._hotSectorsData = sectors;
}

// ── 板块内 Top10 成分股钻取 ──
function showSectorTop10(sectorIdx) {
  const sectors = window._hotSectorsData || [];
  const s = sectors[sectorIdx];
  if (!s || !s.top_stocks || !s.top_stocks.length) return;

  $('#sector-top10-title').textContent = `📈 ${s.name} — 五日涨幅Top10成分股 (共${s.total_stocks||0}只，按近5日涨幅排序)`;
  const all5dZero = s.top_stocks.every(st => !(st.pct_5d));
  const render5d = (v) => all5dZero ? '<span style="color:var(--text-dim);font-size:11px;">--</span>' : ((v||0)>=0?'red':'green');
  $('#table-sector-top10 tbody').innerHTML = s.top_stocks.map((st, i) => `
    <tr>
      <td><strong>#${i+1}</strong></td>
      <td><span style="color:var(--accent);">${st.code}</span></td>
      <td>${st.name}</td>
      <td>${(st.price||0).toFixed(2)}</td>
      <td class="${(st.pct_change||0)>=0?'red':'green'}">${fmtPct(st.pct_change)}</td>
      <td class="${render5d(st.pct_5d)}"><strong>${all5dZero ? '--' : fmtPct(st.pct_5d)}</strong></td>
      <td>${(st.volume||0).toFixed(0)}</td>
      <td>${(st.amount||0).toFixed(2)}</td>
      <td>${(st.turnover||0).toFixed(2)}%</td>
      <td>${(st.pe||0).toFixed(1)}</td>
    </tr>
  `).join('');
  $('#sector-top10-panel').style.display = 'block';

  // 高亮当前选中行
  document.querySelectorAll('.sector-row').forEach(r => r.style.background = '');
  const row = document.querySelector(`.sector-row[data-sector-idx="${sectorIdx}"]`);
  if (row) row.style.background = 'rgba(52,152,219,0.1)';
}

// ── 涨停个股渲染 ──
function renderLimitUpDetails(ztDetails) {
  const stocks = ztDetails.stocks || [];
  summaryZtData = stocks;

  $('#zt-total-label').textContent = `共 ${ztDetails.total_count||stocks.length} 只涨停，首板 ${ztDetails.first_board_count||0} 只`;

  // 高位连板龙头卡片
  const leaders = ztDetails.consecutive_leaders || [];
  const topLeaders = leaders.slice(0, 6);
  $('#zt-top-leaders').innerHTML = topLeaders.length
    ? topLeaders.map(l => `
      <div class="card" style="border-left:3px solid #f39c12;">
        <div class="card-label">${l.consecutive}连板</div>
        <div class="card-value" style="font-size:16px;">${l.name||l.code} <span style="color:${(l.pct_change||0)>=0?'var(--red)':'var(--green)'};font-size:13px;">${fmtPct(l.pct_change)}</span></div>
        <div class="card-sub">${l.sector||''} | 换手${(l.turnover||0).toFixed(1)}% | 封单${(l.seal_amount||0).toFixed(1)}亿</div>
      </div>`).join('')
    : '<div class="empty" style="flex:1">暂无连板龙头</div>';

  // 板块筛选下拉
  const sectors = [...new Set(stocks.map(s => s.sector).filter(Boolean))].sort();
  const sel = $('#zt-sector-filter');
  sel.innerHTML = '<option value="">全部板块</option>' + sectors.map(s => `<option value="${s}">${s}</option>`).join('');

  // 渲染表格
  renderZtTable(stocks);

  // 筛选绑定
  $('#zt-search-btn').onclick = () => renderZtTable(summaryZtData);
  $('#zt-search').oninput = () => renderZtTable(summaryZtData);
  $('#zt-sector-filter').onchange = () => renderZtTable(summaryZtData);
  $('#zt-board-filter').onchange = () => renderZtTable(summaryZtData);
}

function renderZtTable(allStocks) {
  let data = allStocks || [];
  const search = ($('#zt-search')?.value || '').toLowerCase();
  const sectorFilter = $('#zt-sector-filter')?.value || '';
  const boardMin = parseInt($('#zt-board-filter')?.value || '0') || 0;

  if (sectorFilter) data = data.filter(s => s.sector === sectorFilter);
  if (boardMin) data = data.filter(s => (s.consecutive||0) >= boardMin);
  if (search) data = data.filter(s => (s.code||'').includes(search) || (s.name||'').toLowerCase().includes(search));

  if (!data.length) {
    $('#table-zt tbody').innerHTML = '<tr><td colspan="11" class="empty">无匹配涨停个股</td></tr>';
    return;
  }

  $('#table-zt tbody').innerHTML = data.map(s => {
    const conBadge = (s.consecutive||0) >= 5
      ? `<span class="tag" style="background:rgba(241,196,15,0.2);color:#f1c40f;font-weight:700;">${s.consecutive}板</span>`
      : (s.consecutive||0) >= 2
        ? `<span class="tag" style="background:rgba(243,156,18,0.15);color:#f39c12;">${s.consecutive}板</span>`
        : `<span style="color:var(--text-dim);font-size:11px;">首板</span>`;
    return `
    <tr>
      <td>${s.code}</td><td style="font-weight:600;">${s.name}</td>
      <td>${(s.price||0).toFixed(2)}</td>
      <td class="${pnlClass(s.pct_change)}">${fmtPct(s.pct_change)}</td>
      <td>${conBadge}</td>
      <td style="color:var(--text-dim);">${s.sector||'-'}</td>
      <td>${s.first_time||'-'}</td>
      <td>${(s.turnover||0).toFixed(1)}%</td>
      <td>${(s.volume_amount||0).toFixed(2)}</td>
      <td>${(s.float_mv||0).toFixed(1)}</td>
      <td style="max-width:160px;overflow:hidden;text-overflow:ellipsis;" title="${s.reason||''}">${s.reason||'-'}</td>
    </tr>`;
  }).join('');
}

// ═══ 启动 ═══════════════════════════════════════════════════
loadOverview();
</script>
</body>
</html>
'''


# ─── CLI 入口 ────────────────────────────────────────────────

if __name__ == "__main__":
    start_dashboard()
