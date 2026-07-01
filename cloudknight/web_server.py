"""
CloudKnight 数据驱动仪表盘 - Web 服务

基于 FastAPI 提供 REST API，为前端仪表盘供应：
  系统状态、策略赛马、持仓明细、股票池、交割单、收盘总结
"""

import json
import os
import glob
import logging
import time as _time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import datetime
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

# Web API 超时配置
_WEB_KLINE_FETCH_TIMEOUT = 8  # 单只股票K线数据获取超时（秒）
_WEB_POOL_INJECT_MAX_TIME = 30  # 整个池涨幅注入的最大总时间（秒）

try:
    from fastapi import FastAPI, Query, HTTPException
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


from .config import (
    DATA_DIR, LIVE_LOG_DIR, LIVE_TRADE_DIR,
    DEFAULT_CAPITAL, STRATEGIES,
)
from .stock_pool import POOL_DIR, TIER_FOCUS, TIER_WATCH, TIER_BROAD, TIER_LABELS, TIER_THRESHOLDS
from .ops_monitor import ops_collector, SystemMonitor, EngineStateReader, HAS_PSUTIL
from .stock_diagnose import get_diagnoser, StockDiagnosis

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


def _diagnosis_to_dict(d: StockDiagnosis) -> dict:
    """将 StockDiagnosis 序列化为 JSON 兼容字典"""
    def _json_safe(val):
        """转换 numpy 类型为 Python 原生类型"""
        try:
            import numpy as np
            if isinstance(val, (np.integer,)):
                return int(val)
            if isinstance(val, (np.floating,)):
                return float(val)
            if isinstance(val, (np.bool_,)):
                return bool(val)
            if isinstance(val, np.ndarray):
                return val.tolist()
        except ImportError:
            pass
        return val

    def _dict_safe(d):
        """递归转换 dict 中的 numpy 值"""
        if not isinstance(d, dict):
            return d
        return {str(k): _dict_safe(v) if isinstance(v, dict) else _json_safe(v) for k, v in d.items()}

    def ir_to_dict(ir):
        if ir is None:
            return None
        return {"trend": ir.trend, "signal": ir.signal,
                "strength": _json_safe(ir.strength), "details": _dict_safe(ir.details)}

    return {
        "code": d.code,
        "name": d.name,
        "timestamp": d.timestamp,
        "technical": {
            "score": d.technical.score,
            "rating": d.technical.rating,
            "current_price": d.technical.current_price,
            "pct_change_5d": d.technical.pct_change_5d,
            "pct_change_20d": d.technical.pct_change_20d,
            "amplitude_5d": d.technical.amplitude_5d,
            "ma_alignment": d.technical.ma_alignment,
            "ma_positions": d.technical.ma_positions,
            "macd": ir_to_dict(d.technical.macd),
            "kdj": ir_to_dict(d.technical.kdj),
            "rsi": ir_to_dict(d.technical.rsi),
            "bollinger": ir_to_dict(d.technical.bollinger),
            "volume_price": ir_to_dict(d.technical.volume_price),
            "atr": d.technical.atr,
            "support": d.technical.support,
            "resistance": d.technical.resistance,
        },
        "fundamental": {
            "score": d.fundamental.score,
            "rating": d.fundamental.rating,
            "pe": d.fundamental.pe,
            "pb": d.fundamental.pb,
            "roe": d.fundamental.roe,
            "revenue_growth": d.fundamental.revenue_growth,
            "profit_growth": d.fundamental.profit_growth,
            "market_cap": d.fundamental.market_cap,
            "dividend_yield": d.fundamental.dividend_yield,
            "debt_ratio": d.fundamental.debt_ratio,
            "gross_margin": d.fundamental.gross_margin,
            "net_margin": d.fundamental.net_margin,
            "data_available": d.fundamental.data_available,
        },
        "capital": {
            "score": d.capital.score,
            "rating": d.capital.rating,
            "main_force_direction": d.capital.main_force_direction,
            "main_force_5d_net": d.capital.main_force_5d_net,
            "turnover_rate": d.capital.turnover_rate,
            "volume_ratio": d.capital.volume_ratio,
            "data_available": d.capital.data_available,
        },
        "market": {
            "score": d.market.score,
            "rating": d.market.rating,
            "sector_name": d.market.sector_name,
            "sector_rank": d.market.sector_rank,
            "sector_pct": d.market.sector_pct,
            "market_trend": d.market.market_trend,
            "market_sentiment": d.market.market_sentiment,
            "limit_status": d.market.limit_status,
        },
        "composite_score": d.composite_score,
        "composite_rating": d.composite_rating,
        "recommendation": d.recommendation,
        "risk_level": d.risk_level,
        "summary": d.summary,
        "detail_scores": d.detail_scores,
        "data_quality": d.data_quality,
        "strategy_diagnoses": [
            {
                "name": s.name,
                "key": s.key,
                "score": s.score,
                "signal": s.signal,
                "rating": s.rating,
                "match_count": s.match_count,
                "total_conditions": s.total_conditions,
                "reasons": s.reasons,
                "warnings": s.warnings,
            }
            for s in (d.strategy_diagnoses or [])
        ],
    }


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
    """为股票池数据注入累计涨幅、当日涨幅、最大涨幅、层级
    
    VPN 兼容：每只股票的 K 线获取有超时保护（{_WEB_KLINE_FETCH_TIMEOUT}s），
    单只超时则跳过继续处理下一只，确保整体 API 在 {_WEB_POOL_INJECT_MAX_TIME}s 内返回。
    """.format(
        _WEB_KLINE_FETCH_TIMEOUT=_WEB_KLINE_FETCH_TIMEOUT,
        _WEB_POOL_INJECT_MAX_TIME=_WEB_POOL_INJECT_MAX_TIME,
    )
    from .data_manager import DataFetcher
    if not items:
        return items

    fetcher = DataFetcher()
    col_map = {"日期": "date", "开盘": "open", "收盘": "close", "最高": "high",
               "最低": "low", "成交量": "volume", "涨跌幅": "pct_change"}

    deadline = _time.time() + _WEB_POOL_INJECT_MAX_TIME

    for item in items:
        # 总时间超限则立即返回（剩余股票保留原始数据）
        if _time.time() > deadline:
            logger.warning(f"股票池涨幅注入超总时限({_WEB_POOL_INJECT_MAX_TIME}s)，"
                           f"已处理{items.index(item)}/{len(items)}只，剩余跳过")
            break

        code = item.get("code", "")
        entry_price = item.get("entry_price", 0) or 0

        # 确保 tier 字段存在
        tier = item.get("tier", "")
        score = item.get("score", 0) or 0

        try:
            # 带超时的 K 线获取（线程隔离，VPN 下不会无限挂起）
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    fetcher.fetch_daily_kline, str(code), start_date="20240101"
                )
                try:
                    df = future.result(timeout=_WEB_KLINE_FETCH_TIMEOUT)
                except FutureTimeoutError:
                    logger.warning(f"获取 {code} K线数据超时({_WEB_KLINE_FETCH_TIMEOUT}s)，跳过")
                    _set_default_gains(item)
                    item["tier"] = tier or _score_to_tier(score)
                    continue

            if df is None or df.empty or len(df) < 1:
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

    # ── API: 运维监控 ─────────────────────────────

    @app.get("/api/ops/overview")
    async def get_ops_overview():
        """获取运维面板完整数据（系统+引擎+策略+交易+日志+池总览）"""
        try:
            data = ops_collector.collect_all(include_pool_signals=True)
            return {"success": True, **data}
        except Exception as e:
            return JSONResponse({"success": False, "error": str(e)}, status_code=500)

    @app.get("/api/ops/health")
    async def get_ops_health():
        """获取系统健康数据（CPU/内存/磁盘/进程）"""
        try:
            monitor = SystemMonitor()
            return {"success": True, "system": monitor.collect().to_dict(),
                    "has_psutil": HAS_PSUTIL}
        except Exception as e:
            return JSONResponse({"success": False, "error": str(e)}, status_code=500)

    @app.get("/api/ops/engine")
    async def get_ops_engine():
        """获取实时引擎状态"""
        try:
            snapshot = EngineStateReader.read(LIVE_LOG_DIR)
            return {"success": True, "engine": snapshot.to_dict()}
        except Exception as e:
            return JSONResponse({"success": False, "error": str(e)}, status_code=500)

    @app.get("/api/ops/strategies")
    async def get_ops_strategies():
        """获取各策略运行监控数据"""
        try:
            strategies = ops_collector._collect_strategy_monitoring(DATA_DIR, LIVE_LOG_DIR, DEFAULT_CAPITAL, True)
            return {"success": True, "strategies": strategies}
        except Exception as e:
            return JSONResponse({"success": False, "error": str(e)}, status_code=500)

    @app.get("/api/ops/operations")
    async def get_ops_operations(limit: int = Query(50, ge=1, le=200)):
        """获取最近的交易操作记录"""
        try:
            ops = ops_collector._collect_trade_operations(DATA_DIR)
            return {"success": True, "operations": ops[:limit], "total": len(ops)}
        except Exception as e:
            return JSONResponse({"success": False, "error": str(e)}, status_code=500)

    @app.get("/api/ops/logs")
    async def get_ops_logs(limit: int = Query(50, ge=1, le=500)):
        """获取引擎日志"""
        try:
            logs = ops_collector._collect_logs(LIVE_LOG_DIR)
            return {"success": True, "logs": logs[-limit:], "total": len(logs)}
        except Exception as e:
            return JSONResponse({"success": False, "error": str(e)}, status_code=500)

    # ── API: 诊股模块 ─────────────────────────────

    @app.get("/api/diagnose/{code}")
    async def diagnose_stock(code: str, quick: bool = Query(False)):
        """个股多维度综合诊断

        返回技术面/基本面/资金面/市场面四维分析 + 综合评分
        - code: 6位股票代码
        - quick=true: 快速诊断，只做技术面+市场面
        """
        try:
            if not code or len(code) < 6:
                raise HTTPException(400, "请输入6位股票代码")

            diagnoser = get_diagnoser()
            if quick:
                result = diagnoser.quick_diagnose(code)
            else:
                result = diagnoser.diagnose(code)

            return {"success": True, "data": _diagnosis_to_dict(result)}

        except Exception as e:
            return JSONResponse({"success": False, "error": str(e)}, status_code=500)

    @app.get("/api/diagnose/quick/{code}")
    async def diagnose_stock_quick(code: str):
        """快速诊股 — 仅技术面+市场面，响应更快"""
        return await diagnose_stock(code, quick=True)

    # 诊股页面 → 重定向到仪表盘
    @app.get("/diagnose")
    async def diagnose_page():
        """诊股页面已整合至仪表盘"""
        return RedirectResponse("/")

    @app.get("/api/datasource/health")
    async def get_datasource_health():
        """获取数据源健康状态（各数据渠道可用性）"""
        try:
            from .data_manager import DataFetcher
            fetcher = DataFetcher()
            health = fetcher.get_data_source_health()
            return {"success": True, "data": health}
        except Exception as e:
            return JSONResponse({"success": False, "error": str(e)}, status_code=500)

    @app.post("/api/datasource/reset")
    async def reset_datasource():
        """重置不可用数据源标记，强制下次请求重新尝试所有数据源"""
        try:
            from .data_manager import DataFetcher
            fetcher = DataFetcher()
            fetcher.reset_unavailable_sources()
            return {"success": True, "message": "不可用数据源标记已重置，下次请求将重新尝试"}
        except Exception as e:
            return JSONResponse({"success": False, "error": str(e)}, status_code=500)

    return app


# ─── 启动入口 ────────────────────────────────────────────────

def _ensure_echarts_local():
    """下载 ECharts 到本地 static 目录，确保不依赖外网 CDN"""
    dest = os.path.join(STATIC_DIR, "echarts.min.js")
    if os.path.exists(dest) and os.path.getsize(dest) > 100_000:
        return  # 已存在且大小合理

    print("[setup] 正在下载 ECharts 到本地（首次启动需要，后续直接使用本地副本）...")
    try:
        import urllib.request
        # 国内 CDN 镜像（比 jsdelivr 在国内更稳定）+ 国际 CDN 回落
        urls = [
            "https://registry.npmmirror.com/echarts/5.5.0/files/dist/echarts.min.js",
            "https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js",
        ]
        for url in urls:
            try:
                req = urllib.request.Request(url, headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0"
                })
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = resp.read()
                    if len(data) > 100_000:
                        with open(dest, "wb") as f:
                            f.write(data)
                        print(f"[setup] ECharts 已下载到本地: {dest} ({len(data):,} bytes)")
                        return
            except Exception as e:
                print(f"[setup] 尝试 {url[:50]}... 失败: {e}")
                continue
        print("[setup] ⚠ ECharts 下载失败，将回退到 CDN 加载")
    except Exception as e:
        print(f"[setup] ⚠ ECharts 下载异常: {e}，将回退到 CDN 加载")



def start_dashboard(host: str = "0.0.0.0", port: int = 8080, reload: bool = False):
    """启动仪表盘 Web 服务"""
    if not HAS_FASTAPI:
        print("错误: 需要安装 fastapi 和 uvicorn")
        print("  pip install fastapi uvicorn")
        return

    # 确保 static 文件和 dashboard.html 存在
    os.makedirs(STATIC_DIR, exist_ok=True)
    _ensure_echarts_local()
    _ensure_dashboard_html()

    print(f"""
╔══════════════════════════════════════════════╗
║    CloudKnight Dashboard                     ║
║    数据驱动仪表盘                              ║
╚══════════════════════════════════════════════╝

  仪表盘地址: http://0.0.0.0:{port}
  API 文档:   http://0.0.0.0:{port}/docs

  提示: 也可使用 http://localhost:{port} 。VPN 环境下若 localhost 不可用，请用 0.0.0.0
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
<script src="/static/echarts.min.js" onerror="
  var s=document.createElement('script');
  s.src='https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js';
  document.head.appendChild(s);
"></script>
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

/* ═══ 个股诊断 ═══ */
.score-color-strong{color:#4caf50}.score-color-good{color:#8bc34a}.score-color-neutral{color:#ffc107}.score-color-warn{color:#ff9800}.score-color-danger{color:#f44336}
.score-gauge{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:24px;margin-bottom:24px;display:grid;grid-template-columns:1fr 3fr;gap:24px;align-items:center}
.score-circle{text-align:center}
.score-big{font-size:72px;font-weight:700;line-height:1}
.score-label{font-size:16px;color:var(--text-dim);margin-top:8px}
.score-details{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:14px}
.score-item{text-align:center;padding:12px;background:#152535;border-radius:8px}
.score-item .dim-label{font-size:13px;color:var(--text-dim);margin-bottom:4px}
.score-item .val{font-size:22px;font-weight:600}
.diag-panel{background:var(--panel);border:1px solid var(--border);border-radius:10px;margin-bottom:20px;overflow:hidden}
.diag-panel-header{background:#152535;padding:14px 20px;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between}
.diag-panel-header h3{font-size:16px;color:var(--text)}
.diag-panel-header .badge{font-size:12px;padding:3px 10px;border-radius:10px;font-weight:500}
.badge-bullish{background:rgba(76,175,80,.15);color:#4caf50}.badge-bearish{background:rgba(244,67,54,.15);color:#f44336}.badge-neutral{background:rgba(255,193,7,.15);color:#ffc107}
.diag-panel-body{padding:20px}
.diag-table{width:100%;border-collapse:collapse}
.diag-table td{padding:8px 12px;border-bottom:1px solid #1a2e40;font-size:14px}
.diag-table td:first-child{color:var(--text-dim);width:130px}
.diag-table td:last-child{color:var(--text)}
.recommendation-box{background:linear-gradient(135deg,#1a3a2a 0%,var(--panel) 100%);border:1px solid #2a5a3a;border-radius:10px;padding:20px 24px;margin-bottom:20px}
.recommendation-box h3{font-size:16px;color:#4caf50;margin-bottom:8px}
.recommendation-box p{font-size:14px;color:#a0c8a0;line-height:1.8}
.risk-tag{padding:3px 12px;border-radius:10px;font-size:12px;font-weight:500;margin-left:6px}
.risk-low{background:rgba(76,175,80,.15);color:#4caf50}.risk-mid{background:rgba(255,193,7,.15);color:#ffc107}.risk-high{background:rgba(255,152,0,.15);color:#ff9800}.risk-extreme{background:rgba(244,67,54,.15);color:#f44336}
.diag-loading{display:none;position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(15,25,35,.7);z-index:999;align-items:center;justify-content:center}
.diag-loading.active{display:flex}
.diag-spinner{width:40px;height:40px;border:3px solid var(--border);border-top-color:var(--accent);border-radius:50%;animation:diag-spin .8s linear infinite}
@keyframes diag-spin{to{transform:rotate(360deg)}}
.diag-error{background:rgba(244,67,54,.1);border:1px solid rgba(244,67,54,.3);border-radius:8px;padding:16px;color:#ef9a9a;font-size:14px;text-align:center;display:none;margin-bottom:16px}
.diag-error.show{display:block}
.diag-state{text-align:center;padding:40px 0;color:var(--text-dim)}
.diag-state .icon{font-size:64px;margin-bottom:16px;opacity:.6}
.diag-state .text{font-size:16px;color:var(--text-dim)}
.diag-search{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:16px}
.diag-search input{width:200px;padding:8px 12px;border:1px solid var(--border);border-radius:6px;background:var(--panel);color:var(--text);font-size:14px;outline:none}
.diag-search input:focus{border-color:var(--accent)}
.diag-search .btn{padding:8px 18px;border:none;border-radius:6px;cursor:pointer;font-size:14px;font-weight:500;transition:all .2s}
.diag-search .btn-primary{background:var(--accent);color:#fff}
.diag-search .btn-primary:hover{background:#2980b9}
.diag-search .btn-outline{background:transparent;border:1px solid var(--border);color:var(--text-dim)}
.diag-search .btn-outline:hover{border-color:var(--accent);color:var(--accent)}
@media(max-width:768px){.score-gauge{grid-template-columns:1fr;text-align:center}.score-details{grid-template-columns:repeat(2,1fr)}}
.strategy-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px;margin-bottom:20px}
.strategy-card{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:18px}
.strategy-card-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:12px}
.strategy-card-header h4{font-size:15px;color:var(--text);display:flex;align-items:center;gap:6px}
.strategy-score-badge{font-size:22px;font-weight:700;padding:4px 10px;border-radius:6px;min-width:48px;text-align:center}
.stg-sig-buy{background:rgba(76,175,80,.12);color:#4caf50;border:1px solid rgba(76,175,80,.25)}
.stg-sig-hold{background:rgba(255,193,7,.1);color:#ffc107;border:1px solid rgba(255,193,7,.2)}
.stg-sig-sell{background:rgba(244,67,54,.1);color:#f44336;border:1px solid rgba(244,67,54,.2)}
.strategy-card .stg-rating{font-size:13px;color:var(--text-dim);margin-bottom:10px}
.strategy-card .stg-reasons{list-style:none;padding:0;margin:0}
.strategy-card .stg-reasons li{font-size:13px;padding:3px 0;color:#9acd9a;display:flex;align-items:center;gap:5px}
.strategy-card .stg-warnings{list-style:none;padding:0;margin:8px 0 0 0}
.strategy-card .stg-warnings li{font-size:12px;padding:2px 0;color:#d4a76a;display:flex;align-items:center;gap:5px}
.strategy-card .stg-match-bar{height:4px;background:#1a2e40;border-radius:2px;margin-top:10px;overflow:hidden}
.strategy-card .stg-match-fill{height:100%;border-radius:2px;transition:width .5s}
.stg-match-strong{background:#4caf50}.stg-match-good{background:#8bc34a}.stg-match-neutral{background:#ffc107}.stg-match-low{background:#f44336}
.data-quality-tag{font-size:12px;padding:2px 10px;border-radius:10px;margin-left:8px}
.dq-complete{background:rgba(76,175,80,.15);color:#4caf50}
.dq-partial{background:rgba(255,193,7,.15);color:#ffc107}
.dq-sparse{background:rgba(244,67,54,.15);color:#f44336}

/* Ops Dashboard */
.ops-strategy-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; }
@media (max-width:1200px) { .ops-strategy-grid { grid-template-columns:repeat(2,1fr); } }
@media (max-width:700px) { .ops-strategy-grid { grid-template-columns:1fr; } }
.ops-strategy-card {
  background:var(--panel); border:1px solid var(--border); border-radius:8px;
  padding:14px; min-width:0;
}
.ops-strategy-card .card-header {
  display:flex; align-items:center; gap:8px; margin-bottom:10px;
  padding-bottom:8px; border-bottom:1px solid var(--border);
}
.ops-strategy-card .card-header .icon { font-size:20px; }
.ops-strategy-card .card-header .name { font-weight:700; font-size:14px; color:#fff; }
.ops-strategy-card .card-header .badge { font-size:10px; padding:1px 8px; border-radius:10px; }
.ops-strategy-card .metric-row {
  display:flex; justify-content:space-between; padding:3px 0;
  font-size:12px; color:var(--text-dim);
}
.ops-strategy-card .metric-row .val { font-weight:600; color:var(--text); }
.ops-strategy-card .signal-item {
  font-size:11px; padding:2px 0; display:flex; align-items:center; gap:6px;
  border-top:1px solid rgba(255,255,255,0.03);
}
.ops-signal-dot {
  width:6px; height:6px; border-radius:50%; flex-shrink:0;
}
.ops-engine-indicator {
  display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px;
  animation:pulse 2s infinite;
}
.ops-engine-indicator.running { background:#27ae60; }
.ops-engine-indicator.stopped { background:#e74c3c; }
.ops-engine-indicator.paused { background:#f39c12; }
@keyframes pulse {
  0%,100% { opacity:1; } 50% { opacity:0.4; }
}
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
  <button data-tab="ops">运维监控</button>
  <button data-tab="diagnose">个股诊断</button>
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
  <!-- ═══ 运维监控 ═══ -->
  <div class="tab-panel" id="tab-ops">
    <!-- 系统健康 -->
    <div class="chart-box" style="margin-bottom:14px;"><div class="chart-title">🖥️ 系统健康 <span style="color:var(--text-dim);font-size:11px;margin-left:8px;" id="ops-health-time"></span></div>
      <div class="card-row" id="ops-health-cards"></div>
    </div>

    <!-- 引擎状态 + 关键指标 -->
    <div class="card-row" id="ops-engine-status"></div>

    <!-- 策略运行监控 (4列) -->
    <div class="chart-box" style="margin-bottom:14px;"><div class="chart-title">📊 策略运行监控</div>
      <div class="ops-strategy-grid" id="ops-strategies"></div>
    </div>

    <!-- 交易操作记录 -->
    <div class="chart-box" style="margin-bottom:14px;"><div class="chart-title">📋 交易操作记录 <span id="ops-ops-total" style="color:var(--text-dim);font-size:11px;margin-left:8px;"></span></div>
      <div style="margin-bottom:8px;display:flex;gap:6px;align-items:center;" id="ops-ops-filter">
        <span style="color:var(--text-dim);font-size:11px;">筛选:</span>
        <button class="ops-filter-btn active" data-action="" style="background:var(--accent);color:#fff;border:1px solid var(--border);padding:2px 10px;border-radius:3px;cursor:pointer;font-size:11px;">全部</button>
        <button class="ops-filter-btn" data-action="buy" style="background:var(--panel);color:var(--text-dim);border:1px solid var(--border);padding:2px 10px;border-radius:3px;cursor:pointer;font-size:11px;">🟢 建仓</button>
        <button class="ops-filter-btn" data-action="add" style="background:var(--panel);color:var(--text-dim);border:1px solid var(--border);padding:2px 10px;border-radius:3px;cursor:pointer;font-size:11px;">➕ 加仓</button>
        <button class="ops-filter-btn" data-action="sell" style="background:var(--panel);color:var(--text-dim);border:1px solid var(--border);padding:2px 10px;border-radius:3px;cursor:pointer;font-size:11px;">🔴 卖出</button>
        <button class="ops-filter-btn" data-action="stop_loss" style="background:var(--panel);color:var(--text-dim);border:1px solid var(--border);padding:2px 10px;border-radius:3px;cursor:pointer;font-size:11px;">⛔ 止损</button>
        <button class="ops-filter-btn" data-action="take_profit" style="background:var(--panel);color:var(--text-dim);border:1px solid var(--border);padding:2px 10px;border-radius:3px;cursor:pointer;font-size:11px;">✅ 止盈</button>
      </div>
      <div class="table-wrap"><table id="table-ops"><thead><tr><th>时间</th><th>策略</th><th>操作</th><th>代码</th><th>名称</th><th>价格</th><th>数量</th><th>金额</th><th>盈亏</th><th>原因</th></tr></thead><tbody></tbody></table></div>
    </div>

    <!-- 引擎日志 -->
    <div class="chart-box"><div class="chart-title">📝 引擎日志 <span id="ops-log-total" style="color:var(--text-dim);font-size:11px;margin-left:8px;"></span></div>
      <div class="table-wrap" style="max-height:300px;overflow-y:auto;"><table id="table-ops-logs"><thead><tr><th>时间</th><th>阶段</th><th>事件</th></tr></thead><tbody></tbody></table></div>
    </div>
  </div>
  <!-- ═══ 个股诊断 ═══ -->
  <div class="tab-panel" id="tab-diagnose">
    <div class="diag-search">
      <input type="text" id="diagCode" placeholder="输入股票代码 如 000001" maxlength="6" autocomplete="off">
      <button class="btn btn-primary" onclick="doDiagnose(false)">全面诊断</button>
      <button class="btn btn-outline" onclick="doDiagnose(true)">快速诊断</button>
      <span style="font-size:12px;color:var(--text-dim)">提示：快速诊断仅做技术面+市场面，响应更快</span>
    </div>
    <div class="diag-error" id="diagError"></div>
    <div class="diag-state" id="diagInit">
      <div class="icon">🔍</div>
      <div class="text">输入 6 位股票代码，点击"全面诊断"查看个股综合评估报告</div>
    </div>
    <div id="diagResult" style="display:none">
      <div class="card-row" id="diagOverviewCards"></div>
      <div class="score-gauge">
        <div class="score-circle">
          <div class="score-big" id="diagScore">--</div>
          <div class="score-label" id="diagRating">--</div>
        </div>
        <div class="score-details" id="diagScoreDetails"></div>
      </div>
      <div class="recommendation-box">
        <h3>📋 综合建议 <span class="risk-tag" id="diagRisk">--</span></h3>
        <p id="diagRecommendation">--</p>
        <p style="margin-top:8px;font-size:13px;color:#7ca87c" id="diagSummary"></p>
      </div>
      <div class="diag-panel" id="diagTechPanel">
        <div class="diag-panel-header"><h3>📊 技术面分析</h3><span class="badge" id="diagTechBadge">--</span></div>
        <div class="diag-panel-body">
          <div class="chart-box" style="margin-bottom:16px"><div class="chart h300" id="chart-diag-tech"></div></div>
          <table class="diag-table" id="diagTechTable"></table>
        </div>
      </div>
      <div class="diag-panel">
        <div class="diag-panel-header"><h3>🏢 基本面分析</h3><span class="badge" id="diagFundBadge">--</span></div>
        <div class="diag-panel-body"><table class="diag-table" id="diagFundTable"></table></div>
      </div>
      <div class="diag-panel">
        <div class="diag-panel-header"><h3>💰 资金面分析</h3><span class="badge" id="diagCapBadge">--</span></div>
        <div class="diag-panel-body"><table class="diag-table" id="diagCapTable"></table></div>
      </div>
      <div class="diag-panel">
        <div class="diag-panel-header"><h3>🌐 市场环境</h3><span class="badge" id="diagMktBadge">--</span></div>
        <div class="diag-panel-body"><table class="diag-table" id="diagMktTable"></table></div>
      </div>
      <div class="diag-panel">
        <div class="diag-panel-header"><h3>🎯 策略匹配诊断 <span class="data-quality-tag" id="diagDataQuality">--</span></h3></div>
        <div class="diag-panel-body">
          <div class="strategy-grid" id="diagStrategyGrid"></div>
        </div>
      </div>
    </div>
    <div class="diag-loading" id="diagLoading"><div class="diag-spinner"></div></div>
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
    ops: loadOps, diagnose: loadDiagnose,
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

// ═══ 7. 运维监控 ═══════════════════════════════════════════
let opsData = null;
let opsRefreshTimer = null;
async function loadOps() {
  opsData = await API('/ops/overview');
  if (!opsData || !opsData.success) {
    $('#tab-ops').innerHTML = '<div class="loading">运维数据加载失败</div>'; return;
  }
  renderOps();
  // 每30秒自动刷新
  if (opsRefreshTimer) clearInterval(opsRefreshTimer);
  opsRefreshTimer = setInterval(async () => {
    if (!$('#tab-ops')?.classList?.contains('active')) return;
    opsData = await API('/ops/overview');
    if (opsData && opsData.success) renderOps();
  }, 30000);
}

function renderOps() {
  renderOpsHealth(opsData.system);
  renderOpsEngine(opsData.engine);
  renderOpsStrategies(opsData.strategies);
  renderOpsOperations(opsData.operations);
  renderOpsLogs(opsData.logs);
}

// ── 系统健康 ──
function renderOpsHealth(sys) {
  $('#ops-health-time').textContent = `采集时间: ${new Date().toLocaleTimeString()}`;
  const getColor = (val, warn, crit) => val >= crit ? 'var(--red)' : val >= warn ? 'var(--orange)' : 'var(--green)';
  const cards = [
    { label:'CPU使用率', val:sys.cpu_percent+'%', color:getColor(sys.cpu_percent,60,85), sub:`${sys.cpu_count}核` },
    { label:'内存占用', val:sys.memory_percent+'%', color:getColor(sys.memory_percent,70,90), sub:`${sys.memory_used_mb.toFixed(0)}/${sys.memory_total_mb.toFixed(0)} MB` },
    { label:'磁盘空间', val:sys.disk_percent+'%', color:getColor(sys.disk_percent,70,90), sub:`剩余 ${sys.disk_free_gb.toFixed(1)} GB` },
    { label:'进程内存', val:sys.process_memory_mb.toFixed(0)+' MB', color:sys.process_memory_mb>500?'var(--orange)':'var(--green)', sub:`PID ${sys.process_pid} | ${sys.process_threads}线程` },
    { label:'运行时间', val:sys.uptime_str||'--', color:'var(--accent)', sub:`Python ${sys.python_version}` },
  ];
  $('#ops-health-cards').innerHTML = cards.map(c => `
    <div class="card" style="border-left:3px solid ${c.color};">
      <div class="card-label">${c.label}</div>
      <div class="card-value" style="color:${c.color};font-size:20px;">${c.val}</div>
      <div class="card-sub" style="color:var(--text-dim);font-size:11px;">${c.sub}</div>
    </div>
  `).join('');
}

// ── 引擎状态 ──
function renderOpsEngine(eng) {
  const stateColors = { running:'#27ae60', stopped:'#e74c3c', paused:'#f39c12', starting:'#f39c12', stopping:'#e74c3c' };
  const stateLabels = { running:'运行中', stopped:'已停止', paused:'已暂停', starting:'启动中', stopping:'停止中' };
  const sc = stateColors[eng.state]||'var(--text-dim)';
  const sl = stateLabels[eng.state]||eng.state;

  let html = `<div class="card" style="border-left:3px solid ${sc};flex:2;min-width:280px;">
    <div class="card-label">⚙️ 引擎状态</div>
    <div style="display:flex;align-items:center;gap:8px;margin-top:4px;">
      <span class="ops-engine-indicator ${eng.state==='running'?'running':eng.state==='paused'?'paused':'stopped'}"></span>
      <span style="font-weight:700;font-size:16px;color:${sc};">${sl}</span>
      ${eng.available ? '' : '<span style="font-size:10px;color:var(--text-dim);">(离线/未启动)</span>'}
    </div>
    <div class="card-sub" style="margin-top:4px;color:var(--text-dim);">阶段: ${eng.phase_label||'--'} | 交易日: ${eng.is_trading_day?'是':'否'}</div>
  </div>`;

  const metrics = [
    { label:'信号总数', val:eng.signal_count },
    { label:'今日决策', val:eng.decision_count },
    { label:'日志条数', val:eng.log_count },
    { label:'上次扫描', val:eng.last_scan_time||'--' },
  ];
  html += metrics.map(m => `
    <div class="card" style="flex:1;min-width:100px;">
      <div class="card-label">${m.label}</div>
      <div class="card-value" style="font-size:18px;">${m.val}</div>
    </div>
  `).join('');

  $('#ops-engine-status').innerHTML = html;
}

// ── 策略监控 ──
function renderOpsStrategies(strategies) {
  if (!strategies || !strategies.length) {
    $('#ops-strategies').innerHTML = '<div class="empty">暂无策略运行数据</div>'; return;
  }
  const icons = { dragon_head:'🐉', sparrow:'🐦', turtle:'🐢', value_invest:'📈' };
  const colors = { dragon_head:'#e74c3c', sparrow:'#f39c12', turtle:'#27ae60', value_invest:'#3498db' };

  $('#ops-strategies').innerHTML = strategies.map(s => {
    const icon = icons[s.key]||'📊';
    const color = colors[s.key]||'var(--accent)';
    const tierColors = { focus:'#e74c3c', watch:'#f1c40f', broad:'#3498db' };
    const tiers = s.pool_tiers||{};
    const sigActive = s.signal_count > 0;
    const sigHtml = sigActive
      ? `<div style="color:var(--green);font-size:11px;">🟢 信号活跃</div>`
      : `<div style="color:var(--text-dim);font-size:11px;">⏸ 暂无信号</div>`;

    // 信号标签
    let signalsHtml = '';
    if (s.latest_signals && s.latest_signals.length) {
      signalsHtml = '<div style="margin-top:8px;border-top:1px solid var(--border);padding-top:6px;">' +
        s.latest_signals.slice(0,3).map(sig => {
          const typeIcon = sig.type==='buy'?'🟢':sig.type==='sell'?'🔴':sig.type==='add'?'➕':sig.type==='reduce'?'➖':sig.type==='close'?'⏹':'👁';
          return `<div class="signal-item">
            <span class="ops-signal-dot" style="background:${sig.type==='buy'?'#27ae60':'#e74c3c'}"></span>
            <span style="font-weight:600;">${sig.code} ${sig.name||''}</span>
            <span style="color:var(--text-dim);">${typeIcon} ¥${(sig.price||0).toFixed(2)}</span>
            <span style="color:var(--text-dim);font-size:10px;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${sig.reason||''}">${sig.reason||''}</span>
          </div>`;
        }).join('') + '</div>';
    }

    return `
    <div class="ops-strategy-card" style="border-top:3px solid ${color};">
      <div class="card-header">
        <span class="icon">${icon}</span>
        <span class="name">${s.label}</span>
        <span class="badge" style="background:rgba(255,255,255,0.08);color:var(--text-dim);">${s.maintenance_interval_days}天/维护</span>
      </div>
      <div class="metric-row"><span>买/卖信号</span><span class="val"><span style="color:var(--green)">${s.buy_count}</span>/<span style="color:var(--red)">${s.sell_count}</span></span></div>
      <div class="metric-row"><span>扫描耗时</span><span class="val">${s.last_scan_duration||0}秒</span></div>
      <div class="metric-row"><span>池内标的</span><span class="val">🎯${tiers.focus||0} 👀${tiers.watch||0} 📋${tiers.broad||0}</span></div>
      <div class="metric-row"><span>总权益</span><span class="val" style="color:${(s.return_pct||0)>=0?'var(--green)':'var(--red)'}">¥${(s.equity||0).toLocaleString('zh-CN',{maximumFractionDigits:0})}</span></div>
      <div class="metric-row"><span>收益率</span><span class="val" style="color:${(s.return_pct||0)>=0?'var(--green)':'var(--red)'}">${(s.return_pct||0)>=0?'+':''}${(s.return_pct||0).toFixed(2)}%</span></div>
      <div class="metric-row"><span>持仓/交易</span><span class="val">${s.position_count}只 / ${s.trade_count}笔</span></div>
      <div class="metric-row"><span>最大回撤</span><span class="val" style="color:var(--red)">${(s.max_drawdown||0).toFixed(2)}%</span></div>
      ${sigActive ? sigHtml : ''}
      ${signalsHtml}
    </div>`;
  }).join('');
}

// ── 交易操作 ──
let opsFilterAction = '';
function renderOpsOperations(operations) {
  let ops = operations || [];
  $('#ops-ops-total').textContent = `共 ${ops.length} 条记录`;

  // 筛选按钮事件
  document.querySelectorAll('.ops-filter-btn').forEach(b => b.addEventListener('click', function() {
    opsFilterAction = this.dataset.action;
    document.querySelectorAll('.ops-filter-btn').forEach(b => { b.classList.remove('active'); b.style.background='var(--panel)'; b.style.color='var(--text-dim)'; });
    this.classList.add('active'); this.style.background='var(--accent)'; this.style.color='#fff';
    renderOpsOperations(opsData?.operations || []);
  }));

  if (opsFilterAction) {
    ops = ops.filter(o => o.action_type === opsFilterAction);
  }

  if (!ops.length) {
    $('#table-ops tbody').innerHTML = '<tr><td colspan="10" class="empty">暂无交易操作记录</td></tr>'; return;
  }

  const actionBadge = (a, t) => {
    const map = {
      buy: { icon:'🟢', label:'建仓', color:'#27ae60' },
      add: { icon:'➕', label:'加仓', color:'#3498db' },
      sell: { icon:'🔴', label:'卖出', color:'#e74c3c' },
      reduce: { icon:'➖', label:'减持', color:'#e67e22' },
      close: { icon:'⏹', label:'平仓', color:'#95a5a6' },
      stop_loss: { icon:'⛔', label:'止损', color:'#c0392b' },
      take_profit: { icon:'✅', label:'止盈', color:'#27ae60' },
    };
    const m = map[t] || { icon:'📌', label:a||'--', color:'var(--text-dim)' };
    return `<span style="color:${m.color};font-weight:600;">${m.icon} ${m.label}</span>`;
  };

  $('#table-ops tbody').innerHTML = ops.map(o => {
    const pnlStr = (o.pnl||0) !== 0
      ? `<span class="${(o.pnl||0)>=0?'pnl-pos':'pnl-neg'}">${(o.pnl||0)>=0?'+':''}${(o.pnl||0).toFixed(2)}</span>`
      : '<span style="color:var(--text-dim)">--</span>';
    return `
    <tr>
      <td style="color:var(--text-dim);font-size:11px;">${o.date||o.time||'--'}</td>
      <td>${o.strategy||'--'}</td>
      <td>${actionBadge(o.action, o.action_type)}</td>
      <td style="font-weight:600;">${o.code||'--'}</td>
      <td>${o.name||''}</td>
      <td>¥${(o.price||0).toFixed(2)}</td>
      <td>${o.volume||0}</td>
      <td>¥${(o.amount||0).toFixed(0)}</td>
      <td>${pnlStr}</td>
      <td style="color:var(--text-dim);font-size:11px;max-width:160px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${o.reason||''}">${o.reason||'--'}</td>
    </tr>`;
  }).join('');
}

// ── 引擎日志 ──
function renderOpsLogs(logs) {
  const items = logs || [];
  $('#ops-log-total').textContent = `最近 ${items.length} 条`;
  if (!items.length) {
    $('#table-ops-logs tbody').innerHTML = '<tr><td colspan="3" class="empty">暂无引擎日志</td></tr>'; return;
  }
  const phaseLabels = {
    closed:'非交易', pre_market:'盘前', auction:'竞价', auction_result:'竞价结果',
    morning:'早盘', lunch:'午休', afternoon:'午盘', post_market:'收盘',
  };
  const phaseBadge = (p) => {
    const label = phaseLabels[p] || p;
    return `<span style="background:rgba(52,152,219,0.15);color:var(--accent);padding:0 6px;border-radius:3px;font-size:10px;">${label}</span>`;
  };

  $('#table-ops-logs tbody').innerHTML = items.map(l => `
    <tr>
      <td style="color:var(--text-dim);font-size:11px;white-space:nowrap;">${l.time||'--'}</td>
      <td>${phaseBadge(l.phase)}</td>
      <td style="font-size:12px;">${l.event||l.detail||'--'}</td>
    </tr>
  `).join('');
}

// ═══ 8. 个股诊断 ═════════════════════════════════════════════
function loadDiagnose() {
  const input = document.getElementById('diagCode');
  if (input) input.addEventListener('keydown', function(e) {
    if (e.key === 'Enter') doDiagnose(false);
  });
}

function doDiagnose(quick) {
  const code = (document.getElementById('diagCode')?.value || '').trim();
  if (!code || code.length !== 6 || !/^\\d{6}$/.test(code)) {
    showDiagError('请输入6位数字股票代码');
    return;
  }
  document.getElementById('diagResult').style.display = 'none';
  document.getElementById('diagInit').style.display = 'none';
  document.getElementById('diagError').classList.remove('show');
  document.getElementById('diagLoading').style.display = 'flex';

  const url = quick ? '/api/diagnose/quick/' + code : '/api/diagnose/' + code;
  fetch(url)
    .then(r => r.json())
    .then(resp => {
      document.getElementById('diagLoading').style.display = 'none';
      if (!resp.success) { showDiagError(resp.error || '诊断失败'); return; }
      renderDiagResult(resp.data);
    })
    .catch(e => {
      document.getElementById('diagLoading').style.display = 'none';
      showDiagError('网络请求失败: ' + e.message);
    });
}

function showDiagError(msg) {
  const box = document.getElementById('diagError');
  box.textContent = '\u26a0 ' + msg;
  box.classList.add('show');
}

function diagScoreColor(s) {
  if (s >= 80) return 'score-color-strong';
  if (s >= 65) return 'score-color-good';
  if (s >= 50) return 'score-color-neutral';
  if (s >= 35) return 'score-color-warn';
  return 'score-color-danger';
}

function diagScoreEmoji(s) {
  if (s >= 80) return '🟢';
  if (s >= 65) return '🟡';
  if (s >= 50) return '🟠';
  if (s >= 35) return '🟤';
  return '🔴';
}

function diagRiskClass(r) {
  if (r === '\u4f4e') return 'risk-low';
  if (r === '\u4e2d') return 'risk-mid';
  if (r === '\u9ad8') return 'risk-high';
  return 'risk-extreme';
}

function diagFmtVal(v, suffix) {
  if (v == null || v === undefined) return '--';
  if (typeof v === 'number') return v.toFixed(2) + (suffix || '');
  return v + (suffix || '');
}

function diagSignalLabel(ir) {
  if (!ir) return '--';
  const m = {buy:'\u91d1\u53c9\u4e70\u5165 \u2705', sell:'\u6b7b\u53c9\u5356\u51fa \u26a0', hold:'\u6301\u6709 \u27f3'};
  return (m[ir.signal] || ir.signal) + ' (' + (ir.strength||50).toFixed(0) + ')';
}

function diagTrendLabel(ir) {
  if (!ir) return '--';
  const m = {bullish:'\u504f\u591a', bearish:'\u504f\u7a7a', overbought:'\u8d85\u4e70', oversold:'\u8d85\u5356', neutral:'\u4e2d\u6027'};
  return m[ir.trend] || ir.trend;
}

function diagRenderBadge(id, score, rating) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = rating + ' ' + score + '\u5206';
  el.className = 'badge badge-' + (score>=65?'bullish':score>=50?'neutral':'bearish');
}

function renderDiagResult(d) {
  document.getElementById('diagResult').style.display = 'block';
  document.getElementById('diagError').classList.remove('show');

  const t = d.technical, f = d.fundamental, c = d.capital, m = d.market;

  document.getElementById('diagOverviewCards').innerHTML = `
    <div class="card"><div class="card-label">${d.name} ${d.code}</div><div class="card-value" style="font-size:20px">${diagFmtVal(t.current_price)}</div><div class="card-sub">\u5f53\u524d\u4ef7\u683c</div></div>
    <div class="card ${(t.pct_change_5d||0)>=0?'green':'red'}"><div class="card-label">5\u65e5\u6da8\u8dcc</div><div class="card-value" style="font-size:22px">${fmtPct(t.pct_change_5d)}</div></div>
    <div class="card ${(t.pct_change_20d||0)>=0?'green':'red'}"><div class="card-label">20\u65e5\u6da8\u8dcc</div><div class="card-value" style="font-size:22px">${fmtPct(t.pct_change_20d)}</div></div>
    <div class="card"><div class="card-label">\u8bca\u65ad\u65f6\u95f4</div><div class="card-value" style="font-size:14px">${d.timestamp||'--'}</div></div>
  `;

  const cs = d.composite_score;
  document.getElementById('diagScore').textContent = cs;
  document.getElementById('diagScore').className = 'score-big ' + diagScoreColor(cs);
  document.getElementById('diagRating').textContent = diagScoreEmoji(cs) + ' ' + d.composite_rating;
  document.getElementById('diagRisk').textContent = '\u98ce\u9669: ' + d.risk_level;
  document.getElementById('diagRisk').className = 'risk-tag ' + diagRiskClass(d.risk_level);
  document.getElementById('diagRecommendation').textContent = d.recommendation;
  document.getElementById('diagSummary').textContent = d.summary;

  const dims = d.detail_scores || {};
  document.getElementById('diagScoreDetails').innerHTML = Object.entries(dims).map(([k,v]) =>
    `<div class="score-item"><div class="dim-label">${k}</div><div class="val ${diagScoreColor(v)}">${v}\u5206</div></div>`
  ).join('');

  diagRenderBadge('diagTechBadge', t.score, t.rating);
  document.getElementById('diagTechTable').innerHTML = `
    <tr><td>\u5747\u7ebf\u6392\u5217</td><td>${t.ma_alignment||'--'}</td></tr>
    <tr><td>MACD \u4fe1\u53f7</td><td>${diagSignalLabel(t.macd)}</td></tr>
    <tr><td>KDJ \u4fe1\u53f7</td><td>${diagSignalLabel(t.kdj)}</td></tr>
    <tr><td>RSI \u72b6\u6001</td><td>${diagTrendLabel(t.rsi)}</td></tr>
    <tr><td>\u5e03\u6797\u5e26\u4f4d\u7f6e</td><td>${t.bollinger ? (t.bollinger.details ? (t.bollinger.details.position||'--') : '--') : '--'}</td></tr>
    <tr><td>\u91cf\u4ef7\u5173\u7cfb</td><td>${t.volume_price ? (t.volume_price.details ? (t.volume_price.details.pattern||'--') : '--') : '--'}</td></tr>
    <tr><td>ATR \u6ce2\u5e45</td><td>${diagFmtVal(t.atr)}</td></tr>
    <tr><td>\u652f\u6491\u4f4d</td><td style="color:#4caf50">${diagFmtVal(t.support)}</td></tr>
    <tr><td>\u963b\u529b\u4f4d</td><td style="color:#f44336">${diagFmtVal(t.resistance)}</td></tr>
  `;

  diagRenderBadge('diagFundBadge', f.score, f.rating);
  document.getElementById('diagFundTable').innerHTML = f.data_available ? `
    <tr><td>\u5e02\u76c8\u7387 (PE)</td><td>${diagFmtVal(f.pe, ' \u500d')}</td></tr>
    <tr><td>\u5e02\u51c0\u7387 (PB)</td><td>${diagFmtVal(f.pb, ' \u500d')}</td></tr>
    <tr><td>ROE</td><td>${fmtPct(f.roe)}</td></tr>
    <tr><td>\u8425\u6536\u589e\u957f\u7387</td><td>${fmtPct(f.revenue_growth)}</td></tr>
    <tr><td>\u51c0\u5229\u6da6\u589e\u957f\u7387</td><td>${fmtPct(f.profit_growth)}</td></tr>
    <tr><td>\u603b\u5e02\u503c</td><td>${diagFmtVal(f.market_cap, ' \u4ebf')}</td></tr>
    <tr><td>\u80a1\u606f\u7387</td><td>${fmtPct(f.dividend_yield)}</td></tr>
    <tr><td>\u8d44\u4ea7\u8d1f\u503a\u7387</td><td>${fmtPct(f.debt_ratio)}</td></tr>
    <tr><td>\u6bdb\u5229\u7387</td><td>${fmtPct(f.gross_margin)}</td></tr>
    <tr><td>\u51c0\u5229\u7387</td><td>${fmtPct(f.net_margin)}</td></tr>
  ` : `<tr><td colspan="2" style="text-align:center;color:var(--text-dim)">\u6682\u65e0\u8d22\u52a1\u6570\u636e\uff08\u975e\u8d22\u62a5\u5b63\u6216\u6570\u636e\u6e90\u4e0d\u53ef\u7528\uff09</td></tr>`;

  diagRenderBadge('diagCapBadge', c.score, c.rating);
  const mfLabel = {inflow:'主力流入 🔴', outflow:'主力流出 🟢', neutral:'中性', unknown:'--'};
  document.getElementById('diagCapTable').innerHTML = c.data_available ? `
    <tr><td>\u4e3b\u529b\u52a8\u5411</td><td>${mfLabel[c.main_force_direction]||'--'}</td></tr>
    <tr><td>\u8fd15\u65e5\u4e3b\u529b\u51c0\u6d41\u5165</td><td>${diagFmtVal(c.main_force_5d_net, ' \u4ebf')}</td></tr>
    <tr><td>\u6362\u624b\u7387</td><td>${fmtPct(c.turnover_rate)}</td></tr>
    <tr><td>\u91cf\u6bd4</td><td>${diagFmtVal(c.volume_ratio)}</td></tr>
  ` : `<tr><td colspan="2" style="text-align:center;color:var(--text-dim)">\u6682\u65e0\u8d44\u91d1\u6d41\u5411\u6570\u636e</td></tr>`;

  diagRenderBadge('diagMktBadge', m.score, m.rating);
  document.getElementById('diagMktTable').innerHTML = `
    <tr><td>\u5927\u76d8\u8d8b\u52bf</td><td>${m.market_trend||'--'}</td></tr>
    <tr><td>\u5e02\u573a\u60c5\u7eea</td><td>${m.market_sentiment||'--'}</td></tr>
    <tr><td>\u6240\u5c5e\u677f\u5757</td><td>${m.sector_name||'--'}</td></tr>
  `;

  renderDiagTechChart(d);
  renderDiagStrategies(d);
  renderDiagDataQuality(d);
  document.getElementById('tab-diagnose').scrollIntoView({behavior:'smooth', block:'start'});
}

function renderDiagDataQuality(d) {
  const el = document.getElementById('diagDataQuality');
  if (!el) return;
  const q = d.data_quality || '';
  el.textContent = '\u6570\u636e: ' + q;
  el.className = 'data-quality-tag ' + (q === '\u5b8c\u6574' ? 'dq-complete' : q === '\u90e8\u5206\u7f3a\u5931' ? 'dq-partial' : 'dq-sparse');
}

function renderDiagStrategies(d) {
  const grid = document.getElementById('diagStrategyGrid');
  if (!grid) return;
  const strategies = d.strategy_diagnoses || [];
  if (!strategies.length) { grid.innerHTML = '<div class="empty">\u7b56\u7565\u8bca\u65ad\u6570\u636e\u4e0d\u8db3</div>'; return; }

  const emojiMap = {dragon_head:'🐲', sparrow:'🐦', turtle:'🐢', value_invest:'💰'};
  const sigMap = {buy:'✅ 看多', hold:'⟳ 观望', sell:'⚠ 看空'};
  const matchColorMap = [null,'stg-match-low','stg-match-low','stg-match-neutral','stg-match-good','stg-match-strong'];

  grid.innerHTML = strategies.map(s => {
    const pct = s.total_conditions > 0 ? Math.round(s.match_count / Math.max(4, s.total_conditions) * 100) : 0;
    const matchClass = s.score >= 70 ? 'stg-match-strong' : s.score >= 55 ? 'stg-match-good' : s.score >= 40 ? 'stg-match-neutral' : 'stg-match-low';
    const sigClass = 'stg-sig-' + s.signal;
    return `<div class="strategy-card">
      <div class="strategy-card-header">
        <h4>${(emojiMap[s.key]||'')} ${s.name}</h4>
        <span class="strategy-score-badge ${sigClass}">${s.score}</span>
      </div>
      <div class="stg-rating">${sigMap[s.signal]||s.signal} \u00b7 ${s.rating}</div>
      <ul class="stg-reasons">${(s.reasons||[]).map(r => '<li>\u2714 ' + r + '</li>').join('')}</ul>
      ${(s.warnings||[]).length ? '<ul class="stg-warnings">' + s.warnings.map(w => '<li>\u26a0 ' + w + '</li>').join('') + '</ul>' : ''}
      <div class="stg-match-bar"><div class="stg-match-fill ${matchClass}" style="width:${pct}%"></div></div>
    </div>`;
  }).join('');
}

function renderDiagTechChart(d) {
  const dom = document.getElementById('chart-diag-tech');
  if (!dom) return;
  if (charts['diag-tech']) charts['diag-tech'].dispose();
  const c = echarts.init(dom);
  charts['diag-tech'] = c;

  const cats = ['MACD','KDJ','RSI','\u5e03\u6797\u5e26','\u91cf\u4ef7','\u5747\u7ebf','\u652f\u6491\u963b\u529b'];
  const vals = [
    d.technical.macd ? d.technical.macd.strength : 50,
    d.technical.kdj ? d.technical.kdj.strength : 50,
    d.technical.rsi ? d.technical.rsi.strength : 50,
    d.technical.bollinger ? d.technical.bollinger.strength : 50,
    d.technical.volume_price ? d.technical.volume_price.strength : 50,
    d.technical.score||50,
    (d.technical.support && d.technical.resistance) ? 55 : 50,
  ];

  c.setOption({
    tooltip: {trigger:'axis'},
    radar: {
      center: ['50%','55%'], radius: '65%',
      indicator: cats.map(cat => ({name:cat, max:100})),
      axisName: {color:'#6d8099',fontSize:11},
      splitArea: {areaStyle:{color:['rgba(52,152,219,.02)','rgba(52,152,219,.04)']}},
    },
    series: [{
      type: 'radar',
      data: [{value:vals, name:'\u6280\u672f\u6307\u6807', areaStyle:{color:'rgba(52,152,219,.15)'}}],
      symbol: 'circle', symbolSize: 4,
      lineStyle: {color:'#3498db', width: 2},
      itemStyle: {color:'#3498db'},
    }],
  });
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
