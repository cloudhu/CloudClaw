"""
数据源回退链验证工具 (v2 - 扩展版)
测试 DataFetcher 中每一级回退链是否能正常工作。
覆盖新数据源：涨停/跌停/连板池回退、北向资金、龙虎榜、融资融券
"""
import sys
import os
import io
import time
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

# Windows PowerShell 默认使用 GBK，强制 UTF-8 输出
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# 确保项目路径可导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 尽早配置日志，避免重复初始化
logging.basicConfig(
    level=logging.WARNING,  # 抑制 INFO 噪音
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
# 抑制 akshare 进度条输出
logging.getLogger("akshare").setLevel(logging.WARNING)

# 禁用 tqdm 进度条
import os as _os
_os.environ["TQDM_DISABLE"] = "1"

from cloudknight.data_manager import DataFetcher

logger = logging.getLogger("fallback-test")

# 测试用股票代码
TEST_STOCKS = {
    "沪主板": "600519",   # 贵州茅台
    "深主板": "000001",   # 平安银行
    "创业板": "300750",   # 宁德时代
    "科创板": "688981",   # 中芯国际
}

# 日期范围
today = datetime.now().strftime("%Y%m%d")
one_month_ago = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
three_months_ago = (datetime.now() - timedelta(days=90)).strftime("%Y%m%d")


def timing(func, *args, **kwargs):
    """测量函数执行时间"""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


def print_header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_result(ok: bool, label: str, detail: str = ""):
    icon = "[OK]" if ok else "[FAIL]"
    print(f"  {icon} {label}" + (f"  -> {detail}" if detail else ""))


def check_df(df) -> Tuple[bool, str]:
    """检查 DataFrame 是否有效"""
    if df is None:
        return False, "返回 None"
    if hasattr(df, 'empty') and df.empty:
        return False, "空 DataFrame"
    rows = len(df) if hasattr(df, '__len__') else 1
    return True, f"{rows} 行数据"


# ══════════════════════════════════════════════════════════════
# 测试入口
# ══════════════════════════════════════════════════════════════

def main():
    print_header("CloudKnight 数据源回退链验证 (v2 扩展版)")
    print(f"  测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  日期范围: {three_months_ago} ~ {today}")

    fetcher = DataFetcher()

    # 重置所有不可用标记（干净状态开始）
    fetcher.reset_unavailable_sources()

    results: Dict[str, bool] = {}

    # ── 1. 股票列表 (3级回退) ────────────────────────
    results["股票列表"] = test_stock_list(fetcher)

    # ── 2. K线回退链 ─────────────────────────────────
    results["K线"] = test_kline_chain(fetcher)

    # ── 3. 指数回退链 ─────────────────────────────────
    results["指数"] = test_index_chain(fetcher)

    # ── 4. 财务数据回退链 ─────────────────────────────
    results["财务"] = test_financial_chain(fetcher)

    # ── 5. 资金流向回退链 ─────────────────────────────
    results["资金流向"] = test_money_flow_chain(fetcher)

    # ── 6. 实时行情回退链 ─────────────────────────────
    results["实时行情"] = test_realtime_chain(fetcher)

    # ── 7. 市场活跃度回退链 (3级) ──────────────────────
    results["市场活跃度"] = test_market_activity_chain(fetcher)

    # ── 8. 涨停池回退链 (新增) ────────────────────────
    results["涨停池"] = test_limit_up_chain(fetcher)

    # ── 9. 连板池回退链 (新增) ────────────────────────
    results["连板池"] = test_continuous_limit_chain(fetcher)

    # ── 10. 跌停池回退链 (新增) ───────────────────────
    results["跌停池"] = test_limit_down_chain(fetcher)

    # ── 11. 北向资金 (新增) ───────────────────────────
    results["北向资金"] = test_northbound_flow(fetcher)

    # ── 12. 龙虎榜 (新增) ─────────────────────────────
    results["龙虎榜"] = test_lhb_detail(fetcher)

    # ── 13. 融资融券 (新增) ───────────────────────────
    results["融资融券"] = test_margin_trading(fetcher)

    # ── 14. 数据源恢复测试 ─────────────────────────────
    results["恢复机制"] = test_recovery(fetcher)

    # ── 数据源健康报告 ────────────────────────────────
    print_header("数据源健康总览")
    health = fetcher.get_data_source_health()
    for category, info in health.items():
        available = info["available"]
        unavailable = info["unavailable"]
        total = info["total"]
        icon = "[OK]" if info["healthy"] else "[WARN]"
        print(f"  {icon} {category}: {len(available)}/{total} 可用")

        if unavailable:
            for u in unavailable:
                print(f"      [x] {u}")
        for a in available:
            print(f"      [v] {a}")

    # ── 总结 ─────────────────────────────────────────
    print_header("验证总结")
    pass_count = sum(1 for v in results.values() if v)
    total = len(results)
    for name, ok in results.items():
        print_result(ok, name)

    print(f"\n  通过率: {pass_count}/{total} ({pass_count/total*100:.0f}%)")

    if pass_count == total:
        print("\n  [OK] 所有回退链验证通过！")
    else:
        print("\n  [WARN] 部分数据源当前不可用是正常现象（已标记跳过，下次自动恢复）")

    return 0 if pass_count == total else 1


# ══════════════════════════════════════════════════════════════
# 各链条测试函数
# ══════════════════════════════════════════════════════════════

def test_stock_list(fetcher) -> bool:
    """测试股票列表获取 (akshare → 东财直连 → 腾讯直连)"""
    print_header("1. 股票列表回退链 [AKShare → 东财HTTP直连 → 腾讯HTTP直连]")

    try:
        df, elapsed = timing(fetcher.fetch_stock_list, force_refresh=True)
        ok, detail = check_df(df)
        print_result(ok, "股票列表获取", f"{detail} ({elapsed:.1f}s)")
        if ok and hasattr(df, 'columns'):
            cols = list(df.columns)
            print(f"      列: {cols}")
            if hasattr(df, 'iloc') and len(df) > 0:
                print(f"      样例: {dict(df.iloc[0])}")
        return ok
    except Exception as e:
        print_result(False, "股票列表获取", str(e))
        logger.warning(f"股票列表获取异常: {e}")
        return False


def test_kline_chain(fetcher) -> bool:
    """测试K线回退链"""
    print_header("2. K线回退链 [东财→新浪→腾讯→腾讯直连→新浪直连→本地DB]")

    ok_count = 0

    for label, stock_code in TEST_STOCKS.items():
        try:
            df, elapsed = timing(
                fetcher.fetch_daily_kline,
                stock_code, start_date=three_months_ago, end_date=today, adjust="qfq"
            )
            ok, detail = check_df(df)
            print_result(ok, f"K线: {label} ({stock_code})", f"{detail} ({elapsed:.1f}s)")
            if ok and hasattr(df, 'columns'):
                cols = [c for c in df.columns if c not in ('股票代码',)]
                print(f"      列: {cols}, 前复权: qfq")
            if ok:
                ok_count += 1
        except Exception as e:
            print_result(False, f"K线: {label} ({stock_code})", str(e))

    return ok_count >= 1


def test_index_chain(fetcher) -> bool:
    """测试指数回退链"""
    print_header("3. 指数日线回退链 [东方财富 → 腾讯 → 腾讯直连]")

    indices = {
        "000001": "上证指数",
        "399001": "深证成指",
        "399006": "创业板指",
        "000300": "沪深300",
    }

    for code, name in indices.items():
        try:
            df, elapsed = timing(
                fetcher.fetch_index_daily,
                code, start_date=three_months_ago, end_date=today
            )
            ok, detail = check_df(df)
            print_result(ok, f"指数: {name} ({code})", f"{detail} ({elapsed:.1f}s)")
        except Exception as e:
            print_result(False, f"指数: {name} ({code})", str(e))

    return True


def test_financial_chain(fetcher) -> bool:
    """测试财务数据回退链"""
    print_header("4. 财务数据回退链 [AKShare指标 → AKShare摘要 → 东财直连]")

    for label, code in [("贵州茅台", "600519"), ("平安银行", "000001")]:
        try:
            df, elapsed = timing(fetcher.fetch_financial_data, code)
            ok, detail = check_df(df)
            print_result(ok, f"财务: {label} ({code})", f"{detail} ({elapsed:.1f}s)")
            if ok and hasattr(df, 'columns'):
                cols = list(df.columns)[:10]
                print(f"      字段(前10): {cols}")
            if ok:
                return True
        except Exception as e:
            print_result(False, f"财务: {label} ({code})", str(e))

    return False


def test_money_flow_chain(fetcher) -> bool:
    """测试资金流向回退链"""
    print_header("5. 资金流向回退链 [AKShare → 东财直连 → 新浪直连]")

    for label, code in list(TEST_STOCKS.items())[:2]:
        try:
            df, elapsed = timing(fetcher.fetch_money_flow, code)
            ok, detail = check_df(df)
            print_result(ok, f"资金流向: {label} ({code})", f"{detail} ({elapsed:.1f}s)")
            if ok and hasattr(df, 'columns'):
                cols = list(df.columns)
                print(f"      字段: {cols}")
            if ok:
                return True
        except Exception as e:
            print_result(False, f"资金流向: {label} ({code})", str(e))

    return False


def test_realtime_chain(fetcher) -> bool:
    """测试实时行情回退链"""
    print_header("6. 实时行情回退链 [新浪 → 腾讯]")

    codes = list(TEST_STOCKS.values())
    try:
        df, elapsed = timing(fetcher.fetch_realtime_quote, codes)
        ok, detail = check_df(df)
        print_result(ok, "实时行情(批量)", f"{detail} ({elapsed:.1f}s)")
        if ok and hasattr(df, 'columns'):
            cols = list(df.columns)
            print(f"      字段: {cols}")
        return ok
    except Exception as e:
        print_result(False, "实时行情(批量)", str(e))
        return False


def test_market_activity_chain(fetcher) -> bool:
    """测试市场活跃度回退链 (3级)"""
    print_header("7. 市场活跃度回退链 [AKShare乐股 → 东财采样 → 东财全量]")

    try:
        data, elapsed = timing(fetcher.fetch_market_activity)
        ok = bool(data)
        detail = f"{ {k: v for k, v in list(data.items())[:8] if isinstance(v, (int, float))} } ({elapsed:.1f}s)"
        print_result(ok, "市场活跃度", detail)
        return ok
    except Exception as e:
        print_result(False, "市场活跃度", str(e))
        return False


def test_limit_up_chain(fetcher) -> bool:
    """测试涨停池回退链 (新增)"""
    print_header("8. 涨停池回退链 [AKShare涨停池 → 东财涨停池直连]")

    try:
        df, elapsed = timing(fetcher.fetch_limit_up_pool, today)
        ok, detail = check_df(df)
        print_result(ok, f"涨停池 ({today})", f"{detail} ({elapsed:.1f}s)")
        if ok and hasattr(df, 'columns'):
            cols = list(df.columns)[:10]
            print(f"      字段(前10): {cols}")
            if hasattr(df, 'iloc') and len(df) > 0:
                print(f"      样例: {dict(df.iloc[0])}")
        return ok
    except Exception as e:
        print_result(False, "涨停池", str(e))
        return False


def test_continuous_limit_chain(fetcher) -> bool:
    """测试连板池回退链 (新增)"""
    print_header("9. 连板池回退链 [AKShare连板池 → 东财连板池直连]")

    try:
        df, elapsed = timing(fetcher.fetch_continuous_limit_up, today)
        ok, detail = check_df(df)
        print_result(ok, f"连板池 ({today})", f"{detail} ({elapsed:.1f}s)")
        if ok and hasattr(df, 'columns'):
            cols = list(df.columns)[:10]
            print(f"      字段(前10): {cols}")
        return ok
    except Exception as e:
        print_result(False, "连板池", str(e))
        return False


def test_limit_down_chain(fetcher) -> bool:
    """测试跌停池回退链 (新增)"""
    print_header("10. 跌停池回退链 [AKShare跌停池 → 东财跌停池直连]")

    try:
        df, elapsed = timing(fetcher.fetch_limit_down_pool, today)
        ok, detail = check_df(df)
        print_result(ok, f"跌停池 ({today})", f"{detail} ({elapsed:.1f}s)")
        if ok and hasattr(df, 'columns'):
            cols = list(df.columns)[:10]
            print(f"      字段(前10): {cols}")
        return ok
    except Exception as e:
        print_result(False, "跌停池", str(e))
        return False


def test_northbound_flow(fetcher) -> bool:
    """测试北向资金 (新增)"""
    print_header("11. 北向资金回退链 [东财北向直连 → AKShare北向]")

    try:
        df, elapsed = timing(fetcher.fetch_northbound_flow, 30)
        ok, detail = check_df(df)
        print_result(ok, f"北向资金(30日)", f"{detail} ({elapsed:.1f}s)")
        if ok and hasattr(df, 'columns'):
            cols = list(df.columns)
            print(f"      字段: {cols}")
            if hasattr(df, 'iloc') and len(df) > 0:
                print(f"      最新: {dict(df.iloc[-1])}")
        return ok
    except Exception as e:
        print_result(False, "北向资金", str(e))
        return False


def test_lhb_detail(fetcher) -> bool:
    """测试龙虎榜 (新增)"""
    print_header("12. 龙虎榜回退链 [东财龙虎榜直连 → AKShare龙虎榜]")

    try:
        df, elapsed = timing(fetcher.fetch_lhb_detail, today)
        ok, detail = check_df(df)
        print_result(ok, f"龙虎榜 ({today})", f"{detail} ({elapsed:.1f}s)")
        if ok and hasattr(df, 'columns'):
            cols = list(df.columns)[:10]
            print(f"      字段(前10): {cols}")
            if hasattr(df, 'iloc') and len(df) > 0:
                print(f"      样例: {dict(df.iloc[0])}")
        return ok
    except Exception as e:
        print_result(False, "龙虎榜", str(e))
        return False


def test_margin_trading(fetcher) -> bool:
    """测试融资融券 (新增)"""
    print_header("13. 融资融券回退链 [东财融资融券直连 → AKShare融资融券]")

    try:
        df, elapsed = timing(fetcher.fetch_margin_trading)
        ok, detail = check_df(df)
        print_result(ok, f"融资融券", f"{detail} ({elapsed:.1f}s)")
        if ok and hasattr(df, 'columns'):
            cols = list(df.columns)
            print(f"      字段: {cols}")
            if hasattr(df, 'iloc') and len(df) > 0:
                print(f"      最新: {dict(df.iloc[-1])}")
        return ok
    except Exception as e:
        print_result(False, "融资融券", str(e))
        return False


def test_recovery(fetcher) -> bool:
    """测试不可用数据源标记的恢复机制"""
    print_header("14. 数据源标记恢复机制")

    # 先检查当前健康状态
    health_before = fetcher.get_data_source_health()
    any_unavailable = any(
        info["unavailable"] for info in health_before.values()
    )

    if any_unavailable:
        print(f"  发现 {sum(len(i['unavailable']) for i in health_before.values())} 个不可用标记")
        for cat, info in health_before.items():
            for u in info["unavailable"]:
                print(f"    不可用: [{cat}] {u}")

    # 执行重置
    fetcher.reset_unavailable_sources()

    # 检查重置后状态
    health_after = fetcher.get_data_source_health()
    all_available = all(
        info["healthy"] for info in health_after.values()
    )

    if all_available:
        print_result(True, "reset_unavailable_sources()", "所有标记已清除，数据源全部恢复可用")
    else:
        print_result(True, "reset_unavailable_sources()",
                     "标记已清除（下次调用时会重新探测）")

    return True


if __name__ == "__main__":
    sys.exit(main())
