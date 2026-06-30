"""命令行界面 - 基于 AKQuant 的交互式操盘终端"""

import sys
import cmd
import logging
from datetime import datetime
from typing import List

import pandas as pd
from tabulate import tabulate

from .config import STRATEGIES, DEFAULT_CAPITAL, POOL_MAX_SIZE
from .data_manager import DataFetcher
from .indicators import (comprehensive_analysis, trend_score, calc_ma)
from .strategies import (DragonHeadStrategy, SparrowStrategy,
                         TurtleStrategy, ValueInvestStrategy, STRATEGY_CLASSES)
from .backtest_engine import BacktestEngine
from .paper_trader import PaperTrader
from .stock_pool import PoolManager
from .trading_calendar import TradingCalendar, TradingPhase, get_phase_label

logger = logging.getLogger(__name__)


class CloudKnightCLI(cmd.Cmd):
    intro = """
╔══════════════════════════════════════════════╗
║          云侠量化交易系统 v2                 ║
║    CloudKnight Quant Trading [AKQuant]      ║
╚══════════════════════════════════════════════╝
输入 help 查看所有命令，输入 quit 退出
"""
    prompt = "\n云侠> "

    def __init__(self):
        super().__init__()
        self.fetcher = DataFetcher()
        self.engine = BacktestEngine(capital=DEFAULT_CAPITAL)
        self.trader = PaperTrader()
        self.pool_mgr = PoolManager()
        self.current_strategy_cls = None   # 策略类（AKQ 策略不持有实例状态）
        self.stock_pool: List[str] = []
        self.calendar = TradingCalendar()
        self._live_engine = None           # 实时引擎引用（懒加载）

    def do_help(self, arg):
        print("""
═══ 可用命令 ═══
数据查询:
  stock <代码>        查询个股日K线
  search <关键词>     搜索股票
  hot                 查看今日涨停板
  index               查看大盘指数
股票池:
  pool                查看全部策略股票池概览
  pool screen [策略]  筛选刷新股票池（可选指定策略）
  pool list <策略>    查看某策略股票池排名
  pool detail <策略> <代码>  查看个股评分明细
  pool add <策略> <代码>     手动添加
  pool remove <策略> <代码>  移除个股
  pool clear <策略>   清空股票池
技术分析:
  analyze <代码>      全面技术分析报告
  trend <代码>        趋势评分
策略:
  strategy <策略名>   选择策略（dragon|sparrow|turtle|value）
  params              查看策略参数
回测:
  backtest            AKQuant 单策略回测
  compare             AKQuant 多策略对比回测
  report              显示最近回测详细报告
赛马:
  race start [日期]   运行今日赛马（或指定日期）
  race status         查看赛马排名
  race history        从年初开始历史回放赛马
  race reset          重置赛马
  race save           手动保存赛马状态
实时交易:
  live start          启动实时交易引擎
  live status         查看运行状态
  live market         查看市场快照
  live signals        查看最新信号
  live log [行数]     查看引擎日志
  live stop           停止实时引擎
系统:
  status              系统状态
  help                帮助信息
  quit                退出
""")

    # ─── 数据查询 ─────────────────────────────────────

    def do_stock(self, arg):
        code = arg.strip()
        if not code:
            return print("请提供股票代码，如: stock 000001")
        print(f"\n正在获取 {code} 的日K线数据...")
        df = self.fetcher.fetch_daily_kline(code, start_date="20250101")
        if df.empty:
            return print(f"未找到 {code} 的数据")
        print(f"\n{code} 最近10个交易日:")
        cols = ["日期", "开盘", "收盘", "最高", "最低", "成交量", "涨跌幅", "换手率"]
        available = [c for c in cols if c in df.columns]
        display = df[available].tail(10).copy()
        display = display.iloc[::-1]
        print(tabulate(display, headers="keys", tablefmt="grid", floatfmt=".2f"))

    def do_search(self, arg):
        keyword = arg.strip()
        if not keyword:
            return print("请输入搜索关键词")
        df = self.fetcher.fetch_stock_list()
        if df.empty:
            return print("获取股票列表失败")
        mask = df["股票简称"].str.contains(keyword, na=False) | df["股票代码"].str.contains(keyword, na=False)
        results = df[mask]
        if results.empty:
            return print(f"未找到包含 '{keyword}' 的股票")
        print(f"\n搜索结果 ({len(results)}条):")
        print(tabulate(results.head(20), headers="keys", tablefmt="grid"))

    def do_hot(self, arg):
        print("\n正在获取涨停板数据...")
        limit_up = self.fetcher.fetch_limit_up_pool(datetime.now().strftime("%Y%m%d"))
        if not limit_up.empty:
            cols = [c for c in ["代码", "名称", "涨跌幅", "最新价", "连板数", "所属行业"] if c in limit_up.columns]
            print(f"\n今日涨停板 ({len(limit_up)}只):")
            print(tabulate(limit_up[cols].head(30), headers="keys", tablefmt="grid"))
        else:
            print("无法获取涨停数据（可能非交易时间）")

    def do_index(self, arg):
        print("\n正在获取大盘指数...")
        codes = {"sh": "000001", "sz": "399001", "cy": "399006"}
        labels = {"sh": "上证指数", "sz": "深证成指", "cy": "创业板指"}
        for key, code in codes.items():
            df = self.fetcher.fetch_index_daily(code, start_date=datetime.now().replace(day=1).strftime("%Y%m%d"))
            if not df.empty:
                latest = df.iloc[-1]
                close = latest.get("close", latest.get("收盘", 0))
                open_p = latest.get("open", latest.get("开盘", 0))
                pct = (close / open_p - 1) * 100 if open_p > 0 else 0
                print(f"  {labels[key]}: {close:.2f}  {pct:+.2f}%")

    # ─── 技术分析 ─────────────────────────────────────

    def do_analyze(self, arg):
        code = arg.strip()
        if not code:
            return print("请提供股票代码")
        print(f"\n正在分析 {code}...")
        df = self.fetcher.fetch_daily_kline(code, start_date="20240101")
        if df.empty or len(df) < 60:
            return print(f"{code} 数据不足，至少需要60个交易日")
        col_map = {"日期": "date", "开盘": "open", "收盘": "close", "最高": "high",
                   "最低": "low", "成交量": "volume", "涨跌幅": "pct_change", "换手率": "turnover"}
        df_std = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        results = comprehensive_analysis(df_std)
        score = trend_score(df_std)
        print(f"\n═══ {code} 技术分析报告 ═══")
        print(f"最新收盘价: {df_std['close'].iloc[-1]:.2f}")
        print(f"趋势评分: {score['score']} / 100  [{score['rating']}]\n")
        table_data = [[name, r.signal.upper(), r.trend, r.strength, str(r.details)] for name, r in results.items()]
        print(tabulate(table_data, headers=["指标", "信号", "趋势", "强度", "详情"], tablefmt="grid"))
        df_std = calc_ma(df_std, [5, 10, 20, 60])
        latest = df_std.iloc[-1]
        ma_data = [[f"MA{p}", f"{latest[f'MA{p}']:.2f}", "上方" if latest["close"] > latest[f"MA{p}"] else "下方"]
                   for p in [5, 10, 20, 60] if pd.notna(latest.get(f"MA{p}"))]
        if ma_data:
            print("\n均线系统:")
            print(tabulate(ma_data, headers=["均线", "价格", "位置"], tablefmt="simple"))

    def do_trend(self, arg):
        code = arg.strip()
        if not code:
            return print("请提供股票代码")
        df = self.fetcher.fetch_daily_kline(code, start_date="20240101")
        if df.empty:
            return print(f"未找到 {code} 的数据")
        col_map = {"日期": "date", "开盘": "open", "收盘": "close", "最高": "high",
                   "最低": "low", "成交量": "volume", "涨跌幅": "pct_change", "换手率": "turnover"}
        df_std = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        score = trend_score(df_std)
        print(f"\n{code} 趋势评分: {score['score']} / 100")
        print(f"评级: {score['rating']}")
        print(f"各指标信号: {score['details']}")

    # ─── 策略 ─────────────────────────────────────────

    def do_strategy(self, arg):
        name = arg.strip().lower()
        cls = STRATEGY_CLASSES.get(name)
        if cls is None:
            return print(f"未知策略: {name}\n可用: dragon, sparrow, turtle, value")
        self.current_strategy_cls = cls
        inst = cls()
        print(f"\n已选择策略: {inst.description}")
        print(f"参数: {inst.get_params_info()}")
        print(f"预热期: {inst.warmup_period} 根K线")
        print(f"引擎: AKQuant 高性能事件驱动回测")

    def do_params(self, arg):
        if self.current_strategy_cls is None:
            return print("请先选择策略: strategy <策略名>")
        inst = self.current_strategy_cls()
        print(f"\n{inst.description}")
        for k, v in inst.get_params_info().items():
            print(f"  {k}: {v}")
        print(f"\n预热期: {inst.warmup_period} 根K线")

    # ─── 回测 ─────────────────────────────────────────

    def do_backtest(self, arg):
        if self.current_strategy_cls is None:
            return print("请先选择策略: strategy <策略名>")
        if not self.stock_pool:
            return print("请先构建股票池: pool screen")

        start, end = "20240101", datetime.now().strftime("%Y%m%d")
        inst = self.current_strategy_cls()
        print(f"\n[AKQuant] 开始回测: {inst.description}")
        print(f"期间: {start} ~ {end}, 股票池: {len(self.stock_pool)} 只")
        print(f"初始资金: {DEFAULT_CAPITAL:,.0f}, 佣金: 0.03%, 印花税: 0.1%")
        print(f"T+1 交易, 1手=100股\n")

        result = self.engine.run_backtest(
            self.current_strategy_cls, start, end, self.stock_pool[:30], verbose=True
        )
        if result:
            self._last_result = result
            print(f"\n═══ {result.strategy_name} 回测结果 ═══")
            print(f"总收益率: {result.total_return:+.2f}%")
            print(f"年化收益: {result.annual_return:+.2f}%")
            print(f"最大回撤: {result.max_drawdown:.2f}%")
            print(f"夏普比率: {result.sharpe_ratio}")
            print(f"胜率: {result.win_rate:.1f}%")
            print(f"交易次数: {result.total_trades} (盈利: {result.profit_trades})")
            print(f"盈亏比: {result.profit_factor}")
            if result.avg_profit:
                print(f"均盈/均亏: {result.avg_profit:.0f} / {result.avg_loss:.0f}")
            if result.trades:
                trade_data = [[t.get("date", ""), t.get("code", ""), t.get("action", ""),
                              f"{t.get('price', 0):.2f}", t.get("volume", 0),
                              f"{t.get('pnl', 0):.0f}"] for t in result.trades[-10:]]
                print(f"\n最近交易记录:")
                print(tabulate(trade_data, headers=["日期", "代码", "方向", "价格", "数量", "盈亏"], tablefmt="grid"))
        else:
            print("\n回测失败，请检查数据源")

    def do_compare(self, arg):
        if not self.stock_pool:
            return print("请先构建股票池: pool screen")

        print(f"\n[AKQuant] 多策略对比回测...")
        strategy_classes = [DragonHeadStrategy, SparrowStrategy, TurtleStrategy, ValueInvestStrategy]
        start, end = "20240101", datetime.now().strftime("%Y%m%d")
        print(f"期间: {start} ~ {end}, 每策略 {DEFAULT_CAPITAL:,.0f}")

        results = self.engine.compare_strategies(strategy_classes, start, end, self.stock_pool[:30], verbose=True)
        if results:
            print(f"\n═══ 策略对比 ═══")
            table_data = [[r.strategy_name, f"{r.total_return:+.2f}%", f"{r.annual_return:+.2f}%",
                           f"{r.max_drawdown:.2f}%", f"{r.sharpe_ratio:.2f}", f"{r.win_rate:.1f}%",
                           r.total_trades, f"{r.profit_factor:.2f}"] for r in results]
            print(tabulate(table_data, headers=["策略", "总收益", "年化", "最大回撤", "夏普", "胜率", "交易", "盈亏比"], tablefmt="grid"))
            best = max(results, key=lambda r: r.sharpe_ratio)
            print(f"\n[推荐] {best.strategy_name} (夏普: {best.sharpe_ratio})")
            self._last_compare_results = results

    def do_report(self, arg):
        """显示最近一次回测的详细报告"""
        result = getattr(self, "_last_result", None)
        if result is None:
            return print("请先执行回测: backtest 或 compare")

        print(f"\n═══ 详细回测报告: {result.strategy_name} ═══")
        print(f"期间: {result.start_date} ~ {result.end_date}")
        print(f"初始资金: {result.initial_capital:,.0f}")
        print(f"最终资金: {result.final_capital:,.0f}")
        print(f"总收益: {result.total_return:+.2f}%")
        print(f"年化收益: {result.annual_return:+.2f}%")
        print(f"最大回撤: {result.max_drawdown:.2f}%")
        print(f"夏普比率: {result.sharpe_ratio}")
        print(f"胜率: {result.win_rate:.1f}%")
        print(f"交易次数: {result.total_trades} (盈利: {result.profit_trades})")
        if result.avg_profit:
            print(f"平均盈利: {result.avg_profit:,.0f}")
        if result.avg_loss:
            print(f"平均亏损: {result.avg_loss:,.0f}")
        print(f"盈亏比: {result.profit_factor}")

        # AKQuant 原始报告
        if result.raw_result and hasattr(result.raw_result, "report"):
            try:
                print("\n─── AKQuant 引擎详细报告 ───")
                report_text = result.raw_result.report()
                print(report_text[:2000])
            except Exception:
                pass

    # ─── 赛马 ─────────────────────────────────────────

    def do_race(self, arg):
        """赛马系统 - AKQuant 多策略模拟盘竞技"""
        args = arg.strip().split()
        cmd = args[0].lower() if args else "status"

        if cmd == "start":
            date_str = args[1] if len(args) > 1 else None
            if date_str:
                if len(date_str) != 8 or not date_str.isdigit():
                    return print("日期格式错误，应为 YYYYMMDD，如 20250630")
                print(f"\n开始赛马 (指定日期: {date_str})...")
            else:
                print(f"\n开始今日赛马 ({datetime.now().strftime('%Y-%m-%d')})...")

            if not self.stock_pool:
                return print("请先构建股票池: pool screen")

            self.trader.set_stock_pool(self.stock_pool)
            results = self.trader.run_daily(date_str, verbose=True)
            if results:
                self._show_race_rankings()
            else:
                print("赛马执行异常")

        elif cmd == "status":
            self._show_race_rankings()

        elif cmd == "history":
            if not self.stock_pool:
                return print("请先构建股票池: pool screen")
            start = args[1] if len(args) > 1 else f"{datetime.now().year}0101"
            end = args[2] if len(args) > 2 else datetime.now().strftime("%Y%m%d")
            print(f"\n[AKQuant] 赛马回放: {start} ~ {end}")
            print(f"股票池: {len(self.stock_pool)} 只")
            rankings = self.trader.run_history(start, end, self.stock_pool[:50], verbose=True)
            print(f"\n═══ 赛马最终排名 ═══")
            self._print_rankings(rankings)

        elif cmd == "reset":
            self.trader.reset()
            print("\n赛马已重置")

        elif cmd == "save":
            self.trader.save()
            print("\n赛马状态已保存")

        else:
            print(f"用法: race start|status|history|reset|save")

    def _show_race_rankings(self):
        rankings = self.trader.get_rankings()
        snapshot_count = max((len(a.daily_snapshots) for a in self.trader.accounts.values()), default=0)
        print(f"\n═══ 赛马排名 (第 {snapshot_count} 天) ═══")
        self._print_rankings(rankings)

    def _print_rankings(self, rankings):
        table_data = []
        for r in rankings:
            badge = {1: "[1]", 2: "[2]", 3: "[3]"}.get(r.rank, f" {r.rank} ")
            table_data.append([badge, r.strategy_label, f"{r.total_return:+.2f}%",
                              f"{r.total_equity:,.0f}", f"{r.daily_return:+.2f}%" if r.daily_return is not None else "-",
                              f"{r.max_drawdown:.2f}%", r.position_count, r.trades])
        print(tabulate(table_data, headers=["排名", "策略", "总收益", "总权益",
                                            "日收益", "最大回撤", "持仓", "交易"],
                       tablefmt="grid", stralign="center"))

    # ─── 股票池 (保持不变) ────────────────────────────

    def do_pool(self, arg):
        """股票池管理"""
        args = arg.strip().split()
        if not args:
            return self._pool_overview()

        cmd = args[0].lower()

        if cmd == "screen":
            strategy = args[1] if len(args) > 1 else None
            print("\n正在获取全市场股票列表...")
            codes = self.fetcher.build_stock_pool(filter_st=True, filter_new=True)
            self.stock_pool = codes
            print(f"基础股票池: {len(codes)} 只")

            if strategy:
                print(f"\n开始筛选 [{STRATEGIES.get(strategy, strategy)}] 股票池...")
                self.pool_mgr.screen_one(strategy, codes, verbose=True)
                self._show_pool_list(strategy)
            else:
                print(f"\n开始为四种策略筛选独立股票池...")
                results = self.pool_mgr.screen_all(codes, verbose=True)
                print(f"\n═══ 筛选完成 ═══")
                for key, items in results.items():
                    print(f"  {STRATEGIES.get(key, key)}: {len(items)} 只 (最高评分: {items[0].score if items else 0})")

        elif cmd == "list":
            if len(args) < 2:
                return print("请指定策略: pool list dragon|sparrow|turtle|value")
            self._show_pool_list(args[1])

        elif cmd == "detail":
            if len(args) < 3:
                return print("用法: pool detail <策略名> <股票代码>")
            strategy, code = args[1], args[2]
            pool = self.pool_mgr.get_pool(strategy)
            if pool is None:
                return print(f"未知策略: {strategy}")
            item = pool.get(code)
            if item is None:
                return print(f"{code} 不在 {pool.strategy_label} 的股票池中")
            self._show_pool_detail(item, pool.strategy_label)

        elif cmd == "add":
            if len(args) < 3:
                return print("用法: pool add <策略名> <股票代码>")
            strategy, code = args[1], args[2]
            pool = self.pool_mgr.get_pool(strategy)
            if pool is None:
                return print(f"未知策略: {strategy}")
            df = self.fetcher.fetch_daily_kline(code, start_date="20240101")
            if df.empty:
                return print(f"未找到 {code} 的数据")
            df_std = self._normalize_df(df)
            name = self._get_stock_name(code)
            item = pool.scorer.score(code, name, df_std, {})
            pool.items[code] = item
            pool.save()
            print(f"\n已添加 {code} {name} 到 {pool.strategy_label} 池 (评分: {item.score})")

        elif cmd == "remove":
            if len(args) < 3:
                return print("用法: pool remove <策略名> <股票代码>")
            strategy, code = args[1], args[2]
            pool = self.pool_mgr.get_pool(strategy)
            if pool is None:
                return print(f"未知策略: {strategy}")
            if pool.remove(code):
                pool.save()
                print(f"\n已从 {pool.strategy_label} 池移除 {code}")
            else:
                print(f"\n{code} 不在池中")

        elif cmd == "clear":
            if len(args) < 2:
                return print("用法: pool clear <策略名>  或  pool clear all")
            if args[1] == "all":
                for p in self.pool_mgr.pools.values():
                    p.clear()
                    p.save()
                print("\n所有股票池已清空")
            else:
                pool = self.pool_mgr.get_pool(args[1])
                if pool is None:
                    return print(f"未知策略: {args[1]}")
                pool.clear()
                pool.save()
                print(f"\n{pool.strategy_label} 池已清空")
        else:
            print(f"未知子命令: {cmd}")

    def _pool_overview(self):
        print(f"\n═══ 自选股票池概览 ═══")
        print(f"(每种策略独立筛选评分，池容量 {POOL_MAX_SIZE} 只)\n")
        table_data = []
        for s in self.pool_mgr.overview():
            top_codes = ", ".join([f"{t['code']}({t['score']})" for t in s['top5']])
            table_data.append([
                s["strategy"], s["total"],
                s["avg_score"],
                s["last_screened"] or "未筛选",
                top_codes[:60]
            ])
        print(tabulate(table_data,
                       headers=["策略", "数量", "均分", "最近筛选", "Top5 代码(评分)"],
                       tablefmt="grid"))

    def _show_pool_list(self, strategy: str):
        pool = self.pool_mgr.get_pool(strategy)
        if pool is None:
            return print(f"未知策略: {strategy}\n可用: dragon, sparrow, turtle, value")
        ranked = pool.ranked()
        if not ranked:
            return print(f"\n{pool.strategy_label} 股票池为空，请先执行: pool screen {strategy}")
        print(f"\n═══ {pool.strategy_label} 股票池 (最近筛选: {pool.last_screened or '未筛选'}) ═══")
        table_data = []
        for item in ranked:
            comp_str = " | ".join([f"{k}:{v}" for k, v in item.components.items()])
            table_data.append([item.code, item.name, item.score, comp_str[:80]])
        print(tabulate(table_data, headers=["代码", "名称", "评分", "评分明细"], tablefmt="grid"))

    def _show_pool_detail(self, item, label: str):
        print(f"\n═══ {label} - {item.code} {item.name} ═══")
        print(f"综合评分: {item.score} / 100")
        print(f"入池时间: {item.screened_at}")
        print(f"状态: {item.status}")
        print(f"\n评分明细:")
        bar_data = []
        for k, v in item.components.items():
            bar_len = int(v / 100 * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            bar_data.append([k, v, bar])
        print(tabulate(bar_data, headers=["维度", "得分", "分布"], tablefmt="simple"))

    def _normalize_df(self, df):
        col_map = {"日期": "date", "开盘": "open", "收盘": "close", "最高": "high",
                   "最低": "low", "成交量": "volume", "成交额": "amount",
                   "涨跌幅": "pct_change", "换手率": "turnover"}
        return df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    def _get_stock_name(self, code: str) -> str:
        try:
            df = self.fetcher.fetch_stock_list()
            row = df[df["股票代码"].astype(str) == str(code)]
            if not row.empty:
                return str(row.iloc[0]["股票简称"])
        except Exception:
            pass
        return code

    # ─── 实时交易 ────────────────────────────────────

    def do_live(self, arg):
        """实时交易引擎控制"""
        args = arg.strip().split()
        cmd = args[0].lower() if args else "help"

        if cmd == "start":
            self._live_start()
        elif cmd == "status":
            self._live_status()
        elif cmd == "market":
            self._live_market()
        elif cmd == "signals":
            self._live_signals()
        elif cmd == "log":
            n = int(args[1]) if len(args) > 1 and args[1].isdigit() else 20
            self._live_log(n)
        elif cmd == "stop":
            self._live_stop()
        else:
            print("\n实时交易引擎命令:")
            print("  live start    - 启动实时交易引擎")
            print("  live status   - 查看运行状态")
            print("  live market   - 查看市场快照")
            print("  live signals  - 查看最新交易信号")
            print("  live log [N]  - 查看最近 N 条引擎日志")
            print("  live stop     - 停止实时引擎")

    def _live_start(self):
        """启动实时交易引擎"""
        if self._live_engine is not None:
            eng = self._live_engine
            if eng["engine"].state.value in ("running", "starting"):
                return print("\n实时引擎已在运行中！")

        from .live_engine import LiveTradingEngine
        import threading

        engine = LiveTradingEngine()
        engine.pool_mgr = self.pool_mgr  # 共享股票池

        cal = TradingCalendar()
        today = cal.get_current_phase()
        is_trade = cal.is_trading_day()

        print(f"\n═══ 启动实时交易引擎 ═══")
        print(f"当前阶段: {get_phase_label(today)}")
        print(f"交易日: {'是' if is_trade else '否'}")

        # 启动引擎
        engine.start()

        # 保存引用
        self._live_engine = {"engine": engine, "calendar": cal}

        # 后台日志输出线程
        def _log_printer():
            last_idx = 0
            while engine.state.value not in ("stopped", "stopping"):
                logs = engine.get_recent_logs(50)
                for i in range(last_idx, len(logs)):
                    entry = logs[i]
                    phase = get_phase_label(entry.phase)
                    ts = entry.timestamp.strftime("%H:%M:%S")
                    print(f"  [{ts}] [{phase}] {entry.event}")
                last_idx = max(last_idx, len(logs))
                import time
                time.sleep(2)

        threading.Thread(target=_log_printer, daemon=True).start()

    def _live_status(self):
        """查看实时引擎状态"""
        if self._live_engine is None:
            return print("\n实时引擎未启动。使用 live start 启动")

        eng = self._live_engine["engine"]
        status = eng.get_status()

        print(f"\n═══ 实时引擎状态 ═══")
        print(f"运行状态: {status['state']}")
        print(f"当前阶段: {status['phase_label']}")
        print(f"交易日: {'是' if status['is_trading_day'] else '否'}")
        print(f"日志条数: {status['log_count']}")
        print(f"今日决策: {status['decisions_today']} 条")

        # 信号详情
        sd = status.get("signal_details", {})
        if sd:
            print(f"\n最新扫描信号 ({status['latest_signals']} 条):")
            for name, info in sd.items():
                print(f"  [{info['strategy']}] "
                      f"买{info['buy']} 卖{info['sell']} "
                      f"({info['count']}条, {info['duration']}s)")

    def _live_market(self):
        """查看市场快照"""
        from .market_analyzer import MarketAnalyzer

        print("\n正在获取市场全景数据...")
        analyzer = MarketAnalyzer()
        sentiment = analyzer.get_market_sentiment()

        print(f"\n═══ 市场情绪 ═══")
        print(f"情绪: {sentiment['sentiment'].upper()}")
        print(f"评分: {sentiment['score']}/100")

        if "indices" in sentiment:
            print(f"\n核心指数:")
            status = sentiment["indices"]
            for code, info in status.items():
                print(f"  {info['name']}: {info['pct']:+.2f}% "
                      f"趋势{info['trend']} MACD:{info['macd']} KDJ:{info['kdj']}")

        if "factors" in sentiment:
            print(f"\n情绪因子:")
            for k, v in sentiment["factors"].items():
                bar = "█" * (v // 5) + "░" * (20 - v // 5)
                print(f"  {k}: {bar} {v}")

    def _live_signals(self):
        """查看最新交易信号"""
        if self._live_engine is None:
            return print("\n实时引擎未启动")

        eng = self._live_engine["engine"]
        status = eng.get_status()
        sd = status.get("signal_details", {})

        if not sd:
            return print("\n暂无信号数据")

        print(f"\n═══ 最新交易信号 ({status['latest_signals']} 条) ═══")
        plan = eng._trading_plan
        if plan and plan.signals:
            table_data = []
            for s in plan.signals:
                table_data.append([
                    s.code, s.name,
                    STRATEGIES.get(s.strategy, s.strategy),
                    s.signal_type.upper(), s.confidence,
                    f"{s.price:.2f}", s.reason[:50]
                ])
            print(tabulate(table_data,
                           headers=["代码", "名称", "策略", "方向", "置信度", "价格", "原因"],
                           tablefmt="grid"))
        else:
            print("无最新交易信号")

    def _live_log(self, n: int = 20):
        """查看引擎日志"""
        if self._live_engine is None:
            return print("\n实时引擎未启动")

        eng = self._live_engine["engine"]
        logs = eng.get_recent_logs(n)

        print(f"\n═══ 引擎日志 (最近 {len(logs)} 条) ═══")
        for entry in logs:
            phase = get_phase_label(entry.phase)
            ts = entry.timestamp.strftime("%H:%M:%S")
            print(f"  [{ts}] [{phase}] {entry.event}")

    def _live_stop(self):
        """停止实时引擎"""
        if self._live_engine is None:
            return print("\n实时引擎未启动")

        eng = self._live_engine["engine"]
        eng.stop()
        self._live_engine = None
        print("\n实时引擎已停止")

    # ─── 系统 ─────────────────────────────────────────

    def do_status(self, arg):
        from .version import get_version
        from datetime import datetime
        print(f"\n═══ 系统状态 ═══")
        print(f"版本: v{get_version()} | 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        cls = self.current_strategy_cls
        inst = cls() if cls else None
        print(f"策略: {inst.description if inst else '未选择'}")
        print(f"引擎: AKQuant 高性能事件驱动回测")

        # 交易日历状态
        phase = self.calendar.get_current_phase()
        is_trade = self.calendar.is_trading_day()
        print(f"交易日: {'是' if is_trade else '否'} ({get_phase_label(phase)})")

        # 实时引擎状态
        if self._live_engine is not None:
            live_state = self._live_engine["engine"].state.value
            print(f"实时引擎: {live_state}")

        print(f"股票池: 基础{len(self.stock_pool)}只 | 资金: {DEFAULT_CAPITAL:,.0f} | 数据源: akshare")

    def do_clear(self, arg):
        print("\n" * 50)

    def do_quit(self, arg):
        if self._live_engine is not None:
            self._live_stop()
        print("\n感谢使用云侠量化交易系统！")
        return True

    def do_exit(self, arg):
        return self.do_quit(arg)

    def default(self, line):
        print(f"未知命令: {line}\n输入 help 查看可用命令")


def main():
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    if len(sys.argv) > 1:
        code = sys.argv[1]
        fetcher = DataFetcher()
        df = fetcher.fetch_daily_kline(code, start_date="20240101")
        if df.empty:
            return print(f"未找到 {code} 的数据")
        col_map = {"日期": "date", "开盘": "open", "收盘": "close", "最高": "high",
                   "最低": "low", "成交量": "volume", "涨跌幅": "pct_change", "换手率": "turnover"}
        df_std = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        results = comprehensive_analysis(df_std)
        score = trend_score(df_std)
        print(f"\n═══ {code} 技术分析 ═══")
        print(f"最新价: {df_std['close'].iloc[-1]:.2f}")
        print(f"趋势评分: {score['score']}/100 [{score['rating']}]")
        for name, r in results.items():
            print(f"  {name}: {r.signal.upper()} (强度:{r.strength})")
        return
    try:
        CloudKnightCLI().cmdloop()
    except KeyboardInterrupt:
        print("\n\n感谢使用！")


if __name__ == "__main__":
    main()
