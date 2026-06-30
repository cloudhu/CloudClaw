"""命令行界面 - 交互式操盘终端"""

import sys
import cmd
import logging
from datetime import datetime
from typing import List

import pandas as pd
from tabulate import tabulate

from .config import STRATEGIES, DEFAULT_CAPITAL
from .data_manager import DataFetcher
from .indicators import (comprehensive_analysis, trend_score, calc_ma)
from .strategies import (DragonHeadStrategy, SparrowStrategy,
                         TurtleStrategy, ValueInvestStrategy)
from .backtest_engine import BacktestEngine

logger = logging.getLogger(__name__)

STRATEGY_MAP = {"dragon": DragonHeadStrategy, "sparrow": SparrowStrategy,
                "turtle": TurtleStrategy, "value": ValueInvestStrategy}


class CloudClawCLI(cmd.Cmd):
    intro = """
╔══════════════════════════════════════════════╗
║           🦞 CloudClaw 量化交易系统           ║
║          小而精的个人A股量化交易软件          ║
╚══════════════════════════════════════════════╝
输入 help 查看所有命令，输入 quit 退出
"""
    prompt = "\nCloudClaw> "

    def __init__(self):
        super().__init__()
        self.fetcher = DataFetcher()
        self.engine = BacktestEngine(capital=DEFAULT_CAPITAL)
        self.current_strategy = None
        self.stock_pool: List[str] = []

    def do_help(self, arg):
        print("""
═══ 可用命令 ═══
数据查询:
  stock <代码>        查询个股日K线
  search <关键词>     搜索股票
  hot                 查看今日涨停板
  index               查看大盘指数
  pool                构建股票池
技术分析:
  analyze <代码>      全面技术分析报告
  trend <代码>        趋势评分
策略:
  strategy <策略名>   选择策略
  signals <代码>      查看交易信号
  params              查看策略参数
回测:
  backtest            策略回测
  compare             多策略对比
系统:
  status              系统状态
  help                帮助信息
  quit                退出
""")

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

    def do_pool(self, arg):
        codes = self.fetcher.build_stock_pool(filter_st=True, filter_new=True)
        self.stock_pool = codes
        print(f"\n股票池已构建: {len(codes)} 只股票")
        print(f"前10只: {self.stock_pool[:10]}")

    def do_strategy(self, arg):
        name = arg.strip().lower()
        if name not in STRATEGY_MAP:
            return print(f"未知策略: {name}\n可用策略: {', '.join(STRATEGY_MAP.keys())}")
        self.current_strategy = STRATEGY_MAP[name]()
        print(f"\n已选择策略: {self.current_strategy.description}")
        print(f"参数: {self.current_strategy.get_params_info()}")

    def do_signals(self, arg):
        if self.current_strategy is None:
            return print("请先选择策略: strategy <策略名>")
        code = arg.strip()
        if not code:
            return print("请提供股票代码")
        df = self.fetcher.fetch_daily_kline(code, start_date="20240101")
        if df.empty:
            return print(f"未找到 {code} 的数据")
        col_map = {"日期": "date", "开盘": "open", "收盘": "close", "最高": "high",
                   "最低": "low", "成交量": "volume", "涨跌幅": "pct_change", "换手率": "turnover"}
        df_std = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        signals = self.current_strategy.generate_signals(code, df_std, None)
        print(f"\n{code} 交易信号 ({self.current_strategy.description}):")
        for s in signals:
            print(f"  [{s.signal_type.value.upper()}] {s.reason}")
            if s.signal_type.value in ("buy", "add"):
                print(f"    建议仓位: {s.volume_pct*100:.0f}%, 止损: {s.stop_loss_price:.2f}, 止盈: {s.stop_profit_price:.2f}")
            print(f"    置信度: {s.confidence:.0f}%")

    def do_backtest(self, arg):
        if self.current_strategy is None:
            return print("请先选择策略: strategy <策略名>")
        if not self.stock_pool:
            return print("请先构建股票池: pool")
        start, end = "20240101", datetime.now().strftime("%Y%m%d")
        print(f"\n开始回测: {self.current_strategy.description}")
        print(f"期间: {start} ~ {end}, 股票池: {len(self.stock_pool)} 只")
        result = self.engine.run_backtest(self.current_strategy, start, end, self.stock_pool[:30], verbose=True)
        if result:
            print(f"\n═══ 回测结果 ═══")
            print(f"总收益率: {result.total_return:+.2f}%, 年化: {result.annual_return:+.2f}%")
            print(f"最大回撤: {result.max_drawdown:.2f}%, 夏普: {result.sharpe_ratio}")
            print(f"胜率: {result.win_rate:.1f}%, 交易: {result.total_trades}次, 盈亏比: {result.profit_factor}")
            if result.trades:
                trade_data = [[t.date, t.code, t.action, f"{t.price:.2f}", t.volume, t.reason[:30]] for t in result.trades[-10:]]
                print(f"\n最近交易记录:")
                print(tabulate(trade_data, headers=["日期", "代码", "方向", "价格", "数量", "原因"], tablefmt="grid"))

    def do_compare(self, arg):
        if not self.stock_pool:
            return print("请先构建股票池: pool")
        print("\n多策略对比回测...")
        strategies = [DragonHeadStrategy(), SparrowStrategy(), TurtleStrategy(), ValueInvestStrategy()]
        start, end = "20240101", datetime.now().strftime("%Y%m%d")
        results = self.engine.compare_strategies(strategies, start, end, self.stock_pool[:30], verbose=True)
        if results:
            print(f"\n═══ 策略对比 ═══")
            table_data = [[r.strategy_name, f"{r.total_return:+.2f}%", f"{r.annual_return:+.2f}%",
                           f"{r.max_drawdown:.2f}%", f"{r.sharpe_ratio:.2f}", f"{r.win_rate:.1f}%",
                           r.total_trades, f"{r.profit_factor:.2f}"] for r in results]
            print(tabulate(table_data, headers=["策略", "总收益", "年化", "最大回撤", "夏普", "胜率", "交易", "盈亏比"], tablefmt="grid"))
            best = max(results, key=lambda r: r.sharpe_ratio)
            print(f"\n🏆 推荐策略: {best.strategy_name} (夏普: {best.sharpe_ratio})")

    def do_params(self, arg):
        if self.current_strategy is None:
            return print("请先选择策略")
        print(f"\n当前策略: {self.current_strategy.description}")
        for k, v in self.current_strategy.get_params_info().items():
            print(f"  {k}: {v}")

    def do_status(self, arg):
        print(f"\n═══ 系统状态 ═══")
        print(f"版本: 1.0.0 | 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"策略: {self.current_strategy.description if self.current_strategy else '未选择'}")
        print(f"股票池: {len(self.stock_pool)} 只 | 资金: {DEFAULT_CAPITAL:,.0f} | 数据源: akshare")

    def do_clear(self, arg):
        print("\n" * 50)

    def do_quit(self, arg):
        print("\n感谢使用 CloudClaw 量化交易系统！")
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
        CloudClawCLI().cmdloop()
    except KeyboardInterrupt:
        print("\n\n感谢使用！")


if __name__ == "__main__":
    main()
