"""
CloudKnight - 云侠量化交易系统 v2

基于 AKQuant 高性能事件驱动回测引擎。
策略: 龙头战法 | 麻雀战法 | 海龟战法 | 价值投资

用法:
    python main.py                      # 进入交互模式
    python main.py <股票代码>             # 快速分析个股
    python main.py live                 # 启动实时交易引擎
    python main.py --version            # 显示版本号
    python main.py version              # 查看详细版本信息
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def show_version(verbose: bool = False):
    """显示版本信息"""
    from cloudknight.version import get_version, get_semver, VERSION_STRING
    from cloudknight import __engine__
    if verbose:
        sv = get_semver()
        print(f"CloudKnight - 云侠量化交易系统")
        print(f"版本: v{VERSION_STRING}  [引擎: {__engine__}]")
        print(f"SemVer: {sv.major}.{sv.minor}.{sv.patch}")
        print(f"预发布: {sv.prerelease or '无'}")
        print(f"构建: {sv.build or '无'}")
    else:
        print(f"CloudKnight v{get_version()} [{__engine__}]")


def check_dependencies():
    """检查关键依赖"""
    missing = []
    try:
        import akquant
    except ImportError:
        missing.append("akquant (pip install akquant)")
    try:
        import akshare
    except ImportError:
        missing.append("akshare (pip install akshare)")
    if missing:
        print("⚠ 缺少依赖:")
        for m in missing:
            print(f"  - {m}")
        print()
        return False
    return True


def start_live_engine():
    """启动实时交易引擎"""
    import logging
    import signal

    # 检查依赖
    if not check_dependencies():
        print("请先安装缺失的依赖后再启动实时引擎。")
        return

    from cloudknight.live_engine import LiveTradingEngine, get_phase_label
    from cloudknight.trading_calendar import TradingCalendar, TradingPhase

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    print("""
╔══════════════════════════════════════════════╗
║     CloudKnight Live Trading Engine          ║
║     基于 AKQuant 的 A股实时交易引擎           ║
╚══════════════════════════════════════════════╝""")

    engine = LiveTradingEngine()
    calendar = TradingCalendar()

    today = calendar.get_current_phase()
    is_trade = calendar.is_trading_day()

    print(f"\n当前时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"当前阶段: {get_phase_label(today)}")
    print(f"交易日: {'是' if is_trade else '否'}")

    if is_trade:
        print(f"""
A股交易时间线:
  08:30  盘前准备 → 大盘/板块分析, 股票池筛选
  09:15  集合竞价  → 监控竞价走势
  09:26  竞价结果  → 竞价分析, 开盘决策
  09:30  早盘开盘  → 分时跟踪, 信号扫描
  11:30  午间休市  → 15分K分析, 下午计划
  13:00  午盘开盘  → 继续跟踪, 执行交易
  15:00  收盘      → 盘后选股, 次日计划
""")
    else:
        print("\n今日非交易日 → 将执行全策略选股")

    # 注册信号处理
    def handle_signal(sig, frame):
        print("\n\n收到中断信号，正在安全停止引擎...")
        engine.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)

    print("正在启动实时引擎... (Ctrl+C 停止)\n")
    engine.start()

    # 保持主线程存活
    try:
        while engine.state.value != "stopped":
            __import__('time').sleep(1)
    except KeyboardInterrupt:
        engine.stop()

    print("\n实时引擎已停止。")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ("--version", "-v", "-V"):
            show_version(verbose=False)
        elif arg == "version":
            show_version(verbose=True)
        elif arg == "live":
            start_live_engine()
        else:
            check_dependencies()
            from cloudknight.cli import main
            main()
    else:
        check_dependencies()
        from cloudknight.cli import main
        main()
