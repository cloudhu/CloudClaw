"""
CloudKnight - 云侠量化交易系统 v2

基于 AKQuant 高性能事件驱动回测引擎。
策略: 龙头战法 | 麻雀战法 | 海龟战法 | 价值投资

用法:
    python main.py                  # 进入交互模式
    python main.py <股票代码>        # 快速分析个股
    python main.py --version        # 显示版本号
    python main.py version          # 查看详细版本信息
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


if __name__ == "__main__":
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ("--version", "-v", "-V"):
            show_version(verbose=False)
        elif arg == "version":
            show_version(verbose=True)
        else:
            check_dependencies()
            from cloudknight.cli import main
            main()
    else:
        check_dependencies()
        from cloudknight.cli import main
        main()
