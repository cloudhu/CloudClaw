"""
CloudClaw - 小而精的个人A股量化交易软件

用法:
    python main.py                  # 进入交互模式
    python main.py 000001           # 快速分析个股
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cloudclaw.cli import main

if __name__ == "__main__":
    main()
