"""
Interactive Brokers (IBKR) 数据源适配器

通过 ib_insync 连接 TWS/IB Gateway，获取:
- 美股/港股/A股日K线及分钟线
- 实时行情 (Level 1)
- 历史数据 (reqHistoricalData)

前置条件:
  1. 安装 IB Gateway 或 TWS 并登录
  2. pip install ib_insync
  3. TWS/IB Gateway 设置中开启 API 连接 (默认端口 7497/4002)
"""

import logging
import time as _time
from datetime import datetime

import pandas as pd

from .base import DataSourceAdapter, FetchResult

logger = logging.getLogger(__name__)


class IBKRAdapter(DataSourceAdapter):
    """Interactive Brokers 数据源适配器

    连接方式:
    - TWS 实盘: host=127.0.0.1, port=7497, clientId=随机
    - IB Gateway 实盘: host=127.0.0.1, port=4002
    - TWS Paper: host=127.0.0.1, port=7497 (需切换为模拟账户)

    注意: IBKR 没有传统意义上的"API Key"，通过 IP+端口连接。
    在面板中可配置 host/port/clientId。
    """

    name = "IBKR"
    requires_auth = True
    supported_markets = ["A", "HK", "US"]

    def __init__(self, api_key: str = "", api_secret: str = "", **kwargs):
        # IBKR 用 host:port 代替 api_key，clientId 用 api_secret
        super().__init__(api_key=api_key, api_secret=api_secret, **kwargs)
        self._ib = None
        # 默认连接参数
        self._host = kwargs.get("ibkr_host", "127.0.0.1")
        self._port = int(kwargs.get("ibkr_port", 7497))
        self._client_id = int(kwargs.get("ibkr_client_id", 1))
        # 没有额度限制（IBKR 免费数据有限制但不是积分制）
        self._quota.daily_limit = -1

    def _validate_credentials(self) -> bool:
        """IBKR 通过能否成功连接来验证"""
        if self._ib is None:
            return False
        return self._ib.isConnected()

    def _do_connect(self):
        try:
            from ib_insync import IB, util
        except ImportError:
            raise ImportError("请安装 ib_insync: pip install ib_insync")

        self._ib = IB()
        try:
            self._ib.connect(
                host=self._host,
                port=self._port,
                clientId=self._client_id,
                timeout=10,
            )
            # 获取当前时间验证连接
            self._ib.reqCurrentTime()
        except Exception as e:
            self._ib = None
            raise ConnectionError(f"IBKR 连接失败 ({self._host}:{self._port}): {e}")

    def disconnect(self):
        if self._ib and self._ib.isConnected():
            self._ib.disconnect()
        self._ib = None
        super().disconnect()

    # ── 代码转换 ─────────────────────────────────────────
    @staticmethod
    def _to_ib_contract(code: str, market: str = "A"):
        """构造 IBKR Contract"""
        try:
            from ib_insync import Contract, Stock
        except ImportError:
            raise ImportError("请安装 ib_insync")

        if market == "A":
            # A股通过沪港通/深港通
            exchange = "SEHKNTL" if code.startswith(("6", "9")) else "SEHKSZSE"
            return Stock(symbol=code, exchange=exchange, currency="CNH")
        elif market == "HK":
            # 港股: 00700 → 700
            code_stripped = code.lstrip("0") or "0"
            return Stock(symbol=code_stripped, exchange="SEHK", currency="HKD")
        elif market == "US":
            return Stock(symbol=code, exchange="SMART", currency="USD")
        return None

    # ── 日K线 ───────────────────────────────────────────
    def fetch_daily_kline(
        self,
        code: str,
        start_date: str = "20200101",
        end_date: str = "",
        adjust: str = "qfq",
    ) -> FetchResult:
        if not self._ib or not self._ib.isConnected():
            return FetchResult(success=False, source_name=self.name, error="未连接")

        contract = self._to_ib_contract(code)
        if contract is None:
            return FetchResult(success=False, source_name=self.name, error="无法构造合约")

        if not end_date:
            end_date = datetime.now().strftime("%Y%m%d")

        start_dt = datetime.strptime(start_date, "%Y%m%d")
        end_dt = datetime.strptime(end_date, "%Y%m%d")

        # IBKR 单次最多 365 天，需要分批获取
        try:
            all_bars = []
            from ib_insync import util

            # 计算 duration 和 barSize
            total_days = (end_dt - start_dt).days
            if total_days <= 365:
                duration = f"{total_days + 1} D"
            else:
                duration = f"{min(total_days // 365 + 1, 10)} Y"

            bars = self._ib.reqHistoricalData(
                contract=contract,
                endDateTime=end_dt.strftime("%Y%m%d 23:59:59"),
                durationStr=duration,
                barSizeSetting="1 day",
                whatToShow=adjust.upper() if adjust != "qfq" else "ADJUSTED_LAST",
                useRTH=0,
                formatDate=1,
            )

            if not bars:
                return FetchResult(success=False, source_name=self.name, error="无数据")

            df = util.df(bars)
            if df is None or df.empty:
                return FetchResult(success=False, source_name=self.name, error="无数据")

            df_out = pd.DataFrame(
                {
                    "日期": pd.to_datetime(df["date"]).dt.strftime("%Y%m%d"),
                    "开盘": df["open"].astype(float),
                    "最高": df["high"].astype(float),
                    "最低": df["low"].astype(float),
                    "收盘": df["close"].astype(float),
                    "成交量": df["volume"].astype(float),
                    "股票代码": code,
                }
            )

            mask = (df_out["日期"] >= start_date) & (df_out["日期"] <= end_date)
            return FetchResult(
                success=True,
                data=df_out[mask].sort_values("日期").reset_index(drop=True),
                source_name=self.name,
            )
        except Exception as e:
            logger.warning(f"IBKR 获取 {code} K线失败: {e}")
            return FetchResult(success=False, source_name=self.name, error=str(e))

    # ── 财务数据（IBKR 提供基本面摘要） ──────────────────
    def fetch_financial_data(self, code: str) -> FetchResult:
        if not self._ib or not self._ib.isConnected():
            return FetchResult(success=False, source_name=self.name, error="未连接")

        try:
            from ib_insync import FundamentalRatios

            contract = self._to_ib_contract(code)
            if contract is None:
                return FetchResult(success=False, source_name=self.name, error="无法构造合约")

            # 获取基本面数据摘要
            self._ib.reqMarketDataType(1)  # 延迟数据即可

            # 通过 reqFundamentalData 获取
            try:
                xml_data = self._ib.reqFundamentalData(
                    contract=contract,
                    reportType="RESC",  # Ratios
                )
                _time.sleep(1)
            except Exception:
                return FetchResult(success=False, source_name=self.name, error="基本面数据不可用")

            if not xml_data:
                return FetchResult(success=False, source_name=self.name, error="无数据")

            # 简单解析 XML（完整解析需 lxml，这里做关键字段提取）
            result = {"股票代码": code}
            for field, tag in [
                ("市盈率", "P/ERATIO"),
                ("市净率", "P/BRATIO"),
                ("roe", "ROE"),
            ]:
                import re

                match = re.search(f"<{tag}[^>]*>([^<]+)</{tag}>", str(xml_data))
                if match:
                    try:
                        result[field] = float(match.group(1))
                    except ValueError:
                        result[field] = match.group(1)

            return FetchResult(success=True, data=pd.DataFrame([result]), source_name=self.name)
        except Exception as e:
            logger.warning(f"IBKR 获取 {code} 财务数据失败: {e}")
            return FetchResult(success=False, source_name=self.name, error=str(e))

    # ── 实时行情 ─────────────────────────────────────────
    def fetch_realtime_quote(self, codes: list[str]) -> FetchResult:
        if not self._ib or not self._ib.isConnected():
            return FetchResult(success=False, source_name=self.name, error="未连接")

        try:
            from ib_insync import Contract, Stock

            rows = []
            for code in codes[:10]:  # 限制批量请求
                contract = self._to_ib_contract(code)
                if contract is None:
                    continue
                try:
                    self._ib.reqMktData(contract, "", False, False)
                    self._ib.sleep(0.5)  # 等待数据返回
                    ticker = self._ib.ticker(contract)
                    if ticker:
                        rows.append(
                            {
                                "代码": code,
                                "最新价": float(ticker.last) if ticker.last else 0,
                                "今开": float(ticker.open) if ticker.open else 0,
                                "最高": float(ticker.high) if ticker.high else 0,
                                "最低": float(ticker.low) if ticker.low else 0,
                                "昨收": float(ticker.close) if ticker.close else 0,
                                "成交量": float(ticker.volume) if ticker.volume else 0,
                            }
                        )
                except Exception:
                    continue

            if rows:
                return FetchResult(success=True, data=pd.DataFrame(rows), source_name=self.name)
            return FetchResult(success=False, source_name=self.name, error="无数据")
        except Exception as e:
            logger.warning(f"IBKR 获取实时行情失败: {e}")
            return FetchResult(success=False, source_name=self.name, error=str(e))
