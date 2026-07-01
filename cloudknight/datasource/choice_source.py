"""
Choice 金融终端数据源适配器

东方财富 Choice 数据终端提供:
- A股/港股/美股行情数据
- 财务报表（比免费接口更精准）
- 行业分类与估值数据
- 宏观/债券/期货数据

连接方式:
1. 本地已安装 Choice 终端（通过 HTTP 桥接或本地端口）
2. 或使用 Choice 开放 API（需申请）

注意: Choice 终端是付费产品，需要本地安装并登录。
"""

import logging
import json as _json
from datetime import datetime

import pandas as pd

from .base import DataSourceAdapter, FetchResult

logger = logging.getLogger(__name__)


class ChoiceAdapter(DataSourceAdapter):
    """东方财富 Choice 终端适配器

    通过 HTTP API 桥接 Choice 终端数据:
    - 默认端口: 8089 (需在 Choice 终端设置中开启)
    - 接口: http://127.0.0.1:8089/api/

    备选: 若有 Choice 云平台 API Key，可直接使用云端接口
    """

    name = "Choice"
    requires_auth = True
    supported_markets = ["A", "HK", "US"]

    def __init__(self, api_key: str = "", api_secret: str = "", **kwargs):
        super().__init__(api_key=api_key, api_secret=api_secret, **kwargs)
        self._session = None
        self._base_url = kwargs.get("choice_url", "http://127.0.0.1:8089")
        self._use_cloud = kwargs.get("choice_use_cloud", False)
        # Choice 企业版通常无日限额
        self._quota.daily_limit = -1
        self._quota.monthly_limit = -1

    def _validate_credentials(self) -> bool:
        """通过心跳接口验证连接"""
        try:
            r = self._session.get(f"{self._base_url}/api/heartbeat", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def _do_connect(self):
        import requests as _req

        self._session = _req.Session()
        self._session.headers.update(
            {
                "User-Agent": "CloudKnight/1.0",
                "Content-Type": "application/json",
            }
        )
        if self._api_key:
            self._session.headers["X-API-Key"] = self._api_key

        if not self._validate_credentials():
            raise ConnectionError(f"Choice 终端不可达 ({self._base_url})")

    @staticmethod
    def _to_choice_code(code: str) -> str:
        """A股代码转 Choice 格式"""
        if "." in code:
            return code
        if code.startswith(("0", "3")):
            return f"{code}.SZ"
        return f"{code}.SH"

    # ── 日K线 ───────────────────────────────────────────
    def fetch_daily_kline(
        self,
        code: str,
        start_date: str = "20200101",
        end_date: str = "",
        adjust: str = "qfq",
    ) -> FetchResult:
        if not self._session:
            return FetchResult(success=False, source_name=self.name, error="未连接")

        choice_code = self._to_choice_code(code)
        if not end_date:
            end_date = datetime.now().strftime("%Y%m%d")

        try:
            params = {
                "codes": choice_code,
                "startDate": start_date,
                "endDate": end_date,
                "adjust": {"qfq": "1", "hfq": "2", "none": "0"}.get(adjust, "1"),
            }
            r = self._session.post(
                f"{self._base_url}/api/market/daily",
                json=params,
                timeout=15,
            )
            data = r.json()

            if data.get("code") != 0 or not data.get("data"):
                # 降级尝试云端 API
                if self._use_cloud and self._api_key:
                    return self._fetch_daily_cloud(code, start_date, end_date, adjust)
                return FetchResult(success=False, source_name=self.name, error=data.get("msg", "无数据"))

            items = data["data"]
            rows = []
            for row in items:
                rows.append(
                    {
                        "日期": str(row.get("date", "")).replace("-", ""),
                        "开盘": float(row.get("open", 0)),
                        "最高": float(row.get("high", 0)),
                        "最低": float(row.get("low", 0)),
                        "收盘": float(row.get("close", 0)),
                        "成交量": float(row.get("volume", 0)),
                        "成交额": float(row.get("amount", 0)),
                        "涨跌幅": float(row.get("pctChg", 0)),
                        "换手率": float(row.get("turnover", 0)),
                        "股票代码": code,
                    }
                )

            df = pd.DataFrame(rows)
            return FetchResult(
                success=True,
                data=df.sort_values("日期") if not df.empty else df,
                source_name=self.name,
            )
        except Exception as e:
            logger.warning(f"Choice 获取 {code} K线失败: {e}")
            # 云端降级
            if self._use_cloud and self._api_key:
                try:
                    return self._fetch_daily_cloud(code, start_date, end_date, adjust)
                except Exception:
                    pass
            return FetchResult(success=False, source_name=self.name, error=str(e))

    def _fetch_daily_cloud(
        self,
        code: str,
        start_date: str,
        end_date: str,
        adjust: str,
    ) -> FetchResult:
        """通过 Choice 云平台 API 获取日K线（备选）"""
        try:
            choice_code = self._to_choice_code(code)
            r = self._session.get(
                "https://api.choice.eastmoney.com/v1/market/daily",
                params={
                    "code": choice_code,
                    "start": start_date,
                    "end": end_date,
                    "adj": adjust,
                },
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=15,
            )
            data = r.json()
            if data.get("code") != 0:
                return FetchResult(success=False, source_name=self.name, error=data.get("msg", ""))

            rows = []
            for row in data.get("data", []):
                rows.append(
                    {
                        "日期": str(row.get("date", "")).replace("-", ""),
                        "开盘": float(row.get("open", 0)),
                        "最高": float(row.get("high", 0)),
                        "最低": float(row.get("low", 0)),
                        "收盘": float(row.get("close", 0)),
                        "成交量": float(row.get("volume", 0)),
                        "股票代码": code,
                    }
                )
            df = pd.DataFrame(rows)
            return FetchResult(success=True, data=df, source_name=f"{self.name}(云)")
        except Exception as e:
            return FetchResult(success=False, source_name=self.name, error=str(e))

    # ── 财务数据 ─────────────────────────────────────────
    def fetch_financial_data(self, code: str) -> FetchResult:
        if not self._session:
            return FetchResult(success=False, source_name=self.name, error="未连接")

        choice_code = self._to_choice_code(code)

        try:
            r = self._session.post(
                f"{self._base_url}/api/finance/indicator",
                json={"code": choice_code, "periods": 4},
                timeout=10,
            )
            data = r.json()

            if data.get("code") != 0 or not data.get("data"):
                return FetchResult(success=False, source_name=self.name, error=data.get("msg", "无数据"))

            latest = data["data"][0] if data["data"] else {}
            result = {
                "股票代码": code,
                "市盈率": float(latest.get("pe", 0) or 0),
                "市净率": float(latest.get("pb", 0) or 0),
                "roe": float(latest.get("roe", 0) or 0),
                "毛利率": float(latest.get("grossMargin", 0) or 0),
                "净利率": float(latest.get("netMargin", 0) or 0),
                "资产负债率": float(latest.get("debtRatio", 0) or 0),
                "净利润增长率": float(latest.get("profitYoy", 0) or 0),
                "营业收入增长率": float(latest.get("revenueYoy", 0) or 0),
                "总市值": float(latest.get("totalMv", 0) or 0),
                "流通市值": float(latest.get("circMv", 0) or 0),
                "股息率": float(latest.get("dividendYield", 0) or 0),
                "每股收益": float(latest.get("eps", 0) or 0),
            }
            return FetchResult(success=True, data=pd.DataFrame([result]), source_name=self.name)
        except Exception as e:
            logger.warning(f"Choice 获取 {code} 财务数据失败: {e}")
            return FetchResult(success=False, source_name=self.name, error=str(e))

    # ── 指数日线 ─────────────────────────────────────────
    def fetch_index_daily(
        self,
        index_code: str,
        start_date: str = "20200101",
        end_date: str = "",
    ) -> FetchResult:
        if not self._session:
            return FetchResult(success=False, source_name=self.name, error="未连接")

        if not end_date:
            end_date = datetime.now().strftime("%Y%m%d")

        try:
            r = self._session.post(
                f"{self._base_url}/api/index/daily",
                json={
                    "code": index_code,
                    "startDate": start_date,
                    "endDate": end_date,
                },
                timeout=10,
            )
            data = r.json()
            if data.get("code") != 0:
                return FetchResult(success=False, source_name=self.name, error=data.get("msg", ""))

            rows = []
            for row in data.get("data", []):
                rows.append(
                    {
                        "date": str(row.get("date", "")).replace("-", ""),
                        "open": float(row.get("open", 0)),
                        "high": float(row.get("high", 0)),
                        "low": float(row.get("low", 0)),
                        "close": float(row.get("close", 0)),
                        "volume": float(row.get("volume", 0)),
                        "amount": float(row.get("amount", 0)),
                    }
                )
            return FetchResult(success=True, data=pd.DataFrame(rows), source_name=self.name)
        except Exception as e:
            logger.warning(f"Choice 获取指数 {index_code} 失败: {e}")
            return FetchResult(success=False, source_name=self.name, error=str(e))

    # ── 股票列表 ─────────────────────────────────────────
    def fetch_stock_list(self) -> FetchResult:
        if not self._session:
            return FetchResult(success=False, source_name=self.name, error="未连接")

        try:
            r = self._session.get(
                f"{self._base_url}/api/market/stock_list",
                timeout=10,
            )
            data = r.json()
            if data.get("code") != 0:
                return FetchResult(success=False, source_name=self.name, error=data.get("msg", ""))

            rows = []
            for item in data.get("data", []):
                rows.append(
                    {
                        "股票代码": str(item.get("code", "")),
                        "股票简称": str(item.get("name", "")),
                        "行业": str(item.get("industry", "")),
                        "上市时间": str(item.get("listDate", "")).replace("-", ""),
                    }
                )
            return FetchResult(success=True, data=pd.DataFrame(rows), source_name=self.name)
        except Exception as e:
            logger.warning(f"Choice 获取股票列表失败: {e}")
            return FetchResult(success=False, source_name=self.name, error=str(e))
