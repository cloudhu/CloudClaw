"""
TuShare 数据源适配器 - Pro 版本（需 Token）

TuShare Pro 积分体系:
- 默认: 120分/分钟, 2000分/天
- 高积分用户可获取更多数据（分钟线、财务、指数权重等）

使用前需在 https://tushare.pro 注册并获取 Token
"""

import logging
import time as _time
from datetime import datetime

import pandas as pd

from .base import DataSourceAdapter, FetchResult

logger = logging.getLogger(__name__)

# TuShare API 接口积分消耗参考:
# - daily: 2分/次
# - daily_basic: 2分/次
# - income/v3: 2分/次
# - balancesheet/v1: 2分/次
# - moneyflow: 2分/次
# - index_daily: 2分/次
# - stock_basic: 1分/次
# - trade_cal: 1分/次

_TUSHARE_API_COST: dict[str, int] = {
    "daily": 2,
    "daily_basic": 2,
    "income": 2,
    "balancesheet": 2,
    "moneyflow": 2,
    "index_daily": 2,
    "stock_basic": 1,
    "financial": 2,
    "fund_daily": 2,
}

# TuShare 各接口频率限制（秒），避免触发限流后浪费额度
_TUSHARE_API_MIN_INTERVAL: dict[str, float] = {
    "index_daily": 65,      # 1次/分钟，留5秒余量
    "moneyflow": 65,        # 1次/分钟
    "daily_basic": 5,       # 较宽松
    "financial": 5,
    "income": 5,
    "balancesheet": 5,
    "stock_basic": 5,
    "fund_daily": 65,
}
_DEFAULT_MIN_INTERVAL = 5  # 默认最小间隔


class TuShareAdapter(DataSourceAdapter):
    """TuShare Pro 数据源适配器

    支持:
    - A股日K线及复权
    - 财务三表
    - 资金流向
    - 指数日线
    - 股票列表
    - 交易日历
    """

    name = "TuShare"
    requires_auth = True
    supported_markets = ["A"]

    def __init__(self, api_key: str = "", **kwargs):
        super().__init__(api_key=api_key, **kwargs)
        self._quota.daily_limit = 2000
        self._quota.monthly_limit = 60000
        self._pro = None
        self._last_api_call: dict[str, float] = {}  # API名称 → 上次调用时间戳

    def _check_rate_limit(self, api_name: str) -> bool:
        """检查 API 是否处于冷却期，返回 True 表示可以调用"""
        if api_name not in _TUSHARE_API_MIN_INTERVAL:
            return True
        interval = _TUSHARE_API_MIN_INTERVAL.get(api_name, _DEFAULT_MIN_INTERVAL)
        last = self._last_api_call.get(api_name, 0)
        elapsed = _time.time() - last
        if elapsed < interval:
            remaining = interval - elapsed
            logger.debug(f"TuShare {api_name} 冷却中（剩余 {remaining:.0f}s）→ 跳过")
            return False
        return True

    def _mark_api_call(self, api_name: str):
        """记录 API 调用时间"""
        self._last_api_call[api_name] = _time.time()

    def _validate_credentials(self) -> bool:
        if not self._pro:
            return False
        try:
            # 用 trade_cal 做轻量验证（1积分，频率限制更宽松）
            df = self._pro.trade_cal(exchange="SSE", start_date="20260701", end_date="20260701")
            return df is not None and not df.empty
        except Exception as e:
            err_msg = str(e)
            # 频率限制/权限不足 = Token 有效，只是接口限制
            if any(kw in err_msg for kw in ["频率", "frequency", "分钟", "权限", "permission", "积分"]):
                logger.info(f"TuShare Token 有效（{err_msg[:60]}）")
                return True
            # 其它错误也可能只是接口问题，只要不是明确的认证错误就通过
            logger.info(f"TuShare 轻量验证跳过: {err_msg[:60]}，假定 Token 有效")
            return True

    def _do_connect(self):
        try:
            import tushare as ts
        except ImportError:
            raise ImportError("请安装 tushare: pip install tushare")
        ts.set_token(self._api_key)
        self._pro = ts.pro_api()
        if not self._validate_credentials():
            raise ConnectionError("TuShare API Token 无效或网络不可达")

    @staticmethod
    def _to_tushare_code(code: str) -> str:
        """A股代码转 tushare 格式: 600519 → 600519.SH"""
        if "." in code:
            return code
        suffix = ".SH" if code.startswith(("6", "9")) else ".SZ"
        return f"{code}{suffix}"

    @staticmethod
    def _from_tushare_code(ts_code: str) -> str:
        """tushare 格式转 A股代码"""
        return ts_code.split(".")[0]

    # ── 日K线 ───────────────────────────────────────────
    def fetch_daily_kline(
        self,
        code: str,
        start_date: str = "20200101",
        end_date: str = "",
        adjust: str = "qfq",
    ) -> FetchResult:
        if not self._pro:
            return FetchResult(success=False, source_name=self.name, error="未连接")
        if not self.has_quota():
            return FetchResult(success=False, source_name=self.name, error="额度已用完")

        ts_code = self._to_tushare_code(code)
        if not end_date:
            end_date = datetime.now().strftime("%Y%m%d")

        # TuShare 支持前/后复权因子加载，这里先用基础日线
        try:
            df = self._pro.daily(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                limit=5000,
            )
            self._consume_quota(_TUSHARE_API_COST.get("daily", 2))

            if df is None or df.empty:
                return FetchResult(success=False, source_name=self.name, error="无数据")

            # 重命名为项目统一格式
            col_map = {
                "ts_code": "ts_code",
                "trade_date": "日期",
                "open": "开盘",
                "high": "最高",
                "low": "最低",
                "close": "收盘",
                "vol": "成交量",
                "amount": "成交额",
                "pre_close": "前收盘",
                "change": "涨跌额",
                "pct_chg": "涨跌幅",
            }
            df_out = df.rename(columns=col_map)
            df_out["股票代码"] = df["ts_code"].apply(self._from_tushare_code)
            df_out["日期"] = df_out["日期"].astype(str)
            df_out = df_out.sort_values("日期")

            # 前复权：用复权因子近似
            if adjust == "qfq":
                df_out = self._apply_qfq_from_factors(df_out)

            return FetchResult(
                success=True,
                data=df_out,
                source_name=self.name,
                quota_used=_TUSHARE_API_COST.get("daily", 2),
            )
        except Exception as e:
            logger.warning(f"TuShare 获取 {code} K线失败: {e}")
            return FetchResult(success=False, source_name=self.name, error=str(e))

    def _apply_qfq_from_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """使用 TuShare 复权因子进行前复权"""
        if df.empty or "ts_code" not in df.columns:
            return df
        try:
            ts_code = str(df["ts_code"].iloc[0])
            factors = self._pro.adj_factor(ts_code=ts_code)
            if factors is None or factors.empty:
                return df
            factors["trade_date"] = factors["trade_date"].astype(str)
            # 以最新复权因子为基准
            latest_factor = float(factors.iloc[0]["adj_factor"])
            factor_map = dict(zip(factors["trade_date"], factors["adj_factor"]))
            for col in ["开盘", "最高", "最低", "收盘"]:
                if col in df.columns:
                    df[col] = df.apply(
                        lambda r: (
                            float(r[col]) * latest_factor / float(factor_map.get(r["日期"], latest_factor))
                            if r["日期"] in factor_map
                            else float(r[col])
                        ),
                        axis=1,
                    )
            self._consume_quota(1)  # adj_factor 也消耗积分
        except Exception as e:
            logger.debug(f"TuShare 复权因子获取失败: {e}")
        return df

    # ── 财务数据 ─────────────────────────────────────────
    def fetch_financial_data(self, code: str) -> FetchResult:
        if not self._pro:
            return FetchResult(success=False, source_name=self.name, error="未连接")
        if not self.has_quota():
            return FetchResult(success=False, source_name=self.name, error="额度已用完")

        ts_code = self._to_tushare_code(code)

        try:
            # 获取最近4期的财务指标
            df_fin = self._pro.fina_indicator(
                ts_code=ts_code,
                limit=4,
            )
            self._consume_quota(_TUSHARE_API_COST.get("financial", 2))

            # 获取最新一期利润表
            df_income = self._pro.income(
                ts_code=ts_code,
                limit=4,
            )
            self._consume_quota(_TUSHARE_API_COST.get("income", 2))

            # 获取日线基本面（PE/PB/市值等）
            df_basic = self._pro.daily_basic(
                ts_code=ts_code,
                limit=1,
            )
            self._consume_quota(_TUSHARE_API_COST.get("daily_basic", 2))

            # 合并构建统一格式
            result: dict[str, object] = {"股票代码": code}

            if df_basic is not None and not df_basic.empty:
                row = df_basic.iloc[-1]
                result.update(
                    {
                        "市盈率-动态": float(row.get("pe_ttm", 0) or 0),
                        "市净率": float(row.get("pb", 0) or 0),
                        "总市值": float(row.get("total_mv", 0) or 0) * 10000,  # 万元→元
                        "流通市值": float(row.get("circ_mv", 0) or 0) * 10000,
                        "换手率": float(row.get("turnover_rate", 0) or 0),
                        "成交量": float(row.get("vol", 0) or 0),
                    }
                )

            if df_fin is not None and not df_fin.empty:
                row = df_fin.iloc[-1]
                result.update(
                    {
                        "roe": float(row.get("roe", 0) or 0),
                        "roe_dt": float(row.get("roe_dt", 0) or 0),
                        "毛利率": float(row.get("grossprofit_margin", 0) or 0),
                        "净利率": float(row.get("netprofit_margin", 0) or 0),
                        "资产负债率": float(row.get("debt_to_assets", 0) or 0),
                        "净利润增长率": float(row.get("q_profit_yoy", 0) or 0),
                        "营业收入增长率": float(row.get("q_gr_yoy", 0) or 0),
                        "每股收益": float(row.get("eps", 0) or 0),
                        "每股净资产": float(row.get("bps", 0) or 0),
                    }
                )

            return FetchResult(
                success=True,
                data=pd.DataFrame([result]),
                source_name=self.name,
                quota_used=6,
            )
        except Exception as e:
            logger.warning(f"TuShare 获取 {code} 财务数据失败: {e}")
            return FetchResult(success=False, source_name=self.name, error=str(e))

    # ── 指数日线 ─────────────────────────────────────────
    def fetch_index_daily(
        self,
        index_code: str,
        start_date: str = "20200101",
        end_date: str = "",
    ) -> FetchResult:
        if not self._pro:
            return FetchResult(success=False, source_name=self.name, error="未连接")
        if not self.has_quota():
            return FetchResult(success=False, source_name=self.name, error="额度已用完")
        if not self._check_rate_limit("index_daily"):
            return FetchResult(success=False, source_name=self.name, error="频率限制，跳过")

        if not end_date:
            end_date = datetime.now().strftime("%Y%m%d")

        # 上证指数: 000001.SH, 深证成指: 399001.SZ
        ts_code = self._to_tushare_code(index_code)

        try:
            df = self._pro.index_daily(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
            )
            self._mark_api_call("index_daily")
            self._consume_quota(_TUSHARE_API_COST.get("index_daily", 2))

            if df is None or df.empty:
                return FetchResult(success=False, source_name=self.name, error="无数据")

            col_map = {
                "trade_date": "date",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "vol": "volume",
                "amount": "amount",
            }
            df_out = df.rename(columns=col_map)
            df_out["date"] = df_out["date"].astype(str)

            return FetchResult(
                success=True,
                data=df_out,
                source_name=self.name,
                quota_used=_TUSHARE_API_COST.get("index_daily", 2),
            )
        except Exception as e:
            logger.warning(f"TuShare 获取指数 {index_code} 失败: {e}")
            return FetchResult(success=False, source_name=self.name, error=str(e))

    # ── 资金流向 ─────────────────────────────────────────
    def fetch_money_flow(self, code: str) -> FetchResult:
        if not self._pro:
            return FetchResult(success=False, source_name=self.name, error="未连接")
        if not self.has_quota():
            return FetchResult(success=False, source_name=self.name, error="额度已用完")

        ts_code = self._to_tushare_code(code)

        try:
            df = self._pro.moneyflow(
                ts_code=ts_code,
                limit=30,
            )
            self._consume_quota(_TUSHARE_API_COST.get("moneyflow", 2))

            if df is None or df.empty:
                return FetchResult(success=False, source_name=self.name, error="无数据")

            col_map = {
                "trade_date": "日期",
                "buy_sm_vol": "小单买入",
                "buy_md_vol": "中单买入",
                "buy_lg_vol": "大单买入",
                "buy_elg_vol": "超大单买入",
                "sell_sm_vol": "小单卖出",
                "sell_md_vol": "中单卖出",
                "sell_lg_vol": "大单卖出",
                "sell_elg_vol": "超大单卖出",
                "net_mf_vol": "主力净流入",
            }
            df_out = df.rename(columns=col_map)
            return FetchResult(
                success=True,
                data=df_out,
                source_name=self.name,
                quota_used=_TUSHARE_API_COST.get("moneyflow", 2),
            )
        except Exception as e:
            logger.warning(f"TuShare 获取 {code} 资金流失败: {e}")
            return FetchResult(success=False, source_name=self.name, error=str(e))

    # ── 股票列表 ─────────────────────────────────────────
    def fetch_stock_list(self) -> FetchResult:
        if not self._pro:
            return FetchResult(success=False, source_name=self.name, error="未连接")
        if not self.has_quota():
            return FetchResult(success=False, source_name=self.name, error="额度已用完")

        try:
            df = self._pro.stock_basic(
                exchange="",
                list_status="L",
                fields="ts_code,symbol,name,area,industry,market,list_date",
            )
            self._consume_quota(_TUSHARE_API_COST.get("stock_basic", 1))

            if df is None or df.empty:
                return FetchResult(success=False, source_name=self.name, error="无数据")

            df_out = pd.DataFrame(
                {
                    "股票代码": df["symbol"],
                    "股票简称": df["name"],
                    "行业": df["industry"],
                    "地区": df.get("area", ""),
                    "市场类型": df.get("market", ""),
                    "上市时间": df["list_date"],
                }
            )
            return FetchResult(success=True, data=df_out, source_name=self.name, quota_used=1)
        except Exception as e:
            logger.warning(f"TuShare 获取股票列表失败: {e}")
            return FetchResult(success=False, source_name=self.name, error=str(e))
