"""
内置技术指标库 (AKQuant 教材 §16)

全 numpy 实现，零外部依赖。覆盖 6 大类 30+ 种指标：

趋势类:  SMA, EMA, WMA, MACD, TRIX, DMI(ADX/PDI/MDI), Parabolic SAR
动量类:  RSI, KDJ, CCI, MFI, WR, MOM, ROC, Ultimate Oscillator
波动类:  BOLL, ATR, Keltner Channels, Donchian Channels, Historical Volatility
量价类:  OBV, VWAP, VPT, AD (Chaikin), VR (Volume Ratio), Force Index
统计类:  Beta, Correlation, Linear Regression Slope/R², Sharpe (rolling)
形态类:  Hammer, Doji, Engulfing, Morning/Evening Star, Three Soldiers/Crows

设计原则:
- 纯向量化 numpy/pandas 操作，无需 talib
- calc_* 返回计算结果（pd.DataFrame 或 pd.Series）
- analyze_* 返回 IndicatorResult 信号解读
- 所有函数签名带类型标注
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .config import KDJ_M1, KDJ_M2, KDJ_N, MA_PERIODS, MACD_FAST, MACD_SIGNAL, MACD_SLOW, RSI_PERIOD

# ═══════════════════════════════════════════════
# 公共数据结构
# ═══════════════════════════════════════════════


@dataclass
class IndicatorResult:
    """单指标分析结果"""
    trend: str           # bullish / bearish / neutral / overbought / oversold
    signal: str          # buy / sell / hold
    strength: float      # 0-100 信号强度
    details: dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════
# 一、趋势类指标 (Trend Indicators)
# ═══════════════════════════════════════════════

def sma(df: pd.DataFrame, period: int, col: str = "close") -> pd.Series:
    """简单移动平均"""
    return df[col].rolling(period).mean()


def ema(df: pd.DataFrame, period: int, col: str = "close") -> pd.Series:
    """指数移动平均"""
    return df[col].ewm(span=period, adjust=False).mean()


def wma(df: pd.DataFrame, period: int, col: str = "close") -> pd.Series:
    """加权移动平均 (线性权重: 最近权重=period, 最远权重=1)"""
    weights = np.arange(1, period + 1)
    return df[col].rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)


def calc_multi_ma(df: pd.DataFrame, periods: list | None = None, col: str = "close") -> pd.DataFrame:
    """计算多周期均线"""
    if periods is None:
        periods = MA_PERIODS
    df = df.copy()
    for p in periods:
        if len(df) >= p:
            df[f"MA{p}"] = sma(df, p, col)
    return df


# 向后兼容别名
calc_ma = calc_multi_ma
calc_ema = staticmethod(ema)


def calc_macd(
    df: pd.DataFrame, fast: int = MACD_FAST, slow: int = MACD_SLOW, signal: int = MACD_SIGNAL,
    col: str = "close",
) -> pd.DataFrame:
    """MACD 指标"""
    df = df.copy()
    df["EMA_fast"] = ema(df, fast, col)
    df["EMA_slow"] = ema(df, slow, col)
    df["DIF"] = df["EMA_fast"] - df["EMA_slow"]
    df["DEA"] = df["DIF"].ewm(span=signal, adjust=False).mean()
    df["MACD"] = 2 * (df["DIF"] - df["DEA"])
    return df


def analyze_macd(df: pd.DataFrame) -> IndicatorResult:
    df = calc_macd(df)
    if len(df) < 2:
        return IndicatorResult("neutral", "hold", 50, {})
    latest, prev = df.iloc[-1], df.iloc[-2]
    golden_cross = (prev["DIF"] <= prev["DEA"]) and (latest["DIF"] > latest["DEA"])
    death_cross = (prev["DIF"] >= prev["DEA"]) and (latest["DIF"] < latest["DEA"])
    if latest["DIF"] > latest["DEA"] and latest["MACD"] > 0:
        trend, strength = "bullish", min(80, 50 + abs(latest["MACD"]) / latest["close"] * 1000)
    elif latest["DIF"] < latest["DEA"] and latest["MACD"] < 0:
        trend, strength = "bearish", min(80, 50 + abs(latest["MACD"]) / latest["close"] * 1000)
    else:
        trend, strength = "neutral", 50
    signal = "buy" if golden_cross else ("sell" if death_cross else "hold")
    return IndicatorResult(
        trend=trend, signal=signal, strength=round(strength, 1),
        details={"DIF": round(latest["DIF"], 4), "DEA": round(latest["DEA"], 4),
                 "MACD": round(latest["MACD"], 4), "golden_cross": golden_cross, "death_cross": death_cross},
    )


def calc_trix(df: pd.DataFrame, period: int = 15, col: str = "close") -> pd.Series:
    """TRIX 三重指数平滑 (三阶 EMA 的动量)"""
    ema1 = df[col].ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, adjust=False).mean()
    return ema3.pct_change(period) * 100


def calc_dmi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """DMI 趋向指标: ADX / +DI / -DI

    ADX > 25 趋势明显, ADX > 50 强趋势
    +DI > -DI 多头, +DI < -DI 空头
    """
    df = df.copy()
    df["TR"] = np.maximum(
        df["high"] - df["low"],
        np.maximum(abs(df["high"] - df["close"].shift(1)), abs(df["low"] - df["close"].shift(1))),
    )
    df["up_move"] = df["high"] - df["high"].shift(1)
    df["down_move"] = df["low"].shift(1) - df["low"]
    df["+DM"] = np.where((df["up_move"] > df["down_move"]) & (df["up_move"] > 0), df["up_move"], 0)
    df["-DM"] = np.where((df["down_move"] > df["up_move"]) & (df["down_move"] > 0), df["down_move"], 0)

    tr_smooth = df["TR"].ewm(alpha=1 / period, adjust=False).mean()
    pdm_smooth = df["+DM"].ewm(alpha=1 / period, adjust=False).mean()
    ndm_smooth = df["-DM"].ewm(alpha=1 / period, adjust=False).mean()

    df["+DI"] = 100 * pdm_smooth / tr_smooth.replace(0, 1)
    df["-DI"] = 100 * ndm_smooth / tr_smooth.replace(0, 1)
    dx = 100 * abs(df["+DI"] - df["-DI"]) / (df["+DI"] + df["-DI"]).replace(0, 1)
    df["ADX"] = dx.ewm(alpha=1 / period, adjust=False).mean()
    return df[["+DI", "-DI", "ADX"]]


def analyze_dmi(df: pd.DataFrame) -> IndicatorResult:
    dmi = calc_dmi(df)
    if len(dmi) < 1:
        return IndicatorResult("neutral", "hold", 50, {})
    latest = dmi.iloc[-1]
    adx, pdi, ndi = latest["ADX"], latest["+DI"], latest["-DI"]
    if adx > 50 and pdi > ndi:
        trend, signal, strength = "bullish", "buy", 80
    elif adx > 50 and ndi > pdi:
        trend, signal, strength = "bearish", "sell", 80
    elif adx > 25 and pdi > ndi:
        trend, signal, strength = "bullish", "buy", 65
    elif adx > 25 and ndi > pdi:
        trend, signal, strength = "bearish", "sell", 65
    elif pdi > ndi:
        trend, signal, strength = "bullish", "hold", 55
    elif ndi > pdi:
        trend, signal, strength = "bearish", "hold", 45
    else:
        trend, signal, strength = "neutral", "hold", 50
    return IndicatorResult(
        trend=trend, signal=signal, strength=round(strength, 1),
        details={"ADX": round(adx, 2), "+DI": round(pdi, 2), "-DI": round(ndi, 2)},
    )


def calc_sar(df: pd.DataFrame, acceleration: float = 0.02, maximum: float = 0.2) -> pd.Series:
    """Parabolic SAR (威尔斯·威尔德抛物线止损反转指标)

    位于价格下方时趋势向上,位于价格上方时趋势向下。
    """
    n = len(df)
    sar = np.full(n, np.nan)
    high, low, close = df["high"].values, df["low"].values, df["close"].values
    if n < 1:
        return pd.Series(sar, index=df.index)

    # 判断初始趋势: 价格在上升则做多
    bullish = True
    sar[0] = low[0]
    ep = high[0]  # 极点
    af = acceleration

    for i in range(1, n):
        prev_sar = sar[i - 1]
        if bullish:
            sar[i] = prev_sar + af * (ep - prev_sar)
            sar[i] = min(sar[i], low[i - 1], low[i] if i >= 2 else low[i])  # SAR 不能高于前两低点
            if high[i] > ep:
                ep = high[i]
                af = min(af + acceleration, maximum)
            if close[i] < sar[i]:
                bullish = False
                sar[i] = ep
                ep = low[i]
                af = acceleration
        else:
            sar[i] = prev_sar - af * (prev_sar - ep)
            sar[i] = max(sar[i], high[i - 1], high[i] if i >= 2 else high[i])  # SAR 不能低于前两高点
            if low[i] < ep:
                ep = low[i]
                af = min(af + acceleration, maximum)
            if close[i] > sar[i]:
                bullish = True
                sar[i] = ep
                ep = high[i]
                af = acceleration

    return pd.Series(sar, index=df.index)


def analyze_sar(df: pd.DataFrame) -> IndicatorResult:
    sar_vals = calc_sar(df)
    if len(sar_vals) < 2:
        return IndicatorResult("neutral", "hold", 50, {})
    close, sar = df["close"].iloc[-1], sar_vals.iloc[-1]
    if close > sar:
        return IndicatorResult("bullish", "buy", 65, {"SAR": round(sar, 2), "position": "价格在SAR上方"})
    else:
        return IndicatorResult("bearish", "sell", 65, {"SAR": round(sar, 2), "position": "价格在SAR下方"})


# ═══════════════════════════════════════════════
# 二、动量类指标 (Momentum Indicators)
# ═══════════════════════════════════════════════

def calc_rsi(df: pd.DataFrame, period: int = RSI_PERIOD, col: str = "close") -> pd.DataFrame:
    df = df.copy()
    delta = df[col].diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1)
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


def analyze_rsi(df: pd.DataFrame) -> IndicatorResult:
    df = calc_rsi(df)
    if len(df) < 1:
        return IndicatorResult("neutral", "hold", 50, {})
    rsi = df["RSI"].iloc[-1]
    if rsi > 80:
        return IndicatorResult("overbought", "sell", min(95, rsi), {"RSI": round(rsi, 2)})
    elif rsi > 70:
        return IndicatorResult("overbought", "hold", rsi, {"RSI": round(rsi, 2)})
    elif rsi < 20:
        return IndicatorResult("oversold", "buy", max(5, 100 - rsi), {"RSI": round(rsi, 2)})
    elif rsi < 30:
        return IndicatorResult("oversold", "hold", 100 - rsi, {"RSI": round(rsi, 2)})
    elif rsi > 50:
        return IndicatorResult("bullish", "hold", rsi, {"RSI": round(rsi, 2)})
    else:
        return IndicatorResult("bearish", "hold", rsi, {"RSI": round(rsi, 2)})


def calc_kdj(df: pd.DataFrame, n: int = KDJ_N, m1: int = KDJ_M1, m2: int = KDJ_M2) -> pd.DataFrame:
    df = df.copy()
    low_n = df["low"].rolling(n).min()
    high_n = df["high"].rolling(n).max()
    denom = high_n - low_n
    rsv = ((df["close"] - low_n) / denom.replace(0, 1)) * 100
    rsv = rsv.fillna(50)
    df["K"] = rsv.ewm(com=m1 - 1, adjust=False).mean()
    df["D"] = df["K"].ewm(com=m2 - 1, adjust=False).mean()
    df["J"] = 3 * df["K"] - 2 * df["D"]
    return df


def analyze_kdj(df: pd.DataFrame) -> IndicatorResult:
    df = calc_kdj(df)
    if len(df) < 2:
        return IndicatorResult("neutral", "hold", 50, {})
    latest, prev = df.iloc[-1], df.iloc[-2]
    k, d, j = latest["K"], latest["D"], latest["J"]
    golden_cross = (prev["K"] <= prev["D"]) and (latest["K"] > latest["D"])
    death_cross = (prev["K"] >= prev["D"]) and (latest["K"] < latest["D"])
    if k > 80 and d > 80:
        trend, signal, strength = "overbought", "sell" if death_cross else "hold", 70
    elif k < 20 and d < 20:
        trend, signal, strength = "oversold", "buy" if golden_cross else "hold", 70
    elif k > d:
        trend, signal, strength = "bullish", "buy" if golden_cross else "hold", 60
    else:
        trend, signal, strength = "bearish", "sell" if death_cross else "hold", 60
    return IndicatorResult(
        trend=trend, signal=signal, strength=round(strength, 1),
        details={"K": round(k, 2), "D": round(d, 2), "J": round(j, 2),
                 "golden_cross": golden_cross, "death_cross": death_cross},
    )


def calc_cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """CCI 商品通道指数 (Commodity Channel Index)

    CCI > +100 超买, CCI < -100 超卖
    """
    tp = (df["high"] + df["low"] + df["close"]) / 3
    ma = tp.rolling(period).mean()
    md = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    return (tp - ma) / (0.015 * md.replace(0, 1))


def analyze_cci(df: pd.DataFrame) -> IndicatorResult:
    cci = calc_cci(df)
    if len(cci) < 1:
        return IndicatorResult("neutral", "hold", 50, {})
    val = cci.iloc[-1]
    if val > 200:
        return IndicatorResult("overbought", "sell", 85, {"CCI": round(val, 2)})
    elif val > 100:
        return IndicatorResult("overbought", "hold", 65, {"CCI": round(val, 2)})
    elif val < -200:
        return IndicatorResult("oversold", "buy", 85, {"CCI": round(val, 2)})
    elif val < -100:
        return IndicatorResult("oversold", "hold", 65, {"CCI": round(val, 2)})
    elif val > 0:
        return IndicatorResult("bullish", "hold", 55, {"CCI": round(val, 2)})
    else:
        return IndicatorResult("bearish", "hold", 45, {"CCI": round(val, 2)})


def calc_wr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """威廉指标 (Williams %R)

    值范围 [-100, 0]: 高于 -20 超买, 低于 -80 超卖
    """
    high_n = df["high"].rolling(period).max()
    low_n = df["low"].rolling(period).min()
    return (high_n - df["close"]) / (high_n - low_n).replace(0, 1) * (-100)


def analyze_wr(df: pd.DataFrame) -> IndicatorResult:
    wr = calc_wr(df)
    if len(wr) < 1:
        return IndicatorResult("neutral", "hold", 50, {})
    val = wr.iloc[-1]
    if val > -20:
        return IndicatorResult("overbought", "sell", 70, {"WR": round(val, 2)})
    elif val < -80:
        return IndicatorResult("oversold", "buy", 70, {"WR": round(val, 2)})
    elif val > -50:
        return IndicatorResult("bullish", "hold", 55, {"WR": round(val, 2)})
    else:
        return IndicatorResult("bearish", "hold", 45, {"WR": round(val, 2)})


def calc_mom(df: pd.DataFrame, period: int = 10, col: str = "close") -> pd.Series:
    """动量指标 (Momentum): price[t] - price[t-period]"""
    return df[col] - df[col].shift(period)


def calc_roc(df: pd.DataFrame, period: int = 10, col: str = "close") -> pd.Series:
    """变动率 (Rate of Change): (price[t] / price[t-period] - 1) * 100"""
    return df[col].pct_change(period) * 100


def calc_mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """资金流量指数 (Money Flow Index)

    MFI > 80 超买, MFI < 20 超卖
    """
    tp = (df["high"] + df["low"] + df["close"]) / 3
    mf = tp * df["volume"]
    # 判定资金流向
    pos_flow = np.where(tp > tp.shift(1), mf, 0)
    neg_flow = np.where(tp < tp.shift(1), mf, 0)
    pos_sum = pd.Series(pos_flow).rolling(period).sum()
    neg_sum = pd.Series(neg_flow).rolling(period).sum()
    mr = pos_sum / neg_sum.replace(0, 1)
    return 100 - (100 / (1 + mr))


def analyze_mfi(df: pd.DataFrame) -> IndicatorResult:
    mfi = calc_mfi(df)
    if len(mfi) < 1:
        return IndicatorResult("neutral", "hold", 50, {})
    val = mfi.iloc[-1]
    if val > 80:
        return IndicatorResult("overbought", "sell", 75, {"MFI": round(val, 2)})
    elif val < 20:
        return IndicatorResult("oversold", "buy", 75, {"MFI": round(val, 2)})
    elif val > 50:
        return IndicatorResult("bullish", "hold", 55, {"MFI": round(val, 2)})
    else:
        return IndicatorResult("bearish", "hold", 45, {"MFI": round(val, 2)})


def calc_uo(df: pd.DataFrame, period1: int = 7, period2: int = 14, period3: int = 28) -> pd.Series:
    """终极指标 (Ultimate Oscillator)

    综合三个时间框架的动量，减少单一周期假信号。
    UO > 70 超买, UO < 30 超卖
    """
    tl = np.minimum(df["low"], df["close"].shift(1))
    bp = df["close"] - tl  # 买压
    tr = np.maximum(df["high"] - df["low"],
                    np.maximum(abs(df["high"] - df["close"].shift(1)),
                               abs(df["low"] - df["close"].shift(1))))
    # 三周期加权
    avg1 = pd.Series(bp).rolling(period1).sum() / pd.Series(tr).rolling(period1).sum().replace(0, 1)
    avg2 = pd.Series(bp).rolling(period2).sum() / pd.Series(tr).rolling(period2).sum().replace(0, 1)
    avg3 = pd.Series(bp).rolling(period3).sum() / pd.Series(tr).rolling(period3).sum().replace(0, 1)
    return 100 * (4 * avg1 + 2 * avg2 + avg3) / 7


# ═══════════════════════════════════════════════
# 三、波动类指标 (Volatility Indicators)
# ═══════════════════════════════════════════════

def calc_bollinger(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0, col: str = "close") -> pd.DataFrame:
    df = df.copy()
    df["BOLL_MID"] = df[col].rolling(period).mean()
    std = df[col].rolling(period).std()
    df["BOLL_UP"] = df["BOLL_MID"] + std_dev * std
    df["BOLL_DN"] = df["BOLL_MID"] - std_dev * std
    df["BOLL_WIDTH"] = (df["BOLL_UP"] - df["BOLL_DN"]) / df["BOLL_MID"]
    df["%B"] = (df[col] - df["BOLL_DN"]) / (df["BOLL_UP"] - df["BOLL_DN"]).replace(0, 1)
    return df


def analyze_bollinger(df: pd.DataFrame) -> IndicatorResult:
    df = calc_bollinger(df)
    if len(df) < 1:
        return IndicatorResult("neutral", "hold", 50, {})
    latest = df.iloc[-1]
    close, upper, mid, lower, bb, width = latest["close"], latest["BOLL_UP"], latest["BOLL_MID"], latest["BOLL_DN"], latest["%B"], latest["BOLL_WIDTH"]
    if close >= upper:
        return IndicatorResult("overbought", "sell", 75, {"position": "上轨上方", "%B": round(bb, 3)})
    elif close <= lower:
        return IndicatorResult("oversold", "buy", 75, {"position": "下轨下方", "%B": round(bb, 3)})
    elif close > mid:
        return IndicatorResult("bullish", "hold", 55, {"position": "中轨上方", "%B": round(bb, 3), "width": round(width, 3)})
    else:
        return IndicatorResult("bearish", "hold", 45, {"position": "中轨下方", "%B": round(bb, 3), "width": round(width, 3)})


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    df = df.copy()
    df["tr"] = np.maximum(
        df["high"] - df["low"],
        np.maximum(abs(df["high"] - df["close"].shift(1)), abs(df["low"] - df["close"].shift(1))),
    )
    return df["tr"].ewm(span=period, adjust=False).mean()


def calc_keltner(df: pd.DataFrame, period: int = 20, atr_period: int = 14, multiplier: float = 2.0) -> pd.DataFrame:
    """肯特纳通道 (Keltner Channels)

    基于 ATR 的波动通道，比布林带更平滑。
    """
    df = df.copy()
    df["KC_MID"] = df["close"].ewm(span=period, adjust=False).mean()
    atr = calc_atr(df, atr_period)
    df["KC_UP"] = df["KC_MID"] + multiplier * atr
    df["KC_DN"] = df["KC_MID"] - multiplier * atr
    return df


def analyze_keltner(df: pd.DataFrame) -> IndicatorResult:
    df = calc_keltner(df)
    if len(df) < 1:
        return IndicatorResult("neutral", "hold", 50, {})
    close, up, mid, dn = df["close"].iloc[-1], df["KC_UP"].iloc[-1], df["KC_MID"].iloc[-1], df["KC_DN"].iloc[-1]
    if close > up:
        return IndicatorResult("bullish", "hold", 65, {"position": "上轨突破"})
    elif close < dn:
        return IndicatorResult("bearish", "sell", 65, {"position": "下轨跌破"})
    elif close > mid:
        return IndicatorResult("bullish", "hold", 55, {"position": "通道上半"})
    else:
        return IndicatorResult("bearish", "hold", 45, {"position": "通道下半"})


def calc_donchian(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """唐奇安通道"""
    df = df.copy()
    df["DC_UP"] = df["high"].rolling(period).max()
    df["DC_DN"] = df["low"].rolling(period).min()
    df["DC_MID"] = (df["DC_UP"] + df["DC_DN"]) / 2
    return df


def calc_breakout(df: pd.DataFrame, period: int = 20) -> tuple:
    """[兼容旧版] 突破通道"""
    dc = calc_donchian(df, period).iloc[-1]
    return dc.get("DC_UP", float("nan")), dc.get("DC_DN", float("nan"))


def calc_hist_vol(df: pd.DataFrame, period: int = 20, col: str = "close", annualize: int = 252) -> pd.Series:
    """历史波动率 (Historical Volatility)

    对数收益率的标准差，年化。
    """
    log_ret = np.log(df[col] / df[col].shift(1))
    return log_ret.rolling(period).std() * np.sqrt(annualize) * 100  # 百分比


# ═══════════════════════════════════════════════
# 四、量价类指标 (Volume & Price Indicators)
# ═══════════════════════════════════════════════

def calc_obv(df: pd.DataFrame) -> pd.Series:
    """OBV 能量潮 (On-Balance Volume)

    价格涨 + 成交量, 价格跌 - 成交量
    """
    direction = np.where(df["close"] > df["close"].shift(1), 1,
                         np.where(df["close"] < df["close"].shift(1), -1, 0))
    return (direction * df["volume"]).cumsum()


def calc_vwap(df: pd.DataFrame) -> pd.Series:
    """VWAP 成交量加权均价 (Volume Weighted Average Price)

    典型用法: 价格在 VWAP 上方看多, 下方看空
    """
    tp = (df["high"] + df["low"] + df["close"]) / 3
    cumulative_tp_vol = (tp * df["volume"]).cumsum()
    cumulative_vol = df["volume"].cumsum()
    return cumulative_tp_vol / cumulative_vol.replace(0, 1)


def calc_vpt(df: pd.DataFrame) -> pd.Series:
    """VPT 量价趋势 (Volume Price Trend)

    结合价格变动与成交量。
    """
    pct_chg = df["close"].pct_change().fillna(0)
    return (pct_chg * df["volume"]).cumsum()


def calc_ad(df: pd.DataFrame) -> pd.Series:
    """Chaikin A/D 集散指标 (Accumulation/Distribution)

    衡量资金流入流出: CLV * Volume 的累计
    CLV = ((close - low) - (high - close)) / (high - low)
    """
    denom = df["high"] - df["low"]
    clv = np.where(denom > 0, ((df["close"] - df["low"]) - (df["high"] - df["close"])) / denom, 0)
    return (clv * df["volume"]).cumsum()


def calc_vr(df: pd.DataFrame, period: int = 26) -> pd.Series:
    """VR 成交量比率 (Volume Ratio)

    上升日成交 / 下降日成交的累加比值
    VR > 350 超买, VR < 40 超卖
    """
    up_vol = pd.Series(np.where(df["close"] > df["close"].shift(1), df["volume"], 0))
    down_vol = pd.Series(np.where(df["close"] < df["close"].shift(1), df["volume"], 0))
    eq_vol = pd.Series(np.where(df["close"] == df["close"].shift(1), df["volume"], 0))
    up_sum = up_vol.rolling(period).sum()
    down_sum = down_vol.rolling(period).sum()
    eq_sum = eq_vol.rolling(period).sum()
    return 100 * (up_sum + eq_sum * 0.5) / (down_sum + eq_sum * 0.5).replace(0, 1)


def calc_force_index(df: pd.DataFrame, period: int = 13) -> pd.Series:
    """强力指数 (Force Index)

    价格变动 * 成交量，经 EMA 平滑
    """
    fi = (df["close"] - df["close"].shift(1)) * df["volume"]
    return fi.ewm(span=period, adjust=False).mean()


def calc_volume_ma(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """成交量均线"""
    return df["volume"].rolling(period).mean()


def calc_volume_ratio(df: pd.DataFrame, period: int = 5) -> pd.Series:
    """量比: 当前成交量 / 前N日均量"""
    return df["volume"] / df["volume"].rolling(period).mean().replace(0, 1)


def analyze_volume_price(df: pd.DataFrame) -> IndicatorResult:
    """量价关系分析"""
    if len(df) < 5:
        return IndicatorResult("neutral", "hold", 50, {})
    recent = df.tail(5)
    price_pct = (recent["close"].iloc[-1] - recent["close"].iloc[-5]) / recent["close"].iloc[-5] * 100
    avg_vol_first = recent["volume"].iloc[:2].mean()
    avg_vol_last = recent["volume"].iloc[-3:].mean()
    vol_change = (avg_vol_last - avg_vol_first) / avg_vol_first * 100 if avg_vol_first > 0 else 0
    if price_pct > 2 and vol_change > 20:
        return IndicatorResult("bullish", "buy", 75,
                               {"price_pct": round(price_pct, 2), "vol_pct": round(vol_change, 2), "pattern": "价涨量增"})
    elif price_pct > 2 and vol_change < -10:
        return IndicatorResult("bullish_warning", "hold", 40,
                               {"price_pct": round(price_pct, 2), "vol_pct": round(vol_change, 2), "pattern": "价涨量缩"})
    elif price_pct < -2 and vol_change > 20:
        return IndicatorResult("bearish", "sell", 75,
                               {"price_pct": round(price_pct, 2), "vol_pct": round(vol_change, 2), "pattern": "价跌量增"})
    elif price_pct < -2 and vol_change < -10:
        return IndicatorResult("bearish_stop", "hold", 40,
                               {"price_pct": round(price_pct, 2), "vol_pct": round(vol_change, 2), "pattern": "价跌量缩"})
    else:
        return IndicatorResult("neutral", "hold", 50,
                               {"price_pct": round(price_pct, 2), "vol_pct": round(vol_change, 2), "pattern": "正常"})


# ═══════════════════════════════════════════════
# 五、统计类指标 (Statistical Indicators)
# ═══════════════════════════════════════════════

def calc_beta(stock_rets: pd.Series, market_rets: pd.Series, window: int = 252) -> pd.Series:
    """滚动 Beta 系数

    Beta = Cov(stock, market) / Var(market)
    """
    cov = stock_rets.rolling(window).cov(market_rets)
    var = market_rets.rolling(window).var()
    return cov / var.replace(0, 1)


def calc_correlation(stock_rets: pd.Series, market_rets: pd.Series, window: int = 252) -> pd.Series:
    """滚动相关系数"""
    return stock_rets.rolling(window).corr(market_rets)


def calc_linear_reg(df: pd.DataFrame, period: int = 20, col: str = "close") -> pd.DataFrame:
    """线性回归斜率与 R²

    教材参考: §16.6 统计类 - 线性回归
    """
    df = df.copy()
    x = np.arange(period)
    denom = period * np.sum(x**2) - np.sum(x)**2

    def _regress(y: np.ndarray) -> tuple:
        if len(y) < period or np.isnan(y).any():
            return np.nan, np.nan, np.nan
        n = period
        sum_x, sum_y = np.sum(x), np.sum(y)
        sum_xy, sum_y2 = np.sum(x * y), np.sum(y**2)
        slope = (n * sum_xy - sum_x * sum_y) / denom
        intercept = (sum_y - slope * sum_x) / n
        y_pred = intercept + slope * x
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        return intercept + slope * (period - 1), slope, r2

    results = df[col].rolling(period).apply(_regress, raw=True)
    # 分解 tuple 结果
    df["LR_value"] = [r[0] if isinstance(r, tuple) else np.nan for r in results]
    df["LR_slope"] = [r[1] if isinstance(r, tuple) else np.nan for r in results]
    df["LR_R2"] = [r[2] if isinstance(r, tuple) else np.nan for r in results]
    return df


def calc_mean_reversion_zscore(df: pd.DataFrame, period: int = 20, col: str = "close") -> pd.Series:
    """均值回归 Z-Score

    Z > 2 超买 (价格远超均线), Z < -2 超卖
    """
    ma = df[col].rolling(period).mean()
    std = df[col].rolling(period).std()
    return (df[col] - ma) / std.replace(0, 1)


def calc_sharpe_rolling(df: pd.DataFrame, period: int = 252, col: str = "close",
                         risk_free: float = 0.02) -> pd.Series:
    """滚动夏普比率 (年化)

    用于衡量风险调整后收益。
    """
    rets = df[col].pct_change().fillna(0)
    return (rets.rolling(period).mean() - risk_free / 252) / rets.rolling(period).std().replace(0, 1) * np.sqrt(252)


def calc_drawdown(df: pd.DataFrame, col: str = "close") -> pd.DataFrame:
    """回撤分析: 当前回撤百分比"""
    df = df.copy()
    df["dd_peak"] = df[col].expanding().max()
    df["dd_pct"] = (df[col] - df["dd_peak"]) / df["dd_peak"] * 100
    return df


# ═══════════════════════════════════════════════
# 六、K线形态识别 (Candlestick Pattern Recognition)
# ═══════════════════════════════════════════════

def _body(row) -> float:
    """实体大小 (abs)"""
    return abs(row["close"] - row["open"])


def _upper_shadow(row) -> float:
    """上影线长度"""
    return row["high"] - max(row["close"], row["open"])


def _lower_shadow(row) -> float:
    """下影线长度"""
    return min(row["close"], row["open"]) - row["low"]


def _total_range(row) -> float:
    """总振幅"""
    return row["high"] - row["low"]


def detect_doji(df: pd.DataFrame, body_ratio: float = 0.1) -> pd.Series:
    """十字星 (Doji): 实体很小, 上下影线相当"""
    body = df.apply(_body, axis=1)
    total = df.apply(_total_range, axis=1)
    return (body / total.replace(0, 1)) < body_ratio


def detect_hammer(df: pd.DataFrame, body_ratio: float = 0.3, shadow_ratio: float = 2.0) -> pd.Series:
    """锤子线 (Hammer): 下影线长, 实体小且在顶部"""
    body = df.apply(_body, axis=1)
    lower = df.apply(_lower_shadow, axis=1)
    upper = df.apply(_upper_shadow, axis=1)
    total = df.apply(_total_range, axis=1)
    return (body / total.replace(0, 1) < body_ratio) & (lower > shadow_ratio * body) & (upper < body)


def detect_shooting_star(df: pd.DataFrame, body_ratio: float = 0.3, shadow_ratio: float = 2.0) -> pd.Series:
    """射击之星: 上影线长, 实体小且在底部"""
    body = df.apply(_body, axis=1)
    upper = df.apply(_upper_shadow, axis=1)
    lower = df.apply(_lower_shadow, axis=1)
    total = df.apply(_total_range, axis=1)
    return (body / total.replace(0, 1) < body_ratio) & (upper > shadow_ratio * body) & (lower < body)


def detect_engulfing(df: pd.DataFrame) -> pd.Series:
    """吞没形态: 今日实体完全包裹昨日实体"""
    curr_body = df.apply(_body, axis=1)
    prev_body = df.apply(_body, axis=1).shift(1)
    curr_bull = df["close"] > df["open"]
    prev_bull = df["close"].shift(1) > df["open"].shift(1)
    bullish = (~prev_bull) & curr_bull & (df["open"] < df["close"].shift(1)) & (df["close"] > df["open"].shift(1))
    bearish = prev_bull & (~curr_bull) & (df["open"] > df["close"].shift(1)) & (df["close"] < df["open"].shift(1))
    return bullish | bearish


def detect_morning_star(df: pd.DataFrame, threshold: float = 0.3) -> pd.Series:
    """晨星 (Morning Star): 三根K线, 第一根大阴, 第二根小实体跳空, 第三根大阳"""
    if len(df) < 3:
        return pd.Series(False, index=df.index)
    body = df.apply(_body, axis=1)
    total = df.apply(_total_range, axis=1)
    body_ratio = body / total.replace(0, 1)
    c1_bear = df["close"].shift(2) < df["open"].shift(2)  # 大阴线
    c1_big = body_ratio.shift(2) > threshold
    c2_small = body_ratio.shift(1) < (threshold / 2)  # 小实体
    c2_gap = df["open"].shift(1) < df["close"].shift(2)  # 跳空低开
    c3_bull = df["close"] > df["open"]  # 大阳线
    c3_big = body_ratio > threshold
    c3_close = df["close"] > (df["open"].shift(2) + df["close"].shift(2)) / 2  # 回到第一根一半以上
    return c1_bear & c1_big & c2_small & c2_gap & c3_bull & c3_big & c3_close


def detect_evening_star(df: pd.DataFrame, threshold: float = 0.3) -> pd.Series:
    """黄昏星: 晨星的镜像形态"""
    if len(df) < 3:
        return pd.Series(False, index=df.index)
    body = df.apply(_body, axis=1)
    total = df.apply(_total_range, axis=1)
    body_ratio = body / total.replace(0, 1)
    c1_bull = df["close"].shift(2) > df["open"].shift(2)
    c1_big = body_ratio.shift(2) > threshold
    c2_small = body_ratio.shift(1) < (threshold / 2)
    c2_gap = df["open"].shift(1) > df["close"].shift(2)
    c3_bear = df["close"] < df["open"]
    c3_big = body_ratio > threshold
    c3_close = df["close"] < (df["open"].shift(2) + df["close"].shift(2)) / 2
    return c1_bull & c1_big & c2_small & c2_gap & c3_bear & c3_big & c3_close


def detect_three_soldiers(df: pd.DataFrame, body_ratio: float = 0.6) -> pd.Series:
    """三白兵 (Three White Soldiers): 连续三根实体大阳线, 逐步上移"""
    if len(df) < 3:
        return pd.Series(False, index=df.index)
    body = df.apply(_body, axis=1)
    total = df.apply(_total_range, axis=1)
    upper = df.apply(_upper_shadow, axis=1)
    ratio = body / total.replace(0, 1)
    c1, c2, c3 = df["close"], df["close"].shift(1), df["close"].shift(2)
    up = (c1 > c2) & (c2 > c3)
    all_bull = (c1 > df["open"]) & (c2 > df["open"].shift(1)) & (c3 > df["open"].shift(2))
    big_body = (ratio > body_ratio) & (ratio.shift(1) > body_ratio) & (ratio.shift(2) > body_ratio)
    small_upper = (upper < body * 0.2)
    return up & all_bull & big_body & small_upper


def detect_three_crows(df: pd.DataFrame, body_ratio: float = 0.6) -> pd.Series:
    """三只乌鸦: 三白兵的镜像"""
    if len(df) < 3:
        return pd.Series(False, index=df.index)
    body = df.apply(_body, axis=1)
    total = df.apply(_total_range, axis=1)
    lower = df.apply(_lower_shadow, axis=1)
    ratio = body / total.replace(0, 1)
    c1, c2, c3 = df["close"], df["close"].shift(1), df["close"].shift(2)
    down = (c1 < c2) & (c2 < c3)
    all_bear = (c1 < df["open"]) & (c2 < df["open"].shift(1)) & (c3 < df["open"].shift(2))
    big_body = (ratio > body_ratio) & (ratio.shift(1) > body_ratio) & (ratio.shift(2) > body_ratio)
    small_lower = (lower < body * 0.2)
    return down & all_bear & big_body & small_lower


def analyze_candlestick(df: pd.DataFrame) -> IndicatorResult:
    """综合 K 线形态分析，返回最强信号"""
    signals = {
        "morning_star": ("bullish", "buy", 80, "晨星 — 强烈看涨"),
        "three_soldiers": ("bullish", "buy", 75, "三白兵 — 持续看涨"),
        "hammer": ("bullish", "buy", 65, "锤子线 — 底部反转信号"),
        "engulfing_bull": ("bullish", "buy", 70, "看涨吞没 — 反转向上"),
        "evening_star": ("bearish", "sell", 80, "黄昏星 — 强烈看跌"),
        "three_crows": ("bearish", "sell", 75, "三只乌鸦 — 持续看跌"),
        "shooting_star": ("bearish", "sell", 65, "射击之星 — 顶部反转信号"),
        "engulfing_bear": ("bearish", "sell", 70, "看跌吞没 — 反转向下"),
        "doji": ("neutral", "hold", 50, "十字星 — 变盘信号"),
    }

    for name, (trend, signal, strength, desc) in signals.items():
        try:
            if name == "morning_star" and detect_morning_star(df).iloc[-1]:
                return IndicatorResult(trend, signal, strength, {"pattern": desc})
            if name == "three_soldiers" and detect_three_soldiers(df).iloc[-1]:
                return IndicatorResult(trend, signal, strength, {"pattern": desc})
            if name == "hammer" and detect_hammer(df).iloc[-1]:
                return IndicatorResult(trend, signal, strength, {"pattern": desc})
            if name == "engulfing_bull":
                eng = detect_engulfing(df)
                if eng.iloc[-1] and df["close"].iloc[-1] > df["open"].iloc[-1]:
                    return IndicatorResult(trend, signal, strength, {"pattern": desc})
            if name == "evening_star" and detect_evening_star(df).iloc[-1]:
                return IndicatorResult(trend, signal, strength, {"pattern": desc})
            if name == "three_crows" and detect_three_crows(df).iloc[-1]:
                return IndicatorResult(trend, signal, strength, {"pattern": desc})
            if name == "shooting_star" and detect_shooting_star(df).iloc[-1]:
                return IndicatorResult(trend, signal, strength, {"pattern": desc})
            if name == "engulfing_bear":
                eng = detect_engulfing(df)
                if eng.iloc[-1] and df["close"].iloc[-1] < df["open"].iloc[-1]:
                    return IndicatorResult(trend, signal, strength, {"pattern": desc})
            if name == "doji" and detect_doji(df).iloc[-1]:
                return IndicatorResult(trend, signal, strength, {"pattern": desc})
        except (IndexError, KeyError):
            continue

    return IndicatorResult("neutral", "hold", 50, {"pattern": "无明显形态"})


# ═══════════════════════════════════════════════
# 综合分析
# ═══════════════════════════════════════════════

def comprehensive_analysis(df: pd.DataFrame) -> dict[str, IndicatorResult]:
    """综合技术指标分析"""
    return {
        "MACD": analyze_macd(df),
        "KDJ": analyze_kdj(df),
        "RSI": analyze_rsi(df),
        "Bollinger": analyze_bollinger(df),
        "VolumePrice": analyze_volume_price(df),
        "CCI": analyze_cci(df),
        "DMI": analyze_dmi(df),
        "Candlestick": analyze_candlestick(df),
    }


def trend_score(df: pd.DataFrame) -> dict[str, Any]:
    """技术综合评分 (0-100)"""
    results = comprehensive_analysis(df)
    score = 50.0
    weights = {
        "MACD": 0.15, "KDJ": 0.15, "RSI": 0.10, "Bollinger": 0.10,
        "VolumePrice": 0.10, "CCI": 0.10, "DMI": 0.15, "Candlestick": 0.15,
    }
    for name, r in results.items():
        w = weights.get(name, 0.1)
        if r.signal == "buy":
            score += r.strength * w
        elif r.signal == "sell":
            score -= (100 - r.strength) * w
    score = max(0, min(100, score))
    if score >= 70:
        rating = "强势看多"
    elif score >= 55:
        rating = "偏多"
    elif score >= 45:
        rating = "震荡"
    elif score >= 30:
        rating = "偏空"
    else:
        rating = "弱势看空"
    return {"score": round(score, 1), "rating": rating, "details": {k: v.signal for k, v in results.items()}}


# ═══════════════════════════════════════════════
# 指标批量计算
# ═══════════════════════════════════════════════

def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """批量计算所有技术指标到 DataFrame

    返回含以下列的 DataFrame:
      MA5/MA10/MA20/MA60/MA120/MA250, DIF/DEA/MACD, K/D/J, RSI,
      BOLL_UP/MID/DN/width/%B, CCI, WR, MFI, +DI/-DI/ADX,
      SAR, OBV, VWAP, ATR, KC_UP/MID/DN, VR, HistVol_20
    """
    df = df.copy()

    # 趋势
    df = calc_multi_ma(df, col="close")
    df["EMA12"] = ema(df, 12)
    df["EMA26"] = ema(df, 26)
    df = calc_macd(df)
    df["TRIX"] = calc_trix(df)
    dmi = calc_dmi(df)
    df["+DI"] = dmi["+DI"]
    df["-DI"] = dmi["-DI"]
    df["ADX"] = dmi["ADX"]
    df["SAR"] = calc_sar(df)

    # 动量
    df = calc_rsi(df)
    df = calc_kdj(df)
    df["CCI"] = calc_cci(df)
    df["WR"] = calc_wr(df)
    df["MOM"] = calc_mom(df)
    df["ROC"] = calc_roc(df)
    df["MFI"] = calc_mfi(df)

    # 波动
    df = calc_bollinger(df)
    df["ATR"] = calc_atr(df)
    kc = calc_keltner(df)
    df["KC_UP"] = kc["KC_UP"]
    df["KC_MID"] = kc["KC_MID"]
    df["KC_DN"] = kc["KC_DN"]
    df["HistVol_20"] = calc_hist_vol(df, 20)

    # 量价
    df["OBV"] = calc_obv(df)
    df["VWAP"] = calc_vwap(df)
    df["VR"] = calc_vr(df)
    df["VolMA5"] = calc_volume_ma(df, 5)
    df["VolMA20"] = calc_volume_ma(df, 20)

    # 统计
    df["ZScore_20"] = calc_mean_reversion_zscore(df, 20)
    dd = calc_drawdown(df)
    df["DrawdownPct"] = dd["dd_pct"]

    return df
