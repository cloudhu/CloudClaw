"""
技术指标分析模块 - MACD、KDJ、RSI、均线、量价关系等
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from dataclasses import dataclass

from .config import (MACD_FAST, MACD_SLOW, MACD_SIGNAL,
                     KDJ_N, KDJ_M1, KDJ_M2, RSI_PERIOD, MA_PERIODS)


@dataclass
class IndicatorResult:
    trend: str
    signal: str
    strength: float
    details: Dict[str, Any]


def calc_ma(df: pd.DataFrame, periods: list = None) -> pd.DataFrame:
    if periods is None:
        periods = MA_PERIODS
    df = df.copy()
    for p in periods:
        if len(df) >= p:
            df[f"MA{p}"] = df["close"].rolling(p).mean()
    return df


def calc_ema(df: pd.DataFrame, period: int, col: str = "close") -> pd.Series:
    return df[col].ewm(span=period, adjust=False).mean()


def calc_macd(df: pd.DataFrame, fast: int = MACD_FAST,
              slow: int = MACD_SLOW, signal: int = MACD_SIGNAL) -> pd.DataFrame:
    df = df.copy()
    df["EMA_fast"] = calc_ema(df, fast)
    df["EMA_slow"] = calc_ema(df, slow)
    df["DIF"] = df["EMA_fast"] - df["EMA_slow"]
    df["DEA"] = df["DIF"].ewm(span=signal, adjust=False).mean()
    df["MACD"] = 2 * (df["DIF"] - df["DEA"])
    return df


def analyze_macd(df: pd.DataFrame) -> IndicatorResult:
    df = calc_macd(df)
    if len(df) < 2:
        return IndicatorResult("neutral", "hold", 50, {})
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    golden_cross = (prev["DIF"] <= prev["DEA"]) and (latest["DIF"] > latest["DEA"])
    death_cross = (prev["DIF"] >= prev["DEA"]) and (latest["DIF"] < latest["DEA"])
    if latest["DIF"] > latest["DEA"] and latest["MACD"] > 0:
        trend = "bullish"
        strength = min(80, 50 + abs(latest["MACD"]) / latest["close"] * 1000)
    elif latest["DIF"] < latest["DEA"] and latest["MACD"] < 0:
        trend = "bearish"
        strength = min(80, 50 + abs(latest["MACD"]) / latest["close"] * 1000)
    else:
        trend = "neutral"
        strength = 50
    signal = "buy" if golden_cross else ("sell" if death_cross else "hold")
    return IndicatorResult(trend=trend, signal=signal, strength=round(strength, 1),
                           details={"DIF": round(latest["DIF"], 4), "DEA": round(latest["DEA"], 4),
                                    "MACD": round(latest["MACD"], 4),
                                    "golden_cross": golden_cross, "death_cross": death_cross})


def calc_kdj(df: pd.DataFrame, n: int = KDJ_N, m1: int = KDJ_M1, m2: int = KDJ_M2) -> pd.DataFrame:
    df = df.copy()
    low_list = df["low"].rolling(n).min()
    high_list = df["high"].rolling(n).max()
    rsv = (df["close"] - low_list) / (high_list - low_list) * 100
    rsv = rsv.fillna(50)
    df["K"] = rsv.ewm(com=m1 - 1, adjust=False).mean()
    df["D"] = df["K"].ewm(com=m2 - 1, adjust=False).mean()
    df["J"] = 3 * df["K"] - 2 * df["D"]
    return df


def analyze_kdj(df: pd.DataFrame) -> IndicatorResult:
    df = calc_kdj(df)
    if len(df) < 2:
        return IndicatorResult("neutral", "hold", 50, {})
    latest = df.iloc[-1]
    prev = df.iloc[-2]
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
    return IndicatorResult(trend=trend, signal=signal, strength=round(strength, 1),
                           details={"K": round(k, 2), "D": round(d, 2), "J": round(j, 2),
                                    "golden_cross": golden_cross, "death_cross": death_cross})


def calc_rsi(df: pd.DataFrame, period: int = RSI_PERIOD) -> pd.DataFrame:
    df = df.copy()
    delta = df["close"].diff()
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


def calc_bollinger(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    df = df.copy()
    df["BOLL_MID"] = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std()
    df["BOLL_UP"] = df["BOLL_MID"] + std_dev * std
    df["BOLL_DN"] = df["BOLL_MID"] - std_dev * std
    df["BOLL_WIDTH"] = (df["BOLL_UP"] - df["BOLL_DN"]) / df["BOLL_MID"]
    return df


def analyze_bollinger(df: pd.DataFrame) -> IndicatorResult:
    df = calc_bollinger(df)
    if len(df) < 1:
        return IndicatorResult("neutral", "hold", 50, {})
    latest = df.iloc[-1]
    close, upper, mid, lower = latest["close"], latest["BOLL_UP"], latest["BOLL_MID"], latest["BOLL_DN"]
    if close >= upper:
        return IndicatorResult("overbought", "sell", 75, {"position": "上轨上方"})
    elif close <= lower:
        return IndicatorResult("oversold", "buy", 75, {"position": "下轨下方"})
    elif close > mid:
        return IndicatorResult("bullish", "hold", 55, {"position": "中轨上方"})
    else:
        return IndicatorResult("bearish", "hold", 45, {"position": "中轨下方"})


def analyze_volume_price(df: pd.DataFrame) -> IndicatorResult:
    if len(df) < 5:
        return IndicatorResult("neutral", "hold", 50, {})
    recent = df.tail(5)
    price_pct = (recent["close"].iloc[-1] - recent["close"].iloc[-5]) / recent["close"].iloc[-5] * 100
    avg_vol_first = recent["volume"].iloc[:2].mean()
    avg_vol_last = recent["volume"].iloc[-3:].mean()
    vol_change = (avg_vol_last - avg_vol_first) / avg_vol_first * 100 if avg_vol_first > 0 else 0
    if price_pct > 2 and vol_change > 20:
        return IndicatorResult("bullish", "buy", 75, {"price_pct": round(price_pct, 2), "vol_pct": round(vol_change, 2), "pattern": "价涨量增-强势"})
    elif price_pct > 2 and vol_change < -10:
        return IndicatorResult("bullish_warning", "hold", 40, {"price_pct": round(price_pct, 2), "vol_pct": round(vol_change, 2), "pattern": "价涨量缩-背离"})
    elif price_pct < -2 and vol_change > 20:
        return IndicatorResult("bearish", "sell", 75, {"price_pct": round(price_pct, 2), "vol_pct": round(vol_change, 2), "pattern": "价跌量增-弱势"})
    elif price_pct < -2 and vol_change < -10:
        return IndicatorResult("bearish_stop", "hold", 40, {"price_pct": round(price_pct, 2), "vol_pct": round(vol_change, 2), "pattern": "价跌量缩-止跌"})
    else:
        return IndicatorResult("neutral", "hold", 50, {"price_pct": round(price_pct, 2), "vol_pct": round(vol_change, 2), "pattern": "量价正常"})


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    df = df.copy()
    df["tr"] = np.maximum(df["high"] - df["low"],
                           np.maximum(abs(df["high"] - df["close"].shift(1)),
                                      abs(df["low"] - df["close"].shift(1))))
    return df["tr"].ewm(span=period, adjust=False).mean()


def calc_breakout(df: pd.DataFrame, period: int = 20) -> Tuple[pd.Series, pd.Series]:
    return df["high"].rolling(period).max(), df["low"].rolling(period).min()


def comprehensive_analysis(df: pd.DataFrame) -> Dict[str, IndicatorResult]:
    return {"MACD": analyze_macd(df), "KDJ": analyze_kdj(df), "RSI": analyze_rsi(df),
            "Bollinger": analyze_bollinger(df), "VolumePrice": analyze_volume_price(df)}


def trend_score(df: pd.DataFrame) -> Dict[str, Any]:
    results = comprehensive_analysis(df)
    score = 50.0
    weights = {"MACD": 0.15, "KDJ": 0.15, "RSI": 0.10, "Bollinger": 0.10, "VolumePrice": 0.10}
    for name, r in results.items():
        w = weights.get(name, 0.1)
        if r.signal == "buy":
            score += r.strength * w
        elif r.signal == "sell":
            score -= (100 - r.strength) * w
    score = max(0, min(100, score))
    if score >= 70: rating = "强势看多"
    elif score >= 55: rating = "偏多"
    elif score >= 45: rating = "震荡"
    elif score >= 30: rating = "偏空"
    else: rating = "弱势看空"
    return {"score": round(score, 1), "rating": rating,
            "details": {k: v.signal for k, v in results.items()}}
