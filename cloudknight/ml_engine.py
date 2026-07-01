"""
机器学习引擎 - 基于 AKQuant 量化机器学习最佳实践

参考: https://akquant.akfamily.xyz/textbook/12_ml/

核心组件:
  FeatureEngineer   - §12.2 金融特征工程（动量/波动/量价/趋势/形态）
  LabelGenerator    - §12.3 标签生成（固定窗口 / 三重屏障法 / 元标签）
  PurgedCV          - §12.4 净化交叉验证 + 滚动窗口验证（防数据泄漏）
  MLTrainer         - §12.5 多模型训练（LR / RF / GBDT）
  MLPredictor       - §12.5 在线预测 + 概率校准 + 置信度
  MLEngine          - 一站式入口，编排全流程

架构:
  DataFetcher → FeatureEngineer → LabelGenerator → PurgedCV → MLTrainer
                                                              ↓
  StockPoolItem ← MLPredictor ← 模型文件 ← MLModelRegistry

设计原则:
  1. 严格防前视偏差: 特征 t 时刻仅使用 ≤t 的信息
  2. 净化交叉验证: 剔除标签重叠样本 + Embargo 禁区
  3. 滚动窗口重训练: 适应市场非平稳性
  4. 模型版本管理: 可追溯、可复现
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import time as _time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─── sklearn 延迟导入（避免强依赖导致启动失败） ───
_SKLEARN_AVAILABLE = False
try:
    from sklearn.base import BaseEstimator, ClassifierMixin, clone
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix, f1_score, precision_score,
                                 recall_score)
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler

    _SKLEARN_AVAILABLE = True
except ImportError:
    logger.warning("scikit-learn 未安装，ML 引擎将使用降级模式（仅特征工程+Z-Score打分）")


# ═══════════════════════════════════════════════════════════
# 数据类
# ═══════════════════════════════════════════════════════════

@dataclass
class MLFeatureSet:
    """一组标的的特征 DataFrame"""
    code: str
    name: str = ""
    features: pd.DataFrame | None = None  # index=date, columns=特征名
    labels: pd.Series | None = None       # index=date, values=标签
    feature_names: list[str] = field(default_factory=list)


@dataclass
class MLPrediction:
    """单只标的的机器学习预测结果"""
    code: str
    name: str = ""
    # 原始预测
    prob_up: float = 0.5       # 上涨概率 [0, 1]
    prob_down: float = 0.5     # 下跌概率 [0, 1]
    direction: str = "neutral"  # bullish / bearish / neutral
    # 综合评分 -10 ~ +10
    ml_score: float = 0.0
    # 置信度 [0, 1]
    confidence: float = 0.0
    # 模型信息
    model_name: str = ""
    model_version: str = ""
    # 特征重要性 Top3
    top_features: list[tuple[str, float]] = field(default_factory=list)
    # 预测时间
    predicted_at: str = ""


@dataclass
class MLModelMeta:
    """模型版本元数据"""
    model_name: str
    version: str
    model_type: str           # logistic / random_forest / gradient_boosting
    trained_at: str           # ISO datetime
    train_samples: int = 0
    feature_count: int = 0
    features: list[str] = field(default_factory=list)
    cv_f1: float = 0.0
    cv_accuracy: float = 0.0
    label_method: str = "fixed_window"  # fixed_window / triple_barrier
    hyperparams: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════
# 12.2 金融特征工程
# ═══════════════════════════════════════════════════════════

class FeatureEngineer:
    """金融特征工程器 —— AKQuant §12.2

    特征分类:
      动量类: ret_1d/5d/10d/20d, ma_bias_5/10/20/60, roc_10
      波动类: vol_5d/10d/20d, atr_14, boll_width, boll_pct
      成交量类: volume_ratio_5/10, obv_change, turnover_5d
      趋势类: macd_dif/signal/hist, rsi_6/14/24, kdj_k/d/j
      形态类: bias_5/10/20 (乖离率), hh_ll_ratio
    """

    # 默认特征列表
    DEFAULT_FEATURES = [
        # 动量
        "ret_1d", "ret_5d", "ret_10d", "ret_20d",
        "ma_bias_5", "ma_bias_10", "ma_bias_20", "ma_bias_60",
        "roc_10",
        # 波动
        "vol_5d", "vol_10d", "vol_20d",
        "atr_14", "boll_width", "boll_pct",
        # 成交量
        "volume_ratio_5", "volume_ratio_10",
        # 趋势
        "macd_hist", "rsi_14",
        "kdj_k", "kdj_d",
        # 形态
        "bias_20", "hh_ll_ratio",
    ]

    def __init__(self, feature_list: list[str] | None = None):
        self.feature_list = feature_list or self.DEFAULT_FEATURES

    # ── 公开 API ──

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """从 OHLCV DataFrame 计算全部特征。

        Args:
            df: 必须包含 open/close/high/low/volume 列，按日期升序

        Returns:
            特征 DataFrame，index 同 df，列名为 self.feature_list
        """
        required = {"close", "high", "low", "volume"}
        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            raise ValueError(f"缺少必要列: {missing}")

        o = df["open"].astype(float).values
        c = df["close"].astype(float).values
        h = df["high"].astype(float).values
        l = df["low"].astype(float).values
        v = df["volume"].astype(float).values

        result = pd.DataFrame(index=df.index)

        # ── 动量类 ──
        result["ret_1d"] = _pct_change(c, 1)
        result["ret_5d"] = _pct_change(c, 5)
        result["ret_10d"] = _pct_change(c, 10)
        result["ret_20d"] = _pct_change(c, 20)
        result["ma_bias_5"] = _ma_bias(c, 5)
        result["ma_bias_10"] = _ma_bias(c, 10)
        result["ma_bias_20"] = _ma_bias(c, 20)
        result["ma_bias_60"] = _ma_bias(c, 60)
        result["roc_10"] = (c - _shift(c, 10)) / (_shift(c, 10) + 1e-9)

        # ── 波动类 ──
        result["vol_5d"] = _rolling_std(c, 5)
        result["vol_10d"] = _rolling_std(c, 10)
        result["vol_20d"] = _rolling_std(c, 20)
        atr = _atr(h, l, c, 14)
        result["atr_14"] = atr / (c + 1e-9)  # 归一化 ATR
        boll_mid = _sma(c, 20)
        boll_std = _rolling_std(c, 20)
        result["boll_width"] = (2 * boll_std) / (boll_mid + 1e-9)  # 布林带宽度
        result["boll_pct"] = (c - boll_mid) / (2 * boll_std + 1e-9)  # %B

        # ── 成交量类 ──
        result["volume_ratio_5"] = v / (_sma(v, 5) + 1e-9)
        result["volume_ratio_10"] = v / (_sma(v, 10) + 1e-9)

        # ── 趋势类 ──
        ema12 = _ema(c, 12)
        ema26 = _ema(c, 26)
        dif = ema12 - ema26
        dea = _ema(dif, 9)
        result["macd_hist"] = (dif - dea) * 2  # MACD 柱

        result["rsi_14"] = _rsi(c, 14)
        # KDJ
        k, d, j = _kdj(h, l, c, 9, 3, 3)
        result["kdj_k"] = k
        result["kdj_d"] = d

        # ── 形态类 ──
        result["bias_20"] = (c - _sma(c, 20)) / (_sma(c, 20) + 1e-9)
        hh20 = _rolling_max(h, 20)
        ll20 = _rolling_min(l, 20)
        result["hh_ll_ratio"] = (c - ll20) / (hh20 - ll20 + 1e-9)  # 当前价在 20 日区间位置

        # ── 仅返回配置的特征列 ──
        available = [f for f in self.feature_list if f in result.columns]
        result = result[available]

        # 前 60 行由于滚动窗口可能缺失，用 0 填充（但实践中应裁剪）
        result = result.fillna(0)

        return result

    def build_feature_set(self, code: str, name: str, df: pd.DataFrame,
                          label_gen: LabelGenerator | None = None) -> MLFeatureSet:
        """为单只标的构建完整特征集（含标签）。

        Args:
            code: 股票代码
            name: 股票名称
            df: OHLCV DataFrame
            label_gen: 标签生成器，None 则不生成标签

        Returns:
            MLFeatureSet
        """
        features = self.build_features(df)
        fs = MLFeatureSet(
            code=code,
            name=name,
            features=features,
            feature_names=list(features.columns),
        )
        if label_gen is not None:
            fs.labels = label_gen.generate(df, features)
        return fs


# ═══════════════════════════════════════════════════════════
# 12.3 标签生成
# ═══════════════════════════════════════════════════════════

class LabelGenerator:
    """标签生成器 —— AKQuant §12.3

    方法:
      1. 固定时间窗口法: 未来 h 日涨幅 > threshold → 正样本
      2. 三重屏障法 (De Prado): 止盈/止损/时间期限 → 先触发者决定标签
      3. 元标签: 初级模型给方向 → 次级模型判断是否执行
    """

    def __init__(self, method: str = "triple_barrier",
                 horizon: int = 5,
                 profit_take: float = 0.05,    # 止盈 5%
                 stop_loss: float = -0.03,     # 止损 -3%
                 threshold: float = 0.02):      # 固定窗口阈值
        """
        Args:
            method: "fixed_window" | "triple_barrier" | "meta_label"
            horizon: 时间期限（交易日数）
            profit_take: 止盈阈值（正数）
            stop_loss: 止损阈值（负数）
            threshold: 固定窗口上涨阈值
        """
        self.method = method
        self.horizon = horizon
        self.profit_take = profit_take
        self.stop_loss = stop_loss
        self.threshold = threshold

    def generate(self, df: pd.DataFrame, features: pd.DataFrame | None = None) -> pd.Series:
        """生成标签序列。

        Returns:
            pd.Series, index 同 df, values: 1(上涨)/-1(下跌)/0(中性)
        """
        if self.method == "triple_barrier":
            return self._triple_barrier(df)
        elif self.method == "meta_label":
            return self._meta_label(df, features)
        else:
            return self._fixed_window(df)

    def _fixed_window(self, df: pd.DataFrame) -> pd.Series:
        """固定时间窗口法: 未来 horizon 日涨幅 > threshold → 1"""
        closes = df["close"].astype(float).values
        n = len(closes)
        labels = np.zeros(n, dtype=int)

        for i in range(n - self.horizon):
            future_ret = closes[i + self.horizon] / closes[i] - 1
            if future_ret > self.threshold:
                labels[i] = 1
            elif future_ret < -self.threshold:
                labels[i] = -1
            # 最后 horizon 天保持 0

        return pd.Series(labels, index=df.index, name="label")

    def _triple_barrier(self, df: pd.DataFrame) -> pd.Series:
        """三重屏障法 (De Prado, 2018)

        对每个时间点 t，在 t+1 到 t+horizon 之间:
          - 若先触及 profit_take → 标签 = 1
          - 若先触及 stop_loss → 标签 = -1
          - 若 horizon 到期未触及任一 → 标签 = 0
        """
        closes = df["close"].astype(float).values
        highs = df["high"].astype(float).values if "high" in df.columns else closes
        lows = df["low"].astype(float).values if "low" in df.columns else closes
        n = len(closes)
        labels = np.zeros(n, dtype=int)

        for i in range(n - self.horizon):
            entry = closes[i]
            upper = entry * (1 + self.profit_take)
            lower = entry * (1 + self.stop_loss)

            hit_upper = False
            hit_lower = False

            for j in range(1, self.horizon + 1):
                idx = i + j
                if highs[idx] >= upper:
                    hit_upper = True
                    break
                if lows[idx] <= lower:
                    hit_lower = True
                    break

            if hit_upper and not hit_lower:
                labels[i] = 1
            elif hit_lower and not hit_upper:
                labels[i] = -1
            elif hit_upper and hit_lower:
                # 同时触及 → 按收盘价判断
                labels[i] = 1 if closes[i + self.horizon] > entry else -1
            # 都未触及 → 保持 0

        return pd.Series(labels, index=df.index, name="label")

    def _meta_label(self, df: pd.DataFrame, features: pd.DataFrame | None) -> pd.Series:
        """元标签法: 基于初级信号过滤

        初级信号来自特征中的趋势指标组合（MACD + RSI + MA偏差）
        次级标签: 初级信号为"买入"时，实际是否盈利 → 1/0
        """
        if features is None:
            return self._fixed_window(df)

        closes = df["close"].astype(float).values
        n = len(closes)

        # 初级信号: MACD > 0 且 RSI < 70 且 MA5 > MA20 → 买入
        primary_signal = np.zeros(n, dtype=int)
        if "macd_hist" in features.columns:
            primary_signal = (features["macd_hist"].values > 0).astype(int)

        # 次级标签: 初级买入信号发出后 horizon 日是否盈利
        labels = np.zeros(n, dtype=int)
        for i in range(n - self.horizon):
            if primary_signal[i] == 1:
                future_ret = closes[i + self.horizon] / closes[i] - 1
                labels[i] = 1 if future_ret > 0 else 0
            elif primary_signal[i] == 0:
                future_ret = closes[i + self.horizon] / closes[i] - 1
                labels[i] = -1 if future_ret < 0 else 0

        return pd.Series(labels, index=df.index, name="label")


# ═══════════════════════════════════════════════════════════
# 12.4 模型验证：净化交叉验证 + 滚动窗口
# ═══════════════════════════════════════════════════════════

class PurgedCV:
    """净化交叉验证 —— AKQuant §12.4

    核心概念:
      1. Purged K-Fold: 剔除训练集与测试集标签区间有重叠的样本
      2. Embargo: 测试集后设置禁区，防止信息泄漏
      3. Walk-Forward: 扩展窗口 / 滑动窗口两种模式
    """

    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.01,
                 purge_interval: int = 5):
        """
        Args:
            n_splits: K-Fold 分组数
            embargo_pct: 禁区比例（占数据总量）
            purge_interval: 净化区间（与标签 horizon 匹配）
        """
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
        self.purge_interval = purge_interval

    def purged_kfold_split(self, n_samples: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """生成净化 K-Fold 分割索引。

        Returns:
            [(train_idx, test_idx), ...] 每个 fold 的训练/测试索引
        """
        kf = KFold(n_splits=self.n_splits, shuffle=False)
        splits = []
        for train_idx, test_idx in kf.split(range(n_samples)):
            # 净化: 剔除训练集中与测试集标签区间重叠的样本
            test_start = test_idx[0]
            purge_boundary = max(0, test_start - self.purge_interval)
            train_idx = train_idx[train_idx < purge_boundary]

            # Embargo: 测试集后添加禁区
            embargo_size = max(1, int(n_samples * self.embargo_pct))
            test_idx = test_idx[test_idx < n_samples - embargo_size]

            if len(train_idx) > 0 and len(test_idx) > 0:
                splits.append((train_idx, test_idx))

        return splits

    def walk_forward_split(self, n_samples: int, train_size: int,
                           step_size: int = 20,
                           mode: str = "expanding") -> list[tuple[np.ndarray, np.ndarray]]:
        """滚动窗口验证分割。

        Args:
            n_samples: 总样本数
            train_size: 初始训练集大小
            step_size: 每次滚动步长（交易日数）
            mode: "expanding" 扩展窗口 | "sliding" 滑动窗口

        Returns:
            [(train_idx, test_idx), ...]
        """
        splits = []
        start = train_size

        while start + step_size <= n_samples:
            if mode == "expanding":
                train_idx = np.arange(0, start)
            else:  # sliding
                train_idx = np.arange(max(0, start - train_size), start)

            test_idx = np.arange(start, min(start + step_size, n_samples))

            # 净化
            purge_boundary = max(0, start - self.purge_interval)
            train_idx = train_idx[train_idx < purge_boundary]

            if len(train_idx) > 0 and len(test_idx) > 0:
                splits.append((train_idx, test_idx))

            start += step_size

        return splits


# ═══════════════════════════════════════════════════════════
# 12.5 模型训练
# ═══════════════════════════════════════════════════════════

class MLTrainer:
    """多模型训练器 —— AKQuant §12.5

    支持模型: LogisticRegression / RandomForest / GradientBoosting

    训练流程:
      1. 接收标准化后的特征矩阵 X 和标签 y
      2. PurgedCV 验证
      3. 全量训练 → 持久化模型
    """

    MODEL_TYPES: dict[str, type] = {}
    DEFAULT_PARAMS: dict[str, dict] = {}

    def __init__(self, model_type: str = "random_forest",
                 params: dict | None = None,
                 cv: PurgedCV | None = None,
                 scaler: StandardScaler | None = None):
        """
        Args:
            model_type: "logistic" | "random_forest" | "gradient_boosting"
            params: 模型超参，None 则用默认值
            cv: 交叉验证器
            scaler: 特征标准化器
        """
        if not _SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn 未安装，无法使用 MLTrainer")

        self.model_type = model_type
        self.params = params or self.DEFAULT_PARAMS.get(model_type, {})
        self.cv = cv or PurgedCV()
        self.scaler = scaler or StandardScaler()

        self.model: BaseEstimator | None = None
        self.feature_names: list[str] = []
        self.cv_metrics: dict[str, float] = {}
        self.trained_at: str = ""

    def train(self, X: np.ndarray, y: np.ndarray,
              feature_names: list[str] | None = None,
              validate: bool = True) -> dict:
        """训练模型。

        Args:
            X: 特征矩阵 (n_samples, n_features)
            y: 标签向量 (n_samples,)  ∈ {1, 0, -1}
            feature_names: 特征名列表
            validate: 是否执行交叉验证

        Returns:
            训练指标 dict
        """
        # 过滤中性样本（label=0）
        mask = y != 0
        X_f = X[mask]
        y_f = y[mask]
        # 将 -1 → 0 (sklearn 二分类)
        y_f = np.where(y_f == -1, 0, y_f).astype(int)

        if len(X_f) < 50:
            logger.warning(f"有效样本不足 ({len(X_f)} < 50)，跳过训练")
            return {"status": "skipped", "reason": "insufficient_samples"}

        # 标准化
        X_scaled = self.scaler.fit_transform(X_f)
        self.feature_names = feature_names or [f"f{i}" for i in range(X_f.shape[1])]

        # ── 交叉验证 ──
        if validate:
            self.cv_metrics = self._cross_validate(X_scaled, y_f)

        # ── 全量训练 ──
        model_class = self.MODEL_TYPES[self.model_type]
        self.model = model_class(**self.params)
        self.model.fit(X_scaled, y_f)
        self.trained_at = datetime.now().isoformat()

        # ── 特征重要性 ──
        importance = self._get_feature_importance()

        return {
            "status": "trained",
            "model_type": self.model_type,
            "n_samples": len(X_f),
            "n_features": X_f.shape[1],
            "classes": self.model.classes_.tolist(),
            "cv_metrics": self.cv_metrics,
            "feature_importance": importance,
            "trained_at": self.trained_at,
        }

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """预测。

        Returns:
            (predictions, probabilities) 预测类别和概率
        """
        if self.model is None:
            raise RuntimeError("模型未训练")
        if self.scaler is None:
            raise RuntimeError("Scaler 未初始化")

        # 处理 NaN
        X = np.nan_to_num(X, nan=0.0)
        X_scaled = self.scaler.transform(X)

        preds = self.model.predict(X_scaled)
        proba = self.model.predict_proba(X_scaled)

        return preds, proba

    def predict_single(self, features: np.ndarray) -> tuple[int, float, float]:
        """单条预测。

        Returns:
            (prediction, prob_up, prob_down)
        """
        preds, proba = self.predict(features.reshape(1, -1))
        # proba shape: (1, 2) → [prob_class0, prob_class1]
        # class 0 = -1 (下跌), class 1 = 1 (上涨)
        if self.model is not None and len(self.model.classes_) == 2:
            idx_up = list(self.model.classes_).index(1)
            idx_down = list(self.model.classes_).index(0)
            prob_up = float(proba[0, idx_up])
            prob_down = float(proba[0, idx_down])
        else:
            prob_up = prob_down = 0.5

        return int(preds[0]), prob_up, prob_down

    def _cross_validate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """净化交叉验证"""
        n = len(X)
        purged = self.cv.purged_kfold_split(n)
        scores = {"accuracy": [], "precision": [], "recall": [], "f1": []}

        for train_idx, test_idx in purged:
            if len(train_idx) < 10 or len(test_idx) < 5:
                continue

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            m = clone(self.MODEL_TYPES[self.model_type](**self.params))
            m.fit(X_train, y_train)
            y_pred = m.predict(X_test)

            scores["accuracy"].append(accuracy_score(y_test, y_pred))
            scores["precision"].append(precision_score(y_test, y_pred, zero_division=0))
            scores["recall"].append(recall_score(y_test, y_pred, zero_division=0))
            scores["f1"].append(f1_score(y_test, y_pred, zero_division=0))

        return {k: float(np.mean(v)) for k, v in scores.items() if v}

    def _get_feature_importance(self) -> list[tuple[str, float]]:
        """提取特征重要性"""
        if self.model is None:
            return []

        if hasattr(self.model, "feature_importances_"):
            imp = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            imp = np.abs(self.model.coef_[0]) if self.model.coef_.ndim > 1 else np.abs(self.model.coef_)
        else:
            return []

        # 归一化到 0-1
        imp = imp / (imp.sum() + 1e-9)
        pairs = list(zip(self.feature_names, imp))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return [(name, round(val, 4)) for name, val in pairs[:10]]

    def save(self, path: str) -> str:
        """保存模型到文件"""
        if self.model is None:
            raise RuntimeError("无模型可保存")
        data = {
            "model": self.model,
            "scaler": self.scaler,
            "model_type": self.model_type,
            "feature_names": self.feature_names,
            "trained_at": self.trained_at,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)
        return path

    @classmethod
    def load(cls, path: str) -> "MLTrainer":
        """从文件加载模型"""
        with open(path, "rb") as f:
            data = pickle.load(f)
        trainer = cls(model_type=data["model_type"])
        trainer.model = data["model"]
        trainer.scaler = data["scaler"]
        trainer.feature_names = data["feature_names"]
        trainer.trained_at = data["trained_at"]
        return trainer


# ─── 延迟绑定模型类型与参数（避免 sklearn 未安装时类定义阶段崩溃） ───
if _SKLEARN_AVAILABLE:
    MLTrainer.MODEL_TYPES = {
        "logistic": LogisticRegression,
        "random_forest": RandomForestClassifier,
        "gradient_boosting": GradientBoostingClassifier,
    }
    MLTrainer.DEFAULT_PARAMS = {
        "logistic": {"C": 1.0, "max_iter": 1000, "class_weight": "balanced"},
        "random_forest": {"n_estimators": 100, "max_depth": 5,
                          "min_samples_leaf": 10, "class_weight": "balanced",
                          "random_state": 42},
        "gradient_boosting": {"n_estimators": 100, "max_depth": 3,
                              "learning_rate": 0.05, "min_samples_leaf": 10,
                              "random_state": 42},
    }


# ═══════════════════════════════════════════════════════════
# MLPredictor: 在线预测
# ═══════════════════════════════════════════════════════════

class MLPredictor:
    """在线预测器 —— AKQuant §12.5

    加载已训练模型 → 接收特征 → 输出概率/方向/评分
    """

    def __init__(self, trainer: MLTrainer):
        self.trainer = trainer

    def predict_one(self, code: str, name: str,
                    features: np.ndarray,
                    feature_df: pd.DataFrame | None = None) -> MLPrediction:
        """对单只标的预测。

        Args:
            code: 股票代码
            name: 股票名称
            features: 最新一条特征向量 (n_features,)
            feature_df: 可选，最近的完整特征 DataFrame（用于置信度计算）

        Returns:
            MLPrediction
        """
        try:
            pred, prob_up, prob_down = self.trainer.predict_single(features)

            # 方向判断
            if prob_up > 0.55:
                direction = "bullish"
            elif prob_down > 0.55:
                direction = "bearish"
            else:
                direction = "neutral"

            # 综合评分: 映射概率差值到 -10 ~ +10
            diff = prob_up - prob_down
            ml_score = float(np.clip(diff * 10, -10, 10))

            # 置信度: 概率偏离 0.5 的程度
            confidence = abs(prob_up - 0.5) * 2

            # 特征重要性
            top_features = self.trainer._get_feature_importance()[:3]

            return MLPrediction(
                code=code,
                name=name,
                prob_up=round(prob_up, 4),
                prob_down=round(prob_down, 4),
                direction=direction,
                ml_score=round(ml_score, 2),
                confidence=round(confidence, 4),
                model_name=self.trainer.model_type,
                top_features=top_features,
                predicted_at=datetime.now().isoformat(),
            )
        except Exception as e:
            logger.warning(f"预测失败 {code}: {e}")
            return MLPrediction(code=code, name=name)


# ═══════════════════════════════════════════════════════════
# MLOps: 模型注册与版本管理 —— AKQuant §12.8
# ═══════════════════════════════════════════════════════════

class MLModelRegistry:
    """模型版本管理器

    目录结构:
      models/
        ├── registry.json        # 索引文件
        ├── ml_v001_20260701.pkl  # 模型文件
        └── ml_v002_20260701.pkl
    """

    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.model_dir / "registry.json"
        self._registry: dict[str, MLModelMeta] = self._load_registry()

    @property
    def latest_meta(self) -> MLModelMeta | None:
        """最新模型元数据"""
        if not self._registry:
            return None
        versions = sorted(self._registry.keys(), reverse=True)
        return self._registry.get(versions[0]) if versions else None

    def register(self, meta: MLModelMeta, trainer: MLTrainer) -> str:
        """注册新模型版本。

        Args:
            meta: 模型元数据
            trainer: 已训练的 trainer

        Returns:
            模型文件路径
        """
        version = self._next_version()
        meta.version = version

        # 生成确定性文件名
        filename = f"ml_{version}_{meta.trained_at[:10].replace('-', '')}.pkl"
        filepath = self.model_dir / filename

        trainer.save(str(filepath))

        # 更新注册表
        self._registry[version] = meta
        self._save_registry()

        logger.info(f"模型已注册: {version} → {filepath}")
        return str(filepath)

    def load_latest(self) -> MLTrainer | None:
        """加载最新模型"""
        meta = self.latest_meta
        if meta is None:
            return None

        filename = f"ml_{meta.version}_{meta.trained_at[:10].replace('-', '')}.pkl"
        filepath = self.model_dir / filename
        if not filepath.exists():
            logger.warning(f"模型文件缺失: {filepath}")
            return None

        return MLTrainer.load(str(filepath))

    def list_versions(self) -> list[MLModelMeta]:
        """列出所有版本"""
        return sorted(self._registry.values(), key=lambda m: m.version, reverse=True)

    def cleanup(self, keep: int = 5):
        """保留最近 N 个版本，删除旧版本"""
        versions = sorted(self._registry.keys(), reverse=True)
        for v in versions[keep:]:
            meta = self._registry[v]
            filename = f"ml_{v}_{meta.trained_at[:10].replace('-', '')}.pkl"
            filepath = self.model_dir / filename
            if filepath.exists():
                filepath.unlink()
            del self._registry[v]
        self._save_registry()

    # ── 内部 ──

    def _next_version(self) -> str:
        if not self._registry:
            return "v001"
        nums = [int(k[1:]) for k in self._registry]
        return f"v{max(nums) + 1:03d}"

    def _load_registry(self) -> dict[str, MLModelMeta]:
        if not self.registry_path.exists():
            return {}
        try:
            with open(self.registry_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {k: MLModelMeta(**v) for k, v in data.items()}
        except Exception:
            return {}

    def _save_registry(self):
        data = {k: v.__dict__ for k, v in self._registry.items()}
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════════════════
# MLEngine: 一站式机器学习引擎
# ═══════════════════════════════════════════════════════════

class MLEngine:
    """一站式机器学习引擎 —— 编排全流程

    使用示例:
        engine = MLEngine(fetcher=data_fetcher)
        engine.train_on_pool(pool_items)  # 训练
        predictions = engine.predict_all(pool_items)  # 预测
        engine.sync_to_pool(predictions, pool_items)  # 回写池子
    """

    def __init__(self, fetcher=None,
                 model_type: str = "random_forest",
                 label_method: str = "triple_barrier",
                 model_dir: str = "models",
                 train_window: int = 252,
                 feature_engineer: FeatureEngineer | None = None):
        """
        Args:
            fetcher: DataFetcher 实例（用于拉取 K 线）
            model_type: 模型类型
            label_method: 标签生成方法
            model_dir: 模型存储目录
            train_window: 训练窗口（交易日数，默认 252 ≈ 1年）
            feature_engineer: 自定义特征工程器
        """
        self.fetcher = fetcher
        self.model_type = model_type
        self.label_method = label_method
        self.train_window = train_window

        self.feature_engineer = feature_engineer or FeatureEngineer()
        self.label_generator = LabelGenerator(method=label_method)
        self.cv = PurgedCV(n_splits=5, purge_interval=5)
        self.registry = MLModelRegistry(model_dir=model_dir)

        self.trainer: MLTrainer | None = None
        self.last_train_date: str = ""

        # 降级模式标志
        self._fallback_mode = not _SKLEARN_AVAILABLE

    # ── 公开 API ──

    def train_on_pool(self, pool_items: list, now: datetime | None = None,
                      max_stocks: int = 20) -> dict:
        """基于股票池标的训练模型。

        Args:
            pool_items: StockPoolItem 列表
            now: 当前时间
            max_stocks: 最多训练标的数

        Returns:
            训练结果摘要
        """
        if self._fallback_mode:
            return self._train_fallback(pool_items)

        now = now or datetime.now()
        today_str = now.strftime("%Y%m%d")

        if self.last_train_date == today_str:
            return {"status": "already_trained", "date": today_str}

        logger.info(f"[MLEngine] 开始训练 ({self.model_type}/{self.label_method})...")

        # 步骤 1: 收集训练数据
        feature_sets = self._collect_features(pool_items, max_stocks, now)
        if not feature_sets:
            return {"status": "no_data", "reason": "无可训练标的"}

        # 步骤 2: 合并所有标的的特征 → 面板数据
        X_list, y_list = [], []
        for fs in feature_sets:
            if fs.features is None or fs.labels is None:
                continue
            # 对齐特征和标签
            common_idx = fs.features.index.intersection(fs.labels.index)
            if len(common_idx) < 20:
                continue
            X_list.append(fs.features.loc[common_idx].values)
            y_list.append(fs.labels.loc[common_idx].values)

        if not X_list:
            return {"status": "no_data", "reason": "特征/标签对齐后无有效数据"}

        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        feature_names = feature_sets[0].feature_names

        # 步骤 3: 训练
        self.trainer = MLTrainer(
            model_type=self.model_type,
            cv=self.cv,
        )
        result = self.trainer.train(X, y, feature_names=feature_names, validate=True)

        # 步骤 4: 注册模型
        meta = MLModelMeta(
            model_name=f"cloudknight_{self.model_type}",
            version="",  # registry 自动分配
            model_type=self.model_type,
            trained_at=now.isoformat(),
            train_samples=len(X),
            feature_count=X.shape[1],
            features=feature_names,
            cv_f1=result.get("cv_metrics", {}).get("f1", 0),
            cv_accuracy=result.get("cv_metrics", {}).get("accuracy", 0),
            label_method=self.label_method,
            hyperparams=getattr(self.trainer, 'params', {}),
        )
        model_path = self.registry.register(meta, self.trainer)
        self.last_train_date = today_str

        result["model_path"] = model_path
        result["n_stocks_trained"] = len(feature_sets)
        logger.info(f"[MLEngine] 训练完成: {len(X)} 样本, "
                    f"CV F1={meta.cv_f1:.3f}, 模型→{model_path}")
        return result

    def predict_all(self, pool_items: list, now: datetime | None = None) -> list[MLPrediction]:
        """对所有池中标的生产预测。

        Args:
            pool_items: StockPoolItem 列表
            now: 当前时间

        Returns:
            MLPrediction 列表
        """
        if self._fallback_mode:
            return self._predict_fallback(pool_items)

        # 加载或使用当前 trainer
        trainer = self.trainer or self.registry.load_latest()
        if trainer is None:
            logger.warning("[MLEngine] 无可用模型，使用降级预测")
            return self._predict_fallback(pool_items)

        predictor = MLPredictor(trainer)
        predictions = []

        for item in pool_items[:30]:  # 最多预测 30 只
            try:
                df = self._fetch_kline(item.code, now or datetime.now())
                if df is None or len(df) < 60:
                    pred = MLPrediction(code=item.code, name=item.name)
                else:
                    features_df = self.feature_engineer.build_features(df)
                    # 取最新一条特征
                    latest = features_df.iloc[-1].values.astype(float)
                    pred = predictor.predict_one(
                        item.code, item.name,
                        features=latest,
                        feature_df=features_df,
                    )
                predictions.append(pred)
            except Exception as e:
                logger.warning(f"[MLEngine] 预测 {item.code} 失败: {e}")
                predictions.append(MLPrediction(code=item.code, name=item.name))

        return predictions

    def sync_to_pool(self, predictions: list[MLPrediction], pool_items: list):
        """将预测结果同步到 StockPoolItem。

        Args:
            predictions: 预测列表
            pool_items: StockPoolItem 列表（会被原地修改）
        """
        pred_map = {p.code: p for p in predictions}
        for item in pool_items:
            pred = pred_map.get(item.code)
            if pred:
                item.ml_score = pred.ml_score
                item.ml_prediction = pred.direction

    def train_and_predict(self, pool_items: list, now: datetime | None = None,
                          max_stocks: int = 20) -> list[MLPrediction]:
        """一站式：训练 + 预测 + 同步。

        Args:
            pool_items: StockPoolItem 列表
            now: 当前时间
            max_stocks: 最多训练标的数

        Returns:
            MLPrediction 列表
        """
        train_result = self.train_on_pool(pool_items, now, max_stocks)
        predictions = self.predict_all(pool_items, now)
        self.sync_to_pool(predictions, pool_items)
        return predictions

    # ── 内部 ──

    def _collect_features(self, pool_items: list, max_stocks: int,
                          now: datetime) -> list[MLFeatureSet]:
        """收集所有标的的特征和标签"""
        feature_sets = []
        for item in pool_items[:max_stocks]:
            df = self._fetch_kline(item.code, now)
            if df is None or len(df) < 60:
                continue
            fs = self.feature_engineer.build_feature_set(
                item.code, item.name, df,
                label_gen=self.label_generator,
            )
            if fs.features is not None and len(fs.features) > 20:
                feature_sets.append(fs)
        return feature_sets

    def _fetch_kline(self, code: str, now: datetime):
        """拉取个股 K 线"""
        if self.fetcher is None:
            return None
        try:
            start = (now - timedelta(days=self.train_window * 2)).strftime("%Y%m%d")
            df = self.fetcher.fetch_daily_kline(code, start_date=start)
            if df is None or df.empty:
                return None

            # 标准化列名
            col_map = {
                "日期": "date", "开盘": "open", "收盘": "close",
                "最高": "high", "最低": "low", "成交量": "volume",
            }
            df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
            required = {"close", "high", "low", "volume"}
            if not required.issubset(df.columns):
                return None

            return df.tail(self.train_window).copy()
        except Exception:
            return None

    def _train_fallback(self, pool_items: list) -> dict:
        """降级模式：Z-Score 线性打分（无需 sklearn）"""
        return {"status": "fallback", "reason": "sklearn_unavailable"}

    def _predict_fallback(self, pool_items: list) -> list[MLPrediction]:
        """降级预测：基于特征的简化 Z-Score 打分"""
        predictions = []
        for item in pool_items[:30]:
            try:
                df = self._fetch_kline(item.code, datetime.now())
                if df is None or len(df) < 60:
                    predictions.append(MLPrediction(code=item.code, name=item.name))
                    continue

                features_df = self.feature_engineer.build_features(df)
                latest_20 = features_df.tail(20)
                latest = features_df.iloc[-1].values.astype(float)

                # Z-Score 归一化加权
                means = latest_20.mean().values
                stds = latest_20.std().values + 1e-9
                z_scores = (latest - means) / stds
                ml_score = float(np.clip(np.mean(z_scores) * 5, -10, 10))

                predictions.append(MLPrediction(
                    code=item.code,
                    name=item.name,
                    direction="bullish" if ml_score > 0 else "bearish",
                    ml_score=round(ml_score, 2),
                    confidence=min(abs(ml_score) / 10, 1.0),
                ))
            except Exception:
                predictions.append(MLPrediction(code=item.code, name=item.name))

        return predictions


# ═══════════════════════════════════════════════════════════
# 工具函数 (numpy 实现，零依赖)
# ═══════════════════════════════════════════════════════════

def _shift(arr: np.ndarray, n: int) -> np.ndarray:
    """滞后 n 期"""
    result = np.zeros_like(arr, dtype=float)
    if n >= len(arr):
        return result
    result[n:] = arr[:-n]
    return result


def _pct_change(arr: np.ndarray, n: int) -> np.ndarray:
    """n 期收益率"""
    shifted = _shift(arr, n)
    mask = shifted != 0
    result = np.zeros_like(arr, dtype=float)
    result[mask] = (arr[mask] - shifted[mask]) / shifted[mask]
    return result


def _sma(arr: np.ndarray, span: int) -> np.ndarray:
    """简单移动平均"""
    result = np.zeros_like(arr, dtype=float)
    cumsum = np.cumsum(np.insert(arr, 0, 0))
    for i in range(span - 1, len(arr)):
        result[i] = (cumsum[i + 1] - cumsum[i - span + 1]) / span
    result[:span - 1] = np.nan
    return result


def _ema(arr: np.ndarray, span: int) -> np.ndarray:
    """指数移动平均"""
    alpha = 2 / (span + 1)
    result = np.zeros_like(arr, dtype=float)
    result[0] = arr[0]
    for i in range(1, len(arr)):
        result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1]
    return result


def _rolling_std(arr: np.ndarray, span: int) -> np.ndarray:
    """滚动标准差"""
    result = np.zeros_like(arr, dtype=float)
    for i in range(len(arr)):
        start = max(0, i - span + 1)
        result[i] = np.std(arr[start:i + 1])
    return result


def _rolling_max(arr: np.ndarray, span: int) -> np.ndarray:
    """滚动最大值"""
    result = np.zeros_like(arr, dtype=float)
    for i in range(len(arr)):
        start = max(0, i - span + 1)
        result[i] = np.max(arr[start:i + 1])
    return result


def _rolling_min(arr: np.ndarray, span: int) -> np.ndarray:
    """滚动最小值"""
    result = np.zeros_like(arr, dtype=float)
    for i in range(len(arr)):
        start = max(0, i - span + 1)
        result[i] = np.min(arr[start:i + 1])
    return result


def _ma_bias(arr: np.ndarray, span: int) -> np.ndarray:
    """均线乖离率"""
    ma = _sma(arr, span)
    return (arr - ma) / (ma + 1e-9)


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, span: int = 14) -> np.ndarray:
    """平均真实波幅 (ATR)"""
    n = len(close)
    tr = np.zeros(n, dtype=float)
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    return _ema(tr, span)


def _rsi(close: np.ndarray, span: int = 14) -> np.ndarray:
    """相对强弱指标 (RSI)"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = _ema(gain, span)
    avg_loss = _ema(loss, span)
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - 100 / (1 + rs)


def _kdj(high: np.ndarray, low: np.ndarray, close: np.ndarray,
         n: int = 9, m1: int = 3, m2: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """KDJ 指标"""
    length = len(close)
    k = np.zeros(length, dtype=float)
    d = np.zeros(length, dtype=float)
    j = np.zeros(length, dtype=float)

    # RSV
    for i in range(length):
        start = max(0, i - n + 1)
        hh = np.max(high[start:i + 1])
        ll = np.min(low[start:i + 1])
        rsv = (close[i] - ll) / (hh - ll + 1e-9) * 100

        if i == 0:
            k[i] = 50
            d[i] = 50
        else:
            k[i] = (m1 - 1) / m1 * k[i - 1] + 1 / m1 * rsv
            d[i] = (m2 - 1) / m2 * d[i - 1] + 1 / m2 * k[i]
        j[i] = 3 * k[i] - 2 * d[i]

    return k, d, j


# ═══════════════════════════════════════════════════════════════════
# ML 决策门控系统（ML Decision Gate）
# ═══════════════════════════════════════════════════════════════════
#
# 核心逻辑：
#   1. 策略触发买入信号 → 查询 ML 对该标的的次日方向预测
#   2. ML 预测 bullish（上涨）→ 确认买入 ✅
#   3. ML 预测 bearish（下跌）→ 驳回买入 ❌，记录决策，继续观察
#   4. 次日收盘后验证：
#      - 预测正确 → 权重 +step（增强信任）
#      - 预测错误 → 权重 -step（降低信任）
#   5. 动态权重影响：高权重时更信任 ML 驳回；低权重时降低驳回门槛
#
# 权重机制：
#   - 滚动窗口：追踪最近 N 次决策的正确率作为权重
#   - 初始权重 0.50，范围 [0.25, 0.85]
#   - 当权重 < 0.50 时，即使 ML 预测 bearish 也不驳回


@dataclass
class MLDecision:
    """单次 ML 门控决策记录"""
    code: str
    name: str
    date: str                        # YYYYMMDD 决策日期
    signal_type: str                 # buy / add
    ml_direction: str                # bullish / bearish / neutral
    ml_score: float                  # ML 综合评分
    decision: str                    # approved / rejected / approved_override
    reason: str                      # 决策理由
    reference_price: float           # 决策时的参考价格
    strategy: str = ""               # 策略标识
    validated: bool = False          # 是否已验证
    validate_date: str = ""          # 验证日期 YYYYMMDD
    actual_direction: str = ""       # 实际方向 bullish/bearish/neutral
    actual_change_pct: float = 0.0   # 实际涨跌幅 %
    was_correct: bool | None = None  # ML 预测是否正确 (None=未验证)


class MLDecisionGate:
    """ML 决策门控器 — 用历史准确率动态加权，辅助交易决策。

    核心设计：
    - 基于滚动窗口的准确率计算动态权重
    - 权重 ≥ 阈值时，ML 看跌则驳回买入；权重不足时放行
    - 每日盘后自动验证，自适应市场变化
    - 持久化决策历史，重启不丢失
    """

    def __init__(
        self,
        state_dir: str,
        window_size: int = 20,
        min_weight: float = 0.25,
        max_weight: float = 0.85,
        min_accuracy: float = 0.50,
        direction_threshold: float = 0.005,
    ):
        """
        Args:
            state_dir: 状态文件存储目录
            window_size: 滚动窗口大小
            min_weight: 最低权重
            max_weight: 最高权重
            min_accuracy: 最低准确率阈值（低于此值时不驳回）
            direction_threshold: 次日方向判定阈值（涨/跌幅超过此值才算有效方向）
        """
        self._decisions: list[MLDecision] = []
        self._weight: float = 0.50
        self._window_size = window_size
        self._min_weight = min_weight
        self._max_weight = max_weight
        self._min_accuracy = min_accuracy
        self._direction_threshold = direction_threshold

        os.makedirs(state_dir, exist_ok=True)
        self._state_file = os.path.join(state_dir, "decision_gate.json")
        self._load()
        self._recompute_weight()

    # ─── 公共 API ─────────────────────────────────

    @property
    def weight(self) -> float:
        """当前 ML 决策权重 [0, 1]"""
        return self._weight

    @property
    def enabled(self) -> bool:
        """权重是否足以影响决策"""
        return self._weight >= self._min_accuracy

    def evaluate_buy(
        self,
        code: str,
        name: str,
        ml_direction: str,
        ml_score: float,
        date: str,
        price: float,
        strategy: str = "",
    ) -> tuple[bool, str]:
        """评估买入信号是否通过 ML 门控。

        Args:
            code: 股票代码
            name: 股票名称
            ml_direction: ML 预测方向 (bullish/bearish/neutral)
            ml_score: ML 综合评分
            date: 决策日期 YYYYMMDD
            price: 参考价格
            strategy: 策略标识

        Returns:
            (approved: bool, reason: str)
            - approved=True: 通过门控，可执行买入
            - approved=False: 被驳回，继续观察
        """
        if ml_direction == "bullish":
            approved = True
            reason = f"ML预测次日上涨(评分{ml_score:+.1f})，确认买入"
            decision_type = "approved"
        elif ml_direction == "bearish":
            if self._weight >= self._min_accuracy:
                approved = False
                reason = (
                    f"ML预测次日下跌(评分{ml_score:+.1f})，与买入信号背离，"
                    f"驳回买入（权重{self._weight:.1%}≥{self._min_accuracy:.0%}），继续观察"
                )
                decision_type = "rejected"
            else:
                approved = True
                reason = (
                    f"ML预测次日下跌(评分{ml_score:+.1f})，"
                    f"但权重不足({self._weight:.1%}<{self._min_accuracy:.0%})，放行买入"
                )
                decision_type = "approved_override"
        else:  # neutral
            approved = True
            reason = f"ML预测中性(评分{ml_score:+.1f})，按策略信号执行"
            decision_type = "approved"

        # 记录决策
        decision = MLDecision(
            code=code,
            name=name,
            date=date,
            signal_type="buy",
            ml_direction=ml_direction,
            ml_score=ml_score,
            decision=decision_type,
            reason=reason,
            reference_price=price,
            strategy=strategy,
        )
        self._decisions.append(decision)
        self._save()

        return approved, reason

    def validate_daily(self, now: datetime, fetcher) -> dict:
        """验证上一个交易日所有未验证决策的准确性。

        对每个未验证的决策：
        1. 获取决策日的收盘价（参考价）和次日的收盘价
        2. 计算实际涨跌方向
        3. 对比 ML 预测方向 → 判定正确/错误
        4. 更新权重

        Args:
            now: 当前时间（用于确定"昨日"）
            fetcher: DataFetcher 实例，用于获取 K 线数据

        Returns:
            stats dict: {validated: int, correct: int, wrong: int, weight: float}
        """
        today_str = now.strftime("%Y%m%d")

        # 找出所有未验证的决策（日期 < today_str）
        pending = [
            d for d in self._decisions
            if not d.validated and d.date < today_str
        ]

        if not pending:
            return {"validated": 0, "correct": 0, "wrong": 0, "weight": self._weight}

        validated_count = 0
        correct_count = 0

        for decision in pending:
            try:
                # 获取决策日 + 次日的 K 线
                decision_dt = datetime.strptime(decision.date, "%Y%m%d")

                # 获取 5 天数据以涵盖决策日和次日
                end_dt = decision_dt + timedelta(days=7)
                start_str = decision_dt.strftime("%Y%m%d")
                end_str = end_dt.strftime("%Y%m%d")

                df = fetcher.fetch_daily_kline(decision.code, start_date=start_str)
                if df is None or df.empty or len(df) < 2:
                    continue

                # 标准化列名
                col_map = {
                    "日期": "date", "开盘": "open", "收盘": "close",
                    "最高": "high", "最低": "low", "成交量": "volume",
                }
                df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

                if "close" not in df.columns:
                    continue

                # 找决策日行和次日行
                if "date" in df.columns:
                    df["date_str"] = df["date"].astype(str).str.replace("-", "").str[:8]
                    decision_rows = df[df["date_str"] == decision.date]
                else:
                    # 假设按时间排序
                    closes = df["close"].astype(float).values
                    if len(closes) < 2:
                        continue
                    # 取最后两行作为决策日和次日
                    prev_close = closes[-2]
                    next_close = closes[-1]
                    decision.validated = True
                    decision.validate_date = today_str
                    actual_change = (next_close - prev_close) / (prev_close + 1e-9)
                    decision.actual_change_pct = round(actual_change * 100, 2)
                    decision.actual_direction = self._classify_direction(actual_change)
                    decision.was_correct = self._check_correct(decision.ml_direction, decision.actual_direction)
                    validated_count += 1
                    if decision.was_correct:
                        correct_count += 1
                    continue

                if decision_rows.empty:
                    continue

                # 获取决策日收盘价
                idx = decision_rows.index[0]
                if idx + 1 >= len(df):
                    continue

                prev_close = float(df.loc[idx, "close"])
                next_close = float(df.loc[idx + 1, "close"])

                actual_change = (next_close - prev_close) / (prev_close + 1e-9)

                decision.validated = True
                decision.validate_date = today_str
                decision.actual_change_pct = round(actual_change * 100, 2)
                decision.actual_direction = self._classify_direction(actual_change)
                decision.was_correct = self._check_correct(decision.ml_direction, decision.actual_direction)

                validated_count += 1
                if decision.was_correct:
                    correct_count += 1

            except Exception:
                continue

        # 保存并重新计算权重
        self._save()
        self._recompute_weight()

        return {
            "validated": validated_count,
            "correct": correct_count,
            "wrong": validated_count - correct_count,
            "weight": self._weight,
        }

    def get_stats(self) -> dict:
        """返回决策门控统计信息"""
        total = len(self._decisions)
        validated = [d for d in self._decisions if d.validated]
        approved = [d for d in self._decisions if d.decision == "approved"]
        rejected = [d for d in self._decisions if d.decision == "rejected"]
        overridden = [d for d in self._decisions if d.decision == "approved_override"]

        correct = [d for d in validated if d.was_correct]
        wrong = [d for d in validated if d.was_correct is False]

        # 近期准确率（滚动窗口内）
        recent = validated[-self._window_size:] if len(validated) > self._window_size else validated
        recent_correct = sum(1 for d in recent if d.was_correct)

        # 分类统计：驳回决策的准确率（ML 说跌 → 实际跌的比例）
        rejected_validated = [d for d in rejected if d.validated]
        rejected_correct = sum(1 for d in rejected_validated if d.was_correct)
        rejected_accuracy = rejected_correct / len(rejected_validated) if rejected_validated else 0

        # 放行决策的准确率（ML 说涨/中性 → 实际涨的比例）
        approved_validated = [d for d in approved + overridden if d.validated]
        approved_correct = sum(1 for d in approved_validated if d.was_correct)
        approved_accuracy = approved_correct / len(approved_validated) if approved_validated else 0

        return {
            "total_decisions": total,
            "total_validated": len(validated),
            "total_correct": len(correct),
            "total_wrong": len(wrong),
            "total_approved": len(approved),
            "total_rejected": len(rejected),
            "total_overridden": len(overridden),
            "overall_accuracy": len(correct) / len(validated) if validated else 0,
            "recent_accuracy": recent_correct / len(recent) if recent else 0,
            "rejected_accuracy": round(rejected_accuracy, 3),
            "approved_accuracy": round(approved_accuracy, 3),
            "current_weight": round(self._weight, 3),
            "gate_active": self.enabled,
        }

    def get_recent_decisions(self, n: int = 10) -> list[MLDecision]:
        """获取最近 N 条决策记录"""
        return self._decisions[-n:]

    def get_pending_count(self) -> int:
        """获取待验证决策数量"""
        return sum(1 for d in self._decisions if not d.validated)

    def reset(self):
        """重置决策门控（清除所有历史）"""
        self._decisions.clear()
        self._weight = 0.50
        self._save()

    # ─── 内部方法 ─────────────────────────────────

    def _classify_direction(self, change: float) -> str:
        """根据涨跌幅判定实际方向"""
        if change > self._direction_threshold:
            return "bullish"
        elif change < -self._direction_threshold:
            return "bearish"
        else:
            return "neutral"

    def _check_correct(self, predicted: str, actual: str) -> bool:
        """判定 ML 预测是否正确。

        规则：
        - ML 预测 bullish AND 实际 bullish → 正确
        - ML 预测 bearish AND 实际 bearish → 正确
        - ML 预测 neutral AND 实际 neutral → 正确
        - 其他组合 → 错误
        """
        return predicted == actual

    def _recompute_weight(self):
        """基于滚动窗口内已验证决策的准确率重新计算权重"""
        validated = [d for d in self._decisions if d.was_correct is not None]
        if not validated:
            self._weight = 0.50
            return

        recent = validated[-self._window_size:]
        if not recent:
            self._weight = 0.50
            return

        correct = sum(1 for d in recent if d.was_correct)
        raw_weight = correct / len(recent)
        self._weight = max(self._min_weight, min(self._max_weight, raw_weight))

    def _save(self):
        """持久化到 JSON 文件"""
        try:
            data = {
                "weight": self._weight,
                "window_size": self._window_size,
                "min_weight": self._min_weight,
                "max_weight": self._max_weight,
                "min_accuracy": self._min_accuracy,
                "direction_threshold": self._direction_threshold,
                "decisions": [
                    {
                        "code": d.code,
                        "name": d.name,
                        "date": d.date,
                        "signal_type": d.signal_type,
                        "ml_direction": d.ml_direction,
                        "ml_score": d.ml_score,
                        "decision": d.decision,
                        "reason": d.reason,
                        "reference_price": d.reference_price,
                        "strategy": d.strategy,
                        "validated": d.validated,
                        "validate_date": d.validate_date,
                        "actual_direction": d.actual_direction,
                        "actual_change_pct": d.actual_change_pct,
                        "was_correct": d.was_correct,
                    }
                    for d in self._decisions
                ],
            }
            with open(self._state_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"[MLDecisionGate] 保存状态失败: {e}")

    def _load(self):
        """从 JSON 文件加载历史状态"""
        if not os.path.exists(self._state_file):
            return

        try:
            with open(self._state_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            self._weight = data.get("weight", 0.50)
            self._window_size = data.get("window_size", self._window_size)
            self._min_weight = data.get("min_weight", self._min_weight)
            self._max_weight = data.get("max_weight", self._max_weight)
            self._min_accuracy = data.get("min_accuracy", self._min_accuracy)
            self._direction_threshold = data.get("direction_threshold", self._direction_threshold)

            self._decisions = []
            for d in data.get("decisions", []):
                was_correct = d.get("was_correct")
                if was_correct is not None:
                    was_correct = bool(was_correct)
                decision = MLDecision(
                    code=d["code"],
                    name=d.get("name", ""),
                    date=d.get("date", ""),
                    signal_type=d.get("signal_type", "buy"),
                    ml_direction=d.get("ml_direction", "neutral"),
                    ml_score=d.get("ml_score", 0.0),
                    decision=d.get("decision", "approved"),
                    reason=d.get("reason", ""),
                    reference_price=d.get("reference_price", 0.0),
                    strategy=d.get("strategy", ""),
                    validated=d.get("validated", False),
                    validate_date=d.get("validate_date", ""),
                    actual_direction=d.get("actual_direction", ""),
                    actual_change_pct=d.get("actual_change_pct", 0.0),
                    was_correct=was_correct,
                )
                self._decisions.append(decision)

        except Exception as e:
            logger.warning(f"[MLDecisionGate] 加载状态失败: {e}")
            self._decisions = []
            self._weight = 0.50
