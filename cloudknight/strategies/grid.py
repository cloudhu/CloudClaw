"""
网格交易策略 - 震荡市高抛低吸 [AKQuant]
核心：价格区间网格 + ATR动态网格宽度 + 仓位分档管理
"""

from .base import CloudKnightStrategy


class GridStrategy(CloudKnightStrategy):
    name = "grid"
    description = "网格交易 - 震荡区间高抛低吸，ATR动态网格"

    def __init__(self, params: dict | None = None):
        default = {
            "max_position_pct": 0.30,
            "grid_count": 5,  # 网格层数
            "grid_width_atr": 1.0,  # 每格ATR倍数
            "atr_period": 20,
            "center_ma_period": 60,  # 中心线MA周期
            "stop_loss_atr_mult": 3,  # 止损ATR倍数
            "rsi_oversold": 30,  # RSI超卖确认
            "rsi_overbought": 70,  # RSI超买确认
            "volatility_filter_min": 0.5,  # 最低波动率过滤（ATR/close）
        }
        default.update(params or {})
        super().__init__(default)
        # 记录当前持仓对应的网格层级
        self._grid_level: dict[str, int] = {}

    @property
    def warmup_period(self) -> int:
        return 120

    def on_bar(self, bar):
        symbol = bar.symbol
        closes = self.get_history_closes(symbol, 120)
        highs = self.get_history_highs(symbol, 120)
        lows = self.get_history_lows(symbol, 120)

        if len(closes) < 70:
            return

        # ATR 计算
        atr = self._calc_atr(highs, lows, closes, self._params["atr_period"])
        if atr <= 0:
            return

        # 波动率过滤：规避死股
        volatility = atr / self.close
        if volatility < self._params["volatility_filter_min"] / 100:
            return

        # 网格参数
        grid_count = self._params["grid_count"]
        grid_width = atr * self._params["grid_width_atr"]
        ma_center = self._calc_ma(closes, self._params["center_ma_period"])

        if not ma_center or ma_center <= 0:
            return

        # 以MA为中心，上下各 grid_count/2 层网格
        half_grids = grid_count // 2
        grid_step = grid_width / ma_center  # 价格百分比步长

        rsi = self._calc_rsi(closes, 14)
        has_pos = self.has_position(symbol)

        if not has_pos:
            # 建仓条件：价格在MA下方 + RSI超卖区域
            current_below_ma = self.close / ma_center - 1
            # 计算当前价格在MA下方的网格层级
            level_below = int(abs(current_below_ma) / grid_step) if current_below_ma < 0 else -1

            if level_below >= 1 and level_below <= half_grids and rsi < self._params["rsi_oversold"] + 5:
                # 根据网格深度决定仓位（越深仓位越大）
                position_pct = min(
                    self._params["max_position_pct"],
                    self._params["max_position_pct"] * (level_below / half_grids),
                )
                self.enter_long(
                    symbol,
                    position_pct,
                    f"网格建仓: Lv{-level_below} MA下方{abs(current_below_ma) * 100:.1f}% RSI={rsi:.1f}",
                )
                self._grid_level[symbol] = level_below
        else:
            pnl_pct = self.position_pnl_pct(symbol)
            current_diff = self.close / ma_center - 1
            entry_level = self._grid_level.get(symbol, 0)

            # ATR止损
            stop_atr = self._params["stop_loss_atr_mult"] * atr
            entry_price = self._entry_price.get(symbol, self.close)
            if self.close <= entry_price - stop_atr:
                self.exit_position(symbol, f"网格止损: 跌破{self._params['stop_loss_atr_mult']}倍ATR")
                self._grid_level.pop(symbol, None)
                return

            # 网格止盈：回到MA上方
            if current_diff > grid_step and entry_level > 0:
                self.exit_position(symbol, f"网格止盈: 回归MA +{pnl_pct:.1f}%")
                self._grid_level.pop(symbol, None)
                return

            # RSI超买止盈
            if rsi > self._params["rsi_overbought"] and pnl_pct > 3:
                self.reduce_position(symbol, 0.5, f"网格减持: RSI={rsi:.1f}超买")
                self._grid_level[symbol] = max(0, entry_level - 1)

            # 继续深跌加仓（网格下移）
            new_level = int(abs(current_diff) / grid_step) if current_diff < 0 else 0
            if new_level > entry_level and new_level <= half_grids and rsi < 40:
                add_pct = min(
                    self._params["max_position_pct"] * 0.2,
                    self._params["max_position_pct"] - pnl_pct / 100,  # simple guard
                )
                if add_pct > 0.05:
                    self.add_position(symbol, add_pct, f"网格加仓: Lv-{new_level} 深跌加码")
                    self._grid_level[symbol] = new_level

    # ── 诊股接口 ──────────────────────────────────────
    @staticmethod
    def diagnose(closes, highs, lows, volumes, opens, indicators, fund=None, cap=None) -> dict:
        """网格交易诊股评分：价格位置vs MA60、ATR波动率、RSI"""
        reasons, warnings = [], []
        score = 30

        cur = closes[-1]
        ma60 = indicators.get("ma60", 0)
        atr = indicators.get("atr14", 0)
        rsi = indicators.get("rsi", 50)

        # 价格位置 vs MA60
        if ma60 > 0:
            dist_ma60 = (cur - ma60) / ma60 * 100
            if -15 <= dist_ma60 <= -5:
                score += 25
                reasons.append(f"MA60下方{abs(dist_ma60):.1f}%，适合网格建仓")
            elif -5 < dist_ma60 <= 0:
                score += 15
                reasons.append(f"MA60下方{dist_ma60:+.1f}%")
            elif 0 < dist_ma60 <= 5:
                score += 5
                reasons.append(f"MA60上方{dist_ma60:+.1f}%")
            elif dist_ma60 > 15:
                score -= 10
                warnings.append(f"远离MA60上方({dist_ma60:.1f}%)，不适合网格")
            elif dist_ma60 < -20:
                score -= 5
                warnings.append(f"深度跌破MA60({dist_ma60:.1f}%)")
        else:
            warnings.append("MA60无数据")

        # ATR波动率（网格需要适度波动）
        if atr > 0 and cur > 0:
            atr_pct = atr / cur * 100
            if 1.5 <= atr_pct <= 6:
                score += 20
                reasons.append(f"ATR={atr_pct:.1f}%适合网格")
            elif 0.5 <= atr_pct < 1.5:
                score += 8
                warnings.append(f"ATR={atr_pct:.1f}%偏低，网格收益小")
            elif atr_pct > 6:
                score += 5
                warnings.append(f"ATR={atr_pct:.1f}%偏高，风险大")
            else:
                score -= 10
                warnings.append(f"ATR={atr_pct:.1f}%波动不足")
        else:
            score += 5

        # RSI超卖程度
        if 25 <= rsi <= 40:
            score += 25
            reasons.append(f"RSI={rsi:.0f}超卖区，适合低吸")
        elif 40 < rsi <= 55:
            score += 10
        elif rsi > 70:
            score -= 10
            warnings.append(f"RSI={rsi:.0f}超买，不宜建仓")

        score = max(0, min(100, score))
        match = len(reasons)
        total = 3

        if score >= 70:
            signal, rating = "buy", "适合网格"
        elif score >= 55:
            signal, rating = "buy", "可建网格"
        elif score >= 40:
            signal, rating = "hold", "观望"
        else:
            signal, rating = "sell", "不适合"

        return {
            "name": "网格交易",
            "key": "grid",
            "score": score,
            "signal": signal,
            "rating": rating,
            "match_count": match,
            "total_conditions": total,
            "reasons": reasons,
            "warnings": warnings,
            "auto_add_threshold": 65,
        }
