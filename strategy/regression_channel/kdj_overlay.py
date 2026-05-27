import pandas as pd
import numpy as np
from strategy.MA.base import Signal
from strategy.regression_channel.channel_strategy import RegressionChannelWithVolumeStrategy


class ChannelWithKDJStrategy(RegressionChannelWithVolumeStrategy):
    """
    回归通道 + KDJ 叠加策略

    买入:
      1. 回归通道: 放量突破上轨（原逻辑）
      2. KDJ: 30日收盘价线性回归斜率 > 0 且 J < 0（上升趋势中回调买入）

    卖出:
      - 通道买入: 原逻辑（假突破退出 / 跌破上轨 / 缩量观察期）
      - KDJ 买入: J > 80（超买回落），辅以 ATR 追踪止损
    """

    def __init__(self,
                 window: int = 40, vol_window: int = 20,
                 vol_multiplier: float = 1.5, profit_trail_pct: float = 0.10,
                 atr_period: int = 14, atr_multiplier: float = 2.0,
                 partial_profit_target: float = 0.08,
                 vol_active_threshold: float = 1.3,
                 std_multiplier: float = 0.0,
                 false_breakout_retrace: float = 0.03,
                 fundamental_checker=None, min_fundamental_score: float = 4.0,
                 vol_shrink_ratio: float = 0.8,
                 sell_delay_days: int = 2,
                 # KDJ 参数
                 kdj_period: int = 9,
                 kdj_overbought: float = 80.0,
                 kdj_oversold: float = 0.0,
                 kdj_slope_window: int = 30):
        super().__init__(window, vol_window, vol_multiplier, profit_trail_pct,
                         atr_period, atr_multiplier, partial_profit_target,
                         vol_active_threshold, std_multiplier,
                         false_breakout_retrace, fundamental_checker,
                         min_fundamental_score, vol_shrink_ratio, sell_delay_days)
        self.name = f"KDJ_{window}_{int(kdj_period)}_{int(kdj_slope_window)}"
        self.kdj_period = kdj_period
        self.kdj_overbought = kdj_overbought
        self.kdj_oversold = kdj_oversold
        self.kdj_slope_window = kdj_slope_window

    def _calc_kdj(self, high: pd.Series, low: pd.Series, close: pd.Series) -> tuple:
        """计算 KDJ 指标，返回 (K, D, J)"""
        period = self.kdj_period
        low_n = low.rolling(period).min()
        high_n = high.rolling(period).max()
        denom = (high_n - low_n).replace(0, np.nan)
        rsv = (close - low_n) / denom * 100

        k = pd.Series(index=close.index, dtype=float)
        d = pd.Series(index=close.index, dtype=float)
        j = pd.Series(index=close.index, dtype=float)

        first = rsv.first_valid_index()
        if first is None:
            return k, d, j
        k.loc[first] = 50.0
        d.loc[first] = 50.0

        for idx in close.index[close.index.get_loc(first) + 1:]:
            rsv_val = rsv.loc[idx]
            prev_k = k.loc[:idx].iloc[-2]
            prev_d = d.loc[:idx].iloc[-2]

            k_i = (2 / 3) * prev_k + (1 / 3) * rsv_val
            d_i = (2 / 3) * prev_d + (1 / 3) * k_i
            k.loc[idx] = k_i
            d.loc[idx] = d_i
            j.loc[idx] = 3 * k_i - 2 * d_i

        return k, d, j

    def _calc_close_slope(self, close: pd.Series) -> pd.Series:
        """窗口内收盘价线性回归斜率"""
        w = self.kdj_slope_window
        slope = pd.Series(index=close.index, dtype=float)
        for i in range(w, len(close)):
            y = close.iloc[i - w:i].values
            if np.all(np.isfinite(y)):
                slope.iloc[i] = np.polyfit(np.arange(w), y, 1)[0]
        return slope

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        signals = pd.Series(index=data.index, data=0, dtype=int)
        data = data.copy()

        # --- 通道 ---
        upper = self._calc_upper(data['high'])
        data['upper'] = upper

        # --- 成交量 ---
        vol_short = data['volume'].rolling(self.vol_window).mean()
        vol_long = data['volume'].rolling(252).mean()
        active = vol_short > vol_long * self.vol_active_threshold

        # --- KDJ ---
        k, d, j = self._calc_kdj(data['high'], data['low'], data['close'])
        close_slope = self._calc_close_slope(data['close'])

        # --- ATR ---
        atr = self._calc_atr(data['high'], data['low'], data['close'])

        position_pct = 0.0
        buy_price = None
        highest_price = None
        entry_type = None     # 'channel' | 'kdj'
        pending_sell_days = 0

        for i in range(len(data)):
            if pd.isna(upper.iloc[i]) or pd.isna(vol_short.iloc[i]) or i == 0:
                continue

            close = data['close'].iloc[i]
            upper_val = upper.iloc[i]
            volume = data['volume'].iloc[i]
            vol_short_val = vol_short.iloc[i]
            slope_val = self._slope.iloc[i] if not pd.isna(self._slope.iloc[i]) else 0

            j_val = j.iloc[i]
            slope_30 = close_slope.iloc[i]

            # 放量判定
            vol_ok = volume > vol_short_val * self.vol_multiplier
            if not vol_ok and not pd.isna(vol_long.iloc[i]) and not pd.isna(active.iloc[i]):
                vol_ok = active.iloc[i]

            if position_pct == 0.0:
                prev_close = data['close'].iloc[i - 1]
                prev_upper = upper.iloc[i - 1]

                # 条件 A: 通道突破买入
                if prev_close < prev_upper and close >= upper_val and vol_ok:
                    if self.fundamental_checker is not None and self.fundamental_cache is not None and self.symbol is not None:
                        passes, _, _, _ = self.fundamental_checker.check_signal(
                            self.fundamental_cache, self.symbol, data.index[i], price=close)
                        if passes:
                            signals.iloc[i] = Signal.BUY.value
                            position_pct = 1.0; buy_price = close; highest_price = close
                            entry_type = 'channel'
                    else:
                        signals.iloc[i] = Signal.BUY.value
                        position_pct = 1.0; buy_price = close; highest_price = close
                        entry_type = 'channel'

                # 条件 B: KDJ 回调买入（未通过条件 A 时）
                if position_pct == 0.0 and not pd.isna(j_val) and not pd.isna(slope_30):
                    if slope_30 > 0 and j_val < self.kdj_oversold:
                        signals.iloc[i] = Signal.BUY.value
                        position_pct = 1.0; buy_price = close; highest_price = close
                        entry_type = 'kdj'
            else:
                if close > highest_price:
                    highest_price = close

                profit_pct = (close - buy_price) / buy_price
                exit_signal = None

                # --- ATR 追踪止损（通用） ---
                atr_val = atr.iloc[i]
                if not pd.isna(atr_val) and atr_val > 0:
                    if close < highest_price - self.atr_multiplier * atr_val:
                        exit_signal = Signal.SELL.value

                if exit_signal is None and entry_type == 'channel':
                    # 通道卖出原逻辑
                    if slope_val < 0:
                        retrace = (highest_price - close) / highest_price
                        if retrace > self.false_breakout_retrace:
                            exit_signal = Signal.SELL.value
                            pending_sell_days = 0

                    if exit_signal is None:
                        if close <= upper_val:
                            is_low_vol = volume < vol_short_val * self.vol_shrink_ratio
                            if is_low_vol and self.sell_delay_days > 0:
                                pending_sell_days += 1
                                if pending_sell_days >= self.sell_delay_days:
                                    exit_signal = Signal.SELL.value
                                    pending_sell_days = 0
                            else:
                                exit_signal = Signal.SELL.value
                                pending_sell_days = 0
                        elif pending_sell_days > 0:
                            pending_sell_days = 0

                elif exit_signal is None and entry_type == 'kdj':
                    # KDJ 超买卖出（主力退出信号）
                    if not pd.isna(j_val) and j_val > self.kdj_overbought:
                        exit_signal = Signal.SELL.value

                if exit_signal is not None:
                    signals.iloc[i] = exit_signal
                    position_pct = 0.0; buy_price = None; highest_price = None
                    entry_type = None; pending_sell_days = 0
                elif entry_type == 'channel' and position_pct == 1.0 and profit_pct >= self.partial_profit_target:
                    signals.iloc[i] = Signal.SELL_HALF.value
                    position_pct = 0.5

        return signals
