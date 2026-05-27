# 回归通道策略 — 修复未来函数 + 分阶段卖出
import pandas as pd
import numpy as np
from strategy.MA.base import UpperChannelStrategy, Signal


class RegressionChannelWithVolumeStrategy(UpperChannelStrategy):
    """
    回归通道策略：线性回归（无未来函数）+ 放量突破买入 + 分阶段卖出。

    上轨 = 用 t-window 至 t-1 日的最高价预测第 t 天的上轨值。
    买入：放量突破预测上轨。
    卖出 — 按上轨斜率分阶段：
      - 斜率 < 0（下行期）：假突破退出。持仓从最高点回落超过 false_breakout_retrace（默认3%），判定为假突破，清仓。
      - 斜率 ≥ 0（上行/横盘期）：跌破预测上轨卖出。
    """

    def __init__(self, window: int = 60, vol_window: int = 20,
                 vol_multiplier: float = 1.5, profit_trail_pct: float = 0.10,
                 atr_period: int = 14, atr_multiplier: float = 2.0,
                 partial_profit_target: float = 0.08,
                 vol_active_threshold: float = 1.3,
                 std_multiplier: float = 0.0,
                 false_breakout_retrace: float = 0.03,
                 fundamental_checker=None, min_fundamental_score: float = 4.0,
                 vol_shrink_ratio: float = 0.8,
                 sell_delay_days: int = 2):
        name = f"LR_{window}_S{int(std_multiplier*10)}_V{int(vol_multiplier*10)}_PT{int(profit_trail_pct*100)}"
        super().__init__(name, vol_window, vol_multiplier, profit_trail_pct,
                         atr_period, atr_multiplier, partial_profit_target,
                         vol_active_threshold)
        self.window = window
        self.std_multiplier = std_multiplier
        self.false_breakout_retrace = false_breakout_retrace
        # 缩量卖出观察参数（volume < MA20_vol × vol_shrink_ratio → 缩量，延迟卖出）
        self.vol_shrink_ratio = vol_shrink_ratio
        self.sell_delay_days = sell_delay_days
        # 基本面买入点过滤
        self.fundamental_checker = fundamental_checker
        self.min_fundamental_score = min_fundamental_score
        self.fundamental_cache = None   # 由回测脚本在每只股票前设置
        self.symbol = None

    def _calc_upper(self, high: pd.Series) -> pd.Series:
        """无未来函数的回归通道上轨
        对每个点 i，用 high[i-window : i] 预测第 i 天的上轨值（不含当天信息）。
        """
        upper = pd.Series(index=high.index, dtype=float)
        self._slope = pd.Series(index=high.index, dtype=float)
        for i in range(len(high)):
            if i < self.window:
                continue
            x = np.arange(self.window)
            y = high.iloc[i - self.window:i].values
            z = np.polyfit(x, y, 1)
            self._slope.iloc[i] = z[0]
            reg_val = z[0] * (self.window - 1) + z[1]

            if self.std_multiplier > 0:
                reg_line = z[0] * x + z[1]
                residuals = y - reg_line
                std_resid = np.std(residuals)
                upper.iloc[i] = reg_val + self.std_multiplier * std_resid
            else:
                upper.iloc[i] = reg_val
        return upper

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """生成交易信号（无未来函数 + 分阶段卖出）"""
        signals = pd.Series(index=data.index, data=0, dtype=int)
        data = data.copy()

        upper = self._calc_upper(data['high'])
        data['upper'] = upper
        # 均量线
        vol_short = data['volume'].rolling(self.vol_window).mean()
        vol_long = data['volume'].rolling(252).mean()
        active = vol_short > vol_long * self.vol_active_threshold

        position_pct = 0.0
        buy_price = None
        highest_price = None
        pending_sell_days = 0

        for i in range(len(data)):
            if pd.isna(upper.iloc[i]) or pd.isna(vol_short.iloc[i]) or i == 0:
                continue

            close = data['close'].iloc[i]
            upper_val = upper.iloc[i]
            volume = data['volume'].iloc[i]
            vol_short_val = vol_short.iloc[i]
            slope_val = self._slope.iloc[i] if not pd.isna(self._slope.iloc[i]) else 0

            # 放量判定：活跃期自动放量 / 否则卡 1.5x 条件
            vol_ok = volume > vol_short_val * self.vol_multiplier
            if not vol_ok and not pd.isna(vol_long.iloc[i]) and not pd.isna(active.iloc[i]):
                vol_ok = active.iloc[i]

            if position_pct == 0.0:
                prev_close = data['close'].iloc[i - 1]
                prev_upper = upper.iloc[i - 1]

                if prev_close < prev_upper and close >= upper_val and vol_ok:
                    # 放量突破 → 收盘买入
                    if self.fundamental_checker is not None and self.fundamental_cache is not None and self.symbol is not None:
                        passes, _, _, desc = self.fundamental_checker.check_signal(
                            self.fundamental_cache, self.symbol, data.index[i], price=close
                        )
                        if not passes:
                            continue  # 基本面不过关，跳过本次买入
                    signals.iloc[i] = Signal.BUY.value
                    position_pct = 1.0
                    buy_price = close
                    highest_price = close
            else:
                if close > highest_price:
                    highest_price = close

                profit_pct = (close - buy_price) / buy_price
                exit_signal = None

                # === Retrace check (no volume filter — urgency override) ===
                if slope_val < 0:
                    retrace = (highest_price - close) / highest_price
                    if retrace > self.false_breakout_retrace:
                        exit_signal = Signal.SELL.value
                        pending_sell_days = 0

                # === Upper band breakdown check (with volume filter) ===
                if exit_signal is None:
                    if close <= upper_val:
                        # 缩量判定：成交量 < MA20_vol × vol_shrink_ratio
                        is_low_vol = volume < vol_short_val * self.vol_shrink_ratio

                        if is_low_vol and self.sell_delay_days > 0:
                            # 缩量跌破 → 进入观察期
                            pending_sell_days += 1
                            if pending_sell_days >= self.sell_delay_days:
                                # 观察期满，强制卖出
                                exit_signal = Signal.SELL.value
                                pending_sell_days = 0
                        else:
                            # 放量跌破 或 观察期中放量 → 立即执行
                            exit_signal = Signal.SELL.value
                            pending_sell_days = 0
                    elif pending_sell_days > 0:
                        # 价格回到上轨以上 → 取消卖出
                        pending_sell_days = 0

                if exit_signal is not None:
                    signals.iloc[i] = exit_signal
                    position_pct = 0.0
                    buy_price = None
                    highest_price = None
                    pending_sell_days = 0
                elif position_pct == 1.0 and profit_pct >= self.partial_profit_target:
                    # 部分止盈：满仓且盈利达标 → 卖半仓
                    signals.iloc[i] = Signal.SELL_HALF.value
                    position_pct = 0.5

        return signals
