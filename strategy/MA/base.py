# 策略基础模块
from enum import Enum
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class Signal(Enum):
    """交易信号枚举"""
    HOLD = 0
    BUY = 1
    SELL = -1
    SELL_HALF = 2    # 卖出半仓


class UpperChannelStrategy(ABC):
    """
    上轨通道策略基类 — 封装通用的买卖判断逻辑。
    子类只需实现 _calc_upper() 方法计算上轨，信号生成由基类统一完成。
    """

    def __init__(self, name: str, vol_window: int = 20,
                 vol_multiplier: float = 1.5, profit_trail_pct: float = 0.10,
                 atr_period: int = 14, atr_multiplier: float = 2.0,
                 partial_profit_target: float = 0.08,
                 vol_active_threshold: float = 1.3):
        self.name = name
        self.vol_window = vol_window
        self.vol_multiplier = vol_multiplier
        self.profit_trail_pct = profit_trail_pct
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.partial_profit_target = partial_profit_target
        self.vol_active_threshold = vol_active_threshold

    @abstractmethod
    def _calc_upper(self, high: pd.Series) -> pd.Series:
        """子类实现具体的上轨计算方法"""
        ...

    def _calc_atr(self, high: pd.Series, low: pd.Series, close: pd.Series,
                  period: int = None) -> pd.Series:
        """计算 ATR（平均真实波幅）"""
        if period is None:
            period = self.atr_period
        tr = pd.DataFrame({
            'hl': high - low,
            'hc': (high - close.shift(1)).abs(),
            'lc': (low - close.shift(1)).abs(),
        }).max(axis=1)
        return tr.rolling(period).mean()

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """生成交易信号（统一逻辑）

        卖出规则：
          1. ATR 追踪止盈 — 最高点回落 2×ATR 则清仓
          2. 跌破上轨 — 通道突破失败，清仓
          3. 部分止盈 — 盈利达目标（默认 +8%），先卖半仓
        """
        signals = pd.Series(index=data.index, data=0, dtype=int)
        data = data.copy()

        upper = self._calc_upper(data['high'])
        data['upper'] = upper
        # 均量线：短周期(20日) + 长周期(252日 ≈ 年)
        vol_short = data['volume'].rolling(self.vol_window).mean()
        vol_long = data['volume'].rolling(252).mean()
        # 活跃期判定：短均量显著高于年均量
        active = vol_short > vol_long * self.vol_active_threshold
        atr = self._calc_atr(data['high'], data['low'], data['close'])

        position_pct = 0.0   # 0=空仓, 0.5=半仓, 1.0=满仓
        buy_price = None
        highest_price = None

        for i in range(len(data)):
            if pd.isna(upper.iloc[i]) or pd.isna(vol_short.iloc[i]) or i == 0:
                continue

            close = data['close'].iloc[i]
            upper_val = upper.iloc[i]
            volume = data['volume'].iloc[i]
            vol_short_val = vol_short.iloc[i]

            # 放量判定：活跃期自动放量 / 否则卡 1.5x 条件
            vol_ok = volume > vol_short_val * self.vol_multiplier
            if not vol_ok and not pd.isna(vol_long.iloc[i]) and not pd.isna(active.iloc[i]):
                vol_ok = active.iloc[i]

            prev_close = data['close'].iloc[i - 1]
            prev_upper = upper.iloc[i - 1]

            if position_pct == 0.0:
                # 买入: 前一日未突破 + 当日突破 + 放量（或处于活跃期）
                if prev_close < prev_upper and close >= upper_val and vol_ok:
                    signals.iloc[i] = Signal.BUY.value
                    position_pct = 1.0
                    buy_price = close
                    highest_price = close
            else:
                # 更新最高价
                if close > highest_price:
                    highest_price = close

                profit_pct = (close - buy_price) / buy_price

                # 判断是否触发卖出（先检查，再决定是否部分止盈）
                exit_signal = None

                # 1) ATR 追踪止盈
                atr_val = atr.iloc[i]
                if not pd.isna(atr_val) and atr_val > 0:
                    stop_price = highest_price - self.atr_multiplier * atr_val
                    if close < stop_price:
                        exit_signal = Signal.SELL.value

                # 2) 跌破上轨（通道失效）
                if close <= upper_val:
                    exit_signal = Signal.SELL.value

                if exit_signal is not None:
                    signals.iloc[i] = exit_signal
                    position_pct = 0.0
                    buy_price = None
                    highest_price = None
                elif position_pct == 1.0 and profit_pct >= self.partial_profit_target:
                    # 3) 部分止盈：满仓且盈利达标 → 卖半仓
                    signals.iloc[i] = Signal.SELL_HALF.value
                    position_pct = 0.5
                    # highest_price 不变，剩余仓位继续追踪

        return signals


class Strategy(ABC):
    """策略基类（旧版兼容）"""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """生成交易信号"""
        pass
