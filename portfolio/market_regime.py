"""市场状态识别器 — 大盘环境过滤"""
import pandas as pd
import numpy as np
from typing import Optional, Tuple


class MarketRegimeFilter:
    """
    市场状态识别 + 仓位比例建议

    核心逻辑：
      1. 大盘趋势（MA60, MA120）→ 牛/震/熊
      2. 市场波动率（指数ATR）→ 高波动减仓
      3. 成交量活跃度

    输出：允许仓位比例（0%, 50%, 100%）
    """

    def __init__(self,
                 short_ma: int = 60,
                 long_ma: int = 120,
                 bear_threshold: float = -0.20,
                 position_limits: dict = None):
        """
        Args:
            short_ma: 短期均线
            long_ma: 长期均线
            bear_threshold: 从高点回落阈值判定熊市
            position_limits: {regime: max_position_pct}
        """
        self.short_ma = short_ma
        self.long_ma = long_ma
        self.bear_threshold = bear_threshold
        self.position_limits = position_limits or {
            'bull': 1.0,
            'mild_bear': 0.5,
            'bear': 0.0,
            'panic': 0.0,
            'neutral': 0.5,
        }

    def identify_regime(self, index_data: pd.DataFrame) -> Tuple[str, float]:
        """
        识别市场状态

        核心逻辑（基于 A 股实战经验）：
          1. 均线排列为主：MA60 > MA120 + 价格在上 → bull
          2. MA60下方但MA120上方 → mild_bear（减仓）
          3. MA60和MA120双下方 → bear（空仓或极轻仓）
          4. 从高点回落超过 bear_threshold → bear 覆盖

        Args:
            index_data: 大盘指数 OHLCV DataFrame

        Returns:
            (regime, max_position_pct)
            regime: 'bull' | 'mild_bear' | 'bear' | 'panic'
        """
        if index_data.empty or len(index_data) < self.long_ma:
            return 'neutral', self.position_limits.get('neutral', 0.6)

        close = index_data['close']
        high = index_data['high']
        low = index_data['low']

        # 均线
        ma_short = close.rolling(self.short_ma).mean()
        ma_long = close.rolling(self.long_ma).mean()

        latest_close = close.iloc[-1]
        latest_ma_short = ma_short.iloc[-1]
        latest_ma_long = ma_long.iloc[-1]

        if pd.isna(latest_ma_short) or pd.isna(latest_ma_long):
            return 'neutral', self.position_limits.get('neutral', 0.6)

        # 从高点回撤（用1年滚动窗口，避免全时峰值误判）
        peak = close.rolling(252).max()
        drawdown = (close - peak) / peak
        current_dd = drawdown.iloc[-1]

        # 恐慌检测优先
        if self._is_panic(close, high, low):
            return 'panic', self.position_limits.get('panic', 0.0)

        # 从高点回落超过阈值 → bear（覆盖均线判断）
        if current_dd < self.bear_threshold:
            return 'bear', self.position_limits.get('bear', 0.0)

        # === 基于均线排列的判定 ===
        above_ma60 = latest_close > latest_ma_short
        above_ma120 = latest_close > latest_ma_long
        ma60_above_ma120 = latest_ma_short > latest_ma_long

        if above_ma60 and above_ma120 and ma60_above_ma120:
            # 多头排列：价格 > MA60 > MA120
            return 'bull', self.position_limits.get('bull', 1.0)

        if above_ma60 and above_ma120 and not ma60_above_ma120:
            # 价格在双均线上，但MA60 < MA120（死叉排列）
            # 如：大跌后反弹初期，MA60还未跟上 → 中性仓位
            return 'neutral', self.position_limits.get('neutral', 0.5)

        if above_ma60 and not above_ma120:
            # 价格在MA60之上，但MA120之下 → 反弹未过关键线
            return 'mild_bear', self.position_limits.get('mild_bear', 0.5)

        if not above_ma60 and above_ma120:
            # 跌破MA60但还在MA120之上 → 短期走弱
            return 'mild_bear', self.position_limits.get('mild_bear', 0.5)

        # not above_ma60 and not above_ma120
        # 价格在双均线之下 → 熊市
        return 'bear', self.position_limits.get('bear', 0.0)

    def _is_panic(self, close: pd.Series, high: pd.Series, low: pd.Series) -> bool:
        """检测恐慌模式：连续放量暴跌"""
        if len(close) < 20:
            return False
        recent = close.tail(10)
        returns = recent.pct_change().dropna()
        # 近10日超过5天跌幅>2%
        big_drops = sum(1 for r in returns if r < -0.02)
        # 且成交量放大
        vol = pd.Series(high - low, index=close.index).tail(10)
        vol_ma = vol.mean()
        vol_spike = sum(1 for v in vol if v > vol_ma * 1.5)
        return big_drops >= 5 and vol_spike >= 3

    def compute_volatility_adjustment(self, index_data: pd.DataFrame) -> float:
        """
        根据市场波动率计算调整系数

        Returns:
            adj_factor: 0.0 ~ 1.0，乘到仓位上
        """
        if index_data.empty or len(index_data) < 20:
            return 1.0

        high = index_data['high']
        low = index_data['low']
        close = index_data['close']

        # 指数ATR(20) / 指数价格
        tr = pd.DataFrame({
            'hl': high - low,
            'hc': (high - close.shift(1)).abs(),
            'lc': (low - close.shift(1)).abs(),
        }).max(axis=1)
        atr = tr.rolling(20).mean()
        atr_pct = atr / close

        current_vol = atr_pct.iloc[-1]
        # A股指数日波动正常约1-2%，超过3%视为高波动
        if pd.isna(current_vol):
            return 1.0
        if current_vol > 0.03:
            return 0.3  # 高波动 → 30%仓位
        elif current_vol > 0.025:
            return 0.5
        elif current_vol > 0.02:
            return 0.8
        return 1.0

    def get_max_position_pct(self, index_data: pd.DataFrame) -> float:
        """
        综合计算允许的最大仓位

        Args:
            index_data: 大盘指数数据

        Returns:
            max_pct: 0.0 ~ 1.0
        """
        regime, regime_pct = self.identify_regime(index_data)
        vol_adj = self.compute_volatility_adjustment(index_data)
        return min(regime_pct, vol_adj)

    def describe(self, index_data: pd.DataFrame) -> str:
        """返回市场状态描述"""
        regime, max_pct = self.identify_regime(index_data)
        vol_adj = self.compute_volatility_adjustment(index_data)
        final_pct = min(max_pct, vol_adj)

        regime_names = {
            'bull': 'bull',
            'mild_bear': 'mild_bear',
            'bear': 'bear',
            'panic': 'panic',
        }
        lines = [
            f'市场状态: {regime_names.get(regime, regime)}',
            f'允许仓位: {final_pct:.0%}',
            f'趋势限制: {max_pct:.0%}',
            f'波动调整: {vol_adj:.0%}',
        ]
        return '\n'.join(lines)
