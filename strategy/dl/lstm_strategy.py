# LSTM策略模块 — 继承 UpperChannelStrategy 基类
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Optional, Literal
from strategy.MA.base import UpperChannelStrategy
import warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')


class LSTMStrategy(UpperChannelStrategy):
    """
    LSTM上轨策略：用LSTM预测上轨，买卖逻辑由基类统一处理。

    两种模式:
    - 'rolling' (默认): 每天用最近 train_window 天重训模型，精度高但慢
    - 'fast': 用全部数据只训练一次，速度快但精度稍低
    """

    def __init__(self, window: int = 40, lstm_units: int = 64, epochs: int = 50,
                 train_window: int = 200, mode: Literal['rolling', 'fast'] = 'fast',
                 vol_window: int = 20, vol_multiplier: float = 1.5,
                 profit_trail_pct: float = 0.10,
                 atr_period: int = 14, atr_multiplier: float = 2.0,
                 partial_profit_target: float = 0.08,
                 vol_active_threshold: float = 1.3):
        name = f"LSTM_{window}_{lstm_units}_E{epochs}_{mode}_V{int(vol_multiplier*10)}"
        super().__init__(name, vol_window, vol_multiplier, profit_trail_pct,
                         atr_period, atr_multiplier, partial_profit_target,
                         vol_active_threshold)
        self.window = window
        self.lstm_units = lstm_units
        self.epochs = epochs
        self.train_window = train_window
        self.mode = mode
        self.scaler: Optional[MinMaxScaler] = None

    def _calc_upper(self, high: pd.Series) -> pd.Series:
        values = self._lstm_predict_upper(high.values)
        return pd.Series(values, index=high.index)

    def _lstm_predict_upper(self, high: np.ndarray) -> np.ndarray:
        n = len(high)
        upper = np.full(n, np.nan)
        if n < self.window + 50:
            return upper

        if self.mode == 'fast':
            return self._predict_fast(high, n, upper)
        return self._predict_rolling(high, n, upper)

    def _predict_fast(self, high: np.ndarray, n: int, upper: np.ndarray) -> np.ndarray:
        """只训练一次，然后用训练好的模型预测全部"""
        scaler = MinMaxScaler(feature_range=(0, 1))
        high_scaled = scaler.fit_transform(high.reshape(-1, 1)).flatten()

        X, y = self._create_sequences(high_scaled, self.window)
        if len(X) < 10:
            return upper

        X = X.reshape((X.shape[0], X.shape[1], 1))
        model = self._build_model((self.window, 1))
        model.fit(X, y, epochs=self.epochs, batch_size=32, verbose=0)

        for i in range(self.window, n):
            seq = high_scaled[i - self.window:i].reshape(1, self.window, 1)
            pred = model.predict(seq, verbose=0)
            upper[i] = scaler.inverse_transform(pred)[0, 0]

        return upper

    def _predict_rolling(self, high: np.ndarray, n: int, upper: np.ndarray) -> np.ndarray:
        """滚动训练：每天用最近 train_window 天数据训练并预测下一天"""
        scaler = MinMaxScaler(feature_range=(0, 1))
        for i in range(self.window + self.train_window, n):
            train_start = max(0, i - self.train_window)
            high_seg = high[train_start:i]
            scaled = scaler.fit_transform(high_seg.reshape(-1, 1)).flatten()

            X, y = self._create_sequences(scaled, self.window)
            if len(X) < 10:
                continue
            X = X.reshape((X.shape[0], X.shape[1], 1))

            model = self._build_model((self.window, 1))
            model.fit(X, y, epochs=self.epochs, batch_size=16, verbose=0, shuffle=False)

            last_seq = scaled[-self.window:].reshape(1, self.window, 1)
            pred = model.predict(last_seq, verbose=0)
            upper[i] = scaler.inverse_transform(pred)[0, 0]

            tf.keras.backend.clear_session()

        return upper

    def _create_sequences(self, data: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(len(data) - window):
            X.append(data[i:i + window])
            y.append(data[i + window])
        return np.array(X), np.array(y)

    def _build_model(self, input_shape: tuple) -> Sequential:
        model = Sequential([
            LSTM(self.lstm_units, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(self.lstm_units, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
