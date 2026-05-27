# 回测引擎
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from strategy.MA.base import Signal
import os


@dataclass
class BacktestResult:
    """回测结果"""
    total_return: float = 0.0
    annual_return: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    final_value: float = 0.0
    equity_curve: pd.Series = field(default_factory=pd.Series)
    trades: List = field(default_factory=list)


class BacktestEngine:
    """回测引擎"""

    def __init__(self, initial_capital: float = 100000, commission: float = 0.00025,
                 stamp_duty: float = 0.0005):
        """
        Args:
            initial_capital: 初始资金
            commission: 手续费比例（默认万2.5）
            stamp_duty: 印花税比例（默认万5，仅卖出，A股2023.8减半后）
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.stamp_duty = stamp_duty

    def run(self, data: pd.DataFrame, strategy) -> BacktestResult:
        """
        运行回测

        Args:
            data: OHLCV数据
            strategy: 策略实例

        Returns:
            BacktestResult: 回测结果
        """
        # 生成信号
        signals = strategy.generate_signals(data)
        data = data.copy()
        data['signal'] = signals

        # 初始化
        cash = self.initial_capital
        shares = 0
        equity_curve = []
        trades = []

        for idx, (date, row) in enumerate(data.iterrows()):
            signal = row['signal']
            price = row['close']

            if signal == Signal.BUY.value and shares == 0:
                # 买入（收盘价成交，A股买入仅佣金无印花税）
                shares = cash / price * (1 - self.commission)
                cash = 0
                trades.append({
                    'date': str(date.date()),
                    'type': 'BUY',
                    'price': price,
                    'shares': shares
                })

            if signal == Signal.SELL_HALF.value and shares > 0:
                # 卖出半仓（佣金+印花税）
                sell_shares = shares / 2
                cash += sell_shares * price * (1 - self.commission - self.stamp_duty)
                shares = sell_shares
                trades.append({
                    'date': str(date.date()),
                    'type': 'SELL_HALF',
                    'price': price,
                    'shares': sell_shares
                })

            if signal == Signal.SELL.value and shares > 0:
                # 卖出（佣金+印花税）
                cash += shares * price * (1 - self.commission - self.stamp_duty)
                trades.append({
                    'date': str(date.date()),
                    'type': 'SELL',
                    'price': price,
                    'shares': shares
                })
                shares = 0

            # 记录当日资产
            equity = cash + shares * price
            equity_curve.append(equity)

        # 最终资产
        final_value = cash + shares * data.iloc[-1]['close']

        # 计算指标
        equity_series = pd.Series(equity_curve, index=data.index)
        returns = equity_series.pct_change().dropna()

        result = BacktestResult()
        result.final_value = final_value
        result.total_trades = len(trades)
        result.trades = trades
        result.equity_curve = equity_series

        # 计算收益率
        result.total_return = (final_value - self.initial_capital) / self.initial_capital

        # 年化收益率
        days = (data.index[-1] - data.index[0]).days
        if days > 0:
            result.annual_return = (final_value / self.initial_capital) ** (365 / days) - 1

        # 最大回撤
        peak = equity_series.expanding().max()
        drawdown = (equity_series - peak) / peak
        result.max_drawdown = drawdown.min()

        # 夏普比率
        if len(returns) > 0 and returns.std() > 0:
            risk_free = 0.03 / 252  # 日无风险利率
            excess_return = returns - risk_free
            result.sharpe_ratio = np.sqrt(252) * excess_return.mean() / returns.std()

        # 胜率：按交易周期配对，每个买入匹配到下次买入前的所有卖出
        buy_trades = [t for t in trades if t['type'] == 'BUY']
        sell_trades = [t for t in trades if t['type'] in ('SELL', 'SELL_HALF')]
        if len(buy_trades) > 0:
            wins = 0
            total = 0
            for i, bt in enumerate(buy_trades):
                buy_date = bt['date']
                next_buy_date = buy_trades[i + 1]['date'] if i + 1 < len(buy_trades) else None
                if next_buy_date:
                    related_sells = [st for st in sell_trades if buy_date <= st['date'] < next_buy_date]
                else:
                    related_sells = [st for st in sell_trades if st['date'] >= buy_date]
                if related_sells:
                    total_shares = sum(st['shares'] for st in related_sells)
                    if total_shares > 0:
                        avg_sell_price = sum(st['price'] * st['shares'] for st in related_sells) / total_shares
                        if avg_sell_price > bt['price']:
                            wins += 1
                        total += 1
            if total > 0:
                result.win_rate = wins / total

        return result

    def save_results(self, result: BacktestResult, data: pd.DataFrame, strategy_name: str, output_dir: str = 'results'):
        """保存结果"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        result.equity_curve.to_csv(f'{output_dir}/equity.csv')
        trades_df = pd.DataFrame(result.trades)
        trades_df.to_csv(f'{output_dir}/trades.csv', index=False)
        data.to_csv(f'{output_dir}/signals.csv')
