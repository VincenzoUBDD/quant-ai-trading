"""投资组合结果数据类 + 绩效指标计算"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import pandas as pd
import numpy as np


@dataclass
class PortfolioTrade:
    """组合交易记录"""
    date: str                # 交易日期
    symbol: str              # 股票代码
    name: str = ''           # 股票名称
    direction: str = ''      # BUY | SELL | SELL_HALF
    price: float = 0.0       # 成交价格
    shares: float = 0.0      # 成交股数
    value: float = 0.0       # 成交金额
    reason: str = ''         # 交易原因（信号/风控/止损）


@dataclass
class PortfolioResult:
    """组合回测结果"""
    # 净值曲线
    equity_curve: pd.Series = field(default_factory=pd.Series)       # 每日净值
    benchmark_curve: pd.Series = field(default_factory=pd.Series)    # 基准（沪深300）

    # 持仓明细
    daily_positions: pd.DataFrame = field(default_factory=pd.DataFrame)  # 每日持股明细
    trades: List[PortfolioTrade] = field(default_factory=list)       # 交易记录

    # 绩效指标
    total_return: float = 0.0
    annual_return: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    volatility: float = 0.0          # 年化波动率

    # 组合特有指标
    avg_position_count: float = 0.0  # 平均持仓数
    turnover: float = 0.0            # 月均换手率
    sector_hhi: float = 0.0          # 板块集中度（HHI）
    stock_hhi: float = 0.0           # 个股集中度

    # 基准对比
    beta: float = 0.0
    alpha: float = 0.0
    information_ratio: float = 0.0
    tracking_error: float = 0.0

    # 风险指标
    max_consecutive_loss: int = 0    # 最大连续亏损次数
    var_95: float = 0.0              # VaR(95%)
    cvar_95: float = 0.0             # CVaR(95%)
    recovery_days: int = 0           # 最大回撤恢复天数
    win_rate: float = 0.0            # 交易胜率
    profit_loss_ratio: float = 0.0   # 盈亏比
    monthly_returns: pd.Series = field(default_factory=pd.Series)  # 月度收益

    # 资金曲线
    final_capital: float = 0.0
    peak_capital: float = 0.0


def compute_portfolio_metrics(equity_curve: pd.Series,
                              trades: List[PortfolioTrade],
                              daily_positions: pd.DataFrame,
                              benchmark_curve: Optional[pd.Series] = None,
                              risk_free_rate: float = 0.03) -> dict:
    """
    计算投资组合绩效指标

    Args:
        equity_curve: 每日净值
        trades: 交易记录
        daily_positions: 每日持仓（含各只股票的市值）
        benchmark_curve: 基准净值
        risk_free_rate: 年化无风险利率

    Returns:
        dict: {metric_name: value}
    """
    metrics = {}
    if len(equity_curve) < 10:
        return metrics

    # 日收益率
    returns = equity_curve.pct_change().dropna()
    if len(returns) < 5:
        return metrics

    total_days = (equity_curve.index[-1] - equity_curve.index[0]).days
    n_years = total_days / 365.0

    # === 基础指标 ===
    final_value = equity_curve.iloc[-1]
    initial_value = equity_curve.iloc[0]
    total_ret = final_value / initial_value - 1
    metrics['total_return'] = total_ret

    if n_years > 0:
        metrics['annual_return'] = (final_value / initial_value) ** (1 / n_years) - 1
    else:
        metrics['annual_return'] = total_ret

    # 最大回撤
    peak = equity_curve.expanding().max()
    dd = (equity_curve - peak) / peak
    metrics['max_drawdown'] = dd.min()

    # 年化波动率
    ann_vol = returns.std() * np.sqrt(252)
    metrics['volatility'] = ann_vol

    # 夏普
    rf_daily = risk_free_rate / 252
    excess = returns - rf_daily
    if ann_vol > 0:
        metrics['sharpe_ratio'] = np.sqrt(252) * excess.mean() / returns.std()
    else:
        metrics['sharpe_ratio'] = 0

    # Sortino（只考虑下行波动）
    downside = returns[returns < 0]
    if len(downside) > 0 and downside.std() > 0:
        metrics['sortino_ratio'] = np.sqrt(252) * excess.mean() / downside.std()
    else:
        metrics['sortino_ratio'] = 0

    # Calmar
    if metrics['max_drawdown'] < 0 and metrics['annual_return'] > 0:
        metrics['calmar_ratio'] = metrics['annual_return'] / abs(metrics['max_drawdown'])
    else:
        metrics['calmar_ratio'] = 0

    # === VaR / CVaR ===
    metrics['var_95'] = returns.quantile(0.05)
    metrics['cvar_95'] = returns[returns <= returns.quantile(0.05)].mean()

    # === 最大连续亏损 ===
    consecutive_loss = 0
    max_consecutive = 0
    for r in returns:
        if r < 0:
            consecutive_loss += 1
            max_consecutive = max(max_consecutive, consecutive_loss)
        else:
            consecutive_loss = 0
    metrics['max_consecutive_loss'] = max_consecutive

    # === 回撤恢复天数 ===
    dd_series = dd
    if dd.min() < -0.01:
        trough_idx = dd_series.idxmin()
        recovery = dd_series[dd_series.index > trough_idx]
        recovery_above_zero = recovery[recovery >= 0]
        if not recovery_above_zero.empty:
            recovery_date = recovery_above_zero.index[0]
            metrics['recovery_days'] = (recovery_date - trough_idx).days
        else:
            metrics['recovery_days'] = -1  # 未恢复
    else:
        metrics['recovery_days'] = 0

    # === 月度收益 ===
    monthly_ret = equity_curve.resample('ME').last().pct_change().dropna()
    metrics['monthly_returns'] = monthly_ret

    # === 持仓统计 ===
    if 'position_count' in daily_positions.columns:
        metrics['avg_position_count'] = daily_positions['position_count'].mean()
    elif daily_positions.shape[1] >= 1:
        # 推断持仓数
        pos_cols = [c for c in daily_positions.columns if 'shares' in c.lower() or 'value' in c.lower()]
        if pos_cols:
            metrics['avg_position_count'] = (daily_positions[pos_cols] > 0).sum(axis=1).mean()

    # === 换手率 ===
    if trades:
        total_trade_value = sum(t.value for t in trades)
        avg_capital = equity_curve.mean()
        if avg_capital > 0 and n_years > 0:
            metrics['turnover'] = total_trade_value / avg_capital / n_years  # 年化换手率

    # === 胜率 ===
    if trades:
        buy_trades = [t for t in trades if t.direction == 'BUY']
        sell_trades = [t for t in trades if t.direction in ('SELL', 'SELL_HALF')]
        if buy_trades and sell_trades:
            wins = 0
            total_pairs = 0
            for i, bt in enumerate(buy_trades):
                buy_date = bt.date
                next_buy = buy_trades[i + 1].date if i + 1 < len(buy_trades) else None
                related_sells = [st for st in sell_trades
                                 if buy_date <= st.date < (next_buy or '99999999')]
                if related_sells:
                    total_shares = sum(st.shares for st in related_sells)
                    if total_shares > 0:
                        avg_sell_price = sum(st.price * st.shares for st in related_sells) / total_shares
                        if avg_sell_price > bt.price:
                            wins += 1
                        total_pairs += 1
            metrics['win_rate'] = wins / total_pairs if total_pairs > 0 else 0

            # 盈亏比
            sell_values = []
            for i, bt in enumerate(buy_trades):
                buy_date = bt.date
                next_buy = buy_trades[i + 1].date if i + 1 < len(buy_trades) else None
                related_sells = [st for st in sell_trades
                                 if buy_date <= st.date < (next_buy or '99999999')]
                for st in related_sells:
                    pnl = (st.price - bt.price) / bt.price
                    sell_values.append(pnl)
            if sell_values:
                wins_list = [v for v in sell_values if v > 0]
                losses_list = [v for v in sell_values if v < 0]
                avg_win = np.mean(wins_list) if wins_list else 0
                avg_loss = abs(np.mean(losses_list)) if losses_list else 0
                metrics['profit_loss_ratio'] = avg_win / avg_loss if avg_loss > 0 else 0

    # === Beta / Alpha（需要基准） ===
    if benchmark_curve is not None and len(benchmark_curve) == len(equity_curve):
        bench_returns = benchmark_curve.pct_change().dropna()
        port_returns = equity_curve.pct_change().dropna()
        # 对齐索引
        common_idx = port_returns.index.intersection(bench_returns.index)
        if len(common_idx) > 20:
            port_r = port_returns.loc[common_idx]
            bench_r = bench_returns.loc[common_idx]
            cov = np.cov(port_r, bench_r)
            if cov[1, 1] > 0:
                metrics['beta'] = cov[0, 1] / cov[1, 1]
                metrics['alpha'] = (port_r.mean() - bench_r.mean() * metrics['beta']) * 252

            # 跟踪误差 + 信息比率
            diff = port_r - bench_r
            metrics['tracking_error'] = diff.std() * np.sqrt(252)
            if metrics['tracking_error'] > 0:
                metrics['information_ratio'] = (port_r.mean() - bench_r.mean()) / diff.std() * np.sqrt(252)

    return metrics
