"""专业绩效指标计算模块"""
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class TradeRecord:
    """单笔交易记录"""
    entry_date: str
    exit_date: str
    symbol: str = ''
    direction: str = 'long'
    entry_price: float = 0.0
    exit_price: float = 0.0
    shares: float = 0.0
    pnl_pct: float = 0.0
    pnl_value: float = 0.0
    holding_days: int = 0
    exit_reason: str = ''


def compute_calmar_ratio(annual_return: float, max_drawdown: float) -> float:
    """Calmar = 年化收益 / |最大回撤|"""
    if max_drawdown < 0:
        return annual_return / abs(max_drawdown)
    return 0.0


def compute_sortino_ratio(returns: pd.Series, risk_free: float = 0.03) -> float:
    """Sortino: 只考虑下行波动"""
    excess = returns - risk_free / 252
    downside = returns[returns < 0]
    if len(downside) > 0 and downside.std() > 0:
        return np.sqrt(252) * excess.mean() / downside.std()
    return 0.0


def compute_beta_alpha(portfolio_returns: pd.Series,
                       benchmark_returns: pd.Series) -> Tuple[float, float]:
    """
    计算 Beta 和 Alpha（年化）

    Returns:
        (beta, alpha_annual)
    """
    common_idx = portfolio_returns.index.intersection(benchmark_returns.index)
    if len(common_idx) < 20:
        return 0, 0

    pr = portfolio_returns.loc[common_idx]
    br = benchmark_returns.loc[common_idx]
    cov = np.cov(pr, br)
    if cov[1, 1] <= 0:
        return 0, 0
    beta = cov[0, 1] / cov[1, 1]
    alpha = (pr.mean() - br.mean() * beta) * 252
    return beta, alpha


def compute_max_consecutive_losses(returns: pd.Series) -> Tuple[int, float]:
    """
    最大连续亏损次数 + 累计亏损

    Returns:
        (max_streak, total_loss_pct)
    """
    max_streak = 0
    current_streak = 0
    total_loss = 0
    for r in returns:
        if r < 0:
            current_streak += 1
            total_loss += r
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
            total_loss = 0
    return max_streak, total_loss


def compute_recovery_days(equity_curve: pd.Series) -> int:
    """从最大回撤恢复到前高的天数。返回 -1 表示尚未恢复。"""
    if len(equity_curve) < 2:
        return 0
    dd = equity_curve / equity_curve.expanding().max() - 1
    trough_idx = dd.idxmin()
    if dd.min() >= 0:
        return 0
    recovery = dd[dd.index > trough_idx]
    recovered = recovery[recovery >= 0]
    if not recovered.empty:
        return (recovered.index[0] - trough_idx).days
    return -1


def compute_monthly_returns(equity_curve: pd.Series) -> pd.Series:
    """月度收益率序列"""
    return equity_curve.resample('ME').last().pct_change().dropna()


def compute_win_rate(trades: List[TradeRecord]) -> Tuple[float, float, float]:
    """
    胜率 + 盈亏比 + 期望值

    Returns:
        (win_rate, profit_loss_ratio, expectation)
    """
    if not trades:
        return 0, 0, 0

    wins = [t for t in trades if t.pnl_pct > 0]
    losses = [t for t in trades if t.pnl_pct < 0]

    win_rate = len(wins) / len(trades) if trades else 0

    avg_win = np.mean([t.pnl_pct for t in wins]) if wins else 0
    avg_loss = abs(np.mean([t.pnl_pct for t in losses])) if losses else 0
    pl_ratio = avg_win / avg_loss if avg_loss > 0 else 0

    # 期望值 = 胜率 × 平均盈利 - 败率 × 平均亏损
    expectation = win_rate * avg_win - (1 - win_rate) * avg_loss

    return win_rate, pl_ratio, expectation


def compute_information_ratio(portfolio_returns: pd.Series,
                               benchmark_returns: pd.Series) -> Tuple[float, float]:
    """
    跟踪误差 + 信息比率

    Returns:
        (tracking_error, information_ratio)
    """
    common = portfolio_returns.index.intersection(benchmark_returns.index)
    if len(common) < 20:
        return 0, 0

    diff = portfolio_returns.loc[common] - benchmark_returns.loc[common]
    te = diff.std() * np.sqrt(252)
    if te > 0:
        ir = diff.mean() / diff.std() * np.sqrt(252)
    else:
        ir = 0
    return te, ir


def compute_all_metrics(equity_curve: pd.Series,
                        trades: List[TradeRecord] = None,
                        benchmark_curve: pd.Series = None,
                        risk_free: float = 0.03) -> dict:
    """一站式计算全部指标"""
    metrics = {}

    if equity_curve.empty or len(equity_curve) < 5:
        return metrics

    returns = equity_curve.pct_change().dropna()
    initial, final = equity_curve.iloc[0], equity_curve.iloc[-1]
    total_days = (equity_curve.index[-1] - equity_curve.index[0]).days
    n_years = total_days / 365.0

    # 基础
    metrics['total_return'] = final / initial - 1
    metrics['annual_return'] = (final / initial) ** (1 / n_years) - 1 if n_years > 0 else 0

    # 波动率
    metrics['daily_volatility'] = returns.std()
    metrics['annual_volatility'] = returns.std() * np.sqrt(252)

    # 夏普
    rf_daily = risk_free / 252
    excess = returns - rf_daily
    metrics['sharpe_ratio'] = np.sqrt(252) * excess.mean() / returns.std() if returns.std() > 0 else 0

    # Sortino
    metrics['sortino_ratio'] = compute_sortino_ratio(returns, risk_free)

    # 最大回撤
    peak = equity_curve.expanding().max()
    dd = (equity_curve - peak) / peak
    metrics['max_drawdown'] = dd.min()
    metrics['calmar_ratio'] = compute_calmar_ratio(metrics['annual_return'], metrics['max_drawdown'])

    # 尾部风险
    metrics['var_95'] = returns.quantile(0.05)
    metrics['cvar_95'] = returns[returns <= returns.quantile(0.05)].mean()

    # 连续亏损
    max_loss, total_loss = compute_max_consecutive_losses(returns)
    metrics['max_consecutive_losses'] = max_loss

    # 恢复天数
    metrics['recovery_days'] = compute_recovery_days(equity_curve)

    # 月度收益
    metrics['monthly_returns'] = compute_monthly_returns(equity_curve)

    # 基准对比
    if benchmark_curve is not None and not benchmark_curve.empty:
        bench_returns = benchmark_curve.pct_change().dropna()
        metrics['beta'], metrics['alpha'] = compute_beta_alpha(returns, bench_returns)
        metrics['tracking_error'], metrics['information_ratio'] = compute_information_ratio(
            returns, bench_returns)
    else:
        metrics['beta'] = 0
        metrics['alpha'] = 0
        metrics['tracking_error'] = 0
        metrics['information_ratio'] = 0

    # 交易统计
    if trades:
        wr, pl, exp = compute_win_rate(trades)
        metrics['win_rate'] = wr
        metrics['profit_loss_ratio'] = pl
        metrics['expectation'] = exp
        metrics['total_trades'] = len(trades)
    else:
        metrics['win_rate'] = 0
        metrics['profit_loss_ratio'] = 0
        metrics['expectation'] = 0
        metrics['total_trades'] = 0

    return metrics
