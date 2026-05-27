"""投资组合 vs 大盘对比图"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_portfolio_vs_benchmark(equity_curve, benchmark_curve, trades=None, save_path=None):
    """
    组合 vs 大盘双轴对比图

    Args:
        equity_curve: 组合净值 Series
        benchmark_curve: 基准净值 Series（对齐后的）
        trades: 交易记录列表（可选，标记买卖点）
        save_path: 保存路径
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                             gridspec_kw={'height_ratios': [3, 1, 1]})

    # === 上：净值曲线对比 ===
    ax1 = axes[0]
    ax1.plot(equity_curve.index, equity_curve.values, 'b-', linewidth=1.8, label='组合净值')
    ax1.plot(benchmark_curve.index, benchmark_curve.values, 'gray', linewidth=1.2,
             linestyle='--', alpha=0.7, label='上证指数(归一化)')

    # 标注收益率
    port_ret = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
    bench_ret = (benchmark_curve.iloc[-1] / benchmark_curve.iloc[0] - 1) * 100
    ax1.text(0.02, 0.95, f'组合: {port_ret:+.1f}%', transform=ax1.transAxes,
             fontsize=13, fontweight='bold', color='blue', va='top')
    ax1.text(0.02, 0.87, f'大盘: {bench_ret:+.1f}%', transform=ax1.transAxes,
             fontsize=13, color='gray', va='top')

    ax1.set_ylabel('净值')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # 标记买卖点
    if trades:
        buy_dates = [pd.Timestamp(t.date) for t in trades if t.direction == 'BUY'
                     and pd.Timestamp(t.date) in equity_curve.index]
        sell_dates = [pd.Timestamp(t.date) for t in trades if t.direction in ('SELL', 'SELL_HALF')
                      and pd.Timestamp(t.date) in equity_curve.index]
        if buy_dates:
            buy_values = [equity_curve.loc[d] for d in buy_dates if d in equity_curve.index]
            ax1.scatter(buy_dates, buy_values, color='red', marker='^', s=40,
                       alpha=0.6, label='买入', zorder=5)
        if sell_dates:
            sell_values = [equity_curve.loc[d] for d in sell_dates if d in equity_curve.index]
            ax1.scatter(sell_dates, sell_values, color='green', marker='v', s=40,
                       alpha=0.6, label='卖出', zorder=5)

    # === 中：超额收益曲线 ===
    ax2 = axes[1]
    excess = (equity_curve / benchmark_curve - 1) * 100
    ax2.fill_between(excess.index, excess.values, 0,
                     where=excess.values > 0, color='red', alpha=0.3)
    ax2.fill_between(excess.index, excess.values, 0,
                     where=excess.values <= 0, color='green', alpha=0.3)
    ax2.plot(excess.index, excess.values, 'k-', linewidth=0.8)
    ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax2.set_ylabel('超额收益 %')
    excess_final = excess.iloc[-1]
    ax2.text(0.02, 0.9, f'累计超额: {excess_final:+.1f}%', transform=ax2.transAxes,
             fontsize=11, fontweight='bold', va='top')
    ax2.grid(True, alpha=0.3)

    # === 下：回撤曲线 ===
    ax3 = axes[2]
    peak = equity_curve.expanding().max()
    dd = (equity_curve - peak) / peak * 100
    ax3.fill_between(dd.index, dd.values, 0, color='red', alpha=0.5)
    ax3.plot(dd.index, dd.values, 'r-', linewidth=0.8)
    ax3.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax3.set_ylabel('回撤 %')
    ax3.set_xlabel('日期')
    max_dd = dd.min()
    ax3.text(0.02, 0.85, f'最大回撤: {max_dd:.1f}%', transform=ax3.transAxes,
             fontsize=11, fontweight='bold', color='red', va='top')
    ax3.grid(True, alpha=0.3)

    # X轴日期格式
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'图表保存: {save_path}')
    plt.show()


def plot_from_csv(csv_path='results/portfolio_vs_bench.csv', save_path='results/portfolio_comparison.png'):
    """从CSV加载数据绘图"""
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    equity_curve = df['portfolio'].dropna()
    benchmark_curve = df['benchmark'].dropna()

    # 对齐索引
    common = equity_curve.index.intersection(benchmark_curve.index)
    equity_curve = equity_curve.loc[common]
    benchmark_curve = benchmark_curve.loc[common]

    # 加载交易记录
    trades = []
    trades_path = csv_path.replace('_vs_bench.csv', '_trades.csv')
    if os.path.exists(trades_path):
        try:
            tdf = pd.read_csv(trades_path)
            from portfolio.results import PortfolioTrade
            for _, r in tdf.iterrows():
                trades.append(PortfolioTrade(
                    date=str(r['date']), symbol=str(r.get('symbol', '')),
                    direction=str(r['direction']),
                    price=float(r['price']), shares=float(r['shares']),
                    value=float(r['value']), reason=str(r.get('reason', '')),
                ))
        except Exception:
            pass

    plot_portfolio_vs_benchmark(equity_curve, benchmark_curve, trades=trades, save_path=save_path)


if __name__ == '__main__':
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'results/portfolio_vs_bench.csv'
    save_path = sys.argv[2] if len(sys.argv) > 2 else 'results/portfolio_comparison.png'
    plot_from_csv(csv_path, save_path)
