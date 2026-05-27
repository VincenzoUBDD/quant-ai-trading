# 统一可视化模块 — 支持通道图和三面板图
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from utils.data_fetcher import DataFetcher


plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def calc_regression_channel(high: pd.Series, low: pd.Series, window: int = 60):
    """计算上下轨回归通道（无未来函数）"""
    upper = pd.Series(index=high.index, dtype=float)
    lower = pd.Series(index=low.index, dtype=float)

    for i in range(len(high)):
        if i < window:
            continue
        x = np.arange(window)
        z_high = np.polyfit(x, high.iloc[i - window:i].values, 1)
        z_low = np.polyfit(x, low.iloc[i - window:i].values, 1)
        upper.iloc[i] = z_high[0] * (window - 1) + z_high[1]
        lower.iloc[i] = z_low[0] * (window - 1) + z_low[1]

    return upper, lower


def plot_channel(symbol: str, start_date: str, end_date: str, window: int = 60):
    """绘制回归通道图（含买卖信号）"""
    df = DataFetcher.get_stock_hist(symbol, start_date, end_date)
    if df.empty:
        print("获取数据失败!")
        return

    upper, lower = calc_regression_channel(df['high'], df['low'], window)

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(df.index, df['close'], label='Close', color='black', linewidth=1.5)
    ax.plot(df.index, upper, label=f'Upper ({window}d)', color='red', linewidth=1, linestyle='--')
    ax.plot(df.index, lower, label=f'Lower ({window}d)', color='green', linewidth=1, linestyle='--')
    ax.fill_between(df.index, lower, upper, alpha=0.1, color='gray')

    # 标记买卖点
    in_pos = False
    for i in range(len(df)):
        if pd.isna(upper.iloc[i]):
            continue
        close = df['close'].iloc[i]
        if not in_pos and close >= upper.iloc[i]:
            ax.scatter(df.index[i], close, color='red', marker='^', s=100, zorder=5)
            in_pos = True
        elif in_pos and close <= lower.iloc[i]:
            ax.scatter(df.index[i], close, color='green', marker='v', s=100, zorder=5)
            in_pos = False

    ax.set_title(f'{symbol} - Regression Channel (Window={window})', fontsize=14)
    ax.legend(loc='upper left')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    output = f'results/{symbol}/channel_chart_{window}.png'
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"图表已保存: {output}")
    plt.close()


def plot_backtest(symbol: str, start_date: str, end_date: str,
                  channel_window: int = 40, vol_window: int = 20):
    """三面板图: 价格+上轨 / 量比 / 收益曲线"""
    df = DataFetcher.get_stock_hist(symbol, start_date, end_date)
    if df.empty:
        print("获取数据失败!")
        return

    # 计算上轨和量比
    df['channel'] = df['high'].rolling(channel_window).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] * (channel_window - 1) + x[0], raw=True
    )
    df['vol_ma'] = df['volume'].rolling(vol_window).mean()
    df['vol_ratio'] = df['volume'] / df['vol_ma']

    # 模拟交易
    cash = 100000
    shares = 0
    trades = []
    portfolio = [100000] * len(df)
    entry_price = 0

    for i in range(1, len(df)):
        row = df.iloc[i]
        if pd.isna(row['channel']) or pd.isna(row['vol_ma']):
            portfolio[i] = cash + shares * row['close']
            continue
        if row['close'] >= row['channel'] and row['vol_ratio'] > 1.5 and shares == 0:
            shares = int(cash / row['close'])
            cash -= shares * row['close']
            entry_price = row['close']
            trades.append({'date': row.name, 'type': 'BUY', 'price': row['close']})
        elif row['close'] < row['channel'] and shares > 0:
            cash += shares * row['close']
            trades.append({'date': row.name, 'type': 'SELL', 'price': row['close'],
                          'pnl': (row['close'] - entry_price) * shares})
            shares = 0
        portfolio[i] = cash + shares * row['close']

    df['portfolio'] = portfolio
    df['return'] = (df['portfolio'] - 100000) / 100000 * 100

    # 绘图
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    ax1 = axes[0]
    ax1.plot(df.index, df['close'], label='Close', linewidth=1)
    ax1.plot(df.index, df['channel'], label='Upper Channel', linewidth=1, linestyle='--')
    for t in trades:
        marker = '^' if t['type'] == 'BUY' else 'v'
        color = 'green' if t['type'] == 'BUY' else 'red'
        ax1.scatter(t['date'], t['price'], marker=marker, color=color, s=100, zorder=5)
    ax1.set_title(f'{symbol} — Price & Channel', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    colors = ['green' if r > 1.5 else 'gray' for r in df['vol_ratio']]
    ax2.bar(df.index, df['vol_ratio'], color=colors, alpha=0.7)
    ax2.axhline(y=1.5, color='red', linestyle='--', label='Threshold(1.5)')
    ax2.set_title('Volume Ratio', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    ax3.fill_between(df.index, df['return'], 0, where=(df['return'] >= 0), color='red', alpha=0.3)
    ax3.fill_between(df.index, df['return'], 0, where=(df['return'] < 0), color='green', alpha=0.3)
    ax3.plot(df.index, df['return'], linewidth=1.5)
    ax3.axhline(y=0, color='black', linewidth=0.5)
    ax3.set_title('Return (%)', fontsize=12)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    output = f'results/{symbol}/backtest_panel.png'
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"三面板图已保存: {output}")
    plt.close()

    # 打印交易统计
    sell_trades = [t for t in trades if t['type'] == 'SELL']
    win_rate = sum(1 for t in sell_trades if t.get('pnl', 0) > 0) / max(len(sell_trades), 1) * 100
    print(f"\n{symbol} 回测统计:")
    print(f"  初始资金: 100,000 | 最终: {portfolio[-1]:.2f}")
    print(f"  总收益: {df['return'].iloc[-1]:.2f}%")
    print(f"  交易: {len(trades) // 2} 次 | 胜率: {win_rate:.0f}%")


def plot_backtest_v2(symbol: str, start_date: str, end_date: str,
                      channel_window: int = 40, vol_window: int = 20,
                      vol_multiplier: float = 1.5):
    """回测可视化（使用实际策略类）— 价格+上轨+买卖点+量比"""
    from strategy.regression_channel.channel_strategy import RegressionChannelWithVolumeStrategy
    from backtest.engine import BacktestEngine

    # 获取数据
    df = DataFetcher.get_stock_hist(symbol, start_date, end_date)
    if df.empty:
        print("获取数据失败!")
        return

    # 运行策略 & 回测
    strategy = RegressionChannelWithVolumeStrategy(
        window=channel_window, vol_window=vol_window, vol_multiplier=vol_multiplier)
    engine = BacktestEngine(initial_capital=100000)

    df['upper'] = strategy._calc_upper(df['high'])
    df['vol_ma'] = df['volume'].rolling(vol_window).mean()

    result = engine.run(df, strategy)
    signals = strategy.generate_signals(df)

    # 提取买卖点
    buys = [(df.index[i], df['close'].iloc[i], df['upper'].iloc[i])
            for i in range(len(signals)) if signals.iloc[i] == 1]
    sells_half = [(df.index[i], df['close'].iloc[i])
                  for i in range(len(signals)) if signals.iloc[i] == 2]
    sells = [(df.index[i], df['close'].iloc[i])
             for i in range(len(signals)) if signals.iloc[i] == -1]

    # 绘图
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 10),
                                         gridspec_kw={'height_ratios': [3, 1.2, 1]},
                                         sharex=True)

    # === 面板1：价格 & 上轨 & 买卖点 ===
    ax1.plot(df.index, df['close'], label='收盘价', color='black', linewidth=1.5)
    ax1.plot(df.index, df['upper'], label='上轨 (回归通道)', color='red',
             linewidth=1.5, linestyle='--', alpha=0.8)

    # 填充上轨上方区域（突破区）
    ax1.fill_between(df.index, df['close'], df['upper'],
                     where=(df['close'] >= df['upper']),
                     color='red', alpha=0.08, label='突破区')

    # 买入点
    for d, p, u in buys:
        ax1.scatter(d, p, color='green', marker='^', s=120, zorder=6,
                    edgecolors='darkgreen', linewidth=1)
        ax1.annotate(f'买入\n{p:.1f}', (d, p),
                     textcoords="offset points", xytext=(0, 18),
                     ha='center', fontsize=8, color='darkgreen',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    # 半仓卖出点
    for d, p in sells_half:
        ax1.scatter(d, p, color='orange', marker='s', s=120, zorder=6,
                    edgecolors='darkorange', linewidth=1)
        ax1.annotate(f'半仓\n{p:.1f}', (d, p),
                     textcoords="offset points", xytext=(0, -22),
                     ha='center', fontsize=8, color='darkorange',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    # 卖出点
    for d, p in sells:
        ax1.scatter(d, p, color='red', marker='v', s=120, zorder=6,
                    edgecolors='darkred', linewidth=1)
        ax1.annotate(f'卖出\n{p:.1f}', (d, p),
                     textcoords="offset points", xytext=(0, -22),
                     ha='center', fontsize=8, color='darkred',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    ax1.set_title(f'{symbol} — 回归通道策略回测', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.2)
    ax1.set_ylabel('价格')

    # === 面板2：收益曲线 ===
    equity = result.equity_curve
    ret_pct = (equity / equity.iloc[0] - 1) * 100
    ax2.plot(ret_pct.index, ret_pct, color='red', linewidth=1.5)
    ax2.fill_between(ret_pct.index, ret_pct, 0,
                     where=(ret_pct >= 0), color='red', alpha=0.15)
    ax2.fill_between(ret_pct.index, ret_pct, 0,
                     where=(ret_pct < 0), color='green', alpha=0.15)
    ax2.axhline(y=0, color='black', linewidth=0.5)
    # 标注最终收益
    final_ret = ret_pct.iloc[-1]
    ax2.axhline(y=final_ret, color='red', linewidth=0.8, linestyle=':')
    ax2.annotate(f'{final_ret:+.1f}%', (ret_pct.index[-1], final_ret),
                 textcoords="offset points", xytext=(8, 0),
                 fontsize=10, color='red', fontweight='bold')
    ax2.set_ylabel('收益率 (%)')
    ax2.grid(True, alpha=0.2)
    ax2.legend(['收益率', '基准线'], loc='upper left', fontsize=9)

    # === 面板3：成交量 & 量比 ===
    vol_colors = ['red' if v > v_ma * vol_multiplier else 'gray'
                  for v, v_ma in zip(df['volume'], df['vol_ma'])]
    ax3.bar(df.index, df['volume'], color=vol_colors, alpha=0.7, width=1)
    ax3.plot(df.index, df['vol_ma'] * vol_multiplier, color='red',
             linewidth=1, linestyle=':', label=f'放量线 ({vol_multiplier}×)')
    ax3.set_ylabel('成交量')
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.2)

    # 日期格式
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)

    # === 统计信息 ===
    stats_text = (
        f'总收益: {result.total_return * 100:+.1f}%    '
        f'年化: {result.annual_return * 100:+.1f}%    '
        f'夏普: {result.sharpe_ratio:.2f}\n'
        f'最大回撤: {result.max_drawdown * 100:.1f}%    '
        f'胜率: {result.win_rate * 100:.0f}%    '
        f'交易: {result.total_trades} 次'
    )
    ax1.text(0.99, 0.02, stats_text, transform=ax1.transAxes,
             fontsize=10, verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.tight_layout()
    output = f'results/{symbol}/backtest_v2.png'
    os.makedirs(f'results/{symbol}', exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"回测图已保存: {output}")
    plt.close()

    # 打印交易明细（含每笔盈亏）
    print(f'\n{symbol} 交易明细:')
    print(f'{"日期":<12} {"类型":<10} {"价格":>8} {"盈亏":>8}')
    print('-' * 45)
    buy_price = 0
    wins = 0
    total_sells = 0
    for t in result.trades:
        if t['type'] == 'BUY':
            buy_price = t['price']
            print(f'{t["date"]:<12} {"买入":<10} {t["price"]:>8.2f}')
        elif t['type'] in ('SELL_HALF', 'SELL'):
            pnl_pct = (t['price'] - buy_price) / buy_price * 100
            label = '半仓卖出' if t['type'] == 'SELL_HALF' else ('止盈' if pnl_pct > 0 else '止损')
            print(f'{t["date"]:<12} {label:<10} {t["price"]:>8.2f} {pnl_pct:>+7.1f}%')
            total_sells += 1
            if pnl_pct > 0:
                wins += 1
    print(f'\n胜率: {wins}/{total_sells} = {wins/total_sells*100:.0f}%' if total_sells > 0 else '')


if __name__ == '__main__':
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else 'channel'
    symbol = sys.argv[2] if len(sys.argv) > 2 else '300058'

    if mode == 'backtest':
        plot_backtest(symbol, '20240101', '20260101')
    elif mode == 'backtest2':
        plot_backtest_v2(symbol, '20240101', '20260101')
    else:
        plot_channel(symbol, '20240101', '20260101')
