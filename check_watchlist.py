# 快速检查关注列表
import pandas as pd
import numpy as np
from config import CHANNEL_WINDOW, VOL_WINDOW, VOL_MULTIPLIER
from stocks import WATCH_LIST
from utils.data_fetcher import DataFetcher


def calc_upper(high, window):
    """无未来函数上轨：用 i-window 至 i-1 日预测第 i 天上轨值"""
    upper = pd.Series(index=high.index, data=np.nan)
    for i in range(len(high)):
        if i < window:
            continue
        x = np.arange(window)
        z = np.polyfit(x, high.iloc[i-window:i].values, 1)
        upper.iloc[i] = z[0] * (window - 1) + z[1]
    return upper


def check_stock(symbol, shares=0, cost=0):
    df = DataFetcher.get_stock_hist(symbol, '20250101', '20260508')
    if df.empty:
        print(f'{symbol}: 数据获取失败')
        return

    upper = calc_upper(df['high'], CHANNEL_WINDOW)
    vol_ma = df['volume'].rolling(VOL_WINDOW).mean()

    latest = df.iloc[-1]
    close = latest['close']
    u = upper.iloc[-1]
    vol = latest['volume']
    vm = vol_ma.iloc[-1]
    latest_date = df.index[-1].date()

    breakout = close >= u
    vol_ok = vol > vm * VOL_MULTIPLIER

    pnl = (close - cost) / cost * 100 if cost > 0 else 0
    pnl_value = (close - cost) * shares if cost > 0 else 0

    # 信号
    if breakout and vol_ok:
        signal = '【买入】'
    elif breakout:
        signal = '突破待确认'
    else:
        signal = '观望'

    print(f'{symbol}')
    print(f'  日期: {latest_date}')
    print(f'  收盘: {close:.3f} | 上轨: {u:.3f}')
    print(f'  量比: {vol/vm:.2f}x | 突破: {"是" if breakout else "否"}')
    if shares > 0:
        print(f'  持仓: {shares}股 | 成本: {cost:.3f} | 盈亏: {pnl:+.2f}% ({pnl_value:+.0f})')
    print(f'  信号: {signal}')
    print()


if __name__ == '__main__':
    print('='*55)
    print('关注列表检查')
    print('='*55)
    for symbol, info in WATCH_LIST.items():
        check_stock(symbol, info['shares'], info['cost'])
