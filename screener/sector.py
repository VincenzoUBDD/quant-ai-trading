# 板块股票分析工具 — 对任意股票列表运行通道策略分析
# 用法: python screener/sector.py <代码1> <代码2> ... [板块名称]
# 示例: python screener/sector.py 300308 300502 300394 光模块
import sys, os
if sys.stdout.encoding != 'utf-8':
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_fetcher import DataFetcher
from strategy.regression_channel.channel_strategy import RegressionChannelWithVolumeStrategy
from config import CHANNEL_WINDOW, VOL_WINDOW, VOL_MULTIPLIER, TODAY as END_DATE, DATE_RECENT_START as START_DATE
from stocks import WATCH_LIST
import pandas as pd

strategy = RegressionChannelWithVolumeStrategy(
    window=CHANNEL_WINDOW, vol_window=VOL_WINDOW, vol_multiplier=VOL_MULTIPLIER
)


def analyze_stock(symbol, shares=0, cost=0):
    """分析单只股票"""
    df = DataFetcher.get_stock_hist(symbol, START_DATE, END_DATE)
    if df.empty or len(df) < CHANNEL_WINDOW:
        return None

    close = df['close']
    upper = strategy._calc_upper(df['high'])
    vol_ma = df['volume'].rolling(VOL_WINDOW).mean()

    latest = df.iloc[-1]
    c = latest['close']
    u = upper.iloc[-1]
    vol = latest['volume']
    vm = vol_ma.iloc[-1]

    if pd.isna(u) or pd.isna(vm):
        return None

    prev_close = df['close'].iloc[-2]
    prev_upper = upper.iloc[-2]

    prev_below = prev_close < prev_upper
    breakout = c >= u
    vol_ok = vol > vm * VOL_MULTIPLIER
    can_buy = prev_below and breakout and vol_ok
    just_breakout = breakout and not vol_ok

    return {
        'symbol': symbol,
        'close': c,
        'upper': u,
        'pct_from_upper': (c - u) / u * 100,
        'vol_ratio': vol / vm if vm > 0 else 0,
        'prev_below': prev_below,
        'breakout': breakout,
        'vol_ok': vol_ok,
        'can_buy': can_buy,
        'just_breakout': just_breakout,
        'shares': shares,
        'cost': cost,
    }


def analyze_sector(stock_list, sector_name="自定义板块"):
    """分析板块股票"""

    print(f"\n{'=' * 60}")
    print(f"  {sector_name} — 回归通道策略分析")
    print(f"{'=' * 60}")
    print(f"  参数: 通道周期={CHANNEL_WINDOW}天, 放量倍数={VOL_MULTIPLIER}x")
    print()

    buy_signals = []
    watch_list = []

    for symbol in stock_list:
        shares = WATCH_LIST.get(symbol, {}).get('shares', 0)
        cost = WATCH_LIST.get(symbol, {}).get('cost', 0)

        result = analyze_stock(symbol, shares, cost)
        if result is None:
            print(f"  {symbol}: 数据获取失败")
            continue

        print(f"  {result['symbol']}")
        print(f"    收盘: {result['close']:.2f} | 上轨: {result['upper']:.2f} | 距上轨: {result['pct_from_upper']:+.2f}%")
        print(f"    量比: {result['vol_ratio']:.2f}x | {'昨日未突破' if result['prev_below'] else '昨日已突破'} | {'今日突破' if result['breakout'] else '未突破'}")

        if result['can_buy']:
            print(f"    => ★ 【买入信号】")
            buy_signals.append(result)
        elif result['just_breakout']:
            print(f"    => △ 突破待放量确认")
            watch_list.append(result)
        else:
            print(f"    => — 观望")
        print()

    # 总结
    print(f"{'=' * 60}")
    print(f"  分析结果汇总 — {sector_name}")
    print(f"{'=' * 60}")

    if buy_signals:
        print(f"\n  ★ 可买入 ({len(buy_signals)}只)")
        for s in buy_signals:
            print(f"    {s['symbol']}: 收盘{s['close']:.2f} > 上轨{s['upper']:.2f}, 量比{s['vol_ratio']:.2f}x")

    if watch_list:
        print(f"\n  △ 突破待确认 ({len(watch_list)}只)")
        for s in watch_list:
            print(f"    {s['symbol']}: 收盘{s['close']:.2f} > 上轨{s['upper']:.2f}, 量比{s['vol_ratio']:.2f}x (需放量>={VOL_MULTIPLIER}x)")

    if not buy_signals and not watch_list:
        print(f"\n  无突破信号")

    return buy_signals, watch_list


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python screener/sector.py <股票代码1> <股票代码2> ... [板块名称]")
        print("示例: python screener/sector.py 300308 300502 300394 光模块")
        sys.exit(1)

    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    # 最后一个参数如果是非纯数字，视为板块名称
    if len(args) > 1 and not args[-1].isdigit():
        sector_name = args.pop()
    else:
        sector_name = "自定义板块"
    analyze_sector(args, sector_name)
