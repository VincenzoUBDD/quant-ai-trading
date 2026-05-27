"""批量回测：对多支 A 股运行回归通道策略（买入点基本面过滤），按收益率排序输出"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import akshare as ak

from strategy.regression_channel.channel_strategy import RegressionChannelWithVolumeStrategy
from backtest.engine import BacktestEngine
from config import CHANNEL_WINDOW, VOL_WINDOW, VOL_MULTIPLIER, BT_LONG_START, BT_LONG_END, FUNDAMENTAL_MIN_SCORE
from utils.data_fetcher import DataFetcher
from utils.fundamental import FundamentalAnalyzer

STRATEGY = RegressionChannelWithVolumeStrategy(
    window=CHANNEL_WINDOW, vol_window=VOL_WINDOW, vol_multiplier=VOL_MULTIPLIER,
    fundamental_checker=FundamentalAnalyzer(min_score=FUNDAMENTAL_MIN_SCORE),
    min_fundamental_score=FUNDAMENTAL_MIN_SCORE,
)
ENGINE = BacktestEngine(initial_capital=100000)

print("获取股票列表...")
sh = ak.stock_info_sh_name_code()
sz = ak.stock_info_sz_name_code()
all_stocks = []
for _, row in sh.iterrows():
    all_stocks.append((str(row.iloc[0]).zfill(6), str(row.iloc[1])))
for _, row in sz.iterrows():
    all_stocks.append((str(row.iloc[1]).zfill(6), str(row.iloc[2])))
stocks = all_stocks[:300]
print(f"SH {len(sh)} + SZ {len(sz)}, taking 300\n")

# 预取基本面数据（多年代缓存）
print(f"预取基本面数据（{BT_LONG_START[:4]} 起回测所需各年）...")
abstract_cache = STRATEGY.fundamental_checker.prefetch_abstract(stocks)
multi_year_cache = STRATEGY.fundamental_checker.build_multi_year_cache(abstract_cache,
    start_year=int(BT_LONG_START[:4]) - 3,
    end_year=int(BT_LONG_END[:4]))
print(f"原始数据 {len(abstract_cache)} 支，多年代缓存 {len(multi_year_cache)} 支\n")

# 预取机构持仓历史数据（回测基金因子加分）
print("预取机构持仓历史数据（回测基金因子）...")
STRATEGY.fundamental_checker.prefetch_institutional_holdings(
    start_year=int(BT_LONG_START[:4]) - 1,
    end_year=int(BT_LONG_END[:4]))
print()

results = []
for idx, (code, name) in enumerate(stocks, 1):
    print(f"[{idx}/{len(stocks)}] {code} {name} ... ", end="", flush=True)
    df = DataFetcher.get_stock_hist(code, BT_LONG_START, BT_LONG_END)
    if df.empty or len(df) < 100:
        print("skip")
        continue

    STRATEGY.symbol = code
    STRATEGY.fundamental_cache = multi_year_cache

    try:
        result = ENGINE.run(df, STRATEGY)
        results.append({
            'code': code, 'name': name,
            'return': result.total_return * 100,
            'annual': result.annual_return * 100,
            'max_dd': result.max_drawdown * 100,
            'trades': result.total_trades,
            'win_rate': result.win_rate * 100,
            'sharpe': result.sharpe_ratio,
        })
        print(f"{result.total_return*100:+.1f}%")
    except Exception as e:
        print(f"fail: {e}")

results.sort(key=lambda x: x['return'], reverse=True)

print("\n" + "=" * 110)
print(f"{'排名':<4} {'代码':<7} {'名称':<10} {'收益率':<8} {'年化':<8} {'最大回撤':<8} {'交易次数':<8} {'胜率':<6} {'夏普':<6}")
print("=" * 110)
for i, r in enumerate(results, 1):
    print(f"{i:<4} {r['code']:<7} {r['name']:<10} {r['return']:+7.1f}% {r['annual']:+7.1f}% {r['max_dd']:>7.1f}% {r['trades']:<8} {r['win_rate']:>5.1f}% {r['sharpe']:>5.2f}")

print(f"\n有效回测: {len(results)}/{len(stocks)}")
