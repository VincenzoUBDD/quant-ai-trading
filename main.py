# 主入口文件 - 最终优化版
import os
import shutil
from utils.data_fetcher import DataFetcher
from strategy.regression_channel.channel_strategy import RegressionChannelWithVolumeStrategy
from backtest.engine import BacktestEngine
from config import SYMBOL, INITIAL_CAPITAL, CHANNEL_WINDOW, VOL_WINDOW, VOL_MULTIPLIER, BT_LONG_START, BT_LONG_END


def clean_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)


def main():
    # 获取数据
    df = DataFetcher.get_stock_hist(SYMBOL, BT_LONG_START, BT_LONG_END)
    if df.empty:
        return

    # 保存数据
    os.makedirs(f'results/{SYMBOL}/data', exist_ok=True)
    df.to_csv(f'results/{SYMBOL}/data/{SYMBOL}.csv')

    # 清理旧结果（只保留data文件夹）
    for item in os.listdir(f'results/{SYMBOL}'):
        item_path = f'results/{SYMBOL}/{item}'
        if item != 'data' and os.path.isdir(item_path):
            clean_folder(item_path)

    # 运行最终策略
    engine = BacktestEngine(initial_capital=INITIAL_CAPITAL)

    strategy = RegressionChannelWithVolumeStrategy(
        window=CHANNEL_WINDOW,
        vol_window=VOL_WINDOW,
        vol_multiplier=VOL_MULTIPLIER
    )

    result = engine.run(df, strategy)

    # 保存结果
    engine.save_results(result, df, strategy.name, f'results/{SYMBOL}/')


if __name__ == '__main__':
    main()
