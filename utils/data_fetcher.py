# 数据获取模块
import akshare as ak
import pandas as pd
from typing import Optional
import os
import time


class DataFetcher:
    """A股数据获取器"""

    @staticmethod
    def _to_market_symbol(symbol: str) -> str:
        if symbol.startswith(('6', '5')):
            return 'sh' + symbol
        return 'sz' + symbol

    @staticmethod
    def get_stock_hist(symbol: str, start_date: str = '', end_date: str = '',
                       retries: int = 3) -> pd.DataFrame:
        """
        获取单只股票历史数据

        Args:
            symbol: 股票代码，如 '000001'
            start_date: 开始日期 'YYYYMMDD'（可选）
            end_date: 结束日期 'YYYYMMDD'（可选）
            retries: 失败重试次数
        """
        for attempt in range(retries):
            try:
                market = DataFetcher._to_market_symbol(symbol)
                df = ak.stock_zh_a_daily(symbol=market)
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date').sort_index()

                if start_date:
                    df = df[df.index >= pd.Timestamp(start_date)]
                if end_date:
                    df = df[df.index <= pd.Timestamp(end_date)]

                return df
            except Exception as e:
                if attempt < retries - 1:
                    print(f"获取 {symbol} 失败，重试 {attempt + 1}/{retries}...")
                    time.sleep(1)
                else:
                    print(f"获取 {symbol} 数据失败: {e}")
                    return pd.DataFrame()
        return pd.DataFrame()

    @staticmethod
    def get_index_hist(index_code: str, start_date: str = '', end_date: str = '') -> pd.DataFrame:
        """获取指数历史数据"""
        for attempt in range(3):
            try:
                df = ak.stock_zh_index_daily(symbol=f"sh{index_code}")
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date').sort_index()

                if start_date:
                    df = df[df.index >= pd.Timestamp(start_date)]
                if end_date:
                    df = df[df.index <= pd.Timestamp(end_date)]
                return df
            except Exception as e:
                if attempt < 2:
                    time.sleep(1)
                else:
                    print(f"获取指数 {index_code} 数据失败: {e}")
                    return pd.DataFrame()
        return pd.DataFrame()

    @staticmethod
    def save_to_csv(df: pd.DataFrame, filename: str, subdir: str = 'data') -> str:
        """保存数据到CSV"""
        os.makedirs(subdir, exist_ok=True)
        path = f"{subdir}/{filename}"
        df.to_csv(path)
        print(f"数据已保存到: {path}")
        return path

    @staticmethod
    def load_from_csv(filepath: str) -> pd.DataFrame:
        """从CSV加载数据"""
        df = pd.read_csv(filepath)
        df['date'] = pd.to_datetime(df['date'])
        return df.set_index('date').sort_index()
