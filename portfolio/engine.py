"""投资组合回测引擎 — 多股共享资金池，产生组合净值"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from datetime import datetime, timedelta
from collections import defaultdict

from .config import PortfolioConfig
from .results import PortfolioResult, PortfolioTrade, compute_portfolio_metrics
from .position_sizer import PositionSizer
from .exposure_manager import ExposureManager
from .market_regime import MarketRegimeFilter
from .risk_engine import RiskEngine
from strategy.MA.base import Signal
from utils.data_fetcher import DataFetcher


class PortfolioEngine:
    """
    投资组合回测引擎

    数据流（每个交易日）:
      1. 获取 universe 所有股票当天 OHLCV
      2. 对每支股票运行 generate_signals() → 当天信号
      3. 卖出：已有持仓出现卖出信号 → 执行，资金回现金池
      4. 风控检查：止损/回撤/波动率/市场状态
      5. 买入：收集买入信号，经板块/仓位约束后分配资金
      6. 计算当日组合净值
    """

    def __init__(self, config: PortfolioConfig = None):
        self.config = config or PortfolioConfig()
        self.sizer = PositionSizer(self.config)
        self.risk_engine = RiskEngine(self.config)
        self._stopped = False  # 止损熔断标记，市场恢复后可清零

    def run(self,
            universe: List[str],
            strategy,
            start_date: str,
            end_date: str,
            benchmark_code: str = '000001',
            sector_map: Dict[str, List[str]] = None,
            symbol_names: Dict[str, str] = None,
            progress_callback: Callable = None,
            initial_state: dict = None) -> PortfolioResult:
        """
        运行组合回测

        Args:
            universe: 股票代码列表
            strategy: 策略实例（带 generate_signals 方法）
            start_date: 起始日 YYYYMMDD
            end_date: 结束日 YYYYMMDD
            benchmark_code: 基准指数代码
            sector_map: 板块分组 {sector: [symbols]}
            symbol_names: {symbol: name}
            progress_callback: 进度回调 func(current, total)
            initial_state: 接力状态 {cash, positions, equity_curve, daily_positions_records, trades, peak_equity}

        Returns:
            PortfolioResult
        """
        sector_map = sector_map or {}
        symbol_names = symbol_names or {}
        self._strategy = strategy

        # 获取全量历史数据
        print("获取历史数据...")
        all_data: Dict[str, pd.DataFrame] = {}
        total = len(universe)
        for i, symbol in enumerate(universe):
            df = DataFetcher.get_stock_hist(symbol, start_date, end_date)
            if df is not None and not df.empty and len(df) > 60:
                all_data[symbol] = df
            if progress_callback:
                progress_callback(i + 1, total)

        print(f"  有效股票: {len(all_data)}/{total}")

        # 获取基准指数数据
        benchmark_data = DataFetcher.get_index_hist(benchmark_code, start_date, end_date)

        # 获取大盘指数数据（含前120+天历史用于均线计算）
        from datetime import timedelta
        market_start = (pd.Timestamp(start_date) - timedelta(days=250)).strftime('%Y%m%d')
        market_data = DataFetcher.get_index_hist(benchmark_code, market_start, end_date)
        if market_data is not None and not market_data.empty:
            self.risk_engine.market_regime = MarketRegimeFilter()
        else:
            self.risk_engine.market_regime = None

        # 统一交易日索引（取所有股票交易日的并集）
        all_dates = sorted(set(
            d for df in all_data.values() for d in df.index
        ))
        all_dates = [d for d in all_dates if d >= pd.Timestamp(start_date) and d <= pd.Timestamp(end_date)]

        print(f"交易日: {len(all_dates)} 天")

        # ===== 初始化状态（支持接力） =====
        init = initial_state or {}
        cash = init.get('cash', self.config.initial_capital)
        positions: Dict[str, Dict] = init.get('positions', {}).copy() if init.get('positions') else {}
        equity_curve = list(init.get('equity_curve', []))
        daily_positions_records = list(init.get('daily_positions_records', []))
        trades = list(init.get('trades', []))
        peak_equity = init.get('peak_equity', self.config.initial_capital)

        # 缓存每支股票的信号（避免重复计算）
        signal_cache: Dict[str, pd.Series] = {}
        upper_cache: Dict[str, pd.Series] = {}

        print("运行组合回测...")
        for idx, date in enumerate(all_dates):
            date_str = str(date.date())

            # --- 1. 收集当天所有股票的数据 ---
            day_data = {}
            for symbol, df in all_data.items():
                try:
                    row = df.loc[date]
                    if row is not None:
                        day_data[symbol] = row
                except (KeyError, IndexError):
                    continue

            if not day_data:
                equity_curve.append(cash)
                daily_positions_records.append({'date': date, 'equity': cash, 'cash': cash, 'position_count': 0})
                continue

            # --- 2. 获取信号（首次运行时计算全量，后续用缓存） ---
            sell_signals = {}  # {symbol: signal_value}
            buy_signals = {}   # {symbol: signal_value}
            day_prices = {}
            day_atr = {}

            for symbol in day_data:
                if symbol not in signal_cache:
                    # 首次：计算全量信号
                    df = all_data[symbol]
                    signal_cache[symbol] = strategy.generate_signals(df.copy())
                    try:
                        upper_cache[symbol] = strategy._calc_upper(df['high'])
                    except AttributeError:
                        upper_cache[symbol] = pd.Series(index=df.index, dtype=float)

                sig_series = signal_cache[symbol]
                if date in sig_series.index:
                    sig = sig_series.loc[date]
                    if sig == Signal.SELL.value or sig == Signal.SELL_HALF.value:
                        sell_signals[symbol] = sig
                    elif sig == Signal.BUY.value:
                        buy_signals[symbol] = 1.0

                day_prices[symbol] = day_data[symbol]['close']

                # ATR 计算
                df = all_data[symbol]
                if 'high' in df and 'low' in df and 'close' in df:
                    tr = pd.concat([
                        df['high'] - df['low'],
                        (df['high'] - df['close'].shift(1)).abs(),
                        (df['low'] - df['close'].shift(1)).abs(),
                    ], axis=1).max(axis=1)
                    atr = tr.rolling(14).mean()
                    if date in atr.index and day_prices[symbol] > 0:
                        day_atr[symbol] = atr.loc[date] / day_prices[symbol]

            # --- 3. 卖出阶段 ---
            for symbol in list(positions.keys()):
                pos = positions[symbol]
                price = day_prices.get(symbol)
                if price is None or price <= 0:
                    continue

                if symbol in sell_signals:
                    sig = sell_signals[symbol]
                    if sig == Signal.SELL.value:
                        # 全部卖出
                        sell_value = pos['shares'] * price
                        cash += sell_value * (1 - self.config.commission - self.config.stamp_duty)
                        trades.append(PortfolioTrade(
                            date=date_str, symbol=symbol,
                            name=symbol_names.get(symbol, ''),
                            direction='SELL', price=price,
                            shares=pos['shares'],
                            value=sell_value,
                            reason='信号卖出',
                        ))
                        del positions[symbol]
                    elif sig == Signal.SELL_HALF.value:
                        # 卖半仓
                        sell_shares = pos['shares'] / 2
                        sell_value = sell_shares * price
                        cash += sell_value * (1 - self.config.commission - self.config.stamp_duty)
                        pos['shares'] -= sell_shares
                        trades.append(PortfolioTrade(
                            date=date_str, symbol=symbol,
                            name=symbol_names.get(symbol, ''),
                            direction='SELL_HALF', price=price,
                            shares=sell_shares,
                            value=sell_value,
                            reason='半仓止盈',
                        ))

            # --- 4. 风控检查 ---
            current_equity = cash + sum(
                positions[s]['shares'] * day_prices.get(s, 0)
                for s in positions if s in day_prices
            )
            peak_equity = max(peak_equity, current_equity)

            # 组合止损（已熔断则跳过，等市场恢复后解除）
            stop, stop_reason = False, ''
            if not self._stopped:
                stop, stop_reason = self.risk_engine.check_portfolio_stop(
                    current_equity, self.config.initial_capital, peak_equity
                )
            if stop:
                # 强制平仓
                for symbol in list(positions.keys()):
                    pos = positions[symbol]
                    price = day_prices.get(symbol, 0)
                    if price > 0:
                        cash += pos['shares'] * price * (1 - self.config.commission - self.config.stamp_duty)
                        trades.append(PortfolioTrade(
                            date=date_str, symbol=symbol,
                            name=symbol_names.get(symbol, ''),
                            direction='SELL', price=price,
                            shares=pos['shares'],
                            value=pos['shares'] * price,
                            reason=f'风控平仓: {stop_reason}',
                        ))
                positions.clear()
                # 熔断：平仓但不断循环，市场恢复后可重新进场
                self._stopped = True

            # --- 5. 市场状态判断（动态调整仓位上限） ---
            market_max_pct = 1.0
            if hasattr(self.risk_engine, 'market_regime') and self.risk_engine.market_regime is not None:
                try:
                    # 传截至当天的完整行情，保证均线能正确计算
                    day_market_df = market_data.loc[:date]
                    if not day_market_df.empty and len(day_market_df) >= 60:
                        market_max_pct = self.risk_engine.market_regime.get_max_position_pct(day_market_df)
                except (KeyError, ValueError, AttributeError):
                    pass

            # 市场恢复 → 解除熔断，重置峰值基准
            if self._stopped and market_max_pct >= 1.0:
                self._stopped = False
                peak_equity = current_equity
                self.risk_engine.peak_equity = current_equity

            # 按市场状态计算有效仓位上限
            effective_max_pos = int(self.config.max_positions * market_max_pct)
            if effective_max_pos < len(positions):
                # 市场弱势，强制减仓到目标数量
                # 按持仓市值从大到小排序，保留最大的
                pos_values = [(s, positions[s]['shares'] * day_prices.get(s, 0))
                              for s in positions if s in day_prices]
                pos_values.sort(key=lambda x: x[1], reverse=True)
                keep_symbols = {s for s, _ in pos_values[:effective_max_pos]}
                for symbol in list(positions.keys()):
                    if symbol not in keep_symbols:
                        pos = positions[symbol]
                        price = day_prices.get(symbol, 0)
                        if price > 0:
                            cash += pos['shares'] * price * (1 - self.config.commission - self.config.stamp_duty)
                            trades.append(PortfolioTrade(
                                date=date_str, symbol=symbol,
                                name=symbol_names.get(symbol, ''),
                                direction='SELL', price=price,
                                shares=pos['shares'],
                                value=pos['shares'] * price,
                                reason=f'市场减仓(market={market_max_pct:.0%})',
                            ))
                            del positions[symbol]

            # --- 6. 买入阶段 ---
            if buy_signals and not self._stopped:
                valid = len(positions)
                max_pos = self.config.max_positions
                # 应用市场状态限制
                available_slots = max(0, effective_max_pos - valid)

                if available_slots > 0 and cash > self.config.min_capital_per_pos:
                    # 排除已有持仓的（已有仓位不重复买入）
                    new_buys = {s: v for s, v in buy_signals.items() if s not in positions}

                    if new_buys:
                        # 按信号强度排序
                        if hasattr(strategy, '_calc_upper'):
                            # 加分项：偏离上轨幅度
                            scored = {}
                            for s in new_buys:
                                score = 1.0
                                if s in day_prices and s in upper_cache:
                                    try:
                                        u_val = upper_cache[s].loc[date]
                                        if not pd.isna(u_val) and u_val > 0:
                                            score += (day_prices[s] - u_val) / u_val
                                    except (KeyError, TypeError):
                                        pass
                                scored[s] = score
                            sorted_buys = sorted(scored.items(), key=lambda x: x[1], reverse=True)
                        else:
                            sorted_buys = sorted(new_buys.items(), key=lambda x: x[1], reverse=True)

                        # 取 top 候选
                        top_candidates = sorted_buys[:available_slots * 2]

                        # 计算权重
                        signals_dict = {s: score for s, score in top_candidates}
                        current_positions_val = {
                            s: positions[s]['shares'] * day_prices.get(s, 0)
                            for s in positions if s in day_prices
                        }

                        target_weights = self.sizer.compute_weights(
                            available_cash=cash,
                            signals=signals_dict,
                            prices=day_prices,
                            atr_values=day_atr,
                            sector_map=sector_map,
                            current_positions=current_positions_val,
                        )

                        # 计算实际股数
                        target_shares = self.sizer.compute_shares(
                            target_weights, day_prices, cash)

                        # 执行买入
                        for symbol, (shares, cost) in target_shares.items():
                            if len(positions) >= effective_max_pos:
                                break
                            if symbol in day_data and shares >= 100:
                                cash -= cost * (1 + self.config.commission)
                                positions[symbol] = {
                                    'shares': shares,
                                    'avg_cost': cost / shares,
                                }
                                trades.append(PortfolioTrade(
                                    date=date_str, symbol=symbol,
                                    name=symbol_names.get(symbol, ''),
                                    direction='BUY', price=day_prices[symbol],
                                    shares=shares, value=cost,
                                    reason='信号买入',
                                ))

            # --- 7. 计算当日组合净值 ---
            current_equity = cash + sum(
                positions[s]['shares'] * day_prices.get(s, 0)
                for s in positions if s in day_prices
            )
            equity_curve.append(current_equity)

            # 每日持仓记录
            pos_record = {
                'date': date_str,
                'equity': current_equity,
                'cash': cash,
                'position_count': len(positions),
            }
            for s in positions:
                if s in day_prices:
                    pos_record[f'{s}_shares'] = positions[s]['shares']
                    pos_record[f'{s}_value'] = positions[s]['shares'] * day_prices[s]
            daily_positions_records.append(pos_record)

        # ===== 构建结果 =====
        equity_series = pd.Series(
            equity_curve,
            index=[all_dates[i] for i in range(len(equity_curve)) if i < len(all_dates)]
        )
        # 修正索引长度
        if len(equity_series) > 0:
            equity_series.index = all_dates[:len(equity_series)]

        # 基准曲线
        benchmark_curve = pd.Series(dtype=float)
        if benchmark_data is not None and not benchmark_data.empty:
            bench_reindexed = benchmark_data['close'].reindex(equity_series.index, method='ffill')
            if not bench_reindexed.empty and bench_reindexed.iloc[0] > 0:
                benchmark_curve = bench_reindexed / bench_reindexed.iloc[0] * equity_series.iloc[0]

        # 持仓DataFrame
        daily_positions_df = pd.DataFrame(daily_positions_records)
        if not daily_positions_df.empty:
            daily_positions_df.set_index('date', inplace=True)

        # 计算指标
        metrics = compute_portfolio_metrics(
            equity_series, trades, daily_positions_df, benchmark_curve)

        result = PortfolioResult(
            equity_curve=equity_series,
            benchmark_curve=benchmark_curve,
            daily_positions=daily_positions_df,
            trades=trades,
            final_capital=float(equity_series.iloc[-1]) if len(equity_series) > 0 else 0,
            peak_capital=peak_equity,
            **{k: v for k, v in metrics.items() if k in PortfolioResult.__dataclass_fields__},
        )
        # 补全缺少的字段
        for field_name in PortfolioResult.__dataclass_fields__:
            current = getattr(result, field_name, None)
            # 检查是否还是默认值（需要补全）
            is_default = current is None
            if isinstance(current, pd.Series) and current.empty:
                is_default = True
            if isinstance(current, (int, float)) and current == 0 and field_name in metrics:
                is_default = True

            if is_default and field_name in metrics:
                try:
                    setattr(result, field_name, metrics[field_name])
                except Exception:
                    pass

        return result

    def run_batch(self,
                  universe_sets: Dict[str, List[str]],
                  strategy,
                  start_date: str,
                  end_date: str,
                  sector_map: Dict[str, List[str]] = None,
                  symbol_names: Dict[str, str] = None) -> Dict[str, PortfolioResult]:
        """
        批量运行多个组合回测

        Args:
            universe_sets: {set_name: [symbols]}
            strategy: 策略实例
            start_date, end_date: 日期
            sector_map, symbol_names: 同上

        Returns:
            {set_name: PortfolioResult}
        """
        results = {}
        for name, universe in universe_sets.items():
            print(f"\n{'='*60}")
            print(f"组合回测: {name} ({len(universe)} 支)")
            print('=' * 60)
            result = self.run(
                universe=universe,
                strategy=strategy,
                start_date=start_date,
                end_date=end_date,
                sector_map=sector_map,
                symbol_names=symbol_names,
            )
            results[name] = result
        return results
