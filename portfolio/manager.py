#!python
"""实盘组合管理器 — 每日运行：检查持仓 + 候选扫描 + 调仓建议

用法:
    python portfolio/manager.py                  # 显示当前组合状态
    python portfolio/manager.py --portfolio      # 组合回测（从 SCREENER_WATCHLIST 选股）
    python portfolio/manager.py --report         # 详细风险报告
"""
import sys, os
from datetime import datetime
if sys.stdout.encoding != 'utf-8':
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from portfolio.config import PortfolioConfig
from portfolio.engine import PortfolioEngine
from portfolio.exposure_manager import ExposureManager
from portfolio.risk_engine import RiskEngine
from portfolio.position_sizer import PositionSizer
from config import CHANNEL_WINDOW, VOL_WINDOW, VOL_MULTIPLIER, \
    TODAY, DATE_RECENT_START, FUNDAMENTAL_MIN_SCORE
from stocks import WATCH_LIST, SCREENER_WATCHLIST, SECTOR_MAP
from utils.data_fetcher import DataFetcher
from utils.fundamental import FundamentalAnalyzer
from strategy.regression_channel.channel_strategy import RegressionChannelWithVolumeStrategy
from strategy.MA.base import Signal


class PortfolioManager:
    """实盘组合管理器"""

    def __init__(self):
        self.config = PortfolioConfig()
        self.strategy = RegressionChannelWithVolumeStrategy(
            window=CHANNEL_WINDOW, vol_window=VOL_WINDOW, vol_multiplier=VOL_MULTIPLIER,
        )
        self.sizer = PositionSizer(self.config)
        self.risk_engine = RiskEngine(self.config)
        self.exposure_mgr = ExposureManager(
            sector_map=SECTOR_MAP,
            sector_limit=self.config.sector_exposure_limit,
            stock_limit=self.config.max_position_pct,
        )

    def show_portfolio_status(self):
        """显示当前组合状态（持仓 + 盈亏 + 板块分布）"""
        print('=' * 60)
        print(f'【投资组合状态】 {datetime.now().strftime("%Y-%m-%d %H:%M")}')
        print('=' * 60)

        holdings = {code: info for code, info in WATCH_LIST.items() if info['shares'] > 0}
        if not holdings:
            print('当前无持仓')
            return

        total_value = 0
        pos_values = {}

        print(f'\n{"代码":<8} {"名称":<10} {"股数":<8} {"成本":<10} {"现价":<10} {"市值":<10} {"盈亏":<10} {"盈亏%":<8}')
        print('-' * 80)

        for code, info in holdings.items():
            df = DataFetcher.get_stock_hist(code, DATE_RECENT_START, TODAY)
            if not df.empty:
                price = df['close'].iloc[-1]
            else:
                price = info['cost']

            shares = info['shares']
            cost = info['cost']
            value = shares * price
            pnl = value - shares * cost
            pnl_pct = (price - cost) / cost * 100

            # 查找名称
            name = ''
            for c, n in SCREENER_WATCHLIST:
                if c == code:
                    name = n
                    break

            total_value += value
            pos_values[code] = value

            print(f'{code:<8} {name:<10} {shares:<8} {cost:<10.3f} {price:<10.3f} '
                  f'{value:<10.0f} {pnl:<+10.0f} {pnl_pct:<+8.2f}%')

        print('-' * 80)
        print(f'{"总市值":>40} {total_value:>10.0f}')
        print()

        # 板块分布
        print('-- 板块分布 --')
        sector_exp = self.exposure_mgr.compute_sector_exposure(pos_values, total_value)
        for sector, info in sorted(sector_exp.items(), key=lambda x: x[1]['pct'], reverse=True):
            bar = '█' * int(info['pct'] * 30)
            print(f'  {sector:<12} {info["pct"]:>6.2%} {bar} {info["status"]}')

        # 集中度
        conc = self.exposure_mgr.compute_concentration(pos_values, total_value)
        print(f'\n个股HHI: {conc["stock_hhi"]:.4f}  |  板块HHI: {conc["sector_hhi"]:.4f}')
        print(f'Top1: {conc["top1_pct"]:.1%}  |  Top3: {conc["top3_pct"]:.1%}')

        # 告警
        warnings = self.exposure_mgr.check_limits(pos_values, total_value)
        if warnings:
            print('\n-- 告警 --')
            for w in warnings:
                print(f'  ⚠ {w}')

    def show_candidate_signals(self):
        """显示当日候选买入信号"""
        print('\n' + '=' * 60)
        print('【候选买入信号】')
        print('=' * 60)

        signals = {}
        prices = {}
        for code, name in SCREENER_WATCHLIST:
            try:
                df = DataFetcher.get_stock_hist(code, DATE_RECENT_START, TODAY)
                if df.empty or len(df) < 60:
                    continue

                upper = self.strategy._calc_upper(df['high'])
                vol_ma = df['volume'].rolling(VOL_WINDOW).mean()
                latest = df.iloc[-1]
                prev = df.iloc[-2]
                close = latest['close']
                u = upper.iloc[-1]
                vol_ratio = latest['volume'] / vol_ma.iloc[-1] if vol_ma.iloc[-1] > 0 else 0

                if pd.isna(u) or pd.isna(vol_ma.iloc[-1]):
                    continue

                # 放量判定
                vol_ok = latest['volume'] > vol_ma.iloc[-1] * VOL_MULTIPLIER
                if not vol_ok:
                    vol_long = df['volume'].rolling(252).mean()
                    if not pd.isna(vol_long.iloc[-1]):
                        active = vol_ma.iloc[-1] > vol_long.iloc[-1] * 1.3
                        vol_ok = bool(active)

                prev_below = prev['close'] < upper.iloc[-2]
                breakout = close >= u

                if breakout and vol_ok and prev_below:
                    # 信号强度：用偏离上轨幅度作评分
                    strength = (close - u) / u
                    signals[code] = 1.0 + strength
                    prices[code] = close
                    print(f'  ★ {code} {name:<8} 收盘 {close:<8.2f} 上轨 {u:<8.2f} '
                          f'偏离 {strength:<+7.2%} 量比 {vol_ratio:.2f}x')
            except Exception:
                continue

        if not signals:
            print('  当前无买入信号')
            return signals, prices

        # 仓位分配建议
        print('\n-- 仓位分配建议 --')
        holdings_val = {}
        for code, info in WATCH_LIST.items():
            if info['shares'] > 0:
                df = DataFetcher.get_stock_hist(code, DATE_RECENT_START, TODAY)
                price = df['close'].iloc[-1] if not df.empty else info['cost']
                holdings_val[code] = info['shares'] * price

        available_cash = self.config.initial_capital - sum(holdings_val.values())
        if available_cash < self.config.min_capital_per_pos:
            print('  可用资金不足')
            return signals, prices

        weights = self.sizer.compute_weights(
            available_cash=available_cash,
            signals=signals,
            prices=prices,
            sector_map=SECTOR_MAP,
            current_positions=holdings_val,
        )
        shares_info = self.sizer.compute_shares(weights, prices, available_cash)

        print(f'  可用资金: {available_cash:>10.0f}')
        print(f'  {"代码":<8} {"名称":<8} {"建议":<6} {"股数":<8} {"金额":<10} {"板块":<10}')
        print('  ' + '-' * 55)

        for code, (shares, cost) in sorted(shares_info.items(), key=lambda x: x[1][0], reverse=True):
            if shares < 100:
                continue
            # 查找板块和名称
            name = ''
            sector = '其他'
            for c, n in SCREENER_WATCHLIST:
                if c == code:
                    name = n
                    break
            for sec, syms in SECTOR_MAP.items():
                if code in syms:
                    sector = sec
                    break
            action = '加仓' if code in holdings_val else '买入'
            print(f'  {code:<8} {name:<8} {action:<6} {shares:<8} {cost:<10.0f} {sector:<10}')

        return signals, prices

    def run_portfolio_backtest(self):
        """运行组合回测（从 SCREENER_WATCHLIST 选股）"""
        print('=' * 60)
        print('【投资组合回测】')
        print('=' * 60)
        print(f'股票池: SCREENER_WATCHLIST ({len(SCREENER_WATCHLIST)} 支)')
        print(f'最大持仓: {self.config.max_positions} 只')
        print(f'仓位方式: {self.config.sizing_method}')
        print(f'初始资金: {self.config.initial_capital:,.0f}')
        print(f'板块上限: {self.config.sector_exposure_limit:.0%}')
        print()

        universe = [c for c, _ in SCREENER_WATCHLIST]
        symbol_names = {c: n for c, n in SCREENER_WATCHLIST}

        engine = PortfolioEngine(self.config)
        result = engine.run(
            universe=universe,
            strategy=self.strategy,
            start_date='20220101',
            end_date='20260101',
            sector_map=SECTOR_MAP,
            symbol_names=symbol_names,
        )

        # === 输出结果 ===
        print('\n' + '=' * 60)
        print('【回测结果】')
        print('=' * 60)

        print(f'\n总收益率:     {result.total_return:>+8.2%}')
        print(f'年化收益率:   {result.annual_return:>+8.2%}')
        print(f'最大回撤:     {result.max_drawdown:>8.2%}')
        print(f'夏普比率:     {result.sharpe_ratio:>8.2f}')
        print(f'Sortino比率:  {result.sortino_ratio:>8.2f}')
        print(f'Calmar比率:   {result.calmar_ratio:>8.2f}')
        print(f'年化波动率:   {result.volatility:>8.2%}')
        print(f'Beta:         {result.beta:>8.2f}')
        print(f'Alpha:        {result.alpha:>+8.2%}')

        if result.information_ratio != 0:
            print(f'信息比率:     {result.information_ratio:>8.2f}')
        if result.var_95 != 0:
            print(f'VaR(95%):     {result.var_95:>8.2%}')
            print(f'CVaR(95%):    {result.cvar_95:>8.2%}')
        if result.max_consecutive_loss > 0:
            print(f'最大连续亏损: {result.max_consecutive_loss} 次')
        if result.win_rate > 0:
            print(f'交易胜率:     {result.win_rate:>8.2%}')
        if result.profit_loss_ratio > 0:
            print(f'盈亏比:       {result.profit_loss_ratio:>8.2f}')
        if result.avg_position_count > 0:
            print(f'平均持仓数:   {result.avg_position_count:>8.1f}')

        print(f'\n交易次数:     {len(result.trades)}')
        print(f'最终资金:     {result.final_capital:>10,.0f}')

        # 与基准对比
        if not result.benchmark_curve.empty:
            bench_ret = (result.benchmark_curve.iloc[-1] / result.benchmark_curve.iloc[0]) - 1
            print(f'基准收益(沪): {bench_ret:>+8.2%}')
            excess = result.total_return - bench_ret
            print(f'超额收益:     {excess:>+8.2%}')

        # 板块暴露
        if not result.daily_positions.empty:
            latest_pos = {}
            for col in result.daily_positions.columns:
                if col.endswith('_value'):
                    sym = col.replace('_value', '')
                    val = result.daily_positions[col].iloc[-1]
                    if val > 0:
                        latest_pos[sym] = val
            if latest_pos:
                total_val = sum(latest_pos.values())
                print('\n-- 最终持仓 --')
                for sym, val in sorted(latest_pos.items(), key=lambda x: x[1], reverse=True):
                    name = symbol_names.get(sym, '')
                    print(f'  {sym} {name:<8}: {val:>10,.0f} ({val/total_val:>6.2%})')

        return result


def main():
    pm = PortfolioManager()

    if '--portfolio' in sys.argv:
        pm.run_portfolio_backtest()
    elif '--report' in sys.argv:
        pm.show_portfolio_status()
        pm.show_candidate_signals()
    else:
        pm.show_portfolio_status()
        pm.show_candidate_signals()


if __name__ == '__main__':
    main()
