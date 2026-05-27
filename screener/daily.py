# 模块3：开盘前例行检查
# 用法:
#   python screener/daily.py --holdings    检查持仓
#   python screener/daily.py --candidates  回顾昨日候选
#   python screener/daily.py --news        扫描关注股新闻
#   python screener/daily.py --all         全部检查 (默认)
import sys, os
from datetime import datetime
if sys.stdout.encoding != 'utf-8':
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_fetcher import DataFetcher
from utils.fundamental import FundamentalAnalyzer, print_report
from strategy.regression_channel.channel_strategy import RegressionChannelWithVolumeStrategy
from config import CHANNEL_WINDOW, VOL_WINDOW, VOL_MULTIPLIER, TODAY as END_DATE, DATE_RECENT_START as START_DATE, FUNDAMENTAL_MIN_SCORE
from stocks import WATCH_LIST, SCREENER_WATCHLIST
import pandas as pd

CACHE_FILE = os.path.join(os.path.dirname(__file__), 'results', 'latest_screen.csv')

strategy = RegressionChannelWithVolumeStrategy(
    window=CHANNEL_WINDOW, vol_window=VOL_WINDOW, vol_multiplier=VOL_MULTIPLIER
)


def check_holdings():
    """检查持仓股票：卖出信号、回撤止盈"""
    print('=' * 60)
    print('【持仓检查】')
    print('=' * 60)

    holdings = {code: info for code, info in WATCH_LIST.items() if info['shares'] > 0}
    if not holdings:
        print('当前无持仓')
        return

    for code, info in holdings.items():
        shares = info['shares']
        cost = info['cost']
        df = DataFetcher.get_stock_hist(code, START_DATE, END_DATE)
        if df.empty:
            print('%s: 数据获取失败' % code)
            continue

        df['signal'] = strategy.generate_signals(df)
        df['upper'] = strategy._calc_upper(df['high'])
        df['vol_ma'] = df['volume'].rolling(VOL_WINDOW).mean()

        latest = df.iloc[-1]
        close = latest['close']
        u = latest['upper']
        pnl_pct = (close - cost) / cost * 100
        pnl_value = (close - cost) * shares

        # 检查是否有卖出信号
        last_signal = df[df['signal'] != 0]
        sell_signal = False
        if not last_signal.empty and last_signal.iloc[-1]['signal'] == -1:
            sell_signal = True
            sell_date = last_signal.index[-1].date()

        # 检查回撤止盈
        buy_signals = df[df['signal'] == 1].index
        trailing_stop = False
        if len(buy_signals) > 0:
            last_buy = buy_signals[-1]
            sell_after = df[(df.index > last_buy) & (df['signal'] == -1)].index
            if len(sell_after) == 0:
                # 仍在持仓，检查回撤
                since_buy = df.loc[last_buy:]
                highest = since_buy['close'].max()
                dd = (highest - close) / highest * 100
                if dd >= 10:
                    trailing_stop = True

        # 主动预警：跌破/接近上轨 + 斜率下行
        latest_slope = 0
        early_warn = None
        if not pd.isna(u):
            latest_slope = strategy._slope.iloc[-1] if not pd.isna(strategy._slope.iloc[-1]) else 0
            if latest_slope < 0:
                if close <= u:
                    early_warn = '跌破'
                else:
                    above_pct_upper = (close - u) / u * 100
                    if above_pct_upper < 1:
                        early_warn = '接近'

        print()
        print('▶ %s  |  %d股  |  成本 %.2f  |  现价 %.2f' % (code, shares, cost, close))
        print('  盈亏: %+.2f%% (%+.0f)  |  上轨: %.2f  |  斜率: %.4f' % (pnl_pct, pnl_value, u, latest_slope))

        if sell_signal:
            print('  ⚠ 策略信号: 卖出 (%s)' % sell_date)
        if trailing_stop:
            print('  ⚠ 回撤止盈: 从高点回撤 %.1f%% >= 10%%' % dd)
        if not sell_signal and early_warn:
            print('  ⚠ 主动预警: %s上轨 + 斜率下行(%.4f)，建议卖出' % (early_warn, latest_slope))
        if not sell_signal and not trailing_stop:
            if pnl_pct > 0:
                print('  ✅ 持有中，利润垫 %.1f%%' % pnl_pct)
            else:
                print('  ⏳ 持有中，浮亏 %.1f%%' % pnl_pct)

        # 最近信号
        sigs = df[df['signal'] != 0]['signal'].tail(4)
        if not sigs.empty:
            parts = []
            for d, s in sigs.items():
                parts.append('%s %s' % (d.date(), '买' if s == 1 else '卖'))
            print('  最近信号: ' + ' | '.join(parts))


def check_holdings_fundamental():
    """检查持仓股基本面健康度"""
    print('=' * 60)
    print('【持仓基本面检查】')
    print('=' * 60)

    holdings = {code: info for code, info in WATCH_LIST.items() if info['shares'] > 0}
    if not holdings:
        print('当前无持仓')
        return

    analyzer = FundamentalAnalyzer()
    for code, info in holdings.items():
        df = DataFetcher.get_stock_hist(code, START_DATE, END_DATE)
        price = df['close'].iloc[-1] if not df.empty else None
        fund_data = analyzer.fetch_fundamentals(code, price=price)
        if not fund_data:
            print('▶ %s: 基本面数据获取失败' % code)
            continue
        score = analyzer.compute_score(fund_data)
        print()
        print('▶ %s' % code)
        print_report(fund_data, score)
        if score.total_score < FUNDAMENTAL_MIN_SCORE:
            print('  ⚠ 基本面评分 %.1f < %.0f，建议关注风险' % (score.total_score, FUNDAMENTAL_MIN_SCORE))
    print()


def check_candidates():
    """回顾昨日候选买入股票是否仍有效"""
    print('=' * 60)
    print('【候选股回顾】')
    print('=' * 60)

    if not os.path.exists(CACHE_FILE):
        print('无缓存数据，请先运行 python screener/screen.py')
        return

    cache = pd.read_csv(CACHE_FILE, encoding='utf-8-sig', dtype={'code': str})
    buys = cache[cache['signal'] == '★买入']
    if buys.empty:
        print('昨日扫描无买入信号')
        return

    print()
    for _, row in buys.iterrows():
        code = row['code']
        name = row['name']
        prev_close = row['close']
        prev_upper = row['upper']

        # 重新获取最新数据确认
        df = DataFetcher.get_stock_hist(code, START_DATE, END_DATE)
        if df.empty:
            continue

        latest = df.iloc[-1]
        close = latest['close']
        df['upper'] = strategy._calc_upper(df['high'])
        df['vol_ma'] = df['volume'].rolling(VOL_WINDOW).mean()
        u = df['upper'].iloc[-1]
        vol_ratio = latest['volume'] / df['vol_ma'].iloc[-1] if df['vol_ma'].iloc[-1] > 0 else 0
        above_pct = (close - u) / u * 100

        still_valid = '✅ 有效' if close >= u else '⚠ 已回落'
        print('▶ %s %s' % (code, name))
        print('  昨日: 收盘 %.2f  上轨 %.2f' % (prev_close, prev_upper))
        print('  今日: 收盘 %.2f  上轨 %.2f  %+.1f%%  量比 %.2fx' % (
            close, u, above_pct, vol_ratio))
        print('  状态: %s' % still_valid)
        print()


def check_news():
    """扫描关注股票的近期新闻"""
    print('=' * 60)
    print('【相关新闻扫描】')
    print('=' * 60)

    import akshare as ak
    today = datetime.now().strftime('%Y-%m-%d')
    print('日期: %s' % today)
    print()

    for code, name in SCREENER_WATCHLIST:
        try:
            news_df = ak.stock_news_em(symbol=code)
            if news_df.empty:
                continue

            # 筛选近3天的新闻
            recent = news_df[news_df['发布时间'].str.contains(
                '|'.join(['2026-05-0', '2026-04-3']), na=False)]
            if recent.empty:
                # 至少看最近一条
                recent = news_df.head(1)

            print('▶ %s %s' % (code, name))
            for _, row in recent.head(2).iterrows():
                title = str(row['新闻标题'])[:80]
                pub_time = str(row['发布时间'])[:10]
                print('  [%s] %s' % (pub_time, title))
            print()

        except Exception as e:
            pass


def check_portfolio_perspective():
    """组合视角：板块集中度、总仓位比例"""
    print('=' * 60)
    print('【组合视角】')
    print('=' * 60)

    holdings = {code: info for code, info in WATCH_LIST.items() if info['shares'] > 0}
    if not holdings:
        print('当前无持仓')
        return

    from stocks import SECTOR_MAP

    # 计算各股市值
    total_value = 0
    pos_values = {}
    for code, info in holdings.items():
        df = DataFetcher.get_stock_hist(code, START_DATE, END_DATE)
        price = df['close'].iloc[-1] if not df.empty else info['cost']
        value = info['shares'] * price
        total_value += value
        pos_values[code] = value

    print(f'\n总市值: {total_value:,.2f}  持仓数: {len(holdings)}')

    # 板块分布
    sym_to_sector = {}
    for sector, symbols in SECTOR_MAP.items():
        for s in symbols:
            sym_to_sector[s] = sector

    sector_values = {}
    for code, val in pos_values.items():
        sec = sym_to_sector.get(code, '其他')
        sector_values[sec] = sector_values.get(sec, 0) + val

    print('\n板块分布:')
    for sector, val in sorted(sector_values.items(), key=lambda x: x[1], reverse=True):
        pct = val / total_value if total_value > 0 else 0
        bar = '█' * int(pct * 30)
        over = ' ⚠ 超限' if pct > 0.40 else ''
        print(f'  {sector:<12} {pct:>6.2%} {bar}{over}')

    # 集中度
    pcts = [v / total_value for v in pos_values.values()]
    stock_hhi = sum(p ** 2 for p in pcts)
    top1 = max(pcts) if pcts else 0
    top3 = sum(sorted(pcts, reverse=True)[:3])
    print(f'\n个股HHI: {stock_hhi:.4f}  Top1: {top1:.1%}  Top3: {top3:.1%}')

    print()


def run_all():
    """运行全部检查"""
    print('=' * 60)
    print('开盘前例行检查 — %s' % datetime.now().strftime('%Y-%m-%d'))
    print('=' * 60)
    print()
    check_holdings()
    print()
    check_holdings_fundamental()
    print()
    check_portfolio_perspective()
    print()
    check_candidates()
    print()
    check_news()


if __name__ == '__main__':
    mode = '--all'
    if len(sys.argv) > 1:
        mode = sys.argv[1]

    if mode == '--holdings':
        check_holdings()
    elif mode == '--fundamental':
        check_holdings_fundamental()
    elif mode == '--candidates':
        check_candidates()
    elif mode == '--news':
        check_news()
    else:
        run_all()
