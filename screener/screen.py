# 模块1：扫描关注列表，输出买入信号
# 用法: python screener/screen.py
import sys, os
if sys.stdout.encoding != 'utf-8':
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_fetcher import DataFetcher
from utils.fundamental import FundamentalAnalyzer, print_report
from strategy.regression_channel.channel_strategy import RegressionChannelWithVolumeStrategy
from config import CHANNEL_WINDOW, VOL_WINDOW, VOL_MULTIPLIER, VOL_ACTIVE_THRESHOLD, MARKET_INDEX, MARKET_MA_PERIOD, TODAY, DATE_RECENT_START, FUNDAMENTAL_MIN_SCORE
from stocks import SCREENER_WATCHLIST
import pandas as pd


END_DATE = TODAY
START_DATE = DATE_RECENT_START
CACHE_DIR = os.path.join(os.path.dirname(__file__), 'results')
CACHE_FILE = os.path.join(CACHE_DIR, 'latest_screen.csv')


def scan_stock(code, name, strategy):
    """扫描单只股票，返回信号结果"""
    df = DataFetcher.get_stock_hist(code, START_DATE, END_DATE)
    if df.empty or len(df) < 60:
        return None

    df['upper'] = strategy._calc_upper(df['high'])
    df['vol_ma'] = df['volume'].rolling(VOL_WINDOW).mean()
    df['vol_long'] = df['volume'].rolling(252).mean()
    active_period = df['vol_ma'] > df['vol_long'] * VOL_ACTIVE_THRESHOLD

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    close = latest['close']
    u = latest['upper']
    vol_ratio = latest['volume'] / latest['vol_ma'] if latest['vol_ma'] > 0 else 0
    above_pct = (close - u) / u * 100

    prev_below = prev['close'] < prev['upper']
    breakout = close >= u
    # 放量判定：活跃期自动放量 / 否则卡 1.5x
    vol_ok = latest['volume'] > latest['vol_ma'] * VOL_MULTIPLIER
    if not vol_ok and latest['vol_long'] > 0 and not pd.isna(latest['vol_long']):
        vol_ok = bool(active_period.iloc[-1])

    # 信号判断: 新鲜突破 = 前一日未突破 + 当日放量突破
    if breakout and vol_ok and prev_below:
        signal = '★买入'
        priority = 0
    elif breakout and vol_ok:
        signal = '★买入'   # 连续突破也算
        priority = 0
    elif breakout:
        signal = '突破'
        priority = 2
    elif vol_ok:
        signal = '放量'
        priority = 3
    else:
        signal = '-'
        priority = 4

    return {
        'code': code, 'name': name,
        'close': round(close, 2), 'upper': round(u, 2),
        'above_pct': round(above_pct, 1),
        'vol_ratio': round(vol_ratio, 2),
        'signal': signal, 'priority': priority,
        'date': str(df.index[-1].date())
    }


def scan_all(use_fundamental=True):
    """扫描全部关注列表"""
    strategy = RegressionChannelWithVolumeStrategy(
        window=CHANNEL_WINDOW, vol_window=VOL_WINDOW, vol_multiplier=VOL_MULTIPLIER
    )

    analyzer = FundamentalAnalyzer(min_score=FUNDAMENTAL_MIN_SCORE) if use_fundamental else None

    results = []
    for code, name in SCREENER_WATCHLIST:
        try:
            r = scan_stock(code, name, strategy)
            if r:
                # 取最新价格供 PE 计算
                df = DataFetcher.get_stock_hist(code, DATE_RECENT_START, TODAY)
                price = df['close'].iloc[-1] if not df.empty else None
                if analyzer and price:
                    fund = analyzer.fetch_fundamentals(code, name, price=price)
                    if fund:
                        score = analyzer.compute_score(fund)
                        r['fund_score'] = round(score.total_score, 1)
                        r['fund_grade'] = score.grade
                if 'fund_score' not in r:
                    r['fund_score'] = None
                    r['fund_grade'] = 'N/A'
                results.append(r)
        except Exception:
            pass

    # 按优先级、偏离度排序
    results.sort(key=lambda r: (r['priority'], -r['above_pct']))
    return results


def check_market():
    """检查大盘环境"""
    try:
        df = DataFetcher.get_index_hist(MARKET_INDEX)
        if df.empty:
            return None
        ma = df['close'].rolling(MARKET_MA_PERIOD).mean()
        latest = df.iloc[-1]
        return {
            'close': latest['close'],
            'ma': ma.iloc[-1],
            'above': latest['close'] > ma.iloc[-1]
        }
    except:
        return None


def print_results(results):
    """格式化输出"""
    print('数据截止: %s' % END_DATE)
    print('=' * 75)

    # 大盘环境
    market = check_market()
    if market:
        status = 'OK' if market['above'] else '谨慎'
        print('大盘: 上证%.0f  MA%d:%.0f  [%s]' % (
            market['close'], MARKET_MA_PERIOD, market['ma'], status))
    print('=' * 75)
    print('扫描 %d 只 — 策略: 线性回归通道 + 放量突破' % len(results))
    print('=' * 75)
    print()

    # ★ 买入信号
    buys = [r for r in results if r['signal'] == '★买入']
    if buys:
        print('>> ★ 买入信号 (%d只) <<' % len(buys))
        print('%-8s %-8s %8s %8s %7s %6s  %s' % ('代码', '名称', '收盘', '上轨', '偏离', '量比', '基本面'))
        print('-' * 65)
        for r in buys:
            fund_str = '%s/%s' % (r.get('fund_score', 'N/A'), r.get('fund_grade', 'N/A'))
            print('%-8s %-8s %8.2f %8.2f %+5.1f%% %5.2fx  %s' % (
                r['code'], r['name'], r['close'], r['upper'], r['above_pct'], r['vol_ratio'], fund_str))
        print()

    # 突破信号
    breaks = [r for r in results if r['signal'] == '突破']
    if breaks:
        print('>> 突破待确认 (无量) (%d只) <<' % len(breaks))
        print('%-8s %-8s %8s %8s %7s %6s  %s' % ('代码', '名称', '收盘', '上轨', '偏离', '量比', '基本面'))
        print('-' * 65)
        for r in breaks:
            fund_str = '%s/%s' % (r.get('fund_score', 'N/A'), r.get('fund_grade', 'N/A'))
            print('%-8s %-8s %8.2f %8.2f %+5.1f%% %5.2fx  %s' % (
                r['code'], r['name'], r['close'], r['upper'], r['above_pct'], r['vol_ratio'], fund_str))
        print()

    # 其他
    others = [r for r in results if r['signal'] not in ('★买入', '突破')]
    if others:
        print('>> 其他 (%d只) <<' % len(others))
        print('%-8s %-8s %8s %8s %7s %6s  %s' % ('代码', '名称', '收盘', '上轨', '偏离', '量比', '基本面'))
        print('-' * 65)
        for r in others[:10]:
            fund_str = '%s/%s' % (r.get('fund_score', 'N/A'), r.get('fund_grade', 'N/A'))
            print('%-8s %-8s %8.2f %8.2f %+5.1f%% %5.2fx  %s' % (
                r['code'], r['name'], r['close'], r['upper'], r['above_pct'], r['vol_ratio'], fund_str))
        if len(others) > 10:
            print('  ... 还有 %d 只' % (len(others) - 10))
    print()

    # 基本面汇总
    scores = [r['fund_score'] for r in results if r['fund_score'] is not None]
    if scores:
        excellent = sum(1 for s in scores if s >= 8)
        good = sum(1 for s in scores if 6 <= s < 8)
        fair = sum(1 for s in scores if 4 <= s < 6)
        poor = sum(1 for s in scores if s < 4)
        print('>> 基本面评分汇总 <<')
        print('  优秀(>=8): %d只  良好(6-8): %d只  一般(4-6): %d只  较差/很差(<4): %d只' % (
            excellent, good, fair, poor))
        print()


def cache_results(results):
    """缓存结果到 CSV"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    df = pd.DataFrame(results)
    df['code'] = df['code'].astype(str)  # 保留前导零
    df.to_csv(CACHE_FILE, index=False, encoding='utf-8-sig')
    print('结果已缓存: %s' % CACHE_FILE)


if __name__ == '__main__':
    results = scan_all()
    print_results(results)
    cache_results(results)
