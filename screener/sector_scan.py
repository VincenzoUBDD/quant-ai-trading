# 板块异动选股模块
# 收集新闻 → 定位板块异动 → 筛选个股
# 用法: python screener/sector_scan.py
import sys, os
if sys.stdout.encoding != 'utf-8':
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_fetcher import DataFetcher
from strategy.regression_channel.channel_strategy import RegressionChannelWithVolumeStrategy
from config import CHANNEL_WINDOW, VOL_WINDOW, VOL_MULTIPLIER, TODAY, DATE_RECENT_START
from stocks import SECTOR_MAP, SCREENER_WATCHLIST
import pandas as pd

strategy = RegressionChannelWithVolumeStrategy(
    window=CHANNEL_WINDOW, vol_window=VOL_WINDOW, vol_multiplier=VOL_MULTIPLIER
)

# 构建股票名称查询
_STOCK_NAMES = {c: n for c, n in SCREENER_WATCHLIST}


def get_name(code):
    return _STOCK_NAMES.get(code, '')


def scan_stock(code):
    """扫描单只股票，返回信号结果"""
    df = DataFetcher.get_stock_hist(code, DATE_RECENT_START, TODAY)
    if df.empty or len(df) < 60:
        return None

    df['upper'] = strategy._calc_upper(df['high'])
    df['vol_ma'] = df['volume'].rolling(VOL_WINDOW).mean()

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    close = latest['close']
    u = latest['upper']
    vol_ratio = latest['volume'] / latest['vol_ma'] if latest['vol_ma'] > 0 else 0
    above_pct = (close - u) / u * 100

    prev_below = prev['close'] < prev['upper']
    breakout = close >= u
    vol_ok = latest['volume'] > latest['vol_ma'] * VOL_MULTIPLIER

    if breakout and vol_ok and prev_below:
        signal = '★买入'
        priority = 0
    elif breakout and vol_ok:
        signal = '★买入'
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
        'code': code,
        'name': get_name(code),
        'signal': signal,
        'priority': priority,
        'close': round(close, 2),
        'upper': round(u, 2),
        'above_pct': round(above_pct, 1),
        'vol_ratio': round(vol_ratio, 2),
    }


def fetch_news(code, limit=3):
    """获取股票近期新闻"""
    try:
        import akshare as ak
        news_df = ak.stock_news_em(symbol=code)
        if news_df.empty:
            return []
        recent = news_df.head(limit)
        return [str(row['新闻标题'])[:60] for _, row in recent.iterrows()]
    except Exception:
        return []


def score_sector(sector_signals):
    """计算板块热度分数"""
    score = 0
    for s in sector_signals:
        if s['signal'] == '★买入':
            score += 3
            if s['vol_ratio'] >= 2:
                score += 1
        elif s['signal'] == '突破':
            score += 1
    return score


def run():
    """主流程"""
    print('=' * 60)
    print('板块异动选股 — %s' % TODAY)
    print('=' * 60)
    print('策略: 回归通道 + 放量突破')
    print()

    # 扫描每个板块
    sector_results = {}
    all_signals = []

    for sector_name, codes in SECTOR_MAP.items():
        signals = []
        for code in codes:
            try:
                r = scan_stock(code)
                if r and r['signal'] != '-':
                    signals.append(r)
            except Exception:
                pass

        sector_results[sector_name] = signals
        all_signals.extend(signals)

    if not all_signals:
        print('所有板块均无信号')
        return

    # 按热度排序
    ranked = []
    for sector_name, signals in sector_results.items():
        heat = score_sector(signals)
        buy_count = sum(1 for s in signals if s['signal'] == '★买入')
        break_count = sum(1 for s in signals if s['signal'] == '突破')
        vol_count = sum(1 for s in signals if s['signal'] == '放量')
        ranked.append((sector_name, signals, heat, buy_count, break_count, vol_count))

    ranked.sort(key=lambda x: -x[2])

    # 输出
    print('>> 板块热度榜 <<')
    max_heat = max(r[2] for r in ranked) or 1
    for sector_name, signals, heat, buy_count, break_count, vol_count in ranked:
        parts = []
        if buy_count:
            parts.append(f'★{buy_count}')
        if break_count:
            parts.append(f'Δ{break_count}')
        if vol_count:
            parts.append(f'○{vol_count}')
        tag = '  '.join(parts) if parts else '无信号'
        bar_len = int(heat / max_heat * 10)
        bar = '█' * bar_len + '░' * (10 - bar_len)
        print(f'\n  {sector_name}  {tag}  热度: {bar} ({heat}分)')

        if not signals:
            print('    └─ 无信号')
            continue

        for i, s in enumerate(signals):
            prefix = '├─' if i < len(signals) - 1 else '└─'
            print(f'    {prefix} {s["code"]} {s["name"]}  {s["signal"]}  {s["above_pct"]:+.1f}%  量比{s["vol_ratio"]:.2f}x')

        # 为有信号的板块拉新闻
        news_codes = [s['code'] for s in signals[:3]]
        news_items = []
        for nc in news_codes:
            news_items.extend(fetch_news(nc, limit=2))

        if news_items:
            for i, item in enumerate(news_items[:3]):
                prefix = '├─' if i < min(len(news_items[:3]), 3) - 1 else '└─'
                print(f'    {prefix} 📰 {item}')

    # 汇总
    total_buy = sum(1 for s in all_signals if s['signal'] == '★买入')
    total_break = sum(1 for s in all_signals if s['signal'] == '突破')
    hot_sectors = [r[0] for r in ranked if r[2] > 0]

    print()
    print('=' * 60)
    total_vol = sum(1 for s in all_signals if s['signal'] == '放量')
    print('汇总: %d个板块活跃 | ★买入%d 突破%d 放量%d' % (len(hot_sectors), total_buy, total_break, total_vol))
    print('活跃板块: %s' % '  '.join(hot_sectors))
    print('=' * 60)


if __name__ == '__main__':
    run()
