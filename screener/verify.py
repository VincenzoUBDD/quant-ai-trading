# 模块2：回测验证 — 对候选股票跑3年回测，给出买卖建议
# 用法: python screener/verify.py <股票代码>
import sys, os
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_fetcher import DataFetcher
from utils.fundamental import FundamentalAnalyzer, print_report
from strategy.regression_channel.channel_strategy import RegressionChannelWithVolumeStrategy
from backtest.engine import BacktestEngine
from config import CHANNEL_WINDOW, VOL_WINDOW, VOL_MULTIPLIER, TODAY as END_DATE, DATE_BT_START as START_DATE, FUNDAMENTAL_MIN_SCORE


def get_stock_name(code):
    """从 SCREENER_WATCHLIST 中查找股票名称"""
    try:
        from stocks import SCREENER_WATCHLIST
        for c, n in SCREENER_WATCHLIST:
            if c == code:
                return n
    except:
        pass
    return ''


def verify_stock(code, name=''):
    """对单只股票运行回测，输出指标和建议"""
    if not name:
        name = get_stock_name(code)

    print('=' * 60)
    print('回测验证: %s (%s)' % (name, code))
    print('=' * 60)

    # 获取数据
    df = DataFetcher.get_stock_hist(code, START_DATE, END_DATE)
    if df.empty or len(df) < 100:
        print('数据不足 (需要至少100个交易日)')
        return None

    print('数据: %d 天 (%s ~ %s)' % (len(df), df.index[0].date(), df.index[-1].date()))
    print('价格区间: %.2f ~ %.2f' % (df['low'].min(), df['high'].max()))
    print()

    # 运行回测
    strategy = RegressionChannelWithVolumeStrategy(
        window=CHANNEL_WINDOW, vol_window=VOL_WINDOW, vol_multiplier=VOL_MULTIPLIER
    )
    engine = BacktestEngine(initial_capital=100000)
    result = engine.run(df, strategy)

    total_ret = result.total_return * 100
    annual_ret = result.annual_return * 100
    mdd = result.max_drawdown * 100
    sharpe = result.sharpe_ratio
    win_rate = result.win_rate * 100
    trades = result.total_trades

    # 输出
    print('  %-16s %10s' % ('初始资金', format(100000, ',.0f')))
    print('  %-16s %10s' % ('最终资金', format(result.final_value, ',.0f')))
    print('  %-16s %+9.2f%%' % ('总收益率', total_ret))
    print('  %-16s %+9.2f%%' % ('年化收益率', annual_ret))
    print('  %-16s %9.2f%%' % ('最大回撤', mdd))
    print('  %-16s %10.2f' % ('夏普比率', sharpe))
    print('  %-16s %9.1f%%' % ('胜率', win_rate))
    print('  %-16s %10d' % ('交易次数', trades))
    print()

    # 基本面分析
    print('【基本面分析】')
    analyzer = FundamentalAnalyzer()
    fund_data = analyzer.fetch_fundamentals(code, name, price=df['close'].iloc[0])
    fund_passed = False
    if fund_data:
        score = analyzer.compute_score(fund_data)
        print_report(fund_data, score)
        fund_passed = score.total_score >= FUNDAMENTAL_MIN_SCORE
    else:
        print('  基本面数据获取失败')
        print()
    print()

    # 建议
    print('>> 评估建议 <<')
    checks = []
    if annual_ret >= 15:
        checks.append('[OK] 年化>=15%% (%+.1f%%)' % annual_ret)
    elif annual_ret >= 10:
        checks.append('[..] 年化>=10%% (%+.1f%%)' % annual_ret)
    else:
        checks.append('[NO] 年化<10%% (%+.1f%%)' % annual_ret)

    if win_rate >= 45:
        checks.append('[OK] 胜率>=45%% (%.1f%%)' % win_rate)
    elif win_rate >= 35:
        checks.append('[..] 胜率>=35%% (%.1f%%)' % win_rate)
    else:
        checks.append('[NO] 胜率<35%% (%.1f%%)' % win_rate)

    if sharpe >= 0.5:
        checks.append('[OK] 夏普>=0.5 (%.2f)' % sharpe)
    elif sharpe >= 0:
        checks.append('[..] 夏普>0 (%.2f)' % sharpe)
    else:
        checks.append('[NO] 夏普<0 (%.2f)' % sharpe)

    if mdd <= -25:
        checks.append('[NO] 回撤过大 (%.1f%%)' % mdd)
    else:
        checks.append('[OK] 回撤可控 (%.1f%%)' % mdd)

    for c in checks:
        print('  %s' % c)

    print()
    score = 0
    if annual_ret >= 15: score += 2
    elif annual_ret >= 10: score += 1
    if win_rate >= 45: score += 2
    elif win_rate >= 35: score += 1
    if sharpe >= 0.5: score += 2
    elif sharpe >= 0: score += 1
    if mdd > -25: score += 1
    if fund_passed: score += 1

    if score >= 6:
        print('>> 结论: 推荐买入 (得分 %d/8)' % score)
    elif score >= 4:
        print('>> 结论: 谨慎参与 (得分 %d/8)' % score)
    else:
        print('>> 结论: 不推荐 (得分 %d/8)' % score)

    # 最近几笔交易
    if result.trades:
        print()
        print('  最近5笔交易:')
        for t in result.trades[-5:]:
            print('    %s  %s  价格: %.2f' % (t['date'], t['type'], t['price']))

    return {
        'code': code, 'name': name,
        'total_return': total_ret, 'annual_return': annual_ret,
        'max_drawdown': mdd, 'sharpe': sharpe,
        'win_rate': win_rate, 'trades': trades, 'score': score
    }


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('用法: python screener/verify.py <股票代码>')
        print('示例: python screener/verify.py 002050')
        sys.exit(1)

    code = sys.argv[1]
    name = sys.argv[2] if len(sys.argv) > 2 else ''
    verify_stock(code, name)
