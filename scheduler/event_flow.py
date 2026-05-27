"""
事件流调度器
将散落的脚本串联成自动化 pipeline，按交易时段触发。
支持一次性运行（--once）和定时循环（--daemon）。

用法:
    python scheduler/event_flow.py --once          # 运行一次完整流程
    python scheduler/event_flow.py --stage pre     # 只跑盘前
    python scheduler/event_flow.py --stage close   # 只跑收盘后
"""
import sys, os, time
from datetime import datetime, date
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from enum import Enum

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TODAY, DATE_RECENT_START, FUNDAMENTAL_MIN_SCORE, CHANNEL_WINDOW, VOL_WINDOW, VOL_MULTIPLIER, MARKET_INDEX
from stocks import WATCH_LIST, SCREENER_WATCHLIST, SECTOR_MAP
from utils.data_fetcher import DataFetcher
from utils.fundamental import FundamentalAnalyzer
from strategy.regression_channel.channel_strategy import RegressionChannelWithVolumeStrategy
from nl_convert.signal_to_nl import SignalToNL
from memory.trade_memory import TradeMemory, TradeRecord
from analytics.cost_tracker import CostTracker, DecisionLog
from state_machine.trading_fsm import TradingStateMachine, TradingState


class PipelineStage(Enum):
    PRE_MARKET = "pre"          # 盘前检查 (9:00)
    MARKET_OPEN = "open"        # 开盘扫描 (9:30)
    MID_DAY = "mid"             # 盘中监控 (11:00)
    AFTERNOON = "afternoon"     # 尾盘检查 (14:30)
    POST_MARKET = "close"       # 收盘反思 (15:30)
    FULL = "full"               # 完整流程


@dataclass
class PipelineResult:
    stage: PipelineStage
    timestamp: str
    context_text: str            # LLM 可读的完整上下文
    signals: List[dict]          # 信号列表
    decisions: List[dict]        # 待决策事项
    errors: List[str] = field(default_factory=list)


class EventFlow:
    """事件流调度器 — 每个交易时段触发对应的 pipeline stage"""

    def __init__(self):
        self.signal_nl = SignalToNL()
        self.memory = TradeMemory()
        self.cost_tracker = CostTracker()
        self.state_machine = TradingStateMachine()
        self.strategy = RegressionChannelWithVolumeStrategy(
            window=CHANNEL_WINDOW, vol_window=VOL_WINDOW,
            vol_multiplier=VOL_MULTIPLIER,
        )
        self.fundamental = FundamentalAnalyzer(min_score=FUNDAMENTAL_MIN_SCORE)

    # ===== Pipeline Stages =====

    def run_pre_market(self) -> PipelineResult:
        """
        盘前阶段 (9:00)
        - 检查持仓状态
        - 更新基本面数据
        - 回顾昨日候选
        - 检查大盘环境
        """
        result = PipelineResult(
            stage=PipelineStage.PRE_MARKET,
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M'),
            context_text="",
            signals=[],
            decisions=[]
        )
        self.state_machine.transition(TradingState.PRE_MARKET_CHECK)

        lines = []
        lines.append(f"=== 盘前检查 {datetime.now().strftime('%Y-%m-%d')} ===\n")

        # 大盘环境
        market_ctx = self.signal_nl.get_market_context()
        lines.append("## 大盘环境")
        lines.append(market_ctx.description)
        lines.append("")

        # 持仓检查
        holdings = {code: info for code, info in WATCH_LIST.items() if info.get('shares', 0) > 0}
        if holdings:
            lines.append("## 持仓状态")
            for code, info in holdings.items():
                name = self._find_name(code)
                desc = self.signal_nl.describe_sell_signal(code, name, info)
                lines.append(desc)
                lines.append("")

                # 状态机：检测是否需要卖出
                t = self.signal_nl.describe_ticker(code, name, holding_info=info)
                if t.signal_type in ("卖出", "卖出半仓"):
                    result.decisions.append({
                        'type': 'SELL_CHECK',
                        'code': code,
                        'name': name,
                        'reason': t.signal_type,
                        'pnl_pct': t.holding_pnl_pct,
                    })
        else:
            lines.append("## 持仓状态: 空仓\n")
            self.state_machine.transition(TradingState.IDLE)

        # 候选回顾
        lines.append("## 昨日候选股回顾")
        cache_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  'screener', 'results', 'latest_screen.csv')
        if os.path.exists(cache_file):
            import pandas as pd
            cache = pd.read_csv(cache_file, encoding='utf-8-sig', dtype={'code': str})
            buys = cache[cache['signal'] == '★买入'] if 'signal' in cache.columns else pd.DataFrame()
            if not buys.empty:
                for _, row in buys.iterrows():
                    code = str(row['code'])
                    name = str(row.get('name', ''))
                    t = self.signal_nl.describe_ticker(code, name)
                    still_valid = "有效" if t.signal_type == "★买入" else "已回落"
                    lines.append(f"- {code} {name}: 昨日买入信号 → 今日{'仍' if still_valid == '有效' else '已'}有效 ({t.close:.2f}, 上轨{t.upper:.2f})")
            else:
                lines.append("昨日无买入信号候选")
        else:
            lines.append("无缓存数据")
        lines.append("")

        result.context_text = "\n".join(lines)
        self.state_machine.transition(TradingState.IDLE)
        return result

    def run_market_open(self) -> PipelineResult:
        """
        开盘扫描阶段 (9:30)
        - 扫描全量关注列表
        - 生成买入信号 + NL 描述
        - 记录信号到记忆
        """
        result = PipelineResult(
            stage=PipelineStage.MARKET_OPEN,
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M'),
            context_text="",
            signals=[],
            decisions=[]
        )
        self.state_machine.transition(TradingState.SCREENING)

        # 扫描
        holdings = {code: info for code, info in WATCH_LIST.items() if info.get('shares', 0) > 0}
        sector_vals = self._compute_sector_values(holdings)

        for code, name in SCREENER_WATCHLIST:
            info = holdings.get(code, {})
            t = self.signal_nl.describe_ticker(
                code, name,
                holding_info=info if info else None,
                sector_positions=sector_vals
            )
            if not t.close:
                continue

            sig_record = {
                'code': code, 'name': name,
                'date': t.date, 'signal': t.signal_type,
                'close': t.close, 'upper': t.upper,
                'above_pct': t.above_pct, 'vol_ratio': t.vol_ratio,
                'fund_score': t.fund_score, 'fund_grade': t.fund_grade,
                'slope': t.slope, 'slope_status': t.slope_status,
                'sector': t.sector,
            }
            result.signals.append(sig_record)

            if t.signal_type == "★买入":
                result.decisions.append({
                    'type': 'BUY_CANDIDATE',
                    'code': code, 'name': name,
                    'close': t.close, 'upper': t.upper,
                    'above_pct': t.above_pct,
                    'vol_ratio': t.vol_ratio,
                    'fund_score': t.fund_score,
                    'fund_grade': t.fund_grade,
                    'sector': t.sector,
                })

        # 批量记录信号到记忆
        self.memory.batch_record_signals(result.signals)

        # 生成完整上下文
        context_lines = []
        context_lines.append(f"=== 开盘扫描 {datetime.now().strftime('%Y-%m-%d')} ===\n")

        market_ctx = self.signal_nl.get_market_context()
        context_lines.append(f"## 大盘: {market_ctx.description}\n")

        # 检索相关历史记忆
        buy_codes = [d['code'] for d in result.decisions if d['type'] == 'BUY_CANDIDATE']
        if buy_codes:
            memory_ctx = self.memory.retrieve_for_decision(buy_codes)
            if memory_ctx:
                context_lines.append(memory_ctx)
                context_lines.append("")

        if result.decisions:
            context_lines.append(f"## 买入候选 ({len([d for d in result.decisions if d['type'] == 'BUY_CANDIDATE'])}只)")
            for i, d in enumerate([d for d in result.decisions if d['type'] == 'BUY_CANDIDATE']):
                context_lines.append(f"\n#{i+1} {d['code']} {d['name']} ({d['sector']})")
                context_lines.append(f"  收盘 {d['close']:.2f} | 上轨 {d['upper']:.2f} | 偏离 +{d['above_pct']:.1f}%")
                context_lines.append(f"  量比 {d['vol_ratio']:.2f}x | 基本面 {d['fund_score']}分/{d['fund_grade']}")
        else:
            context_lines.append("## 今日无买入信号")

        result.context_text = "\n".join(context_lines)
        self.state_machine.transition(TradingState.IDLE)
        return result

    def run_post_market(self) -> PipelineResult:
        """
        收盘反思阶段 (15:30)
        - 更新持仓记忆
        - 对有交易的标的生成反思
        - 记录当日决策和成本
        - 生成次日关注清单
        """
        result = PipelineResult(
            stage=PipelineStage.POST_MARKET,
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M'),
            context_text="",
            signals=[],
            decisions=[]
        )
        self.state_machine.transition(TradingState.REFLECTING)

        lines = []
        lines.append(f"=== 收盘反思 {datetime.now().strftime('%Y-%m-%d')} ===\n")

        # 更新持仓股信号到记忆 + 生成反思提示
        holdings = {code: info for code, info in WATCH_LIST.items() if info.get('shares', 0) > 0}
        for code, info in holdings.items():
            name = self._find_name(code)
            t = self.signal_nl.describe_ticker(code, name, holding_info=info)

            # 记录当日信号
            self.memory.record_signal(code, {
                'date': t.date, 'signal': t.signal_type,
                'close': t.close, 'upper': t.upper,
                'above_pct': t.above_pct, 'vol_ratio': t.vol_ratio,
            }, name=name)

            # 生成反思 prompt（可喂给 LLM）
            reflection_prompt = self.memory.generate_reflection_prompt(code, name)
            result.decisions.append({
                'type': 'REFLECTION_NEEDED',
                'code': code,
                'name': name,
                'prompt': reflection_prompt,
            })

            lines.append(f"## {code} {name}")
            lines.append(f"持仓 {info['shares']}股 | 成本 {info['cost']:.2f} | 现价 {t.close:.2f}")
            lines.append(f"浮动盈亏: {t.holding_pnl_pct:+.1f}%")
            lines.append(f"信号: {t.signal_type} | 斜率: {t.slope_status}\n")

        # 成本日报
        daily_report = self.cost_tracker.generate_daily_report()
        lines.append(daily_report)

        # 重置状态
        self.state_machine.transition(TradingState.IDLE)

        # 更新记忆索引
        self.memory._update_index("", "")

        result.context_text = "\n".join(lines)
        return result

    def run_full_pipeline(self) -> Dict[str, PipelineResult]:
        """运行完整 pipeline（一次性）"""
        results = {}

        print(">>> 阶段1: 盘前检查")
        results['pre'] = self.run_pre_market()
        print(f"  持仓 {len([d for d in results['pre'].decisions if d['type'] == 'SELL_CHECK'])} 只需要关注")

        print(">>> 阶段2: 开盘扫描")
        results['open'] = self.run_market_open()
        buy_count = len([d for d in results['open'].decisions if d['type'] == 'BUY_CANDIDATE'])
        print(f"  扫描 {len(results['open'].signals)} 只, 发现 {buy_count} 个买入候选")

        print(">>> 阶段3: 收盘反思")
        results['close'] = self.run_post_market()
        print(f"  待反思标的: {len([d for d in results['close'].decisions if d['type'] == 'REFLECTION_NEEDED'])} 只")

        return results

    # ===== 工具方法 =====

    def _find_name(self, code: str) -> str:
        for c, n in SCREENER_WATCHLIST:
            if c == code:
                return n
        return ""

    def _compute_sector_values(self, holdings: dict) -> Dict[str, float]:
        sector_vals = {}
        for code, info in holdings.items():
            if info.get('shares', 0) <= 0:
                continue
            for sec, syms in SECTOR_MAP.items():
                if code in syms:
                    df = DataFetcher.get_stock_hist(code, DATE_RECENT_START, TODAY)
                    price = float(df['close'].iloc[-1]) if not df.empty else info.get('cost', 0)
                    val = info['shares'] * price
                    sector_vals[sec] = sector_vals.get(sec, 0) + val
                    break
        return sector_vals

    def export_llm_context(self, results: Dict[str, PipelineResult]) -> str:
        """
        将所有 pipeline 结果合并为一份完整的 LLM 上下文。
        直接把返回值塞给 LLM 做决策。
        """
        parts = []
        parts.append(f"# 交易系统日度报告 — {datetime.now().strftime('%Y-%m-%d')}")
        parts.append("")

        # 盘前
        if 'pre' in results:
            parts.append(results['pre'].context_text)
            parts.append("")

        # 开盘扫描
        if 'open' in results:
            parts.append(results['open'].context_text)
            parts.append("")

        # 收盘
        if 'close' in results:
            parts.append(results['close'].context_text)
            parts.append("")

        parts.append("---")
        parts.append("请基于以上信息，给出今日的交易决策总结和明日关注要点。")
        return "\n".join(parts)


# ===== CLI 入口 =====

def main():
    import argparse
    parser = argparse.ArgumentParser(description='事件流调度器')
    parser.add_argument('--once', action='store_true', help='运行一次完整流程')
    parser.add_argument('--stage', choices=['pre', 'open', 'close', 'full'],
                       default='full', help='运行指定阶段')
    parser.add_argument('--output', type=str, default='', help='输出 LLM 上下文到文件')
    args = parser.parse_args()

    flow = EventFlow()

    if args.stage == 'pre':
        result = flow.run_pre_market()
        print(result.context_text)
    elif args.stage == 'open':
        result = flow.run_market_open()
        print(result.context_text)
    elif args.stage == 'close':
        result = flow.run_post_market()
        print(result.context_text)
    else:
        results = flow.run_full_pipeline()
        llm_ctx = flow.export_llm_context(results)
        print(llm_ctx)

        if args.output:
            output_path = args.output
            if not os.path.isabs(output_path):
                output_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'results', args.output
                )
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(llm_ctx)
            print(f"\nLLM 上下文已导出到: {output_path}")


if __name__ == '__main__':
    main()
