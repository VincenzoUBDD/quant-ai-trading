"""
RAG 交易记忆系统
每个标的维护独立记忆文件，记录历史决策、PnL、反思。
LLM 下次遇到同一标的时可检索历史记忆来做"实时强化学习"。
"""
import os, json
from datetime import datetime, date
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
import pandas as pd


MEMORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ticker_memories')
INDEX_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MEMORY_INDEX.md')


@dataclass
class TradeRecord:
    """单笔交易记录"""
    date: str                      # 交易日期 YYYY-MM-DD
    direction: str                 # BUY / SELL / SELL_HALF
    price: float
    shares: int
    value: float                   # 成交金额
    reason: str = ""               # 决策依据
    signal_desc: str = ""          # 触发信号描述
    pnl: Optional[float] = None    # 平仓时的盈亏（仅SELL时有值）
    pnl_pct: Optional[float] = None
    llm_model: str = ""            # 做决策的模型
    llm_cost: float = 0.0          # 该决策消耗的token费用


@dataclass
class TickerMemory:
    """单只标的的完整记忆"""
    symbol: str
    name: str = ""
    sector: str = ""
    trades: List[TradeRecord] = field(default_factory=list)
    reflections: List[dict] = field(default_factory=list)
    signal_history: List[dict] = field(default_factory=list)
    summary_stats: dict = field(default_factory=dict)
    updated_at: str = ""

    def to_markdown(self) -> str:
        """导出为 LLM 可读的 Markdown 格式"""
        lines = []
        lines.append(f"---")
        lines.append(f"symbol: {self.symbol}")
        lines.append(f"name: {self.name}")
        lines.append(f"sector: {self.sector}")
        lines.append(f"updated_at: {self.updated_at}")
        lines.append(f"---")
        lines.append("")

        # 统计摘要
        stats = self._compute_stats()
        lines.append("## 统计摘要")
        lines.append(f"- 总交易次数: {stats['total_trades']}")
        lines.append(f"- 胜率: {stats['win_rate']:.1%}" if stats['total_trades'] > 0 else "- 胜率: N/A")
        lines.append(f"- 累计已实现盈亏: {stats['total_pnl']:+.2f}")
        lines.append(f"- 平均持仓天数: {stats['avg_hold_days']:.1f}" if stats['avg_hold_days'] > 0 else "- 平均持仓天数: N/A")
        lines.append(f"- 最大单笔盈利: {stats['max_win']:+.2f}")
        lines.append(f"- 最大单笔亏损: {stats['max_loss']:+.2f}")
        lines.append("")

        # 交易记录
        if self.trades:
            lines.append("## 交易记录")
            lines.append("| 日期 | 方向 | 价格 | 股数 | 金额 | 盈亏 | 盈亏% | 依据 |")
            lines.append("|------|------|------|------|------|------|-------|------|")
            for t in self.trades:
                pnl_str = f"{t.pnl:+.0f}" if t.pnl is not None else "-"
                pnl_pct_str = f"{t.pnl_pct:+.1f}%" if t.pnl_pct is not None else "-"
                reason_short = t.reason[:30] if t.reason else "-"
                lines.append(f"| {t.date} | {t.direction} | {t.price:.2f} | {t.shares} | {t.value:.0f} | {pnl_str} | {pnl_pct_str} | {reason_short} |")
            lines.append("")

        # 反思记录
        if self.reflections:
            lines.append("## 反思记录")
            for r in self.reflections[-5:]:  # 最近5条
                lines.append(f"### {r.get('date', '')}")
                lines.append(f"**触发原因**: {r.get('trigger', '')}")
                lines.append(f"**反思内容**: {r.get('content', '')}")
                if r.get('lessons'):
                    lines.append(f"**经验教训**: {r['lessons']}")
                lines.append("")

        # 信号历史摘要
        if self.signal_history:
            lines.append("## 近期信号历史")
            lines.append("| 日期 | 信号 | 收盘 | 上轨 | 偏离% | 量比 |")
            lines.append("|------|------|------|------|-------|------|")
            for s in self.signal_history[-10:]:
                lines.append(f"| {s.get('date', '')} | {s.get('signal', '')} | {s.get('close', 0):.2f} | {s.get('upper', 0):.2f} | {s.get('above_pct', 0):+.1f} | {s.get('vol_ratio', 0):.2f}x |")
            lines.append("")

        return "\n".join(lines)

    def _compute_stats(self) -> dict:
        sells = [t for t in self.trades if t.direction in ('SELL', 'SELL_HALF') and t.pnl is not None]
        buys = [t for t in self.trades if t.direction == 'BUY']
        wins = [t for t in sells if (t.pnl or 0) > 0]

        # 平均持仓天数
        avg_hold_days = 0.0
        if buys and sells:
            hold_days = []
            for i, buy in enumerate(buys):
                sell_dates = [s.date for s in sells if s.date >= buy.date]
                if i + 1 < len(buys):
                    sell_dates = [d for d in sell_dates if d < buys[i + 1].date]
                if sell_dates:
                    bd = datetime.strptime(buy.date, '%Y-%m-%d')
                    sd = datetime.strptime(sell_dates[0], '%Y-%m-%d')
                    hold_days.append((sd - bd).days)
            if hold_days:
                avg_hold_days = sum(hold_days) / len(hold_days)

        return {
            'total_trades': len(sells),
            'win_rate': len(wins) / len(sells) if sells else 0.0,
            'total_pnl': sum(t.pnl or 0 for t in sells),
            'avg_hold_days': avg_hold_days,
            'max_win': max((t.pnl or 0 for t in sells), default=0),
            'max_loss': min((t.pnl or 0 for t in sells), default=0),
        }


class TradeMemory:
    """RAG 记忆管理器"""

    def __init__(self):
        os.makedirs(MEMORY_DIR, exist_ok=True)
        self._index: Dict[str, str] = {}  # {symbol: filepath}

    # ===== 记忆文件路径 =====

    def _memory_path(self, symbol: str) -> str:
        return os.path.join(MEMORY_DIR, f'{symbol}.md')

    # ===== 读取 =====

    def load(self, symbol: str, name: str = "", sector: str = "") -> TickerMemory:
        """加载单只标的的记忆"""
        path = self._memory_path(symbol)
        if not os.path.exists(path):
            return TickerMemory(symbol=symbol, name=name, sector=sector)

        mem = TickerMemory(symbol=symbol, name=name, sector=sector)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            mem = self._parse_markdown(content, symbol, name, sector)
        except Exception:
            pass
        return mem

    def load_all(self) -> Dict[str, TickerMemory]:
        """加载所有标的的记忆"""
        result = {}
        if not os.path.exists(MEMORY_DIR):
            return result
        for fname in os.listdir(MEMORY_DIR):
            if fname.endswith('.md'):
                symbol = fname.replace('.md', '')
                result[symbol] = self.load(symbol)
        return result

    # ===== 写入 =====

    def save(self, mem: TickerMemory):
        """保存单只标的的记忆"""
        mem.updated_at = datetime.now().strftime('%Y-%m-%d %H:%M')
        path = self._memory_path(mem.symbol)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(mem.to_markdown())
        self._update_index(mem.symbol, path)

    # ===== 记录交易 =====

    def record_trade(self, symbol: str, trade: TradeRecord,
                     name: str = "", sector: str = ""):
        """记录一笔交易"""
        mem = self.load(symbol, name, sector)
        mem.name = name or mem.name
        mem.sector = sector or mem.sector
        mem.trades.append(trade)
        self.save(mem)

    def record_signal(self, symbol: str, signal_info: dict,
                      name: str = "", sector: str = ""):
        """记录一条信号历史"""
        mem = self.load(symbol, name, sector)
        mem.name = name or mem.name
        mem.sector = sector or mem.sector
        mem.signal_history.append(signal_info)
        # 只保留最近 30 条
        if len(mem.signal_history) > 30:
            mem.signal_history = mem.signal_history[-30:]
        self.save(mem)

    def record_reflection(self, symbol: str, reflection: dict,
                          name: str = "", sector: str = ""):
        """记录一条反思"""
        mem = self.load(symbol, name, sector)
        mem.name = name or mem.name
        mem.sector = sector or mem.sector
        if 'date' not in reflection:
            reflection['date'] = datetime.now().strftime('%Y-%m-%d')
        mem.reflections.append(reflection)
        # 只保留最近 20 条反思
        if len(mem.reflections) > 20:
            mem.reflections = mem.reflections[-20:]
        self.save(mem)

    # ===== 批量记录 =====

    def batch_record_signals(self, signals: List[dict]):
        """批量记录当日信号扫描结果"""
        for sig in signals:
            code = sig.get('code', '')
            if not code:
                continue
            self.record_signal(code, {
                'date': sig.get('date', datetime.now().strftime('%Y-%m-%d')),
                'signal': sig.get('signal', '-'),
                'close': sig.get('close', 0),
                'upper': sig.get('upper', 0),
                'above_pct': sig.get('above_pct', 0),
                'vol_ratio': sig.get('vol_ratio', 0),
                'fund_score': sig.get('fund_score'),
            }, name=sig.get('name', ''))

    # ===== 检索（RAG 核心） =====

    def retrieve_context(self, symbol: str, name: str = "", sector: str = "") -> str:
        """
        检索某只标的的历史上下文，返回适合塞入 LLM prompt 的文本。
        这是 RAG 的 retrieval 步骤。
        """
        mem = self.load(symbol, name, sector)
        if not mem.trades and not mem.reflections:
            return f"({symbol} {name}: 无历史交易记录)"

        lines = []
        lines.append(f"## {symbol} {mem.name} 历史交易记忆")

        # 统计
        stats = mem._compute_stats()
        if stats['total_trades'] > 0:
            lines.append(f"累计交易{stats['total_trades']}次, "
                        f"胜率{stats['win_rate']:.0%}, "
                        f"累计盈亏{stats['total_pnl']:+.0f}元")

        # 最近交易
        recent_trades = [t for t in mem.trades if t.pnl is not None][-3:]
        for t in recent_trades:
            lines.append(f"- {t.date}: {t.reason} → 盈亏{t.pnl:+.0f}元({t.pnl_pct:+.1f}%)" if t.pnl else "")

        # 最近反思（最重要）
        if mem.reflections:
            latest = mem.reflections[-1]
            lines.append(f"\n最近反思({latest.get('date', '')}):")
            lines.append(f"  {latest.get('content', '')}")
            if latest.get('lessons'):
                lines.append(f"  经验: {latest['lessons']}")

        return "\n".join(lines)

    def retrieve_for_decision(self, symbols: List[str]) -> str:
        """
        批量检索多只标的的历史上下文，用于 LLM 决策。
        自动按标的聚合，标注重要信息。
        """
        if not symbols:
            return ""

        parts = []
        parts.append("## 历史交易记忆（按标的）")
        parts.append("以下是各标的的历史交易记录和反思，请在决策时参考：\n")

        for sym in symbols:
            ctx = self.retrieve_context(sym)
            if ctx:
                parts.append(ctx)
                parts.append("")

        return "\n".join(parts)

    # ===== 收盘后反思模板 =====

    def generate_reflection_prompt(self, symbol: str, name: str = "") -> str:
        """
        生成收盘后反思的 prompt 模板。
        可以把这个 prompt 发给 LLM，LLM 的回复再存入记忆。
        """
        mem = self.load(symbol, name)

        # 找到最近一笔已完成的交易轮回
        lines = []
        lines.append(f"请对 {symbol} {mem.name} 最近的交易进行反思：")

        recent = mem.trades[-10:] if len(mem.trades) > 10 else mem.trades
        for t in recent:
            pnl_info = f" 盈亏{t.pnl:+.0f}" if t.pnl is not None else ""
            lines.append(f"- {t.date} {t.direction} @{t.price:.2f} x{t.shares}股{pnl_info} | {t.reason}")

        lines.append("\n请分析：")
        lines.append("1. 这次交易的入场和出场时机是否合适？")
        lines.append("2. 信号质量如何？是否存在噪音干扰？")
        lines.append("3. 下次该标的再次触发信号时，应该注意什么？")
        lines.append("4. 这次交易有什么可以改进的地方？")
        return "\n".join(lines)

    # ===== 内部方法 =====

    def _parse_markdown(self, content: str, symbol: str, name: str, sector: str) -> TickerMemory:
        """解析记忆 Markdown 文件（简化实现）"""
        mem = TickerMemory(symbol=symbol, name=name, sector=sector)

        # 解析 frontmatter
        if content.startswith('---'):
            end = content.find('---', 3)
            if end > 0:
                fm = content[3:end].strip()
                for line in fm.split('\n'):
                    if ':' in line:
                        k, v = line.split(':', 1)
                        k, v = k.strip(), v.strip()
                        if k == 'name':
                            mem.name = v
                        elif k == 'sector':
                            mem.sector = v
                        elif k == 'updated_at':
                            mem.updated_at = v

        return mem

    def _update_index(self, symbol: str, filepath: str):
        """更新记忆索引文件"""
        self._index[symbol] = filepath
        try:
            lines = []
            lines.append("# 交易记忆索引\n")
            lines.append(f"最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            for sym in sorted(self._index.keys()):
                mem = self.load(sym)
                stats = mem._compute_stats()
                trade_count = stats['total_trades']
                total_pnl = stats['total_pnl']
                lines.append(f"- [{sym}](ticker_memories/{sym}.md) — "
                           f"{mem.name} | {trade_count}笔交易 | "
                           f"累计盈亏{total_pnl:+.0f}元")
            with open(INDEX_FILE, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
        except Exception:
            pass
