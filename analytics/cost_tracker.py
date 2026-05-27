"""
成本追踪 + 决策日志
追踪每笔决策的 token 成本、服务器成本、电力成本，
与市场 alpha 对比计算净收益。
"""
import os, json, time
from datetime import datetime, date
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional


LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'results', 'analytics')
DAILY_LOG_FILE = os.path.join(LOG_DIR, 'daily_logs.jsonl')
COST_SUMMARY_FILE = os.path.join(LOG_DIR, 'cost_summary.json')


# ===== 模型成本参考（元/1M tokens） =====
MODEL_COSTS = {
    # 输入价格 / 输出价格（元/百万token）
    'qwen-plus':       {'input': 0.8,  'output': 2.0},     # 免费额度大
    'qwen-turbo':      {'input': 0.3,  'output': 0.6},     # 更便宜
    'qwen-max':        {'input': 2.0,  'output': 6.0},
    'deepseek-v3':     {'input': 1.0,  'output': 4.0},
    'deepseek-r1':     {'input': 4.0,  'output': 16.0},
    'gpt-4o':          {'input': 17.5, 'output': 70.0},
    'gpt-4o-mini':     {'input': 1.05, 'output': 4.2},
    'claude-sonnet-4': {'input': 21.0, 'output': 105.0},
    'gemini-2.5-flash': {'input': 1.05, 'output': 4.2},
    'gemini-2.5-pro':  {'input': 8.75, 'output': 43.75},
}


@dataclass
class DecisionLog:
    """单条决策日志"""
    id: str = ""                        # 决策 ID
    timestamp: str = ""                 # 时间戳
    decision_type: str = ""             # BUY / SELL / HOLD / ANALYZE / REFLECT
    ticker: str = ""                    # 标的代码
    ticker_name: str = ""

    # LLM 信息
    llm_model: str = ""                 # 使用的模型
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    llm_cost: float = 0.0              # 该决策的 token 费用

    # 决策内容
    decision_summary: str = ""          # 决策摘要
    confidence: float = 0.0             # 置信度 (0-1)
    reasoning: str = ""                 # 推理过程摘要

    # 结果
    pnl_outcome: Optional[float] = None # 该决策最终盈亏（事后填写）
    pnl_date: str = ""                  # 盈亏确认日期

    # 系统成本
    server_cost_per_hour: float = 0.042 # 服务器时租约 ¥1/天 = ¥0.042/h
    runtime_seconds: float = 0.0        # 该决策耗时
    electricity_cost: float = 0.0       # 电费分摊


@dataclass
class DailyReport:
    """每日成本与收益报告"""
    date: str = ""
    decisions: List[DecisionLog] = field(default_factory=list)

    # 收益
    pnl_realized: float = 0.0           # 已实现盈亏
    pnl_unrealized: float = 0.0         # 持仓浮盈
    benchmark_return: float = 0.0       # 基准收益（上证）
    alpha: float = 0.0                  # 超额收益

    # 成本
    total_token_cost: float = 0.0
    total_server_cost: float = 0.0
    total_electricity_cost: float = 0.0
    total_operational_cost: float = 0.0

    # 净收益
    net_alpha: float = 0.0              # alpha - 运营成本

    def summary(self) -> str:
        lines = []
        lines.append(f"【{self.date} 成本收益日报】")
        lines.append(f"  已实现盈亏:    {self.pnl_realized:>+10.2f} 元")
        lines.append(f"  持仓浮盈:      {self.pnl_unrealized:>+10.2f} 元")
        lines.append(f"  超额收益(α):   {self.alpha:>+10.2f} 元")
        lines.append(f"  " + "-" * 30)
        lines.append(f"  Token 费用:    {self.total_token_cost:>10.4f} 元")
        lines.append(f"  服务器费用:    {self.total_server_cost:>10.4f} 元")
        lines.append(f"  电费分摊:      {self.total_electricity_cost:>10.4f} 元")
        lines.append(f"  运营总成本:    {self.total_operational_cost:>10.4f} 元")
        lines.append(f"  " + "-" * 30)
        lines.append(f"  净 Alpha:      {self.net_alpha:>+10.2f} 元")
        lines.append(f"  决策次数:      {len(self.decisions)}")
        if self.total_operational_cost > 0:
            roi = self.net_alpha / self.total_operational_cost
            lines.append(f"  成本收益率(ROI): {roi:+.1f}x")
        return "\n".join(lines)


class CostTracker:
    """成本追踪器"""

    def __init__(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        self._decisions_today: List[DecisionLog] = []
        self._decision_counter = 0
        self._server_start_time = time.time()
        # 默认成本假设
        self.server_cost_per_hour = 1.0 / 24  # 约 ¥1/天
        self.electricity_cost_per_hour = 1.0 / 24  # 约 ¥1/天

    def estimate_token_cost(self, model: str, input_tokens: int,
                            output_tokens: int) -> float:
        """估算 token 费用"""
        costs = MODEL_COSTS.get(model)
        if not costs:
            return 0.0
        input_cost = (input_tokens / 1_000_000) * costs['input']
        output_cost = (output_tokens / 1_000_000) * costs['output']
        return input_cost + output_cost

    def estimate_tokens(self, text: str) -> int:
        """粗略估算 token 数（中文约 1.5 字/token，英文约 4 字/token）"""
        # 粗略估计：混合中英文约 2-3 字符/token
        return max(1, len(text) // 2)

    def log_decision(self,
                     decision_type: str,
                     ticker: str = "",
                     ticker_name: str = "",
                     llm_model: str = "",
                     input_text: str = "",
                     output_text: str = "",
                     decision_summary: str = "",
                     confidence: float = 0.0,
                     reasoning: str = "",
                     runtime_seconds: float = 0.0) -> DecisionLog:
        """记录一条决策"""
        self._decision_counter += 1

        input_tokens = self.estimate_tokens(input_text)
        output_tokens = self.estimate_tokens(output_text)
        token_cost = self.estimate_token_cost(llm_model, input_tokens, output_tokens)

        runtime = runtime_seconds if runtime_seconds > 0 else 1.0
        server_cost = (runtime / 3600) * self.server_cost_per_hour
        electricity_cost = (runtime / 3600) * self.electricity_cost_per_hour

        log = DecisionLog(
            id=f"{datetime.now().strftime('%Y%m%d')}-{self._decision_counter:04d}",
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            decision_type=decision_type,
            ticker=ticker,
            ticker_name=ticker_name,
            llm_model=llm_model,
            llm_input_tokens=input_tokens,
            llm_output_tokens=output_tokens,
            llm_cost=round(token_cost, 6),
            decision_summary=decision_summary,
            confidence=confidence,
            reasoning=reasoning,
            server_cost_per_hour=self.server_cost_per_hour,
            runtime_seconds=runtime,
            electricity_cost=round(electricity_cost, 6),
        )
        self._decisions_today.append(log)
        return log

    def update_pnl(self, decision_id: str, pnl: float):
        """事后更新决策的盈亏结果"""
        for d in self._decisions_today:
            if d.id == decision_id:
                d.pnl_outcome = pnl
                d.pnl_date = datetime.now().strftime('%Y-%m-%d')
                break

    def get_today_stats(self) -> dict:
        """获取今日统计"""
        total_token = sum(d.llm_cost for d in self._decisions_today)
        total_server = sum(d.server_cost_per_hour * d.runtime_seconds / 3600
                          for d in self._decisions_today)
        total_electricity = sum(d.electricity_cost for d in self._decisions_today)

        uptime_hours = (time.time() - self._server_start_time) / 3600
        ongoing_server = uptime_hours * self.server_cost_per_hour
        ongoing_electricity = uptime_hours * self.electricity_cost_per_hour

        return {
            'decisions_count': len(self._decisions_today),
            'total_token_cost': total_token,
            'estimated_server_cost': max(total_server, ongoing_server),
            'estimated_electricity_cost': max(total_electricity, ongoing_electricity),
            'uptime_hours': uptime_hours,
        }

    def generate_daily_report(self,
                               pnl_realized: float = 0.0,
                               pnl_unrealized: float = 0.0,
                               benchmark_return: float = 0.0) -> str:
        """生成每日成本收益报告"""
        stats = self.get_today_stats()
        report = DailyReport(
            date=datetime.now().strftime('%Y-%m-%d'),
            decisions=self._decisions_today.copy(),
            pnl_realized=pnl_realized,
            pnl_unrealized=pnl_unrealized,
            benchmark_return=benchmark_return,
            alpha=pnl_realized - benchmark_return,
            total_token_cost=stats['total_token_cost'],
            total_server_cost=stats['estimated_server_cost'],
            total_electricity_cost=stats['estimated_electricity_cost'],
        )
        report.total_operational_cost = (
            report.total_token_cost +
            report.total_server_cost +
            report.total_electricity_cost
        )
        report.net_alpha = report.alpha - report.total_operational_cost

        # 持久化
        self._save_daily_log(report)

        return report.summary()

    def _save_daily_log(self, report: DailyReport):
        """保存日报到 JSONL 文件"""
        record = {
            'date': report.date,
            'pnl_realized': report.pnl_realized,
            'alpha': report.alpha,
            'token_cost': report.total_token_cost,
            'server_cost': report.total_server_cost,
            'electricity_cost': report.total_electricity_cost,
            'total_cost': report.total_operational_cost,
            'net_alpha': report.net_alpha,
            'decision_count': len(report.decisions),
            'decisions': [asdict(d) for d in report.decisions],
        }
        with open(DAILY_LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

        # 更新汇总
        self._update_summary()

    def _update_summary(self):
        """更新成本汇总文件"""
        if not os.path.exists(DAILY_LOG_FILE):
            return

        records = []
        with open(DAILY_LOG_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

        if not records:
            return

        total_alpha = sum(r.get('alpha', 0) for r in records)
        total_cost = sum(r.get('total_cost', 0) for r in records)
        total_net = sum(r.get('net_alpha', 0) for r in records)
        total_decisions = sum(r.get('decision_count', 0) for r in records)

        # 按模型统计
        model_stats = {}
        for r in records:
            for d in r.get('decisions', []):
                model = d.get('llm_model', 'unknown')
                if model not in model_stats:
                    model_stats[model] = {'count': 0, 'cost': 0.0}
                model_stats[model]['count'] += 1
                model_stats[model]['cost'] += d.get('llm_cost', 0)

        summary = {
            'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'total_days': len(records),
            'cumulative': {
                'alpha': round(total_alpha, 2),
                'total_cost': round(total_cost, 4),
                'net_alpha': round(total_net, 2),
                'total_decisions': total_decisions,
                'roi': round(total_net / total_cost, 1) if total_cost > 0 else 0,
            },
            'model_usage': model_stats,
            'avg_daily': {
                'alpha': round(total_alpha / len(records), 2),
                'cost': round(total_cost / len(records), 4),
                'net': round(total_net / len(records), 2),
                'decisions': round(total_decisions / len(records), 1),
            },
        }
        with open(COST_SUMMARY_FILE, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    def get_cumulative_stats(self) -> Optional[dict]:
        """获取累计统计"""
        if not os.path.exists(COST_SUMMARY_FILE):
            return None
        with open(COST_SUMMARY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)

    def print_cumulative_stats(self) -> str:
        """打印累计统计"""
        stats = self.get_cumulative_stats()
        if not stats:
            return "暂无累计数据"

        c = stats['cumulative']
        a = stats['avg_daily']
        lines = []
        lines.append("=" * 50)
        lines.append("【累计成本收益统计】")
        lines.append(f"统计天数: {stats['total_days']} 天")
        lines.append(f"累计 Alpha:    {c['alpha']:>+10.2f} 元")
        lines.append(f"累计成本:      {c['total_cost']:>10.4f} 元")
        lines.append(f"累计净收益:    {c['net_alpha']:>+10.2f} 元")
        lines.append(f"累计决策:      {c['total_decisions']} 次")
        lines.append(f"成本 ROI:      {c['roi']:+.1f}x")
        lines.append("")
        lines.append("日均:")
        lines.append(f"  日均 Alpha:   {a['alpha']:>+10.2f} 元")
        lines.append(f"  日均成本:     {a['cost']:>10.4f} 元")
        lines.append(f"  日均净收益:   {a['net']:>+10.2f} 元")
        lines.append(f"  日均决策:     {a['decisions']:.0f} 次")
        lines.append("")
        lines.append("模型使用分布:")
        for model, info in stats.get('model_usage', {}).items():
            lines.append(f"  {model}: {info['count']}次, 费用 ¥{info['cost']:.4f}")
        return "\n".join(lines)
