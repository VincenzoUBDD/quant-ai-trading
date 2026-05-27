"""
交易状态机
定义交易系统在整个交易日中的合法状态和转换规则。
每个状态有明确的进入/退出条件、允许的动作、和 LLM prompt 提示。
"""
from enum import Enum, auto
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable


class TradingState(Enum):
    """交易系统全局状态"""
    # 系统级状态
    INIT = auto()               # 系统初始化
    IDLE = auto()               # 空闲等待

    # 盘前
    PRE_MARKET_CHECK = auto()   # 盘前检查（基本面/新闻/隔夜消息）

    # 盘中
    SCREENING = auto()          # 扫描选股
    SIGNAL_DETECTED = auto()    # 发现买入/卖出信号
    ANALYZING = auto()          # 深入分析信号（LLM 介入）
    DECISION_PENDING = auto()   # 等待最终决策

    # 交易执行
    BUY_PLANNED = auto()        # 计划买入
    BUY_EXECUTED = auto()       # 已执行买入
    SELL_PLANNED = auto()       # 计划卖出
    SELL_EXECUTED = auto()      # 已执行卖出

    # 持仓监控
    MONITORING = auto()         # 监控持仓
    ALERT_TRIGGERED = auto()    # 触发预警

    # 风控
    RISK_STOP = auto()          # 风控止损/熔断
    MARKET_HALT = auto()        # 市场暂停（熔断恢复中）

    # 收盘
    REFLECTING = auto()         # 收盘反思
    DAILY_SUMMARY = auto()      # 日报生成


@dataclass
class Transition:
    """状态转换记录"""
    from_state: TradingState
    to_state: TradingState
    timestamp: str
    reason: str = ""
    metadata: dict = field(default_factory=dict)


class TradingStateMachine:
    """
    交易状态机

    合法状态转换图:
      INIT → IDLE
      IDLE ↔ PRE_MARKET_CHECK → IDLE
      IDLE ↔ SCREENING → SIGNAL_DETECTED → ANALYZING → DECISION_PENDING
        → BUY_PLANNED | SELL_PLANNED
      BUY_PLANNED → BUY_EXECUTED → MONITORING
      SELL_PLANNED → SELL_EXECUTED → IDLE
      MONITORING → ALERT_TRIGGERED → ANALYZING | SELL_PLANNED
      MONITORING → REFLECTING
      IDLE → REFLECTING → DAILY_SUMMARY → IDLE
      * → RISK_STOP → MARKET_HALT → IDLE (熔断恢复)
    """

    # 合法转换表
    VALID_TRANSITIONS = {
        TradingState.INIT: {TradingState.IDLE},

        TradingState.IDLE: {
            TradingState.PRE_MARKET_CHECK,
            TradingState.SCREENING,
            TradingState.REFLECTING,
            TradingState.RISK_STOP,
        },

        TradingState.PRE_MARKET_CHECK: {
            TradingState.IDLE,
            TradingState.SCREENING,
        },

        TradingState.SCREENING: {
            TradingState.SIGNAL_DETECTED,
            TradingState.IDLE,           # 无信号
            TradingState.ALERT_TRIGGERED,
        },

        TradingState.SIGNAL_DETECTED: {
            TradingState.ANALYZING,
            TradingState.IDLE,           # 放弃信号
        },

        TradingState.ANALYZING: {
            TradingState.DECISION_PENDING,
            TradingState.IDLE,           # 分析后放弃
        },

        TradingState.DECISION_PENDING: {
            TradingState.BUY_PLANNED,
            TradingState.SELL_PLANNED,
            TradingState.IDLE,
        },

        TradingState.BUY_PLANNED: {
            TradingState.BUY_EXECUTED,
            TradingState.IDLE,           # 取消买入
        },

        TradingState.BUY_EXECUTED: {
            TradingState.MONITORING,
            TradingState.IDLE,
        },

        TradingState.SELL_PLANNED: {
            TradingState.SELL_EXECUTED,
            TradingState.IDLE,
        },

        TradingState.SELL_EXECUTED: {
            TradingState.IDLE,
            TradingState.SCREENING,      # 卖出后立即扫描下一机会
        },

        TradingState.MONITORING: {
            TradingState.ALERT_TRIGGERED,
            TradingState.SELL_PLANNED,
            TradingState.REFLECTING,
            TradingState.RISK_STOP,
        },

        TradingState.ALERT_TRIGGERED: {
            TradingState.ANALYZING,
            TradingState.SELL_PLANNED,
            TradingState.MONITORING,     # 解除预警
        },

        TradingState.RISK_STOP: {
            TradingState.MARKET_HALT,
            TradingState.IDLE,
        },

        TradingState.MARKET_HALT: {
            TradingState.IDLE,           # 熔断恢复
        },

        TradingState.REFLECTING: {
            TradingState.DAILY_SUMMARY,
            TradingState.IDLE,
        },

        TradingState.DAILY_SUMMARY: {
            TradingState.IDLE,
        },
    }

    # 每个状态的 LLM prompt 提示
    STATE_PROMPTS = {
        TradingState.IDLE:
            "系统空闲，等待下一交易时段。",
        TradingState.PRE_MARKET_CHECK:
            "盘前检查中。请关注：隔夜消息、外盘走势、持仓股基本面变化。",
        TradingState.SCREENING:
            "正在扫描关注列表，检测突破信号。关注点：放量突破、上轨偏离度、板块轮动。",
        TradingState.SIGNAL_DETECTED:
            "发现交易信号。请分析信号质量：是否新鲜突破、量能配合、基本面支撑。",
        TradingState.ANALYZING:
            "深入分析中。对比历史相似信号的表现，评估胜率和风险收益比。",
        TradingState.DECISION_PENDING:
            "等待最终决策。请综合考虑：市场环境、仓位状态、板块集中度、历史记忆。",
        TradingState.BUY_PLANNED:
            "买入计划已生成。确认：价格、仓位、止损位、止盈目标。",
        TradingState.MONITORING:
            "监控持仓中。关注：上轨斜率变化、缩量预警、回撤是否触发止损。",
        TradingState.ALERT_TRIGGERED:
            "持仓预警触发。检查：是缩量洗盘还是趋势反转？是否需要加仓/减仓？",
        TradingState.REFLECTING:
            "收盘反思。回顾今日决策质量，记录经验教训，更新标的记忆。",
        TradingState.RISK_STOP:
            "风控止损触发。暂停交易，等待市场恢复信号。",
    }

    def __init__(self):
        self.current_state = TradingState.INIT
        self.state_history: List[Transition] = []
        self.state_start_time: str = ""
        self.state_data: dict = {}  # 状态上下文数据

        # 转换到初始空闲状态
        self.transition(TradingState.IDLE, "系统启动")

    def transition(self, to_state: TradingState, reason: str = "", **metadata) -> bool:
        """
        尝试状态转换。返回 True 表示成功，False 表示非法转换。
        """
        # 检查合法性
        valid_targets = self.VALID_TRANSITIONS.get(self.current_state, set())
        if to_state not in valid_targets:
            print(f"⚠ 非法状态转换: {self.current_state.name} → {to_state.name}")
            print(f"  合法目标: {[s.name for s in valid_targets]}")
            return False

        # 执行转换
        transition = Transition(
            from_state=self.current_state,
            to_state=to_state,
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            reason=reason,
            metadata=metadata,
        )
        self.state_history.append(transition)
        self.current_state = to_state
        self.state_start_time = transition.timestamp
        self.state_data = metadata

        return True

    def get_prompt(self) -> str:
        """获取当前状态的 LLM prompt"""
        return self.STATE_PROMPTS.get(self.current_state, "")

    def get_current_context(self) -> dict:
        """获取当前状态的完整上下文"""
        return {
            'state': self.current_state.name,
            'prompt': self.get_prompt(),
            'start_time': self.state_start_time,
            'data': self.state_data,
        }

    def is_trading_allowed(self) -> bool:
        """当前状态是否允许交易"""
        allowed_states = {
            TradingState.DECISION_PENDING,
            TradingState.BUY_PLANNED,
            TradingState.SELL_PLANNED,
            TradingState.BUY_EXECUTED,
            TradingState.SELL_EXECUTED,
            TradingState.MONITORING,
        }
        return self.current_state in allowed_states

    def is_risk_active(self) -> bool:
        """风控是否激活"""
        return self.current_state in {TradingState.RISK_STOP, TradingState.MARKET_HALT}

    def summary(self) -> str:
        """状态机运行摘要"""
        lines = []
        lines.append(f"当前状态: {self.current_state.name}")
        lines.append(f"状态开始: {self.state_start_time}")
        lines.append(f"提示: {self.get_prompt()}")

        # 今日状态转换历史
        today = datetime.now().strftime('%Y-%m-%d')
        today_transitions = [t for t in self.state_history if t.timestamp.startswith(today)]
        if today_transitions:
            lines.append(f"\n今日状态转换 ({len(today_transitions)}次):")
            for t in today_transitions[-10:]:
                lines.append(f"  {t.timestamp}  {t.from_state.name} → {t.to_state.name}  ({t.reason})")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            'current_state': self.current_state.name,
            'state_start_time': self.state_start_time,
            'history_length': len(self.state_history),
        }
