"""投资组合配置"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class PortfolioConfig:
    """投资组合配置"""
    # 资金
    initial_capital: float = 1_000_000       # 初始资金
    reserve_cash_pct: float = 0.10           # 保留现金比例

    # 仓位限制
    max_positions: int = 5                   # 最大同时持仓数
    max_position_pct: float = 0.25           # 单只上限（T+1风控）
    min_capital_per_pos: float = 10_000      # 单只最低占用
    sector_exposure_limit: float = 0.40      # 单板块上线
    sizing_method: str = 'equal'             # equal | signal_weighted | volatility_parity

    # 交易成本 (A股实盘费率)
    commission: float = 0.00025              # 手续费（万2.5）
    stamp_duty: float = 0.0005               # 印花税（万5，仅卖出，2023.8减半）

    # 风控
    portfolio_stop_loss: float = -0.25       # 组合止损（总亏损，安全网）
    portfolio_max_drawdown: float = -0.25    # 组合最大回撤（从峰值回落，安全网）
    volatility_limit: float = 0.35           # 年化波动率上限
    consecutive_loss_limit: int = 5          # 最大连续亏损次数

    # 市场过滤器
    market_ma_short: int = 60                # 大盘短期均线
    market_ma_long: int = 120                # 大盘长期均线
    market_bear_threshold: float = -0.10     # 大盘进入熊市阈值
    market_position_limits: dict = field(default_factory=lambda: {
        'bull': 1.0,      # 牛市满仓
        'neutral': 0.6,   # 震荡半仓以上
        'bear': 0.0,      # 熊市空仓
    })

    # 回测
    lookback_days: int = 120                 # 信号计算回溯天数

    # 自选股（默认使用 SCREENER_WATCHLIST）
    use_default_universe: bool = True
    custom_universe: List[str] = field(default_factory=list)
