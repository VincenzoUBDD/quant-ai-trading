"""风险控制引擎 — 组合级 + 个股级风控"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from .config import PortfolioConfig
from .position_sizer import PositionSizer


class RiskEngine:
    """
    风险控制引擎

    功能：
      - 组合级止损 / 最大回撤限制
      - 个股级止损（ATR追踪 / 固定比例）
      - 波动率限制（年化波动率 > 阈值 → 减仓）
      - 市场状态联动（自动减仓）
      - 最大连续亏损计数
      - VaR(95%) / CVaR(95%)
    """

    def __init__(self, config: PortfolioConfig):
        self.config = config
        self.consecutive_losses = 0
        self.peak_equity = 0
        self.daily_returns = []  # 近20日收益缓存

    def check_portfolio_stop(self,
                             current_equity: float,
                             initial_capital: float,
                             peak_equity: float = None) -> Tuple[bool, str]:
        """
        检查组合级止损

        Args:
            current_equity: 当前组合净值
            initial_capital: 初始资金
            peak_equity: 历史最高净值（None则自动累计）

        Returns:
            (should_stop: bool, reason: str)
        """
        if peak_equity is not None:
            self.peak_equity = max(self.peak_equity, peak_equity)

        self.peak_equity = max(self.peak_equity, current_equity)

        # 总亏损止损
        total_loss = (current_equity - initial_capital) / initial_capital
        if total_loss <= self.config.portfolio_stop_loss:
            return True, f'组合亏损 {total_loss:.1%} 触发止损线 {self.config.portfolio_stop_loss:.0%}'

        # 最大回撤止损
        if self.peak_equity > 0:
            dd = (current_equity - self.peak_equity) / self.peak_equity
            if dd <= self.config.portfolio_max_drawdown:
                return True, f'组合回撤 {dd:.1%} 触发最大回撤线 {self.config.portfolio_max_drawdown:.0%}'

        return False, ''

    def check_consecutive_loss(self, daily_return: float) -> Tuple[bool, int]:
        """
        检查连续亏损

        Args:
            daily_return: 当日收益率

        Returns:
            (should_stop: bool, consecutive_count: int)
        """
        if daily_return < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        if self.consecutive_losses >= self.config.consecutive_loss_limit:
            return True, self.consecutive_losses
        return False, self.consecutive_losses

    def check_volatility_limit(self,
                                equity_curve: pd.Series) -> Tuple[bool, float]:
        """
        检查波动率限制

        Args:
            equity_curve: 近期净值序列

        Returns:
            (breached: bool, annualized_vol: float)
        """
        if len(equity_curve) < 20:
            return False, 0

        returns = equity_curve.pct_change().dropna().tail(60)  # 近60日
        if len(returns) < 20:
            return False, 0

        ann_vol = returns.std() * np.sqrt(252)
        if ann_vol > self.config.volatility_limit:
            return True, ann_vol
        return False, ann_vol

    def estimate_var(self, equity_curve: pd.Series) -> dict:
        """
        估算 VaR / CVaR

        Returns:
            {'var_95': float, 'cvar_95': float, 'daily_vol': float}
        """
        if len(equity_curve) < 20:
            return {'var_95': 0, 'cvar_95': 0, 'daily_vol': 0}

        returns = equity_curve.pct_change().dropna().tail(252)
        if len(returns) < 20:
            return {'var_95': 0, 'cvar_95': 0, 'daily_vol': 0}

        var_95 = returns.quantile(0.05)
        cvar_95 = returns[returns <= var_95].mean()
        return {
            'var_95': float(var_95),
            'cvar_95': float(cvar_95),
            'daily_vol': float(returns.std()),
        }

    def adjust_for_regime(self,
                           base_position_pct: float,
                           regime_pct: float,
                           vol_adj: float) -> float:
        """根据市场状态调整仓位"""
        return min(base_position_pct, regime_pct, vol_adj)

    def generate_risk_report(self,
                              current_equity: float,
                              initial_capital: float,
                              equity_curve: pd.Series,
                              positions: Dict[str, float] = None) -> str:
        """生成风险报告"""
        lines = []
        lines.append('=' * 60)
        lines.append('【风险报告】')
        lines.append('=' * 60)

        total_return = (current_equity - initial_capital) / initial_capital * 100
        peak = max(self.peak_equity, current_equity)
        current_dd = (current_equity - peak) / peak * 100 if peak > 0 else 0

        lines.append(f'总收益率: {total_return:+.2f}%')
        lines.append(f'当前回撤: {current_dd:.2f}%')
        lines.append(f'连续亏损: {self.consecutive_losses} 次 (上限: {self.config.consecutive_loss_limit})')
        lines.append('')

        if len(equity_curve) > 20:
            var_info = self.estimate_var(equity_curve)
            lines.append(f'VaR(95%): {var_info["var_95"]:.2%}')
            lines.append(f'CVaR(95%): {var_info["cvar_95"]:.2%}')
            lines.append(f'日波动率: {var_info["daily_vol"]:.2%}')
            lines.append('')

            vol_check, ann_vol = self.check_volatility_limit(equity_curve)
            lines.append(f'年化波动率: {ann_vol:.1%} (上限: {self.config.volatility_limit:.0%})')
            if vol_check:
                lines.append('  ⚠ 波动率超限！')

        # 止损检查
        stop, reason = self.check_portfolio_stop(current_equity, initial_capital, peak)
        if stop:
            lines.append('')
            lines.append(f'🚨 {reason}')

        return '\n'.join(lines)
