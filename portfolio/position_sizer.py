"""仓位计算器 — 三种模式 + A股实盘约束"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from .config import PortfolioConfig


class PositionSizer:
    """
    仓位计算器

    三种模式：
      - equal: 等权重，available / max_positions
      - signal_weighted: 按信号强度分配
      - volatility_parity: ATR倒数加权（波动率平价）

    A股实盘约束：
      - T+1 → 单只上限 25%
      - 最小100股
      - 板块上限 40%
    """

    def __init__(self, config: PortfolioConfig):
        self.config = config

    def compute_weights(self,
                        method: str = None,
                        available_cash: float = 0,
                        signals: Dict[str, float] = None,
                        prices: Dict[str, float] = None,
                        atr_values: Dict[str, float] = None,
                        sector_map: Dict[str, List[str]] = None,
                        current_positions: Dict[str, float] = None) -> Dict[str, float]:
        """
        计算目标权重

        Args:
            method: 仓位计算方法（equal/signal_weighted/volatility_parity）
            available_cash: 可用资金
            signals: {symbol: signal_strength} 信号强度（仅 signal_weighted 用）
            prices: {symbol: current_price} 当前价格（取整用）
            atr_values: {symbol: atr_pct} ATR百分比（volatility_parity 用）
            sector_map: {sector: [symbols]} 板块分组
            current_positions: {symbol: current_value} 当前持仓市值

        Returns:
            {symbol: target_weight} 目标权重（总和 <= 1-reserve_cash_pct）
        """
        method = method or self.config.sizing_method
        n_positions = self.config.max_positions
        signals = signals or {}
        prices = prices or {}
        atr_values = atr_values or {}
        current_positions = current_positions or {}

        if not signals:
            return {}

        # 按信号强度排序
        candidates = sorted(signals.items(), key=lambda x: x[1], reverse=True)

        # 限制候选数量
        candidates = candidates[:n_positions * 2]

        weights = {}
        selected = [s for s, _ in candidates]

        if method == 'equal':
            # 等权重
            per_pos = 1.0 / max(len(selected), 1)
            for sym in selected:
                weights[sym] = per_pos

        elif method == 'signal_weighted':
            # 按信号强度加权
            total_strength = sum(abs(s[1]) for s in candidates)
            if total_strength > 0:
                for sym, strength in candidates:
                    weights[sym] = abs(strength) / total_strength
            else:
                per_pos = 1.0 / max(len(selected), 1)
                for sym in selected:
                    weights[sym] = per_pos

        elif method == 'volatility_parity':
            # 波动率平价（ATR倒数加权）
            atr_vals = {sym: atr_values.get(sym, 1) for sym in selected}
            # 限制极端值
            for sym in atr_vals:
                atr_vals[sym] = max(atr_vals[sym], 0.01)

            inv_vol_sum = sum(1.0 / v for v in atr_vals.values())
            if inv_vol_sum > 0:
                for sym in selected:
                    weights[sym] = (1.0 / atr_vals[sym]) / inv_vol_sum
            else:
                per_pos = 1.0 / max(len(selected), 1)
                for sym in selected:
                    weights[sym] = per_pos

        # 应用单只上限
        max_pos = self.config.max_position_pct
        for sym, w in weights.items():
            if available_cash > 0 and current_positions:
                current_value = current_positions.get(sym, 0)
                if current_value > 0 and current_value / (available_cash + sum(current_positions.values())) > max_pos:
                    weights[sym] = max_pos
            else:
                weights[sym] = min(w, max_pos)

        # 应用板块限制
        if sector_map:
            weights = self._apply_sector_limit(weights, sector_map)

        # 重新归一化
        total = sum(weights.values())
        if total > 0:
            max_total = 1.0 - self.config.reserve_cash_pct
            scale = min(1.0, max_total / total)
            weights = {s: w * scale for s, w in weights.items()}

        return weights

    def compute_shares(self,
                       weights: Dict[str, float],
                       prices: Dict[str, float],
                       total_capital: float) -> Dict[str, Tuple[int, float]]:
        """
        根据权重计算实际买入股数（100股取整）

        Returns:
            {symbol: (shares, cost)}
        """
        result = {}
        for sym, w in weights.items():
            price = prices.get(sym, 0)
            if price <= 0:
                continue
            alloc = total_capital * w
            if alloc < self.config.min_capital_per_pos:
                continue
            shares = int(alloc / price / 100) * 100  # 向下取整到100股
            if shares >= 100:
                result[sym] = (shares, shares * price)
        return result

    def _apply_sector_limit(self,
                             weights: Dict[str, float],
                             sector_map: Dict[str, List[str]]) -> Dict[str, float]:
        """应用板块限制"""
        sector_weights = {}
        for sector, symbols in sector_map.items():
            sector_w = sum(weights.get(s, 0) for s in symbols)
            if sector_w > self.config.sector_exposure_limit:
                # 按比例缩减该板块权重
                scale = self.config.sector_exposure_limit / sector_w
                for s in symbols:
                    if s in weights:
                        weights[s] *= scale
        return weights

    def compute_target_positions(self,
                                  available_cash: float,
                                  signals: Dict[str, float],
                                  prices: Dict[str, float],
                                  current_holdings: Dict[str, float] = None,
                                  atr_values: Dict[str, float] = None,
                                  sector_map: Dict[str, List[str]] = None) -> Dict[str, Dict]:
        """
        一站式计算目标持仓（仓位+股数）

        Returns:
            {symbol: {'weight': float, 'shares': int, 'cost': float, 'action': str}}
        """
        current_holdings = current_holdings or {}
        weights = self.compute_weights(
            available_cash=available_cash,
            signals=signals,
            prices=prices,
            atr_values=atr_values,
            sector_map=sector_map,
            current_positions=current_holdings,
        )

        shares_info = self.compute_shares(weights, prices, available_cash)
        total_capital = available_cash + sum(current_holdings.values())

        result = {}
        for sym, (shares, cost) in shares_info.items():
            current_shares = current_holdings.get(sym, 0)
            if current_shares > 0 and current_shares >= shares:
                action = 'HOLD'
            elif current_shares > 0 and shares > current_shares:
                action = 'ADD'
            elif shares > 0:
                action = 'BUY'
            else:
                action = 'HOLD'
            result[sym] = {
                'weight': weights.get(sym, 0),
                'shares': shares,
                'cost': cost,
                'action': action,
            }
        return result
