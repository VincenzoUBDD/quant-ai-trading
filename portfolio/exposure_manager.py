"""板块/个股暴露管理"""
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from collections import defaultdict


class ExposureManager:
    """
    暴露管理

    功能：
      - 板块暴露计算与监控
      - 个股集中度（HHI）
      - 风格暴露（预留）
      - 暴露报告生成
    """

    def __init__(self, sector_map: Dict[str, List[str]] = None,
                 sector_limit: float = 0.40,
                 stock_limit: float = 0.25):
        """
        Args:
            sector_map: {sector_name: [symbols]}
            sector_limit: 单板块上限
            stock_limit: 单只上限
        """
        self.sector_map = sector_map or {}
        self.sector_limit = sector_limit
        self.stock_limit = stock_limit

    def compute_sector_exposure(self,
                                 positions: Dict[str, float],
                                 total_value: float) -> Dict[str, dict]:
        """
        计算各板块暴露

        Args:
            positions: {symbol: market_value}
            total_value: 组合总市值

        Returns:
            {sector_name: {'value': float, 'pct': float, 'limit': float, 'status': str}}
        """
        if total_value <= 0:
            return {}

        # 建立symbol->sector的逆映射
        sym_to_sector = {}
        for sector, symbols in self.sector_map.items():
            for sym in symbols:
                sym_to_sector[sym] = sector

        sector_exposure = defaultdict(float)
        for sym, val in positions.items():
            if val > 0:
                sector = sym_to_sector.get(sym, '其他')
                sector_exposure[sector] += val

        result = {}
        for sector, value in sector_exposure.items():
            pct = value / total_value
            over_limit = pct > self.sector_limit
            result[sector] = {
                'value': value,
                'pct': pct,
                'limit': self.sector_limit,
                'status': '⚠ 超限' if over_limit else 'OK',
                'excess': pct - self.sector_limit if over_limit else 0,
            }
        return result

    def compute_concentration(self,
                               positions: Dict[str, float],
                               total_value: float) -> dict:
        """
        计算集中度指标

        Returns:
            {'stock_hhi': float, 'sector_hhi': float, 'top1_pct': float, 'top3_pct': float}
        """
        if total_value <= 0 or not positions:
            return {'stock_hhi': 0, 'sector_hhi': 0, 'top1_pct': 0, 'top3_pct': 0}

        pcts = np.array([v / total_value for v in positions.values() if v > 0])

        # HHI = Σ(weight_i^2)，值越大越集中
        stock_hhi = float(np.sum(pcts ** 2))

        # 前N名集中度
        sorted_pcts = sorted(pcts, reverse=True)
        top1 = sorted_pcts[0] if sorted_pcts else 0
        top3 = sum(sorted_pcts[:3]) if len(sorted_pcts) >= 3 else sum(sorted_pcts)

        # 板块HHI
        sector_exposure = self.compute_sector_exposure(positions, total_value)
        sector_pcts = [v['pct'] for v in sector_exposure.values() if v['pct'] > 0]
        sector_hhi = float(np.sum(np.array(sector_pcts) ** 2)) if sector_pcts else 0

        return {
            'stock_hhi': round(stock_hhi, 4),
            'sector_hhi': round(sector_hhi, 4),
            'top1_pct': float(top1),
            'top3_pct': float(top3),
        }

    def check_limits(self,
                     positions: Dict[str, float],
                     total_value: float) -> List[str]:
        """
        检查所有风控限制，返回告警列表

        Returns:
            [warning_message, ...]
        """
        warnings = []

        if total_value <= 0:
            return warnings

        # 个股限制
        for sym, val in positions.items():
            pct = val / total_value
            if pct > self.stock_limit:
                warnings.append(f'{sym} 仓位 {pct:.1%} 超过单只上限 {self.stock_limit:.0%}')

        # 板块限制
        sector_exp = self.compute_sector_exposure(positions, total_value)
        for sector, info in sector_exp.items():
            if info['status'] == '⚠ 超限':
                warnings.append(f'{sector} 板块暴露 {info["pct"]:.1%} 超过上限 {self.sector_limit:.0%}')

        return warnings

    def generate_report(self,
                        positions: Dict[str, float],
                        total_value: float,
                        symbol_names: Dict[str, str] = None) -> str:
        """生成暴露分析报告"""
        lines = []
        lines.append('=' * 60)
        lines.append('【暴露分析报告】')
        lines.append('=' * 60)
        lines.append(f'组合总市值: {total_value:,.2f}')
        lines.append(f'持仓个数: {sum(1 for v in positions.values() if v > 0)}')
        lines.append('')

        if total_value > 0:
            # 个股持仓排名
            lines.append('-- 个股持仓 Top 5 --')
            sorted_pos = sorted([(s, v) for s, v in positions.items() if v > 0],
                                key=lambda x: x[1], reverse=True)
            for sym, val in sorted_pos[:5]:
                name = symbol_names.get(sym, '') if symbol_names else ''
                pct = val / total_value
                lines.append(f'  {sym} {name}: {val:>10,.2f} ({pct:>6.2%})')
            lines.append('')

            # 板块暴露
            lines.append('-- 板块暴露 --')
            sector_exp = self.compute_sector_exposure(positions, total_value)
            for sector, info in sorted(sector_exp.items(), key=lambda x: x[1]['pct'], reverse=True):
                lines.append(f'  {sector}: {info["pct"]:>6.2%}  {info["status"]}')
            lines.append('')

            # 集中度
            conc = self.compute_concentration(positions, total_value)
            lines.append(f'个股HHI: {conc["stock_hhi"]:.4f}')
            lines.append(f'板块HHI: {conc["sector_hhi"]:.4f}')
            lines.append(f'Top1仓位: {conc["top1_pct"]:.2%}')
            lines.append(f'Top3仓位: {conc["top3_pct"]:.2%}')

            # 告警
            warnings = self.check_limits(positions, total_value)
            if warnings:
                lines.append('')
                lines.append('-- 告警 --')
                for w in warnings:
                    lines.append(f'  ⚠ {w}')

        return '\n'.join(lines)
