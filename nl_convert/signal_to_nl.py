"""
信号 → 自然语言转换层
将结构化交易信号转化为 LLM 可理解的自然语言描述。
"""
import sys, os
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    CHANNEL_WINDOW, VOL_WINDOW, VOL_MULTIPLIER, VOL_ACTIVE_THRESHOLD,
    MARKET_INDEX, MARKET_MA_PERIOD, TODAY, DATE_RECENT_START,
    FUNDAMENTAL_MIN_SCORE
)
from stocks import SECTOR_MAP
from utils.data_fetcher import DataFetcher
from utils.fundamental import FundamentalAnalyzer
from strategy.regression_channel.channel_strategy import RegressionChannelWithVolumeStrategy
from strategy.MA.base import Signal


@dataclass
class MarketContext:
    """大盘环境上下文"""
    index_name: str = "上证指数"
    close: float = 0.0
    ma60: float = 0.0
    ma120: float = 0.0
    regime: str = "unknown"        # bull / mild_bear / bear / panic
    vol_adj: float = 1.0           # 波动率调整系数
    max_position_pct: float = 1.0  # 允许仓位比例
    description: str = ""


@dataclass
class TickerSignal:
    """单只股票信号描述"""
    code: str = ""
    name: str = ""
    # 价格信息
    close: float = 0.0
    upper: float = 0.0
    above_pct: float = 0.0
    # 成交量
    volume: float = 0.0
    vol_ma20: float = 0.0
    vol_ratio: float = 0.0
    is_active_period: bool = False
    # 上轨斜率
    slope: float = 0.0
    slope_status: str = ""  # 上行/横盘/下行
    # 信号
    signal_type: str = ""   # ★买入 / 突破 / 放量 / - / 卖出
    signal_value: int = 0
    # 基本面
    fund_score: Optional[float] = None
    fund_grade: str = "N/A"
    fund_details: Dict[str, float] = field(default_factory=dict)
    # 板块
    sector: str = "其他"
    # 组合上下文
    already_holding: bool = False
    holding_pnl_pct: float = 0.0
    sector_current_pct: float = 0.0  # 组合中该板块当前占比
    # 最近交易日
    date: str = ""


class SignalToNL:
    """将结构化信号转化为 LLM 可理解的自然语言"""

    def __init__(self):
        self.strategy = RegressionChannelWithVolumeStrategy(
            window=CHANNEL_WINDOW, vol_window=VOL_WINDOW,
            vol_multiplier=VOL_MULTIPLIER,
            vol_active_threshold=VOL_ACTIVE_THRESHOLD,
        )
        self.fundamental = FundamentalAnalyzer(min_score=FUNDAMENTAL_MIN_SCORE)

    # ========== 大盘环境 ==========

    def get_market_context(self) -> MarketContext:
        """获取大盘环境自然语言描述"""
        ctx = MarketContext()
        try:
            df = DataFetcher.get_index_hist(MARKET_INDEX, DATE_RECENT_START, TODAY)
            if df.empty:
                ctx.description = "大盘数据获取失败"
                return ctx

            close = df['close']
            ma60 = close.rolling(60).mean()
            ma120 = close.rolling(120).mean()

            latest = close.iloc[-1]
            ma60_val = ma60.iloc[-1]
            ma120_val = ma120.iloc[-1]

            ctx.close = float(latest)
            ctx.ma60 = float(ma60_val) if not pd.isna(ma60_val) else 0.0
            ctx.ma120 = float(ma120_val) if not pd.isna(ma120_val) else 0.0

            # 均线排列判定
            above_ma60 = latest > ma60_val
            above_ma120 = latest > ma120_val
            ma60_above_ma120 = ma60_val > ma120_val

            if above_ma60 and above_ma120 and ma60_above_ma120:
                ctx.regime = "bull"
                ctx.max_position_pct = 1.0
            elif not above_ma60 and not above_ma120:
                ctx.regime = "bear"
                ctx.max_position_pct = 0.0
            else:
                ctx.regime = "mild_bear"
                ctx.max_position_pct = 0.5

            # 波动率调整
            tr = pd.DataFrame({
                'hl': df['high'] - df['low'],
                'hc': (df['high'] - df['close'].shift(1)).abs(),
                'lc': (df['low'] - df['close'].shift(1)).abs(),
            }).max(axis=1)
            atr = tr.rolling(20).mean()
            atr_pct = float(atr.iloc[-1] / latest) if not pd.isna(atr.iloc[-1]) else 0.01
            if atr_pct > 0.03:
                ctx.vol_adj = 0.3
            elif atr_pct > 0.025:
                ctx.vol_adj = 0.5
            elif atr_pct > 0.02:
                ctx.vol_adj = 0.8
            else:
                ctx.vol_adj = 1.0

            ctx.max_position_pct = min(ctx.max_position_pct, ctx.vol_adj)

            # 生成文本描述
            regime_names = {'bull': '多头排列，市场处于上升趋势',
                           'mild_bear': '震荡偏弱，部分均线承压',
                           'bear': '空头排列，市场处于下行趋势'}
            parts = []
            parts.append(f"上证指数 {latest:.0f}点")
            parts.append(f"MA60={ma60_val:.0f}, MA120={ma120_val:.0f}")
            parts.append(f"市场状态: {regime_names.get(ctx.regime, ctx.regime)}")
            parts.append(f"允许仓位: {ctx.max_position_pct:.0%}")
            if atr_pct > 0.025:
                parts.append(f"高波动({atr_pct:.1%})，建议控制仓位")
            ctx.description = "；".join(parts)

        except Exception as e:
            ctx.description = f"大盘分析异常: {e}"

        return ctx

    # ========== 单只股票信号 → NL ==========

    def describe_ticker(self, code: str, name: str,
                        price: float = None,
                        holding_info: dict = None,
                        sector_positions: Dict[str, float] = None) -> TickerSignal:
        """
        对单只股票生成完整自然语言信号描述。

        Args:
            code: 股票代码
            name: 股票名称
            price: 当前价格（可选，不传则自动获取）
            holding_info: {'shares': int, 'cost': float} 持仓信息
            sector_positions: {sector: value} 组合中各板块当前市值
        """
        ts = TickerSignal(code=code, name=name)

        df = DataFetcher.get_stock_hist(code, DATE_RECENT_START, TODAY)
        if df.empty or len(df) < 60:
            return ts

        df['upper'] = self.strategy._calc_upper(df['high'])
        df['vol_ma'] = df['volume'].rolling(VOL_WINDOW).mean()
        df['vol_long'] = df['volume'].rolling(252).mean()

        latest = df.iloc[-1]
        prev = df.iloc[-2]
        close = float(latest['close'])
        u = float(latest['upper']) if not pd.isna(latest['upper']) else 0.0
        vol = float(latest['volume'])
        vol_ma20 = float(latest['vol_ma']) if not pd.isna(latest['vol_ma']) else 0.0

        ts.close = close
        ts.upper = u
        ts.above_pct = (close - u) / u * 100 if u > 0 else 0.0
        ts.volume = vol
        ts.vol_ma20 = vol_ma20
        ts.vol_ratio = vol / vol_ma20 if vol_ma20 > 0 else 0.0

        # 活跃期判定
        vol_long = float(latest['vol_long']) if not pd.isna(latest['vol_long']) else 0.0
        ts.is_active_period = vol_ma20 > vol_long * VOL_ACTIVE_THRESHOLD if vol_long > 0 else False

        # 斜率
        try:
            ts.slope = float(self.strategy._slope.iloc[-1]) if not pd.isna(self.strategy._slope.iloc[-1]) else 0.0
        except (AttributeError, IndexError):
            ts.slope = 0.0

        if ts.slope > 0.001:
            ts.slope_status = "上行"
        elif ts.slope < -0.001:
            ts.slope_status = "下行"
        else:
            ts.slope_status = "横盘"

        # 信号类型
        prev_below = float(prev['close']) < float(prev['upper']) if not pd.isna(prev['upper']) else False
        breakout = close >= u
        vol_ok = vol > vol_ma20 * VOL_MULTIPLIER
        if not vol_ok and ts.is_active_period:
            vol_ok = True

        if breakout and vol_ok and prev_below:
            ts.signal_type = "★买入"
            ts.signal_value = Signal.BUY.value
        elif breakout and vol_ok:
            ts.signal_type = "★买入"
            ts.signal_value = Signal.BUY.value
        elif breakout:
            ts.signal_type = "突破(无量)"
        elif vol_ok:
            ts.signal_type = "放量(未突破)"

        # 检查卖出信号
        signals = self.strategy.generate_signals(df.copy())
        last_sig = signals.iloc[-1] if len(signals) > 0 else 0
        if last_sig == Signal.SELL.value:
            ts.signal_type = "卖出"
            ts.signal_value = Signal.SELL.value
        elif last_sig == Signal.SELL_HALF.value:
            ts.signal_type = "卖出半仓"
            ts.signal_value = Signal.SELL_HALF.value

        # 基本面
        try:
            fund = self.fundamental.fetch_fundamentals(code, name, price=close)
            if fund:
                score = self.fundamental.compute_score(fund)
                ts.fund_score = round(score.total_score, 1)
                ts.fund_grade = score.grade
                ts.fund_details = score.details
        except Exception:
            pass

        # 板块
        for sec, syms in SECTOR_MAP.items():
            if code in syms:
                ts.sector = sec
                break

        # 持仓上下文
        if holding_info:
            ts.already_holding = holding_info.get('shares', 0) > 0
            if ts.already_holding:
                cost = holding_info.get('cost', close)
                ts.holding_pnl_pct = (close - cost) / cost * 100 if cost > 0 else 0.0

        if sector_positions:
            sec_val = sector_positions.get(ts.sector, 0)
            total = sum(sector_positions.values())
            ts.sector_current_pct = sec_val / total if total > 0 else 0.0

        ts.date = str(df.index[-1].date())
        return ts

    # ========== 批量扫描 → NL 报告 ==========

    def scan_to_context(self,
                        watchlist: List[Tuple[str, str]],
                        holdings: dict = None,
                        portfolio_value: float = 0.0) -> str:
        """
        扫描关注列表，生成完整的自然语言上下文报告。
        这个报告可以直接喂给 LLM 做决策。

        Args:
            watchlist: [(code, name), ...]
            holdings: {code: {'shares': int, 'cost': float}}
            portfolio_value: 组合总市值

        Returns:
            完整的 NL 上下文字符串
        """
        holdings = holdings or {}
        lines = []
        lines.append("=" * 60)
        lines.append(f"【交易决策上下文】 {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append("=" * 60)

        # 1. 大盘环境
        market = self.get_market_context()
        lines.append("\n## 大盘环境")
        lines.append(market.description)

        # 2. 当前持仓
        if holdings:
            lines.append("\n## 当前持仓")
            sector_vals = {}
            total_holding_value = 0.0
            for code, info in holdings.items():
                if info.get('shares', 0) <= 0:
                    continue
                t = self.describe_ticker(code, '', holding_info=info)
                val = t.close * info['shares']
                total_holding_value += val
                # 找板块
                for sec, syms in SECTOR_MAP.items():
                    if code in syms:
                        sector_vals[sec] = sector_vals.get(sec, 0) + val
                        break
                pnl_icon = "+" if t.holding_pnl_pct > 0 else ""
                lines.append(
                    f"- {code} {t.name}: {info['shares']}股 × {t.close:.2f} = {val:.0f} | "
                    f"浮动{pnl_icon}{t.holding_pnl_pct:.1f}% | "
                    f"上轨{t.upper:.2f}(偏离{t.above_pct:+.1f}%) | "
                    f"斜率{t.slope_status} | 板块:{t.sector}"
                )

            lines.append(f"\n总持仓市值: {total_holding_value:.0f}")
            lines.append(f"现金: {portfolio_value - total_holding_value:.0f}" if portfolio_value > 0 else "现金: 未知")

            # 板块分布
            if sector_vals:
                lines.append("\n板块分布:")
                for sec, val in sorted(sector_vals.items(), key=lambda x: x[1], reverse=True):
                    pct = val / total_holding_value * 100 if total_holding_value > 0 else 0
                    warn = " ⚠超限" if pct > 40 else ""
                    lines.append(f"  {sec}: {pct:.1f}%{warn}")

        # 3. 买入信号
        lines.append("\n## 信号扫描")
        buy_signals = []
        other_signals = []

        for code, name in watchlist:
            # 跳过已持有的
            info = holdings.get(code, {})
            t = self.describe_ticker(
                code, name,
                holding_info=info if info else None,
                sector_positions=sector_vals if holdings else None
            )
            if not t.close:
                continue

            if t.signal_type == "★买入":
                buy_signals.append(t)
            elif t.signal_type and t.signal_type not in ("", "-"):
                other_signals.append(t)

        if buy_signals:
            lines.append(f"\n### ★ 买入信号 ({len(buy_signals)}只)")
            lines.append("以下股票触发放量突破买入信号：\n")
            for i, t in enumerate(buy_signals, 1):
                lines.append(self._format_buy_signal(i, t))

        if other_signals:
            lines.append(f"\n### 其他值得关注的信号 ({len(other_signals)}只)")
            for t in other_signals:
                lines.append(f"- {t.code} {t.name}: {t.signal_type} | "
                           f"收盘{t.close:.2f} | 上轨{t.upper:.2f} | 偏离{t.above_pct:+.1f}%")

        # 4. 决策要点
        lines.append("\n## 决策要点")
        lines.append("请基于以上信息，对以下问题给出判断：")
        lines.append("1. 当前市场环境是否适合开仓/加仓？")
        if buy_signals:
            lines.append("2. 买入信号中哪些最值得参与？排序并说明理由")
            lines.append("3. 建议的仓位分配（考虑板块集中度和可用资金）")
        if holdings:
            lines.append("4. 现有持仓是否有需要减仓/清仓的？")

        return "\n".join(lines)

    def _format_buy_signal(self, index: int, t: TickerSignal) -> str:
        """格式化单条买入信号的详细描述"""
        lines = []
        lines.append(f"#{index} {t.code} {t.name} ({t.sector})")
        lines.append(f"  价格: 收盘 {t.close:.2f} | 预测上轨 {t.upper:.2f} | 偏离 +{t.above_pct:.1f}%")
        lines.append(f"  成交量: 量比 {t.vol_ratio:.2f}x | {'活跃期自动放量' if t.is_active_period else '标准放量'}")
        lines.append(f"  趋势: 上轨斜率 {t.slope:.4f} ({t.slope_status})")

        if t.fund_score is not None:
            fund_icon = "✓" if t.fund_score >= FUNDAMENTAL_MIN_SCORE else "✗"
            lines.append(f"  基本面: {t.fund_score}分/{t.fund_grade} {fund_icon} (阈值{FUNDAMENTAL_MIN_SCORE})")
            detail_str = ", ".join(f"{k}={v:.0f}" for k, v in t.fund_details.items() if v > 0)
            if detail_str:
                lines.append(f"  评分明细: {detail_str}")

        if t.already_holding:
            lines.append(f"  注意: 已持有该股，盈亏 {t.holding_pnl_pct:+.1f}%")

        # 板块风险提示
        if t.sector_current_pct > 0.35:
            lines.append(f"  ⚠ 板块集中度风险: {t.sector}已占组合 {t.sector_current_pct:.0%}，加仓可能超限")

        lines.append("")
        return "\n".join(lines)

    # ========== 卖出信号 → NL ==========

    def describe_sell_signal(self, code: str, name: str,
                             holding_info: dict) -> str:
        """
        对持仓股生成卖出/预警信号的自然语言描述。
        """
        t = self.describe_ticker(code, name, holding_info=holding_info)
        lines = []
        lines.append(f"{code} {name}: 持有{holding_info['shares']}股, "
                    f"成本{holding_info['cost']:.2f}, "
                    f"现价{t.close:.2f}, 浮{'盈' if t.holding_pnl_pct > 0 else '亏'}{abs(t.holding_pnl_pct):.1f}%")

        if t.signal_type in ("卖出", "卖出半仓"):
            lines.append(f"  ⚠ 策略发出卖出信号: {t.signal_type}")
            if t.slope_status == "下行":
                lines.append(f"  原因: 上轨斜率下行，假突破风险上升")

        # 预警条件
        if t.slope_status == "下行" and t.above_pct < 1.0:
            lines.append(f"  ⚠ 主动预警: 逼近上轨（仅偏离{t.above_pct:+.1f}%）+ 斜率下行")
        if abs(t.holding_pnl_pct) > 10:
            lines.append(f"  ⚠ 风险提示: 浮{'亏' if t.holding_pnl_pct < 0 else '盈'}已超10%")

        return "\n".join(lines)
