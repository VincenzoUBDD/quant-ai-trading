# 统一参数配置
# 日期自动计算，无需每日修改；修改策略参数或回测标的在此文件调整
from datetime import datetime, timedelta

# ===== 策略参数 =====
CHANNEL_WINDOW = 40       # 回归通道天数
VOL_WINDOW = 20           # 成交量均线周期
VOL_MULTIPLIER = 1.5      # 放量倍数
VOL_ACTIVE_THRESHOLD = 1.3  # 活跃期判定：20日均量 > 年均量 × 该值，突破自动认定放量
INITIAL_CAPITAL = 100000  # 初始资金

# ===== 回测默认标的 =====
SYMBOL = '002230'

# ===== 动态日期（各模块自动使用，无需手动修改） =====
_NOW = datetime.now()
TODAY = _NOW.strftime('%Y%m%d')                                        # 当天
DATE_RECENT_START = (_NOW - timedelta(days=120)).strftime('%Y%m%d')    # 扫描用：近120天
DATE_BT_START    = (_NOW - timedelta(days=3*365)).strftime('%Y%m%d')   # 回测用：近3年

# ===== 交易成本（A股实盘费率） =====
COMMISSION = 0.00025           # 佣金（万2.5，买卖双向）
STAMP_DUTY = 0.0005            # 印花税（万5，仅卖出，2023.8减半）

# ===== 风险控制参数 =====
ATR_PERIOD = 14                # ATR 计算周期
ATR_MULTIPLIER = 2.0           # ATR 追踪止盈倍数
PARTIAL_PROFIT_TARGET = 0.08   # 部分止盈目标（盈利 +8% 卖半仓）
FALSE_BREAKOUT_RETRACE = 0.03  # 假突破退出：从高点回落阈值（斜率<0时生效）
VOL_SHRINK_RATIO = 0.8       # 缩量阈值：成交量 < MA20 × 该值 → 认定为缩量
SELL_DELAY_DAYS = 2          # 缩量跌破上轨后最长观察天数（到期强制卖出）

# ===== KDJ 叠加策略参数 =====
KDJ_PERIOD = 9             # KDJ 周期（标准9日）
KDJ_OVERBOUGHT = 80.0      # J 值超买阈值（超过此值卖出）
KDJ_OVERSOLD = -10.0        # J 值超卖阈值（低于此值买入，调优后最优值 -10）
KDJ_SLOPE_WINDOW = 30      # 收盘价线性回归窗口（用于趋势判定）

# ===== 大盘环境过滤 =====
MARKET_INDEX = '000001'        # 上证指数
MARKET_MA_PERIOD = 60          # 大盘均线周期

# ===== 基本面分析参数 =====
FUNDAMENTAL_MIN_SCORE = 4.0   # 最低基本面评分 (0-10)，低于此值过滤
FUNDAMENTAL_BT_YEAR = 2021    # 回测使用哪年年报（BT_LONG_START 前一年，避免未来函数）

# ===== 固定日期（长周期回测用，按需调整） =====
BT_LONG_START = '20220101'
BT_LONG_END   = '20260101'

# ===== 投资组合参数 =====
PORTFOLIO_INITIAL_CAPITAL = 1_000_000    # 组合初始资金
PORTFOLIO_MAX_POSITIONS = 5              # 最大同时持仓数
PORTFOLIO_SECTOR_LIMIT = 0.40            # 单板块上线
PORTFOLIO_RESERVE_CASH = 0.10            # 保留现金比例
PORTFOLIO_STOP_LOSS = -0.15              # 组合止损线
PORTFOLIO_MAX_DRAWDOWN = -0.25           # 组合最大回撤线
