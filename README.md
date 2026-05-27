# A股趋势跟踪策略系统

基于线性回归通道 + 放量突破 + 买入点基本面过滤 + 缩量卖出观察的 A 股量化系统。
支持单股回测、扫描选股、投资组合回测、批量回测。

## 策略逻辑

### 上轨计算（无未来函数）

上轨使用历史数据预测，不引入当天信息：

```
上轨[i] = 对 high[i-window : i] 做线性回归，取回归线最后一点的值
```

即用 **第 i-window 天到第 i-1 天** 的最高价预测第 i 天的上轨，不包含第 i 天信息。

### 买入

股价放量突破预测上轨，且在买入信号触发时**检查基本面评分**：

- 前一日收盘 < 前一日上轨（预测值）← 新鲜突破
- 当日收盘 ≥ 当日上轨（预测值）← 站上预测上轨
- 成交量判定
- 基本面评分 >= 阈值(默认4.0) ← 买入点基本面过滤

### 成交量判定（双条件）

```
① volume > 20日均量 × 1.5       ← 标准放量
② 20日均量 > 年均量 × 1.3       ← 活跃期自动放量（不另卡量）
```

### 卖出（分阶段 + 缩量观察）

**缩量跌破上轨 → 延迟卖出**：
```
跌破上轨时检查成交量：
  缩量（volume < MA20_vol × 0.8）→ 多给 2 日观察，期内回到上轨以上则取消卖出
  放量跌破 / 观察期中出现放量 → 立即执行
```

**下行期（上轨斜率 < 0）→ 假突破退出**：
斜率 < 0 时，从持仓最高点回落 > 3% 即清仓（不受缩量观察影响，强制退出）。

**上行/横盘期（上轨斜率 ≥ 0）→ 跌破预测上轨卖出**。

**始终生效**：盈利 +8% 卖半仓。

## 基本面过滤系统

在 **每次买入信号触发时** 检查基本面，保留所有交易机会，只在买入时点根据当时财务数据决策。

### 评分体系

| 指标 | 权重 | 评分逻辑 |
|------|------|----------|
| ROE | 0.25 | >=25%→10分, >=20%→8分, >=15%→6分, >=10%→4分, >=5%→2分 |
| 净利润增长率 | 0.20 | >=50%→10, >=30%→8, >=20%→6, >=10%→4, >=0%→2 |
| 营收增长率 | 0.15 | 同上 |
| PE | 0.10 | <=15→10, <=30→8, <=50→6, <=80→4; 负值→0 |
| 毛利率 | 0.10 | >=60%→10, >=40%→8, >=30%→6, >=20%→4, >=10%→2 |
| 资产负债率 | 0.10 | 反向: <=20%→10, <=40%→8, <=60%→6, <=70%→4, <=85%→2 |
| 质押比例 | 0.10 | 反向: <=5%→10, <=10%→8, <=20%→6, <=30%→4, <=50%→2 |

总分 0-10 分，默认阈值 4.0。缺失指标权重等比再分配。

### 基金持仓因子

当基金持仓数据可用时，基金持仓比例增长作为加分项（最高 +2.0 分）：

| 季度持仓比例增长 | 加分 |
|------------------|------|
| >5 个百分点 | +2.0 |
| >2 个百分点 | +1.5 |
| >0 | +1.0 |

### 数据缓存

- `data/fundamental_cache/abstract_raw.pkl` — `stock_financial_abstract` 原始数据
- `data/fundamental_cache/institutional_holdings.pkl` — 机构持仓历史数据
- 首次运行约 5 分钟，后续秒级加载

## 投资组合系统

多股共享资金池，按信号 + 风控 + 市场状态动态分配仓位。

### 核心组件

| 模块 | 功能 |
|------|------|
| `portfolio/engine.py` | 组合回测引擎（多股资金池、信号执行、持仓接力） |
| `portfolio/config.py` | 组合配置（资金、仓位限制、交易成本、风控参数） |
| `portfolio/position_sizer.py` | 仓位计算器（等权重/信号加权/波动率平价） |
| `portfolio/risk_engine.py` | 风控引擎（组合止损、回撤限制、波动率上限、连续亏损） |
| `portfolio/exposure_manager.py` | 板块暴露管理（板块上限、集中度检查） |
| `portfolio/market_regime.py` | 市场状态识别（MA60/MA120、波动率调整、恐慌检测） |
| `portfolio/results.py` | 结果数据类 + 绩效指标计算 |
| `portfolio/manager.py` | 实盘组合管理（每日调仓建议） |

### 组合回测数据流

```
每个交易日 t:
  1. 获取 universe 中所有股票的 OHLCV[t]
  2. 对每支股票运行 generate_signals() → 当天信号
  3. 卖出阶段：已有持仓出现卖出信号 → 执行，资金回现金池
  4. 风控检查：止损/回撤/波动率/市场状态
  5. 买入阶段：收集买入信号，经板块/仓位约束后分配资金
  6. 计算组合净值 = 现金 + Σ(shares_i × close_i)
```

### 风控机制

- **组合级止损**（-25%总亏损硬止损）和 **最大回撤限制**（-25%从峰值回落）
- **熔断恢复**：触发后平仓但不退出循环，市场恢复（MA60/MA120多头排列）后自动重新入场
- **市场状态联动**：大盘MA60/MA120空头排列 → 自动降仓位至50%或0%
- **波动率调整**：指数ATR > 3% → 仓位降至30%
- **板块暴露限制**：单板块上限40%，避免行业黑天鹅

### 回测结果（2022-2026.05，原版22支 + PortfolioEngine）

| 配置 | 总收益 | 年化 | 最大回撤 | 夏普 | 说明 |
|------|--------|------|---------|------|------|
| **max5-等权** | **+132.92%** | **+21.65%** | **-15.50%** | **1.07** | 默认配置，推荐 |
| max5-信号加权 | +132.97% | +21.66% | -15.56% | 1.07 | 同质化强，等权已够用 |
| max5-波动率平价 | +128.57% | +21.12% | -15.41% | 1.06 | 同质化强效果接近 |
| max3-等权 | +91.88% | +16.31% | -14.95% | 0.92 | 仓位太少，漏掉同步上涨 |

> **说明：** 之前录得的 +62.57% 来自止损熔断永久退出循环的 bug 版本。修复后引擎支持熔断恢复，回测结果大幅提升至 +132.92%。

**结论：** 策略的超额收益集中在原版22支股票池，扩大选股范围反而稀释收益。22支中混入的非科技票（宇通客车、中国中免等）在策略里很少触发信号，客观上起到了"自动减仓"的压舱石作用。

### 滚动回测验证

每年用过去2年数据从200支候选股中选收益率Top 50，持仓接力回测：

| 年份 | 收益率 | 股票池 | Top50重叠度 |
|------|--------|--------|------------|
| 2024 | +12.42% | 2022-2023选出的Top50 | — |
| 2025 | -8.16% | 2023-2024重选，继承1支 | 18/50与前一年重叠 |
| 2026 | -5.50% | 2024-2025重选，继承4支 | 24/50与前一年重叠 |
| **合计** | **-3.76%** | — | 仅10/50三年全上榜 |

说明该策略的选股结果高度不稳定，Top50名单每年剧烈变化。最佳实践是固定持有AI/半导体板块，而非滚动重选。

## 项目结构

```
├── config.py                  # 统一参数配置
├── stocks.py                  # 自选股（22支，默认）
│
├── strategy/                  # 策略模块
│   ├── MA/base.py             # 策略基类（Signal枚举、半仓、活跃期判定）
│   ├── regression_channel/    # 线性回归通道策略（核心策略）
│   │   └── channel_strategy.py
│   └── dl/                    # LSTM策略
│
├── portfolio/                 # 投资组合系统
│   ├── engine.py              # 组合回测引擎（核心）
│   ├── manager.py             # 实盘组合管理
│   ├── risk_engine.py         # 风险控制引擎
│   ├── position_sizer.py      # 仓位计算器
│   ├── exposure_manager.py    # 板块暴露管理
│   ├── market_regime.py       # 市场状态识别
│   └── results.py             # 结果数据类
│
├── backtest/                  # 单股回测
├── screener/                  # 筛选与验证
│   ├── screen.py              # 每日扫描选股
│   ├── verify.py              # 单股回测验证
│   └── daily.py               # 开盘前检查
│
├── nl_convert/                # [NEW] 信号→自然语言转换层
│   └── signal_to_nl.py        # 结构化信号转 LLM 可读文本
│
├── memory/                    # [NEW] RAG 交易记忆系统
│   ├── trade_memory.py        # 记忆管理器（读写/检索/反思）
│   └── ticker_memories/       # 每标的独立记忆文件
│
├── scheduler/                 # [NEW] 事件流调度器
│   └── event_flow.py          # 盘前→开盘→收盘自动化 pipeline
│
├── state_machine/             # [NEW] 交易状态机
│   └── trading_fsm.py         # 交易生命周期状态管理
│
├── analytics/                 # [NEW] 成本追踪 + 决策日志
│   └── cost_tracker.py        # token/电费/服务器 vs alpha 追踪
│
├── batch_backtest.py          # 批量回测（龙头股）
├── visualize.py               # 单股可视化
├── main.py                    # 单股回测主入口
├── check_watchlist.py         # 持仓快速检查
│
├── utils/
│   ├── data_fetcher.py        # akshare数据获取
│   └── fundamental.py         # 基本面分析器
│
├── data/fundamental_cache/    # 基本面数据缓存
└── results/                   # 回测结果 + 分析输出
```

## 安装

```bash
pip install akshare pandas numpy matplotlib
# LSTM 策略需要额外安装:
pip install tensorflow scikit-learn
```

## 使用方法

### ① 每日选股扫描

```bash
python screener/screen.py
```

扫描 `stocks.py` 中配置的 22 只科技/AI 股，输出买入信号、突破待确认等分类，附带基本面评分。

### ② 单股回测验证

```bash
python screener/verify.py 002050
```

对候选股跑近3年回测，输出收益率/胜率/夏普等指标，含基本面分析。

### ③ 事件流调度（NEW）

```bash
# 运行完整 pipeline（盘前 → 开盘扫描 → 收盘反思）
python scheduler/event_flow.py --once

# 只跑盘前检查
python scheduler/event_flow.py --stage pre

# 导出 LLM 上下文到文件（喂给 LLM 做决策）
python scheduler/event_flow.py --once --output daily_context.txt
```

自动串联：持仓检查 + 信号扫描 + 历史记忆检索 + 反思提示生成。

### ④ 信号 NL 转换（NEW）

```python
from nl_convert import SignalToNL

nl = SignalToNL()
# 获取大盘环境描述
market = nl.get_market_context()
print(market.description)

# 获取单只标的的完整 NL 描述
t = nl.describe_ticker('002230', '科大讯飞')
# t对象包含: close, upper, above_pct, vol_ratio, slope, signal_type, fund_score ...

# 批量扫描生成 LLM 上下文
ctx = nl.scan_to_context(SCREENER_WATCHLIST, holdings=WATCH_LIST)
# 把 ctx 直接喂给 LLM
```

### ⑤ RAG 交易记忆（NEW）

```python
from memory import TradeMemory, TradeRecord

mem = TradeMemory()

# 记录一笔交易
mem.record_trade('002230', TradeRecord(
    date='2026-05-27', direction='BUY', price=72.64, shares=500,
    value=36320, reason='放量突破上轨,基本面6.8分'
))

# 记录反思（收盘后跑）
mem.record_reflection('002230', {
    'trigger': '持仓中触发缩量预警但未跌破',
    'content': '缩量回踩上轨后反弹，下次类似情况可适当放宽观察期',
    'lessons': '该标的缩量洗盘概率高，建议观察3天而非2天',
})

# 检索历史上下文（塞入 LLM prompt）
ctx = mem.retrieve_context('002230')

# 批量检索多个标的
ctx = mem.retrieve_for_decision(['002230', '002371', '300308'])

# 生成反思 prompt（发给 LLM）
prompt = mem.generate_reflection_prompt('002230')
```

### ⑥ 成本追踪（NEW）

```python
from analytics import CostTracker

tracker = CostTracker()

# 记录一条 LLM 决策
log = tracker.log_decision(
    decision_type='BUY', ticker='002230', ticker_name='科大讯飞',
    llm_model='qwen-plus',
    input_text=context_prompt,
    output_text=llm_response,
    decision_summary='买入500股@72.64',
)

# 收盘后更新盈亏
tracker.update_pnl(log.id, pnl=+1250.0)

# 生成日报
print(tracker.generate_daily_report(pnl_realized=1250.0))

# 查看累计统计
print(tracker.print_cumulative_stats())
```

### ⑦ 组合回测

```bash
# 最优配置：原版22支 + max5-等权
python -c "
from portfolio.engine import PortfolioEngine
from portfolio.config import PortfolioConfig
from strategy.regression_channel.channel_strategy import RegressionChannelWithVolumeStrategy
from stocks import SCREENER_WATCHLIST, SECTOR_MAP

config = PortfolioConfig(
    initial_capital=1_000_000,
    max_positions=5,              # 最优仓位数量
    sizing_method='equal',        # 等权重（对同质化股票池已够用）
)
strategy = RegressionChannelWithVolumeStrategy(window=20, vol_window=20, vol_multiplier=2.0)
engine = PortfolioEngine(config)
result = engine.run(
    universe=[c for c, n in SCREENER_WATCHLIST],
    strategy=strategy,
    start_date='20220101',
    end_date='20260520',
    sector_map=SECTOR_MAP,
)
print(f'总收益: {((result.final_capital/config.initial_capital)-1)*100:.2f}%')
# 预期: +132.92%, 夏普1.07, 最大回撤-15.50%
"
```

### ⑧ 回测可视化

```bash
python visualize.py backtest2 002050    # 三面板回测图
python visualize.py channel 300058      # 通道图
```

### ⑨ 开盘前检查

```bash
python screener/daily.py --all
python screener/daily.py --holdings      # 持仓信号检查
python screener/daily.py --fundamental   # 持仓基本面健康度
python screener/daily.py --candidates    # 昨日候选回顾
```

### ⑩ 批量回测

```bash
python batch_backtest.py              # 龙头股批量回测（市值Top300）
```

### ⑪ 持仓检查

```bash
python check_watchlist.py
```

## 配置说明

所有参数集中在 `config.py`：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `CHANNEL_WINDOW` | 40 | 回归通道计算天数 |
| `VOL_WINDOW` | 20 | 成交量短均线周期 |
| `VOL_MULTIPLIER` | 1.5 | 标准放量倍数 |
| `VOL_ACTIVE_THRESHOLD` | 1.3 | 活跃期判定：20日均量 > 年均量 × 该值 |
| `VOL_SHRINK_RATIO` | 0.8 | 缩量阈值：成交量 < MA20 × 该值 → 缩量 |
| `SELL_DELAY_DAYS` | 2 | 缩量跌破上轨后最长观察天数 |
| `PARTIAL_PROFIT_TARGET` | 0.08 | 部分止盈目标（+8% 卖半仓） |
| `FALSE_BREAKOUT_RETRACE` | 0.03 | 下行期从高点回落阈值 |
| `FUNDAMENTAL_MIN_SCORE` | 4.0 | 基本面最低评分（0-10） |
| `BT_LONG_START` | 20220101 | 长周期回测起始日 |
| `BT_LONG_END` | 20260101 | 长周期回测结束日 |

自选股在 `stocks.py` 中维护：`WATCH_LIST`（持仓跟踪）、`SCREENER_WATCHLIST`（扫描列表）、`SECTOR_MAP`（板块分组）。

## 依赖

- akshare — A 股数据
- pandas / numpy — 数据处理
- matplotlib — 可视化
- tensorflow / scikit-learn — LSTM 策略（可选）
