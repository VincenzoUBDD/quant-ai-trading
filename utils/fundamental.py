# 基本面分析模块：数据获取 + 评分过滤
import os, csv, time, pickle, re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import pandas as pd
import akshare as ak

# ===== 评分权重 =====
DEFAULT_WEIGHTS = {
    'roe': 0.25,
    'net_profit_growth': 0.20,
    'revenue_growth': 0.15,
    'pe': 0.10,
    'gross_margin': 0.10,
    'debt_ratio': 0.10,
    'pledge_ratio': 0.10,
}

# ===== 评分阈值（每项 0-10 分） =====
SCORE_BREAKPOINTS = {
    'roe':          [(25, 10), (20, 8), (15, 6), (10, 4), (5, 2)],
    'revenue_growth': [(50, 10), (30, 8), (20, 6), (10, 4), (0, 2)],
    'net_profit_growth': [(50, 10), (30, 8), (20, 6), (10, 4), (0, 2)],
    'gross_margin': [(60, 10), (40, 8), (30, 6), (20, 4), (10, 2)],
    'debt_ratio':   [(20, 10), (40, 8), (60, 6), (70, 4), (85, 2)],
    'pledge_ratio': [(5, 10), (10, 8), (20, 6), (30, 4), (50, 2)],
}

# ===== akshare 字段映射 =====
COLUMN_MAPPING = {
    '净资产收益率': 'roe',
    '营业收入增长率': 'revenue_growth',
    '净利润增长率': 'net_profit_growth',
    '销售毛利率': 'gross_margin',
    '资产负债率': 'debt_ratio',
    '加权每股收益': 'eps',
}
# 部分版本 akshare 字段名可能不同，补充映射
COLUMN_MAPPING_ALT = {
    '净资产收益率(%)': 'roe',
    '营业收入增长率(%)': 'revenue_growth',
    '主营业务收入增长率(%)': 'revenue_growth',
    '净利润增长率(%)': 'net_profit_growth',
    '销售毛利率(%)': 'gross_margin',
    '资产负债率(%)': 'debt_ratio',
    '加权每股收益(元)': 'eps',
    '基本每股收益(元)': 'eps',
}

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'data', 'fundamental_cache')


@dataclass
class FundamentalData:
    """单只股票基本面数据"""
    symbol: str = ''
    name: str = ''
    roe: Optional[float] = None
    pe: Optional[float] = None
    revenue_growth: Optional[float] = None
    net_profit_growth: Optional[float] = None
    gross_margin: Optional[float] = None
    debt_ratio: Optional[float] = None
    pledge_ratio: Optional[float] = None
    eps: Optional[float] = None
    data_date: str = ''
    is_annual: bool = False
    # 基金持仓
    fund_count: Optional[int] = None      # 最新季度持有该股的基金家数
    fund_ratio: Optional[float] = None    # 最新季度基金持仓占流通股比例(%)
    fund_ratio_change: Optional[float] = None  # 持仓比例较上季度变化(百分点)


@dataclass
class ScoreResult:
    """评分结果"""
    total_score: float = 0.0
    grade: str = 'N/A'
    details: Dict[str, float] = field(default_factory=dict)
    raw_values: Dict[str, float] = field(default_factory=dict)


GRADE_MAP = [
    (8.0, '优秀'), (6.0, '良好'), (4.0, '一般'), (2.0, '较差'), (-1, '很差'),
]


def _score_metric(metric: str, value: float) -> float:
    """对单个指标评分 0-10"""
    if value is None or pd.isna(value):
        return 0
    # 负 PE 直接 0
    if metric == 'pe':
        if value <= 0:
            return 0
        if value <= 15: return 10
        if value <= 30: return 8
        if value <= 50: return 6
        if value <= 80: return 4
        return 2
    breakpoints = SCORE_BREAKPOINTS.get(metric, [])
    for threshold, score in breakpoints:
        if metric in ('debt_ratio', 'pledge_ratio'):
            if value <= threshold:
                return score
        else:
            if value >= threshold:
                return score
    return 0


def _get_grade(score: float) -> str:
    for threshold, grade in GRADE_MAP:
        if score >= threshold:
            return grade
    return '很差'


class FundamentalAnalyzer:
    """基本面分析器：数据获取 + 评分"""

    def __init__(self, min_score: float = 4.0):
        self.min_score = min_score
        self._pledge_map: Optional[Dict[str, float]] = None
        self._weights = DEFAULT_WEIGHTS.copy()
        self._institutional_cache: Optional[Dict[str, Dict[str, float]]] = None
        # {symbol: {'count': float, 'ratio': float, 'count_change': float, 'ratio_change': float}}

    # ---- 基金持仓数据 ----

    FUND_HOLD_CACHE_FILE = os.path.join(CACHE_DIR, 'fund_holdings.csv')

    def _fetch_fund_holdings(self, symbol: str) -> Tuple[Optional[int], Optional[float], Optional[float]]:
        """
        获取基金持仓数据及季度变化。

        Returns:
            (fund_count, fund_ratio, ratio_change_pct)
            ratio_change_pct: 持仓比例较上季度变化（百分点）
        """
        # 先检查磁盘缓存
        cache = self._load_fund_cache()
        if symbol in cache:
            cached_ts, f_cnt, f_ratio, f_change = cache[symbol]
            age_hours = (time.time() - cached_ts) / 3600
            if age_hours < 24:  # 24小时内有效
                return f_cnt, f_ratio, f_change

        try:
            df = ak.stock_fund_stock_holder(symbol=symbol)
            if df is None or df.empty:
                return None, None, None

            dates = sorted(df['截止日期'].unique())
            if len(dates) >= 2:
                latest = dates[-1]
                prev = dates[-2]
                latest_data = df[df['截止日期'] == latest]
                prev_data = df[df['截止日期'] == prev]

                f_cnt = len(latest_data)
                f_ratio = float(latest_data['占流通股比例'].sum())
                prev_ratio = float(prev_data['占流通股比例'].sum())
                f_change = f_ratio - prev_ratio
            elif len(dates) == 1:
                f_cnt = len(df)
                f_ratio = float(df['占流通股比例'].sum())
                f_change = None
            else:
                return None, None, None

            # 写入缓存
            cache[symbol] = (time.time(), f_cnt, f_ratio, f_change)
            self._save_fund_cache(cache)
            return f_cnt, f_ratio, f_change
        except Exception:
            return None, None, None

    def _load_fund_cache(self) -> dict:
        """加载基金持仓磁盘缓存"""
        if not os.path.exists(self.FUND_HOLD_CACHE_FILE):
            return {}
        try:
            df = pd.read_csv(self.FUND_HOLD_CACHE_FILE, dtype={'symbol': str})
            cache = {}
            for _, r in df.iterrows():
                cache[r['symbol']] = (r['timestamp'], r['fund_count'], r['fund_ratio'], r['ratio_change'])
            return cache
        except Exception:
            return {}

    def _save_fund_cache(self, cache: dict):
        """保存基金持仓磁盘缓存"""
        os.makedirs(CACHE_DIR, exist_ok=True)
        rows = []
        for sym, (ts, cnt, ratio, chg) in cache.items():
            rows.append([sym, ts, cnt, ratio, chg])
        pd.DataFrame(rows, columns=['symbol', 'timestamp', 'fund_count', 'fund_ratio', 'ratio_change']
                     ).to_csv(self.FUND_HOLD_CACHE_FILE, index=False, encoding='utf-8-sig')

    # ---- 数据获取 ----

    def fetch_fundamentals(self, symbol: str, name: str = '',
                           year: Optional[int] = None,
                           price: Optional[float] = None) -> Optional[FundamentalData]:
        """
        获取单只股票基本面数据

        Args:
            symbol: 股票代码
            name: 股票名称
            year: 指定年份年报（回测规避未来函数）；None 则取最新季度
            price: 当前价格（用于计算 PE），不传则跳过 PE
        """
        try:
            fund = FundamentalData(symbol=symbol, name=name)

            # ---- 财务指标 ----
            start_year = str(year if year else datetime.now().year - 3)
            df = ak.stock_financial_analysis_indicator(symbol=symbol, start_year=start_year)
            if df is None or df.empty:
                return None

            # 合并字段映射
            col_map = {}
            for cn, en in COLUMN_MAPPING.items():
                if cn in df.columns:
                    col_map[cn] = en
            for cn, en in COLUMN_MAPPING_ALT.items():
                if cn in df.columns:
                    col_map[cn] = en

            # 统一日期列为字符串
            dates = df['日期'].astype(str)
            target_row = None
            if year:
                target_date = f'{year}-12-31'
                mask = dates == target_date
                if not mask.any():
                    # 找该年最新一行
                    yr_mask = dates.str.startswith(str(year))
                    if yr_mask.any():
                        idx = dates[yr_mask].index[-1]
                        target_row = df.loc[[idx]]
                else:
                    target_row = df[mask]
            else:
                target_row = df.iloc[[0]]  # 最新季度

            if target_row is None or target_row.empty:
                return None

            row = target_row.iloc[0]

            # 解析指标
            for cn, en in col_map.items():
                val = row.get(cn)
                if val is not None:
                    try:
                        setattr(fund, en, float(val))
                    except (ValueError, TypeError):
                        pass

            fund.data_date = str(row.get('日期', ''))
            fund.is_annual = year is not None and '12-31' in fund.data_date

            # ---- PE = price / eps ----
            if price and fund.eps and fund.eps > 0:
                fund.pe = price / fund.eps

            # ---- 质押比例 ----
            pledge = self._get_pledge(symbol)
            if pledge is not None:
                fund.pledge_ratio = pledge

            # ---- 基金持仓 ----
            if year is None:  # 仅在非回测模式下获取（回测没有历史数据）
                f_cnt, f_ratio, f_change = self._fetch_fund_holdings(symbol)
                fund.fund_count = f_cnt
                fund.fund_ratio = f_ratio
                fund.fund_ratio_change = f_change

            return fund

        except Exception as e:
            return None

    def _get_pledge(self, symbol: str) -> Optional[float]:
        """获取单只股票质押比例（从缓存）"""
        if self._pledge_map is None:
            self._load_pledge_data()
        if self._pledge_map and symbol in self._pledge_map:
            return self._pledge_map[symbol]
        return None

    def _load_pledge_data(self):
        """加载全市场质押数据"""
        cache_path = os.path.join(CACHE_DIR, 'pledge_ratios.csv')
        if os.path.exists(cache_path):
            try:
                mtime = os.path.getmtime(cache_path)
                age_hours = (time.time() - mtime) / 3600
                if age_hours < 24:
                    df = pd.read_csv(cache_path, dtype={'股票代码': str})
                    self._pledge_map = dict(zip(df['股票代码'], df['质押比例']))
                    return
            except Exception:
                pass

        try:
            df = ak.stock_gpzy_pledge_ratio_em()
            if df is not None and not df.empty:
                self._pledge_map = dict(zip(df['股票代码'].astype(str), df['质押比例'].values))
                os.makedirs(CACHE_DIR, exist_ok=True)
                df.to_csv(cache_path, index=False, encoding='utf-8-sig')
        except Exception:
            self._pledge_map = {}

    # ---- 批量获取 ----

    def fetch_batch(self, stocks: List[Tuple[str, str]],
                    year: Optional[int] = None,
                    use_cache: bool = True,
                    force_refresh: bool = False) -> Dict[str, FundamentalData]:
        """
        批量获取基本面数据（带磁盘缓存）

        Args:
            stocks: [(code, name), ...]
            year: 年份
            use_cache: 是否使用缓存
            force_refresh: 强制刷新缓存
        Returns:
            {symbol: FundamentalData}
        """
        result: Dict[str, FundamentalData] = {}

        if use_cache and not force_refresh:
            result = self._load_cache(year)

        cached_symbols = set(result.keys())
        all_symbols = {s for s, _ in stocks}
        missing = all_symbols - cached_symbols

        if not missing:
            return result

        # 需要获取缺失的
        for i, (symbol, name) in enumerate(stocks):
            if symbol in cached_symbols:
                continue
            fund = self.fetch_fundamentals(symbol, name, year=year)
            if fund:
                result[symbol] = fund

        if use_cache:
            self._save_cache(result, year)

        return result

    def _cache_path(self, year: Optional[int]) -> str:
        suffix = str(year) if year else 'latest'
        return os.path.join(CACHE_DIR, f'fundamental_{suffix}.csv')

    def _load_cache(self, year: Optional[int]) -> Dict[str, FundamentalData]:
        path = self._cache_path(year)
        if not os.path.exists(path):
            return {}
        try:
            df = pd.read_csv(path, dtype={'symbol': str})
            result = {}
            for _, row in df.iterrows():
                d = FundamentalData(
                    symbol=str(row['symbol']),
                    name=str(row.get('name', '')),
                    roe=_safe_float(row, 'roe'),
                    pe=_safe_float(row, 'pe'),
                    revenue_growth=_safe_float(row, 'revenue_growth'),
                    net_profit_growth=_safe_float(row, 'net_profit_growth'),
                    gross_margin=_safe_float(row, 'gross_margin'),
                    debt_ratio=_safe_float(row, 'debt_ratio'),
                    pledge_ratio=_safe_float(row, 'pledge_ratio'),
                    eps=_safe_float(row, 'eps'),
                    data_date=str(row.get('data_date', '')),
                    fund_count=_safe_int(row, 'fund_count'),
                    fund_ratio=_safe_float(row, 'fund_ratio'),
                    fund_ratio_change=_safe_float(row, 'fund_ratio_change'),
                )
                result[d.symbol] = d
            return result
        except Exception:
            return {}

    def _save_cache(self, data: Dict[str, FundamentalData], year: Optional[int]):
        path = self._cache_path(year)
        os.makedirs(CACHE_DIR, exist_ok=True)
        rows = []
        for d in data.values():
            rows.append([d.symbol, d.name, d.roe, d.pe, d.revenue_growth,
                         d.net_profit_growth, d.gross_margin, d.debt_ratio,
                         d.pledge_ratio, d.eps, d.data_date,
                         d.fund_count, d.fund_ratio, d.fund_ratio_change])
        columns = ['symbol', 'name', 'roe', 'pe', 'revenue_growth',
                   'net_profit_growth', 'gross_margin', 'debt_ratio',
                   'pledge_ratio', 'eps', 'data_date',
                   'fund_count', 'fund_ratio', 'fund_ratio_change']
        pd.DataFrame(rows, columns=columns).to_csv(path, index=False, encoding='utf-8-sig')

    # ---- 机构持仓历史数据（回测用） ----

    INST_CACHE_FILE = os.path.join(CACHE_DIR, 'institutional_holdings.pkl')

    def prefetch_institutional_holdings(self, start_year: int = 2018, end_year: int = 2025,
                                        use_cache: bool = True, force_refresh: bool = False):
        """
        预取机构持仓历史数据，构建 {symbol: {quarter: data}} 缓存，用于回测。

        数据来源: ak.stock_institute_hold，按季度覆盖全市场。

        Args:
            start_year: 起始年份（含）
            end_year: 结束年份（含）
        """
        if use_cache and not force_refresh and os.path.exists(self.INST_CACHE_FILE):
            import pickle
            with open(self.INST_CACHE_FILE, 'rb') as f:
                self._institutional_cache = pickle.load(f)
            return

        from collections import defaultdict
        cache: Dict[str, Dict[str, float]] = defaultdict(dict)

        for year in range(start_year, end_year + 1):
            for quarter in [1, 2, 3, 4]:
                qcode = f'{year}{quarter}'
                try:
                    df = ak.stock_institute_hold(symbol=qcode)
                    if df is None or df.empty:
                        continue
                    cols = list(df.columns)
                    # columns: 证券代码, 证券名称, 家数, 家数变化, 持股比例, 持股比例变化, ...
                    for _, row in df.iterrows():
                        code = str(row.iloc[0]).zfill(6)
                        cache[code][qcode] = {
                            'count': _safe_float(row, df.columns[2]),
                            'count_change': _safe_float(row, df.columns[3]),
                            'ratio': _safe_float(row, df.columns[4]),
                            'ratio_change': _safe_float(row, df.columns[5]),
                        }
                except Exception:
                    continue
                if (year * 10 + quarter) % 2 == 0:
                    print(f'  获取机构持仓: {year}Q{quarter}')

        self._institutional_cache = dict(cache)
        if use_cache:
            import pickle
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(self.INST_CACHE_FILE, 'wb') as f:
                pickle.dump(self._institutional_cache, f)
        print(f'机构持仓缓存: {len(self._institutional_cache)} 支股票, {start_year}-{end_year}')

    def _get_institutional_bonus(self, symbol: str, as_of_date) -> float:
        """
        根据信号日期的机构持仓变化计算加分。

        比较信号日所在最新季度与上季度的机构家数和持仓比例变化。
        Returns: 0 ~ 2.0 的加分值
        """
        if self._institutional_cache is None:
            return 0.0

        stock_data = self._institutional_cache.get(symbol, {})
        if not stock_data:
            return 0.0

        # 确定信号日期对应的最新完整季度 + 往前找最近的可用数据
        dt = pd.to_datetime(as_of_date)
        y, m = dt.year, dt.month

        # 从信号日期对应的季度往前找最近的3个季度
        if m <= 3:
            candidates = [f'{y}1', f'{y-1}4', f'{y-1}3']
        elif m <= 6:
            candidates = [f'{y}2', f'{y}1', f'{y-1}4']
        elif m <= 9:
            candidates = [f'{y}3', f'{y}2', f'{y}1']
        else:
            candidates = [f'{y}4', f'{y}3', f'{y}2']

        latest = None
        prev = None
        for q in candidates:
            if q in stock_data:
                if latest is None:
                    latest = stock_data[q]
                    latest_q = q
                else:
                    prev = stock_data[q]
                    break

        if latest is None:
            return 0.0
        if prev is None:
            return 0.0

        l_ratio = latest.get('ratio')
        p_ratio = prev.get('ratio')
        l_count = latest.get('count')
        p_count = prev.get('count')

        if l_ratio is None or p_ratio is None:
            return 0.0

        ratio_change = l_ratio - p_ratio

        # 加分逻辑
        if ratio_change > 5:
            return 2.0
        elif ratio_change > 2:
            return 1.5
        elif ratio_change > 0:
            return 1.0
        return 0.0

    def compute_score(self, data: FundamentalData) -> ScoreResult:
        """计算综合基本面评分（含基金持仓加分）"""
        raw = {
            'roe': data.roe,
            'revenue_growth': data.revenue_growth,
            'net_profit_growth': data.net_profit_growth,
            'pe': data.pe,
            'gross_margin': data.gross_margin,
            'debt_ratio': data.debt_ratio,
            'pledge_ratio': data.pledge_ratio,
        }
        scores = {}
        active_metrics = {}
        for k, v in raw.items():
            if v is not None and not pd.isna(v) and v != 0:
                scores[k] = _score_metric(k, v)
                active_metrics[k] = v
            else:
                scores[k] = 0

        # 缺失指标权重重新分配
        total_weight = sum(self._weights.get(k, 0) for k in scores)
        weighted = 0.0
        for k, s in scores.items():
            w = self._weights.get(k, 0) / total_weight if total_weight > 0 else 0
            weighted += s * w

        # 基金持仓加分（仅最新数据，非回测模式）
        fund_bonus = 0.0
        if data.fund_ratio_change is not None:
            if data.fund_ratio_change > 5:
                fund_bonus = 2.0
            elif data.fund_ratio_change > 2:
                fund_bonus = 1.5
            elif data.fund_ratio_change > 0:
                fund_bonus = 1.0

        total = min(weighted + fund_bonus, 10)
        grade = _get_grade(total)
        return ScoreResult(
            total_score=round(total, 2),
            grade=grade,
            details={k: scores[k] for k in scores},
            raw_values={**active_metrics, '_fund_bonus': fund_bonus},
        )

    def passes(self, data: FundamentalData) -> Tuple[bool, ScoreResult]:
        """是否通过基本面过滤"""
        score = self.compute_score(data)
        return score.total_score >= self.min_score, score

    # ========== 新增：基于 stock_financial_abstract（东方财富）的数据获取 ==========

    def prefetch_abstract(self, stocks: List[Tuple[str, str]],
                          use_cache: bool = True,
                          force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
        """
        预取全量财务摘要数据，使用 ak.stock_financial_abstract（覆盖率 ~100%）

        Args:
            stocks: [(code, name), ...]
            use_cache: 是否使用磁盘缓存
            force_refresh: 强制刷新缓存
        Returns:
            {symbol: DataFrame}  — 原始 DataFrame 来自 akshare
        """
        cache_file = os.path.join(CACHE_DIR, 'abstract_raw.pkl')

        if use_cache and not force_refresh and os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                cached = pickle.load(f)
            # 校验缓存是否包含所有需要的股票
            if isinstance(cached, dict):
                cached_symbols = set(cached.keys())
                needed_symbols = {s for s, _ in stocks}
                if cached_symbols.issuperset(needed_symbols):
                    return cached
                # 只取不存在的补取
                result = dict(cached)
            else:
                result = {}
        else:
            result = {}
        total = len(stocks)
        for i, (code, name) in enumerate(stocks):
            if code in result:
                continue  # 已有缓存，跳过
            try:
                df = ak.stock_financial_abstract(symbol=code)
                if df is not None and not df.empty:
                    result[code] = df
            except Exception:
                pass
            if (i + 1) % 10 == 0:
                print(f"  预取基本面: {i+1}/{total}")
            time.sleep(0.3)

        if use_cache and result:
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)

        return result

    def _extract_fund_from_abstract(self, df: pd.DataFrame,
                                    symbol: str, name: str,
                                    year: int) -> Optional[FundamentalData]:
        """
        从 stock_financial_abstract 的 DataFrame 中提取指定年份的年报数据。

        DataFrame 列: ['选项', '指标', '20251231', '20250930', '20241231', ...]
        """
        metric_col = df.columns[1]  # '指标'

        # 找目标年份的 12-31 列，如果缺则用该年最后一期
        target_col = f'{year}1231'
        if target_col not in df.columns:
            year_cols = [c for c in df.columns[2:] if str(c).startswith(str(year))]
            if not year_cols:
                return None
            target_col = year_cols[-1]

        def _get(metric_name: str) -> Optional[float]:
            mask = df[metric_col].apply(lambda x: metric_name in str(x) if pd.notna(x) else False)
            if not mask.any():
                return None
            row = df[mask].iloc[0]
            val = row.get(target_col)
            if pd.notna(val):
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return None
            return None

        roe = _get('净资产收益率(ROE)')
        rev_growth = _get('营业总收入增长率')
        profit_growth = _get('归属母公司净利润增长率')
        gross_margin = _get('毛利率')
        debt_ratio = _get('资产负债率')
        eps = _get('基本每股收益')

        # 至少有一项有效才返回
        if all(v is None for v in [roe, rev_growth, profit_growth, gross_margin, debt_ratio]):
            return None

        # 数据日期
        data_date = f'{year}-12-31'
        if target_col != f'{year}1231':
            data_date = f'{target_col[:4]}-{target_col[4:6]}-{target_col[6:8]}'

        return FundamentalData(
            symbol=symbol, name=name,
            roe=roe, pe=None,
            revenue_growth=rev_growth, net_profit_growth=profit_growth,
            gross_margin=gross_margin, debt_ratio=debt_ratio,
            pledge_ratio=None, eps=eps,
            data_date=data_date, is_annual=target_col.endswith('1231'),
        )

    def build_multi_year_cache(self, abstract_cache: Dict[str, pd.DataFrame],
                               start_year: int = 2018,
                               end_year: int = 2025) -> Dict[str, Dict[int, FundamentalData]]:
        """
        从原始 abstract 数据构建 {symbol: {year: FundamentalData}} 多年代缓存。

        结果可传递给 check_signal() 供回测信号时使用。
        """
        result: Dict[str, Dict[int, FundamentalData]] = {}
        for symbol, df in abstract_cache.items():
            for year in range(start_year, end_year + 1):
                fund = self._extract_fund_from_abstract(df, symbol, '', year)
                if fund is not None:
                    if symbol not in result:
                        result[symbol] = {}
                    result[symbol][year] = fund
        return result

    @staticmethod
    def get_year_for_date(as_of_date) -> int:
        """
        根据信号日期确定应使用哪一年的年报数据。
        年报通常在次年 4 月底前发布，保守起见 6 月前仍用前两年数据。
        """
        dt = pd.to_datetime(as_of_date)
        if dt.month >= 6:
            return dt.year - 1
        else:
            return dt.year - 2

    def check_signal(self, multi_year_cache: Dict[str, Dict[int, FundamentalData]],
                     symbol: str, as_of_date, price: float = None) -> Tuple[bool, Optional[float], Optional[str], str]:
        """
        在买入信号触发时检查基本面（含机构持仓变化加分）。

        Args:
            multi_year_cache: build_multi_year_cache() 的输出
            symbol: 股票代码
            as_of_date: 信号日期
            price: 当前价格（用于计算 PE），可选

        Returns:
            (passes, score, grade, reason)
            - passes: True=通过可买, False=不通过
            - score: 评分或 None
            - grade: 等级或 None
            - reason: 描述
        """
        year = self.get_year_for_date(as_of_date)

        stock_data = multi_year_cache.get(symbol, {})
        fund = stock_data.get(year)

        # 如果当年数据缺失，降级取前一年
        if fund is None:
            fund = stock_data.get(year - 1)
        if fund is None:
            fund = stock_data.get(year - 2)

        if fund is None:
            # 无数据 → 放行（不阻塞交易）
            return (True, None, None, '无基本面数据，放行')

        # 如果有价格和 EPS，实时计算 PE
        if price is not None and fund.eps is not None and fund.eps > 0:
            fund.pe = price / fund.eps

        # 回测模式：从机构持仓缓存获取季度变化加分
        inst_bonus = 0.0
        if self._institutional_cache is not None:
            inst_bonus = self._get_institutional_bonus(symbol, as_of_date)
            if inst_bonus > 0:
                # 把机构加分注入 FundamentalData，让 compute_score 能读到
                fund.fund_ratio_change = inst_bonus * 2  # 映射到加分阈值

        result = self.compute_score(fund)

        # 回测模式加分修正（compute_score 用 fund_ratio_change 打分，但回测时用的是机构数据）
        if inst_bonus > 0 and '_fund_bonus' in result.raw_values:
            # 改用机构持仓的实际加分
            result.total_score = min(result.total_score - result.raw_values['_fund_bonus'] + inst_bonus, 10)
            result.raw_values['_fund_bonus'] = inst_bonus

        passes = result.total_score >= self.min_score
        if passes:
            desc = f'基本面 {result.total_score:.1f}分({result.grade})，通过'
        else:
            desc = f'基本面 {result.total_score:.1f}分({result.grade})，低于阈值 {self.min_score}，跳过'
        return (passes, result.total_score, result.grade, desc)


def _safe_float(row, col):
    try:
        v = row[col]
        if pd.isna(v):
            return None
        return float(v)
    except (KeyError, ValueError, TypeError):
        return None


def _safe_int(row, col):
    try:
        v = row[col]
        if pd.isna(v):
            return None
        return int(v)
    except (KeyError, ValueError, TypeError):
        return None


def print_report(data: FundamentalData, score: ScoreResult):
    """打印基本面分析报告"""
    print(f'  基本面分析: {data.symbol} {data.name}')
    print(f'  评分: {score.total_score:.1f} / 10  [{score.grade}]')
    print('  ' + '-' * 50)

    labels = {
        'roe': 'ROE', 'pe': 'PE', 'revenue_growth': '营收增长率',
        'net_profit_growth': '净利增长率', 'gross_margin': '毛利率',
        'debt_ratio': '资产负债率', 'pledge_ratio': '质押比例',
    }
    for k, v in score.raw_values.items():
        if k == '_fund_bonus':
            continue
        s = score.details.get(k, 0)
        w = DEFAULT_WEIGHTS.get(k, 0)
        label = labels.get(k, k)
        if k == 'pe':
            print(f'  {label}: {"N/A" if v is None else f"{v:.1f}"}      {"":>8} → {s:.0f}分 (w={w:.2f})')
        else:
            print(f'  {label}: {v:>7.1f}%   → {s:.0f}分 (w={w:.2f})')

    # 基金持仓
    if data.fund_count is not None:
        chg_str = f'({data.fund_ratio_change:+.1f}pp)' if data.fund_ratio_change is not None else ''
        print(f'  基金持仓: {data.fund_count}家  占比{data.fund_ratio:.2f}%{chg_str}')
        bonus = score.raw_values.get('_fund_bonus', 0)
        if bonus > 0:
            print(f'           -> 基金持仓增加 +{bonus:.0f}分加分')

    if data.data_date:
        print(f'  数据基准日: {data.data_date}')
    print()
