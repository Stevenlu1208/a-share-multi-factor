# config.py

# 数据开始时间，要比策略开始时间早，因为要计算过去6个月收益率
DATA_START_DATE = "2013-01-01"

# 策略正式开始回测时间
START_DATE = "2014-01-01"

# 策略结束时间
END_DATE = "2024-12-31"

# 股票池
UNIVERSE_INDEX = ["000300", "000905"]
# 基准指数（净值对比 + 均线择时用）
INDEX_SYMBOL = "sh000300"
MAX_STOCKS = None                       # 全量

# 每个月选择前 N 只股票
TOP_N = 30

# 缓存目录
CACHE_DIR = "data/cache"

# 输出目录
OUTPUT_DIR = "output"

# 因子方向
# 1 表示因子越大越好
# -1 表示因子越小越好
FACTOR_DIRECTION = {
    "pb": -1,
    "roe": 1,
    "gross_margin": 1,
    "revenue_yoy": 1,
    "mcap": -1,       # 规模：越小越好（小市值效应）
    "turnover": -1,   # 换手率：越低越好（低换手异象）
    "mom_1m": -1,
    "vol_1m": -1,
} 

# 不做市值/行业中性化的因子（保留原始截面信号）
NO_NEUTRALIZE_FACTORS = ["mcap"]

# 一只股票至少要有多少个因子有效，否则不打分
MIN_FACTORS = 1

# 中性化开关
NEUTRALIZE_INDUSTRY = True
NEUTRALIZE_MCAP = True

# 因子加权方式
USE_IC_WEIGHTS = True      # False 则退回等权
IC_WEIGHT_WINDOW = 12      # 用过去12个月的IC算权重

# 单边交易成本（含佣金、印花税、滑点），双边即为 0.006
TRADE_COST_SINGLE = 0.003 

# ---------------- 仓位管理 ----------------
USE_POSITION_CONTROL = True
POSITION_METHOD = "ma"        # "ma": N月均线择时 | "drawdown": 回撤分档降仓
POSITION_MA_MONTHS = 10       # 月均线周期
POSITION_LOW = 0.5            # 跌破均线后的仓位
POSITION_DD_STEPS = [(-0.10, 0.5), (-0.20, 0.2)]   # (回撤阈值, 仓位)

# 无风险利率（夏普比率改为超额口径）
RISK_FREE_ANNUAL = 0.02

# 选股风控：单一行业最多持有几只
MAX_PER_INDUSTRY = 6

# 停牌判定：调仓日距离最后交易日超过该天数(自然日)视为停牌
SUSPENSION_GAP_DAYS = 7
# 停牌期间价格向前填充的月数上限（冻结期记0收益，复牌捕捉跳空）
FFILL_LIMIT = 3