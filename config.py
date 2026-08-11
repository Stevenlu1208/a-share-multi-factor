# config.py

# 数据开始时间，要比策略开始时间早，因为要计算过去6个月收益率
DATA_START_DATE = "2013-01-01"

# 策略正式开始回测时间
START_DATE = "2014-01-01"

# 策略结束时间
END_DATE = "2024-12-31"

# 股票池指数，沪深300
UNIVERSE_INDEX = "000300"

# 指数行情代码，沪深300通常用 sh000300
INDEX_SYMBOL = "sh000300"

MAX_STOCKS = None

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
    "pb": -1,            # 价值因子，PB 越低越好
    "roe": 1,            # 质量因子，ROE 越高越好
    "gross_margin": 1,    # 质量因子，毛利率越高越好
    "revenue_yoy": 1,     # 成长因子，营收同比增速越高越好
    "debt_to_asset": -1,  # 风险因子，资产负债率越低越好
    "mom_6m": -1,         # 过去6个月收益率，这里先按反转处理
}

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