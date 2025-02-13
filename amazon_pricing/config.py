from pathlib import Path

# 路径配置
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'outputs'
FIGURE_DIR = OUTPUT_DIR / 'figures'
RESULTS_DIR = OUTPUT_DIR / 'results'

# 数据文件
DATA_PATH = DATA_DIR / 'amazon.csv'

# 创建必要的目录
for dir_path in [DATA_DIR, FIGURE_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# 分析参数
ANALYSIS_PARAMS = {
    'price_segments': 5,
    'demand_segments': 5,
    'min_data_points': 10
}

# 模型参数
MODEL_PARAMS = {
    'min_margin': 0.1,
    'max_price_change': 0.2,
    'learning_rate': 0.01,
    'n_estimators': 100
}

# 可视化参数
VIZ_PARAMS = {
    'figure_size': (15, 12),
    'dpi': 300,
    'style': 'seaborn'
} 