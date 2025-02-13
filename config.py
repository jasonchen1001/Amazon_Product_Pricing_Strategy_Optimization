"""模型配置文件"""

# XGBoost模型参数
XGBOOST_PARAMS = {
    'n_estimators': 80,
    'learning_rate': 0.1,
    'max_depth': 3,
    'min_child_weight': 10,
    'subsample': 0.6,
    'colsample_bytree': 0.6,
    'reg_alpha': 1.0,
    'reg_lambda': 2.0,
    'random_state': 42,
    'objective': 'reg:squarederror'
}

# 特征工程参数
FEATURE_ENGINEERING_PARAMS = {
    'price_segments': 3,          # 减少分段避免过度细分
    'popularity_segments': 3,     
    'use_log_transform': True,
    'add_polynomial_features': False,  # 暂时关闭多项式特征
    'standardize_features': True,
    'polynomial_degree': 2,        # 保留参数但暂时不使用
    'interaction_only': True       # 保留参数但暂时不使用
}

# 模型评估参数
EVALUATION_PARAMS = {
    'test_size': 0.2,           # 保持20%测试集
    'cv_folds': 5,              # 保持5折交叉验证
    'early_stopping_rounds': 20,
    'stratify_by': 'price_segment'
}

# 价格优化参数
OPTIMIZATION_PARAMS = {
    'max_price_change': {
        'increase': 0.15,     # 最大涨价15%
        'decrease': 0.20      # 最大降价20%
    },
    'price_steps': 50,        # 价格测试点数量
    'cost_ratio': 0.70,       # 成本比例
    'min_profit_margin': 0.10,  # 最小利润率
    'min_profit_improvement': 0.02  # 最小利润改进
}

# 特征重要性阈值
FEATURE_IMPORTANCE_THRESHOLD = 0.01

# 模型性能目标
PERFORMANCE_TARGETS = {
    'min_r2': 0.75,
    'max_mape': 0.15,
    'min_profit_increase': 0.05
} 