import os
import sys
import pandas as pd
import numpy as np

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.models.price_elasticity import PriceElasticityAnalyzer
from src.models.sentiment_analyzer import SentimentAnalyzer

def test_price_elasticity():
    """测试价格敏感度分析"""
    print("\n[INFO] Loading data...")
    
    # 使用已经包含情感分析结果的数据文件
    data_path = os.path.join(project_root, 'data', 'processed', 'amazon_with_sentiment.csv')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Please run sentiment analysis first to generate data file: {data_path}")
    
    # 加载数据
    df = pd.read_csv(data_path)
    print(f"Loaded data: {len(df)} records")
    
    # 打印情感分布
    sentiment_dist = df['sentiment'].value_counts()
    total = len(df)
    
    print("\nSentiment Distribution:")
    for label, count in sentiment_dist.items():
        percentage = count / total * 100
        print(f"{label}: {percentage:.1f}%")
    
    # 价格敏感度分析
    print("\n[INFO] Starting price elasticity analysis...")
    analyzer = PriceElasticityAnalyzer()
    elasticities = analyzer.calculate_price_sensitivity(df)
    
    # 验证结果
    assert isinstance(elasticities, dict), "Return value should be a dictionary"
    assert len(elasticities) > 0, "Should have at least one category elasticity"
    assert all(0 <= e <= 1 for e in elasticities.values()), "Elasticity should be between 0 and 1"
    
    # 验证每个品类都有合理的敏感度
    for category, elasticity in elasticities.items():
        assert isinstance(category, str), f"Category name {category} should be string"
        assert isinstance(elasticity, float), f"Elasticity {elasticity} should be float"
        assert 0.05 <= elasticity <= 0.35, f"Category {category} elasticity {elasticity} out of reasonable range"
    
    # 验证分析器状态
    assert hasattr(analyzer, 'category_elasticities'), "Analyzer should save category elasticities"
    assert hasattr(analyzer, 'elasticity'), "Analyzer should save average elasticity"
    
    # 验证get_elasticity方法
    test_category = list(elasticities.keys())[0]
    assert analyzer.get_elasticity(test_category) == elasticities[test_category], \
        "get_elasticity method return value doesn't match stored value"
    assert analyzer.get_elasticity('non_existent_category') == 0.15, \
        "Should return default value 0.15 for non-existent category"
    
    print("\n=== Test passed ===")

if __name__ == '__main__':
    test_price_elasticity() 