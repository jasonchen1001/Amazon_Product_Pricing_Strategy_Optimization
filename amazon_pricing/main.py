import os
import sys

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from src.data import DataPreprocessor
from src.models import (
    PriceElasticityAnalyzer, 
    PricingOptimizer,
    SentimentAnalyzer
)
from src.analysis import MarketAnalyzer, Dashboard
import config

def main():
    # 1. 加载和预处理数据
    print("[INFO] Loading and preprocessing data...")
    preprocessor = DataPreprocessor()
    df = preprocessor.load_and_preprocess(config.DATA_PATH)
    
    # 2. 情感分析
    print("[INFO] Analyzing sentiments...")
    sentiment_analyzer = SentimentAnalyzer()
    df = sentiment_analyzer.analyze_reviews(df)
    
    # 3. 市场分析
    print("[INFO] Analyzing market...")
    analyzer = MarketAnalyzer(df)
    market_analysis = analyzer.analyze_market()
    
    # 4. 价格优化
    print("[INFO] Optimizing prices...")
    optimizer = PricingOptimizer(df)
    recommendations = optimizer.optimize_prices()
    
    # 5. 生成仪表板
    print("[INFO] Generating dashboard...")
    dashboard = Dashboard(
        df, 
        market_analysis, 
        recommendations,
        sentiment_analyzer.get_sentiment_summary(df)
    )
    dashboard.generate()
    
    print("[INFO] Analysis completed. Check outputs/ directory for results.")

if __name__ == "__main__":
    main() 