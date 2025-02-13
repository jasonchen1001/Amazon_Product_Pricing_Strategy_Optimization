import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.data.preprocessor import DataPreprocessor
from src.models.sentiment_analyzer import SentimentAnalyzer
from src.models.price_elasticity import PriceElasticityAnalyzer
from src.models.pricing_model import DynamicPricingModel
from src.models.pricing_optimizer import PricingOptimizer
import config

def test_pricing_optimization():
    """测试价格优化效果"""
    # 1. 数据准备
    print("\n[INFO] 加载和预处理数据...")
    preprocessor = DataPreprocessor()
    df = preprocessor.load_and_preprocess(config.DATA_PATH)
    
    # 2. 情感分析
    print("\n[INFO] 进行情感分析...")
    sentiment_analyzer = SentimentAnalyzer()
    df = sentiment_analyzer.analyze_reviews(df)
    
    # 3. 价格敏感度分析
    print("\n[INFO] 分析价格敏感度...")
    elasticity_analyzer = PriceElasticityAnalyzer()
    sensitivity_metrics = elasticity_analyzer.calculate_price_sensitivity(df)
    
    # 4. 训练定价模型
    print("\n[INFO] 训练定价模型...")
    pricing_model = DynamicPricingModel()
    model_metrics = pricing_model.train(df)
    
    # 5. 创建价格优化器
    optimizer = PricingOptimizer(pricing_model, elasticity_analyzer)
    
    # 6. 按类别分析价格优化
    print("\n=== 各类别价格优化分析 ===")
    category_results = {}
    
    for category in df['main_category'].unique():
        category_df = df[df['main_category'] == category]
        if len(category_df) < 10:
            continue
            
        print(f"\n类别: {category}")
        print(f"产品数量: {len(category_df)}")
        
        # 计算类别统计
        stats = {
            '当前均价': category_df['discounted_price'].mean(),
            '平均折扣': category_df['real_discount'].mean(),
            '平均评分': category_df['rating'].mean(),
            '平均评论数': category_df['rating_count'].mean(),
            '平均情感得分': category_df['sentiment'].mean()
        }
        
        # 优化每个产品的价格
        optimized_prices = []
        price_changes = []
        sentiment_impacts = []
        
        for idx, product in category_df.iterrows():
            product_df = pd.DataFrame([product])
            result = optimizer.optimize_price(product_df)
            
            current_price = result['current_price']
            optimal_price = result['optimal_price']
            price_change = result['price_change']
            
            optimized_prices.append(optimal_price)
            price_changes.append(price_change)
            
            # 预测情感影响
            sentiment_impact = _predict_sentiment_impact(
                price_change, 
                product['sentiment'],
                elasticity_analyzer.elasticity
            )
            sentiment_impacts.append(sentiment_impact)
        
        # 保存类别结果
        category_results[category] = {
            'stats': stats,
            'optimized_prices': optimized_prices,
            'price_changes': price_changes,
            'sentiment_impacts': sentiment_impacts
        }
        
        # 打印类别分析结果
        print("\n价格优化结果:")
        print(f"当前均价: ₹{stats['当前均价']:.2f}")
        print(f"平均折扣: {stats['平均折扣']:.1f}%")
        print(f"平均调价幅度: {np.mean(price_changes):.1f}%")
        print(f"最大上调: {max(price_changes):.1f}%")
        print(f"最大下调: {min(price_changes):.1f}%")
        print(f"预计情感变化: {np.mean(sentiment_impacts):.3f}")
    
    # 7. 生成可视化
    _generate_optimization_visualizations(category_results)
    
    print("\n[INFO] 分析完成，可视化结果已保存到 outputs/figures/")

def _predict_sentiment_impact(price_change, current_sentiment, elasticity):
    """预测价格变化对情感的影响"""
    # 简单的线性影响模型
    impact = -price_change * elasticity * 0.01
    new_sentiment = current_sentiment + impact
    return max(min(new_sentiment, 1), -1)  # 确保在[-1, 1]范围内

def _generate_optimization_visualizations(results):
    """生成优化结果可视化"""
    # 1. 价格调整分布图
    plt.figure(figsize=(12, 6))
    
    price_changes = []
    categories = []
    for category, data in results.items():
        price_changes.extend(data['price_changes'])
        categories.extend([category] * len(data['price_changes']))
    
    plt.subplot(1, 2, 1)
    sns.boxplot(x=categories, y=price_changes)
    plt.xticks(rotation=45)
    plt.xlabel('产品类别')
    plt.ylabel('价格调整幅度 (%)')
    plt.title('各类别价格调整分布')
    
    # 2. 情感影响分析
    plt.subplot(1, 2, 2)
    sentiment_changes = []
    for category, data in results.items():
        current_sentiment = data['stats']['平均情感得分']
        predicted_sentiment = np.mean(data['sentiment_impacts'])
        sentiment_changes.append({
            'category': category,
            'current': current_sentiment,
            'predicted': predicted_sentiment
        })
    
    sentiment_df = pd.DataFrame(sentiment_changes)
    sentiment_df.plot(x='category', y=['current', 'predicted'], 
                     kind='bar', width=0.8)
    plt.xticks(rotation=45)
    plt.xlabel('产品类别')
    plt.ylabel('情感得分')
    plt.title('价格调整对情感的影响')
    plt.legend(['当前情感', '预测情感'])
    
    plt.tight_layout()
    plt.savefig('outputs/figures/pricing_optimization_analysis.png')
    plt.close()
    
    # 3. 价格-情感关系图
    plt.figure(figsize=(10, 6))
    all_price_changes = []
    all_sentiment_impacts = []
    all_categories = []
    
    for category, data in results.items():
        all_price_changes.extend(data['price_changes'])
        all_sentiment_impacts.extend(data['sentiment_impacts'])
        all_categories.extend([category] * len(data['price_changes']))
    
    plt.scatter(all_price_changes, all_sentiment_impacts, 
               c=[hash(c) % 256 for c in all_categories], 
               alpha=0.6)
    
    plt.xlabel('价格调整幅度 (%)')
    plt.ylabel('预测情感得分')
    plt.title('价格调整与情感变化关系')
    plt.grid(True)
    plt.savefig('outputs/figures/price_sentiment_relationship.png')
    plt.close()

if __name__ == "__main__":
    test_pricing_optimization() 