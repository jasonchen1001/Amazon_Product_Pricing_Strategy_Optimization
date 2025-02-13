import pandas as pd
from data.data_preprocessing import load_data, extract_features
from price_elasticity import PriceElasticityAnalyzer
from sentiment_analysis import SentimentAnalyzer
from pricing_model import DynamicPricingModel
from pricing_optimizer import PricingOptimizer

def main():
    # 1. 数据预处理
    print("[INFO] Loading data...")
    df = load_data('amazon.csv')
    
    # 打印数据集基本信息
    print("\n[INFO] 数据集概况:")
    print(f"总产品数: {len(df)}")
    print(f"有效评论数 > 0 的产品: {(df['rating_count'] > 0).sum()}")
    print(f"有效价格 > 0 的产品: {(df['discounted_price'] > 0).sum()}")
    print("\n价格分布:")
    print(df['discounted_price'].describe())
    
    # 检查是否有重复的产品ID
    duplicates = df['product_id'].duplicated().sum()
    if duplicates > 0:
        print(f"\n[WARNING] 发现 {duplicates} 个重复的产品ID")
        df = df.drop_duplicates(subset='product_id', keep='first')
    
    features = extract_features(df)
    
    # 2. 情感分析
    print("\n[INFO] Starting sentiment analysis...")
    sentiment_analyzer = SentimentAnalyzer()
    df = sentiment_analyzer.analyze_reviews(df)
    print("[INFO] Sentiment analysis completed")
    
    # 3. 价格优化
    print("\n[INFO] Building price optimization model...")
    optimizer = PricingOptimizer()
    recommendations = optimizer.optimize_price(df)
    segments = optimizer.analyze_segments(df)
    
    # 4. 生成报告
    generate_report(df, optimizer)

def generate_report(df, optimizer):
    """生成分析报告"""
    # 获取所有必要数据
    recommendations = optimizer.recommendations
    segments = optimizer.segments
    elasticity = optimizer.elasticity_analyzer.calculate_elasticity(
        df['discounted_price'].values,
        df['rating_count'].values
    )
    
    # 确保弹性值有效
    elasticity_text = f"{elasticity:.2f}" if elasticity is not None else "未知"
    
    # 计算情感分布
    sentiment_stats = {
        'positive': (df['sentiment'] > 0).mean() * 100,
        'neutral': (df['sentiment'] == 0).mean() * 100,
        'negative': (df['sentiment'] < 0).mean() * 100
    }
    
    # 准备TOP5产品数据
    top5_products = recommendations.nlargest(5, 'expected_change')[
        ['product_id', 'current_price', 'recommended_price', 'expected_change']
    ].round(2)
    
    # 生成表格内容
    table_rows = []
    for _, row in top5_products.iterrows():
        table_rows.append(
            f"| {row['product_id']} | {row['current_price']:.2f} | "
            f"{row['recommended_price']:.2f} | {row['expected_change']:.2f} |"
        )
    table_content = '\n'.join(table_rows)
    
    # 生成报告内容
    report = f"""# 印度电商线缆产品定价策略分析报告

## 1. 市场概况 📊
- **产品总数**: {len(df)} 个
- **平均折扣率**: {df['real_discount'].mean():.1f}%
- **平均评分**: {df['rating'].mean():.2f} ⭐
- **价格弹性系数**: {elasticity_text}

## 2. 情感分析 💭
- **正面评价占比**: {sentiment_stats['positive']:.1f}%
- **中性评价占比**: {sentiment_stats['neutral']:.1f}%
- **负面评价占比**: {sentiment_stats['negative']:.1f}%

## 3. 定价模型表现 🎯
- **R² Score**: {optimizer.metrics['r2']:.2%}
- **平均绝对误差**: {optimizer.metrics['mape']:.2%}
- **测试样本数**: {optimizer.metrics['test_size']}

## 4. 价格优化建议 💡
- **建议提价产品数**: {len(recommendations[recommendations['expected_change'] > 0]):,} 个
- **建议降价产品数**: {len(recommendations[recommendations['expected_change'] < 0]):,} 个
- **预期平均利润提升**: {recommendations['expected_change'].mean():.2f}%

## 5. 重点关注产品 TOP5 ⭐
| 产品ID | 当前价格 (₹) | 建议价格 (₹) | 预期变化 (%) |
|--------|-------------|--------------|--------------|
{table_content}

## 6. 策略建议 📈

### 价格弹性分析
{optimizer.elasticity_analyzer.interpret_elasticity()}

### 市场定位建议
1. **高端市场**: 
   - 重点关注产品质量和品牌建设
   - 强调产品差异化
   - 维持较高利润率

2. **中端市场**:
   - 平衡价格和质量
   - 关注竞品定价
   - 保持稳定市场份额

3. **低端市场**:
   - 优化成本结构
   - 提高运营效率
   - 通过规模效应获利

---
*报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    # 保存报告
    with open('report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("报告已生成到 report.md")

if __name__ == "__main__":
    main() 