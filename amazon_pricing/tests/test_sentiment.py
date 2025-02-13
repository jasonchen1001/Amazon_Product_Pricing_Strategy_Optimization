import os
import sys
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.data.preprocessor import DataPreprocessor
from src.models.sentiment_analyzer import SentimentAnalyzer
import config

def generate_wordcloud(text, analyzer, title, output_path, sentiment_type='positive'):
    """生成词云图"""
    # 1. 分词和预处理
    words = []
    for review in text:
        if not isinstance(review, str):
            continue
        
        # 使用预处理器清理文本
        cleaned_text = analyzer.preprocess_text(review)
        # 分词
        review_words = cleaned_text.split()
        words.extend(review_words)
    
    # 2. 计算词频
    word_freq = Counter(words)
    
    # 3. 生成词频文本（每个词只出现一次）
    text_for_cloud = ' '.join(word_freq.keys())
    
    # 4. 根据情感类型设置词云参数
    if sentiment_type == 'positive':
        color = 'YlGn'  # 使用黄绿色系
    else:
        color = 'OrRd'  # 使用橙红色系
    
    # 5. 生成词云
    wordcloud = WordCloud(
        width=1200, 
        height=800,
        background_color='white',
        max_words=100,  # 增加显示的词数
        min_font_size=10,
        max_font_size=150,
        colormap=color,
        collocations=False,  # 避免重复的词组
        prefer_horizontal=0.7,  # 70%的词水平显示
        random_state=42
    ).generate(text_for_cloud)
    
    # 6. 保存图片
    plt.figure(figsize=(15, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=20, pad=20)
    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def test_sentiment_analyzer():
    """测试情感分析功能"""
    # 1. 加载和预处理数据
    print("\n[INFO] 加载和预处理数据...")
    
    # 设置输入输出路径
    input_path = os.path.join(project_root, 'data', 'amazon.csv')
    output_path = os.path.join(project_root, 'data', 'processed', 'amazon_clean.csv')
    
    # 如果已经有清理后的数据，直接使用
    if os.path.exists(output_path):
        df = pd.read_csv(output_path)
    else:
        # 否则进行预处理
        preprocessor = DataPreprocessor()
        df = preprocessor.preprocess(input_path, output_path)
    
    print(f"加载数据: {len(df)} 条记录")
    
    # 2. 初始化情感分析器
    print("\n[INFO] 开始情感分析...")
    analyzer = SentimentAnalyzer()
    
    # 3. 分析评论
    df = analyzer.analyze_reviews(df)
    
    # 4. 打印分析结果
    print("\n=== 情感分析结果 ===")
    
    # 情感分布统计
    sentiment_counts = df['sentiment'].value_counts()
    total = len(df)
    
    print("\n1. 整体情感分布:")
    print(f"正面评价: {sentiment_counts.get('positive', 0)} ({sentiment_counts.get('positive', 0)/total*100:.1f}%)")
    print(f"中性评价: {sentiment_counts.get('neutral', 0)} ({sentiment_counts.get('neutral', 0)/total*100:.1f}%)")
    print(f"负面评价: {sentiment_counts.get('negative', 0)} ({sentiment_counts.get('negative', 0)/total*100:.1f}%)")
    
    # 按类别统计
    print("\n2. 各类别情感分布:")
    category_sentiment = pd.crosstab(
        df['main_category'], 
        df['sentiment'],
        normalize='index'
    ) * 100
    print(category_sentiment)
    
    # 价格区间情感分析
    df['price_segment'] = pd.qcut(df['discounted_price'], q=4, labels=[
        '低价', '中低价', '中高价', '高价'
    ])
    
    print("\n3. 价格区间情感分布:")
    price_sentiment = pd.crosstab(
        df['price_segment'], 
        df['sentiment'],
        normalize='index'
    ) * 100
    print(price_sentiment)
    
    # 情感分数分析
    print("\n4. 情感分数统计:")
    print("\n平均情感分数:")
    print(f"综合得分: {df['sentiment_compound'].mean():.3f}")
    print(f"正面强度: {df['sentiment_positive'].mean():.3f}")
    print(f"中性强度: {df['sentiment_neutral'].mean():.3f}")
    print(f"负面强度: {df['sentiment_negative'].mean():.3f}")
    
    # 按价格区间的平均情感分数
    print("\n各价格区间的平均情感分数:")
    price_scores = df.groupby('price_segment')[
        ['sentiment_compound', 'sentiment_positive', 'sentiment_negative']
    ].mean()
    print(price_scores)
    
    # 保存分析结果
    output_file = os.path.join(project_root, 'data', 'processed', 'amazon_with_sentiment.csv')
    df.to_csv(output_file, index=False)
    print(f"\n分析结果已保存到: {output_file}")
    
    # 生成词云
    print("\n[INFO] 生成词云...")
    
    # 创建输出目录
    figures_dir = os.path.join(project_root, 'outputs', 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # 打印列名以进行调试
    print("\n数据列名:", df.columns.tolist())
    
    # 生成正面评价词云
    positive_reviews = df[df['sentiment'] == 'positive']['reviews']
    analyzer.generate_wordcloud(
        positive_reviews,
        os.path.join(figures_dir, 'positive_wordcloud.png'),
        'positive'
    )
    
    # 生成负面评价词云
    negative_reviews = df[df['sentiment'] == 'negative']['reviews']
    analyzer.generate_wordcloud(
        negative_reviews,
        os.path.join(figures_dir, 'negative_wordcloud.png'),
        'negative'
    )
    
    print(f"词云已保存到: {figures_dir}")
    
    print("\n=== 情感分析测试完成 ===")

if __name__ == "__main__":
    test_sentiment_analyzer()
