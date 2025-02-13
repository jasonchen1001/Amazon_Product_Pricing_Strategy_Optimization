from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import re
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt

class SentimentAnalyzer:
    def __init__(self):
        # 下载必要的NLTK数据
        try:
            nltk.data.find('corpora/stopwords')
            nltk.download('punkt')
            nltk.download('vader_lexicon')
        except LookupError:
            nltk.download('stopwords')
            nltk.download('punkt')
            nltk.download('vader_lexicon')
        
        # 初始化VADER分析器
        self.sia = SentimentIntensityAnalyzer()
        
        # 定义产品相关的中性词（这些词不应影响情感判断）
        self.neutral_words = {
            # 产品描述词
            'quality', 'value', 'money', 'price', 'cost',
            'size', 'small', 'big', 'large', 'little',
            'color', 'colour', 'black', 'white', 'blue',
            'weight', 'heavy', 'light', 'average', 'normal',
            'original', 'genuine', 'authentic',
            'charged', 'charging', 'charge', 'battery', 'power',
            
            # 动作词
            'buy', 'bought', 'purchase', 'ordered', 'received',
            'use', 'using', 'used', 'keep', 'keeping',
            'want', 'need', 'try', 'tried',
            'able', 'unable', 'can', 'cannot', 'could',
            'work', 'working', 'worked', 'start',
            
            # 时间词
            'time', 'month', 'day', 'year', 'week',
            'long', 'short', 'fast', 'slow', 'quick',
            'now', 'later', 'before', 'after', 'during',
            
            # 产品相关
            'product', 'item', 'thing', 'stuff', 'material',
            'cable', 'charger', 'wire', 'cord', 'adapter',
            'device', 'amazon', 'delivery', 'package', 'box',
            'brand', 'company', 'seller', 'shop', 'store'
        }
    
    def preprocess_text(self, text):
        """预处理文本，移除中性词"""
        if not isinstance(text, str):
            return ""
        
        # 转换为小写
        text = text.lower()
        
        # 分词并移除中性词
        words = text.split()
        words = [w for w in words if w not in self.neutral_words]
        
        return ' '.join(words)
    
    def analyze_text(self, text):
        """分析文本情感，返回详细的情感分数"""
        try:
            # 预处理文本
            processed_text = self.preprocess_text(text)
            if not processed_text:
                return {
                    'compound': 0.0,
                    'pos': 0.0,
                    'neu': 1.0,
                    'neg': 0.0,
                    'sentiment': 'neutral'
                }
            
            # 获取VADER情感分数
            scores = self.sia.polarity_scores(processed_text)
            
            # 添加情感标签
            if scores['compound'] >= 0.05:
                sentiment = 'positive'
            elif scores['compound'] <= -0.05:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            scores['sentiment'] = sentiment
            return scores
            
        except Exception as e:
            print(f"Error analyzing text: {e}")
            return {
                'compound': 0.0,
                'pos': 0.0,
                'neu': 1.0,
                'neg': 0.0,
                'sentiment': 'neutral'
            }
    
    def analyze_reviews(self, df):
        """分析所有评论并添加情感分析结果"""
        # 存储详细的情感分析结果
        sentiments = []
        
        for _, row in df.iterrows():
            text = str(row['reviews'])
            
            # 安全地获取评分
            try:
                rating_str = str(row.get('rating', '5.0'))
                # 清理评分字符串，只保留数字和小数点
                rating_str = ''.join(c for c in rating_str if c.isdigit() or c == '.')
                rating = float(rating_str) if rating_str else 5.0
            except:
                rating = 5.0  # 如果无法解析评分，使用默认值
            
            # 分析文本情感
            sentiment_scores = self.analyze_text(text)
            
            # 如果评分低于4，倾向于判定为负面
            if rating < 4.0:
                sentiment_scores['sentiment'] = 'negative'
                sentiment_scores['neg'] = max(sentiment_scores['neg'], 0.2)
            
            sentiments.append(sentiment_scores)
        
        # 添加情感分析结果到DataFrame
        df['sentiment_scores'] = sentiments
        df['sentiment'] = [s['sentiment'] for s in sentiments]
        df['sentiment_compound'] = [s['compound'] for s in sentiments]
        df['sentiment_positive'] = [s['pos'] for s in sentiments]
        df['sentiment_neutral'] = [s['neu'] for s in sentiments]
        df['sentiment_negative'] = [s['neg'] for s in sentiments]
        
        # 打印分布情况
        sentiment_dist = df['sentiment'].value_counts(normalize=True) * 100
        print("\nSentiment Distribution:")
        print(f"Positive: {sentiment_dist.get('positive', 0):.1f}%")
        print(f"Neutral: {sentiment_dist.get('neutral', 0):.1f}%")
        print(f"Negative: {sentiment_dist.get('negative', 0):.1f}%")
        
        return df

    def get_sentiment_words(self, text):
        """
        分析文本中的情感词
        Args:
            text (str): 输入文本
        Returns:
            list: 情感词及其得分的列表 [(word, score)]
        """
        if not isinstance(text, str):
            return []
        
        # 简单分词（用空格分割）
        words = text.lower().split()
        
        # 过滤停用词和标点符号
        stop_words = set(stopwords.words('english'))
        words = [word for word in words 
                if word.isalnum() and  # 只保留字母数字
                word not in stop_words and  # 过滤停用词
                len(word) > 2]  # 过滤短词
        
        # 计算每个词的情感得分
        word_scores = []
        for word in words:
            if word not in self.neutral_words:  # 过滤中性词
                score = self.sia.polarity_scores(word)['compound']
                word_scores.append((word, score))
        
        return word_scores

    def generate_wordcloud(self, text, output_path, sentiment_type='positive'):
        """
        生成词云图
        Args:
            text: 评论文本列表
            output_path: 输出路径
            sentiment_type: 情感类型 ('positive' 或 'negative')
        """
        # 1. 获取情感词
        words = []
        for review in text:
            if not isinstance(review, str):
                continue
            
            # 获取情感词和得分
            sentiment_words = self.get_sentiment_words(review)
            
            # 根据情感类型筛选词
            if sentiment_type == 'positive':
                filtered_words = [word for word, score in sentiment_words if score > 0.2]
            else:
                filtered_words = [word for word, score in sentiment_words if score < -0.2]
            
            words.extend(filtered_words)
        
        # 2. 计算词频
        word_freq = Counter(words)
        
        # 3. 生成词频文本（每个词只出现一次）
        text_for_cloud = ' '.join(word_freq.keys())
        
        # 4. 生成词云
        wordcloud = WordCloud(
            width=1200, 
            height=800,
            background_color='white',
            max_words=100,
            min_font_size=10,
            max_font_size=150,
            colormap='YlGn' if sentiment_type == 'positive' else 'OrRd',
            collocations=False,
            prefer_horizontal=0.7,
            random_state=42
        ).generate(text_for_cloud)
        
        # 5. 保存图片
        plt.figure(figsize=(15, 10))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()