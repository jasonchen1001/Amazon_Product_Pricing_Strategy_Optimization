import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, r2_score
import os
from scipy import stats

class PriceElasticityAnalyzer:
    """价格敏感度分析器"""
    
    def __init__(self):
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.metrics = {}
        self.elasticity = None
        self.category_elasticities = {}
        
    def calculate_price_sensitivity(self, df):
        """计算价格敏感度"""
        print("\n[INFO] 分析价格敏感度...")
        
        # 将情感标签转换为数值
        sentiment_map = {
            'positive': 1,
            'neutral': 0,
            'negative': -1
        }
        df['sentiment_score'] = df['sentiment'].map(sentiment_map)
        
        # 按品类分组计算价格敏感度
        elasticities = {}
        
        for category in df['main_category'].unique():
            category_df = df[df['main_category'] == category].copy()
            
            try:
                # 1. 计算相对价格和相对需求
                category_df['price_ratio'] = category_df['discounted_price'] / category_df['actual_price']
                category_df['log_demand'] = np.log(category_df['rating_count'])
                
                # 2. 移除异常值
                price_ratio_q1, price_ratio_q3 = category_df['price_ratio'].quantile([0.25, 0.75])
                price_ratio_iqr = price_ratio_q3 - price_ratio_q1
                price_ratio_upper = price_ratio_q3 + 1.5 * price_ratio_iqr
                price_ratio_lower = price_ratio_q1 - 1.5 * price_ratio_iqr
                
                valid_data = category_df[
                    (category_df['price_ratio'] >= price_ratio_lower) &
                    (category_df['price_ratio'] <= price_ratio_upper)
                ]
                
                # 3. 计算弹性
                if len(valid_data) >= 10:  # 确保有足够的数据点
                    # 使用稳健回归
                    slope, _, _, _, _ = stats.linregress(
                        valid_data['price_ratio'],
                        valid_data['log_demand']
                    )
                    elasticity = -slope  # 取负值，因为价格和需求通常是负相关
                else:
                    elasticity = 0.15  # 默认中等敏感度
                
                # 4. 限制弹性范围
                elasticity = max(0.05, min(elasticity, 0.35))  # 限制在0.05-0.35之间
                
            except Exception as e:
                print(f"[WARNING] {category} 品类弹性计算失败: {str(e)}")
                elasticity = 0.15  # 默认中等敏感度
            
            elasticities[category] = elasticity
            print(f"品类 {category} 的价格敏感度: {elasticity:.3f}")
        
        # 保存品类弹性
        self.category_elasticities = elasticities
        
        # 计算平均弹性
        self.elasticity = np.mean(list(elasticities.values()))
        
        # 打印分析结果
        print("\n=== 价格敏感度分析结果 ===")
        print(f"平均价格敏感度: {self.elasticity:.3f}")
        print("\n品类敏感度分布:")
        for category, elasticity in sorted(elasticities.items(), key=lambda x: x[1], reverse=True):
            print(f"{category}: {elasticity:.3f}")
        
        return elasticities
    
    def get_elasticity(self, category):
        """获取特定品类的价格敏感度"""
        return self.category_elasticities.get(category, 0.15)
    
    def _calculate_price_elasticity(self, prices, demand):
        """计算价格弹性"""
        # 使用对数变换处理非线性关系
        log_prices = np.log(prices)
        log_demand = np.log(demand)
        
        # 线性回归
        X = log_prices.reshape(-1, 1)
        y = log_demand
        
        self.model.fit(X, y)
        elasticity = self.model.coef_[0]
        
        return abs(elasticity)
    
    def _analyze_discount_sensitivity(self, df):
        """分析折扣敏感度"""
        try:
            # 按折扣力度分组，处理重复值并设置 observed=True
            discount_bins = pd.qcut(df['real_discount'], q=5, duplicates='drop')
            discount_analysis = df.groupby(discount_bins, observed=True).agg({
                'rating_count': 'mean',
                'sentiment': 'mean',
                'discounted_price': 'mean'
            })
            
            # 计算折扣对需求的影响
            correlation = df['real_discount'].corr(df['rating_count'])
            
            # 归一化到0-1范围
            sensitivity = abs(correlation)
            return min(sensitivity, 1.0)
        except Exception as e:
            print(f"[WARNING] 折扣敏感度分析失败: {str(e)}")
            return 0.0
    
    def _analyze_sentiment_impact(self, df):
        """分析情感对销量的影响"""
        # 计算不同情感评分的平均销量
        sentiment_impact = df.groupby('sentiment').agg({
            'rating_count': 'mean',
            'discounted_price': 'mean'
        })
        
        # 计算情感和销量的相关性
        correlation = df['sentiment'].corr(df['rating_count'])
        
        # 归一化到0-1范围
        impact = abs(correlation)
        return min(impact, 1.0)
    
    def _analyze_price_segments(self, df):
        """分析价格区间敏感度"""
        try:
            # 按价格分段，处理重复值并设置 observed=True
            price_bins = pd.qcut(df['discounted_price'], q=5, duplicates='drop')
            segment_analysis = df.groupby(price_bins, observed=True).agg({
                'rating_count': 'mean',
                'sentiment': 'mean',
                'real_discount': 'mean'
            })
            
            # 计算高价区间和低价区间的需求差异
            high_price_demand = segment_analysis['rating_count'].iloc[-2:].mean()
            low_price_demand = segment_analysis['rating_count'].iloc[:2].mean()
            
            # 计算价格区间敏感度
            if low_price_demand > 0:
                sensitivity = (low_price_demand - high_price_demand) / low_price_demand
                return min(max(sensitivity, 0), 1)
            return 0.5
        except Exception as e:
            print(f"[WARNING] 价格区间分析失败: {str(e)}")
            return 0.0
    
    def _calculate_composite_sensitivity(self, metrics):
        """计算综合敏感度指标"""
        weights = {
            'price_elasticity': 0.4,      # 价格弹性权重
            'discount_sensitivity': 0.3,   # 折扣敏感度权重
            'sentiment_impact': 0.2,       # 情感影响权重
            'segment_sensitivity': 0.1     # 价格区间敏感度权重
        }
        
        composite = sum(metrics[k] * weights[k] for k in weights)
        return min(composite, 1.0)
    
    def _print_analysis_results(self, metrics):
        """打印分析结果"""
        print("\n=== 价格敏感度分析结果 ===")
        print(f"\n1. 价格弹性: {metrics['price_elasticity']:.3f}")
        print(f"2. 折扣敏感度: {metrics['discount_sensitivity']:.3f}")
        print(f"3. 情感影响: {metrics['sentiment_impact']:.3f}")
        print(f"4. 价格区间敏感度: {metrics['segment_sensitivity']:.3f}")
        print(f"\n综合敏感度指标: {self.elasticity:.3f}")
        
        # 解释结果
        if self.elasticity > 0.7:
            print("\n市场高度敏感，建议:")
            print("- 谨慎调整价格")
            print("- 关注竞品定价")
            print("- 强化差异化优势")
        elif self.elasticity > 0.4:
            print("\n市场中度敏感，建议:")
            print("- 灵活调整价格")
            print("- 优化促销策略")
            print("- 提升产品价值")
        else:
            print("\n市场低度敏感，建议:")
            print("- 提高定价空间")
            print("- 注重品牌建设")
            print("- 扩大市场份额")
    
    def _generate_visualizations(self, df):
        """生成分析可视化"""
        try:
            # 确保输出目录存在
            os.makedirs('outputs/figures', exist_ok=True)
            
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac系统
            plt.rcParams['axes.unicode_minus'] = False
            
            plt.figure(figsize=(15, 10))
            
            # 1. 价格-需求关系
            plt.subplot(2, 2, 1)
            plt.scatter(df['discounted_price'], df['rating_count'], alpha=0.5)
            plt.xlabel('价格')
            plt.ylabel('需求(评论数)')
            plt.title('价格-需求关系')
            
            # 2. 折扣-需求关系
            plt.subplot(2, 2, 2)
            plt.scatter(df['real_discount'], df['rating_count'], alpha=0.5)
            plt.xlabel('折扣率(%)')
            plt.ylabel('需求(评论数)')
            plt.title('折扣-需求关系')
            
            # 3. 情感-需求关系
            plt.subplot(2, 2, 3)
            sentiment_demand = df.groupby('sentiment', observed=True)['rating_count'].mean()
            sentiment_demand.plot(kind='bar')
            plt.xlabel('情感得分')
            plt.ylabel('平均需求')
            plt.title('情感-需求关系')
            
            # 4. 价格区间分析
            plt.subplot(2, 2, 4)
            price_bins = pd.qcut(df['discounted_price'], q=5, duplicates='drop')
            segment_analysis = df.groupby(price_bins, observed=True)['rating_count'].mean()
            segment_analysis.plot(kind='bar')
            plt.xlabel('价格区间')
            plt.ylabel('平均需求')
            plt.title('价格区间分析')
            
            plt.tight_layout()
            plt.savefig('outputs/figures/price_sensitivity_analysis.png', dpi=300)
            plt.close()
        except Exception as e:
            print(f"[WARNING] 可视化生成失败: {str(e)}")

    def get_model_performance(self):
        return {
            'R² Score': f"{self.metrics['r2']:.2%}",
            'MAPE': f"{self.metrics['mape']:.2%}"
        } 