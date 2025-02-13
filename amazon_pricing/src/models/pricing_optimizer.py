import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_absolute_percentage_error
import xgboost as xgb
from .price_elasticity import PriceElasticityAnalyzer
from .sentiment_analyzer import SentimentAnalyzer
from config import *
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# 配置参数
XGBOOST_PARAMS = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 4,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
}

FEATURE_ENGINEERING_PARAMS = {
    'add_polynomial_features': True,
    'use_log_transform': True,
    'price_segments': 5,
    'popularity_segments': 5,
    'standardize_features': True
}

OPTIMIZATION_PARAMS = {
    'max_price_change': {
        'increase': 0.2,  # 最大涨价20%
        'decrease': 0.3   # 最大降价30%
    },
    'price_steps': 20,    # 价格搜索步数
    'cost_ratio': 0.7,    # 成本占原价的比例
}

class PricingOptimizer:
    def __init__(self, pricing_model, elasticity_analyzer):
        self.pricing_model = pricing_model
        self.elasticity_analyzer = elasticity_analyzer
        self.optimization_results = {}
        
    def optimize_price(self, product_data, constraints=None):
        """优化产品价格"""
        print("\n[INFO] 优化产品价格...")
        
        # 1. 获取产品特征
        features = self.pricing_model.prepare_features(product_data)
        current_price = product_data['discounted_price'].values[0]
        list_price = product_data['actual_price'].values[0]
        sentiment = product_data['sentiment'].values[0]
        category = product_data['main_category'].values[0]
        
        # 获取品类特定的价格敏感度
        elasticity = self.elasticity_analyzer.get_elasticity(category)
        
        # 2. 确定定价策略
        if elasticity < 0.15:
            strategy = 'premium'      # 低敏感度，溢价策略
        elif elasticity < 0.25:
            strategy = 'balanced'     # 中等敏感度，平衡策略
        else:
            strategy = 'aggressive'   # 高敏感度，竞争策略
        
        print(f"采用策略: {strategy} (品类: {category}, 价格敏感度: {elasticity:.3f})")
        
        # 3. 计算当前收益
        current_demand = self._predict_demand(features, strategy)
        current_revenue = current_price * current_demand
        
        def objective(price):
            features_copy = features.copy()
            features_copy['current_price'] = price
            predicted_demand = self._predict_demand(features_copy, strategy)
            new_revenue = price * predicted_demand
            
            # 计算收益变化
            revenue_change = (new_revenue - current_revenue) / current_revenue
            price_change = (price - current_price) / current_price
            
            # 根据策略差异化处理
            if strategy == 'premium':
                # 低敏感度：更强烈地鼓励涨价
                if price > current_price:
                    revenue_bonus = new_revenue * (1 + revenue_change * 1.2)  # 提高涨价奖励
                else:
                    revenue_bonus = new_revenue * (1 + revenue_change * 0.1)  # 降低降价奖励
                    
            elif strategy == 'aggressive':
                # 高敏感度：保持不变
                if price < current_price:
                    revenue_bonus = new_revenue * (1 + revenue_change * 1.5)
                else:
                    revenue_bonus = new_revenue * (1 + revenue_change * 0.2)
                    
            else:  # balanced
                # 中等敏感度：轻微偏好降价
                if price < current_price:
                    revenue_bonus = new_revenue * (1 + revenue_change * 1.1)
                else:
                    revenue_bonus = new_revenue * (1 + revenue_change * 0.9)
            
            # 品牌价值保护
            brand_penalty = 0
            if price < list_price * 0.7:
                brand_penalty = revenue_bonus * 0.3
            
            return revenue_bonus - brand_penalty
        
        # 4. 设置价格约束
        if constraints is None:
            # 根据策略设置变动范围
            if strategy == 'premium':
                max_increase = 0.08  # 最大涨价8%
                max_decrease = 0.03  # 最大降价3%
            elif strategy == 'aggressive':
                max_increase = 0.03  # 最大涨价3%
                max_decrease = 0.08  # 最大降价8%
            else:
                max_increase = 0.05  # 最大涨价5%
                max_decrease = 0.05  # 最大降价5%
            
            constraints = {
                'min_price': current_price * (1 - max_decrease),
                'max_price': current_price * (1 + max_increase),
                'min_margin': 0.1
            }
        
        # 5. 执行优化
        bounds = [(constraints['min_price'], constraints['max_price'])]
        result = minimize(
            lambda x: -objective(x),
            x0=current_price,
            bounds=bounds,
            method='L-BFGS-B'
        )
        
        # 6. 保存结果
        optimal_revenue = -result.fun
        self.optimization_results = {
            'current_price': current_price,
            'optimal_price': result.x[0],
            'price_change': ((result.x[0] / current_price - 1) * 100),
            'expected_revenue': optimal_revenue,
            'current_revenue': current_revenue,
            'revenue_change': ((optimal_revenue - current_revenue) / current_revenue * 100),
            'strategy': strategy,
            'elasticity': elasticity,
            'success': result.success
        }
        
        return self.optimization_results
    
    def _predict_demand(self, features, strategy):
        """预测需求"""
        try:
            # 基础需求
            base_demand = features['rating_count'].values[0]
            price = features['current_price'].values[0]
            current_price = self.optimization_results['current_price']  # 使用当前价格作为基准
            category = features['main_category'].values[0]
            elasticity = self.elasticity_analyzer.get_elasticity(category)
            
            # 价格变化（相对于当前价格）
            price_change = (price - current_price) / current_price
            
            # 根据策略差异化处理需求变化
            if strategy == 'premium':
                if price_change > 0:
                    demand_factor = 1 - (elasticity * price_change * 0.8)
                else:
                    demand_factor = 1 + (abs(elasticity * price_change) * 0.5)
            elif strategy == 'aggressive':
                if price_change > 0:
                    demand_factor = 1 - (elasticity * price_change * 1.5)
                else:
                    demand_factor = 1 + (abs(elasticity * price_change) * 0.2)
            else:  # balanced
                if price_change > 0:
                    demand_factor = 1 - (elasticity * price_change * 1.2)
                else:
                    demand_factor = 1 + (abs(elasticity * price_change) * 0.8)
            
            # 情感影响
            sentiment_factor = 1 + (features['sentiment'].values[0] * 0.2)
            
            # 最终需求
            final_demand = base_demand * demand_factor * sentiment_factor
            return max(final_demand, base_demand * 0.1)
        
        except Exception as e:
            print(f"[WARNING] 需求预测失败: {str(e)}")
            return features['rating_count'].values[0]
    
    def _calculate_elasticity_penalty(self, price, original_price):
        """计算价格敏感度惩罚"""
        elasticity = self.elasticity_analyzer.elasticity
        price_change = abs(price - original_price) / original_price
        return elasticity * price_change * price
    
    def _print_optimization_results(self):
        """打印优化结果"""
        print("\n=== 价格优化结果 ===")
        print(f"当前价格: ₹{self.optimization_results['current_price']:.2f}")
        print(f"最优价格: ₹{self.optimization_results['optimal_price']:.2f}")
        print(f"价格变化: {self.optimization_results['price_change']:.1f}%")
        print(f"预期收益: ₹{self.optimization_results['expected_revenue']:.2f}")
        print(f"优化状态: {self.optimization_results['success']}")
        
    def _plot_revenue_curve(self, product_data, min_price, max_price):
        """绘制价格-收益曲线"""
        prices = np.linspace(min_price, max_price, 100)
        revenues = []
        features = self.pricing_model.prepare_features(product_data)
        
        for price in prices:
            features_copy = features.copy()
            features_copy['current_price'] = price
            demand = self._predict_demand(features_copy)
            revenue = price * demand
            revenues.append(revenue)
        
        plt.figure(figsize=(10, 6))
        plt.plot(prices, revenues, 'b-', label='收益曲线')
        plt.axvline(x=self.optimization_results['optimal_price'], 
                   color='r', linestyle='--', label='最优价格')
        plt.axvline(x=self.optimization_results['current_price'], 
                   color='g', linestyle='--', label='当前价格')
        
        plt.xlabel('价格')
        plt.ylabel('预期收益')
        plt.title('价格-收益关系')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('outputs/figures/revenue_curve.png')
        plt.close()