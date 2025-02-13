import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

class DynamicPricingModel:
    """动态定价模型"""
    
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.metrics = {}
        self.feature_names = None  # 添加特征列名属性
        
    def prepare_features(self, df):
        """准备特征"""
        features = pd.DataFrame()
        
        # 1. 基础特征
        features['current_price'] = df['discounted_price']  # 实际售价
        features['list_price'] = df['actual_price']        # 原价
        features['rating'] = df['rating']
        features['rating_count'] = df['rating_count']
        features['sentiment'] = df['sentiment']
        
        # 2. 类别特征
        features = pd.concat([
            features,
            pd.get_dummies(df['main_category'], prefix='category')
        ], axis=1)
        
        # 3. 衍生特征
        features['price_per_rating'] = features['current_price'] / features['rating']
        features['review_sentiment'] = features['rating_count'] * features['sentiment']
        
        return features
    
    def train(self, df):
        """训练模型"""
        print("\n[INFO] 训练定价模型...")
        
        # 1. 准备数据
        X = self.prepare_features(df)
        self.feature_names = X.columns  # 保存特征列名
        y = df['discounted_price']
        
        # 2. 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 3. 特征标准化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 4. 训练模型
        self.model.fit(X_train_scaled, y_train)
        
        # 5. 评估模型
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)
        
        self.metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
            'train_r2': r2_score(y_train, train_pred),
            'test_r2': r2_score(y_test, test_pred)
        }
        
        # 6. 打印结果
        print("\n模型性能:")
        print(f"训练集 RMSE: ₹{self.metrics['train_rmse']:.2f}")
        print(f"测试集 RMSE: ₹{self.metrics['test_rmse']:.2f}")
        print(f"训练集 R²: {self.metrics['train_r2']:.3f}")
        print(f"测试集 R²: {self.metrics['test_r2']:.3f}")
        
        # 7. 特征重要性分析
        self._analyze_feature_importance(X.columns)
        
        return self.metrics
    
    def predict_price(self, features):
        """预测价格"""
        # 确保特征列与训练时一致
        missing_cols = set(self.feature_names) - set(features.columns)
        
        # 添加缺失的类别列，填充0
        for col in missing_cols:
            features[col] = 0
        
        # 按训练时的列顺序重排
        features = features[self.feature_names]
        
        # 标准化并预测
        features_scaled = self.scaler.transform(features)
        return self.model.predict(features_scaled)
    
    def _analyze_feature_importance(self, feature_names):
        """分析特征重要性"""
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("\n特征重要性:")
        for f in range(min(10, len(feature_names))):
            print(f"{feature_names[indices[f]]}: {importances[indices[f]]:.3f}")
        
        # 可视化
        plt.figure(figsize=(10, 6))
        plt.title("特征重要性")
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), 
                  [feature_names[i] for i in indices], 
                  rotation=45)
        plt.tight_layout()
        plt.savefig('outputs/figures/feature_importance.png')
        plt.close()

class PricingModel:
    def __init__(self):
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.feature_importance = None
        
    def prepare_features(self, df):
        """准备特征"""
        features = pd.DataFrame()
        
        # 1. 基础特征
        features['current_price'] = df['discounted_price']  # 实际售价
        features['list_price'] = df['actual_price']        # 原价
        features['rating'] = df['rating']
        features['rating_count'] = df['rating_count']
        features['sentiment'] = df['sentiment']
        
        # 2. 类别特征
        features = pd.concat([
            features,
            pd.get_dummies(df['main_category'], prefix='category')
        ], axis=1)
        
        # 3. 衍生特征
        # 折扣率
        features['discount_rate'] = (features['list_price'] - features['current_price']) / features['list_price']
        # 价格评分比
        features['price_per_rating'] = features['current_price'] / features['rating']
        # 情感影响
        features['review_sentiment'] = features['rating_count'] * features['sentiment']
        
        return features
        
    def train(self, df):
        """训练定价模型"""
        print("\n[INFO] 开始训练定价模型...")
        
        # 1. 准备特征和目标变量
        X = self.prepare_features(df)
        y = df['discounted_price']
        
        # 2. 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 3. 特征标准化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 4. 训练模型
        self.model.fit(X_train_scaled, y_train)
        
        # 5. 评估模型
        y_pred = self.model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n模型评估结果:")
        print(f"MAE: ₹{mae:.2f}")
        print(f"R² Score: {r2:.2%}")
        
        # 6. 特征重要性分析
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n最重要的特征:")
        print(self.feature_importance.head())
        
        # 7. 可视化
        self._plot_results(y_test, y_pred, X.columns)
        
        return self
    
    def optimize_price(self, df, min_margin=0.1, max_change=0.2):
        """生成价格建议"""
        # 1. 准备特征
        features = self.prepare_features(df)
        X_scaled = self.scaler.transform(features)
        
        # 2. 预测最优价格
        base_price = self.model.predict(X_scaled)
        
        # 3. 应用业务约束
        recommendations = []
        for i, row in df.iterrows():
            current_price = row['discounted_price']
            cost = row['actual_price'] * 0.7  # 假设成本是原价的70%
            
            # 3.1 计算价格范围
            min_price = max(cost / (1 - min_margin), current_price * (1 - max_change))
            max_price = current_price * (1 + max_change)
            
            # 3.2 预测最优价格
            optimal_price = np.clip(base_price[i], min_price, max_price)
            
            # 3.3 计算变化
            price_change = (optimal_price - current_price) / current_price
            
            recommendations.append({
                'product_id': row['product_id'],
                'current_price': current_price,
                'optimal_price': optimal_price,
                'price_change': price_change,
                'min_price': min_price,
                'max_price': max_price
            })
        
        return pd.DataFrame(recommendations)
    
    def _plot_results(self, y_true, y_pred, feature_names):
        """可视化模型结果"""
        plt.figure(figsize=(15, 10))
        
        # 1. 预测vs实际价格
        plt.subplot(2, 2, 1)
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        plt.xlabel('实际价格')
        plt.ylabel('预测价格')
        plt.title('预测价格 vs 实际价格')
        
        # 2. 特征重要性
        plt.subplot(2, 2, 2)
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=True)
        plt.barh(range(len(importance_df)), importance_df['importance'])
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('特征重要性')
        plt.title('特征重要性排序')
        
        # 3. 预测误差分布
        plt.subplot(2, 2, 3)
        errors = y_pred - y_true
        plt.hist(errors, bins=50)
        plt.xlabel('预测误差')
        plt.ylabel('频次')
        plt.title('预测误差分布')
        
        plt.tight_layout()
        plt.savefig('outputs/pricing_model_analysis.png')
        plt.close()

def main():
    # 1. 加载数据
    df = pd.read_csv('outputs/analyzed_data.csv')
    
    # 2. 训练模型
    model = PricingModel()
    model.train(df)
    
    # 3. 生成价格建议
    recommendations = model.optimize_price(df)
    
    # 4. 保存结果
    recommendations.to_csv('outputs/price_recommendations.csv', index=False)
    print("\n价格建议已保存到 outputs/price_recommendations.csv")

if __name__ == "__main__":
    main() 