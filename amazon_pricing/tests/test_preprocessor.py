import os
import sys
import pandas as pd

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.data.preprocessor import DataPreprocessor

def test_preprocessor():
    """测试数据预处理"""
    # 设置输入输出路径
    input_path = os.path.join(project_root, 'data', 'amazon.csv')
    output_path = os.path.join(project_root, 'data', 'processed', 'amazon_clean.csv')
    
    # 创建预处理器
    preprocessor = DataPreprocessor()
    
    # 执行预处理
    clean_data = preprocessor.preprocess(input_path, output_path)
    
    # 验证结果
    assert os.path.exists(output_path), "清理后的数据文件未生成"
    assert len(clean_data) > 0, "清理后的数据不能为空"
    assert all(col in clean_data.columns for col in [
        'main_category', 'reviews', 'discounted_price', 'actual_price', 'rating_count'
    ]), "缺少必要的列"
    
    # 验证数据质量
    assert clean_data['discounted_price'].dtype in ['float64', 'int64'], "价格数据类型不正确"
    assert clean_data['actual_price'].dtype in ['float64', 'int64'], "价格数据类型不正确"
    assert clean_data['rating_count'].dtype in ['float64', 'int64'], "评分数据类型不正确"
    assert all(clean_data['discounted_price'] > 0), "存在非正价格"
    assert all(clean_data['actual_price'] > 0), "存在非正价格"
    
    print("\n=== 数据预处理测试通过 ===")
    print(f"清理后的数据记录数: {len(clean_data)}")

if __name__ == '__main__':
    test_preprocessor()
