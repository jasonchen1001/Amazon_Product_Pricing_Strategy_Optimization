import os
import sys

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# 运行测试
from tests.test_preprocessor import test_preprocessor
from tests.test_sentiment import test_sentiment_analyzer

if __name__ == "__main__":
    print("Running preprocessor tests...")
    test_preprocessor()
    
    print("\nRunning sentiment analyzer tests...")
    test_sentiment_analyzer() 