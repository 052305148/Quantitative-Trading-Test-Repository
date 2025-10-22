"""
测试情感分析器
"""

import ssl
import os
import warnings

# 设置SSL环境
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
warnings.filterwarnings("ignore", category=UserWarning)

# 导入模块
from 中期分析.预期差分析.expectation_gap_analysis import SentimentAnalyzer

# 测试情感分析器
print("初始化情感分析器...")
analyzer = SentimentAnalyzer()

# 测试文本
test_text = "公司获得国家级小巨人认证，技术突破显著，Q3订单超预期，新产品获得市场认可，营收大幅增长。"
print(f"测试文本: {test_text}")

# 分析情感
sentiment = analyzer.analyze_sentiment(test_text)
print(f"情感分数: {sentiment}")

print("测试完成!")