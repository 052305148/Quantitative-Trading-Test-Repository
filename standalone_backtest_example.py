# 临时导入路径配置
import sys
import os

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 临时创建config模块
if not os.path.exists("config"):
    os.makedirs("config")

# 创建config_manager.py
if not os.path.exists("config/config_manager.py"):
    with open("config/config_manager.py", "w", encoding="utf-8") as f:
        f.write("""
# 临时配置管理器
class ConfigManager:
    def __init__(self, config_path=None):
        self.config = {
            "api": {
                "qwen_api_key": "your_api_key_here",
                "qwen_api_url": "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
            },
            "analysis": {
                "long_term_keywords": ["专精特新", "小巨人", "隐形冠军"],
                "mid_term_keywords": ["业绩预告", "财报", "业绩快报"],
                "short_term_keywords": ["突发", "重大", "紧急"]
            },
            "trading": {
                "initial_capital": 1000000,
                "max_position_size": 0.2,
                "stop_loss": 0.08,
                "take_profit": 0.15
            },
            "backtest": {
                "start_date": "2023-01-01",
                "end_date": "2023-12-31",
                "benchmark": "000300.SH"
            }
        }
    
    def get_config(self, key=None):
        if key:
            return self.config.get(key, {})
        return self.config
    
    def update_config(self, key, value):
        self.config[key] = value
""")

# 创建__init__.py
if not os.path.exists("config/__init__.py"):
    with open("config/__init__.py", "w", encoding="utf-8") as f:
        f.write("# 配置模块\n")

# 现在导入优化版回测系统
from optimized_backtest_system import OptimizedBacktestEngine, OptimizedBacktestConfig
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import json

def create_sample_data():
    """创建示例股票数据"""
    # 创建日期范围
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    dates = pd.date_range(start_date, end_date)
    
    # 创建5只股票的价格数据
    np.random.seed(42)
    stocks = ['600000.SH', '000001.SZ', '002415.SZ', '300750.SZ', '688981.SH']
    
    price_data = {}
    for stock in stocks:
        # 生成随机价格走势
        initial_price = np.random.uniform(10, 100)
        returns = np.random.normal(0.001, 0.02, len(dates))  # 日收益率
        prices = [initial_price]
        
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        price_data[stock] = prices[1:]  # 去掉初始价格
    
    # 创建DataFrame
    df = pd.DataFrame(price_data, index=dates)
    
    # 创建基准指数数据
    benchmark_initial = 3000
    benchmark_returns = np.random.normal(0.0008, 0.015, len(dates))
    benchmark_prices = [benchmark_initial]
    
    for ret in benchmark_returns:
        benchmark_prices.append(benchmark_prices[-1] * (1 + ret))
    
    benchmark_data = pd.Series(benchmark_prices[1:], index=dates, name='benchmark')
    
    return df, benchmark_data

def create_sample_signals(price_data):
    """创建示例交易信号"""
    signals = []
    dates = price_data.index.tolist()
    stocks = price_data.columns.tolist()
    
    # 为每个股票生成随机信号
    for date in dates[10:]:  # 跳过前10天，确保有足够的历史数据
        for stock in stocks:
            # 30%概率生成信号
            if np.random.random() < 0.3:
                # 随机生成信号强度和类型
                strength = np.random.uniform(0.5, 1.0)
                signal_type = np.random.choice(['buy', 'sell'])
                
                # 生成随机分析结果
                analysis_result = {
                    "专精特新度": np.random.uniform(0.6, 0.95),
                    "成长性": np.random.uniform(0.5, 0.9),
                    "估值": np.random.uniform(0.4, 0.8),
                    "技术面": np.random.uniform(0.3, 0.9)
                }
                
                # 生成随机市场环境
                market_environment = np.random.choice(['bull', 'bear', 'neutral'])
                
                signals.append({
                    'date': date.strftime('%Y-%m-%d'),  # 转换为字符串
                    'symbol': stock,  # 修改为symbol字段
                    'action': signal_type,  # 修改为action字段
                    'strength': strength,
                    'position_size': 0,  # 添加position_size字段
                    'reason': f"示例信号-{signal_type}",  # 添加reason字段
                    'analysis_result': analysis_result,
                    'market_environment': market_environment
                })
    
    return signals

def run_backtest_example():
    """运行回测示例"""
    print("创建示例数据...")
    price_data, benchmark_data = create_sample_data()
    
    print("生成交易信号...")
    signals = create_sample_signals(price_data)
    
    print(f"生成了 {len(signals)} 个交易信号")
    
    # 创建回测配置
    config = OptimizedBacktestConfig(
        start_date="2023-01-01",
        end_date="2023-12-31",
        initial_cash=1000000,
        commission_ratio=0.0001,
        slippage_ratio=0.0001,
        benchmark="SHSE.000300"
    )
    
    # 创建回测系统
    backtest = OptimizedBacktestEngine(config)
    
    # 设置价格数据 - 优化版回测引擎需要使用load_price_data方法
    # 这里我们不需要手动设置，因为run_backtest方法会自动加载
    
    # 运行回测
    print("开始回测...")
    results = backtest.run_backtest(signals)
    
    # 打印结果
    print("\n回测结果:")
    portfolio = results.get("portfolio", {})
    performance_metrics = results.get("performance_metrics", {})
    
    print(f"最终资产价值: {portfolio.get('final_value', 0):,.2f}")
    print(f"总收益率: {performance_metrics.get('total_return', 0):.2%}")
    print(f"年化收益率: {performance_metrics.get('annualized_return', 0):.2%}")
    print(f"最大回撤: {performance_metrics.get('max_drawdown', 0):.2%}")
    print(f"夏普比率: {performance_metrics.get('sharpe_ratio', 0):.2f}")
    print(f"总交易次数: {performance_metrics.get('total_trades', 0)}")
    print(f"胜率: {performance_metrics.get('win_rate', 0):.2%}")
    
    # 保存结果
    backtest.save_results(results)
    
    # 绘制图表
    backtest.plot_results(results)
    
    return results

def run_integrated_backtest_example():
    """运行整合分析回测示例"""
    print("\n\n=== 整合分析与回测示例 ===")
    
    # 这里可以整合专精特新分析结果与回测系统
    # 由于需要实际的分析数据，这里仅作示例
    
    # 模拟分析结果
    analysis_results = {
        "long_term": {
            "top_companies": [
                {"stock": "688981.SH", "score": 0.95, "name": "中芯国际"},
                {"stock": "300750.SZ", "score": 0.92, "name": "宁德时代"},
                {"stock": "002415.SZ", "score": 0.89, "name": "海康威视"}
            ]
        },
        "mid_term": {
            "positive_expectation_gap": [
                {"stock": "600000.SH", "gap": 0.15, "name": "浦发银行"},
                {"stock": "000001.SZ", "gap": 0.12, "name": "平安银行"}
            ]
        },
        "short_term": {
            "news_signals": [
                {"stock": "688981.SH", "sentiment": "positive", "impact": "high"},
                {"stock": "300750.SZ", "sentiment": "positive", "impact": "medium"}
            ]
        }
    }
    
    print("分析结果示例:")
    print(f"长期分析Top公司: {[c['name'] for c in analysis_results['long_term']['top_companies']]}")
    print(f"中期正向预期差: {[c['name'] for c in analysis_results['mid_term']['positive_expectation_gap']]}")
    print(f"短期新闻信号: {[c['stock'] for c in analysis_results['short_term']['news_signals']]}")
    
    # 基于分析结果生成交易信号
    signals = []
    
    # 长期分析信号
    for company in analysis_results["long_term"]["top_companies"]:
        signals.append({
            'date': datetime(2023, 6, 1).strftime('%Y-%m-%d'),  # 转换为字符串
            'symbol': company["stock"],  # 修改为symbol字段
            'action': 'buy',  # 修改为action字段
            'strength': company["score"],
            'position_size': 0,  # 添加position_size字段
            'reason': '长期分析-专精特新度高',  # 添加reason字段
            'analysis_result': {"专精特新度": company["score"]},
            'market_environment': 'bull',
            'source': 'long_term_analysis'
        })
    
    # 中期分析信号
    for company in analysis_results["mid_term"]["positive_expectation_gap"]:
        signals.append({
            'date': datetime(2023, 6, 15).strftime('%Y-%m-%d'),  # 转换为字符串
            'symbol': company["stock"],  # 修改为symbol字段
            'action': 'buy',  # 修改为action字段
            'strength': 0.7 + company["gap"],
            'position_size': 0,  # 添加position_size字段
            'reason': '中期分析-正向预期差',  # 添加reason字段
            'analysis_result': {"预期差": company["gap"]},
            'market_environment': 'neutral',
            'source': 'mid_term_analysis'
        })
    
    # 短期分析信号
    for company in analysis_results["short_term"]["news_signals"]:
        impact_multiplier = 1.0 if company["impact"] == "high" else 0.8
        signals.append({
            'date': datetime(2023, 6, 20).strftime('%Y-%m-%d'),  # 转换为字符串
            'symbol': company["stock"],  # 修改为symbol字段
            'action': 'buy' if company["sentiment"] == "positive" else 'sell',  # 修改为action字段
            'strength': impact_multiplier,
            'position_size': 0,  # 添加position_size字段
            'reason': f'短期分析-{company["sentiment"]}情绪-{company["impact"]}影响',  # 添加reason字段
            'analysis_result': {"情绪": company["sentiment"], "影响": company["impact"]},
            'market_environment': 'neutral',
            'source': 'short_term_analysis'
        })
    
    print(f"\n基于分析结果生成了 {len(signals)} 个交易信号")
    
    # 运行回测
    return run_backtest_with_signals(signals)

def run_backtest_with_signals(signals):
    """使用给定信号运行回测"""
    print("创建示例数据...")
    # 注意：优化版回测引擎会在内部生成价格数据
    
    # 创建回测配置
    config = OptimizedBacktestConfig(
        start_date="2023-01-01",
        end_date="2023-12-31",
        initial_cash=1000000,
        commission_ratio=0.0001,
        slippage_ratio=0.0001,
        benchmark="SHSE.000300"
    )
    
    # 创建回测系统
    backtest = OptimizedBacktestEngine(config)
    
    # 运行回测
    print("开始回测...")
    results = backtest.run_backtest(signals)
    
    # 打印结果
    print("\n回测结果:")
    portfolio = results.get("portfolio", {})
    performance_metrics = results.get("performance_metrics", {})
    
    print(f"最终资产价值: {portfolio.get('final_value', 0):,.2f}")
    print(f"总收益率: {performance_metrics.get('total_return', 0):.2%}")
    print(f"年化收益率: {performance_metrics.get('annualized_return', 0):.2%}")
    print(f"最大回撤: {performance_metrics.get('max_drawdown', 0):.2%}")
    print(f"夏普比率: {performance_metrics.get('sharpe_ratio', 0):.2f}")
    print(f"总交易次数: {performance_metrics.get('total_trades', 0)}")
    print(f"胜率: {performance_metrics.get('win_rate', 0):.2%}")
    
    # 保存结果
    backtest.save_results(results)
    
    # 绘制图表
    backtest.plot_results(results)
    
    return results

if __name__ == "__main__":
    print("=== 优化版专精特新文本分析交易框架回测示例 ===\n")
    
    # 运行基本回测示例
    print("1. 运行基本回测示例...")
    basic_results = run_backtest_example()
    
    # 运行整合分析回测示例
    print("\n\n2. 运行整合分析回测示例...")
    integrated_results = run_integrated_backtest_example()
    
    print("\n\n=== 回测完成 ===")
    print("优化版回测系统已成功运行，请查看生成的结果文件和图表。")