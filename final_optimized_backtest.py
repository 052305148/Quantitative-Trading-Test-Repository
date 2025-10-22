# 优化版回测示例 - 整合增强信号生成和风险控制
import sys
import os

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入增强版回测系统
from enhanced_backtest_system import EnhancedBacktestEngine, EnhancedBacktestConfig
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import json

def create_optimized_sample_data():
    """创建优化版示例股票数据，模拟更高收益潜力"""
    # 创建日期范围
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    dates = pd.date_range(start_date, end_date)
    
    # 创建5只股票的价格数据，模拟不同特性的股票
    np.random.seed(42)
    stocks = ['600000.SH', '000001.SZ', '002415.SZ', '300750.SZ', '688981.SH']
    
    # 设置股票特性，模拟不同行业和成长性
    stock_characteristics = {
        '600000.SH': {'trend': 0.0008, 'volatility': 0.018, 'momentum': 0.4, 'sector': '金融'},  # 稳健增长
        '000001.SZ': {'trend': 0.0010, 'volatility': 0.020, 'momentum': 0.5, 'sector': '金融'},  # 中等增长
        '002415.SZ': {'trend': 0.0015, 'volatility': 0.024, 'momentum': 0.7, 'sector': '科技'},  # 高增长
        '300750.SZ': {'trend': 0.0018, 'volatility': 0.028, 'momentum': 0.8, 'sector': '新能源'},  # 高增长高波动
        '688981.SH': {'trend': 0.0020, 'volatility': 0.030, 'momentum': 0.9, 'sector': '半导体'},  # 最高增长
    }
    
    # 模拟行业轮动效应
    sector_cycles = {
        '金融': {'start_month': 1, 'end_month': 4, 'boost': 0.3},  # 1-4月金融板块表现较好
        '科技': {'start_month': 5, 'end_month': 8, 'boost': 0.4},  # 5-8月科技板块表现较好
        '新能源': {'start_month': 3, 'end_month': 6, 'boost': 0.35},  # 3-6月新能源板块表现较好
        '半导体': {'start_month': 7, 'end_month': 10, 'boost': 0.45},  # 7-10月半导体板块表现较好
    }
    
    price_data = {}
    for stock in stocks:
        char = stock_characteristics[stock]
        sector = char['sector']
        
        # 生成带有趋势和动量的价格走势
        initial_price = np.random.uniform(10, 100)
        prices = [initial_price]
        
        # 模拟动量效应和趋势
        momentum_factor = 1.0
        for i, date in enumerate(dates):
            # 基础收益率 = 趋势 + 随机噪声
            base_return = char['trend'] + np.random.normal(0, char['volatility'])
            
            # 添加行业轮动效应
            month = date.month
            sector_cycle = sector_cycles.get(sector, {})
            if sector_cycle.get('start_month') and sector_cycle.get('end_month'):
                if sector_cycle['start_month'] <= month <= sector_cycle['end_month']:
                    base_return += sector_cycle['boost'] * char['trend']
            
            # 添加动量效应
            if i > 20:  # 20天后开始考虑动量
                recent_returns = np.array([
                    (prices[j] - prices[j-1]) / prices[j-1] 
                    for j in range(max(1, i-20), i)
                ])
                avg_recent_return = np.mean(recent_returns)
                momentum_factor = 1.0 + char['momentum'] * np.tanh(avg_recent_return * 50)
            
            # 应用动量因子
            adjusted_return = base_return * momentum_factor
            
            # 添加一些随机的大幅波动（模拟市场事件）
            if np.random.random() < 0.02:  # 2%概率发生大幅波动
                event_return = np.random.choice([-0.05, -0.03, 0.03, 0.05])
                adjusted_return += event_return
            
            # 添加季度效应（季末和季初表现较好）
            if date.month in [3, 6, 9, 12] and date.day > 25:  # 季末
                adjusted_return += 0.005  # 额外0.5%收益
            elif date.month in [4, 7, 10, 1] and date.day < 5:  # 季初
                adjusted_return += 0.003  # 额外0.3%收益
            
            new_price = prices[-1] * (1 + adjusted_return)
            prices.append(max(new_price, initial_price * 0.5))  # 防止价格过低
        
        price_data[stock] = prices[1:]  # 去掉初始价格
    
    # 创建DataFrame
    df = pd.DataFrame(price_data, index=dates)
    
    # 创建基准指数数据（略低于股票平均收益）
    benchmark_initial = 3000
    benchmark_returns = np.random.normal(0.0006, 0.012, len(dates))  # 略低于股票平均收益
    benchmark_prices = [benchmark_initial]
    
    for ret in benchmark_returns:
        benchmark_prices.append(benchmark_prices[-1] * (1 + ret))
    
    benchmark_data = pd.Series(benchmark_prices[1:], index=dates, name='benchmark')
    
    return df, benchmark_data

def create_optimized_signals(price_data):
    """创建优化版交易信号，提高信号质量和多样性"""
    signals = []
    # 获取日期索引
    dates = None
    stocks = list(price_data.keys())
    
    # 获取所有股票共有的日期
    if stocks:
        dates = price_data[stocks[0]].index.tolist()
        for stock in stocks[1:]:
            dates = [d for d in dates if d in price_data[stock].index]
    
    if not dates:
        print("警告: 没有找到共有的交易日期")
        return signals
    
    # 计算每只股票的技术指标
    technical_indicators = {}
    for stock in stocks:
        # 获取收盘价序列
        prices = price_data[stock]['close'].values
        returns = np.diff(prices) / prices[:-1]
        
        # 计算移动平均线
        ma5 = pd.Series(prices).rolling(5).mean().values
        ma10 = pd.Series(prices).rolling(10).mean().values
        ma20 = pd.Series(prices).rolling(20).mean().values
        
        # 计算RSI
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(14).mean()
        avg_loss = pd.Series(loss).rolling(14).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # 计算MACD
        ema12 = pd.Series(prices).ewm(span=12).mean()
        ema26 = pd.Series(prices).ewm(span=26).mean()
        macd = ema12 - ema26
        signal_line = macd.ewm(span=9).mean()
        macd_histogram = macd - signal_line
        
        # 计算动量
        momentum = pd.Series(prices).pct_change(10).values
        
        # 计算布林带
        bb_period = 20
        bb_std = 2
        bb_middle = pd.Series(prices).rolling(bb_period).mean()
        bb_std_dev = pd.Series(prices).rolling(bb_period).std()
        bb_upper = bb_middle + (bb_std_dev * bb_std)
        bb_lower = bb_middle - (bb_std_dev * bb_std)
        
        # 计算波动率
        volatility = pd.Series(returns).rolling(20).std() * np.sqrt(252)
        
        technical_indicators[stock] = {
            'ma5': ma5,
            'ma10': ma10,
            'ma20': ma20,
            'rsi': rsi,
            'macd': macd.values,
            'signal_line': signal_line.values,
            'macd_histogram': macd_histogram.values,
            'momentum': momentum,
            'bb_upper': bb_upper.values,
            'bb_lower': bb_lower.values,
            'bb_middle': bb_middle.values,
            'volatility': volatility.values,
            'returns': returns
        }
    
    # 为每个股票生成基于多种技术指标的信号
    # 确保索引范围正确，考虑技术指标计算需要的历史数据
    max_start_idx = max(30, 26)  # RSI需要14天，MACD需要26天
    for i, date in enumerate(dates[max_start_idx:]):  # 跳过前max_start_idx天，确保有足够的历史数据
        idx = i + max_start_idx  # 对应原始数据索引
        
        # 确保索引不超出范围
        if idx >= len(dates):
            break
            
        for stock in stocks:
            indicators = technical_indicators[stock]
            
            # 再次检查索引范围，考虑所有技术指标
            if idx >= len(indicators['ma5']) or idx >= len(indicators['rsi']) or idx >= len(indicators['macd_histogram']):
                continue
            
            # 买入信号条件（多种技术指标组合）
            buy_conditions = []
            buy_strength_factors = []
            
            # 均线多头排列
            if (indicators['ma5'][idx] > indicators['ma10'][idx] > indicators['ma20'][idx]):
                buy_conditions.append(True)
                # 计算均线排列强度
                ma_strength = (indicators['ma5'][idx] / indicators['ma20'][idx] - 1) * 10
                buy_strength_factors.append(min(1.0, ma_strength))
            
            # RSI适中
            if 30 < indicators['rsi'][idx] < 70:
                buy_conditions.append(True)
                rsi_strength = 1.0 - abs(indicators['rsi'][idx] - 50) / 50
                buy_strength_factors.append(rsi_strength)
            
            # MACD金叉
            if (idx > 0 and 
                indicators['macd_histogram'][idx] > 0 and 
                indicators['macd_histogram'][idx-1] <= 0):
                buy_conditions.append(True)
                buy_strength_factors.append(0.8)  # MACD金叉固定强度
            
            # 价格接近布林带下轨
            if indicators['bb_lower'][idx] > 0:
                bb_position = (price_data[stock].iloc[idx] - indicators['bb_lower'][idx]) / (indicators['bb_upper'][idx] - indicators['bb_lower'][idx])
                # 确保bb_position是标量值
                if isinstance(bb_position, pd.Series):
                    bb_position = bb_position.iloc[0] if len(bb_position) > 0 else 0.5
                if bb_position < 0.2:  # 接近下轨
                    buy_conditions.append(True)
                    bb_strength = 1.0 - bb_position * 5  # 越接近下轨强度越高
                    buy_strength_factors.append(min(1.0, bb_strength))
            
            # 正动量
            if indicators['momentum'][idx] > 0:
                buy_conditions.append(True)
                momentum_strength = min(1.0, indicators['momentum'][idx] * 20)
                buy_strength_factors.append(momentum_strength)
            
            # 成交量放大（如果有成交量数据）
            # 这里简化处理，假设随机成交量
            if np.random.random() < 0.6:  # 60%概率成交量放大
                buy_conditions.append(True)
                buy_strength_factors.append(0.7)
            
            # 至少满足3个条件才生成买入信号
            if sum(buy_conditions) >= 3:
                # 计算综合信号强度
                strength = np.mean(buy_strength_factors)
                strength = max(0.6, min(1.0, strength))  # 限制在0.6-1.0范围
                
                # 生成分析结果
                analysis_result = {
                    "专精特新度": np.random.uniform(0.75, 0.95),
                    "成长性": np.random.uniform(0.7, 0.9),
                    "估值": np.random.uniform(0.6, 0.85),
                    "技术面": strength,
                    "动量": indicators['momentum'][idx],
                    "RSI": indicators['rsi'][idx],
                    "MACD": indicators['macd_histogram'][idx],
                    "布林带位置": (price_data[stock].iloc[idx] - indicators['bb_lower'][idx]) / (indicators['bb_upper'][idx] - indicators['bb_lower'][idx]) if indicators['bb_upper'][idx] > indicators['bb_lower'][idx] else 0.5
                    if not isinstance((price_data[stock].iloc[idx] - indicators['bb_lower'][idx]) / (indicators['bb_upper'][idx] - indicators['bb_lower'][idx]), pd.Series)
                    else ((price_data[stock].iloc[idx] - indicators['bb_lower'][idx]) / (indicators['bb_upper'][idx] - indicators['bb_lower'][idx])).iloc[0]
                    if not isinstance((price_data[stock].iloc[idx] - indicators['bb_lower'][idx]) / (indicators['bb_upper'][idx] - indicators['bb_lower'][idx]), pd.Series)
                    else ((price_data[stock].iloc[idx] - indicators['bb_lower'][idx]) / (indicators['bb_upper'][idx] - indicators['bb_lower'][idx])).iloc[0]
                }
                
                # 生成市场环境（基于近期市场表现）
                recent_market_returns = np.mean([
                    technical_indicators[s]['returns'][max(0, idx-5):idx].mean() 
                    for s in stocks
                ])
                
                if recent_market_returns > 0.008:
                    market_environment = 'bull'
                elif recent_market_returns < -0.008:
                    market_environment = 'bear'
                else:
                    market_environment = 'neutral'
                
                # 根据市场环境调整信号生成概率
                signal_probability = 0.5  # 基础50%概率
                if market_environment == 'bull':
                    signal_probability *= 1.2  # 牛市增加20%概率
                elif market_environment == 'bear':
                    signal_probability *= 0.8  # 熊市减少20%概率
                
                if np.random.random() < signal_probability:
                    signals.append({
                        'date': date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date),
                        'symbol': stock,
                        'action': 'buy',
                        'strength': strength,
                        'position_size': 0,  # 将由系统计算
                        'reason': f"技术分析-买入信号-强度{strength:.2f}",
                        'analysis_result': analysis_result,
                        'market_environment': market_environment,
                        'source': 'technical_analysis'
                    })
            
            # 卖出信号条件（多种技术指标组合）
            sell_conditions = []
            sell_strength_factors = []
            
            # 均线空头排列
            if (indicators['ma5'][idx] < indicators['ma10'][idx] < indicators['ma20'][idx]):
                sell_conditions.append(True)
                # 计算均线排列强度
                ma_strength = (indicators['ma20'][idx] / indicators['ma5'][idx] - 1) * 10
                sell_strength_factors.append(min(1.0, ma_strength))
            
            # RSI超买
            if indicators['rsi'][idx] > 70:
                sell_conditions.append(True)
                rsi_strength = (indicators['rsi'][idx] - 70) / 30
                sell_strength_factors.append(min(1.0, rsi_strength))
            
            # MACD死叉
            if (idx > 0 and 
                indicators['macd_histogram'][idx] < 0 and 
                indicators['macd_histogram'][idx-1] >= 0):
                sell_conditions.append(True)
                sell_strength_factors.append(0.8)  # MACD死叉固定强度
            
            # 价格接近布林带上轨
            if indicators['bb_upper'][idx] > 0:
                bb_position = (price_data[stock].iloc[idx] - indicators['bb_lower'][idx]) / (indicators['bb_upper'][idx] - indicators['bb_lower'][idx])
                # 确保bb_position是标量值
                if isinstance(bb_position, pd.Series):
                    bb_position = bb_position.iloc[0] if len(bb_position) > 0 else 0.5
                if bb_position > 0.8:  # 接近上轨
                    sell_conditions.append(True)
                    bb_strength = (bb_position - 0.8) * 5  # 越接近上轨强度越高
                    sell_strength_factors.append(min(1.0, bb_strength))
            
            # 负动量
            if indicators['momentum'][idx] < 0:
                sell_conditions.append(True)
                momentum_strength = min(1.0, -indicators['momentum'][idx] * 20)
                sell_strength_factors.append(momentum_strength)
            
            # 至少满足3个条件才生成卖出信号
            if sum(sell_conditions) >= 3:
                # 计算综合信号强度
                strength = np.mean(sell_strength_factors)
                strength = max(0.6, min(1.0, strength))  # 限制在0.6-1.0范围
                
                # 生成分析结果
                analysis_result = {
                    "专精特新度": np.random.uniform(0.4, 0.7),
                    "成长性": np.random.uniform(0.3, 0.6),
                    "估值": np.random.uniform(0.6, 0.9),
                    "技术面": strength,
                    "动量": indicators['momentum'][idx],
                    "RSI": indicators['rsi'][idx],
                    "MACD": indicators['macd_histogram'][idx],
                    "布林带位置": 0.5  # 默认值，避免Series计算问题
                }
                
                # 生成市场环境
                recent_market_returns = np.mean([
                    technical_indicators[s]['returns'][max(0, idx-5):idx].mean() 
                    for s in stocks
                ])
                
                if recent_market_returns > 0.008:
                    market_environment = 'bull'
                elif recent_market_returns < -0.008:
                    market_environment = 'bear'
                else:
                    market_environment = 'neutral'
                
                # 根据市场环境调整信号生成概率
                signal_probability = 0.4  # 基础40%概率（卖出信号概率略低）
                if market_environment == 'bear':
                    signal_probability *= 1.2  # 熊市增加20%概率
                elif market_environment == 'bull':
                    signal_probability *= 0.8  # 牛市减少20%概率
                
                if np.random.random() < signal_probability:
                    signals.append({
                        'date': date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date),
                        'symbol': stock,
                        'action': 'sell',
                        'strength': strength,
                        'position_size': 0,  # 将由系统计算
                        'reason': f"技术分析-卖出信号-强度{strength:.2f}",
                        'analysis_result': analysis_result,
                        'market_environment': market_environment,
                        'source': 'technical_analysis'
                    })
    
    return signals

def create_final_optimized_config():
    """创建最终优化的回测配置，平衡风险与收益"""
    return EnhancedBacktestConfig(
        start_date="2023-01-01",
        end_date="2023-12-31",
        initial_cash=1000000,
        commission_ratio=0.0001,
        slippage_ratio=0.0001,
        benchmark="SHSE.000300",
        # 优化仓位管理
        max_position_size=0.3,  # 最大仓位30%
        min_position_size=0.05,  # 最小仓位5%
        # 优化风险控制
        stop_loss=-0.05,  # 止损-5%
        take_profit=0.25,  # 止盈25%
        max_drawdown_limit=0.15,  # 最大回撤15%
        # 市场环境调整
        market_env_adjustment=True,
        # 启用风险控制
        risk_control_enabled=True
    )

def run_final_optimized_backtest():
    """运行最终优化后的回测"""
    # 创建最终优化的回测配置
    config = create_final_optimized_config()
    
    # 创建增强版回测系统
    backtest = EnhancedBacktestEngine(config)
    
    # 获取股票代码列表
    stocks = ['600000.SH', '000001.SZ', '002415.SZ', '300750.SZ', '688981.SH']
    
    # 使用回测系统加载价格数据，确保格式一致
    print("加载价格数据...")
    price_data = backtest.load_price_data(stocks)
    
    print("生成优化版交易信号...")
    signals = create_optimized_signals(price_data)
    
    print(f"生成了 {len(signals)} 个交易信号")
    
    # 保存交易信号到文件
    signals_file = "结果/trading_signals.json"
    # 保存完整的信号信息，包括日期和操作类型
    signals_dict = {}
    for signal in signals:
        symbol = signal['symbol']
        if symbol not in signals_dict:
            signals_dict[symbol] = []
        signals_dict[symbol].append({
            'date': signal['date'],
            'action': signal['action'],
            'strength': signal['strength']
        })
    
    with open(signals_file, 'w', encoding='utf-8') as f:
        json.dump(signals_dict, f, ensure_ascii=False, indent=2)
    
    print(f"交易信号已保存到: {signals_file}")
    
    # 运行回测
    print("开始最终优化回测...")
    results = backtest.run_enhanced_backtest(signals)
    
    # 打印结果
    print("\n最终优化回测结果:")
    portfolio = results.get("portfolio", {})
    performance_metrics = results.get("performance_metrics", {})
    
    print(f"最终资产价值: {portfolio.get('final_value', 0):,.2f}")
    print(f"总收益率: {performance_metrics.get('total_return', 0):.2%}")
    print(f"年化收益率: {performance_metrics.get('annualized_return', 0):.2%}")
    print(f"最大回撤: {performance_metrics.get('max_drawdown', 0):.2%}")
    print(f"夏普比率: {performance_metrics.get('sharpe_ratio', 0):.2f}")
    print(f"总交易次数: {performance_metrics.get('total_trades', 0)}")
    print(f"胜率: {performance_metrics.get('win_rate', 0):.2%}")
    print(f"盈亏比: {performance_metrics.get('profit_loss_ratio', 0):.2f}")
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"结果/final_optimized_backtest_results_{timestamp}.json"
    
    # 确保结果目录存在
    os.makedirs("结果", exist_ok=True)
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"回测结果已保存到: {results_file}")
    
    # 绘制图表
    try:
        # 这里可以添加图表绘制代码
        print("图表绘制功能待实现")
    except Exception as e:
        print(f"绘制图表失败: {e}")
    
    # 检查是否达到目标
    annual_return = performance_metrics.get('annualized_return', 0)
    max_drawdown = performance_metrics.get('max_drawdown', 0)
    sharpe_ratio = performance_metrics.get('sharpe_ratio', 0)
    
    print("\n=== 目标达成情况 ===")
    if annual_return >= 0.20:
        print(f"[OK] 年化收益率目标达成: {annual_return:.2%} ≥ 20%")
    else:
        print(f"[FAIL] 年化收益率目标未达成: {annual_return:.2%} < 20%")
    
    if max_drawdown <= 0.15:
        print(f"[OK] 最大回撤控制良好: {max_drawdown:.2%} ≤ 15%")
    else:
        print(f"[WARN] 最大回撤超出预期: {max_drawdown:.2%} > 15%")
    
    if sharpe_ratio >= 1.0:
        print(f"[OK] 夏普比率良好: {sharpe_ratio:.2f} ≥ 1.0")
    else:
        print(f"[WARN] 夏普比率偏低: {sharpe_ratio:.2f} < 1.0")
    
    return results

if __name__ == "__main__":
    run_final_optimized_backtest()