import pandas as pd
import numpy as np
from enhanced_backtest_system import EnhancedBacktestEngine, EnhancedBacktestConfig
import json

# 加载交易信号
with open('结果/trading_signals.json', 'r') as f:
    signals = json.load(f)

# 创建增强版回测配置
config = EnhancedBacktestConfig(
    start_date='2023-01-01',
    end_date='2023-12-31',
    initial_cash=1000000,
    commission_ratio=0.001,
    slippage_ratio=0.001,
    max_position_size=0.3,
    stop_loss=-0.05,
    take_profit=0.25,
    max_drawdown_limit=0.15
)

# 创建回测引擎
engine = EnhancedBacktestEngine(config)

# 转换信号格式
signals_list = []
for symbol, signal_list in signals.items():
    for signal_info in signal_list:
        signals_list.append({
            'symbol': symbol,
            'date': signal_info['date'],
            'action': signal_info['action'],
            'strength': signal_info['strength'],
            'position_size': 0,
            'reason': f"技术分析-{signal_info['action']}-强度{signal_info['strength']:.2f}",
            'market_environment': 'neutral',
            'source': 'technical_analysis'
        })

# 从信号中提取股票代码
symbols = list(set(signal['symbol'] for signal in signals_list))

# 加载价格数据
price_data = engine.load_price_data(symbols)

# 按日期分组信号
signals_by_date = {}
for signal in signals_list:
    date = signal['date']
    if date not in signals_by_date:
        signals_by_date[date] = []
    signals_by_date[date].append(signal)

# 获取所有交易日期
all_dates = sorted(set(
    list(signals_by_date.keys()) + 
    [d.strftime('%Y-%m-%d') for d in pd.date_range(
        start=config.start_date, 
        end=config.end_date, 
        freq='D'
    ) if d.weekday() < 5]  # 只包含工作日
))

# 检查前5个信号的处理
print("检查前5个信号的处理:")
for i, date in enumerate(sorted(signals_by_date.keys())[:5]):
    print(f"\n日期: {date}")
    if date in signals_by_date:
        for signal in signals_by_date[date]:
            symbol = signal['symbol']
            action = signal['action']
            strength = signal['strength']
            
            print(f"  信号: {symbol} {action} 强度={strength}")
            
            # 检查价格数据
            if symbol in price_data:
                # 将字符串日期转换为datetime对象
                date_obj = pd.to_datetime(date)
                
                # 检查日期是否在价格数据中
                if date_obj in price_data[symbol].index:
                    current_price = price_data[symbol]['close'].loc[date_obj]
                    print(f"    价格: {current_price}")
                else:
                    # 尝试使用字符串格式查找
                    try:
                        date_str = date_obj.strftime('%Y-%m-%d')
                        if date_str in price_data[symbol].index.astype(str):
                            idx = np.where(price_data[symbol].index.astype(str) == date_str)[0][0]
                            current_price = price_data[symbol].iloc[idx]['close']
                            print(f"    价格(字符串查找): {current_price}")
                        else:
                            print(f"    错误: 找不到价格数据")
                            continue
                    except Exception as e:
                        print(f"    错误: {e}")
                        continue
                
                # 计算仓位大小
                market_env = 0.0  # 假设中性市场
                position_size = engine.calculate_enhanced_position_size(
                    strength, market_env, engine.portfolio.total_value, 
                    current_price, symbol,
                    pd.DataFrame({s: df['close'] for s, df in price_data.items()})
                )
                print(f"    仓位大小: {position_size}")
                
                if position_size == 0:
                    print(f"    跳过: 仓位大小为0")
                    continue
                
                # 计算交易数量
                if action == "buy":
                    available_cash = engine.portfolio.cash * 0.95
                    max_value = available_cash * position_size
                    quantity = int(max_value / (current_price * (1 + config.slippage_ratio)))
                    print(f"    可用现金: {available_cash}, 最大价值: {max_value}, 数量: {quantity}")
                else:
                    quantity = 0
                    print(f"    卖出信号，但无持仓")
                
                if quantity <= 0:
                    print(f"    跳过: 数量<=0")
                    continue
                
                print(f"    应该执行交易: {action} {symbol} {quantity}股")
            else:
                print(f"    错误: 找不到股票价格数据")