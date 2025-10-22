import pandas as pd
import json

# 检查交易信号的日期格式
with open('结果/trading_signals.json', 'r') as f:
    signals = json.load(f)

print('交易信号日期示例:')
for symbol, signal_list in list(signals.items())[:1]:
    for signal in signal_list[:3]:
        print(f'  {symbol}: {signal["date"]} (类型: {type(signal["date"])})')

# 检查价格数据的日期格式
from optimized_backtest_system import OptimizedBacktestEngine, OptimizedBacktestConfig
config = OptimizedBacktestConfig(start_date='2023-01-01', end_date='2023-12-31')
engine = OptimizedBacktestEngine(config)
price_data = engine.load_price_data(list(signals.keys())[:1])

print('\n价格数据索引示例:')
symbol = list(signals.keys())[0]
print(f'  {symbol} 索引类型: {type(price_data[symbol].index)}')
print(f'  前3个索引: {price_data[symbol].index[:3].tolist()}')
print(f'  索引[0]类型: {type(price_data[symbol].index[0])}')

# 检查信号日期是否在价格数据索引中
signal_date = signals[list(signals.keys())[0]][0]["date"]
print(f'\n检查信号日期 {signal_date} 是否在价格数据索引中:')
print(f'  直接查找: {signal_date in price_data[symbol].index}')
print(f'  转换为Timestamp后查找: {pd.to_datetime(signal_date) in price_data[symbol].index}')

# 检查索引的字符串表示
print(f'\n价格数据索引的字符串表示:')
print(f'  前3个索引的字符串: {[str(idx) for idx in price_data[symbol].index[:3]]}')
print(f'  信号日期字符串: {signal_date}')
print(f'  信号日期是否在字符串索引中: {signal_date in [str(idx) for idx in price_data[symbol].index]}')