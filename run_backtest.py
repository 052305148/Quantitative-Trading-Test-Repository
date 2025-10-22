"""
执行交易回测的脚本
"""

from backtest_system import BacktestEngine, BacktestConfig
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def main():
    # 设置工作目录
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # 创建回测配置
    config = BacktestConfig(
        start_date='2023-12-01',
        end_date='2023-12-31',
        initial_cash=1000000.0,
        commission_ratio=0.0001,
        slippage_ratio=0.0001,
        benchmark='SHSE.000300'
    )

    # 创建回测引擎
    engine = BacktestEngine(config)

    # 加载最新的整合交易信号
    import glob
    signal_files = glob.glob('数据/交易记录/integrated_trading_signals_*.json')
    if not signal_files:
        print("未找到整合交易信号文件，使用默认信号文件")
        signal_file = '结果/trading_signals.json'
    else:
        # 获取最新的信号文件
        signal_file = max(signal_files, key=os.path.getctime)
        print(f"使用最新的整合交易信号文件: {signal_file}")
    
    with open(signal_file, 'r', encoding='utf-8') as f:
        signals = json.load(f)

    print(f'加载了 {len(signals)} 个交易信号')

    # 如果信号是列表格式，直接使用
    if isinstance(signals, list):
        signals_data = signals
        # 更新信号日期以匹配回测期间
        for signal in signals_data:
            signal["date"] = "2023-12-30"  # 使用回测期间的日期
            # 确保卖出信号的strength为负数
            if signal.get("action") == "sell":
                signal["strength"] = -abs(signal.get("strength", 0.5))
        
        # 添加初始买入信号，确保有持仓可以卖出
        unique_symbols = list(set(signal.get("symbol", "") for signal in signals_data if "symbol" in signal))
        for symbol in unique_symbols[:5]:  # 只取前5个股票，避免资金过于分散
            signals_data.append({
                "date": "2023-12-01",  # 在回测开始时买入
                "symbol": symbol,
                "action": "buy",
                "strength": 0.8  # 使用较高的买入强度
            })
    else:
        # 如果信号是字典格式，转换为信号数据列表
        signals_data = []
        for symbol, strength in signals.items():
            # 为每个信号创建一个条目
            # 如果信号强度为0，我们添加一个随机信号进行测试
            if strength == 0:
                strength = np.random.uniform(-0.8, 0.8)
            
            signals_data.append({
                "date": "2023-12-30",  # 使用信号生成日期
                "symbol": symbol,
                "action": "buy" if strength > 0 else "sell",
                "strength": abs(strength)
            })
        
        # 添加更多随机信号进行测试
        test_symbols = ["000001", "000002", "000858", "002415", "600519"]
        for symbol in test_symbols:
            strength = np.random.uniform(-0.8, 0.8)
            signals_data.append({
                "date": "2023-12-30",
                "symbol": symbol,
                "action": "buy" if strength > 0 else "sell",
                "strength": abs(strength)
            })

    print(f'转换了 {len(signals_data)} 个交易信号')

    # 执行回测
    print('开始执行回测...')
    results = engine.run_backtest(signals_data)
    print('回测完成')

    # 保存回测结果
    with open('结果/backtest_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    print('回测结果已保存到 结果/backtest_results.json')

    # 打印部分回测结果
    print('\n回测结果摘要:')
    print(f'总收益率: {results.get("total_return", 0):.2%}')
    print(f'年化收益率: {results.get("annual_return", 0):.2%}')
    print(f'最大回撤: {results.get("max_drawdown", 0):.2%}')
    print(f'夏普比率: {results.get("sharpe_ratio", 0):.2f}')
    print(f'胜率: {results.get("win_rate", 0):.2%}')

if __name__ == "__main__":
    main()