from trading_manager import TradingManager, TradingConfig
import json

# 加载配置
with open('trading_config.json', 'r', encoding='utf-8') as f:
    config_dict = json.load(f)

# 创建TradingConfig实例
config = TradingConfig(**config_dict)

# 创建交易管理器
try:
    manager = TradingManager(config)
    print('交易管理器创建成功')
    
    # 测试获取策略状态
    status = manager.get_strategy_status()
    print(f'当前策略状态: 运行状态={status["is_running"]}, 策略ID={status["config"]["strategy_id"]}')
    
    # 测试新闻分析
    print('开始测试新闻分析...')
    result = manager.run_news_analysis()
    print(f'新闻分析结果: {result}')
    
    # 测试交易信号生成
    print('开始测试交易信号生成...')
    signals = manager.generate_trading_signals()
    print(f'生成交易信号数量: {len(signals)}')
    
    # 打印部分交易信号
    for symbol, signal in list(signals.items())[:3]:
        print(f'{symbol}: 信号强度={signal["strength"]:.2f}')
    
    print('交易管理器功能测试成功')
except Exception as e:
    print(f'交易管理器测试失败: {e}')
    import traceback
    traceback.print_exc()