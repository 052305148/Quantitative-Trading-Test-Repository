import json
import os
from datetime import datetime
from enhanced_backtest_system import EnhancedBacktestEngine, EnhancedBacktestConfig

def analyze_trades(trades):
    """分析交易记录"""
    print("\n交易记录分析:")
    print(f"总交易数: {len(trades)}")
    
    if not trades:
        print("没有交易记录")
        return
    
    buy_trades = [t for t in trades if t['action'] == 'buy']
    sell_trades = [t for t in trades if t['action'] == 'sell']
    
    print(f"买入交易: {len(buy_trades)}")
    print(f"卖出交易: {len(sell_trades)}")
    
    print("\n前5个交易:")
    for i, trade in enumerate(trades[:5]):
        print(f"  {i+1}. {trade.get('timestamp', trade.get('date', 'N/A'))} {trade['action']} {trade['symbol']} "
              f"{trade['quantity']}股 @ {trade['price']:.2f}")

def main():
    # 配置回测参数 - 最终优化版本
    config = EnhancedBacktestConfig(
        start_date="2023-01-01",
        end_date="2023-12-31",
        initial_cash=1000000,  # 100万初始资金
        commission_ratio=0.001,  # 0.1%手续费
        slippage_ratio=0.001,  # 0.1%滑点
        
        # 优化版参数
        max_position_size=0.22,  # 单只股票最大仓位22%
        min_position_size=0.025,  # 最小仓位2.5%
        max_drawdown_limit=0.18,  # 最大回撤限制18%
        stop_loss=-0.09,  # 止损-9%
        take_profit=0.18,  # 止盈18%
        
        # 增强版参数
        market_env_adjustment=True,  # 市场环境调整
        risk_control_enabled=True,  # 风险控制
        max_total_position=0.85,  # 最大总仓位85%
        dynamic_position_sizing=True,  # 动态仓位调整
        volatility_adjustment=True,  # 波动率调整
        correlation_adjustment=True,  # 相关性调整
        bull_market_multiplier=1.6,  # 牛市仓位倍数
        bear_market_multiplier=0.55,  # 熊市仓位倍数
        max_daily_loss=0.055  # 最大日损失5.5%
    )
    
    # 初始化引擎
    engine = EnhancedBacktestEngine(config)
    
    # 运行回测
    print("开始运行最终优化版回测...")
    
    # 加载交易信号
    with open('结果/trading_signals.json', 'r', encoding='utf-8') as f:
        signals = json.load(f)
    
    results = engine.run_enhanced_backtest(signals)
    
    # 打印结果
    print("\n回测结果:")
    print(f"最终总价值: {results['portfolio']['final_value']:,.2f}")
    print(f"现金: {results['portfolio']['cash']:,.2f}")
    print(f"持仓数量: {len(results['portfolio']['positions'])}")
    print(f"交易次数: {len(results['trades'])}")
    
    # 打印绩效指标
    metrics = results['performance_metrics']
    print(f"\n绩效指标:")
    print(f"总收益率: {metrics.get('total_return', 0):.2%}")
    print(f"年化收益率: {metrics.get('annualized_return', 0):.2%}")
    print(f"最大回撤: {metrics.get('max_drawdown', 0):.2%}")
    print(f"夏普比率: {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"胜率: {metrics.get('win_rate', 0):.2%}")
    print(f"盈亏比: {metrics.get('profit_loss_ratio', 0):.2f}")
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"结果/final_optimized_backtest_results_{timestamp}.json"
    
    # 确保结果目录存在
    os.makedirs("结果", exist_ok=True)
    
    # 保存结果到JSON文件
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n结果已保存到: {results_file}")
    
    # 检查是否达到目标
    if metrics.get('annualized_return', 0) >= 0.20:
        print("\n✅ 年化收益率达到20%以上，优化成功！")
    else:
        print(f"\n❌ 年化收益率未达到20%目标，当前为{metrics.get('annualized_return', 0):.2%}")
        
    # 检查最大回撤
    if metrics.get('max_drawdown', 0) <= -0.15:
        print("✅ 最大回撤控制在15%以内")
    else:
        print(f"❌ 最大回撤超过15%，当前为{metrics.get('max_drawdown', 0):.2%}")
    
    # 分析交易记录
    analyze_trades(results['trades'])
    
    return results

if __name__ == "__main__":
    main()