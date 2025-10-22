#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
完整回测测试脚本
运行增强版回测系统并检查结果
"""

import sys
import os
import json
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_backtest_system import EnhancedBacktestEngine, EnhancedBacktestConfig

def main():
    print("=" * 50)
    print("运行增强版回测系统")
    print("=" * 50)
    
    # 创建增强版配置
    config = EnhancedBacktestConfig(
        start_date='2023-01-01',
        end_date='2023-12-31',
        initial_cash=1000000,  # 使用initial_cash而不是initial_capital
        max_position_size=0.3,  # 提高最大仓位至30%
        min_position_size=0.05,  # 降低最小仓位至5%
        max_total_position=0.9,  # 最大总仓位90%
        stop_loss=-0.05,  # 收紧止损至-5%
        take_profit=0.20,  # 提高止盈至20%
        max_drawdown_limit=0.15,  # 提高最大回撤限制至15%
        risk_control_enabled=True,  # 启用风险控制
        commission_ratio=0.001,  # 降低手续费至0.1%
        slippage_ratio=0.001  # 降低滑点至0.1%
    )
    
    # 创建增强版回测引擎
    engine = EnhancedBacktestEngine(config)
    
    # 运行回测
    print("开始运行回测...")
    
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
    print(f"总收益率: {metrics.get('total_return_rate', 0):.2%}")
    print(f"年化收益率: {metrics.get('annual_return_rate', 0):.2%}")
    print(f"最大回撤: {metrics.get('max_drawdown_rate', 0):.2%}")
    print(f"夏普比率: {metrics.get('sharpe', 0):.2f}")
    print(f"胜率: {metrics.get('win_ratio', 0):.2%}")
    print(f"盈亏比: {metrics.get('profit_loss_ratio', 0):.2f}")
    
    # 保存结果
    with open('enhanced_backtest_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n结果已保存到 enhanced_backtest_results.json")
    
    # 检查是否达到目标
    metrics = results['performance_metrics']
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
    if 'trades' in results and len(results['trades']) > 0:
        print(f"\n交易记录分析:")
        print(f"总交易数: {len(results['trades'])}")
        
        buy_trades = [t for t in results['trades'] if t['action'] == 'buy']
        sell_trades = [t for t in results['trades'] if t['action'] == 'sell']
        
        print(f"买入交易: {len(buy_trades)}")
        print(f"卖出交易: {len(sell_trades)}")
        
        # 显示前5个交易
        print("\n前5个交易:")
        for i, trade in enumerate(results['trades'][:5]):
            print(f"  {i+1}. {trade.get('timestamp', trade.get('date', 'N/A'))} {trade['action']} {trade['symbol']} "
                  f"{trade['quantity']}股 @ {trade['price']:.2f}")
    else:
        print("\n⚠️ 警告: 没有找到交易记录")

if __name__ == "__main__":
    main()