"""
优化版专精特新文本分析交易框架回测示例
演示如何使用优化版系统进行回测
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from optimized_main import OptimizedSpecializedInnovativeFramework
from optimized_signal_generator import OptimizedSignalGenerator
from optimized_backtest_system import OptimizedBacktestEngine, OptimizedBacktestConfig


def create_sample_data():
    """创建示例数据用于回测"""
    
    # 创建股票列表
    stock_codes = ["000001", "000002", "000858", "002415", "300750"]
    stock_names = ["平安银行", "万科A", "五粮液", "海康威视", "宁德时代"]
    
    # 创建价格数据
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    price_data = []
    
    for i, (code, name) in enumerate(zip(stock_codes, stock_names)):
        # 生成随机价格走势
        base_price = 10 + i * 5
        dates = []
        prices = []
        volumes = []
        
        current_date = start_date
        current_price = base_price
        
        while current_date <= end_date:
            # 跳过周末
            if current_date.weekday() < 5:
                dates.append(current_date.strftime("%Y-%m-%d"))
                
                # 随机价格变动
                change = np.random.normal(0, 0.02)  # 2%标准差
                current_price *= (1 + change)
                
                # 确保价格为正
                current_price = max(current_price, 1.0)
                
                prices.append(round(current_price, 2))
                volumes.append(np.random.randint(1000000, 10000000))
            
            current_date += timedelta(days=1)
        
        # 创建DataFrame
        df = pd.DataFrame({
            "date": dates,
            "close": prices,
            "volume": volumes
        })
        
        price_data.append({
            "stock_code": code,
            "stock_name": name,
            "price_data": df
        })
    
    return price_data


def create_sample_signals(price_data):
    """创建示例交易信号"""
    
    signals = []
    
    for stock in price_data:
        stock_code = stock["stock_code"]
        price_df = stock["price_data"]
        
        # 为每个交易日生成信号
        for i, row in price_df.iterrows():
            if i < 20:  # 前20天不生成信号
                continue
            
            # 获取最近20天的价格
            recent_prices = price_df.iloc[i-20:i+1]["close"].values
            
            # 计算移动平均线
            ma5 = np.mean(recent_prices[-5:])
            ma20 = np.mean(recent_prices)
            
            # 生成信号
            if ma5 > ma20 * 1.02:  # 短期均线上穿长期均线
                signal_strength = min(0.8, 0.2 + (ma5 / ma20 - 1) * 10)
                signal_type = "buy"
            elif ma5 < ma20 * 0.98:  # 短期均线下穿长期均线
                signal_strength = max(-0.8, -0.2 - (ma5 / ma20 - 1) * 10)
                signal_type = "sell"
            else:
                signal_strength = 0
                signal_type = "hold"
            
            # 添加随机噪声
            signal_strength += np.random.normal(0, 0.05)
            signal_strength = max(-1, min(1, signal_strength))
            
            signals.append({
                "stock_code": stock_code,
                "date": row["date"],
                "signal": signal_strength,
                "signal_type": signal_type,
                "price": row["close"]
            })
    
    return signals


def run_backtest_example():
    """运行回测示例"""
    
    print("=" * 60)
    print("优化版专精特新文本分析交易框架回测示例")
    print("=" * 60)
    
    # 创建示例数据
    print("\n1. 创建示例数据...")
    price_data = create_sample_data()
    signals = create_sample_signals(price_data)
    
    print(f"   创建了{len(price_data)}只股票的价格数据")
    print(f"   创建了{len(signals)}个交易信号")
    
    # 初始化优化版回测引擎
    print("\n2. 初始化优化版回测引擎...")
    
    config = OptimizedBacktestConfig(
        initial_capital=1000000,  # 100万初始资金
        commission_rate=0.001,    # 0.1%手续费
        slippage_rate=0.001,      # 0.1%滑点
        max_position_size=0.2,    # 单只股票最大仓位20%
        stop_loss_pct=0.08,       # 8%止损
        take_profit_pct=0.15,     # 15%止盈
        max_drawdown_pct=0.1,     # 最大回撤10%
        max_daily_loss_pct=0.03   # 单日最大亏损3%
    )
    
    backtest_engine = OptimizedBacktestEngine(config)
    
    # 加载价格数据
    print("\n3. 加载价格数据...")
    for stock in price_data:
        backtest_engine.load_price_data(
            stock_code=stock["stock_code"],
            price_data=stock["price_data"]
        )
    
    # 加载基准数据（使用第一只股票作为基准）
    benchmark_data = price_data[0]["price_data"].copy()
    benchmark_data["benchmark"] = benchmark_data["close"]
    backtest_engine.load_benchmark_data(benchmark_data)
    
    # 执行回测
    print("\n4. 执行回测...")
    backtest_results = backtest_engine.run_backtest(
        signals=signals,
        start_date="2022-01-01",
        end_date="2023-12-31"
    )
    
    # 显示回测结果
    print("\n5. 回测结果:")
    print("-" * 40)
    
    performance_metrics = backtest_results["performance_metrics"]
    print(f"总收益率: {performance_metrics['total_return']:.2%}")
    print(f"年化收益率: {performance_metrics['annualized_return']:.2%}")
    print(f"最大回撤: {performance_metrics['max_drawdown']:.2%}")
    print(f"夏普比率: {performance_metrics['sharpe_ratio']:.2f}")
    print(f"胜率: {performance_metrics['win_rate']:.2%}")
    print(f"总交易次数: {performance_metrics['total_trades']}")
    print(f"盈利交易次数: {performance_metrics['winning_trades']}")
    print(f"亏损交易次数: {performance_metrics['losing_trades']}")
    
    # 显示交易记录（前10条）
    print("\n6. 交易记录（前10条）:")
    print("-" * 40)
    
    trade_records = backtest_results["trade_records"]
    for i, record in enumerate(trade_records[:10]):
        print(f"{i+1}. {record['date']} {record['stock_code']} "
              f"{record['action']} {record['quantity']}股 "
              f"@{record['price']:.2f} 金额:{record['amount']:.2f}")
    
    # 保存回测结果
    print("\n7. 保存回测结果...")
    
    # 创建结果目录
    results_dir = "回测结果"
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存交易记录
    trades_df = pd.DataFrame(trade_records)
    trades_file = os.path.join(results_dir, f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    trades_df.to_csv(trades_file, index=False)
    print(f"   交易记录已保存至: {trades_file}")
    
    # 保存每日净值
    portfolio_values = backtest_results["portfolio_values"]
    portfolio_df = pd.DataFrame(portfolio_values)
    portfolio_file = os.path.join(results_dir, f"portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    portfolio_df.to_csv(portfolio_file, index=False)
    print(f"   每日净值已保存至: {portfolio_file}")
    
    # 生成回测报告
    report_file = os.path.join(results_dir, f"backtest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# 优化版专精特新文本分析交易框架回测报告\n\n")
        f.write(f"回测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## 回测配置\n\n")
        f.write(f"- 初始资金: {config.initial_capital:,.0f}元\n")
        f.write(f"- 手续费率: {config.commission_rate:.2%}\n")
        f.write(f"- 滑点率: {config.slippage_rate:.2%}\n")
        f.write(f"- 单只股票最大仓位: {config.max_position_size:.2%}\n")
        f.write(f"- 止损比例: {config.stop_loss_pct:.2%}\n")
        f.write(f"- 止盈比例: {config.take_profit_pct:.2%}\n")
        f.write(f"- 最大回撤限制: {config.max_drawdown_pct:.2%}\n")
        f.write(f"- 单日最大亏损限制: {config.max_daily_loss_pct:.2%}\n\n")
        f.write("## 回测结果\n\n")
        f.write(f"- 总收益率: {performance_metrics['total_return']:.2%}\n")
        f.write(f"- 年化收益率: {performance_metrics['annualized_return']:.2%}\n")
        f.write(f"- 最大回撤: {performance_metrics['max_drawdown']:.2%}\n")
        f.write(f"- 夏普比率: {performance_metrics['sharpe_ratio']:.2f}\n")
        f.write(f"- 胜率: {performance_metrics['win_rate']:.2%}\n")
        f.write(f"- 总交易次数: {performance_metrics['total_trades']}\n")
        f.write(f"- 盈利交易次数: {performance_metrics['winning_trades']}\n")
        f.write(f"- 亏损交易次数: {performance_metrics['losing_trades']}\n\n")
        f.write("## 优化策略说明\n\n")
        f.write("本次回测使用了优化版交易策略，包含以下六项优化:\n\n")
        f.write("1. 调整信号阈值：提高买入信号阈值，降低卖出信号敏感度\n")
        f.write("2. 增加持仓管理：引入8%止损、15%止盈机制\n")
        f.write("3. 优化资金分配：根据信号强度调整仓位大小(5%-20%)\n")
        f.write("4. 增加市场环境判断：在整体市场下跌时降低仓位\n")
        f.write("5. 完善风控机制：设置最大回撤10%、单日最大亏损3%限制\n")
        f.write("6. 结合Barra CNE5模型：小盘股、价值和动量因子正向调整\n\n")
        f.write("## 文件说明\n\n")
        f.write(f"- 交易记录: {trades_file}\n")
        f.write(f"- 每日净值: {portfolio_file}\n")
    
    print(f"   回测报告已保存至: {report_file}")
    
    print("\n" + "=" * 60)
    print("回测示例完成！")
    print("=" * 60)


def run_integrated_backtest_example():
    """运行整合分析回测示例"""
    
    print("=" * 60)
    print("优化版专精特新文本分析交易框架整合分析回测示例")
    print("=" * 60)
    
    # 初始化框架
    print("\n1. 初始化优化版框架...")
    framework = OptimizedSpecializedInnovativeFramework(use_api=False)  # 不使用API，使用示例数据
    
    # 运行整合分析和回测
    print("\n2. 运行整合分析和回测...")
    results = framework.run_integrated_analysis_with_backtest(
        phases=["long_term", "mid_term", "short_term"]
    )
    
    # 显示结果
    print("\n3. 分析结果:")
    print("-" * 40)
    
    # 显示各期分析统计
    if "long_term" in results and results["long_term"]:
        stats = results["long_term"].get("statistics", {})
        print(f"长期分析: 评估企业数量 {stats.get('evaluated_companies', 0)}")
    
    if "mid_term" in results and results["mid_term"]:
        stats = results["mid_term"].get("statistics", {})
        print(f"中期分析: 预期差分析数量 {stats.get('total_gaps', 0)}")
    
    if "short_term" in results and results["short_term"]:
        stats = results["short_term"].get("statistics", {})
        print(f"短期分析: 风险预警总数 {stats.get('total_risk_alerts', 0)}")
    
    # 显示交易信号统计
    if "integrated_signals" in results and results["integrated_signals"]:
        signals = results["integrated_signals"]
        print(f"\n交易信号统计:")
        print(f"  生成信号数量: {len(signals.get('signals', []))}")
        
        # 统计买入和卖出信号
        buy_signals = sum(1 for s in signals.get('signals', []) if s.get('signal', 0) > 0.2)
        sell_signals = sum(1 for s in signals.get('signals', []) if s.get('signal', 0) < -0.3)
        print(f"  买入信号数量: {buy_signals}")
        print(f"  卖出信号数量: {sell_signals}")
    
    # 显示回测结果
    if "backtest_results" in results and results["backtest_results"]:
        backtest = results["backtest_results"]
        performance = backtest.get("performance_metrics", {})
        print(f"\n回测结果:")
        print(f"  总收益率: {performance.get('total_return', 0):.2%}")
        print(f"  年化收益率: {performance.get('annualized_return', 0):.2%}")
        print(f"  最大回撤: {performance.get('max_drawdown', 0):.2%}")
        print(f"  夏普比率: {performance.get('sharpe_ratio', 0):.2f}")
        print(f"  胜率: {performance.get('win_rate', 0):.2%}")
    
    # 显示报告路径
    if "report_path" in results:
        print(f"\n综合报告: {results['report_path']}")
    
    print("\n" + "=" * 60)
    print("整合分析回测示例完成！")
    print("=" * 60)


if __name__ == "__main__":
    # 运行简单回测示例
    run_backtest_example()
    
    print("\n\n")
    
    # 运行整合分析回测示例
    run_integrated_backtest_example()