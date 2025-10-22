"""
交易策略命令行管理工具
提供简单的命令行界面，用于管理和监控基于新闻分析的量化交易策略
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from typing import Dict, Any

# 导入交易管理器
from trading_manager import TradingManager, TradingConfig

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def print_status(status: Dict[str, Any]):
    """打印策略状态"""
    print("\n========== 策略状态 ==========")
    print(f"运行状态: {'运行中' if status.get('is_running', False) else '已停止'}")
    print(f"策略ID: {status.get('config', {}).get('strategy_id', '未知')}")
    print(f"策略文件: {status.get('config', {}).get('strategy_file', '未知')}")
    print(f"运行模式: {status.get('config', {}).get('mode', '未知')}")
    
    if status.get('start_time'):
        print(f"启动时间: {status['start_time']}")
    
    if status.get('stop_time'):
        print(f"停止时间: {status['stop_time']}")
    
    if status.get('last_news_analysis'):
        print(f"上次新闻分析: {status['last_news_analysis']}")
    
    if status.get('last_position_evaluation'):
        print(f"上次持仓评估: {status['last_position_evaluation']}")
    
    print("==============================\n")


def print_signals(signals: Dict[str, Any]):
    """打印交易信号"""
    print("\n========== 交易信号 ==========")
    
    if not signals:
        print("无交易信号")
        print("==============================\n")
        return
    
    for symbol, signal in signals.items():
        strength = signal.get('strength', 0)
        timestamp = signal.get('timestamp', '未知时间')
        
        # 根据信号强度确定显示颜色
        if strength > 0.1:
            direction = "买入"
            strength_str = f"+{strength:.2f}"
        elif strength < -0.1:
            direction = "卖出"
            strength_str = f"{strength:.2f}"
        else:
            direction = "持有"
            strength_str = f"{strength:.2f}"
        
        print(f"{symbol}: {direction} ({strength_str}) - {timestamp}")
    
    print("==============================\n")


def cmd_start(args):
    """启动策略命令"""
    try:
        # 加载配置
        config = TradingConfig()
        
        # 如果提供了配置文件，则从文件加载
        if args.config:
            with open(args.config, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
                for key, value in config_dict.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
        
        # 创建交易管理器
        manager = TradingManager(config)
        
        # 启动策略
        if manager.start_strategy():
            print("策略启动成功")
            
            # 显示状态
            status = manager.get_strategy_status()
            print_status(status)
        else:
            print("策略启动失败")
            
    except Exception as e:
        print(f"启动策略时发生错误: {e}")


def cmd_stop(args):
    """停止策略命令"""
    try:
        # 加载配置
        config = TradingConfig()
        
        # 如果提供了配置文件，则从文件加载
        if args.config:
            with open(args.config, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
                for key, value in config_dict.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
        
        # 创建交易管理器
        manager = TradingManager(config)
        
        # 停止策略
        if manager.stop_strategy():
            print("策略停止成功")
            
            # 显示状态
            status = manager.get_strategy_status()
            print_status(status)
        else:
            print("策略停止失败")
            
    except Exception as e:
        print(f"停止策略时发生错误: {e}")


def cmd_status(args):
    """查看状态命令"""
    try:
        # 加载配置
        config = TradingConfig()
        
        # 如果提供了配置文件，则从文件加载
        if args.config:
            with open(args.config, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
                for key, value in config_dict.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
        
        # 创建交易管理器
        manager = TradingManager(config)
        
        # 获取状态
        status = manager.get_strategy_status()
        print_status(status)
        
    except Exception as e:
        print(f"获取状态时发生错误: {e}")


def cmd_analyze(args):
    """分析新闻命令"""
    try:
        # 加载配置
        config = TradingConfig()
        
        # 如果提供了配置文件，则从文件加载
        if args.config:
            with open(args.config, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
                for key, value in config_dict.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
        
        # 创建交易管理器
        manager = TradingManager(config)
        
        # 运行新闻分析
        if manager.run_news_analysis():
            print("新闻分析完成")
            
            # 生成交易信号
            signals = manager.generate_trading_signals()
            print_signals(signals)
        else:
            print("新闻分析失败")
            
    except Exception as e:
        print(f"分析新闻时发生错误: {e}")


def cmd_signals(args):
    """生成交易信号命令"""
    try:
        # 加载配置
        config = TradingConfig()
        
        # 如果提供了配置文件，则从文件加载
        if args.config:
            with open(args.config, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
                for key, value in config_dict.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
        
        # 创建交易管理器
        manager = TradingManager(config)
        
        # 生成交易信号
        signals = manager.generate_trading_signals()
        print_signals(signals)
        
    except Exception as e:
        print(f"生成交易信号时发生错误: {e}")


def cmd_config(args):
    """生成配置文件命令"""
    try:
        # 创建默认配置
        config = TradingConfig()
        config_dict = {
            'strategy_id': config.strategy_id,
            'strategy_file': config.strategy_file,
            'token': config.token,
            'mode': config.mode,
            'backtest_start_time': config.backtest_start_time,
            'backtest_end_time': config.backtest_end_time,
            'backtest_adjust': config.backtest_adjust,
            'backtest_initial_cash': config.backtest_initial_cash,
            'backtest_commission_ratio': config.backtest_commission_ratio,
            'backtest_slippage_ratio': config.backtest_slippage_ratio,
            'backtest_match_mode': config.backtest_match_mode,
            'target_symbols': config.target_symbols,
            'max_position_ratio': config.max_position_ratio,
            'rebalance_frequency': config.rebalance_frequency,
            'news_analysis_time': config.news_analysis_time,
            'position_evaluation_time': config.position_evaluation_time
        }
        
        # 保存配置文件
        config_file = args.output or 'trading_config.json'
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=2)
        
        print(f"配置文件已生成: {config_file}")
        print("请根据需要修改配置文件中的参数，特别是token参数")
        
    except Exception as e:
        print(f"生成配置文件时发生错误: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='基于新闻分析的量化交易策略管理工具')
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 启动策略命令
    start_parser = subparsers.add_parser('start', help='启动交易策略')
    start_parser.add_argument('--config', '-c', help='配置文件路径')
    start_parser.set_defaults(func=cmd_start)
    
    # 停止策略命令
    stop_parser = subparsers.add_parser('stop', help='停止交易策略')
    stop_parser.add_argument('--config', '-c', help='配置文件路径')
    stop_parser.set_defaults(func=cmd_stop)
    
    # 查看状态命令
    status_parser = subparsers.add_parser('status', help='查看策略状态')
    status_parser.add_argument('--config', '-c', help='配置文件路径')
    status_parser.set_defaults(func=cmd_status)
    
    # 分析新闻命令
    analyze_parser = subparsers.add_parser('analyze', help='分析新闻并生成交易信号')
    analyze_parser.add_argument('--config', '-c', help='配置文件路径')
    analyze_parser.set_defaults(func=cmd_analyze)
    
    # 生成交易信号命令
    signals_parser = subparsers.add_parser('signals', help='生成交易信号')
    signals_parser.add_argument('--config', '-c', help='配置文件路径')
    signals_parser.set_defaults(func=cmd_signals)
    
    # 生成配置文件命令
    config_parser = subparsers.add_parser('config', help='生成配置文件')
    config_parser.add_argument('--output', '-o', help='输出文件路径')
    config_parser.set_defaults(func=cmd_config)
    
    # 解析参数
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # 执行命令
    args.func(args)


if __name__ == '__main__':
    main()