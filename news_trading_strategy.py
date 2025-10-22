# coding=utf-8
from __future__ import print_function, absolute_import
from typing import List, NoReturn, Text
from gm.api import *
from gm.csdk.c_sdk import BarLikeDict2, TickLikeDict2
from gm.model import DictLikeAccountStatus, DictLikeExecRpt, DictLikeIndicator, DictLikeOrder, DictLikeParameter
from gm.pb.account_pb2 import AccountStatus, ExecRpt, Order
from gm.pb.performance_pb2 import Indicator
from gm.pb.rtconf_pb2 import Parameter
from gm.utils import gmsdklogger

from datetime import datetime, timedelta
import sys
import os
import json
import logging

# 添加短期分析模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), '短期分析'))
from short_term_analysis import ShortTermAnalyzer, ShortTermAnalysisConfig

"""
基于新闻分析的量化交易策略
该策略整合了新闻爬虫、文本分类和风险分析系统，根据新闻分析结果进行交易决策
"""

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def init(context):
    # type: (Context) -> NoReturn
    """
    策略初始化函数
    """
    logger.info("初始化基于新闻分析的量化交易策略")
    
    # 初始化新闻分析系统
    context.news_analyzer = None
    try:
        config = ShortTermAnalysisConfig()
        context.news_analyzer = ShortTermAnalyzer(config)
        logger.info("新闻分析系统初始化成功")
    except Exception as e:
        logger.error(f"新闻分析系统初始化失败: {e}")
    
    # 设置定时任务，每天9:00执行新闻分析
    schedule(schedule_func=analyze_news_and_trade, date_rule='1d', time_rule='09:00:00')
    
    # 设置定时任务，每天15:00评估持仓
    schedule(schedule_func=evaluate_positions, date_rule='1d', time_rule='15:00:00')
    
    # 订阅专精特新相关股票的行情
    # 这里使用一些示例专精特新股票代码
    context.target_symbols = [
        'SZSE.000001',  # 平安银行
        'SZSE.000002',  # 万科A
        'SZSE.000858',  # 五粮液
        'SHSE.600000',  # 浦发银行
        'SHSE.600036',  # 招商银行
        'SHSE.600519',  # 贵州茅台
        'SZSE.002415',  # 海康威视
        'SZSE.000725',  # 京东方A
        'SHSE.600276',  # 恒瑞医药
        'SZSE.002594'   # 比亚迪
    ]
    
    # 订阅股票行情，每5分钟更新一次
    subscribe(symbols=','.join(context.target_symbols), frequency='300s')
    
    # 初始化交易决策变量
    context.trade_signals = {}  # 存储交易信号
    context.position_limits = {symbol: 0.1 for symbol in context.target_symbols}  # 每只股票最大持仓比例10%
    
    logger.info(f"策略初始化完成，目标股票数量: {len(context.target_symbols)}")


def analyze_news_and_trade(context):
    # type: (Context) -> NoReturn
    """
    分析新闻并生成交易信号
    """
    logger.info("开始执行每日新闻分析和交易决策")
    
    if not context.news_analyzer:
        logger.error("新闻分析系统未初始化，跳过新闻分析")
        return
    
    try:
        # 爬取新闻
        news_data = context.news_analyzer.crawl_news()
        logger.info(f"爬取到 {len(news_data)} 条新闻")
        
        if not news_data:
            logger.warning("未获取到新闻数据，不进行交易")
            return
        
        # 分类新闻
        classified_news = context.news_analyzer.classify_news(news_data)
        logger.info(f"分类了 {len(classified_news)} 条新闻")
        
        # 分析风险
        risk_events, risk_alerts = context.news_analyzer.analyze_risks(classified_news)
        logger.info(f"识别出 {len(risk_events)} 个风险事件，{len(risk_alerts)} 个风险预警")
        
        # 基于新闻分析结果生成交易信号
        generate_trade_signals(context, classified_news, risk_alerts)
        
        # 执行交易
        execute_trades(context)
        
    except Exception as e:
        logger.error(f"新闻分析和交易决策过程中出错: {e}")


def generate_trade_signals(context, classified_news, risk_alerts):
    # type: (Context, List, List) -> NoReturn
    """
    基于新闻分类和风险预警生成交易信号
    """
    logger.info("开始生成交易信号")
    
    # 重置交易信号
    context.trade_signals = {}
    
    # 统计各类新闻数量
    category_counts = {}
    for news in classified_news:
        category = news.get('category', '其他')
        category_counts[category] = category_counts.get(category, 0) + 1
    
    # 统计风险预警
    high_risk_count = sum(1 for alert in risk_alerts if alert.get('level') == '高')
    medium_risk_count = sum(1 for alert in risk_alerts if alert.get('level') == '中')
    
    # 基于新闻分类和风险预警生成交易信号
    for symbol in context.target_symbols:
        signal_strength = 0
        
        # 技术创新类新闻正面影响
        tech_innovation_count = category_counts.get('技术创新', 0)
        if tech_innovation_count > 0:
            signal_strength += min(tech_innovation_count * 0.1, 0.3)  # 最多增加0.3信号强度
        
        # 市场表现类新闻中性影响
        market_performance_count = category_counts.get('市场表现', 0)
        
        # 政策动态类新闻正面影响
        policy_count = category_counts.get('政策动态', 0)
        if policy_count > 0:
            signal_strength += min(policy_count * 0.15, 0.3)  # 最多增加0.3信号强度
        
        # 公司公告类新闻中性影响
        company_announcement_count = category_counts.get('公司公告', 0)
        
        # 行业动态类新闻轻微正面影响
        industry_count = category_counts.get('行业动态', 0)
        if industry_count > 0:
            signal_strength += min(industry_count * 0.05, 0.2)  # 最多增加0.2信号强度
        
        # 风险预警负面影响
        if high_risk_count > 0:
            signal_strength -= min(high_risk_count * 0.2, 0.5)  # 最多减少0.5信号强度
        if medium_risk_count > 0:
            signal_strength -= min(medium_risk_count * 0.1, 0.3)  # 最多减少0.3信号强度
        
        # 限制信号强度在[-1, 1]范围内
        signal_strength = max(-1, min(1, signal_strength))
        
        # 存储交易信号
        context.trade_signals[symbol] = {
            'strength': signal_strength,
            'category_counts': category_counts,
            'risk_counts': {'high': high_risk_count, 'medium': medium_risk_count}
        }
        
        logger.info(f"股票 {symbol} 交易信号强度: {signal_strength:.2f}")
    
    logger.info("交易信号生成完成")


def execute_trades(context):
    # type: (Context) -> NoReturn
    """
    执行交易
    """
    logger.info("开始执行交易")
    
    # 获取账户信息
    account = context.account()
    if not account:
        logger.error("无法获取账户信息")
        return
    
    cash = account.cash
    total_value = account.market_value + cash
    
    for symbol, signal in context.trade_signals.items():
        try:
            # 获取当前持仓
            position = context.position(symbol=symbol)
            current_position = position.volume if position else 0
            
            # 计算目标持仓
            signal_strength = signal['strength']
            max_position_value = total_value * context.position_limits[symbol]
            target_position_value = max_position_value * signal_strength
            
            # 计算目标股数（向下取整）
            current_price = context.current(symbol).price
            target_shares = int(target_position_value / current_price / 100) * 100  # A股最小交易单位100股
            
            # 计算需要交易的股数
            shares_to_trade = target_shares - current_position
            
            if abs(shares_to_trade) >= 100:  # 至少交易100股
                if shares_to_trade > 0:
                    # 买入
                    order_volume(symbol=symbol, volume=shares_to_trade, side=OrderSide_Buy, 
                                order_type=OrderType_Market, position_effect=PositionEffect_Open)
                    logger.info(f"买入 {symbol} {shares_to_trade} 股，信号强度: {signal_strength:.2f}")
                else:
                    # 卖出
                    order_volume(symbol=symbol, volume=abs(shares_to_trade), side=OrderSide_Sell, 
                                order_type=OrderType_Market, position_effect=PositionEffect_Close)
                    logger.info(f"卖出 {symbol} {abs(shares_to_trade)} 股，信号强度: {signal_strength:.2f}")
            else:
                logger.info(f"股票 {symbol} 无需交易，当前持仓: {current_position}，目标持仓: {target_shares}")
        
        except Exception as e:
            logger.error(f"执行股票 {symbol} 交易时出错: {e}")
    
    logger.info("交易执行完成")


def evaluate_positions(context):
    # type: (Context) -> NoReturn
    """
    评估持仓情况
    """
    logger.info("开始评估持仓情况")
    
    try:
        # 获取账户信息
        account = context.account()
        if not account:
            logger.error("无法获取账户信息")
            return
        
        # 记录账户状态
        logger.info(f"账户总资产: {account.market_value + account.cash:.2f}")
        logger.info(f"可用资金: {account.cash:.2f}")
        logger.info(f"持仓市值: {account.market_value:.2f}")
        
        # 评估各股票持仓
        for symbol in context.target_symbols:
            position = context.position(symbol=symbol)
            if position and position.volume > 0:
                current_price = context.current(symbol).price
                market_value = position.volume * current_price
                profit_loss = market_value - position.volume * position.vwap
                profit_loss_pct = profit_loss / (position.volume * position.vwap) * 100
                
                logger.info(f"股票 {symbol}: 持仓 {position.volume} 股, "
                           f"当前价 {current_price:.2f}, "
                           f"成本价 {position.vwap:.2f}, "
                           f"盈亏 {profit_loss:.2f} ({profit_loss_pct:.2f}%)")
        
        # 保存交易记录
        save_trading_records(context)
        
    except Exception as e:
        logger.error(f"评估持仓时出错: {e}")


def save_trading_records(context):
    # type: (Context) -> NoReturn
    """
    保存交易记录
    """
    try:
        # 创建交易记录目录
        records_dir = os.path.join('数据', '交易记录')
        os.makedirs(records_dir, exist_ok=True)
        
        # 获取当前日期
        current_date = datetime.now().strftime('%Y-%m-%d')
        
        # 准备交易记录数据
        records = {
            'date': current_date,
            'account': {
                'total_value': context.account().market_value + context.account().cash,
                'cash': context.account().cash,
                'market_value': context.account().market_value
            },
            'positions': [],
            'signals': context.trade_signals
        }
        
        # 记录各股票持仓
        for symbol in context.target_symbols:
            position = context.position(symbol=symbol)
            if position and position.volume > 0:
                current_price = context.current(symbol).price
                records['positions'].append({
                    'symbol': symbol,
                    'volume': position.volume,
                    'current_price': current_price,
                    'vwap': position.vwap,
                    'market_value': position.volume * current_price
                })
        
        # 保存交易记录
        record_file = os.path.join(records_dir, f"trading_record_{current_date}.json")
        with open(record_file, 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        
        logger.info(f"交易记录已保存到: {record_file}")
        
    except Exception as e:
        logger.error(f"保存交易记录时出错: {e}")


def on_bar(context, bars):
    # type: (Context, List[BarLikeDict2]) -> NoReturn
    """
    K线数据推送事件，可用于实时监控
    """
    # 这里可以添加实时监控逻辑
    pass


def on_order_status(context, order):
    # type: (Context, DictLikeOrder) -> NoReturn
    """
    委托状态更新事件
    """
    logger.info(f"委托状态更新: {order.symbol} {order.side} {order.volume} "
               f"状态: {order.status} 成交量: {order.filled_volume}")


def on_execution_report(context, execrpt):
    # type: (Context, DictLikeExecRpt) -> NoReturn
    """
    委托执行回报事件
    """
    logger.info(f"委托执行回报: {execrpt.symbol} {execrpt.side} {execrpt.volume} "
               f"成交价: {execrpt.price} 成交量: {execrpt.volume}")


def on_error(context, code, info):
    # type: (Context, int, Text) -> NoReturn
    """
    错误回调函数
    """
    logger.error(f"策略错误: 错误码 {code}, 错误信息: {info}")


if __name__ == '__main__':
    """
    运行策略
    """
    # 设置回测参数
    backtest_start_time = str(datetime.now() - timedelta(days=30))[:19]
    backtest_end_time = str(datetime.now())[:19]
    
    # 运行策略
    run(strategy_id='news_based_trading_strategy',
        filename='news_trading_strategy.py',
        mode=MODE_BACKTEST,
        token='{{token}}',
        backtest_start_time=backtest_start_time,
        backtest_end_time=backtest_end_time,
        backtest_adjust=ADJUST_PREV,
        backtest_initial_cash=10000000,
        backtest_commission_ratio=0.0001,
        backtest_slippage_ratio=0.0001,
        backtest_match_mode=1)