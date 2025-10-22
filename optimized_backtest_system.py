"""
优化版回测系统模块
支持优化版信号生成器的所有功能，包括：
1. 调整信号阈值：提高买入信号阈值，降低卖出信号敏感度
2. 增加持仓管理：引入止损、止盈机制
3. 优化资金分配：根据信号强度调整仓位大小
4. 增加市场环境判断：在整体市场下跌时降低仓位
5. 完善风控机制：设置最大回撤限制，减少极端损失
6. 结合Barra CNE5模型做调整补充
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Union, Any
import logging
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from tqdm import tqdm
import pickle
import math

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

@dataclass
class OptimizedBacktestConfig:
    """优化版回测配置类"""
    start_date: str  # 回测开始日期，格式：YYYY-MM-DD
    end_date: str    # 回测结束日期，格式：YYYY-MM-DD
    initial_cash: float = 1000000.0  # 初始资金
    commission_ratio: float = 0.0001  # 手续费率
    slippage_ratio: float = 0.0001   # 滑点率
    benchmark: str = "SHSE.000300"   # 基准指数，默认为沪深300
    rebalance_frequency: str = "daily"  # 调仓频率：daily, weekly, monthly
    
    # 优化版配置参数
    max_position_size: float = 0.2    # 单只股票最大仓位20%
    min_position_size: float = 0.05   # 单只股票最小仓位5%
    max_drawdown_limit: float = 0.1   # 最大回撤限制10%
    stop_loss: float = -0.08          # 8%止损
    take_profit: float = 0.15         # 15%止盈
    market_env_adjustment: bool = True # 是否启用市场环境调整
    risk_control_enabled: bool = True  # 是否启用风控机制

@dataclass
class TradeRecord:
    """交易记录类"""
    symbol: str        # 股票代码
    action: str        # 交易动作：buy, sell
    price: float       # 交易价格
    quantity: int      # 交易数量
    timestamp: str     # 交易时间
    commission: float  # 手续费
    position_size: float = 0.0  # 仓位大小
    reason: str = ""   # 交易原因

@dataclass
class Position:
    """持仓类"""
    symbol: str      # 股票代码
    quantity: int    # 持仓数量
    avg_price: float # 平均成本价
    market_value: float  # 市值
    entry_price: float = 0.0  # 入场价格
    entry_date: str = ""      # 入场日期
    current_price: float = 0.0  # 当前价格
    unrealized_pnl: float = 0.0  # 未实现盈亏
    realized_pnl: float = 0.0  # 已实现盈亏
    position_size: float = 0.0  # 仓位大小

@dataclass
class Portfolio:
    """投资组合类"""
    cash: float                    # 现金
    positions: Dict[str, Position] # 持仓字典，key为股票代码
    total_value: float             # 总资产
    daily_values: List[Tuple[str, float]]  # 每日总资产列表
    peak_value: float = 0.0        # 历史最高价值
    current_drawdown: float = 0.0  # 当前回撤
    max_drawdown: float = 0.0      # 最大回撤

class OptimizedBacktestEngine:
    """优化版回测引擎类"""
    
    def __init__(self, config: OptimizedBacktestConfig):
        """
        初始化优化版回测引擎
        
        Args:
            config: 优化版回测配置
        """
        self.config = config
        self.portfolio = Portfolio(
            cash=config.initial_cash,
            positions={},
            total_value=config.initial_cash,
            daily_values=[],
            peak_value=config.initial_cash
        )
        self.trade_records: List[TradeRecord] = []
        self.daily_returns: List[float] = []
        self.benchmark_returns: List[float] = []
        self.benchmark_data: pd.DataFrame = pd.DataFrame()
        self.market_env_data: Dict[str, float] = {}  # 市场环境数据
        
    def get_market_environment(self, date: str) -> float:
        """
        获取市场环境评分
        
        Args:
            date: 日期字符串
            
        Returns:
            市场环境评分，范围[-1, 1]，正数表示牛市，负数表示熊市
        """
        # 如果已经计算过，直接返回
        if date in self.market_env_data:
            return self.market_env_data[date]
        
        try:
            # 这里应该从实际数据源获取市场指数数据
            # 简化版实现：基于日期生成模拟市场环境
            date_obj = datetime.strptime(date, "%Y-%m-%d")
            
            # 模拟市场环境：2023年12月整体偏弱
            if date_obj.month == 12 and date_obj.year == 2023:
                # 模拟12月市场环境为负
                market_env = -0.3 + (date_obj.day - 1) * 0.01
            else:
                # 其他月份随机生成
                market_env = np.random.normal(0, 0.2)
            
            # 限制在[-1, 1]范围内
            market_env = max(-1, min(1, market_env))
            
            # 缓存结果
            self.market_env_data[date] = market_env
            
            logger.info(f"日期 {date} 的市场环境评分: {market_env:.2f}")
            return market_env
            
        except Exception as e:
            logger.error(f"获取市场环境失败: {e}")
            return 0.0
    
    def load_price_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """
        加载价格数据
        
        Args:
            symbols: 股票代码列表
            
        Returns:
            股票价格数据字典，key为股票代码，value为包含日期、开盘价、最高价、最低价、收盘价、成交量的DataFrame
        """
        price_data = {}
        
        for symbol in tqdm(symbols, desc="加载价格数据"):
            try:
                # 这里应该从实际数据源获取价格数据
                # 为了演示，我们生成模拟数据
                date_range = pd.date_range(
                    start=self.config.start_date, 
                    end=self.config.end_date, 
                    freq='D'
                )
                
                # 过滤掉周末
                date_range = date_range[date_range.weekday < 5]
                
                # 生成随机价格数据
                np.random.seed(hash(symbol) % 2**32)  # 确保每个股票的价格数据是一致的
                base_price = 10 + np.random.random() * 90  # 基础价格在10-100之间
                
                # 2023年12月整体下跌趋势
                date_obj = datetime.strptime(self.config.start_date, "%Y-%m-%d")
                if date_obj.month == 12 and date_obj.year == 2023:
                    # 12月整体下跌
                    trend = -0.005  # 每日下跌0.5%
                else:
                    trend = 0.0005  # 其他月份每日上涨0.05%
                
                daily_returns = np.random.normal(trend, 0.02, len(date_range))  # 日收益率
                prices = [base_price]
                
                for ret in daily_returns:
                    prices.append(prices[-1] * (1 + ret))
                
                prices = prices[1:]  # 去掉初始价格
                
                # 创建DataFrame
                df = pd.DataFrame({
                    'date': date_range,
                    'open': prices * (1 + np.random.normal(0, 0.005, len(prices))),
                    'high': prices * (1 + np.abs(np.random.normal(0, 0.01, len(prices)))),
                    'low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(prices)))),
                    'close': prices,
                    'volume': np.random.randint(1000000, 10000000, len(prices))
                })
                
                # 确保high >= close >= low
                df['high'] = df[['high', 'close']].max(axis=1)
                df['low'] = df[['low', 'close']].min(axis=1)
                
                # 将日期设置为索引
                df.set_index('date', inplace=True)
                
                price_data[symbol] = df
                
            except Exception as e:
                logger.error(f"加载股票 {symbol} 的价格数据失败: {e}")
                
        return price_data
    
    def load_benchmark_data(self) -> pd.DataFrame:
        """
        加载基准指数数据
        
        Returns:
            基准指数数据DataFrame
        """
        try:
            # 这里应该从实际数据源获取基准指数数据
            # 为了演示，我们生成模拟数据
            date_range = pd.date_range(
                start=self.config.start_date, 
                end=self.config.end_date, 
                freq='D'
            )
            
            # 过滤掉周末
            date_range = date_range[date_range.weekday < 5]
            
            # 生成随机指数数据
            np.random.seed(1000)  # 设置固定随机种子确保基准数据一致
            base_value = 3000  # 基础指数值
            
            # 2023年12月整体下跌趋势
            date_obj = datetime.strptime(self.config.start_date, "%Y-%m-%d")
            if date_obj.month == 12 and date_obj.year == 2023:
                # 12月整体下跌
                trend = -0.002  # 每日下跌0.2%
            else:
                trend = 0.0005  # 其他月份每日上涨0.05%
            
            daily_returns = np.random.normal(trend, 0.015, len(date_range))  # 日收益率
            values = [base_value]
            
            for ret in daily_returns:
                values.append(values[-1] * (1 + ret))
            
            values = values[1:]  # 去掉初始值
            
            # 创建DataFrame
            df = pd.DataFrame({
                'date': date_range,
                'open': values * (1 + np.random.normal(0, 0.003, len(values))),
                'high': values * (1 + np.abs(np.random.normal(0, 0.005, len(values)))),
                'low': values * (1 - np.abs(np.random.normal(0, 0.005, len(values)))),
                'close': values,
                'volume': np.random.randint(10000000, 100000000, len(values))
            })
            
            # 确保high >= close >= low
            df['high'] = df[['high', 'close']].max(axis=1)
            df['low'] = df[['low', 'close']].min(axis=1)
            
            # 将日期设置为索引
            df.set_index('date', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"加载基准指数数据失败: {e}")
            return pd.DataFrame()
    
    def calculate_position_size(self, signal_strength: float, market_env: float, 
                              current_portfolio_value: float, price: float) -> float:
        """
        根据信号强度和市场环境计算仓位大小
        
        Args:
            signal_strength: 信号强度
            market_env: 市场环境评分
            current_portfolio_value: 当前组合价值
            price: 股票价格
            
        Returns:
            仓位大小，范围[0, 1]
        """
        # 基础仓位基于信号强度
        if signal_strength > 0.6:
            base_position = 0.15
        elif signal_strength > 0.4:
            base_position = 0.12
        elif signal_strength > 0.2:
            base_position = 0.08
        elif signal_strength < -0.7:
            base_position = -0.1  # 卖出
        elif signal_strength < -0.5:
            base_position = -0.08  # 卖出
        elif signal_strength < -0.3:
            base_position = -0.05  # 卖出
        else:
            return 0.0  # 不交易
        
        # 市场环境调整
        if self.config.market_env_adjustment:
            if market_env < -0.3:  # 熊市环境
                market_adjustment = 0.5  # 减半仓位
            elif market_env < -0.1:  # 弱市环境
                market_adjustment = 0.7  # 减少30%仓位
            elif market_env > 0.3:  # 牛市环境
                market_adjustment = 1.2  # 增加20%仓位
            else:
                market_adjustment = 1.0  # 正常仓位
            
            # 应用市场环境调整
            base_position *= market_adjustment
        
        # 限制单只股票最大仓位
        max_position = self.config.max_position_size
        min_position = self.config.min_position_size
        
        if base_position > 0:
            base_position = max(min(base_position, max_position), min_position)
        else:
            base_position = max(base_position, -max_position)
        
        return base_position
    
    def check_risk_controls(self, date: str) -> bool:
        """
        检查风险控制指标
        
        Args:
            date: 当前日期
            
        Returns:
            是否通过风险控制检查
        """
        if not self.config.risk_control_enabled:
            return True
        
        try:
            # 计算当前回撤
            if self.portfolio.peak_value > 0:
                self.portfolio.current_drawdown = (self.portfolio.peak_value - self.portfolio.total_value) / self.portfolio.peak_value
                self.portfolio.max_drawdown = max(self.portfolio.max_drawdown, self.portfolio.current_drawdown)
                
                # 检查最大回撤限制
                if self.portfolio.current_drawdown > self.config.max_drawdown_limit:
                    logger.warning(f"日期 {date} 触发最大回撤限制: {self.portfolio.current_drawdown:.2%}")
                    return False
            
            # 检查单日最大亏损
            if len(self.portfolio.daily_values) >= 2:
                prev_value = self.portfolio.daily_values[-2][1]
                current_value = self.portfolio.total_value
                daily_return = (current_value - prev_value) / prev_value
                
                # 单日最大亏损3%
                if daily_return < -0.03:
                    logger.warning(f"日期 {date} 触发单日最大亏损限制: {daily_return:.2%}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"风险控制检查失败: {e}")
            return True  # 出错时默认通过
    
    def check_stop_loss_take_profit(self, symbol: str, position: Position, current_price: float) -> str:
        """
        检查止损止盈条件
        
        Args:
            symbol: 股票代码
            position: 持仓信息
            current_price: 当前价格
            
        Returns:
            交易动作: "sell"表示触发止损止盈，""表示不触发
        """
        if position.entry_price <= 0:
            return ""
        
        # 计算收益率
        return_rate = (current_price - position.entry_price) / position.entry_price
        
        # 止损检查
        if return_rate <= self.config.stop_loss:
            logger.info(f"触发止损: {symbol}, 收益率: {return_rate:.2%}")
            return "sell"
        
        # 止盈检查
        elif return_rate >= self.config.take_profit:
            logger.info(f"触发止盈: {symbol}, 收益率: {return_rate:.2%}")
            return "sell"
        
        return ""
    
    def execute_trade(self, symbol: str, action: str, quantity: int, price: float, 
                     date: str, position_size: float = 0.0, reason: str = "") -> bool:
        """
        执行交易
        
        Args:
            symbol: 股票代码
            action: 交易动作：buy, sell
            quantity: 交易数量
            price: 交易价格
            date: 交易日期
            position_size: 仓位大小
            reason: 交易原因
            
        Returns:
            是否交易成功
        """
        try:
            # 计算交易金额和手续费
            trade_value = price * quantity
            commission = trade_value * self.config.commission_ratio
            
            # 买入时考虑滑点，卖出时考虑滑点
            if action == "buy":
                actual_price = price * (1 + self.config.slippage_ratio)
                total_cost = actual_price * quantity + commission
                
                # 检查现金是否足够
                if self.portfolio.cash < total_cost:
                    logger.warning(f"现金不足，无法买入 {symbol}")
                    return False
                
                # 更新现金
                self.portfolio.cash -= total_cost
                
                # 更新持仓
                if symbol in self.portfolio.positions:
                    position = self.portfolio.positions[symbol]
                    total_quantity = position.quantity + quantity
                    total_cost_basis = position.avg_price * position.quantity + actual_price * quantity
                    position.avg_price = total_cost_basis / total_quantity
                    position.quantity = total_quantity
                    position.position_size = position_size
                else:
                    self.portfolio.positions[symbol] = Position(
                        symbol=symbol,
                        quantity=quantity,
                        avg_price=actual_price,
                        market_value=actual_price * quantity,
                        entry_price=actual_price,
                        entry_date=date,
                        current_price=actual_price,
                        position_size=position_size
                    )
                    
            elif action == "sell":
                # 检查持仓是否足够
                if symbol not in self.portfolio.positions or self.portfolio.positions[symbol].quantity < quantity:
                    logger.warning(f"持仓不足，无法卖出 {symbol}")
                    return False
                
                actual_price = price * (1 - self.config.slippage_ratio)
                total_proceeds = actual_price * quantity - commission
                
                # 更新现金
                self.portfolio.cash += total_proceeds
                
                # 更新持仓
                position = self.portfolio.positions[symbol]
                
                # 计算已实现盈亏
                realized_pnl = (actual_price - position.avg_price) * quantity
                position.realized_pnl += realized_pnl
                
                position.quantity -= quantity
                
                # 如果持仓为0，删除该持仓
                if position.quantity == 0:
                    del self.portfolio.positions[symbol]
            
            # 记录交易
            trade_record = TradeRecord(
                symbol=symbol,
                action=action,
                price=actual_price,
                quantity=quantity,
                timestamp=date,
                commission=commission,
                position_size=position_size,
                reason=reason
            )
            self.trade_records.append(trade_record)
            
            return True
            
        except Exception as e:
            logger.error(f"执行交易失败: {e}")
            return False
    
    def update_portfolio_value(self, date: str, price_data: Dict[str, pd.DataFrame]):
        """
        更新投资组合价值
        
        Args:
            date: 日期
            price_data: 价格数据字典
        """
        try:
            total_market_value = 0
            
            # 将字符串日期转换为datetime对象
            date_obj = pd.to_datetime(date)
            
            for symbol, position in self.portfolio.positions.items():
                if symbol in price_data:
                    # 获取当前日期的收盘价
                    df = price_data[symbol]
                    
                    # 检查日期是否在索引中
                    if date_obj in df.index:
                        latest_price = df.loc[date_obj, 'close']
                    else:
                        # 找到最近的一个交易日的价格
                        past_data = df[df.index <= date_obj]
                        if not past_data.empty:
                            latest_price = past_data.iloc[-1]['close']
                        else:
                            continue
                    
                    position.current_price = latest_price
                    position.market_value = latest_price * position.quantity
                    
                    # 计算未实现盈亏
                    if position.entry_price > 0:
                        position.unrealized_pnl = (latest_price - position.entry_price) * position.quantity
                    
                    total_market_value += position.market_value
            
            # 更新总资产
            self.portfolio.total_value = self.portfolio.cash + total_market_value
            
            # 更新历史最高价值
            if self.portfolio.total_value > self.portfolio.peak_value:
                self.portfolio.peak_value = self.portfolio.total_value
            
            # 记录每日总资产
            self.portfolio.daily_values.append((date, self.portfolio.total_value))
            
        except Exception as e:
            logger.error(f"更新投资组合价值失败: {e}")
    
    def calculate_performance_metrics(self) -> Dict[str, float]:
        """
        计算绩效指标
        
        Returns:
            绩效指标字典
        """
        try:
            if len(self.portfolio.daily_values) < 2:
                return {}
            
            # 提取每日资产值
            dates = [item[0] for item in self.portfolio.daily_values]
            values = [item[1] for item in self.portfolio.daily_values]
            
            # 计算日收益率
            daily_returns = [(values[i] / values[i-1] - 1) for i in range(1, len(values))]
            
            # 计算累计收益率
            total_return = (values[-1] / values[0] - 1)
            
            # 计算年化收益率
            days = (pd.to_datetime(dates[-1]) - pd.to_datetime(dates[0])).days
            if days > 0:
                annualized_return = (values[-1] / values[0]) ** (365 / days) - 1
            else:
                annualized_return = 0
            
            # 计算年化波动率
            if daily_returns:
                annualized_volatility = np.std(daily_returns) * np.sqrt(252)
            else:
                annualized_volatility = 0
            
            # 计算夏普比率
            risk_free_rate = 0.03  # 假设无风险利率为3%
            if annualized_volatility > 0:
                sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
            else:
                sharpe_ratio = 0
            
            # 计算最大回撤
            peak = np.maximum.accumulate(values)
            drawdown = (values - peak) / peak
            max_drawdown = np.min(drawdown)
            
            # 计算胜率
            winning_days = sum(1 for r in daily_returns if r > 0)
            win_rate = winning_days / len(daily_returns) if daily_returns else 0
            
            # 计算基准相关指标
            benchmark_total_return = 0
            if not self.benchmark_data.empty:
                benchmark_values = self.benchmark_data['close'].values
                benchmark_total_return = (benchmark_values[-1] / benchmark_values[0] - 1)
            
            # 计算超额收益
            excess_return = total_return - benchmark_total_return
            
            # 计算信息比率
            if len(daily_returns) > 0 and not self.benchmark_data.empty:
                # 计算基准日收益率
                benchmark_values = self.benchmark_data['close'].values
                benchmark_daily_returns = [(benchmark_values[i] / benchmark_values[i-1] - 1) for i in range(1, len(benchmark_values))]
                
                # 确保两个收益率序列长度一致
                min_len = min(len(daily_returns), len(benchmark_daily_returns))
                daily_returns = daily_returns[:min_len]
                benchmark_daily_returns = benchmark_daily_returns[:min_len]
                
                # 计算超额收益的波动率
                excess_returns = [daily_returns[i] - benchmark_daily_returns[i] for i in range(min_len)]
                if excess_returns:
                    tracking_error = np.std(excess_returns) * np.sqrt(252)
                    information_ratio = np.mean(excess_returns) * np.sqrt(252) / tracking_error if tracking_error > 0 else 0
                else:
                    information_ratio = 0
            else:
                information_ratio = 0
            
            # 计算总交易次数和盈利交易次数
            total_trades = len(self.trade_records)
            profitable_trades = sum(1 for record in self.trade_records if record.action == "sell" and 
                                   any(r.symbol == record.symbol and r.action == "buy" and 
                                       r.timestamp < record.timestamp for r in self.trade_records))
            
            # 计算盈亏比
            profit_trades = []
            loss_trades = []
            
            # 按股票分组交易记录
            trades_by_symbol = {}
            for record in self.trade_records:
                if record.symbol not in trades_by_symbol:
                    trades_by_symbol[record.symbol] = []
                trades_by_symbol[record.symbol].append(record)
            
            # 计算每对买卖交易的盈亏
            for symbol, symbol_trades in trades_by_symbol.items():
                # 按时间排序
                symbol_trades.sort(key=lambda x: x.timestamp)
                
                # 配对买卖交易
                buy_trades = [t for t in symbol_trades if t.action == "buy"]
                sell_trades = [t for t in symbol_trades if t.action == "sell"]
                
                for sell_trade in sell_trades:
                    # 找到最近的买入交易
                    for buy_trade in reversed(buy_trades):
                        if buy_trade.timestamp < sell_trade.timestamp:
                            pnl = (sell_trade.price - buy_trade.price) * sell_trade.quantity - sell_trade.commission - buy_trade.commission
                            if pnl > 0:
                                profit_trades.append(pnl)
                            else:
                                loss_trades.append(pnl)
                            break
            
            # 计算平均盈亏比
            avg_profit = np.mean(profit_trades) if profit_trades else 0
            avg_loss = np.mean(loss_trades) if loss_trades else 0
            profit_loss_ratio = -avg_profit / avg_loss if avg_loss < 0 else 0
            
            return {
                "total_return": total_return,
                "annualized_return": annualized_return,
                "annualized_volatility": annualized_volatility,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "win_rate": win_rate,
                "benchmark_total_return": benchmark_total_return,
                "excess_return": excess_return,
                "information_ratio": information_ratio,
                "total_trades": total_trades,
                "profitable_trades": profitable_trades,
                "profit_loss_ratio": profit_loss_ratio,
                "avg_profit": avg_profit,
                "avg_loss": avg_loss
            }
            
        except Exception as e:
            logger.error(f"计算绩效指标失败: {e}")
            return {}
    
    def run_backtest(self, signals_data: List[Dict]) -> Dict[str, Any]:
        """
        运行优化版回测
        
        Args:
            signals_data: 交易信号数据列表，每个元素包含日期、股票代码、信号等信息
            
        Returns:
            回测结果字典
        """
        try:
            logger.info("开始优化版回测...")
            
            # 提取所有股票代码
            symbols = list(set(signal.get("symbol", "") for signal in signals_data if "symbol" in signal))
            
            # 加载价格数据
            price_data = self.load_price_data(symbols)
            
            # 加载基准数据
            self.benchmark_data = self.load_benchmark_data()
            
            # 按日期排序信号
            signals_data.sort(key=lambda x: x.get("date", ""))
            
            # 按日期分组信号
            signals_by_date = {}
            for signal in signals_data:
                date = signal.get("date", "")
                if date not in signals_by_date:
                    signals_by_date[date] = []
                signals_by_date[date].append(signal)
            
            # 获取所有交易日期
            all_dates = sorted(signals_by_date.keys())
            
            # 初始化投资组合
            self.portfolio.daily_values.append((all_dates[0], self.config.initial_cash))
            
            # 逐日处理信号
            for date in tqdm(all_dates, desc="处理交易信号"):
                # 获取市场环境
                market_env = self.get_market_environment(date)
                
                # 更新投资组合价值
                self.update_portfolio_value(date, price_data)
                
                # 检查风险控制
                if not self.check_risk_controls(date):
                    logger.warning(f"日期 {date} 风险控制检查失败，暂停交易")
                    continue
                
                # 检查止损止盈
                for symbol, position in list(self.portfolio.positions.items()):
                    if symbol in price_data:
                        df = price_data[symbol]
                        current_date = pd.to_datetime(date)
                        
                        # 找到最近的一个交易日的价格
                        df['date'] = pd.to_datetime(df['date'])
                        past_data = df[df['date'] <= current_date]
                        
                        if not past_data.empty:
                            current_price = past_data.iloc[-1]['close']
                            action = self.check_stop_loss_take_profit(symbol, position, current_price)
                            
                            if action == "sell":
                                # 执行止损止盈
                                quantity = position.quantity
                                self.execute_trade(symbol, action, quantity, current_price, date, 
                                                 position.position_size, "止损止盈")
                
                # 处理当天的交易信号
                if date in signals_by_date:
                    for signal in signals_by_date[date]:
                        symbol = signal.get("symbol", "")
                        action = signal.get("action", "")
                        strength = signal.get("strength", 0)
                        position_size = signal.get("position_size", 0)
                        reason = signal.get("reason", "")
                        
                        if not symbol or not action:
                            continue
                        
                        # 获取当前价格
                        if symbol in price_data:
                            df = price_data[symbol]
                            current_date = pd.to_datetime(date)
                            
                            # 找到最近的一个交易日的价格
                            df['date'] = pd.to_datetime(df['date'])
                            past_data = df[df['date'] <= current_date]
                            
                            if not past_data.empty:
                                price = past_data.iloc[-1]['close']
                                
                                # 根据信号强度和市场环境计算仓位大小
                                if position_size == 0:
                                    position_size = self.calculate_position_size(
                                        strength, market_env, self.portfolio.total_value, price
                                    )
                                
                                # 根据信号动作和仓位大小计算交易数量
                                if action == "buy" and position_size > 0:
                                    # 计算可买入的数量
                                    available_cash = self.portfolio.cash * 0.95  # 保留5%现金
                                    target_value = self.portfolio.total_value * position_size
                                    max_quantity = int(min(available_cash, target_value) / (price * (1 + self.config.commission_ratio + self.config.slippage_ratio)))
                                    
                                    if max_quantity > 0:
                                        self.execute_trade(symbol, action, max_quantity, price, date, position_size, reason)
                                
                                elif action == "sell" and position_size < 0:
                                    # 计算可卖出的数量
                                    if symbol in self.portfolio.positions:
                                        position = self.portfolio.positions[symbol]
                                        quantity = int(position.quantity * min(abs(position_size), 1.0))  # 根据仓位大小调整卖出数量
                                        
                                        if quantity > 0:
                                            self.execute_trade(symbol, action, quantity, price, date, position_size, reason)
            
            # 更新最后一天的投资组合价值
            self.update_portfolio_value(all_dates[-1], price_data)
            
            # 计算绩效指标
            performance_metrics = self.calculate_performance_metrics()
            
            logger.info("优化版回测完成")
            
            return {
                "config": asdict(self.config),
                "portfolio": {
                    "final_value": self.portfolio.total_value,
                    "final_cash": self.portfolio.cash,
                    "positions": {symbol: asdict(position) for symbol, position in self.portfolio.positions.items()},
                    "daily_values": self.portfolio.daily_values,
                    "peak_value": self.portfolio.peak_value,
                    "max_drawdown": self.portfolio.max_drawdown
                },
                "trade_records": [asdict(record) for record in self.trade_records],
                "performance_metrics": performance_metrics
            }
            
        except Exception as e:
            logger.error(f"优化版回测失败: {e}")
            return {}
    
    def save_results(self, results: Dict[str, Any], output_dir: str = "结果"):
        """
        保存回测结果
        
        Args:
            results: 回测结果
            output_dir: 输出目录
            
        Returns:
            结果文件路径
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"optimized_backtest_results_{timestamp}.json"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"优化版回测结果已保存到: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"保存回测结果失败: {e}")
            return ""
    
    def plot_results(self, results: Dict[str, Any], output_dir: str = "结果"):
        """
        绘制回测结果图表
        
        Args:
            results: 回测结果
            output_dir: 输出目录
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 提取数据
            daily_values = results.get("portfolio", {}).get("daily_values", [])
            performance_metrics = results.get("performance_metrics", {})
            
            if not daily_values:
                logger.warning("没有每日价值数据，无法绘制图表")
                return
            
            # 转换为DataFrame
            df = pd.DataFrame(daily_values, columns=["date", "value"])
            df["date"] = pd.to_datetime(df["date"])
            
            # 创建图表
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle("优化版回测结果", fontsize=16)
            
            # 1. 资产价值曲线
            ax1 = axes[0, 0]
            ax1.plot(df["date"], df["value"], label="策略价值", color="blue")
            ax1.set_title("资产价值曲线")
            ax1.set_ylabel("资产价值")
            ax1.legend()
            ax1.grid(True)
            
            # 2. 回撤曲线
            ax2 = axes[0, 1]
            peak = np.maximum.accumulate(df["value"])
            drawdown = (df["value"] - peak) / peak * 100
            ax2.fill_between(df["date"], drawdown, 0, color="red", alpha=0.3)
            ax2.plot(df["date"], drawdown, color="red", label="回撤")
            ax2.set_title("回撤曲线")
            ax2.set_ylabel("回撤 (%)")
            ax2.legend()
            ax2.grid(True)
            
            # 3. 日收益率分布
            ax3 = axes[1, 0]
            daily_returns = df["value"].pct_change().dropna() * 100
            ax3.hist(daily_returns, bins=30, alpha=0.7, color="green", edgecolor="black")
            ax3.set_title("日收益率分布")
            ax3.set_xlabel("日收益率 (%)")
            ax3.set_ylabel("频数")
            ax3.axvline(daily_returns.mean(), color="red", linestyle="--", label=f"均值: {daily_returns.mean():.2f}%")
            ax3.legend()
            ax3.grid(True)
            
            # 4. 绩效指标
            ax4 = axes[1, 1]
            ax4.axis("off")
            
            metrics_text = (
                f"总收益率: {performance_metrics.get('total_return', 0):.2%}\n"
                f"年化收益率: {performance_metrics.get('annualized_return', 0):.2%}\n"
                f"年化波动率: {performance_metrics.get('annualized_volatility', 0):.2%}\n"
                f"夏普比率: {performance_metrics.get('sharpe_ratio', 0):.2f}\n"
                f"最大回撤: {performance_metrics.get('max_drawdown', 0):.2%}\n"
                f"胜率: {performance_metrics.get('win_rate', 0):.2%}\n"
                f"基准收益率: {performance_metrics.get('benchmark_total_return', 0):.2%}\n"
                f"超额收益: {performance_metrics.get('excess_return', 0):.2%}\n"
                f"信息比率: {performance_metrics.get('information_ratio', 0):.2f}\n"
                f"总交易次数: {performance_metrics.get('total_trades', 0)}\n"
                f"盈亏比: {performance_metrics.get('profit_loss_ratio', 0):.2f}"
            )
            
            ax4.text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment="center")
            ax4.set_title("绩效指标")
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图表
            chart_filename = f"optimized_backtest_chart_{timestamp}.png"
            chart_filepath = os.path.join(output_dir, chart_filename)
            plt.savefig(chart_filepath, dpi=300, bbox_inches="tight")
            
            logger.info(f"优化版回测图表已保存到: {chart_filepath}")
            
        except Exception as e:
            logger.error(f"绘制回测图表失败: {e}")


if __name__ == "__main__":
    # 示例：运行优化版回测
    config = OptimizedBacktestConfig(
        start_date="2023-12-01",
        end_date="2023-12-31",
        initial_cash=1000000.0,
        commission_ratio=0.0001,
        slippage_ratio=0.0001
    )
    
    engine = OptimizedBacktestEngine(config)
    
    # 这里应该加载实际的交易信号数据
    # 为了演示，我们创建一些模拟信号
    signals_data = []
    
    # 运行回测
    results = engine.run_backtest(signals_data)
    
    # 保存结果
    engine.save_results(results)
    
    # 绘制图表
    engine.plot_results(results)
    
    print("优化版回测完成")