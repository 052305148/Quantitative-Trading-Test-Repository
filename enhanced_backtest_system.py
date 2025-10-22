# 优化版回测系统 - 增强仓位管理和风险控制
import sys
import os

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from optimized_backtest_system import OptimizedBacktestEngine, OptimizedBacktestConfig, Portfolio, Position, TradeRecord
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import json
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedBacktestConfig(OptimizedBacktestConfig):
    """增强版回测配置，优化仓位管理和风险控制"""
    
    def __init__(self, **kwargs):
        # 提取父类接受的参数
        parent_kwargs = {
            'start_date': kwargs.get('start_date'),
            'end_date': kwargs.get('end_date'),
            'initial_cash': kwargs.get('initial_cash', 1000000.0),
            'commission_ratio': kwargs.get('commission_ratio', 0.0001),
            'slippage_ratio': kwargs.get('slippage_ratio', 0.0001),
            'benchmark': kwargs.get('benchmark', "SHSE.000300"),
            'rebalance_frequency': kwargs.get('rebalance_frequency', "daily"),
            'max_position_size': kwargs.get('max_position_size', 0.2),
            'min_position_size': kwargs.get('min_position_size', 0.05),
            'max_drawdown_limit': kwargs.get('max_drawdown_limit', 0.1),
            'stop_loss': kwargs.get('stop_loss', -0.08),
            'take_profit': kwargs.get('take_profit', 0.15),
            'market_env_adjustment': kwargs.get('market_env_adjustment', True),
            'risk_control_enabled': kwargs.get('risk_control_enabled', True)
        }
        
        # 调用父类初始化
        super().__init__(**parent_kwargs)
        
        # 优化仓位管理参数（覆盖父类默认值）
        self.max_position_size = kwargs.get('max_position_size', 0.3)  # 提高最大仓位至30%
        self.min_position_size = kwargs.get('min_position_size', 0.05)  # 降低最小仓位至5%
        self.max_total_position = kwargs.get('max_total_position', 0.9)  # 最大总仓位90%
        
        # 优化风险控制参数（覆盖父类默认值）
        self.stop_loss = kwargs.get('stop_loss', -0.05)  # 收紧止损至-5%
        self.take_profit = kwargs.get('take_profit', 0.20)  # 提高止盈至20%
        self.max_drawdown_limit = kwargs.get('max_drawdown_limit', 0.15)  # 提高最大回撤限制至15%
        self.risk_control_enabled = kwargs.get('risk_control_enabled', True)  # 启用风险控制
        
        # 内部使用的参数，不传递给父类
        self.dynamic_position_sizing = True  # 启用动态仓位管理
        self.volatility_adjustment = True  # 启用波动率调整
        self.correlation_adjustment = True  # 启用相关性调整
        self.bull_market_multiplier = 1.3  # 牛市仓位倍数
        self.bear_market_multiplier = 0.6  # 熊市仓位倍数
        self.max_daily_loss = kwargs.get('max_daily_loss', 0.04)  # 最大单日亏损4%（内部使用）

class EnhancedBacktestEngine(OptimizedBacktestEngine):
    """增强版回测引擎，优化仓位管理和风险控制"""
    
    def __init__(self, config: EnhancedBacktestConfig):
        # 调用父类初始化
        super().__init__(config)
        
        # 存储历史波动率
        self.historical_volatility = {}
        
        # 存储股票相关性矩阵
        self.correlation_matrix = None
        
    def calculate_historical_volatility(self, symbol: str, price_data: pd.DataFrame, window: int = 20) -> float:
        """
        计算历史波动率
        
        Args:
            symbol: 股票代码
            price_data: 价格数据
            window: 计算窗口
            
        Returns:
            历史波动率
        """
        if symbol not in price_data.columns:
            return 0.02  # 默认波动率
            
        prices = price_data[symbol].values
        if len(prices) < window:
            window = len(prices) - 1
            
        if window <= 1:
            return 0.02  # 默认波动率
            
        # 计算日收益率
        returns = np.diff(prices) / prices[:-1]
        
        # 计算波动率（年化）
        volatility = np.std(returns) * np.sqrt(252)
        
        return max(0.05, min(volatility, 1.0))  # 限制在5%-100%之间
    
    def calculate_correlation_matrix(self, price_data: pd.DataFrame, window: int = 60) -> pd.DataFrame:
        """
        计算股票相关性矩阵
        
        Args:
            price_data: 价格数据
            window: 计算窗口
            
        Returns:
            相关性矩阵
        """
        # 计算日收益率
        returns = price_data.pct_change().dropna()
        
        if len(returns) < window:
            window = len(returns)
            
        if window <= 1:
            # 如果数据不足，返回单位矩阵
            symbols = price_data.columns.tolist()
            return pd.DataFrame(np.eye(len(symbols)), index=symbols, columns=symbols)
        
        # 使用最近window天的数据计算相关性
        recent_returns = returns.tail(window)
        
        # 计算相关性矩阵
        corr_matrix = recent_returns.corr()
        
        return corr_matrix
    
    def calculate_enhanced_position_size(self, signal_strength: float, market_env: float, 
                                       current_portfolio_value: float, price: float,
                                       symbol: str, price_data: pd.DataFrame = None) -> float:
        """
        计算增强版仓位大小，考虑波动率和相关性
        
        Args:
            signal_strength: 信号强度
            market_env: 市场环境评分
            current_portfolio_value: 当前组合价值
            price: 股票价格
            symbol: 股票代码
            price_data: 价格数据
            
        Returns:
            仓位大小，范围[0, 1]
        """
        # 基础仓位基于信号强度
        if signal_strength > 0.8:
            base_position = 0.25  # 高信号强度，25%基础仓位
        elif signal_strength > 0.6:
            base_position = 0.20  # 中高信号强度，20%基础仓位
        elif signal_strength > 0.4:
            base_position = 0.15  # 中等信号强度，15%基础仓位
        elif signal_strength > 0.2:
            base_position = 0.10  # 低信号强度，10%基础仓位
        elif signal_strength > 0:
            base_position = 0.05  # 极低信号强度，5%基础仓位
        elif signal_strength < -0.7:
            base_position = -0.15  # 强卖出信号
        elif signal_strength < -0.5:
            base_position = -0.12  # 中等卖出信号
        elif signal_strength < -0.3:
            base_position = -0.08  # 弱卖出信号
        else:
            base_position = 0.0  # 信号强度在-0.3到0之间，不交易
        
        # 市场环境调整
        if self.config.market_env_adjustment:
            if market_env < -0.3:  # 熊市环境
                market_adjustment = self.config.bear_market_multiplier
            elif market_env < -0.1:  # 弱市环境
                market_adjustment = 0.8
            elif market_env > 0.3:  # 牛市环境
                market_adjustment = self.config.bull_market_multiplier
            else:
                market_adjustment = 1.0  # 正常仓位
            
            # 应用市场环境调整
            base_position *= market_adjustment
        
        # 波动率调整
        if self.config.volatility_adjustment and price_data is not None:
            volatility = self.calculate_historical_volatility(symbol, price_data)
            
            # 低波动率股票增加仓位，高波动率股票减少仓位
            if volatility < 0.15:  # 低波动率
                volatility_adjustment = 1.2
            elif volatility < 0.25:  # 中等波动率
                volatility_adjustment = 1.0
            elif volatility < 0.35:  # 中高波动率
                volatility_adjustment = 0.8
            else:  # 高波动率
                volatility_adjustment = 0.6
                
            base_position *= volatility_adjustment
        
        # 相关性调整
        if self.config.correlation_adjustment and price_data is not None:
            # 计算与现有持仓的相关性
            if self.correlation_matrix is None:
                self.correlation_matrix = self.calculate_correlation_matrix(price_data)
            
            if symbol in self.correlation_matrix.columns and len(self.portfolio.positions) > 0:
                # 计算与现有持仓的平均相关性
                avg_correlation = 0
                count = 0
                
                for pos_symbol in self.portfolio.positions:
                    if pos_symbol in self.correlation_matrix.columns:
                        correlation = self.correlation_matrix.loc[symbol, pos_symbol]
                        avg_correlation += correlation
                        count += 1
                
                if count > 0:
                    avg_correlation /= count
                    
                    # 高相关性降低仓位，低相关性增加仓位
                    if avg_correlation > 0.7:  # 高相关性
                        correlation_adjustment = 0.7
                    elif avg_correlation > 0.4:  # 中等相关性
                        correlation_adjustment = 0.85
                    elif avg_correlation > 0.1:  # 低相关性
                        correlation_adjustment = 1.0
                    else:  # 负相关或无相关
                        correlation_adjustment = 1.2
                        
                    base_position *= correlation_adjustment
        
        # 限制单只股票最大仓位
        max_position = self.config.max_position_size
        min_position = self.config.min_position_size
        
        # 只有当base_position不为0时才应用仓位限制
        if base_position != 0:
            if base_position > 0:
                # 确保最小仓位不被完全过滤
                if base_position < min_position and base_position > 0:
                    base_position = min_position
                base_position = min(base_position, max_position)
            else:
                # 对于卖出信号，使用相同的绝对值限制
                base_position = max(base_position, -max_position)
        
        # 检查总仓位限制
        current_total_position = sum(pos.position_size for pos in self.portfolio.positions.values())
        available_position = self.config.max_total_position - current_total_position
        
        if base_position > 0 and base_position > available_position:
            # 只有当可用仓位大于最小仓位时才调整
            if available_position >= min_position:
                base_position = available_position
            else:
                # 如果可用仓位不足最小仓位，但base_position不为0，仍然使用最小仓位
                if base_position > 0:
                    base_position = min_position
                else:
                    base_position = 0
        
        return max(0, base_position)  # 确保返回非负值
    
    def enhanced_check_risk_controls(self, date: str) -> bool:
        """
        增强版风险控制检查
        
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
                
                # 单日最大亏损
                if daily_return < -self.config.max_daily_loss:
                    logger.warning(f"日期 {date} 触发单日最大亏损限制: {daily_return:.2%}")
                    return False
            
            # 检查集中度风险（单一持仓占比过高）
            for symbol, position in self.portfolio.positions.items():
                position_ratio = position.market_value / self.portfolio.total_value
                if position_ratio > self.config.max_position_size * 1.2:  # 允许20%的超出
                    logger.warning(f"日期 {date} 股票 {symbol} 持仓占比过高: {position_ratio:.2%}")
                    # 不直接拒绝交易，但记录警告
            
            return True
            
        except Exception as e:
            logger.error(f"风险控制检查失败: {e}")
            return True  # 出错时默认通过
    
    def run_enhanced_backtest(self, signals: list) -> dict:
        """
        运行增强版回测
        
        Args:
            signals: 交易信号列表
            
        Returns:
            回测结果
        """
        # 初始化
        self.portfolio = Portfolio(
            cash=self.config.initial_cash,
            positions={},
            total_value=self.config.initial_cash,
            daily_values=[],
            peak_value=self.config.initial_cash
        )
        
        self.trade_records = []
        
        # 如果信号是字典格式（从文件读取的），转换为列表格式
        if isinstance(signals, dict):
            signals_list = []
            for symbol, date_dict in signals.items():
                for date_str, signal_info in date_dict.items():
                    # 检查signal_info是否是字典
                    if isinstance(signal_info, dict):
                        signals_list.append({
                            'symbol': symbol,
                            'date': date_str,
                            'action': signal_info.get('action', 'hold'),
                            'strength': signal_info.get('strength', 0),
                            'position_size': 0,  # 将由系统计算
                            'reason': f"技术分析-{signal_info.get('action', 'hold')}-强度{signal_info.get('strength', 0):.2f}",
                            'market_environment': 'neutral',
                            'source': 'technical_analysis'
                        })
            signals = signals_list
        
        # 从信号中提取股票代码
        symbols = list(set(signal['symbol'] for signal in signals))
        
        # 加载价格数据
        price_data = self.load_price_data(symbols)
        
        # 计算相关性矩阵
        if self.config.correlation_adjustment:
            self.correlation_matrix = self.calculate_correlation_matrix(
                pd.DataFrame({symbol: df['close'] for symbol, df in price_data.items()})
            )
        
        # 加载基准数据
        benchmark_data = self.load_benchmark_data()
        
        # 按日期分组信号
        signals_by_date = {}
        for signal in signals:
            date = signal['date']
            if date not in signals_by_date:
                signals_by_date[date] = []
            signals_by_date[date].append(signal)
        
        # 按日期排序
        sorted_dates = sorted(signals_by_date.keys())
        
        # 获取所有交易日期
        all_dates = sorted(set(
            list(signals_by_date.keys()) + 
            [d.strftime('%Y-%m-%d') for d in pd.date_range(
                start=self.config.start_date, 
                end=self.config.end_date, 
                freq='D'
            ) if d.weekday() < 5]  # 只包含工作日
        ))
        
        # 逐日处理
        for date in all_dates:
            # 更新投资组合价值
            self.update_portfolio_value(date, price_data)
            
            # 记录每日价值
            self.portfolio.daily_values.append((date, self.portfolio.total_value))
            
            # 更新峰值价值
            if self.portfolio.total_value > self.portfolio.peak_value:
                self.portfolio.peak_value = self.portfolio.total_value
            
            # 增强版风险控制检查
            if not self.enhanced_check_risk_controls(date):
                logger.warning(f"日期 {date} 未通过风险控制检查，跳过交易")
                continue
            
            # 处理当日信号
            if date in signals_by_date:
                # 获取市场环境
                market_env = self.get_market_environment(date)
                
                # 处理每个信号
                for signal in signals_by_date[date]:
                    try:
                        # 提取信号信息
                        symbol = signal['symbol']
                        action = signal['action']
                        strength = signal['strength']
                        reason = signal.get('reason', '')
                        
                        # 获取当前价格
                        if symbol not in price_data:
                            continue
                            
                        # 将字符串日期转换为datetime对象
                        date_obj = pd.to_datetime(date)
                        
                        # 检查日期是否在价格数据中
                        if date_obj not in price_data[symbol].index:
                            # 尝试使用字符串格式查找
                            try:
                                # 尝试将索引转换为字符串格式并查找
                                date_str = date_obj.strftime('%Y-%m-%d')
                                if date_str in price_data[symbol].index.astype(str):
                                    # 找到字符串格式的索引
                                    idx = np.where(price_data[symbol].index.astype(str) == date_str)[0][0]
                                    current_price = price_data[symbol].iloc[idx]['close']
                                else:
                                    # 尝试直接使用字符串查找
                                    if date in price_data[symbol].index.astype(str):
                                        idx = np.where(price_data[symbol].index.astype(str) == date)[0][0]
                                        current_price = price_data[symbol].iloc[idx]['close']
                                    else:
                                        continue
                            except Exception:
                                # 如果所有方法都失败，跳过此信号
                                continue
                        else:
                            # 使用datetime索引
                            current_price = price_data[symbol]['close'].loc[date_obj]
                        
                        # 检查止损止盈
                        if symbol in self.portfolio.positions:
                            position = self.portfolio.positions[symbol]
                            stop_action = self.check_stop_loss_take_profit(symbol, position, current_price)
                            if stop_action:
                                # 执行止损止盈
                                quantity = position.quantity
                                self.execute_trade(symbol, stop_action, quantity, current_price, date, 0, f"止损止盈-{reason}")
                                continue
                        
                        # 计算仓位大小
                        position_size = self.calculate_enhanced_position_size(
                            strength, market_env, self.portfolio.total_value, 
                            current_price, symbol,
                            pd.DataFrame({s: df['close'] for s, df in price_data.items()})
                        )
                        
                        if position_size == 0:
                            continue
                        
                        # 计算交易数量
                        if action == "buy":
                            # 买入
                            available_cash = self.portfolio.cash * 0.95  # 保留5%现金
                            max_value = available_cash * position_size
                            quantity = int(max_value / (current_price * (1 + self.config.slippage_ratio)))
                        else:
                            # 卖出
                            if symbol in self.portfolio.positions:
                                quantity = int(self.portfolio.positions[symbol].quantity * position_size)
                            else:
                                quantity = 0
                        
                        if quantity <= 0:
                            continue
                        
                        # 执行交易
                        success = self.execute_trade(
                            symbol, action, quantity, current_price, date, position_size, reason
                        )
                        
                        if success:
                            logger.info(f"日期 {date} 执行交易: {action} {symbol} {quantity}股 价格{current_price:.2f} 仓位{position_size:.2%}")
                    
                    except Exception as e:
                        logger.error(f"处理信号失败: {e}, 信号: {signal}")
        
        # 计算绩效指标
        performance_metrics = self.calculate_performance_metrics()
        
        # 返回结果
        return {
            "config": self.config.__dict__,
            "portfolio": {
                "final_value": self.portfolio.total_value,
                "cash": self.portfolio.cash,
                "positions": {symbol: {
                    "quantity": pos.quantity,
                    "avg_price": pos.avg_price,
                    "market_value": pos.market_value,
                    "position_size": pos.position_size,
                    "realized_pnl": pos.realized_pnl
                } for symbol, pos in self.portfolio.positions.items()},
                "daily_values": self.portfolio.daily_values,
                "max_drawdown": self.portfolio.max_drawdown
            },
            "trades": [trade.__dict__ for trade in self.trade_records],
            "performance_metrics": performance_metrics
        }