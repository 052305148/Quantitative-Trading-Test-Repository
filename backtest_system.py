"""
回测系统模块
用于评估交易策略的历史表现，提供详细的回测报告和可视化结果
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

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

@dataclass
class BacktestConfig:
    """回测配置类"""
    start_date: str  # 回测开始日期，格式：YYYY-MM-DD
    end_date: str    # 回测结束日期，格式：YYYY-MM-DD
    initial_cash: float = 1000000.0  # 初始资金
    commission_ratio: float = 0.0001  # 手续费率
    slippage_ratio: float = 0.0001   # 滑点率
    benchmark: str = "SHSE.000300"   # 基准指数，默认为沪深300
    rebalance_frequency: str = "daily"  # 调仓频率：daily, weekly, monthly
    
@dataclass
class TradeRecord:
    """交易记录类"""
    symbol: str        # 股票代码
    action: str        # 交易动作：buy, sell
    price: float       # 交易价格
    quantity: int      # 交易数量
    timestamp: str     # 交易时间
    commission: float  # 手续费
    
@dataclass
class Position:
    """持仓类"""
    symbol: str      # 股票代码
    quantity: int    # 持仓数量
    avg_price: float # 平均成本价
    market_value: float  # 市值
    
@dataclass
class Portfolio:
    """投资组合类"""
    cash: float                    # 现金
    positions: Dict[str, Position] # 持仓字典，key为股票代码
    total_value: float             # 总资产
    daily_values: List[Tuple[str, float]]  # 每日总资产列表
    
class BacktestEngine:
    """回测引擎类"""
    
    def __init__(self, config: BacktestConfig):
        """
        初始化回测引擎
        
        Args:
            config: 回测配置
        """
        self.config = config
        self.portfolio = Portfolio(
            cash=config.initial_cash,
            positions={},
            total_value=config.initial_cash,
            daily_values=[]
        )
        self.trade_records: List[TradeRecord] = []
        self.daily_returns: List[float] = []
        self.benchmark_returns: List[float] = []
        self.benchmark_data: pd.DataFrame = pd.DataFrame()
        
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
                daily_returns = np.random.normal(0, 0.02, len(date_range))  # 日收益率，均值0，标准差2%
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
            base_value = 3000  # 基础指数值
            daily_returns = np.random.normal(0.0005, 0.015, len(date_range))  # 日收益率，均值0.05%，标准差1.5%
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
            
            return df
            
        except Exception as e:
            logger.error(f"加载基准指数数据失败: {e}")
            return pd.DataFrame()
    
    def execute_trade(self, symbol: str, action: str, quantity: int, price: float, date: str) -> bool:
        """
        执行交易
        
        Args:
            symbol: 股票代码
            action: 交易动作：buy, sell
            quantity: 交易数量
            price: 交易价格
            date: 交易日期
            
        Returns:
            是否交易成功
        """
        try:
            # 计算交易金额和手续费
            trade_value = price * quantity
            commission = trade_value * self.config.commission_ratio
            slippage = trade_value * self.config.slippage_ratio
            
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
                else:
                    self.portfolio.positions[symbol] = Position(
                        symbol=symbol,
                        quantity=quantity,
                        avg_price=actual_price,
                        market_value=actual_price * quantity
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
                commission=commission
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
            
            for symbol, position in self.portfolio.positions.items():
                if symbol in price_data:
                    # 获取当前日期的收盘价
                    df = price_data[symbol]
                    current_date = pd.to_datetime(date)
                    
                    # 找到最近的一个交易日的价格
                    df['date'] = pd.to_datetime(df['date'])
                    past_data = df[df['date'] <= current_date]
                    
                    if not past_data.empty:
                        latest_price = past_data.iloc[-1]['close']
                        position.market_value = latest_price * position.quantity
                        total_market_value += position.market_value
            
            # 更新总资产
            self.portfolio.total_value = self.portfolio.cash + total_market_value
            
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
            
            return {
                "total_return": total_return,
                "annualized_return": annualized_return,
                "annualized_volatility": annualized_volatility,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "win_rate": win_rate,
                "benchmark_total_return": benchmark_total_return,
                "excess_return": excess_return,
                "total_trades": len(self.trade_records)
            }
            
        except Exception as e:
            logger.error(f"计算绩效指标失败: {e}")
            return {}
    
    def run_backtest(self, signals_data: List[Dict]) -> Dict[str, Any]:
        """
        运行回测
        
        Args:
            signals_data: 交易信号数据列表，每个元素包含日期、股票代码、信号等信息
            
        Returns:
            回测结果字典
        """
        try:
            logger.info("开始回测...")
            
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
                # 更新投资组合价值
                self.update_portfolio_value(date, price_data)
                
                # 处理当天的交易信号
                if date in signals_by_date:
                    for signal in signals_by_date[date]:
                        symbol = signal.get("symbol", "")
                        action = signal.get("action", "")
                        strength = signal.get("strength", 0)
                        
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
                                
                                # 根据信号强度计算交易数量
                                # 这里使用简单的策略：信号强度为正数时买入，为负数时卖出
                                # 交易数量与信号强度成正比
                                if action == "buy" and strength > 0:
                                    # 计算可买入的数量
                                    available_cash = self.portfolio.cash * 0.95  # 保留5%现金
                                    max_quantity = int(available_cash / (price * (1 + self.config.commission_ratio + self.config.slippage_ratio)))
                                    quantity = int(max_quantity * min(strength, 1.0))  # 根据信号强度调整买入数量
                                    
                                    if quantity > 0:
                                        self.execute_trade(symbol, "buy", quantity, price, date)
                                
                                elif action == "sell" and strength < 0:
                                    # 计算可卖出的数量
                                    if symbol in self.portfolio.positions:
                                        position = self.portfolio.positions[symbol]
                                        quantity = int(position.quantity * min(abs(strength), 1.0))  # 根据信号强度调整卖出数量
                                        
                                        if quantity > 0:
                                            self.execute_trade(symbol, "sell", quantity, price, date)
            
            # 更新最后一天的投资组合价值
            self.update_portfolio_value(all_dates[-1], price_data)
            
            # 计算绩效指标
            performance_metrics = self.calculate_performance_metrics()
            
            logger.info("回测完成")
            
            return {
                "config": asdict(self.config),
                "portfolio": {
                    "final_value": self.portfolio.total_value,
                    "final_cash": self.portfolio.cash,
                    "positions": {symbol: asdict(position) for symbol, position in self.portfolio.positions.items()},
                    "daily_values": self.portfolio.daily_values
                },
                "trade_records": [asdict(record) for record in self.trade_records],
                "performance_metrics": performance_metrics
            }
            
        except Exception as e:
            logger.error(f"回测失败: {e}")
            return {}
    
    def save_results(self, results: Dict[str, Any], output_dir: str = "回测结果"):
        """
        保存回测结果
        
        Args:
            results: 回测结果
            output_dir: 输出目录
        """
        try:
            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存回测结果为JSON
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = os.path.join(output_dir, f"backtest_results_{timestamp}.json")
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"回测结果已保存到: {results_file}")
            
            # 生成可视化报告
            self.generate_visual_report(results, output_dir, timestamp)
            
        except Exception as e:
            logger.error(f"保存回测结果失败: {e}")
    
    def generate_visual_report(self, results: Dict[str, Any], output_dir: str, timestamp: str):
        """
        生成可视化报告
        
        Args:
            results: 回测结果
            output_dir: 输出目录
            timestamp: 时间戳
        """
        try:
            # 提取数据
            daily_values = results.get("portfolio", {}).get("daily_values", [])
            performance_metrics = results.get("performance_metrics", {})
            
            if not daily_values:
                logger.warning("没有每日资产数据，无法生成可视化报告")
                return
            
            # 提取日期和资产值
            dates = [pd.to_datetime(item[0]) for item in daily_values]
            values = [item[1] for item in daily_values]
            
            # 创建图表
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('回测结果报告', fontsize=16)
            
            # 资产曲线
            axes[0, 0].plot(dates, values, label='策略资产', color='blue')
            axes[0, 0].set_title('资产曲线')
            axes[0, 0].set_xlabel('日期')
            axes[0, 0].set_ylabel('资产值')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # 设置x轴日期格式
            axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            axes[0, 0].xaxis.set_major_locator(mdates.MonthLocator())
            plt.setp(axes[0, 0].xaxis.get_majorticklabels(), rotation=45)
            
            # 日收益率分布
            daily_returns = [(values[i] / values[i-1] - 1) for i in range(1, len(values))]
            axes[0, 1].hist(daily_returns, bins=30, alpha=0.75, color='green')
            axes[0, 1].set_title('日收益率分布')
            axes[0, 1].set_xlabel('日收益率')
            axes[0, 1].set_ylabel('频数')
            axes[0, 1].grid(True)
            
            # 回撤分析
            peak = np.maximum.accumulate(values)
            drawdown = (values - peak) / peak * 100
            axes[1, 0].fill_between(dates, drawdown, color='red', alpha=0.3)
            axes[1, 0].plot(dates, drawdown, color='red')
            axes[1, 0].set_title('回撤分析')
            axes[1, 0].set_xlabel('日期')
            axes[1, 0].set_ylabel('回撤 (%)')
            axes[1, 0].grid(True)
            
            # 设置x轴日期格式
            axes[1, 0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            axes[1, 0].xaxis.set_major_locator(mdates.MonthLocator())
            plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45)
            
            # 绩效指标表格
            metrics_text = ""
            if performance_metrics:
                metrics_text = (
                    f"总收益率: {performance_metrics.get('total_return', 0):.2%}\n"
                    f"年化收益率: {performance_metrics.get('annualized_return', 0):.2%}\n"
                    f"年化波动率: {performance_metrics.get('annualized_volatility', 0):.2%}\n"
                    f"夏普比率: {performance_metrics.get('sharpe_ratio', 0):.2f}\n"
                    f"最大回撤: {performance_metrics.get('max_drawdown', 0):.2%}\n"
                    f"胜率: {performance_metrics.get('win_rate', 0):.2%}\n"
                    f"基准收益率: {performance_metrics.get('benchmark_total_return', 0):.2%}\n"
                    f"超额收益: {performance_metrics.get('excess_return', 0):.2%}\n"
                    f"总交易次数: {performance_metrics.get('total_trades', 0)}"
                )
            
            axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center')
            axes[1, 1].set_title('绩效指标')
            axes[1, 1].axis('off')
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图表
            chart_file = os.path.join(output_dir, f"backtest_report_{timestamp}.png")
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"可视化报告已保存到: {chart_file}")
            
        except Exception as e:
            logger.error(f"生成可视化报告失败: {e}")

def load_trading_signals(trading_signals_file: str) -> List[Dict]:
    """
    加载交易信号数据
    
    Args:
        trading_signals_file: 交易信号文件路径
        
    Returns:
        交易信号数据列表
    """
    try:
        print(f"正在加载交易信号文件: {trading_signals_file}")
        with open(trading_signals_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"加载的数据类型: {type(data)}")
        
        # 检查数据格式并处理
        signals = []
        
        # 情况1: 数据是列表格式（新的整合信号格式）
        if isinstance(data, list):
            print("检测到列表格式的交易信号")
            for item in data:
                if isinstance(item, dict):
                    date = item.get("date", datetime.now().strftime("%Y-%m-%d"))
                    symbol = item.get("symbol", "")
                    action = item.get("action", "hold")
                    strength = item.get("strength", 0)
                    reason = item.get("reason", "")
                    
                    if symbol and action != "hold":
                        signals.append({
                            "date": date,
                            "symbol": symbol,
                            "action": action,
                            "strength": strength,
                            "price": 0,  # 回测时会获取实际价格
                            "reason": reason
                        })
        
        # 情况2: 数据是字典格式（旧的信号格式）
        elif isinstance(data, dict):
            print("检测到字典格式的交易信号")
            for symbol, signal_data in data.items():
                if symbol == "timestamp":
                    continue
                    
                # 从文件名提取日期
                file_name = os.path.basename(trading_signals_file)
                file_date = file_name.split("_")[-1].split(".")[0]
                # 确保日期格式正确
                if len(file_date) == 8 and file_date.isdigit():
                    date = f"{file_date[:4]}-{file_date[4:6]}-{file_date[6:8]}"
                else:
                    # 如果文件名格式不符合预期，使用当前日期
                    date = datetime.now().strftime("%Y-%m-%d")
                
                # 根据信号强度决定买卖方向
                signal_strength = signal_data.get("strength", 0)
                action = "buy" if signal_strength > 0.3 else "hold"  # 设置阈值为0.3
                
                print(f"股票 {symbol}: 信号强度 {signal_strength}, 动作 {action}")
                
                if action != "hold":  # 只添加非持有的信号
                    signals.append({
                        "date": date,
                        "symbol": symbol,
                        "action": action,
                        "strength": signal_strength,
                        "price": 0,  # 回测时会获取实际价格
                        "reason": f"信号强度: {signal_strength:.2f}"
                    })
        
        print(f"总共加载了 {len(signals)} 个交易信号")
        return signals
        
    except Exception as e:
        logger.error(f"加载交易信号数据失败: {e}")
        print(f"加载交易信号数据失败: {e}")
        return []

def main():
    """
    主函数
    """
    # 创建回测配置
    config = BacktestConfig(
        start_date="2025-09-18",
        end_date="2025-10-18",
        initial_cash=1000000.0,
        commission_ratio=0.0001,
        slippage_ratio=0.0001,
        benchmark="SHSE.000300",
        rebalance_frequency="daily"
    )
    
    # 创建回测引擎
    engine = BacktestEngine(config)
    
    # 加载交易信号数据
    trading_signals_file = "数据/交易记录/trading_signals_20251018_194106.json"
    signals_data = load_trading_signals(trading_signals_file)
    
    if not signals_data:
        logger.error("没有可用的交易信号数据")
        return
    
    # 运行回测
    results = engine.run_backtest(signals_data)
    
    if results:
        # 保存结果
        engine.save_results(results)
        
        # 打印绩效指标
        performance_metrics = results.get("performance_metrics", {})
        print("\n回测绩效指标:")
        for key, value in performance_metrics.items():
            if isinstance(value, float):
                if key in ["total_return", "annualized_return", "annualized_volatility", "max_drawdown", "win_rate", "benchmark_total_return", "excess_return"]:
                    print(f"{key}: {value:.2%}")
                else:
                    print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
    else:
        logger.error("回测失败")

if __name__ == "__main__":
    main()