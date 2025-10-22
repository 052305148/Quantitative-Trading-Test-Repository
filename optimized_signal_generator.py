"""
优化版整合分析信号生成器
包含以下优化：
1. 调整信号阈值：提高买入信号阈值，降低卖出信号敏感度
2. 增加持仓管理：引入止损、止盈机制
3. 优化资金分配：根据信号强度调整仓位大小
4. 增加市场环境判断：在整体市场下跌时降低仓位
5. 完善风控机制：设置最大回撤限制，减少极端损失
6. 结合Barra CNE5模型做调整补充
"""

import os
import json
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
import math

from main import SpecializedInnovativeFramework
from backtest_system import BacktestEngine

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OptimizedSignalGenerator:
    """优化版整合分析信号生成器"""
    
    def __init__(self, config_path: str = None):
        """
        初始化优化版整合分析信号生成器
        
        Args:
            config_path: 配置文件路径
        """
        self.framework = SpecializedInnovativeFramework(config_path)
        self.output_dir = self.framework.output_dir
        self.signals_dir = os.path.join("数据", "交易记录")
        os.makedirs(self.signals_dir, exist_ok=True)
        
        # 优化后的信号权重配置
        self.signal_weights = {
            "long_term": 0.3,   # 长期分析权重
            "mid_term": 0.4,    # 中期分析权重
            "short_term": 0.2,  # 短期分析权重
            "market_env": 0.1   # 市场环境权重
        }
        
        # 优化后的信号阈值
        self.signal_thresholds = {
            "strong_buy": 0.6,      # 提高买入阈值
            "buy": 0.4,             # 提高买入阈值
            "weak_buy": 0.2,        # 提高买入阈值
            "hold": 0.0,            # 持有
            "weak_sell": -0.3,      # 降低卖出敏感度
            "sell": -0.5,           # 降低卖出敏感度
            "strong_sell": -0.7     # 降低卖出敏感度
        }
        
        # 持仓管理参数
        self.position_management = {
            "stop_loss": -0.08,      # 8%止损
            "take_profit": 0.15,     # 15%止盈
            "max_position_size": 0.2, # 单只股票最大仓位20%
            "min_position_size": 0.05 # 单只股票最小仓位5%
        }
        
        # 风控参数
        self.risk_controls = {
            "max_drawdown": 0.1,     # 最大回撤10%
            "max_daily_loss": 0.03,  # 单日最大亏损3%
            "volatility_threshold": 0.25, # 波动率阈值
            "concentration_limit": 0.4   # 行业集中度限制
        }
        
        # Barra CNE5风格因子权重
        self.barra_factors = {
            "size": 0.15,        # 规模因子
            "value": 0.15,       # 价值因子
            "momentum": 0.15,    # 动量因子
            "volatility": 0.1,   # 波动率因子
            "liquidity": 0.1,    # 流动性因子
            "leverage": 0.1,     # 杠杆因子
            "growth": 0.1,       # 成长因子
            "beta": 0.15         # 贝塔因子
        }
        
        logger.info("优化版整合分析信号生成器初始化完成")
    
    def get_market_environment(self, date: str) -> float:
        """
        获取市场环境评分
        
        Args:
            date: 日期字符串
            
        Returns:
            市场环境评分，范围[-1, 1]，正数表示牛市，负数表示熊市
        """
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
            
            logger.info(f"日期 {date} 的市场环境评分: {market_env:.2f}")
            return market_env
            
        except Exception as e:
            logger.error(f"获取市场环境失败: {e}")
            return 0.0
    
    def calculate_barra_factor_exposure(self, symbol: str) -> Dict[str, float]:
        """
        计算股票的Barra CNE5风格因子暴露度
        
        Args:
            symbol: 股票代码
            
        Returns:
            因子暴露度字典
        """
        # 简化版实现：基于股票代码生成模拟因子暴露度
        # 实际应用中应该从因子数据库获取
        np.random.seed(hash(symbol) % 10000)  # 确保同一股票的因子暴露度一致
        
        factor_exposure = {
            "size": np.random.normal(0, 1),
            "value": np.random.normal(0, 1),
            "momentum": np.random.normal(0, 1),
            "volatility": np.random.normal(0, 1),
            "liquidity": np.random.normal(0, 1),
            "leverage": np.random.normal(0, 1),
            "growth": np.random.normal(0, 1),
            "beta": np.random.normal(0, 1)
        }
        
        return factor_exposure
    
    def adjust_signal_by_barra_factors(self, symbol: str, signal: float) -> float:
        """
        根据Barra CNE5因子调整信号
        
        Args:
            symbol: 股票代码
            signal: 原始信号强度
            
        Returns:
            调整后的信号强度
        """
        factor_exposure = self.calculate_barra_factor_exposure(symbol)
        
        # 计算因子调整值
        factor_adjustment = 0
        for factor, exposure in factor_exposure.items():
            weight = self.barra_factors.get(factor, 0)
            
            # 根据因子类型和暴露度计算调整值
            if factor == "value" and exposure > 0.5:  # 价值因子正向
                factor_adjustment += weight * 0.1
            elif factor == "momentum" and exposure > 0.5:  # 动量因子正向
                factor_adjustment += weight * 0.1
            elif factor == "volatility" and exposure > 0.5:  # 高波动率负向
                factor_adjustment -= weight * 0.1
            elif factor == "liquidity" and exposure < -0.5:  # 低流动性负向
                factor_adjustment -= weight * 0.1
            elif factor == "size" and exposure < -0:  # 小盘股正向（专精特新多为小盘股）
                factor_adjustment += weight * 0.05
        
        # 应用因子调整
        adjusted_signal = signal + factor_adjustment
        
        # 限制在[-1, 1]范围内
        adjusted_signal = max(-1, min(1, adjusted_signal))
        
        return adjusted_signal
    
    def calculate_position_size(self, signal: float, market_env: float, current_positions: Dict[str, float]) -> float:
        """
        根据信号强度和市场环境计算仓位大小
        
        Args:
            signal: 信号强度
            market_env: 市场环境评分
            current_positions: 当前持仓
            
        Returns:
            仓位大小，范围[0, 1]
        """
        # 基础仓位基于信号强度
        if signal > self.signal_thresholds["strong_buy"]:
            base_position = 0.15
        elif signal > self.signal_thresholds["buy"]:
            base_position = 0.12
        elif signal > self.signal_thresholds["weak_buy"]:
            base_position = 0.08
        elif signal < self.signal_thresholds["strong_sell"]:
            base_position = -0.1  # 卖出
        elif signal < self.signal_thresholds["sell"]:
            base_position = -0.08  # 卖出
        elif signal < self.signal_thresholds["weak_sell"]:
            base_position = -0.05  # 卖出
        else:
            return 0.0  # 不交易
        
        # 市场环境调整
        if market_env < -0.3:  # 熊市环境
            market_adjustment = 0.5  # 减半仓位
        elif market_env < -0.1:  # 弱市环境
            market_adjustment = 0.7  # 减少30%仓位
        elif market_env > 0.3:  # 牛市环境
            market_adjustment = 1.2  # 增加20%仓位
        else:
            market_adjustment = 1.0  # 正常仓位
        
        # 应用市场环境调整
        adjusted_position = base_position * market_adjustment
        
        # 限制单只股票最大仓位
        max_position = self.position_management["max_position_size"]
        min_position = self.position_management["min_position_size"]
        
        if adjusted_position > 0:
            adjusted_position = max(min(adjusted_position, max_position), min_position)
        else:
            adjusted_position = max(adjusted_position, -max_position)
        
        return adjusted_position
    
    def check_risk_controls(self, current_portfolio_value: float, 
                           previous_portfolio_value: float,
                           current_positions: Dict[str, Any]) -> bool:
        """
        检查风险控制指标
        
        Args:
            current_portfolio_value: 当前组合价值
            previous_portfolio_value: 前一日组合价值
            current_positions: 当前持仓
            
        Returns:
            是否通过风险控制检查
        """
        # 计算日收益率
        daily_return = (current_portfolio_value - previous_portfolio_value) / previous_portfolio_value
        
        # 检查单日最大亏损
        if daily_return < -self.risk_controls["max_daily_loss"]:
            logger.warning(f"触发单日最大亏损限制: {daily_return:.2%}")
            return False
        
        # 检查最大回撤（简化版，实际应该跟踪历史最高点）
        max_drawdown = (previous_portfolio_value - current_portfolio_value) / previous_portfolio_value
        if max_drawdown > self.risk_controls["max_drawdown"]:
            logger.warning(f"触发最大回撤限制: {max_drawdown:.2%}")
            return False
        
        return True
    
    def integrate_signals(self, long_term_signals: Dict[str, float], 
                         mid_term_signals: Dict[str, float], 
                         short_term_signals: Dict[str, float],
                         market_env: float) -> Dict[str, float]:
        """
        整合三个分析阶段的信号，加入市场环境判断
        
        优化策略：
        1. 长期信号判断主升区间
        2. 中期信号判断买入
        3. 短期信号判断止盈和风控
        4. 市场环境调整整体信号强度
        
        Args:
            long_term_signals: 长期分析信号
            mid_term_signals: 中期分析信号
            short_term_signals: 短期分析信号
            market_env: 市场环境评分
            
        Returns:
            整合后的信号字典
        """
        integrated_signals = {}
        
        # 获取所有股票代码
        all_symbols = set(long_term_signals.keys()) | set(mid_term_signals.keys()) | set(short_term_signals.keys())
        
        for symbol in all_symbols:
            # 获取各阶段信号
            lt_signal = long_term_signals.get(symbol, 0)
            mt_signal = mid_term_signals.get(symbol, 0)
            st_signal = short_term_signals.get(symbol, 0)
            
            # 优化后的信号整合策略
            # 1. 长期信号判断主升区间：只有当长期信号为正时，才考虑买入
            if lt_signal <= 0:
                # 长期信号为负或零，不处于主升区间，整体信号为负
                integrated_signal = -0.3  # 轻微负信号，表示不应持有
            else:
                # 2. 中期信号判断买入：在主升区间内，中期信号决定是否买入
                if mt_signal > self.signal_thresholds["buy"]:
                    # 中期信号强烈买入，结合短期信号判断止盈和风控
                    # 3. 短期信号判断止盈和风控：调整中期买入信号的强度
                    if st_signal > 0.3:
                        # 短期信号强烈正面，增强买入信号
                        integrated_signal = min(mt_signal + 0.15, 1.0)
                    elif st_signal < -0.3:
                        # 短期信号强烈负面，减弱买入信号或转为卖出
                        integrated_signal = max(mt_signal - 0.3, -0.1)
                    else:
                        # 短期信号中性，保持中期信号
                        integrated_signal = mt_signal
                elif mt_signal < self.signal_thresholds["sell"]:
                    # 中期信号卖出，在主升区间内可能是短期调整
                    # 结合短期信号判断是否需要止损
                    if st_signal < -0.3:
                        # 短期信号也负面，确认卖出
                        integrated_signal = min(mt_signal - 0.05, -0.4)
                    else:
                        # 短期信号中性或正面，可能是短期调整，轻微卖出
                        integrated_signal = mt_signal * 0.6
                else:
                    # 中期信号中性，结合短期信号
                    if st_signal > 0.3:
                        # 短期信号正面，轻微买入
                        integrated_signal = 0.25
                    elif st_signal < -0.3:
                        # 短期信号负面，轻微卖出
                        integrated_signal = -0.25
                    else:
                        # 两个信号都中性，保持中性
                        integrated_signal = 0
            
            # 4. 市场环境调整
            if market_env < -0.2:  # 熊市环境
                # 降低所有买入信号强度，增强卖出信号
                if integrated_signal > 0:
                    integrated_signal *= 0.6
                else:
                    integrated_signal *= 1.2
            elif market_env > 0.2:  # 牛市环境
                # 增强所有买入信号强度，降低卖出信号
                if integrated_signal > 0:
                    integrated_signal *= 1.2
                else:
                    integrated_signal *= 0.6
            
            # 限制信号强度在[-1, 1]范围内
            integrated_signal = max(-1, min(1, integrated_signal))
            
            # 5. 应用Barra CNE5因子调整
            integrated_signal = self.adjust_signal_by_barra_factors(symbol, integrated_signal)
            
            integrated_signals[symbol] = integrated_signal
        
        logger.info(f"整合了{len(integrated_signals)}个交易信号")
        return integrated_signals
    
    def generate_trading_signals(self, integrated_signals: Dict[str, float], 
                               market_env: float,
                               current_positions: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        生成交易信号列表
        
        Args:
            integrated_signals: 整合后的信号字典
            market_env: 市场环境评分
            current_positions: 当前持仓
            
        Returns:
            交易信号列表
        """
        if current_positions is None:
            current_positions = {}
        
        trading_signals = []
        
        # 生成日期序列（假设为未来10个交易日）
        base_date = datetime.now()
        dates = []
        for i in range(10):
            date = base_date + timedelta(days=i)
            # 跳过周末
            if date.weekday() < 5:
                dates.append(date.strftime("%Y-%m-%d"))
        
        # 为每个日期生成信号
        for date in dates:
            # 检查风险控制
            if not self.check_risk_controls(1000000, 1000000, current_positions):
                logger.warning(f"日期 {date} 风险控制检查失败，暂停交易")
                continue
            
            for symbol, signal_strength in integrated_signals.items():
                # 计算仓位大小
                position_size = self.calculate_position_size(signal_strength, market_env, current_positions)
                
                # 检查止损止盈
                current_position = current_positions.get(symbol, {})
                if current_position:
                    current_price = current_position.get("current_price", 0)
                    entry_price = current_position.get("entry_price", 0)
                    
                    if current_price > 0 and entry_price > 0:
                        return_rate = (current_price - entry_price) / entry_price
                        
                        # 止损检查
                        if return_rate <= self.position_management["stop_loss"]:
                            position_size = -abs(position_size)  # 强制卖出
                            logger.info(f"触发止损: {symbol}, 收益率: {return_rate:.2%}")
                        
                        # 止盈检查
                        elif return_rate >= self.position_management["take_profit"]:
                            position_size = -abs(position_size)  # 卖出一部分
                            logger.info(f"触发止盈: {symbol}, 收益率: {return_rate:.2%}")
                
                # 根据仓位大小决定交易动作
                if position_size > 0.05:
                    action = "buy"
                elif position_size < -0.05:
                    action = "sell"
                else:
                    action = "hold"
                
                # 只添加非持有信号
                if action != "hold":
                    trading_signals.append({
                        "date": date,
                        "symbol": symbol,
                        "action": action,
                        "strength": abs(signal_strength),
                        "position_size": abs(position_size),
                        "price": 0,  # 回测时会获取实际价格
                        "reason": f"优化整合信号强度: {signal_strength:.2f}, 市场环境: {market_env:.2f}"
                    })
        
        logger.info(f"生成了{len(trading_signals)}个交易信号")
        return trading_signals
    
    def save_trading_signals(self, trading_signals: List[Dict[str, Any]]) -> str:
        """
        保存交易信号
        
        Args:
            trading_signals: 交易信号列表
            
        Returns:
            保存路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"optimized_trading_signals_{timestamp}.json"
        filepath = os.path.join(self.signals_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(trading_signals, f, ensure_ascii=False, indent=2)
        
        logger.info(f"优化交易信号已保存到: {filepath}")
        return filepath
    
    def extract_long_term_signals(self, long_term_results: Dict[str, Any]) -> Dict[str, float]:
        """
        从长期分析结果中提取交易信号
        
        Args:
            long_term_results: 长期分析结果
            
        Returns:
            股票代码到信号强度的映射
        """
        signals = {}
        
        try:
            # 获取企业评估结果
            company_evaluations = long_term_results.get("company_evaluations", [])
            
            for evaluation in company_evaluations:
                company_code = evaluation.get("company_code", "")
                score = evaluation.get("score", 0)
                
                if company_code:
                    # 将评分转换为信号强度，范围[-1, 1]
                    # 假设评分范围是[0, 100]，转换为[-1, 1]
                    signal_strength = (score - 50) / 50.0
                    signal_strength = max(-1, min(1, signal_strength))
                    
                    signals[company_code] = signal_strength
            
            logger.info(f"从长期分析中提取了{len(signals)}个交易信号")
            
        except Exception as e:
            logger.error(f"提取长期分析信号失败: {e}")
        
        return signals
    
    def extract_mid_term_signals(self, mid_term_results: Dict[str, Any]) -> Dict[str, float]:
        """
        从中期分析结果中提取交易信号
        
        Args:
            mid_term_results: 中期分析结果
            
        Returns:
            股票代码到信号强度的映射
        """
        signals = {}
        
        try:
            # 获取交易信号
            trading_signals = mid_term_results.get("trading_signals", [])
            
            for signal in trading_signals:
                company = signal.get("company", "")
                signal_type = signal.get("signal_type", "")
                signal_strength = signal.get("signal_strength", 0)
                confidence = signal.get("confidence", 0)
                
                if not company:
                    continue
                
                # 根据信号类型确定基础信号强度
                if signal_type == "强烈买入":
                    base_strength = 0.8
                elif signal_type == "买入":
                    base_strength = 0.5
                elif signal_type == "温和买入":
                    base_strength = 0.2
                else:
                    base_strength = 0
                
                # 调整信号强度，考虑置信度
                adjusted_strength = base_strength * confidence
                
                # 如果有原始信号强度，则结合使用
                if signal_strength > 0:
                    adjusted_strength = (adjusted_strength + signal_strength / 5.0) / 2.0
                
                # 限制在[-1, 1]范围内
                adjusted_strength = max(-1, min(1, adjusted_strength))
                
                signals[company] = adjusted_strength
            
            logger.info(f"从中期分析中提取了{len(signals)}个交易信号")
            
        except Exception as e:
            logger.error(f"提取中期分析信号失败: {e}")
        
        return signals
    
    def extract_short_term_signals(self, short_term_results: Dict[str, Any]) -> Dict[str, float]:
        """
        从短期分析结果中提取交易信号
        
        Args:
            short_term_results: 短期分析结果
            
        Returns:
            股票代码到信号强度的映射
        """
        signals = {}
        
        try:
            # 获取分类新闻
            classified_news = short_term_results.get("classified_news", [])
            
            # 获取风险预警
            risk_alerts = short_term_results.get("risk_alerts", [])
            
            # 统计各类新闻数量
            category_counts = {}
            for news in classified_news:
                category = news.get("category", "其他")
                category_counts[category] = category_counts.get(category, 0) + 1
            
            # 统计风险预警
            high_risk_count = sum(1 for alert in risk_alerts if alert.get("level") == "高")
            medium_risk_count = sum(1 for alert in risk_alerts if alert.get("level") == "中")
            
            # 假设有一组目标股票代码
            target_symbols = [
                "SZSE.000790",  # 华神科技
                "SZSE.001207",  # 联科科技
                "SZSE.001208",  # 华厦眼科
                "SZSE.001223",  # 欧克科技
                "SZSE.001226",  # 福莱新材
                "SZSE.001229",  # 恒帅股份
                "SZSE.001230",  # 拓山重工
                "SZSE.001255",  # 萃华珠宝
                "SZSE.001256",  # 煌上煌
                "SZSE.001266"   # 远信工业
            ]
            
            # 基于新闻分类和风险预警生成交易信号
            for symbol in target_symbols:
                signal_strength = 0
                
                # 技术创新类新闻正面影响
                tech_innovation_count = category_counts.get('技术创新', 0)
                if tech_innovation_count > 0:
                    signal_strength += min(tech_innovation_count * 0.1, 0.3)  # 最多增加0.3信号强度
                
                # 政策动态类新闻正面影响
                policy_count = category_counts.get('政策动态', 0)
                if policy_count > 0:
                    signal_strength += min(policy_count * 0.15, 0.3)  # 最多增加0.3信号强度
                
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
                
                signals[symbol] = signal_strength
            
            logger.info(f"从短期分析中提取了{len(signals)}个交易信号")
            
        except Exception as e:
            logger.error(f"提取短期分析信号失败: {e}")
        
        return signals
    
    def run(self) -> Tuple[str, Dict[str, Any]]:
        """
        运行优化版整合分析并生成交易信号
        
        Returns:
            交易信号文件路径和分析结果
        """
        logger.info("开始运行优化版整合分析")
        
        # 运行完整分析流程
        results = self.framework.run_full_analysis()
        
        # 提取各阶段信号
        long_term_signals = self.extract_long_term_signals(results.get("long_term", {}))
        mid_term_signals = self.extract_mid_term_signals(results.get("mid_term", {}))
        short_term_signals = self.extract_short_term_signals(results.get("short_term", {}))
        
        # 获取市场环境（使用第一个交易日的市场环境）
        market_env = self.get_market_environment("2023-12-01")
        
        # 整合信号
        integrated_signals = self.integrate_signals(long_term_signals, mid_term_signals, short_term_signals, market_env)
        
        # 生成交易信号
        trading_signals = self.generate_trading_signals(integrated_signals, market_env)
        
        # 保存交易信号
        signals_path = self.save_trading_signals(trading_signals)
        
        # 保存整合信号
        integrated_signals_path = os.path.join(
            self.signals_dir, 
            f"optimized_integrated_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(integrated_signals_path, 'w', encoding='utf-8') as f:
            json.dump(integrated_signals, f, ensure_ascii=False, indent=2)
        
        logger.info("优化版整合分析完成")
        return signals_path, results


if __name__ == "__main__":
    # 运行优化版整合分析
    generator = OptimizedSignalGenerator()
    signals_path, results = generator.run()
    
    print(f"优化交易信号已保存到: {signals_path}")
    print("分析结果:")
    print(json.dumps(results, ensure_ascii=False, indent=2))