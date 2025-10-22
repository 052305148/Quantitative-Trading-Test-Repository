"""
整合长期、中期和短期分析并生成交易信号的程序
"""

import os
import json
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import pandas as pd

from main import SpecializedInnovativeFramework
from backtest_system import BacktestEngine

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class IntegratedAnalysisSignalGenerator:
    """整合分析信号生成器"""
    
    def __init__(self, config_path: str = None):
        """
        初始化整合分析信号生成器
        
        Args:
            config_path: 配置文件路径
        """
        self.framework = SpecializedInnovativeFramework(config_path)
        self.output_dir = self.framework.output_dir
        self.signals_dir = os.path.join("数据", "交易记录")
        os.makedirs(self.signals_dir, exist_ok=True)
        
        # 信号权重配置
        self.signal_weights = {
            "long_term": 0.4,   # 长期分析权重
            "mid_term": 0.4,    # 中期分析权重
            "short_term": 0.2   # 短期分析权重
        }
        
        logger.info("整合分析信号生成器初始化完成")
    
    def run_integrated_analysis(self) -> Dict[str, Any]:
        """
        运行整合分析
        
        Returns:
            分析结果字典
        """
        logger.info("开始运行整合分析")
        
        # 运行完整分析流程
        results = self.framework.run_full_analysis()
        
        logger.info("整合分析完成")
        return results
    
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
    
    def integrate_signals(self, long_term_signals: Dict[str, float], 
                         mid_term_signals: Dict[str, float], 
                         short_term_signals: Dict[str, float]) -> Dict[str, float]:
        """
        整合三个分析阶段的信号
        
        新策略：
        1. 长期信号判断主升区间
        2. 中期信号判断买入
        3. 短期信号判断止盈和风控
        
        Args:
            long_term_signals: 长期分析信号
            mid_term_signals: 中期分析信号
            short_term_signals: 短期分析信号
            
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
            
            # 新信号整合策略
            # 1. 长期信号判断主升区间：只有当长期信号为正时，才考虑买入
            if lt_signal <= 0:
                # 长期信号为负或零，不处于主升区间，整体信号为负
                integrated_signal = -0.5  # 轻微负信号，表示不应持有
            else:
                # 2. 中期信号判断买入：在主升区间内，中期信号决定是否买入
                if mt_signal > 0.2:
                    # 中期信号强烈买入，结合短期信号判断止盈和风控
                    # 3. 短期信号判断止盈和风控：调整中期买入信号的强度
                    if st_signal > 0.3:
                        # 短期信号强烈正面，增强买入信号
                        integrated_signal = min(mt_signal + 0.2, 1.0)
                    elif st_signal < -0.3:
                        # 短期信号强烈负面，减弱买入信号或转为卖出
                        integrated_signal = max(mt_signal - 0.4, -0.2)
                    else:
                        # 短期信号中性，保持中期信号
                        integrated_signal = mt_signal
                elif mt_signal < -0.2:
                    # 中期信号卖出，在主升区间内可能是短期调整
                    # 结合短期信号判断是否需要止损
                    if st_signal < -0.3:
                        # 短期信号也负面，确认卖出
                        integrated_signal = min(mt_signal - 0.1, -0.5)
                    else:
                        # 短期信号中性或正面，可能是短期调整，轻微卖出
                        integrated_signal = mt_signal * 0.5
                else:
                    # 中期信号中性，结合短期信号
                    if st_signal > 0.3:
                        # 短期信号正面，轻微买入
                        integrated_signal = 0.3
                    elif st_signal < -0.3:
                        # 短期信号负面，轻微卖出
                        integrated_signal = -0.3
                    else:
                        # 两个信号都中性，保持中性
                        integrated_signal = 0
            
            integrated_signals[symbol] = integrated_signal
        
        logger.info(f"整合了{len(integrated_signals)}个交易信号")
        return integrated_signals
    
    def generate_trading_signals(self, integrated_signals: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        生成交易信号列表
        
        Args:
            integrated_signals: 整合后的信号字典
            
        Returns:
            交易信号列表
        """
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
            for symbol, signal_strength in integrated_signals.items():
                # 根据信号强度决定交易动作
                if signal_strength > 0.2:
                    action = "buy"
                elif signal_strength < -0.2:
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
                        "price": 0,  # 回测时会获取实际价格
                        "reason": f"整合信号强度: {signal_strength:.2f}"
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
        filename = f"integrated_trading_signals_{timestamp}.json"
        filepath = os.path.join(self.signals_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(trading_signals, f, ensure_ascii=False, indent=2)
        
        logger.info(f"交易信号已保存到: {filepath}")
        return filepath
    
    def run(self) -> Tuple[str, Dict[str, Any]]:
        """
        运行整合分析并生成交易信号
        
        Returns:
            交易信号文件路径和分析结果
        """
        # 运行整合分析
        results = self.run_integrated_analysis()
        
        # 提取各阶段信号
        long_term_signals = self.extract_long_term_signals(results.get("long_term", {}))
        mid_term_signals = self.extract_mid_term_signals(results.get("mid_term", {}))
        short_term_signals = self.extract_short_term_signals(results.get("short_term", {}))
        
        # 整合信号
        integrated_signals = self.integrate_signals(long_term_signals, mid_term_signals, short_term_signals)
        
        # 生成交易信号
        trading_signals = self.generate_trading_signals(integrated_signals)
        
        # 保存交易信号
        signals_path = self.save_trading_signals(trading_signals)
        
        # 保存整合信号
        integrated_signals_path = os.path.join(
            self.signals_dir, 
            f"integrated_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(integrated_signals_path, 'w', encoding='utf-8') as f:
            json.dump(integrated_signals, f, ensure_ascii=False, indent=2)
        
        return signals_path, results


def main():
    """主函数"""
    # 创建整合分析信号生成器
    generator = IntegratedAnalysisSignalGenerator()
    
    # 运行整合分析并生成交易信号
    signals_path, results = generator.run()
    
    print(f"\n整合分析完成！")
    print(f"交易信号已保存到: {signals_path}")
    
    # 打印统计信息
    print("\n各阶段分析统计:")
    
    if "long_term" in results and results["long_term"]:
        stats = results["long_term"].get("statistics", {})
        print(f"\n长期分析:")
        print(f"  学术论文数量: {stats.get('total_papers', 0)}")
        print(f"  企业年报数量: {stats.get('total_reports', 0)}")
        print(f"  评估企业数量: {stats.get('evaluated_companies', 0)}")
    
    if "mid_term" in results and results["mid_term"]:
        stats = results["mid_term"].get("statistics", {})
        print(f"\n中期分析:")
        print(f"  季报数量: {stats.get('total_quarterly_reports', 0)}")
        print(f"  月报数量: {stats.get('total_monthly_reports', 0)}")
        print(f"  提取事件数量: {stats.get('total_events', 0)}")
        print(f"  预期差分析数量: {stats.get('total_gaps', 0)}")
        print(f"  交易信号数量: {stats.get('trading_signals', 0)}")
    
    if "short_term" in results and results["short_term"]:
        stats = results["short_term"].get("statistics", {})
        print(f"\n短期分析:")
        print(f"  爬取新闻数量: {stats.get('total_news', 0)}")
        print(f"  有效分类新闻数: {stats.get('classified_news', 0)}")
        print(f"  风险事件总数: {stats.get('total_risk_events', 0)}")
        print(f"  风险预警总数: {stats.get('total_risk_alerts', 0)}")
    
    # 打印交易信号统计
    with open(signals_path, 'r', encoding='utf-8') as f:
        signals = json.load(f)
    
    buy_signals = sum(1 for s in signals if s.get("action") == "buy")
    sell_signals = sum(1 for s in signals if s.get("action") == "sell")
    
    print(f"\n整合交易信号统计:")
    print(f"  买入信号: {buy_signals}")
    print(f"  卖出信号: {sell_signals}")
    print(f"  总信号数: {len(signals)}")
    
    # 返回信号路径，供后续使用
    return signals_path


if __name__ == "__main__":
    main()