"""
交易程序管理器
用于管理基于新闻分析的量化交易策略的执行和监控
"""

import os
import sys
import json
import time
import logging
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

# 添加短期分析模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), '短期分析'))
from short_term_analysis import ShortTermAnalyzer, ShortTermAnalysisConfig

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TradingConfig:
    """交易配置类"""
    # 策略配置
    strategy_id: str = "news_based_trading_strategy"
    strategy_file: str = "news_trading_strategy.py"
    
    # 掘金平台配置
    token: str = "{{token}}"  # 需要替换为实际token
    mode: str = "MODE_BACKTEST"  # MODE_BACKTEST 或 MODE_LIVE
    
    # 回测配置
    backtest_start_time: str = None  # 自动设置为30天前
    backtest_end_time: str = None    # 自动设置为当前时间
    backtest_adjust: str = "ADJUST_PREV"  # ADJUST_NONE, ADJUST_PREV, ADJUST_POST
    backtest_initial_cash: int = 10000000  # 初始资金
    backtest_commission_ratio: float = 0.0001  # 佣金比例
    backtest_slippage_ratio: float = 0.0001    # 滑点比例
    backtest_match_mode: int = 1  # 0: 下一tick/bar开盘价撮合, 1: 当前tick/bar收盘价撮合
    
    # 目标股票
    target_symbols: List[str] = None
    
    # 交易配置
    max_position_ratio: float = 0.1  # 单只股票最大持仓比例
    rebalance_frequency: str = "1d"  # 调仓频率
    news_analysis_time: str = "09:00:00"  # 新闻分析时间
    position_evaluation_time: str = "15:00:00"  # 持仓评估时间
    
    def __post_init__(self):
        """后处理初始化"""
        if self.target_symbols is None:
            self.target_symbols = [
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
        
        if self.backtest_start_time is None:
            self.backtest_start_time = str(datetime.now() - timedelta(days=30))[:19]
        
        if self.backtest_end_time is None:
            self.backtest_end_time = str(datetime.now())[:19]


class TradingManager:
    """交易程序管理器"""
    
    def __init__(self, config: TradingConfig):
        """
        初始化交易管理器
        
        Args:
            config: 交易配置
        """
        self.config = config
        self.is_running = False
        self.process = None
        self.news_analyzer = None
        
        # 创建必要的目录
        os.makedirs("数据/交易记录", exist_ok=True)
        os.makedirs("结果/交易报告", exist_ok=True)
        
        # 初始化新闻分析系统
        try:
            news_config = ShortTermAnalysisConfig()
            self.news_analyzer = ShortTermAnalyzer(news_config)
            logger.info("新闻分析系统初始化成功")
        except Exception as e:
            logger.error(f"新闻分析系统初始化失败: {e}")
        
        logger.info("交易管理器初始化完成")
    
    def start_strategy(self) -> bool:
        """
        启动交易策略
        
        Returns:
            bool: 启动是否成功
        """
        if self.is_running:
            logger.warning("交易策略已在运行中")
            return False
        
        try:
            # 准备策略参数
            strategy_params = {
                'strategy_id': self.config.strategy_id,
                'filename': self.config.strategy_file,
                'mode': self.config.mode,
                'token': self.config.token,
                'backtest_start_time': self.config.backtest_start_time,
                'backtest_end_time': self.config.backtest_end_time,
                'backtest_adjust': self.config.backtest_adjust,
                'backtest_initial_cash': self.config.backtest_initial_cash,
                'backtest_commission_ratio': self.config.backtest_commission_ratio,
                'backtest_slippage_ratio': self.config.backtest_slippage_ratio,
                'backtest_match_mode': self.config.backtest_match_mode
            }
            
            # 构建命令行参数
            cmd = [
                'python', self.config.strategy_file,
                f"--strategy_id={strategy_params['strategy_id']}",
                f"--filename={strategy_params['filename']}",
                f"--mode={strategy_params['mode']}",
                f"--token={strategy_params['token']}",
                f"--backtest_start_time={strategy_params['backtest_start_time']}",
                f"--backtest_end_time={strategy_params['backtest_end_time']}",
                f"--backtest_adjust={strategy_params['backtest_adjust']}",
                f"--backtest_initial_cash={strategy_params['backtest_initial_cash']}",
                f"--backtest_commission_ratio={strategy_params['backtest_commission_ratio']}",
                f"--backtest_slippage_ratio={strategy_params['backtest_slippage_ratio']}",
                f"--backtest_match_mode={strategy_params['backtest_match_mode']}"
            ]
            
            # 启动策略进程
            logger.info(f"启动交易策略: {self.config.strategy_file}")
            self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            self.is_running = True
            
            # 保存启动记录
            self._save_start_record()
            
            logger.info("交易策略启动成功")
            return True
            
        except Exception as e:
            logger.error(f"启动交易策略失败: {e}")
            return False
    
    def stop_strategy(self) -> bool:
        """
        停止交易策略
        
        Returns:
            bool: 停止是否成功
        """
        if not self.is_running:
            logger.warning("交易策略未在运行")
            return False
        
        try:
            if self.process:
                self.process.terminate()
                self.process.wait(timeout=10)
                self.process = None
            
            self.is_running = False
            
            # 保存停止记录
            self._save_stop_record()
            
            logger.info("交易策略已停止")
            return True
            
        except Exception as e:
            logger.error(f"停止交易策略失败: {e}")
            return False
    
    def get_strategy_status(self) -> Dict[str, Any]:
        """
        获取策略状态
        
        Returns:
            Dict: 策略状态信息
        """
        status = {
            'is_running': self.is_running,
            'config': asdict(self.config),
            'start_time': None,
            'stop_time': None,
            'last_news_analysis': None,
            'last_position_evaluation': None
        }
        
        # 从记录文件中获取状态信息
        try:
            status_file = os.path.join("数据", "交易记录", "strategy_status.json")
            if os.path.exists(status_file):
                with open(status_file, 'r', encoding='utf-8') as f:
                    saved_status = json.load(f)
                    status.update(saved_status)
        except Exception as e:
            logger.error(f"读取策略状态失败: {e}")
        
        return status
    
    def run_news_analysis(self) -> bool:
        """
        运行新闻分析
        
        Returns:
            bool: 分析是否成功
        """
        if not self.news_analyzer:
            logger.error("新闻分析系统未初始化")
            return False
        
        try:
            logger.info("开始执行新闻分析")
            
            # 爬取新闻
            news_data = self.news_analyzer.crawl_news()
            logger.info(f"爬取到 {len(news_data)} 条新闻")
            
            if not news_data:
                logger.warning("未获取到新闻数据")
                return False
            
            # 分类新闻
            classified_news = self.news_analyzer.classify_news(news_data)
            logger.info(f"分类了 {len(classified_news)} 条新闻")
            
            # 分析风险
            risk_events, risk_alerts = self.news_analyzer.analyze_risks(classified_news)
            logger.info(f"识别出 {len(risk_events)} 个风险事件，{len(risk_alerts)} 个风险预警")
            
            # 保存分析结果
            self._save_news_analysis_result(news_data, classified_news, risk_events, risk_alerts)
            
            logger.info("新闻分析完成")
            return True
            
        except Exception as e:
            logger.error(f"新闻分析失败: {e}")
            return False
    
    def generate_trading_signals(self) -> Dict[str, Any]:
        """
        生成交易信号
        
        新策略：
        1. 长期信号判断主升区间
        2. 中期信号判断买入
        3. 短期信号判断止盈和风控
        
        Returns:
            Dict: 交易信号
        """
        try:
            # 获取最新的新闻分析结果
            analysis_result = self._get_latest_news_analysis_result()
            if not analysis_result:
                logger.warning("未找到新闻分析结果，无法生成交易信号")
                return {}
            
            classified_news = analysis_result.get('classified_news', [])
            risk_alerts = analysis_result.get('risk_alerts', [])
            
            # 统计各类新闻数量
            category_counts = {}
            for news in classified_news:
                category = news.get('category', '其他')
                category_counts[category] = category_counts.get(category, 0) + 1
            
            # 统计风险预警
            high_risk_count = sum(1 for alert in risk_alerts if alert.get('level') == '高')
            medium_risk_count = sum(1 for alert in risk_alerts if alert.get('level') == '中')
            
            # 获取长期和中期分析结果
            long_term_result = self._get_latest_long_term_result()
            mid_term_result = self._get_latest_mid_term_result()
            
            # 基于新闻分类和风险预警生成交易信号
            signals = {}
            for symbol in self.config.target_symbols:
                # 短期信号（基于新闻）
                short_term_signal = 0
                
                # 技术创新类新闻正面影响
                tech_innovation_count = category_counts.get('技术创新', 0)
                if tech_innovation_count > 0:
                    short_term_signal += min(tech_innovation_count * 0.1, 0.3)  # 最多增加0.3信号强度
                
                # 政策动态类新闻正面影响
                policy_count = category_counts.get('政策动态', 0)
                if policy_count > 0:
                    short_term_signal += min(policy_count * 0.15, 0.3)  # 最多增加0.3信号强度
                
                # 行业动态类新闻轻微正面影响
                industry_count = category_counts.get('行业动态', 0)
                if industry_count > 0:
                    short_term_signal += min(industry_count * 0.05, 0.2)  # 最多增加0.2信号强度
                
                # 风险预警负面影响
                if high_risk_count > 0:
                    short_term_signal -= min(high_risk_count * 0.2, 0.5)  # 最多减少0.5信号强度
                if medium_risk_count > 0:
                    short_term_signal -= min(medium_risk_count * 0.1, 0.3)  # 最多减少0.3信号强度
                
                # 限制短期信号强度在[-1, 1]范围内
                short_term_signal = max(-1, min(1, short_term_signal))
                
                # 获取长期信号（判断主升区间）
                long_term_signal = self._extract_long_term_signal(symbol, long_term_result)
                
                # 获取中期信号（判断买入）
                mid_term_signal = self._extract_mid_term_signal(symbol, mid_term_result)
                
                # 应用新信号整合策略
                integrated_signal = self._integrate_signals(long_term_signal, mid_term_signal, short_term_signal)
                
                # 存储交易信号
                signals[symbol] = {
                    'strength': integrated_signal,
                    'long_term_strength': long_term_signal,
                    'mid_term_strength': mid_term_signal,
                    'short_term_strength': short_term_signal,
                    'category_counts': category_counts,
                    'risk_counts': {'high': high_risk_count, 'medium': medium_risk_count},
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            
            # 保存交易信号
            self._save_trading_signals(signals)
            
            logger.info(f"生成交易信号完成，共 {len(signals)} 只股票")
            return signals
            
        except Exception as e:
            logger.error(f"生成交易信号失败: {e}")
            return {}
    
    def _extract_long_term_signal(self, symbol: str, long_term_result: Dict[str, Any]) -> float:
        """
        从长期分析结果中提取信号
        
        Args:
            symbol: 股票代码
            long_term_result: 长期分析结果
            
        Returns:
            长期信号强度
        """
        try:
            if not long_term_result:
                return 0
            
            company_evaluations = long_term_result.get("company_evaluations", [])
            
            for evaluation in company_evaluations:
                company_code = evaluation.get("company_code", "")
                score = evaluation.get("score", 0)
                
                # 检查是否匹配当前股票
                if symbol.endswith(company_code) or company_code.endswith(symbol.split('.')[-1]):
                    # 将评分转换为信号强度，范围[-1, 1]
                    # 假设评分范围是[0, 100]，转换为[-1, 1]
                    signal_strength = (score - 50) / 50.0
                    return max(-1, min(1, signal_strength))
            
            return 0
            
        except Exception as e:
            logger.error(f"提取长期信号失败: {e}")
            return 0
    
    def _extract_mid_term_signal(self, symbol: str, mid_term_result: Dict[str, Any]) -> float:
        """
        从中期分析结果中提取信号
        
        Args:
            symbol: 股票代码
            mid_term_result: 中期分析结果
            
        Returns:
            中期信号强度
        """
        try:
            if not mid_term_result:
                return 0
            
            trading_signals = mid_term_result.get("trading_signals", [])
            
            for signal in trading_signals:
                company = signal.get("company", "")
                signal_type = signal.get("signal_type", "")
                signal_strength = signal.get("signal_strength", 0)
                confidence = signal.get("confidence", 0)
                
                # 检查是否匹配当前股票
                if symbol.endswith(company) or company.endswith(symbol.split('.')[-1]):
                    # 根据信号类型确定基础信号强度
                    if signal_type == "强烈买入":
                        base_strength = 0.8
                    elif signal_type == "买入":
                        base_strength = 0.5
                    elif signal_type == "温和买入":
                        base_strength = 0.2
                    elif signal_type == "卖出":
                        base_strength = -0.5
                    elif signal_type == "强烈卖出":
                        base_strength = -0.8
                    else:
                        base_strength = 0
                    
                    # 调整信号强度，考虑置信度
                    adjusted_strength = base_strength * confidence
                    
                    # 如果有原始信号强度，则结合使用
                    if signal_strength > 0:
                        adjusted_strength = (adjusted_strength + signal_strength / 5.0) / 2.0
                    
                    # 限制在[-1, 1]范围内
                    return max(-1, min(1, adjusted_strength))
            
            return 0
            
        except Exception as e:
            logger.error(f"提取中期信号失败: {e}")
            return 0
    
    def _integrate_signals(self, long_term_signal: float, mid_term_signal: float, short_term_signal: float) -> float:
        """
        整合三个阶段的信号
        
        新策略：
        1. 长期信号判断主升区间
        2. 中期信号判断买入
        3. 短期信号判断止盈和风控
        
        Args:
            long_term_signal: 长期信号
            mid_term_signal: 中期信号
            short_term_signal: 短期信号
            
        Returns:
            整合后的信号
        """
        # 1. 长期信号判断主升区间：只有当长期信号为正时，才考虑买入
        if long_term_signal <= 0:
            # 长期信号为负或零，不处于主升区间，整体信号为负
            integrated_signal = -0.5  # 轻微负信号，表示不应持有
        else:
            # 2. 中期信号判断买入：在主升区间内，中期信号决定是否买入
            if mid_term_signal > 0.2:
                # 中期信号强烈买入，结合短期信号判断止盈和风控
                # 3. 短期信号判断止盈和风控：调整中期买入信号的强度
                if short_term_signal > 0.3:
                    # 短期信号强烈正面，增强买入信号
                    integrated_signal = min(mid_term_signal + 0.2, 1.0)
                elif short_term_signal < -0.3:
                    # 短期信号强烈负面，减弱买入信号或转为卖出
                    integrated_signal = max(mid_term_signal - 0.4, -0.2)
                else:
                    # 短期信号中性，保持中期信号
                    integrated_signal = mid_term_signal
            elif mid_term_signal < -0.2:
                # 中期信号卖出，在主升区间内可能是短期调整
                # 结合短期信号判断是否需要止损
                if short_term_signal < -0.3:
                    # 短期信号也负面，确认卖出
                    integrated_signal = min(mid_term_signal - 0.1, -0.5)
                else:
                    # 短期信号中性或正面，可能是短期调整，轻微卖出
                    integrated_signal = mid_term_signal * 0.5
            else:
                # 中期信号中性，结合短期信号
                if short_term_signal > 0.3:
                    # 短期信号正面，轻微买入
                    integrated_signal = 0.3
                elif short_term_signal < -0.3:
                    # 短期信号负面，轻微卖出
                    integrated_signal = -0.3
                else:
                    # 两个信号都中性，保持中性
                    integrated_signal = 0
        
        return integrated_signal
    
    def _get_latest_long_term_result(self) -> Dict[str, Any]:
        """获取最新的长期分析结果"""
        try:
            # 查找最新的长期分析结果文件
            result_dir = os.path.join("结果", "长期分析")
            if not os.path.exists(result_dir):
                return {}
            
            files = [f for f in os.listdir(result_dir) if f.endswith('.json')]
            if not files:
                return {}
            
            # 按修改时间排序，获取最新的文件
            files.sort(key=lambda x: os.path.getmtime(os.path.join(result_dir, x)), reverse=True)
            latest_file = os.path.join(result_dir, files[0])
            
            with open(latest_file, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        except Exception as e:
            logger.error(f"获取长期分析结果失败: {e}")
            return {}
    
    def _get_latest_mid_term_result(self) -> Dict[str, Any]:
        """获取最新的中期分析结果"""
        try:
            # 查找最新的中期分析结果文件
            result_dir = os.path.join("结果", "中期分析")
            if not os.path.exists(result_dir):
                return {}
            
            files = [f for f in os.listdir(result_dir) if f.endswith('.json')]
            if not files:
                return {}
            
            # 按修改时间排序，获取最新的文件
            files.sort(key=lambda x: os.path.getmtime(os.path.join(result_dir, x)), reverse=True)
            latest_file = os.path.join(result_dir, files[0])
            
            with open(latest_file, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        except Exception as e:
            logger.error(f"获取中期分析结果失败: {e}")
            return {}
    
    def _save_start_record(self):
        """保存策略启动记录"""
        try:
            record = {
                'action': 'start',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'config': asdict(self.config)
            }
            
            record_file = os.path.join("数据", "交易记录", "strategy_operations.json")
            records = []
            
            if os.path.exists(record_file):
                with open(record_file, 'r', encoding='utf-8') as f:
                    records = json.load(f)
            
            records.append(record)
            
            with open(record_file, 'w', encoding='utf-8') as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
            
            # 更新状态文件
            status_file = os.path.join("数据", "交易记录", "strategy_status.json")
            status = self.get_strategy_status()
            status['start_time'] = record['timestamp']
            status['is_running'] = True
            
            with open(status_file, 'w', encoding='utf-8') as f:
                json.dump(status, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"保存启动记录失败: {e}")
    
    def _save_stop_record(self):
        """保存策略停止记录"""
        try:
            record = {
                'action': 'stop',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            record_file = os.path.join("数据", "交易记录", "strategy_operations.json")
            records = []
            
            if os.path.exists(record_file):
                with open(record_file, 'r', encoding='utf-8') as f:
                    records = json.load(f)
            
            records.append(record)
            
            with open(record_file, 'w', encoding='utf-8') as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
            
            # 更新状态文件
            status_file = os.path.join("数据", "交易记录", "strategy_status.json")
            status = self.get_strategy_status()
            status['stop_time'] = record['timestamp']
            status['is_running'] = False
            
            with open(status_file, 'w', encoding='utf-8') as f:
                json.dump(status, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"保存停止记录失败: {e}")
    
    def _save_news_analysis_result(self, news_data, classified_news, risk_events, risk_alerts):
        """保存新闻分析结果"""
        try:
            result = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'news_count': len(news_data),
                'classified_news_count': len(classified_news),
                'risk_events_count': len(risk_events),
                'risk_alerts_count': len(risk_alerts),
                'category_counts': {},
                'risk_level_counts': {},
                # 保存完整的分类新闻和风险预警数据
                'classified_news': classified_news,
                'risk_alerts': risk_alerts
            }
            
            # 统计新闻分类
            for news in classified_news:
                category = news.get('category', '其他')
                result['category_counts'][category] = result['category_counts'].get(category, 0) + 1
            
            # 统计风险等级
            for alert in risk_alerts:
                level = alert.get('level', '未知')
                result['risk_level_counts'][level] = result['risk_level_counts'].get(level, 0) + 1
            
            # 保存结果
            result_file = os.path.join("数据", "交易记录", f"news_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            # 更新状态文件
            status_file = os.path.join("数据", "交易记录", "strategy_status.json")
            status = self.get_strategy_status()
            status['last_news_analysis'] = result['timestamp']
            
            with open(status_file, 'w', encoding='utf-8') as f:
                json.dump(status, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"保存新闻分析结果失败: {e}")
    
    def _get_latest_news_analysis_result(self) -> Optional[Dict[str, Any]]:
        """获取最新的新闻分析结果"""
        try:
            records_dir = os.path.join("数据", "交易记录")
            files = [f for f in os.listdir(records_dir) if f.startswith('news_analysis_') and f.endswith('.json')]
            
            if not files:
                return None
            
            # 按时间排序，获取最新的文件
            files.sort(reverse=True)
            latest_file = os.path.join(records_dir, files[0])
            
            with open(latest_file, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        except Exception as e:
            logger.error(f"获取最新新闻分析结果失败: {e}")
            return None
    
    def _save_trading_signals(self, signals: Dict[str, Any]):
        """保存交易信号"""
        try:
            signals_file = os.path.join("数据", "交易记录", f"trading_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(signals_file, 'w', encoding='utf-8') as f:
                json.dump(signals, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"保存交易信号失败: {e}")


if __name__ == '__main__':
    # 示例用法
    config = TradingConfig()
    manager = TradingManager(config)
    
    # 运行新闻分析
    manager.run_news_analysis()
    
    # 生成交易信号
    signals = manager.generate_trading_signals()
    print(f"生成交易信号: {len(signals)} 只股票")
    
    # 获取策略状态
    status = manager.get_strategy_status()
    print(f"策略状态: {status['is_running']}")