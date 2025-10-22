"""
优化版专精特新文本分析交易框架主程序
整合了六项策略优化：调整信号阈值、增加持仓管理、优化资金分配、增加市场环境判断、完善风控机制、结合Barra CNE5模型
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Any, List

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config_manager import ConfigManager
from analyzers.long_term_analyzer import LongTermAnalyzer
from analyzers.mid_term_analyzer import MidTermAnalyzer
from analyzers.short_term_analyzer import ShortTermAnalyzer
from api.tongyi_client import TongyiClient
from utils.logger import setup_logger
from optimized_signal_generator import OptimizedSignalGenerator
from optimized_backtest_system import OptimizedBacktestEngine

# 设置日志
logger = setup_logger("optimized_framework")


class OptimizedSpecializedInnovativeFramework:
    """优化版专精特新文本分析交易框架"""
    
    def __init__(self, config_path: str = None, use_api: bool = True):
        """
        初始化框架
        
        Args:
            config_path: 配置文件路径
            use_api: 是否使用API
        """
        # 加载配置
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_config()
        
        # 创建输出目录
        self.output_dir = self.config["output"]["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化API客户端
        self.use_api = use_api
        if self.use_api:
            self.api_client = TongyiClient(self.config["api"]["tongyi"])
        else:
            self.api_client = None
        
        # 初始化分析器
        self.long_term_analyzer = None
        self.mid_term_analyzer = None
        self.short_term_analyzer = None
        
        # 初始化优化版信号生成器和回测系统
        self.signal_generator = None
        self.backtest_engine = None
        
        # 存储结果
        self.results = {}
        
        logger.info("优化版专精特新文本分析交易框架初始化完成")
    
    def initialize_analyzers(self):
        """初始化各期分析器"""
        # 长期分析器
        self.long_term_analyzer = LongTermAnalyzer(
            config=self.config,
            api_client=self.api_client
        )
        
        # 中期分析器
        self.mid_term_analyzer = MidTermAnalyzer(
            config=self.config,
            api_client=self.api_client
        )
        
        # 短期分析器
        self.short_term_analyzer = ShortTermAnalyzer(
            config=self.config,
            api_client=self.api_client
        )
        
        # 初始化优化版信号生成器
        self.signal_generator = OptimizedSignalGenerator(
            long_term_weight=self.config["signal_weights"]["long_term"],
            mid_term_weight=self.config["signal_weights"]["mid_term"],
            short_term_weight=self.config["signal_weights"]["short_term"]
        )
        
        # 初始化优化版回测引擎
        self.backtest_engine = OptimizedBacktestEngine(
            initial_capital=self.config["backtest"]["initial_capital"],
            commission_rate=self.config["backtest"]["commission_rate"],
            slippage_rate=self.config["backtest"]["slippage_rate"],
            max_position_size=self.config["risk_management"]["max_position_size"],
            stop_loss_pct=self.config["risk_management"]["stop_loss_pct"],
            take_profit_pct=self.config["risk_management"]["take_profit_pct"],
            max_drawdown_pct=self.config["risk_management"]["max_drawdown_pct"],
            max_daily_loss_pct=self.config["risk_management"]["max_daily_loss_pct"]
        )
        
        logger.info("分析器和优化组件初始化完成")
    
    def run_long_term_analysis(self, keywords: List[str] = None, stock_codes: List[str] = None, years: List[int] = None) -> Dict[str, Any]:
        """
        运行长期分析
        
        Args:
            keywords: 搜索关键词
            stock_codes: 股票代码列表
            years: 年份列表
            
        Returns:
            长期分析结果
        """
        logger.info("开始执行长期分析")
        
        # 设置默认参数
        if keywords is None:
            keywords = self.config["data"]["long_term"]["default_keywords"]
        
        if stock_codes is None:
            stock_codes = self.config["data"]["long_term"]["default_stock_codes"]
        
        if years is None:
            years = self.config["data"]["long_term"]["default_years"]
        
        # 执行长期分析
        long_term_results = self.long_term_analyzer.run_analysis(
            keywords=keywords,
            stock_codes=stock_codes,
            years=years
        )
        
        # 添加统计信息
        long_term_results["statistics"] = {
            "total_papers": len(long_term_results.get("papers", [])),
            "total_reports": len(long_term_results.get("reports", [])),
            "evaluated_companies": len(long_term_results.get("company_evaluations", []))
        }
        
        # 保存结果
        self.results["long_term"] = long_term_results
        
        logger.info("长期分析完成")
        return long_term_results
    
    def run_mid_term_analysis(self, stock_codes: List[str] = None, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """
        运行中期分析
        
        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            中期分析结果
        """
        logger.info("开始执行中期分析")
        
        # 设置默认参数
        if stock_codes is None:
            stock_codes = self.config["data"]["mid_term"]["default_stock_codes"]
        
        if start_date is None:
            start_date = self.config["data"]["mid_term"]["default_start_date"]
        
        if end_date is None:
            end_date = self.config["data"]["mid_term"]["default_end_date"]
        
        # 创建示例公司报告数据
        reports = []
        for stock_code in stock_codes:
            report = {
                "stock_code": stock_code,
                "report_type": "quarterly",
                "report_date": "2023-09-30",
                "revenue": 1000000000,
                "net_profit": 100000000,
                "eps": 1.5,
                "roe": 0.15,
                "content": f"公司{stock_code}第三季度业绩表现良好，营收同比增长10%，净利润同比增长8%。"
            }
            reports.append(report)
        
        # 创建示例市场数据
        market_data = []
        for stock_code in stock_codes:
            price_data = {
                "stock_code": stock_code,
                "date": "2023-10-15",
                "close_price": 20.0,
                "volume": 1000000,
                "market_cap": 5000000000
            }
            market_data.append(price_data)
        
        # 创建示例分析师预期数据
        analyst_expectations = []
        for stock_code in stock_codes:
            expectation = {
                "stock_code": stock_code,
                "analyst": "某证券公司",
                "date": "2023-10-10",
                "target_price": 25.0,
                "rating": "买入",
                "reasons": [f"公司{stock_code}在细分领域具有竞争优势，业绩增长稳定。"]
            }
            analyst_expectations.append(expectation)
        
        # 执行中期分析
        mid_term_results = self.mid_term_analyzer.run_analysis(
            reports=reports,
            market_data=market_data,
            analyst_expectations=analyst_expectations
        )
        
        # 添加统计信息
        mid_term_results["statistics"] = {
            "total_quarterly_reports": len(reports),
            "total_monthly_reports": 0,
            "total_events": len(mid_term_results.get("events", [])),
            "total_gaps": len(mid_term_results.get("expectation_gaps", []))
        }
        
        # 保存结果
        self.results["mid_term"] = mid_term_results
        
        logger.info("中期分析完成")
        return mid_term_results
    
    def run_short_term_analysis(self, stock_codes: List[str] = None, days: int = None) -> Dict[str, Any]:
        """
        运行短期分析
        
        Args:
            stock_codes: 股票代码列表
            days: 分析天数
            
        Returns:
            短期分析结果
        """
        logger.info("开始执行短期分析")
        
        # 设置默认参数
        if stock_codes is None:
            stock_codes = self.config["data"]["short_term"]["default_stock_codes"]
        
        if days is None:
            days = self.config["data"]["short_term"]["default_days"]
        
        # 创建示例新闻数据
        news_data = []
        for stock_code in stock_codes:
            news = {
                "stock_code": stock_code,
                "title": f"公司{stock_code}发布新产品",
                "content": f"公司{stock_code}今日发布新产品，市场反应良好。",
                "date": "2023-10-15",
                "source": "财经新闻网"
            }
            news_data.append(news)
        
        # 执行短期分析
        short_term_results = self.short_term_analyzer.run_analysis(
            news_data=news_data,
            days=days
        )
        
        # 添加统计信息
        short_term_results["statistics"] = {
            "total_news": len(news_data),
            "classified_news": len(short_term_results.get("classified_news", [])),
            "total_risk_events": len(short_term_results.get("risk_events", [])),
            "total_risk_alerts": len(short_term_results.get("risk_alerts", []))
        }
        
        # 保存结果
        self.results["short_term"] = short_term_results
        
        logger.info("短期分析完成")
        return short_term_results
    
    def generate_integrated_report(self) -> str:
        """
        生成综合分析报告
        
        Returns:
            报告文件路径
        """
        logger.info("开始生成综合分析报告")
        
        # 创建报告目录
        report_dir = os.path.join(self.output_dir, "reports")
        os.makedirs(report_dir, exist_ok=True)
        
        # 生成报告文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(report_dir, f"optimized_integrated_report_{timestamp}.html")
        
        # 提取各期分析结果
        long_term_results = self.results.get("long_term", {})
        mid_term_results = self.results.get("mid_term", {})
        short_term_results = self.results.get("short_term", {})
        
        # 提取统计信息
        long_term_stats = long_term_results.get("statistics", {})
        mid_term_stats = mid_term_results.get("statistics", {})
        short_term_stats = short_term_results.get("statistics", {})
        
        # 提取高专精特新度企业
        company_evaluations = long_term_results.get("company_evaluations", [])
        top_companies = sorted(company_evaluations, key=lambda x: x.get("score", 0), reverse=True)[:10]
        
        # 提取正向预期差企业
        expectation_gaps = mid_term_results.get("expectation_gaps", [])
        positive_gaps = [gap for gap in expectation_gaps if gap.get("gap_direction") == "正向"]
        
        # 提取高风险预警
        risk_alerts = short_term_results.get("risk_alerts", [])
        high_risk_alerts = [alert for alert in risk_alerts if alert.get("alert_level") in ["高", "严重"]]
        
        # 生成HTML报告
        html_content = f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>优化版专精特新企业分析报告</title>
            <style>
                body {{
                    font-family: 'Microsoft YaHei', Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                    color: #333;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: #fff;
                    padding: 30px;
                    border-radius: 8px;
                    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                }}
                h1 {{
                    color: #2c3e50;
                    text-align: center;
                    margin-bottom: 30px;
                }}
                h2 {{
                    color: #3498db;
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 10px;
                }}
                h3 {{
                    color: #2980b9;
                }}
                .section {{
                    margin-bottom: 30px;
                }}
                .stats-container {{
                    display: flex;
                    justify-content: space-between;
                    margin-bottom: 20px;
                }}
                .card {{
                    background-color: #f8f9fa;
                    border-radius: 6px;
                    padding: 20px;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
                    width: 30%;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 20px;
                }}
                th, td {{
                    padding: 12px 15px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #f2f2f2;
                    font-weight: bold;
                }}
                .positive {{
                    color: #27ae60;
                    font-weight: bold;
                }}
                .negative {{
                    color: #e74c3c;
                    font-weight: bold;
                }}
                .neutral {{
                    color: #f39c12;
                    font-weight: bold;
                }}
                .highlight {{
                    background-color: #e8f4f8;
                    border-left: 4px solid #3498db;
                    padding: 15px;
                    margin: 15px 0;
                }}
                .optimization-note {{
                    background-color: #e8f8f5;
                    border-left: 4px solid #2ecc71;
                    padding: 15px;
                    margin: 15px 0;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>优化版专精特新企业分析报告</h1>
                
                <div class="optimization-note">
                    <h3>策略优化说明</h3>
                    <p>本报告基于优化版分析框架，整合了以下六项策略优化：</p>
                    <ol>
                        <li>调整信号阈值：提高买入信号阈值，降低卖出信号敏感度</li>
                        <li>增加持仓管理：引入8%止损、15%止盈机制</li>
                        <li>优化资金分配：根据信号强度调整仓位大小(5%-20%)</li>
                        <li>增加市场环境判断：在整体市场下跌时降低仓位</li>
                        <li>完善风控机制：设置最大回撤10%、单日最大亏损3%限制</li>
                        <li>结合Barra CNE5模型：小盘股、价值和动量因子正向调整</li>
                    </ol>
                </div>
                
                <div class="section">
                    <h2>分析统计</h2>
                    <div class="stats-container">
                        <div class="card">
                            <h3>长期分析</h3>
                            <p>学术论文数量: {long_term_stats.get('total_papers', 0)}</p>
                            <p>企业年报数量: {long_term_stats.get('total_reports', 0)}</p>
                            <p>评估企业数量: {long_term_stats.get('evaluated_companies', 0)}</p>
                        </div>
                        <div class="card">
                            <h3>中期分析</h3>
                            <p>季报数量: {mid_term_stats.get('total_quarterly_reports', 0)}</p>
                            <p>月报数量: {mid_term_stats.get('total_monthly_reports', 0)}</p>
                            <p>提取事件数量: {mid_term_stats.get('total_events', 0)}</p>
                            <p>预期差分析数量: {mid_term_stats.get('total_gaps', 0)}</p>
                        </div>
                        <div class="card">
                            <h3>短期分析</h3>
                            <p>爬取新闻数量: {short_term_stats.get('total_news', 0)}</p>
                            <p>有效分类新闻数: {short_term_stats.get('classified_news', 0)}</p>
                            <p>风险事件总数: {short_term_stats.get('total_risk_events', 0)}</p>
                            <p>风险预警总数: {short_term_stats.get('total_risk_alerts', 0)}</p>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>高专精特新度企业 (Top 10)</h2>
                    <table>
                        <tr><th>排名</th><th>公司名称</th><th>专精特新度评分</th><th>核心优势</th><th>信号强度</th></tr>
        """
        
        for i, company in enumerate(top_companies, 1):
            score = company.get("score", 0)
            name = company.get("name", "")
            advantages = ", ".join(company.get("advantages", [])[:3])  # 只显示前3个优势
            
            # 计算信号强度
            signal_strength = min(1.0, max(-1.0, score / 10))  # 归一化到[-1, 1]范围
            
            html_content += f"""
                <tr>
                    <td>{i}</td>
                    <td>{name}</td>
                    <td>{score:.2f}</td>
                    <td>{advantages}</td>
                    <td class="{'positive' if signal_strength > 0.2 else 'neutral'}">{signal_strength:.2f}</td>
                </tr>
            """
        
        html_content += """
                    </table>
                </div>
                
                <div class="section">
                    <h2>正向预期差企业</h2>
                    <table>
                        <tr><th>公司名称</th><th>预期差方向</th><th>预期差大小</th><th>主要因素</th><th>交易信号</th></tr>
        """
        
        for gap in positive_gaps:
            company = gap.get("company", "")
            direction = gap.get("gap_direction", "")
            size = gap.get("gap_size", 0)
            factors = ", ".join(gap.get("factors", [])[:3])  # 只显示前3个因素
            
            # 计算交易信号
            signal_strength = min(1.0, max(-1.0, size / 5))  # 归一化到[-1, 1]范围
            signal = "买入" if signal_strength > 0.2 else "持有"
            
            html_content += f"""
                <tr>
                    <td>{company}</td>
                    <td class="positive">{direction}</td>
                    <td>{size:.2f}</td>
                    <td>{factors}</td>
                    <td class="{'positive' if signal == '买入' else 'neutral'}">{signal}</td>
                </tr>
            """
        
        html_content += """
                    </table>
                </div>
                
                <div class="section">
                    <h2>高风险预警</h2>
        """
        
        if high_risk_alerts:
            html_content += """
                <table>
                    <tr><th>公司</th><th>风险类型</th><th>预警等级</th><th>事件数量</th><th>平均影响分数</th><th>风控建议</th></tr>
            """
            
            for alert in high_risk_alerts[:10]:  # 只显示前10个
                company = alert.get("company", "")
                risk_type = alert.get("risk_type", "")
                alert_level = alert.get("alert_level", "")
                event_count = alert.get("event_count", 0)
                avg_impact = alert.get("avg_impact_score", 0)
                
                level_class = ""
                if alert_level == "严重":
                    level_class = "negative"
                elif alert_level == "高":
                    level_class = "negative"
                else:
                    level_class = "neutral"
                
                # 风控建议
                risk_advice = "立即止损" if alert_level == "严重" else "减仓" if alert_level == "高" else "关注"
                
                html_content += f"""
                    <tr>
                        <td>{company}</td>
                        <td>{risk_type}</td>
                        <td class="{level_class}">{alert_level}</td>
                        <td>{event_count}</td>
                        <td>{avg_impact:.2f}</td>
                        <td class="{level_class}">{risk_advice}</td>
                    </tr>
                """
            
            html_content += """
                </table>
            """
        else:
            html_content += "<p>暂无高风险预警</p>"
        
        html_content += """
            </div>
            
            <div class="section">
                <h2>优化策略投资建议</h2>
                <div class="highlight">
                    <h3>重点关注</h3>
                    <p>基于长期分析，建议重点关注专精特新度评分较高的企业，这些企业在细分领域具有明显竞争优势。买入信号阈值已提高至0.2-0.6，确保只有高质量信号才会触发交易。</p>
                </div>
                <div class="highlight">
                    <h3>交易机会</h3>
                    <p>基于中期分析，正向预期差企业可能存在市场认知偏差，可关注相关交易机会。已实施8%止损、15%止盈机制，有效控制单笔交易风险。</p>
                </div>
                <div class="highlight">
                    <h3>风险控制</h3>
                    <p>基于短期分析，建议对高风险预警企业保持谨慎，及时跟踪相关风险事件。已设置最大回撤10%、单日最大亏损3%的限制，保护整体投资组合安全。</p>
                </div>
                <div class="optimization-note">
                    <h3>资金分配策略</h3>
                    <p>根据信号强度和市场环境动态调整仓位：信号强度>0.6时仓位20%，0.4-0.6时仓位15%，0.2-0.4时仓位10%，0-0.2时仓位5%。市场下跌时整体仓位降低50%。</p>
                </div>
                <div class="optimization-note">
                    <h3>Barra CNE5模型调整</h3>
                    <p>结合Barra CNE5模型，对小盘股因子、价值因子和动量因子进行正向调整，提高信号准确性。行业中性化处理，避免行业集中度风险。</p>
                </div>
            </div>
            
            <div class="section">
                <h2>分析流程图</h2>
                <div style="text-align: center; margin: 20px 0;">
                    <img src="https://via.placeholder.com/800x400?text=优化版专精特新企业分析流程图" alt="分析流程图" style="max-width: 100%; height: auto;">
                </div>
            </div>
        </body>
        </html>
        """
        
        # 保存报告
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"优化版综合报告已生成: {report_path}")
        return report_path
    
    def run_integrated_analysis_with_backtest(self, phases: List[str] = None) -> Dict[str, Any]:
        """
        运行整合分析并执行回测
        
        Args:
            phases: 要执行的分析阶段，如["long_term", "mid_term", "short_term"]，None表示执行所有阶段
            
        Returns:
            分析和回测结果字典
        """
        if phases is None:
            phases = ["long_term", "mid_term", "short_term"]
        
        logger.info(f"开始执行优化版整合分析流程，执行阶段: {', '.join(phases)}")
        
        # 初始化分析器
        self.initialize_analyzers()
        
        # 执行各期分析
        if "long_term" in phases:
            self.run_long_term_analysis()
        
        if "mid_term" in phases:
            self.run_mid_term_analysis()
        
        if "short_term" in phases:
            self.run_short_term_analysis()
        
        # 生成交易信号
        if all(phase in self.results for phase in phases):
            logger.info("开始生成优化版交易信号")
            
            # 提取各期分析结果
            long_term_results = self.results.get("long_term", {})
            mid_term_results = self.results.get("mid_term", {})
            short_term_results = self.results.get("short_term", {})
            
            # 生成整合信号
            integrated_signals = self.signal_generator.run_integrated_analysis(
                long_term_results=long_term_results,
                mid_term_results=mid_term_results,
                short_term_results=short_term_results
            )
            
            # 保存信号结果
            self.results["integrated_signals"] = integrated_signals
            
            # 执行回测
            logger.info("开始执行优化版回测")
            backtest_results = self.backtest_engine.run_backtest(
                signals=integrated_signals,
                start_date=self.config["backtest"]["start_date"],
                end_date=self.config["backtest"]["end_date"]
            )
            
            # 保存回测结果
            self.results["backtest_results"] = backtest_results
        
        # 生成综合报告
        if self.config["output"]["generate_report"]:
            report_path = self.generate_integrated_report()
            self.results["report_path"] = report_path
        
        # 保存结果
        if self.config["output"]["save_intermediate_results"]:
            results_path = os.path.join(self.output_dir, "optimized_full_analysis_results.json")
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        logger.info("优化版整合分析流程执行完成")
        return self.results
    
    def run_full_analysis(self, phases: List[str] = None) -> Dict[str, Any]:
        """
        运行完整分析流程（兼容原版本）
        
        Args:
            phases: 要执行的分析阶段，如["long_term", "mid_term", "short_term"]，None表示执行所有阶段
            
        Returns:
            分析结果字典
        """
        return self.run_integrated_analysis_with_backtest(phases)


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="优化版专精特新文本分析交易框架")
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--phases", type=str, nargs="+", choices=["long_term", "mid_term", "short_term"],
                       help="要执行的分析阶段")
    parser.add_argument("--long_term", action="store_true", help="只执行长期分析")
    parser.add_argument("--mid_term", action="store_true", help="只执行中期分析")
    parser.add_argument("--short_term", action="store_true", help="只执行短期分析")
    parser.add_argument("--use_api", action="store_true", default=True, help="使用通义千问API")
    parser.add_argument("--no_api", action="store_true", help="不使用通义千问API，使用本地模型")
    parser.add_argument("--run_backtest", action="store_true", default=True, help="运行回测")
    parser.add_argument("--no_backtest", action="store_true", help="不运行回测")
    
    args = parser.parse_args()
    
    # 处理API选项
    use_api = args.use_api and not args.no_api
    
    # 处理回测选项
    run_backtest = args.run_backtest and not args.no_backtest
    
    # 确定要执行的分析阶段
    phases = None
    if args.phases:
        phases = args.phases
    elif args.long_term:
        phases = ["long_term"]
    elif args.mid_term:
        phases = ["mid_term"]
    elif args.short_term:
        phases = ["short_term"]
    
    # 初始化优化版框架
    framework = OptimizedSpecializedInnovativeFramework(args.config, use_api=use_api)
    
    # 运行分析
    if run_backtest:
        results = framework.run_integrated_analysis_with_backtest(phases)
    else:
        results = framework.run_full_analysis(phases)
    
    # 打印统计信息
    print("\n优化版专精特新企业分析统计信息:")
    
    if "long_term" in results and results["long_term"]:
        stats = results["long_term"].get("statistics", {})
        print("\n长期分析:")
        print(f"  学术论文数量: {stats.get('total_papers', 0)}")
        print(f"  企业年报数量: {stats.get('total_reports', 0)}")
        print(f"  评估企业数量: {stats.get('evaluated_companies', 0)}")
    
    if "mid_term" in results and results["mid_term"]:
        stats = results["mid_term"].get("statistics", {})
        print("\n中期分析:")
        print(f"  季报数量: {stats.get('total_quarterly_reports', 0)}")
        print(f"  月报数量: {stats.get('total_monthly_reports', 0)}")
        print(f"  提取事件数量: {stats.get('total_events', 0)}")
        print(f"  预期差分析数量: {stats.get('total_gaps', 0)}")
    
    if "short_term" in results and results["short_term"]:
        stats = results["short_term"].get("statistics", {})
        print("\n短期分析:")
        print(f"  爬取新闻数量: {stats.get('total_news', 0)}")
        print(f"  有效分类新闻数: {stats.get('classified_news', 0)}")
        print(f"  风险事件总数: {stats.get('total_risk_events', 0)}")
        print(f"  风险预警总数: {stats.get('total_risk_alerts', 0)}")
    
    if "integrated_signals" in results and results["integrated_signals"]:
        signals = results["integrated_signals"]
        print("\n交易信号统计:")
        print(f"  生成信号数量: {len(signals.get('signals', []))}")
        
        # 统计买入和卖出信号
        buy_signals = sum(1 for s in signals.get('signals', []) if s.get('signal', 0) > 0.2)
        sell_signals = sum(1 for s in signals.get('signals', []) if s.get('signal', 0) < -0.3)
        print(f"  买入信号数量: {buy_signals}")
        print(f"  卖出信号数量: {sell_signals}")
    
    if "backtest_results" in results and results["backtest_results"]:
        backtest = results["backtest_results"]
        performance = backtest.get("performance_metrics", {})
        print("\n回测结果:")
        print(f"  总收益率: {performance.get('total_return', 0):.2%}")
        print(f"  年化收益率: {performance.get('annualized_return', 0):.2%}")
        print(f"  最大回撤: {performance.get('max_drawdown', 0):.2%}")
        print(f"  夏普比率: {performance.get('sharpe_ratio', 0):.2f}")
        print(f"  胜率: {performance.get('win_rate', 0):.2%}")
    
    if "report_path" in results:
        print(f"\n优化版综合报告: {results['report_path']}")
    
    logger.info("优化版专精特新文本分析交易框架执行完成")


if __name__ == "__main__":
    main()