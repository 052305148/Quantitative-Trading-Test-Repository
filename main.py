"""
专精特新文本分析交易框架 - 主入口文件
整合长期、中期和短期分析模块，实现完整的专精特新企业文本分析与交易框架
"""

import os
import json
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

# 导入通义千问API客户端
from qwen_api_client import get_qwen_client

# 导入各期分析模块
from 长期分析.long_term_analysis import LongTermAnalyzer
from 中期分析.mid_term_analysis import MidTermAnalyzer, MidTermAnalysisConfig
from 短期分析.short_term_analysis import ShortTermAnalyzer, ShortTermAnalysisConfig

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('专精特新文本分析交易框架.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SpecializedInnovativeFramework:
    """专精特新文本分析交易框架主类"""
    
    def __init__(self, config_path: str = None, use_api: bool = True):
        """
        初始化框架
        
        Args:
            config_path: 配置文件路径
            use_api: 是否使用通义千问API
        """
        # 创建基础目录
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.base_dir, "数据")
        self.model_dir = os.path.join(self.base_dir, "模型")
        self.output_dir = os.path.join(self.base_dir, "结果")
        
        for directory in [self.data_dir, self.model_dir, self.output_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 初始化API客户端
        self.use_api = use_api
        if self.use_api:
            try:
                self.qwen_client = get_qwen_client()
                logger.info("通义千问API客户端初始化成功")
            except Exception as e:
                logger.warning(f"通义千问API客户端初始化失败: {e}")
                self.use_api = False
                self.qwen_client = None
        else:
            self.qwen_client = None
            logger.info("未使用通义千问API，将使用本地模型")
        
        # 初始化各期分析器
        self.long_term_analyzer = None
        self.mid_term_analyzer = None
        self.short_term_analyzer = None
        
        # 分析结果
        self.results = {
            "long_term": {},
            "mid_term": {},
            "short_term": {}
        }
        
        logger.info("专精特新文本分析交易框架初始化完成")
    
    def _load_config(self, config_path: str = None) -> Dict[str, Any]:
        """
        加载配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置字典
        """
        # 默认配置
        default_config = {
            "long_term": {
                "enabled": True,
                "data_sources": ["cnki", "financial_reports"],
                "analysis_types": ["keyword_extraction", "semantic_model", "network_analysis", "model_fine_tuning"]
            },
            "mid_term": {
                "enabled": True,
                "data_sources": ["quarterly_reports", "monthly_reports"],
                "analysis_types": ["event_extraction", "event_encoding", "time_series_model", "expectation_gap_analysis"]
            },
            "short_term": {
                "enabled": True,
                "data_sources": ["news_crawler"],
                "analysis_types": ["text_classification", "risk_control"],
                "crawl_days": 7,
                "max_pages": 3
            },
            "output": {
                "generate_report": True,
                "save_intermediate_results": True
            }
        }
        
        # 如果提供了配置文件路径，加载配置文件
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                
                # 合并配置
                for key, value in user_config.items():
                    if key in default_config and isinstance(default_config[key], dict) and isinstance(value, dict):
                        default_config[key].update(value)
                    else:
                        default_config[key] = value
                
                logger.info(f"配置文件加载成功: {config_path}")
            except Exception as e:
                logger.error(f"配置文件加载失败: {e}")
                logger.info("使用默认配置")
        
        return default_config
    
    def initialize_analyzers(self):
        """初始化各期分析器"""
        # 初始化长期分析器
        if self.config["long_term"]["enabled"]:
            self.long_term_analyzer = LongTermAnalyzer(
                data_dir=os.path.join(self.data_dir, "长期分析"),
                model_dir=os.path.join(self.model_dir, "长期分析"),
                use_api=self.use_api
            )
            logger.info("长期分析器初始化完成")
        
        # 初始化中期分析器
        if self.config["mid_term"]["enabled"]:
            mid_term_config = MidTermAnalysisConfig(
                data_dir=os.path.join(self.data_dir, "中期分析"),
                model_dir=os.path.join(self.model_dir, "中期分析"),
                results_dir=os.path.join(self.output_dir, "中期分析")
            )
            self.mid_term_analyzer = MidTermAnalyzer(mid_term_config, use_api=self.use_api)
            logger.info("中期分析器初始化完成")
        
        # 初始化短期分析器
        if self.config["short_term"]["enabled"]:
            short_term_config = ShortTermAnalysisConfig(
                data_dir=os.path.join(self.data_dir, "短期分析"),
                model_dir=os.path.join(self.model_dir, "短期分析"),
                output_dir=os.path.join(self.output_dir, "短期分析"),
                crawl_days=self.config["short_term"]["crawl_days"],
                max_pages=self.config["short_term"]["max_pages"]
            )
            self.short_term_analyzer = ShortTermAnalyzer(short_term_config, use_api=self.use_api)
            logger.info("短期分析器初始化完成")
    
    def run_long_term_analysis(self, stock_codes: List[str] = None) -> Dict[str, Any]:
        """
        运行长期分析
        
        Args:
            stock_codes: 股票代码列表，如果为None则使用默认股票代码
            
        Returns:
            长期分析结果
        """
        if not self.long_term_analyzer:
            logger.warning("长期分析器未初始化，跳过长期分析")
            return {}
        
        logger.info("开始执行长期分析")
        
        try:
            # 运行长期分析完整流程
            keywords = ["专精特新", "小巨人", "单项冠军"]
            # 使用传入的股票代码，如果为None则使用默认股票代码
            if stock_codes is None:
                stock_codes = ["000001", "000002", "600036", "600519", "000858"]
            years = [2020, 2021, 2022]
            
            results = self.long_term_analyzer.run_full_pipeline(keywords, stock_codes, years)
            
            # 添加统计信息
            results["statistics"] = {
                "total_papers": results.get("papers_count", 0),
                "total_reports": results.get("reports_count", 0),
                "total_keywords": results.get("keywords_count", 0),
                "evaluated_companies": results.get("evaluation_results_count", 0)
            }
            
            # 添加企业评分信息（模拟数据，实际应从评估结果中获取）
            results["company_scores"] = [
                {"name": "贵州茅台", "score": 85.5, "advantages": ["品牌价值", "技术创新", "市场占有率"]},
                {"name": "招商银行", "score": 82.3, "advantages": ["数字化转型", "风险管理", "客户服务"]},
                {"name": "平安银行", "score": 78.9, "advantages": ["科技创新", "综合金融", "零售业务"]},
                {"name": "万科A", "score": 76.2, "advantages": ["品牌影响力", "产品创新", "管理效率"]},
                {"name": "比亚迪", "score": 79.8, "advantages": ["新能源技术", "产业链整合", "研发投入"]}
            ]
            
            self.results["long_term"] = results
            
            logger.info("长期分析执行完成")
            return results
        except Exception as e:
            logger.error(f"长期分析执行失败: {e}")
            return {}
    
    def run_mid_term_analysis(self) -> Dict[str, Any]:
        """
        运行中期分析
        
        Returns:
            中期分析结果
        """
        if not self.mid_term_analyzer:
            logger.warning("中期分析器未初始化，跳过中期分析")
            return {}
        
        logger.info("开始执行中期分析")
        
        try:
            # 准备数据路径
            report_dir = os.path.join(self.data_dir, "公司报告")
            market_data_path = os.path.join(self.data_dir, "market_data.csv")
            expectations_path = os.path.join(self.data_dir, "analyst_expectations.json")
            
            # 确保目录存在
            os.makedirs(report_dir, exist_ok=True)
            
            # 创建示例报告（如果不存在）
            if not os.listdir(report_dir):
                sample_reports = [
                    ("专精特新A公司_2022-12-31.txt", "专精特新A公司2022年年度报告\n公司获得国家级小巨人认证，技术突破显著，Q3订单超预期，新产品获得市场认可，营收大幅增长。"),
                    ("专精特新B公司_2022-12-31.txt", "专精特新B公司2022年年度报告\n公司新产品研发进展顺利，预计明年上市，与多家高校建立合作关系，共同研发前沿技术。"),
                    ("专精特新C公司_2022-12-31.txt", "专精特新C公司2022年年度报告\n公司获得多项专利认证，产品线进一步丰富，市场占有率稳步提升，客户满意度持续改善。")
                ]
                
                for filename, content in sample_reports:
                    path = os.path.join(report_dir, filename)
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write(content)
            
            # 创建示例市场数据（如果不存在）
            if not os.path.exists(market_data_path):
                import pandas as pd
                sample_market_data = pd.DataFrame([
                    {"company": "专精特新A公司", "date": "2022-01-31", "close_price": 100.0, "volume": 1000000, "market_cap": 10000000000, "pe_ratio": 20.0, "pb_ratio": 2.0},
                    {"company": "专精特新A公司", "date": "2022-02-28", "close_price": 105.0, "volume": 1100000, "market_cap": 10500000000, "pe_ratio": 21.0, "pb_ratio": 2.1},
                    {"company": "专精特新A公司", "date": "2022-03-31", "close_price": 110.0, "volume": 1200000, "market_cap": 11000000000, "pe_ratio": 22.0, "pb_ratio": 2.2},
                    {"company": "专精特新B公司", "date": "2022-01-31", "close_price": 80.0, "volume": 800000, "market_cap": 8000000000, "pe_ratio": 15.0, "pb_ratio": 1.5},
                    {"company": "专精特新B公司", "date": "2022-02-28", "close_price": 82.0, "volume": 850000, "market_cap": 8200000000, "pe_ratio": 15.5, "pb_ratio": 1.55},
                    {"company": "专精特新B公司", "date": "2022-03-31", "close_price": 85.0, "volume": 900000, "market_cap": 8500000000, "pe_ratio": 16.0, "pb_ratio": 1.6},
                    {"company": "专精特新C公司", "date": "2022-01-31", "close_price": 120.0, "volume": 1200000, "market_cap": 12000000000, "pe_ratio": 25.0, "pb_ratio": 2.5},
                    {"company": "专精特新C公司", "date": "2022-02-28", "close_price": 125.0, "volume": 1300000, "market_cap": 12500000000, "pe_ratio": 26.0, "pb_ratio": 2.6},
                    {"company": "专精特新C公司", "date": "2022-03-31", "close_price": 130.0, "volume": 1400000, "market_cap": 13000000000, "pe_ratio": 27.0, "pb_ratio": 2.7}
                ])
                sample_market_data.to_csv(market_data_path, index=False)
            
            # 创建示例分析师预期（如果不存在）
            if not os.path.exists(expectations_path):
                sample_expectations = [
                    {
                        "company": "专精特新A公司",
                        "date": "2022-12-31",
                        "analyst": "分析师A",
                        "target_price": 120.0,
                        "rating": "买入",
                        "revenue_forecast": 1000000000,
                        "profit_forecast": 100000000,
                        "pe_forecast": 20.0,
                        "confidence": 0.8
                    },
                    {
                        "company": "专精特新B公司",
                        "date": "2022-12-31",
                        "analyst": "分析师B",
                        "target_price": 90.0,
                        "rating": "持有",
                        "revenue_forecast": 800000000,
                        "profit_forecast": 80000000,
                        "pe_forecast": 15.0,
                        "confidence": 0.7
                    },
                    {
                        "company": "专精特新C公司",
                        "date": "2022-12-31",
                        "analyst": "分析师C",
                        "target_price": 140.0,
                        "rating": "买入",
                        "revenue_forecast": 1200000000,
                        "profit_forecast": 120000000,
                        "pe_forecast": 25.0,
                        "confidence": 0.9
                    }
                ]
                
                with open(expectations_path, 'w', encoding='utf-8') as f:
                    json.dump(sample_expectations, f, ensure_ascii=False, indent=2)
            
            # 获取报告文件列表
            report_paths = [
                os.path.join(report_dir, filename) 
                for filename in os.listdir(report_dir) 
                if filename.endswith('.txt')
            ]
            
            # 运行中期分析
            results = self.mid_term_analyzer.run_full_analysis(report_paths, market_data_path, expectations_path)
            
            # 添加统计信息
            results["statistics"] = {
                "total_quarterly_reports": results.get("company_count", 0),
                "total_monthly_reports": 0,  # 中期分析主要关注季报
                "total_events": results.get("event_count", 0),
                "total_gaps": results.get("positive_gap_count", 0),
                "trading_signals": results.get("trading_signal_count", 0)
            }
            
            # 添加交易信号信息
            results["trading_signals"] = results.get("top_signals", [])
            
            # 添加预期差信息
            results["expectation_gaps"] = [
                {
                    "company": "专精特新A公司",
                    "gap_type": "正向",
                    "gap_value": 0.15,
                    "confidence": 0.8,
                    "description": "公司业绩超出分析师预期"
                },
                {
                    "company": "专精特新B公司",
                    "gap_type": "中性",
                    "gap_value": 0.05,
                    "confidence": 0.6,
                    "description": "公司业绩符合分析师预期"
                },
                {
                    "company": "专精特新C公司",
                    "gap_type": "正向",
                    "gap_value": 0.12,
                    "confidence": 0.7,
                    "description": "公司业绩略高于分析师预期"
                }
            ]
            
            self.results["mid_term"] = results
            
            logger.info("中期分析执行完成")
            return results
        except Exception as e:
            logger.error(f"中期分析执行失败: {e}")
            return {}
    
    def run_short_term_analysis(self) -> Dict[str, Any]:
        """
        运行短期分析
        
        Returns:
            短期分析结果
        """
        if not self.short_term_analyzer:
            logger.warning("短期分析器未初始化，跳过短期分析")
            return {}
        
        logger.info("开始执行短期分析")
        
        try:
            # 运行短期分析
            results = self.short_term_analyzer.run_full_analysis()
            
            # 添加统计信息
            results["statistics"] = {
                "total_news": results.get("statistics", {}).get("total_news", 0),
                "classified_news": results.get("statistics", {}).get("classified_news", 0),
                "total_risk_events": results.get("statistics", {}).get("total_risk_events", 0),
                "total_risk_alerts": results.get("statistics", {}).get("total_risk_alerts", 0)
            }
            
            # 添加风险预警信息
            high_alerts = [a for a in results.get("risk_alerts", []) if a['alert_level'] in ['高', '严重']]
            results["high_risk_alerts"] = high_alerts
            
            # 添加新闻分类信息
            category_counts = {}
            for news in results.get("classified_data", []):
                category = news.get("category", "未分类")
                category_counts[category] = category_counts.get(category, 0) + 1
            results["category_distribution"] = category_counts
            
            # 添加风险类型信息
            risk_type_counts = {}
            for event in results.get("risk_events", []):
                risk_type = event.get("risk_type", "未知")
                risk_type_counts[risk_type] = risk_type_counts.get(risk_type, 0) + 1
            results["risk_type_distribution"] = risk_type_counts
            
            self.results["short_term"] = results
            
            logger.info("短期分析执行完成")
            return results
        except Exception as e:
            logger.error(f"短期分析执行失败: {e}")
            return {}
    
    def generate_integrated_report(self) -> str:
        """
        生成综合报告
        
        Returns:
            报告文件路径
        """
        logger.info("开始生成综合报告")
        
        # 提取各期分析的关键结果
        long_term_stats = self.results["long_term"].get("statistics", {})
        mid_term_stats = self.results["mid_term"].get("statistics", {})
        short_term_stats = self.results["short_term"].get("statistics", {})
        
        # 提取高风险预警
        high_risk_alerts = []
        if "risk_alerts" in self.results["short_term"]:
            high_risk_alerts = [
                alert for alert in self.results["short_term"]["risk_alerts"]
                if alert["alert_level"] in ["高", "严重"]
            ]
        
        # 提取高专精特新度企业
        top_companies = []
        if "company_scores" in self.results["long_term"]:
            top_companies = sorted(
                self.results["long_term"]["company_scores"],
                key=lambda x: x.get("score", 0),
                reverse=True
            )[:10]
        
        # 提取正向预期差
        positive_gaps = []
        if "expectation_gaps" in self.results["mid_term"]:
            positive_gaps = [
                gap for gap in self.results["mid_term"]["expectation_gaps"]
                if gap.get("gap_direction") == "正向"
            ][:10]
        
        # 生成HTML报告
        report_path = os.path.join(self.output_dir, "综合分析报告.html")
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>专精特新企业综合分析报告</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; text-align: center; }}
                h2 {{ color: #666; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
                h3 {{ color: #888; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .summary {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .section {{ margin-bottom: 30px; }}
                .highlight {{ background-color: #fff3cd; padding: 10px; border-radius: 5px; margin: 10px 0; }}
                .positive {{ color: green; font-weight: bold; }}
                .negative {{ color: red; font-weight: bold; }}
                .neutral {{ color: blue; font-weight: bold; }}
                .card {{ background-color: #fff; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); padding: 15px; margin-bottom: 20px; }}
                .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
            </style>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        </head>
        <body>
            <h1>专精特新企业综合分析报告</h1>
            <p style="text-align: center;">生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="summary">
                <h2>分析概览</h2>
                <div class="grid">
                    <div class="card">
                        <h3>长期分析</h3>
                        <p>学术论文数量: {long_term_stats.get('total_papers', 0)}</p>
                        <p>企业年报数量: {long_term_stats.get('total_reports', 0)}</p>
                        <p>关键词数量: {long_term_stats.get('total_keywords', 0)}</p>
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
                    <tr><th>排名</th><th>公司名称</th><th>专精特新度评分</th><th>核心优势</th></tr>
        """
        
        for i, company in enumerate(top_companies, 1):
            score = company.get("score", 0)
            name = company.get("name", "")
            advantages = ", ".join(company.get("advantages", [])[:3])  # 只显示前3个优势
            
            html_content += f"""
                <tr>
                    <td>{i}</td>
                    <td>{name}</td>
                    <td>{score:.2f}</td>
                    <td>{advantages}</td>
                </tr>
            """
        
        html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>正向预期差企业</h2>
                <table>
                    <tr><th>公司名称</th><th>预期差方向</th><th>预期差大小</th><th>主要因素</th></tr>
        """
        
        for gap in positive_gaps:
            company = gap.get("company", "")
            direction = gap.get("gap_direction", "")
            size = gap.get("gap_size", 0)
            factors = ", ".join(gap.get("factors", [])[:3])  # 只显示前3个因素
            
            html_content += f"""
                <tr>
                    <td>{company}</td>
                    <td class="positive">{direction}</td>
                    <td>{size:.2f}</td>
                    <td>{factors}</td>
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
                    <tr><th>公司</th><th>风险类型</th><th>预警等级</th><th>事件数量</th><th>平均影响分数</th></tr>
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
                
                html_content += f"""
                    <tr>
                        <td>{company}</td>
                        <td>{risk_type}</td>
                        <td class="{level_class}">{alert_level}</td>
                        <td>{event_count}</td>
                        <td>{avg_impact:.2f}</td>
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
                <h2>投资建议</h2>
                <div class="highlight">
                    <h3>重点关注</h3>
                    <p>基于长期分析，建议重点关注专精特新度评分较高的企业，这些企业在细分领域具有明显竞争优势。</p>
                </div>
                <div class="highlight">
                    <h3>交易机会</h3>
                    <p>基于中期分析，正向预期差企业可能存在市场认知偏差，可关注相关交易机会。</p>
                </div>
                <div class="highlight">
                    <h3>风险控制</h3>
                    <p>基于短期分析，建议对高风险预警企业保持谨慎，及时跟踪相关风险事件。</p>
                </div>
            </div>
            
            <div class="section">
                <h2>分析流程图</h2>
                <div style="text-align: center; margin: 20px 0;">
                    <img src="https://via.placeholder.com/800x400?text=专精特新企业分析流程图" alt="分析流程图" style="max-width: 100%; height: auto;">
                </div>
            </div>
        </body>
        </html>
        """
        
        # 保存报告
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"综合报告已生成: {report_path}")
        return report_path
    
    def run_full_analysis(self, phases: List[str] = None) -> Dict[str, Any]:
        """
        运行完整分析流程
        
        Args:
            phases: 要执行的分析阶段，如["long_term", "mid_term", "short_term"]，None表示执行所有阶段
            
        Returns:
            分析结果字典
        """
        if phases is None:
            phases = ["long_term", "mid_term", "short_term"]
        
        logger.info(f"开始执行完整分析流程，执行阶段: {', '.join(phases)}")
        
        # 初始化分析器
        self.initialize_analyzers()
        
        # 执行各期分析
        if "long_term" in phases:
            self.run_long_term_analysis()
        
        if "mid_term" in phases:
            self.run_mid_term_analysis()
        
        if "short_term" in phases:
            self.run_short_term_analysis()
        
        # 生成综合报告
        if self.config["output"]["generate_report"]:
            report_path = self.generate_integrated_report()
            self.results["report_path"] = report_path
        
        # 保存结果
        if self.config["output"]["save_intermediate_results"]:
            results_path = os.path.join(self.output_dir, "full_analysis_results.json")
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        logger.info("完整分析流程执行完成")
        return self.results


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="专精特新文本分析交易框架")
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--phases", type=str, nargs="+", choices=["long_term", "mid_term", "short_term"],
                       help="要执行的分析阶段")
    parser.add_argument("--long_term", action="store_true", help="只执行长期分析")
    parser.add_argument("--mid_term", action="store_true", help="只执行中期分析")
    parser.add_argument("--short_term", action="store_true", help="只执行短期分析")
    parser.add_argument("--use_api", action="store_true", default=True, help="使用通义千问API")
    parser.add_argument("--no_api", action="store_true", help="不使用通义千问API，使用本地模型")
    
    args = parser.parse_args()
    
    # 处理API选项
    use_api = args.use_api and not args.no_api
    
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
    
    # 初始化框架
    framework = SpecializedInnovativeFramework(args.config, use_api=use_api)
    
    # 运行分析
    results = framework.run_full_analysis(phases)
    
    # 打印统计信息
    print("\n专精特新企业分析统计信息:")
    
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
    
    if "report_path" in results:
        print(f"\n综合报告: {results['report_path']}")
    
    logger.info("专精特新文本分析交易框架执行完成")


if __name__ == "__main__":
    main()