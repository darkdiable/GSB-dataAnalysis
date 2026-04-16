"""
PDF报告生成模块
生成完整的电商销售分析PDF报告
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
import os
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import (SimpleDocTemplate, Table, TableStyle, Paragraph,
                                Spacer, Image, PageBreak, ListFlowable, ListItem)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PDFReportGenerator:
    """PDF报告生成器"""

    def __init__(self, output_path: str = 'output/sales_analysis_report.pdf'):
        self.output_path = output_path
        self.styles = getSampleStyleSheet()
        self.setup_styles()
        self.story = []

    def setup_styles(self):
        """设置样式"""
        # 标题样式
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f4e79'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )

        # 章节标题样式
        self.heading_style = ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#2e75b5'),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        )

        # 子标题样式
        self.subheading_style = ParagraphStyle(
            'CustomSubHeading',
            parent=self.styles['Heading3'],
            fontSize=13,
            textColor=colors.HexColor('#5b9bd5'),
            spaceAfter=10,
            spaceBefore=10,
            fontName='Helvetica-Bold'
        )

        # 正文样式
        self.body_style = ParagraphStyle(
            'CustomBody',
            parent=self.styles['Normal'],
            fontSize=10,
            leading=14,
            alignment=TA_JUSTIFY,
            spaceAfter=10
        )

        # 表格标题样式
        self.table_header_style = ParagraphStyle(
            'TableHeader',
            parent=self.styles['Normal'],
            fontSize=9,
            textColor=colors.white,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )

        # 表格内容样式
        self.table_body_style = ParagraphStyle(
            'TableBody',
            parent=self.styles['Normal'],
            fontSize=8,
            alignment=TA_CENTER
        )

    def add_title(self, title: str, subtitle: str = ""):
        """添加标题页"""
        self.story.append(Spacer(1, 2*inch))
        self.story.append(Paragraph(title, self.title_style))

        if subtitle:
            subtitle_style = ParagraphStyle(
                'Subtitle',
                parent=self.styles['Normal'],
                fontSize=14,
                textColor=colors.grey,
                alignment=TA_CENTER,
                spaceAfter=20
            )
            self.story.append(Paragraph(subtitle, subtitle_style))

        # 添加生成日期
        date_style = ParagraphStyle(
            'Date',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.grey,
            alignment=TA_CENTER
        )
        self.story.append(Spacer(1, 1*inch))
        self.story.append(Paragraph(f"生成日期: {datetime.now().strftime('%Y-%m-%d %H:%M')}", date_style))
        self.story.append(PageBreak())

    def add_heading(self, text: str):
        """添加章节标题"""
        self.story.append(Paragraph(text, self.heading_style))
        self.story.append(Spacer(1, 0.1*inch))

    def add_subheading(self, text: str):
        """添加子标题"""
        self.story.append(Paragraph(text, self.subheading_style))
        self.story.append(Spacer(1, 0.05*inch))

    def add_paragraph(self, text: str):
        """添加段落"""
        self.story.append(Paragraph(text, self.body_style))
        self.story.append(Spacer(1, 0.1*inch))

    def add_table(self, df: pd.DataFrame, title: str = "", max_rows: int = 20):
        """添加表格"""
        try:
            if title:
                self.add_subheading(title)

            # 限制行数
            display_df = df.head(max_rows)

            # 准备数据
            headers = list(display_df.columns)
            data = [headers]

            for _, row in display_df.iterrows():
                formatted_row = []
                for val in row:
                    if isinstance(val, float):
                        formatted_row.append(f"{val:.2f}")
                    else:
                        formatted_row.append(str(val))
                data.append(formatted_row)

            # 创建表格
            col_widths = [2*cm] * len(headers)
            table = Table(data, colWidths=col_widths, repeatRows=1)

            # 设置表格样式
            table_style = TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2e75b5')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')])
            ])
            table.setStyle(table_style)

            self.story.append(table)
            self.story.append(Spacer(1, 0.2*inch))

        except Exception as e:
            logger.error(f"添加表格出错: {e}")

    def add_image(self, image_path: str, title: str = "", width: float = 6*inch):
        """添加图片"""
        try:
            if os.path.exists(image_path):
                if title:
                    self.add_subheading(title)

                img = Image(image_path, width=width, height=width*0.6)
                self.story.append(img)
                self.story.append(Spacer(1, 0.2*inch))
            else:
                logger.warning(f"图片不存在: {image_path}")
        except Exception as e:
            logger.error(f"添加图片出错: {e}")

    def add_kpi_summary(self, kpis: Dict):
        """添加KPI摘要"""
        self.add_heading("一、关键指标概览")

        kpi_data = [
            ['指标', '数值'],
            ['总销售额', f"¥{kpis.get('total_revenue', 0):,.2f}"],
            ['总订单数', f"{kpis.get('total_transactions', 0):,}"],
            ['总销量', f"{kpis.get('total_quantity', 0):,}"],
            ['平均订单金额', f"¥{kpis.get('avg_order_value', 0):.2f}"],
            ['独立客户数', f"{kpis.get('unique_customers', 0):,}"],
            ['独立商品数', f"{kpis.get('unique_products', 0):,}"],
            ['客户平均价值', f"¥{kpis.get('avg_customer_value', 0):.2f}"],
        ]

        if 'daily_avg_revenue' in kpis:
            kpi_data.append(['日均销售额', f"¥{kpis.get('daily_avg_revenue', 0):.2f}"])

        table = Table(kpi_data, colWidths=[3*inch, 3*inch])
        table_style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2e75b5')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')])
        ])
        table.setStyle(table_style)

        self.story.append(table)
        self.story.append(Spacer(1, 0.3*inch))

    def add_sales_trend_section(self, trend_results: Dict, image_paths: Dict):
        """添加销售趋势分析章节"""
        self.add_heading("二、销售趋势分析")

        # 月度趋势数据
        if 'monthly_trend' in trend_results:
            self.add_subheading("月度销售趋势")
            monthly = trend_results['monthly_trend'].copy()
            monthly['revenue'] = monthly['revenue'].apply(lambda x: f"¥{x:,.2f}")
            monthly['revenue_growth'] = monthly['revenue_growth'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "-")
            self.add_table(monthly[['year_month', 'revenue', 'transactions', 'revenue_growth']], max_rows=15)

        # 添加趋势图
        if 'sales_trend' in image_paths:
            self.add_image(image_paths['sales_trend'], "销售趋势图")

        # 星期分析
        if 'weekday_analysis' in trend_results:
            self.add_subheading("星期销售分析")
            self.add_table(trend_results['weekday_analysis'][['weekday_name', 'total_revenue', 'transactions']])

        if 'weekday_analysis' in image_paths:
            self.add_image(image_paths['weekday_analysis'], "星期分析图")

        self.story.append(PageBreak())

    def add_customer_analysis_section(self, customer_results: Dict, image_paths: Dict):
        """添加客户分析章节"""
        self.add_heading("三、客户价值分析")

        # RFM分析
        if 'rfm' in customer_results:
            self.add_subheading("RFM分析")
            rfm_summary = customer_results['rfm'].groupby('customer_segment').agg({
                'customer_id': 'count',
                'recency': 'mean',
                'frequency': 'mean',
                'monetary': 'mean'
            }).reset_index()
            rfm_summary.columns = ['客户分层', '客户数量', '平均最近购买天数', '平均购买频次', '平均消费金额']
            self.add_table(rfm_summary)

        # 客户分群图
        if 'customer_segmentation' in image_paths:
            self.add_image(image_paths['customer_segmentation'], "客户分群分析")

        if 'rfm_distribution' in image_paths:
            self.add_image(image_paths['rfm_distribution'], "RFM分布")

        # 流失分析
        if 'churn_stats' in customer_results:
            self.add_subheading("客户流失分析")
            churn = customer_results['churn_stats']
            self.add_paragraph(f"总客户数: {churn['total_customers']:,}")
            self.add_paragraph(f"流失客户数: {churn['churned_customers']:,}")
            self.add_paragraph(f"流失率: {churn['churn_rate']:.2f}%")

        self.story.append(PageBreak())

    def add_product_analysis_section(self, product_results: Dict, image_paths: Dict):
        """添加商品分析章节"""
        self.add_heading("四、商品关联分析")

        # 品类表现
        if 'category_performance' in product_results:
            self.add_subheading("品类表现")
            cat_perf = product_results['category_performance'].copy()
            cat_perf['total_revenue'] = cat_perf['total_revenue'].apply(lambda x: f"¥{x:,.2f}")
            cat_perf['revenue_share'] = cat_perf['revenue_share'].apply(lambda x: f"{x:.2f}%")
            self.add_table(cat_perf[['category', 'total_revenue', 'revenue_share', 'transactions', 'unique_customers']])

        if 'category_analysis' in image_paths:
            self.add_image(image_paths['category_analysis'], "品类分析")

        # 热销商品
        if 'top_products' in product_results:
            self.add_subheading("热销商品 Top 10")
            top_prod = product_results['top_products'].copy()
            top_prod['revenue'] = top_prod['revenue'].apply(lambda x: f"¥{x:,.2f}")
            col_name = 'product_name' if 'product_name' in top_prod.columns else 'product_id'
            self.add_table(top_prod[[col_name, 'revenue', 'quantity', 'transactions']])

        if 'top_products' in image_paths:
            self.add_image(image_paths['top_products'], "热销商品")

        self.story.append(PageBreak())

    def add_prediction_section(self, prediction_results: Dict, image_paths: Dict):
        """添加预测分析章节"""
        self.add_heading("五、销售预测分析")

        # 模型对比
        if 'metrics' in prediction_results and not prediction_results['metrics'].empty:
            self.add_subheading("模型性能对比")
            metrics = prediction_results['metrics'].copy()
            for col in ['mae', 'rmse', 'r2', 'mape']:
                if col in metrics.columns:
                    metrics[col] = metrics[col].apply(lambda x: f"{x:.4f}")
            self.add_table(metrics)

        if 'model_comparison' in image_paths:
            self.add_image(image_paths['model_comparison'], "模型对比")

        # 预测结果
        if 'forecast' in prediction_results and not prediction_results['forecast'].empty:
            self.add_subheading("未来30天销售预测")
            forecast = prediction_results['forecast'].copy()
            forecast['predicted_sales'] = forecast['predicted_sales'].apply(lambda x: f"¥{x:,.2f}")
            forecast['date'] = forecast['date'].apply(lambda x: x.strftime('%Y-%m-%d') if hasattr(x, 'strftime') else str(x)[:10])
            self.add_table(forecast[['date', 'predicted_sales']].head(15))

        if 'prediction_results' in image_paths:
            self.add_image(image_paths['prediction_results'], "预测结果")

        # 特征重要性
        if 'feature_importance' in prediction_results and not prediction_results['feature_importance'].empty:
            self.add_subheading("特征重要性")
            feat_imp = prediction_results['feature_importance'].head(10).copy()
            feat_imp['importance'] = feat_imp['importance'].apply(lambda x: f"{x:.4f}")
            self.add_table(feat_imp)

        if 'feature_importance' in image_paths:
            self.add_image(image_paths['feature_importance'], "特征重要性")

        self.story.append(PageBreak())

    def add_conclusion(self):
        """添加结论"""
        self.add_heading("六、分析结论与建议")

        conclusions = [
            "1. 销售趋势：通过分析可以看出销售的季节性波动规律，建议在促销旺季前做好库存准备。",
            "2. 客户价值：重要价值客户是核心资产，应重点维护；对于流失风险客户，建议采取挽回措施。",
            "3. 商品策略：热销商品应保持充足库存，同时关注关联商品，制定交叉销售策略。",
            "4. 预测结果：基于历史数据的预测模型可以帮助制定未来的销售和库存计划。",
            "5. 持续优化：建议定期更新数据并重新训练模型，以提高预测准确性。"
        ]

        for conclusion in conclusions:
            self.add_paragraph(conclusion)

    def generate_report(self, data: Dict):
        """
        生成完整报告

        Args:
            data: 包含所有分析结果的字典
        """
        try:
            logger.info("开始生成PDF报告...")

            # 创建文档
            doc = SimpleDocTemplate(
                self.output_path,
                pagesize=A4,
                rightMargin=2*cm,
                leftMargin=2*cm,
                topMargin=2*cm,
                bottomMargin=2*cm
            )

            # 添加标题页
            self.add_title(
                "电商销售数据分析报告",
                "基于多维度数据挖掘与机器学习预测"
            )

            # 添加KPI摘要
            if 'kpis' in data:
                self.add_kpi_summary(data['kpis'])

            # 添加销售趋势分析
            if 'sales_trend' in data:
                self.add_sales_trend_section(
                    data['sales_trend'],
                    data.get('images', {})
                )

            # 添加客户分析
            if 'customer_value' in data:
                self.add_customer_analysis_section(
                    data['customer_value'],
                    data.get('images', {})
                )

            # 添加商品分析
            if 'product_association' in data:
                self.add_product_analysis_section(
                    data['product_association'],
                    data.get('images', {})
                )

            # 添加预测分析
            if 'prediction' in data:
                self.add_prediction_section(
                    data['prediction'],
                    data.get('images', {})
                )

            # 添加结论
            self.add_conclusion()

            # 生成PDF
            doc.build(self.story)
            logger.info(f"PDF报告已生成: {self.output_path}")
            return self.output_path

        except Exception as e:
            logger.error(f"生成PDF报告出错: {e}")
            raise


def generate_pdf_report(data: Dict, output_path: str = 'output/sales_analysis_report.pdf') -> str:
    """
    便捷函数：生成PDF报告

    Args:
        data: 分析结果数据
        output_path: 输出路径

    Returns:
        生成的PDF文件路径
    """
    generator = PDFReportGenerator(output_path)
    return generator.generate_report(data)
