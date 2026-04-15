import os
from datetime import datetime
from typing import Dict, List, Optional
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import pandas as pd


class PDFReportGenerator:
    def __init__(self, output_dir: str = 'output'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.styles = getSampleStyleSheet()
        self._setup_styles()
        
    def _setup_styles(self):
        try:
            font_paths = [
                '/System/Library/Fonts/PingFang.ttc',
                '/System/Library/Fonts/STHeiti Light.ttc',
                '/Library/Fonts/Arial Unicode.ttf',
                '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
                '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'
            ]
            
            font_registered = False
            for font_path in font_paths:
                if os.path.exists(font_path):
                    try:
                        pdfmetrics.registerFont(TTFont('ChineseFont', font_path))
                        font_registered = True
                        break
                    except:
                        continue
            
            if font_registered:
                self.styles.add(ParagraphStyle(
                    name='ChineseTitle',
                    fontName='ChineseFont',
                    fontSize=24,
                    alignment=TA_CENTER,
                    spaceAfter=30,
                    textColor=colors.darkblue
                ))
                
                self.styles.add(ParagraphStyle(
                    name='ChineseHeading',
                    fontName='ChineseFont',
                    fontSize=16,
                    alignment=TA_LEFT,
                    spaceBefore=20,
                    spaceAfter=10,
                    textColor=colors.darkblue
                ))
                
                self.styles.add(ParagraphStyle(
                    name='ChineseBody',
                    fontName='ChineseFont',
                    fontSize=10,
                    alignment=TA_JUSTIFY,
                    spaceBefore=6,
                    spaceAfter=6,
                    leading=14
                ))
            else:
                self.styles.add(ParagraphStyle(
                    name='ChineseTitle',
                    fontName='Helvetica-Bold',
                    fontSize=24,
                    alignment=TA_CENTER,
                    spaceAfter=30,
                    textColor=colors.darkblue
                ))
                
                self.styles.add(ParagraphStyle(
                    name='ChineseHeading',
                    fontName='Helvetica-Bold',
                    fontSize=16,
                    alignment=TA_LEFT,
                    spaceBefore=20,
                    spaceAfter=10,
                    textColor=colors.darkblue
                ))
                
                self.styles.add(ParagraphStyle(
                    name='ChineseBody',
                    fontName='Helvetica',
                    fontSize=10,
                    alignment=TA_JUSTIFY,
                    spaceBefore=6,
                    spaceAfter=6,
                    leading=14
                ))
        except Exception as e:
            print(f"字体设置警告: {str(e)}")
    
    def create_report(self, analysis_results: Dict, figures: Dict, 
                     filename: str = 'sales_analysis_report.pdf') -> str:
        filepath = os.path.join(self.output_dir, filename)
        
        doc = SimpleDocTemplate(
            filepath,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        story = []
        
        story.append(Paragraph("电商销售分析报告", self.styles['ChineseTitle']))
        story.append(Paragraph(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                              self.styles['ChineseBody']))
        story.append(Spacer(1, 20))
        
        story = self._add_executive_summary(story, analysis_results)
        story.append(PageBreak())
        
        story = self._add_sales_trend_section(story, analysis_results, figures)
        story.append(PageBreak())
        
        story = self._add_customer_analysis_section(story, analysis_results, figures)
        story.append(PageBreak())
        
        story = self._add_product_analysis_section(story, analysis_results, figures)
        story.append(PageBreak())
        
        story = self._add_prediction_section(story, analysis_results, figures)
        story.append(PageBreak())
        
        story = self._add_recommendations_section(story, analysis_results)
        
        doc.build(story)
        
        print(f"PDF报告已生成: {filepath}")
        return filepath
    
    def _add_executive_summary(self, story: list, analysis_results: Dict) -> list:
        story.append(Paragraph("一、执行摘要", self.styles['ChineseHeading']))
        
        sales_summary = analysis_results.get('sales_summary', {})
        
        summary_text = f"""
        本报告基于电商销售数据进行全面分析，涵盖销售趋势、客户价值、商品关联等多个维度。
        
        关键指标概览:
        - 总销售额: ¥{sales_summary.get('total_revenue', 0):,.2f}
        - 总订单数: {sales_summary.get('total_orders', 0):,}
        - 总客户数: {sales_summary.get('total_customers', 0):,}
        - 平均订单价值: ¥{sales_summary.get('avg_order_value', 0):,.2f}
        """
        
        story.append(Paragraph(summary_text, self.styles['ChineseBody']))
        story.append(Spacer(1, 10))
        
        growth = sales_summary.get('growth_metrics', {})
        if growth:
            growth_text = f"""
            增长指标:
            - 月度收入增长率: {growth.get('monthly_revenue_growth', 0):.2f}%
            - 月度订单增长率: {growth.get('monthly_order_growth', 0):.2f}%
            - 周度收入增长率: {growth.get('weekly_revenue_growth', 0):.2f}%
            """
            story.append(Paragraph(growth_text, self.styles['ChineseBody']))
        
        return story
    
    def _add_sales_trend_section(self, story: list, analysis_results: Dict, 
                                figures: Dict) -> list:
        story.append(Paragraph("二、销售趋势分析", self.styles['ChineseHeading']))
        
        if 'time_series' in figures and os.path.exists(figures['time_series']):
            story.append(Paragraph("2.1 销售趋势时间序列", self.styles['ChineseBody']))
            img = Image(figures['time_series'], width=6*inch, height=3.5*inch)
            story.append(img)
            story.append(Spacer(1, 10))
        
        if 'sales_heatmap_2023' in figures and os.path.exists(figures['sales_heatmap_2023']):
            story.append(Paragraph("2.2 销售热力图", self.styles['ChineseBody']))
            img = Image(figures['sales_heatmap_2023'], width=6*inch, height=3*inch)
            story.append(img)
            story.append(Spacer(1, 10))
        
        if 'weekly_pattern' in figures and os.path.exists(figures['weekly_pattern']):
            story.append(Paragraph("2.3 周销售模式", self.styles['ChineseBody']))
            img = Image(figures['weekly_pattern'], width=5*inch, height=3*inch)
            story.append(img)
            story.append(Spacer(1, 10))
        
        sales_summary = analysis_results.get('sales_summary', {})
        peak = sales_summary.get('peak_periods', {})
        if peak:
            peak_text = f"""
            销售高峰分析:
            - 最高销售日: {peak.get('peak_day', {}).get('date', 'N/A')}, 
              销售额: ¥{peak.get('peak_day', {}).get('revenue', 0):,.2f}
            - 最高销售月: {peak.get('peak_month', {}).get('month', 'N/A')}, 
              销售额: ¥{peak.get('peak_month', {}).get('revenue', 0):,.2f}
            - 最佳销售日: {peak.get('peak_weekday', {}).get('day', 'N/A')}, 
              平均销售额: ¥{peak.get('peak_weekday', {}).get('avg_revenue', 0):,.2f}
            """
            story.append(Paragraph(peak_text, self.styles['ChineseBody']))
        
        return story
    
    def _add_customer_analysis_section(self, story: list, analysis_results: Dict, 
                                       figures: Dict) -> list:
        story.append(Paragraph("三、客户价值分析", self.styles['ChineseHeading']))
        
        if 'customer_scatter' in figures and os.path.exists(figures['customer_scatter']):
            story.append(Paragraph("3.1 客户RFM分群散点图", self.styles['ChineseBody']))
            img = Image(figures['customer_scatter'], width=6*inch, height=3*inch)
            story.append(img)
            story.append(Spacer(1, 10))
        
        if 'customer_segment' in figures and os.path.exists(figures['customer_segment']):
            story.append(Paragraph("3.2 客户分群分布", self.styles['ChineseBody']))
            img = Image(figures['customer_segment'], width=6*inch, height=3*inch)
            story.append(img)
            story.append(Spacer(1, 10))
        
        segment_summary = analysis_results.get('customer_analysis', {}).get('segment_summary')
        if segment_summary is not None and len(segment_summary) > 0:
            story.append(Paragraph("3.3 客户分群详情", self.styles['ChineseBody']))
            
            table_data = [['客户分群', '客户数量', '客户占比%', '收入占比%']]
            for _, row in segment_summary.iterrows():
                table_data.append([
                    str(row['segment']),
                    str(int(row['customer_count'])),
                    f"{row['customer_share']:.1f}%",
                    f"{row['revenue_share']:.1f}%"
                ])
            
            table = Table(table_data, colWidths=[2*inch, 1.2*inch, 1.2*inch, 1.2*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(table)
        
        return story
    
    def _add_product_analysis_section(self, story: list, analysis_results: Dict, 
                                      figures: Dict) -> list:
        story.append(Paragraph("四、商品关联分析", self.styles['ChineseHeading']))
        
        if 'category_distribution' in figures and os.path.exists(figures['category_distribution']):
            story.append(Paragraph("4.1 品类销售分布", self.styles['ChineseBody']))
            img = Image(figures['category_distribution'], width=6*inch, height=3*inch)
            story.append(img)
            story.append(Spacer(1, 10))
        
        if 'top_products' in figures and os.path.exists(figures['top_products']):
            story.append(Paragraph("4.2 热销商品Top 10", self.styles['ChineseBody']))
            img = Image(figures['top_products'], width=5*inch, height=4*inch)
            story.append(img)
            story.append(Spacer(1, 10))
        
        if 'product_association' in figures and os.path.exists(figures['product_association']):
            story.append(Paragraph("4.3 品类关联度热力图", self.styles['ChineseBody']))
            img = Image(figures['product_association'], width=5*inch, height=4*inch)
            story.append(img)
            story.append(Spacer(1, 10))
        
        cross_sell = analysis_results.get('product_analysis', {}).get('cross_sell_opportunities')
        if cross_sell is not None and len(cross_sell) > 0:
            story.append(Paragraph("4.4 捆绑销售机会", self.styles['ChineseBody']))
            
            table_data = [['商品A', '商品B', 'Lift值', '置信度']]
            for _, row in cross_sell.head(5).iterrows():
                table_data.append([
                    str(row.get('product_a_name', 'N/A'))[:15],
                    str(row.get('product_b_name', 'N/A'))[:15],
                    f"{row['lift']:.2f}",
                    f"{row['confidence_a_to_b']:.2%}"
                ])
            
            table = Table(table_data, colWidths=[1.5*inch, 1.5*inch, 1*inch, 1*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(table)
        
        return story
    
    def _add_prediction_section(self, story: list, analysis_results: Dict, 
                               figures: Dict) -> list:
        story.append(Paragraph("五、销售预测分析", self.styles['ChineseHeading']))
        
        model_metrics = analysis_results.get('prediction_results', {}).get('metrics', {})
        if model_metrics:
            test_metrics = model_metrics.get('test', {})
            metrics_text = f"""
            预测模型性能指标:
            - R² 分数: {test_metrics.get('r2', 0):.4f}
            - 均方根误差 (RMSE): {test_metrics.get('rmse', 0):,.2f}
            - 平均绝对误差 (MAE): {test_metrics.get('mae', 0):,.2f}
            """
            story.append(Paragraph(metrics_text, self.styles['ChineseBody']))
            story.append(Spacer(1, 10))
        
        if 'prediction_comparison' in figures and os.path.exists(figures['prediction_comparison']):
            story.append(Paragraph("5.1 销售预测对比图", self.styles['ChineseBody']))
            img = Image(figures['prediction_comparison'], width=6*inch, height=3*inch)
            story.append(img)
            story.append(Spacer(1, 10))
        
        pred_summary = analysis_results.get('prediction_results', {}).get('summary', {})
        if pred_summary:
            pred_text = f"""
            未来30天预测概览:
            - 预测总销售额: ¥{pred_summary.get('total_predicted_revenue', 0):,.2f}
            - 预测日均销售额: ¥{pred_summary.get('avg_predicted_revenue', 0):,.2f}
            - 预测最高销售额: ¥{pred_summary.get('max_predicted_revenue', 0):,.2f}
            - 预测最低销售额: ¥{pred_summary.get('min_predicted_revenue', 0):,.2f}
            - 预测期间: {pred_summary.get('prediction_period', 'N/A')}
            """
            story.append(Paragraph(pred_text, self.styles['ChineseBody']))
        
        return story
    
    def _add_recommendations_section(self, story: list, analysis_results: Dict) -> list:
        story.append(Paragraph("六、业务建议", self.styles['ChineseHeading']))
        
        recommendations = []
        
        sales_summary = analysis_results.get('sales_summary', {})
        growth = sales_summary.get('growth_metrics', {})
        
        if growth.get('monthly_revenue_growth', 0) > 0:
            recommendations.append("销售呈增长趋势，建议加大营销投入，扩大市场份额。")
        else:
            recommendations.append("销售增长放缓，建议分析原因，优化产品和营销策略。")
        
        customer_analysis = analysis_results.get('customer_analysis', {})
        segment_summary = customer_analysis.get('segment_summary')
        if segment_summary is not None and len(segment_summary) > 0:
            high_value = segment_summary[segment_summary['segment'].str.contains('重要')]
            if len(high_value) > 0:
                recommendations.append("重点关注高价值客户群体，提供个性化服务和专属优惠。")
        
        cross_sell = analysis_results.get('product_analysis', {}).get('cross_sell_opportunities')
        if cross_sell is not None and len(cross_sell) > 0:
            recommendations.append("利用商品关联分析结果，制定捆绑销售策略，提升客单价。")
        
        recommendations.append("持续监控销售预测模型，及时调整库存和营销计划。")
        recommendations.append("建立客户生命周期管理体系，提升客户留存率和复购率。")
        
        for i, rec in enumerate(recommendations, 1):
            story.append(Paragraph(f"{i}. {rec}", self.styles['ChineseBody']))
        
        story.append(Spacer(1, 20))
        story.append(Paragraph("报告结束", self.styles['ChineseBody']))
        
        return story
