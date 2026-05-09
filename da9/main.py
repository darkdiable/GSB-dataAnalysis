#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
电商销售数据分析系统 - 主程序入口
"""

import os
import sys
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_generator import generate_datasets, load_datasets
from data_cleaner import DataCleaner
from data_analyzer import DataAnalyzer
from data_visualizer import DataVisualizer


class ECommerceAnalyzer:
    """
    电商销售分析主类
    """
    
    def __init__(self, project_dir='.'):
        self.project_dir = project_dir
        self.data_dir = os.path.join(project_dir, 'data')
        self.output_dir = os.path.join(project_dir, 'output')
        
        # 确保目录存在
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 数据存储
        self.users = None
        self.products = None
        self.orders = None
        self.order_details = None
        
        self.cleaned_users = None
        self.cleaned_products = None
        self.cleaned_orders = None
        self.cleaned_order_details = None
        
        self.analysis_results = None
        self.generated_charts = None
    
    def run_data_generation(self, n_orders=1000):
        """
        步骤1: 生成模拟数据集
        """
        print("\n" + "="*60)
        print("步骤1: 生成模拟数据集")
        print("="*60)
        
        self.users, self.products, self.orders, self.order_details = generate_datasets(
            n_orders=n_orders,
            n_users=200,
            n_products=100
        )
        
        print(f"\n数据集生成完成:")
        print(f"  - 用户数据: {len(self.users)} 条记录")
        print(f"  - 产品数据: {len(self.products)} 条记录")
        print(f"  - 订单数据: {len(self.orders)} 条记录")
        print(f"  - 订单详情: {len(self.order_details)} 条记录")
        
        return True
    
    def run_data_cleaning(self):
        """
        步骤2: 数据清洗
        """
        print("\n" + "="*60)
        print("步骤2: 数据清洗")
        print("="*60)
        
        if self.users is None:
            print("错误: 请先生成或加载数据")
            return False
        
        cleaner = DataCleaner(self.users, self.products, self.orders, self.order_details)
        self.cleaned_users, self.cleaned_products, self.cleaned_orders, self.cleaned_order_details = cleaner.clean_all()
        
        print(f"\n清洗后数据统计:")
        print(f"  - 用户数据: {len(self.cleaned_users)} 条记录")
        print(f"  - 产品数据: {len(self.cleaned_products)} 条记录")
        print(f"  - 订单数据: {len(self.cleaned_orders)} 条记录")
        print(f"  - 订单详情: {len(self.cleaned_order_details)} 条记录")
        
        return True
    
    def run_analysis(self):
        """
        步骤3: 数据分析
        """
        print("\n" + "="*60)
        print("步骤3: 数据分析")
        print("="*60)
        
        if self.cleaned_orders is None:
            print("错误: 请先执行数据清洗")
            return False
        
        analyzer = DataAnalyzer(
            self.cleaned_users,
            self.cleaned_products,
            self.cleaned_orders,
            self.cleaned_order_details
        )
        
        self.analysis_results = analyzer.run_all_analyses()
        
        print("\n分析完成!")
        return True
    
    def run_visualization(self):
        """
        步骤4: 数据可视化
        """
        print("\n" + "="*60)
        print("步骤4: 数据可视化")
        print("="*60)
        
        if self.analysis_results is None:
            print("错误: 请先执行数据分析")
            return False
        
        visualizer = DataVisualizer(self.analysis_results, self.output_dir)
        self.generated_charts = visualizer.plot_all()
        
        return True
    
    def generate_report(self):
        """
        步骤5: 生成汇总报告 (HTML格式)
        """
        print("\n" + "="*60)
        print("步骤5: 生成汇总报告")
        print("="*60)
        
        if self.analysis_results is None:
            print("错误: 请先执行数据分析")
            return False
        
        # 构建HTML报告
        report_html = self._build_html_report()
        
        # 保存报告
        report_path = os.path.join(self.output_dir, '分析报告.html')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_html)
        
        print(f"报告已生成: {report_path}")
        
        # 同时生成文本版报告
        text_report = self._build_text_report()
        text_report_path = os.path.join(self.output_dir, '分析报告.txt')
        with open(text_report_path, 'w', encoding='utf-8') as f:
            f.write(text_report)
        
        print(f"文本报告已生成: {text_report_path}")
        
        return True
    
    def _build_html_report(self):
        """
        构建HTML格式报告
        """
        # 准备数据
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # RFM数据
        rfm_stats = None
        if 'rfm' in self.analysis_results:
            rfm_stats = self.analysis_results['rfm']['segment_stats']
        
        # 月度销售数据
        monthly_data = None
        if 'monthly_sales' in self.analysis_results:
            monthly_data = self.analysis_results['monthly_sales']
        
        # 类别贡献度
        category_stats = None
        if 'category_contribution' in self.analysis_results:
            category_stats = self.analysis_results['category_contribution']['category_stats']
        
        # 价格相关性
        price_corr = None
        if 'price_quantity_correlation' in self.analysis_results:
            price_corr = self.analysis_results['price_quantity_correlation']['overall_correlation']
        
        # 构建HTML
        html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>电商销售数据分析报告</title>
    <style>
        body {{
            font-family: 'Microsoft YaHei', 'SimHei', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            border-bottom: 3px solid #3498db;
            padding-bottom: 15px;
        }}
        h2 {{
            color: #34495e;
            border-left: 4px solid #3498db;
            padding-left: 15px;
            margin-top: 30px;
        }}
        h3 {{
            color: #7f8c8d;
        }}
        .metadata {{
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .metadata p {{
            margin: 5px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .chart-container {{
            text-align: center;
            margin: 20px 0;
            padding: 15px;
            background-color: #fafafa;
            border-radius: 8px;
        }}
        .chart-container img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        .highlight {{
            background-color: #fff3cd;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }}
        .stat-box {{
            display: inline-block;
            background-color: #3498db;
            color: white;
            padding: 15px 25px;
            margin: 10px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-box .value {{
            font-size: 24px;
            font-weight: bold;
        }}
        .stat-box .label {{
            font-size: 14px;
            margin-top: 5px;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #7f8c8d;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>电商销售数据分析报告</h1>
        
        <div class="metadata">
            <p><strong>报告生成时间:</strong> {current_time}</p>
            <p><strong>分析周期:</strong> 过去12个月</p>
        </div>
        
        <h2>一、数据概览</h2>
        <div class="stat-box">
            <div class="value">{len(self.users) if self.users is not None else 0}</div>
            <div class="label">总用户数</div>
        </div>
        <div class="stat-box">
            <div class="value">{len(self.products) if self.products is not None else 0}</div>
            <div class="label">总产品数</div>
        </div>
        <div class="stat-box">
            <div class="value">{len(self.cleaned_orders) if self.cleaned_orders is not None else 0}</div>
            <div class="label">有效订单数</div>
        </div>
        
        <h2>二、月度销售额趋势分析</h2>
"""
        
        # 添加月度销售数据
        if monthly_data:
            total_sales = monthly_data['total_sales']
            total_orders = monthly_data['total_orders']
            peak_month = monthly_data['peak_month']
            peak_sales = monthly_data['peak_sales']
            
            html += f"""
        <div class="stat-box">
            <div class="value">¥{total_sales:,.2f}</div>
            <div class="label">总销售额</div>
        </div>
        <div class="stat-box">
            <div class="value">{total_orders}</div>
            <div class="label">总订单数</div>
        </div>
        <div class="stat-box">
            <div class="value">{peak_month}</div>
            <div class="label">销售峰值月份</div>
        </div>
        
        <div class="highlight">
            <strong>关键发现:</strong> 销售峰值出现在 {peak_month}，当月销售额为 ¥{peak_sales:,.2f}
        </div>
        
        <h3>月度销售额趋势图</h3>
        <div class="chart-container">
            <img src="1_月度销售额趋势.png" alt="月度销售额趋势图">
        </div>
"""
        
        # 添加RFM分析
        html += f"""
        <h2>三、RFM用户分层分析</h2>
"""
        
        if rfm_stats is not None:
            # 构建RFM表格
            html += """
        <h3>用户分层统计</h3>
        <table>
            <tr>
                <th>用户分层</th>
                <th>用户数</th>
                <th>用户数占比</th>
                <th>平均消费金额</th>
                <th>平均消费次数</th>
            </tr>
"""
            for _, row in rfm_stats.iterrows():
                html += f"""
            <tr>
                <td>{row['Segment']}</td>
                <td>{row['用户数']}</td>
                <td>{row['用户数占比']:.2f}%</td>
                <td>¥{row['平均消费金额']:,.2f}</td>
                <td>{row['平均消费次数']:.1f}</td>
            </tr>
"""
            
            html += """
        </table>
        
        <div class="highlight">
            <strong>营销建议:</strong> 
            <ul>
                <li><strong>重要价值用户</strong>: 提供VIP专属服务，保持高忠诚度</li>
                <li><strong>重要发展用户</strong>: 推荐相关产品，提升消费频率</li>
                <li><strong>重要保持用户</strong>: 发送召回邮件，唤醒沉睡用户</li>
                <li><strong>一般用户</strong>: 发送优惠券，刺激消费</li>
            </ul>
        </div>
        
        <h3>RFM用户分层图</h3>
        <div class="chart-container">
            <img src="2_RFM用户分层.png" alt="RFM用户分层图">
        </div>
"""
        
        # 添加类别贡献度
        html += f"""
        <h2>四、产品类别贡献度分析</h2>
"""
        
        if category_stats is not None:
            top_category = category_stats.iloc[0]
            html += f"""
        <div class="stat-box">
            <div class="value">{top_category['category']}</div>
            <div class="label">销售冠军类别</div>
        </div>
        <div class="stat-box">
            <div class="value">¥{top_category['销售额']:,.0f}</div>
            <div class="label">冠军类别销售额</div>
        </div>
        <div class="stat-box">
            <div class="value">{top_category['销售额占比']:.1f}%</div>
            <div class="label">冠军类别占比</div>
        </div>
        
        <h3>各类别销售统计</h3>
        <table>
            <tr>
                <th>类别</th>
                <th>销售额</th>
                <th>销售额占比</th>
                <th>销量</th>
                <th>平均单价</th>
            </tr>
"""
            for _, row in category_stats.iterrows():
                html += f"""
            <tr>
                <td>{row['category']}</td>
                <td>¥{row['销售额']:,.2f}</td>
                <td>{row['销售额占比']:.2f}%</td>
                <td>{row['销量']}</td>
                <td>¥{row['平均单价']:,.2f}</td>
            </tr>
"""
            
            html += """
        </table>
        
        <h3>产品类别贡献度图</h3>
        <div class="chart-container">
            <img src="3_产品类别贡献度.png" alt="产品类别贡献度图">
        </div>
"""
        
        # 添加价格与销量相关性
        html += f"""
        <h2>五、价格与销量相关性分析</h2>
"""
        
        if price_corr:
            pq_corr = price_corr['价格-销量相关系数']
            po_corr = price_corr['价格-订单数相关系数']
            
            corr_interpretation = ""
            if pq_corr > 0.3:
                corr_interpretation = f"价格与销量呈<strong>正相关</strong>（相关系数: {pq_corr:.4f}），价格越高，销量反而越高，可能是高端产品受欢迎"
            elif pq_corr < -0.3:
                corr_interpretation = f"价格与销量呈<strong>负相关</strong>（相关系数: {pq_corr:.4f}），价格越低，销量越高，符合一般价格规律"
            else:
                corr_interpretation = f"价格与销量<strong>相关性较弱</strong>（相关系数: {pq_corr:.4f}），价格不是影响销量的主要因素"
            
            html += f"""
        <div class="highlight">
            <strong>相关分析结果:</strong> {corr_interpretation}
        </div>
        
        <div class="stat-box">
            <div class="value">{pq_corr:.4f}</div>
            <div class="label">价格-销量相关系数</div>
        </div>
        <div class="stat-box">
            <div class="value">{po_corr:.4f}</div>
            <div class="label">价格-订单数相关系数</div>
        </div>
        
        <h3>价格与销量相关性图</h3>
        <div class="chart-container">
            <img src="4_价格与销量相关性.png" alt="价格与销量相关性图">
        </div>
"""
        
        # 报告总结
        html += """
        <h2>六、总结与建议</h2>
        <div class="highlight">
            <h3>主要发现</h3>
            <ol>
                <li><strong>销售趋势</strong>: 关注销售峰值月份，提前备货和营销</li>
                <li><strong>用户价值</strong>: 重点维护"重要价值用户"，同时关注"重要保持用户"的召回</li>
                <li><strong>产品结构</strong>: 销售冠军类别贡献最大，可考虑加大投入；长尾类别可评估是否需要优化</li>
                <li><strong>定价策略</strong>: 根据价格与销量的相关性，制定合理的定价和促销策略</li>
            </ol>
            
            <h3>行动建议</h3>
            <ul>
                <li>对不同RFM分层用户制定差异化营销策略</li>
                <li>在销售淡季推出促销活动，提升整体销售额</li>
                <li>优化产品组合，重点扶持高贡献度类别</li>
                <li>定期进行数据分析，及时调整经营策略</li>
            </ul>
        </div>
        
        <div class="footer">
            <p>本报告由电商销售数据分析系统自动生成</p>
            <p>技术栈: Python + Pandas + Matplotlib + Seaborn</p>
        </div>
    </div>
</body>
</html>
"""
        
        return html
    
    def _build_text_report(self):
        """
        构建文本格式报告
        """
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        text = f"""
================================================================================
                        电商销售数据分析报告
================================================================================

报告生成时间: {current_time}
分析周期: 过去12个月

--------------------------------------------------------------------------------
一、数据概览
--------------------------------------------------------------------------------
"""
        if self.users is not None:
            text += f"""
  总用户数: {len(self.users)}
  总产品数: {len(self.products)}
  有效订单数: {len(self.cleaned_orders)}
"""
        
        # 月度销售
        text += """

--------------------------------------------------------------------------------
二、月度销售额趋势分析
--------------------------------------------------------------------------------
"""
        if 'monthly_sales' in self.analysis_results:
            ms = self.analysis_results['monthly_sales']
            text += f"""
  总销售额: ¥{ms['total_sales']:,.2f}
  总订单数: {ms['total_orders']}
  平均月销售额: ¥{ms['avg_monthly_sales']:,.2f}
  销售峰值月份: {ms['peak_month']} (¥{ms['peak_sales']:,.2f})
  销售低谷月份: {ms['low_month']} (¥{ms['low_sales']:,.2f})
"""
        
        # RFM分析
        text += """

--------------------------------------------------------------------------------
三、RFM用户分层分析
--------------------------------------------------------------------------------
"""
        if 'rfm' in self.analysis_results:
            rfm_stats = self.analysis_results['rfm']['segment_stats']
            text += "\n  用户分层统计:\n"
            text += "  " + "-"*80 + "\n"
            text += f"  {'用户分层':<15} {'用户数':<10} {'占比':<10} {'平均金额':<15} {'平均次数':<10}\n"
            text += "  " + "-"*80 + "\n"
            for _, row in rfm_stats.iterrows():
                text += f"  {row['Segment']:<15} {row['用户数']:<10} {row['用户数占比']:.2f}%{'':<6} ¥{row['平均消费金额']:,.2f}{'':<6} {row['平均消费次数']:.1f}\n"
            
            text += """
  营销建议:
  - 重要价值用户: 提供VIP专属服务，保持高忠诚度
  - 重要发展用户: 推荐相关产品，提升消费频率
  - 重要保持用户: 发送召回邮件，唤醒沉睡用户
  - 一般用户: 发送优惠券，刺激消费
"""
        
        # 类别贡献度
        text += """

--------------------------------------------------------------------------------
四、产品类别贡献度分析
--------------------------------------------------------------------------------
"""
        if 'category_contribution' in self.analysis_results:
            cc = self.analysis_results['category_contribution']
            category_stats = cc['category_stats']
            
            text += f"""
  销售冠军类别: {cc['top_category']}
  冠军类别销售额: ¥{cc['top_category_sales']:,.2f}
  冠军类别占比: {cc['top_category_share']:.2f}%

  各类别销售统计:
"""
            text += "  " + "-"*70 + "\n"
            text += f"  {'类别':<12} {'销售额':<15} {'占比':<10} {'销量':<10} {'平均单价':<12}\n"
            text += "  " + "-"*70 + "\n"
            for _, row in category_stats.iterrows():
                text += f"  {row['category']:<12} ¥{row['销售额']:,.2f}{'':<5} {row['销售额占比']:.2f}%{'':<4} {row['销量']:<10} ¥{row['平均单价']:,.2f}\n"
        
        # 价格相关性
        text += """

--------------------------------------------------------------------------------
五、价格与销量相关性分析
--------------------------------------------------------------------------------
"""
        if 'price_quantity_correlation' in self.analysis_results:
            pq = self.analysis_results['price_quantity_correlation']
            corr = pq['overall_correlation']
            
            text += f"""
  价格-销量相关系数: {corr['价格-销量相关系数']:.4f}
  价格-订单数相关系数: {corr['价格-订单数相关系数']:.4f}
"""
            
            pq_corr = corr['价格-销量相关系数']
            if pq_corr > 0.3:
                text += f"\n  解读: 价格与销量呈正相关（{pq_corr:.4f}），高端产品可能更受欢迎"
            elif pq_corr < -0.3:
                text += f"\n  解读: 价格与销量呈负相关（{pq_corr:.4f}），价格越低销量越高"
            else:
                text += f"\n  解读: 价格与销量相关性较弱（{pq_corr:.4f}），价格不是主要影响因素"
        
        # 总结
        text += """

================================================================================
六、总结与建议
================================================================================

主要发现:
1. 关注销售峰值月份，提前备货和营销
2. 重点维护"重要价值用户"，同时关注"重要保持用户"的召回
3. 销售冠军类别贡献最大，可考虑加大投入
4. 根据价格与销量的相关性，制定合理的定价策略

行动建议:
- 对不同RFM分层用户制定差异化营销策略
- 在销售淡季推出促销活动，提升整体销售额
- 优化产品组合，重点扶持高贡献度类别
- 定期进行数据分析，及时调整经营策略

================================================================================
                              报告结束
================================================================================
"""
        
        return text
    
    def run_full_analysis(self, n_orders=1000):
        """
        执行完整分析流程
        """
        print("\n" + "="*60)
        print("电商销售数据分析系统")
        print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        # 执行所有步骤
        success = True
        
        if not self.run_data_generation(n_orders):
            success = False
        
        if success and not self.run_data_cleaning():
            success = False
        
        if success and not self.run_analysis():
            success = False
        
        if success and not self.run_visualization():
            success = False
        
        if success and not self.generate_report():
            success = False
        
        # 最终总结
        print("\n" + "="*60)
        if success:
            print("分析完成!")
            print(f"\n输出文件:")
            if self.generated_charts:
                print(f"  图表:")
                for chart in self.generated_charts:
                    print(f"    - {chart}")
            print(f"  报告:")
            print(f"    - {os.path.join(self.output_dir, '分析报告.html')}")
            print(f"    - {os.path.join(self.output_dir, '分析报告.txt')}")
        else:
            print("分析过程中出现错误，请检查日志。")
        print("="*60)
        
        return success


def main():
    """
    主函数
    """
    # 获取项目目录
    project_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 创建分析器
    analyzer = ECommerceAnalyzer(project_dir)
    
    # 执行完整分析（1000条订单）
    analyzer.run_full_analysis(n_orders=1000)


if __name__ == '__main__':
    main()
