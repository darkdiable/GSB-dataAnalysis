# -*- coding: utf-8 -*-
"""
数据可视化模块
生成各类图表并保存
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
import warnings
import os
warnings.filterwarnings('ignore')

# 查找系统中文字体
def get_chinese_font():
    """获取系统中文字体"""
    # 尝试的字体名称列表
    font_names = ['PingFang SC', 'Heiti SC', 'Songti SC', 'STHeiti', 'SimHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC']
    
    # 首先尝试通过名称查找
    for font_name in font_names:
        try:
            fp = FontProperties(family=font_name)
            # 测试字体是否可用
            test_text = fp.get_name()
            if test_text:
                print(f"使用字体: {font_name}")
                return fp
        except:
            continue
    
    # 回退到默认
    return None

# 获取中文字体
CHINESE_FONT = get_chinese_font()

# 辅助函数：设置文本时使用中文字体
def set_text_with_font(text_obj, text, fontproperties=None, **kwargs):
    """设置文本，使用中文字体"""
    if fontproperties is None:
        fontproperties = CHINESE_FONT
    if fontproperties:
        text_obj.set_text(text)
        text_obj.set_fontproperties(fontproperties)
    else:
        text_obj.set_text(text)


class Visualizer:
    """电商销售数据可视化器"""
    
    def __init__(self, output_dir='../output'):
        """
        初始化可视化器
        
        参数:
            output_dir: 图表输出目录
        """
        self.output_dir = output_dir
        self.figures = []
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置图表样式
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['figure.dpi'] = 100
        
    def _save_figure(self, filename, formats=['png', 'pdf']):
        """保存图表"""
        saved_files = []
        for fmt in formats:
            filepath = os.path.join(self.output_dir, f"{filename}.{fmt}")
            plt.savefig(filepath, format=fmt, bbox_inches='tight', dpi=150)
            saved_files.append(filepath)
            print(f"图表已保存: {filepath}")
        return saved_files
    
    def plot_sales_trend(self, daily_sales, title="销售额趋势分析"):
        """
        绘制销售额趋势折线图（含移动平均）
        
        参数:
            daily_sales: 日销售数据DataFrame
            title: 图表标题
        """
        print("\n生成图表: 销售额趋势折线图...")
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # 绘制原始销售额
        ax.plot(daily_sales['日期'], daily_sales['销售额'], 
               label='Daily Sales', color='#3498db', alpha=0.6, linewidth=1)
        
        # 绘制移动平均线
        if 'MA_7' in daily_sales.columns:
            ax.plot(daily_sales['日期'], daily_sales['MA_7'], 
                   label='7-Day MA', color='#e74c3c', linewidth=2)
        if 'MA_14' in daily_sales.columns:
            ax.plot(daily_sales['日期'], daily_sales['MA_14'], 
                   label='14-Day MA', color='#f39c12', linewidth=2)
        if 'MA_30' in daily_sales.columns:
            ax.plot(daily_sales['日期'], daily_sales['MA_30'], 
                   label='30-Day MA', color='#2ecc71', linewidth=2)
        
        # 绘制趋势线
        if '趋势值' in daily_sales.columns:
            ax.plot(daily_sales['日期'], daily_sales['趋势值'], 
                   label='Trend', color='#9b59b6', linestyle='--', linewidth=2)
        
        # 设置标签 - 使用英文避免字体问题
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Sales Amount (CNY)', fontsize=12)
        ax.set_title('Sales Trend Analysis', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 格式化x轴日期
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        self._save_figure('01_sales_trend')
        plt.close()
        
    def plot_amount_distribution(self, df, title="交易金额分布"):
        """
        绘制金额分布直方图+箱线图
        
        参数:
            df: 销售数据DataFrame
            title: 图表标题
        """
        print("生成图表: 金额分布直方图+箱线图...")
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), 
                                gridspec_kw={'height_ratios': [3, 1]})
        
        # 直方图
        ax1 = axes[0]
        n, bins, patches = ax1.hist(df['交易金额'], bins=50, 
                                    color='#3498db', alpha=0.7, edgecolor='white')
        ax1.axvline(df['交易金额'].mean(), color='#e74c3c', 
                   linestyle='--', linewidth=2, label=f'Mean: {df["交易金额"].mean():.2f}')
        ax1.axvline(df['交易金额'].median(), color='#2ecc71', 
                   linestyle='--', linewidth=2, label=f'Median: {df["交易金额"].median():.2f}')
        ax1.set_xlabel('Transaction Amount (CNY)', fontsize=11)
        ax1.set_ylabel('Frequency', fontsize=11)
        ax1.set_title('Transaction Amount Distribution', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 箱线图
        ax2 = axes[1]
        box_plot = ax2.boxplot(df['交易金额'], vert=False, patch_artist=True,
                              widths=0.5, showmeans=True,
                              meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
        box_plot['boxes'][0].set_facecolor('#3498db')
        box_plot['boxes'][0].set_alpha(0.7)
        ax2.set_xlabel('Transaction Amount (CNY)', fontsize=11)
        ax2.set_title('Transaction Amount Box Plot', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # 添加统计信息文本
        stats_text = f"Min: {df['交易金额'].min():.2f}\n"
        stats_text += f"Q1: {df['交易金额'].quantile(0.25):.2f}\n"
        stats_text += f"Median: {df['交易金额'].median():.2f}\n"
        stats_text += f"Q3: {df['交易金额'].quantile(0.75):.2f}\n"
        stats_text += f"Max: {df['交易金额'].max():.2f}"
        ax2.text(0.98, 0.5, stats_text, transform=ax2.transAxes, 
                fontsize=9, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                horizontalalignment='right')
        
        plt.tight_layout()
        self._save_figure('02_amount_distribution')
        plt.close()
        
    def plot_category_sales(self, df, title="各类别销售额对比"):
        """
        绘制类别销售额条形图
        
        参数:
            df: 销售数据DataFrame
            title: 图表标题
        """
        print("生成图表: 类别销售额条形图...")
        
        # 计算类别统计
        category_stats = df.groupby('商品类别').agg({
            '交易金额': ['sum', 'count', 'mean']
        }).round(2)
        category_stats.columns = ['Total Sales', 'Order Count', 'Avg Order Value']
        category_stats = category_stats.sort_values('Total Sales', ascending=True)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(category_stats)))
        
        # 总销售额
        ax1 = axes[0]
        bars1 = ax1.barh(range(len(category_stats)), category_stats['Total Sales'], color=colors)
        ax1.set_yticks(range(len(category_stats)))
        ax1.set_yticklabels(category_stats.index)
        ax1.set_xlabel('Total Sales (CNY)', fontsize=11)
        ax1.set_title('Total Sales by Category', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')
        # 添加数值标签
        for i, bar in enumerate(bars1):
            width = bar.get_width()
            ax1.text(width, bar.get_y() + bar.get_height()/2, 
                    f'{width:,.0f}', ha='left', va='center', fontsize=9)
        
        # 订单数
        ax2 = axes[1]
        bars2 = ax2.barh(range(len(category_stats)), category_stats['Order Count'], color=colors)
        ax2.set_yticks(range(len(category_stats)))
        ax2.set_yticklabels(category_stats.index)
        ax2.set_xlabel('Order Count', fontsize=11)
        ax2.set_title('Order Count by Category', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        for i, bar in enumerate(bars2):
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2, 
                    f'{int(width)}', ha='left', va='center', fontsize=9)
        
        # 平均订单金额
        ax3 = axes[2]
        bars3 = ax3.barh(range(len(category_stats)), category_stats['Avg Order Value'], color=colors)
        ax3.set_yticks(range(len(category_stats)))
        ax3.set_yticklabels(category_stats.index)
        ax3.set_xlabel('Avg Order Value (CNY)', fontsize=11)
        ax3.set_title('Avg Order Value by Category', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')
        for i, bar in enumerate(bars3):
            width = bar.get_width()
            ax3.text(width, bar.get_y() + bar.get_height()/2, 
                    f'{width:.0f}', ha='left', va='center', fontsize=9)
        
        plt.suptitle('Category Sales Analysis', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save_figure('03_category_sales')
        plt.close()
        
    def plot_user_clustering(self, rfm_data, title="用户价值聚类分析"):
        """
        绘制用户聚类散点图
        
        参数:
            rfm_data: RFM数据DataFrame（包含聚类标签）
            title: 图表标题
        """
        print("生成图表: 用户价值聚类散点图...")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # 使用PCA降维后的散点图
        ax1 = axes[0]
        cluster_labels = rfm_data['聚类标签'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(cluster_labels)))
        
        cluster_name_map = {
            '高价值客户': 'High Value',
            '新客户': 'New Customer',
            '低价值客户': 'Low Value',
            '频繁购买客户': 'Frequent Buyer'
        }
        
        for i, label in enumerate(sorted(cluster_labels)):
            cluster_data = rfm_data[rfm_data['聚类标签'] == label]
            customer_type = cluster_data['客户类型'].iloc[0]
            english_name = cluster_name_map.get(customer_type, f'Cluster {label}')
            ax1.scatter(cluster_data['PCA1'], cluster_data['PCA2'], 
                       c=[colors[i]], label=f'{english_name}',
                       alpha=0.6, s=50, edgecolors='white', linewidth=0.5)
        
        ax1.set_xlabel('Principal Component 1', fontsize=11)
        ax1.set_ylabel('Principal Component 2', fontsize=11)
        ax1.set_title('User Clustering (PCA)', fontsize=12, fontweight='bold')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # RFM三维散点图（使用颜色和大小）
        ax2 = axes[1]
        scatter = ax2.scatter(rfm_data['F_购买频次'], rfm_data['M_消费金额'],
                            c=rfm_data['R_最近购买天数'], s=100, 
                            cmap='RdYlGn_r', alpha=0.6, edgecolors='white')
        ax2.set_xlabel('Frequency (F)', fontsize=11)
        ax2.set_ylabel('Monetary (M)', fontsize=11)
        ax2.set_title('User RFM Distribution (Color = Recency)', fontsize=12, fontweight='bold')
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Recency (R) - Days', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('User Value Clustering Analysis', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save_figure('04_user_clustering')
        plt.close()
        
    def plot_correlation_heatmap(self, corr_matrix, title="相关性热力图"):
        """
        绘制相关性热力图
        
        参数:
            corr_matrix: 相关性矩阵
            title: 图表标题
        """
        print("生成图表: 相关性热力图...")
        
        # 重命名列和索引为英文
        corr_matrix_en = corr_matrix.copy()
        name_map = {
            '交易金额': 'Amount',
            '单价': 'Unit Price',
            '数量': 'Quantity',
            '用户评分': 'Rating'
        }
        corr_matrix_en.columns = [name_map.get(c, c) for c in corr_matrix_en.columns]
        corr_matrix_en.index = [name_map.get(i, i) for i in corr_matrix_en.index]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 创建热力图
        mask = np.triu(np.ones_like(corr_matrix_en, dtype=bool))
        heatmap = sns.heatmap(corr_matrix_en, mask=mask, annot=True, 
                             fmt='.3f', cmap='RdBu_r', center=0,
                             square=True, linewidths=0.5, 
                             cbar_kws={"shrink": 0.8}, ax=ax,
                             annot_kws={'size': 11})
        
        ax.set_title('Correlation Heatmap', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        self._save_figure('05_correlation_heatmap')
        plt.close()
        
    def plot_sales_prediction(self, prediction_results, title="销售预测"):
        """
        绘制销售预测图
        
        参数:
            prediction_results: 预测结果字典
            title: 图表标题
        """
        print("生成图表: 销售预测图...")
        
        daily_sales = prediction_results['daily_sales']
        y_pred_test = prediction_results['y_pred_test']
        future_predictions = prediction_results['future_predictions']
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # 历史数据
        train_size = int(len(daily_sales) * 0.8)
        train_data = daily_sales[:train_size]
        test_data = daily_sales[train_size:]
        
        ax.plot(train_data['交易日期'], train_data['交易金额'], 
               label='Training Data', color='#3498db', linewidth=1.5)
        ax.plot(test_data['交易日期'], test_data['交易金额'], 
               label='Test Data', color='#2ecc71', linewidth=1.5)
        ax.plot(test_data['交易日期'], y_pred_test, 
               label='Test Prediction', color='#e74c3c', linestyle='--', linewidth=2)
        
        # 未来预测
        ax.plot(future_predictions['日期'], future_predictions['预测销售额'], 
               label='Future Prediction', color='#9b59b6', linestyle='--', linewidth=2, marker='o', markersize=4)
        
        # 添加预测区间阴影
        ax.fill_between(future_predictions['日期'], 
                       future_predictions['预测销售额'] * 0.9,
                       future_predictions['预测销售额'] * 1.1,
                       alpha=0.2, color='#9b59b6', label='Prediction Interval (±10%)')
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Sales Amount (CNY)', fontsize=12)
        ax.set_title('Sales Prediction', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 格式化x轴
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        
        # 添加评估指标文本
        metrics_text = f"MAE: {prediction_results['test_mae']:.2f}\n"
        metrics_text += f"RMSE: {prediction_results['test_rmse']:.2f}\n"
        metrics_text += f"R²: {prediction_results['test_r2']:.4f}"
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        self._save_figure('06_sales_prediction')
        plt.close()
        
    def plot_weekly_pattern(self, df, title="周内销售模式"):
        """
        绘制周内销售模式图
        
        参数:
            df: 销售数据DataFrame
            title: 图表标题
        """
        print("生成图表: 周内销售模式图...")
        
        # 添加星期信息
        df['星期'] = pd.to_datetime(df['交易日期']).dt.day_name()
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        weekday_sales = df.groupby('星期').agg({
            '交易金额': ['sum', 'mean', 'count']
        }).round(2)
        weekday_sales.columns = ['Total Sales', 'Avg Sales', 'Order Count']
        
        # 按星期顺序排序
        weekday_sales = weekday_sales.reindex(weekday_order)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 总销售额
        ax1 = axes[0]
        bars = ax1.bar(weekday_sales.index, weekday_sales['Total Sales'], 
                      color=plt.cm.viridis(np.linspace(0, 1, 7)))
        ax1.set_ylabel('Total Sales (CNY)', fontsize=11)
        ax1.set_title('Total Sales by Day of Week', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:,.0f}', ha='center', va='bottom', fontsize=9)
        
        # 平均销售额
        ax2 = axes[1]
        bars = ax2.bar(weekday_sales.index, weekday_sales['Avg Sales'], 
                      color=plt.cm.plasma(np.linspace(0, 1, 7)))
        ax2.set_ylabel('Avg Order Value (CNY)', fontsize=11)
        ax2.set_title('Avg Order Value by Day of Week', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}', ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('Weekly Sales Pattern', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save_figure('07_weekly_pattern')
        plt.close()
        
    def generate_all_visualizations(self, df, analysis_results):
        """
        生成所有可视化图表
        
        参数:
            df: 清洗后的数据
            analysis_results: 分析结果字典
        """
        print("\n" + "=" * 50)
        print("开始生成所有可视化图表...")
        print("=" * 50)
        
        # 1. 销售额趋势图
        if 'time_series' in analysis_results:
            self.plot_sales_trend(analysis_results['time_series']['daily_sales'])
        
        # 2. 金额分布图
        self.plot_amount_distribution(df)
        
        # 3. 类别销售图
        self.plot_category_sales(df)
        
        # 4. 用户聚类图
        if 'clustering' in analysis_results:
            self.plot_user_clustering(analysis_results['clustering']['rfm_data'])
        
        # 5. 相关性热力图
        if 'correlation' in analysis_results:
            self.plot_correlation_heatmap(analysis_results['correlation']['correlation_matrix'])
        
        # 6. 销售预测图
        if 'prediction' in analysis_results:
            self.plot_sales_prediction(analysis_results['prediction'])
        
        # 7. 周内模式图
        self.plot_weekly_pattern(df)
        
        print("\n" + "=" * 50)
        print("所有图表生成完成!")
        print(f"图表保存位置: {self.output_dir}")
        print("=" * 50)


if __name__ == '__main__':
    # 测试可视化功能
    import sys
    sys.path.append('..')
    from modules.data_generator import DataGenerator
    from modules.data_cleaner import DataCleaner
    from modules.data_analyzer import DataAnalyzer
    
    # 生成并处理数据
    generator = DataGenerator(n_records=600)
    raw_data = generator.generate_data()
    cleaner = DataCleaner(raw_data)
    clean_data = cleaner.clean_data()
    
    # 分析数据
    analyzer = DataAnalyzer(clean_data)
    analyzer.descriptive_statistics()
    analyzer.correlation_analysis()
    analyzer.time_series_analysis()
    analyzer.user_clustering(n_clusters=4)
    analyzer.sales_prediction(forecast_days=30)
    
    # 生成可视化
    results = analyzer.get_all_results()
    visualizer = Visualizer(output_dir='../output')
    visualizer.generate_all_visualizations(clean_data, results)
