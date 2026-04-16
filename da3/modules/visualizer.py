"""
数据可视化模块
生成各类分析图表并保存
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib import font_manager
import os
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 设置样式
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8-whitegrid')


class Visualizer:
    """数据可视化器"""
    
    def __init__(self, output_dir='output'):
        """
        初始化可视化器
        
        Args:
            output_dir: 图表输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.generated_files = []
        
        # 设置图表默认参数
        self.default_figsize = (12, 6)
        self.dpi = 150
        
    def _save_figure(self, fig, filename, formats=None):
        """
        保存图表
        
        Args:
            fig: matplotlib图表对象
            filename: 文件名（不含扩展名）
            formats: 保存格式列表，默认['png', 'pdf']
        """
        if formats is None:
            formats = ['png', 'pdf']
        
        saved_files = []
        for fmt in formats:
            filepath = os.path.join(self.output_dir, f"{filename}.{fmt}")
            try:
                fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
                saved_files.append(filepath)
                print(f"  图表已保存: {filepath}")
            except Exception as e:
                print(f"  保存失败 {filepath}: {str(e)}")
        
        self.generated_files.extend(saved_files)
        plt.close(fig)
        
        return saved_files
    
    def plot_sales_trend(self, df, ts_results=None, filename='01_sales_trend'):
        """
        绘制销售额趋势图（含移动平均）
        
        Args:
            df: 原始数据
            ts_results: 时间序列分析结果
            filename: 输出文件名
        """
        print("\n生成图表: 销售额趋势图...")
        
        fig, ax = plt.subplots(figsize=self.default_figsize)
        
        # 按天聚合销售额
        daily_sales = df.set_index('订单时间').resample('D')['总金额'].sum().fillna(0)
        
        # 绘制原始数据
        ax.plot(daily_sales.index, daily_sales.values, 
               color='steelblue', alpha=0.5, linewidth=1, label='日销售额')
        
        # 绘制移动平均
        ma7 = daily_sales.rolling(window=7, center=True).mean()
        ax.plot(ma7.index, ma7.values, 
               color='red', linewidth=2, label='7日移动平均')
        
        # 如果有STL分解结果，绘制趋势
        if ts_results and 'STL分解' in ts_results and ts_results['STL分解'] is not None:
            trend = ts_results['STL分解']['趋势']
            ax.plot(trend.index, trend.values, 
                   color='green', linewidth=2, linestyle='--', label='趋势成分')
        
        ax.set_xlabel('日期', fontsize=12)
        ax.set_ylabel('销售额 (元)', fontsize=12)
        ax.set_title('销售额趋势分析', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # 格式化x轴日期
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        self._save_figure(fig, filename)
    
    def plot_amount_distribution(self, df, filename='02_amount_distribution'):
        """
        绘制金额分布直方图+箱线图
        
        Args:
            df: 原始数据
            filename: 输出文件名
        """
        print("生成图表: 金额分布图...")
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        amounts = df['总金额']
        
        # 直方图
        ax1 = axes[0]
        n, bins, patches = ax1.hist(amounts, bins=50, color='skyblue', 
                                     edgecolor='black', alpha=0.7)
        ax1.axvline(amounts.mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'均值: ¥{amounts.mean():.2f}')
        ax1.axvline(amounts.median(), color='green', linestyle='--', 
                   linewidth=2, label=f'中位数: ¥{amounts.median():.2f}')
        ax1.set_xlabel('订单金额 (元)', fontsize=12)
        ax1.set_ylabel('频数', fontsize=12)
        ax1.set_title('订单金额分布直方图', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 箱线图
        ax2 = axes[1]
        bp = ax2.boxplot(amounts, vert=False, patch_artist=True,
                        boxprops=dict(facecolor='lightblue', color='blue'),
                        whiskerprops=dict(color='blue'),
                        capprops=dict(color='blue'),
                        medianprops=dict(color='red', linewidth=2))
        ax2.set_xlabel('订单金额 (元)', fontsize=12)
        ax2.set_title('订单金额箱线图', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 添加统计信息
        stats_text = f"最小值: ¥{amounts.min():.2f}\n"
        stats_text += f"Q1: ¥{amounts.quantile(0.25):.2f}\n"
        stats_text += f"中位数: ¥{amounts.median():.2f}\n"
        stats_text += f"Q3: ¥{amounts.quantile(0.75):.2f}\n"
        stats_text += f"最大值: ¥{amounts.max():.2f}"
        ax2.text(0.98, 0.5, stats_text, transform=ax2.transAxes,
                fontsize=10, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                horizontalalignment='right')
        
        plt.tight_layout()
        self._save_figure(fig, filename)
    
    def plot_category_sales(self, df, filename='03_category_sales'):
        """
        绘制类别销售额条形图
        
        Args:
            df: 原始数据
            filename: 输出文件名
        """
        print("生成图表: 类别销售额图...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 按类别汇总
        category_sales = df.groupby('商品类别')['总金额'].sum().sort_values(ascending=True)
        category_counts = df.groupby('商品类别').size().sort_values(ascending=True)
        
        # 销售额条形图
        ax1 = axes[0]
        colors = plt.cm.Set3(np.linspace(0, 1, len(category_sales)))
        bars1 = ax1.barh(category_sales.index, category_sales.values, color=colors)
        ax1.set_xlabel('销售额 (元)', fontsize=12)
        ax1.set_title('各类别销售额', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # 添加数值标签
        for bar in bars1:
            width = bar.get_width()
            ax1.text(width, bar.get_y() + bar.get_height()/2,
                    f'¥{width:,.0f}', ha='left', va='center', fontsize=10)
        
        # 订单数量条形图
        ax2 = axes[1]
        bars2 = ax2.barh(category_counts.index, category_counts.values, color=colors)
        ax2.set_xlabel('订单数量', fontsize=12)
        ax2.set_title('各类别订单数量', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # 添加数值标签
        for bar in bars2:
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2,
                    f'{int(width)}', ha='left', va='center', fontsize=10)
        
        plt.tight_layout()
        self._save_figure(fig, filename)
    
    def plot_customer_clusters(self, cluster_results, filename='04_customer_clusters'):
        """
        绘制用户价值聚类散点图
        
        Args:
            cluster_results: 聚类分析结果
            filename: 输出文件名
        """
        print("生成图表: 用户聚类散点图...")
        
        if cluster_results is None or '聚类数据' not in cluster_results:
            print("  警告: 缺少聚类数据，跳过此图表")
            return
        
        rfm = cluster_results['聚类数据']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 颜色映射
        n_clusters = rfm['聚类标签'].nunique()
        colors = plt.cm.Set1(np.linspace(0, 1, n_clusters))
        
        # 图1: 消费金额 vs 购买频率
        ax1 = axes[0]
        for i in range(n_clusters):
            cluster_data = rfm[rfm['聚类标签'] == i]
            ax1.scatter(cluster_data['购买频率'], cluster_data['消费金额'],
                       c=[colors[i]], label=f'聚类 {i}', alpha=0.6, s=50)
        
        ax1.set_xlabel('购买频率', fontsize=12)
        ax1.set_ylabel('消费金额 (元)', fontsize=12)
        ax1.set_title('用户聚类: 消费金额 vs 购买频率', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 图2: 最近购买天数 vs 消费金额（气泡大小为购买频率）
        ax2 = axes[1]
        for i in range(n_clusters):
            cluster_data = rfm[rfm['聚类标签'] == i]
            scatter = ax2.scatter(cluster_data['最近购买天数'], 
                               cluster_data['消费金额'],
                               s=cluster_data['购买频率'] * 20,
                               c=[colors[i]], label=f'聚类 {i}', alpha=0.6)
        
        ax2.set_xlabel('最近购买天数', fontsize=12)
        ax2.set_ylabel('消费金额 (元)', fontsize=12)
        ax2.set_title('用户聚类: RFM三维可视化\n(气泡大小=购买频率)', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_figure(fig, filename)
    
    def plot_correlation_heatmap(self, df, filename='05_correlation_heatmap'):
        """
        绘制相关性热力图
        
        Args:
            df: 原始数据
            filename: 输出文件名
        """
        print("生成图表: 相关性热力图...")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 选择数值型字段
        numeric_cols = ['单价', '数量', '总金额', '用户评分']
        corr_matrix = df[numeric_cols].corr()
        
        # 绘制热力图
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.3f',
                   cmap='RdYlBu_r', center=0, square=True,
                   linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
        
        ax.set_title('数值字段相关性热力图', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        self._save_figure(fig, filename)
    
    def plot_payment_distribution(self, df, filename='06_payment_distribution'):
        """
        绘制支付方式分布饼图
        
        Args:
            df: 原始数据
            filename: 输出文件名
        """
        print("生成图表: 支付方式分布图...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 支付方式占比（按订单数）
        payment_counts = df['支付方式'].value_counts()
        ax1 = axes[0]
        colors = plt.cm.Set2(np.linspace(0, 1, len(payment_counts)))
        wedges, texts, autotexts = ax1.pie(payment_counts.values, labels=payment_counts.index,
                                           autopct='%1.1f%%', colors=colors,
                                           startangle=90, textprops={'fontsize': 10})
        ax1.set_title('支付方式分布（按订单数）', fontsize=14, fontweight='bold')
        
        # 支付方式占比（按金额）
        payment_amounts = df.groupby('支付方式')['总金额'].sum().sort_values(ascending=False)
        ax2 = axes[1]
        colors = plt.cm.Set2(np.linspace(0, 1, len(payment_amounts)))
        wedges, texts, autotexts = ax2.pie(payment_amounts.values, labels=payment_amounts.index,
                                           autopct='%1.1f%%', colors=colors,
                                           startangle=90, textprops={'fontsize': 10})
        ax2.set_title('支付方式分布（按金额）', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        self._save_figure(fig, filename)
    
    def plot_regional_sales(self, df, filename='07_regional_sales'):
        """
        绘制地区销售分布图
        
        Args:
            df: 原始数据
            filename: 输出文件名
        """
        print("生成图表: 地区销售分布图...")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 按地区汇总
        region_stats = df.groupby('地区').agg({
            '总金额': 'sum',
            '订单ID': 'count'
        }).reset_index()
        region_stats.columns = ['地区', '销售额', '订单数']
        region_stats = region_stats.sort_values('销售额', ascending=True)
        
        # 创建双轴图
        x_pos = np.arange(len(region_stats))
        width = 0.35
        
        bars = ax.barh(x_pos, region_stats['销售额'], color='steelblue', alpha=0.7)
        ax.set_yticks(x_pos)
        ax.set_yticklabels(region_stats['地区'])
        ax.set_xlabel('销售额 (元)', fontsize=12, color='steelblue')
        ax.tick_params(axis='x', labelcolor='steelblue')
        
        # 添加销售额标签
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2,
                   f'¥{width:,.0f}', ha='left', va='center', fontsize=9)
        
        # 第二个y轴显示订单数
        ax2 = ax.twiny()
        ax2.plot(region_stats['订单数'], x_pos, 'ro-', markersize=8, linewidth=2)
        ax2.set_xlabel('订单数量', fontsize=12, color='red')
        ax2.tick_params(axis='x', labelcolor='red')
        
        ax.set_title('各地区销售情况', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        self._save_figure(fig, filename)
    
    def plot_sales_forecast(self, pred_results, filename='08_sales_forecast'):
        """
        绘制销售预测图
        
        Args:
            pred_results: 预测结果
            filename: 输出文件名
        """
        print("生成图表: 销售预测图...")
        
        if pred_results is None:
            print("  警告: 缺少预测数据，跳过此图表")
            return
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # 历史数据
        historical = pred_results['历史数据']
        ax.plot(historical['订单时间'], historical['总金额'],
               color='blue', label='历史销售额', linewidth=1.5)
        
        # 测试集预测
        test_pred = pred_results['测试集预测']
        # 找到测试集对应的时间
        test_size = len(test_pred)
        test_dates = historical['订单时间'].iloc[-test_size:]
        ax.plot(test_dates, test_pred['预测值'],
               color='green', label='测试集预测', linewidth=2, linestyle='--')
        ax.plot(test_dates, test_pred['实际值'],
               color='orange', label='测试集实际', linewidth=1.5, alpha=0.7)
        
        # 未来预测
        future = pred_results['未来预测']
        ax.plot(future['订单时间'], future['预测销售额'],
               color='red', label='未来30天预测', linewidth=2, linestyle='--')
        
        ax.set_xlabel('日期', fontsize=12)
        ax.set_ylabel('销售额 (元)', fontsize=12)
        ax.set_title('销售趋势预测', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # 添加模型评估信息
        model_eval = pred_results['模型评估']
        eval_text = f"模型评估:\n"
        eval_text += f"R² = {model_eval['R²']}\n"
        eval_text += f"RMSE = {model_eval['RMSE']}\n"
        eval_text += f"MAPE = {model_eval['MAPE(%)']}"
        ax.text(0.02, 0.98, eval_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        self._save_figure(fig, filename)
    
    def plot_rfm_distribution(self, rfm_data, filename='09_rfm_distribution'):
        """
        绘制RFM分布图
        
        Args:
            rfm_data: RFM分析结果
            filename: 输出文件名
        """
        print("生成图表: RFM分布图...")
        
        if rfm_data is None:
            print("  警告: 缺少RFM数据，跳过此图表")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # R分布
        ax1 = axes[0, 0]
        ax1.hist(rfm_data['最近购买天数'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        ax1.axvline(rfm_data['最近购买天数'].mean(), color='red', linestyle='--', 
                   label=f'均值: {rfm_data["最近购买天数"].mean():.1f}天')
        ax1.set_xlabel('最近购买天数', fontsize=11)
        ax1.set_ylabel('用户数', fontsize=11)
        ax1.set_title('Recency分布', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # F分布
        ax2 = axes[0, 1]
        ax2.hist(rfm_data['购买频率'], bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
        ax2.axvline(rfm_data['购买频率'].mean(), color='red', linestyle='--',
                   label=f'均值: {rfm_data["购买频率"].mean():.1f}次')
        ax2.set_xlabel('购买频率', fontsize=11)
        ax2.set_ylabel('用户数', fontsize=11)
        ax2.set_title('Frequency分布', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # M分布
        ax3 = axes[1, 0]
        ax3.hist(rfm_data['消费金额'], bins=30, color='salmon', edgecolor='black', alpha=0.7)
        ax3.axvline(rfm_data['消费金额'].mean(), color='red', linestyle='--',
                   label=f'均值: ¥{rfm_data["消费金额"].mean():.2f}')
        ax3.set_xlabel('消费金额 (元)', fontsize=11)
        ax3.set_ylabel('用户数', fontsize=11)
        ax3.set_title('Monetary分布', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 客户分层饼图
        ax4 = axes[1, 1]
        segment_counts = rfm_data['客户分层'].value_counts()
        colors = plt.cm.Set3(np.linspace(0, 1, len(segment_counts)))
        wedges, texts, autotexts = ax4.pie(segment_counts.values, labels=segment_counts.index,
                                           autopct='%1.1f%%', colors=colors,
                                           startangle=90, textprops={'fontsize': 10})
        ax4.set_title('客户分层分布', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        self._save_figure(fig, filename)
    
    def generate_all_visualizations(self, df, analyzer_results):
        """
        生成所有可视化图表
        
        Args:
            df: 原始数据
            analyzer_results: 分析器的结果字典
        """
        print("\n" + "=" * 60)
        print("开始生成可视化图表...")
        print("=" * 60)
        
        # 1. 销售额趋势图
        ts_results = analyzer_results.get('时间序列分析')
        self.plot_sales_trend(df, ts_results)
        
        # 2. 金额分布直方图+箱线图
        self.plot_amount_distribution(df)
        
        # 3. 类别销售额条形图
        self.plot_category_sales(df)
        
        # 4. 用户价值聚类散点图
        cluster_results = analyzer_results.get('用户聚类')
        self.plot_customer_clusters(cluster_results)
        
        # 5. 相关性热力图
        self.plot_correlation_heatmap(df)
        
        # 6. 支付方式分布图
        self.plot_payment_distribution(df)
        
        # 7. 地区销售分布图
        self.plot_regional_sales(df)
        
        # 8. 销售预测图
        pred_results = analyzer_results.get('销售预测')
        self.plot_sales_forecast(pred_results)
        
        # 9. RFM分布图
        rfm_data = analyzer_results.get('RFM分析')
        self.plot_rfm_distribution(rfm_data)
        
        print("\n" + "=" * 60)
        print(f"可视化完成! 共生成 {len(self.generated_files)} 个文件")
        print("=" * 60)
        
        return self.generated_files


def visualize_data(df, analyzer_results, output_dir='output'):
    """
    可视化数据的便捷函数
    
    Args:
        df: 数据框
        analyzer_results: 分析结果
        output_dir: 输出目录
        
    Returns:
        list: 生成的文件列表
    """
    visualizer = Visualizer(output_dir)
    files = visualizer.generate_all_visualizations(df, analyzer_results)
    return files


if __name__ == '__main__':
    # 测试可视化功能
    import sys
    sys.path.append('..')
    from modules.data_generator import generate_raw_data
    from modules.data_cleaner import clean_data
    from modules.data_analyzer import analyze_data
    
    # 生成、清洗、分析数据
    raw_df = generate_raw_data('data/test_raw.csv', n_records=500)
    cleaned_df, _ = clean_data('data/test_raw.csv', 'data/test_cleaned.csv')
    analyzer = analyze_data(cleaned_df)
    
    # 生成可视化
    files = visualize_data(cleaned_df, analyzer.analysis_results, 'output/test')
    print(f"\n生成的文件: {files}")
