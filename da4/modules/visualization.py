"""
可视化模块 - 生成销售趋势图、预测对比图和特征重要性图
"""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class Visualizer:
    """可视化类，生成各种图表"""
    
    def __init__(self, output_dir='output'):
        """
        初始化可视化器
        
        Parameters:
            output_dir: 输出目录路径
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置图表样式
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['figure.dpi'] = 100
    
    def plot_sales_trend(self, df, title="销售趋势分析", filename="sales_trend.png"):
        """
        绘制销售趋势折线图
        
        Parameters:
            df: 销售数据DataFrame
            title: 图表标题
            filename: 输出文件名
            
        Returns:
            str: 保存的文件路径
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # 绘制销售趋势线
        ax.plot(df['date'], df['sales'], linewidth=1.5, color='#2E86AB', alpha=0.8)
        
        # 添加7日移动平均线
        if len(df) >= 7:
            df['ma_7'] = df['sales'].rolling(window=7).mean()
            ax.plot(df['date'], df['ma_7'], linewidth=2, color='#F24236', 
                   label='7日移动平均', linestyle='--')
        
        # 添加30日移动平均线
        if len(df) >= 30:
            df['ma_30'] = df['sales'].rolling(window=30).mean()
            ax.plot(df['date'], df['ma_30'], linewidth=2, color='#F6AE2D', 
                   label='30日移动平均', linestyle='-')
        
        # 设置图表属性
        ax.set_xlabel('日期', fontsize=12)
        ax.set_ylabel('销售额', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        
        # 格式化x轴日期
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.xticks(rotation=45)
        
        # 添加网格
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"销售趋势图已保存: {filepath}")
        return filepath
    
    def plot_prediction_comparison(self, historical_df, prediction_df, 
                                   title="销售预测对比", filename="prediction_comparison.png"):
        """
        绘制预测结果对比图
        
        Parameters:
            historical_df: 历史销售数据
            prediction_df: 预测结果数据
            title: 图表标题
            filename: 输出文件名
            
        Returns:
            str: 保存的文件路径
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # 绘制历史数据（最近90天）
        recent_data = historical_df.tail(90) if len(historical_df) > 90 else historical_df
        ax.plot(recent_data['date'], recent_data['sales'], 
               linewidth=2, color='#2E86AB', label='历史销售额', marker='o', markersize=3)
        
        # 绘制预测数据
        ax.plot(prediction_df['date'], prediction_df['predicted_sales'], 
               linewidth=2, color='#F24236', label='预测销售额', 
               marker='s', markersize=4, linestyle='--')
        
        # 添加预测区间阴影
        ax.fill_between(prediction_df['date'], 
                       prediction_df['predicted_sales'] * 0.9,
                       prediction_df['predicted_sales'] * 1.1,
                       alpha=0.2, color='#F24236', label='预测区间(±10%)')
        
        # 添加垂直分隔线
        if len(recent_data) > 0:
            split_date = recent_data['date'].max()
            ax.axvline(x=split_date, color='gray', linestyle=':', linewidth=2, alpha=0.7)
            ax.text(split_date, ax.get_ylim()[1] * 0.95, '预测起点', 
                   ha='right', va='top', fontsize=10, color='gray')
        
        # 设置图表属性
        ax.set_xlabel('日期', fontsize=12)
        ax.set_ylabel('销售额', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        
        # 格式化x轴日期
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        plt.xticks(rotation=45)
        
        # 添加网格
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"预测对比图已保存: {filepath}")
        return filepath
    
    def plot_feature_importance(self, feature_importance_df, 
                                title="特征重要性分析", filename="feature_importance.png"):
        """
        绘制特征重要性柱状图
        
        Parameters:
            feature_importance_df: 特征重要性DataFrame
            title: 图表标题
            filename: 输出文件名
            
        Returns:
            str: 保存的文件路径
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 取前15个重要特征
        top_features = feature_importance_df.head(15)
        
        # 创建水平柱状图
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
        bars = ax.barh(range(len(top_features)), top_features['importance'], color=colors)
        
        # 设置y轴标签
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        
        # 添加数值标签
        for i, (idx, row) in enumerate(top_features.iterrows()):
            ax.text(row['importance'], i, f' {row["importance"]:.3f}', 
                   va='center', fontsize=9)
        
        # 设置图表属性
        ax.set_xlabel('重要性得分', fontsize=12)
        ax.set_ylabel('特征', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # 反转y轴，使最重要的特征在顶部
        ax.invert_yaxis()
        
        # 添加网格
        ax.grid(True, axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"特征重要性图已保存: {filepath}")
        return filepath
    
    def plot_seasonal_analysis(self, df, title="季节性分析", filename="seasonal_analysis.png"):
        """
        绘制季节性分析图
        
        Parameters:
            df: 销售数据DataFrame
            title: 图表标题
            filename: 输出文件名
            
        Returns:
            str: 保存的文件路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 按月统计
        monthly_sales = df.groupby('month')['sales'].mean()
        axes[0, 0].bar(monthly_sales.index, monthly_sales.values, color='#2E86AB')
        axes[0, 0].set_xlabel('月份')
        axes[0, 0].set_ylabel('平均销售额')
        axes[0, 0].set_title('月度平均销售额')
        axes[0, 0].set_xticks(range(1, 13))
        
        # 2. 按星期统计
        weekday_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
        weekday_sales = df.groupby('dayofweek')['sales'].mean()
        axes[0, 1].bar(range(7), [weekday_sales.get(i, 0) for i in range(7)], 
                      color='#F6AE2D')
        axes[0, 1].set_xlabel('星期')
        axes[0, 1].set_ylabel('平均销售额')
        axes[0, 1].set_title('星期平均销售额')
        axes[0, 1].set_xticks(range(7))
        axes[0, 1].set_xticklabels(weekday_names, rotation=45)
        
        # 3. 按日期统计
        day_sales = df.groupby('day')['sales'].mean()
        axes[1, 0].plot(day_sales.index, day_sales.values, marker='o', color='#F24236')
        axes[1, 0].set_xlabel('日期')
        axes[1, 0].set_ylabel('平均销售额')
        axes[1, 0].set_title('日度平均销售额')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 周末vs工作日
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        weekend_sales = df.groupby('is_weekend')['sales'].mean()
        labels = ['工作日', '周末']
        axes[1, 1].bar(labels, weekend_sales.values, color=['#2E86AB', '#F24236'])
        axes[1, 1].set_ylabel('平均销售额')
        axes[1, 1].set_title('周末vs工作日平均销售额')
        
        # 添加数值标签
        for i, v in enumerate(weekend_sales.values):
            axes[1, 1].text(i, v, f'{v:.0f}', ha='center', va='bottom')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # 保存图表
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"季节性分析图已保存: {filepath}")
        return filepath
    
    def generate_all_visualizations(self, historical_df, prediction_df, 
                                   feature_importance_df):
        """
        生成所有可视化图表
        
        Parameters:
            historical_df: 历史销售数据
            prediction_df: 预测结果数据
            feature_importance_df: 特征重要性数据
            
        Returns:
            list: 所有生成的文件路径
        """
        files = []
        
        # 1. 销售趋势图
        files.append(self.plot_sales_trend(historical_df))
        
        # 2. 预测对比图
        files.append(self.plot_prediction_comparison(historical_df, prediction_df))
        
        # 3. 特征重要性图
        if feature_importance_df is not None and len(feature_importance_df) > 0:
            files.append(self.plot_feature_importance(feature_importance_df))
        
        # 4. 季节性分析图
        files.append(self.plot_seasonal_analysis(historical_df))
        
        print(f"\n所有可视化图表已生成，共 {len(files)} 个文件")
        return files
