import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


class Visualizer:
    def __init__(self, figsize=(12, 8), dpi=300):
        self.figsize = figsize
        self.dpi = dpi
        self.output_dir = '/Users/bilei/work/LargeModelAnnotation/GBS/260410/GSB-dataAnalysis/da7/output/figures'
        os.makedirs(self.output_dir, exist_ok=True)
        
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = figsize
        plt.rcParams['figure.dpi'] = dpi
        
    def load_data(self, processed_path='data/processed_data.csv', importance_path='output/feature_importance.csv'):
        self.data = pd.read_csv(os.path.join('/Users/bilei/work/LargeModelAnnotation/GBS/260410/GSB-dataAnalysis/da7', processed_path))
        if os.path.exists(os.path.join('/Users/bilei/work/LargeModelAnnotation/GBS/260410/GSB-dataAnalysis/da7', importance_path)):
            self.feature_importance = pd.read_csv(os.path.join('/Users/bilei/work/LargeModelAnnotation/GBS/260410/GSB-dataAnalysis/da7', importance_path))
        else:
            self.feature_importance = None
        print("数据加载成功！")
        return self.data
    
    def plot_category_completion_rate(self, save=True):
        print("\n绘制各类别课程平均完成率柱状图...")
        
        category_stats = self.data.groupby('course_category').agg({
            'is_completed': ['mean', 'count']
        }).reset_index()
        category_stats.columns = ['course_category', 'completion_rate', 'course_count']
        category_stats = category_stats.sort_values('completion_rate', ascending=True)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = plt.cm.RdYlGn(category_stats['completion_rate'])
        bars = ax.barh(category_stats['course_category'], category_stats['completion_rate'], color=colors)
        
        ax.set_xlabel('平均完成率', fontsize=12, fontweight='bold')
        ax.set_ylabel('课程类别', fontsize=12, fontweight='bold')
        ax.set_title('各类别课程的平均完成率', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlim(0, 1)
        
        for i, (bar, rate, count) in enumerate(zip(bars, category_stats['completion_rate'], category_stats['course_count'])):
            ax.text(rate + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{rate:.1%} (n={count})', 
                   va='center', fontsize=10)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'category_completion_rate.png')
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight', facecolor='white')
            print(f"  图表已保存: {filepath}")
        
        plt.close()
        return fig
    
    def plot_duration_vs_score_scatter(self, save=True):
        print("\n绘制用户观看时长与测验分数关系散点图...")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        completed = self.data[self.data['is_completed'] == 1]
        not_completed = self.data[self.data['is_completed'] == 0]
        
        ax.scatter(not_completed['watch_duration_minutes'], not_completed['quiz_score'], 
                  alpha=0.5, c='#e74c3c', s=30, label='未完成', edgecolors='white', linewidth=0.5)
        ax.scatter(completed['watch_duration_minutes'], completed['quiz_score'], 
                  alpha=0.6, c='#2ecc71', s=30, label='已完成', edgecolors='white', linewidth=0.5)
        
        z = np.polyfit(self.data['watch_duration_minutes'], self.data['quiz_score'], 1)
        p = np.poly1d(z)
        ax.plot(self.data['watch_duration_minutes'].sort_values(), 
               p(self.data['watch_duration_minutes'].sort_values()), 
               "b--", alpha=0.8, linewidth=2, label=f'趋势线 (y={z[0]:.3f}x+{z[1]:.2f})')
        
        correlation = self.data['watch_duration_minutes'].corr(self.data['quiz_score'])
        
        ax.set_xlabel('观看时长 (分钟)', fontsize=12, fontweight='bold')
        ax.set_ylabel('测验分数', fontsize=12, fontweight='bold')
        ax.set_title(f'用户观看时长与测验分数的关系\n(相关系数: {correlation:.3f})', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'duration_vs_score_scatter.png')
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight', facecolor='white')
            print(f"  图表已保存: {filepath}")
        
        plt.close()
        return fig
    
    def plot_feature_importance(self, feature_importance_df=None, save=True):
        print("\n绘制模型特征重要性条形图...")
        
        if feature_importance_df is not None:
            self.feature_importance = feature_importance_df
        
        if self.feature_importance is None:
            print("  错误: 未提供特征重要性数据")
            return None
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        df = self.feature_importance.sort_values('importance', ascending=True)
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(df)))
        bars = ax.barh(df['feature'], df['importance'], color=colors)
        
        ax.set_xlabel('重要性得分', fontsize=12, fontweight='bold')
        ax.set_ylabel('特征', fontsize=12, fontweight='bold')
        ax.set_title('随机森林模型特征重要性排序', fontsize=14, fontweight='bold', pad=20)
        
        for bar, importance in zip(bars, df['importance']):
            ax.text(importance + 0.005, bar.get_y() + bar.get_height()/2, 
                   f'{importance:.3f}', 
                   va='center', fontsize=10)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'feature_importance.png')
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight', facecolor='white')
            print(f"  图表已保存: {filepath}")
        
        plt.close()
        return fig
    
    def create_all_visualizations(self, feature_importance_df=None):
        print("="*50)
        print("开始生成所有可视化图表")
        print("="*50)
        
        self.load_data()
        self.plot_category_completion_rate()
        self.plot_duration_vs_score_scatter()
        self.plot_feature_importance(feature_importance_df)
        
        print("\n所有图表生成完成！")
        print(f"图表保存位置: {self.output_dir}")
        
        return self.output_dir


if __name__ == '__main__':
    visualizer = Visualizer()
    visualizer.create_all_visualizations()
