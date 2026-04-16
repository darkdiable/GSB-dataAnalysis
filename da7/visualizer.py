import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.rcParams['font.sans-serif'] = ['PingFang SC', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class Visualizer:
    def __init__(self, output_dir='plots', dpi=300):
        self.output_dir = output_dir
        self.dpi = dpi
        os.makedirs(output_dir, exist_ok=True)
        
    def plot_category_completion_rate(self, df, figsize=(12, 8)):
        print("\n生成各类别课程的平均完成率柱状图...")
        
        category_completion = df.groupby('课程类别')['是否完成课程'].agg([
            ('平均完成率', 'mean'),
            ('样本数', 'count')
        ]).sort_values('平均完成率', ascending=True).reset_index()
        
        plt.figure(figsize=figsize)
        ax = sns.barplot(data=category_completion, x='平均完成率', y='课程类别', 
                        palette='viridis', edgecolor='black')
        
        for i, v in enumerate(category_completion['平均完成率']):
            ax.text(v + 0.01, i, f'{v:.1%}', va='center', fontweight='bold', fontsize=10)
        
        plt.title('各类别课程的平均完成率', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('平均完成率', fontsize=12, labelpad=10)
        plt.ylabel('课程类别', fontsize=12, labelpad=10)
        plt.xlim(0, 1)
        plt.grid(axis='x', alpha=0.3, linestyle='--')
        
        for i, (n, v) in enumerate(zip(category_completion['样本数'], category_completion['平均完成率'])):
            ax.text(v/2, i, f'n={n}', ha='center', va='center', color='white', fontsize=8)
        
        plt.tight_layout()
        
        filepath = f'{self.output_dir}/category_completion_rate.png'
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"图表已保存至: {filepath}")
        
    def plot_duration_vs_score(self, df, figsize=(12, 8)):
        print("\n生成观看时长与测验分数的关系散点图...")
        
        plt.figure(figsize=figsize)
        
        scatter = plt.scatter(df['观看时长(分钟)'], df['测验分数'], 
                            c=df['是否完成课程'], cmap='coolwarm', 
                            alpha=0.6, s=50, edgecolors='white', linewidth=0.5)
        
        sns.regplot(data=df, x='观看时长(分钟)', y='测验分数', 
                   scatter=False, color='black', line_kws={'linestyle':'--', 'alpha':0.7})
        
        legend = plt.legend(*scatter.legend_elements(), 
                          title='是否完成课程',
                          loc='lower right')
        legend.get_texts()[0].set_text('未完成')
        legend.get_texts()[1].set_text('完成')
        
        plt.title('观看时长与测验分数的关系', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('观看时长 (分钟)', fontsize=12, labelpad=10)
        plt.ylabel('测验分数', fontsize=12, labelpad=10)
        plt.grid(alpha=0.3, linestyle='--')
        
        corr = df['观看时长(分钟)'].corr(df['测验分数'])
        plt.annotate(f'相关系数: {corr:.3f}', 
                    xy=(0.05, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.5),
                    fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        filepath = f'{self.output_dir}/duration_vs_score_scatter.png'
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"图表已保存至: {filepath}")
        
    def plot_feature_importance(self, feature_importance, top_n=10, figsize=(12, 8)):
        print("\n生成模型特征重要性条形图...")
        
        top_features = feature_importance.head(top_n).sort_values('重要性', ascending=True)
        
        plt.figure(figsize=figsize)
        bars = plt.barh(top_features['特征'], top_features['重要性'], 
                       color=sns.color_palette('magma', top_n),
                       edgecolor='black')
        
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{width:.4f}', va='center', fontweight='bold', fontsize=9)
        
        plt.title(f'特征重要性排序 (Top {top_n})', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('重要性得分', fontsize=12, labelpad=10)
        plt.ylabel('特征名称', fontsize=12, labelpad=10)
        plt.grid(axis='x', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        filepath = f'{self.output_dir}/feature_importance.png'
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"图表已保存至: {filepath}")
        
    def generate_all_plots(self, df, feature_importance):
        print("\n" + "="*50)
        print("开始生成所有可视化图表...")
        print("="*50)
        
        self.plot_category_completion_rate(df)
        self.plot_duration_vs_score(df)
        self.plot_feature_importance(feature_importance)
        
        print("="*50)
        print("所有可视化图表生成完成!")
        print("="*50)
