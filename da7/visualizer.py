import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class Visualizer:
    def __init__(self, output_dir='figures', dpi=300):
        self.output_dir = output_dir
        self.dpi = dpi
        os.makedirs(output_dir, exist_ok=True)
        
    def plot_completion_rate_by_category(self, df, save_name='completion_rate_by_category.png'):
        completion_rate = df.groupby('course_category')['completed'].mean().sort_values(ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = sns.color_palette('viridis', len(completion_rate))
        bars = ax.bar(completion_rate.index, completion_rate.values, color=colors)
        
        ax.set_xlabel('课程类别', fontsize=12)
        ax.set_ylabel('平均完成率', fontsize=12)
        ax.set_title('各类别课程的平均完成率', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1)
        
        for bar, rate in zip(bars, completion_rate.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{rate:.1%}', ha='center', va='bottom', fontsize=10)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, save_name)
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"图表已保存: {filepath}")
        return filepath
    
    def plot_watch_duration_vs_quiz_score(self, df, save_name='watch_duration_vs_quiz_score.png'):
        fig, ax = plt.subplots(figsize=(10, 8))
        
        scatter = ax.scatter(
            df['watch_duration'],
            df['quiz_score'],
            c=df['completed'],
            cmap='coolwarm',
            alpha=0.6,
            s=30
        )
        
        cbar = plt.colorbar(scatter)
        cbar.set_label('是否完成课程 (0=未完成, 1=完成)', fontsize=10)
        
        ax.set_xlabel('观看时长 (分钟)', fontsize=12)
        ax.set_ylabel('测验分数', fontsize=12)
        ax.set_title('用户观看时长与测验分数的关系', fontsize=14, fontweight='bold')
        
        z = np.polyfit(df['watch_duration'], df['quiz_score'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df['watch_duration'].min(), df['watch_duration'].max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, label='趋势线')
        ax.legend()
        
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, save_name)
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"图表已保存: {filepath}")
        return filepath
    
    def plot_feature_importance(self, feature_importance, save_name='feature_importance.png'):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        feature_importance = feature_importance.sort_values('importance', ascending=True)
        
        colors = sns.color_palette('Blues_r', len(feature_importance))
        bars = ax.barh(feature_importance['feature'], feature_importance['importance'], color=colors)
        
        ax.set_xlabel('重要性', fontsize=12)
        ax.set_ylabel('特征', fontsize=12)
        ax.set_title('模型特征重要性排序', fontsize=14, fontweight='bold')
        
        for bar, importance in zip(bars, feature_importance['importance']):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{importance:.4f}', ha='left', va='center', fontsize=10)
        
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, save_name)
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"图表已保存: {filepath}")
        return filepath
    
    def plot_all(self, df, feature_importance=None):
        print("\n" + "=" * 50)
        print("开始生成可视化图表...")
        print("=" * 50)
        
        self.plot_completion_rate_by_category(df)
        self.plot_watch_duration_vs_quiz_score(df)
        
        if feature_importance is not None:
            self.plot_feature_importance(feature_importance)
        
        print("=" * 50)
        print("所有可视化图表生成完成!")
        print("=" * 50)


def create_visualizations(df, feature_importance=None, output_dir='figures'):
    visualizer = Visualizer(output_dir=output_dir)
    visualizer.plot_all(df, feature_importance)
    return visualizer


if __name__ == "__main__":
    import numpy as np
    from data_generator import generate_course_data
    from data_preprocessor import DataPreprocessor
    from model_trainer import train_completion_model
    
    data = generate_course_data(n_samples=1000)
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.preprocess(data)
    
    feature_columns = preprocessor.get_feature_columns()
    trainer = train_completion_model(processed_data, feature_columns)
    
    create_visualizations(processed_data, trainer.feature_importance)
