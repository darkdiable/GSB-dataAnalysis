import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False
rcParams['figure.dpi'] = 150
rcParams['savefig.dpi'] = 300
rcParams['figure.figsize'] = (10, 6)


class Visualizer:
    def __init__(self, output_dir='../output/figures'):
        self.output_dir = output_dir
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    
    def plot_conversion_funnel(self, df):
        print("\n=== 绘制用户行为转化漏斗图 ===")
        
        behavior_counts = df['behavior_type'].value_counts().reindex(['pv', 'fav', 'cart', 'buy'])
        behavior_names = ['浏览', '收藏', '加购', '购买']
        values = behavior_counts.values
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        y_pos = np.arange(len(behavior_names))
        max_width = values[0]
        
        for i, (name, value) in enumerate(zip(behavior_names, values)):
            width = value / max_width * 0.8
            rect = plt.Rectangle(
                ((0.8 - width) / 2, i - 0.35), 
                width, 0.7,
                color=self.colors[i], alpha=0.8, edgecolor='white', linewidth=2
            )
            ax.add_patch(rect)
            
            conversion_rate = value / values[0] * 100
            ax.text(
                0.4, i,
                f'{name}\n{value:,}  ({conversion_rate:.1f}%)',
                ha='center', va='center', fontsize=12, fontweight='bold', color='white'
            )
        
        ax.set_xlim(0, 0.8)
        ax.set_ylim(-0.5, len(behavior_names) - 0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([])
        ax.set_xticks([])
        
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        plt.title('电商用户行为转化漏斗', fontsize=16, pad=20, fontweight='bold')
        
        for i in range(len(values) - 1):
            rate = values[i + 1] / values[i] * 100
            ax.annotate(
                f'',
                xy=(0.4, i + 0.35), xytext=(0.4, i + 0.65),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2)
            )
            ax.text(0.62, i + 0.5, f'步骤转化: {rate:.1f}%', ha='center', va='center', 
                    fontsize=10, bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3))
        
        plt.tight_layout()
        filepath = f'{self.output_dir}/conversion_funnel.png'
        plt.savefig(filepath, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"漏斗图已保存: {filepath}")
    
    def plot_rfm_scatter(self, rfm_df):
        print("\n=== 绘制RFM用户价值聚类散点图 ===")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        segment_colors = {
            '重要价值用户': '#FF6B6B',
            '新用户': '#4ECDC4',
            '重要召回用户': '#45B7D1',
            '流失用户': '#95a5a6',
            '一般用户': '#BDC3C7'
        }
        
        ax1 = axes[0]
        for segment, color in segment_colors.items():
            segment_data = rfm_df[rfm_df['User_Segment'] == segment]
            ax1.scatter(segment_data['Recency'], segment_data['Frequency'], 
                       c=color, label=segment, alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
        
        ax1.set_xlabel('最近购买间隔 (天)', fontsize=12)
        ax1.set_ylabel('购买频率', fontsize=12)
        ax1.set_title('R-F 散点图 - 用户分群', fontsize=14, pad=15, fontweight='bold')
        ax1.legend(fontsize=9, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[1]
        for segment, color in segment_colors.items():
            segment_data = rfm_df[rfm_df['User_Segment'] == segment]
            ax2.scatter(segment_data['Frequency'], segment_data['Monetary'], 
                       c=color, label=segment, alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
        
        ax2.set_xlabel('购买频率', fontsize=12)
        ax2.set_ylabel('消费金额 (估算)', fontsize=12)
        ax2.set_title('F-M 散点图 - 用户分群', fontsize=14, pad=15, fontweight='bold')
        ax2.legend(fontsize=9, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('RFM用户价值聚类分析', fontsize=18, y=1.02, fontweight='bold')
        plt.tight_layout()
        
        filepath = f'{self.output_dir}/rfm_scatter.png'
        plt.savefig(filepath, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"RFM散点图已保存: {filepath}")
    
    def plot_repurchase_trend(self, df):
        print("\n=== 绘制复购率趋势图 ===")
        
        df = df.copy()
        df['week'] = df['timestamp'].dt.isocalendar().week
        
        purchase_df = df[df['behavior_type'] == 'buy'].copy()
        user_week_purchases = purchase_df.groupby(['user_id', 'week']).size().reset_index(name='purchase_count')
        user_total_purchases = purchase_df.groupby('user_id').size()
        
        weekly_data = []
        for week in sorted(df['week'].unique()):
            week_users = user_week_purchases[user_week_purchases['week'] == week]['user_id'].unique()
            repurchase_count = 0
            for user in week_users:
                if user_total_purchases.get(user, 0) >= 2:
                    repurchase_count += 1
            
            if len(week_users) > 0:
                repurchase_rate = repurchase_count / len(week_users)
                weekly_data.append({
                    'week': week,
                    'repurchase_rate': repurchase_rate,
                    'total_users': len(week_users)
                })
        
        trend_df = pd.DataFrame(weekly_data)
        
        fig, ax1 = plt.subplots(figsize=(12, 7))
        
        color1 = self.colors[0]
        ax1.set_xlabel('周数', fontsize=12)
        ax1.set_ylabel('复购率', fontsize=12, color=color1)
        line = ax1.plot(trend_df['week'], trend_df['repurchase_rate'], color=color1, 
                       linewidth=3, marker='o', markersize=8, label='复购率')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, alpha=0.3)
        
        for i, row in trend_df.iterrows():
            ax1.annotate(f"{row['repurchase_rate']:.1%}", 
                        (row['week'], row['repurchase_rate']),
                        textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)
        
        ax2 = ax1.twinx()
        color2 = self.colors[1]
        ax2.set_ylabel('购买用户数', fontsize=12, color=color2)
        bar = ax2.bar(trend_df['week'], trend_df['total_users'], color=color2, 
                      alpha=0.4, label='购买用户数')
        ax2.tick_params(axis='y', labelcolor=color2)
        
        plt.title('周度复购率趋势', fontsize=16, pad=20, fontweight='bold')
        
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        filepath = f'{self.output_dir}/repurchase_trend.png'
        plt.savefig(filepath, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"复购率趋势图已保存: {filepath}")
    
    def plot_feature_importance(self, feature_importance):
        print("\n=== 绘制特征重要性图 ===")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        feature_importance = feature_importance.sort_values('importance', ascending=True)
        
        bars = ax.barh(feature_importance['feature'], feature_importance['importance'], 
                       color=self.colors[:len(feature_importance)])
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.005, bar.get_y() + bar.get_height()/2,
                   f'{feature_importance.iloc[i]["importance"]:.4f}',
                   va='center', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('重要性得分', fontsize=12)
        ax.set_title('复购预测模型特征重要性', fontsize=16, pad=20, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        filepath = f'{self.output_dir}/feature_importance.png'
        plt.savefig(filepath, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"特征重要性图已保存: {filepath}")
    
    def generate_all_charts(self, df, rfm_df, feature_importance=None):
        self.plot_conversion_funnel(df)
        self.plot_rfm_scatter(rfm_df)
        self.plot_repurchase_trend(df)
        if feature_importance is not None:
            self.plot_feature_importance(feature_importance)
        print("\n=== 所有图表生成完成 ===")


if __name__ == '__main__':
    from data_preprocessor import DataPreprocessor
    preprocessor = DataPreprocessor()
    df, rfm_df = preprocessor.preprocess('../data/ecommerce_behavior.csv')
    
    visualizer = Visualizer()
    visualizer.generate_all_charts(df, rfm_df)
