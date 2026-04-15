import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class StatisticalModeling:
    def __init__(self, df):
        self.df = df.copy()
        self.scaler = StandardScaler()
        self.pca_result = None
        self.kmeans_result = None
        self.correlation_matrix = None
        
    def calculate_correlation(self):
        numeric_cols = ['Horsepower', 'Price_USD', 'Sales_Units', 'Production_Units',
                       'Fuel_Efficiency_L100km', 'CO2_Emissions_gkm', 
                       'Acceleration_0_100_s', 'Top_Speed_kmh']
        numeric_cols = [c for c in numeric_cols if c in self.df.columns]
        
        self.correlation_matrix = self.df[numeric_cols].corr()
        return self
    
    def plot_correlation_heatmap(self, save_path):
        if self.correlation_matrix is None:
            self.calculate_correlation()
            
        fig, ax = plt.subplots(figsize=(12, 10))
        
        mask = np.triu(np.ones_like(self.correlation_matrix, dtype=bool))
        
        sns.heatmap(self.correlation_matrix, mask=mask, annot=True, fmt='.2f',
                   cmap='RdYlBu_r', center=0, square=True, linewidths=0.5,
                   cbar_kws={'shrink': 0.8}, ax=ax)
        
        ax.set_title('特征相关性热力图', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"相关性热力图已保存: {save_path}")
        
    def plot_lorenz_curve(self, save_path):
        country_sales = self.df.groupby('Country')['Sales_Units'].sum().sort_values()
        total = country_sales.sum()
        
        cumsum = country_sales.cumsum()
        cumsum_pct = cumsum / total * 100
        country_pct = np.arange(1, len(country_sales) + 1) / len(country_sales) * 100
        
        perfect_equality_x = np.linspace(0, 100, len(country_sales) + 1)
        perfect_equality_y = np.linspace(0, 100, len(country_sales) + 1)
        
        area_under_lorenz = np.trapz([0] + list(cumsum_pct), [0] + list(country_pct))
        area_under_equality = 0.5 * 100 * 100
        gini = (area_under_equality - area_under_lorenz) / area_under_equality
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        ax.plot([0] + list(country_pct), [0] + list(cumsum_pct), 'b-', linewidth=2, label='洛伦兹曲线')
        ax.plot(perfect_equality_x, perfect_equality_y, 'r--', linewidth=2, label='完全平等线')
        
        ax.fill_between([0] + list(country_pct), [0] + list(cumsum_pct), 
                       perfect_equality_y[:len(country_pct)+1], alpha=0.3, color='blue')
        
        ax.set_xlabel('国家累计百分比 (%)', fontsize=12)
        ax.set_ylabel('销量累计百分比 (%)', fontsize=12)
        ax.set_title(f'洛伦兹曲线 - 市场集中度分析\n基尼系数: {gini:.3f}', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"洛伦兹曲线已保存: {save_path}")
        
        return gini
    
    def kmeans_clustering(self, n_clusters=4):
        country_features = self.df.groupby('Country').agg({
            'Sales_Units': 'sum',
            'Price_USD': 'mean',
            'Horsepower': 'mean',
            'Growth_Rate_Percent': 'mean',
            'Market_Share_Percent': 'mean'
        }).reset_index()
        
        feature_cols = ['Sales_Units', 'Price_USD', 'Horsepower', 'Growth_Rate_Percent', 'Market_Share_Percent']
        X = country_features[feature_cols].values
        X_scaled = self.scaler.fit_transform(X)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        country_features['Cluster'] = kmeans.fit_predict(X_scaled)
        
        self.kmeans_result = {
            'data': country_features,
            'model': kmeans,
            'features_scaled': X_scaled
        }
        
        return country_features
    
    def plot_kmeans_clusters(self, save_path):
        if self.kmeans_result is None:
            self.kmeans_clustering()
            
        country_features = self.kmeans_result['data']
        X_scaled = self.kmeans_result['features_scaled']
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        country_features['PCA1'] = X_pca[:, 0]
        country_features['PCA2'] = X_pca[:, 1]
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        ax1 = axes[0]
        scatter = ax1.scatter(country_features['PCA1'], country_features['PCA2'],
                            c=country_features['Cluster'], cmap='viridis', 
                            s=100, alpha=0.7, edgecolors='black')
        
        for i, row in country_features.iterrows():
            ax1.annotate(row['Country'], (row['PCA1'], row['PCA2']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax1.set_xlabel('主成分1')
        ax1.set_ylabel('主成分2')
        ax1.set_title('K-Means聚类结果 (PCA降维可视化)', fontsize=14, fontweight='bold')
        plt.colorbar(scatter, ax=ax1, label='聚类')
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[1]
        cluster_stats = country_features.groupby('Cluster').agg({
            'Sales_Units': 'mean',
            'Price_USD': 'mean',
            'Growth_Rate_Percent': 'mean'
        }).reset_index()
        
        x = np.arange(len(cluster_stats))
        width = 0.25
        
        ax2.bar(x - width, cluster_stats['Sales_Units']/1000, width, label='平均销量(千)', color='steelblue')
        ax2.bar(x, cluster_stats['Price_USD']/1000, width, label='平均价格(千)', color='coral')
        ax2.bar(x + width, cluster_stats['Growth_Rate_Percent'], width, label='平均增长率(%)', color='green')
        
        ax2.set_xlabel('聚类')
        ax2.set_ylabel('数值')
        ax2.set_title('各聚类特征统计', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'Cluster {i}' for i in cluster_stats['Cluster']])
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"K-Means聚类图已保存: {save_path}")
        
    def pca_analysis(self, n_components=2):
        numeric_cols = ['Horsepower', 'Price_USD', 'Sales_Units', 'Production_Units',
                       'Fuel_Efficiency_L100km', 'CO2_Emissions_gkm', 
                       'Acceleration_0_100_s', 'Top_Speed_kmh']
        numeric_cols = [c for c in numeric_cols if c in self.df.columns]
        
        X = self.df[numeric_cols].dropna()
        X_scaled = self.scaler.fit_transform(X)
        
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        self.pca_result = {
            'pca': pca,
            'X_pca': X_pca,
            'explained_variance': pca.explained_variance_ratio_,
            'components': pca.components_,
            'feature_names': numeric_cols
        }
        
        return self.pca_result
    
    def plot_pca_visualization(self, save_path):
        if self.pca_result is None:
            self.pca_analysis()
            
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        ax1 = axes[0]
        explained_var = self.pca_result['explained_variance']
        cumsum_var = np.cumsum(explained_var)
        
        n_components = len(explained_var)
        ax1.bar(range(1, n_components + 1), explained_var, alpha=0.7, 
               label='解释方差', color='steelblue')
        ax1.plot(range(1, n_components + 1), cumsum_var, 'r-o', 
                label='累计解释方差')
        
        ax1.set_xlabel('主成分')
        ax1.set_ylabel('解释方差比例')
        ax1.set_title('PCA解释方差', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(range(1, n_components + 1))
        
        ax2 = axes[1]
        components = self.pca_result['components']
        feature_names = self.pca_result['feature_names']
        
        x = np.arange(len(feature_names))
        width = 0.35
        
        ax2.bar(x - width/2, components[0], width, label='PC1', color='steelblue')
        ax2.bar(x + width/2, components[1], width, label='PC2', color='coral')
        
        ax2.set_xlabel('特征')
        ax2.set_ylabel('载荷')
        ax2.set_title('主成分载荷图', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(feature_names, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"PCA可视化图已保存: {save_path}")
        
    def create_interactive_modeling(self, save_path):
        if self.correlation_matrix is None:
            self.calculate_correlation()
        if self.pca_result is None:
            self.pca_analysis()
            
        fig = make_subplots(rows=2, cols=2,
                           subplot_titles=('相关性矩阵', 'PCA散点图',
                                          '解释方差', '特征重要性'))
        
        fig.add_trace(go.Heatmap(z=self.correlation_matrix.values,
                                x=self.correlation_matrix.columns,
                                y=self.correlation_matrix.columns,
                                colorscale='RdBu',
                                zmid=0),
                     row=1, col=1)
        
        X_pca = self.pca_result['X_pca']
        fig.add_trace(go.Scatter(x=X_pca[:, 0], y=X_pca[:, 1],
                                mode='markers', marker=dict(size=5, opacity=0.5),
                                name='样本'),
                     row=1, col=2)
        
        explained_var = self.pca_result['explained_variance']
        fig.add_trace(go.Bar(x=[f'PC{i+1}' for i in range(len(explained_var))],
                            y=explained_var, name='解释方差'),
                     row=2, col=1)
        
        components = self.pca_result['components']
        feature_names = self.pca_result['feature_names']
        fig.add_trace(go.Bar(x=feature_names, y=np.abs(components[0]),
                            name='PC1载荷'),
                     row=2, col=2)
        
        fig.update_layout(height=800, width=1200,
                         title_text="统计建模分析",
                         showlegend=False)
        
        fig.write_html(save_path)
        print(f"交互式统计建模图已保存: {save_path}")
        
    def generate_report(self):
        if self.correlation_matrix is None:
            self.calculate_correlation()
        if self.pca_result is None:
            self.pca_analysis()
            
        report = "# 统计建模分析报告\n\n"
        
        report += "## 1. 相关性分析\n\n"
        report += "主要相关性发现：\n"
        
        corr_pairs = []
        cols = self.correlation_matrix.columns
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                corr_pairs.append((cols[i], cols[j], self.correlation_matrix.iloc[i, j]))
        
        corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        for feat1, feat2, corr in corr_pairs[:5]:
            report += f"- **{feat1}** 与 **{feat2}**: r = {corr:.3f}\n"
        report += "\n"
        
        report += "## 2. PCA分析\n\n"
        explained_var = self.pca_result['explained_variance']
        report += f"- PC1解释方差: {explained_var[0]*100:.2f}%\n"
        report += f"- PC2解释方差: {explained_var[1]*100:.2f}%\n"
        report += f"- 累计解释方差: {sum(explained_var)*100:.2f}%\n\n"
        
        report += "## 3. 聚类分析\n\n"
        if self.kmeans_result:
            cluster_counts = self.kmeans_result['data']['Cluster'].value_counts()
            for cluster, count in cluster_counts.items():
                report += f"- Cluster {cluster}: {count} 个国家\n"
        
        return report


if __name__ == "__main__":
    df = pd.read_csv("../data/fuel_performance_cars.csv")
    modeling = StatisticalModeling(df)
    modeling.calculate_correlation()
    modeling.plot_correlation_heatmap("../output/correlation_heatmap.png")
    modeling.plot_lorenz_curve("../output/lorenz_curve.png")
    modeling.kmeans_clustering()
    modeling.plot_kmeans_clusters("../output/kmeans_clusters.png")
    modeling.pca_analysis()
    modeling.plot_pca_visualization("../output/pca_visualization.png")
