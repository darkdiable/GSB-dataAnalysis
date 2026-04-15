"""
统计建模分析模块
功能：相关性热力图、洛伦兹曲线、K-Means国家聚类、PCA降维可视化
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class StatisticalModelingAnalyzer:
    """统计建模分析类"""

    def __init__(self, df, output_dir='../output'):
        """
        初始化
        Args:
            df: pandas DataFrame
            output_dir: 输出目录
        """
        self.df = df.copy()
        self.output_dir = output_dir
        self.recent_years = [2023, 2024, 2025]

    def plot_correlation_heatmap(self, save_path=None, dpi=300):
        """
        绘制相关性热力图
        Args:
            save_path: 保存路径
            dpi: 图像分辨率
        """
        numeric_cols = ['Horsepower', 'Torque_Nm', 'Fuel_Efficiency_L_100km',
                       'Price_USD', 'Units_Sold', 'Weight_kg',
                       'Acceleration_0_100_s', 'CO2_Emissions_g_km']

        corr_matrix = self.df[numeric_cols].corr()

        fig, ax = plt.subplots(figsize=(12, 10))

        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                   center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                   annot_kws={'size': 10}, ax=ax)

        ax.set_title('性能车指标相关性热力图', fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
            print(f"相关性热力图已保存至: {save_path}")

        print("\n强相关性指标对 (|r| > 0.5):")
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:
                    print(f"  {corr_matrix.columns[i]} - {corr_matrix.columns[j]}: {corr_val:.3f}")

        return fig, corr_matrix

    def plot_lorenz_curve(self, save_path=None, dpi=300):
        """
        绘制洛伦兹曲线（市场集中度分析）
        Args:
            save_path: 保存路径
            dpi: 图像分辨率
        """
        recent_data = self.df[self.df['Year'].isin(self.recent_years)]

        brand_sales = recent_data.groupby('Brand')['Units_Sold'].sum().sort_values()

        cumsum_sales = np.cumsum(brand_sales)
        cumsum_pct = cumsum_sales / cumsum_sales.iloc[-1] * 100

        n_brands = len(brand_sales)
        population_pct = np.arange(1, n_brands + 1) / n_brands * 100

        gini = self._calculate_gini(brand_sales.values)

        fig, ax = plt.subplots(figsize=(10, 8))

        ax.plot([0, 100], [0, 100], 'k--', linewidth=1, label='绝对平等线')
        ax.plot(population_pct, cumsum_pct, 'b-', linewidth=2.5, label='洛伦兹曲线')
        ax.fill_between(population_pct, cumsum_pct, population_pct, alpha=0.3, color='red')

        ax.set_xlabel('品牌累计占比 (%)', fontsize=12)
        ax.set_ylabel('销量累计占比 (%)', fontsize=12)
        ax.set_title(f'品牌销量洛伦兹曲线 (基尼系数: {gini:.3f})', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)

        for pct in [20, 40, 60, 80]:
            idx = int(n_brands * pct / 100) - 1
            if idx >= 0:
                ax.annotate(f'{pct}%品牌占{cumsum_pct.iloc[idx]:.1f}%',
                           xy=(population_pct[idx], cumsum_pct.iloc[idx]),
                           xytext=(population_pct[idx]+5, cumsum_pct.iloc[idx]-10),
                           fontsize=9,
                           arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
            print(f"洛伦兹曲线已保存至: {save_path}")

        print(f"\n洛伦兹曲线分析:")
        print(f"  基尼系数: {gini:.3f}")
        print(f"  Top 20%品牌销量占比: {cumsum_pct.iloc[int(n_brands*0.2)-1]:.1f}%")
        print(f"  Top 50%品牌销量占比: {cumsum_pct.iloc[int(n_brands*0.5)-1]:.1f}%")

        return fig, gini

    def _calculate_gini(self, values):
        """计算基尼系数"""
        values = np.array(values)
        values = np.sort(values)
        n = len(values)
        cumsum = np.cumsum(values)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n

    def kmeans_country_clustering(self, save_path=None, dpi=300):
        """
        K-Means国家聚类分析
        Args:
            save_path: 保存路径
            dpi: 图像分辨率
        """
        recent_data = self.df[self.df['Year'].isin(self.recent_years)]

        country_features = recent_data.groupby('Country').agg({
            'Units_Sold': 'sum',
            'Price_USD': 'mean',
            'Horsepower': 'mean',
            'Market_Share_Percent': 'sum'
        }).reset_index()

        feature_cols = ['Units_Sold', 'Price_USD', 'Horsepower', 'Market_Share_Percent']
        X = country_features[feature_cols].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        inertias = []
        silhouettes = []
        K_range = range(2, min(6, len(country_features)))

        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            inertias.append(kmeans.inertia_)
            silhouettes.append(silhouette_score(X_scaled, labels))

        optimal_k = list(K_range)[np.argmax(silhouettes)]

        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        country_features['Cluster'] = kmeans.fit_predict(X_scaled)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        ax1 = axes[0, 0]
        ax1.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('聚类数 (K)', fontsize=11)
        ax1.set_ylabel('惯性 (Inertia)', fontsize=11)
        ax1.set_title('肘部法则确定最优K值', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.axvline(x=optimal_k, color='r', linestyle='--', label=f'最优K={optimal_k}')
        ax1.legend()

        ax2 = axes[0, 1]
        ax2.plot(K_range, silhouettes, 'go-', linewidth=2, markersize=8)
        ax2.set_xlabel('聚类数 (K)', fontsize=11)
        ax2.set_ylabel('轮廓系数', fontsize=11)
        ax2.set_title('轮廓系数确定最优K值', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.axvline(x=optimal_k, color='r', linestyle='--', label=f'最优K={optimal_k}')
        ax2.legend()

        ax3 = axes[1, 0]
        colors = plt.cm.Set1(np.linspace(0, 1, optimal_k))
        for i in range(optimal_k):
            cluster_data = country_features[country_features['Cluster'] == i]
            ax3.scatter(cluster_data['Units_Sold'], cluster_data['Price_USD'],
                       c=[colors[i]], s=200, alpha=0.7, edgecolors='black',
                       label=f'聚类 {i+1}')
            for _, row in cluster_data.iterrows():
                ax3.annotate(row['Country'], (row['Units_Sold'], row['Price_USD']),
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
        ax3.set_xlabel('销量（辆）', fontsize=11)
        ax3.set_ylabel('平均价格（USD）', fontsize=11)
        ax3.set_title('K-Means聚类结果 (销量 vs 价格)', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        ax4 = axes[1, 1]
        cluster_summary = country_features.groupby('Cluster')[feature_cols].mean()
        im = ax4.imshow(cluster_summary.T.values, cmap='RdYlGn', aspect='auto')
        ax4.set_xticks(range(optimal_k))
        ax4.set_xticklabels([f'聚类 {i+1}' for i in range(optimal_k)])
        ax4.set_yticks(range(len(feature_cols)))
        ax4.set_yticklabels(feature_cols)
        ax4.set_title('聚类特征热力图', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax4)

        for i in range(optimal_k):
            for j in range(len(feature_cols)):
                text = ax4.text(i, j, f'{cluster_summary.iloc[i, j]:.1f}',
                              ha="center", va="center", color="black", fontsize=9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
            print(f"K-Means聚类图已保存至: {save_path}")

        print(f"\nK-Means聚类分析 (K={optimal_k}):")
        for i in range(optimal_k):
            cluster_countries = country_features[country_features['Cluster'] == i]['Country'].tolist()
            print(f"  聚类 {i+1}: {', '.join(cluster_countries)}")

        return fig, country_features

    def pca_analysis(self, save_path=None, dpi=300):
        """
        PCA降维可视化
        Args:
            save_path: 保存路径
            dpi: 图像分辨率
        """
        recent_data = self.df[self.df['Year'].isin(self.recent_years)]

        numeric_cols = ['Horsepower', 'Torque_Nm', 'Fuel_Efficiency_L_100km',
                       'Price_USD', 'Units_Sold', 'Weight_kg',
                       'Acceleration_0_100_s', 'CO2_Emissions_g_km']

        brand_features = recent_data.groupby('Brand')[numeric_cols].mean()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(brand_features.values)

        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)

        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)

        n_components_90 = np.argmax(cumulative_variance >= 0.9) + 1

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        ax1 = axes[0, 0]
        ax1.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio,
               alpha=0.7, color='steelblue', edgecolor='black')
        ax1.set_xlabel('主成分', fontsize=11)
        ax1.set_ylabel('解释方差比例', fontsize=11)
        ax1.set_title('各主成分解释方差比例', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')

        ax2 = axes[0, 1]
        ax2.plot(range(1, len(cumulative_variance) + 1), cumulative_variance,
                'ro-', linewidth=2, markersize=8)
        ax2.axhline(y=0.9, color='g', linestyle='--', label='90%阈值')
        ax2.axvline(x=n_components_90, color='r', linestyle='--', label=f'{n_components_90}个主成分')
        ax2.set_xlabel('主成分数量', fontsize=11)
        ax2.set_ylabel('累计解释方差比例', fontsize=11)
        ax2.set_title('累计解释方差比例', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        ax3 = axes[1, 0]
        scatter = ax3.scatter(X_pca[:, 0], X_pca[:, 1], c=brand_features['Units_Sold'],
                             s=200, alpha=0.7, cmap='viridis', edgecolors='black')
        for i, brand in enumerate(brand_features.index):
            ax3.annotate(brand, (X_pca[i, 0], X_pca[i, 1]),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        ax3.set_xlabel(f'PC1 ({explained_variance_ratio[0]*100:.1f}%)', fontsize=11)
        ax3.set_ylabel(f'PC2 ({explained_variance_ratio[1]*100:.1f}%)', fontsize=11)
        ax3.set_title('PCA降维散点图 (PC1 vs PC2)', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax3, label='销量')

        ax4 = axes[1, 1]
        components_df = pd.DataFrame(
            pca.components_[:2].T,
            columns=['PC1', 'PC2'],
            index=numeric_cols
        )
        sns.heatmap(components_df, annot=True, fmt='.2f', cmap='RdBu_r',
                   center=0, ax=ax4, cbar_kws={"shrink": 0.8})
        ax4.set_title('主成分载荷矩阵', fontsize=12, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
            print(f"PCA分析图已保存至: {save_path}")

        print(f"\nPCA分析结果:")
        print(f"  前2个主成分解释方差: {(explained_variance_ratio[0] + explained_variance_ratio[1])*100:.1f}%")
        print(f"  达到90%方差需要主成分数: {n_components_90}")
        print(f"  PC1主要载荷: ")
        pc1_loadings = pd.Series(pca.components_[0], index=numeric_cols).abs().sort_values(ascending=False)
        for var, loading in pc1_loadings.head(3).items():
            print(f"    {var}: {loading:.3f}")

        return fig, pca


def run_statistical_modeling(df, output_dir='../output'):
    """
    运行完整的统计建模分析
    Args:
        df: DataFrame
        output_dir: 输出目录
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    analyzer = StatisticalModelingAnalyzer(df, output_dir)

    print("=" * 60)
    print("统计建模分析")
    print("=" * 60)

    analyzer.plot_correlation_heatmap(save_path=f'{output_dir}/13_correlation_heatmap.png')

    analyzer.plot_lorenz_curve(save_path=f'{output_dir}/14_lorenz_curve.png')

    analyzer.kmeans_country_clustering(save_path=f'{output_dir}/15_kmeans_clustering.png')

    analyzer.pca_analysis(save_path=f'{output_dir}/16_pca_analysis.png')

    print("\n统计建模分析完成！")

    return analyzer


if __name__ == '__main__':
    df = pd.read_csv('../data/fuel_performance_cars.csv')
    run_statistical_modeling(df)
