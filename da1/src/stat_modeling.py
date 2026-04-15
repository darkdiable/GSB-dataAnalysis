import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from .config import OUTPUT_DIR, DPI, FIGSIZE

def plot_correlation_heatmap(df, interactive=False):
    numeric_df = df[['production', 'sales', 'year']].copy()
    corr = numeric_df.corr()
    
    if interactive:
        fig = px.imshow(corr, text_auto=True, aspect='auto',
                       title='相关性热力图', color_continuous_scale='RdBu_r')
        fig.write_html(f'{OUTPUT_DIR}/stat_correlation_interactive.html')
    else:
        plt.figure(figsize=FIGSIZE)
        sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0,
                   square=True, fmt='.3f', cbar_kws={'shrink': 0.8})
        plt.title('变量相关性热力图', fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/stat_correlation.png', dpi=DPI)
        plt.close()
    
    return corr

def plot_lorenz_curve(df):
    country_sales = df.groupby('country')['sales'].sum().sort_values()
    sorted_sales = country_sales.values
    cum_sales = np.cumsum(sorted_sales)
    cum_sales = cum_sales / cum_sales[-1]
    cum_pop = np.arange(1, len(cum_sales) + 1) / len(cum_sales)
    
    gini = 2 * np.trapz(cum_pop - cum_sales, cum_pop)
    
    plt.figure(figsize=FIGSIZE)
    plt.plot(cum_pop, cum_sales, 'b-', linewidth=2, label='洛伦兹曲线')
    plt.plot([0, 1], [0, 1], 'r--', label='绝对平均线')
    plt.fill_between(cum_pop, cum_sales, cum_pop, alpha=0.3,
                    label=f'基尼系数 = {gini:.3f}')
    plt.title('销量分布洛伦兹曲线', fontsize=14, pad=20)
    plt.xlabel('累计国家比例', fontsize=12)
    plt.ylabel('累计销量比例', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/stat_lorenz.png', dpi=DPI)
    plt.close()
    
    return gini

def kmeans_country_clustering(df):
    country_features = df.groupby('country').agg({
        'sales': ['sum', 'mean', 'std'],
        'production': ['sum', 'mean']
    }).reset_index()
    country_features.columns = ['country', 'sales_sum', 'sales_mean', 'sales_std',
                               'prod_sum', 'prod_mean']
    country_features = country_features.fillna(0)
    
    X = country_features.drop('country', axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    country_features['cluster'] = clusters
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=FIGSIZE)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters,
                         cmap='viridis', s=200, alpha=0.7)
    
    for i, country in enumerate(country_features['country']):
        plt.annotate(country, (X_pca[i, 0], X_pca[i, 1]),
                    ha='center', va='center')
    
    plt.legend(*scatter.legend_elements(), title='聚类')
    plt.title('K-Means国家聚类结果 (PCA降维可视化)', fontsize=14, pad=20)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} 方差)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} 方差)', fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/stat_kmeans.png', dpi=DPI)
    plt.close()
    
    return country_features, kmeans

def pca_visualization(df, interactive=False):
    country_features = df.groupby('country').agg({
        'sales': ['sum', 'mean', 'std'],
        'production': ['sum', 'mean']
    }).reset_index()
    country_features.columns = ['country', 'sales_sum', 'sales_mean', 'sales_std',
                               'prod_sum', 'prod_mean']
    country_features = country_features.fillna(0)
    
    X = country_features.drop('country', axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    
    pca_df = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'PC3': X_pca[:, 2],
        'country': country_features['country']
    })
    
    var_ratio = pca.explained_variance_ratio_
    
    if interactive:
        fig = px.scatter_3d(pca_df, x='PC1', y='PC2', z='PC3',
                           text='country', title='PCA 3D降维可视化',
                           color='PC1', color_continuous_scale='viridis')
        fig.update_traces(textposition='top center')
        fig.write_html(f'{OUTPUT_DIR}/stat_pca_3d_interactive.html')
    else:
        plt.figure(figsize=FIGSIZE)
        plt.scatter(X_pca[:, 0], X_pca[:, 1], s=100, alpha=0.7)
        for i, country in enumerate(country_features['country']):
            plt.annotate(country, (X_pca[i, 0], X_pca[i, 1]))
        
        plt.title(f'PCA 2D降维可视化\n累计方差解释率: {sum(var_ratio[:2]):.1%}',
                 fontsize=14, pad=20)
        plt.xlabel(f'PC1 ({var_ratio[0]:.1%})', fontsize=12)
        plt.ylabel(f'PC2 ({var_ratio[1]:.1%})', fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/stat_pca_2d.png', dpi=DPI)
        plt.close()
    
    return pca, var_ratio

def run_statistical_modeling(df):
    print("=== 开始统计建模分析 ===")
    
    corr = plot_correlation_heatmap(df)
    plot_correlation_heatmap(df, interactive=True)
    print("相关性热力图已生成")
    
    gini = plot_lorenz_curve(df)
    print(f"基尼系数: {gini:.3f}")
    print("洛伦兹曲线图已生成")
    
    clusters, kmeans = kmeans_country_clustering(df)
    print("K-Means国家聚类已完成")
    print("聚类结果:")
    print(clusters[['country', 'cluster']])
    
    pca, var_ratio = pca_visualization(df)
    pca_visualization(df, interactive=True)
    print(f"PCA累计方差解释率: {sum(var_ratio[:2]):.1%}")
    print("PCA可视化已完成")
    
    print("=== 统计建模分析完成 ===")
    return {
        'correlation': corr,
        'gini_coefficient': gini,
        'country_clusters': clusters,
        'pca_results': {'pca': pca, 'variance_ratio': var_ratio}
    }
