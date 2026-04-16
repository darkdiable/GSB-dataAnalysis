import pandas as pd
import numpy as np
from scipy import stats


class FeatureAnalyzer:
    def __init__(self, data):
        self.data = data.copy()
        self.analysis_results = {}

    def analyze_temperature_seasonality(self):
        """
        分析温度的季节性特征
        """
        results = {}
        
        # 按月统计
        monthly_stats = self.data.groupby('month')['temperature'].agg([
            'mean', 'std', 'min', 'max'
        ]).round(2)
        results['monthly'] = monthly_stats
        
        # 按季节统计
        seasonal_stats = self.data.groupby('season')['temperature'].agg([
            'mean', 'std', 'min', 'max'
        ]).round(2)
        results['seasonal'] = seasonal_stats
        
        # 年度趋势
        yearly_avg = self.data.groupby('year')['temperature'].mean().round(2)
        results['yearly_trend'] = yearly_avg
        
        # 计算季节性指数
        overall_mean = self.data['temperature'].mean()
        seasonal_index = monthly_stats['mean'] / overall_mean
        results['seasonal_index'] = seasonal_index.round(4)
        
        self.analysis_results['temperature_seasonality'] = results
        return results

    def analyze_correlations(self):
        """
        分析气象要素间的相关性
        """
        numeric_columns = ['temperature', 'humidity', 'precipitation', 'wind_speed', 'pressure']
        
        # 计算相关系数矩阵
        corr_matrix = self.data[numeric_columns].corr().round(4)
        
        # 特定相关性分析
        specific_corrs = {
            'temperature_humidity': {
                'pearson': round(self.data['temperature'].corr(self.data['humidity']), 4),
                'spearman': round(stats.spearmanr(self.data['temperature'], self.data['humidity'])[0], 4)
            },
            'humidity_precipitation': {
                'pearson': round(self.data['humidity'].corr(self.data['precipitation']), 4),
                'spearman': round(stats.spearmanr(self.data['humidity'], self.data['precipitation'])[0], 4)
            },
            'temperature_precipitation': {
                'pearson': round(self.data['temperature'].corr(self.data['precipitation']), 4),
                'spearman': round(stats.spearmanr(self.data['temperature'], self.data['precipitation'])[0], 4)
            }
        }
        
        results = {
            'correlation_matrix': corr_matrix,
            'specific_correlations': specific_corrs
        }
        
        self.analysis_results['correlations'] = results
        return results

    def analyze_extreme_weather(self):
        """
        分析极端天气事件
        """
        results = {}
        
        # 极端高温 (超过95百分位数)
        temp_high_threshold = self.data['temperature'].quantile(0.95)
        extreme_heat = self.data[self.data['temperature'] > temp_high_threshold]
        results['extreme_heat'] = {
            'threshold': round(temp_high_threshold, 2),
            'count': len(extreme_heat),
            'dates': extreme_heat['date'].tolist()[:10]  # 前10个日期
        }
        
        # 极端低温 (低于5百分位数)
        temp_low_threshold = self.data['temperature'].quantile(0.05)
        extreme_cold = self.data[self.data['temperature'] < temp_low_threshold]
        results['extreme_cold'] = {
            'threshold': round(temp_low_threshold, 2),
            'count': len(extreme_cold),
            'dates': extreme_cold['date'].tolist()[:10]
        }
        
        # 暴雨事件 (超过95百分位数)
        rain_threshold = self.data['precipitation'].quantile(0.95)
        heavy_rain = self.data[self.data['precipitation'] > rain_threshold]
        results['heavy_rain'] = {
            'threshold': round(rain_threshold, 2),
            'count': len(heavy_rain),
            'dates': heavy_rain['date'].tolist()[:10]
        }
        
        self.analysis_results['extreme_weather'] = results
        return results

    def analyze_trends(self):
        """
        分析长期趋势
        """
        results = {}
        
        # 按年统计各项指标
        yearly_data = self.data.groupby('year').agg({
            'temperature': ['mean', 'max', 'min'],
            'precipitation': 'sum',
            'humidity': 'mean'
        }).round(2)
        
        results['yearly_summary'] = yearly_data
        
        # 计算线性趋势
        years = yearly_data.index.values
        for col in ['temperature', 'precipitation', 'humidity']:
            if col == 'precipitation':
                values = yearly_data[(col, 'sum')].values
            else:
                values = yearly_data[(col, 'mean')].values
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(years, values)
            results[f'{col}_trend'] = {
                'slope': round(slope, 6),
                'r_squared': round(r_value**2, 4),
                'p_value': round(p_value, 4),
                'trend': '上升' if slope > 0 else '下降'
            }
        
        self.analysis_results['trends'] = results
        return results

    def get_all_analysis(self):
        """
        获取所有分析结果
        """
        return self.analysis_results

    def generate_analysis_summary(self):
        """
        生成分析摘要文本
        """
        summary = []
        summary.append("=" * 60)
        summary.append("气象特征分析摘要")
        summary.append("=" * 60)
        
        # 温度季节性
        if 'temperature_seasonality' in self.analysis_results:
            ts = self.analysis_results['temperature_seasonality']
            summary.append("\n【温度季节性分析】")
            summary.append(f"  最高月均温: {ts['monthly']['max'].max()}°C ({ts['monthly']['max'].idxmax()}月)")
            summary.append(f"  最低月均温: {ts['monthly']['min'].min()}°C ({ts['monthly']['min'].idxmin()}月)")
            summary.append(f"  年温差: {ts['monthly']['mean'].max() - ts['monthly']['mean'].min():.2f}°C")
        
        # 相关性
        if 'correlations' in self.analysis_results:
            corr = self.analysis_results['correlations']
            summary.append("\n【气象要素相关性】")
            summary.append(f"  温度-湿度相关系数: {corr['specific_correlations']['temperature_humidity']['pearson']}")
            summary.append(f"  湿度-降水相关系数: {corr['specific_correlations']['humidity_precipitation']['pearson']}")
        
        # 极端天气
        if 'extreme_weather' in self.analysis_results:
            ew = self.analysis_results['extreme_weather']
            summary.append("\n【极端天气统计】")
            summary.append(f"  极端高温事件: {ew['extreme_heat']['count']} 次 (阈值: {ew['extreme_heat']['threshold']}°C)")
            summary.append(f"  极端低温事件: {ew['extreme_cold']['count']} 次 (阈值: {ew['extreme_cold']['threshold']}°C)")
            summary.append(f"  暴雨事件: {ew['heavy_rain']['count']} 次 (阈值: {ew['heavy_rain']['threshold']}mm)")
        
        # 趋势
        if 'trends' in self.analysis_results:
            trends = self.analysis_results['trends']
            summary.append("\n【长期趋势分析】")
            for var in ['temperature', 'precipitation', 'humidity']:
                trend_key = f'{var}_trend'
                if trend_key in trends:
                    t = trends[trend_key]
                    summary.append(f"  {var}趋势: {t['trend']} (斜率: {t['slope']}, R²: {t['r_squared']})")
        
        summary.append("\n" + "=" * 60)
        
        return "\n".join(summary)
