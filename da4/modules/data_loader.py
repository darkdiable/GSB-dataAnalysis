"""
数据加载模块 - 负责加载和生成零售销售数据
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class DataLoader:
    """数据加载器类，支持从CSV加载或生成模拟数据"""
    
    def __init__(self, file_path=None):
        """
        初始化数据加载器
        
        Parameters:
            file_path: 数据文件路径，若为None则生成模拟数据
        """
        self.file_path = file_path
    
    def load_data(self):
        """
        加载数据，如果文件路径存在则读取CSV，否则生成模拟数据
        
        Returns:
            DataFrame: 包含日期和销售额的数据框
        """
        if self.file_path:
            return self._load_from_csv()
        else:
            return self._generate_sample_data()
    
    def _load_from_csv(self):
        """从CSV文件加载数据"""
        try:
            df = pd.read_csv(self.file_path)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            print(f"成功从 {self.file_path} 加载 {len(df)} 条数据")
            return df
        except Exception as e:
            print(f"从CSV加载数据失败: {e}，将生成模拟数据")
            return self._generate_sample_data()
    
    def _generate_sample_data(self):
        """
        生成模拟零售销售数据（包含趋势、季节性和噪声）
        
        Returns:
            DataFrame: 模拟销售数据
        """
        np.random.seed(42)
        
        # 生成365天的历史数据
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # 构建时间序列特征
        t = np.arange(len(dates))
        
        # 趋势项（线性增长）
        trend = 1000 + 2 * t
        
        # 季节性（周效应和年效应）
        weekly_pattern = 200 * np.sin(2 * np.pi * t / 7)  # 周周期
        yearly_pattern = 500 * np.sin(2 * np.pi * t / 365)  # 年周期
        
        # 周末效应（周末销售额更高）
        dayofweek = dates.dayofweek
        weekend_effect = np.where(dayofweek >= 5, 300, 0)
        
        # 节假日效应（模拟促销）
        holiday_effect = np.zeros(len(dates))
        # 每月1号促销
        holiday_effect[dates.day == 1] = 800
        
        # 随机噪声
        noise = np.random.normal(0, 150, len(dates))
        
        # 组合所有成分
        sales = trend + weekly_pattern + yearly_pattern + weekend_effect + holiday_effect + noise
        sales = np.maximum(sales, 100)  # 确保销售额为正
        
        # 创建DataFrame
        df = pd.DataFrame({
            'date': dates,
            'sales': sales.round(2),
            'dayofweek': dayofweek,
            'month': dates.month,
            'day': dates.day
        })
        
        print(f"成功生成 {len(df)} 条模拟销售数据")
        return df
    
    def get_data_summary(self, df):
        """
        获取数据摘要信息
        
        Parameters:
            df: 销售数据DataFrame
            
        Returns:
            dict: 数据摘要统计
        """
        summary = {
            '总记录数': len(df),
            '日期范围': f"{df['date'].min().strftime('%Y-%m-%d')} 至 {df['date'].max().strftime('%Y-%m-%d')}",
            '平均日销售额': f"{df['sales'].mean():.2f}",
            '最高日销售额': f"{df['sales'].max():.2f}",
            '最低日销售额': f"{df['sales'].min():.2f}",
            '销售额标准差': f"{df['sales'].std():.2f}"
        }
        return summary
