"""
数据预处理模块
功能：多重插补法处理缺失值、IQR方法检测异常值、数据标准化
"""

import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """数据预处理类"""
    
    def __init__(self, df):
        """
        初始化
        Args:
            df: pandas DataFrame
        """
        self.original_df = df.copy()
        self.df = df.copy()
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
    def add_missing_values(self, missing_rate=0.05, random_state=42):
        """
        随机添加缺失值（用于演示多重插补）
        Args:
            missing_rate: 缺失率
            random_state: 随机种子
        """
        np.random.seed(random_state)
        df_missing = self.df.copy()
        
        numeric_cols_for_missing = [col for col in self.numeric_cols 
                                     if col not in ['Year', 'Units_Sold']]
        
        for col in numeric_cols_for_missing:
            mask = np.random.rand(len(df_missing)) < missing_rate
            df_missing.loc[mask, col] = np.nan
            
        self.df = df_missing
        return self
    
    def multiple_imputation(self, max_iter=10, random_state=42):
        """
        使用多重插补法(MICE)处理缺失值
        Args:
            max_iter: 最大迭代次数
            random_state: 随机种子
        """
        numeric_cols = [col for col in self.df.columns 
                       if self.df[col].dtype in ['float64', 'int64']]
        
        if self.df[numeric_cols].isnull().sum().sum() == 0:
            print("没有缺失值需要处理")
            return self
        
        imputer = IterativeImputer(max_iter=max_iter, random_state=random_state)
        
        df_imputed = self.df.copy()
        imputed_values = imputer.fit_transform(df_imputed[numeric_cols])
        df_imputed[numeric_cols] = imputed_values
        
        missing_before = self.df.isnull().sum().sum()
        self.df = df_imputed
        missing_after = self.df.isnull().sum().sum()
        
        print(f"多重插补完成：处理前缺失值 {missing_before}，处理后缺失值 {missing_after}")
        return self
    
    def detect_outliers_iqr(self, columns=None, k=1.5, remove=False):
        """
        使用IQR方法检测异常值
        Args:
            columns: 需要检测的列，None表示所有数值列
            k: IQR倍数，默认1.5
            remove: 是否移除异常值
        Returns:
            异常值索引字典
        """
        if columns is None:
            columns = [col for col in self.numeric_cols 
                      if col not in ['Year']]
        
        outlier_indices = {}
        outlier_summary = {}
        
        for col in columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - k * IQR
            upper_bound = Q3 + k * IQR
            
            outliers = self.df[(self.df[col] < lower_bound) | 
                              (self.df[col] > upper_bound)]
            
            outlier_indices[col] = outliers.index.tolist()
            outlier_summary[col] = {
                'count': len(outliers),
                'percentage': len(outliers) / len(self.df) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
        
        print("\nIQR异常值检测结果：")
        for col, info in outlier_summary.items():
            print(f"  {col}: {info['count']}个异常值 ({info['percentage']:.2f}%)")
        
        if remove:
            all_outliers = set()
            for indices in outlier_indices.values():
                all_outliers.update(indices)
            self.df = self.df.drop(index=list(all_outliers))
            print(f"\n已移除 {len(all_outliers)} 个异常值样本")
        
        return outlier_indices
    
    def standardize(self, columns=None, method='zscore'):
        """
        数据标准化
        Args:
            columns: 需要标准化的列
            method: 'zscore' 或 'minmax'
        """
        if columns is None:
            columns = [col for col in self.numeric_cols 
                      if col not in ['Year']]
        
        if method == 'zscore':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError("method必须是 'zscore' 或 'minmax'")
        
        self.df[columns] = scaler.fit_transform(self.df[columns])
        
        print(f"\n使用 {method} 方法标准化完成，涉及列: {columns}")
        return self
    
    def get_processed_data(self):
        """获取处理后的数据"""
        return self.df.copy()
    
    def get_summary(self):
        """获取数据摘要"""
        summary = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'numeric_columns': len(self.numeric_cols),
            'categorical_columns': len(self.categorical_cols),
            'missing_values': self.df.isnull().sum().sum(),
            'memory_usage': self.df.memory_usage(deep=True).sum() / 1024**2
        }
        return summary


def preprocess_pipeline(file_path, add_missing=False):
    """
    完整的数据预处理流程
    Args:
        file_path: CSV文件路径
        add_missing: 是否添加缺失值（用于演示）
    Returns:
        处理后的DataFrame
    """
    df = pd.read_csv(file_path)
    print(f"原始数据加载完成：{df.shape[0]}行 x {df.shape[1]}列")
    
    preprocessor = DataPreprocessor(df)
    
    if add_missing:
        preprocessor.add_missing_values(missing_rate=0.03)
    
    preprocessor.multiple_imputation()
    
    preprocessor.detect_outliers_iqr(remove=False)
    
    print("\n数据预处理完成！")
    print(f"数据摘要: {preprocessor.get_summary()}")
    
    return preprocessor.get_processed_data()


if __name__ == '__main__':
    df_processed = preprocess_pipeline('../data/fuel_performance_cars.csv', add_missing=True)
    print(f"\n处理后的数据前5行：")
    print(df_processed.head())
