import pandas as pd
import numpy as np

class DataPreprocessor:
    def __init__(self):
        pass
    
    def handle_missing_values(self, df):
        df_processed = df.copy()
        
        missing_before = df_processed['观看时长(分钟)'].isnull().sum()
        df_processed['观看时长(分钟)'].fillna(df_processed['观看时长(分钟)'].median(), inplace=True)
        
        print(f"处理缺失值: 填充 {missing_before} 个缺失值")
        return df_processed
    
    def handle_outliers(self, df, column='观看时长(分钟)', method='iqr'):
        df_processed = df.copy()
        
        if method == 'iqr':
            Q1 = df_processed[column].quantile(0.25)
            Q3 = df_processed[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_before = ((df_processed[column] < lower_bound) | 
                             (df_processed[column] > upper_bound)).sum()
            
            df_processed[column] = np.clip(df_processed[column], lower_bound, upper_bound)
            print(f"处理异常值: {column} 列修正 {outliers_before} 个异常值")
        
        return df_processed
    
    def create_features(self, df):
        df_processed = df.copy()
        
        df_processed['学习效率'] = df_processed['测验分数'] / df_processed['观看时长(分钟)'].replace(0, np.nan)
        df_processed['学习效率'].fillna(0, inplace=True)
        
        category_completion = df_processed.groupby('课程类别')['是否完成课程'].transform('mean')
        df_processed['类别平均完成率'] = category_completion
        
        user_course_count = df_processed.groupby('用户ID')['课程ID'].transform('count')
        df_processed['用户学习课程数'] = user_course_count
        
        df_processed['观看时长等级'] = pd.qcut(df_processed['观看时长(分钟)'], 
                                            q=5, labels=['很短', '较短', '中等', '较长', '很长'])
        
        df_processed['测验等级'] = pd.cut(df_processed['测验分数'],
                                       bins=[0, 60, 75, 85, 100],
                                       labels=['不及格', '及格', '良好', '优秀'],
                                       include_lowest=True)
        
        print("特征工程完成: 新增[学习效率, 类别平均完成率, 用户学习课程数, 观看时长等级, 测验等级]特征")
        return df_processed
    
    def encode_categorical(self, df):
        df_processed = df.copy()
        
        df_encoded = pd.get_dummies(df_processed, columns=['课程类别', '观看时长等级', '测验等级'], 
                                   drop_first=True)
        
        print("类别变量编码完成")
        return df_encoded
    
    def preprocess(self, df, encode=True):
        print("="*50)
        print("开始数据预处理...")
        print("="*50)
        
        df_processed = self.handle_missing_values(df)
        df_processed = self.handle_outliers(df_processed)
        df_processed = self.create_features(df_processed)
        
        if encode:
            df_processed = self.encode_categorical(df_processed)
        
        print("="*50)
        print("数据预处理完成!")
        print(f"处理后数据集形状: {df_processed.shape}")
        print("="*50)
        
        return df_processed
    
    def get_basic_info(self, df):
        print("\n" + "="*50)
        print("数据集基本信息")
        print("="*50)
        print(f"数据量: {len(df)} 条")
        print(f"字段数: {len(df.columns)} 个")
        print("\n字段列表:")
        for col in df.columns:
            print(f"  - {col}")
        print("\n缺失值统计:")
        print(df.isnull().sum())
        print("="*50)
