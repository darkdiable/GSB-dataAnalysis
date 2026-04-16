import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


class DataPreprocessor:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.category_mapping = None
        
    def handle_missing_values(self, df):
        df = df.copy()
        
        if 'quiz_score' in df.columns:
            median_score = df['quiz_score'].median()
            df['quiz_score'].fillna(median_score, inplace=True)
            print(f"测验分数缺失值已用中位数 {median_score:.2f} 填充")
        
        return df
    
    def handle_outliers(self, df, column='watch_duration', method='iqr'):
        df = df.copy()
        
        if column not in df.columns:
            return df
            
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_count = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
            df[column] = df[column].clip(lower_bound, upper_bound)
            print(f"观看时长异常值处理: 使用IQR方法，修正了 {outlier_count} 个异常值")
        
        return df
    
    def create_features(self, df):
        df = df.copy()
        
        df['learning_efficiency'] = df['quiz_score'] / (df['watch_duration'] + 1e-6)
        df['learning_efficiency'] = df['learning_efficiency'].clip(upper=10)
        
        df['watch_duration_category'] = pd.cut(
            df['watch_duration'],
            bins=[0, 30, 60, 120, 300],
            labels=['短', '中', '长', '超长']
        )
        
        df['quiz_score_category'] = pd.cut(
            df['quiz_score'],
            bins=[0, 60, 80, 100],
            labels=['不及格', '及格', '优秀']
        )
        
        print("新特征创建完成: learning_efficiency, watch_duration_category, quiz_score_category")
        
        return df
    
    def encode_categorical(self, df, columns=None):
        df = df.copy()
        
        if columns is None:
            columns = ['course_category']
        
        for col in columns:
            if col in df.columns:
                df[f'{col}_encoded'] = self.label_encoder.fit_transform(df[col])
                self.category_mapping = dict(zip(
                    self.label_encoder.classes_,
                    self.label_encoder.transform(self.label_encoder.classes_)
                ))
                print(f"类别特征 '{col}' 编码完成")
        
        return df
    
    def preprocess(self, df):
        print("=" * 50)
        print("开始数据预处理...")
        print("=" * 50)
        
        print(f"\n原始数据形状: {df.shape}")
        print(f"缺失值统计:\n{df.isnull().sum()}")
        
        df = self.handle_missing_values(df)
        df = self.handle_outliers(df)
        df = self.create_features(df)
        df = self.encode_categorical(df)
        
        print(f"\n预处理后数据形状: {df.shape}")
        print("=" * 50)
        
        return df
    
    def get_feature_columns(self):
        return ['watch_duration', 'quiz_score', 'course_category_encoded', 'learning_efficiency']


def load_data(filepath):
    df = pd.read_csv(filepath)
    print(f"数据加载成功: {filepath}")
    print(f"数据形状: {df.shape}")
    return df


if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    
    from data_generator import generate_course_data
    data = generate_course_data(n_samples=100)
    
    processed_data = preprocessor.preprocess(data)
    print(processed_data.head())
