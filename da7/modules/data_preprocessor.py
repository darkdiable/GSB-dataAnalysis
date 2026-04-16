import pandas as pd
import numpy as np
from datetime import datetime
import os


class DataPreprocessor:
    def __init__(self):
        self.data = None
        self.processed_data = None
        self.numeric_features = ['watch_duration_minutes', 'quiz_score']
        self.categorical_features = ['course_category']
        
    def load_data(self, filepath='data/raw_data.csv'):
        full_path = os.path.join('/Users/bilei/work/LargeModelAnnotation/GBS/260410/GSB-dataAnalysis/da7', filepath)
        self.data = pd.read_csv(full_path)
        print(f"数据加载成功！形状: {self.data.shape}")
        return self.data
    
    def check_missing_values(self):
        print("\n缺失值统计:")
        missing = self.data.isnull().sum()
        missing_pct = (missing / len(self.data)) * 100
        missing_df = pd.DataFrame({
            '缺失数量': missing,
            '缺失比例(%)': missing_pct.round(2)
        })
        print(missing_df[missing_df['缺失数量'] > 0])
        return missing_df
    
    def check_outliers(self):
        print("\n异常值检测:")
        outliers_info = {}
        
        for col in self.numeric_features:
            if col in self.data.columns:
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = self.data[(self.data[col] < lower_bound) | (self.data[col] > upper_bound)]
                outliers_info[col] = {
                    '下限': lower_bound,
                    '上限': upper_bound,
                    '异常值数量': len(outliers)
                }
                print(f"{col}: 下限={lower_bound:.2f}, 上限={upper_bound:.2f}, 异常值={len(outliers)}")
        
        return outliers_info
    
    def handle_missing_values(self):
        print("\n处理缺失值...")
        
        for col in self.numeric_features:
            if col in self.data.columns:
                median_val = self.data[col].median()
                missing_count = self.data[col].isnull().sum()
                self.data[col].fillna(median_val, inplace=True)
                print(f"  {col}: 使用中位数 {median_val:.2f} 填充 {missing_count} 个缺失值")
        
        return self.data
    
    def handle_outliers(self):
        print("\n处理异常值...")
        
        for col in self.numeric_features:
            if col in self.data.columns:
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers_count = len(self.data[(self.data[col] < lower_bound) | (self.data[col] > upper_bound)])
                
                self.data[col] = np.where(self.data[col] < lower_bound, lower_bound, self.data[col])
                self.data[col] = np.where(self.data[col] > upper_bound, upper_bound, self.data[col])
                
                print(f"  {col}: 将 {outliers_count} 个异常值限制在 [{lower_bound:.2f}, {upper_bound:.2f}]")
        
        return self.data
    
    def engineer_features(self):
        print("\n构建新特征...")
        
        self.data['learning_efficiency'] = np.where(
            self.data['watch_duration_minutes'] > 0,
            self.data['quiz_score'] / self.data['watch_duration_minutes'],
            0
        ).round(4)
        
        self.data['engagement_level'] = pd.cut(
            self.data['watch_duration_minutes'],
            bins=[0, 30, 90, 180, 300],
            labels=['低', '中', '高', '极高']
        )
        
        self.data['score_level'] = pd.cut(
            self.data['quiz_score'],
            bins=[0, 60, 80, 90, 100],
            labels=['不及格', '及格', '良好', '优秀']
        )
        
        category_completion = self.data.groupby('course_category')['is_completed'].mean()
        self.data['category_avg_completion'] = self.data['course_category'].map(category_completion).round(4)
        
        self.data['days_since_registration'] = (
            datetime.now() - pd.to_datetime(self.data['registration_date'])
        ).dt.days
        
        user_activity = self.data.groupby('user_id').agg({
            'course_id': 'count',
            'is_completed': 'sum'
        }).reset_index()
        user_activity.columns = ['user_id', 'total_courses', 'completed_courses']
        user_activity['user_completion_rate'] = (
            user_activity['completed_courses'] / user_activity['total_courses']
        ).round(4)
        
        self.data = self.data.merge(user_activity[['user_id', 'user_completion_rate']], on='user_id', how='left')
        
        print("  - 学习效率 (quiz_score / watch_duration_minutes)")
        print("  - 参与度等级 (engagement_level)")
        print("  - 分数等级 (score_level)")
        print("  - 类别平均完成率 (category_avg_completion)")
        print("  - 注册天数 (days_since_registration)")
        print("  - 用户完成率 (user_completion_rate)")
        
        return self.data
    
    def preprocess(self):
        print("="*50)
        print("开始数据预处理")
        print("="*50)
        
        self.check_missing_values()
        self.check_outliers()
        self.handle_missing_values()
        self.handle_outliers()
        self.engineer_features()
        
        self.processed_data = self.data.copy()
        
        print("\n预处理完成！")
        print(f"最终数据形状: {self.processed_data.shape}")
        print(f"特征列表: {list(self.processed_data.columns)}")
        
        return self.processed_data
    
    def save_processed_data(self, filepath='data/processed_data.csv'):
        full_path = os.path.join('/Users/bilei/work/LargeModelAnnotation/GBS/260410/GSB-dataAnalysis/da7', filepath)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        self.processed_data.to_csv(full_path, index=False, encoding='utf-8-sig')
        print(f"\n处理后的数据已保存至: {full_path}")
        return full_path
    
    def get_feature_columns(self):
        feature_cols = [
            'watch_duration_minutes', 'quiz_score', 'learning_efficiency',
            'category_avg_completion', 'days_since_registration', 'user_completion_rate'
        ]
        return feature_cols
    
    def get_target_column(self):
        return 'is_completed'


if __name__ == '__main__':
    preprocessor = DataPreprocessor()
    preprocessor.load_data()
    preprocessor.preprocess()
    preprocessor.save_processed_data()
