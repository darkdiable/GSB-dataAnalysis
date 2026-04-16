import pandas as pd
import numpy as np
import os

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.cleaning_report = {}
    
    def remove_duplicates(self):
        """去除重复记录"""
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates()
        final_count = len(self.df)
        removed = initial_count - final_count
        self.cleaning_report['重复记录移除'] = removed
        print(f"移除重复记录: {removed} 条")
        return self
    
    def handle_missing_values(self):
        """处理缺失值"""
        missing_before = self.df.isnull().sum().to_dict()
        
        self.df['订单金额'] = self.df['订单金额'].fillna(self.df['订单金额'].median())
        self.df['评分'] = self.df['评分'].fillna(self.df['评分'].mode()[0])
        
        missing_after = self.df.isnull().sum().to_dict()
        total_missing = sum(missing_before.values())
        self.cleaning_report['缺失值处理'] = total_missing
        print(f"处理缺失值: {total_missing} 个")
        return self
    
    def handle_outliers(self, columns=None, method='iqr'):
        """处理异常值"""
        if columns is None:
            columns = ['订单金额', '单价']
        
        outliers_handled = 0
        for col in columns:
            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_mask = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
                outliers_count = outlier_mask.sum()
                
                self.df.loc[self.df[col] > upper_bound, col] = upper_bound
                self.df.loc[self.df[col] < lower_bound, col] = lower_bound
                
                outliers_handled += outliers_count
        
        self.cleaning_report['异常值处理'] = outliers_handled
        print(f"处理异常值: {outliers_handled} 个")
        return self
    
    def convert_data_types(self):
        """转换数据类型"""
        if '订单时间' in self.df.columns:
            self.df['订单时间'] = pd.to_datetime(self.df['订单时间'])
            self.df['日期'] = self.df['订单时间'].dt.date
            self.df['月份'] = self.df['订单时间'].dt.to_period('M')
            self.df['星期'] = self.df['订单时间'].dt.day_name()
            self.df['小时'] = self.df['订单时间'].dt.hour
        print("数据类型转换完成")
        return self
    
    def clean(self):
        """执行完整清洗流程"""
        print("=" * 50)
        print("开始数据清洗...")
        print("=" * 50)
        
        self.remove_duplicates()
        self.handle_missing_values()
        self.handle_outliers()
        self.convert_data_types()
        
        print("=" * 50)
        print("数据清洗完成!")
        print("清洗报告:", self.cleaning_report)
        print("=" * 50)
        
        return self.df
    
    def get_cleaning_report(self):
        """获取清洗报告"""
        return self.cleaning_report
    
    def save_cleaned_data(self, output_path='../data/cleaned_ecommerce_data.csv'):
        """保存清洗后的数据"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"清洗后数据已保存至: {output_path}")
        print(f"清洗后数据形状: {self.df.shape}")

if __name__ == '__main__':
    from data_generator import generate_ecommerce_data
    
    df_raw = generate_ecommerce_data(n_records=2000)
    print("原始数据形状:", df_raw.shape)
    print("原始数据缺失值:\n", df_raw.isnull().sum())
    
    cleaner = DataCleaner(df_raw)
    df_cleaned = cleaner.clean()
    
    print("\n清洗后数据预览:")
    print(df_cleaned.head())
    print("\n清洗后数据信息:")
    print(df_cleaned.info())
