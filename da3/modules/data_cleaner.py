"""
数据清洗模块
处理缺失值、异常值、重复值等数据质量问题
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os


class DataCleaner:
    """数据清洗器"""
    
    def __init__(self, df):
        """
        初始化数据清洗器
        
        Args:
            df: 待清洗的DataFrame
        """
        self.original_df = df.copy()
        self.cleaned_df = None
        self.cleaning_log = []
        
    def check_data_quality(self):
        """
        检查数据质量，返回质量报告
        
        Returns:
            dict: 数据质量报告
        """
        df = self.original_df
        report = {
            '总记录数': len(df),
            '总字段数': len(df.columns),
            '缺失值统计': df.isnull().sum().to_dict(),
            '缺失值比例': (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
            '重复记录数': df.duplicated().sum(),
            '数据类型': df.dtypes.to_dict()
        }
        
        # 检查数值型字段的异常值
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outliers_info = {}
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outliers_info[col] = len(outliers)
        
        report['异常值统计'] = outliers_info
        
        return report
    
    def handle_missing_values(self, df, strategy='auto'):
        """
        处理缺失值
        
        Args:
            df: 数据框
            strategy: 处理策略，'auto'为自动选择
            
        Returns:
            DataFrame: 处理后的数据
        """
        df_clean = df.copy()
        missing_before = df_clean.isnull().sum().sum()
        
        if strategy == 'auto':
            # 数值型字段：用中位数填充
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df_clean[col].isnull().sum() > 0:
                    median_val = df_clean[col].median()
                    df_clean[col].fillna(median_val, inplace=True)
                    self.cleaning_log.append(f"字段 '{col}': 用中位数 {median_val} 填充 {df[col].isnull().sum()} 个缺失值")
            
            # 分类型字段：用众数填充
            categorical_cols = df_clean.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if df_clean[col].isnull().sum() > 0:
                    mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else '未知'
                    df_clean[col].fillna(mode_val, inplace=True)
                    self.cleaning_log.append(f"字段 '{col}': 用众数 '{mode_val}' 填充 {df[col].isnull().sum()} 个缺失值")
        
        missing_after = df_clean.isnull().sum().sum()
        print(f"缺失值处理完成: {missing_before} -> {missing_after}")
        
        return df_clean
    
    def handle_duplicates(self, df, subset=None):
        """
        处理重复值
        
        Args:
            df: 数据框
            subset: 用于判断重复的字段列表，None表示所有字段
            
        Returns:
            DataFrame: 去重后的数据
        """
        df_clean = df.copy()
        duplicates_before = df_clean.duplicated(subset=subset).sum()
        
        # 删除重复记录，保留第一条
        df_clean = df_clean.drop_duplicates(subset=subset, keep='first')
        
        duplicates_after = df_clean.duplicated(subset=subset).sum()
        removed = duplicates_before - duplicates_after
        
        self.cleaning_log.append(f"删除重复记录: {removed} 条")
        print(f"重复值处理完成: 删除 {removed} 条重复记录")
        
        return df_clean
    
    def handle_outliers(self, df, method='iqr', columns=None):
        """
        处理异常值
        
        Args:
            df: 数据框
            method: 异常值检测方法，'iqr'为四分位距法
            columns: 需要处理的数值型字段，None表示所有数值型字段
            
        Returns:
            DataFrame: 处理后的数据
        """
        df_clean = df.copy()
        
        if columns is None:
            columns = df_clean.select_dtypes(include=[np.number]).columns
        
        total_outliers = 0
        
        for col in columns:
            if method == 'iqr':
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # 标记异常值
                outliers_mask = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
                outliers_count = outliers_mask.sum()
                
                if outliers_count > 0:
                    # 使用上下界替换异常值
                    df_clean.loc[df_clean[col] < lower_bound, col] = lower_bound
                    df_clean.loc[df_clean[col] > upper_bound, col] = upper_bound
                    
                    self.cleaning_log.append(
                        f"字段 '{col}': 用边界值 [{lower_bound:.2f}, {upper_bound:.2f}] 替换 {outliers_count} 个异常值"
                    )
                    total_outliers += outliers_count
        
        print(f"异常值处理完成: 处理 {total_outliers} 个异常值")
        return df_clean
    
    def validate_data(self, df):
        """
        数据验证，确保数据符合业务规则
        
        Args:
            df: 数据框
            
        Returns:
            DataFrame: 验证后的数据
        """
        df_clean = df.copy()
        invalid_count = 0
        
        # 验证规则1: 总金额应该等于单价乘以数量
        amount_mismatch = abs(df_clean['总金额'] - df_clean['单价'] * df_clean['数量']) > 0.01
        if amount_mismatch.sum() > 0:
            df_clean.loc[amount_mismatch, '总金额'] = (
                df_clean.loc[amount_mismatch, '单价'] * df_clean.loc[amount_mismatch, '数量']
            )
            invalid_count += amount_mismatch.sum()
            self.cleaning_log.append(f"修正金额不匹配: {amount_mismatch.sum()} 条记录")
        
        # 验证规则2: 用户评分应该在1-5之间
        invalid_ratings = (df_clean['用户评分'] < 1) | (df_clean['用户评分'] > 5)
        if invalid_ratings.sum() > 0:
            df_clean.loc[df_clean['用户评分'] < 1, '用户评分'] = 1
            df_clean.loc[df_clean['用户评分'] > 5, '用户评分'] = 5
            invalid_count += invalid_ratings.sum()
            self.cleaning_log.append(f"修正无效评分: {invalid_ratings.sum()} 条记录")
        
        # 验证规则3: 数量应该为正整数
        invalid_quantities = df_clean['数量'] <= 0
        if invalid_quantities.sum() > 0:
            df_clean.loc[invalid_quantities, '数量'] = 1
            # 重新计算总金额
            df_clean.loc[invalid_quantities, '总金额'] = (
                df_clean.loc[invalid_quantities, '单价'] * 
                df_clean.loc[invalid_quantities, '数量']
            )
            invalid_count += invalid_quantities.sum()
            self.cleaning_log.append(f"修正无效数量: {invalid_quantities.sum()} 条记录")
        
        # 验证规则4: 单价应该为正数
        invalid_prices = df_clean['单价'] <= 0
        if invalid_prices.sum() > 0:
            # 用该类别的中位数替换
            for idx in df_clean[invalid_prices].index:
                category = df_clean.loc[idx, '商品类别']
                median_price = df_clean[df_clean['商品类别'] == category]['单价'].median()
                df_clean.loc[idx, '单价'] = median_price if median_price > 0 else 100
            # 重新计算总金额
            df_clean.loc[invalid_prices, '总金额'] = (
                df_clean.loc[invalid_prices, '单价'] * 
                df_clean.loc[invalid_prices, '数量']
            )
            invalid_count += invalid_prices.sum()
            self.cleaning_log.append(f"修正无效单价: {invalid_prices.sum()} 条记录")
        
        print(f"数据验证完成: 修正 {invalid_count} 条无效记录")
        return df_clean
    
    def clean(self):
        """
        执行完整的数据清洗流程
        
        Returns:
            DataFrame: 清洗后的数据
        """
        print("=" * 50)
        print("开始数据清洗...")
        print("=" * 50)
        
        # 1. 检查数据质量
        print("\n1. 数据质量检查:")
        quality_report = self.check_data_quality()
        print(f"   - 总记录数: {quality_report['总记录数']}")
        print(f"   - 缺失值: {sum(quality_report['缺失值统计'].values())} 个")
        print(f"   - 重复记录: {quality_report['重复记录数']} 条")
        
        # 2. 处理缺失值
        print("\n2. 处理缺失值...")
        df = self.handle_missing_values(self.original_df)
        
        # 3. 处理重复值
        print("\n3. 处理重复值...")
        df = self.handle_duplicates(df)
        
        # 4. 处理异常值
        print("\n4. 处理异常值...")
        df = self.handle_outliers(df, columns=['单价', '总金额', '用户评分'])
        
        # 5. 数据验证
        print("\n5. 数据验证...")
        df = self.validate_data(df)
        
        self.cleaned_df = df
        
        print("\n" + "=" * 50)
        print("数据清洗完成!")
        print(f"原始记录数: {len(self.original_df)}")
        print(f"清洗后记录数: {len(self.cleaned_df)}")
        print("=" * 50)
        
        return self.cleaned_df
    
    def get_cleaning_report(self):
        """
        获取清洗报告
        
        Returns:
            str: 清洗报告文本
        """
        report = []
        report.append("=" * 60)
        report.append("数据清洗报告")
        report.append("=" * 60)
        report.append(f"清洗时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"原始记录数: {len(self.original_df)}")
        report.append(f"清洗后记录数: {len(self.cleaned_df)}")
        report.append(f"删除记录数: {len(self.original_df) - len(self.cleaned_df)}")
        report.append("-" * 60)
        report.append("清洗操作明细:")
        for log in self.cleaning_log:
            report.append(f"  - {log}")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def save_cleaned_data(self, filepath):
        """
        保存清洗后的数据
        
        Args:
            filepath: 保存路径
        """
        try:
            if self.cleaned_df is None:
                raise ValueError("请先执行数据清洗")
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            self.cleaned_df.to_csv(filepath, index=False, encoding='utf-8-sig')
            print(f"清洗后的数据已保存至: {filepath}")
        except Exception as e:
            print(f"保存数据失败: {str(e)}")
            raise


def clean_data(input_path, output_path=None):
    """
    清洗数据的便捷函数
    
    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径，None则不保存
        
    Returns:
        DataFrame: 清洗后的数据
    """
    # 读取数据
    df = pd.read_csv(input_path, encoding='utf-8-sig')
    
    # 转换日期字段
    df['订单时间'] = pd.to_datetime(df['订单时间'])
    
    # 创建清洗器并执行清洗
    cleaner = DataCleaner(df)
    cleaned_df = cleaner.clean()
    
    # 保存清洗后的数据
    if output_path:
        cleaner.save_cleaned_data(output_path)
    
    return cleaned_df, cleaner


if __name__ == '__main__':
    # 测试数据清洗
    from data_generator import generate_raw_data
    
    # 生成测试数据
    raw_df = generate_raw_data('data/test_raw.csv', n_records=500)
    
    # 清洗数据
    cleaned_df, cleaner = clean_data('data/test_raw.csv', 'data/test_cleaned.csv')
    
    # 打印清洗报告
    print("\n" + cleaner.get_cleaning_report())
