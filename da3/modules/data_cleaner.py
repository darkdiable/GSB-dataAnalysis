import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')


def clean_data(df, output_path='data/cleaned_sales.csv'):
    """
    数据清洗主函数：处理重复值、缺失值、异常值
    
    参数:
        df: 原始DataFrame
        output_path: 清洗后数据保存路径
    
    返回:
        DataFrame: 清洗后的数据
        dict: 清洗报告统计信息
    """
    try:
        print("\n开始数据清洗...")
        cleaning_report = {}
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        initial_count = len(df)
        cleaning_report['初始记录数'] = initial_count
        print(f"初始记录数: {initial_count}")
        
        df = remove_duplicates(df, cleaning_report)
        
        df = handle_missing_values(df, cleaning_report)
        
        df = handle_outliers(df, cleaning_report)
        
        df = format_data(df)
        
        final_count = len(df)
        cleaning_report['最终记录数'] = final_count
        cleaning_report['总删除记录数'] = initial_count - final_count
        
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n数据清洗完成！清洗后数据已保存至 {output_path}")
        print(f"清洗后记录数: {final_count}")
        print(f"总删除记录数: {initial_count - final_count}")
        
        return df, cleaning_report
    
    except Exception as e:
        print(f"数据清洗过程中发生错误: {str(e)}")
        raise


def remove_duplicates(df, report):
    """
    处理重复值
    """
    print("\n处理重复值...")
    dup_count = df.duplicated().sum()
    report['重复记录数'] = int(dup_count)
    
    if dup_count > 0:
        df = df.drop_duplicates(keep='first')
        print(f"删除重复记录: {dup_count} 条")
    else:
        print("未发现重复记录")
    
    return df


def handle_missing_values(df, report):
    """
    处理缺失值
    """
    print("\n处理缺失值...")
    missing_stats = df.isnull().sum()
    report['缺失值统计'] = missing_stats.to_dict()
    
    total_missing = missing_stats.sum()
    report['缺失值总数'] = int(total_missing)
    
    if total_missing > 0:
        print("缺失值分布:")
        for col, count in missing_stats[missing_stats > 0].items():
            print(f"  {col}: {count} 条 ({count/len(df)*100:.2f}%)")
        
        df['订单金额'] = df['订单金额'].fillna(df.groupby('商品类别')['订单金额'].transform('median'))
        df['用户评分'] = df['用户评分'].fillna(df['用户评分'].median())
        df['商品类别'] = df['商品类别'].fillna(df['商品类别'].mode()[0])
        
        print("缺失值填充完成")
    else:
        print("未发现缺失值")
    
    return df


def handle_outliers(df, report):
    """
    处理异常值（使用IQR方法）
    """
    print("\n处理异常值...")
    outliers_count = 0
    
    for col in ['订单金额', '商品数量', '用户评分']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        col_outliers_count = len(outliers)
        
        if col_outliers_count > 0:
            print(f"{col} 异常值: {col_outliers_count} 条")
            outliers_count += col_outliers_count
            
            df.loc[df[col] > upper_bound, col] = upper_bound
            df.loc[df[col] < lower_bound, col] = max(lower_bound, df[col].min())
    
    report['异常值处理数'] = outliers_count
    print(f"异常值处理完成，共处理 {outliers_count} 条异常记录")
    
    return df


def format_data(df):
    """
    数据格式标准化
    """
    print("\n数据格式标准化...")
    
    df['下单时间'] = pd.to_datetime(df['下单时间'])
    df['订单金额'] = df['订单金额'].round(2)
    
    df['年份'] = df['下单时间'].dt.year
    df['月份'] = df['下单时间'].dt.month
    df['日期'] = df['下单时间'].dt.date
    df['星期'] = df['下单时间'].dt.dayofweek
    df['小时'] = df['下单时间'].dt.hour
    
    print("数据格式标准化完成")
    
    return df


def data_quality_report(df, report, output_path='output/cleaning_report.txt'):
    """
    生成数据质量报告
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 50 + "\n")
        f.write("数据清洗质量报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"初始记录数: {report['初始记录数']}\n")
        f.write(f"重复记录数: {report['重复记录数']}\n")
        f.write(f"缺失值总数: {report['缺失值总数']}\n")
        f.write(f"异常值处理数: {report['异常值处理数']}\n")
        f.write(f"总删除记录数: {report['总删除记录数']}\n")
        f.write(f"最终记录数: {report['最终记录数']}\n\n")
        
        f.write("数据概览:\n")
        f.write(f"  时间范围: {df['下单时间'].min()} 至 {df['下单时间'].max()}\n")
        f.write(f"  用户数: {df['用户ID'].nunique()}\n")
        f.write(f"  商品类别数: {df['商品类别'].nunique()}\n")
        f.write(f"  总销售额: {df['订单金额'].sum():.2f}\n")
        f.write(f"  平均订单金额: {df['订单金额'].mean():.2f}\n")
    
    print(f"\n数据质量报告已保存至: {output_path}")


if __name__ == '__main__':
    from data_generator import generate_sales_data
    df = generate_sales_data()
    cleaned_df, report = clean_data(df)
    data_quality_report(cleaned_df, report)
