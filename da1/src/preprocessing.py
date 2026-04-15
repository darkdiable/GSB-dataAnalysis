import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def mice_imputation(df, numeric_cols=['production', 'sales']):
    imputer = IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=50, random_state=42),
        max_iter=50,
        random_state=42
    )
    df_imputed = df.copy()
    df_imputed[numeric_cols] = imputer.fit_transform(df_imputed[numeric_cols])
    return df_imputed

def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

def handle_outliers(df, columns=['production', 'sales']):
    outlier_report = {}
    for col in columns:
        outliers, lower, upper = detect_outliers_iqr(df, col)
        outlier_report[col] = {
            'count': len(outliers),
            'lower_bound': lower,
            'upper_bound': upper,
            'outlier_values': outliers[col].tolist()
        }
        df.loc[df[col] < lower, col] = lower
        df.loc[df[col] > upper, col] = upper
    return df, outlier_report

def standardize_data(df, numeric_cols=['production', 'sales']):
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])
    return df_scaled, scaler

def preprocess_pipeline(file_path):
    print("=== 开始数据预处理 ===")
    
    df = load_data(file_path)
    print(f"原始数据形状: {df.shape}")
    print(f"缺失值统计:\n{df.isnull().sum()}")
    
    df_imputed = mice_imputation(df)
    print(f"多重插补后缺失值: {df_imputed[['production', 'sales']].isnull().sum().sum()}")
    
    df_cleaned, outlier_report = handle_outliers(df_imputed)
    print(f"异常值处理报告:")
    for col, report in outlier_report.items():
        print(f"  {col}: 发现 {report['count']} 个异常值")
    
    df_scaled, scaler = standardize_data(df_cleaned)
    
    print("=== 数据预处理完成 ===")
    return df_cleaned, df_scaled, outlier_report, scaler
