import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class DataPreprocessor:
    def __init__(self, df):
        self.df = df.copy()
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.outliers_info = {}
        
    def multiple_imputation(self):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        imputer = IterativeImputer(max_iter=10, random_state=42)
        self.df[numeric_cols] = imputer.fit_transform(self.df[numeric_cols])
        print("多重插补完成")
        return self
    
    def detect_outliers_iqr(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        for col in columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_mask = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
            outlier_count = outliers_mask.sum()
            
            self.outliers_info[col] = {
                'count': outlier_count,
                'percentage': outlier_count / len(self.df) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'mask': outliers_mask
            }
            
        return self
    
    def remove_outliers(self, columns=None):
        if columns is None:
            columns = list(self.outliers_info.keys())
            
        for col in columns:
            if col in self.outliers_info:
                mask = self.outliers_info[col]['mask']
                self.df = self.df[~mask]
                
        print(f"异常值移除完成，剩余 {len(self.df)} 条记录")
        return self
    
    def cap_outliers(self, columns=None):
        if columns is None:
            columns = list(self.outliers_info.keys())
            
        for col in columns:
            if col in self.outliers_info:
                lower = self.outliers_info[col]['lower_bound']
                upper = self.outliers_info[col]['upper_bound']
                self.df[col] = self.df[col].clip(lower=lower, upper=upper)
                
        print("异常值盖帽处理完成")
        return self
    
    def standardize_data(self, columns=None):
        if columns is None:
            columns = ['Horsepower', 'Price_USD', 'Sales_Units', 'Production_Units',
                      'Fuel_Efficiency_L100km', 'CO2_Emissions_gkm', 
                      'Acceleration_0_100_s', 'Top_Speed_kmh']
            columns = [c for c in columns if c in self.df.columns]
            
        scaled_data = self.scaler.fit_transform(self.df[columns])
        scaled_df = pd.DataFrame(scaled_data, columns=[f'{c}_scaled' for c in columns], 
                                index=self.df.index)
        self.df = pd.concat([self.df, scaled_df], axis=1)
        print("数据标准化完成")
        return self
    
    def encode_categorical(self, columns=None):
        if columns is None:
            columns = ['Country', 'Brand', 'Engine_Type', 'HP_Category']
            columns = [c for c in columns if c in self.df.columns]
            
        for col in columns:
            le = LabelEncoder()
            self.df[f'{col}_encoded'] = le.fit_transform(self.df[col])
            self.label_encoders[col] = le
            
        print("分类变量编码完成")
        return self
    
    def get_processed_data(self):
        return self.df
    
    def plot_outliers_summary(self, save_path):
        if not self.outliers_info:
            print("请先运行 detect_outliers_iqr()")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        outlier_counts = {k: v['count'] for k, v in self.outliers_info.items() 
                         if v['count'] > 0}
        
        if outlier_counts:
            ax1 = axes[0, 0]
            cols = list(outlier_counts.keys())[:8]
            counts = [outlier_counts[c] for c in cols]
            bars = ax1.barh(cols, counts, color='coral')
            ax1.set_xlabel('异常值数量')
            ax1.set_title('各特征异常值数量统计')
            for bar, count in zip(bars, counts):
                ax1.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                        f'{count}', va='center')
        
        ax2 = axes[0, 1]
        numeric_cols = ['Price_USD', 'Horsepower', 'Sales_Units']
        numeric_cols = [c for c in numeric_cols if c in self.df.columns]
        if numeric_cols:
            box_data = [self.df[col].dropna() for col in numeric_cols]
            bp = ax2.boxplot(box_data, labels=numeric_cols, patch_artist=True)
            colors = ['lightblue', 'lightgreen', 'lightyellow']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            ax2.set_title('关键特征箱线图（异常值可视化）')
            ax2.tick_params(axis='x', rotation=15)
        
        ax3 = axes[1, 0]
        missing_before = self.df.isnull().sum()
        missing_before = missing_before[missing_before > 0]
        if len(missing_before) > 0:
            ax3.bar(missing_before.index, missing_before.values, color='steelblue')
            ax3.set_xlabel('特征')
            ax3.set_ylabel('缺失值数量')
            ax3.set_title('缺失值分布（处理前）')
            ax3.tick_params(axis='x', rotation=45)
        else:
            ax3.text(0.5, 0.5, '无缺失值', ha='center', va='center', fontsize=14)
            ax3.set_title('缺失值分布')
        
        ax4 = axes[1, 1]
        stats_text = "数据预处理统计\n" + "="*30 + "\n"
        stats_text += f"总记录数: {len(self.df)}\n"
        stats_text += f"特征数量: {len(self.df.columns)}\n"
        stats_text += f"缺失值: {self.df.isnull().sum().sum()}\n"
        stats_text += f"异常值特征数: {len([k for k,v in self.outliers_info.items() if v['count']>0])}\n"
        ax4.text(0.1, 0.5, stats_text, fontsize=12, family='monospace',
                verticalalignment='center', transform=ax4.transAxes)
        ax4.axis('off')
        ax4.set_title('预处理统计摘要')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"异常值分析图已保存: {save_path}")
        
    def generate_report(self):
        report = "# 数据预处理报告\n\n"
        report += "## 1. 数据概览\n"
        report += f"- 总记录数: {len(self.df)}\n"
        report += f"- 特征数量: {len(self.df.columns)}\n\n"
        
        report += "## 2. 缺失值处理\n"
        report += "采用多重插补法（MICE）处理缺失值，该方法通过迭代回归模型预测缺失值。\n\n"
        
        report += "## 3. 异常值检测\n"
        report += "使用IQR（四分位距）方法检测异常值：\n\n"
        for col, info in self.outliers_info.items():
            if info['count'] > 0:
                report += f"- **{col}**: {info['count']}个异常值 ({info['percentage']:.2f}%)\n"
        report += "\n"
        
        report += "## 4. 数据标准化\n"
        report += "对数值型特征进行Z-score标准化处理。\n"
        
        return report


if __name__ == "__main__":
    df = pd.read_csv("../data/fuel_performance_cars.csv")
    preprocessor = DataPreprocessor(df)
    preprocessor.multiple_imputation()
    preprocessor.detect_outliers_iqr()
    preprocessor.cap_outliers()
    preprocessor.standardize_data()
    preprocessor.encode_categorical()
    processed_df = preprocessor.get_processed_data()
    print(processed_df.head())
