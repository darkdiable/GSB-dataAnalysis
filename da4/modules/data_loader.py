import pandas as pd
import os


class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"数据文件不存在: {self.file_path}")
        
        file_ext = os.path.splitext(self.file_path)[1].lower()
        
        if file_ext == '.csv':
            self.data = pd.read_csv(self.file_path)
        elif file_ext in ['.xlsx', '.xls']:
            self.data = pd.read_excel(self.file_path)
        else:
            raise ValueError(f"不支持的文件格式: {file_ext}")
        
        print(f"成功加载数据，共 {len(self.data)} 条记录")
        return self.data

    def get_data_info(self):
        if self.data is None:
            raise ValueError("数据尚未加载")
        
        info = {
            'shape': self.data.shape,
            'columns': self.data.columns.tolist(),
            'dtypes': self.data.dtypes.to_dict(),
            'missing_values': self.data.isnull().sum().to_dict()
        }
        return info

    def validate_date_column(self, date_column):
        if self.data is None:
            raise ValueError("数据尚未加载")
        
        if date_column not in self.data.columns:
            raise ValueError(f"日期列 '{date_column}' 不存在")
        
        try:
            self.data[date_column] = pd.to_datetime(self.data[date_column])
            print(f"日期列 '{date_column}' 格式验证通过")
            return True
        except Exception as e:
            raise ValueError(f"日期列格式转换失败: {e}")

    def validate_sales_column(self, sales_column):
        if self.data is None:
            raise ValueError("数据尚未加载")
        
        if sales_column not in self.data.columns:
            raise ValueError(f"销售额列 '{sales_column}' 不存在")
        
        if not pd.api.types.is_numeric_dtype(self.data[sales_column]):
            try:
                self.data[sales_column] = pd.to_numeric(self.data[sales_column], errors='coerce')
                print(f"销售额列 '{sales_column}' 已转换为数值类型")
            except Exception as e:
                raise ValueError(f"销售额列转换失败: {e}")
        
        return True
