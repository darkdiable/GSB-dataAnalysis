"""
数据加载模块
提供健壮的数据加载功能，包含异常处理
"""
import pandas as pd
import os
import logging
from typing import Dict, Optional, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataLoader:
    """数据加载器类"""

    def __init__(self, data_dir: str = 'data'):
        self.data_dir = data_dir
        self.data = {}

    def load_csv(self, filename: str, **kwargs) -> Optional[pd.DataFrame]:
        """
        加载CSV文件

        Args:
            filename: CSV文件名
            **kwargs: 传递给pd.read_csv的额外参数

        Returns:
            DataFrame或None（如果加载失败）
        """
        filepath = os.path.join(self.data_dir, filename)

        try:
            if not os.path.exists(filepath):
                logger.error(f"文件不存在: {filepath}")
                return None

            df = pd.read_csv(filepath, encoding='utf-8-sig', **kwargs)
            logger.info(f"成功加载 {filename}, 共 {len(df)} 行, {len(df.columns)} 列")
            return df

        except pd.errors.EmptyDataError:
            logger.error(f"文件为空: {filepath}")
            return None
        except pd.errors.ParserError as e:
            logger.error(f"解析错误 {filename}: {e}")
            return None
        except Exception as e:
            logger.error(f"加载 {filename} 时发生错误: {e}")
            return None

    def load_sales_data(self) -> Optional[pd.DataFrame]:
        """加载销售数据"""
        df = self.load_csv('sales.csv')
        if df is not None:
            try:
                df['date'] = pd.to_datetime(df['date'])
                df['year'] = df['date'].dt.year
                df['month'] = df['date'].dt.month
                df['quarter'] = df['date'].dt.quarter
                df['weekday'] = df['date'].dt.weekday
                df['year_month'] = df['date'].dt.to_period('M')
            except Exception as e:
                logger.error(f"处理销售数据日期时出错: {e}")
        self.data['sales'] = df
        return df

    def load_products_data(self) -> Optional[pd.DataFrame]:
        """加载商品数据"""
        df = self.load_csv('products.csv')
        self.data['products'] = df
        return df

    def load_customers_data(self) -> Optional[pd.DataFrame]:
        """加载客户数据"""
        df = self.load_csv('customers.csv')
        if df is not None:
            try:
                df['register_date'] = pd.to_datetime(df['register_date'])
            except Exception as e:
                logger.error(f"处理客户数据日期时出错: {e}")
        self.data['customers'] = df
        return df

    def load_all_data(self) -> Dict[str, Optional[pd.DataFrame]]:
        """加载所有数据"""
        self.load_sales_data()
        self.load_products_data()
        self.load_customers_data()
        return self.data

    def merge_data(self) -> Optional[pd.DataFrame]:
        """
        合并所有数据

        Returns:
            合并后的DataFrame
        """
        if not self.data:
            self.load_all_data()

        sales = self.data.get('sales')
        products = self.data.get('products')
        customers = self.data.get('customers')

        if sales is None:
            logger.error("销售数据未加载")
            return None

        merged_df = sales.copy()

        if products is not None:
            try:
                merged_df = merged_df.merge(
                    products,
                    on='product_id',
                    how='left',
                    suffixes=('', '_product')
                )
                logger.info("成功合并商品数据")
            except Exception as e:
                logger.error(f"合并商品数据时出错: {e}")

        if customers is not None:
            try:
                merged_df = merged_df.merge(
                    customers,
                    on='customer_id',
                    how='left',
                    suffixes=('', '_customer')
                )
                logger.info("成功合并客户数据")
            except Exception as e:
                logger.error(f"合并客户数据时出错: {e}")

        return merged_df

    def validate_data(self) -> Dict[str, bool]:
        """
        验证数据完整性

        Returns:
            各数据集的验证结果
        """
        results = {}

        for name, df in self.data.items():
            if df is None:
                results[name] = False
                continue

            checks = {
                'has_data': len(df) > 0,
                'no_empty': not df.empty,
                'has_columns': len(df.columns) > 0
            }

            missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
            checks['low_missing'] = missing_ratio < 0.5

            results[name] = all(checks.values())

        return results

    def get_data_summary(self) -> Dict:
        """获取数据摘要信息"""
        summary = {}

        for name, df in self.data.items():
            if df is not None:
                summary[name] = {
                    'rows': len(df),
                    'columns': len(df.columns),
                    'column_names': list(df.columns),
                    'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
                }

        return summary


def load_and_prepare_data(data_dir: str = 'data') -> Tuple[Optional[pd.DataFrame], Dict]:
    """
    便捷函数：加载并准备数据

    Args:
        data_dir: 数据目录

    Returns:
        (merged_df, data_dict)
    """
    loader = DataLoader(data_dir)
    data_dict = loader.load_all_data()
    merged_df = loader.merge_data()

    validation = loader.validate_data()
    logger.info(f"数据验证结果: {validation}")

    summary = loader.get_data_summary()
    logger.info(f"数据摘要: {summary}")

    return merged_df, data_dict
