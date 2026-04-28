import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


class DataAnalyzer:
    """
    数据分析类，包含RFM用户分层、月度销售额趋势、类别贡献度、价格与销量相关性分析
    """
    
    def __init__(self, users, products, orders, order_details):
        self.users = users
        self.products = products
        self.orders = orders
        self.order_details = order_details
        self.analysis_results = {}
    
    def analyze_rfm(self):
        """
        RFM用户分层分析
        """
        print("\n开始RFM用户分层分析...")
        
        # 确保订单日期是datetime类型
        if not pd.api.types.is_datetime64_any_dtype(self.orders['order_date']):
            self.orders['order_date'] = pd.to_datetime(self.orders['order_date'])
        
        # 计算RFM指标
        # R: 最近一次购买距离现在的天数
        # F: 购买频率
        # M: 购买总金额
        
        # 分析日期设为数据中的最新日期+1天
        analysis_date = self.orders['order_date'].max() + pd.Timedelta(days=1)
        
        # 按用户聚合
        rfm = self.orders.groupby('user_id').agg(
            # R: 最近一次购买日期与分析日期的差值
            Recency=('order_date', lambda x: (analysis_date - x.max()).days),
            # F: 订单数量
            Frequency=('order_id', 'count'),
            # M: 总金额
            Monetary=('total_amount', 'sum')
        ).reset_index()
        
        # 移除总金额为0的用户
        rfm = rfm[rfm['Monetary'] > 0]
        
        # 计算RFM得分（使用四分位数）
        # R: 越小越好（最近购买），所以反向打分
        # F: 越大越好
        # M: 越大越好
        
        def r_score(x, quantiles):
            if x <= quantiles[0.25]:
                return 4
            elif x <= quantiles[0.5]:
                return 3
            elif x <= quantiles[0.75]:
                return 2
            else:
                return 1
        
        def fm_score(x, quantiles):
            if x <= quantiles[0.25]:
                return 1
            elif x <= quantiles[0.5]:
                return 2
            elif x <= quantiles[0.75]:
                return 3
            else:
                return 4
        
        # 计算四分位数
        r_quantiles = rfm['Recency'].quantile([0.25, 0.5, 0.75])
        f_quantiles = rfm['Frequency'].quantile([0.25, 0.5, 0.75])
        m_quantiles = rfm['Monetary'].quantile([0.25, 0.5, 0.75])
        
        # 计算RFM得分
        rfm['R_Score'] = rfm['Recency'].apply(lambda x: r_score(x, r_quantiles))
        rfm['F_Score'] = rfm['Frequency'].apply(lambda x: fm_score(x, f_quantiles))
        rfm['M_Score'] = rfm['Monetary'].apply(lambda x: fm_score(x, m_quantiles))
        
        # 计算总得分
        rfm['RFM_Total'] = rfm['R_Score'] + rfm['F_Score'] + rfm['M_Score']
        
        # 用户分层
        def segment_user(row):
            r = row['R_Score']
            f = row['F_Score']
            m = row['M_Score']
            
            # 重要价值用户：R高，F高，M高
            if r >= 3 and f >= 3 and m >= 3:
                return '重要价值用户'
            # 重要发展用户：R高，F低，M高
            elif r >= 3 and f < 3 and m >= 3:
                return '重要发展用户'
            # 重要保持用户：R低，F高，M高
            elif r < 3 and f >= 3 and m >= 3:
                return '重要保持用户'
            # 重要挽留用户：R低，F低，M高
            elif r < 3 and f < 3 and m >= 3:
                return '重要挽留用户'
            # 一般价值用户：R高，F高，M低
            elif r >= 3 and f >= 3 and m < 3:
                return '一般价值用户'
            # 一般发展用户：R高，F低，M低
            elif r >= 3 and f < 3 and m < 3:
                return '一般发展用户'
            # 一般保持用户：R低，F高，M低
            elif r < 3 and f >= 3 and m < 3:
                return '一般保持用户'
            # 一般挽留用户：R低，F低，M低
            else:
                return '一般挽留用户'
        
        rfm['Segment'] = rfm.apply(segment_user, axis=1)
        
        # 统计各分层用户数量
        segment_stats = rfm.groupby('Segment').agg(
            用户数=('user_id', 'count'),
            平均消费金额=('Monetary', 'mean'),
            平均消费次数=('Frequency', 'mean'),
            平均最近消费天数=('Recency', 'mean')
        ).reset_index()
        
        segment_stats['用户数占比'] = segment_stats['用户数'] / segment_stats['用户数'].sum() * 100
        
        # 保存结果
        self.analysis_results['rfm'] = {
            'rfm_data': rfm,
            'segment_stats': segment_stats,
            'quantiles': {
                'Recency': r_quantiles.to_dict(),
                'Frequency': f_quantiles.to_dict(),
                'Monetary': m_quantiles.to_dict()
            }
        }
        
        print("RFM分析完成。")
        print(f"  - 共分析 {len(rfm)} 个有效用户")
        print(f"  - 分层统计: {segment_stats['用户数'].to_dict()}")
        
        return rfm, segment_stats
    
    def analyze_monthly_sales(self):
        """
        月度销售额趋势分析
        """
        print("\n开始月度销售额趋势分析...")
        
        # 确保订单日期是datetime类型
        if not pd.api.types.is_datetime64_any_dtype(self.orders['order_date']):
            self.orders['order_date'] = pd.to_datetime(self.orders['order_date'])
        
        # 提取月份
        self.orders['order_month'] = self.orders['order_date'].dt.to_period('M')
        
        # 按月聚合销售额
        monthly_sales = self.orders.groupby('order_month').agg(
            销售额=('total_amount', 'sum'),
            订单数=('order_id', 'count'),
            客单价=('total_amount', 'mean')
        ).reset_index()
        
        # 转换为字符串格式
        monthly_sales['order_month'] = monthly_sales['order_month'].astype(str)
        
        # 计算环比增长率
        monthly_sales['销售额环比增长率'] = monthly_sales['销售额'].pct_change() * 100
        
        # 计算订单数环比增长率
        monthly_sales['订单数环比增长率'] = monthly_sales['订单数'].pct_change() * 100
        
        # 保存结果
        self.analysis_results['monthly_sales'] = {
            'monthly_data': monthly_sales,
            'total_sales': monthly_sales['销售额'].sum(),
            'total_orders': monthly_sales['订单数'].sum(),
            'avg_monthly_sales': monthly_sales['销售额'].mean(),
            'peak_month': monthly_sales.loc[monthly_sales['销售额'].idxmax(), 'order_month'],
            'peak_sales': monthly_sales['销售额'].max(),
            'low_month': monthly_sales.loc[monthly_sales['销售额'].idxmin(), 'order_month'],
            'low_sales': monthly_sales['销售额'].min()
        }
        
        print("月度销售额趋势分析完成。")
        print(f"  - 总销售额: {monthly_sales['销售额'].sum():.2f}")
        print(f"  - 总订单数: {monthly_sales['订单数'].sum()}")
        print(f"  - 峰值月份: {self.analysis_results['monthly_sales']['peak_month']}")
        
        return monthly_sales
    
    def analyze_category_contribution(self):
        """
        产品类别贡献度分析
        """
        print("\n开始产品类别贡献度分析...")
        
        # 合并订单详情和产品信息
        order_product = self.order_details.merge(
            self.products[['product_id', 'category', 'brand']],
            on='product_id',
            how='left'
        )
        
        # 按类别聚合
        category_stats = order_product.groupby('category').agg(
            销售额=('line_amount', 'sum'),
            销量=('quantity', 'sum'),
            订单数=('order_id', 'nunique'),
            产品数=('product_id', 'nunique')
        ).reset_index()
        
        # 计算各指标占比
        total_sales = category_stats['销售额'].sum()
        total_quantity = category_stats['销量'].sum()
        
        category_stats['销售额占比'] = category_stats['销售额'] / total_sales * 100
        category_stats['销量占比'] = category_stats['销量'] / total_quantity * 100
        
        # 计算平均单价
        category_stats['平均单价'] = category_stats['销售额'] / category_stats['销量']
        
        # 按销售额排序
        category_stats = category_stats.sort_values('销售额', ascending=False).reset_index(drop=True)
        
        # 按品牌聚合（补充分析）
        brand_stats = order_product.groupby('brand').agg(
            销售额=('line_amount', 'sum'),
            销量=('quantity', 'sum')
        ).reset_index()
        
        brand_stats['销售额占比'] = brand_stats['销售额'] / brand_stats['销售额'].sum() * 100
        brand_stats = brand_stats.sort_values('销售额', ascending=False).reset_index(drop=True)
        
        # 保存结果
        self.analysis_results['category_contribution'] = {
            'category_stats': category_stats,
            'brand_stats': brand_stats,
            'top_category': category_stats.iloc[0]['category'],
            'top_category_sales': category_stats.iloc[0]['销售额'],
            'top_category_share': category_stats.iloc[0]['销售额占比']
        }
        
        print("产品类别贡献度分析完成。")
        print(f"  - 类别数量: {len(category_stats)}")
        print(f"  - 销售冠军类别: {category_stats.iloc[0]['category']}")
        print(f"  - 占比: {category_stats.iloc[0]['销售额占比']:.2f}%")
        
        return category_stats, brand_stats
    
    def analyze_price_quantity_correlation(self):
        """
        价格与销量相关性分析
        """
        print("\n开始价格与销量相关性分析...")
        
        # 合并订单详情和产品信息
        # 注意：order_details和products都有unit_price列，需要重命名避免冲突
        products_for_merge = self.products[['product_id', 'category', 'unit_price']].rename(
            columns={'unit_price': 'product_unit_price'}
        )
        
        order_product = self.order_details.merge(
            products_for_merge,
            on='product_id',
            how='left'
        )
        
        # 按产品聚合，计算每个产品的总销量和平均价格
        # 使用产品表中的定价（product_unit_price）而不是订单详情中的单价
        product_stats = order_product.groupby(['product_id', 'category', 'product_unit_price']).agg(
            总销量=('quantity', 'sum'),
            订单数=('order_id', 'nunique')
        ).reset_index()
        
        # 重命名列以便后续使用
        product_stats = product_stats.rename(columns={'product_unit_price': 'unit_price'})
        
        # 计算相关系数
        price_quantity_corr = product_stats['unit_price'].corr(product_stats['总销量'])
        price_orders_corr = product_stats['unit_price'].corr(product_stats['订单数'])
        
        # 按类别分析相关性
        category_correlations = {}
        for category in product_stats['category'].unique():
            category_data = product_stats[product_stats['category'] == category]
            if len(category_data) >= 5:  # 至少5个样本
                corr = category_data['unit_price'].corr(category_data['总销量'])
                category_correlations[category] = {
                    '相关系数': corr,
                    '产品数': len(category_data),
                    '平均价格': category_data['unit_price'].mean(),
                    '平均销量': category_data['总销量'].mean()
                }
        
        # 价格区间分析
        # 定义价格区间
        def get_price_range(price):
            if price < 50:
                return '低价 (<50)'
            elif price < 200:
                return '中低价 (50-200)'
            elif price < 500:
                return '中高价 (200-500)'
            elif price < 2000:
                return '高价 (500-2000)'
            else:
                return '超高价 (≥2000)'
        
        product_stats['价格区间'] = product_stats['unit_price'].apply(get_price_range)
        
        price_range_stats = product_stats.groupby('价格区间').agg(
            产品数=('product_id', 'nunique'),
            总销量=('总销量', 'sum'),
            平均销量=('总销量', 'mean'),
            平均价格=('unit_price', 'mean')
        ).reset_index()
        
        # 保存结果
        self.analysis_results['price_quantity_correlation'] = {
            'product_stats': product_stats,
            'price_range_stats': price_range_stats,
            'category_correlations': category_correlations,
            'overall_correlation': {
                '价格-销量相关系数': price_quantity_corr,
                '价格-订单数相关系数': price_orders_corr
            }
        }
        
        print("价格与销量相关性分析完成。")
        print(f"  - 整体价格-销量相关系数: {price_quantity_corr:.4f}")
        print(f"  - 整体价格-订单数相关系数: {price_orders_corr:.4f}")
        
        return product_stats, price_range_stats, category_correlations
    
    def run_all_analyses(self):
        """
        执行所有分析
        """
        print("\n" + "="*50)
        print("开始数据分析")
        print("="*50)
        
        self.analyze_rfm()
        self.analyze_monthly_sales()
        self.analyze_category_contribution()
        self.analyze_price_quantity_correlation()
        
        print("\n" + "="*50)
        print("数据分析完成")
        print("="*50)
        
        return self.analysis_results
    
    def get_analysis_results(self):
        """
        获取分析结果
        """
        return self.analysis_results


if __name__ == '__main__':
    from data_generator import load_datasets
    from data_cleaner import DataCleaner
    
    # 加载并清洗数据
    users, products, orders, order_details = load_datasets()
    cleaner = DataCleaner(users, products, orders, order_details)
    cleaned_users, cleaned_products, cleaned_orders, cleaned_order_details = cleaner.clean_all()
    
    # 执行分析
    analyzer = DataAnalyzer(cleaned_users, cleaned_products, cleaned_orders, cleaned_order_details)
    results = analyzer.run_all_analyses()
    
    # 打印关键指标
    print("\n" + "="*50)
    print("分析关键指标汇总")
    print("="*50)
    
    if 'rfm' in results:
        segment_stats = results['rfm']['segment_stats']
        print(f"\nRFM用户分层:")
        for _, row in segment_stats.iterrows():
            print(f"  - {row['Segment']}: {row['用户数']}人 ({row['用户数占比']:.1f}%)")
    
    if 'monthly_sales' in results:
        ms = results['monthly_sales']
        print(f"\n月度销售趋势:")
        print(f"  - 总销售额: ¥{ms['total_sales']:,.2f}")
        print(f"  - 总订单数: {ms['total_orders']}")
        print(f"  - 峰值月份: {ms['peak_month']} (¥{ms['peak_sales']:,.2f})")
    
    if 'category_contribution' in results:
        cc = results['category_contribution']
        print(f"\n类别贡献度:")
        print(f"  - 销售冠军: {cc['top_category']} ({cc['top_category_share']:.1f}%)")
