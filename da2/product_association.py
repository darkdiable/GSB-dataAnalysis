"""
商品关联分析模块
提供关联规则挖掘、商品推荐、品类分析等功能
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict, Counter
import logging
from itertools import combinations

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ProductAssociationAnalyzer:
    """商品关联分析器"""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.results = {}

    def market_basket_analysis(self, min_support: float = 0.01, min_confidence: float = 0.5) -> pd.DataFrame:
        """
        购物篮分析（关联规则挖掘）

        Args:
            min_support: 最小支持度
            min_confidence: 最小置信度
        """
        try:
            # 构建购物篮（每个订单的商品集合）
            baskets = self.df.groupby('transaction_id')['product_id'].apply(list).values
            total_baskets = len(baskets)

            # 计算单项支持度
            item_counts = Counter()
            for basket in baskets:
                item_counts.update(basket)

            frequent_items = {item: count / total_baskets
                             for item, count in item_counts.items()
                             if count / total_baskets >= min_support}

            # 计算二项集支持度
            pair_counts = Counter()
            for basket in baskets:
                if len(basket) >= 2:
                    for pair in combinations(basket, 2):
                        if pair[0] in frequent_items and pair[1] in frequent_items:
                            pair_counts[tuple(sorted(pair))] += 1

            frequent_pairs = {pair: count / total_baskets
                             for pair, count in pair_counts.items()
                             if count / total_baskets >= min_support}

            # 生成关联规则
            rules = []
            for (item_a, item_b), pair_support in frequent_pairs.items():
                support_a = frequent_items[item_a]
                support_b = frequent_items[item_b]

                # 置信度: P(B|A) = P(A,B) / P(A)
                confidence_a_to_b = pair_support / support_a
                confidence_b_to_a = pair_support / support_b

                # 提升度: Lift(A->B) = P(B|A) / P(B)
                lift_a_to_b = confidence_a_to_b / support_b
                lift_b_to_a = confidence_b_to_a / support_a

                if confidence_a_to_b >= min_confidence:
                    rules.append({
                        'antecedent': item_a,
                        'consequent': item_b,
                        'support': pair_support,
                        'confidence': confidence_a_to_b,
                        'lift': lift_a_to_b
                    })

                if confidence_b_to_a >= min_confidence:
                    rules.append({
                        'antecedent': item_b,
                        'consequent': item_a,
                        'support': pair_support,
                        'confidence': confidence_b_to_a,
                        'lift': lift_b_to_a
                    })

            rules_df = pd.DataFrame(rules)
            if not rules_df.empty:
                rules_df = rules_df.sort_values('lift', ascending=False)

            self.results['association_rules'] = rules_df
            self.results['frequent_items'] = frequent_items
            logger.info(f"购物篮分析完成，发现{len(rules_df)}条关联规则")
            return rules_df

        except Exception as e:
            logger.error(f"购物篮分析出错: {e}")
            return pd.DataFrame()

    def product_co_occurrence_matrix(self) -> pd.DataFrame:
        """商品共现矩阵"""
        try:
            # 获取商品列表
            products = self.df['product_id'].unique()

            # 构建共现矩阵
            co_occurrence = pd.DataFrame(0, index=products, columns=products)

            baskets = self.df.groupby('transaction_id')['product_id'].apply(list)

            for basket in baskets:
                if len(basket) >= 2:
                    for prod_a, prod_b in combinations(basket, 2):
                        co_occurrence.loc[prod_a, prod_b] += 1
                        co_occurrence.loc[prod_b, prod_a] += 1

            self.results['co_occurrence'] = co_occurrence
            logger.info("商品共现矩阵构建完成")
            return co_occurrence

        except Exception as e:
            logger.error(f"商品共现矩阵构建出错: {e}")
            return pd.DataFrame()

    def category_association(self) -> pd.DataFrame:
        """品类关联分析"""
        try:
            if 'category' not in self.df.columns:
                logger.warning("缺少category列")
                return pd.DataFrame()

            # 构建品类购物篮
            baskets = self.df.groupby('transaction_id')['category'].apply(lambda x: list(set(x))).values
            total_baskets = len(baskets)

            # 计算品类共现
            category_pairs = Counter()
            for basket in baskets:
                if len(basket) >= 2:
                    for pair in combinations(basket, 2):
                        category_pairs[tuple(sorted(pair))] += 1

            # 转换为DataFrame
            category_assoc = []
            for (cat_a, cat_b), count in category_pairs.most_common():
                category_assoc.append({
                    'category_a': cat_a,
                    'category_b': cat_b,
                    'co_occurrence': count,
                    'support': count / total_baskets
                })

            category_df = pd.DataFrame(category_assoc)
            self.results['category_association'] = category_df
            logger.info("品类关联分析完成")
            return category_df

        except Exception as e:
            logger.error(f"品类关联分析出错: {e}")
            return pd.DataFrame()

    def product_recommendations(self, product_id: str, top_n: int = 5) -> pd.DataFrame:
        """
        基于关联规则的商品推荐

        Args:
            product_id: 目标商品ID
            top_n: 推荐数量
        """
        try:
            if 'association_rules' not in self.results:
                self.market_basket_analysis()

            rules = self.results['association_rules']

            if rules.empty:
                return pd.DataFrame()

            # 查找以该商品为先导的关联规则
            recommendations = rules[rules['antecedent'] == product_id].head(top_n)

            # 获取商品名称
            if 'product_name' in self.df.columns:
                product_names = dict(zip(self.df['product_id'], self.df['product_name']))
                recommendations['antecedent_name'] = recommendations['antecedent'].map(product_names)
                recommendations['consequent_name'] = recommendations['consequent'].map(product_names)

            return recommendations

        except Exception as e:
            logger.error(f"商品推荐出错: {e}")
            return pd.DataFrame()

    def cross_sell_opportunities(self, min_lift: float = 1.5) -> pd.DataFrame:
        """
        交叉销售机会分析

        Args:
            min_lift: 最小提升度阈值
        """
        try:
            if 'association_rules' not in self.results:
                self.market_basket_analysis()

            rules = self.results['association_rules']

            if rules.empty:
                return pd.DataFrame()

            # 筛选高提升度的规则
            cross_sell = rules[rules['lift'] >= min_lift].copy()

            # 添加商品信息
            if 'product_name' in self.df.columns:
                product_info = self.df.groupby('product_id').agg({
                    'product_name': 'first',
                    'category': 'first'
                }).reset_index()

                cross_sell = cross_sell.merge(
                    product_info.rename(columns={'product_id': 'antecedent', 'product_name': 'antecedent_name', 'category': 'antecedent_category'}),
                    on='antecedent',
                    how='left'
                )
                cross_sell = cross_sell.merge(
                    product_info.rename(columns={'product_id': 'consequent', 'product_name': 'consequent_name', 'category': 'consequent_category'}),
                    on='consequent',
                    how='left'
                )

            self.results['cross_sell'] = cross_sell
            logger.info(f"交叉销售机会分析完成，发现{len(cross_sell)}个机会")
            return cross_sell

        except Exception as e:
            logger.error(f"交叉销售分析出错: {e}")
            return pd.DataFrame()

    def category_performance(self) -> pd.DataFrame:
        """品类表现分析"""
        try:
            if 'category' not in self.df.columns:
                logger.warning("缺少category列")
                return pd.DataFrame()

            category_perf = self.df.groupby('category').agg({
                'total_amount': ['sum', 'mean'],
                'quantity': ['sum', 'mean'],
                'transaction_id': 'count',
                'product_id': 'nunique',
                'customer_id': 'nunique'
            }).reset_index()

            category_perf.columns = ['category', 'total_revenue', 'avg_order_value',
                                    'total_quantity', 'avg_quantity', 'transactions',
                                    'unique_products', 'unique_customers']

            # 计算占比
            category_perf['revenue_share'] = category_perf['total_revenue'] / category_perf['total_revenue'].sum() * 100
            category_perf['transaction_share'] = category_perf['transactions'] / category_perf['transactions'].sum() * 100

            # 计算品类效率指标
            category_perf['revenue_per_product'] = category_perf['total_revenue'] / category_perf['unique_products']
            category_perf['revenue_per_customer'] = category_perf['total_revenue'] / category_perf['unique_customers']

            category_perf = category_perf.sort_values('total_revenue', ascending=False)

            self.results['category_performance'] = category_perf
            logger.info("品类表现分析完成")
            return category_perf

        except Exception as e:
            logger.error(f"品类表现分析出错: {e}")
            return pd.DataFrame()

    def brand_analysis(self) -> pd.DataFrame:
        """品牌分析"""
        try:
            if 'brand' not in self.df.columns:
                logger.warning("缺少brand列")
                return pd.DataFrame()

            brand_perf = self.df.groupby('brand').agg({
                'total_amount': ['sum', 'mean'],
                'quantity': 'sum',
                'transaction_id': 'count',
                'product_id': 'nunique'
            }).reset_index()

            brand_perf.columns = ['brand', 'total_revenue', 'avg_order_value',
                                 'total_quantity', 'transactions', 'product_count']

            brand_perf['revenue_share'] = brand_perf['total_revenue'] / brand_perf['total_revenue'].sum() * 100
            brand_perf = brand_perf.sort_values('total_revenue', ascending=False)

            self.results['brand_performance'] = brand_perf
            logger.info("品牌分析完成")
            return brand_perf

        except Exception as e:
            logger.error(f"品牌分析出错: {e}")
            return pd.DataFrame()

    def product_affinity_network(self, min_cooccurrence: int = 5) -> Dict:
        """
        构建商品关联网络

        Args:
            min_cooccurrence: 最小共现次数
        """
        try:
            if 'co_occurrence' not in self.results:
                self.product_co_occurrence_matrix()

            co_occurrence = self.results['co_occurrence']

            # 构建网络边
            edges = []
            products = co_occurrence.index

            for i, prod_a in enumerate(products):
                for prod_b in products[i+1:]:
                    weight = co_occurrence.loc[prod_a, prod_b]
                    if weight >= min_cooccurrence:
                        edges.append({
                            'source': prod_a,
                            'target': prod_b,
                            'weight': weight
                        })

            # 计算每个商品的连接数（度）
            nodes = []
            product_stats = self.df.groupby('product_id').agg({
                'total_amount': 'sum',
                'quantity': 'sum',
                'transaction_id': 'count'
            })

            for prod in products:
                degree = (co_occurrence.loc[prod] >= min_cooccurrence).sum()
                if degree > 0:
                    node = {
                        'id': prod,
                        'degree': degree,
                        'total_revenue': product_stats.loc[prod, 'total_amount'],
                        'total_quantity': product_stats.loc[prod, 'quantity'],
                        'transactions': product_stats.loc[prod, 'transaction_id']
                    }

                    # 添加商品名称
                    if 'product_name' in self.df.columns:
                        name = self.df[self.df['product_id'] == prod]['product_name'].iloc[0]
                        node['name'] = name

                    nodes.append(node)

            network = {
                'nodes': nodes,
                'edges': edges
            }

            self.results['affinity_network'] = network
            logger.info(f"商品关联网络构建完成，包含{len(nodes)}个节点，{len(edges)}条边")
            return network

        except Exception as e:
            logger.error(f"商品关联网络构建出错: {e}")
            return {'nodes': [], 'edges': []}

    def run_all_analysis(self) -> Dict:
        """运行所有商品关联分析"""
        logger.info("开始商品关联分析...")

        self.market_basket_analysis()
        self.product_co_occurrence_matrix()
        self.category_association()
        self.cross_sell_opportunities()
        self.category_performance()
        self.brand_analysis()
        self.product_affinity_network()

        logger.info("商品关联分析全部完成")
        return self.results

    def get_results(self) -> Dict:
        """获取分析结果"""
        return self.results
