import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from itertools import combinations
from collections import defaultdict


class ProductAssociationAnalyzer:
    def __init__(self, orders_df: pd.DataFrame, products_df: pd.DataFrame):
        self.orders_df = orders_df.copy()
        self.products_df = products_df.copy()
        self.completed_orders = self.orders_df[self.orders_df['order_status'] == '已完成']
        
    def create_basket_data(self) -> pd.DataFrame:
        basket = self.completed_orders.groupby(['order_id', 'product_id'])['quantity'].sum().unstack(fill_value=0)
        basket_binary = (basket > 0).astype(int)
        return basket_binary
    
    def calculate_support(self, basket_binary: pd.DataFrame, min_support: float = 0.01) -> pd.DataFrame:
        n_transactions = len(basket_binary)
        
        product_support = basket_binary.sum() / n_transactions
        
        support_df = pd.DataFrame({
            'product_id': product_support.index,
            'support': product_support.values
        })
        
        support_df = support_df[support_df['support'] >= min_support]
        
        return support_df.sort_values('support', ascending=False)
    
    def find_product_pairs(self, min_support: float = 0.005) -> pd.DataFrame:
        order_products = self.completed_orders.groupby('order_id')['product_id'].apply(list)
        
        pair_counts = defaultdict(int)
        product_counts = defaultdict(int)
        n_orders = len(order_products)
        
        for products in order_products:
            unique_products = list(set(products))
            for product in unique_products:
                product_counts[product] += 1
            
            for pair in combinations(sorted(unique_products), 2):
                pair_counts[pair] += 1
        
        pair_df = pd.DataFrame([
            {
                'product_a': pair[0],
                'product_b': pair[1],
                'pair_count': count,
                'support': count / n_orders
            }
            for pair, count in pair_counts.items()
        ])
        
        if len(pair_df) > 0:
            pair_df = pair_df[pair_df['support'] >= min_support]
            
            pair_df['support_a'] = pair_df['product_a'].map(product_counts) / n_orders
            pair_df['support_b'] = pair_df['product_b'].map(product_counts) / n_orders
            
            pair_df['confidence_a_to_b'] = pair_df['support'] / pair_df['support_a']
            pair_df['confidence_b_to_a'] = pair_df['support'] / pair_df['support_b']
            
            pair_df['lift'] = pair_df['support'] / (pair_df['support_a'] * pair_df['support_b'])
            
            pair_df = pair_df.sort_values('lift', ascending=False)
        
        return pair_df
    
    def analyze_category_association(self) -> pd.DataFrame:
        order_categories = self.completed_orders.groupby('order_id')['category'].apply(list)
        
        pair_counts = defaultdict(int)
        category_counts = defaultdict(int)
        n_orders = len(order_categories)
        
        for categories in order_categories:
            unique_categories = list(set(categories))
            for category in unique_categories:
                category_counts[category] += 1
            
            for pair in combinations(sorted(unique_categories), 2):
                pair_counts[pair] += 1
        
        pair_df = pd.DataFrame([
            {
                'category_a': pair[0],
                'category_b': pair[1],
                'pair_count': count,
                'support': count / n_orders
            }
            for pair, count in pair_counts.items()
        ])
        
        if len(pair_df) > 0:
            pair_df['support_a'] = pair_df['category_a'].map(category_counts) / n_orders
            pair_df['support_b'] = pair_df['category_b'].map(category_counts) / n_orders
            
            pair_df['confidence_a_to_b'] = pair_df['support'] / pair_df['support_a']
            pair_df['confidence_b_to_a'] = pair_df['support'] / pair_df['support_b']
            
            pair_df['lift'] = pair_df['support'] / (pair_df['support_a'] * pair_df['support_b'])
            
            pair_df = pair_df.sort_values('lift', ascending=False)
        
        return pair_df
    
    def analyze_product_performance(self) -> pd.DataFrame:
        performance = self.completed_orders.groupby('product_id').agg({
            'order_id': 'count',
            'quantity': ['sum', 'mean'],
            'total_amount': ['sum', 'mean'],
            'customer_id': 'nunique'
        })
        
        performance.columns = [
            'order_count', 'total_quantity', 'avg_quantity',
            'total_revenue', 'avg_revenue', 'unique_customers'
        ]
        
        performance['repeat_purchase_rate'] = (
            (performance['order_count'] - performance['unique_customers']) / 
            performance['order_count'] * 100
        ).fillna(0)
        
        performance = performance.merge(
            self.products_df[['product_id', 'product_name', 'category', 'price']],
            on='product_id',
            how='left'
        )
        
        return performance.reset_index(drop=True)
    
    def get_top_products(self, metric: str = 'total_revenue', n: int = 20) -> pd.DataFrame:
        performance = self.analyze_product_performance()
        top_products = performance.nlargest(n, metric)
        return top_products
    
    def analyze_cross_sell_opportunities(self, min_lift: float = 1.5) -> pd.DataFrame:
        product_pairs = self.find_product_pairs()
        
        opportunities = pd.DataFrame()
        
        if len(product_pairs) > 0:
            opportunities = product_pairs[product_pairs['lift'] >= min_lift]
            
            if len(opportunities) > 0:
                opportunities = opportunities.merge(
                    self.products_df[['product_id', 'product_name', 'category']].rename(
                        columns={'product_id': 'product_a', 'product_name': 'product_a_name', 'category': 'category_a'}
                    ),
                    on='product_a',
                    how='left'
                )
                
                opportunities = opportunities.merge(
                    self.products_df[['product_id', 'product_name', 'category']].rename(
                        columns={'product_id': 'product_b', 'product_name': 'product_b_name', 'category': 'category_b'}
                    ),
                    on='product_b',
                    how='left'
                )
        
        return opportunities
    
    def create_co_occurrence_matrix(self) -> pd.DataFrame:
        order_products = self.completed_orders.groupby('order_id')['product_id'].apply(list)
        
        unique_products = self.completed_orders['product_id'].unique()
        n_products = len(unique_products)
        
        co_occurrence = pd.DataFrame(0, index=unique_products, columns=unique_products)
        
        for products in order_products:
            unique_in_order = list(set(products))
            for pair in combinations(unique_in_order, 2):
                co_occurrence.loc[pair[0], pair[1]] += 1
                co_occurrence.loc[pair[1], pair[0]] += 1
        
        return co_occurrence
    
    def get_product_recommendations(self, product_id: str, n: int = 5) -> pd.DataFrame:
        product_pairs = self.find_product_pairs()
        
        if len(product_pairs) == 0:
            return pd.DataFrame()
        
        related = product_pairs[
            (product_pairs['product_a'] == product_id) | 
            (product_pairs['product_b'] == product_id)
        ].copy()
        
        related['related_product'] = related.apply(
            lambda row: row['product_b'] if row['product_a'] == product_id else row['product_a'],
            axis=1
        )
        
        related = related.merge(
            self.products_df[['product_id', 'product_name', 'category']],
            left_on='related_product',
            right_on='product_id',
            how='left'
        )
        
        related = related.sort_values('lift', ascending=False).head(n)
        
        return related[['related_product', 'product_name', 'category', 'lift', 'confidence_a_to_b', 'confidence_b_to_a']]
    
    def get_all_analysis(self) -> Dict:
        basket = self.create_basket_data()
        
        return {
            'product_support': self.calculate_support(basket),
            'product_pairs': self.find_product_pairs(),
            'category_association': self.analyze_category_association(),
            'product_performance': self.analyze_product_performance(),
            'top_products': self.get_top_products(),
            'cross_sell_opportunities': self.analyze_cross_sell_opportunities(),
            'co_occurrence_matrix': self.create_co_occurrence_matrix()
        }
