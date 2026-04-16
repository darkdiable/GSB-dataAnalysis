import pandas as pd
import numpy as np

class DataGenerator:
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
        
    def generate_data(self, n_samples=5000):
        course_categories = ['编程开发', '数据科学', '人工智能', '产品设计', 
                           '市场营销', '语言学习', '职业发展', '金融理财']
        
        user_ids = np.random.randint(1000, 2000, size=n_samples)
        course_ids = np.random.randint(100, 500, size=n_samples)
        categories = np.random.choice(course_categories, size=n_samples)
        
        watch_duration = np.random.normal(loc=120, scale=60, size=n_samples)
        watch_duration = np.clip(watch_duration, 10, 360)
        
        base_score = 60 + (watch_duration / 360) * 35
        quiz_scores = base_score + np.random.normal(loc=0, scale=10, size=n_samples)
        quiz_scores = np.clip(quiz_scores, 0, 100)
        
        completion_prob = 1 / (1 + np.exp(-(quiz_scores - 70) / 10))
        is_completed = np.random.binomial(1, completion_prob, size=n_samples)
        
        mask_missing = np.random.choice([True, False], size=n_samples, p=[0.02, 0.98])
        watch_duration[mask_missing] = np.nan
        
        mask_outlier = np.random.choice([True, False], size=n_samples, p=[0.01, 0.99])
        watch_duration[mask_outlier] = watch_duration[mask_outlier] * 5
        
        df = pd.DataFrame({
            '用户ID': user_ids,
            '课程ID': course_ids,
            '课程类别': categories,
            '观看时长(分钟)': watch_duration.round(2),
            '测验分数': quiz_scores.round(2),
            '是否完成课程': is_completed
        })
        
        return df
    
    def save_data(self, df, filepath='data/course_data.csv'):
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"数据已保存至: {filepath}")
        
    def load_data(self, filepath='data/course_data.csv'):
        return pd.read_csv(filepath, encoding='utf-8-sig')
