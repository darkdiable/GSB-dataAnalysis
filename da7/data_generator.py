import numpy as np
import pandas as pd
import os


class DataGenerator:
    def __init__(self, n_samples=5000, random_state=42):
        self.n_samples = n_samples
        self.random_state = random_state
        np.random.seed(random_state)
        
        self.course_categories = ['编程开发', '数据科学', '人工智能', '产品设计', '商业管理', '语言学习']
        self.n_users = 500
        self.n_courses = 100
        
    def generate(self):
        user_ids = np.random.randint(1, self.n_users + 1, self.n_samples)
        course_ids = np.random.randint(1, self.n_courses + 1, self.n_samples)
        categories = np.random.choice(self.course_categories, self.n_samples)
        
        watch_duration = np.random.exponential(scale=60, size=self.n_samples)
        watch_duration = np.clip(watch_duration, 5, 300)
        
        base_scores = np.random.normal(loc=70, scale=15, size=self.n_samples)
        score_adjustment = (watch_duration - 60) * 0.1
        quiz_scores = base_scores + score_adjustment
        quiz_scores = np.clip(quiz_scores, 0, 100)
        
        completion_prob = (
            0.3 + 
            0.003 * watch_duration + 
            0.005 * quiz_scores +
            np.random.normal(0, 0.1, self.n_samples)
        )
        completion_prob = np.clip(completion_prob, 0, 1)
        completed = (np.random.random(self.n_samples) < completion_prob).astype(int)
        
        missing_mask = np.random.random(self.n_samples) < 0.02
        quiz_scores_with_missing = quiz_scores.copy()
        quiz_scores_with_missing[missing_mask] = np.nan
        
        watch_duration_with_outliers = watch_duration.copy()
        outlier_mask = np.random.random(self.n_samples) < 0.01
        watch_duration_with_outliers[outlier_mask] = watch_duration[outlier_mask] * 3
        
        data = pd.DataFrame({
            'user_id': user_ids,
            'course_id': course_ids,
            'course_category': categories,
            'watch_duration': watch_duration_with_outliers,
            'quiz_score': quiz_scores_with_missing,
            'completed': completed
        })
        
        return data
    
    def save_data(self, data, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data.to_csv(filepath, index=False)
        print(f"数据已保存至: {filepath}")
        print(f"数据集大小: {len(data)} 条记录")


def generate_course_data(n_samples=5000, save_path=None, random_state=42):
    generator = DataGenerator(n_samples=n_samples, random_state=random_state)
    data = generator.generate()
    
    if save_path:
        generator.save_data(data, save_path)
    
    return data


if __name__ == "__main__":
    data = generate_course_data(
        n_samples=5000,
        save_path="data/course_data.csv"
    )
    print(data.head())
    print(f"\n数据集形状: {data.shape}")
    print(f"\n缺失值统计:\n{data.isnull().sum()}")
