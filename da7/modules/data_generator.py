import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os


class DataGenerator:
    def __init__(self, n_samples=5000, random_state=42):
        self.n_samples = n_samples
        self.random_state = random_state
        np.random.seed(random_state)
        
        self.course_categories = [
            '编程开发', '数据科学', '人工智能', '产品设计',
            '市场营销', '语言学习', '商业管理', '职业技能'
        ]
        
    def generate_user_ids(self):
        return [f'USER_{str(i).zfill(6)}' for i in range(1, self.n_samples + 1)]
    
    def generate_course_ids(self):
        return [f'COURSE_{str(i).zfill(5)}' for i in range(1, 201)]
    
    def generate_data(self):
        print(f"正在生成 {self.n_samples} 条模拟数据...")
        
        user_ids = self.generate_user_ids()
        course_ids = self.generate_course_ids()
        
        data = {
            'user_id': np.random.choice(user_ids, self.n_samples),
            'course_id': np.random.choice(course_ids, self.n_samples),
            'course_category': np.random.choice(self.course_categories, self.n_samples),
            'watch_duration_minutes': [],
            'quiz_score': [],
            'is_completed': [],
            'registration_date': [],
            'last_activity_date': []
        }
        
        base_date = datetime(2024, 1, 1)
        
        for i in range(self.n_samples):
            category = data['course_category'][i]
            
            if category in ['编程开发', '数据科学', '人工智能']:
                watch_duration = np.random.normal(120, 40)
                quiz_score_base = 75
            elif category in ['产品设计', '商业管理']:
                watch_duration = np.random.normal(90, 30)
                quiz_score_base = 80
            else:
                watch_duration = np.random.normal(60, 25)
                quiz_score_base = 85
            
            watch_duration = max(5, min(300, watch_duration))
            data['watch_duration_minutes'].append(round(watch_duration, 2))
            
            quiz_score = np.random.normal(quiz_score_base, 15)
            quiz_score = max(0, min(100, quiz_score))
            data['quiz_score'].append(round(quiz_score, 2))
            
            completion_prob = 0.3 + 0.4 * (watch_duration / 300) + 0.3 * (quiz_score / 100)
            is_completed = np.random.random() < completion_prob
            data['is_completed'].append(int(is_completed))
            
            reg_days = np.random.randint(0, 365)
            reg_date = base_date + timedelta(days=int(reg_days))
            data['registration_date'].append(reg_date.strftime('%Y-%m-%d'))
            
            last_days = np.random.randint(0, 30)
            last_date = datetime.now() - timedelta(days=int(last_days))
            data['last_activity_date'].append(last_date.strftime('%Y-%m-%d'))
        
        df = pd.DataFrame(data)
        
        missing_indices = np.random.choice(df.index, size=int(self.n_samples * 0.02), replace=False)
        df.loc[missing_indices[:len(missing_indices)//2], 'quiz_score'] = np.nan
        df.loc[missing_indices[len(missing_indices)//2:], 'watch_duration_minutes'] = np.nan
        
        outlier_indices = np.random.choice(df.index, size=int(self.n_samples * 0.01), replace=False)
        for idx in outlier_indices[:len(outlier_indices)//3]:
            df.loc[idx, 'watch_duration_minutes'] = np.random.choice([500, 600, 2, 1])
        for idx in outlier_indices[len(outlier_indices)//3:2*len(outlier_indices)//3]:
            df.loc[idx, 'quiz_score'] = np.random.choice([150, -20, 200])
        
        print(f"数据生成完成！数据形状: {df.shape}")
        print(f"\n数据预览:")
        print(df.head(10))
        print(f"\n数据统计:")
        print(df.describe())
        
        return df
    
    def save_data(self, df, filepath='data/raw_data.csv'):
        full_path = os.path.join('/Users/bilei/work/LargeModelAnnotation/GBS/260410/GSB-dataAnalysis/da7', filepath)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        df.to_csv(full_path, index=False, encoding='utf-8-sig')
        print(f"\n数据已保存至: {full_path}")
        return full_path


if __name__ == '__main__':
    generator = DataGenerator(n_samples=5000)
    df = generator.generate_data()
    generator.save_data(df)
