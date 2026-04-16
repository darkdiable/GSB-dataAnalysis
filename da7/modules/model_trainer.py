import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
import os
import json


class ModelTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.label_encoders = {}
        self.scaler = None
        self.feature_importance = None
        self.metrics = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.y_pred_proba = None
        
    def load_data(self, filepath='data/processed_data.csv'):
        full_path = os.path.join('/Users/bilei/work/LargeModelAnnotation/GBS/260410/GSB-dataAnalysis/da7', filepath)
        self.data = pd.read_csv(full_path)
        print(f"数据加载成功！形状: {self.data.shape}")
        return self.data
    
    def prepare_features(self, feature_cols=None, target_col='is_completed'):
        print("\n准备特征...")
        
        if feature_cols is None:
            feature_cols = [
                'watch_duration_minutes', 'quiz_score', 'learning_efficiency',
                'category_avg_completion', 'days_since_registration', 'user_completion_rate'
            ]
        
        self.feature_cols = feature_cols
        self.target_col = target_col
        
        X = self.data[feature_cols].copy()
        y = self.data[target_col].copy()
        
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
        
        X = X.fillna(X.median())
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        print(f"训练集大小: {self.X_train.shape}")
        print(f"测试集大小: {self.X_test.shape}")
        print(f"特征: {feature_cols}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_model(self, n_estimators=100, max_depth=10):
        print("\n训练随机森林模型...")
        
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=self.random_state,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        self.model.fit(self.X_train, self.y_train)
        
        print(f"模型训练完成！")
        print(f"  估计器数量: {n_estimators}")
        print(f"  最大深度: {max_depth}")
        
        return self.model
    
    def evaluate_model(self):
        print("\n模型评估...")
        
        self.y_pred = self.model.predict(self.X_test)
        self.y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        accuracy = accuracy_score(self.y_test, self.y_pred)
        precision = precision_score(self.y_test, self.y_pred, average='weighted')
        recall = recall_score(self.y_test, self.y_pred, average='weighted')
        f1 = f1_score(self.y_test, self.y_pred, average='weighted')
        auc = roc_auc_score(self.y_test, self.y_pred_proba)
        
        self.metrics = {
            '准确率 (Accuracy)': round(accuracy, 4),
            '精确率 (Precision)': round(precision, 4),
            '召回率 (Recall)': round(recall, 4),
            'F1分数 (F1 Score)': round(f1, 4),
            'AUC分数': round(auc, 4)
        }
        
        print("\n" + "="*50)
        print("模型评估报告")
        print("="*50)
        for metric, value in self.metrics.items():
            print(f"{metric}: {value}")
        
        print("\n分类报告:")
        print(classification_report(self.y_test, self.y_pred, target_names=['未完成', '已完成']))
        
        print("混淆矩阵:")
        cm = confusion_matrix(self.y_test, self.y_pred)
        print(cm)
        
        return self.metrics
    
    def get_feature_importance(self):
        print("\n特征重要性分析...")
        
        importance = self.model.feature_importances_
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print("\n特征重要性排序:")
        print("-" * 40)
        for idx, row in self.feature_importance.iterrows():
            print(f"{row['feature']:30s}: {row['importance']:.4f}")
        
        return self.feature_importance
    
    def save_results(self, output_dir='output'):
        full_output_dir = os.path.join('/Users/bilei/work/LargeModelAnnotation/GBS/260410/GSB-dataAnalysis/da7', output_dir)
        os.makedirs(full_output_dir, exist_ok=True)
        
        metrics_path = os.path.join(full_output_dir, 'model_metrics.json')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, ensure_ascii=False, indent=2)
        
        importance_path = os.path.join(full_output_dir, 'feature_importance.csv')
        self.feature_importance.to_csv(importance_path, index=False, encoding='utf-8-sig')
        
        print(f"\n结果已保存至:")
        print(f"  - {metrics_path}")
        print(f"  - {importance_path}")
        
        return full_output_dir
    
    def run_full_pipeline(self, feature_cols=None):
        self.load_data()
        self.prepare_features(feature_cols)
        self.train_model()
        self.evaluate_model()
        self.get_feature_importance()
        self.save_results()
        
        return {
            'metrics': self.metrics,
            'feature_importance': self.feature_importance,
            'model': self.model
        }


if __name__ == '__main__':
    trainer = ModelTrainer()
    results = trainer.run_full_pipeline()
