import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix

class ModelTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.feature_importance = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        
    def prepare_data(self, df, target_col='是否完成课程'):
        X = df.drop([target_col, '用户ID', '课程ID'], axis=1, errors='ignore')
        y = df[target_col]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state, stratify=y
        )
        
        print(f"训练集大小: {self.X_train.shape[0]} 样本")
        print(f"测试集大小: {self.X_test.shape[0]} 样本")
        print(f"特征数量: {self.X_train.shape[1]} 个")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_model(self, n_estimators=100, max_depth=None):
        print("\n" + "="*50)
        print("开始训练随机森林模型...")
        print("="*50)
        
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)
        
        importances = self.model.feature_importances_
        feature_names = self.X_train.columns
        self.feature_importance = pd.DataFrame({
            '特征': feature_names,
            '重要性': importances
        }).sort_values('重要性', ascending=False).reset_index(drop=True)
        
        print("模型训练完成!")
        print("="*50)
        
        return self.model
    
    def evaluate_model(self):
        if self.y_pred is None:
            raise ValueError("请先训练模型!")
            
        print("\n" + "="*60)
        print("模型评估报告")
        print("="*60)
        
        accuracy = accuracy_score(self.y_test, self.y_pred)
        precision = precision_score(self.y_test, self.y_pred)
        recall = recall_score(self.y_test, self.y_pred)
        f1 = f1_score(self.y_test, self.y_pred)
        
        print(f"准确率 (Accuracy):   {accuracy:.4f}")
        print(f"精确率 (Precision):  {precision:.4f}")
        print(f"召回率 (Recall):     {recall:.4f}")
        print(f"F1分数 (F1-Score):   {f1:.4f}")
        
        print("\n" + "-"*60)
        print("详细分类报告:")
        print(classification_report(self.y_test, self.y_pred, target_names=['未完成', '完成']))
        
        print("\n混淆矩阵:")
        cm = confusion_matrix(self.y_test, self.y_pred)
        cm_df = pd.DataFrame(cm, 
                           index=['实际: 未完成', '实际: 完成'], 
                           columns=['预测: 未完成', '预测: 完成'])
        print(cm_df)
        print("="*60)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def show_feature_importance(self, top_n=10):
        if self.feature_importance is None:
            raise ValueError("请先训练模型!")
            
        print("\n" + "="*60)
        print(f"特征重要性排序 (Top {top_n})")
        print("="*60)
        
        display_df = self.feature_importance.head(top_n).copy()
        display_df['重要性'] = display_df['重要性'].round(4)
        print(display_df.to_string(index=False))
        print("="*60)
        
        return self.feature_importance
