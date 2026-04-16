import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import os


class ModelTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.feature_importance = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def prepare_data(self, df, feature_columns, target_column='completed', test_size=0.2):
        X = df[feature_columns]
        y = df[target_column]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        print(f"训练集大小: {len(self.X_train)}")
        print(f"测试集大小: {len(self.X_test)}")
        print(f"目标变量分布 (训练集):\n{self.y_train.value_counts()}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train(self, n_estimators=100, max_depth=10):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        self.model.fit(self.X_train, self.y_train)
        print(f"模型训练完成: Random Forest (n_estimators={n_estimators}, max_depth={max_depth})")
        
        return self.model
    
    def evaluate(self):
        y_pred = self.model.predict(self.X_test)
        
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        
        report = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': classification_report(self.y_test, y_pred),
            'confusion_matrix': confusion_matrix(self.y_test, y_pred)
        }
        
        print("\n" + "=" * 50)
        print("模型评估报告")
        print("=" * 50)
        print(f"准确率 (Accuracy): {accuracy:.4f}")
        print(f"精确率 (Precision): {precision:.4f}")
        print(f"召回率 (Recall): {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        print("\n分类报告:")
        print(report['classification_report'])
        print("\n混淆矩阵:")
        print(report['confusion_matrix'])
        
        return report
    
    def get_feature_importance(self, feature_names):
        importance = self.model.feature_importances_
        self.feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print("\n" + "=" * 50)
        print("特征重要性排序")
        print("=" * 50)
        for idx, row in self.feature_importance.iterrows():
            print(f"{row['feature']}: {row['importance']:.4f}")
        
        return self.feature_importance
    
    def save_results(self, output_dir='output'):
        os.makedirs(output_dir, exist_ok=True)
        
        if self.feature_importance is not None:
            importance_path = os.path.join(output_dir, 'feature_importance.csv')
            self.feature_importance.to_csv(importance_path, index=False)
            print(f"特征重要性已保存至: {importance_path}")
        
        report = self.evaluate()
        report_path = os.path.join(output_dir, 'model_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 50 + "\n")
            f.write("模型评估报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"准确率 (Accuracy): {report['accuracy']:.4f}\n")
            f.write(f"精确率 (Precision): {report['precision']:.4f}\n")
            f.write(f"召回率 (Recall): {report['recall']:.4f}\n")
            f.write(f"F1分数: {report['f1_score']:.4f}\n\n")
            f.write("分类报告:\n")
            f.write(report['classification_report'])
            f.write("\n特征重要性:\n")
            for idx, row in self.feature_importance.iterrows():
                f.write(f"{row['feature']}: {row['importance']:.4f}\n")
        
        print(f"模型报告已保存至: {report_path}")


def train_completion_model(df, feature_columns, target_column='completed', output_dir='output'):
    trainer = ModelTrainer()
    trainer.prepare_data(df, feature_columns, target_column)
    trainer.train()
    trainer.evaluate()
    trainer.get_feature_importance(feature_columns)
    trainer.save_results(output_dir)
    
    return trainer


if __name__ == "__main__":
    from data_generator import generate_course_data
    from data_preprocessor import DataPreprocessor
    
    data = generate_course_data(n_samples=1000)
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.preprocess(data)
    
    feature_columns = preprocessor.get_feature_columns()
    trainer = train_completion_model(processed_data, feature_columns)
