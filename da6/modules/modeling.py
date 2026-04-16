import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, classification_report,
                             confusion_matrix)


class RepurchasePredictor:
    """用户复购预测模型"""

    def __init__(self, X, y, feature_names):
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}

    def split_data(self, test_size=0.3, random_state=42):
        """划分训练集和测试集"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        print(f"训练集大小: {len(self.X_train)}")
        print(f"测试集大小: {len(self.X_test)}")
        print(f"训练集复购率: {self.y_train.mean():.2%}")
        print(f"测试集复购率: {self.y_test.mean():.2%}")

    def train_logistic_regression(self):
        """训练逻辑回归模型"""
        print("\n训练逻辑回归模型...")
        lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        lr.fit(self.X_train, self.y_train)
        self.models['logistic_regression'] = lr
        return lr

    def train_random_forest(self):
        """训练随机森林模型"""
        print("\n训练随机森林模型...")
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        rf.fit(self.X_train, self.y_train)
        self.models['random_forest'] = rf
        return rf

    def evaluate_model(self, model_name):
        """评估模型性能"""
        model = self.models[model_name]
        y_pred = model.predict(self.X_test)
        y_prob = model.predict_proba(self.X_test)[:, 1]

        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred),
            'recall': recall_score(self.y_test, y_pred),
            'f1': f1_score(self.y_test, y_pred),
            'auc': roc_auc_score(self.y_test, y_prob)
        }

        self.results[model_name] = metrics

        print(f"\n{model_name} 模型评估结果:")
        print(f"  准确率 (Accuracy): {metrics['accuracy']:.4f}")
        print(f"  精确率 (Precision): {metrics['precision']:.4f}")
        print(f"  召回率 (Recall): {metrics['recall']:.4f}")
        print(f"  F1分数: {metrics['f1']:.4f}")
        print(f"  AUC: {metrics['auc']:.4f}")

        return metrics

    def get_feature_importance(self, model_name='random_forest'):
        """获取特征重要性"""
        if model_name not in self.models:
            print(f"模型 {model_name} 未训练")
            return None

        model = self.models[model_name]

        if model_name == 'random_forest':
            importances = model.feature_importances_
        else:
            importances = np.abs(model.coef_[0])

        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        print(f"\n{model_name} 特征重要性:")
        for idx, row in feature_importance.iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")

        return feature_importance

    def cross_validate(self, model_name, cv=5):
        """交叉验证"""
        model = self.models[model_name]
        scores = cross_val_score(model, self.X, self.y, cv=cv, scoring='roc_auc')

        print(f"\n{model_name} {cv}折交叉验证 AUC: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
        return scores

    def get_classification_report(self, model_name):
        """获取详细分类报告"""
        model = self.models[model_name]
        y_pred = model.predict(self.X_test)

        report = classification_report(self.y_test, y_pred, target_names=['未复购', '复购'])
        print(f"\n{model_name} 分类报告:")
        print(report)

        return report

    def run_modeling(self):
        """运行完整建模流程"""
        print("=" * 50)
        print("开始建模分析...")
        print("=" * 50)

        # 1. 划分数据
        self.split_data()

        # 2. 训练逻辑回归
        self.train_logistic_regression()
        self.evaluate_model('logistic_regression')
        self.get_feature_importance('logistic_regression')
        self.cross_validate('logistic_regression')

        # 3. 训练随机森林
        self.train_random_forest()
        self.evaluate_model('random_forest')
        self.get_feature_importance('random_forest')
        self.cross_validate('random_forest')

        # 4. 详细分类报告
        self.get_classification_report('random_forest')

        print("\n建模分析完成!")

        return self.results

    def get_best_model(self):
        """获取最佳模型"""
        if not self.results:
            return None

        best_model = max(self.results.items(), key=lambda x: x[1]['auc'])
        print(f"\n最佳模型: {best_model[0]}, AUC: {best_model[1]['auc']:.4f}")
        return best_model


if __name__ == '__main__':
    # 测试代码
    from data_generator import ECommerceDataGenerator
    from data_preprocessor import DataPreprocessor

    generator = ECommerceDataGenerator(n_users=1000, n_records=5000)
    df = generator.generate()

    preprocessor = DataPreprocessor(df)
    preprocessor.run_preprocessing()

    X, y, feature_names = preprocessor.prepare_modeling_data()

    predictor = RepurchasePredictor(X, y, feature_names)
    predictor.run_modeling()
    predictor.get_best_model()
