import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')


class RepurchasePredictor:
    
    def __init__(self, model_type='random_forest', random_state=42):
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.feature_importance = None
        
    def create_target_variable(self, df):
        purchase_counts = df[df['behavior_type'] == 'buy'].groupby('user_id').size()
        
        target = (purchase_counts >= 2).astype(int).reset_index()
        target.columns = ['user_id', 'is_repurchase']
        
        return target
    
    def prepare_features(self, behavior_features, rfm_features):
        features = behavior_features.merge(rfm_features, on='user_id', how='inner')
        
        feature_cols = [
            'pv', 'fav', 'cart', 'buy', 'total_actions',
            'conversion_rate', 'cart_to_buy_rate', 'fav_to_buy_rate',
            'category_diversity', 'recency', 'frequency', 'monetary',
            'r_score', 'f_score', 'm_score'
        ]
        
        available_cols = [col for col in feature_cols if col in features.columns]
        self.feature_names = available_cols
        
        return features[['user_id'] + available_cols]
    
    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=self.random_state
            )
        
        self.model.fit(X_train_scaled, y_train)
        
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        self._calculate_feature_importance()
        
        return metrics
    
    def _calculate_feature_importance(self):
        if self.model_type == 'random_forest':
            importance = self.model.feature_importances_
        else:
            importance = np.abs(self.model.coef_[0])
        
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
    
    def get_feature_importance(self):
        return self.feature_importance
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)


class ModelEvaluator:
    
    def __init__(self):
        self.results = {}
        
    def evaluate_model(self, predictor, X, y, model_name):
        y_pred = predictor.predict(X)
        y_pred_proba = predictor.predict_proba(X)[:, 1]
        
        self.results[model_name] = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y, y_pred_proba)
        }
        
        return self.results[model_name]
    
    def compare_models(self):
        return pd.DataFrame(self.results).T
    
    def generate_report(self, metrics, feature_importance):
        report = []
        report.append("=" * 60)
        report.append("复购预测模型评估报告")
        report.append("=" * 60)
        report.append("")
        report.append("一、模型性能指标")
        report.append("-" * 40)
        report.append(f"准确率 (Accuracy): {metrics['accuracy']:.4f}")
        report.append(f"精确率 (Precision): {metrics['precision']:.4f}")
        report.append(f"召回率 (Recall): {metrics['recall']:.4f}")
        report.append(f"F1分数 (F1-Score): {metrics['f1']:.4f}")
        report.append(f"AUC-ROC: {metrics['roc_auc']:.4f}")
        report.append("")
        report.append("二、混淆矩阵")
        report.append("-" * 40)
        cm = metrics['confusion_matrix']
        report.append(f"真负例 (TN): {cm[0][0]}")
        report.append(f"假正例 (FP): {cm[0][1]}")
        report.append(f"假负例 (FN): {cm[1][0]}")
        report.append(f"真正例 (TP): {cm[1][1]}")
        report.append("")
        report.append("三、特征重要性排名")
        report.append("-" * 40)
        for idx, row in feature_importance.iterrows():
            report.append(f"{row['feature']}: {row['importance']:.4f}")
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)


if __name__ == '__main__':
    from data_generator import EcommerceDataGenerator
    from preprocessing import DataPreprocessor, RFMFeatureBuilder, UserBehaviorFeatures
    
    generator = EcommerceDataGenerator(n_users=1000, n_records=50000)
    df = generator.generate(inject_issues=True)
    
    preprocessor = DataPreprocessor(df)
    clean_df = preprocessor.handle_missing_values().handle_outliers().get_processed_data()
    
    rfm_builder = RFMFeatureBuilder(clean_df)
    rfm_df = rfm_builder.calculate_rfm_scores()
    
    feature_builder = UserBehaviorFeatures(clean_df)
    behavior_features = feature_builder.build_features()
    
    predictor = RepurchasePredictor(model_type='random_forest')
    target = predictor.create_target_variable(clean_df)
    
    features = predictor.prepare_features(behavior_features, rfm_df)
    
    data = features.merge(target, on='user_id', how='inner')
    X = data[predictor.feature_names]
    y = data['is_repurchase']
    
    metrics = predictor.train(X, y)
    
    feature_importance = predictor.get_feature_importance()
    
    evaluator = ModelEvaluator()
    report = evaluator.generate_report(metrics, feature_importance)
    print(report)
