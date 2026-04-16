import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


class RepurchasePredictor:
    def __init__(self, model_type='random_forest', random_seed=42):
        self.model_type = model_type
        self.random_seed = random_seed
        self.model = None
        self.scaler = None
        self.feature_importance = None
        self.metrics = {}
    
    def prepare_data(self, rfm_df, test_size=0.3):
        feature_cols = ['Recency', 'Frequency', 'Monetary', 'R_Score', 'F_Score', 'M_Score',
                       'count_pv', 'count_fav', 'count_cart', 'count_buy',
                       'cart_conversion', 'pv_cart_rate']
        
        X = rfm_df[feature_cols]
        y = rfm_df['Repurchase']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_seed, stratify=y
        )
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols
    
    def train_model(self, X_train, y_train, feature_cols):
        print(f"\n=== 训练{self.model_type}模型 ===")
        
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100, max_depth=8, random_state=self.random_seed
            )
        elif self.model_type == 'logistic_regression':
            self.model = LogisticRegression(
                max_iter=1000, random_state=self.random_seed
            )
        
        self.model.fit(X_train, y_train)
        
        if self.model_type == 'random_forest':
            importances = self.model.feature_importances_
        else:
            importances = np.abs(self.model.coef_[0])
        
        self.feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': importances
        }).sort_values('importance', ascending=False).reset_index(drop=True)
        
        print("特征重要性排名:")
        for i, row in self.feature_importance.iterrows():
            print(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
        
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        print("\n=== 模型评估结果 ===")
        print(f"准确率: {self.metrics['accuracy']:.4f}")
        print(f"精确率: {self.metrics['precision']:.4f}")
        print(f"召回率: {self.metrics['recall']:.4f}")
        print(f"F1分数: {self.metrics['f1']:.4f}")
        print(f"AUC: {self.metrics['auc']:.4f}")
        
        return self.metrics
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled), self.model.predict_proba(X_scaled)[:, 1]
    
    def run(self, rfm_df):
        X_train, X_test, y_train, y_test, feature_cols = self.prepare_data(rfm_df)
        self.train_model(X_train, y_train, feature_cols)
        self.evaluate_model(X_test, y_test)
        
        return self.model, self.feature_importance, self.metrics


if __name__ == '__main__':
    from data_preprocessor import DataPreprocessor
    preprocessor = DataPreprocessor()
    _, rfm_df = preprocessor.preprocess('../data/ecommerce_behavior.csv')
    
    predictor = RepurchasePredictor(model_type='random_forest')
    model, feature_imp, metrics = predictor.run(rfm_df)
