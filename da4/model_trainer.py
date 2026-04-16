import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def train_model(df, feature_columns, target_column='sales'):
    print("=" * 50)
    print("开始模型训练...")
    
    X = df[feature_columns]
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False, random_state=42
    )
    
    print(f"训练集大小: {X_train.shape[0]} 样本")
    print(f"测试集大小: {X_test.shape[0]} 样本")
    
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    metrics = calculate_metrics(y_train, y_train_pred, y_test, y_test_pred)
    
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("模型训练完成!")
    print_metrics(metrics)
    print("=" * 50)
    
    return model, metrics, feature_importance


def calculate_metrics(y_train, y_train_pred, y_test, y_test_pred):
    metrics = {
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'train_r2': r2_score(y_train, y_train_pred),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'test_r2': r2_score(y_test, y_test_pred)
    }
    return metrics


def print_metrics(metrics):
    print("\n模型评估指标:")
    print(f"训练集 - MAE: {metrics['train_mae']:.2f}, RMSE: {metrics['train_rmse']:.2f}, R²: {metrics['train_r2']:.4f}")
    print(f"测试集 - MAE: {metrics['test_mae']:.2f}, RMSE: {metrics['test_rmse']:.2f}, R²: {metrics['test_r2']:.4f}")


def predict_future(model, future_df, feature_columns, last_known_sales):
    print("\n开始预测未来30天销售额...")
    
    future_df = future_df.copy()
    predictions = []
    current_sales = last_known_sales
    
    sales_lag_7 = [last_known_sales] * 7
    sales_lag_14 = [last_known_sales] * 14
    sales_lag_30 = [last_known_sales] * 30
    
    for i in range(len(future_df)):
        row = future_df.iloc[i].copy()
        
        row['sales_lag_1'] = current_sales
        row['sales_lag_7'] = sales_lag_7[-7]
        row['sales_lag_14'] = sales_lag_14[-14]
        row['sales_lag_30'] = sales_lag_30[-30]
        
        row['sales_rolling_7'] = np.mean(sales_lag_7[-7:])
        row['sales_rolling_14'] = np.mean(sales_lag_14[-14:])
        row['sales_rolling_30'] = np.mean(sales_lag_30[-30:])
        
        row['sales_rolling_std_7'] = np.std(sales_lag_7[-7:])
        row['sales_rolling_std_30'] = np.std(sales_lag_30[-30:])
        
        row['sales_diff_1'] = current_sales - sales_lag_7[-1]
        row['sales_diff_7'] = current_sales - sales_lag_7[-7]
        
        X_pred = pd.DataFrame([row[feature_columns]])
        pred_sales = model.predict(X_pred)[0]
        
        predictions.append(pred_sales)
        current_sales = pred_sales
        
        sales_lag_7.append(pred_sales)
        sales_lag_14.append(pred_sales)
        sales_lag_30.append(pred_sales)
    
    future_df['predicted_sales'] = predictions
    
    print(f"未来30天预测完成! 预测总销售额: {sum(predictions):.2f}")
    print("=" * 50)
    
    return future_df
