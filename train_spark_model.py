import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import joblib

def main():
    # 1. 加载历史数据
    df = pd.read_csv('spark_history.csv')

    # 2. 编码 job_type
    le = LabelEncoder()
    df['job_type_enc'] = le.fit_transform(df['job_type'])

    # 3. 构造特征矩阵 X 和目标 y
    feature_cols = [
        'data_size_gb', 'job_type_enc',
        'executorMemoryMB', 'executorCores',
        'driverMemoryMB', 'driverCores',
        'numPartitions'
    ]
    X = df[feature_cols]
    y = df['executionTimeSec']

    # 4. 划分训练集/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 5. 建立模型并做网格搜索
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }
    rf = RandomForestRegressor(random_state=42)
    grid = GridSearchCV(
        rf, param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train, y_train)

    # 6. 评估模型
    best_rf = grid.best_estimator_
    y_pred = best_rf.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f'Best params: {grid.best_params_}')
    print(f'Test RMSE: {rmse:.2f} seconds')

    # 7. 保存模型和 LabelEncoder
    joblib.dump(best_rf, 'rf_spark_exec_time.pkl')
    joblib.dump(le, 'job_type_encoder.pkl')
    print('Model and encoder saved to disk.')

if __name__ == '__main__':
    main()
