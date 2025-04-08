import joblib
import numpy as np
import pandas as pd

# 1. 加载模型和编码器
rf_model = joblib.load('rf_spark_exec_time.pkl')
le = joblib.load('job_type_encoder.pkl')

# 2. 新的 objective
def objective_ml(params, data_size_mb, job_type):
    """
    用训练好的随机森林模型预测执行时间
    params: [executorMemoryMB, executorCores, driverMemoryMB, driverCores, numPartitions]
    """
    em, ec, dm, dc, npart = params
    # 对 job_type 做编码
    jt_enc = le.transform([job_type])[0]
    features = np.array([data_size_mb, jt_enc, em, ec, dm, dc, npart]).reshape(1, -1)

    # 将特征转换为 pandas DataFrame，并给它们添加列名，确保与训练时一致
    feature_names = ['data_size_mb', 'job_type_enc', 'executorMemoryMB', 'executorCores', 'driverMemoryMB',
                     'driverCores', 'numPartitions']
    features_df = pd.DataFrame(features, columns=feature_names)

    # 用模型预测执行时间
    pred_time = rf_model.predict(features_df)[0]
    return pred_time
