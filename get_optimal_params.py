import os

import joblib
import numpy as np
import pandas as pd

from objective import objective_ml


class Particle:
    def __init__(self, pos, vel):
        self.position     = np.array(pos, dtype=float)
        self.velocity     = np.array(vel, dtype=float)
        self.best_position= self.position.copy()
        self.best_value   = float('inf')

def pso_optimize(initial, bounds, data_size_mb, job_type,
                 num_particles=20, max_iter=100):
    dim = len(initial)
    # 初始化粒子
    particles = []
    for _ in range(num_particles):
        pos = []
        vel = []
        for i in range(dim):
            lo, hi = bounds[i]
            perturb = np.random.uniform(-0.1,0.1)*(hi-lo)
            pos_val = np.clip(initial[i]+perturb, lo, hi)
            pos.append(pos_val)
            vel.append(np.random.uniform(-0.05,0.05)*(hi-lo))
        p = Particle(pos, vel)
        p.best_value = objective_ml(p.position, data_size_mb, job_type)
        p.best_position = p.position.copy()
        particles.append(p)
    # 全局最优
    gbest = min(particles, key=lambda p: p.best_value)
    gbest_pos, gbest_val = gbest.best_position.copy(), gbest.best_value

    # PSO 参数
    w, c1, c2 = 0.6, 1.5, 1.5

    for it in range(max_iter):
        for p in particles:
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            # 更新速度
            p.velocity = (w*p.velocity +
                          c1*r1*(p.best_position - p.position) +
                          c2*r2*(gbest_pos - p.position))
            # 更新位置
            p.position += p.velocity
            # 边界约束
            for i in range(dim):
                lo, hi = bounds[i]
                p.position[i] = np.clip(p.position[i], lo, hi)
            # 评估
            val = objective_ml(p.position, data_size_mb, job_type)
            if val < p.best_value:
                p.best_value = val
                p.best_position = p.position.copy()
            if val < gbest_val:
                gbest_val = val
                gbest_pos = p.position.copy()
        # （可选）自适应 w：w = w*0.99
    return gbest_pos, gbest_val

def predict_initial_params(data_size_mb, job_type):  # 参数改为 data_size_mb
    # 默认配置
    base_mem = 4096  # MB
    base_cores = 2
    ideal_partition_size = 256  # MB (每个分区的理想大小，按需调整)
    base_partitions = max(1, int(data_size_mb / ideal_partition_size))
    if job_type == 'join':
        executor_memory = base_mem + data_size_mb * 25  # join操作需要更多的资源，每MB增加25MB的内存
        executor_cores  = base_cores + 2
        driver_memory   = 2048 + data_size_mb * 12.5
        driver_cores    = 2
        base_partitions  = base_partitions * 3
    else:
        executor_memory = base_mem + data_size_mb * 12.5
        executor_cores  = base_cores + 1
        driver_memory   = 2048 + data_size_mb * 5
        driver_cores    = 1

    num_partitions = max(1, base_partitions)  # 确保至少1个分区
    return [executor_memory, executor_cores, driver_memory, driver_cores, num_partitions]


def update_model(features, pred_time, feature_names, model_path='rf_spark_exec_time.pkl'):
    """
    用于增量更新模型的函数。每次调用时都将新的数据添加到训练数据中，并更新模型。
    """
    # 加载模型
    rf_model = joblib.load(model_path)
    # 将新数据追加到 CSV 文件
    new_data = [features[0], features[1]] + features[2:] + [pred_time]
    new_data_df = pd.DataFrame([new_data], columns=feature_names + ['executionTimeSec'])

    if os.path.exists('train_data.csv'):
        new_data_df.to_csv('train_data.csv', mode='a', header=False, index=False)
    else:
        new_data_df.to_csv('train_data.csv', mode='w', header=True, index=False)

    # 每当收集到 100 条数据时进行增量训练
    train_data = pd.read_csv('train_data.csv')
    if len(train_data) >= 100:
        X_train = train_data[feature_names].values
        y_train = train_data['executionTimeSec'].values

        # 使用随机森林的 fit() 方法进行全量训练，这里可以替换成增量学习方法（如在线学习模型）
        rf_model.fit(X_train, y_train)  # 注意，fit()方法是全量训练的

        # 保存更新后的模型
        joblib.dump(rf_model, model_path)

        # 清空 CSV 文件中的数据（如果需要清理训练数据）
        os.remove('train_data.csv')

if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_size_mb', type=float, required=True)
    parser.add_argument('--job_type', type=str, required=True)
    args = parser.parse_args()

    # 1. 初始预测
    init = predict_initial_params(args.data_size_mb, args.job_type)

    # 2. 定义范围
    bounds = [
      (2048, 32768),
      (1, 8),
      (1024, 8192),
      (1, 4),
      (10, 500)
    ]

    # 3. PSO 优化
    best_params, best_val = pso_optimize(init, bounds,
                                         args.data_size_mb, args.job_type)
    # 四舍五入整数参数
    best = [
      int(best_params[0]),
      int(round(best_params[1])),
      int(best_params[2]),
      int(round(best_params[3])),
      int(round(best_params[4]))
    ]

    best_params_with_metadata = [args.data_size_mb, args.job_type] + best

    # 4. 输出可直接用于 spark-submit
    print(f"--executor-memory {best[0]}m "
          f"--executor-cores {best[1]} "
          f"--driver-memory {best[2]}m "
          f"--driver-cores {best[3]} "
          f"--conf spark.sql.shuffle.partitions={best[4]}")

    feature_names = ['data_size_mb', 'job_type_enc', 'executorMemoryMB', 'executorCores', 'driverMemoryMB',
                     'driverCores', 'numPartitions']
    update_model(best_params_with_metadata, best_val, feature_names)