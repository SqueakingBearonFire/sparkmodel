import numpy as np

from objective import objective_ml


class Particle:
    def __init__(self, pos, vel):
        self.position     = np.array(pos, dtype=float)
        self.velocity     = np.array(vel, dtype=float)
        self.best_position= self.position.copy()
        self.best_value   = float('inf')

def pso_optimize(initial, bounds, data_size_gb, job_type,
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
        p.best_value = objective_ml(p.position, data_size_gb, job_type)
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
            val = objective_ml(p.position, data_size_gb, job_type)
            if val < p.best_value:
                p.best_value = val
                p.best_position = p.position.copy()
            if val < gbest_val:
                gbest_val = val
                gbest_pos = p.position.copy()
        # （可选）自适应 w：w = w*0.99
    return gbest_pos, gbest_val

def predict_initial_params(data_size_gb, job_type):
    # 简单规则示例
    base_mem = 4096  # MB
    base_cores = 2
    base_partitions = int(data_size_gb * 10)
    if job_type == 'join':
        executor_memory = base_mem + data_size_gb * 100
        executor_cores  = base_cores + 2
        driver_memory   = 2048 + data_size_gb * 50
        driver_cores    = 2
    else:
        executor_memory = base_mem + data_size_gb * 50
        executor_cores  = base_cores + 1
        driver_memory   = 2048 + data_size_gb * 20
        driver_cores    = 1
    num_partitions = max(1, base_partitions)
    return [executor_memory, executor_cores, driver_memory, driver_cores, num_partitions]



if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_size_gb', type=float, required=True)
    parser.add_argument('--job_type', type=str, required=True)
    args = parser.parse_args()

    # 1. 初始预测
    init = predict_initial_params(args.data_size_gb, args.job_type)

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
                                         args.data_size_gb, args.job_type)
    # 四舍五入整数参数
    best = [
      int(best_params[0]),
      int(round(best_params[1])),
      int(best_params[2]),
      int(round(best_params[3])),
      int(round(best_params[4]))
    ]

    # 4. 输出可直接用于 spark-submit
    print(f"--executor-memory {best[0]}m "
          f"--executor-cores {best[1]} "
          f"--driver-memory {best[2]}m "
          f"--driver-cores {best[3]} "
          f"--conf spark.sql.shuffle.partitions={best[4]}")
