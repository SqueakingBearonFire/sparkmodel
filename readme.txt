data_size_gb：输入数据量，单位 GB
job_type：作业类型（如 join、filter、agg）
executorMemoryMB：每个 Executor 内存，单位 MB
executorCores：每个 Executor 核心数
driverMemoryMB：Driver 内存，单位 MB
driverCores：Driver 核心数
numPartitions：Shuffle 分区数（通常通过 spark.sql.shuffle.partitions 设置）
executionTimeSec：作业实际执行时间，单位 秒

rf_spark_exec_time.pkl —— 序列化的随机森林模型
job_type_encoder.pkl —— 序列化的 LabelEncoder
