import os
import pandas as pd
import Sprayer_PINN as SP

# 设置数据目录路径
base_dir = 'data/prepared_data'

# 初始化存储结构
data_train = pd.DataFrame()
data_test = pd.DataFrame()
data_sprayer_train = []

# 遍历 base_dir 下的所有子目录
for subdir in os.listdir(base_dir):
    subdir_path = os.path.join(base_dir, subdir)
    
    # 确保是目录
    if os.path.isdir(subdir_path):
        print(f"Processing Directory: {subdir_path}")
        data_train = pd.DataFrame()
        data_test = pd.DataFrame()
        data_sprayer_train = []
        # 遍历子目录中的所有文件
        for file in os.listdir(subdir_path):
            file_path = os.path.join(subdir_path, file)

            # 检查是否是 CSV 文件
            if file_path.endswith('.csv'):
                try:
                    # 读取 CSV 文件
                    df = pd.read_csv(file_path)

                    # 根据文件名决定存储方式
                    if 'data_train.csv' in file:
                        data_train = df
                    elif 'data_test.csv' in file:
                        data_test = df
                    elif 'data_sprayer_train' in file:
                        data_sprayer_train.append(df)

                    print(f"  Processed file: {file}")
                except Exception as e:
                    print(f"    Error reading {file}: {e}")
        model = SP.InformedNN(now_t = int(subdir_path[-6:]))
        model.train(data_train, data_sprayer_train, show = False)
        data_test["prediction_result"] = model.forward(data_test).detach().numpy()
#             train_range_show = 60 
#             test_range_show = 60
#             update_time_range_show = 10
#             model.test_show(train_range_show, test_range_show, update_time_range_show, result_path, time_start)
        data_test.to_csv(result_path + "/data_test_" + subdir_path[-6:] + ".csv", index = False)        

# 输出结果的简要信息

