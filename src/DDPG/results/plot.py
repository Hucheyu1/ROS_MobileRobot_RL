import csv  # 导入csv模块
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


pkg_path = 'F:/catkin_ws/src/DDPG/results'
stage_1 = '/ddpg/stage_none_nowaypoint'
stage_2 = '/ddpg/stage_sparse'
stage_3 = '/ddpg/stage_sparse_rh'
stage_4 = '/ddpg/stage_sparse_rv'
stage_5 = '/ddpg/stage_sparse_rh_rv'
stage_6 = '/ddpg/stage_sparse_rh_rv_per'
stage_7 = '/ddpg/stage_sparse_lstm'
# filename_1 = pkg_path + stage_1 + '/ddpg_test_trajectory.csv'
filename_1 = pkg_path + stage_1 + '/ddpg_training_trajectory.csv'
filename_2 = pkg_path + stage_2 + '/ddpg_training_trajectory.csv'
filename_3 = pkg_path + stage_4 + '/ddpg_training_trajectory.csv'
filename_4 = pkg_path + stage_5 + '/ddpg_training_trajectory.csv'
filename_5 = pkg_path + stage_6 + '/ddpg_training_trajectory_8.csv'

def moving_average(data, arrival_data, window_size):
    moving_avg = []
    arrival_avg = []
    for i in range(len(data) - window_size + 1):
        window = data[i:i+window_size]
        window_1 = arrival_data[i:i+window_size]
        avg = sum(window) / window_size
        avg_1 = sum(window_1) / window_size
        moving_avg.append(avg)
        arrival_avg.append(avg_1)
    return moving_avg , arrival_avg

def read_data(filename , window_size = 20):
    with open(filename) as f:
        reader = csv.reader(f)
        header_row = next(reader)  # 返回文件的下一行，在这便是首行，即文件头
        episode_number = []
        episode_reward = []
        episode_arrival = []
        total_arrival_rate = []
        Timeout_rate = 0
        Collision_rate = 0
        true_number = 0
        step_average = 0
        for i, row in enumerate(reader):
            # print(row)
            episode_reward.append(int(row[3]))
            episode_number.append(int(row[0]))
            if row[1] == 'True':
                true_number += 1
                step_average = step_average + int(row[4])
                episode_arrival.append(1)
            else:
                episode_arrival.append(0)
                if int(row[4]) == 300 or int(row[4]) == 200:
                    Timeout_rate += 1
                else:
                    Collision_rate += 1
            total_arrival_rate.append(round(sum(episode_arrival)/len(episode_number) ,2))
        Timeout_rate = round(Timeout_rate / len(episode_number) , 3)
        Collision_rate = round(Collision_rate / len(episode_number) , 3)
        smoothed_data , smoothed_arrival= moving_average(episode_reward, episode_arrival, window_size)
        episode_number = episode_number[:len(smoothed_data)]
        total_arrival_rate = total_arrival_rate[:len(smoothed_data)]
        arrival_rate = round(true_number / len(episode_reward) , 3)
        step_average = round(step_average / true_number)
        return episode_number , smoothed_data, smoothed_arrival ,total_arrival_rate, arrival_rate, step_average, Timeout_rate, Collision_rate

episode_number_1 , smoothed_data_1 , smoothed_arrival_1 ,total_arrival_rate_1 ,arrival_rate_1 , step_average_1, Timeout_rate_1, Collision_rate_1= read_data(filename_1 , 1)
episode_number_2 , smoothed_data_2 , smoothed_arrival_2 ,total_arrival_rate_2 ,arrival_rate_2 , step_average_2, Timeout_rate_2, Collision_rate_2= read_data(filename_2 , 1)
episode_number_3 , smoothed_data_3 , smoothed_arrival_3 ,total_arrival_rate_3 ,arrival_rate_3 , step_average_3, Timeout_rate_3, Collision_rate_3= read_data(filename_3 , 1)
episode_number_4 , smoothed_data_4 , smoothed_arrival_4 ,total_arrival_rate_4 ,arrival_rate_4 , step_average_4, Timeout_rate_4, Collision_rate_4= read_data(filename_4 , 1)
episode_number_5 , smoothed_data_5 , smoothed_arrival_5 ,total_arrival_rate_5 ,arrival_rate_5 , step_average_5, Timeout_rate_5, Collision_rate_5= read_data(filename_5 , 1)


plt.figure(figsize=(12,5))
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False
plt.plot(episode_number_1, total_arrival_rate_1, c='red', alpha=0.5)
plt.plot(episode_number_2, total_arrival_rate_2, c='blue', alpha=0.5)
plt.plot(episode_number_3, total_arrival_rate_3, c='green', alpha=0.5)
plt.plot(episode_number_4, total_arrival_rate_4, c='black', alpha=0.5)
plt.plot(episode_number_5, total_arrival_rate_5, c='orange', alpha=0.8)
plt.xlabel('迭代次数',  fontsize=16)
plt.ylabel('到达率',  fontsize=16)
# plt.title("reward",  fontsize=24)
# plt.legend(["stage_sparse_rh_rv","stage_sparse_rh_rv_per","stage_sparse_rv","stage_sparse_rh_rv","stage_sparse_rh_rv_per"])
plt.legend(["stage_none","stage_obs","stage_rv","stage_rh_rv","stage_lstm_per"],fontsize=16)

# plt.subplot(2, 1, 2)
# # plt.plot(episode_number_1, total_arrival_rate_1, c='red', alpha=0.5)
# # plt.plot(episode_number_2, total_arrival_rate_2, c='blue', alpha=0.5)
# plt.plot(episode_number_3, total_arrival_rate_3, c='green', alpha=0.5)
# plt.plot(episode_number_4, total_arrival_rate_4, c='orange', alpha=1.0)
# plt.plot(episode_number_5, total_arrival_rate_5, c='black', alpha=0.5)
# plt.xlabel('episode_number',  fontsize=16)
# plt.ylabel('arrival_rate',  fontsize=16)
# plt.title("arrival_rate",  fontsize=24)
# plt.legend(["stage_sparse_rh_rv","stage_sparse_rh_rv_per","stage_sparse_rv","stage_sparse_rh_rv","stage_sparse_rh_rv_per"])


print(arrival_rate_1 , arrival_rate_2, arrival_rate_3, arrival_rate_4, arrival_rate_5)
print(step_average_1 , step_average_2, step_average_3, step_average_4, step_average_5)
print(Timeout_rate_1 , Timeout_rate_2, Timeout_rate_3, Timeout_rate_4, Timeout_rate_5)
print(Collision_rate_1 , Collision_rate_2, Collision_rate_3, Collision_rate_4, Collision_rate_5)  
plt.show()

