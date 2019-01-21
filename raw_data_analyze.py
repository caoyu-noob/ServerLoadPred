import json
from tqdm import tqdm
import numpy as np

# with open('dev_data.json', 'r') as f:
#     dev_data = json.load(f)
# with open('test_data.json', 'r') as f:
#     test_data = json.load(f)
# with open('train_data.json', 'r') as f:
#     train_data = json.load(f)
# train_data.extend(dev_data)
# train_data.extend(test_data)
# data = train_data
# max_len = 0
# avg_len = 0
# time_interval = {}
# data_lens = []
# cpu_change = {}
# for d in data:
#     seq = d['data']
#     data_lens.append(len(seq))
#     if len(seq) > max_len:
#         max_len = len(seq)
#     avg_len += len(seq)
#     prev_time_step = seq[0][0]
#     prev_cpu =seq[0][1]
#     for i in range(1, len(seq)):
#         cur_time_interval = seq[i][0] - prev_time_step
#         if not time_interval.__contains__(cur_time_interval):
#             time_interval[cur_time_interval] = 1
#         else:
#             time_interval[cur_time_interval] += 1
#         cur_cpu_change = seq[i][1] - prev_cpu
#         prev_time_step, prev_cpu = seq[i][0], seq[i][1]
#         if not cpu_change.__contains__(cur_cpu_change):
#             cpu_change[cur_cpu_change] = 1
#         else:
#             cpu_change[cur_cpu_change] += 1
# with open('data_len.json', 'w') as f:
#     json.dump(data_lens, f)
# with open('time_interval.json', 'w') as f:
#     json.dump(time_interval, f)
# with open('cpu_change.json', 'w') as f:
#     json.dump(cpu_change, f)
# print(max_len)
# print(avg_len/len(data))
# print(time_interval)
with open('cpu_change.json', 'r') as f:
    data = json.load(f)
max_len, min_len = 0, 100000
list_data = []
for key, value in data.items():
    # if int(key) > 0:
    list_data.extend([int(key)] * value)
max_val = np.max(list_data)
min_val = np.min(list_data)
mean_val = np.mean(list_data)
std_val = np.std(list_data)
print(mean_val)