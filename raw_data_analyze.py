import json

with open('dev_data.json', 'r') as f:
    data = json.load(f)
max_len = 0
avg_len = 0
time_interval = []
for d in data:
    seq = d['data']
    if len(seq) > max_len:
        max_len = len(seq)
    avg_len += len(seq)
    prev_time_step = seq[0][0]
    for i in range(1, len(seq)):
        cur_time_interval = seq[i][0] - prev_time_step
        if not cur_time_interval in time_interval:
            time_interval.append(cur_time_interval)
print(max_len)
print(avg_len/len(data))
print(time_interval)