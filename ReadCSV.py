import json
from tqdm import tqdm

with open('machine_usage.csv', 'r') as f:
    data = f.readlines()
json_data = []
previous_id = data[0].split(',')[0]
current_data = {}
current_data['id'] = previous_id
for d in tqdm(data):
    line = d.split(',')
    if previous_id != line[0]:
        json_data.append(current_data)
        current_data = {}
        current_data['id'] = line[0]
        previous_id = line[0]
    if not current_data.__contains__('data'):
        current_data['data'] = []
    if line[1] != '' and line[2] != '' and line[3] != '' and line[6] != '' and line[7] != '' and line[8] != '':
        current_data['data'].append([int(line[1]), int(line[2]), int(line[3]), float(line[6]), float(line[7]),
                                     float(line[8])])
with open('data.json', 'w') as f:
    json.dump(json_data, f)
print('aaa')