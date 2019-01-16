import json
import pickle
import random

with open('index.pickle', 'rb') as f:
    index = pickle.load(f)
train_index = index['train'][:15000]
train_data = []
# dev_index = index['dev']
# test_index = index['test']
with open('data.json', 'r') as f:
    data = json.load(f)
for index in train_index:
    train_data.append(data[index])
with open('train_data.json', 'w') as f:
    json.dump(train_data, f)
# dev_data = []
# for index in dev_index:
#     dev_data.append(data[index])
# test_data = []
# for index in test_index:
#     test_data.append(data[index])
# with open('dev_data.json', 'w') as f:
#     json.dump(dev_data, f)
# with open('test_data.json', 'w') as f:
#     json.dump(test_data, f)
print('aaa')