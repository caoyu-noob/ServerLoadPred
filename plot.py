import json

import matplotlib.pyplot as plt
import numpy as np

def plot_func():
    data_num = 10
    with open('../train_data/train_data_' + str(data_num) + '.json', 'r') as f:
        raw_data = json.load(f)
    x = [d[0] for d in raw_data['data']]
    x = x[:500]
    y_raw = [d[1] for d in raw_data['data']]
    y_raw = y_raw[:500]
    plt.plot(x, y_raw)
    plt.show()

if __name__ == '__main__':
    plot_func()