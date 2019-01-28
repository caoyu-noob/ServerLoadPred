from statsmodels.tsa.arima_model import ARIMA, ARIMAResults
from tqdm import tqdm
import numpy as np
import json

def fit_ARIMA():
    init_bic = float("inf")
    init_p = 0
    init_q = 0
    init_d = 1
    maxLag = 8
    with open('dev_data.json', 'r') as f:
        raw_data = json.load(f)
    data = []
    for index in range(len(raw_data)):
        if index % 10 == 0:
            data.extend([float(di[1]) for di in raw_data[index]['data']])
        # data.extend([float(di[1]) for di in d['data']])

    init_properModel = None
    for p in np.arange(0, maxLag):
        for q in np.arange(0, maxLag):
            for d in np.arange(1, 3):
                print('test p = %d, d = %d, q = %d'%(p, d, q))
                model = ARIMA(data, order=(p, d, q))
                ## bug cuased by statsmodels lib, if do not set them then error occurs during saving
                model.dates = None
                model.freq = None
                model.missing = None
                try:
                    results_ARIMA = model.fit(disp=-1, method='css')
                except:
                    continue
                bic = results_ARIMA.bic
                print('current bic = %.4f'%(bic))
                if bic < init_bic:
                    print('It BEATS the previous best p, d and q, current p = %d, d = %d, q = %d'%(p, d, q))
                    init_p = p
                    init_q = q
                    init_d = d
                    init_properModel = results_ARIMA
                    init_bic = bic
                    results_ARIMA.save('ARIMA_best')
                    print('Saved the best model')
    print('p is :' + str(init_p))
    print('q is :' + str(init_q))
    print('d is :' + str(init_d))

if __name__ == '__main__':
    fit_ARIMA()
