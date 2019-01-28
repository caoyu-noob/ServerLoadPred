from statsmodels.tsa.arima_model import ARMA, ARMAResults
from tqdm import tqdm
import numpy as np
import json

def fit_ARMA():
    init_bic = 6933580.6994
    init_p = 0
    init_q = 0
    maxLag = 8
    with open('dev_data.json', 'r') as f:
        raw_data = json.load(f)
    data = []
    for index in range(len(raw_data)):
        if index % 10 == 0:
            data.extend([float(di[1]) for di in raw_data[index]['data']])
        # data.extend([float(di[1]) for di in d['data']])

    init_properModel = None
    for p in np.arange(5, maxLag):
        for q in np.arange(0, maxLag):
            print('test p = %d, q = %d'%(p, q))
            model = ARMA(data, order=(p, q))
            try:
                results_ARMA = model.fit(disp=-1, method='css')
            except:
                continue
            bic = results_ARMA.bic
            print('current bic = %.4f'%(bic))
            if bic < init_bic:
                print('It BEATS the previous best p and q, current p = %d, q = %d'%(p, q))
                init_p = p
                init_q = q
                init_properModel = results_ARMA
                init_bic = bic
                results_ARMA.save('ARMA_best')
                print('Saved the best model')
    print('p is :' + str(init_p))
    print('q is :' + str(init_q))

def predict(coef, history):
	yhat = 0.0
	for i in range(1, len(coef)+1):
		yhat += coef[i-1] * history[-i]
	return yhat

def test_ARMA():
    fited_model = ARMAResults.load('best')
    ar_coef, ma_coef = fited_model.arparams, fited_model.maparams
    resid = fited_model.resid
    ar_num, ma_num = ar_coef.size, ma_coef.size
    with open('dev_data.json', 'r') as f:
        test_data = json.load(f)
    rmse_sum, mae_sum, mape_sum = 0, 0, 0
    skip_mape_count = 0
    for data in tqdm(test_data):
        history = []
        label = []
        pred = []
        for index, d in enumerate(data['data']):
            if index >= ar_num:
                label.append(d[1])
                cur_resid = np.random.choice(resid, ma_num)
                yhat = predict(ar_coef, history) + predict(ma_coef, cur_resid)
                pred.append(yhat)
            history.append(d[1])
        label = np.array(label)
        pred = np.array(pred)
        rmse = np.sqrt(np.mean(np.square(label - pred)))
        mae = np.mean(np.abs(label - pred))
        mape_mask = label > 1
        mape = np.sum(np.abs(label - pred) / (label + 0.001) * mape_mask)
        mape_count = np.sum(mape_mask)
        if mape_count != 0:
            mape = mape / np.sum(mape_mask)
            mape_sum += mape
        else:
            skip_mape_count += 1
        rmse_sum += rmse
        mae_sum += mae
    print('RMSE is {:.3f}'.format(rmse_sum / len(test_data)))
    print('MAE is {:.3f}'.format(mae_sum / len(test_data)))
    print('MAPE is {:.5f}'.format(mape_sum / (len(test_data) - skip_mape_count)))


if __name__ == '__main__':
    test_ARMA()