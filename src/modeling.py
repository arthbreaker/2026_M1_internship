import numpy as np
from lmfit import Model
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import minmax_scale


def sim_func(t, D_0, D_1, t_0, tau_1, tau_2):
    delta = np.where(t < t_0, 0, 1)
    
    D = D_0 - D_1*delta*((t-t_0)/tau_1)*np.exp(-((t-t_0)/tau_2))
    return D


def model_fit(x, y):
    x = np.linspace(0, len(y)/300, len(y))
    gmodel = Model(sim_func)
    params = gmodel.make_params()

    params['D_1'].set(value=250.0, min=0.0, max=2000)
    params['t_0'].set(value=0.3, min=0.0, max=6.0)
    params['tau_1'].set(value=1.0, min=0.01, max=50.0)
    params['tau_2'].set(value=0.8, min=0.01, max=50.0)
    
    result = gmodel.fit(y, params, t=x, D_0=y[0], weights=np.repeat([20.0, 0.2], [300, 1492]))

    # plt.plot(x, y, '-')
    # plt.plot(x, result.init_fit, '--', label='initial fit')
    # plt.plot(x, result.best_fit, '-', label='best fit')
    # plt.legend()
    # sns.despine()
    # plt.show()
    
    return result.best_values



def sim_model(D_0, D_1, t_0, tau_1, tau_2, length):
    D = []
    for t in range(int(length)):
        delta = 1
        if t < t_0:
            delta = 0
        D.append(D_0 - D_1*delta*((t-t_0)/tau_1)*np.exp(-((t-t_0)/tau_2)))
    return D

def model_fit(x, y):
    gmodel = Model(sim_model)
    params = gmodel.make_params()

    params['D_1'].set(value=4.0, min=0.05, max=7.0)
    params['t_0'].set(value=20, min=0.0, max=200)
    params['tau_1'].set(value=10.0, min=7.0, max=20.0)
    params['tau_2'].set(value=400, min=100, max=800.0)
    params['length'].set(value=len(x))

    result = gmodel.fit(y, params, x=x, D_0=y[10])

    return result.best_values

def calc_mse(data, params):
    real = minmax_scale(data)
    predict = minmax_scale(sim_model(data.iloc[10], params['D_1'], params['t_0'], params['tau_1'], params['tau_2'], len(data)))
    mse = mean_squared_error(real, predict)
    return True if mse > 0.1 else False

# def remove_bad(data, params):
#     bad_channels = []
#     for i in data:
#         if calc_mse(i, params):
#             continue
#         else:
#             bad_channels.append(i)
#     return bad_channels

def remove_bad(data, order, window, params):
    neg_idx = [i for i, x in enumerate(order) if x == "negative"]
    neu_idx = [i for i, x in enumerate(order) if x == "neutre"]
    pos_idx = [i for i, x in enumerate(order) if x == "positive"]

    neg_bad = []
    neu_bad = []
    pos_bad = []

    for i, v in enumerate(neg_idx):
        if calc_mse(data.iloc[window[i][0]: window[i][1]], params[0]):
            neg_bad.append(data.iloc[window[i][0]: window[i][1]])


    # pupil = []
    # for stim in window:
    #     pupil.append(data.iloc[stim[0]: stim[0] + 1792])
    #
    # neg = [pupil[i][0:1792] for i in neg_idx]
    # neu = [pupil[i][0:1792] for i in neu_idx]
    # pos = [pupil[i][0:1792] for i in pos_idx]
    #
    # neg_bad = []
    # neu_bad = []
    # pos_bad = []
    #
    # for i in neg:
    #     if calc_mse(i, params[0]):
    #         neg_bad.append(i)
    #     else:
    #         continue
    #
    # for i in neu:
    #     if calc_mse(i, params[1]):
    #         neu_bad.append(i)
    #     else:
    #         continue
    #
    # for i in pos:
    #     if calc_mse(i, params[2]):
    #         pos_bad.append(i)
    #     else:
    #         continue   #

    return len(neg_bad), 6-len(neg_bad)
