import numpy as np
from lmfit import Model
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import minmax_scale


def sim_func(x, D_0, D_1, t_0, tau_1, tau_2):
    t = np.linspace(0, x, x)
    delta = np.where(t < t_0, 0, 1)
    D = D_0 - D_1*delta*((t-t_0)/tau_1)*np.exp(-((t-t_0)/tau_2))

    return D


def sim_prf(x, D_0, t_0, n, t_max):
    t = np.linspace(0, x, x)
    delta = np.where(t < t_0, 0, 1)
    D = D_0 - (delta*(t**n) * np.exp((-n*t)/t_max))

    return D


# c: peak amplitude
# k and theta: shape and time of peak amplitude
def sim_gamma(x, t_0, c, k, theta):
    t = np.linspace(0, x, x)
    delta = np.where(t < t_0, 0, 1)
    D = -delta *c*(t**(k-1)*np.exp(-t/theta))
    return D

### Examples parameters
# func = sim_func(x=1800, D_0=0, D_1=6, t_0=100, tau_1=20, tau_2=400)
# prf = sim_prf(x=1800, D_0=0, t_0=100, n=0.8, t_max=300)
# gamma = sim_gamma(x=400, t_0=50, c=0.4, k=3, theta=50)



def model_fit(x, y, weight):
    gmodel = Model(sim_func)
    params = gmodel.make_params()

    params['D_1'].set(value=4.0, min=0.05, max=7.0)
    params['t_0'].set(value=20, min=0.0, max=200)
    params['tau_1'].set(value=10.0, min=7.0, max=20.0)
    params['tau_2'].set(value=400, min=100, max=800.0)
    # params['length'].set(value=len(x))

    result = gmodel.fit(y, params, x=x, D_0=y.iloc[10], weights=weight)
    return result.best_values, result.best_fit




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
