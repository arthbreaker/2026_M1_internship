import numpy as np
from numpy.f2py.crackfortran import expectbegin


def order_emotion(data, stim_window):
    order = []
    for stim in stim_window:
        try:
            order.append(data.loc[stim[0]].split('_')[2])
        except AttributeError:
            print(data.loc[stim[0]])
    return order


def mean_emotion(data, order, window):
    neg_idx = [i for i, x in enumerate(order) if x == "negative"]
    neu_idx = [i for i, x in enumerate(order) if x == "neutre"]
    pos_idx = [i for i, x in enumerate(order) if x == "positive"]

    pupil = []
    for stim in window:
        pupil.append(data.iloc[stim[0]: stim[0] + 1792])

    neg = [pupil[i][0:1792] for i in neg_idx] # the [0:1792] ensures they're all the same length, easier to average later on
    neu = [pupil[i][0:1792] for i in neu_idx]
    pos = [pupil[i][0:1792] for i in pos_idx]

    neg_ave = np.mean(neg, axis=0)
    neu_ave = np.mean(neu, axis=0)
    pos_ave = np.mean(pos, axis=0)

    return neg_ave, neu_ave, pos_ave

def pupil_area(data):
    return (data.iloc[:, 0]/2) * (data.iloc[:, 1]/2) * np.pi