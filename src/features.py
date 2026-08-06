import numpy as np

def mean(data, category, valence, eye):
    ave = []
    trials=np.unique(data.loc[:, 'trial'][(data['category']==category) & (data['valence']==valence)])

    for trial in trials:
        ave.append(data.loc[:, eye][data['trial']==trial])

    return np.mean(ave, axis=0)

def latency(data):
    return np.argmax(np.gradient(data[:300]))

def max_diam(data):
    return max(data)

def mean_variation(data):
    return data[-1]/data[300]