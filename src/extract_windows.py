import numpy as np

def extract_window(data):
    stim_windows = []
    stim_on = np.where(data.loc[:, 'Label']=='croix centrale=off')[0]+1
    if np.size(stim_on) > 0:
        for i in stim_on:
            try:
                stim_off = np.where(data.loc[i: i+1805, 'Label']=='consignes=on')[0][0]-1
                stim_windows.append((i, i+stim_off))
            except IndexError:
                print("consignes=on not found")
    else:
        raise ValueError('croix centrale=off does not exist')
    return stim_windows