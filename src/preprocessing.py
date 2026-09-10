import numpy as np
import pandas as pd
import scipy as sp
from sklearn.preprocessing import OneHotEncoder

def extract_windows(data):
    """ Extracts the time windows during the stimulus presentation based on when the croix centrale=off in the label column

    Args:
        data (pandas.DataFrame): dataset

    Returns:
        list: list of tuples for onset and offset of the stimulus presentation
    """
    stim_windows = []
    data['stim_window'] = 0

    stim_on = np.where(data.loc[:, 'Label']=='croix centrale=off')[0]+1

    for i, v in enumerate(stim_on):
        if pd.notna(data.loc[v, 'Label']):
            data.loc[v:v+1791, 'stim_window'] = 1
            stim_windows.append((v, v+1791))
        else:
            j=0
            while pd.isna(data.loc[v+j, 'Label']):
                j+=1
            stim_windows.append((v+j, v+j+1791))
            data[v+j:v+j+1791, 'stim_window'] = 1
    return stim_windows, data


def detect_blink(data, sampling_rate=300, ms2add=60):
    """ Detects the blinks in each pupil response by applying a threshold determined by 1.5 times the mean of the pupil response

    Args:
        data (pandas.DataFrame): singular pupillary response
        sampling_rate (int, optional): sampling rate of eyetracking headset
        ms2add (int, optional): padding added to detected blink. Defaults to 60ms

    Returns:
        list: list of tuples of blink onsets and offsets
    """
    mean = np.mean(data)/1.5
    
    # extract blink windows
    blink_windows = []

    blink_duration_extension = int(sampling_rate / 1000 * ms2add)
    
    # extract points that cross the threshold
    blink = []
    for i in range(len(data)-1):
        if data.iloc[i]<=mean:
            blink.append(i)
            if data.iloc[i+1]>mean:
                blink_windows.append((blink[0], blink[-1]))
                blink = []
    
    # extract time window
    for i in range(len(blink_windows)):
        mid = (blink_windows[i][1] - blink_windows[i][0])//2 + blink_windows[i][1]
        blink_windows[i] = (mid-blink_duration_extension, mid+blink_duration_extension)

    return blink_windows


def detect_blink_pd(data):

    data['blink'] = 0.0

    indexes = data[data['RDY (pix)']==0].index.values
    data.loc[indexes, 'blink'] = 1.0

    return data


def remove_blink_pd(data, sampling_rate, ms2add):
    data = data.copy()
    buffer = int(sampling_rate / 1000 * ms2add)
    indexes = data[data.blink==1.0].index.values.tolist()

    for i in indexes:
        data.loc[i-buffer:i+buffer+20, 'RDY (pix)'] = np.nan

    # data['RDY (pix)'] = data['RDY (pix)'].interpolate()
    ## .rolling() does not need interpolation

    return data


def remove_blink_padding(data, blink_windows):
    """ Replaces values where blinks are detected to NaNs

    Args:
        data (pandas.DataFrame): dataset
        blink_windows (list): list of tuples of blink onsets and offsets

    Returns:
        pandas.DataFrame: returns dataset with NaNs where blinks are
    """
    for j in blink_windows:
        data.iloc[j[0]-15: j[1]+15, :] = np.nan
    
    nan_proportion = np.sum(np.isnan(data.iloc[:, 0]))/len(data.iloc[:, 0])

    if nan_proportion > 0.3:
        data.iloc[:, :] = np.nan
        return data
    else:
        return data


def remove_blink(data, stim_windows):
    """ Detects blink, removes blink and interpolates blink

    Args:
        data (pandas.DataFrame): dataset
        stim_windows (list): list of tuples of stimulus onset and offset

    Returns:
        pandas.DataFrame: dataset with detected, removed and interpolated blinks
    """
    df_clean = data.copy()
    for v in stim_windows:
        blinks = detect_blink(df_clean.loc[v[0]:v[1], 'RDX (pix)'])
        if blinks:
            df_clean.iloc[v[0]:v[1], :] = remove_blink_padding(df_clean.iloc[v[0]:v[1], :], blinks)
            try:
                df_clean.iloc[v[0]:v[1], :] = df_clean.iloc[v[0]:v[1], :].interpolate(method='linear')
            except IndexError:
                continue
    return df_clean


def extract_stim(data, stim_windows):
    """ Get the order of the presentation of the valence and category

    Args:
        data (pandas.DataFrame): dataset
        stim_windows (list): list of tuples of stimulus onset and offset

    Returns:
        list: list of tuples of category and valence for each stimulus presentation
    """
    data['category'] = None
    data['valence'] = None
    data['trial'] = None

    order_stim = []
    for i, stim in enumerate(stim_windows):
        a = data.loc[stim[0], 'Label'].split('_')
        order_stim.append((a[1], a[2]))
        data.loc[stim[0]:stim[1], 'trial'] = int(a[0])
        data.loc[stim[0]:stim[1], 'category'] = a[1]
        data.loc[stim[0]:stim[1], 'valence'] = a[2]
    return order_stim, data.drop(columns='stim_window')


def add_stim_columns(data, stim_windows, order_stim, trial_len, sampling_rate):
    """ Adds category, valence as well as trial and time columns to dataset

    Args:
        data (pandas.DataFrame): dataset
        stim_windows (list): list of tuples of stimulus onset and offset
        order_stim (list): list of tuples of category and valence for each stimulus presentation
        trial_len (int): length of trial of pupillary response (6 seconds, 300Hz, hence around 1800) -> 1792 is used since some trials are slightly shorter -> ensures all response are exactly the same length
        sampling_rate (int): sampling rate of eyetracking headset

    Returns:
        pandas.DataFrame: dataset with added columns
    """
    data['category'] = None
    data['valence'] = None
    data['trial'] = None
    data['time'] = None

    x = np.linspace(0, trial_len/sampling_rate, trial_len)

    for i, v in enumerate(stim_windows):
        data.loc[v[0]:v[1], 'category'] = order_stim[i][0]
        data.loc[v[0]:v[1], 'valence'] = order_stim[i][1]
        data.loc[v[0]:v[0]+(trial_len-1), 'trial'] = i
        for j in range(trial_len):
            data.loc[v[0]+j, 'time'] = x[j]

    return data


def smooth(data, stim_windows, column, lfreq):
    """ Smooth the timeseries

    Args:
        data (pandas.DataFrame): A given timeseries
        stim_windows (list): list of tuples of stimulus onset and offset
        column (list): columns to apply the filter to
        lfreq (int): low pass frequency

    Returns:
        pandas.DataFrame: returns the smoothed timeseries
    """
    
    df_smooth = data.copy()
    for v in stim_windows:
        for col in column:
            sos = sp.signal.butter(5, Wn=lfreq, fs=300, btype='low', output='sos')
            df_smooth.loc[v[0]:v[1], col] = sp.signal.sosfiltfilt(sos, df_smooth.loc[v[0]:v[1], col])
    return df_smooth


def baseline_correct(data, stim_windows, column):
    for col in column:
        for v in stim_windows:
            mean = np.mean(data.loc[v[0]-60:v[0], col])
            data.loc[v[0]:v[1], col] = data.loc[v[0]:v[1], col]-mean

    return data


def find_bad_signals(data, stim_windows, column):
    removed = 0
    total = 0
    for col in column:
        for i, v in enumerate(stim_windows):
            if data.loc[v[0]:v[1], col].mean() > 0:
                data.loc[v[0]:v[1], col] = np.nan
            if data.loc[v[0]:v[0]+250, col].abs().max() > 100:
                data.loc[v[0]:v[1], col] = np.nan
    return data


def trials_only(data):
    return data.dropna(axis=0)


def remove_bad_signals(data, column, threshold):
    """ Removes 

    Args:
        data (_type_): _description_
        column (_type_): _description_
        threshold (_type_): _description_

    Returns:
        _type_: _description_
    """
    for col in column:
        col_removed = f'{col}_removed'
        bad_ratio = data.groupby('id')[col_removed].transform('mean')
        data.loc[bad_ratio > threshold, col]=np.nan

    return data


def one_hot_encoder(data):
    ohe = OneHotEncoder(sparse_output=False).set_output(transform="pandas")
    ohetransform = ohe.fit_transform(data[['category', 'valence']])
    data_ohe = pd.concat([data, ohetransform], axis=1).drop(columns=['LDX (pix)_removed', 'LDY (pix)_removed', 'RDX (pix)_removed', 'RDY (pix)_removed'])

    return data_ohe