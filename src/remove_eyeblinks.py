import numpy as np

def detect_blink(data, sampling_rate=300, ms2add=60):
    """
    --- Parameters ---
    data: all data streams from a time window
    """

    # Set threshold (threshold is set at: mean - 1.5*mean)
    series = data.loc[:, 'RDX (pix)'] # RDX channel is used because eye blinks are the most clear in this channel
    mean = np.mean(series ) /1.5

    # extract blink windows
    blink_windows = []

    blink_duration_extension = int(sampling_rate / 1000 * ms2add)

    # extract points that cross the threshold
    blink = []
    for i in range(len(series) - 1):
        if series.iloc[i] <= mean:
            blink.append(i)
            if series.iloc[i+1] > mean:
                blink_windows.append((blink[0], blink[-1]))
                blink = []

    # extract time window
    for i in range(len(blink_windows)):
        mid = (blink_windows[i][1] - blink_windows[i][0] )//2 + blink_windows[i][1]
        blink_windows[i] = (mid -blink_duration_extension, mid +blink_duration_extension)

    return blink_windows

def remove_blink_padding(data, blink_windows):
    for blink in blink_windows:
        data.iloc[blink[0]-15:blink[1]+15] = np.nan
    return data


# The following function was adapted from Sina Kling's eyeprep repository
# URL: https://github.com/sinaklg/eyeprep
# Retrieved on: June 2, 2026

def interpol_nans(eyetracking_data):
    """
    Interpolate missing (NaN) values in eye-tracking data, filling gaps with the nearest valid data.

    Args:
        eyetracking_data (np.array): Eye-tracking data containing NaN values.

    Returns:
        np.array: Interpolated data with NaNs replaced.
    """

    nan_indices = np.isnan(eyetracking_data)

    # Fill NaNs at the start and end with nearest valid values
    if nan_indices[0]:  # If the first value is NaN
        first_valid_idx = np.where(~nan_indices)[0][0]
        eyetracking_data[:first_valid_idx] = eyetracking_data[first_valid_idx]

    if nan_indices[-1]:  # If the last value is NaN
        last_valid_idx = np.where(~nan_indices)[0][-1]
        eyetracking_data[last_valid_idx + 1:] = eyetracking_data[last_valid_idx]

    # Now interpolate remaining NaNs
    eyetracking_no_nans = np.nan_to_num(eyetracking_data)
    eyetracking_signal_interpolated = np.interp(np.arange(len(eyetracking_data)),
                                                np.where(~nan_indices)[0],
                                                eyetracking_no_nans[~nan_indices])

    return eyetracking_signal_interpolated


def interpol_blinks(data, window):
    for v in window:
        blinks = detect_blink(data.iloc[v[0]:v[1], :])
        if len(blinks) > 0:
            data.iloc[v[0]:v[1], :] = remove_blink_padding(data.iloc[v[0]:v[1], :], blinks)
            for i in range(len(data.T)):
                data.iloc[v[0]:v[1], i] = interpol_nans(np.array(data.iloc[v[0]:v[1], i]))
    return data