import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.extract_windows import extract_window
from src.remove_eyeblinks import interpol_blinks
from src.extract_average import mean_emotion, order_emotion, pupil_area
from src.model_data import model_fit, remove_bad
from pandas.errors import EmptyDataError

# parameters
frequency = 300
screen_distance_mm = 600
screen_size_mm = [478, 269]
resolution = [1920, 1080]
conversion = screen_size_mm[0]/resolution[0]
time_axis = np.arange(1792)*1/frequency

def run_pipeline(file):
    negative_dilation = []
    neutral_dilation = []
    positive_dilation = []
    for i, x in enumerate(file.iterdir()):
        # import data
        print(f'starting {i}')
        try:
            df = pd.read_csv(x, sep=';', skiprows=21, encoding='latin-1')
            window = extract_window(df)

            # remove and interpolate eye blinks
            data_no_blink = interpol_blinks(df.loc[:, ['LDX (pix)', 'LDY (pix)', 'RDX (pix)', 'RDY (pix)']], window)

            # detect emotion
            order = order_emotion(df.loc[:, 'Label'], window)
            # area = pupil_area(data_no_blink.loc[:, ['RDX (pix)', 'RDY (pix)']])
            neg, neu, pos = mean_emotion(data_no_blink.loc[:, 'RDX (pix)'], order, window)

            negative_dilation.append(neg)
            neutral_dilation.append(neu)
            positive_dilation.append(pos)
        except EmptyDataError:
            print(x)

    # parameter = [model_fit(np.mean(negative_dilation, axis=0), time_axis),
    #           model_fit(np.mean(neutral_dilation, axis=0), time_axis),
    #           model_fit(np.mean(positive_dilation, axis=0), time_axis)]

    return np.mean(negative_dilation, axis=0), np.mean(neutral_dilation, axis=0), np.mean(positive_dilation, axis=0)
    # return negative_dilation, neutral_dilation, positive_dilation
    # return parameter


def extract_params(file):
    negative_params= []
    neutral_params = []
    positive_params = []
    for i, x in enumerate(file.iterdir()):
        # import data
        print(f'starting {i}')
        try:
            df = pd.read_csv(x, sep=';', skiprows=21, encoding='latin-1')
            window = extract_window(df)

            # remove and interpolate eye blinks
            data_no_blink = interpol_blinks(df.loc[:, ['LDX (pix)', 'LDY (pix)', 'RDX (pix)', 'RDY (pix)']], window)

            # detect emotion
            order = order_emotion(df.loc[:, 'Label'], window)
            # area = pupil_area(data_no_blink.loc[:, ['RDX (pix)', 'RDY (pix)']])
            neg, neu, pos = mean_emotion(data_no_blink.loc[:, 'RDX (pix)'], order, window)

            negative_params.append(model_fit(time_axis, neg))
            neutral_params.append(model_fit(time_axis, neu))
            positive_params.append(model_fit(time_axis, pos))
        except EmptyDataError:
            print(x)

    return negative_params, neutral_params, positive_params
    # return parameter


if __name__ == '__main__':
    project_dir = Path.cwd().parent.parent
    PCA_dir = project_dir / "data/DFT"
    neg_param, neu_param, pos_param = extract_params(PCA_dir)

    # neg_param = model_fit(time_axis, neg_ave)
    # neu_param = model_fit(time_axis, neu_ave)
    # pos_param = model_fit(time_axis, pos_ave)


    print(pd.DataFrame(neg_param))

    # time_axis = np.arange(len(neg_ave)) * 1 / frequency
    #
    # plt.plot(time_axis, neg_ave, label='Negative')
    # plt.plot(time_axis, neu_ave, label='Neutral')
    # plt.plot(time_axis, pos_ave, label='Positive')
    # # plt.plot(np.vstack(bad_neg).T, alpha=0.6)
    #
    # plt.xlabel('Time (seconds)')
    # plt.ylabel('Pupil Diameter (pixels)')
    # plt.title('Average right eye vertical pupil diameter in response to different emotional stimuli in DFT patients')
    #
    # plt.tight_layout()
    # plt.legend()
    # plt.show()
    # plt.savefig('DFT_RDY.png', bbox_inches='tight', dpi=600)