from scipy.signal import butter, sosfilt
import numpy as np


def butter_bandpass(lowcut, highcut, fs, order):
    """Butterworth Bandpass Filter Creator, returns a bandpass filter

    Keyword arguments:
        lowcut -- lower bound of the filter -1 for no lower bound  (lowpass)
        highcut -- higher bound of the filter -1 for no upper bound (highpass)
        fs -- frequency of recorded data
        order -- the order of the filter
    """

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    if lowcut == -1:
        sos = butter(order, high, analog=False, btype="low", output="sos")
    elif highcut == -1:
        sos = butter(order, low, analog=False, btype="high", output="sos")
    else:
        sos = butter(order, [low, high], analog=False, btype="band", output="sos")
    return sos


def butter_bandpass_filter(data, fs, lowcut=-1, highcut=-1, order=6):
    """Applies a butterworth filter to a channel of data and returns the filtered data

    Keyword arguments:
        data -- single channel of data for filtering
        fs -- frequency of recording
        lowcut -- lower bound for filter defaults to -1 meaning no lower bound
        highcut -- higher bound for filter defaults to -1 meaning no higher bound
        order -- the order of the filter
    """
    sos = butter_bandpass(lowcut, highcut, fs, order)
    y = sosfilt(sos, data)
    return y


def filter_waterfall(some_data, fs, lowcut=-1, highcut=-1, order=6):
    """Applies a Butterworth filter to a full 2D TDMS np array

    Keyword arguments:
        some_data -- TDMS np array
        fs -- frequency of recording
        lowcut -- lower bound for filter defaults to -1 meaning no lower bound
        highcut -- higher bound for filter defaults to -1 meaning no higher bound
        order -- the order of the filter
    """
    filtered_data = np.empty(some_data.shape, type(some_data[0, 0]))

    for samp_num in range(0, len(some_data[0])):
        # print(samp_num)
        time_sample = some_data.transpose()[samp_num, :]

        filtered_signal = butter_bandpass_filter(
            time_sample, lowcut=lowcut, highcut=highcut, fs=fs, order=order
        )

        for i, point in enumerate(filtered_signal):
            filtered_data[i, samp_num] = point

    return filtered_data
