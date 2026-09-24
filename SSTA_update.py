from numpy.typing import NDArray
from tqdm import tqdm
import numpy as np


def ssta(
    data: NDArray[np.floating],
    start_channel: int = 1650,
    end_channel: int = 8500,
    num: int = 50,
    thresh: int = 2,
    get_mask: bool = True,
):
    """Performs the spatial short term average on an array of DAS data

    Keyword arguments:
        data -- A 2D array of DAS data
        start_channel -- An int. First column (channel) of data the method will use
        end_channel -- An int. Last column (channel of data the method will use)
        num -- An int. Number of rows (time samples) used to perform the SSTA
        thresh -- An int. The threshold ratio above which indicates an event has occured
        get_mask -- A boolean flag indicating the returned arrays format:
            True - -1 (Not processed), 0 (No Event), 1 (Event)
            False - -1 (Not processed), ratio value
    """

    n_channels = data.shape[1]
    mask: list[list[int]] = []
    with tqdm(total=len(data)) as pbar:
        for i in range(0, len(data)):
            if not (i - num <= 0):
                temp = np.array(data[i - num : i, start_channel:end_channel])
                temp = abs(temp)
                mean = temp.mean()

                channel_mask: list[int] = []
                for channel in range(len(temp[0])):
                    cdata = np.array(temp.transpose()[channel, :])
                    cdata = abs(cdata)
                    if cdata.mean() / mean > thresh:

                        if get_mask:
                            # 1 or 0 mask
                            channel_mask.append(1)
                        else:
                            # gives actual amplitude of the data
                            channel_mask.append(abs(data[i, channel]))
                    else:
                        channel_mask.append(0)

                mask.append(channel_mask)
            pbar.update()

    return np.pad(
        np.array(mask),
        [(num, 0), (start_channel, (n_channels - end_channel))],
        mode="constant",
        constant_values=-1,
    )
