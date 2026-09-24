from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


type Number = int | float | np.integer | np.floating
type NumberArray = NDArray[np.integer | np.floating]


@dataclass(slots=True)
class Cluster:
    sample_start: Number
    sample_end: Number
    channel_start: Number
    channel_end: Number
    point_count: Number
    points: NumberArray

    def __getitem__(self, key: int) -> Number | NumberArray:
        match key:
            case 0: return self.sample_start
            case 1: return self.sample_end
            case 2: return self.channel_start
            case 3: return self.channel_end
            case 4: return self.point_count
            case 5: return self.points
            case _: raise ValueError(f"{key} should be an integer between 0 and 5")


def _intervals_overlap(
    a_start: Number, a_end: Number, b_start: Number, b_end: Number
) -> bool:
    # bool cast needed due to numpy bool not being python bool
    return bool(
        b_start <= a_start <= b_end <= a_end
        or a_start <= b_start <= a_end <= b_end
        or b_start <= a_start <= a_end <= b_end
        or a_start <= b_start <= b_end <= a_end
    )


def _channels_overlap(a: Cluster, b: Cluster, channel_offset: float) -> bool:
    midpoint = b.channel_start + int((b.channel_end - b.channel_start) / 2)
    start = midpoint - channel_offset
    end = midpoint + channel_offset
    extended_start = start if start < b.channel_start else b.channel_start
    extended_end = end if end > b.channel_end else b.channel_end
    return _intervals_overlap(
        a.channel_start, a.channel_end, extended_start, extended_end
    )


def _samples_overlap(a: Cluster, b: Cluster, sample_offset: float) -> bool:
    return _intervals_overlap(
        a.sample_start, a.sample_end,
        b.sample_start - sample_offset, b.sample_end + sample_offset,
    )


def cluster_association_points(
    cluster_data: Iterable[Cluster],
    sample_overlap: float,
    channel_overlap: float,
    fs: float,
    min_width: float,
    max_width: float,
    min_length: float,
    max_length: float,
) -> list[Cluster]:
    # massive clusters need filtering or they may skew the association
    copy: list[Cluster] = []
    for cluster in cluster_data:
        if (
            min_width <= cluster.channel_end - cluster.channel_start < max_width
            and min_length < cluster.sample_end - cluster.sample_start < max_length
        ):
            copy.append(cluster)

    stable = False

    # averaging distance for a signal
    channel_offset = channel_overlap / 2

    # fixed time diff
    sample_offset = (sample_overlap * fs) / 2

    while not stable:
        stable = True

        for i, a in enumerate(copy):
            if i == len(copy):
                print("end")
            else:
                for j, b in enumerate(copy):
                    if i != j:
                        x_overlap = False
                        y_overlap = False
                        c: list[Number] = [0, 0, 0, 0]

                        if _channels_overlap(a, b, channel_offset):
                            c[2] = min([a.channel_start, b.channel_start])
                            c[3] = max([a.channel_end, b.channel_end])
                            x_overlap = True

                        if _samples_overlap(a, b, sample_offset):
                            c[0] = min([a.sample_start, b.sample_start])
                            c[1] = max([a.sample_end, b.sample_end])
                            y_overlap = True

                        if x_overlap and y_overlap:
                            stable = False

                            copy[i].sample_start = copy[i].sample_start if c[0] == 0 else c[0]
                            copy[i].sample_end = copy[i].sample_end if c[1] == 0 else c[1]
                            copy[i].channel_start = copy[i].channel_start if c[2] == 0 else c[2]
                            copy[i].channel_end = copy[i].channel_end if c[3] == 0 else c[3]
                            copy[i].point_count = copy[i].point_count + copy[j].point_count

                            copy[i].points = np.array(
                                [
                                    np.append(copy[i].points[0], copy[j].points[0]),
                                    np.append(copy[i].points[1], copy[j].points[1]),
                                ]
                            )

                            # print(b)
                            # print(copy[i+j+1])
                            try:
                                del copy[j]
                            except IndexError:
                                print("Index Error")
                            break

            if not stable:
                break
    return copy
