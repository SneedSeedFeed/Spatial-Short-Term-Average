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
    xOff = channel_overlap / 2

    # fixed time diff
    yOff = (sample_overlap * fs) / 2

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

                        # find the midpoint
                        b_mp = b.channel_start + int((b.channel_end - b.channel_start) / 2)

                        # midpoint extended overlap
                        if (
                            (b_mp - xOff if b_mp - xOff < b.channel_start else b.channel_start)
                            <= a.channel_start
                            <= (b_mp + xOff if b_mp + xOff > b.channel_end else b.channel_end)
                            <= a.channel_end
                            or a.channel_start
                            <= (b_mp - xOff if b_mp - xOff < b.channel_start else b.channel_start)
                            <= a.channel_end
                            <= (b_mp + xOff if b_mp + xOff > b.channel_end else b.channel_end)
                            or (b_mp - xOff if b_mp - xOff < b.channel_start else b.channel_start)
                            <= a.channel_start
                            <= a.channel_end
                            <= (b_mp + xOff if b_mp + xOff > b.channel_end else b.channel_end)
                            or a.channel_start
                            <= (b_mp - xOff if b_mp - xOff < b.channel_start else b.channel_start)
                            <= (b_mp + xOff if b_mp + xOff > b.channel_end else b.channel_end)
                            <= a.channel_end
                        ):
                            c[2] = min([a.channel_start, b.channel_start])
                            c[3] = max([a.channel_end, b.channel_end])
                            x_overlap = True

                        # adaptive time difference
                        # yOff = b.sample_end - b.sample_start

                        # boundry extended overlap
                        if (
                            b.sample_start - yOff <= a.sample_start <= b.sample_end + yOff <= a.sample_end
                            or a.sample_start <= b.sample_start - yOff <= a.sample_end <= b.sample_end + yOff
                            or b.sample_start - yOff <= a.sample_start <= a.sample_end <= b.sample_end + yOff
                            or a.sample_start <= b.sample_start - yOff <= b.sample_end + yOff <= a.sample_end
                        ):
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
