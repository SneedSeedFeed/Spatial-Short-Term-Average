import numpy as np


def cluster_association_points(
    cluster_data,
    sample_overlap,
    channel_overlap,
    fs,
    min_width,
    max_width,
    min_length,
    max_length,
):

    # massive clusters need filtering or they may skew the association
    copy = []
    for c in cluster_data:
        if (
            min_width <= c[3] - c[2] < max_width
            and min_length < c[1] - c[0] < max_length
        ):
            copy.append(c)

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
                        c = [0, 0, 0, 0]

                        # find the midpoint
                        b_mp = b[2] + int((b[3] - b[2]) / 2)

                        # midpoint extended overlap
                        if (
                            (b_mp - xOff if b_mp - xOff < b[2] else b[2])
                            <= a[2]
                            <= (b_mp + xOff if b_mp + xOff > b[3] else b[3])
                            <= a[3]
                            or a[2]
                            <= (b_mp - xOff if b_mp - xOff < b[2] else b[2])
                            <= a[3]
                            <= (b_mp + xOff if b_mp + xOff > b[3] else b[3])
                            or (b_mp - xOff if b_mp - xOff < b[2] else b[2])
                            <= a[2]
                            <= a[3]
                            <= (b_mp + xOff if b_mp + xOff > b[3] else b[3])
                            or a[2]
                            <= (b_mp - xOff if b_mp - xOff < b[2] else b[2])
                            <= (b_mp + xOff if b_mp + xOff > b[3] else b[3])
                            <= a[3]
                        ):
                            c[2] = min([a[2], b[2]])
                            c[3] = max([a[3], b[3]])
                            x_overlap = True

                        # adaptive time difference
                        # yOff = b[1] - b[0]

                        # boundry extended overlap
                        if (
                            b[0] - yOff <= a[0] <= b[1] + yOff <= a[1]
                            or a[0] <= b[0] - yOff <= a[1] <= b[1] + yOff
                            or b[0] - yOff <= a[0] <= a[1] <= b[1] + yOff
                            or a[0] <= b[0] - yOff <= b[1] + yOff <= a[1]
                        ):
                            c[0] = min([a[0], b[0]])
                            c[1] = max([a[1], b[1]])
                            y_overlap = True

                        if x_overlap and y_overlap:
                            stable = False

                            copy[i][0] = copy[i][0] if c[0] == 0 else c[0]
                            copy[i][1] = copy[i][1] if c[1] == 0 else c[1]
                            copy[i][2] = copy[i][2] if c[2] == 0 else c[2]
                            copy[i][3] = copy[i][3] if c[3] == 0 else c[3]
                            copy[i][4] = copy[i][4] + copy[j][4]

                            copy[i][5] = np.array(
                                [
                                    np.append(copy[i][5][0], copy[j][5][0]),
                                    np.append(copy[i][5][1], copy[j][5][1]),
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
