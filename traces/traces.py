#!/usr/bin/env python

# The value in seconds of the cumulative duration of the 2004 and 2005 LBNL
# network packet traces. These values are necessary to calculate the
# traffic demand in bits per second.
trace_durations= {
        '2004': 229763.868056,
        '2005': 147981.683537
        }

def trace_duration(year):
    """
    Reads a text file consisting of lines of timestamps in pairs, followed by
    a line containing anything other than a numeric timestamp:

        # Begin list of timestamps
        123123.20
        123213.22
        -----Anything
        ...
        123123.42
        123123.55
        ...
        EOF

    Returns the sum of the elapsed time between all timestamp pairs year.
    """
    year = str(year)
    total = 0
    timestamps = []
    file = open("times-{year}.txt".format(year=year), "r")

    for line in file.readlines():
        assert len(timestamps) < 3, "File Format Violation: Too many consecutive timestamps"
        if line.startswith("1"):
            timestamps.append(float(line))
        elif len(timestamps) == 2:
            diff = timestamps[1] - timestamps[0]
            assert diff > 0, "File Format Violation: Timestamps must increase"
            total += diff
            timestamps = []

    return total
