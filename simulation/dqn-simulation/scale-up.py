import math
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


# For timeseries
CONFERENCE_COUNT_SCALE = 15
PARTICIPANT_COUNT_SCALE = 40

# For timeseries-V2
#CONFERENCE_COUNT_SCALE = 5
#PARTICIPANT_COUNT_SCALE = 10

df = pd.read_csv('dbV3.csv')
conference_count_over_time = df['conferences'] * CONFERENCE_COUNT_SCALE
participant_count_over_time = df['participants'] * PARTICIPANT_COUNT_SCALE
smoothed_conferences = savgol_filter(conference_count_over_time, 201, 1)
smoothed_participants = savgol_filter(participant_count_over_time, 201, 1)
for i in range(len(smoothed_conferences)):
    if int(smoothed_conferences[i]) == 0:
        smoothed_participants[i] = 0

with open('timeseries.csv', 'w') as f:
    f.write("conference_count,participant_count\n")
    for i in range(len(smoothed_conferences)):
        f.write(f"{int(smoothed_conferences[i])},{int(smoothed_participants[i])}\n")
