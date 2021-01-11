import math
import json
import random
import time
import calendar
import pickle
import os
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, chi
from collections import namedtuple
from statistics import pvariance, pstdev


df = pd.read_csv('dbV3.csv')
timeseries = pd.read_csv('timeseries.csv')

# Loss approximation
def loss_from_nearest_points(c, p, tj, ij):
    PARTITIONS = 3
    losses = []
    #conf_partitions = [0, 1, 2, 3]
    part_partitions = [1, 5, 9, 13]
    tj_partitions = [1, 3, 5, 7]
    ij_partitions = [0, 2, 4, 7]

    for i in range(PARTITIONS):
        #curr_c = conf_partitions[i]
        #d = df[df['conferences'] == curr_c]
        flag = True
        for curr_p in range(part_partitions[i], part_partitions[i+1]):
            if not flag:
                break
            d1 = df[df['participants'] == curr_p]
            for curr_tj in range(tj_partitions[i], tj_partitions[i+1]):
                if not flag:
                    break
                d2 = d1[d1['jvb_num'] == curr_tj]
                for curr_ij in range(ij_partitions[i], ij_partitions[i+1]):
                    d3 = d2[d2['zero_conf'] == curr_ij]
                    if len(d3) > 0:
                        loss = d3['loss'].mean()
                        participants_scale = p / curr_p
                        curr_active_jvb_count = curr_tj - curr_ij
                        if (tj - ij) == 0 or curr_active_jvb_count == 0:
                            continue
                        active_jvbs_scale = (tj - ij) / curr_active_jvb_count
                        loss_scale = participants_scale / active_jvbs_scale
                        estimated_loss = loss * loss_scale
                        losses.append(estimated_loss)
                        flag = False
                        break

    return np.mean(losses)


##############
# Simulation #
##############
curr_time = time.time()
print("Starting simulation...")

# list of [jvb id, conference count] pair of currently running JVBs
# selected with round-robin, removed with graceful shutdown
curr_jvbs = [[i, 0] for i in range(25)]
losses_history = []
losses_dict = pickle.load(open('losses_dict.pkl', 'rb'))

conf_count_over_time = timeseries['conference_count']
part_count_over_time = timeseries['participant_count']

for i in range(len(conf_count_over_time)):
    c1 = int(conf_count_over_time[i])
    p1 = int(part_count_over_time[i])

    # update conferences
    try:
        new_c = c1 - int(conf_count_over_time[i-1])
    except:
        new_c = c1
    if new_c > 0:
        # assign conferences
        for c in range(new_c):
            jvb_conferences = [x[1] for x in curr_jvbs]
            least_loaded_idx = np.argmin(jvb_conferences)
            curr_jvbs[least_loaded_idx][1] += 1
    elif new_c < 0:
        # remove conferences
        for c in range(abs(new_c)):
            for j in curr_jvbs:
                if j[1] > 0:
                    j[1] -= 1
                    break

    j1 = len(curr_jvbs)
    z1 = len(list(filter(lambda x: x[1] == 0, curr_jvbs)))
    avg_loss = losses_dict.get(c1, {}).get(p1, {}).get(j1, {}).get(z1, -1)
    if avg_loss == -1:
        avg_loss = df[
            (df['conferences'] == c1)
            & (df['participants'] == p1)
            & (df['jvb_num'] == j1)
            & (df['zero_conf'] == z1)
        ]['loss'].mean()
        if pd.isna(avg_loss):
            if c1 == 0 or p1 == 0:
                avg_loss = 0
            else:
                avg_loss = df[
                    (df['conferences'] >= c1-1) & (df['conferences'] <= c1+1)
                    & (df['participants'] >= p1-1) & (df['participants'] <= p1+1)
                    & (df['jvb_num'] >= j1-1) & (df['jvb_num'] <= j1+1)
                    & (df['zero_conf'] >= z1-1) & (df['zero_conf'] <= z1+1)
                ]['loss'].mean()
                if pd.isna(avg_loss):
                    avg_loss = loss_from_nearest_points(c1, p1, j1, z1)
        losses_dict.setdefault(c1, {}).setdefault(p1, {}).setdefault(j1, {})[z1] = avg_loss
    losses_history.append(avg_loss)

    if (i+1) % 500 == 0:
        print(f"Timesteps passed: {i+1}", end="\r")
print(f"\nSimulation finished in {time.time() - curr_time} seconds")


#################
# Visualization #
#################
plt.figure(figsize=(16, 9))
plt.subplot(411)
plt.title("Conferences")
plt.plot(np.arange(len(conf_count_over_time)), conf_count_over_time)
plt.subplot(412)
plt.title("Participants")
plt.plot(np.arange(len(part_count_over_time)), part_count_over_time)
ax = plt.subplot(413)
plt.title("Losses")
plt.plot(np.arange(len(losses_history)), losses_history)
avg = np.mean(losses_history)
plt.text(0.95, 0.95, 'Mean Losses = ' + str(avg),
    horizontalalignment='right',
    verticalalignment='top',
    transform=ax.transAxes)
std = pstdev(losses_history)
plt.text(0.95, 0.85, 'Stdev Losses = ' + str(std),
    horizontalalignment='right',
    verticalalignment='top',
    transform=ax.transAxes)
chi_r = chi.fit(losses_history)
avg_r = chi.mean(chi_r[0], loc=chi_r[1], scale=chi_r[2])
plt.text(0.95, 0.75, 'Overall Mean Losses (Chi-square) = ' + str(avg_r),
    horizontalalignment='right',
    verticalalignment='top',
    transform=ax.transAxes)
std_r = chi.std(chi_r[0], loc=chi_r[1], scale=chi_r[2])
plt.text(0.95, 0.65, 'Overall STDEV Losses (Chi-square) = ' + str(std_r),
    horizontalalignment='right',
    verticalalignment='top',
    transform=ax.transAxes)
int_r = chi.interval(0.95, chi_r[0], loc=chi_r[1], scale=chi_r[2])
plt.text(0.95, 0.55, '95% Confidence Interval (Chi-square) = ' + str(int_r),
    horizontalalignment='right',
    verticalalignment='top',
    transform=ax.transAxes)
int_r = chi.interval(0.75, chi_r[0], loc=chi_r[1], scale=chi_r[2])
plt.text(0.95, 0.45, '75% Confidence Interval (Chi-square) = ' + str(int_r),
    horizontalalignment='right',
    verticalalignment='top',
    transform=ax.transAxes)
ax = plt.subplot(414)
plt.title("Losses Distribution")
l_map = {}
for i in losses_history:
    try:
        l_map[i] += 1
    except:
        l_map[i] = 1
plt.plot(sorted(l_map.keys()), [l_map[k] for k in sorted(l_map.keys())])
plt.show()
