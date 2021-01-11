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
from collections import namedtuple

###############################
# Hyperparameters & Utilities #
###############################
df = pd.read_csv('dbV3.csv')
timeseries = pd.read_csv('timeseries-V2.csv')
min_jvb_num = 1
max_jvb_num = 50
W1 = 3000
action_cooldown = 15
lookback = 5
loss_thresholds = [0.000235, 0.00305]

# define reward function
def calc_reward(state, action):
    curr_jvb = state[2]
    curr_loss = state[4]
    reward = -W1 * curr_loss - math.log(curr_jvb)
    return reward

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
curr_jvbs = [[0, 0], ]
is_shutting_down = []
prev_state = np.array([0, 1, 1, 0])
prev_action = -1
latest_losses = []
jvb_num_history = []
rewards_history = []
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
            jvb_conferences = [x[1] if x[0] not in is_shutting_down else 1e10 for x in curr_jvbs]
            least_loaded_idx = np.argmin(jvb_conferences)
            curr_jvbs[least_loaded_idx][1] += 1
    elif new_c < 0:
        # remove conferences
        for c in range(abs(new_c)):
            for j in curr_jvbs:
                if j[1] > 0:
                    j[1] -= 1
                    break

    # update jvbs (check shutting down jvbs)
    for idx in range(len(is_shutting_down) - 1, -1, -1):
        for j in curr_jvbs:
            if j[0] == is_shutting_down[idx] and j[1] == 0:
                curr_jvbs.remove(j)
                is_shutting_down.pop(idx)
                break

    j1 = len(curr_jvbs)
    jvb_num_history.append(j1)
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
    latest_losses.append(avg_loss)
    losses_history.append(avg_loss)

    assert j1 <= max_jvb_num and j1 >= min_jvb_num
    assert z1 <= max_jvb_num and j1 >= 0
    assert z1 <= j1

    if (i+1) % action_cooldown == 0:
        # Cooldown finished, Agent act
        l1 = np.mean(latest_losses[-lookback:])
        latest_losses = []
        state = np.array([c1, p1, j1, z1, l1])

        # select action
        if l1 > loss_thresholds[1]:
            curr_action = 0
        elif l1 <= loss_thresholds[0]:
            curr_action = 1
        else:
            curr_action = 2

        # apply action
        if curr_action == 0:
            # 'Add' class
            if len(curr_jvbs) < max_jvb_num:
                curr_jvbs.append([i+1, 0])
        elif curr_action == 1:
            # 'Remove' class
            if len(curr_jvbs) - len(is_shutting_down) > min_jvb_num:
                jvb_pair = None
                for j in curr_jvbs:
                    if j[1] == 0:
                        jvb_pair = j
                        break
                if jvb_pair:
                    curr_jvbs.remove(jvb_pair)
                else:
                    if curr_jvbs[0][0] not in is_shutting_down:
                        is_shutting_down.append(curr_jvbs[0][0])
        else:
            # 'Maintain' class
            pass

        if prev_action != -1:
            # calculate reward
            reward = calc_reward(state, prev_action)
            rewards_history.append(reward)

        # Save current state & action
        prev_action = curr_action

    if (i+1) % 500 == 0:
        print(f"Timesteps passed: {i+1}", end="\r")
print(f"\nSimulation finished in {time.time() - curr_time} seconds")

jvb_num_history2 = pickle.load(open('final-diff-test-train-data/jvb_num_history.pkl', 'rb'))
rewards_history2 = pickle.load(open('final-diff-test-train-data/rewards_history.pkl', 'rb'))
losses_history2 = pickle.load(open('final-diff-test-train-data/losses_history.pkl', 'rb'))

#################
# Visualization #
#################
plt.figure(figsize=(16, 9))
plt.subplot(511)
plt.title("Conferences")
plt.plot(np.arange(len(conf_count_over_time)), conf_count_over_time)
plt.subplot(512)
plt.title("Participants")
plt.plot(np.arange(len(part_count_over_time)), part_count_over_time)
ax = plt.subplot(513)
plt.title("JVB Count")
plt.plot(np.arange(len(jvb_num_history)), jvb_num_history, label='Threshold-based')
plt.plot(np.arange(len(jvb_num_history2)), jvb_num_history2, label='DQN-based')
jvb_improvement = abs(np.mean(jvb_num_history) - np.mean(jvb_num_history2)) * 100 / np.mean(jvb_num_history)
t = plt.text(0.99, 0.9, f'Total JVB Count Baseline = {sum(jvb_num_history):.2f}'
        f'\nAverage JVB Count Baseline = {np.mean(jvb_num_history):.2f}'
        f'\nTotal JVB Count DQN = {sum(jvb_num_history2):.2f}'
        f'\nAverage JVB Count DQN = {np.mean(jvb_num_history2):.2f}'
        f'\n% Improvement = {jvb_improvement:.2f}%',
    horizontalalignment='right',
    verticalalignment='top',
    transform=ax.transAxes)
t.set_bbox(dict(facecolor='white', boxstyle='round', edgecolor='#BBBBBB', alpha=0.8))
plt.legend(loc='upper left')
ax = plt.subplot(514)
plt.title("Rewards")
plt.plot(np.arange(len(rewards_history)), rewards_history, label='Threshold-based')
plt.plot(np.arange(len(rewards_history2)), rewards_history2, label='DQN-based')
rewards_improvement = abs((np.mean(rewards_history) - np.mean(rewards_history2)) * 100 / np.mean(rewards_history))
t = plt.text(0.99, 0.9, f'Total Reward Baseline = {sum(rewards_history):.2f}'
        f'\nAverage Reward Baseline = {np.mean(rewards_history):.2f}'
        f'\nTotal Reward DQN = {sum(rewards_history2):.2f}'
        f'\nAverage Reward DQN = {np.mean(rewards_history2):.2f}'
        f'\n% Improvement = {rewards_improvement:.2f}%',
    horizontalalignment='right',
    verticalalignment='top',
    transform=ax.transAxes)
t.set_bbox(dict(facecolor='white', boxstyle='round', edgecolor='#BBBBBB', alpha=0.8))
plt.legend(loc='upper left')
ax = plt.subplot(515)
plt.title("Losses")
plt.plot(np.arange(len(losses_history)), losses_history, label='Threshold-based')
plt.plot(np.arange(len(losses_history2)), losses_history2, label='DQN-based')
loss_improvement = abs(np.mean(losses_history) - np.mean(losses_history2)) * 100 / np.mean(losses_history)
t = plt.text(0.99, 0.9, f'Total Loss Baseline = {sum(losses_history):.4f}'
        f'\nAverage Loss Baseline = {np.mean(losses_history):.6f}'
        f'\nTotal Loss DQN = {sum(losses_history2):.4f}'
        f'\nAverage Loss DQN = {np.mean(losses_history2):.6f}'
        f'\n% Improvement = {loss_improvement:.2f}%',
    horizontalalignment='right',
    verticalalignment='top',
    transform=ax.transAxes)
t.set_bbox(dict(facecolor='white', boxstyle='round', edgecolor='#BBBBBB', alpha=0.8))
plt.legend(loc='upper left')
plt.show()
