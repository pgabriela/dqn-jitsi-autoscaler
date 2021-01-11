import math
import random
import time
import calendar
import os
import pickle
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



#############
# Q-Network #
#############
class DQN(nn.Module):
    def __init__(self, num_in_features, num_out_features):
        super(DQN, self).__init__()
        self.linear1 = nn.Linear(num_in_features, 32)
        self.ln1 = nn.LayerNorm(32)
        self.linear2 = nn.Linear(32, 64)
        self.ln2 = nn.LayerNorm(64)
        self.linear3 = nn.Linear(64, 64)
        self.ln3 = nn.LayerNorm(64)
        self.linear4 = nn.Linear(64, 32)
        self.ln4 = nn.LayerNorm(32)
        self.out_layer = nn.Linear(32, num_out_features)


    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[add_class,remove_class, maintain_class]]).
    def forward(self, x):
        x = F.leaky_relu(self.ln1(self.linear1(x)))
        x = F.leaky_relu(self.ln2(self.linear2(x)))
        x = F.leaky_relu(self.ln3(self.linear3(x)))
        x = F.leaky_relu(self.ln4(self.linear4(x)))
        return self.out_layer(x)



###############################
# Hyperparameters & Utilities #
###############################
# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv('../dbV3.csv')
timeseries = pd.read_csv('../timeseries.csv')
MIN_JVB_NUM = 1
MAX_JVB_NUM = 50
W1 = 3000
ACTION_COOLDOWN = 15
LOOKBACK = 5

# Q-Network paramteres
N_FEATURES = 5
N_ACTIONS = 3

# Initialize
curr_time = time.time()
print("Loading Pre-trained model ...")
policy_net = DQN(N_FEATURES, N_ACTIONS).to(device)
policy_net.load_state_dict(torch.load("parameters"))
print(f"Pre-trained model loaded in {time.time() - curr_time} seconds")

# define reward function
def calc_reward(state, action):
    curr_loss = state[0][4].item()
    if action == 0:
        jvb_num_delta = 1
    elif action == 1:
        jvb_num_delta = -1
    else:
        jvb_num_delta = 0
    reward = math.exp(-W1 * curr_loss) - 1e-2 * jvb_num_delta
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
latest_losses = []
jvb_num_history = []
idle_jvb_history = []
rewards_history = []
losses_history = []
losses_dict = pickle.load(open('../losses_dict.pkl', 'rb'))

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
    idle_jvb_history.append(z1)
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

    assert j1 <= MAX_JVB_NUM and j1 >= MIN_JVB_NUM
    assert z1 <= MAX_JVB_NUM and j1 >= 0
    assert z1 <= j1

    if (i+1) % ACTION_COOLDOWN == 0:
        # Cooldown finished, Agent act
        l1 = np.mean(latest_losses[-LOOKBACK:])
        latest_losses = []
        curr_state = np.array([p1, j1, z1, l1])

        state = curr_state - prev_state
        
        p_delta = state[0]
        j_delta = state[1]
        z_delta = state[2]
        l_delta = state[3]
        state = [[p_delta, j_delta, z_delta, l_delta, l1]]
        state = torch.tensor(state, dtype=torch.float)

        # select action
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            policy_net.eval()
            curr_action = policy_net(state).max(1)[1].view(1, 1).item()

        # apply action
        if curr_action == 0:
            # 'Add' class
            if len(curr_jvbs) < MAX_JVB_NUM:
                curr_jvbs.append([i+1, 0])
        elif curr_action == 1:
            # 'Remove' class
            if len(curr_jvbs) - len(is_shutting_down) > MIN_JVB_NUM:
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
        
        # calculate reward
        reward = calc_reward(state, curr_action)
        rewards_history.append(reward)

        # Save current state & action
        prev_state = curr_state

    if (i+1) % 500 == 0:
        print(f"Timesteps passed: {i+1}", end="\r")
print(f"\nSimulation finished in {time.time() - curr_time} seconds")


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
plt.plot(np.arange(len(jvb_num_history)), jvb_num_history)
plt.text(0.95, 0.95, 'Total JVB Count = ' + str(sum(jvb_num_history)),
    horizontalalignment='right',
    verticalalignment='top',
    transform=ax.transAxes)
plt.text(0.95, 0.85, 'Average JVB Count = ' + str(np.mean(jvb_num_history)),
    horizontalalignment='right',
    verticalalignment='top',
    transform=ax.transAxes)
pickle.dump(jvb_num_history, open('jvb_num_history.pkl', 'wb'))
pickle.dump(idle_jvb_history, open('idle_jvb_history.pkl', 'wb'))
ax = plt.subplot(514)
plt.title("Rewards")
plt.plot(np.arange(len(rewards_history)), rewards_history)
plt.text(0.95, 0.95, 'Total Reward = ' + str(sum(rewards_history)),
    horizontalalignment='right',
    verticalalignment='top',
    transform=ax.transAxes)
plt.text(0.95, 0.85, 'Average Reward = ' + str(np.mean(rewards_history)),
    horizontalalignment='right',
    verticalalignment='top',
    transform=ax.transAxes)
ax = plt.subplot(515)
plt.title("Losses")
plt.plot(np.arange(len(losses_history)), losses_history)
plt.text(0.95, 0.95, 'Total Loss = ' + str(sum(losses_history)),
    horizontalalignment='right',
    verticalalignment='top',
    transform=ax.transAxes)
plt.text(0.95, 0.85, 'Average Loss = ' + str(np.mean(losses_history)),
    horizontalalignment='right',
    verticalalignment='top',
    transform=ax.transAxes)
pickle.dump(losses_history, open('losses_history.pkl', 'wb'))
plt.show()
