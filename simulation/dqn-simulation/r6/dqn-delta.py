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

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


SEED = 2701
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


#################
# Replay Memory #
#################
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



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
W1 = 30

ACTION_COOLDOWN = 30
LOOKBACK = 10
MEMORY_CAPACITY = 2000
BATCH_SIZE = 64
GAMMA = 0.2
TARGET_UPDATE = 200
N_EPISODES = 300
EPS_START = 1.0
EPS_END = 0.05
EXPLORATION_DUR = (80000 / ACTION_COOLDOWN) * (N_EPISODES / 1.5)
EPS_DECAY = (EPS_START - EPS_END) / EXPLORATION_DUR
EPS_THRESHOLD = EPS_START

# Q-Network paramteres
N_FEATURES = 5
N_ACTIONS = 3

# Initialize
policy_net = DQN(N_FEATURES, N_ACTIONS).to(device)
target_net = DQN(N_FEATURES, N_ACTIONS).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(policy_net.parameters())
memory = ReplayMemory(MEMORY_CAPACITY)

# define reward function
def calc_reward(state, action):
    loss_delta = state[0][3].item()
    curr_loss = state[0][4].item()
    if action == 0:
        jvb_num_delta = 1
    elif action == 1:
        jvb_num_delta = -1
    else:
        jvb_num_delta = 0
    reward = loss_delta * jvb_num_delta
    if loss_delta == 0:
        reward = W1 * curr_loss * jvb_num_delta
        if curr_loss == 0:
            reward = -jvb_num_delta
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
                        estimated_loss = (loss + 1e-10) * loss_scale
                        losses.append(estimated_loss)
                        flag = False
                        break

    return np.mean(losses)


#################
# Training Func #
#################
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    policy_net.train()
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    return loss


##############
# Simulation #
##############
print("Starting simulation...")
curr_time = time.time()
cummulative_rewards_history = []
epsilon_history = []
losses_dict = pickle.load(open('losses_dict.pkl', 'rb'))

counter = 0
for i_episode in range(N_EPISODES):
    # list of [jvb id, conference count] pair of currently running JVBs
    # selected with round-robin, removed with graceful shutdown
    curr_jvbs = [[0, 0], ]
    is_shutting_down = []
    prev_state = np.array([0, 1, 1, 0])
    prev_action = -1
    prev_delta_state = None
    latest_losses = []
    jvb_num_history = []
    rewards_history = []
    losses_history = []
    miss_count = 0

    conf_count_over_time = timeseries['conference_count']
    part_count_over_time = timeseries['participant_count']

    with open('../logs/conference_count.txt', 'w') as f:
        pass
    with open('../logs/participant_count.txt', 'w') as f:
        pass
    with open('../logs/jvb_count.txt', 'w') as f:
        pass
    with open('../logs/rewards.txt', 'w') as f:
        pass
    with open('../logs/losses.txt', 'w') as f:
        pass

    episode_start_time = time.time()
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
            miss_count += 1
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

            if prev_action != -1:
                # Push to memory
                memory.push(prev_delta_state, prev_action, state, torch.tensor([rewards_history[-1]], dtype=torch.float))

            # select action
            sample = random.random()
            if sample > EPS_THRESHOLD:
                with torch.no_grad():
                    # t.max(1) will return largest column value of each row.
                    # second column on max result is index of where max element was
                    # found, so we pick action with the larger expected reward.
                    policy_net.eval()
                    curr_action = policy_net(state).max(1)[1].view(1, 1).item()
            else:
                curr_action = random.randrange(N_ACTIONS)
            EPS_THRESHOLD = max(EPS_THRESHOLD - EPS_DECAY, EPS_END)

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
            # Save metrics to log files for live visualization
            with open('../logs/rewards.txt', 'a') as reward_f:
                reward_f.write(f"{rewards_history[-1]}\n")

            # Save current state & action
            prev_state = curr_state
            prev_action = torch.tensor([[curr_action]], dtype=torch.long)
            prev_delta_state = state

            # Train Q-Network
            optimize_model()

            # Update the target network, copying all weights and biases in DQN
            counter += 1
            if counter % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())
                counter = 0

        if (i+1) % 500 == 0:
            print(f"Timesteps passed: {i+1}", end="\r")

        # Save metrics to log files for live visualization
        with open('../logs/conference_count.txt', 'a') as conf_f:
            conf_f.write(f"{conf_count_over_time[i]}\n")
        with open('../logs/participant_count.txt', 'a') as part_f:
            part_f.write(f"{part_count_over_time[i]}\n")
        with open('../logs/jvb_count.txt', 'a') as jvb_f:
            jvb_f.write(f"{jvb_num_history[-1]}\n")
        with open('../logs/losses.txt', 'a') as loss_f:
            loss_f.write(f"{losses_history[-1]}\n")

    cummulative_rewards_history.append(sum(rewards_history))
    epsilon_history.append(EPS_THRESHOLD)
    print(f"Episode {i_episode+1} - Cummulative Reward: {cummulative_rewards_history[-1]} - Epsilon: {EPS_THRESHOLD} - Miss Count: {miss_count} - Duration: {time.time() - episode_start_time} seconds")
print(f"\nSimulation finished in {time.time() - curr_time} seconds")

torch.save(policy_net.state_dict(), "parameters")
print("Latest model parameters has been saved to parameters")

pickle.dump(losses_dict, open('losses_dict.pkl', 'wb'))
print("Losses dictionary has been saved to losses_dict.pkl")
pickle.dump(cummulative_rewards_history, open('cum_rewards_history.pkl', 'wb'))
print("Cummulative rewards history has been saved to cum_rewards_history.pkl")
pickle.dump(epsilon_history, open('epsilon_history.pkl', 'wb'))
print("Epsilon thresholds history has been saved to epsilon_history.pkl")

plt.subplot(211)
plt.title("Cummulative Rewards over Episodes")
plt.plot(np.arange(len(cummulative_rewards_history)) + 1, cummulative_rewards_history)
plt.subplot(212)
plt.title("Epsilon Thresholds over Episodes")
plt.plot(np.arange(len(epsilon_history)) + 1, epsilon_history)
plt.show()
