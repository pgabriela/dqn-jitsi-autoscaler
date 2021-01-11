import math
import random
import time
import calendar
import os
import requests
import numpy as np
import pandas as pd
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from kubernetes import client, config
from influxdb import InfluxDBClient


# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config.load_kube_config()

v1 = client.CoreV1Api()
api_instance = client.AppsV1Api()
influxdb_jcf = InfluxDBClient(
    host=os.environ.get("INFLUXDB_HOST"),
    port=8086,
    database="jicofo_stats")
influxdb = InfluxDBClient(
    host=os.environ.get("INFLUXDB_HOST"),
    port=8086,
    database="jitsi_stats")

XMPP_SERVER = os.environ.get("XMPP_SERVER")
DOCKER_HOST_ADDR0 = os.environ.get("DOCKER_HOST_ADDR0")
DOCKER_HOST_ADDR1 = os.environ.get("DOCKER_HOST_ADDR1")
DOCKER_HOST_ADDR2 = os.environ.get("DOCKER_HOST_ADDR2")
DOCKER_HOST_ADDR3 = os.environ.get("DOCKER_HOST_ADDR3")
DOCKER_HOST_ADDR4 = os.environ.get("DOCKER_HOST_ADDR4")
DOCKER_HOST_ADDR5 = os.environ.get("DOCKER_HOST_ADDR5")
AUTOSCALER_IP = os.environ.get("AUTOSCALER_IP")

LOAD_MODEL = False
MODEL_STATE_PATH = os.environ.get("MODEL_STATE_PATH")
if MODEL_STATE_PATH:
    LOAD_MODEL = True
    print("Load Model State Mode On")
    print("Model State Path: " + MODEL_STATE_PATH)


####################
# JVB STS Template #
####################
# MAKE SURE TO UPDATE THE JVB STS TEMPLATE HERE
def get_jvb_sts_body(idx):
    i = -1
    try:
        i = int(idx)
    except:
        raise Exception("JVB Index should be an integer")
    min_idx = MIN_JVB_NUM - 1
    max_idx = MAX_JVB_NUM - 1
    if i < min_idx or i > max_idx:
        raise Exception("JVB Index should be between " + str(min_idx) + " and " + str(max_idx) + " (both inclusive)")

    return {
        "metadata": {
            "name": "jvb-" + str(i),
        },
        "spec": {
            "serviceName": "jvb-" + str(i),
            "replicas": 1,
            "selector": {
              "matchLabels": {
                "app": "jvb"
              }
            },
            "template": {
                "metadata": {
                    "labels": {
                        "app": "jvb"
                    }
                },
                "spec": {
                    "volumes": [
                        {
                            "name": "telegraf-config-volume",
                            "configMap": {
                                "name": "telegraf-config"
                            }
                        }
                    ],
                    "containers": [
                        {
                            "name": "telegraf",
                            "image": "telegraf:1.10.0",
                            "envFrom": [
                                {
                                    "secretRef": {
                                        "name": "telegraf-secrets"
                                    }
                                }
                            ],
                            "env": [
                                {
                                    "name": "JVB_ADDR",
                                    "valueFrom": {
                                        "fieldRef": {
                                            "fieldPath": "status.podIP"
                                        }
                                    }
                                },
                                {
                                    "name": "INPUT_ADDR",
                                    "value": "http://$(JVB_ADDR):8080/colibri/stats"
                                }
                            ],
                            "volumeMounts": [
                                {
                                    "name": "telegraf-config-volume",
                                    "mountPath": "/etc/telegraf/telegraf.conf",
                                    "subPath": "telegraf.conf",
                                    "readOnly": True
                                }
                            ]
                        },
                        {
                            "name": "jvb",
                            "image": "jitsiacr.azurecr.io/jvb:1.3.0",
                            "imagePullPolicy": "Always",
                            "resources": {
                                "requests": {
                                    "cpu": "0.2",
                                    "memory": "500Mi",
                                },
                                "limits": {
                                    "cpu": "0.3",
                                    "memory": "600Mi",
                                },
                            },
                            "env": [
                                {
                                    "name": "XMPP_SERVER",
                                    "value": XMPP_SERVER
                                },
                                {
                                    "name": "DOCKER_HOST_ADDR0",
                                    "value": DOCKER_HOST_ADDR0
                                },
                                {
                                    "name": "DOCKER_HOST_ADDR1",
                                    "value": DOCKER_HOST_ADDR1
                                },
                                {
                                    "name": "DOCKER_HOST_ADDR2",
                                    "value": DOCKER_HOST_ADDR2
                                },
                                {
                                    "name": "DOCKER_HOST_ADDR3",
                                    "value": DOCKER_HOST_ADDR3
                                },
                                {
                                    "name": "DOCKER_HOST_ADDR4",
                                    "value": DOCKER_HOST_ADDR4
                                },
                                {
                                    "name": "DOCKER_HOST_ADDR5",
                                    "value": DOCKER_HOST_ADDR5
                                },
                                {
                                    "name": "DOCKER_LOCAL_ADDR",
                                    "valueFrom": {
                                        "fieldRef": {
                                            "fieldPath": "status.podIP"
                                        }
                                    }
                                },
                                {
                                    "name": "XMPP_DOMAIN",
                                    "value": "jitsicluster.eastus.cloudapp.azure.com"
                                },
                                {
                                    "name": "XMPP_GUEST_DOMAIN",
                                    "value": "guest.jitsicluster.eastus.cloudapp.azure.com"
                                },
                                {
                                    "name": "XMPP_AUTH_DOMAIN",
                                    "value": "auth.jitsicluster.eastus.cloudapp.azure.com"
                                },
                                {
                                    "name": "XMPP_INTERNAL_MUC_DOMAIN",
                                    "value": "internal-muc.jitsicluster.eastus.cloudapp.azure.com"
                                },
                                {
                                    "name": "JICOFO_AUTH_USER",
                                    "value": "focus"
                                },
                                {
                                    "name": "JVB_TCP_HARVESTER_DISABLED",
                                    "value": "true"
                                },
                                {
                                    "name": "JVB_AUTH_USER",
                                    "value": "jvb"
                                },
                                {
                                    "name": "JVB_ENABLE_APIS",
                                    "value": "rest,colibri"
                                },
                                {
                                    "name": "JVB_AUTH_PASSWORD",
                                    "valueFrom": {
                                        "secretKeyRef": {
                                            "name": "jitsi-config",
                                            "key": "JVB_AUTH_PASSWORD"
                                        }
                                    }
                                },
                                {
                                    "name": "JICOFO_AUTH_PASSWORD",
                                    "valueFrom": {
                                        "secretKeyRef": {
                                            "name": "jitsi-config",
                                            "key": "JICOFO_AUTH_PASSWORD"
                                        }
                                    }
                                },
                                {
                                    "name": "JVB_BREWERY_MUC",
                                    "value": "jvbbrewery"
                                },
                                {
                                    "name": "JVB_KILLER_IP",
                                    "value": AUTOSCALER_IP
                                }
                            ]
                        }
                    ]
                }
            }
        }
    }


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
        self.lstm = nn.LSTM(num_in_features, 32, batch_first=True)
        self.ln1 = nn.LayerNorm(32)
        self.linear = nn.Linear(32, 16)
        self.ln2 = nn.LayerNorm(16)
        self.out_layer = nn.Linear(16, num_out_features)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[add_class,remove_class, maintain_class]]).
    def forward(self, x):
        _, (ht, _) = self.lstm(x)
        x = F.leaky_relu(self.ln1(ht.view(x.size(0), 32)))
        x = F.leaky_relu(self.ln2(self.linear(x)))
        return self.out_layer(x)


###############
# Environment #
###############
class Environment:
    def __init__(self, min_jvb_num, max_jvb_num):
        # List of JVB IPs that are in graceful shutdown mode
        self.is_shutting_down = []
        # Sorted list of pods stored as [JVB Pod Idx, JVB Pod IP] that are currently running
        self.curr_jvbs = []
        # A [JVB Pod Idx, JVB Pod IP] pair that tells which JVB Pod will be removed should the action to be applied is the 'Remove' class
        self.next_removed = None
        # Min & Max number of JVBs allowed to run
        self.min_jvb_num = min_jvb_num
        self.max_jvb_num = max_jvb_num
        # Prev state & action
        self.prev_state = None
        self.prev_action = None

    def get_prev_state(self):
        return self.prev_state

    def get_prev_action(self):
        return self.prev_action

    def set_prev_action(self, action):
        self.prev_action = action

    def get_state(self):
        # Check the JVBs
        print("Listing JVBs with their IPs:")
        try:
            ret = v1.list_namespaced_pod(namespace='jitsi', label_selector='app=jvb', watch=False)
        except:
            print("Error listing JVB Pods")
            return None
        self.curr_jvbs = []
        for i in ret.items:
            print("%s\t%s\t%s" % (i.status.pod_ip, i.metadata.namespace, i.metadata.name))
            curr_jvb_idx = int(i.metadata.name.split('-')[1])
            self.curr_jvbs.append([curr_jvb_idx, i.status.pod_ip])
        self.curr_jvbs.sort(key=lambda x: x[0])
        for p in self.curr_jvbs:
            if p[1] is None:
                print("There is an unready JVB Pod")
                print("Skipping current loop")
                return None

        # Try to remove the pod whose JVB has been shut down gracefully
        if len(self.is_shutting_down) > 0 and len(self.curr_jvbs) > 0:
            print("Checking if the JVB is still running")
            try:
                r = requests.get('http://' + self.is_shutting_down[-1][1] + ':8080/colibri/stats', timeout=3)
            except:
                # JVB is not running anymore
                # Remove Pod
                print("Remove")
                try:
                    api_response = api_instance.delete_namespaced_stateful_set(
                        name='jvb-' + str(self.is_shutting_down[-1][0]),
                        namespace='jitsi',
                    )

                    print("Deployment updated. status='%s'" % str(api_response.status))
                    self.curr_jvbs.remove(self.is_shutting_down[-1])
                    self.is_shutting_down.pop()
                    time.sleep(EXTRA_COOLDOWN)
                except:
                    print("Failed removing the JVB pod")

        # Check if there is at least 1 JVB running
        if len(self.curr_jvbs) == 0:
            print("JVB StatefulSet has not been deployed yet")
            print("Do nothing")
            return None

        # Decide which JVB should be removed next should the action to be applied is the 'Remove' class
        self.next_removed = None
        idle_count = 0
        for c, p in enumerate(self.curr_jvbs):
            try:
                r = requests.get('http://' + p[1] + ':8080/colibri/stats', timeout=3)
                d = r.json()
                conferences_num = d['conferences']
                if conferences_num == 0:
                    if self.next_removed is None:
                        self.next_removed = p
                    idle_count += 1
                elif c == len(self.curr_jvbs) - 1:
                    if self.next_removed is None:
                        self.next_removed = self.curr_jvbs[0]
            except:
                continue

        # Get latest time
        try:
            newest = influxdb.query('select overall_loss from jitsi_stats order by time desc limit 1;')
        except Exception as e:
            print("Error querying influxdb newest")
            print(e)
            return None
        newest = list(newest.get_points())
        if len(newest) > 0:
            newest = round(calendar.timegm(time.strptime(newest[0]['time'], '%Y-%m-%dT%H:%M:%SZ'))) * 1e9
            newest -= 1e9 # -1s
            newest = int(newest)
        else:
            print("Empty query result newest")
            return None

        # Get states
        try:
            curr_time = newest
            st_time = newest - LOOKBACK * 1e9 # -LOOKBACK seconds
            monitoring_data = influxdb_jcf.query('select conferences, participants, ' + \
                'bridge_selector_bridge_count as jvb_count' + \
                'from jicofo_stats ' + \
                'where time >= ' + str(st_time) + \
                ' and time <= ' + str(curr_time) + \
                ' limit ' + str(LOOKBACK) + ';')
            monitoring_data2 = influxdb_jvb.query("select count(conferences) as idle_jvbs from jitsi_stats " + \
                'where time >= ' + str(st_time) + \
                ' and time <= ' + str(curr_time) + \
                " and conferences = 0 group by time(1s) fill(0) limit " + str(LOOKBACK) + ";")
            losses = influxdb.query('select mean(overall_loss) from jitsi_stats ' + \
                'where time >= ' + str(st_time) + \
                ' and time <= ' + str(curr_time) + \
                ' group by time(1s) fill(0) limit ' + str(LOOKBACK) + ';')
        except:
            print("Error querying influxdb features data")
            return None
        d = list(monitoring_data.get_points())
        conferences = []
        participants = []
        jvb_counts = []
        if len(d) >= LOOKBACK:
            df = pd.DataFrame(d)
            conferences = df['conferences']
            participants = df['participants']
            jvb_counts = df['jvb_count']
        else:
            print("Empty query result monitoring data")
            return None
        d = list(monitoring_data2.get_points())
        idle_jvbs = []
        if len(d) >= LOOKBACK:
            df = pd.DataFrame(d)
            idle_jvbs = df['idle_jvbs']
        else:
            print("Empty query result monitoring data 2")
            return None
        losses = list(losses.get_points())
        if len(losses) > LOOKBACK:
            losses = pd.DataFrame(losses)['mean']
        else:
            print("Empty query result losses")
            return None

        state = [[
            conferences[i],
            participants[i],
            jvb_counts[i],
            idle_jvbs[i],
            losses[i]] for i in range(LOOKBACK)]

       self.prev_state = torch.tensor([state], device=device, dtype=torch.float)
       return self.prev_state

    def step(self, action_item):
        # Calculate reward
        reward = calc_reward(self.prev_state)

        # Perform action & save whether the action is successfully applied
        if action_item == 0:
            # Add Class
            if len(self.curr_jvbs) >= self.max_jvb_num:
                print("Maximum number of JVBs limit has been reached")
                print("Maintain")
            else:
                print("Add JVB Pod")
                # Add JVB
                new_idx = -1
                curr_indexes = [x[0] for x in self.curr_jvbs]
                for i in range(MAX_JVB_NUM - MIN_JVB_NUM + 1):
                    if i not in curr_indexes:
                        new_idx = i
                        break
                try:
                    api_response = api_instance.create_namespaced_stateful_set(
                        namespace='jitsi',
                        body=get_jvb_sts_body(new_idx))
                    print("Deployment updated. status='%s'" % str(api_response.status))
                except:
                    print("Failed adding a JVB pod")
                self.prev_action_success = True
        elif action_item  == 1:
            # Remove Class
            if len(self.curr_jvbs) - len(self.is_shutting_down) <= self.min_jvb_num:
                print("Minimum number of JVBs limit has been reached")
                print("Maintain")
            elif not self.next_removed:
                print("Maintain")
            else:
                # Remove JVB
                try:
                    r = requests.get('http://' + self.next_removed[1] + ':8080/colibri/stats', timeout=3)
                    # JVB is still running
                    # Gracefully shutdown
                    print("Gracefully shutting down jvb-%d-0" % self.next_removed[0])
                    r = requests.post('http://' + self.next_removed[1] + ':8080/colibri/shutdown',
                            data='{"graceful-shutdown": "true"}',
                            headers={"Content-Type": "application/json"})
                    if self.next_removed not in self.is_shutting_down:
                        self.is_shutting_down.append(self.next_removed)
                except:
                    # JVB is not running anymore
                    # Remove Pod
                    print("Remove JVB Pod")
                    try:
                        api_response = api_instance.delete_namespaced_stateful_set(
                            name='jvb-' + str(self.next_removed[0]),
                            namespace='jitsi',
                        )

                        # REQUIRED
                        self.is_shutting_down.pop()
                        print("Deployment updated. status='%s'" % str(api_response.status))
                    except:
                        print("Failed removing the JVB pod")
        else:
            # Maintain Class
            print("Maintain")

        return reward


###############################
# Hyperparameters & Utilities #
###############################
MEMORY_CAPACITY = 2000
BATCH_SIZE = 64
GAMMA = 0.5
EPS_START = 1.0
EPS_END = 0.05
if LOAD_MODEL:
    EPS_START = 0.5
    EPS_END = 0.05
EPS_DECAY = 1e-5
EPS_THRESHOLD = EPS_START
TARGET_UPDATE = 200
N_EPISODES = 200

ACTION_COOLDOWN = 15
EXTRA_COOLDOWN = 5
LOOKBACK = 5

# Model saving parameters
MODEL_SAVE_INTERVAL = 60 # every ~30m
MODEL_SAVE_PATH = 'model_state_dict'

# Reward saving parameters
REWARD_SAVE_PATH = 'rewards'

# Q-Network paramteres
N_FEATURES = 5
N_ACTIONS = 3

# State parameters
LIMIT_JVB_CPU = 0.3 * 1e9 # in n
LIMIT_JVB_MEM = 600 * 1e3 # in Ki

# Environment parameters
MIN_JVB_NUM = 1
MAX_JVB_NUM = 6
W1 = 3000

# Initialize
policy_net = DQN(N_FEATURES, N_ACTIONS).to(device)
target_net = DQN(N_FEATURES, N_ACTIONS).to(device)
if LOAD_MODEL:
    policy_net.load_state_dict(torch.load(MODEL_STATE_PATH))
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters())
memory = ReplayMemory(MEMORY_CAPACITY)

# define reward function
def calc_reward(state):
    curr_jvb = state[0][-1][2].item()
    curr_loss = state[0][-1][4].item()
    reward = -W1 * curr_loss - math.log(curr_jvb)
    return reward

def select_action(state):
    global EPS_THRESHOLD
    sample = random.random()

    if sample > EPS_THRESHOLD:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            policy_net.eval()
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(N_ACTIONS)]], device=device, dtype=torch.long)


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


#############
# Main Loop #
#############
batch_rewards = []
env = Environment(MIN_JVB_NUM, MAX_JVB_NUM)
c1 = 0
c2 = 0
print("Starting the first JVB before going into the main loop")
while True:
    try:
        api_response = api_instance.create_namespaced_stateful_set(
            namespace='jitsi',
            body=get_jvb_sts_body(0))
        print("Deployment updated. status='%s'" % str(api_response.status))
        time.sleep(EXTRA_COOLDOWN)
        break
    except:
        print("Failed creating a JVB pod")
while True:
    # Get previous state & action
    prev_state = env.get_prev_state()
    prev_action = env.get_prev_action()

    # Get current state + save current state as prev state
    state = env.get_state()
    if state is None:
        print()
        time.sleep(EXTRA_COOLDOWN)
        continue

    # Select and perform an action
    action = select_action(state)
    env.set_prev_action(action)
    reward = env.step(action.item())
    batch_rewards.append(reward)
    reward = torch.tensor([reward], device=device, dtype=torch.float)

    if prev_action is not None:
        # Store the transition in memory
        memory.push(prev_state, prev_action, state, reward)

    # Perform one step of the optimization (on the target network)
    optimize_model()

    c1 += 1
    c2 += 1

    # Update the target network, copying all weights and biases in DQN
    if c1 % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
        c1 = 0

        # Also save the rewards for visualizing the rewards convergence
        rf = open(REWARD_SAVE_PATH, 'a')
        for r in batch_rewards:
            rf.write(str(r) + '\n')
        rf.close()
        batch_rewards = []

        # Might (might not) help with the Azure kube api-server being unreachable
        config.load_kube_config()
        v1 = None
        api_instance = None
        v1 = client.CoreV1Api()
        api_instance = client.AppsV1Api()

    # Save model learned parameters
    if c2 % MODEL_SAVE_INTERVAL == 0:
        ct = int(time.time())
        save_path = MODEL_SAVE_PATH + "-" + str(ct)
        torch.save(policy_net.state_dict(), save_path)
        print("Latest model parameters has been saved to " + save_path)
        c2 = 0

    # Cooldown
    print()
    time.sleep(ACTION_COOLDOWN)
