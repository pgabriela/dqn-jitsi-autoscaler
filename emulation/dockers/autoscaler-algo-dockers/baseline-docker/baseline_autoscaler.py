import time
import calendar
import os
import requests
import numpy as np
import pandas as pd
from kubernetes import client, config
from influxdb import InfluxDBClient

# Configs can be set in Configuration class directly or using helper utility
config.load_kube_config()

v1 = client.CoreV1Api()
api_instance = client.AppsV1Api()
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
    min_idx = MIN_JVB_COUNT - 1
    max_idx = MAX_JVB_COUNT - 1
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


###############
# Environment #
###############
class Environment:
    def __init__(self, min_jvb_count, max_jvb_count, lookback):
        # Performance metric
        self.loss = 0.0
        # List of JVB IPs that are in graceful shutdown mode
        self.is_shutting_down = []
        # Sorted list of pods stored as [JVB Pod Idx, JVB Pod IP] that are currently running
        self.curr_jvbs = []
        # A [JVB Pod Idx, JVB Pod IP] pair that tells which JVB Pod will be removed should the action to be applied is the 'Remove' class
        self.next_removed = None
        # Min & Max number of JVBs allowed to run
        self.min_jvb_count = min_jvb_count
        self.max_jvb_count = max_jvb_count
        # Number of Losses to be averaged
        self.lookback = lookback

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
        if len(self.is_shutting_down) > 0 and len(self.curr_jvbs) > self.min_jvb_count:
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
        for c, p in enumerate(self.curr_jvbs):
            try:
                r = requests.get('http://' + p[1] + ':8080/colibri/stats', timeout=3)
                d = r.json()
                conferences_count = d['conferences']
                if conferences_count == 0:
                    self.next_removed = p
                    break
                elif c == len(self.curr_jvbs) - 1:
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
            end_time = int(curr_time - (self.lookback * 1e9)) # -lookback s
            loss = influxdb.query('select mean(mean) ' + \
                'from (select mean(overall_loss) from jitsi_stats ' + \
                'where time <= ' + str(curr_time) + \
                ' and time > ' + str(end_time) + \
                ' group by host order by time desc limit ' + str(self.lookback) + \
                ') order by time desc;')
        except:
            print("Error querying influxdb features data")
            return None
        loss = list(loss.get_points())
        if len(loss) > 0:
            loss = loss[0]['mean']
        else:
            print("Empty query result loss")
            return None

        self.loss = loss
        state = np.array([self.loss])

        return state

    def step(self, state, action):
        # Perform action
        if action == 0:
            # Add Class
            if len(self.curr_jvbs) >= self.max_jvb_count:
                print("Maximum number of JVBs limit has been reached")
                print("Maintain")
            else:
                print("Add JVB Pod")
                # Add JVB
                new_idx = -1
                curr_indexes = [x[0] for x in self.curr_jvbs]
                for i in range(self.max_jvb_count - self.min_jvb_count + 1):
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
        elif action == 1:
            # Remove Class
            if len(self.curr_jvbs) - len(self.is_shutting_down) <= self.min_jvb_count:
                print("Minimum number of JVBs limit has been reached")
                print("Maintain")
            elif not self.next_removed:
                print("Unexpected case")
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

        # Calculate reward
        reward = 0

        return reward


###############################
# Hyperparameters & Utilities #
###############################
ACTION_COOLDOWN = 15
EXTRA_COOLDOWN = 5

# Environment parameters
W1 = 3000
MIN_JVB_COUNT = 1
MAX_JVB_COUNT = 6
LOOKBACK = 5

# Preset Thresholds
THRESHOLDS = [0, 0] # UPDATE THIS

def select_action(state):
    curr_action = 2
    if state[0] >= THRESHOLDS[1]:
        curr_action = 0
    elif state[0] <= THRESHOLDS[0]:
        curr_action = 1
    return curr_action


#############
# Main Loop #
#############
env = Environment(MIN_JVB_COUNT, MAX_JVB_COUNT, LOOKBACK)
counter = 0
flag = True
print("Starting the first JVB before going into the main loop")
while flag:
    try:
        api_response = api_instance.create_namespaced_stateful_set(
            namespace='jitsi',
            body=get_jvb_sts_body(0))
        print("Deployment updated. status='%s'" % str(api_response.status))
        time.sleep(EXTRA_COOLDOWN)
        break
    except:
        print("Failed creating a JVB pod")
        try:
            ret = v1.list_namespaced_pod(namespace='jitsi', label_selector='app=jvb', watch=False)
        except:
            raise Exception("Error checking JVB Pods")
        curr_jvbs = []
        for i in ret.items:
            curr_jvbs.append(i.metadata.name)
        if len(curr_jvbs) > 0:
            print("JVB(s) already running")
            flag = False
while True:
    # Get curr state
    curr_state = env.get_state()
    if curr_state is None:
        print()
        time.sleep(EXTRA_COOLDOWN)
        continue

    # Select and perform an action
    curr_action = select_action(curr_state)
    curr_reward = env.step(curr_state, curr_action)

    counter += 1

    if counter % 100 == 0:
        # Might (might not) help with the Azure kube api-server being unreachable
        config.load_kube_config()
        v1 = None
        api_instance = None
        metrics_api = None
        v1 = client.CoreV1Api()
        api_instance = client.AppsV1Api()
        metrics_api = client.CustomObjectsApi()
        # reset
        counter = 0

    # Cooldown
    print()
    time.sleep(ACTION_COOLDOWN)
