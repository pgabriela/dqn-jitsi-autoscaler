import time
import os
import requests
import json
from kubernetes import client, config

# Configs can be set in Configuration class directly or using helper utility
config.load_kube_config()

v1 = client.CoreV1Api()
api_instance = client.BatchV1Api()
iterations_num = 17280 # from the original vmeeting data, 24h = 86,400s / 5s = 17,280 data points
seconds_per_iter = 0.3 # compressed 16.67x, total ~ 1h27m (originally 5s in the vmeeting data, total = 24h)

conf_list_f = open('conferences_list.json', 'r') # Choose the conferences_list json file
conferences_list = json.load(conf_list_f)

idx = -1
image_path = "421379781443.dkr.ecr.us-east-1.amazonaws.com/torture-selenium:1.0.0" # UPDATE THIS
jitsi_url = "https://jitsicluster.eastus.cloudapp.azure.com" # CHANGE TO THE JITSI HOST

for i in range(iterations_num):
    for c_name, c in conferences_list.items():
        if i >= c['start'] and i < c['start'] + c['duration']:
            for p in c['participants']:
                if i == p['start']:
                    idx += 1
                    job_name = "torture-selenium-" + str(idx)
                    participants_num = p['num']
                    duration = int(round((c['start'] + c['duration'] - p['start']) * seconds_per_iter)) # in seconds
                    req_cpu = "1.2"
                    req_mem = "2000Mi"
                    if participants_num > 1:
                        req_cpu = "3"
                        req_mem = str(2000 * participants_num) + "Mi"
                    api_response = api_instance.create_namespaced_job(
                        namespace='jitsi',
                        body={
                            "metadata": {
                                "labels": {
                                    "app": job_name,
                                    "name": job_name,
                                },
                                "name": job_name,
                                "namespace": "jitsi",
                            },
                            "spec": {
                                "template": {
                                    "metadata": {
                                        "labels": {
                                            "app": job_name,
                                            "name": job_name,
                                        },
                                    },
                                    "spec": {
                                        "containers": [{
                                            "name": "torture-selenium",
                                            "image": image_path,
                                            "imagePullPolicy": "Always", # Update
                                            "resources": {
                                                "requests": {
                                                    "cpu": req_cpu,
                                                    "memory": req_mem
                                                },
                                            },
                                            "args": [
                                                "--instance-url=" + jitsi_url,
                                                "--conferences=1",
                                                "--room-name-prefix=" + c_name + "-loadtest",
                                                "--audio-senders=" + str(participants_num),
                                                "--senders=" + str(participants_num),
                                                "--participants=" + str(participants_num),
                                                "--duration=" + str(duration),
                                            ],
                                        }],
                                        "restartPolicy": "Never",
                                    }
                                },
                            },
                        }
                    )
                    print("Executed " + job_name + " with room name prefix " + c_name + "-loadtest and " + str(participants_num) + " participants for " + str(duration) + " seconds")

    time.sleep(seconds_per_iter)
print("All torture-selenium have been executed")
