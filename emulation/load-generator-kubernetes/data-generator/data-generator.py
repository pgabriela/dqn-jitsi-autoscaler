import time
import os
import requests
import json
import matplotlib.pyplot as plt
import numpy as np
import dateutil.parser
import matplotlib.dates as mdates
from datetime import timezone, timedelta, datetime
from scipy.stats import norm
from influxdb import InfluxDBClient
import statistics
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler


influxdb = InfluxDBClient(
    host="127.0.0.1",
    port=8086,
    database="telegraf")

curr_time = time.time()
# Query InfluxDB
day_idx = 22
month_idx = 10
weekends = [[23, 10], [24, 10], [30, 10], [31, 10], [6, 11], [7, 11], [13, 11], [14, 11], [20, 11], [21, 11]]
last_day = [24, 11]
data_per_day_no_weekend = []
while(True):
    if day_idx > last_day[0] and month_idx >= last_day[1]:
        break
    start_month = month_idx
    start_day = day_idx
    end_month = start_month
    end_day = start_day + 1
    if end_month == 10 and end_day > 31 or end_month == 11 and end_day > 30:
        end_day = 1
        end_month += 1
    if [start_day, start_month] in weekends:
        day_idx += 1
        if day_idx > 31:
            day_idx = 1
            month_idx += 1
        continue
    str_start_day = str(start_day)
    if start_day < 10:
        str_start_day = "0" + str(start_day)
    str_end_day = str(end_day)
    if end_day < 10:
        str_end_day = "0" + str(end_day)
    monitoring_data = influxdb.query("select conferences, participants, host from jitsi_stats where time >= '2020-" + \
            str(start_month) + "-" + \
            str_start_day + "T15:00:00Z' AND time < '2020-" + \
            str(end_month) + "-" + \
            str_end_day + "T15:00:00Z';")
    data_per_day_no_weekend.append(list(monitoring_data.get_points()))
    day_idx += 1
    if day_idx > 31:
        day_idx = 1
        month_idx += 1
print("Query time: %d seconds" % (time.time() - curr_time))

all_conferences = [[], []]
all_participants = [[], []]
conferences = [[], []]
participants = [[], []]
avg_part_per_conf = [[], []]
conf_norm_dist_funcs = []
part_norm_dist_funcs = []

curr_time = time.time()
for i, data in enumerate(data_per_day_no_weekend):
    j = 0
    for row in data:
        if row['host']:
            continue
        if i == 0:
            all_conferences[1].append([])
            all_participants[1].append([])

            dt = dateutil.parser.parse(row['time']).astimezone(timezone(timedelta(hours=9)))
            all_conferences[0].append(dt)
            conferences[0].append(dt)
            all_participants[0].append(dt)
            participants[0].append(dt)

        all_conferences[1][j].append(int(row['conferences']))
        all_participants[1][j].append(int(row['participants']))
        j += 1
print("Cleaning time: %d seconds" % (time.time() - curr_time))

curr_time = time.time()
for d in all_conferences[1]:
    conf_norm_dist_funcs.append(norm.fit(d))
print("First Normal Distribution Functions Fitting time: %d seconds" % (time.time() - curr_time))

curr_time = time.time()
for d in all_participants[1]:
    part_norm_dist_funcs.append(norm.fit(d))
print("Second Normal Distribution Functions Fitting time: %d seconds" % (time.time() - curr_time))

curr_time = time.time()
for loc, scale in conf_norm_dist_funcs:
    conferences[1].append(norm.rvs(loc=loc, scale=scale, random_state=8192))
conferences[1] = np.clip(conferences[1], 0, None)
conferences[1] = savgol_filter(conferences[1], 91, 1)
conferences[1] = np.around(conferences[1])
print("First Data Generation time: %d seconds" % (time.time() - curr_time))

curr_time = time.time()
for loc, scale in part_norm_dist_funcs:
    participants[1].append(norm.rvs(loc=loc, scale=scale, random_state=8192))
participants[1] = np.clip(participants[1], 0, None)
participants[1] = savgol_filter(participants[1], 61, 1)
participants[1] = np.around(participants[1])
print("Second Data Generation time: %d seconds" % (time.time() - curr_time))

conferences2 = [[], []]
participants2 = [[], []]

scaler1 = MinMaxScaler((0, 4))
conferences2[1] = [[x] for x in conferences[1]]
conferences2[1] = scaler1.fit_transform(conferences2[1])
conferences2[1] = [round(x[0]) for x in conferences2[1]]
scaler2 = MinMaxScaler((0, 18))
participants2[1] = [[x] for x in participants[1]]
participants2[1] = scaler2.fit_transform(participants2[1])
participants2[1] = [round(x[0]) for x in participants2[1]]

xformatter = mdates.DateFormatter('%H:%M', tz=timezone(timedelta(hours=9)))

part_per_conf = []
for i in range(len(conferences[1])):
    if int(participants[1][i]) != 0:
        part_per_conf.append(conferences[1][i]/participants[1][i])

avg = statistics.mean(part_per_conf)
std = statistics.pstdev(part_per_conf)

last_conf_id = 0
stagnant_conf_num = 0
conferences_time = {}
active_conferences = []
for i, n in enumerate(conferences2[1]):
    if len(active_conferences) < n:
        stagnant_conf_num = 0
        curr_total_participants = 0
        for ac in active_conferences:
            for p in ac['participants']:
                curr_total_participants += p['num']
        expected_total_participants = participants2[1][i]
        approx_part_per_conf = (expected_total_participants - curr_total_participants) / (n - len(active_conferences))
        flag = False
        if approx_part_per_conf < 1:
            flag = True
        for m in range(int(n - len(active_conferences))):
            if flag:
                approx_part_per_conf = np.clip(norm.rvs(loc=avg, scale=std, random_state=8192), 2, None)
            active_conferences.append({
                'id': 'conf'+str(last_conf_id),
                'participants': [{
                    'start': i,
                    'num': int(round(approx_part_per_conf)),
                }],
            })
            conferences_time['conf'+str(last_conf_id)] = {
                'start': i,
                'duration': 0,
                'participants': [{
                    'start': i,
                    'num': int(round(approx_part_per_conf)),
                }],
            }
            last_conf_id += 1
    elif len(active_conferences) > n:
        stagnant_conf_num = 0
        for m in range(int(len(active_conferences) - n)):
            active_conferences.pop(0)
    else:
        if stagnant_conf_num > 2: # recheck every 10 seconds for more dynamic num of participants
            curr_total_participants = 0
            for ac in active_conferences:
                for p in ac['participants']:
                    curr_total_participants += p['num']
            expected_total_participants = participants2[1][i]
            approx_part_per_conf = round(np.clip((expected_total_participants - curr_total_participants) / len(active_conferences), 0, None))
            for m in range(len(active_conferences)):
                if approx_part_per_conf > 0:
                    active_conferences[m]['participants'].append({
                        'start': i,
                        'num': int(approx_part_per_conf),
                    })
                    conferences_time[active_conferences[m]['id']]['participants'].append({
                        'start': i,
                        'num': int(approx_part_per_conf),
                    })
            stagnant_conf_num = 0
        stagnant_conf_num += 1
    for c in active_conferences:
        conferences_time[c['id']]['duration'] += 1

# write to conferences_list.json file
f = open('conferences_list.json', 'w')
json.dump(conferences_time, f)
print("Saved the conferences list as JSON file in conferences_list.json")

reconstructed = []
part_reconstructed = []
for i in range(17280):
    curr_total_conferences = 0
    curr_total_participants = 0
    for c in conferences_time:
        if conferences_time[c]['start'] <= i and conferences_time[c]['start'] + conferences_time[c]['duration'] > i:
            curr_total_conferences += 1
            for p in conferences_time[c]['participants']:
                if p['start'] <= i:
                    curr_total_participants += p['num']
    reconstructed.append(curr_total_conferences)
    part_reconstructed.append(curr_total_participants)
avg_part_per_conf_recon = []
for i in range(len(reconstructed)):
    if reconstructed[i] == 0:
        continue
    avg = part_reconstructed[i] / reconstructed[i]
    avg_part_per_conf_recon.append(avg)

plt.subplot(221)
plt.plot(np.arange(len(conferences[0])), conferences[1])
plt.title("Generated Total Conferences")
plt.grid(True)
plt.subplot(222)
plt.plot(np.arange(0, len(participants[0])), participants[1])
plt.title("Generated Total Participants")
plt.grid(True)
plt.subplot(223)
plt.plot(np.arange(0, len(reconstructed)), reconstructed)
plt.title("Total Conferences for torture-selenium (Scaled Down)")
plt.grid(True)
plt.subplot(224)
plt.plot(np.arange(0, len(part_reconstructed)), part_reconstructed)
plt.title("Total Participants for torture-selenium (Scaled Down)")
plt.grid(True)
plt.show()
