import time
import os
import requests
import matplotlib.pyplot as plt
import numpy as np
import dateutil.parser
import matplotlib.dates as mdates
from datetime import timezone, timedelta, datetime
from scipy.stats import gamma
from influxdb import InfluxDBClient
from statistics import pvariance, pstdev, mean


influxdb = InfluxDBClient(
    host="127.0.0.1",
    port=8086,
    database="telegraf")

# Query InfluxDB
monitoring_data_thu = influxdb.query("select conferences, participants, host from jitsi_stats where time >= '2020-11-18T15:00:00Z' AND time < '2020-11-19T15:00:00Z';")
monitoring_data_fri = influxdb.query("select conferences, participants, host from jitsi_stats where time >= '2020-11-19T15:00:00Z' AND time < '2020-11-20T15:00:00Z';")
monitoring_data_sat = influxdb.query("select conferences, participants, host from jitsi_stats where time >= '2020-11-20T15:00:00Z' AND time < '2020-11-21T15:00:00Z';")
monitoring_data_sun = influxdb.query("select conferences, participants, host from jitsi_stats where time >= '2020-11-21T15:00:00Z' AND time < '2020-11-22T15:00:00Z';")
monitoring_data_mon = influxdb.query("select conferences, participants, host from jitsi_stats where time >= '2020-11-22T15:00:00Z' AND time < '2020-11-23T15:00:00Z';")
monitoring_data_tue = influxdb.query("select conferences, participants, host from jitsi_stats where time >= '2020-11-23T15:00:00Z' AND time < '2020-11-24T15:00:00Z';")
monitoring_data_wed = influxdb.query("select conferences, participants, host from jitsi_stats where time >= '2020-11-24T15:00:00Z' AND time < '2020-11-25T15:00:00Z';")

# Print data
week_data = [
    list(monitoring_data_thu.get_points()),
    list(monitoring_data_fri.get_points()),
    list(monitoring_data_sat.get_points()),
    list(monitoring_data_sun.get_points()),
    list(monitoring_data_mon.get_points()),
    list(monitoring_data_tue.get_points()),
    list(monitoring_data_wed.get_points()),
]
#conf_data_per_host = {
#    'vmeeting': [[], []],
#    'worker1': [[], []],
#    'worker2': [[], []],
#    'worker3': [[], []],
#    'worker4': [[], []],
#    'worker5': [[], []],
#}
#part_data_per_host = {
#    'vmeeting': [[], []],
#    'worker1': [[], []],
#    'worker2': [[], []],
#    'worker3': [[], []],
#    'worker4': [[], []],
#    'worker5': [[], []],
#}
conferences = [[], []]
participants = [[], []]
#jvb_util_spread = [[], []]
avg_part_per_conf = [[], []]

day_idx = 0
for data in week_data:
    i = 0
    for row in data:
        if row['host']:
            pass
            #conf_data_per_host[row['host']][0].append(dateutil.parser.parse(row['time']))
            #conf_data_per_host[row['host']][1].append(int(row['conferences']))
            #part_data_per_host[row['host']][0].append(dateutil.parser.parse(row['time']))
            #part_data_per_host[row['host']][1].append(int(row['participants']))
        else:
            if day_idx > 0:
                conferences[1][i] += int(row['conferences'])
                participants[1][i] += int(row['participants'])
            else:
                conferences[0].append(dateutil.parser.parse(row['time']).astimezone(timezone(timedelta(hours=9))))
                conferences[1].append(int(row['conferences']))
                participants[0].append(dateutil.parser.parse(row['time']).astimezone(timezone(timedelta(hours=9))))
                participants[1].append(int(row['participants']))
            i += 1
    day_idx += 1
for x in range(len(conferences[1])):
    conferences[1][x] /= 7
    participants[1][x] /= 7

#for i in range(len(part_data_per_host['vmeeting'][0])):
#    jvb_util_spread[0].append(part_data_per_host['vmeeting'][0][i])
#    jvb_util = []
#    for k in part_data_per_host:
#        jvb_util.append(part_data_per_host[k][1][i])
#    spread = pvariance(jvb_util)
#    jvb_util_spread[1].append(spread)
delta_conferences = []
for i in range(len(conferences[0])):
    try:
        avg = participants[1][i] / conferences[1][i]
        avg_part_per_conf[0].append(participants[0][i])
        avg_part_per_conf[1].append(avg)
    except:
        pass
    if i == len(conferences[0]) - 1:
        break
    delta_conferences.append(conferences[1][i+1]-conferences[1][i])


fit_alpha, fit_loc, fit_beta = gamma.fit(delta_conferences)
#print(fit_alpha, fit_loc, fit_beta)

#colors = {
#    'vmeeting': 'r',
#    'worker2': 'b',
#    'worker3': 'y',
#    'worker4': 'k',
#    'worker5': 'p',
#}

xformatter = mdates.DateFormatter('%H:%M', tz=timezone(timedelta(hours=9)))

fig = plt.figure()

ax = plt.subplot(411)
plt.plot(conferences[0], conferences[1])
plt.title('Average Total Conferences (19/11/2020 - 25/11/2020)')
ax.xaxis.set_major_formatter(xformatter)

ax = plt.subplot(412)
plt.plot(participants[0], participants[1])
plt.title('Average Total Participants (19/11/2020 - 25/11/2020)')
ax.xaxis.set_major_formatter(xformatter)

#plt.subplot(313)
#plt.plot(jvb_util_spread[0], jvb_util_spread[1])
#plt.title('Variance of Participants per JVB')
ax = plt.subplot(413)
plt.plot(avg_part_per_conf[0], avg_part_per_conf[1])
plt.title('Mean Average Participants per Conference (19/11/2020 - 25/11/2020)')
ax.xaxis.set_major_formatter(xformatter)

avg = mean(avg_part_per_conf[1])
plt.text(1.0, 1.0, 'Overall Average Participants per Conference = ' + str(avg),
    horizontalalignment='right',
    verticalalignment='top',
    transform=ax.transAxes)
std = pstdev(avg_part_per_conf[1])
plt.text(1.0, 0.9, 'Overall STDEV Participants per Conference = ' + str(std),
    horizontalalignment='right',
    verticalalignment='top',
    transform=ax.transAxes)

cummulative = [0,]
plt.subplot(414)
d = gamma.rvs(fit_alpha, loc=fit_loc, scale=fit_beta, size=17279)
for x in d:
    cummulative.append(cummulative[-1] + x)
plt.plot(np.arange(0, len(delta_conferences)), delta_conferences)
plt.title('Reconstructed Conference Number per Day')

plt.show()
