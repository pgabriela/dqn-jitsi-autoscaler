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
monitoring_data = influxdb.query("select conferences, participants, host from jitsi_stats where time >= '2020-11-20T15:00:00Z' AND time < '2020-11-21T15:00:00Z';")

conferences = [[], []]
participants = [[], []]
avg_part_per_conf = [[], []]

for row in monitoring_data.get_points():
    if row['host']:
        continue
    else:
        conferences[0].append(dateutil.parser.parse(row['time']).astimezone(timezone(timedelta(hours=9))))
        conferences[1].append(int(row['conferences']))
        participants[0].append(dateutil.parser.parse(row['time']).astimezone(timezone(timedelta(hours=9))))
        participants[1].append(int(row['participants']))

for i in range(len(conferences[0])):
    if conferences[1][i] == 0:
        continue
    avg = participants[1][i] / conferences[1][i]
    avg_part_per_conf[0].append(participants[0][i])
    avg_part_per_conf[1].append(avg)

xformatter = mdates.DateFormatter('%H:%M', tz=timezone(timedelta(hours=9)))

fig = plt.figure()

ax = plt.subplot(311)
plt.plot(conferences[0], conferences[1])
plt.title('Total Conferences (21/10/2020)')
ax.xaxis.set_major_formatter(xformatter)

ax = plt.subplot(312)
plt.plot(participants[0], participants[1])
plt.title('Total Participants (21/10/2020)')
ax.xaxis.set_major_formatter(xformatter)

ax = plt.subplot(313)
plt.plot(avg_part_per_conf[0], avg_part_per_conf[1])
plt.title('Average Participants per Conference (21/10/2020)')
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

plt.show()
