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
monitoring_data = influxdb.query("select conferences, participants, host from jitsi_stats where time >= '2020-11-23T15:00:00Z' AND time < '2020-11-24T15:00:00Z';")
monitoring_data2 = influxdb.query("select conferences, participants, host from jitsi_stats where time >= '2020-11-19T15:00:00Z' AND time < '2020-11-20T15:00:00Z';")

conferences = [[], []]
participants = [[], []]
conferences2 = [[], []]
participants2 = [[], []]

for row in monitoring_data.get_points():
    if row['host']:
        continue
    else:
        conferences[0].append(dateutil.parser.parse(row['time']).astimezone(timezone(timedelta(hours=9))))
        conferences[1].append(int(row['conferences']))
        participants[0].append(dateutil.parser.parse(row['time']).astimezone(timezone(timedelta(hours=9))))
        participants[1].append(int(row['participants']))

for row in monitoring_data2.get_points():
    if row['host']:
        continue
    else:
        conferences2[0].append(dateutil.parser.parse(row['time']).astimezone(timezone(timedelta(hours=9))))
        conferences2[1].append(int(row['conferences']))
        participants2[0].append(dateutil.parser.parse(row['time']).astimezone(timezone(timedelta(hours=9))))
        participants2[1].append(int(row['participants']))

xformatter = mdates.DateFormatter('%H:%M', tz=timezone(timedelta(hours=9)))

ax = plt.subplot(221)
plt.plot(conferences2[0], conferences2[1])
plt.title('Total Conferences (20/11/2020)')
ax.xaxis.set_major_formatter(xformatter)

ax = plt.subplot(222)
plt.plot(participants2[0], participants2[1])
plt.title('Total Participants (20/11/2020)')
ax.xaxis.set_major_formatter(xformatter)

ax = plt.subplot(223)
plt.plot(conferences[0], conferences[1])
plt.title('Total Conferences (24/11/2020)')
ax.xaxis.set_major_formatter(xformatter)

ax = plt.subplot(224)
plt.plot(participants[0], participants[1])
plt.title('Total Participants (24/11/2020)')
ax.xaxis.set_major_formatter(xformatter)

plt.show()
