import time
import os
import requests
import json
import matplotlib.pyplot as plt
import numpy as np
import dateutil.parser
import matplotlib.dates as mdates
from datetime import timezone, timedelta, datetime
from scipy.stats import norm, chi
from influxdb import InfluxDBClient
import statistics
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
import random


influxdb_jcf = InfluxDBClient(
    host="127.0.0.1",
    port=8086,
    database="jicofo_stats")

influxdb_jvb = InfluxDBClient(
    host="127.0.0.1",
    port=8086,
    database="jitsi_stats")

f = open('dbV3.csv', 'a')

start_time = '2020-12-22T11:16:00Z'
end_time = '2020-12-22T15:25:00Z'
curr_time = time.time()
# Query InfluxDB
monitoring_data = influxdb_jcf.query("select sum(conferences) as conferences, sum(participants) as participants, sum(bridge_selector_bridge_count) as jvb_num from jicofo_stats where time >= '" + \
    start_time + "' and time < '" + \
    end_time + "' group by time(1s) fill(0);")
monitoring_data2 = influxdb_jvb.query("select mean(overall_loss) as loss from jitsi_stats where time >= '" + \
    start_time + "' and time <  '" + \
    end_time + "' group by time(1s) fill(0);")
monitoring_data3 = influxdb_jvb.query("select count(conferences) as zero_conf from jitsi_stats where time >= '" + \
    start_time + "' and time < '" + \
    end_time + "' and conferences = 0 group by time(1s) fill(0);")
print("Query time: %d seconds" % (time.time() - curr_time))

conferences = [[], []]
participants = [[], []]
jvb_num = [[], []]
losses = [[], []]
zero_conferences = [[], []]

curr_time = time.time()
for row in monitoring_data.get_points():
    dt = dateutil.parser.parse(row['time']).astimezone(timezone(timedelta(hours=7)))
    conferences[0].append(dt)
    participants[0].append(dt)
    jvb_num[0].append(dt)
    conferences[1].append(int(row['conferences']))
    participants[1].append(int(row['participants']))
    jvb_num[1].append(int(row['jvb_num']))
for row in monitoring_data2.get_points():
    dt = dateutil.parser.parse(row['time']).astimezone(timezone(timedelta(hours=7)))
    losses[0].append(dt)
    losses[1].append(row['loss'])
for row in monitoring_data3.get_points():
    dt = dateutil.parser.parse(row['time']).astimezone(timezone(timedelta(hours=7)))
    zero_conferences[0].append(dt)
    zero_conferences[1].append(int(row['zero_conf']))
print("Cleaning time: %d seconds" % (time.time() - curr_time))

for i in range(len(conferences[0])):
    f.write("%d,%d,%d,%f,%d\n" % (conferences[1][i], participants[1][i], jvb_num[1][i], losses[1][i], zero_conferences[1][i]))
r = random.randrange(120, 600)
for i in range(r):
    f.write("0,0,1,0,1\n")
f.close()
