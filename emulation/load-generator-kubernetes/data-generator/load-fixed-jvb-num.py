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


influxdb_jcf = InfluxDBClient(
    host="127.0.0.1",
    port=8086,
    database="jicofo_stats")

influxdb_jvb = InfluxDBClient(
    host="127.0.0.1",
    port=8086,
    database="jitsi_stats")

curr_time = time.time()
# Query InfluxDB
monitoring_data = influxdb_jcf.query("select conferences, participants from jicofo_stats where time >= '2020-12-07T09:50:00Z';")
monitoring_data2 = influxdb_jvb.query("select host, jitter_aggregate, overall_loss, rtt_aggregate from jitsi_stats where time >= '2020-12-07T09:50:00Z';")
print("Query time: %d seconds" % (time.time() - curr_time))

conferences = [[], []]
participants = [[], []]
jitters = {
    'jvb-0': [[], []],
    'jvb-1': [[], []],
    'jvb-2': [[], []],
    'avg': [[], []],
}
losses = {
    'jvb-0': [[], []],
    'jvb-1': [[], []],
    'jvb-2': [[], []],
    'avg': [[], []],
}
rtts = {
    'jvb-0': [[], []],
    'jvb-1': [[], []],
    'jvb-2': [[], []],
    'avg': [[], []],
}
avg_part_per_conf = [[], []]

curr_time = time.time()
for row in monitoring_data.get_points():
    dt = dateutil.parser.parse(row['time']).astimezone(timezone(timedelta(hours=7)))
    conferences[0].append(dt)
    participants[0].append(dt)
    conferences[1].append(int(row['conferences']))
    participants[1].append(int(row['participants']))
for row in monitoring_data2.get_points():
    dt = dateutil.parser.parse(row['time']).astimezone(timezone(timedelta(hours=7)))
    jitters[row['host']][0].append(dt)
    losses[row['host']][0].append(dt)
    rtts[row['host']][0].append(dt)
    jitters[row['host']][1].append(row['jitter_aggregate'])
    losses[row['host']][1].append(row['overall_loss'])
    rtts[row['host']][1].append(row['rtt_aggregate'])
for i in range(len(jitters['jvb-0'][0])):
    jitters['avg'][0].append(jitters['jvb-0'][0][i])
    losses['avg'][0].append(jitters['jvb-0'][0][i])
    rtts['avg'][0].append(jitters['jvb-0'][0][i])
    combined_jitters = []
    combined_losses = []
    combined_rtts = []
    for h in jitters:
        if h != 'avg':
            combined_jitters.append(jitters[h][1][i])
            combined_losses.append(losses[h][1][i])
            combined_rtts.append(rtts[h][1][i])
    jitters['avg'][1].append(statistics.mean(combined_jitters))
    losses['avg'][1].append(statistics.mean(combined_losses))
    rtts['avg'][1].append(statistics.mean(combined_rtts))
print("Cleaning time: %d seconds" % (time.time() - curr_time))


curr_time = time.time()
for i in range(len(conferences[0])):
    if conferences[1][i] == 0:
        continue
    avg = participants[1][i] / conferences[1][i]
    avg_part_per_conf[0].append(participants[0][i])
    avg_part_per_conf[1].append(avg)
print("Average calculation time: %d seconds" % (time.time() - curr_time))

xformatter = mdates.DateFormatter('%H:%M:%S', tz=timezone(timedelta(hours=7)))

fig = plt.figure()

curr_time = time.time()
ax = plt.subplot(321)
plt.plot(conferences[0], conferences[1])
plt.title('Resulted Total Conferences (Fixed JVB)')
ax.xaxis.set_major_formatter(xformatter)
plt.grid(True)
print("First Plotting time: %d seconds" % (time.time() - curr_time))

curr_time = time.time()
ax = plt.subplot(322)
plt.plot(jitters['avg'][0], jitters['avg'][1])
plt.title('Resulted Overall Jitters (Fixed JVB)')
ax.xaxis.set_major_formatter(xformatter)
plt.grid(True)
print("Second Plotting time: %d seconds" % (time.time() - curr_time))

chi_j = chi.fit(jitters['avg'][1])
#avg_j = statistics.mean(jitters['avg'][1])
avg_j = chi.mean(chi_j[0], loc=chi_j[1], scale=chi_j[2])
plt.text(0.95, 0.95, 'Overall Mean Jitter (Chi-square) = ' + str(avg_j),
    horizontalalignment='right',
    verticalalignment='top',
    transform=ax.transAxes)
#std_j = statistics.pstdev(jitters['avg'][1])
std_j = chi.std(chi_j[0], loc=chi_j[1], scale=chi_j[2])
plt.text(0.95, 0.85, 'Overall STDEV Jitter (Chi-square) = ' + str(std_j),
    horizontalalignment='right',
    verticalalignment='top',
    transform=ax.transAxes)
int_j = chi.interval(0.95, chi_j[0], loc=chi_j[1], scale=chi_j[2])
plt.text(0.95, 0.75, '95% Confidence Interval (Chi-square) = ' + str(int_j),
    horizontalalignment='right',
    verticalalignment='top',
    transform=ax.transAxes)

curr_time = time.time()
ax = plt.subplot(323)
plt.plot(participants[0], participants[1])
plt.title('Resulted Total Participants (Fixed JVB)')
ax.xaxis.set_major_formatter(xformatter)
plt.grid(True)
print("Third Plotting time: %d seconds" % (time.time() - curr_time))

curr_time = time.time()
ax = plt.subplot(324)
plt.plot(losses['avg'][0], losses['avg'][1])
plt.title('Resulted Overall Losses (Fixed JVB)')
ax.xaxis.set_major_formatter(xformatter)
plt.grid(True)
print("Fourth Plotting time: %d seconds" % (time.time() - curr_time))

chi_l = chi.fit(losses['avg'][1])
#avg_l = statistics.mean(losses['avg'][1])
avg_l = chi.mean(chi_l[0], loc=chi_l[1], scale=chi_l[2])
plt.text(0.95, 0.95, 'Overall Mean Loss (Chi-square) = ' + str(avg_l),
    horizontalalignment='right',
    verticalalignment='top',
    transform=ax.transAxes)
#std_l = statistics.pstdev(losses['avg'][1])
std_l = chi.std(chi_l[0], loc=chi_l[1], scale=chi_l[2])
plt.text(0.95, 0.85, 'Overall STDEV Loss (Chi-square) = ' + str(std_l),
    horizontalalignment='right',
    verticalalignment='top',
    transform=ax.transAxes)
int_l = chi.interval(0.95, chi_l[0], loc=chi_l[1], scale=chi_l[2])
plt.text(0.95, 0.75, '95% Confidence Interval (Chi-square) = ' + str(int_l),
    horizontalalignment='right',
    verticalalignment='top',
    transform=ax.transAxes)

curr_time = time.time()
ax = plt.subplot(325)
plt.plot(avg_part_per_conf[0], avg_part_per_conf[1])
plt.title('Resulted Average Participants per Conference (Fixed JVB)')
ax.xaxis.set_major_formatter(xformatter)
plt.grid(True)
print("Fifth Plotting time: %d seconds" % (time.time() - curr_time))

avg = statistics.mean(avg_part_per_conf[1])
plt.text(0.95, 0.95, 'Overall Average Participants per Conference = ' + str(avg),
    horizontalalignment='right',
    verticalalignment='top',
    transform=ax.transAxes)
std = statistics.pstdev(avg_part_per_conf[1])
plt.text(0.95, 0.85, 'Overall STDEV Participants per Conference = ' + str(std),
    horizontalalignment='right',
    verticalalignment='top',
    transform=ax.transAxes)

curr_time = time.time()
ax = plt.subplot(326)
plt.plot(rtts['avg'][0], rtts['avg'][1])
plt.title('Resulted Overall RTTs (Fixed JVB)')
ax.xaxis.set_major_formatter(xformatter)
plt.grid(True)
print("Sixth Plotting time: %d seconds" % (time.time() - curr_time))

chi_r = chi.fit(rtts['avg'][1])
#avg_l = statistics.mean(rtts['avg'][1])
avg_r = chi.mean(chi_r[0], loc=chi_r[1], scale=chi_r[2])
plt.text(0.95, 0.95, 'Overall Mean RTT (Chi-square) = ' + str(avg_r),
    horizontalalignment='right',
    verticalalignment='top',
    transform=ax.transAxes)
#std_l = statistics.pstdev(rtts['avg'][1])
std_r = chi.std(chi_r[0], loc=chi_r[1], scale=chi_r[2])
plt.text(0.95, 0.85, 'Overall STDEV RTT (Chi-square) = ' + str(std_r),
    horizontalalignment='right',
    verticalalignment='top',
    transform=ax.transAxes)
int_r = chi.interval(0.95, chi_r[0], loc=chi_r[1], scale=chi_r[2])
plt.text(0.95, 0.75, '95% Confidence Interval (Chi-square) = ' + str(int_r),
    horizontalalignment='right',
    verticalalignment='top',
    transform=ax.transAxes)

plt.show()
