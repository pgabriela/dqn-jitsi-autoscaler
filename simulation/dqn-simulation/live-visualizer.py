import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


def animate(i):
    conf_count_over_time = []
    part_count_over_time = []
    jvb_num_history = []
    rewards_history = []
    losses_history = []

    try:
        conf_f = open('logs/conference_count.txt', 'r')
        for row in conf_f:
            conf_count_over_time.append(int(row.rstrip()))
    except:
        pass
    try:
        part_f = open('logs/participant_count.txt', 'r')
        for row in part_f:
            part_count_over_time.append(int(row.rstrip()))
    except:
        pass
    try:
        jvb_f = open('logs/jvb_count.txt', 'r')
        for row in jvb_f:
            jvb_num_history.append(int(row.rstrip()))
    except:
        pass
    try:
        reward_f = open('logs/rewards.txt', 'r')
        for row in reward_f:
            rewards_history.append(float(row.rstrip()))
    except:
        pass
    try:
        loss_f = open('logs/losses.txt', 'r')
        for row in loss_f:
            losses_history.append(float(row.rstrip()))
    except:
        pass
    plt.clf()
    plt.subplot(511)
    plt.title("Conferences")
    plt.plot(np.arange(len(conf_count_over_time)), conf_count_over_time)
    plt.subplot(512)
    plt.title("Participants")
    plt.plot(np.arange(len(part_count_over_time)), part_count_over_time)
    plt.subplot(513)
    plt.title("JVB Count")
    plt.plot(np.arange(len(jvb_num_history)), jvb_num_history)
    plt.subplot(514)
    plt.title("Rewards")
    plt.plot(np.arange(len(rewards_history)), rewards_history)
    plt.subplot(515)
    plt.title("Losses")
    plt.plot(np.arange(len(losses_history)), losses_history)

ani = FuncAnimation(plt.gcf(), animate, None, interval=5000)
plt.show()
