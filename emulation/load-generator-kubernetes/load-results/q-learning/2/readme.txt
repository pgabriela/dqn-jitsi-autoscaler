Use loss rate only to estimate QoS
w1 = 20
eps_decay = 200
alpha = 0.1
gamma = 0.5
with selective scale down
back to 1h27m (0.3s per point)
uses only 5 features: participant_count_delta, total_jvb_count_delta, idle_jvb_count_delta, loss_delta, & curr_loss
for random action, prioritize selecting unexplored state-action pairs

# define reward function
def calc_reward(state, action):
    loss_delta = state[3]
    curr_loss = state[4]
    if action == 0:
        jvb_num_delta = 1
    elif action == 1:
        jvb_num_delta = -1
    else:
        jvb_num_delta = 0
    reward = loss_delta * jvb_num_delta
    if loss_delta == 0:
        reward = W1 * curr_loss * jvb_num_delta
        if curr_loss == 0:
            reward = -jvb_num_delta
    return reward

23-12-2020
1: Bugfix the curr_loss used was always 0.0 in the q-table
2:40pm autoscaler deployed
2:45pm torture deployed
4:03pm influxdb etc backed up

* Load generator run out of credits in the middle of the session
