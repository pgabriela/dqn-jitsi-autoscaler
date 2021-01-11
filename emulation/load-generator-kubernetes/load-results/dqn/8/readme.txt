Use loss rate only to estimate QoS
reward = (self.loss_delta * self.jvb_num_delta) + \
	(self.alpha * sigmoid(-abs(self.loss_delta) * 1e20) * -self.jvb_num_delta) + \
	(v * self.beta)
alpha = 1e-2
beta = 1e-6
target_update = 10
v = +-1 (based on prev_action_success)
eps_decay = 200
with selective scale down
back to 1h27m (0.3s per point)
added 4 features: conferences_delta, participants_delta, jvb_num_delta, loss_delta
removed 2 features: rtt, jitter
for random action, 67% prev action, 17% for other action -> More exploration on JVB num
Adam Optimizer
128 Batch Size
bugfix: limit query to 30 for loss rate (previously 10)

:pm influxdb backed up
:pm torture deployed
:pm torture ended
:pm influxdb backed up
