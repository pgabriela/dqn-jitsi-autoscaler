Use loss rate only to estimate QoS
reward = (self.loss_delta * self.jvb_num_delta) + \
	(self.alpha * sigmoid(-abs(self.loss_delta) * 1e20) * -self.jvb_num_delta) + \
	(v * self.beta)
alpha = 1e-2
beta = 1e-2
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

17-12-2020
1: Restarted near end because DoubleTensor expected FloatTensor, JVBs could not connect to the XMPP server
12:17pm autoscaler deployed
12:27pm torture deployed
1:54pm torture ended
2:27pm influxdb backed up

2: 
2:48pm autoscaler deployed
2:58pm torture deployed
4:25pm torture ended
4:35pm influxdb backed up
5:05pm torture deployed
6:45pm torture ended
6:48pm influxdb backed up
