w1 = 1e-1
w2 = 150
b = 0.5
eps_decay = 220
with selective scale down
back to 1h27m (0.3s per point)
reward = math.exp(-self.w1 * (self.rtt + self.jitter + self.loss)) * ((self.w2+1-self.jvb_num) / self.w2)* 10 / (1 + self.jvb_load_spread) + b
added 3 features: conferences_diff, participants_diff, jvb_num_diff
for random action, 50% prev action, 25% for other action -> More exploration on JVB num
Adam Optimizer
128 Batch Size

16-12-2020
2:27pm autoscaler deployed
2:33pm torture deployed
4:01pm torture ended
4:09pm influxdb backed up
4:48pm torture deployed
6:15pm torture ended
6:24pm influxdb backed up
