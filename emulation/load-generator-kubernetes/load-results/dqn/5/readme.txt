w1 = 1e-1
w2 = 25
b = 0.05
eps_decay = 150
with selective scale down
back to 1h27m (0.3s per point)
reward = math.exp(-self.w1 * (self.rtt + self.jitter + self.loss)) * (self.w2+1-self.jvb_num) / self.w2 + b
added 3 features: conferences_diff, participants_diff, jvb_num_diff

16-12-2020
11:45am autoscaler deployed
11:55am torture deployed
1:23pm torture ended
1:35pm influxdb backed up
