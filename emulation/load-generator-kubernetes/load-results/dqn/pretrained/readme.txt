Dynamic [1, 6]# JVB (0.2 vCPU & 500 MB RAM)
compressed 0.5s/point ~ 2h 24m
interval 12s extra cooldown 7s

7:23pm - autoscaler deployed
7:28pm - torture deployed
9:53pm - torture ended
10:23pm - influxdb backed up

*Bug:
Kube api-server was often unreachable for minutes, making the autoscaler stop applying actions for minutes
sometimes to the point where it was timeout
The code has not handled the timeout case such that it was restarted, affecting the EPS (restart back to EPS_START again)
