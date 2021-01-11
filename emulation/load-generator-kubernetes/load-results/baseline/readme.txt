Dynamic [1, 6]# JVB (0.2 vCPU & 500 MB RAM)
compressed 0.5s/point ~ 2h 24m
> First: 4:30pm (bug: cannot remove jvb)
> Second: 9:50pm - 00:28am (bug: still using 0.3s/point; accidentally include previous data from all host)
> Third: 5:37pm - 9:15pm (Bug fix: 0.5s/point, include only some host, use latest available data instead of current time bcs telegraf data may take some time to be written)
