# Pytorch Docker
FROM bitnami/pytorch

WORKDIR /app
ADD . /app/
RUN pip install --upgrade pip
RUN pip install kubernetes
RUN pip install influxdb

CMD ["python", "-u", "/app/dqn_autoscaler.py"]
