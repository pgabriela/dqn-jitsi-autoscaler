FROM python:3.7

RUN mkdir /app
WORKDIR /app
ADD . /app/
RUN pip install kubernetes
RUN pip install influxdb

CMD ["python", "-u", "/app/baseline_autoscaler.py"]
