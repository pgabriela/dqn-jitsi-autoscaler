FROM jrottenberg/ffmpeg:4.1-ubuntu AS builder

ENV RESOURCES_DIR=/usr/share/jitsi-meet-torture/resources

RUN apt-get update && \
	apt-get install -y wget

RUN mkdir -p $RESOURCES_DIR

RUN wget -q https://media.xiph.org/video/derf/webm/FourPeople_1280x720_60.webm -O /FourPeople_1280x720_60.webm && \
	ffmpeg -i /FourPeople_1280x720_60.webm -filter:v fps=fps=30 -q:v 5 $RESOURCES_DIR/fakeVideoStream.mjpeg

RUN wget -q https://github.com/jitsi/jitsi-meet-torture/raw/master/resources/fakeAudioStream.wav -O $RESOURCES_DIR/fakeAudioStream.wav


# Main container
FROM selenium/standalone-chrome:3.141

USER root

ENV START_XVFB=false
# 100 is more than enough for a single server
ENV NODE_MAX_SESSION=100
ENV NODE_MAX_INSTANCES=100
ENV RESOURCES_DIR=/usr/share/jitsi-meet-torture/resources

RUN mkdir -p /usr/share/man/man1 && \
	apt-get update && \
	apt-get install -y openjdk-8-jdk-headless maven vim git && \
	apt-get clean

COPY jitsi-meet-torture/ /jitsi-meet-torture/
RUN cd /jitsi-meet-torture && \
	mvn test -Djitsi-meet.tests.toRun=MalleusJitsificus -Dmaven.test.skip.exec=true

RUN mkdir -p $RESOURCES_DIR
COPY --from=builder $RESOURCES_DIR/* $RESOURCES_DIR/

RUN mv /usr/bin/google-chrome /usr/bin/google-chrome-bin
COPY rootfs/ /
RUN chmod 755 /usr/bin/google-chrome
RUN chmod 755 /start.sh

ENTRYPOINT ["/start.sh"]


