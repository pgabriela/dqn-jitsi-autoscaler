#cutomized jibri image to use pulseaudio
FROM jitsi/jibri
COPY 10-config /etc/cont-init.d
COPY client.conf /defaults
COPY daemon.conf /defaults
COPY ffmpeg /defaults
COPY google-chrome /defaults
COPY services.d/ /etc/services.d/
