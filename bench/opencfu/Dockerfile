# OpenCFU 4.0.0 (Debian package) for the colonycount benchmark; see README.md.
FROM debian:stable-slim@sha256:fbfb4059d2127120aa05b6e6064523d2102552f0cf0026ddf0701c12f1f8f3e0
RUN apt-get update \
 && apt-get install -y --no-install-recommends opencfu=4.0.0+dfsg-2+b2 \
 && rm -rf /var/lib/apt/lists/* \
 && useradd bench
USER bench
# opencfu loads its colony classifier from ./data/trainedClassifier*.xml
WORKDIR /usr/share/opencfu
