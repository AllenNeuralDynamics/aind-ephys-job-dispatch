# hash:sha256:fa37cf12007b767e187292f5061fc09879abd45276daeb15ac45a5cdd4d954e3
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential=12.9ubuntu3 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
    aind-log-utils==0.2.3 \
    hdmf-zarr==0.11.0 \
    pynwb==2.8.3 \
    spikeinterface==0.103.2 \
    wavpack-numcodecs==0.2.2
