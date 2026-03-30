# hash:sha256:73c763f51118485a77d3e2318e8d34ab9389021d80fd46c060f2d960023c0f5d
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
    spikeinterface==0.104.0 \
    wavpack-numcodecs==0.2.2
