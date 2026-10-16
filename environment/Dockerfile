# hash:sha256:093db53833e252a95e2ee33643a732e09cf0fbd1750698a9b0406c374e0d27c0
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
    hdmf-zarr==0.13.0 \
    log-schema==0.2.10.dev1 \
    pynwb==3.1.3 \
    spikeinterface==0.105.2 \
    wavpack-numcodecs==0.2.2
