# hash:sha256:7cce7b22f0ff6e0a3797e0a806f0ce2bc2baa5339d752050b55fd6f414e86fcb
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
    hdmf-zarr==0.13.0 \
    pynwb==3.1.3 \
    spikeinterface==0.104.7 \
    wavpack-numcodecs==0.2.2
