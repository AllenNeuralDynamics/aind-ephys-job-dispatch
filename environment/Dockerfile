# hash:sha256:dad681911c89d8b2ba1788388485da09c27b69194506a4a985480d3ea9b2f36b
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
    spikeinterface==0.104.7 \
    wavpack-numcodecs==0.2.2
