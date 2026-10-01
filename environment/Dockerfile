# hash:sha256:de96277e12fd95f2cd49fc343efd9d1c870484a59a16aaa736029130870937d2
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
    spikeinterface==0.105.0 \
    wavpack-numcodecs==0.2.2
