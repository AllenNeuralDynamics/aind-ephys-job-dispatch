# hash:sha256:d709a4d9f1e4b18aa033244f7387c327f52f1de22b68a88db01aa5856307fe1c
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
    s3fs==2025.5.1 \
    spikeinterface==0.105.1 \
    universal-pathlib==0.3.10 \
    wavpack-numcodecs==0.2.2
