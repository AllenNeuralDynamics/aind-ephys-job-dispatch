# hash:sha256:2913135f3facc6b603f0d5a1b176a99e0e61f62c6be8d52d99b2f1178fe4250e
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG AWS_ACCESS_KEY_ID
ARG AWS_DEFAULT_REGION
ARG AWS_SECRET_ACCESS_KEY

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-log-utils \
    hdmf-zarr==0.8.0 \
    pynwb==2.8.0 \
    spikeinterface[full]==0.101.2 \
    wavpack-numcodecs==0.2.1
