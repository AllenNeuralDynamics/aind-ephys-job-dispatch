# hash:sha256:ba5ec287206d8abc9baa62296006b7a15649c1e56363cddb9c6602f6a2e2ac09
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
    hdmf-zarr==0.8.0 \
    pynwb==2.8.0 \
    spikeinterface[full]==0.101.1 \
    wavpack-numcodecs==0.1.5
