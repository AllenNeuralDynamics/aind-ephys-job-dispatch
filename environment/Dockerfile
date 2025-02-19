# hash:sha256:7bec27e5aa81e9bd9eed54d349a26b9f3f2b4249c57177ace21c0cb2ac507243
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
    aind-log-utils==0.2.3 \
    hdmf-zarr==0.11.0 \
    pynwb==2.8.3 \
    spikeinterface==0.102.1 \
    wavpack-numcodecs==0.2.1
