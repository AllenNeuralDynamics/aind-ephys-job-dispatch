# hash:sha256:52b7d4b66b1ceb4c6ad2195adcf3e38fc4aeefcbc67eedba509e629a46ca352c
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
    probeinterface==0.2.23 \
    pynwb==2.8.0 \
    spikeinterface[full]==0.100.7 \
    wavpack-numcodecs==0.1.5
