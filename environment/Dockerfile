# hash:sha256:b599e0ef9358cdd088dc2a78a3e6c3a380f9427771a3e7ae698e687bc4195966
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

COPY postInstall /
RUN /postInstall
