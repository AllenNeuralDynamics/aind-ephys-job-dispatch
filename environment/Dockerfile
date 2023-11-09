# hash:sha256:f113041840f74d75f765bf896f763dfbaa9b40a3742e6ddfcdc579ac1788bdea
FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    spikeinterface[full]==0.99.0 \
    wavpack-numcodecs==0.1.5
