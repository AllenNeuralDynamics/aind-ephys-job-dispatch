# hash:sha256:812efd5f51a54c377e18170de95f7454685fd8492823361b85966c685a6f10d3
FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    wavpack-numcodecs==0.1.5

COPY postInstall /
RUN /postInstall
