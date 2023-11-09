# hash:sha256:dffab5c9d94f182d3a99bd855ad0b34dd11c65f1b4ff2cf1d2c8c431c4d685de
FROM registry.codeocean.allenneuraldynamics.org/codeocean/py-r:python3.9-R4.2.2-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    wavpack-numcodecs==0.1.5

COPY postInstall /
RUN /postInstall
