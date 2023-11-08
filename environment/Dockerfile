FROM continuumio/miniconda3:23.9.0-0

ARG DEBIAN_FRONTEND=noninteractive

RUN pip install -U --no-cache-dir \
    wavpack-numcodecs==0.1.5

COPY postInstall /
RUN /postInstall
