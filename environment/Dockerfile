# hash:sha256:707d5f81deab1de27fced7c5989724897859ab0dceeacd79b37ec08931448ec9
FROM registry.codeocean.com/codeocean/matlab:2021b-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG MLM_LICENSE_FILE

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3=3.8.2-0ubuntu2 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install -U --no-cache-dir \
    matplotlib==3.5.3 \
    pandas==1.4.3 \
    pyyaml==6.0 \
    scikit-learn==1.1.2 \
    seaborn==0.11.2 \
    tabulate==0.8.10
