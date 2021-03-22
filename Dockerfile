FROM debian:bullseye-slim

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get dist-upgrade -yq && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get install -yq \
    git \
    build-essential \
    wget \
    bash \
    coreutils \
    ca-certificates \
    curl \
    && \
    rm -rf /var/lib/apt/lists/*

RUN curl -L https://micromamba.snakepit.net/api/micromamba/linux-64/0.7.14 | \
    tar -xj -C / bin/micromamba

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH "$MAMBA_ROOT_PREFIX/bin:$PATH"
RUN mkdir $MAMBA_ROOT_PREFIX && \
    yes | /bin/micromamba shell init -s bash -p $MAMBA_ROOT_PREFIX

WORKDIR /opt/batchlogit

RUN micromamba install -y \
       -n base \
       -c conda-forge \
       python=3.8.8

COPY environment.yml ./

RUN micromamba install -y \
       -n base \
       -c rapidsai-nightly \
       -c nvidia \
       -c conda-forge \
       -c defaults \
       -c anaconda \
       -f ./environment.yml \
       && \
    rm /opt/conda/pkgs/cache/*

COPY . /opt/batchlogit

RUN echo "/opt/batchlogit" > \
    /opt/conda/lib/python3.8/site-packages/batchlogit.pth

