FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin

RUN apt update \
    && apt-get install -y \
    g++ \
    cmake \
    libatlas-base-dev \
    python3 \
    python3-dev \
    python3-pip \
    swig \
    git\
    libhdf5-serial-dev\
    && ln -sf /usr/bin/swig4.0 /usr/bin/swig

RUN pip3 install python-libsbml>=5.17.0 numpy

COPY container_files.tar.gz /tmp

RUN pip3 install -U --upgrade pip wheel \
    && mkdir -p /tmp/container_files/ \
    && cd /tmp/container_files \
    && tar -xzf ../container_files.tar.gz

RUN pip3 install aesara arviz pymc==5.1.1 theano numpy matplotlib scipy pandas seaborn arviz jupyter
