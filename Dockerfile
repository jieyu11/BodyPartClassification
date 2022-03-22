ARG UBUNTU_VERSION=18.04
ARG ARCH=
ARG CUDA=10.1
FROM nvidia/cuda${ARCH:+-$ARCH}:${CUDA}-base-ubuntu${UBUNTU_VERSION} as base
# specify ARCH and CUDA again because the FROM directive resets ARGs
ARG ARCH
ARG CUDA
ARG CUDNN=7.6.5.32-1
ARG CUDNN_MAJOR_VERSION=7
ARG LIB_DIR_PREFIX=x86_64
ARG LIBNVINFER=6.0.1-1
ARG LIBNVINFER_MAJOR_VERSION=6
# for string substitution
SHELL ["/bin/bash", "-c"]
# dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    cuda-command-line-tools-${CUDA/./-} \
    libcublas10=10.2.1.243-1 \ 
    cuda-nvrtc-${CUDA/./-} \
    cuda-cufft-${CUDA/./-} \
    cuda-curand-${CUDA/./-} \
    cuda-cusolver-${CUDA/./-} \
    cuda-cusparse-${CUDA/./-} \
    curl \
    libcudnn7=${CUDNN}+cuda${CUDA} \
    libfreetype6-dev \
    libhdf5-serial-dev \
    libzmq3-dev \
    pkg-config \
    software-properties-common \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \        
    unzip
# install TensorRT if not building for PowerPC
RUN [[ "${ARCH}" = "ppc64le" ]] || { apt-get update && \
    apt-get install -y --no-install-recommends libnvinfer${LIBNVINFER_MAJOR_VERSION}=${LIBNVINFER}+cuda${CUDA} \
    libnvinfer-plugin${LIBNVINFER_MAJOR_VERSION}=${LIBNVINFER}+cuda${CUDA} \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*; }
# TF requires CUPTI for CUDA profiling
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
# link the libcuda stub to the location where TF is searching for it and reconfigure
# dynamic linker run-time bindings
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 \
    && echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/z-cuda-stubs.conf \
    && ldconfig
# avoid unicode error
ENV LANG C.UTF-8
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip
RUN python3 -m pip --no-cache-dir install --upgrade \
    pip \
    setuptools
# TF tools expect "python" binary
RUN ln -s $(which python3) /usr/local/bin/python
# COPY requirement.txt /requirement.txt
# RUN pip install -r /requirement.txt

RUN apt-get install -y vim \
    ffmpeg \
    libsm6 \
    libxext6 
RUN rm -rf /var/lib/apt/lists/*

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
RUN pip3 install --upgrade pip
RUN pip3 install numpy==1.18.5 \
    pandas==1.1.5 \
    opencv-python==4.5.1.48 \
    h5py==2.10.0 \
    toml==0.10.2 \
    scikit-learn==0.24.2 \
    tensorflow==2.3.1 \
    h5py==2.10.0 

ENV WORK_ROOT /work
RUN mkdir $WORK_ROOT
WORKDIR $WORK_ROOT

COPY *.py Modeling/*.py models/*.h5 ./
COPY libs_internal libs_internal
COPY hbm hbm