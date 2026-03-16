FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Europe/Berlin \
    CUDA_HOME=/usr/local/cuda \
    FORCE_CUDA=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

SHELL ["/bin/bash", "-lc"]

ARG MAX_JOBS=8
ARG POINTCEPT_REF=v1.6.1
# Broad default for mixed GPU fleets (V100/T4-2080/A100/RTX30/RTX40/H100).
# Override with --build-arg TORCH_CUDA_ARCH_LIST="8.9" for single-machine optimized build.
ARG TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0"
ARG INSTALL_FLASH_ATTN=0

ENV MAX_JOBS=${MAX_JOBS} \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    INSTALL_FLASH_ATTN=${INSTALL_FLASH_ATTN}

RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata ca-certificates \
    git wget curl vim \
    build-essential ninja-build cmake pkg-config \
    python3 python3-dev python3-pip python3-venv \
    libglib2.0-0 libsm6 libxrender1 libxext6 libgl1 \
    libsparsehash-dev \
    && ln -snf /usr/share/zoneinfo/${TZ} /etc/localtime \
    && echo ${TZ} > /etc/timezone \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && python3 -m pip install --upgrade pip setuptools wheel \
    && rm -rf /var/lib/apt/lists/*

# PTv3 / Pointcept recommended CUDA 12.4 + PyTorch 2.5.0 stack.
RUN python -m pip install \
    torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Common runtime deps from Pointcept environment.
RUN python -m pip install \
    numpy==1.26.4 \
    h5py pyyaml scipy tensorboard tensorboardx \
    sharedarray easydict termcolor tqdm scikit-learn \
    addict einops plyfile timm yapf wandb \
    ftfy regex matplotlib open3d

# PYG ecosystem matched to torch 2.5.0 + cu124.
RUN python -m pip install \
    --find-links https://data.pyg.org/whl/torch-2.5.0+cu124.html \
    torch-cluster torch-scatter torch-sparse torch-geometric

# Sparse convolution backend for PTv3 sparse modules.
RUN python -m pip install spconv-cu124

# Required by Pointcept default model imports (v1.6.x).
RUN python -m pip install peft

# FlashAttention is optional in PTv3. For RTX 2080 (sm75), FlashAttention 2 is not the
# recommended path, so we keep it opt-in via build arg INSTALL_FLASH_ATTN=1.
RUN python -m pip install packaging \
    && if [ "${INSTALL_FLASH_ATTN}" = "1" ]; then \
        python -m pip install --no-build-isolation flash-attn; \
    else \
        echo "Skip flash-attn installation (INSTALL_FLASH_ATTN=${INSTALL_FLASH_ATTN})"; \
    fi

# Install Pointcept and compile CUDA ops used by PTv3 training.
RUN git clone --recursive https://github.com/Pointcept/Pointcept.git /opt/Pointcept \
    && cd /opt/Pointcept \
    && git checkout ${POINTCEPT_REF} \
    && git submodule update --init --recursive \
    && cd libs/pointops \
    && python setup.py install

# Keep PTv3 detached repo as a reference implementation.
RUN git clone https://github.com/Pointcept/PointTransformerV3.git /opt/PointTransformerV3

ENV PYTHONPATH=/opt/Pointcept

WORKDIR /workspace

CMD ["/bin/bash"]
