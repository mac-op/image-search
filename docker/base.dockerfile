FROM nvcr.io/nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04
LABEL authors="httq"

RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    pip \
    libssl-dev \
    curl ca-certificates \
    build-essential
ADD https://astral.sh/uv/install.sh /uv-installer.sh

RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

WORKDIR /workspace
COPY . .

RUN uv venv
ENV PATH="/workspace/.venv/bin/:$PATH"
RUN uv pip install --upgrade pip setuptools wheel \
    && uv pip install torch==2.9.0 torchvision --index-url https://download.pytorch.org/whl/cu130 \
    && uv pip install torch-tensorrt tensorrt transformers pillow open-clip-torch\
    && uv pip install -r pyproject.toml