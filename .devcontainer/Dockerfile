FROM nvcr.io/nvidia/tritonserver:24.03-py3

ARG USERNAME=triton-server

RUN apt-get update \
    && apt-get install -y sudo

# Create the user
RUN apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

RUN pip3 install pre-commit ipdb \
    && pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu130 \
    && pip install --no-cache-dir torch-tensorrt tensorrt transformers pillow open-clip-torch openvino

RUN #apt-get install -y cmake rapidjson-dev

USER ${USERNAME}