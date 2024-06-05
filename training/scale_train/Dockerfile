# syntax=docker/dockerfile:1.3

FROM nvcr.io/nvidia/pytorch:24.04-py3
# create dir
RUN mkdir -p /workspace/codeprm
WORKDIR /workspace/codeprm

RUN apt-get update && apt-get install -y tmux jq cargo git-lfs nvtop htop libaio-dev bsdmainutils

# needs to be in this specific order
RUN pip install wheel 
RUN pip install packaging
RUN pip install flash-attn --no-build-isolation

# copy requirements.txt and install them
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

RUN git config --global credential.helper store

ENV WANDB_BASE_URL=https://scaleai.wandb.io/

# copy the rest of the files
COPY . /workspace/codeprm

# install the package
RUN pip install -e .

# install huggingface token
RUN bash -c "huggingface-cli login --api-key $(cat /workspace/codeprm/training/scale_train/hf_token)"
