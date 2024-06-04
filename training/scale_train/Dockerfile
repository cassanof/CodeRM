# syntax=docker/dockerfile:1.3

FROM nvcr.io/nvidia/pytorch:24.04-py3
COPY . /workspace/codeprm
WORKDIR /workspace/codeprm

RUN apt-get update && apt-get install -y tmux jq cargo git-lfs nvtop htop libaio-dev bsdmainutils

# needs to be in this specific order
RUN pip install wheel 
RUN pip install packaging
RUN pip install pytorch==2.2.1
RUN pip install flash-attn --no-build-isolation
RUN pip install -r requirements.txt
