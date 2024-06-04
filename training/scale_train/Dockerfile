# syntax=docker/dockerfile:1.3

FROM nvcr.io/nvidia/pytorch:24.04-py3
COPY . /workspace/codeprm
WORKDIR /workspace/codeprm

RUN apt-get update && apt-get install -y tmux jq cargo git-lfs nvtop htop libaio-dev bsdmainutils

RUN pip install -r requirements.txt
