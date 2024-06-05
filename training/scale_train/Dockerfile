FROM nvidia/cuda:12.1.0-devel-ubuntu20.04

# create dir
RUN mkdir -p /workspace/codeprm
WORKDIR /workspace/codeprm

RUN apt-get update && apt-get install -y tmux \
      jq \
      cargo \
      git-lfs \
      nvtop \
      htop \
      libaio-dev \
      bsdmainutils \
      make \
      build-essential \
      libssl-dev \
      zlib1g-dev \
      libbz2-dev \
      libreadline-dev \
      libsqlite3-dev \
      wget \
      curl \
      python3 \
      python3-pip \
      python3-venv \
      python3-dev

# needs to be in this specific order
RUN pip install wheel 
RUN pip install packaging
RUN pip install torch==2.2.1
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
RUN bash -c "huggingface-cli login --token $(cat /workspace/codeprm/training/scale_train/hf_token) --add-to-git-credential"

# copy wandb netrc
RUN cp /workspace/codeprm/training/scale_train/wandb_netrc /root/.netrc
