FROM mcr.microsoft.com/devcontainers/rust:0-1-bullseye

# Include lld linker to improve build times either by using environment variable
# RUSTFLAGS="-C link-arg=-fuse-ld=lld" or with Cargo's configuration file (i.e see .cargo/config.toml).
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install clang lld lshw \
    && apt-get autoremove -y && apt-get clean -y

RUN apt-get update && apt-get -y install --no-install-recommends \
    ffmpeg \
    gcc \
    pciutils \
    cuda-nvcc-12-2 \
    cuda-toolkit-12-2

#create global virtual environment using python standard library tools of virtualenv
ARG USER="codespace"
ARG VENV_PATH="/home/${USER}/venv"
COPY requirements.txt /tmp/
COPY Makefile /tmp/
RUN su $USER -c "/usr/bin/python -m venv /home/${USER}/venv" \
    && su $USER -c "${VENV_PATH}/bin/pip --disable-pip-version-check --no-cache-dir install -r /tmp/requirements.txt" \
    && rm -rf /tmp/requirements.txt
