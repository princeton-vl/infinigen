ARG APP_IMAGE=condaforge/miniforge3
FROM ${APP_IMAGE}
ARG APP_IMAGE
ARG TARGETARCH
# /root/miniforge3/bin is used only when miniforge is manually installed in the CUDA path;
# the condaforge/miniforge3 base image already includes /opt/conda/bin in PATH.
ENV PATH="/root/miniforge3/bin:${PATH}"
RUN if [ "$APP_IMAGE" = "nvidia/cuda:12.0.0-devel-ubuntu22.04" ]; then \
    echo "Using CUDA image" && \
    apt-get update && \
    apt-get install -y unzip sudo git g++ libglm-dev libglew-dev libglfw3-dev libgles2-mesa-dev zlib1g-dev wget cmake vim libxi6 libgconf-2-4 && \
    if [ "${TARGETARCH}" = "arm64" ]; then \
        CONDA_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh"; \
    else \
        CONDA_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"; \
    fi && \
    wget "${CONDA_URL}" -O /tmp/miniforge.sh && \
    mkdir -p /root/.conda && \
    bash /tmp/miniforge.sh -b -p /root/miniforge3 && \
    rm -f /tmp/miniforge.sh && \
    apt-get install -y libxkbcommon-x11-0; \
else \
    echo "Using Conda image" && \
    apt-get update -yq && \
    apt-get install -yq cmake g++ libgconf-2-4 libgles2-mesa-dev libglew-dev libglfw3-dev libglm-dev libxi6 sudo unzip vim zlib1g-dev && \
    apt-get install -y libxkbcommon-x11-0; \
fi

RUN mkdir /opt/infinigen
WORKDIR /opt/infinigen
COPY . .
RUN conda init bash && \
    . ~/.bashrc && \
    conda create --name infinigen python=3.11 -y && \
    conda activate infinigen && \
    pip install -e ".[dev]"
