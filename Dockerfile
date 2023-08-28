ARG APP_IMAGE=continuumio/miniconda3
FROM ${APP_IMAGE}
ARG APP_IMAGE
ENV PATH="/root/miniconda3/bin:${PATH}"
RUN if [ "$APP_IMAGE" = "nvidia/cuda:12.0.0-devel-ubuntu22.04" ]; then \
    echo "Using CUDA image" && \
    apt-get update && \
    apt-get install -y unzip sudo git g++ libglm-dev libglew-dev libglfw3-dev libgles2-mesa-dev zlib1g-dev wget cmake vim libxi6 libgconf-2-4 && \
    wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh; \
    && apt-get install libxkbcommon-x11-0 \
else \
    echo "Using Conda image" && \
    apt-get update -yq \
    && apt-get install -yq \
        cmake \
        g++ \
        libgconf-2-4 \
        libgles2-mesa-dev \
        libglew-dev \
        libglfw3-dev \
        libglm-dev \
        libxi6 \
        sudo \
        unzip \
        vim \
        zlib1g-dev; \
    && apt-get install libxkbcommon-x11-0 \
fi

RUN mkdir /opt/infinigen
WORKDIR /opt/infinigen
COPY . .
RUN conda init bash \
    && . ~/.bashrc \
    && conda create --name infinigen python=3.10 \
    && conda activate infinigen \
    && pip install -e .[dev]