ARG APP_IMAGE=continuumio/miniconda3
FROM ${APP_IMAGE}
ARG APP_IMAGE
ENV PATH="/root/miniconda3/bin:${PATH}"
RUN if [ "$APP_IMAGE" = "nvidia/cuda:12.0.0-devel-ubuntu22.04" ]; then \
        echo "Using CUDA image" && \
        apt-get update && \
        apt-get install -y libxkbcommon-x11-0 unzip sudo git g++ libglm-dev libglew-dev libglfw3-dev libgles2-mesa-dev zlib1g-dev wget cmake vim libxi6 libgconf-2-4 && \
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
        mkdir /root/.conda && \
        bash Miniconda3-latest-Linux-x86_64.sh -b && \
        rm -f Miniconda3-latest-Linux-x86_64.sh; \
    else \
        echo "Using Conda image" && \
        apt-get update && \
        apt-get install -yq \
            cmake \
            g++ \
            libgconf-2-4 \
            libgles2-mesa-dev \
            libglew-dev \
            libglfw3-dev \
            libglm-dev \
            libxi6 \
            libxrender1 \
            libxxf86vm-dev \
            libxfixes3 \
            xorg \
            sudo \
            unzip \
            vim \
            dbus \
            software-properties-common \
            zlib1g-dev \
            libxkbcommon-x11-0 \
            wget && \
        wget https://www.python.org/ftp/python/3.11.3/Python-3.11.3.tgz && \
        tar -xf Python-3.11.3.tgz && \
        cd Python-3.11.3 && \
        ./configure --enable-optimizations && \
        make -j$(nproc) && \
        make altinstall && \
        update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.11 1 && \
        update-alternatives --install /usr/bin/python python /usr/local/bin/python3.11 1 && \
        python3.11 -m pip install --upgrade pip && \
        apt-get clean && \
        rm -rf /var/lib/apt/lists/*; \
    fi

RUN mkdir /opt/infinigen
WORKDIR /opt/infinigen
COPY . .
RUN conda init bash \
    && . ~/.bashrc \
    && conda create --name infinigen python=3.11 \
    && conda activate infinigen \
    && pip install -e .[dev]