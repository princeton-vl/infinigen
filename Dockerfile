FROM continuumio/miniconda3

RUN apt-get update -yq \
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
        zlib1g-dev

RUN mkdir /opt/infinigen
WORKDIR /opt/infinigen

COPY . .
RUN ./install.sh
