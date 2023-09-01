cleanpip:
	rm -rf *.egg-info
	rm -rf build

clean_terrain:
	rm -rf infinigen/terrain/*.egg-info
	rm -rf infinigen/terrain/__pycache__
	rm -rf infinigen/terrain/build

terrain: clean_terrain
	bash scripts/install/compile_terrain.sh

customgt:
	bash scripts/install/compile_opengl.sh

flip_fluids:
	bash scripts/install/compile_flip_fluids.sh

DOCKER_BUILD_PROGRESS ?= auto
DOCKER_TAG ?= infinigen_docker_img

PWD = $(shell pwd)

XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth

default:

docker-build:
	docker build \
		--tag $(DOCKER_TAG) \
		--progress $(DOCKER_BUILD_PROGRESS) .

docker-build-cuda:
	docker build \
		--tag $(DOCKER_TAG) \
		--progress $(DOCKER_BUILD_PROGRESS) \
		--build-arg APP_IMAGE=nvidia/cuda:12.0.0-devel-ubuntu22.04 .

docker-clean:
	echo "Removing infinigen docker image if already exists..."
	-docker rmi -f $(DOCKER_TAG)

docker-setup:
	sudo apt-get install x11-xserver-utils \
		&& touch ~/.Xauthority \
		&& xauth add $(HOST):0 . $(shell xxd -l 16 -p /dev/urandom) \
		&& touch "$(XAUTH)" \
		&& xauth nlist "$(DISPLAY)" | sed -e 's/^..../ffff/' | xauth -f "$(XAUTH)" nmerge - \
		&& xhost +local:docker

docker-run:
	docker run -td --privileged --net=host --ipc=host \
		--name="infinigen" \
		--gpus=all \
		--env NVIDIA_DISABLE_REQUIRE=1 \
		-e "DISPLAY=$(DISPLAY)" \
		-e "QT_X11_NO_MITSHM=1" \
		-v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
		-v $(PWD)/outputs:/opt/infinigen/outputs \
		-e "XAUTHORITY=$(XAUTH)" \
		-e ROS_IP=127.0.0.1 \
		--cap-add=SYS_PTRACE \
		-v /etc/group:/etc/group:ro \
		"$(DOCKER_TAG)" /bin/bash \
	|| docker run -td --privileged --net=host --ipc=host \
		--name="infinigen" \
		--device /dev/dri \
		-e "DISPLAY=$(DISPLAY)" \
		-e "QT_X11_NO_MITSHM=1" \
		-v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
		-v $(PWD)/outputs:/opt/infinigen/outputs \
		-e "XAUTHORITY=$(XAUTH)" \
		-e ROS_IP=127.0.0.1 \
		--cap-add=SYS_PTRACE \
		-v /etc/group:/etc/group:ro \
		"$(DOCKER_TAG)" bash


docker-run-no-opengl:
	echo "Launching Docker image without OpenGL ground truth"
	docker run -td --rm --privileged --net=host --ipc=host \
		--name="infinigen" \
		--gpus=all \
		--env NVIDIA_DISABLE_REQUIRE=1 \
		-v $(PWD)/outputs:/opt/infinigen/outputs \
		"$(DOCKER_TAG)" /bin/bash

docker-run-no-gpu:
	echo "Launching Docker image without GPU passthrough"
	docker run -td --privileged --net=host --ipc=host \
		--name="infinigen" \
		-e "DISPLAY=$(DISPLAY)" \
		-e "QT_X11_NO_MITSHM=1" \
		-v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
		-v $(PWD)/outputs:/opt/infinigen/outputs \
		-e "XAUTHORITY=$(XAUTH)" \
		-e ROS_IP=127.0.0.1 \
		--cap-add=SYS_PTRACE \
		-v /etc/group:/etc/group:ro \
		"$(DOCKER_TAG)" /bin/bash \
]

docker-run-no-gpu-opengl:
	echo "Launching Docker image without GPU passthrough or OpenGL"
	docker run -td --rm --privileged --net=host --ipc=host \
		--name="infinigen" \
		-e "BLENDER=/opt/infinigen/blender/blender" \
		-v $(PWD)/outputs:/opt/infinigen/outputs \
		"$(DOCKER_TAG)" /bin/bash
