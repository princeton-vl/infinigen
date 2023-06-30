DOCKER_BUILD_PROGRESS ?= auto
DOCKER_TAG ?= infinigen_docker_img

PWD = $(shell pwd)

XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth

default:

docker-build:
	git submodule init
	git submodule update
	docker build \
		--tag $(DOCKER_TAG) \
		--progress $(DOCKER_BUILD_PROGRESS) .

docker-build-cuda:
	git submodule init
	git submodule update
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
		-e "BLENDER=/opt/infinigen/blender/blender" \
		-e "DISPLAY=$(DISPLAY)" \
		-e "QT_X11_NO_MITSHM=1" \
		-v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
		-v $(PWD)/worldgen/outputs:/opt/infinigen/worldgen/outputs \
		-e "XAUTHORITY=$(XAUTH)" \
		-e ROS_IP=127.0.0.1 \
		--cap-add=SYS_PTRACE \
		-v /etc/group:/etc/group:ro \
		"$(DOCKER_TAG)" /bin/bash \
	|| docker run -td --privileged --net=host --ipc=host \
		--name="infinigen" \
		--device /dev/dri \
		-e "BLENDER=/opt/infinigen/blender/blender" \
		-e "DISPLAY=$(DISPLAY)" \
		-e "QT_X11_NO_MITSHM=1" \
		-v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
		-v $(PWD)/worldgen/outputs:/opt/infinigen/worldgen/outputs \
		-e "XAUTHORITY=$(XAUTH)" \
		-e ROS_IP=127.0.0.1 \
		--cap-add=SYS_PTRACE \
		-v /etc/group:/etc/group:ro \
		"$(DOCKER_TAG)" bash

	docker exec infinigen /bin/bash -c worldgen/tools/compile_opengl.sh

docker-run-no-opengl:
	echo "Launching Docker image without OpenGL ground truth"
	docker run -td --rm --privileged --net=host --ipc=host \
		--name="infinigen" \
		--gpus=all \
		--env NVIDIA_DISABLE_REQUIRE=1 \
		-e "BLENDER=/opt/infinigen/blender/blender" \
		-v $(PWD)/worldgen/outputs:/opt/infinigen/worldgen/outputs \
		"$(DOCKER_TAG)" /bin/bash

docker-run-no-gpu:
	echo "Launching Docker image without GPU passthrough"
	docker run -td --privileged --net=host --ipc=host \
		--name="infinigen" \
		-e "BLENDER=/opt/infinigen/blender/blender" \
		-e "DISPLAY=$(DISPLAY)" \
		-e "QT_X11_NO_MITSHM=1" \
		-v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
		-v $(PWD)/worldgen/outputs:/opt/infinigen/worldgen/outputs \
		-e "XAUTHORITY=$(XAUTH)" \
		-e ROS_IP=127.0.0.1 \
		--cap-add=SYS_PTRACE \
		-v /etc/group:/etc/group:ro \
		"$(DOCKER_TAG)" /bin/bash \

	docker exec infinigen /bin/bash -c worldgen/tools/compile_opengl.sh

docker-run-no-gpu-opengl:
	echo "Launching Docker image without GPU passthrough or OpenGL"
	docker run -td --rm --privileged --net=host --ipc=host \
		--name="infinigen" \
		-e "BLENDER=/opt/infinigen/blender/blender" \
		-v $(PWD)/worldgen/outputs:/opt/infinigen/worldgen/outputs \
		"$(DOCKER_TAG)" /bin/bash
	