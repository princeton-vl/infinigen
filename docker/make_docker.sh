set -e

cp -av ~/.ssh/id_ed25519 .

echo "Removing infinigen docker image if already exists..."
docker rm -f infinigen 2> /dev/null
docker rmi -f infinigen_docker_img 2> /dev/null
docker build --tag infinigen_docker_img .

rm -fv id_ed25519

# UI permisions
if [ "$1" != "--noGPU" ]; then
    sudo apt-get install x11-xserver-utils
    touch ~/.Xauthority
    xauth add ${HOST}:0 . $(xxd -l 16 -p /dev/urandom)
    XSOCK=/tmp/.X11-unix
    XAUTH=/tmp/.docker.xauth
    touch $XAUTH
    xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -

    xhost +local:docker

        # Create a new container
    docker run -td --privileged --net=host --ipc=host \
        --name="infinigen" \
        --gpus=all \
        --env NVIDIA_DISABLE_REQUIRE=1 \
        -e "DISPLAY=$DISPLAY" \
        -e "QT_X11_NO_MITSHM=1" \
        -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
        -e "XAUTHORITY=$XAUTH" \
        -e ROS_IP=127.0.0.1 \
        --cap-add=SYS_PTRACE \
        -v /etc/group:/etc/group:ro \
        infinigen_docker_img bash || \
    docker run -td --privileged --net=host --ipc=host \
        --name="infinigen" \
        --device /dev/dri \
        -e "DISPLAY=$DISPLAY" \
        -e "QT_X11_NO_MITSHM=1" \
        -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
        -e "XAUTHORITY=$XAUTH" \
        -e ROS_IP=127.0.0.1 \
        --cap-add=SYS_PTRACE \
        -v /etc/group:/etc/group:ro \
        infinigen_docker_img bash
else
    docker run -td --privileged --net=host --ipc=host \
        --name="infinigen" \
        infinigen_docker_img bash
    echo "Created Docker image without GPU passthrough"
fi

