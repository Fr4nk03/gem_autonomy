FROM osrf/ros:noetic-desktop-full

ARG USER_UID=1000
ARG USER_GID=1000
ARG USER=$(whoami)
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get upgrade -y

RUN apt-get install -y \
    git \
    sudo \
    python3-pip \
    build-essential \
    cmake \
    pkg-config \
    libjpeg-dev \
    libtiff5-dev \
    libpng-dev \
    make \
    ros-noetic-ros-controllers \
    ros-noetic-gazebo-ros-control \
    ros-noetic-rosserial \
    ros-noetic-actionlib-tools \
    ros-noetic-ackermann-msgs \
    ros-noetic-geometry2 \
    ros-noetic-hector-gazebo \
    ros-noetic-hector-models \
    ros-noetic-jsk-rviz-plugins \
    ros-noetic-velodyne-simulator \
    ros-noetic-septentrio-gnss-driver \
    ros-noetic-novatel-gps-msgs \
    ros-noetic-nodelet \
    ros-noetic-nodelet-topic-tools \
    ros-noetic-jsk-topic-tools \
    ros-noetic-jsk-recognition-msgs \
    && rm -rf /var/lib/apt/lists/*

ARG USER=Fr4nk
ARG USER_UID=1000
ARG USER_GID=1000

# RUN groupadd -g ${USER_GID} ${USER} || groupmod -n ${USER} $(getent group ${USER_GID} | cut -d: -f1)
# RUN useradd -l -m -u ${USER_UID} -g ${USER_GID} ${USER} || usermod -l ${USER} -m -u ${USER_UID} -g ${USER_GID} $(getent passwd ${USER_UID} | cut -d: -f1)
# Hardcode 'Fr4nk' and the ID 1000 directly to bypass Windows environment issues
RUN groupadd -g 1000 Fr4nk || groupmod -n Fr4nk $(getent group 1000 | cut -d: -f1)
RUN useradd -l -m -u 1000 -g 1000 Fr4nk || usermod -l Fr4nk -m -u 1000 -g 1000 $(getent passwd 1000 | cut -d: -f1)
RUN echo "Fr4nk ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

RUN chown -R Fr4nk:Fr4nk /home/Fr4nk

RUN mkdir -p /usr/local/lib/python3.8/dist-packages \
    && chown -R Fr4nk:Fr4nk /usr/local/lib/python3.8/dist-packages/

RUN mkdir -p /catkin_ws/src && chown -R Fr4nk:Fr4nk /catkin_ws

USER root
RUN chown -R Fr4nk:Fr4nk /opt/ros/noetic

RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && cd /catkin_ws && catkin_make -DCMAKE_BUILD_TYPE=Release"

USER root
RUN chown -R Fr4nk:Fr4nk /home/Fr4nk
USER Fr4nk

# Add ROS setup to user's bashrc
RUN echo 'export PATH=$HOME/.local/bin:$PATH' >> /home/Fr4nk/.bashrc
RUN echo "source /opt/ros/noetic/setup.bash" >> /home/Fr4nk/.bashrc

WORKDIR /home/Fr4nk

# Set container entrypoint
ENTRYPOINT ["/bin/bash"]
