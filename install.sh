#/usr/bin/env bash

# Python dependencies
echo "Install dependencies..."
apt-get update
apt-get install -y python-pip

# BlueROV2 related
apt-get install -y socat minicom ros-${ROS_DISTRO}-move-base ros-${ROS_DISTRO}-joy ros-${ROS_DISTRO}-cv-bridge
pip install mavproxy pymavlink
# Bruce related
pip install "numpy==1.16.5" "scipy==1.2.2" "matplotlib==2.2.4" catkin-tools "tqdm==4.30.0" "scikit-learn==0.20.4" shapely cython
apt-get install -y ros-${ROS_DISTRO}-nav-core ros-${ROS_DISTRO}-navfn ros-${ROS_DISTRO}-teb-local-planner ros-${ROS_DISTRO}-grid-map ros-${ROS_DISTRO}-move-base ros-${ROS_DISTRO}-move-base-msgs 

echo "Create workspace in current folder..."
mkdir -p bruce_ws/src
cd bruce_ws

source /opt/ros/${ROS_DISTRO}/setup.bash
catkin init
catkin config --merge-devel
catkin config --cmake-args -DCMAKE_BUILD_TYPE=Release

cd src
echo "Donwload packages..."

# Bluerov ROS package
git clone https://github.com/jinkunw/bluerov
git clone https://github.com/jinkunw/bruce

# GTSAM (note: latest code on github is not compatible)
git clone -b emex --single-branch https://bitbucket.com/jinkunw/gtsam
git clone https://github.com/pybind/pybind11.git bruce/bruce_slam/src/bruce_slam/cpp/pybind11

# Other libraries
git clone https://github.com/ethz-asl/libnabo.git
git clone https://github.com/ethz-asl/libpointmatcher.git
git clone https://github.com/jinkunw/rviz_satellite.git

echo "Build packages..."
catkin build
