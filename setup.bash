bash run_docker_container.sh
bash stop_docker_container.sh

docker exec -u root -it ros-noetic-container chown -R qiu:qiu /home/qiu

echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
echo "source ~/host/gem_simulation_ws/devel/setup.bash" >> ~/.bashrc


source /opt/ros/noetic/setup.bash
catkin_make
source devel/setup.bash

rosservice call /gazebo/reset_world

# stop sign testing
roslaunch gem_launch gem_init.launch world_name:="highbay_track.world" x:=12.5 y:=-21 yaw:=3.1416 custom_scene:=true
roslaunch gem_dwa_sim dwa_sim.launch goal.x:=-20.0 yaml_path:=$(rospack find gem_gazebo)/scenes/highbay_track.yaml
rostopic pub -r 20 /dwa_cmd ackermann_msgs/AckermannDrive "{speed: 1.5, steering_angle: 0.0}"
roslaunch gem_gazebo yolo_detector.launch
rosrun gem_gazebo stop_sign_behavior.py