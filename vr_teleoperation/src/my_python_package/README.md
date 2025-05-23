# code controlling robot

# how to use
## ros part
ros neotic python 3.8
1. put the package in catkin_ws/src
2. run catkin_make
3. source devel/setup.bash 
4. roslaunch mocap2robot mocap_manager.launch  

**ros topic is affordable now**
## vr part
5. create new terminal
6. create new venv python 3.8
7. cd this_package/src
8. install numpy==1.24 and https://github.com/DanielBok/nlopt-python/tree/2.7.1.2 then install all requirements -> src/mocap2robot_src/TeleVision/requirements.txt
9. python src/vr_mocap_data_wrapper_udp_client.py
10. how to use quest -> src/mocap2robot_src/TeleVision/README.md


# deprecate
code path: src/my_python_package/src/pub_tf2.py

msg recv from vision pro and send to port 5015: src/my_python_package/src/other_file/streamer.py