cmake_minimum_required(VERSION 2.8.3)

project(openloong_mw)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(catkin REQUIRED)


find_package(catkin REQUIRED COMPONENTS
joint_state_publisher
robot_state_publisher
roscpp
rospy
rviz
xacro
tf
sensor_msgs
std_msgs 
message_generation
)
find_package(HDF5 REQUIRED COMPONENTS CXX)
find_package(Boost REQUIRED COMPONENTS date_time system thread)
find_package(yaml-cpp REQUIRED)
add_message_files(
  FILES
  ObjectInfo.msg
  ObjectInfoList.msg
  ObjNameAndInt.msg
  ControlMessage.msg
)
include_directories(${PROJECT_SOURCE_DIR}/external_include)

## 查找 Boost 库，包含 filesystem 和 system 组件
find_package(Boost REQUIRED COMPONENTS filesystem system)

if(Boost_FOUND)
  include_directories(
    ${Boost_INCLUDE_DIRS}
  )
endif()


generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)
 
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES demo
CATKIN_DEPENDS controller_manager message_runtime std_msgs geometry_msgs gazebo_ros gazebo_ros_control joint_state_publisher robot_state_publisher roscpp rospy rviz xacro
DEPENDS Boost 
#  DEPENDS system_lib
)

# set_target_properties(${PROJECT_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/${PROJECT_NAME}/lib)


include_directories(
  include ${catkin_INCLUDE_DIRS}
  include ${CMAKE_CURRENT_SOURCE_DIR}/inc/
  include ${CMAKE_CURRENT_SOURCE_DIR}/gen-cpp/
  ${HDF5_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
)

# 设置库的输出路径
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)

# 设置为共享库而不是可执行文件
add_library(open_br SHARED
    src/openloong_br.cpp
    src/comm/key_come.cpp
    src/comm/sys_timer_come.cpp
    src/comm/calculate.cpp
    src/comm/Information.cpp
    src/comm/mission.cpp
    src/comm/trajectory.cpp    
)

# 添加依赖（如果有的话）
add_dependencies(open_br ${catkin_EXPORTED_TARGETS})

# 链接其他库 libanother.so
target_link_libraries(open_br
  ${catkin_LIBRARIES}
  yaml-cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/libk_OLR2.so  # 这里替换为 libanother.so 的路径
)
# 安装共享库到指定路径
install(TARGETS open_br
    LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/lib
)

add_executable(openloongIf_br
src/openloongIf_br.cpp
)

add_dependencies(openloongIf_br  ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(openloongIf_br
${catkin_LIBRARIES} 
yaml-cpp
${CMAKE_CURRENT_SOURCE_DIR}/lib/libopen_br.so
${CMAKE_CURRENT_SOURCE_DIR}/lib/libsdk.so
)

foreach(dir config launch meshes urdf)
	install(DIRECTORY ${dir}/
		DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/${dir})
endforeach(dir)
