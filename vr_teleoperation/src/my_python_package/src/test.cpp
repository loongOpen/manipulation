#include <ros/ros.h>

int main(int argc, char **argv)
{
    // Initialize the ROS system
    ros::init(argc, argv, "hello_world_node");

    // Create a node handle
    ros::NodeHandle nh;

    // Print "Hello, World!" to the console
    ROS_INFO("Hello, World!");

    // Spin once to process callbacks (if any)
    ros::spinOnce();

    return 0;
}
