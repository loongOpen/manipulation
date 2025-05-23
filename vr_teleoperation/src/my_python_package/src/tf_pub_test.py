#!/usr/bin/env python

import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped

import struct


import datetime
import numpy as np



def create_tf_xyz_quat(x, y, z, qw, qx, qy, qz, parent, child):
    tf_msg = TransformStamped()

    # Set the frame IDs
    tf_msg.header.frame_id = parent  # Parent frame
    tf_msg.child_frame_id = child  # Child frame

    # Set the initial position (x, y, z)
    tf_msg.transform.translation.x = x
    tf_msg.transform.translation.y = y
    tf_msg.transform.translation.z = z

    # Set the initial orientation (roll, pitch, yaw)

    tf_msg.transform.rotation.x = qx
    tf_msg.transform.rotation.y = qy
    tf_msg.transform.rotation.z = qz
    tf_msg.transform.rotation.w = qw
    return tf_msg


class TfPubTestNode:
    def __init__(self):
        rospy.init_node('tf_pub_test')

        # Create a TF broadcaster
        # self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.pub = rospy.Publisher('custom_transform_topic', TransformStamped, queue_size=10)

        # Create a TransformStamped message

        # read tf

        # self.tf_msg = create_tf_xyz_rpy(1, 2, 3, 0, 0, 0, 'map', 'camera_link')

        # Publish the TF message at a rate of 10 Hz
        self.tf_publish_rate = rospy.Rate(60)

    def publish_tf(self):

        while not rospy.is_shutdown():

                    # tf_msg = create_tf_xyz_quat(float_array[1]/1000,float_array[2]/1000,float_array[3]/1000,float_array[4],float_array[5],float_array[6],float_array[7],'map',link)
            tf_msg = create_tf_xyz_quat(0,0,0,
                                       1,0, 0,0, 'map', '1')

            tf_msg.header.stamp = rospy.Time.now()

            # Publish the TF message
            # self.tf_broadcaster.sendTransform(tf_msg)
            self.pub.publish(tf_msg)


            # Sleep to maintain the publishing rate
            self.tf_publish_rate.sleep()


if __name__ == '__main__':
    # TFPublisher()
    my_node = TfPubTestNode()
    my_node.publish_tf()


