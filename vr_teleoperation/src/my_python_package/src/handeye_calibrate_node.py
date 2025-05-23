#!/usr/bin/env python

import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped
from tf.transformations import quaternion_from_euler,quaternion_from_matrix
import struct
import socket
import receiver
import datetime
import numpy as np

def create_tf_xyz_szyx(x,y,z,rz,ry,rx,parent,child):
    tf_msg = TransformStamped()

    # Set the frame IDs
    tf_msg.header.frame_id = parent # Parent frame
    tf_msg.child_frame_id = child # Child frame

    # Set the initial position (x, y, z)
    tf_msg.transform.translation.x = x
    tf_msg.transform.translation.y = y
    tf_msg.transform.translation.z = z

    # Set the initial orientation (roll, pitch, yaw)
   
    quaternion = quaternion_from_euler(rz, ry, rx,'szyx')
    tf_msg.transform.rotation.x = quaternion[0]
    tf_msg.transform.rotation.y = quaternion[1]
    tf_msg.transform.rotation.z = quaternion[2]
    tf_msg.transform.rotation.w = quaternion[3]
    return tf_msg

def create_tf_xyz_quat(x,y,z,qw,qx,qy,qz,parent,child):
    tf_msg = TransformStamped()

    # Set the frame IDs
    tf_msg.header.frame_id = parent # Parent frame
    tf_msg.child_frame_id = child # Child frame

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




class CameraArmMsgPublisher:
    def __init__(self):
        self.start_time = datetime.datetime.now()
        self.end_time = datetime.datetime.now()
        rospy.init_node('CameraArmMsgPublisher')

        # Create a TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        # rospy.sleep(0.01)

        # Create a TransformStamped message
        
        # read tf


        self.tf_msg = create_tf_xyz_szyx(1,2,3,0,0,0,'map','camera_link')

        
        UDP_IP = "0.0.0.0"  # IP address to listen on
        UDP_PORT = 8006  # Port to listen on
        BUFFER_SIZE = 1024  # Buffer size for incoming messages

        # Create UDP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((UDP_IP, UDP_PORT))

        print("UDP receiver started")
        links=[]

        while not rospy.is_shutdown():
            
            # Receive message
            data, addr = sock.recvfrom(BUFFER_SIZE)
            if not data:
                break

            print(len(data))
            
            # camera tf message
            if len(data)==64:
                # Unpack the received data into a float array
                float_array = struct.unpack('16f', data)

                rt=np.array(float_array).reshape((4,4))
                q = quaternion_from_matrix(rt)

                link="link"+str(int(float_array[0]))
                tf_msg = create_tf_xyz_quat(rt[0,-1]/1000,rt[1,-1]/1000,rt[2,-1]/1000,q[3],q[0],q[1],q[2],'optical_origin',"optical_target")

                # Update the timestamp
                tf_msg.header.stamp = rospy.Time.now()

                # Publish the TF message
                self.tf_broadcaster.sendTransform(tf_msg)
            
            # arm tf message
            if len(data)==112:
                # Unpack the received data into a float array
                float_array = struct.unpack('14d', data)
                # print(float_array)
                float_array=np.array(float_array)
                # Update the timestamp
                # link="link"+str(int(float_array[0]))
                left_arm=float_array[:7]
                right_arm=float_array[7:]
                print(right_arm)
                # tf_msg = create_tf_xyz_quat(float_array[1]/1000,float_array[2]/1000,float_array[3]/1000,float_array[4],float_array[5],float_array[6],float_array[7],'map',link)
                tf_msg = create_tf_xyz_szyx(right_arm[3]/1000,right_arm[4]/1000,right_arm[5]/1000,
                right_arm[0],right_arm[1],right_arm[2],'base_link','ee_link')
            
                tf_msg.header.stamp = rospy.Time.now()

                # Publish the TF message
                self.tf_broadcaster.sendTransform(tf_msg)

            # print("Received float array:", float_array)

            # Close the socket
            # sock.close()
            # Update the timestamp
            # self.tf_msg.header.stamp = rospy.Time.now()
            

            # Publish the TF message
            # self.tf_broadcaster.sendTransform(self.tf_msg)

            # Sleep to maintain the publishing rate
            # self.tf_publish_rate.sleep()
            rospy.sleep(0.00)

    

if __name__ == '__main__':
    
        # TFPublisher()
    cam_arm_msg_publisher_node=CameraArmMsgPublisher()
        

