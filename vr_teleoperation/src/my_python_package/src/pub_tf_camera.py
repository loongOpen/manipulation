import datetime

import cv2
import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation
from tf.transformations import quaternion_from_euler, quaternion_from_matrix

from src.HelpFuction import rpy2rotation_matrix

def fast_mat_inv(mat):
    ret = np.eye(4)
    ret[:3, :3] = mat[:3, :3].T
    ret[:3, 3] = -mat[:3, :3].T @ mat[:3, 3]
    return ret
def create_tf_xyz_qxyzw(x, y, z, qx, qy, qz, qw, parent, child):
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

    tf_msg.header.stamp = rospy.Time.now()

    return tf_msg


class TFPublisher:
    def __init__(self):
        rospy.init_node('tf_publisher')
        # print("asdfagehbr")
        # Create a TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        # Create a TransformStamped message

        # read tf

        # self.tf_msg = create_tf_xyz_rpy(1,2,3,0,0,0,'map','camera_link')

        # Publish the TF message at a rate of 10 Hz
        self.tf_publish_rate = rospy.Rate(30)

        UDP_IP = "127.0.0.1"  # IP address to listen on
        UDP_PORT = 10000  # Port to listen on
        self.BUFFER_SIZE = 2888  # Buffer size for incoming messages

        self.left_theta_rad = 0.3
        self.right_theta_rad = 0.3
        self.start_time = datetime.datetime.now()

        self.data = {}
        self.data_len_f = 0
        while not rospy.is_shutdown():

            def arm(d):
                zyx=d[:3]
                t_vec=np.array(d[3:6])
                t_vec /= 1000

                R = np.eye(4)

                R[:3, :3] = Rotation.from_euler("zyx", zyx).as_matrix()
                q = quaternion_from_matrix(R)
                return t_vec,q

            d1 = [1.62603, 1.09267, -2.85307,-292.904, 246.681, 204.256]
            t_vec,q1= arm(d1)

            # tf_msg = create_tf_xyz_quat(float_array[1]/1000,float_array[2]/1000,float_array[3]/1000,float_array[4],float_array[5],float_array[6],float_array[7],'map',link)
            tf_msg = create_tf_xyz_qxyzw(*t_vec, *q1, 'map',
                                         'left')

            # Publish the TF message
            self.tf_broadcaster.sendTransform(tf_msg)

            c2s=np.array(
                [[-0.008719762646023366, -0.7001636870570678, -0.7139291120734955, 0.11116224604079081],
                 [0.0707962231144128, -0.7125969215639691, 0.6979924943509699, 0.15845845344952939],
                 [-0.9974526858614053, -0.0444571558267835, 0.05578262062382677, -0.1958717228683168],
                 [0.0, 0.0, 0.0, 1.0]]

            )
            # H_cam=fast_mat_inv(H_cam)

            q2 = quaternion_from_matrix(c2s)
            tf_msg = create_tf_xyz_qxyzw(*c2s[:3,-1], *q2, 'map',
                                         'cam_pred')
            self.tf_broadcaster.sendTransform(tf_msg)

            r = rpy2rotation_matrix(np.deg2rad(-90), np.deg2rad(90), np.deg2rad(45))

            s2c = np.array([[1.0, 0, 0, 0.120],
                            [0, 1, 0, 0.130],
                            [0, 0, 1, -0.200],
                            [0, 0, 0, 1]])
            s2c[:3, :3] = r

            q2 = quaternion_from_matrix(s2c)
            tf_msg = create_tf_xyz_qxyzw(*s2c[:3, -1], *q2, 'map',
                                         'camera_maybe')

            # Publish the TF message
            self.tf_broadcaster.sendTransform(tf_msg)

            t2c = np.array([[-9.96876880e-01, 7.89488135e-02, 1.88951885e-03, -2.00076357e+02/1000],
                            [5.47168142e-02, 7.07759518e-01, -7.04331268e-01, 7.90271676e+01/1000],
                            [-5.69434429e-02, -7.02028168e-01, -7.09868928e-01, 8.94103866e+02/1000],
                            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
            q2 = quaternion_from_matrix(t2c)
            tf_msg = create_tf_xyz_qxyzw(*t2c[:3, -1], *q2, 'cam_pred',
                                         'target')
            self.tf_broadcaster.sendTransform(tf_msg)



            tf_msg = create_tf_xyz_qxyzw(*t2c[:3, -1], *q2, 'camera_maybe',
                                         'target_maybe')
            self.tf_broadcaster.sendTransform(tf_msg)

            left1 = [1.57552, 0.455111, 2.19918, -173.716, 368.395, -83.9987, 1.15448]
            t_vec, q2 = arm(left1)

            tf_msg = create_tf_xyz_qxyzw(*t_vec, *q2, 'map',
                                         'left1')
            self.tf_broadcaster.sendTransform(tf_msg)

            e2s=np.array([[1.23971019e-02,  6.41634308e-01,  7.66910507e-01, - 2.98751426e+02/1000],
             [7.52166698e-01, - 5.11345415e-01,  4.15657461e-01,  3.35218670e+02/1000],
             [6.58856259e-01,  5.71691596e-01, - 4.88955160e-01,  9.15574857e+02/1000],
             [0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
            e2s=fast_mat_inv(e2s)
            q2 = quaternion_from_matrix(e2s)
            tf_msg = create_tf_xyz_qxyzw(*e2s[:3, -1], *q2, 'map',
                                         'end')
            self.tf_broadcaster.sendTransform(tf_msg)

            # eTt=np.array([[-0.6175570862272052, -0.43778012731962196, -0.6534307961632815, 0.0056024225506829395], [0.7864760312235004, -0.3343398801715716, -0.519299814016711, 0.13135329499486093], [0.008871164607645532, -0.8346049392681991, 0.5507775392911638, 0.5621401579590395], [0.0, 0.0, 0.0, 1.0]])
            # q2 = quaternion_from_matrix(eTt)
            #
            # tf_msg = create_tf_xyz_qxyzw(*eTt[:3, -1], *q2, 'left1',
            #                              'target_esimate')
            # self.tf_broadcaster.sendTransform(tf_msg)


            # rospy.sleep(0.00)
            self.tf_publish_rate.sleep()


if __name__ == '__main__':
    print("UDP start!")
    rec = TFPublisher()
