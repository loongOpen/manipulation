from transformations import quaternion_from_euler, euler_from_matrix,quaternion_matrix,quaternion_from_matrix,euler_matrix
import struct
from mocap2robot_src.HelpFuction import matrix3d_to_euler_angles_zyx, xyz_quaternion_to_homogeneous, rpy2rotation_matrix, rotation_matrix_to_rpy, \
    find_axis_angle, calc_dist


import struct
import numpy as np



from params import head_to_left_S,head_to_right_S

head_pose_new=np.eye(4)
head_pose_save=np.eye(4)


def lim_angle(angle):

    if angle <0:
        angle=0
    if angle>90:
        angle=90

    return int(angle)

def calibrate_shoudler():
     global head_pose_save
     head_pose_save=head_pose_new[:,:]

def get_hand_tf_quest_fixed_head(group_to_head_new,group_to_left_hand,group_to_right_hand):

    global head_pose_new

    group_to_head=np.eye(4)

    # group_to_head[0,3]=0 #left right offset
    # group_to_head[1,3]=0 # front back offset

    head_pose_new=group_to_head_new[:,:]
    # head_position_save=group_to_head[:3,3]

    head_r_zyx=matrix3d_to_euler_angles_zyx(group_to_head_new[:3,:3])
    head_r_rpy_new=[0,0,head_r_zyx[0]]
    head_r_new=rpy2rotation_matrix(*head_r_rpy_new)
    head_pose_new[:3,:3]=head_r_new

    group_to_head=head_pose_save[:,:] # if using floor level
    
    # group_to_head[2,3]=0 # if using eye level

    
    # head_to_left_S = np.array([[1, 0, 0, -0.2],
    #                         [0, 1, 0, -0.2],
    #                         [0, 0, 1, -0.2],
    #                         [0, 0, 0, 1]])

    # head_to_right_S = np.array([[1, 0, 0, 0.2],
    #                         [0, 1, 0, -0.2],
    #                         [0, 0, 1, -0.2],
    #                         [0, 0, 0, 1]])

    should_to_sun_left_should = np.array([[0.0000000,  0.0000000, -1.0000000, 0],
                            [0.0000000,  1.0000000,  0.0000000, 0],
                            [1.0000000,  0.0000000,  0.0000000, 0],
                            [0, 0, 0, 1]])


    should_to_sun_right_should = np.array([[0.0000000, -0.0000000,  1.0000000, 0],
                            [0.0000000, -1.0000000, -0.0000000, 0],
                            [1.0000000,  0.0000000, -0.0000000, 0],
                            [0, 0, 0, 1]])


    should_to_hand_left = fast_mat_inv(should_to_sun_left_should)@fast_mat_inv(head_to_left_S)@fast_mat_inv(group_to_head)@group_to_left_hand


    should_to_hand_right = fast_mat_inv(should_to_sun_right_should)@fast_mat_inv(head_to_right_S)@fast_mat_inv(group_to_head)@group_to_right_hand

    zyx_left=matrix3d_to_euler_angles_zyx(should_to_hand_left)
    zyx_right=matrix3d_to_euler_angles_zyx(should_to_hand_right)


    return should_to_hand_left,should_to_hand_right,zyx_left,zyx_right



def handle_raw_data(data_bytes):
     '''
     if l == 4 * 48 * 7:
                    ...
                    fun(data_bytes)
     '''
     float_array = struct.unpack(f'{48 * 7}f', data_bytes)
     xyzqwqxqyqz = np.array(float_array).reshape((48, 7))
     return xyzqwqxqyqz

ee_init_value_l=np.array([-1.5708, 1.5708, 0.0, -0.250 ,0.350, 0.05, 0.5233])
ee_init_value_r=np.array([1.5708, 1.5708, 0.0, -0.250 ,-0.3500, 0.05, -0.5233])

ee_init_rt_l=np.eye(4)
ee_init_rt_l=euler_matrix(*ee_init_value_l[:3],'szyx')
ee_init_rt_l[:3,-1]=ee_init_value_l[3:6]


ee_init_rt_r=np.eye(4)
ee_init_rt_r=euler_matrix(*ee_init_value_r[:3],'szyx')
ee_init_rt_r[:3,-1]=ee_init_value_r[3:6]

ee_cur_rt_l=ee_init_rt_l.copy()
ee_cur_rt_r=ee_init_rt_r.copy()

ee_last_rt_l=ee_init_rt_l.copy()
ee_last_rt_r=ee_init_rt_r.copy()

def fast_mat_inv(mat):
    ret = np.eye(4)
    ret[:3, :3] = mat[:3, :3].T
    ret[:3, 3] = -mat[:3, :3].T @ mat[:3, 3]
    return ret


def process(xyzqwqxqyqz):
    global ee_cur_rt_l,ee_cur_rt_r,ee_last_rt_l,ee_last_rt_r

    # if save_once:
    #     np.save('/home/jyw/posedata.npy',xyzqwqxqyqz)
    #     save_once=False

    cmd= xyzqwqxqyqz[1,:].copy()
    print(cmd)
    # 0 left grasp
    # 1 right grasp
    # 2 left move
    # 3 right move
    # 4 left reset
    # 5 right reset
    if cmd[2]==0:
        ee_cur_rt_l=ee_last_rt_l.copy()
    if cmd[3]==0:
        ee_cur_rt_r=ee_last_rt_r.copy()
        
    if cmd[4]==1:
        ee_cur_rt_l=ee_init_rt_l.copy()
        ee_last_rt_l=ee_init_rt_l.copy()
    if cmd[5]==1:
        ee_cur_rt_r=ee_init_rt_r.copy()
        ee_last_rt_r=ee_init_rt_r.copy()
    
    left_grasp=cmd[0]*50
    right_grasp=cmd[1]*50

    # unity left frame to right frame
    xyzqwqxqyqz[:,3]*=-1

    xyzqwqxqyqz=xyzqwqxqyqz[:,[0,2,1,3,4,6,5]]

    xyz=xyzqwqxqyqz[:,:3]
    # qwqxqyqz=xyzqwqxqyqz[:,3:]
    qxqyqzqw=xyzqwqxqyqz[:,[4,5,6,3]]

    rt_list=[]
    for i in range(48):
        if i==1:
            rt_list.append(np.eye(4))
        else:
            rt_base_quest_2_part_quest = quaternion_matrix(qxqyqzqw[i,:])
            rt_base_quest_2_part_quest[:3,-1]=xyz[i,:]
            rt_list.append(rt_base_quest_2_part_quest)
    

    rt_l_hand_quest_2_l_hand_robot=np.eye(4)
    # rt_l_hand_quest_2_l_hand_robot[:3,:3]=rpy2rotation_matrix(np.deg2rad(180),0,np.deg2rad(0))
    rt_l_hand_quest_2_l_hand_robot[:3,:3]=rpy2rotation_matrix(0,np.deg2rad(-90),0)

    rt_r_hand_quest_2_r_hand_robot=np.eye(4)
    # rt_r_hand_quest_2_r_hand_robot[:3,:3]=rpy2rotation_matrix(np.deg2rad(90),0,np.deg2rad(180))
    # rt_r_hand_quest_2_r_hand_robot[:3,:3]=rpy2rotation_matrix(np.deg2rad(0),np.deg2rad(180),0)
    rt_r_hand_quest_2_r_hand_robot[:3,:3]=rpy2rotation_matrix(np.deg2rad(180),np.deg2rad(-90),0)

    if cmd[2]>0:
        rt_l_ee1_quest_2_l_ee2_quest=rt_list[0]
        ee_new_rt_l=np.eye(4)
        delta_ee_l= fast_mat_inv(rt_l_hand_quest_2_l_hand_robot) @ rt_l_ee1_quest_2_l_ee2_quest @ rt_l_hand_quest_2_l_hand_robot

        # result correct but method not clear
        ee_new_rt_l[:3,-1] = ee_cur_rt_l[:3,-1] +delta_ee_l[:3,-1]
        ee_new_rt_l[:3,:3] = delta_ee_l[:3,:3] @ ee_cur_rt_l[:3,:3] 

        # update last
        ee_last_rt_l=ee_new_rt_l.copy()
    
    print(cmd[3])
    if cmd[3]>0:
        rt_r_ee1_quest_2_r_ee2_quest=rt_list[24+0]
        print(rt_r_ee1_quest_2_r_ee2_quest)
        ee_new_rt_r=np.eye(4)
        delta_ee_r= fast_mat_inv(rt_r_hand_quest_2_r_hand_robot) @ rt_r_ee1_quest_2_r_ee2_quest @ rt_r_hand_quest_2_r_hand_robot
        ee_new_rt_r[:3,-1] = ee_cur_rt_r[:3,-1] +delta_ee_r[:3,-1]
        ee_new_rt_r[:3,:3] = delta_ee_r[:3,:3] @ ee_cur_rt_r[:3,:3] 
        # update last
        ee_last_rt_r=ee_new_rt_r.copy()



    # print(ee_new_rt_l[:3,-1].flatten(),delta_ee_l[:3,-1].flatten())
    # print(ee_new_rt_r[:3,-1].flatten(),delta_ee_r[:3,-1].flatten())

    # here we use last result

    zyx_left_robot=matrix3d_to_euler_angles_zyx(ee_last_rt_l)
    zyx_right_robot=matrix3d_to_euler_angles_zyx(ee_last_rt_r)


    
    left_wrist_t=ee_last_rt_l[:3,-1]*1000
    right_wrist_t=ee_last_rt_r[:3,-1]*1000
    # print("left_theta_rad",self.left_theta_rad)
    # print(left_data,left_wrist_t)
    # print([zyx_left_robot,
    #         left_wrist_t,
    #         zyx_right_robot,
    #         right_wrist_t,
    #         left_grasp,0,0,0,0,0,
    #         right_grasp,0,0,0,0,0,])

    return [zyx_left_robot,
            left_wrist_t,
            zyx_right_robot,
            right_wrist_t,
            80,left_grasp,left_grasp,left_grasp,left_grasp,left_grasp,
            80,right_grasp,right_grasp,right_grasp,right_grasp,right_grasp,]