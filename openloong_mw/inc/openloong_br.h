#ifndef _OPENLOONG_BR_H_
#define _OPENLOONG_BR_H_
#include <ros/ros.h>
#include <ros/package.h>
#include <iostream>
#include <termio.h>
#include <stdio.h>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <sys/types.h>   
#include <sys/socket.h>   
#include <netinet/in.h>   
#include <arpa/inet.h> 
#include <fcntl.h> 
#include <unordered_set>
#include <mutex>
#include <algorithm>
#include <string>
#include <chrono>
#include <thread>
#include <math.h>
#include <yaml-cpp/yaml.h>
#include <stdint.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h> 
#include <tuple>
#include "std_msgs/Float32.h"
#include "std_msgs/Bool.h"
#include "std_msgs/String.h"
#include "openloong_mw/ObjectInfo.h"
#include "openloong_mw/ObjectInfoList.h"
#include "openloong_mw/ObjNameAndInt.h"
#include "openloong_mw/ControlMessage.h"


class openloong_br {
public:
    static const int LIMIT_MIN = 0;
    static const int LIMIT_MAX = 100;
    static const int GET_TIME_NUM = 50;
    static const int KEY_ = 32;
    static const int KEY_1 = 49;
    static const int KEY_2 = 50;
    static const int KEY_3 = 51;
    static const int KEY_4 = 52;
    static const int KEY_5 = 53;
    static const int KEY_6 = 54;
    static const int KEY_7 = 55;
    static const int KEY_8 = 56;
    static const int KEY_9 = 57;
    static const int KEY_0 = 48;
    static const bool TEST_MODE = true;
    // 静态函数模拟宏
    static int LIMIT(int x, int min, int max) {
        return (x < min) ? min : ((x > max) ? max : x);
    }
    void Thread_KEY();
    void Thread_ROS();
    void Thread_UDP_ROBOT();
    void Thread_UDP_Float();
    // 启动所有线程
    void startAllThreads();
    // 等待所有线程结束
    void joinAllThreads();

	openloong_br();
	~openloong_br();
private:

    enum
    {
        NOW = 0,
        OLD,
        NEW,
        DT_LAST
    };

    std::mutex msg_mutex;

    typedef struct
    {
        int clss;
        int x;
        int y;
        int width;
        int height;
        float depth;
        float conf;
    }BOUDING_BOX;
    typedef struct
    {
        float x;
        float y;
        float z;
        float zz;
    }END_POS;

    struct ObjectDetails {
        int id;
        std::string name;
        int side;
        END_POS pose_left;
        END_POS pose_left_app;
        END_POS pose_left_app_s;
        END_POS att_left;
        END_POS att_left_app;
        END_POS att_left_app_s;
        END_POS pose_right;
        END_POS pose_right_app;
        END_POS pose_right_app_s;
        END_POS att_right;
        END_POS att_right_app;
        END_POS att_right_app_s;
        int clss;
        int16_t x, y, width, height;
        float conf;
        float depth;  // 如果有深度信息
    };

#pragma pack(push, 1)
struct remote_msg_end {
    uint8_t  head;                  // 0xAA = aloha, 0xBB = quest
    uint8_t  mode;                  // 0 = joint, 1 = epos
    float base_vel[3];
    float arm_pos_exp_l[3];
    float arm_att_exp_l[3];
    float arm_q_exp_l[7];
    float cap_l;
    float arm_pos_exp_r[3];
    float arm_att_exp_r[3];
    float arm_q_exp_r[7];
    float cap_r;
    float waist_exp;
    float head_exp[3];
    float flt_rate;
};
#pragma pack(pop)

    typedef struct{
        int obj_num;
        int obj_state[10];
        int obj_id[10];
        int obj_side[10];
        END_POS obj_pos_from_left[10];
        END_POS obj_att_from_left[10];
        END_POS wrist_pos_from_left[10];
        END_POS wrist_att_from_left[10];
        END_POS imminent_pos_from_left[10];
        END_POS imminent_att_from_left[10];
        END_POS obj_pos_from_right[10];
        END_POS obj_att_from_right[10];
        END_POS wrist_pos_from_right[10];
        END_POS wrist_att_from_right[10];
        END_POS imminent_pos_from_right[10];
        END_POS imminent_att_from_right[10];
        BOUDING_BOX obj_box[10];
        std::string obj_name[10];
    }ENV;
    typedef struct
    {
        int lsm_main;
        int arm_control_mode;
        int reset_arm;
        int get_grasp_pose=0;
        END_POS arm_epos_h_exp_safe[2];
        END_POS arm_att_h_exp_safe[2];
        float cap_set_safe[2];
        //----------cmd
        END_POS arm_epos_h_exp[2];
        END_POS arm_att_h_exp[2];

        END_POS arm_epos_h_exp_flt[2];
        END_POS arm_att_h_exp_flt[2];

        float cap_set[2];
        float cap_set_flt[2];
        //-----------grasp
        int max_grasp_num;
        int grasp_cnt;
        int grasp_mission[2];
        int grasp_action[2];
        int grasp_try_flag[2];

        END_POS arm_epos_h_grasp;//convert to grasp arm's hip by Tf
        END_POS arm_att_h_grasp;
        END_POS arm_epos_h_grasp_app;
        END_POS arm_att_h_grasp_app;

        END_POS arm_epos_h_grasp_off;
        END_POS arm_epos_h_grasp_gpd_off;

        END_POS arm_epos_h_grasp_right;//convert to grasp arm's hip by Tf
        END_POS arm_att_h_grasp_right;
        END_POS arm_epos_h_grasp_app_right;
        END_POS arm_att_h_grasp_app_right;

        END_POS arm_epos_h_grasp_off_right;
        END_POS arm_epos_h_grasp_gpd_off_right;
        
        ENV _env;
    }robotTypeDef;
    struct Package_Socket_Arm//send to robot
    {
        long time_stamp=0;
        int arm_mode;
        float ee_pose_l[6];
        float ee_pose_r[6];
        float q_exp_l[7];
        float q_exp_r[7];
        float cap_set[2];
        float hand_q[2][6];
    };
    struct Package_Socket_Waist_Head//send to robot
    {
        float waist_q_exp_flt[3];
        float head_q_exp_flt[2];
        float waist_q_exp[3];
        float head_q_exp[2];
    };
    struct Hand_Force//send to robot
    {
        float left_hand_force[5];
        float right_hand_force[5];
    };

    struct Action {
        std::string action;
        bool com_left;
        bool com_right;
        bool eyes;
        bool arm_bezier;
        bool waist_bezier;
        float speed;
        float sleep;
    };
    remote_msg_end send_msg_pose;
    struct Actions {
        std::vector<std::vector<float>> pose;
        std::vector<std::vector<float>> older;
        bool com_left;
        bool com_right;
        bool eyes;
        bool arm_bezier;
        bool waist_bezier;
        float speed;
        float sleep;
    };
    struct Tasks {
        std::string task_name;
        std::vector<Actions> task_config_list;
    };

    Actions task_config;
    Tasks small_task_list;
    Tasks* found_task;
    // Actions& action_store;
    std::vector<Tasks> big_task_list;
    std::filesystem::path filePath;

    std::string package_path, custom_task_name, left_none, right_none, extension, action_name, txt_line,
                base_path, base_path_pose, base_path_older;
    // 用于存储所有动作的 vector
    std::vector<Action> actions_list;
    Action action;
    YAML::Node config;

    ros::NodeHandle* nh1, *nh2;
    ros::Publisher pub;
    ros::Subscriber sub_obj_list, Message;
    bool task_arm_bezier, task_waist_bezier;


    volatile float Cycle_T[GET_TIME_NUM][4];
    struct timeval tv;
    std::vector<ObjectDetails> robotwb_env_left, robotwb_env_right;
    Package_Socket_Arm _send_msg_arm;
    Hand_Force hand_force;
    Package_Socket_Waist_Head Waist_Head;
    robotTypeDef robotwb;
    std::thread thread_key, thread_ros, thread_float, thread_send;
    
    Eigen::Vector3d position_to_max_xyz, position_to_max_rpy, max_to_position_taget_right_p,
                    max_to_position_taget_right_r, quet_to_max_quet_p, max_to_quet_translation,
                    bezier_original_LPose0, bezier_original_LPose1, bezier_original_LRot0, bezier_original_LRot1,
                    bezier_original_RPose0, bezier_original_RPose1, bezier_original_RRot0, bezier_original_RRot1,
                    LinterpolatedPosition, RinterpolatedPosition, LinterpolatedRxyz, RinterpolatedRxyz,
                    bezier_dan_Pose0, bezier_dan_Pose1, bezier_dan_Rot0, bezier_dan_Rot1,
                    bezier_dan_interpolatedPosition, bezier_dan_interpolatedRxyz;

    Eigen::Vector4d quet_to_max_quet_q;

    Eigen::VectorXd exp_pose_left, exp_pose_right, exp_pose_waist, current_pose_left, current_pose_right, 
                    current_waist_pose, bezier_arm_pose_left, bezier_arm_pose_right, bezier_waist_pose,
                    bezier_dan_arm_pose, bezierInterpolationEEF_arm_pose;

    Eigen::Matrix3d LinterpolatedRotationMatrix, RinterpolatedRotationMatrix, rotation_to_homogeneous_Matrix,
                    bezier_dan_interpolatedRotationMatrix;


    Eigen::Matrix4d bezier_original_Lmax0, bezier_original_Lmax1, bezier_original_Rmax0, bezier_original_Rmax1,
                    max_old_robot_to_erweima_left, qu2_left, qu_new2_left, old_to_new2_left,
                    max_old_robot_to_erweima_right, qu2_right, qu_new2_right, old_to_new2_right,
                    bezier_dan_max0, bezier_dan_max1, mat_quat, HomogeneousMatrix, S_E,
                    point_bianhuan_guihua, point_bianhuan_guihua_dan;


    double w_quat, x_quat, y_quat, z_quat, beta_y, alpha_z, gamma_x, r_dt, dur, start_time, end_time, tel;

    Eigen::Quaterniond bezier_original_LQ0, bezier_original_LQ1, bezier_original_RQ0, bezier_original_RQ1, 
                       LinterpolatedOrientation, RinterpolatedOrientation, max_to_quet_quaternion,
                       bezier_dan_Q0, bezier_dan_Q1, bezier_dan_interpolatedOrientation;
    
    std::vector<float> erweima_left, erweima_right, old_robot_to_erweima_left, old_robot_to_erweima_right,
                       erweima_left_new, erweima_right_new, old_robot_to_erweima_left_new,
                       old_robot_to_erweima_right_new, max_to_position_positions, max_to_quet_result,
                       new_er_left, new_er_right, waist_point, pos_guihua_dan, test_guihua_dan, txt_row,
                       combined_row, pos_left_guihua, pos_right_guihua, test_left_guihua, test_right_guihua;

    std::vector<std::vector<float>> target_left_up_new, target_right_up_new, target_new_guihua, 
                                    target_pose_dan_new, txt_target_data, left_table, right_table,
                                    target_pose, order, target_pose_left, target_pose_right, target_pose_waist,
                                    target_pose_new_left, target_pose_new_right;

    int batch_size, total_rows, processed_rows, current_batch_size, init,
        in_scanKeyboard, key_updateKey, force_exit, lsm_main, key, get_grasp_pose;

    std::unordered_set<std::string> hamburger_namesToKeep, bread_namesToKeep;
    ObjectDetails firstObject;

    float wai_hea[5], task_speed, timer_grasp, txt_value;

    std::ifstream inputFile; 
    std::stringstream txt_ss;
    char comma_remove;  // 用于跳过逗号

    struct termios new_settings;
    struct termios stored_settings;
	uint32_t GetSysTime_us_ms, GetSysTime_us_value;
    
    int updateKey(void);
    void classifyAndSortObjects_left();
    void classifyAndSortObjects_right();
    void calculate_trajectory_config(const Actions task_config);
    Tasks* findTaskByName(std::vector<Tasks>& big_task_list, const std::string& task_name);
    float Get_Cycle_T(int item);
    void Cycle_Time_Init();
    void param_init(void);
    int scanKeyboard();
    uint32_t GetSysTime_us(void);
    float Get_T_Trig(void);
    int containsName_left(const std::string& nameToFind);
    int containsName_right(const std::string& nameToFind);
    static void Perror(const char *s);
    static void setnonblocking(int sockfd);
    void keepSpecificObjectName_left(const std::string& specificName);
    void keepSpecificObjectName_right(const std::string& specificName);
    void keepSpecificObjectNames_left_duo(const std::unordered_set<std::string>& names);
    void keepSpecificObjectNames_right_duo(const std::unordered_set<std::string>& names);
    void waesdfrgthObjectDetails(const std::vector<ObjectDetails>& objects, const std::string& side);

    void udpSendData(const float* data, const std::string& ip, int port);
    std::vector<std::vector<float>> guihua(const Eigen::Matrix4d& binahuan, 
    const Eigen::Matrix4d& old_robot_to_erweima, std::vector<std::vector<float>> target_pose);
    void find_yaml(const std::string& directoryPath);
    void read_yaml_config(const std::string& name);
    void objectInfoListCallback(const openloong_mw::ObjectInfoList::ConstPtr& msg);
    void messageCallback(const openloong_mw::ControlMessage::ConstPtr& msg);
    void grasp_mission_auto_bofang(float dt, const std::string& task_name);
    void grasp_mission_auto_bofang_interaction(float dt, const std::string& task_name);
    void grasp_mission_auto_bofang_alone(float dt, const std::string& task_name);
    float bezierInterpolationwaist(const float& P0, const float& P1, double t);
    void state_machine_main(float dt);
    std::vector<std::vector<float>> guihua_dan(const Eigen::Matrix4d& binahuan, const Eigen::Matrix4d& old_robot_to_erweima, std::vector<std::vector<float>> target_pose);
    std::vector<std::vector<float>> readDataFromFile(const std::string& filename);
    Eigen::VectorXd bezierInterpolationEEF(const Eigen::VectorXd& P0, const Eigen::VectorXd& P1, double t);
    Eigen::VectorXd bezierInterpolationEEF_dan(const Eigen::VectorXd& P0, const Eigen::VectorXd& P1, double t);
    std::vector<float> bianhua(const Eigen::Matrix4d& S_E, const std::vector<float>& position);
    Eigen::VectorXd bezierInterpolationJoints(const Eigen::VectorXd& P0, const Eigen::VectorXd& P1, double t);
    Eigen::Vector3d matrix3dToEulerAnglesZYX(const Eigen::Matrix3d& m3dr);
    Eigen::Matrix4d quaternionToMatrix(const Eigen::Vector4d& quat);
    Eigen::Matrix4d poseToHomogeneousMatrix(const Eigen::Vector3d& translation, const Eigen::Vector3d& euler_angles);
    Eigen::Matrix4d position_to_max(const std::vector<float>& position);
    Eigen::VectorXd line_angles(const Eigen::VectorXd& angle);
    std::vector<float> max_to_position(const Eigen::Matrix4d& max);
    Eigen::Matrix4d quet_to_max(const std::vector<float>& quet);
    std::vector<float> max_to_quet(const Eigen::Matrix4d& matrix);
    void grasp_mission_auto_bofang_dps(float dt, const std::string& task_name);
};

 #endif  