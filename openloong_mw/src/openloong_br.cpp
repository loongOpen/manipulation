#include "openloong_br.h"
#include "ArmController.h"
#include "TorsoController.h"

openloong_br::openloong_br()
    : exp_pose_left(6),               // 初始化为长度为6的动态向量
      exp_pose_right(6),
      exp_pose_waist(3),
      current_pose_left(6),
      current_pose_right(6),
      current_waist_pose(3),
      bezier_arm_pose_left(6),
      bezier_arm_pose_right(6),
      bezier_waist_pose(3),
      max_old_robot_to_erweima_left(Eigen::Matrix4d::Zero()), // 初始化为零矩阵
      qu2_left(Eigen::Matrix4d::Zero()),
      qu_new2_left(Eigen::Matrix4d::Zero()),
      old_to_new2_left(Eigen::Matrix4d::Zero()),
      max_old_robot_to_erweima_right(Eigen::Matrix4d::Zero()),
      qu2_right(Eigen::Matrix4d::Zero()),
      qu_new2_right(Eigen::Matrix4d::Zero()),
      old_to_new2_right(Eigen::Matrix4d::Zero()),
      new_er_left(6, 0.0f),           // 初始化为长度6，所有元素为0.0f
      new_er_right(6, 0.0f),
      waist_point(3, 0.0f),
      max_to_position_positions(6, 0.0f),
      bezierInterpolationEEF_arm_pose(12),
      bezier_dan_arm_pose(6)
{
    SDK& sdkInstance = SDK::createInstance();
	nh1 = new ros::NodeHandle();
	nh2 = new ros::NodeHandle();
	pub = nh1->advertise<std_msgs::Bool>("/arm_msg", 1);
	sub_obj_list = nh2->subscribe("/processed_pose_list", 1, &openloong_br::objectInfoListCallback, this);
    Message = nh2->subscribe("/control_msg", 1, &openloong_br::messageCallback, this);
    package_path = ros::package::getPath("openloong_mw");
    std::string base_path       = package_path + "/config/yaml";
    std::string base_path_pose  = package_path + "/config/pose";
    std::string base_path_older = package_path + "/config/older";
    force_exit=0;
    lsm_main=0;
    key=0;
    r_dt = 0.005;
    find_yaml(base_path);
    startAllThreads();
    
    
}

openloong_br::~openloong_br() {
    joinAllThreads(); 
	delete nh1;
	delete nh2;
}

void openloong_br::Perror(const char *s)
{
    perror(s);
    exit(EXIT_FAILURE);
}

void openloong_br::setnonblocking(int sockfd) {
    int flag = fcntl(sockfd, F_GETFL, 0);
    if (flag < 0) {
        Perror("fcntl F_GETFL fail");
        return;
    }
    if (fcntl(sockfd, F_SETFL, flag | O_NONBLOCK) < 0) {
        Perror("fcntl F_SETFL fail");
    }
}

void openloong_br::messageCallback(const openloong_mw::ControlMessage::ConstPtr& msg)
{
    if (msg->code_ready)
    {
        if ((msg->action == "reset_one") || (msg->action == "reset_two") || (msg->action == "up") || (msg->action == "down"))
        {
            custom_task_name = msg->action;
            lsm_main = 75;
        }
        else
        {
            lsm_main = 100;
        }
        
    }
    
}

void openloong_br::objectInfoListCallback(const openloong_mw::ObjectInfoList::ConstPtr& msg)
{

    for(int i=0;i<10;i++){
        if(robotwb._env.obj_state[i]!=2){
            robotwb._env.obj_state[i]=0;
            robotwb._env.obj_name[i]="null";
        }
    }
    robotwb._env.obj_num=0;
    for(int i=0;i<msg->objects.size();i++){
        for(int j=0;j<10;j++){
            if(robotwb._env.obj_state[j]==0){
                robotwb._env.obj_name[j]=msg->objects[i].name.c_str();
                robotwb._env.obj_id[j]=j;
                robotwb._env.obj_state[j]=1;//set check flag

                robotwb._env.obj_pos_from_left[j].x=msg->objects[i].pose_left.position.x;
                robotwb._env.obj_pos_from_left[j].y=msg->objects[i].pose_left.position.y;
                robotwb._env.obj_pos_from_left[j].z=msg->objects[i].pose_left.position.z;

                robotwb._env.obj_att_from_left[j].x=msg->objects[i].pose_left.orientation.x;
                robotwb._env.obj_att_from_left[j].y=msg->objects[i].pose_left.orientation.y;
                robotwb._env.obj_att_from_left[j].z=msg->objects[i].pose_left.orientation.z;

                robotwb._env.wrist_pos_from_left[j].x=msg->objects[i].pose_left_app.position.x;
                robotwb._env.wrist_pos_from_left[j].y=msg->objects[i].pose_left_app.position.y;
                robotwb._env.wrist_pos_from_left[j].z=msg->objects[i].pose_left_app.position.z;

                robotwb._env.wrist_att_from_left[j].x=msg->objects[i].pose_left_app.orientation.x;
                robotwb._env.wrist_att_from_left[j].y=msg->objects[i].pose_left_app.orientation.y;
                robotwb._env.wrist_att_from_left[j].z=msg->objects[i].pose_left_app.orientation.z;

                robotwb._env.imminent_pos_from_left[j].x=msg->objects[i].pose_left_app_s.position.x;
                robotwb._env.imminent_pos_from_left[j].y=msg->objects[i].pose_left_app_s.position.y;
                robotwb._env.imminent_pos_from_left[j].z=msg->objects[i].pose_left_app_s.position.z;

                robotwb._env.imminent_att_from_left[j].x=msg->objects[i].pose_left_app_s.orientation.x;
                robotwb._env.imminent_att_from_left[j].y=msg->objects[i].pose_left_app_s.orientation.y;
                robotwb._env.imminent_att_from_left[j].z=msg->objects[i].pose_left_app_s.orientation.z;

                robotwb._env.obj_pos_from_right[j].x=msg->objects[i].pose_right.position.x;
                robotwb._env.obj_pos_from_right[j].y=msg->objects[i].pose_right.position.y;
                robotwb._env.obj_pos_from_right[j].z=msg->objects[i].pose_right.position.z;

                robotwb._env.obj_att_from_right[j].x=msg->objects[i].pose_right.orientation.x;
                robotwb._env.obj_att_from_right[j].y=msg->objects[i].pose_right.orientation.y;
                robotwb._env.obj_att_from_right[j].z=msg->objects[i].pose_right.orientation.z;

                robotwb._env.wrist_pos_from_right[j].x=msg->objects[i].pose_right_app.position.x;
                robotwb._env.wrist_pos_from_right[j].y=msg->objects[i].pose_right_app.position.y;
                robotwb._env.wrist_pos_from_right[j].z=msg->objects[i].pose_right_app.position.z;

                robotwb._env.wrist_att_from_right[j].x=msg->objects[i].pose_right_app.orientation.x;
                robotwb._env.wrist_att_from_right[j].y=msg->objects[i].pose_right_app.orientation.y;
                robotwb._env.wrist_att_from_right[j].z=msg->objects[i].pose_right_app.orientation.z;

                robotwb._env.imminent_pos_from_right[j].x=msg->objects[i].pose_right_app_s.position.x;
                robotwb._env.imminent_pos_from_right[j].y=msg->objects[i].pose_right_app_s.position.y;
                robotwb._env.imminent_pos_from_right[j].z=msg->objects[i].pose_right_app_s.position.z;

                robotwb._env.imminent_att_from_right[j].x=msg->objects[i].pose_right_app_s.orientation.x;
                robotwb._env.imminent_att_from_right[j].y=msg->objects[i].pose_right_app_s.orientation.y;
                robotwb._env.imminent_att_from_right[j].z=msg->objects[i].pose_right_app_s.orientation.z;


                robotwb._env.obj_side[j]=1;
                robotwb._env.obj_box[j].clss=msg->objects[i].clss;//
                robotwb._env.obj_box[j].x=msg->objects[i].x1;
                robotwb._env.obj_box[j].y=msg->objects[i].y1;
                robotwb._env.obj_box[j].width=msg->objects[i].width;
                robotwb._env.obj_box[j].height=msg->objects[i].height;
                robotwb._env.obj_box[j].conf=msg->objects[i].conf;


#if 0
                ROS_INFO("  obj_state %lu:", robotwb._env.obj_state[j] );
                ROS_INFO("  Object %lu:", i );
                ROS_INFO("  Name: %s", msg->objects[i].name.c_str());
                ROS_INFO("  Position: [%f, %f, %f]", msg->objects[i].pose.position.x, msg->objects[i].pose.position.y, msg->objects[i].pose.position.z);
                ROS_INFO("  Orientation: [%f, %f, %f, %f]", msg->objects[i].pose.orientation.x, msg->objects[i].pose.orientation.y, msg->objects[i].pose.orientation.z, msg->objects[i].pose.orientation.w);
#endif
                break;
            }
        }     
    }
 
    for(int i=0;i<10;i++){
        if(robotwb._env.obj_state[i]!=0){
            robotwb._env.obj_num++;
        }
    }
   
}

// 线程函数，接收 UDP 数据
void openloong_br::Thread_UDP_Float() {  // 这里的参数不使用
    int sockfd;
    struct sockaddr_in server_addr, client_addr;
    socklen_t addr_len = sizeof(client_addr);

    // 创建一个 UDP 套接字
    sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) {
        std::cerr << "Socket creation failed" << std::endl;

    }

    // 设置服务器地址
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(9091);  // 假设监听端口 9091

    // 绑定套接字到端口
    if (bind(sockfd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        std::cerr << "Bind failed" << std::endl;
        close(sockfd);

    }

    float data[15];  // 用于存储接收到的浮点数数据

    while (true) {
        // 接收 UDP 数据
        int n = recvfrom(sockfd, data, sizeof(data), 0, (struct sockaddr*)&client_addr, &addr_len);
        if (n < 0) {
            std::cerr << "Receive failed" << std::endl;
            break;
        }
            wai_hea[0] = data[0];
            wai_hea[1] = data[1];
            wai_hea[2] = data[2];
            wai_hea[3] = data[3];
            wai_hea[4] = data[4];
            hand_force.left_hand_force[0] = data[5];
            hand_force.left_hand_force[1] = data[6];
            hand_force.left_hand_force[2] = data[7];
            hand_force.left_hand_force[3] = data[8];
            hand_force.left_hand_force[4] = data[9];
            hand_force.right_hand_force[0] = data[10];
            hand_force.right_hand_force[1] = data[11];
            hand_force.right_hand_force[2] = data[12];
            hand_force.right_hand_force[3] = data[13];
            hand_force.right_hand_force[4] = data[14];
    }
    close(sockfd);
}
void openloong_br::Thread_ROS()//内存管理线程
{
    ros::Rate loop_rate(200);
    ROS_INFO("Thread_ROS Started");
    //-----------------------------------------enviroment-------------------------------------------
    
    int cnt_p=9999;
    int cnt_p1=9999; 
    Cycle_Time_Init();

    std_msgs::Bool shi;
    shi.data = true;
    std_msgs::Bool fou;
    fou.data = false;

    while (ros::ok()&&!force_exit)
    {  
        float dt=Get_Cycle_T(0);
        //printf("dt=%f\n",dt);
        if (robotwb.arm_control_mode == 0)
        {
            pub.publish(shi);
        }
        else if (robotwb.arm_control_mode == 1)
        {
            pub.publish(fou);
        }
        state_machine_main(dt);

        ros::spinOnce();              
        loop_rate.sleep();
    } 
}
void openloong_br::Thread_UDP_ROBOT()//as client:send epose command to robot
{
    ROS_INFO("Thread_ROBOT UDP Started");

	int sock_fd;  
 
	sock_fd = socket(AF_INET, SOCK_DGRAM, 0);  
	if(sock_fd < 0)  
	{  
		exit(1);  
	}  

	int sock_fd_wh;  
 
	sock_fd_wh = socket(AF_INET, SOCK_DGRAM, 0);  
	if(sock_fd_wh < 0)  
	{  
		exit(1);  
	}  
     
   /* 设置address */  
    int SERV_PORT= 3334 ;//
	struct sockaddr_in addr_serv;  
	int len;  
	memset(&addr_serv, 0, sizeof(addr_serv));  
	addr_serv.sin_family = AF_INET;  
	std::string UDP_IP="192.168.1.188";//local test
    // string UDP_IP="192.168.1.44";

	addr_serv.sin_addr.s_addr = inet_addr(UDP_IP.c_str());  
	addr_serv.sin_port = htons(SERV_PORT);  
	len = sizeof(addr_serv);  


   /* 设置address */  
    int SERV_PORT_wh= 9015 ;//
	struct sockaddr_in addr_serv_wh;  
	int len_wh;  
	memset(&addr_serv_wh, 0, sizeof(addr_serv_wh));  
	addr_serv_wh.sin_family = AF_INET;  
	std::string UDP_IP_wh="127.0.0.1";//local test
    // string UDP_IP_wh="192.168.1.44";

	addr_serv_wh.sin_addr.s_addr = inet_addr(UDP_IP_wh.c_str());  
	addr_serv_wh.sin_port = htons(SERV_PORT_wh);  
	len_wh = sizeof(addr_serv_wh);  
    int recv_num=0,send_num=0;  
    int connect=0,loss_cnt=0;
    int recv_num_wh=0,send_num_wh=0;  
    int connect_wh=0,loss_cnt_wh=0;
    char send_buf[500]={0},recv_buf[500]={0}; 
    char send_buf_wh[500]={0},recv_buf_wh[500]={0}; 


        robotwb.arm_att_h_exp[0].x = 0.0;
        robotwb.arm_att_h_exp[0].y = 2.0;
        robotwb.arm_att_h_exp[0].z = -1.5708;
        robotwb.arm_epos_h_exp[0].x = -0.1;
        robotwb.arm_epos_h_exp[0].y = 0.2;
        robotwb.arm_epos_h_exp[0].z = 0.3;

        robotwb.arm_att_h_exp[1].x = 0.0;
        robotwb.arm_att_h_exp[1].y = 2.0;
        robotwb.arm_att_h_exp[1].z = 1.5708;
        robotwb.arm_epos_h_exp[1].x = -0.1;
        robotwb.arm_epos_h_exp[1].y = -0.2;
        robotwb.arm_epos_h_exp[1].z = 0.3;

        robotwb.cap_set[0] = 0.0;
        robotwb.cap_set[1] = 0.0;

    #if 1
        // float flt_pos[3]={0.01,0.01,0.01};
        // float flt_att[3]={0.01,0.01,0.01};
        float flt_pos[3]={1.0,1.0,1.0};
        float flt_att[3]={1.0,1.0,1.0};
        float flt_q=0.3;
        float flt_cap=0.7;
        // float flt_wait=0.01;
        // float flt_head=0.01;
        float flt_wait=1.0;
        float flt_head=1.0;
        float zu_waist[3]={0.0,0.0,0.0};
        float zu_head[3]={0.0,0.0};
    #else
        float flt_pos[3]={1,1,1};
        float flt_att[3]={1,1,1};
        float flt_q=1;
        float flt_cap=1;
        float flt_wait=1;
        float flt_head=1;
        float zu_waist[3]={0.0,0.0,0.0};
        float zu_head[3]={0.0,0.0};
    #endif
    while(!force_exit){
// std::cout<<"1"<<std::endl;

{
        std::lock_guard<std::mutex> lock(msg_mutex);
        if(robotwb.arm_control_mode>=1&&(robotwb.grasp_mission>0)){
            // std::cout<<"yes"<<std::endl;

            _send_msg_arm.arm_mode=robotwb.arm_control_mode;//0->disable  1->e_pose  2->joint

            for (int i = 0; i < 3; i++)
            {
                Waist_Head.waist_q_exp_flt[i]=Waist_Head.waist_q_exp[i]*flt_wait+(1-flt_wait)*Waist_Head.waist_q_exp_flt[i];
            }
            for (int i = 0; i < 2; i++)
            {
                Waist_Head.head_q_exp_flt[i]=Waist_Head.head_q_exp[i]*flt_head+(1-flt_head)*Waist_Head.head_q_exp_flt[i];
            }    
            for(int i=0;i<2;i++){//end
                robotwb.arm_epos_h_exp_flt[i].x=robotwb.arm_epos_h_exp[i].x*flt_pos[0]+(1-flt_pos[0])*robotwb.arm_epos_h_exp_flt[i].x;
                robotwb.arm_epos_h_exp_flt[i].y=robotwb.arm_epos_h_exp[i].y*flt_pos[1]+(1-flt_pos[1])*robotwb.arm_epos_h_exp_flt[i].y;
                robotwb.arm_epos_h_exp_flt[i].z=robotwb.arm_epos_h_exp[i].z*flt_pos[2]+(1-flt_pos[2])*robotwb.arm_epos_h_exp_flt[i].z;
                robotwb.arm_att_h_exp_flt[i].x=robotwb.arm_att_h_exp[i].x*flt_att[0]+(1-flt_att[0])*robotwb.arm_att_h_exp_flt[i].x;
                robotwb.arm_att_h_exp_flt[i].y=robotwb.arm_att_h_exp[i].y*flt_att[1]+(1-flt_att[1])*robotwb.arm_att_h_exp_flt[i].y;
                robotwb.arm_att_h_exp_flt[i].z=robotwb.arm_att_h_exp[i].z*flt_att[2]+(1-flt_att[2])*robotwb.arm_att_h_exp_flt[i].z;
                robotwb.cap_set_flt[i]=robotwb.cap_set[i]*flt_cap+(1-flt_cap)*robotwb.cap_set_flt[i];
            }

            //output filter----------------------------------------------------------------------

            _send_msg_arm.ee_pose_l[0]=robotwb.arm_att_h_exp_flt[0].x;
            _send_msg_arm.ee_pose_l[1]=robotwb.arm_att_h_exp_flt[0].y;
            _send_msg_arm.ee_pose_l[2]=robotwb.arm_att_h_exp_flt[0].z;
            _send_msg_arm.ee_pose_l[3]=robotwb.arm_epos_h_exp_flt[0].x*1000;
            _send_msg_arm.ee_pose_l[4]=robotwb.arm_epos_h_exp_flt[0].y*1000;
            _send_msg_arm.ee_pose_l[5]=robotwb.arm_epos_h_exp_flt[0].z*1000;
            _send_msg_arm.ee_pose_r[0]=robotwb.arm_att_h_exp_flt[1].x;
            _send_msg_arm.ee_pose_r[1]=robotwb.arm_att_h_exp_flt[1].y;
            _send_msg_arm.ee_pose_r[2]=robotwb.arm_att_h_exp_flt[1].z;
            _send_msg_arm.ee_pose_r[3]=robotwb.arm_epos_h_exp_flt[1].x*1000;
            _send_msg_arm.ee_pose_r[4]=robotwb.arm_epos_h_exp_flt[1].y*1000;
            _send_msg_arm.ee_pose_r[5]=robotwb.arm_epos_h_exp_flt[1].z*1000;
            _send_msg_arm.cap_set[0]=robotwb.cap_set_flt[0];
            _send_msg_arm.cap_set[1]=robotwb.cap_set_flt[1];
            float udp_9015[5] = {Waist_Head.waist_q_exp_flt[0] * 1000.0,
                        Waist_Head.waist_q_exp_flt[1] * 1000.0,
                        Waist_Head.waist_q_exp_flt[2] * 1000.0,
                        Waist_Head.head_q_exp_flt[0] * 1000.0,
                        Waist_Head.head_q_exp_flt[1] * 1000.0};
            send_msg_pose.head = 0xBB;
            send_msg_pose.mode = 1;
            send_msg_pose.base_vel[0] = 0.0f;
            send_msg_pose.base_vel[1] = 0.0f;
            send_msg_pose.base_vel[2] = 0.0f;
            // 左手部分示例

            send_msg_pose.arm_pos_exp_l[0] = _send_msg_arm.ee_pose_r[3];
            send_msg_pose.arm_pos_exp_l[1] = _send_msg_arm.ee_pose_l[4];
            send_msg_pose.arm_pos_exp_l[2] = _send_msg_arm.ee_pose_l[5];

            send_msg_pose.arm_att_exp_l[0] = _send_msg_arm.ee_pose_l[2]*180.0/3.1415;
            send_msg_pose.arm_att_exp_l[1] = _send_msg_arm.ee_pose_l[1]*180.0/3.1415;
            send_msg_pose.arm_att_exp_l[2] = _send_msg_arm.ee_pose_l[0]*180.0/3.1415;

            for (int i = 0; i < 7; ++i) send_msg_pose.arm_q_exp_l[i] = 0.0f;

            send_msg_pose.cap_l = _send_msg_arm.cap_set[0];

            // 右手部分
            send_msg_pose.arm_pos_exp_r[0] = _send_msg_arm.ee_pose_l[3];
            send_msg_pose.arm_pos_exp_r[1] = _send_msg_arm.ee_pose_r[4];
            send_msg_pose.arm_pos_exp_r[2] = _send_msg_arm.ee_pose_r[5];

            send_msg_pose.arm_att_exp_r[0] = _send_msg_arm.ee_pose_r[2]*180.0/3.1415;
            send_msg_pose.arm_att_exp_r[1] = _send_msg_arm.ee_pose_r[1]*180.0/3.1415;
            send_msg_pose.arm_att_exp_r[2] = _send_msg_arm.ee_pose_r[0]*180.0/3.1415;

            for (int i = 0; i < 7; ++i) send_msg_pose.arm_q_exp_r[i] = 0.0f;

            send_msg_pose.cap_r = _send_msg_arm.cap_set[1];
            send_msg_pose.waist_exp = Waist_Head.waist_q_exp[0]/180.0*3.1415;
            send_msg_pose.head_exp[0] = send_msg_pose.head_exp[1] = send_msg_pose.head_exp[2] = 0.0f;
            send_msg_pose.flt_rate = 1.0f;

            std::cout<<"left_pose:  ";
            for (size_t i = 0; i < 3; i++)
            {
                std::cout<<send_msg_pose.arm_pos_exp_l[i]<<"  ";
            }
            std::cout<<std::endl;
            std::cout<<"right_pose:  ";
            for (size_t i = 0; i < 3; i++)
            {
                std::cout<<send_msg_pose.arm_pos_exp_r[i]<<"  ";
            }
            std::cout<<std::endl;

            // 发送数据
            ssize_t sent = sendto(
                sock_fd,
                reinterpret_cast<char*>(&send_msg_pose),
                sizeof(remote_msg_end),
                0,
                reinterpret_cast<sockaddr*>(&addr_serv),
                sizeof(addr_serv)
            );

            if (sent == -1) {
                perror("sendto failed");
            } 

        }else{//reset flt as now pose

            robotwb.arm_epos_h_exp_flt[0]=robotwb.arm_epos_h_exp[0];
            robotwb.arm_epos_h_exp_flt[1]=robotwb.arm_epos_h_exp[1];
            robotwb.arm_att_h_exp_flt[0]=robotwb.arm_att_h_exp[0];
            robotwb.arm_att_h_exp_flt[1]=robotwb.arm_att_h_exp[1];
            robotwb.cap_set_flt[0]=robotwb.cap_set[0];
            robotwb.cap_set_flt[1]=robotwb.cap_set[1];
            Waist_Head.waist_q_exp_flt[0] = wai_hea[0];
            Waist_Head.waist_q_exp_flt[1] = wai_hea[1];
            Waist_Head.waist_q_exp_flt[2] = wai_hea[2];
            Waist_Head.head_q_exp_flt[0] = wai_hea[3];
            Waist_Head.head_q_exp_flt[1] = wai_hea[4];

            // 发送数据
            ssize_t sent = sendto(
                sock_fd,
                reinterpret_cast<char*>(&send_msg_pose),
                sizeof(remote_msg_end),
                0,
                reinterpret_cast<sockaddr*>(&addr_serv),
                sizeof(addr_serv)
            );

            if (sent == -1) {
                perror("sendto failed");
            } 
        }
        }
        usleep(1000*2);
    }
    close(sock_fd);  
}

void openloong_br::Thread_KEY()//内存管理线程
{
    ROS_INFO("Thread_KEY Started");
    while(!force_exit){
        key=updateKey();
        //printf("key=%d\n",key);
        usleep(1000*2);

        if(key==113){//force quit
            ROS_ERROR("Key interupted force quit!!");
            force_exit=1;
        }
        key=0;
    }
}

void openloong_br::startAllThreads() {
    // 启动四个不同功能的线程
    thread_key = std::thread(&openloong_br::Thread_KEY, this);
    thread_ros = std::thread(&openloong_br::Thread_ROS, this);
    thread_float = std::thread(&openloong_br::Thread_UDP_Float, this);
    thread_send = std::thread(&openloong_br::Thread_UDP_ROBOT, this);
}

void openloong_br::joinAllThreads() {
    // 等待所有线程结束
    if (thread_key.joinable()) {
        thread_key.join();
    }
    if (thread_ros.joinable()) {
        thread_ros.join();
    }
    if (thread_float.joinable()) {
        thread_float.join();
    }
    if (thread_send.joinable()) {
        thread_send.join();
    }
}
