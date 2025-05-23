#include "openloong_br.h"


void openloong_br::param_init(void){
    //left safe
    robotwb.arm_epos_h_exp_safe[0].x=-0.2;
    robotwb.arm_epos_h_exp_safe[0].y=0.35;
    robotwb.arm_epos_h_exp_safe[0].z=0.18;
    robotwb.arm_att_h_exp_safe[0].x=0.0;
    robotwb.arm_att_h_exp_safe[0].y=0.0;
    robotwb.arm_att_h_exp_safe[0].z=0.0;
    robotwb.cap_set_safe[0]=0;//open
    //right safe
    robotwb.arm_epos_h_exp_safe[1].x=-0.2;
    robotwb.arm_epos_h_exp_safe[1].y=-0.35;
    robotwb.arm_epos_h_exp_safe[1].z=0.18;
    robotwb.arm_att_h_exp_safe[1].x=0.0;
    robotwb.arm_att_h_exp_safe[1].y=0.0;
    robotwb.arm_att_h_exp_safe[1].z=0.0;


    robotwb.cap_set_safe[1]=0;//open

    robotwb.arm_control_mode=0;//disable arm_cmd send
    robotwb.grasp_mission[0]=0;
    robotwb.grasp_action[0]=0;
    robotwb.grasp_mission[1]=0;
    robotwb.grasp_action[1]=0;

    robotwb.grasp_mission[0]=0;
    robotwb.grasp_action[0]=0;
    robotwb.grasp_mission[1]=0;
    robotwb.grasp_action[1]=0;
    robotwb.reset_arm=1;

    robotwb.max_grasp_num=1000;

    ROS_INFO("Brain Param init done");
}


void openloong_br::state_machine_main(float dt){
    static float timer=0;
    static int init=0;
    if(!init){
        init=1;
        lsm_main=0;
        param_init();
    }
    
    switch(lsm_main){
        case 0:
            robotwb.arm_control_mode = 0;
            if(key==KEY_2){
                ROS_INFO("Active fenjian Mode");
                lsm_main=20;
                timer=0;
            }
            if(key==KEY_){
                ROS_INFO("Active LLM skill safe");
                lsm_main=100;
                timer=0;
            }
        break;



        //grasp-----------------------------------------------------
        case 20:
            if(TEST_MODE){
                ROS_INFO("Get Grasp Data");
                robotwb.arm_control_mode=1;   //初始化
                robotwb.grasp_mission[0]=0;
                robotwb.grasp_action[0]=0;
                robotwb.grasp_mission[1]=0;
                robotwb.grasp_action[1]=0;
                robotwb.reset_arm=1;
                robotwb.grasp_cnt=0;   //初始化
                lsm_main++;   //前往下一步
            }else{
                ROS_ERROR("Nan Grasp Data");
                lsm_main=0;    //如果出错返回初始lsm_main
            }
        break;
        case 21:
        timer+=dt;   //模拟时间步
        robotwb.arm_control_mode=1;   //将控制模式置1，即末端控制
        grasp_mission_auto_bofang_dps(dt, "Minute_Maid");  //grasp_mission_auto_bofang_dps智能抓取函数，Minute_Maid即在yaml文件夹中定义的一个任务
            if(robotwb.arm_control_mode == 0)    //在任务完成时 robotwb.arm_control_mode会置成0
            {   timer=0;    //模拟时间置0，即重新开始
                // ROS_ERROR("Grasp mission finish");
                lsm_main++;    //前往下一步
            }
        break;
        case 22:     //跳到第100步：复位
                timer+=dt;
                robotwb.arm_control_mode=1;
                robotwb.grasp_mission[0]=0;
                robotwb.grasp_action[0]=0;
                robotwb.grasp_mission[1]=0;
                robotwb.grasp_action[1]=0;
                robotwb.reset_arm=1;
                robotwb.grasp_cnt=0;
            if(timer>=0.2){
                timer=0;
                lsm_main = 100;   //lsm_main = 100即为复位
            }
        break;



        case 100://fail safe
            timer+=dt;
            robotwb.arm_control_mode=1;
            // restrict = true;
            for(int i=0;i<2;i++){
                robotwb.arm_epos_h_exp[i]=robotwb.arm_epos_h_exp_safe[i];
                robotwb.arm_att_h_exp[i]=robotwb.arm_att_h_exp_safe[i];
                robotwb.cap_set[i]=robotwb.cap_set_safe[i];
            }
            for (int i = 0; i < 3; i++)
            {
                Waist_Head.waist_q_exp[i]=0.0;
            }
            for (int i = 0; i < 2; i++)
            {
                Waist_Head.head_q_exp[i]=0.0;
            }    

            if(timer>0.6){
                timer=0;
                lsm_main++;
            }
        break;
        case 101:// 
            timer+=dt;
            for(int i=0;i<2;i++){
                robotwb.arm_epos_h_exp[i]=robotwb.arm_epos_h_exp_safe[i];
                robotwb.arm_att_h_exp[i]=robotwb.arm_att_h_exp_safe[i];
                robotwb.cap_set[i]=robotwb.cap_set_safe[i];
            }
            for (int i = 0; i < 3; i++)
            {
                Waist_Head.waist_q_exp[i]=0.0;
            }
            for (int i = 0; i < 2; i++)
            {
                Waist_Head.head_q_exp[i]=0.0;
            }    

            if(timer>2){
                timer=0;
                lsm_main=0;
                robotwb.arm_control_mode=0;
                robotwb.grasp_mission[0]=0;
                robotwb.grasp_action[0]=0;
                robotwb.grasp_mission[1]=0;
                robotwb.grasp_action[1]=0;
                ROS_ERROR("Reach Safe Pose & Exit");
            }

        break;

    }

    if(lsm_main>0&&lsm_main<100){//safe fail reset
        if(key==KEY_)
        {
            // std::cout<<lsm_main<<std::endl;
            ROS_ERROR("Fail Safe to pose");
            lsm_main=100;
            timer=0;
        }
    }
    robotwb.lsm_main=lsm_main;
}
