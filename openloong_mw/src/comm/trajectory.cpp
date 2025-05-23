#include "openloong_br.h"


void openloong_br::calculate_trajectory_config(const Actions task_config)
{
    target_pose = task_config.pose;
    order = task_config.older;
    task_speed = task_config.speed;
    task_arm_bezier = task_config.arm_bezier;
    task_waist_bezier = task_config.waist_bezier;
    if (order.size() >= 3 && 
        order[0].size() >= 12 && 
        order[1].size() >= 12 && 
        order[2].size() >= 12 && 
        order[3].size() >= 12) {
        erweima_left.assign(order[0].begin(), order[0].begin() + 6);
        erweima_right.assign(order[0].begin() + 6, order[0].begin() + 12);
        old_robot_to_erweima_left.assign(order[1].begin(), order[1].begin() + 6);
        old_robot_to_erweima_right.assign(order[1].begin() + 6, order[1].begin() + 12);
        erweima_left_new.assign(order[2].begin(), order[2].begin() + 6);
        erweima_right_new.assign(order[2].begin() + 6, order[2].begin() + 12);
        old_robot_to_erweima_left_new.assign(order[3].begin(), order[3].begin() + 6);
        old_robot_to_erweima_right_new.assign(order[3].begin() + 6, order[3].begin() + 12);
    }
    else {
        std::cerr << "Error: 'order' 结构不满足要求。" << std::endl;
    }
    if (task_config.eyes)
    {
        erweima_left_new[0] = robotwb.arm_epos_h_grasp.x;
        erweima_left_new[1] = robotwb.arm_epos_h_grasp.y;
        erweima_left_new[2] = robotwb.arm_epos_h_grasp.z;
        erweima_left_new[3] = robotwb.arm_att_h_grasp.x;
        erweima_left_new[4] = robotwb.arm_att_h_grasp.y;
        erweima_left_new[5] = robotwb.arm_att_h_grasp.z;

        erweima_right_new[0] = robotwb.arm_epos_h_grasp_right.x;
        erweima_right_new[1] = robotwb.arm_epos_h_grasp_right.y;
        erweima_right_new[2] = robotwb.arm_epos_h_grasp_right.z;
        erweima_right_new[3] = robotwb.arm_att_h_grasp_right.x;
        erweima_right_new[4] = robotwb.arm_att_h_grasp_right.y;
        erweima_right_new[5] = robotwb.arm_att_h_grasp_right.z;

    }
    if (!(task_config.com_left == false))
    {
        new_er_left[0] = erweima_left_new[0];
        new_er_left[1] = erweima_left_new[1];
        new_er_left[2] = erweima_left_new[2];
        new_er_left[3] = erweima_left_new[3];
        new_er_left[4] = erweima_left_new[4];
        new_er_left[5] = erweima_left_new[5];
    }
    else
    {
        new_er_left[0] = erweima_left[0];
        new_er_left[1] = erweima_left[1];
        new_er_left[2] = erweima_left[2];
        new_er_left[3] = erweima_left[3];
        new_er_left[4] = erweima_left[4];
        new_er_left[5] = erweima_left[5];
    }
    if (!(task_config.com_right == false))
    {
        new_er_right[0] = erweima_right_new[0];
        new_er_right[1] = erweima_right_new[1];
        new_er_right[2] = erweima_right_new[2];
        new_er_right[3] = erweima_right_new[3];
        new_er_right[4] = erweima_right_new[4];
        new_er_right[5] = erweima_right_new[5];
    }
    else
    {
        new_er_right[0] = erweima_right[0];
        new_er_right[1] = erweima_right[1];
        new_er_right[2] = erweima_right[2];
        new_er_right[3] = erweima_right[3];
        new_er_right[4] = erweima_right[4];
        new_er_right[5] = erweima_right[5];
    }
    batch_size = 10;
    total_rows = target_pose.size();
    processed_rows = 0;
    while (processed_rows < total_rows) {
        current_batch_size = std::min(batch_size, total_rows - processed_rows);
        // 清空目标容器，以避免保存旧数据
        target_pose_left.clear();
        target_pose_right.clear();
        target_pose_waist.clear();
        for (size_t i = processed_rows; i < processed_rows + current_batch_size; i++) {
            const auto& row = target_pose[i];
            if (row.size() >= 13) {
                target_pose_left.push_back(std::vector<float>(row.begin(), row.begin() + 7));
                target_pose_right.push_back(std::vector<float>(row.begin() + 7, row.begin() + 14));
                if (row.size()<17)
                {
                    waist_point = {Waist_Head.waist_q_exp_flt[0], Waist_Head.head_q_exp_flt[0], Waist_Head.head_q_exp_flt[1]};
                    target_pose_waist.push_back(waist_point);
                }
                else
                {
                    waist_point = {row.begin() + 15, row.begin() + 18};
                    waist_point[0] = waist_point[0]*180/3.1415;
                    waist_point[1] = waist_point[1]*180/3.1415;
                    waist_point[2] = waist_point[2]*180/3.1415;
                    target_pose_waist.push_back(waist_point);
                }
            } else {
                std::cerr << "Row size is less than expected: " << row.size() << std::endl;
            }
        }
    max_old_robot_to_erweima_left = position_to_max(old_robot_to_erweima_left);
    qu2_left = position_to_max(erweima_left);
    qu_new2_left = position_to_max(new_er_left);
    old_to_new2_left = qu2_left.inverse() * qu_new2_left;

    current_pose_left[0] = robotwb.arm_epos_h_exp_flt[0].x;
    current_pose_left[1] = robotwb.arm_epos_h_exp_flt[0].y;
    current_pose_left[2] = robotwb.arm_epos_h_exp_flt[0].z;
    current_pose_left[3] = robotwb.arm_att_h_exp_flt[0].x;
    current_pose_left[4] = robotwb.arm_att_h_exp_flt[0].y;
    current_pose_left[5] = robotwb.arm_att_h_exp_flt[0].z;

    current_pose_right[0] = robotwb.arm_epos_h_exp_flt[1].x;
    current_pose_right[1] = robotwb.arm_epos_h_exp_flt[1].y;
    current_pose_right[2] = robotwb.arm_epos_h_exp_flt[1].z;
    current_pose_right[3] = robotwb.arm_att_h_exp_flt[1].x;
    current_pose_right[4] = robotwb.arm_att_h_exp_flt[1].y;
    current_pose_right[5] = robotwb.arm_att_h_exp_flt[1].z;

    current_waist_pose[0] = Waist_Head.waist_q_exp_flt[0];
    current_waist_pose[1] = Waist_Head.head_q_exp_flt[0];
    current_waist_pose[2] = Waist_Head.head_q_exp_flt[1];


    max_old_robot_to_erweima_right = position_to_max(old_robot_to_erweima_right);
    qu2_right = position_to_max(erweima_right);
    qu_new2_right = position_to_max(new_er_right);
    old_to_new2_right = qu2_right.inverse() * qu_new2_right;

    target_pose_new_left = guihua_dan(old_to_new2_left, max_old_robot_to_erweima_left, target_pose_left);
    target_pose_new_right = guihua_dan(old_to_new2_right, max_old_robot_to_erweima_right, target_pose_right);

    for (size_t p = 0; p < current_batch_size; p++)
    {
        dur = (double)(target_pose)[p + processed_rows][14];
        start_time = 0;
        end_time = start_time + dur;
        tel = 0.0;
        for (size_t j = 0; j < 6; j++)
        {
            exp_pose_left[j] = target_pose_new_left[p][j];  
            exp_pose_right[j] = target_pose_new_right[p][j];
        }
        for (size_t j = 0; j < 3; j++)
        {
            exp_pose_waist[j] = target_pose_waist[p][j];
        }
        for (double t = 0.0; t < end_time; t = t + r_dt)
        {
            tel = t/dur;
            if (tel<=0)
                tel = 0;
            else if (tel>=1)
                tel = 1;
            if (task_arm_bezier)
            {
                bezier_arm_pose_left = bezierInterpolationEEF_dan(current_pose_left, exp_pose_left, tel);
                bezier_arm_pose_right = bezierInterpolationEEF_dan(current_pose_right, exp_pose_right, tel);
            }
            else
            {
                bezier_arm_pose_left = bezierInterpolationEEF_dan(current_pose_left, exp_pose_left, 1.0);
                bezier_arm_pose_right = bezierInterpolationEEF_dan(current_pose_right, exp_pose_right, 1.0);
            }
            if (task_waist_bezier)
            {
                bezier_waist_pose = bezierInterpolationJoints(current_waist_pose, exp_pose_waist, tel);
            }
            else
            {
                bezier_waist_pose = bezierInterpolationJoints(current_waist_pose, exp_pose_waist, 1.0);
            }
               robotwb.arm_att_h_exp[0].x=bezier_arm_pose_left[3];
               robotwb.arm_att_h_exp[0].y=bezier_arm_pose_left[4];
               robotwb.arm_att_h_exp[0].z=bezier_arm_pose_left[5];
                robotwb.arm_epos_h_exp[0].x=bezier_arm_pose_left[0];
                robotwb.arm_epos_h_exp[0].y=bezier_arm_pose_left[1];
                robotwb.arm_epos_h_exp[0].z=bezier_arm_pose_left[2];
                robotwb.cap_set[0] = target_pose_left[p][6];
               robotwb.arm_att_h_exp[1].x=bezier_arm_pose_right[3];
               robotwb.arm_att_h_exp[1].y=bezier_arm_pose_right[4];
               robotwb.arm_att_h_exp[1].z=bezier_arm_pose_right[5];
                robotwb.arm_epos_h_exp[1].x=bezier_arm_pose_right[0];
                robotwb.arm_epos_h_exp[1].y=bezier_arm_pose_right[1];
                robotwb.arm_epos_h_exp[1].z=bezier_arm_pose_right[2];
                robotwb.cap_set[1] = target_pose_right[p][6];
                Waist_Head.waist_q_exp[0] = bezier_waist_pose[0];
                Waist_Head.head_q_exp[0] = bezier_waist_pose[1];
                Waist_Head.head_q_exp[1] = bezier_waist_pose[2];
                std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(r_dt * 1000 / task_speed)));
    }
    }
    processed_rows += current_batch_size;
}
std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(task_config.sleep * 1000)));
}


// 函数：根据 task_name 查找匹配的 Tasks，返回指向 Tasks 的指针
openloong_br::Tasks* openloong_br::findTaskByName(std::vector<Tasks>& big_task_list, const std::string& task_name) {
    for (auto& task : big_task_list) {
        if (task.task_name == task_name) {
            return &task; // 返回指向匹配任务的指针
        }
    }
    return nullptr; // 如果未找到，返回 nullptr
}


void openloong_br::grasp_mission_auto_bofang_alone(float dt, const std::string& task_name){

    timer_grasp=0;
    init=0;
    if(!init){
        init=1;

    }
        switch(robotwb.grasp_mission[0]){
        case 0:

            if(robotwb.grasp_action[0]==0){//start grasping
                robotwb.grasp_mission[0]++;

                timer_grasp=0;
                printf("[L]start grasping goto check pose\n");
            }
        break;
        case 1://goto check pose
            timer_grasp+=dt;
            classifyAndSortObjects_right();
                timer_grasp=0;
                robotwb.grasp_mission[0]++;
                robotwb.grasp_action[0]=1;//set 1 to trigger the target pose reg
            
        break;
        case 2://check target
        {
                found_task = findTaskByName(big_task_list, task_name);

                if (found_task != nullptr) {
                    std::cout << "找到任务: " << found_task->task_name << std::endl;
                    // 检查 task_config_list 是否为空
                    if (found_task->task_config_list.empty()) {
                        std::cout << "  该任务没有 Actions 配置。" << std::endl;
                    } else {
                        std::cout<<"robotwb_env_right"<<std::endl;
                        std::cout<<robotwb_env_right.size()<<std::endl;
                        if (!robotwb_env_right.empty())
                        {
                            firstObject = robotwb_env_right.front();
                            robotwb.arm_epos_h_grasp_right.x=firstObject.pose_right.x;//left hand
                            robotwb.arm_epos_h_grasp_right.y=firstObject.pose_right.y;
                            robotwb.arm_epos_h_grasp_right.z=firstObject.pose_right.z;
                            robotwb.arm_att_h_grasp_right.x=firstObject.att_right.x;
                            robotwb.arm_att_h_grasp_right.y=firstObject.att_right.y;
                            robotwb.arm_att_h_grasp_right.z=firstObject.att_right.z;
                            
                            std::cout<<firstObject.pose_right.x<<std::endl;
                            std::cout<<firstObject.pose_right.y<<std::endl;
                            std::cout<<firstObject.pose_right.z<<std::endl;
                            std::cout<<firstObject.att_right.x<<std::endl;
                            std::cout<<firstObject.att_right.y<<std::endl;
                            std::cout<<firstObject.att_right.z<<std::endl;
                        }

                        // 遍历 task_config_list 并输出 eyes 成员
                        for (size_t i = 0; i < found_task->task_config_list.size(); ++i) {
                            calculate_trajectory_config(found_task->task_config_list[i]);
                        }
                    }
                } else {
                    std::cout << "未找到任务名为 \"" << task_name << "\" 的任务。" << std::endl;
                }

               {//locate good<<---------------------
                   timer_grasp=0;
                   robotwb.grasp_mission[0]++;
                   printf("[R][0]locate target done\n");
               }
        }

        break;

        case 3://reset and grasp next target
            robotwb.arm_control_mode=0; 
        break;
        default:
            robotwb.grasp_mission[0]=0;
            robotwb.grasp_mission[1]=0;
        break;
    }
}

void openloong_br::grasp_mission_auto_bofang_interaction(float dt, const std::string& task_name){

    timer_grasp=0;
    init=0;
    if(!init){
        init=1;

    }
        switch(robotwb.grasp_mission[0]){
        case 0:

            if(robotwb.grasp_action[0]==0){//start grasping
                robotwb.grasp_mission[0]++;

                timer_grasp=0;
                printf("[L]start grasping goto check pose\n");
            }
        break;
        case 1://goto check pose
            timer_grasp+=dt;
            if (task_name == "tidy")
            {
                classifyAndSortObjects_left();
                classifyAndSortObjects_right();
                hamburger_namesToKeep = {"hamburger", "hamburger2", "bread", "bread2"};
                bread_namesToKeep = {"hamburger", "hamburger2", "bread", "bread2"};
                keepSpecificObjectNames_left_duo(hamburger_namesToKeep);
                keepSpecificObjectNames_right_duo(bread_namesToKeep);
                if (!robotwb_env_left.empty())
                {
                    ObjectDetails& firstObject = robotwb_env_left.front();
                    robotwb.arm_epos_h_grasp.x=firstObject.pose_left.x;//left hand
                    robotwb.arm_epos_h_grasp.y=firstObject.pose_left.y;
                    robotwb.arm_epos_h_grasp.z=firstObject.pose_left.z;
                    robotwb.arm_att_h_grasp.x=firstObject.att_left.x;
                    robotwb.arm_att_h_grasp.y=firstObject.att_left.y;
                    robotwb.arm_att_h_grasp.z=firstObject.att_left.z;
                        if((firstObject.name == "bread") || (firstObject.name == "bread2"))
                        {
                            left_none = "bread";
                        }
                        else if ((firstObject.name == "hamburger") || (firstObject.name == "hamburger2"))
                        {
                            left_none = "hamburger";
                        }
                        else
                        {
                            left_none = "none";
                        }
                }
                else
                {
                    std::cout << "No objects available in robotwb_env_left" << std::endl;
                    left_none = "none";
                }
                if (!robotwb_env_right.empty())
                {
                    firstObject = robotwb_env_right.front();
                    robotwb.arm_epos_h_grasp_right.x=firstObject.pose_right.x;//left hand
                    robotwb.arm_epos_h_grasp_right.y=firstObject.pose_right.y;
                    robotwb.arm_epos_h_grasp_right.z=firstObject.pose_right.z;
                    robotwb.arm_att_h_grasp_right.x=firstObject.att_right.x;
                    robotwb.arm_att_h_grasp_right.y=firstObject.att_right.y;
                    robotwb.arm_att_h_grasp_right.z=firstObject.att_right.z;
                    if((firstObject.name == "bread") || (firstObject.name == "bread2") || (firstObject.name == "hamburger") || (firstObject.name == "hamburger2"))
                    {
                        right_none = "have";
                    }
                    else
                    {
                        right_none = "none";
                    }
                }
                else
                {
                    std::cout << "No objects available in robotwb_env_right" << std::endl;
                    right_none = "none";
                }
                if ((left_none == "none") && (right_none == "none"))
                {
                    std::cout << "No objects available in robotwb_env" << std::endl;
                    robotwb.arm_control_mode=0;
                    break;
                }
            }
            else
            {
                classifyAndSortObjects_left();
                classifyAndSortObjects_right();
                hamburger_namesToKeep = {task_name};
                bread_namesToKeep = {task_name};
                keepSpecificObjectNames_left_duo(hamburger_namesToKeep);
                keepSpecificObjectNames_right_duo(bread_namesToKeep);
                if (!robotwb_env_left.empty())
                {
                    firstObject = robotwb_env_left.front();
                    robotwb.arm_epos_h_grasp.x=firstObject.pose_left.x;//left hand
                    robotwb.arm_epos_h_grasp.y=firstObject.pose_left.y;
                    robotwb.arm_epos_h_grasp.z=firstObject.pose_left.z;
                    robotwb.arm_att_h_grasp.x=firstObject.att_left.x;
                    robotwb.arm_att_h_grasp.y=firstObject.att_left.y;
                    robotwb.arm_att_h_grasp.z=firstObject.att_left.z;
                    left_none = task_name;
                }
                else
                {
                    std::cout << "No objects available in robotwb_env_left" << std::endl;
                    left_none = "none";
                }
                if (!robotwb_env_right.empty())
                {
                    firstObject = robotwb_env_right.front();
                    robotwb.arm_epos_h_grasp_right.x=firstObject.pose_right.x;//left hand
                    robotwb.arm_epos_h_grasp_right.y=firstObject.pose_right.y;
                    robotwb.arm_epos_h_grasp_right.z=firstObject.pose_right.z;
                    robotwb.arm_att_h_grasp_right.x=firstObject.att_right.x;
                    robotwb.arm_att_h_grasp_right.y=firstObject.att_right.y;
                    robotwb.arm_att_h_grasp_right.z=firstObject.att_right.z;
                    right_none = task_name;
                }
                else
                {
                    std::cout << "No objects available in robotwb_env_right" << std::endl;
                    right_none = "none";
                }
                if ((left_none == "none") && (right_none == "none"))
                {
                    std::cout << "No objects available in robotwb_env" << std::endl;
                    robotwb.arm_control_mode=0;
                    break;
                }
            }
            if(timer_grasp>2.0)
            {
                timer_grasp=0;
                robotwb.grasp_mission[0]++;
                robotwb.grasp_action[0]=1;//set 1 to trigger the target pose reg
            }
            
        break;
        case 2://check target
        {
                found_task = findTaskByName(big_task_list, left_none + left_none);

                if (found_task != nullptr) {
                    std::cout << "找到任务: " << found_task->task_name << std::endl;
                    // 检查 task_config_list 是否为空
                    if (found_task->task_config_list.empty()) {
                        std::cout << "  该任务没有 Actions 配置。" << std::endl;
                    } else {
                        // 遍历 task_config_list 并输出 eyes 成员
                        for (size_t i = 0; i < found_task->task_config_list.size(); ++i) {
                            calculate_trajectory_config(found_task->task_config_list[i]);
                        }
                    }
                } else {
                    std::cout << "未找到任务名为 \"" << left_none + left_none << "\" 的任务。" << std::endl;
                }

               {//locate good<<---------------------
                   timer_grasp=0;
                   robotwb.grasp_mission[0]++;
                   printf("[R][0]locate target done\n");
               }
        }

        break;

        case 3://reset and grasp next target
            timer_grasp+=dt;
                timer_grasp=0;
                robotwb.grasp_cnt++;
                robotwb.grasp_mission[0]=1;
                robotwb.grasp_mission[1]=1;
                printf("[L]reset done!\n");

        break;
        default:
            robotwb.grasp_mission[0]=0;
            robotwb.grasp_mission[1]=0;
        break;
    }
}

void openloong_br::grasp_mission_auto_bofang_dps(float dt, const std::string& task_name){

    timer_grasp=0;
    init=0;
    if(!init){
        init=1;

    }
        switch(robotwb.grasp_mission[0]){
        case 0:

            if(robotwb.grasp_action[0]==0){//start grasping
                robotwb.grasp_mission[0]++;

                timer_grasp=0;
                printf("[L]start grasping goto check pose\n");
            }
        break;
        case 1://goto check pose
            timer_grasp+=dt;
                classifyAndSortObjects_right();
                keepSpecificObjectName_right(task_name);
                if (!robotwb_env_right.empty())
                {
                    firstObject = robotwb_env_right.front();
                    robotwb.arm_epos_h_grasp_right.x=firstObject.pose_right.x;//left hand
                    robotwb.arm_epos_h_grasp_right.y=firstObject.pose_right.y;
                    robotwb.arm_epos_h_grasp_right.z=firstObject.pose_right.z;
                    robotwb.arm_att_h_grasp_right.x=0.0;
                    robotwb.arm_att_h_grasp_right.y=0.0;
                    robotwb.arm_att_h_grasp_right.z=0.0;
                    right_none = "have";
                }
                else
                {
                    std::cout << "No objects available in robotwb_env_right" << std::endl;
                    right_none = "none";
                }
                if (right_none == "none")
                {
                    std::cout << "No objects available in robotwb_env" << std::endl;
                    robotwb.arm_control_mode=0;
                    break;
                }
                else
                {
                    timer_grasp=0;
                    robotwb.grasp_mission[0]++;
                    robotwb.grasp_action[0]=1;//set 1 to trigger the target pose reg

                }
            
        break;
        case 2://check target
        {
                found_task = findTaskByName(big_task_list, task_name);
                if (found_task != nullptr) {
                    std::cout << "找到任务: " << found_task->task_name << std::endl;
                    // 检查 task_config_list 是否为空
                    if (found_task->task_config_list.empty()) {
                        std::cout << "  该任务没有 Actions 配置。" << std::endl;
                    } else {
                        // 遍历 task_config_list 并输出 eyes 成员
                        for (size_t i = 0; i < found_task->task_config_list.size(); ++i) {
                            calculate_trajectory_config(found_task->task_config_list[i]);
                        }
                    }
                } else {
                    std::cout << "未找到任务名为 \"" << task_name << "\" 的任务。" << std::endl;
                }
               {//locate good<<---------------------
                   timer_grasp=0;
                   robotwb.grasp_mission[0]++;
                   printf("[R][0]locate target done\n");
               }
        }

        break;

        case 3://reset and grasp next target
            robotwb.arm_control_mode=0; 

        break;
        default:
            robotwb.grasp_mission[0]=0;
            robotwb.grasp_mission[1]=0;
        break;
    }
}
