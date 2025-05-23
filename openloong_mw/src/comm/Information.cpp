#include <openloong_br.h>


void openloong_br::find_yaml(const std::string& directoryPath)
{
        big_task_list.clear();
        // 创建一个字符串序列来存储文件名
        try {
            // 检查目录是否存在且确实是一个目录
            if (!std::filesystem::exists(directoryPath)) {
                std::cerr << "错误: 目录不存在。" << std::endl;
                std::exit(EXIT_FAILURE);
            }
            if (!std::filesystem::is_directory(directoryPath)) {
                std::cerr << "错误: 提供的路径不是一个目录。" << std::endl;
                std::exit(EXIT_FAILURE);
            }
            // 遍历目录中的每一个条目
            for (const auto& entry : std::filesystem::directory_iterator(directoryPath)) {
                // 只处理常规文件
                if (entry.is_regular_file()) {
                    
                    filePath = entry.path();
                    // 获取文件扩展名并转换为小写以实现不区分大小写的比较
                    extension = filePath.extension().string();
                    for (auto & c: extension) c = std::tolower(c);

                    if (extension == ".yaml" || extension == ".yml") {
                        // 获取文件名（不包括扩展名）
                        small_task_list.task_name = filePath.stem().string();
                        small_task_list.task_config_list.clear();
                        read_yaml_config(small_task_list.task_name);
                        big_task_list.push_back(small_task_list);
                    }
                }
            }
        }
        catch (const std::filesystem::filesystem_error& e) {
            std::cerr << "文件系统错误: " << e.what() << std::endl;
            std::exit(EXIT_FAILURE);
        }
        catch (const std::exception& e) {
            std::cerr << "常规错误: " << e.what() << std::endl;
            std::exit(EXIT_FAILURE);
        }

    }


void openloong_br::read_yaml_config(const std::string& name) 
{
    
    try {
        // 使用拼接后的完整路径加载 YAML 文件
        config = YAML::LoadFile(base_path + "/" + name + ".yaml");
        // 检查文件内容是否为列表格式
        if (!config.IsSequence()) {
            std::cerr << "YAML file is not a sequence!" << std::endl;
            std::exit(EXIT_FAILURE);
        }
        // 遍历每个动作节点
        for (const auto& node : config) {
            action_name = node["action"].as<std::string>();
            task_config.pose.clear();
            task_config.older.clear();
            task_config.pose = readDataFromFile(base_path_pose + "/" + action_name + ".txt");
            task_config.older = readDataFromFile(base_path_older + "/" + action_name + ".txt");
            task_config.com_left = node["com_left"].as<bool>();  // 读取为 bool 类型
            task_config.com_right = node["com_right"].as<bool>();  // 读取为 bool 类型
            task_config.eyes = node["eyes"].as<bool>();
            task_config.arm_bezier = node["arm_bezier"].as<bool>();  // 读取为 bool 类型
            task_config.waist_bezier = node["waist_bezier"].as<bool>();  // 读取为 bool 类型
            task_config.speed = node["speed"].as<float>();
            task_config.sleep = node["sleep"].as<float>();
            // 将每个动作存储到 vector 中
            small_task_list.task_config_list.push_back(task_config);
        }
    } catch (const YAML::Exception& e) {
        std::cerr << "Error reading YAML file: " << e.what() << std::endl;
        std::exit(EXIT_FAILURE);
    }
}




std::vector<std::vector<float>> openloong_br::readDataFromFile(const std::string& filename)
{
    inputFile.clear();  
    inputFile.open(filename);      // 再调用 open() 打开文件
    if (!inputFile.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return {};  // 返回空 vector 表示错误
    }
    txt_target_data.clear();
    // 逐行读取文件内容
    while (std::getline(inputFile, txt_line)) {
        txt_ss.str("");
        txt_ss.clear();
        txt_row.clear();
        txt_ss.str(txt_line);

        // 逐个读取浮点数并存储到当前行的 vector 中
        while (txt_ss >> txt_value) {
            txt_row.push_back(txt_value);  // 将读取到的浮点数加入到当前行
            txt_ss >> comma_remove;  // 跳过逗号
        }

        txt_target_data.push_back(txt_row);  // 将当前行加入到总的数据容器中
    }

    inputFile.close();  // 关闭文件
    return txt_target_data;  // 返回读取到的数据
}

std::vector<std::vector<float>> openloong_br::guihua(const Eigen::Matrix4d& binahuan, const Eigen::Matrix4d& old_robot_to_erweima, std::vector<std::vector<float>> target_pose)
{
    target_left_up_new.clear();
    target_right_up_new.clear();
    target_new_guihua.clear();
    point_bianhuan_guihua = old_robot_to_erweima * binahuan * old_robot_to_erweima.inverse();
    for (size_t i = 0; i < (target_pose.size()); i++)
        {
            pos_left_guihua.clear();
            pos_right_guihua.clear();
            test_left_guihua.clear();
            test_right_guihua.clear();
            pos_left_guihua.assign(target_pose[i].begin(), target_pose[i].begin() + 6);
            pos_right_guihua.assign(target_pose[i].begin() + 6, target_pose[i].begin() + 12);
            test_left_guihua = bianhua(point_bianhuan_guihua, pos_left_guihua);
            test_right_guihua = bianhua(point_bianhuan_guihua, pos_right_guihua);
            target_left_up_new.push_back(test_left_guihua);
            target_right_up_new.push_back(test_right_guihua);
        }
        // 遍历每一行，合并对应的行
        for (size_t i = 0; i < target_left_up_new.size(); ++i) {
            // 创建一个新行，并将左、右部分的元素依次添加到该行
            combined_row.clear();
            combined_row.insert(combined_row.end(), target_left_up_new[i].begin(), target_left_up_new[i].end());
            combined_row.insert(combined_row.end(), target_right_up_new[i].begin(), target_right_up_new[i].end());
            // 将合并后的行添加到目标二维向量中
            target_new_guihua.push_back(combined_row);
        }
        return target_new_guihua;
}

std::vector<std::vector<float>> openloong_br::guihua_dan(const Eigen::Matrix4d& binahuan, const Eigen::Matrix4d& old_robot_to_erweima, std::vector<std::vector<float>> target_pose)
{
    point_bianhuan_guihua_dan = old_robot_to_erweima * binahuan * old_robot_to_erweima.inverse();
    target_pose_dan_new.clear();
    for (size_t i = 0; i < (target_pose.size()); i++)
        {
            test_guihua_dan.clear();
            pos_guihua_dan.assign(target_pose[i].begin(), target_pose[i].begin() + 6);
            test_guihua_dan = bianhua(point_bianhuan_guihua_dan, pos_guihua_dan);
            target_pose_dan_new.push_back(test_guihua_dan);
        }
        return target_pose_dan_new;
}


void openloong_br::classifyAndSortObjects_left() {
    // 先清除旧数据
    robotwb_env_left.clear();

    // 根据side属性分配对象到左侧或右侧
    for (int i = 0; i < 10; i++) {
        if (robotwb._env.obj_state[i] != 0) {
            ObjectDetails obj;
            obj.id = i;
            obj.name = robotwb._env.obj_name[i];
            obj.side = robotwb._env.obj_side[i];

            obj.pose_left = robotwb._env.obj_pos_from_left[i];
            obj.pose_left_app = robotwb._env.wrist_pos_from_left[i];
            obj.pose_left_app_s = robotwb._env.imminent_pos_from_left[i];

            obj.att_left = robotwb._env.obj_att_from_left[i];
            obj.att_left_app = robotwb._env.wrist_att_from_left[i];
            obj.att_left_app_s = robotwb._env.imminent_att_from_left[i];

            obj.pose_right = robotwb._env.obj_pos_from_right[i];
            obj.pose_right_app = robotwb._env.wrist_pos_from_right[i];
            obj.pose_right_app_s = robotwb._env.imminent_pos_from_right[i];

            obj.att_right = robotwb._env.obj_att_from_right[i];
            obj.att_right_app = robotwb._env.wrist_att_from_right[i];
            obj.att_right_app_s = robotwb._env.imminent_att_from_right[i];

            obj.clss = robotwb._env.obj_box[i].clss;
            obj.x = robotwb._env.obj_box[i].x;
            obj.y = robotwb._env.obj_box[i].y;
            obj.width = robotwb._env.obj_box[i].width;
            obj.height = robotwb._env.obj_box[i].height;
            obj.conf = robotwb._env.obj_box[i].conf;

            if (obj.side == 0) {
                robotwb_env_left.push_back(obj);
            } 
            // else if (obj.side == 1) {
            //     robotwb_env_right.push_back(obj);
            // }
        }
    }

    // 排序左侧和右侧列表
    auto sortByY = [](const ObjectDetails &a, const ObjectDetails &b) {
        return a.y < b.y;
    };
    std::sort(robotwb_env_left.begin(), robotwb_env_left.end(), sortByY);
    // std::sort(robotwb_env_right.begin(), robotwb_env_right.end(), sortByY);
}

void openloong_br::classifyAndSortObjects_right() {
    // 先清除旧数据
    // robotwb_env_left.clear();
    robotwb_env_right.clear();

    // 根据side属性分配对象到左侧或右侧
    for (int i = 0; i < 10; i++) {
        if (robotwb._env.obj_state[i] != 0) {
            ObjectDetails obj;
            obj.id = i;
            obj.name = robotwb._env.obj_name[i];
            obj.side = robotwb._env.obj_side[i];

            obj.pose_left = robotwb._env.obj_pos_from_left[i];
            obj.pose_left_app = robotwb._env.wrist_pos_from_left[i];
            obj.pose_left_app_s = robotwb._env.imminent_pos_from_left[i];

            obj.att_left = robotwb._env.obj_att_from_left[i];
            obj.att_left_app = robotwb._env.wrist_att_from_left[i];
            obj.att_left_app_s = robotwb._env.imminent_att_from_left[i];

            obj.pose_right = robotwb._env.obj_pos_from_right[i];
            obj.pose_right_app = robotwb._env.wrist_pos_from_right[i];
            obj.pose_right_app_s = robotwb._env.imminent_pos_from_right[i];

            obj.att_right = robotwb._env.obj_att_from_right[i];
            obj.att_right_app = robotwb._env.wrist_att_from_right[i];
            obj.att_right_app_s = robotwb._env.imminent_att_from_right[i];

            obj.clss = robotwb._env.obj_box[i].clss;
            obj.x = robotwb._env.obj_box[i].x;
            obj.y = robotwb._env.obj_box[i].y;
            obj.width = robotwb._env.obj_box[i].width;
            obj.height = robotwb._env.obj_box[i].height;
            obj.conf = robotwb._env.obj_box[i].conf;
            std::cout<<"obj.side:   "<<obj.side<<std::endl;
            if (obj.side == 1) {
                robotwb_env_right.push_back(obj);
            }
        }
    }

    // 排序左侧和右侧列表
    auto sortByY = [](const ObjectDetails &a, const ObjectDetails &b) {
        return a.y < b.y;
    };
    // std::sort(robotwb_env_left.begin(), robotwb_env_left.end(), sortByY);
    std::sort(robotwb_env_right.begin(), robotwb_env_right.end(), sortByY);
}

void openloong_br::waesdfrgthObjectDetails(const std::vector<ObjectDetails>& objects, const std::string& side) {
    std::cout << "Number of objects on the " << side << " side: " << objects.size() << std::endl;
    for (const auto& obj : objects) {
        std::cout << "Object Name: " << obj.name << std::endl;
    }
}

void openloong_br::keepSpecificObjectName_left(const std::string& specificName) {
    auto newEnd = std::remove_if(robotwb_env_left.begin(), robotwb_env_left.end(),
                                 [&specificName](const ObjectDetails& obj) {
                                     return obj.name != specificName; // 移除不匹配的名称
                                 });
    robotwb_env_left.erase(newEnd, robotwb_env_left.end());

    // 打印结果，查看当前列表中剩余的对象
    std::cout << "Objects left after removal:" << std::endl;
    for (const auto& obj : robotwb_env_left) {
        std::cout << "Object Name: " << obj.name << ", ID: " << obj.id << std::endl;
    }
}

void openloong_br::keepSpecificObjectName_right(const std::string& specificName) {
    auto newEnd = std::remove_if(robotwb_env_right.begin(), robotwb_env_right.end(),
                                 [&specificName](const ObjectDetails& obj) {
                                     return obj.name != specificName; // 移除不匹配的名称
                                 });
    robotwb_env_right.erase(newEnd, robotwb_env_right.end());

    // 打印结果，查看当前列表中剩余的对象
    std::cout << "Objects right after removal:" << std::endl;
    for (const auto& obj : robotwb_env_right) {
        std::cout << "Object Name: " << obj.name << ", ID: " << obj.id << std::endl;
    }
}

void openloong_br::keepSpecificObjectNames_left_duo(const std::unordered_set<std::string>& names) {
    auto newEnd = std::remove_if(robotwb_env_left.begin(), robotwb_env_left.end(),
                                 [&names](const ObjectDetails& obj) {
                                     return names.find(obj.name) == names.end(); // 如果名字不在集合中，则移除
                                 });
    robotwb_env_left.erase(newEnd, robotwb_env_left.end());

    // 打印结果，查看当前列表中剩余的对象
    std::cout << "Objects left after removal:" << std::endl;
    for (const auto& obj : robotwb_env_left) {
        std::cout << "Object Name: " << obj.name << ", ID: " << obj.id << std::endl;
    }
}

void openloong_br::keepSpecificObjectNames_right_duo(const std::unordered_set<std::string>& names) {
    auto newEnd = std::remove_if(robotwb_env_right.begin(), robotwb_env_right.end(),
                                 [&names](const ObjectDetails& obj) {
                                     return names.find(obj.name) == names.end(); // 如果名字不在集合中，则移除
                                 });
    robotwb_env_right.erase(newEnd, robotwb_env_right.end());

    // 打印结果，查看当前列表中剩余的对象
    std::cout << "Objects right after removal:" << std::endl;
    for (const auto& obj : robotwb_env_right) {
        std::cout << "Object Name: " << obj.name << ", ID: " << obj.id << std::endl;
    }
}

int openloong_br::containsName_left(const std::string& nameToFind) {
    auto it = std::find_if(robotwb_env_left.begin(), robotwb_env_left.end(), 
                           [&nameToFind](const ObjectDetails& obj) {
                               return obj.name == nameToFind;
                           });
    return it != robotwb_env_left.end() ? 1 : 0; // 返回 1 如果找到，否则返回 0
}

int openloong_br::containsName_right(const std::string& nameToFind) {
    auto it = std::find_if(robotwb_env_right.begin(), robotwb_env_right.end(), 
                           [&nameToFind](const ObjectDetails& obj) {
                               return obj.name == nameToFind;
                           });
    return it != robotwb_env_right.end() ? 1 : 0; // 返回 1 如果找到，否则返回 0
}
