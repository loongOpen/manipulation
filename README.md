#### 介绍

# openloong_mw为原子技能调用包，需要结合vr_teleoperation一起使用

## openloong_mw依赖安装：

```bash
cmake>=3.23
sudo apt-get update
sudo apt-get install build-essential git zlib1g-dev
git clone https://github.com/HDFGroup/hdf5.git
cd hdf5
mkdir build
cd build
cmake ..
make -j4
sudo make install
sudo apt-get install libssl-dev
git clone https://github.com/libssh2/libssh2.git
cd libssh2
mkdir build
cd build
cmake ..
make -j4
sudo make install
```

## vr_teleoperation使用方法：

```bash
roscore
python3 vr_teleoperation/src/my_python_package/src/vr_udp_mini_gai_jilu.py
```

> 本程序可以遥操作并记录运动轨迹
> O 开始记录/终止记录并开始下一条
> P 终止记录
> 记录的轨迹文件在 `vr_teleoperation/src/my_python_package/config` 中以 `vrmocap_data_log_X.txt` 的形式保存，保存显示可能有延时，等0-30s会显示的。

- `vr_udp_mini_gai_jilu.py` 中，`self.udp_ip` 和 `self.udp_port` 是下位机的 IP 和端口
- `Mocap2Robot_quest3Controller2Qinlong.py` 中 `receive_data` 函数的 `broadcast_ip` 和 `port` 为操控器的 IP 和端口

## openloong_mw使用方法是：

```bash
roscore
source ./devel/setup.bash
rosrun openloong_mw openloongIf_br
```

> 注意：`openloongIf_br` 和 `vr_udp_mini_gai_jilu` 不能同时开启，且每次重启必须长按空格复位

---

## openloong_mw/config/yaml 是任务的选项配置，如下所示：

### Minute_Maid.yaml

```yaml
- action: "Minute_Maid" #名字在older和pose中需要有对应，如果有一部分不想开可以和下面一样注释掉
  com_left: false       #左手操作是否进行偏置
  com_right: true       #右手操作是否进行偏置
  eyes: true            #是否获取位姿估计信息
  arm_bezier: false     #手臂操作是否进行插值
  waist_bezier: false   #腰部操作是否进行插值
  speed: 1.0            #操作速度倍率
  sleep: 0.0            #操作结束后的延迟时间
# - action: "Turn_Right_Waist"
#   com_left: false
#   com_right: false
#   eyes: false
#   arm_bezier: false
#   waist_bezier: false
#   speed: 1.0
#   sleep: 0.0
# - action: "Let_Go"
#   com_left: false
#   com_right: false
#   eyes: false
#   arm_bezier: false
#   waist_bezier: false
#   speed: 1.0
#   sleep: 0.0
# - action: "Turn_Left_Waist"
#   com_left: false
#   com_right: false
#   eyes: false
#   arm_bezier: false
#   waist_bezier: false
#   speed: 1.0
#   sleep: 0.0
```

---

## pose文件是记录的轨迹信息的txt

由先前的 `vr_udp_mini_gai_jilu.py` 遥操作记录得到，其列数有 18：

- 0-6 左手状态（x、y、z、rx、ry、rz、夹爪）
- 7-13 右手状态（x、y、z、rx、ry、rz、夹爪）
- 14 执行时间
- 15-17 腰部关节0、头部关节0、头部关节1

## older 是默认物体位置信息的txt

每次采集前需记录好，共 12 列 4 行，四行一致：

- 0-5 在左臂坐标系下的目标物体位姿（x、y、z、rx、ry、rz）
- 6-11 在右臂坐标系下的目标物体位姿（x、y、z、rx、ry、rz）

收集后可以使用技能调用。

---

## 键盘按键说明

- 空格（长按）：复位
- 其他按键需要根据不同的任务在 `openloong_mw/src/comm/mission.cpp` 文件中修改

例如，本实例中对按键 2 示例如下：

```cpp
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
```

在 `lsm_main` 的位置上，以智能抓取按键 2 为例：

```cpp
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
    grasp_mission_auto_bofang_dps(dt, "Minute_Maid");  //轨迹任务执行函数
    if(robotwb.arm_control_mode == 0) {
        timer=0;
        lsm_main++;
    }
break;

case 22:  //跳到第100步：复位
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
        lsm_main = 100;   //lsm_main = 100 即为复位
    }
break;
```

---

## 修改机器人复位位置

可以在以下位置修改：

```cpp
robotwb.arm_epos_h_exp_safe
```

