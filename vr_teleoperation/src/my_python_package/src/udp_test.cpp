#include <iostream>
#include <vector>
#include <deque>
#include <mutex>
#include <thread>
#include <chrono>
#include <cmath>
#include <optional>
#include <atomic>
#include <Eigen/Dense>
#include <ros/ros.h>
#include <sys/socket.h>  // 套接字函数
#include <arpa/inet.h>   // 地址转换
#include <netinet/in.h>  // 套接字地址

#define SERVER_PORT 5015
#define UDP_IP "192.168.1.119"
#define UDP_PORT 3334
#define BUFFER_SIZE 8888
#define BROADCAST_IP "255.255.255.255"
#define BROADCAST_PORT 5005
#define MAX_RETRIES 5

using Eigen::Matrix3f;
using Eigen::Vector3f;

class VRMocapManager {
public:
    VRMocapManager() : stop_flag(false), udp_ip(UDP_IP), udp_port(UDP_PORT), retry_count(0) {
        ros::init(argc, argv, "udp_test");
        nh = ros::NodeHandle();
    }

    ~VRMocapManager() {
        stop_flag = true;
        if (data_receive_thread.joinable()) data_receive_thread.join();
    }

    void run() {
        data_receive_thread = std::thread(&VRMocapManager::receive_data, this);
        std::cout << "启动主循环" << std::endl;
        ros::Rate loop_rate(60);  // 60 Hz

        while (ros::ok() && !stop_flag) {
            auto data = get_data();
            if (data.has_value()) {
                process_and_send(data.value());
            }
            loop_rate.sleep();
        }
    }

private:
    std::atomic<bool> stop_flag;
    std::string udp_ip;
    int udp_port;
    int retry_count;
    std::thread data_receive_thread;
    ros::NodeHandle nh;
    std::mutex shared_resource_lock;
    std::deque<std::vector<float>> data_queue;

    // 接收数据
    void receive_data() {
        while (!stop_flag) {
            int sock = socket(AF_INET, SOCK_DGRAM, 0);
            if (sock < 0) {
                std::cerr << "接收套接字创建失败。" << std::endl;
                return;
            }
            setsockopt(sock, SOL_SOCKET, SO_BROADCAST, &broadcast_option, sizeof(broadcast_option));
            struct sockaddr_in server_address;
            memset(&server_address, 0, sizeof(server_address));
            server_address.sin_family = AF_INET;
            server_address.sin_port = htons(SERVER_PORT);
            server_address.sin_addr.s_addr = INADDR_ANY;

            if (bind(sock, (struct sockaddr *)&server_address, sizeof(server_address)) < 0) {
                std::cerr << "绑定套接字失败。" << std::endl;
                close(sock);
                return;
            }

            struct timeval timeout;
            timeout.tv_sec = 1;
            timeout.tv_usec = 0;
            setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));

            try {
                if (retry_count >= MAX_RETRIES) {
                    std::cout << "重新广播消息以重新连接..." << std::endl;
                    struct sockaddr_in broadcast_addr;
                    memset(&broadcast_addr, 0, sizeof(broadcast_addr));
                    broadcast_addr.sin_family = AF_INET;
                    broadcast_addr.sin_port = htons(BROADCAST_PORT);
                    inet_pton(AF_INET, BROADCAST_IP, &broadcast_addr.sin_addr);
                    sendto(sock, "Hello, devices!", strlen("Hello, devices!"), 0, (struct sockaddr *)&broadcast_addr, sizeof(broadcast_addr));
                    retry_count = 0;
                }

                char buffer[BUFFER_SIZE];
                int bytes_received = recvfrom(sock, buffer, BUFFER_SIZE, 0, nullptr, nullptr);
                if (bytes_received > 0) {
                    std::vector<float> data(buffer, buffer + bytes_received / sizeof(float));
                    std::lock_guard<std::mutex> lock(shared_resource_lock);
                    data_queue.push_back(data);
                    retry_count = 0;
                } else if (bytes_received < 0) {
                    std::cerr << "无数据接收" << std::endl;
                    retry_count++;
                }
            } catch (...) {
                std::cerr << "数据接收错误" << std::endl;
                break;
            }
            close(sock);
        }
    }

    // 获取接收的数据
    std::optional<std::vector<float>> get_data() {
        std::lock_guard<std::mutex> lock(shared_resource_lock);
        if (!data_queue.empty()) {
            auto data_out = data_queue.front();
            data_queue.pop_front();
            return data_out;
        }
        return std::nullopt;
    }

    // 处理并发送数据
    void process_and_send(const std::vector<float>& received_data) {
        if (received_data.size() < 18) return;

        // 提取传入的数据
        std::vector<float> left_wrist_t = {received_data[0], received_data[1], received_data[2]};
        std::vector<float> zyx_left_robot = {received_data[3], received_data[4], received_data[5]};
        std::vector<float> right_wrist_t = {received_data[6], received_data[7], received_data[8]};
        std::vector<float> zyx_right_robot = {received_data[9], received_data[10], received_data[11]};
        std::vector<float> joint_hand_left = {received_data[12], received_data[13], received_data[14]};
        std::vector<float> joint_hand_right = {received_data[15], received_data[16], received_data[17]};

        send_udp_message(zyx_left_robot, left_wrist_t, zyx_right_robot, right_wrist_t, joint_hand_left, joint_hand_right, udp_ip, udp_port);
    }

    // 发送 UDP 消息
    void send_udp_message(const std::vector<float>& zyx_left_robot, const std::vector<float>& left_wrist_t,
                          const std::vector<float>& zyx_right_robot, const std::vector<float>& right_wrist_t,
                          const std::vector<float>& joint_hand_left, const std::vector<float>& joint_hand_right,
                          const std::string& udp_ip, int udp_port) {
        int sock = socket(AF_INET, SOCK_DGRAM, 0);
        if (sock < 0) {
            std::cerr << "创建套接字失败。" << std::endl;
            return;
        }
        struct sockaddr_in server_address;
        memset(&server_address, 0, sizeof(server_address));
        server_address.sin_family = AF_INET;
        server_address.sin_port = htons(udp_port);
        inet_pton(AF_INET, udp_ip.c_str(), &server_address.sin_addr);

        float data[16] = {
            left_wrist_t[0], left_wrist_t[1], left_wrist_t[2], zyx_left_robot[0], zyx_left_robot[1], zyx_left_robot[2],
            right_wrist_t[0], right_wrist_t[1], right_wrist_t[2], zyx_right_robot[0], zyx_right_robot[1], zyx_right_robot[2],
            joint_hand_left[2], joint_hand_right[2], 99.0f, 99.0f
        };
    std::cout<<"left_xyz:   "<<left_wrist_t[0]<<"   "<<left_wrist_t[1]<<"   "<<left_wrist_t[2]<<std::endl;
    std::cout<<"left_rpy:   "<<zyx_left_robot[0]<<"   "<<zyx_left_robot[1]<<"   "<<zyx_left_robot[2]<<std::endl;
    std::cout<<"right_xyz:   "<<right_wrist_t[0]<<"   "<<right_wrist_t[1]<<"   "<<right_wrist_t[2]<<std::endl;
    std::cout<<"right_rpy:   "<<zyx_right_robot[0]<<"   "<<zyx_right_robot[1]<<"   "<<zyx_right_robot[2]<<std::endl;
        int bytes_sent = sendto(sock, data, sizeof(data), 0, (struct sockaddr*)&server_address, sizeof(server_address));
        if (bytes_sent < 0) {
            std::cerr << "发送 UDP 消息失败。" << std::endl;
        } else {
            std::cout << "已发送 UDP 数据到 " << udp_ip << ":" << udp_port << std::endl;
        }
        close(sock);
    }
};

int main(int argc, char** argv) {

    VRMocapManager vrMocapManager;
    vrMocapManager.run();
    return 0;
}
