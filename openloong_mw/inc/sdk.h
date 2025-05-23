// sdk.h
#ifndef SDK_H
#define SDK_H

#include <ros/ros.h>
#include <ros/callback_queue.h>
#include <libssh2_sftp.h>
#include <fstream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <condition_variable>
#include "sdk/PuppetState.h"
#include "sdk/ManageNode.h"

// 单例类
class SDK {
public:
	static SDK& createInstance() {
		static SDK instance;
		return instance;
	}
	static bool uploadFile(std::string const& local_file, std::string const& remote_folder);
	static bool manageNode(std::string const& node_name, std::string const& package_name, std::string const& executable_name, std::string const& parameter, int const action);
	static sdk::PuppetState getPuppetState();
private:
	ros::NodeHandle* nh1, * nh2;
	static ros::CallbackQueue cq;
	static ros::Subscriber sub;
	static sdk::PuppetState puppet_state;
	static bool ready;
	static ros::ServiceClient manage_node_client;

	// 私有构造函数，确保只能通过 createInstance() 访问
	SDK();
	// 私有析构函数，确保在程序结束时释放资源
	~SDK();
	static void jointStateCallback(sdk::PuppetState::ConstPtr const& msg);
	// 禁止拷贝构造和赋值
	SDK(SDK const&) = delete;
	SDK& operator=(SDK const&) = delete;
};

#endif	// SDK_H
