// ArmController.h
#ifndef ARMCONTROLLER_H
#define ARMCONTROLLER_H

#include "sdk.h"
#include "k_OLR2.h"
#include <queue>
#include <H5Cpp.h>
#include "sdk/AlohaCmd.h"
#include "sdk/RosControl.h"
#include "sdk/BoudingBox.h"
#include "sdk/Environment.h"
#include "sdk/EndPos.h"
#include "sdk/GetEnvironment.h"

// ArmController 单例类定义
class ArmController {
public:
	struct ArmEEPose {
		double left_arm_tip[7];
		double right_arm_tip[7];
		double left_hand_fingers[6];
		double right_hand_fingers[6];
	};

	static ArmController& getInstance() {
		static ArmController instance;
		return instance;
	}
	ArmEEPose getArmEEPose();
	sdk::Environment getObjectPose();
	void sendArmEECmd(double const left_pose[6], double const right_pose[6], double const grasp_left, double const grasp_right);
	void sendArmCmd(double const left_pose[7], double const right_pose[7], double const left_hand[6], double const right_hand[6]);
	void replayRecord(std::string const& filename, double const frequency);
	bool uploadModel(std::string const& local_file, int const mode);
	bool startInference(std::string const& parameter, int const mode);
	bool stopInference(int const mode);
	std::vector<float> getArmPose();
private:
	ros::NodeHandle* nh1, *nh2;
	ros::Publisher pub_tip;
	ros::Publisher pub_angle;
	ros::ServiceClient environment_client;

	ArmController();
	~ArmController();
	void setCommonFields(sdk::RosControl& msg);
	void setCommonAngles(sdk::AlohaCmd& msg);
	std::queue<std::vector<double>> readJointAngles(std::string const& filename, std::string const& dataset_name);
	ArmController(ArmController const&) = delete;
	ArmController& operator=(ArmController const&) = delete;
};

#endif	// ARMCONTROLLER_H
