// TorsoController.h
#ifndef TORSOCONTROLLER_H
#define TORSOCONTROLLER_H

#include "sdk.h"
#include <queue>
#include <H5Cpp.h>
#include "sdk/NeckControl.h"
#include "sdk/WaistControl.h"

// TorsoController 单例类定义
class TorsoController {
public:
	struct TorsoPose {
		double neck_angle[2];
		double waist_angle[3];
	};

	static TorsoController& getInstance() {
		static TorsoController instance;
		return instance;
	}
    TorsoPose getTorsoPose();
    void sendNeckCmd(double const neck_pose[2]);
    void sendWaistCmd(double const waist_pose[3]);

private:
	ros::NodeHandle* nh;
	ros::Publisher pub_neck;
	ros::Publisher pub_waist;

	TorsoController();
	~TorsoController();
	TorsoController(TorsoController const&) = delete;
	TorsoController& operator=(TorsoController const&) = delete;
};

#endif	// TORSOCONTROLLER_H
