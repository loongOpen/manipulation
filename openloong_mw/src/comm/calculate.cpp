#include <openloong_br.h>

Eigen::Matrix4d openloong_br::quaternionToMatrix(const Eigen::Vector4d& quat)
{
    w_quat = quat(0);
    x_quat = quat(1);
    y_quat = quat(2);
    z_quat = quat(3);
    mat_quat.setZero();
    mat_quat << 1 - 2 * y_quat * y_quat - 2 * z_quat * z_quat,   2 * x_quat * y_quat - 2 * w_quat * z_quat,   2 * x_quat * z_quat + 2 * w_quat * y_quat,   0,
                 2 * x_quat * y_quat + 2 * w_quat * z_quat,       1 - 2 * x_quat * x_quat - 2 * z_quat * z_quat, 2 * y_quat * z_quat - 2 * w_quat * x_quat,   0,
                 2 * x_quat * z_quat - 2 * w_quat * y_quat,       2 * y_quat * z_quat + 2 * w_quat * x_quat,   1 - 2 * x_quat * x_quat - 2 * y_quat * y_quat, 0,
                 0,                                               0,                                           0,                                           1;
    return mat_quat;
}

Eigen::Matrix4d openloong_br::poseToHomogeneousMatrix(const Eigen::Vector3d& translation, const Eigen::Vector3d& euler_angles)
{
    // 构造旋转矩阵
    rotation_to_homogeneous_Matrix.setZero();
    HomogeneousMatrix = Eigen::Matrix4d::Identity();
    rotation_to_homogeneous_Matrix = Eigen::AngleAxisd(euler_angles[0], Eigen::Vector3d::UnitX()) *
                                    Eigen::AngleAxisd(euler_angles[1], Eigen::Vector3d::UnitY()) *
                                    Eigen::AngleAxisd(euler_angles[2], Eigen::Vector3d::UnitZ());
    // 构造齐次变换矩阵
    HomogeneousMatrix.block<3, 3>(0, 0) = rotation_to_homogeneous_Matrix;
    HomogeneousMatrix.block<3, 1>(0, 3) = translation;

    return HomogeneousMatrix;
}

Eigen::Matrix4d openloong_br::position_to_max(const std::vector<float>& position)
{
    for (size_t i = 0; i < 3; i++)
    {
        position_to_max_xyz[i] = position[i];
        position_to_max_rpy[i] = position[i+3];
    }
    return poseToHomogeneousMatrix(position_to_max_xyz,position_to_max_rpy);
}

std::vector<float> openloong_br::max_to_position(const Eigen::Matrix4d& max)
{
    max_to_position_taget_right_p = max.block<3, 1>(0, 3);
    max_to_position_taget_right_r = matrix3dToEulerAnglesZYX(max.block<3, 3>(0, 0));

    for (size_t i = 0; i < 3; i++)
    {
        max_to_position_positions[i] = max_to_position_taget_right_p[i];
        max_to_position_positions[i+3] = max_to_position_taget_right_r[i];
    }
    return max_to_position_positions;
}

Eigen::Matrix4d openloong_br::quet_to_max(const std::vector<float>& quet)
{

    for (size_t i = 0; i < 3; i++)
    {
        quet_to_max_quet_p[i] = quet[i];
    }
    for (size_t i = 0; i < 4; i++)
    {
        quet_to_max_quet_q[i] = quet[i+3];
    }
    S_E = quaternionToMatrix(quet_to_max_quet_q);
    S_E.block<3, 1>(0, 3) = quet_to_max_quet_p;
    return S_E;
}

std::vector<float> openloong_br::max_to_quet(const Eigen::Matrix4d& matrix)
{
    max_to_quet_result.clear();
    // 提取xyz位置
    max_to_quet_translation = matrix.block<3,1>(0, 3);
    max_to_quet_result.push_back(max_to_quet_translation.x());
    max_to_quet_result.push_back(max_to_quet_translation.y());
    max_to_quet_result.push_back(max_to_quet_translation.z());

    // 提取旋转部分，并转换为四元数
    max_to_quet_quaternion = Eigen::Quaterniond(matrix.block<3,3>(0,0));;

    max_to_quet_result.push_back(max_to_quet_quaternion.w());
    max_to_quet_result.push_back(max_to_quet_quaternion.x());
    max_to_quet_result.push_back(max_to_quet_quaternion.y());
    max_to_quet_result.push_back(max_to_quet_quaternion.z());
    return max_to_quet_result;
}

Eigen::Vector3d openloong_br::matrix3dToEulerAnglesZYX(const Eigen::Matrix3d& m3dr)
{
	beta_y=atan2(m3dr(0,2),sqrt(m3dr(0,0)*m3dr(0,0)+m3dr(0,1)*m3dr(0,1)));
	alpha_z=atan2(-m3dr(0,1)/cos(beta_y),m3dr(0,0)/cos(beta_y));
	gamma_x=atan2(-m3dr(1,2)/cos(beta_y),m3dr(2,2)/cos(beta_y));
	if(abs(beta_y - M_PI/2) < 10e-4)
	{
		gamma_x = 0;
		alpha_z = atan2(m3dr(1,0),m3dr(1,1));
	}
    if(abs(beta_y + M_PI/2) < 10e-4)
	{
		gamma_x = 0;
		alpha_z = atan2(m3dr(1,0),m3dr(1,1));
	}
	if (gamma_x>M_PI)
	{
		gamma_x=gamma_x-2*M_PI;
	}
	if (gamma_x<-M_PI)
	{
		gamma_x=gamma_x+2*M_PI;
	}
	if (beta_y>M_PI)
	{
		beta_y=beta_y-2*M_PI;
	}
	if (beta_y<-M_PI)
	{
		beta_y=beta_y+2*M_PI;
	}
	if (alpha_z>M_PI)
	{
		alpha_z=alpha_z-2*M_PI;
	}
	if (alpha_z<-M_PI)
	{
		alpha_z=alpha_z+2*M_PI;
	}
return Eigen::Vector3d(gamma_x, beta_y, alpha_z);
}

Eigen::VectorXd openloong_br::bezierInterpolationEEF(const Eigen::VectorXd& P0, const Eigen::VectorXd& P1, double t)
{
    for (size_t i = 0; i < 3; i++)
    {
        bezier_original_LPose0[i] = P0[i];
        bezier_original_LRot0[i] = P0[i+3];
        bezier_original_LPose1[i] = P1[i];
        bezier_original_LRot1[i] = P1[i+3];

        bezier_original_RPose0[i] = P0[i+6];
        bezier_original_RRot0[i] = P0[i+9];
        bezier_original_RPose1[i] = P1[i+6];
        bezier_original_RRot1[i] = P1[i+9];
    }
    bezier_original_Lmax0 = poseToHomogeneousMatrix(bezier_original_LPose0 ,bezier_original_LRot0);
    bezier_original_Lmax1 = poseToHomogeneousMatrix(bezier_original_LPose1 ,bezier_original_LRot1);
    bezier_original_Rmax0 = poseToHomogeneousMatrix(bezier_original_RPose0 ,bezier_original_RRot0);
    bezier_original_Rmax1 = poseToHomogeneousMatrix(bezier_original_RPose1 ,bezier_original_RRot1);
    // 将旋转矩阵转换为四元数
    bezier_original_LQ0 = Eigen::Quaterniond(bezier_original_Lmax0.block<3,3>(0, 0));
    bezier_original_LQ1 = Eigen::Quaterniond(bezier_original_Lmax1.block<3,3>(0, 0));
    bezier_original_RQ0 = Eigen::Quaterniond(bezier_original_Rmax0.block<3,3>(0, 0));
    bezier_original_RQ1 = Eigen::Quaterniond(bezier_original_Rmax1.block<3,3>(0, 0));
    // 使用贝塞尔曲线对位置进行插值
    LinterpolatedPosition = (1 - t) * bezier_original_LPose0 + t * bezier_original_LPose1;
    RinterpolatedPosition = (1 - t) * bezier_original_RPose0 + t * bezier_original_RPose1;
    // 使用Slerp对四元数进行插值
    LinterpolatedOrientation = bezier_original_LQ0.slerp(t, bezier_original_LQ1);
    LinterpolatedRotationMatrix = LinterpolatedOrientation.toRotationMatrix();
    LinterpolatedRxyz = matrix3dToEulerAnglesZYX(LinterpolatedRotationMatrix);
    RinterpolatedOrientation = bezier_original_RQ0.slerp(t, bezier_original_RQ1);
    RinterpolatedRotationMatrix = RinterpolatedOrientation.toRotationMatrix();
    RinterpolatedRxyz = matrix3dToEulerAnglesZYX(RinterpolatedRotationMatrix);
    for (size_t i = 0; i < 3; i++)
    {
        bezierInterpolationEEF_arm_pose[i] = LinterpolatedPosition[i];
        bezierInterpolationEEF_arm_pose[i+3] = LinterpolatedRxyz[i];
        bezierInterpolationEEF_arm_pose[i+6] = RinterpolatedPosition[i];
        bezierInterpolationEEF_arm_pose[i+9] = RinterpolatedRxyz[i];
    }
    return bezierInterpolationEEF_arm_pose;
}

Eigen::VectorXd openloong_br::bezierInterpolationEEF_dan(const Eigen::VectorXd& P0, const Eigen::VectorXd& P1, double t)
{
    for (size_t i = 0; i < 3; i++)
    {
        bezier_dan_Pose0[i] = P0[i];
        bezier_dan_Rot0[i] = P0[i+3];
        bezier_dan_Pose1[i] = P1[i];
        bezier_dan_Rot1[i] = P1[i+3];

    }
    bezier_dan_max0 = poseToHomogeneousMatrix(bezier_dan_Pose0 ,bezier_dan_Rot0);
    bezier_dan_max1 = poseToHomogeneousMatrix(bezier_dan_Pose1 ,bezier_dan_Rot1);
    // 将旋转矩阵转换为四元数
    bezier_dan_Q0 = Eigen::Quaterniond(bezier_dan_max0.block<3,3>(0, 0));
    bezier_dan_Q1 = Eigen::Quaterniond(bezier_dan_max1.block<3,3>(0, 0));
    // 使用贝塞尔曲线对位置进行插值
    bezier_dan_interpolatedPosition[0] = (1.0 - t) * bezier_dan_Pose0[0] + t * bezier_dan_Pose1[0];
    bezier_dan_interpolatedPosition[1] = (1.0 - t) * bezier_dan_Pose0[1] + t * bezier_dan_Pose1[1];
    bezier_dan_interpolatedPosition[2] = (1.0 - t) * bezier_dan_Pose0[2] + t * bezier_dan_Pose1[2];

    // 使用Slerp对四元数进行插值
    bezier_dan_interpolatedOrientation = bezier_dan_Q0.slerp(t, bezier_dan_Q1);
    bezier_dan_interpolatedRotationMatrix = bezier_dan_interpolatedOrientation.toRotationMatrix();
    bezier_dan_interpolatedRxyz = matrix3dToEulerAnglesZYX(bezier_dan_interpolatedRotationMatrix);
    for (size_t i = 0; i < 3; i++)
    {
        bezier_dan_arm_pose[i] = bezier_dan_interpolatedPosition[i];
        bezier_dan_arm_pose[i+3] = bezier_dan_interpolatedRxyz[i];
    }
    return bezier_dan_arm_pose;
}

std::vector<float> openloong_br::bianhua(const Eigen::Matrix4d& binahuan, const std::vector<float>& position)
{
    return max_to_position(binahuan * position_to_max(position));
}

Eigen::VectorXd openloong_br::bezierInterpolationJoints(const Eigen::VectorXd& P0, const Eigen::VectorXd& P1, double t) 
{
    return ((1 - t) * P0 + t * P1);
}

float openloong_br::bezierInterpolationwaist(const float& P0, const float& P1, double t) 
{
    return ((1 - t) * P0 + t * P1);
}

