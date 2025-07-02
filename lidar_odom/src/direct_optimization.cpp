#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/Point.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <sophus/se3.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

using PointT = pcl::PointXYZ;
using PointCloudT = pcl::PointCloud<PointT>;
using namespace std;
using namespace Eigen;

std::string result_path = "/home/syx/my_lio/lidar_odom_ws/src/lidar_odom/tmp/pcl_kf_distance_1";

double time_thresh = 30.0; 
double distance_thresh = 25.0; 
int min_id_interval = 50;
int skip_id = 0;

struct Keyframe {
    int id;
    double timestamp;
    Eigen::Vector3d t;
    Eigen::Quaterniond q;
};

std::vector<Keyframe> keyframes;

struct LoopConstraint {
    int id_a;                       // 起点
    int id_b;                       // 终点
    Eigen::Matrix4d T_a_to_b;       // 优化后的相对位姿
    double rmse;                    // 置信度

    LoopConstraint(int a, int b, const Eigen::Matrix4d& T, double e)
        : id_a(a), id_b(b), T_a_to_b(T), rmse(e) {}
};

Eigen::Matrix3d SkewSymmetric(const Eigen::Vector3d& v) {
    Eigen::Matrix3d m;
    m << 0, -v.z(), v.y(),
         v.z(), 0, -v.x(),
        -v.y(), v.x(), 0;
    return m;
}

Eigen::Matrix3d ExpSO3(const Eigen::Vector3d& omega) {
    double theta = omega.norm();
    if (theta < 1e-10) {
        return Eigen::Matrix3d::Identity();
    }
    
    Eigen::Vector3d axis = omega / theta;
    Eigen::Matrix3d axis_skew = SkewSymmetric(axis);

    return Eigen::Matrix3d::Identity() + std::sin(theta) * axis_skew + (1 - std::cos(theta)) * axis_skew * axis_skew;
}

Eigen::Vector3d LogSO3_q(const Eigen::Quaterniond& q) {
    Eigen::Quaterniond q_norm = q.normalized();
    double w = q_norm.w();
    Eigen::Vector3d v = q_norm.vec(); 

    double norm_v = v.norm();
    double theta = 2.0 * atan2(norm_v, w);

    if (norm_v < 1e-10) {
        return Eigen::Vector3d::Zero();
    } else {
        return theta * v.normalized();
    }
}

template <typename T>
Eigen::Matrix<T,3,1> LogSO3(const Eigen::Quaternion<T>& q) {
    // Normalize to ensure unit quaternion
    Eigen::Quaternion<T> qn = q.normalized();
    const T w = qn.w();
    const Eigen::Matrix<T,3,1> v = qn.coeffs().template tail<3>();  // 使用 coeffs().template tail<3>() 避免 const 问题
    const T norm_v = v.norm();
    const T eps = T(1e-10);
    if (norm_v < eps) {
        // small angle approx
        return T(2) * v;
    } else {
        // full log
        const T theta = T(2) * atan2(norm_v, w);
        return theta * (v / norm_v);
    }
}

struct PoseGraphErrorTerm {
    PoseGraphErrorTerm(const Eigen::Quaterniond& q_ij, const Eigen::Vector3d& t_ij)
        : q_ij_(q_ij), t_ij_(t_ij) {}

    template <typename T>
    bool operator()(const T* const q_i, const T* const t_i, const T* const q_j, const T* const t_j, T* residuals) const {
        Eigen::Map<const Eigen::Quaternion<T>> Qi(q_i);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> Ti(t_i);
        
        Eigen::Map<const Eigen::Quaternion<T>> Qj(q_j);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> Tj(t_j);

        Eigen::Quaternion<T> Qij_meas = q_ij_.cast<T>();
        Eigen::Matrix<T, 3, 1> Tij_meas = t_ij_.cast<T>();

        Eigen::Quaternion<T> Q_err = Qij_meas.inverse() * (Qi.inverse() * Qj);
        Eigen::Matrix<T, 3, 1> T_err = Qi.inverse() * (Tj - Ti) - Tij_meas;

        // Eigen::Matrix<T, 3, 1> r_rot = LogSO3(Q_err);
        // residuals[0] = r_rot(0);
        // residuals[1] = r_rot(1);
        // residuals[2] = r_rot(2);

        residuals[0] = T(2.0) * Q_err.x();
        residuals[1] = T(2.0) * Q_err.y();
        residuals[2] = T(2.0) * Q_err.z();        
        residuals[3] = T_err.x();
        residuals[4] = T_err.y();
        residuals[5] = T_err.z();
        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Quaterniond& q_ij, const Eigen::Vector3d& t_ij) {
        return (new ceres::AutoDiffCostFunction<PoseGraphErrorTerm, 6, 4, 3, 4, 3>(
            new PoseGraphErrorTerm(q_ij, t_ij)));
    }

    Eigen::Quaterniond q_ij_;
    Eigen::Vector3d t_ij_;
};

struct PoseGraphErrorTermNumeric {
    PoseGraphErrorTermNumeric(const Eigen::Quaterniond& q_ij, const Eigen::Vector3d& t_ij)
        : q_ij_(q_ij), t_ij_(t_ij) {}

    bool operator()(const double* const q_i, const double* const t_i,
                    const double* const q_j, const double* const t_j,
                    double* residuals) const {
        Eigen::Map<const Eigen::Quaterniond> Qi(q_i);
        Eigen::Map<const Eigen::Vector3d> Ti(t_i);
        
        Eigen::Map<const Eigen::Quaterniond> Qj(q_j);
        Eigen::Map<const Eigen::Vector3d> Tj(t_j);

        Eigen::Quaterniond Q_err = q_ij_.inverse() * (Qi.inverse() * Qj);
        Eigen::Vector3d T_err = Qi.inverse() * (Tj - Ti) - t_ij_;

        residuals[0] = 2.0 * Q_err.x();
        residuals[1] = 2.0 * Q_err.y();
        residuals[2] = 2.0 * Q_err.z();
        residuals[3] = T_err.x();
        residuals[4] = T_err.y();
        residuals[5] = T_err.z();
        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Quaterniond& q_ij, const Eigen::Vector3d& t_ij) {
        return new ceres::NumericDiffCostFunction<PoseGraphErrorTermNumeric, ceres::CENTRAL, 6, 4, 3, 4, 3>(
            new PoseGraphErrorTermNumeric(q_ij, t_ij));
    }

    Eigen::Quaterniond q_ij_;
    Eigen::Vector3d t_ij_;
};

// Helper function: Jacobian of quaternion log map with respect to quaternion
Eigen::Matrix<double, 3, 4> LogQuaternionJacobian(const Eigen::Quaterniond& q) {
    Eigen::Matrix<double, 3, 4> J;
    J.setZero();
    
    double w = q.w();
    Eigen::Vector3d v = q.vec();  // [x, y, z]
    double v_norm = v.norm();
    
    if (v_norm < 1e-6) {
        // Small angle approximation: log(q) ≈ 2 * [x, y, z]
        J.block<3,3>(0,0) = 2.0 * Eigen::Matrix3d::Identity();  // d/d[x,y,z]
        J.col(3).setZero();  // d/dw = 0
    } else {
        double theta = 2.0 * atan2(v_norm, w);
        Eigen::Vector3d axis = v / v_norm;
        
        // Full jacobian computation
        Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
        Eigen::Matrix3d outer = axis * axis.transpose();
        
        // d(log(q))/d[x,y,z]
        double coeff1 = theta / v_norm;
        double coeff2 = (2.0 * w) / (w*w + v_norm*v_norm) / v_norm;
        J.block<3,3>(0,0) = coeff1 * I + (coeff2 - coeff1/v_norm) * outer;
        
        // d(log(q))/dw  
        J.col(3) = -2.0 * v / (w*w + v_norm*v_norm);
    }
    
    return J;
}

// 右乘雅可比：vec(q1 ⊗ q2) = R(q2) * vec(q1)
Eigen::Matrix4d QuaternionMultRightJacobian(const Eigen::Quaterniond& q2) {
    double w = q2.w(), x = q2.x(), y = q2.y(), z = q2.z();
    Eigen::Matrix4d J;
    J <<  w, -x, -y, -z,
          x,  w,  z, -y,
          y, -z,  w,  x,
          z,  y, -x,  w;
    return J;
}

// 左乘雅可比：vec(q1 ⊗ q2) = L(q1) * vec(q2)
Eigen::Matrix4d QuaternionMultLeftJacobian(const Eigen::Quaterniond& q1) {
    double w = q1.w(), x = q1.x(), y = q1.y(), z = q1.z();
    Eigen::Matrix4d J;
    J <<  w, -x, -y, -z,
          x,  w, -z,  y,
          y,  z,  w, -x,
          z, -y,  x,  w;
    return J;
}


class PoseGraphErrorAnalytic_Quaternion : public ceres::SizedCostFunction<6, 4, 3, 4, 3> {
public:
    PoseGraphErrorAnalytic_Quaternion(const Eigen::Quaterniond& q_ij, const Eigen::Vector3d& t_ij)
        : q_ij_(q_ij), t_ij_(t_ij) {}

    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const override {

        Eigen::Map<const Eigen::Quaterniond> qi(parameters[0]);
        Eigen::Map<const Eigen::Vector3d> ti(parameters[1]);
        Eigen::Map<const Eigen::Quaterniond> qj(parameters[2]);
        Eigen::Map<const Eigen::Vector3d> tj(parameters[3]);

        // Compute residuals
        Eigen::Quaterniond qi_inv = qi.inverse();
        Eigen::Quaterniond q_rel = qi_inv * qj;  // relative rotation
        Eigen::Quaterniond q_err = q_ij_.inverse() * q_rel;  // rotation error
        Eigen::Vector3d r_rot = LogSO3_q(q_err);
        Eigen::Vector3d r_trans = qi_inv * (tj - ti) - t_ij_;

        Eigen::Map<Eigen::Matrix<double, 6, 1>> res(residuals);
        res.head<3>() = r_rot;
        res.tail<3>() = r_trans;

        if (jacobians) {
            Eigen::Matrix3d Ri = qi.toRotationMatrix();
            Eigen::Vector3d dt = tj - ti;
            
            Eigen::Matrix<double, 3, 4> J_log = LogQuaternionJacobian(q_err);
            
            if (jacobians[0]) {
                Eigen::Map<Eigen::Matrix<double,6,4,Eigen::RowMajor>> J_qi(jacobians[0]);
                J_qi.setZero();
                
                Eigen::Matrix<double, 4, 4> J_qi_inv;
                J_qi_inv.setZero();
                J_qi_inv(0,0) = -1.0; 
                J_qi_inv(1,1) = -1.0; 
                J_qi_inv(2,2) = -1.0; 
                J_qi_inv(3,3) = 1.0;

                Eigen::Quaterniond qi_inv = qi.inverse();
                Eigen::Quaterniond temp = qi_inv * qj;

                Eigen::Matrix<double, 4, 4> J_qerr_temp = QuaternionMultLeftJacobian(q_ij_.inverse());
                Eigen::Matrix<double, 4, 4> J_temp_qiinv = QuaternionMultRightJacobian(qj);
                Eigen::Matrix<double, 4, 4> J_qerr_qi = J_qerr_temp * J_temp_qiinv * J_qi_inv;
                
                J_qi.block<3,4>(0,0) = J_log * J_qerr_qi;

                Eigen::Vector3d v = tj - ti;
                double qx = qi.x(), qy = qi.y(), qz = qi.z(), qw = qi.w();
                double vx = v.x(), vy = v.y(), vz = v.z();
                
                Eigen::Matrix<double, 3, 4> J_trans_qi;
                J_trans_qi << 
                    2*(qw*vx + qz*vy - qy*vz), 2*(-qx*vx - qy*vy - qz*vz), 2*(qy*vx - qx*vy - qw*vz), 2*(qz*vx + qw*vy - qx*vz),
                    2*(qw*vy - qz*vx + qx*vz), 2*(qy*vx - qx*vy - qw*vz), 2*(-qx*vx - qy*vy - qz*vz), 2*(-qw*vx - qz*vy + qy*vz),
                    2*(qw*vz + qy*vx - qx*vy), 2*(qz*vx + qw*vy - qx*vz), 2*(qw*vx + qz*vy - qy*vz), 2*(-qx*vx - qy*vy - qz*vz);
                
                J_qi.block<3,4>(3,0) = J_trans_qi;
            }

            if (jacobians[1]) {
                Eigen::Map<Eigen::Matrix<double,6,3,Eigen::RowMajor>> J_ti(jacobians[1]);
                J_ti.setZero();
                J_ti.block<3,3>(3,0) = -Ri.transpose();
            }

            if (jacobians[2]) {
                Eigen::Map<Eigen::Matrix<double,6,4,Eigen::RowMajor>> J_qj(jacobians[2]);
                J_qj.setZero();

                Eigen::Quaterniond qi_inv = qi.inverse();
                Eigen::Matrix<double, 4, 4> J_qerr_temp = QuaternionMultLeftJacobian(q_ij_.inverse());
                Eigen::Matrix<double, 4, 4> J_qi_temp = QuaternionMultLeftJacobian(qi.inverse());
                Eigen::Matrix<double, 4, 4> J_qerr_qj = J_qerr_temp * J_qi_temp;
                J_qj.block<3,4>(0,0) = J_log * J_qerr_qj;
            }

            if (jacobians[3]) {
                Eigen::Map<Eigen::Matrix<double,6,3,Eigen::RowMajor>> J_tj(jacobians[3]);
                J_tj.setZero();
                J_tj.block<3,3>(3,0) = Ri.transpose();
            }
        }

        return true;
    }

private:
    Eigen::Quaterniond q_ij_;
    Eigen::Vector3d t_ij_;
};

// 混合雅可比版本：平移部分解析，旋转部分数值
class PoseGraphErrorHybrid_Quaternion : public ceres::SizedCostFunction<6, 4, 3, 4, 3> {
public:
    PoseGraphErrorHybrid_Quaternion(const Eigen::Quaterniond& q_ij, const Eigen::Vector3d& t_ij)
        : q_ij_(q_ij), t_ij_(t_ij) {}

    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const override {

        Eigen::Map<const Eigen::Quaterniond> qi(parameters[0]);
        Eigen::Map<const Eigen::Vector3d> ti(parameters[1]);
        Eigen::Map<const Eigen::Quaterniond> qj(parameters[2]);
        Eigen::Map<const Eigen::Vector3d> tj(parameters[3]);

        Eigen::Quaterniond q_err = q_ij_.inverse() * (qi.inverse() * qj);
        Eigen::Vector3d r_rot = LogSO3_q(q_err);
        Eigen::Vector3d r_trans = qi.inverse() * (tj - ti) - t_ij_;
        
        Eigen::Map<Eigen::Matrix<double, 6, 1>> res(residuals);
        res.head<3>() = r_rot;
        res.tail<3>() = r_trans;

        if (jacobians) {
            Eigen::Matrix3d Ri = qi.toRotationMatrix();
            Eigen::Matrix3d Rj = qj.toRotationMatrix();
            Eigen::Vector3d dt = tj - ti;

            const double eps = 1e-8;

            if (jacobians[0]) {
                Eigen::Map<Eigen::Matrix<double,6,4,Eigen::RowMajor>> J_qi(jacobians[0]);
                J_qi.setZero();
                
                for (int i = 0; i < 4; i++) {
                    Eigen::Vector4d delta = Eigen::Vector4d::Zero();
                    delta(i) = eps;
                    
                    Eigen::Quaterniond qi_plus(qi.coeffs() + delta);
                    qi_plus.normalize();
                    
                    Eigen::Quaterniond q_err_plus = q_ij_.inverse() * (qi_plus.inverse() * qj);
                    Eigen::Vector3d r_rot_plus = LogSO3_q(q_err_plus);
                    Eigen::Vector3d r_trans_plus = qi_plus.inverse() * dt - t_ij_;

                    J_qi.col(i).head<3>() = (r_rot_plus - r_rot) / eps;
                    J_qi.col(i).tail<3>() = (r_trans_plus - r_trans) / eps;
                }
            }

            if (jacobians[1]) {
                Eigen::Map<Eigen::Matrix<double,6,3,Eigen::RowMajor>> J_ti(jacobians[1]);
                J_ti.setZero();
                J_ti.bottomRows<3>() = - Ri.transpose();
            }

            if (jacobians[2]) {
                Eigen::Map<Eigen::Matrix<double,6,4,Eigen::RowMajor>> J_qj(jacobians[2]);
                J_qj.setZero();
                
                for (int i = 0; i < 4; i++) {
                    Eigen::Vector4d delta = Eigen::Vector4d::Zero();
                    delta(i) = eps;
                    
                    Eigen::Quaterniond qj_plus(qj.coeffs() + delta);
                    qj_plus.normalize();
                    
                    Eigen::Quaterniond q_err_plus = q_ij_.inverse() * (qi.inverse() * qj_plus);
                    Eigen::Vector3d r_rot_plus = LogSO3_q(q_err_plus);
                    
                    J_qj.col(i).head<3>() = (r_rot_plus - r_rot) / eps;
                    J_qj.col(i).tail<3>().setZero();
                }
            }

            if (jacobians[3]) {
                Eigen::Map<Eigen::Matrix<double,6,3,Eigen::RowMajor>> J_tj(jacobians[3]);
                J_tj.setZero();
                J_tj.bottomRows<3>() = Ri.transpose();
            }
        }

        return true;
    }

private:
    Eigen::Quaterniond q_ij_;
    Eigen::Vector3d t_ij_;
};

void OptimizePoseGraph(std::map<int, Keyframe>& keyframes,
                       const std::vector<std::pair<int, int>>& edges,
                       const std::map<std::pair<int, int>, Eigen::Isometry3d>& relative_poses) {

    ceres::Problem problem;
    for (auto& kf : keyframes) {
        problem.AddParameterBlock(kf.second.q.coeffs().data(), 4, new ceres::EigenQuaternionManifold());
        problem.AddParameterBlock(kf.second.t.data(), 3, new ceres::EuclideanManifold<3>());// .data()表示Eigen格式读取，取出其底层 double* 指针
    }

    // 固定第一个关键帧
    problem.SetParameterBlockConstant(keyframes.begin()->second.q.coeffs().data());
    problem.SetParameterBlockConstant(keyframes.begin()->second.t.data());

    for (auto& edge : edges) {
        const Eigen::Isometry3d& T = relative_poses.at(edge);
        Eigen::Quaterniond q(T.rotation());
        Eigen::Vector3d t = T.translation();

        // ceres::CostFunction* cost_function = PoseGraphErrorTerm::Create(q, t);
        // ceres::CostFunction* cost_function = PoseGraphErrorTermNumeric::Create(q, t);
        // ceres::CostFunction* cost_function = new PoseGraphErrorAnalytic_Quaternion(q, t);
        ceres::CostFunction* cost_function = new PoseGraphErrorHybrid_Quaternion(q, t);

        // nullptr 是 loss function
        problem.AddResidualBlock(cost_function, nullptr,
                                 keyframes[edge.first].q.coeffs().data(), keyframes[edge.first].t.data(),
                                 keyframes[edge.second].q.coeffs().data(), keyframes[edge.second].t.data());

        // ceres::CostFunction* cost_function = new PoseGraphErrorAnalytic(q, t);
        // problem.AddResidualBlock(cost_function, nullptr,
        //                          keyframes[edge.first].q.coeffs().data(), keyframes[edge.first].t.data(),
        //                          keyframes[edge.second].q.coeffs().data(), keyframes[edge.second].t.data());

    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.max_num_iterations = 200;

    options.function_tolerance = 1e-8;
    options.gradient_tolerance = 1e-4;
    options.parameter_tolerance = 1e-6;

    options.initial_trust_region_radius = 1e1;
    options.max_trust_region_radius = 1e6;
    options.min_trust_region_radius = 1e-8;

    options.use_nonmonotonic_steps = false;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";
}

// 读取关键帧位姿
bool LoadKeyframesFromFile(const std::string& path) {
    std::ifstream fin(path);
    if (!fin.is_open()) {
        ROS_ERROR("Cannot open keyframe file: %s", path.c_str());
        return false;
    }

    ROS_INFO("keyframe file is opened from: %s", path.c_str());

    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        Keyframe kf;
        double qx, qy, qz, qw;
        iss >> kf.id >> kf.timestamp >> kf.t.x() >> kf.t.y() >> kf.t.z() >> qx >> qy >> qz >> qw;
        kf.q = Eigen::Quaterniond(qw, qx, qy, qz);
        keyframes.push_back(kf);
    }

    return true;
}

void PublishLoopEdges(ros::Publisher& pub_graph, const std::vector<Keyframe>& keyframes, const std::vector<LoopConstraint>& loop_constraints) {
    visualization_msgs::Marker loop_edge_marker;
    loop_edge_marker.header.frame_id = "map";
    loop_edge_marker.header.stamp = ros::Time::now();
    loop_edge_marker.ns = "loop_edges";
    loop_edge_marker.id = 999; 
    loop_edge_marker.type = visualization_msgs::Marker::LINE_LIST;
    loop_edge_marker.action = visualization_msgs::Marker::ADD;
    loop_edge_marker.scale.x = 0.25;
    loop_edge_marker.color.r = 0.0;
    loop_edge_marker.color.g = 1.0;
    loop_edge_marker.color.b = 0.0;
    loop_edge_marker.color.a = 1.0;

    for (const auto& loop : loop_constraints) {
        geometry_msgs::Point p1, p2;
        const auto& kf1 = keyframes[loop.id_a-1];
        const auto& kf2 = keyframes[loop.id_b-1];

        p1.x = kf1.t.x();
        p1.y = kf1.t.y();
        p1.z = kf1.t.z();

        p2.x = kf2.t.x();
        p2.y = kf2.t.y();
        p2.z = kf2.t.z();

        loop_edge_marker.points.push_back(p1);
        loop_edge_marker.points.push_back(p2);
    }

    pub_graph.publish(loop_edge_marker);
}

void PublishKeyframeGraph(ros::Publisher& pub_graph, const std::vector<Keyframe>& keyframes) {

    visualization_msgs::Marker line_list;
    line_list.header.frame_id = "map";
    line_list.header.stamp = ros::Time::now();
    line_list.ns = "keyframe_graph";
    line_list.id = 0;
    line_list.type = visualization_msgs::Marker::LINE_LIST;
    line_list.action = visualization_msgs::Marker::ADD;
    line_list.scale.x = 0.05;
    line_list.color.r = 1.0;
    line_list.color.g = 1.0;
    line_list.color.b = 0.0;
    line_list.color.a = 1.0;

    for (size_t i = 1; i < keyframes.size(); ++i) {
        geometry_msgs::Point p1, p2;
        p1.x = keyframes[i - 1].t.x();
        p1.y = keyframes[i - 1].t.y();
        p1.z = keyframes[i - 1].t.z();
        p2.x = keyframes[i].t.x();
        p2.y = keyframes[i].t.y();
        p2.z = keyframes[i].t.z();
        line_list.points.push_back(p1);
        line_list.points.push_back(p2);
    }
    pub_graph.publish(line_list);


    for (size_t i = 0; i < keyframes.size(); ++i) {
        const auto& kf = keyframes[i];

        visualization_msgs::Marker text;
        text.header.frame_id = "map";
        text.header.stamp = ros::Time::now();
        text.ns = "keyframe_text";
        text.id = i;
        text.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
        text.action = visualization_msgs::Marker::ADD;
        text.scale.z = 0.3;
        text.color.r = 1.0;
        text.color.g = 1.0;
        text.color.b = 1.0;
        text.color.a = 1.0;
        text.pose.position.x = kf.t.x();
        text.pose.position.y = kf.t.y();
        text.pose.position.z = kf.t.z() + 0.3;
        text.text = std::to_string(kf.id);
        pub_graph.publish(text);

        visualization_msgs::Marker sphere;
        sphere.header.frame_id = "map";
        sphere.header.stamp = ros::Time::now();
        sphere.ns = "keyframe_sphere";
        sphere.id = i;
        sphere.type = visualization_msgs::Marker::SPHERE;
        sphere.action = visualization_msgs::Marker::ADD;
        sphere.pose.position.x = kf.t.x();
        sphere.pose.position.y = kf.t.y();
        sphere.pose.position.z = kf.t.z();
        sphere.scale.x = 0.5;
        sphere.scale.y = 0.5;
        sphere.scale.z = 0.5;
        sphere.color.r = 1.0;
        sphere.color.g = 0.0;
        sphere.color.b = 0.0;
        sphere.color.a = 1.0;
        pub_graph.publish(sphere);
    }
}

bool LoadLoopConstraintsFromFile(const std::string& path, std::vector<LoopConstraint>& loop_constraints) {
    std::ifstream fin(path);
    if (!fin.is_open()) {
        ROS_ERROR("Cannot open loop constraint file: %s", path.c_str());
        return false;
    }

    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        int id_a, id_b;
        iss >> id_a >> id_b;

        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        for (int i = 0; i < 4; ++i) {
            std::getline(fin, line);
            std::istringstream row_stream(line);
            for (int j = 0; j < 4; ++j) {
                row_stream >> T(i, j);
            }
        }

        if (!std::getline(fin, line)) {
            ROS_ERROR("Missing RMSE for loop constraint");
            return false;
        }
        double rmse;
        std::istringstream rmse_stream(line);
        std::string rmse_prefix;
        rmse_stream >> rmse_prefix >> rmse;

        loop_constraints.emplace_back(id_a, id_b, T, rmse);
    }

    ROS_INFO("Loaded %zu loop constraints from file.", loop_constraints.size());
    return true;
}


int main(int argc, char** argv) {
    ros::init(argc, argv, "keyframe_graph_node");
    ros::NodeHandle nh;

    ros::Publisher pub_graph = nh.advertise<visualization_msgs::Marker>("keyframe_graph", 1);
    ros::Publisher pub_loop_edge = nh.advertise<visualization_msgs::Marker>("loop_edges", 1);

    std::string keyframe_file = result_path + "/keyframes.txt";
    std::string loop_file = result_path + "/loop_constraints.txt";

    if (!LoadKeyframesFromFile(keyframe_file)) return -1;

    std::vector<LoopConstraint> loop_constraints;
    if (!LoadLoopConstraintsFromFile(loop_file, loop_constraints)) return -1;

    // 构造 odometry 边
    std::vector<std::pair<int, int>> edges;
    std::map<std::pair<int, int>, Eigen::Isometry3d> relative_poses;

    for (size_t i = 1; i < keyframes.size(); ++i) {
        const auto& kf1 = keyframes[i - 1];
        const auto& kf2 = keyframes[i];

        Eigen::Quaterniond dq = kf1.q.inverse() * kf2.q;
        Eigen::Vector3d dt = kf1.q.inverse() * (kf2.t - kf1.t);
        Eigen::Isometry3d T;
        T.linear() = dq.toRotationMatrix();
        T.translation() = dt;

        edges.emplace_back(kf1.id, kf2.id);
        relative_poses[{kf1.id, kf2.id}] = T;
    }

    // 加入回环边
    for (const auto& loop : loop_constraints) {
        Eigen::Isometry3d T(loop.T_a_to_b);
        edges.emplace_back(loop.id_a, loop.id_b);
        relative_poses[{loop.id_a, loop.id_b}] = T;
    }

    // 构造 keyframe map
    std::map<int, Keyframe> keyframes_map;
    for (const auto& kf : keyframes) {
        keyframes_map[kf.id] = kf;
    }

    OptimizePoseGraph(keyframes_map, edges, relative_poses);

    // 更新 keyframes 并保存
    for (auto& kf : keyframes) {
        kf.q = keyframes_map[kf.id].q.normalized();
        kf.t = keyframes_map[kf.id].t;
    }

    std::ofstream fout_opt(result_path + "/optimized_keyframes.txt");
    for (const auto& kf : keyframes) {
        fout_opt << kf.id << " " << kf.timestamp << " "
                 << kf.t.x() << " " << kf.t.y() << " " << kf.t.z() << " "
                 << kf.q.x() << " " << kf.q.y() << " " << kf.q.z() << " " << kf.q.w() << "\n";
    }
    fout_opt.close();

    ros::Rate rate(1);
    while (ros::ok()) {
        PublishLoopEdges(pub_loop_edge, keyframes, loop_constraints);
        PublishKeyframeGraph(pub_graph, keyframes);
        ros::spinOnce();
        rate.sleep();
    }

    return 0;
}
