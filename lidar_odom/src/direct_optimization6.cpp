#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/Point.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>

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
#include <sophus/se3.hpp>

using PointT = pcl::PointXYZ;
using PointCloudT = pcl::PointCloud<PointT>;
using namespace std;
using namespace Eigen;

std::string result_path = "/home/syx/my_lio/lidar_odom_ws/src/lidar_odom/tmp/pcl_kf_distance_1";

struct Pose3d {
    Eigen::Vector3d p;
    Eigen::Quaterniond q;

    static std::string name() { return "VERTEX_SE3:QUAT"; }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

inline std::istream& operator>>(std::istream& in, Pose3d& pose) {
    
    in >> pose.p.x() >> pose.p.y() >> pose.p.z() >> pose.q.x() >> pose.q.y() >> pose.q.z() >> pose.q.w();
    pose.q.normalize();

    return in;
}

struct LoopConstraint {
    int id_a;
    int id_b;
    Pose3d t_be;
    Eigen::Matrix<double,6,6> information;

    static std::string name() { return "EDGE_SE3:QUAT"; }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

inline std::istream& operator>>(std::istream& in, LoopConstraint& constraint) {
    
    Pose3d& t_be = constraint.t_be;
    in >> constraint.id_a >> constraint.id_b >> t_be;

    for (int i = 0; i < 6; ++i) {
        for (int j = i; j < 6; ++j) {
            in >> constraint.information(i,j);
            constraint.information(j, i) = constraint.information(i, j);
        }
    }

    return in;
}

class PGOManifold : public ceres::Manifold {
public:
    bool Plus(const double *x, const double *delta, double *x_plus_delta) const override {

        Eigen::Map<const Eigen::Matrix<double,6,1>> xi (x);
        Eigen::Map<const Eigen::Matrix<double,6,1>> dpi(delta);

        Sophus::SE3d T_x = Sophus::SE3d::exp(xi);
        Sophus::SE3d T_d = Sophus::SE3d::exp(dpi);
        Sophus::SE3d T_p = T_d * T_x; //左乘

        Eigen::Matrix<double,6,1> log6 = T_p.log();
        for (int i = 0; i < 6; ++i) {
            x_plus_delta[i] = log6[i];
        }
        return true;
    }

    bool PlusJacobian(const double* x, double* jacobian) const override {
        Eigen::Map<Eigen::Matrix<double,6,6,Eigen::RowMajor>> J(jacobian);
        J.setIdentity();
        return true;
    }

    bool Minus(const double* y, const double* x, double* y_minus_x) const override {
        
        Eigen::Map<const Eigen::Matrix<double,6,1>> yi(y);
        Eigen::Map<const Eigen::Matrix<double,6,1>> xi(x);

        Sophus::SE3d T_y = Sophus::SE3d::exp(yi);
        Sophus::SE3d T_x = Sophus::SE3d::exp(xi);
        Sophus::SE3d T_d = T_y * T_x.inverse();

        Eigen::Matrix<double,6,1> log6 = T_d.log();
        for(int i = 0; i < 6; ++i) {
            y_minus_x[i] = log6[i];
        }
        return true;
    }

    bool MinusJacobian(const double* x, double* jacobian) const override {
        Eigen::Map<Eigen::Matrix<double,6,6,Eigen::RowMajor>> J(jacobian);
        J.setIdentity();
        return true;
    }

    int AmbientSize() const override { return 6; }
    int TangentSize() const override { return 6; }
};

struct PoseGraphErrorAD {
  PoseGraphErrorAD(const Sophus::SE3d& T_meas) : T_meas_(T_meas) {}

    template <typename T>
    bool operator()(const T* const xi, const T* const xj, T* residuals) const {

        Eigen::Matrix<T,6,1> xi_T = Eigen::Map<const Eigen::Matrix<T,6,1>>(xi);
        Eigen::Matrix<T,6,1> xj_T = Eigen::Map<const Eigen::Matrix<T,6,1>>(xj);

        Sophus::SE3<T> Ti = Sophus::SE3<T>::exp(xi_T);
        Sophus::SE3<T> Tj = Sophus::SE3<T>::exp(xj_T);

        Sophus::SE3<T> Tmeas_inv = T_meas_.cast<T>().inverse();
        Sophus::SE3<T> Tij   = Ti.inverse() * Tj;
        Sophus::SE3<T> err   = Tmeas_inv * Tij;

        // Sophus::SE3<T> Tmeas = T_meas_.template cast<T>();
        // Sophus::SE3<T> Tji   = Tj.inverse() * Ti;
        // Sophus::SE3<T> err   = Tji * Tmeas;

        Eigen::Matrix<T,6,1> v = err.log();
        for (int k = 0; k < 6; ++k) {
            residuals[k] = v[k];
        }
        return true;
    }
    static ceres::CostFunction* Create(const Sophus::SE3d& T_meas) {
        return new ceres::AutoDiffCostFunction<PoseGraphErrorAD, 6, 6, 6>(
            new PoseGraphErrorAD(T_meas));
    }

    Sophus::SE3d T_meas_;
};


class PoseGraphError6D : public ceres::SizedCostFunction<6, 6, 6> {
public:
        PoseGraphError6D(const Sophus::SE3d& T_meas) : T_meas_(T_meas) {}

    virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override {

        Eigen::Map<const Eigen::Matrix<double, 6, 1>> xi(parameters[0]);
        Eigen::Map<const Eigen::Matrix<double, 6, 1>> xj(parameters[1]);
        Sophus::SE3d T_i = Sophus::SE3d::exp(xi);
        Sophus::SE3d T_j = Sophus::SE3d::exp(xj);

        Sophus::SE3d T_ij = T_i.inverse() * T_j;
        Sophus::SE3d err = T_meas_.inverse() * T_ij;
        Eigen::Matrix<double, 6, 1> v_err = err.log();
        Eigen::Map<Eigen::Matrix<double, 6, 1>> r(residuals);
        r = v_err;

        if (jacobians) {
            Eigen::Matrix3d skew_rho = Sophus::SO3d::hat(v_err.head<3>());
            Eigen::Matrix3d skew_omega = Sophus::SO3d::hat(v_err.tail<3>());

            Eigen::Matrix<double, 6, 6> I6 = Eigen::Matrix<double, 6, 6>::Identity();
            Eigen::Matrix<double,6,6> Adj = (T_j.inverse()).Adj();

            Eigen::Matrix<double, 6, 6> J_i;
            J_i.setZero();
            J_i.block<3, 3>(0, 0).noalias() = 1.0 * skew_omega;
            J_i.block<3, 3>(0, 3).noalias() = 1.0 * skew_rho;
            J_i.block<3, 3>(3, 3).noalias() = 1.0 * skew_omega;

            if (jacobians[0]) {
                Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> J_xi(jacobians[0]);
                J_xi.setZero();
                J_xi.block<6, 6>(0, 0).noalias() = -(I6 + 0.5 * J_i) * Adj;
            }

            if (jacobians[1]) {
                Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> J_xj(jacobians[1]);
                J_xj.setZero();
                J_xj.block<6, 6>(0, 0).noalias() = (I6 + 0.5 * J_i) * Adj;
            }
        }
        return true;
    }
private:
    Sophus::SE3d T_meas_;
};

void BuildOptimizationProblem(const std::map<int, Pose3d>& poses, const std::vector<LoopConstraint>& edges,
                              std::map<int, Sophus::SE3d>& optimized_map, ceres::Problem* problem) {

    auto* manifold = new PGOManifold();

    struct SE3Param { double data[6]; };
    std::map<int, SE3Param> param_map;
    for(auto const& [id, p] : poses) {
        Sophus::SE3d T(p.q, p.p);
        optimized_map[id] = T;
        Eigen::Matrix<double,6,1> xi = T.log();
        for (int k=0; k<6; ++k) {
            param_map[id].data[k] = xi[k];
        }
        problem->AddParameterBlock(param_map[id].data, 6, manifold);
    }

    problem->SetParameterBlockConstant(param_map.begin()->second.data);

    for (auto const& c : edges) {
        Sophus::SE3d Tmeas(c.t_be.q, c.t_be.p);
        // ceres::CostFunction* cost = PoseGraphErrorAD::Create(Tmeas);
        ceres::CostFunction* cost = new PoseGraphError6D(Tmeas);
        problem->AddResidualBlock(cost, nullptr, param_map[c.id_a].data, param_map[c.id_b].data);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.max_num_iterations = 200;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, problem, &summary);
    std::cout << "Optimization Summary:\n" << summary.FullReport() << "\n";

    for (auto& [id, T] : optimized_map) {
        Eigen::Matrix<double,6,1> xi;
        for (int k = 0; k < 6; ++k) {
            xi[k] = param_map[id].data[k];
        }
        T = Sophus::SE3d::exp(xi);
    }
};

bool LoadGraphFile(const std::string& filename, std::map<int,Pose3d>& out_poses, std::vector<LoopConstraint>& out_constraints)
{
    std::ifstream fin(filename);
    if (!fin.is_open()) {
        std::cerr << "无法打开文件: " << filename << "\n";
        return false;
    }

    out_poses.clear();
    out_constraints.clear();

    std::string tag;
    while (fin >> tag) {
        if (tag == Pose3d::name()) {
            int id;
            fin >> id;
            Pose3d p;
            fin >> p;
            out_poses[id] = p;
        }
        else if (tag == LoopConstraint::name()) {
            LoopConstraint lc;
            fin >> lc;
            out_constraints.push_back(lc);
        }
    }

    return true;
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "keyframe_graph_node");
    ros::NodeHandle nh;

    ros::Publisher pub_graph = nh.advertise<visualization_msgs::Marker>("keyframe_graph", 1);
    ros::Publisher pub_loop_edge = nh.advertise<visualization_msgs::Marker>("loop_edges", 1);

    std::string framefile_path = result_path + "/graph_car.g2o";
    
    std::map<int,Pose3d>         poses;
    std::vector<LoopConstraint>  edges;
    
    if (!LoadGraphFile(framefile_path, poses, edges)) {
        ROS_ERROR("failed to load graph file: %s", framefile_path.c_str());
        return -1;
    };
    ROS_INFO("Loaded %zu poses, %zu loop constraints", poses.size(), edges.size());

    std::map<int,Sophus::SE3d> optimized_map;
    for (const auto& [id, p3d] : poses) {
        optimized_map[id] = Sophus::SE3d(p3d.q, p3d.p);
    }
    
    ceres::Problem problem;
    BuildOptimizationProblem(poses, edges, optimized_map, &problem);

    std::ofstream fout(result_path + "/optimized_keyframes.txt");
    if (!fout) {
        ROS_ERROR("cannot open file for writing: %s", (result_path + "/optimized_keyframes.txt").c_str());
        return -1;
    }
    for (auto const& [id, T] : optimized_map) {
        auto t = T.translation();
        auto q = T.unit_quaternion();
        fout << id << " " << t.x() << " " << t.y() << " " << t.z() << " "
             << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
    }
    fout.close();
    ROS_INFO("saved optimized keyframes to %s", (result_path + "/optimized_keyframes.txt").c_str());

    

    ros::Rate rate(1);
    while (ros::ok()) {

        {
            visualization_msgs::Marker marker_edges;
            marker_edges.header.frame_id = "map";
            marker_edges.header.stamp    = ros::Time::now();
            marker_edges.ns              = "loop_edges";
            marker_edges.id              = 0;
            marker_edges.type            = visualization_msgs::Marker::LINE_LIST;
            marker_edges.action          = visualization_msgs::Marker::ADD;
            marker_edges.scale.x         = 0.05;
            marker_edges.color.r         = 1;
            marker_edges.color.g         = 1; 
            marker_edges.color.a         = 1;

            for (auto const& lc : edges) {
                const auto& Ta = optimized_map.at(lc.id_a);
                const auto& Tb = optimized_map.at(lc.id_b);
                geometry_msgs::Point A, B;
                A.x = Ta.translation().x();
                A.y = Ta.translation().y();
                A.z = Ta.translation().z();
                B.x = Tb.translation().x();
                B.y = Tb.translation().y();
                B.z = Tb.translation().z();
                marker_edges.points.push_back(A);
                marker_edges.points.push_back(B);
            }
            pub_loop_edge.publish(marker_edges);
        }

        {
            visualization_msgs::Marker marker_sphere;
            marker_sphere.header.frame_id = "map";
            marker_sphere.header.stamp    = ros::Time::now();
            marker_sphere.ns              = "keyframe_graph";
            marker_sphere.id              = 1;
            marker_sphere.type            = visualization_msgs::Marker::SPHERE_LIST;
            marker_sphere.action          = visualization_msgs::Marker::ADD;
            marker_sphere.scale.x = marker_sphere.scale.y = marker_sphere.scale.z = 0.3;
            marker_sphere.color.r = 1; marker_sphere.color.a = 1;

            for (auto const& [id, T] : optimized_map) {
                geometry_msgs::Point p;
                p.x = T.translation().x();
                p.y = T.translation().y();
                p.z = T.translation().z();
                marker_sphere.points.push_back(p);
            }
            pub_graph.publish(marker_sphere);
        }

        {
            visualization_msgs::Marker marker_text;
            marker_text.header.frame_id = "map";
            marker_text.header.stamp    = ros::Time::now();
            marker_text.ns              = "keyframe_labels";
            marker_text.type            = visualization_msgs::Marker::TEXT_VIEW_FACING;
            marker_text.action          = visualization_msgs::Marker::ADD;
            marker_text.scale.z         = 0.5;
            marker_text.color.r         = 1;
            marker_text.color.g         = 1;
            marker_text.color.b         = 1;
            marker_text.color.a         = 1;

            int text_id = 1000;
            for (auto const& [id, T] : optimized_map) {
                marker_text.id = text_id++;
                marker_text.pose.position.x = T.translation().x();
                marker_text.pose.position.y = T.translation().y();
                marker_text.pose.position.z = T.translation().z() + 0.5;
                marker_text.text = std::to_string(id);
                pub_graph.publish(marker_text);
            }
        }

        ros::spinOnce();
        rate.sleep();
    }

    return 0;
}
