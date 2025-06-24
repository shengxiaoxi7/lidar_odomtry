// 只用于回环检测，并构建回环边
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

using PointT = pcl::PointXYZ;
using PointCloudT = pcl::PointCloud<PointT>;

std::string result_path = "/home/syx/my_lio/lidar_odom_ws/src/lidar_odom/tmp/car";

double time_thresh = 30.0; 
double distance_thresh = 25.0; 
int min_id_interval = 50;
int skip_id = 50;

struct Keyframe {
    int id;
    double timestamp;
    Eigen::Vector3d t;
    Eigen::Quaterniond q;
};

std::vector<Keyframe> keyframes;

struct LoopCandidate {
    int id_a;
    int id_b;
    Eigen::Matrix4d T_a_to_b;

    LoopCandidate(int a, int b, const Eigen::Matrix4d& T) : id_a(a), id_b(b), T_a_to_b(T) {}
};

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

bool IsLoopCandidate(const Keyframe& a, const Keyframe& b, double distance_thresh, double time_thresh) {
    if (std::abs(a.timestamp - b.timestamp) < time_thresh) return false; // 排除时间上相近的帧
    double dist = (a.t - b.t).norm();
    return dist < distance_thresh;
}

void DetectLoopCandidates(const std::vector<Keyframe>& keyframes, std::vector<LoopCandidate>& loop_candidates) {
    const Keyframe* check_first = nullptr;
    const Keyframe* check_second = nullptr;

    for (size_t i = 0; i < keyframes.size(); ++i) {
        const auto& kf_first = keyframes[i];

        if (check_first && std::abs(kf_first.id - check_first->id) <= skip_id) continue;

        for (size_t j = 0; j < i; ++j) {
            const auto& kf_second = keyframes[j];

            if (check_second && std::abs(kf_second.id - check_second->id) <= skip_id) continue;

            if (std::abs(kf_first.id - kf_second.id) < min_id_interval) continue;

            if (IsLoopCandidate(kf_first, kf_second, distance_thresh, time_thresh)) {
                Eigen::Matrix4d T1 = Eigen::Matrix4d::Identity();
                T1.block<3, 3>(0, 0) = kf_first.q.toRotationMatrix();
                T1.block<3, 1>(0, 3) = kf_first.t;

                Eigen::Matrix4d T2 = Eigen::Matrix4d::Identity();
                T2.block<3, 3>(0, 0) = kf_second.q.toRotationMatrix();
                T2.block<3, 1>(0, 3) = kf_second.t;

                Eigen::Matrix4d relative_pose = T1.inverse() * T2;

                loop_candidates.emplace_back(kf_first.id, kf_second.id, relative_pose);

                check_first = &kf_first;
                check_second = &kf_second;
            }
        }
    }

    std::cout << "Detected " << loop_candidates.size() << " loop candidates." << std::endl;
}

bool GnAlignPoint2Plane(const PointCloudT::Ptr& cloud_a, const PointCloudT::Ptr& cloud_b, const LoopCandidate& loop, Eigen::Matrix4d& T_refined, double& rmse_out) {
    if (!cloud_a || !cloud_b || cloud_a->empty() || cloud_b->empty()) return false;

    pcl::KdTreeFLANN<PointT>::Ptr kdtree(new pcl::KdTreeFLANN<PointT>());
    kdtree->setInputCloud(cloud_b);

    Eigen::Matrix4d T = loop.T_a_to_b;
    double residual_sum = 0.0;
    int effective_count = 0;

    for (int iter = 0; iter < 20; ++iter) {
        Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
        Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Zero();

        effective_count = 0; 
        residual_sum = 0.0;

        for (size_t i = 0; i < cloud_a->size(); ++i) {
            const auto& pt = cloud_a->points[i];
            Eigen::Vector4d pa(pt.x, pt.y, pt.z, 1.0);
            Eigen::Vector3d transformed = (T * pa).head<3>();

            PointT query_pt(transformed.x(), transformed.y(), transformed.z());
            std::vector<int> indices(5);
            std::vector<float> dists(5);

            if (kdtree->nearestKSearch(query_pt, 5, indices, dists) >= 3) {
                Eigen::Vector3d pj(cloud_b->points[indices[0]].x, cloud_b->points[indices[0]].y, cloud_b->points[indices[0]].z);
                Eigen::Vector3d pl(cloud_b->points[indices[1]].x, cloud_b->points[indices[1]].y, cloud_b->points[indices[1]].z);
                Eigen::Vector3d pm(cloud_b->points[indices[2]].x, cloud_b->points[indices[2]].y, cloud_b->points[indices[2]].z);

                Eigen::Vector3d n = (pj - pl).cross(pj - pm);
                if (n.norm() < 1e-6) continue;
                n.normalize();

                double r = n.dot(transformed - pj);
                if (std::abs(r) > 1.0) continue;

                Eigen::Matrix<double, 1, 6> J;
                J.block<1,3>(0,0) = -n.transpose() * T.block<3,3>(0,0) * SkewSymmetric(pa.head<3>());
                J.block<1,3>(0,3) = n.transpose();

                H += J.transpose() * J;
                b += J.transpose() * (-r);
                residual_sum += r * r;
                effective_count++;
            }
        }

        if (effective_count < 10) {
            std::cout << " Too few effective correspondences." << std::endl;
            return false;
        }

        Eigen::Matrix<double, 6, 1> dx = H.ldlt().solve(b);
        if (dx.norm() < 0.01) {
            break;
        }

        Eigen::Matrix3d dR = ExpSO3(dx.head<3>());
        Eigen::Vector3d dt = dx.tail<3>();
        T.block<3,3>(0,0) = T.block<3,3>(0,0) * dR;
        T.block<3,1>(0,3) += dt;
    }

    T_refined = T;
    rmse_out = std::sqrt(residual_sum / effective_count);
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

int main(int argc, char** argv) {
    ros::init(argc, argv, "keyframe_graph_node");
    ros::NodeHandle nh;

    ros::Publisher pub_graph = nh.advertise<visualization_msgs::Marker>("keyframe_graph", 1);
    ros::Publisher pub_loop_edge = nh.advertise<visualization_msgs::Marker>("loop_edges", 1);

    std::string keyframe_file = result_path + "/keyframes.txt";


    if (!LoadKeyframesFromFile(keyframe_file)) {
        return -1;
    }

    std::vector<LoopCandidate> loop_candidates;
    DetectLoopCandidates(keyframes, loop_candidates);

    std::vector<LoopConstraint> loop_constraints;
    for (const auto& loop : loop_candidates) {
        std::string cloud_file_a = result_path + "/" + std::to_string(loop.id_a) + ".pcd";
        std::string cloud_file_b = result_path + "/" + std::to_string(loop.id_b) + ".pcd";

        PointCloudT::Ptr cloud_a(new PointCloudT);
        PointCloudT::Ptr cloud_b(new PointCloudT);

        if (pcl::io::loadPCDFile(cloud_file_a, *cloud_a) == -1) {
            std::cerr << "Failed to load " << cloud_file_a << std::endl;
            continue;
        }
        if (pcl::io::loadPCDFile(cloud_file_b, *cloud_b) == -1) {
            std::cerr << "Failed to load " << cloud_file_b << std::endl;
            continue;
        }

        Eigen::Matrix4d T_refined;
        double rmse;
        if (GnAlignPoint2Plane(cloud_a, cloud_b, loop, T_refined, rmse)) {
            std::cout << "Loop detected between " << loop.id_a << " and " << loop.id_b << ", RMSE: " << rmse << std::endl;
            if (rmse < 0.4) {
                loop_constraints.emplace_back(loop.id_a, loop.id_b, T_refined, rmse);
            }
        } else {
            std::cout << "Failed to align loop between " << loop.id_a << " and " << loop.id_b << std::endl;
        }
    }
    std::cout << "Total loop constraints: " << loop_constraints.size() << std::endl;

    // std::ofstream fout(result_path + "/loop_constraints.txt");
    // for (const auto& c : loop_constraints) {
    //     fout << c.id_a << " " << c.id_b << "\n";
    //     fout << c.T_a_to_b << "\n";
    //     fout << "rmse: " << c.rmse << "\n\n";
    // }
    // fout.close();

    ros::Rate rate(1);  // 每秒发布一次
    while (ros::ok()) {
        PublishLoopEdges(pub_loop_edge, keyframes, loop_constraints);
        PublishKeyframeGraph(pub_graph, keyframes);
        ros::spinOnce();
        rate.sleep();
    }

    return 0;
}