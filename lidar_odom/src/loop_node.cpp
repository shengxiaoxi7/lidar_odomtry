#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Path.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <fstream>
#include <sstream>
#include <vector>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>

using PointT = pcl::PointXYZ;
using PointCloudT = pcl::PointCloud<PointT>;
using PointCloudPtr = pcl::PointCloud<PointT>::Ptr;

// 订阅关键帧位姿话题
ros::Publisher pub_marker;
ros::Publisher pub_path;

int skip_id = 20;
int min_id_interval = 30;
double time_thresh = 10.0;
double distance_thresh = 30.0;
std::string result_path = "/home/syx/my_lio/lidar_odom_ws/src/lidar_odom/tmp/loop";

struct Keyframe {
    int id;
    double timestamp;
    Eigen::Vector3d t;
    Eigen::Quaterniond q;
    PointCloudPtr cloud;  // 追加：关键帧对应的点云
};


struct LoopCandidate {
    int id_a, id_b;
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

std::vector<Keyframe> keyframes;


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

bool GnAlignPoint2Plane(const PointCloudT::Ptr& cloud_a, const PointCloudT::Ptr& cloud_b, const LoopCandidate& loop,
                        Eigen::Matrix4d& T_refined, double& rmse_out) {
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

void KeyframePoseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg) {
    static int id_counter = 0;

    Keyframe kf;
    kf.id = id_counter++;
    kf.timestamp = msg->header.stamp.toSec();
    kf.t = Eigen::Vector3d(msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
    kf.q = Eigen::Quaterniond(
        msg->pose.orientation.w,
        msg->pose.orientation.x,
        msg->pose.orientation.y,
        msg->pose.orientation.z
    );

    keyframes.push_back(kf);
}

void KeyframeCloudCallback(const geometry_msgs::PoseStamped::ConstPtr& msg) {
    static int id_counter = 0;

    Keyframe kf;
    kf.id = id_counter++;
    kf.timestamp = msg->header.stamp.toSec();
    kf.t = Eigen::Vector3d(msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
    kf.q = Eigen::Quaterniond(
        msg->pose.orientation.w,
        msg->pose.orientation.x,
        msg->pose.orientation.y,
        msg->pose.orientation.z
    );

    keyframes.push_back(kf);
}

void PublishVisualization() {
    // 1. 清除旧 Marker
    visualization_msgs::Marker clear_all;
    clear_all.action = visualization_msgs::Marker::DELETEALL;
    pub_marker.publish(clear_all);

    if (keyframes.size() < 2) return;

    ros::Time stamp = ros::Time::now();

    // 2. 发布轨迹线
    visualization_msgs::Marker line_list;
    line_list.header.frame_id = "map";
    line_list.header.stamp = stamp;
    line_list.ns = "trajectory_line";
    line_list.id = 0;
    line_list.type = visualization_msgs::Marker::LINE_STRIP;
    line_list.action = visualization_msgs::Marker::ADD;
    line_list.scale.x = 0.05;
    line_list.color.r = 1.0;
    line_list.color.g = 1.0;
    line_list.color.b = 0.0;
    line_list.color.a = 1.0;

    for (const auto& kf : keyframes) {
        geometry_msgs::Point p;
        p.x = kf.t.x();
        p.y = kf.t.y();
        p.z = kf.t.z();
        line_list.points.push_back(p);
    }
    pub_marker.publish(line_list);

    // 3. 发布最新关键帧小球
    const auto& kf = keyframes.back();
    visualization_msgs::Marker sphere;
    sphere.header.frame_id = "map";
    sphere.header.stamp = stamp;
    sphere.ns = "keyframe_sphere";
    sphere.id = 1;
    sphere.type = visualization_msgs::Marker::SPHERE;
    sphere.action = visualization_msgs::Marker::ADD;
    sphere.pose.position.x = kf.t.x();
    sphere.pose.position.y = kf.t.y();
    sphere.pose.position.z = kf.t.z();
    sphere.scale.x = 0.2;
    sphere.scale.y = 0.2;
    sphere.scale.z = 0.2;
    sphere.color.r = 1.0;
    sphere.color.g = 0.0;
    sphere.color.b = 0.0;
    sphere.color.a = 1.0;
    pub_marker.publish(sphere);

    // 4. 发布文字标签
    visualization_msgs::Marker text;
    text.header.frame_id = "map";
    text.header.stamp = stamp;
    text.ns = "keyframe_text";
    text.id = 2;
    text.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    text.action = visualization_msgs::Marker::ADD;
    text.pose.position.x = kf.t.x();
    text.pose.position.y = kf.t.y();
    text.pose.position.z = kf.t.z() + 0.3;
    text.scale.z = 0.3;
    text.color.r = 1.0;
    text.color.g = 1.0;
    text.color.b = 1.0;
    text.color.a = 1.0;
    text.text = std::to_string(kf.id);
    pub_marker.publish(text);

    // 5. 发布 Path 消息（用于轨迹线）
    nav_msgs::Path path;
    path.header.frame_id = "map";
    path.header.stamp = stamp;
    for (const auto& kf : keyframes) {
        geometry_msgs::PoseStamped pose;
        pose.header.frame_id = "map";
        pose.header.stamp = stamp;
        pose.pose.position.x = kf.t.x();
        pose.pose.position.y = kf.t.y();
        pose.pose.position.z = kf.t.z();
        pose.pose.orientation.x = kf.q.x();
        pose.pose.orientation.y = kf.q.y();
        pose.pose.orientation.z = kf.q.z();
        pose.pose.orientation.w = kf.q.w();
        path.poses.push_back(pose);
    }
    pub_path.publish(path);
}

void PublishLoopEdges(const std::vector<LoopConstraint>& loops) {
    visualization_msgs::Marker loop_line;
    loop_line.header.frame_id = "map";
    loop_line.header.stamp = ros::Time::now();
    loop_line.ns = "loop_edges";
    loop_line.id = 10;
    loop_line.type = visualization_msgs::Marker::LINE_LIST;
    loop_line.action = visualization_msgs::Marker::ADD;
    loop_line.scale.x = 0.05;
    loop_line.color.r = 0.0;
    loop_line.color.g = 1.0;
    loop_line.color.b = 1.0;
    loop_line.color.a = 1.0;

    for (const auto& loop : loops) {
        geometry_msgs::Point p1, p2;
        p1.x = keyframes[loop.id_a].t.x();
        p1.y = keyframes[loop.id_a].t.y();
        p1.z = keyframes[loop.id_a].t.z();
        p2.x = keyframes[loop.id_b].t.x();
        p2.y = keyframes[loop.id_b].t.y();
        p2.z = keyframes[loop.id_b].t.z();
        loop_line.points.push_back(p1);
        loop_line.points.push_back(p2);
    }

    pub_marker.publish(loop_line);
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "keyframe_loop_node");
    ros::NodeHandle nh;

    // 订阅关键帧话题
    ros::Subscriber sub_kf_pose = nh.subscribe("/keyframe_pose", 100, KeyframePoseCallback);
    ros::Subscriber sub_kf_cloud = nh.subscribe("/keyframe_cloud", 100, KeyframeCloudCallback);

    // 发布轨迹 Marker 和 Path
    pub_marker = nh.advertise<visualization_msgs::Marker>("keyframe_graph", 10);
    pub_path   = nh.advertise<nav_msgs::Path>("keyframe_path", 1);

    ros::Rate rate(1.0);  // 1Hz 可视化频率
    static int last_detect_id = -100;

    while (ros::ok()) {
        ros::spinOnce();
        PublishVisualization();

        if (!keyframes.empty() && keyframes.back().id - last_detect_id >= 5) {
            last_detect_id = keyframes.back().id;

            std::vector<LoopCandidate> loop_candidates;
            DetectLoopCandidates(keyframes, loop_candidates);

            std::vector<LoopConstraint> loop_constraints;
            for (const auto& loop : loop_candidates) {

                PointCloudPtr cloud_a = keyframes[loop.id_a].cloud;
                PointCloudPtr cloud_b = keyframes[loop.id_b].cloud;

                if (!cloud_a || !cloud_b || cloud_a->empty() || cloud_b->empty()) {
                    ROS_WARN("Empty cloud for loop (%d, %d)", loop.id_a, loop.id_b);
                    continue;
                }

                Eigen::Matrix4d T_refined;
                double rmse;
                if (GnAlignPoint2Plane(cloud_a, cloud_b, loop, T_refined, rmse)) {
                    if (rmse < 0.5) {
                        loop_constraints.emplace_back(loop.id_a, loop.id_b, T_refined, rmse);
                    }
                }
            }

            PublishLoopEdges(loop_constraints);
        }

        rate.sleep();
    }

    return 0;
}