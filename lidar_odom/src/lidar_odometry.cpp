#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/ndt.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/features/normal_3d.h>

#include <Eigen/Dense>
#include <deque>
#include <unordered_map>
#include <vector>
#include <cmath>

using PointT = pcl::PointXYZ;
using PointCloudT = pcl::PointCloud<PointT>;
using PointCloudPtr = pcl::PointCloud<PointT>::Ptr;

using Vec3d = Eigen::Vector3d;
using Vec6d = Eigen::Matrix<double, 6, 1>;
using Mat3d = Eigen::Matrix3d;
using Mat6d = Eigen::Matrix<double, 6, 6>;
using KeyType = Eigen::Vector3i;

Eigen::Matrix4d T_curr_last = Eigen::Matrix4d::Identity();
Eigen::Matrix4d curr_pose_w = Eigen::Matrix4d::Identity();  // 当前位姿
Eigen::Matrix4d last_pose_w = Eigen::Matrix4d::Identity();     // 上一帧位姿

PointCloudPtr last_cloud = nullptr;         // 上一帧点云
PointCloudPtr global_map = nullptr;       // 全局地图
PointCloudPtr local_map = nullptr;
PointCloudPtr localmap_cloud = nullptr;
std::deque<PointCloudPtr> local_map_frames;

int max_local_frames;  // 最多保留帧数
double keyframe_distance_thresh;
double keyframe_angle_thresh_deg;

// 算法参数
int max_iterations;
double convergence_threshold;
float global_voxel_size;                   // 全局地图降采样大小
float local_voxel_size;             // 局部地图降采样大小
float source_voxel_size;            // 源点云降采样大小


// NDT参数
float ndt_resolution;
float ndt_step_size;
float ndt_trans_epsilon;
float ndt_voxel_size;
int ndt_max_iterations;

bool use_pcl_ndt;
bool use_local_map;

// 发布器
ros::Publisher pub_odom;
ros::Publisher pub_path;
ros::Publisher pub_global_map;
ros::Publisher pub_local_map;
ros::Publisher pub_debug_marker;
nav_msgs::Path laser_path;  // 轨迹路径
ros::Publisher pub_source;
ros::Publisher pub_target;
ros::Publisher pub_aligned;
ros::Publisher pub_localmap;

Mat3d ComputeCovariance(const std::vector<Vec3d>& points, const Vec3d& mean) {
    Mat3d cov = Mat3d::Zero();
    for (const auto& pt : points) {
        Vec3d delta = pt - mean;
        cov += delta * delta.transpose();
    }
    return cov / points.size();
}

// 计算反对称矩阵
Eigen::Matrix3d SkewSymmetric(const Eigen::Vector3d& v) {
    Eigen::Matrix3d m;
    m << 0, -v.z(), v.y(),
         v.z(), 0, -v.x(),
        -v.y(), v.x(), 0;
    return m;
}

// SO(3)
Eigen::Matrix3d ExpSO3(const Eigen::Vector3d& omega) {
    double theta = omega.norm();
    if (theta < 1e-10) {
        return Eigen::Matrix3d::Identity();
    }
    
    Eigen::Vector3d axis = omega / theta;
    Eigen::Matrix3d axis_skew = SkewSymmetric(axis);

    return Eigen::Matrix3d::Identity() + std::sin(theta) * axis_skew + (1 - std::cos(theta)) * axis_skew * axis_skew;
}

bool isMatrixValid(const Eigen::Matrix4d& T) {
    return T.allFinite();
}

// 降采样点云
PointCloudPtr DownsampleCloud(const PointCloudPtr& cloud, float leaf_size) {
    pcl::VoxelGrid<PointT> voxel;
    voxel.setLeafSize(leaf_size, leaf_size, leaf_size);
    voxel.setInputCloud(cloud);

    PointCloudPtr filtered(new PointCloudT);
    voxel.filter(*filtered);
    return filtered;
}

// // 高斯牛顿法ICP配准
bool GnAlignPoint2Plane(const PointCloudPtr& source, const PointCloudPtr& target, Eigen::Matrix4d& T_output) {
    if (!source || !target || source->empty() || target->empty()) return false;
    
    // ros::Time t_norm_start = ros::Time::now();

    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());
    pcl::NormalEstimation<PointT, pcl::Normal> ne;
    ne.setInputCloud(target);
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
    ne.setSearchMethod(tree);
    ne.setKSearch(20);  // 邻域大小
    ne.compute(*normals);

    // ros::Duration t_norm_time = ros::Time::now() - t_norm_start;
    // ROS_INFO("Normal estimation time: %.3f ms", t_norm_time.toSec() * 1000.0);
    // ros::Time t_kdtree_start = ros::Time::now();

    pcl::KdTreeFLANN<PointT>::Ptr kdtree(new pcl::KdTreeFLANN<PointT>());
    kdtree->setInputCloud(target);

    // ros::Duration t_kdtree_time = ros::Time::now() - t_kdtree_start;
    // ROS_INFO("KdTree build time: %.3f ms", t_kdtree_time.toSec() * 1000.0);

    Eigen::Matrix4d T = T_output;
    ros::Time t_iter_start = ros::Time::now();
    for (int iter = 0; iter < max_iterations; ++iter) {

        Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
        Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Zero();

        for (size_t i = 0; i < source->size(); ++i) {
            const auto& all_curr_p_w = source->points[i];
            Eigen::Vector4d curr_p_w(all_curr_p_w.x, all_curr_p_w.y, all_curr_p_w.z, 1.0);
            Eigen::Vector3d trans_curr_p_w =  (T * curr_p_w).head<3>();

            
            PointT search_point;
            search_point.x = trans_curr_p_w.x();
            search_point.y = trans_curr_p_w.y();
            search_point.z = trans_curr_p_w.z();

            std::vector<int> indices(1);
            std::vector<float> dists(1);

            if (kdtree->nearestKSearch(search_point, 1, indices, dists) == 0)
                continue;

            int idx = indices[0];
            const auto& pt = target->points[idx];
            const auto& n = normals->points[idx];

            Eigen::Vector3d last_p_w(pt.x, pt.y, pt.z);
            Eigen::Vector3d ni(n.normal_x, n.normal_y, n.normal_z);

            if (ni.norm() < 1e-3) continue;

            // 残差：点到面距离
            double r = (trans_curr_p_w - last_p_w).dot(ni);
            if (std::abs(r) > 1.0) continue;

            // 雅可比
            Eigen::Matrix<double, 1, 6> J;
            J.block<1,3>(0,0) = -ni.transpose() * T.block<3,3>(0,0) * SkewSymmetric(curr_p_w.head<3>());
            J.block<1,3>(0,3) = ni.transpose();

            H += J.transpose() * J;
            b += J.transpose() * (-r);

        }

        Eigen::Matrix<double, 6, 1> dx = H.ldlt().solve(b);
        if (dx.norm() < convergence_threshold) {
            // ROS_INFO("Point2Plane converged at iter %d with delta %.6f", iter, dx.norm());
            break;
        }

        Eigen::Matrix3d dR = ExpSO3(dx.head<3>());
        Eigen::Vector3d dt = dx.tail<3>();

        T.block<3,3>(0,0) = T.block<3,3>(0,0) * dR;
        T.block<3,1>(0,3) += dt;
        T(3, 3) = 1.0;
    

    }
    // ros::Duration t_iter_time = ros::Time::now() - t_iter_start;
    // ROS_INFO("Iter time: %.3f ms", t_iter_time.toSec() * 1000.0);

    T_output = T;
    return true;
}

// bool GnAlignPoint2Plane(const PointCloudPtr& source, const PointCloudPtr& target, Eigen::Matrix4d& T_output) {
//     if (!source || !target || source->empty() || target->empty()) return false;

//     // ros::Time t_kdtree_start = ros::Time::now();
    
//     pcl::KdTreeFLANN<PointT>::Ptr kdtree(new pcl::KdTreeFLANN<PointT>());
//     kdtree->setInputCloud(target);

//     // ros::Duration t_kdtree_time = ros::Time::now() - t_kdtree_start;
//     // ROS_INFO("KdTree build time: %.3f ms", t_kdtree_time.toSec() * 1000.0);
//     // ros::Time t_iter_start = ros::Time::now();

//     Eigen::Matrix4d T = T_output;
//     for (int iter = 0; iter < max_iterations; ++iter) {

//         Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
//         Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Zero();

//         int effective_count = 0;

//         for (size_t i = 0; i < source->size(); ++i) {
//             const auto& all_curr_p_w = source->points[i];
//             Eigen::Vector4d curr_p_w(all_curr_p_w.x, all_curr_p_w.y, all_curr_p_w.z, 1.0);
//             Eigen::Vector3d trans_curr_p_w =  (T * curr_p_w).head<3>();

          
//             PointT p_query(trans_curr_p_w.x(), trans_curr_p_w.y(), trans_curr_p_w.z());

//             std::vector<int> indices(5);
//             std::vector<float> dists(5);

//             int K = 5;
//             if (kdtree->nearestKSearch(p_query, K, indices, dists) >= 3) {
//                 Eigen::Vector3d pj(target->points[indices[0]].x, target->points[indices[0]].y, target->points[indices[0]].z);
//                 Eigen::Vector3d pl(target->points[indices[1]].x, target->points[indices[1]].y, target->points[indices[1]].z);
//                 Eigen::Vector3d pm(target->points[indices[2]].x, target->points[indices[2]].y, target->points[indices[2]].z);
//                 Eigen::Vector3d n = (pj - pl).cross(pj - pm);
                
//                 if (n.norm() < 1e-6) continue; // 平面退化
//                 n.normalize();
                
//                 // 残差：点到面距离
//                 double r = n.dot(trans_curr_p_w - pj);
//                 if (std::abs(r) > 1.0) continue;

//                 // 雅可比
//                 Eigen::Matrix<double, 1, 6> J;
//                 J.block<1,3>(0,0) = -n.transpose() * T.block<3,3>(0,0) * SkewSymmetric(curr_p_w.head<3>());
//                 J.block<1,3>(0,3) = n.transpose();

//                 H += J.transpose() * J;
//                 b += J.transpose() * (-r);
//                 effective_count++;
//             }
//         }

//         if (effective_count < 10) {
//             ROS_WARN("Too few correspondences: %d", effective_count);
//             continue;
//         }

//         Eigen::Matrix<double, 6, 1> dx = H.ldlt().solve(b);
//         if (dx.norm() < convergence_threshold) {
//             break;
//         }

//         Eigen::Matrix3d dR = ExpSO3(dx.head<3>());
//         Eigen::Vector3d dt = dx.tail<3>();

//         T.block<3,3>(0,0) = T.block<3,3>(0,0) * dR;
//         T.block<3,1>(0,3) += dt;
//         T(3, 3) = 1.0;
    
        
//     }

//     // ros::Duration t_iter_time = ros::Time::now() - t_iter_start;
//     // ROS_INFO("Iter time: %.3f ms", t_iter_time.toSec() * 1000.0);

//     T_output = T;
//     return true;
// }

// NDT配准函数
bool pclNdtAlign(const PointCloudPtr& source, const PointCloudPtr& target, Eigen::Matrix4d& T_output) {
    if (!source || !target || source->empty() || target->empty()) return false;

    PointCloudPtr filtered_source(new PointCloudT());
    PointCloudPtr filtered_target(new PointCloudT());

    pcl::ApproximateVoxelGrid<PointT> voxel;

    voxel.setLeafSize(ndt_voxel_size, ndt_voxel_size, ndt_voxel_size);
    voxel.setInputCloud(source);
    voxel.filter(*filtered_source);
    voxel.setInputCloud(target);
    voxel.filter(*filtered_target);

    // NDT配准
    pcl::NormalDistributionsTransform<PointT, PointT> ndt;
    ndt.setResolution(ndt_resolution);
    ndt.setStepSize(ndt_step_size);
    ndt.setTransformationEpsilon(ndt_trans_epsilon);
    ndt.setMaximumIterations(ndt_max_iterations);

    ndt.setInputSource(filtered_source);
    ndt.setInputTarget(filtered_target);

    PointCloudT output;
    Eigen::Matrix4f init_guess = T_output.cast<float>();  // 使用当前位姿作为初始猜测

    ndt.align(output, init_guess);

    if (!ndt.hasConverged()) {
        return false;
    }

    T_output = ndt.getFinalTransformation().cast<double>();
    return true;
}

bool IsKeyframe(const Eigen::Matrix4d& last_pose_w, const Eigen::Matrix4d& curr_pose_w) {
    Eigen::Matrix4d delta = last_pose_w.inverse() * curr_pose_w;
    double trans = delta.block<3,1>(0,3).norm();

    Eigen::Matrix3d R = delta.block<3,3>(0,0);
    double angle_rad = std::acos(std::min(1.0, std::max(-1.0, (R.trace() - 1.0) / 2.0))); //trace(R)=1+2cosθ
    double angle_deg = angle_rad * 180.0 / M_PI;

    return trans > keyframe_distance_thresh || angle_deg > keyframe_angle_thresh_deg;
}

void UpdateGlobalMap(const PointCloudPtr& current_cloud, const Eigen::Matrix4d& curr_pose_w) {
    // 初始化地图
    if (!global_map) {
        PointCloudPtr transformed(new PointCloudT());
        pcl::transformPointCloud(*current_cloud, *transformed, curr_pose_w);
        global_map = DownsampleCloud(transformed, global_voxel_size);
        return;
    }

    // 当前帧变换到全局系
    PointCloudPtr cloud_transformed(new PointCloudT());
    pcl::transformPointCloud(*current_cloud, *cloud_transformed, curr_pose_w);

    *global_map += *cloud_transformed;
    global_map = DownsampleCloud(global_map, global_voxel_size);

}

void UpdateLocalMap(const PointCloudPtr& localmap_cloud, const Eigen::Matrix4d& curr_pose_w) {

    if (!local_map) {
        PointCloudPtr tmp(new PointCloudT());
        pcl::transformPointCloud(*localmap_cloud, *tmp, curr_pose_w);
        local_map = DownsampleCloud(tmp, local_voxel_size);
        return;
    }
    PointCloudPtr tmp(new PointCloudT());
    pcl::transformPointCloud(*localmap_cloud, *tmp, curr_pose_w);
    *local_map += *tmp;
    local_map = DownsampleCloud(local_map, local_voxel_size);
}

// 发布里程计和轨迹
void PublishOdometry(const ros::Time& stamp, const Eigen::Matrix4d& curr_pose_w) {
    
    // 提取旋转和平移
    Eigen::Matrix3d R = curr_pose_w.block<3,3>(0,0);
    Eigen::Vector3d t = curr_pose_w.block<3,1>(0,3);
    
    // 转换为四元数
    Eigen::Quaterniond q(R);

    // 发布里程计
    nav_msgs::Odometry odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = "map";
    odom.child_frame_id = "base_link";
    odom.pose.pose.orientation.x = q.x();
    odom.pose.pose.orientation.y = q.y();
    odom.pose.pose.orientation.z = q.z();
    odom.pose.pose.orientation.w = q.w();
    odom.pose.pose.position.x = t.x();
    odom.pose.pose.position.y = t.y();
    odom.pose.pose.position.z = t.z();
    pub_odom.publish(odom);

    // 发布轨迹
    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header = odom.header;
    pose_stamped.pose = odom.pose.pose;
    laser_path.header.stamp = stamp;
    laser_path.header.frame_id = "map";
    laser_path.poses.push_back(pose_stamped);
    pub_path.publish(laser_path);
}


// 发布地图
void PublishMaps(const ros::Time& stamp, const PointCloudPtr& global_map, const PointCloudPtr& local_map) {
    // 发布全局地图
    if (global_map) {
        sensor_msgs::PointCloud2 map_msg;
        pcl::toROSMsg(*global_map, map_msg);
        map_msg.header.stamp = stamp;
        map_msg.header.frame_id = "map";
        pub_global_map.publish(map_msg);
    }

    if (local_map) {
        sensor_msgs::PointCloud2 local_msg;
        pcl::toROSMsg(*local_map, local_msg);
        local_msg.header.stamp = stamp;
        local_msg.header.frame_id = "map"; 
        pub_local_map.publish(local_msg);
    }
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr ColorizePointCloud(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud,
    uint8_t r, uint8_t g, uint8_t b) {
    
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    for (const auto& pt : input_cloud->points) {
        pcl::PointXYZRGB pt_rgb;
        pt_rgb.x = pt.x;
        pt_rgb.y = pt.y;
        pt_rgb.z = pt.z;
        pt_rgb.r = r;
        pt_rgb.g = g;
        pt_rgb.b = b;
        colored_cloud->push_back(pt_rgb);
    }
    return colored_cloud;
}

// 点云回调函数
void CloudCallback(const sensor_msgs::PointCloud2::ConstPtr& msg) {
    
    bool success = false;

    static int frame_count = 0;
    frame_count++;
    if(frame_count % 100== 0){
        ROS_INFO("global_cloud size: %lu", global_map->size());
    }
    // 转换为PCL点云
    PointCloudPtr cloud(new PointCloudT());
    pcl::fromROSMsg(*msg, *cloud);

    auto &current_cloud = cloud;

    // 第一帧处理
    if (!last_cloud) {

        last_cloud = current_cloud;
        last_pose_w = curr_pose_w;   // 改变current_pose的初始值，last_pose对应改变
        
        local_map_frames.push_back(current_cloud);
        localmap_cloud.reset(new PointCloudT(*current_cloud));

        // ROS_INFO("Initialized with first frame.");
        return;
    }

    PointCloudPtr source_cloud = DownsampleCloud(current_cloud, source_voxel_size);

    if (use_pcl_ndt) {
        ros::Time start = ros::Time::now();
        success = pclNdtAlign(source_cloud, last_cloud, T_curr_last);
        // success = CustomNdtAlign(used_cloud, last_cloud, T_curr_last);
        ros::Duration elapsed = ros::Time::now() - start;
        if(frame_count % 10 == 0){
            ROS_INFO("NDT Runtime: %.3f ms", elapsed.toSec() * 1000.0);
        }
    } else {
        ros::Time start = ros::Time::now();
        if(use_local_map){
            success = GnAlignPoint2Plane(source_cloud, localmap_cloud, T_curr_last);
        } else{
            // success = GnAlignPoint2Point(used_cloud, last_cloud, T_curr_last);
            success = GnAlignPoint2Plane(source_cloud, last_cloud, T_curr_last);
        }
        ros::Duration elapsed = ros::Time::now() - start;
        if(frame_count % 10 == 0){
            ROS_INFO("ICPsuccess! Runtime: %.3f ms", elapsed.toSec() * 1000.0);
        }
        // ROS_INFO_STREAM("T_curr_last: \n" << T_curr_last);
    }

    PointCloudPtr target_for_viz(new PointCloudT(*last_cloud));  // 深拷贝上一帧

    if (success) {
        // 更新当前位姿
        // curr_pose_w = T_curr_last * last_pose_w;
        curr_pose_w = last_pose_w * T_curr_last;
        static int keyframe_num = 0;

        if (IsKeyframe(last_pose_w, curr_pose_w)) {
            PointCloudPtr current_in_map(new PointCloudT());
            pcl::transformPointCloud(*current_cloud, *current_in_map, curr_pose_w);
            local_map_frames.push_back(current_in_map);
            
            if (local_map_frames.size() > max_local_frames) {
                local_map_frames.pop_front();
            }
            keyframe_num++;
            ROS_INFO("Added a new keyframe. Total: %d", keyframe_num);
            // ROS_INFO("Current keyframe count: %lu", local_map_frames.size());
        }

        localmap_cloud->clear();
        for (const auto& frame : local_map_frames) {
            *localmap_cloud += *frame;
        }
        Eigen::Matrix4f map_to_lidar = curr_pose_w.inverse().cast<float>();
        pcl::transformPointCloud(*localmap_cloud, *localmap_cloud, map_to_lidar);
        localmap_cloud = DownsampleCloud(localmap_cloud, local_voxel_size);
        // ROS_INFO("localmap_cloud size: %lu", localmap_cloud->size());

        last_pose_w = curr_pose_w;
        
        // 计算 aligned 点云
        PointCloudPtr aligned_cloud(new PointCloudT());
        pcl::transformPointCloud(*current_cloud, *aligned_cloud, T_curr_last);

        PointCloudPtr source_in_map(new PointCloudT());
        PointCloudPtr target_in_map(new PointCloudT());
        PointCloudPtr aligned_in_map(new PointCloudT());

        pcl::transformPointCloud(*current_cloud, *source_in_map, curr_pose_w);
        pcl::transformPointCloud(*target_for_viz, *target_in_map, curr_pose_w);  
        pcl::transformPointCloud(*aligned_cloud, *aligned_in_map, curr_pose_w);     

        // 构建 RGB 点云
        auto source_rgb = ColorizePointCloud(source_in_map, 255, 0, 0);       // 红色
        auto target_rgb = ColorizePointCloud(target_in_map, 0, 255, 0);      // 绿色
        auto aligned_rgb = ColorizePointCloud(aligned_in_map, 0, 0, 255);    // 蓝色

        // 转换为 ROS 消息
        sensor_msgs::PointCloud2 msg_source, msg_target, msg_aligned;
        pcl::toROSMsg(*source_rgb, msg_source);
        pcl::toROSMsg(*target_rgb, msg_target);
        pcl::toROSMsg(*aligned_rgb, msg_aligned);

        // 设置时间戳和坐标系
        msg_source.header.stamp = msg->header.stamp;
        msg_target.header.stamp = msg->header.stamp;
        msg_aligned.header.stamp = msg->header.stamp;

        msg_source.header.frame_id = "map";
        msg_target.header.frame_id = "map";
        msg_aligned.header.frame_id = "map";

        // 发布
        pub_source.publish(msg_source);
        pub_target.publish(msg_target);
        pub_aligned.publish(msg_aligned);

        // 更新地图
        UpdateGlobalMap(current_cloud, curr_pose_w);
        UpdateLocalMap(localmap_cloud, curr_pose_w);
        
        // 发布数据
        PublishOdometry(msg->header.stamp, curr_pose_w);
        PublishMaps(msg->header.stamp, global_map, local_map);

    }
    else {
        ROS_WARN("Registration failed. Skipping this frame.");
        return;
    }
    // 更新上一帧点云
    last_cloud.reset(new PointCloudT(*current_cloud));

}


int main(int argc, char** argv) {
    // 初始化ROS节点
    ros::init(argc, argv, "odometry");
    ros::NodeHandle nh;

    nh.param("use_pcl_ndt", use_pcl_ndt, false);
    nh.param("use_local_map", use_local_map, false);
    nh.param("max_iterations", max_iterations, 10);
    nh.param("convergence_threshold", convergence_threshold, 1e-4);
    nh.param("global_voxel_size", global_voxel_size, 0.1f);
    nh.param("local_voxel_size", local_voxel_size, 0.2f);
    nh.param("source_voxel_size", source_voxel_size, 0.5f);
    
    nh.param("max_local_frames", max_local_frames, 10);
    nh.param("keyframe_distance_thresh", keyframe_distance_thresh, 0.5);
    nh.param("keyframe_angle_thresh_deg", keyframe_angle_thresh_deg, 10.0);

    nh.param("ndt_resolution", ndt_resolution, 1.5f);
    nh.param("ndt_voxel_size", ndt_voxel_size, 0.5f);
    nh.param("ndt_step_size", ndt_step_size, 0.1f);
    nh.param("ndt_trans_epsilon", ndt_trans_epsilon, 0.01f);
    nh.param("ndt_max_iterations", ndt_max_iterations, 10);

    
    // 创建发布器
    pub_odom = nh.advertise<nav_msgs::Odometry>("/laser_odom", 100);
    pub_path = nh.advertise<nav_msgs::Path>("/laser_path", 100);
    pub_global_map = nh.advertise<sensor_msgs::PointCloud2>("/global_map", 10);
    pub_local_map = nh.advertise<sensor_msgs::PointCloud2>("/local_map", 10);
    pub_debug_marker = nh.advertise<visualization_msgs::Marker>("/debug_marker", 10);
    pub_source = nh.advertise<sensor_msgs::PointCloud2>("/source_cloud", 10);
    pub_target = nh.advertise<sensor_msgs::PointCloud2>("/target_cloud", 10);
    pub_aligned = nh.advertise<sensor_msgs::PointCloud2>("/aligned_cloud", 10);
    pub_localmap = nh.advertise<sensor_msgs::PointCloud2>("/localmap_cloud", 10);

    
    // 创建订阅者
    ros::Subscriber sub_cloud = nh.subscribe("/velodyne_cloud", 100, CloudCallback);
    
    std::cout << "使用" << (use_pcl_ndt ? "NDT" : "ICP") << "配准" << std::endl;
    
    // 循环处理回调
    ros::spin();
    return 0;
}