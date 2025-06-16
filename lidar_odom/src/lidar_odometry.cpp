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




// 全局变量
PointCloudPtr last_cloud = nullptr;         // 上一帧点云
Eigen::Matrix4d current_pose = Eigen::Matrix4d::Identity();  // 当前位姿
Eigen::Matrix4d last_pose = Eigen::Matrix4d::Identity();     // 上一帧位姿
PointCloudPtr global_map = nullptr;       // 全局地图
Eigen::Matrix4d T_curr_last = Eigen::Matrix4d::Identity();

// 算法参数
int max_iterations;
double convergence_threshold;
float voxel_size;                   // 全局地图降采样大小
float local_voxel_size;             // 局部地图降采样大小


// NDT参数
float ndt_resolution;
float ndt_step_size;
float ndt_trans_epsilon;
float ndt_voxel_size;
int ndt_max_iterations;

bool use_pcl_ndt;

// 发布器
ros::Publisher pub_odom;
ros::Publisher pub_path;
ros::Publisher pub_global_map;
ros::Publisher pub_debug_marker;
nav_msgs::Path laser_path;  // 轨迹路径
ros::Publisher pub_source;
ros::Publisher pub_target;
ros::Publisher pub_aligned;


struct VoxelData {
    std::vector<int> indices;
    Vec3d mu;
    Mat3d sigma;
    Mat3d info;
};

//------------------ 配置项 ------------------//
int min_pts_in_voxel = 5;
double convergence_eps = 1e-4;
double residual_threshold = 3.0;

//------------------ 哈希函数 ------------------//
namespace std {
    template<>
    struct hash<KeyType> {
        size_t operator()(const KeyType& k) const {
            return std::hash<int>()(k[0]) ^ std::hash<int>()(k[1]) ^ std::hash<int>()(k[2]);
        }
    };
}

//------------------ 工具函数 ------------------//
KeyType ToKey(const Vec3d& pt) {
    return KeyType((int)std::floor(pt[0]), (int)std::floor(pt[1]), (int)std::floor(pt[2]));
}

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

Eigen::Vector3d LogSO3(const Eigen::Matrix3d& R) {
    double cos_theta = (R.trace() - 1.0) / 2.0;
    double theta = std::acos(std::min(std::max(cos_theta, -1.0), 1.0));  // clamp to [-1, 1]

    if (theta < 1e-10) {
        return Eigen::Vector3d::Zero();
    }

    Eigen::Matrix3d lnR = (theta / (2.0 * std::sin(theta))) * (R - R.transpose());
    return Eigen::Vector3d(lnR(2,1), lnR(0,2), lnR(1,0));  // vee operator
}

// SE（3）
Eigen::Matrix4d ExpSE3(const Eigen::Matrix<double, 6, 1>& xi) {
    Eigen::Vector3d rho = xi.tail<3>();   // 平移量
    Eigen::Vector3d phi = xi.head<3>();   // 旋转量
    double theta = phi.norm();

    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d J = Eigen::Matrix3d::Identity();

    if (theta < 1e-10) {
        // θ → 0，使用泰勒展开
        R = Eigen::Matrix3d::Identity();
        J = Eigen::Matrix3d::Identity();
    } else {
        Eigen::Matrix3d phi_hat = SkewSymmetric(phi);
        R = ExpSO3(phi);

        double theta2 = theta * theta;
        double theta3 = theta2 * theta;

        J = Eigen::Matrix3d::Identity()
            + (1 - std::cos(theta)) / theta2 * phi_hat
            + (theta - std::sin(theta)) / theta3 * (phi_hat * phi_hat);
    }

    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.topLeftCorner<3, 3>() = R;
    T.topRightCorner<3, 1>() = J * rho;

    return T;
}


Eigen::Matrix<double, 6, 1> LogSE3(const Eigen::Matrix4d& T) {
    Eigen::Matrix3d R = T.topLeftCorner<3,3>();
    Eigen::Vector3d t = T.topRightCorner<3,1>();

    Eigen::Vector3d phi = LogSO3(R);
    double theta = phi.norm();

    Eigen::Matrix3d J_inv = Eigen::Matrix3d::Identity();
    if (theta < 1e-10) {
        // J ≈ I → J_inv ≈ I
        J_inv = Eigen::Matrix3d::Identity();
    } else {
        Eigen::Matrix3d phi_hat = SkewSymmetric(phi / theta);
        double half_theta = 0.5 * theta;
        double cot_half_theta = 1.0 / std::tan(half_theta);
        J_inv = Eigen::Matrix3d::Identity()
              - 0.5 * phi_hat
              + (1.0 / (theta * theta)) * (1 - theta * cot_half_theta / 2.0) * phi_hat * phi_hat;
    }

    Eigen::Vector3d rho = J_inv * t;

    Eigen::Matrix<double, 6, 1> xi;
    xi.head<3>() = rho;
    xi.tail<3>() = phi;
    return xi;
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

// 高斯牛顿法ICP配准
bool GnAlignPoint2Point(const PointCloudPtr& source, const PointCloudPtr& target, Eigen::Matrix4d& T_output) {
    if (!source || !target || source->empty() || target->empty()) return false;

    // 构建KD树
    pcl::KdTreeFLANN<PointT>::Ptr kdtree(new pcl::KdTreeFLANN<PointT>());
    kdtree->setInputCloud(target);

    Eigen::Matrix4d T = T_output;

    std::vector<int> index(source->size());
    for (int i = 0; i < source->size(); ++i) {
        index[i] = i;
    }

    std::vector<Eigen::Vector3d> src_pts(source->size());
    std::vector<Eigen::Vector3d> tgt_pts(source->size());

    for (int iter = 0; iter < max_iterations; ++iter) {
        int effective_count = 0;

        for (int i : index) {
            const auto& p = source->points[i];
            Eigen::Vector4d ps_h(p.x, p.y, p.z, 1.0);
            Eigen::Vector3d ps = (T * ps_h).head<3>();

            PointT search_point;
            search_point.x = ps.x();
            search_point.y = ps.y();
            search_point.z = ps.z();

            std::vector<int> indices(1);
            std::vector<float> dists(1);
            if (kdtree->nearestKSearch(search_point, 1, indices, dists) > 0) {
                float max_nn_distance = 1.0;  // 最近邻距离阈值
                if (dists[0] > max_nn_distance * max_nn_distance) continue;

                const auto& pt = target->points[indices[0]];
                tgt_pts[effective_count] = Eigen::Vector3d(pt.x, pt.y, pt.z);
                src_pts[effective_count] = ps;
                ++effective_count;
            }
        }

        if (effective_count < 15) continue;

        // 构建Hessian和b向量
        Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
        Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Zero();

        for (int i = 0; i < effective_count; ++i) {
            const Eigen::Vector3d& pi = src_pts[i];
            const Eigen::Vector3d& qi = tgt_pts[i];
            Eigen::Vector3d r = pi - qi;

            double residual_thresh = 1.0;
            if (r.norm() > residual_thresh) continue;

            Eigen::Matrix<double, 3, 6> J;
            J.block<3,3>(0,0) = -SkewSymmetric(pi);
            J.block<3,3>(0,3) = Eigen::Matrix3d::Identity();

            double weight = 1.0 / (1.0 + r.norm());
            H += weight * J.transpose() * J;
            b += weight * J.transpose() * (-r);
        }

        // 求解增量
        Eigen::Matrix<double, 6, 1> dx = H.ldlt().solve(b);
        if (dx.norm() < convergence_threshold) {
            ROS_INFO("Converged at iter %d with delta norm %.6f", iter, dx.norm());
            break;
        }

        // 更新位姿
        Eigen::Matrix4d dT = ExpSE3(dx);
        T = dT * T;
    }

    T_output = T;
    return true;
}

bool GnAlignPoint2Plane(const PointCloudPtr& source, const PointCloudPtr& target, Eigen::Matrix4d& T_output) {
    if (!source || !target || source->empty() || target->empty()) return false;

    // 法向量估计
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());
    pcl::NormalEstimation<PointT, pcl::Normal> ne;
    ne.setInputCloud(target);
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
    ne.setSearchMethod(tree);
    ne.setKSearch(20);  // 邻域大小
    ne.compute(*normals);

    // 构建 kd-tree
    pcl::KdTreeFLANN<PointT>::Ptr kdtree(new pcl::KdTreeFLANN<PointT>());
    kdtree->setInputCloud(target);

    Eigen::Matrix4d T = T_output;

    for (int iter = 0; iter < max_iterations; ++iter) {
        Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
        Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Zero();

        int effective_count = 0;

        for (size_t i = 0; i < source->size(); ++i) {
            const auto& p = source->points[i];
            Eigen::Vector4d ps_h(p.x, p.y, p.z, 1.0);
            Eigen::Vector3d ps = (T * ps_h).head<3>();

            // 最近邻
            PointT search_point;
            search_point.x = ps.x();
            search_point.y = ps.y();
            search_point.z = ps.z();

            std::vector<int> indices(1);
            std::vector<float> dists(1);
            if (kdtree->nearestKSearch(search_point, 1, indices, dists) == 0)
                continue;

            int idx = indices[0];
            const auto& pt = target->points[idx];
            const auto& n = normals->points[idx];

            Eigen::Vector3d qi(pt.x, pt.y, pt.z);
            Eigen::Vector3d ni(n.normal_x, n.normal_y, n.normal_z);

            if (ni.norm() < 1e-3) continue;

            // 残差：点到面距离
            double r = (ps - qi).dot(ni);
            if (std::abs(r) > 1.0) continue;

            // 雅可比
            Eigen::Matrix<double, 1, 6> J;
            J.block<1,3>(0,0) = -ni.transpose() * SkewSymmetric(ps);
            J.block<1,3>(0,3) = ni.transpose();

            double w = 1.0 / (1.0 + std::abs(r));  // 鲁棒核

            H += w * J.transpose() * J;
            b += w * J.transpose() * (-r);
            effective_count++;
        }

        if (effective_count < 10) {
            ROS_WARN("Too few correspondences: %d", effective_count);
            continue;
        }

        Eigen::Matrix<double, 6, 1> dx = H.ldlt().solve(b);
        if (dx.norm() < convergence_threshold) {
            ROS_INFO("Point2Plane converged at iter %d with delta %.6f", iter, dx.norm());
            break;
        }

        Eigen::Matrix4d dT = ExpSE3(dx);
        T = dT * T;
    }

    T_output = T;
    return true;
}



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

//------------------ 主函数 ------------------//
bool CustomNdtAlign(const PointCloudPtr& source, const PointCloudPtr& target, Eigen::Matrix4d& T_output) {
    if (!source || !target || source->empty() || target->empty()) return false;

    // 构建体素地图
    std::unordered_map<KeyType, VoxelData> voxel_map;
    for (int i = 0; i < target->points.size(); ++i) {
        Vec3d pt(target->points[i].x, target->points[i].y, target->points[i].z);
        KeyType key = ToKey(pt / ndt_voxel_size);
        voxel_map[key].indices.push_back(i);
    }

    for (auto it = voxel_map.begin(); it != voxel_map.end();) {
        auto& voxel = it->second;
        if (voxel.indices.size() < min_pts_in_voxel) {
            it = voxel_map.erase(it);
            continue;
        }

        std::vector<Vec3d> pts;
        Vec3d mu = Vec3d::Zero();
        for (int idx : voxel.indices) {
            Vec3d p(target->points[idx].x, target->points[idx].y, target->points[idx].z);
            pts.push_back(p);
            mu += p;
        }
        mu /= pts.size();
        voxel.mu = mu;
        voxel.sigma = ComputeCovariance(pts, mu);

        Eigen::JacobiSVD<Mat3d> svd(voxel.sigma, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Vector3d lambda = svd.singularValues();
        lambda = lambda.cwiseMax(lambda[0] * 1e-3);
        Mat3d inv_lambda = lambda.cwiseInverse().asDiagonal();
        voxel.info = svd.matrixV() * inv_lambda * svd.matrixU().transpose();

        ++it;
    }

    // Gauss-Newton 优化
    Eigen::Matrix4d T = T_output;

    for (int iter = 0; iter < max_iterations; ++iter) {
        Mat6d H = Mat6d::Zero();
        Vec6d b = Vec6d::Zero();
        int effective_count = 0;

        for (const auto& pt : source->points) {
            Vec3d ps(pt.x, pt.y, pt.z);
            Vec3d ps_t = (T.block<3,3>(0,0) * ps + T.block<3,1>(0,3));
            KeyType key = ToKey(ps_t / ndt_voxel_size);

            auto it = voxel_map.find(key);
            if (it == voxel_map.end()) continue;
            auto& voxel = it->second;

            Vec3d r = ps_t - voxel.mu;
            if (r.transpose() * voxel.info * r > residual_threshold) continue;

            Eigen::Matrix<double, 3, 6> J;
            J.block<3,3>(0,0) = -T.block<3,3>(0,0) * SkewSymmetric(ps);
            J.block<3,3>(0,3) = Eigen::Matrix3d::Identity();

            H += J.transpose() * voxel.info * J;
            b += -J.transpose() * voxel.info * r;
            ++effective_count;
        }

        if (effective_count < 10) break;

        Vec6d dx = H.ldlt().solve(b);
        if (dx.norm() < convergence_eps) break;

        Eigen::Matrix3d dR = ExpSO3(dx.head<3>());
        Eigen::Vector3d dt = dx.tail<3>();
        Eigen::Matrix4d dT = Eigen::Matrix4d::Identity();
        dT.block<3,3>(0,0) = dR;
        dT.block<3,1>(0,3) = dt;

        T = dT * T;
    }

    T_output = T;
    return true;
}


void UpdateGlobalMap(const PointCloudPtr& cloud, const Eigen::Matrix4d& pose) {
    // 初始化地图
    if (!global_map) {
        PointCloudPtr transformed(new PointCloudT());
        pcl::transformPointCloud(*cloud, *transformed, pose);
        global_map = DownsampleCloud(transformed, voxel_size);
        return;
    }

    // 当前帧变换到全局系
    PointCloudPtr cloud_transformed(new PointCloudT());
    pcl::transformPointCloud(*cloud, *cloud_transformed, pose);

    PointCloudPtr downsampled(new PointCloudT());
    downsampled = DownsampleCloud(cloud_transformed, voxel_size);
    *global_map += *downsampled;
}


// 发布里程计和轨迹
void PublishOdometry(const ros::Time& stamp, const Eigen::Matrix4d& pose) {
    // 提取旋转和平移
    Eigen::Matrix3d R = pose.block<3,3>(0,0);
    Eigen::Vector3d t = pose.block<3,1>(0,3);
    
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
void PublishMaps(const ros::Time& stamp, const PointCloudPtr& global_map) {
    // 发布全局地图
    if (global_map) {
        sensor_msgs::PointCloud2 map_msg;
        pcl::toROSMsg(*global_map, map_msg);
        map_msg.header.stamp = stamp;
        map_msg.header.frame_id = "map";
        pub_global_map.publish(map_msg);
    }
}

void VisualizeTwoClouds(const PointCloudPtr& source, const PointCloudPtr& target, const Eigen::Matrix4d& T) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr source_rgb(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr target_rgb(new pcl::PointCloud<pcl::PointXYZRGB>());

    for (auto& pt : *source) {
        pcl::PointXYZRGB pt_rgb;
        pt_rgb.x = pt.x;
        pt_rgb.y = pt.y;
        pt_rgb.z = pt.z;
        pt_rgb.r = 255; pt_rgb.g = 0; pt_rgb.b = 0;  // 红色：当前帧
        source_rgb->push_back(pt_rgb);
    }

    for (auto& pt : *target) {
        pcl::PointXYZRGB pt_rgb;
        pt_rgb.x = pt.x;
        pt_rgb.y = pt.y;
        pt_rgb.z = pt.z;
        pt_rgb.r = 0; pt_rgb.g = 255; pt_rgb.b = 0;  // 绿色：上一帧
        target_rgb->push_back(pt_rgb);
    }

    // 变换 source_cloud
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr source_transformed(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::transformPointCloud(*source_rgb, *source_transformed, T);

    // 可视化
    pcl::visualization::PCLVisualizer viewer("ICP Result");
    viewer.addPointCloud(source_transformed, "source");
    viewer.addPointCloud(target_rgb, "target");

    viewer.addCoordinateSystem(1.0);
    viewer.spin();
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
    // 转换为PCL点云
    PointCloudPtr cloud(new PointCloudT());
    pcl::fromROSMsg(*msg, *cloud);
    
    // 降采样输入点云
    // PointCloudPtr filtered_cloud = DownsampleCloud(cloud, voxel_size);
    auto &used_cloud = cloud;  // 使用原始点云，避免降采样

    // 第一帧处理
    if (!last_cloud) {
        // last_cloud = filtered_cloud;
        last_cloud = used_cloud;
        last_pose = current_pose;
        return;
    }
    static int frame_id = 0;
    frame_id ++;
    if (frame_id % 10 == 0) {
        std::cout << "处理第 " << frame_id << " 帧点云" << std::endl;
        Eigen::Matrix4d T;
        std::string last_path = "/home/syx/my_lio/lidar_odom_ws/src/lidar_odom/tmp/last_cloud_filter_" + std::to_string(frame_id) + ".pcd";
        std::string curr_path = "/home/syx/my_lio/lidar_odom_ws/src/lidar_odom/tmp/current_cloud_filter_" + std::to_string(frame_id) + ".pcd";
        pcl::io::savePCDFileBinary(last_path, *last_cloud);
        pcl::io::savePCDFileBinary(curr_path, *used_cloud);

        std::cout << "两帧点云保存完毕" << std::endl;
        // GnAlignPoint2Point(used_cloud, last_cloud, T);
        // VisualizeTwoClouds(used_cloud, last_cloud, T);
    }

    bool success = false;
    
    if (use_pcl_ndt) {
        ros::Time start = ros::Time::now();
        success = pclNdtAlign(used_cloud, last_cloud, T_curr_last);
        // success = CustomNdtAlign(used_cloud, last_cloud, T_curr_last);
        ros::Duration elapsed = ros::Time::now() - start;
        ROS_INFO("NDT Runtime: %.3f ms", elapsed.toSec() * 1000.0);
    } else {
        // ros::Time start = ros::Time::now();
        // // success = AlignNdt( );
        // ros::Duration elapsed = ros::Time::now() - start;
        // ROS_INFO("NDT Runtime: %.3f ms", elapsed.toSec() * 1000.0);

        ros::Time start = ros::Time::now();
        // success = GnAlignPoint2Point(used_cloud, last_cloud, T_curr_last);
        success = GnAlignPoint2Plane(used_cloud, last_cloud, T_curr_last);
        ros::Duration elapsed = ros::Time::now() - start;
        ROS_INFO("ICPsuccess! Runtime: %.3f ms", elapsed.toSec() * 1000.0);
    }

    PointCloudPtr target_for_viz(new PointCloudT(*last_cloud));  // 深拷贝上一帧

    if (success) {
        std::cout << "T_curr_last:\n" << T_curr_last << std::endl;

        // 更新当前位姿
        current_pose = T_curr_last * last_pose;
        last_pose = current_pose;     

        // 计算 aligned 点云
        PointCloudPtr aligned_cloud(new PointCloudT());
        pcl::transformPointCloud(*used_cloud, *aligned_cloud, T_curr_last);

        PointCloudPtr source_in_map(new PointCloudT());
        PointCloudPtr target_in_map(new PointCloudT());
        PointCloudPtr aligned_in_map(new PointCloudT());

        pcl::transformPointCloud(*used_cloud, *source_in_map, current_pose);
        pcl::transformPointCloud(*target_for_viz, *target_in_map, current_pose);  
        pcl::transformPointCloud(*aligned_cloud, *aligned_in_map, current_pose);     

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
        UpdateGlobalMap(used_cloud, current_pose);

        
        // 发布数据
        PublishOdometry(msg->header.stamp, current_pose);
        PublishMaps(msg->header.stamp, global_map);

    }
    else {
        ROS_WARN("Registration failed. Skipping this frame.");
        return;
    }
    // 更新上一帧点云
    last_cloud.reset(new PointCloudT(*used_cloud));
}

int main(int argc, char** argv) {
    // 初始化ROS节点
    ros::init(argc, argv, "odometry");
    ros::NodeHandle nh;

    nh.param("use_pcl_ndt", use_pcl_ndt, false);
    nh.param("max_iterations", max_iterations, 10);
    nh.param("convergence_threshold", convergence_threshold, 1e-4);
    nh.param("voxel_size", voxel_size, 0.3f);
    nh.param("local_voxel_size", local_voxel_size, 0.3f);

    nh.param("ndt_resolution", ndt_resolution, 1.5f);
    nh.param("ndt_voxel_size", ndt_voxel_size, 0.5f);
    nh.param("ndt_step_size", ndt_step_size, 0.1f);
    nh.param("ndt_trans_epsilon", ndt_trans_epsilon, 0.01f);
    nh.param("ndt_max_iterations", ndt_max_iterations, 10);
    
    // 创建发布器
    pub_odom = nh.advertise<nav_msgs::Odometry>("/laser_odom", 100);
    pub_path = nh.advertise<nav_msgs::Path>("/laser_path", 100);
    pub_global_map = nh.advertise<sensor_msgs::PointCloud2>("/global_map", 10);
    pub_debug_marker = nh.advertise<visualization_msgs::Marker>("/debug_marker", 10);
    pub_source = nh.advertise<sensor_msgs::PointCloud2>("/source_cloud", 10);
    pub_target = nh.advertise<sensor_msgs::PointCloud2>("/target_cloud", 10);
    pub_aligned = nh.advertise<sensor_msgs::PointCloud2>("/aligned_cloud", 10);

    
    // 创建订阅者
    ros::Subscriber sub_cloud = nh.subscribe("/velodyne_cloud", 100, CloudCallback);
    
    // 初始化地图
    global_map = nullptr;
    
    std::cout << "使用" << (use_pcl_ndt ? "NDT" : "ICP") << "配准" << std::endl;
    
    // 循环处理回调
    ros::spin();
    return 0;
}