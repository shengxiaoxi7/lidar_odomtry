// #pragma once

// #include <pcl/point_cloud.h>
// #include <pcl/point_types.h>
// #include <Eigen/Dense>
// #include <vector>
// #include <deque>
// #include <map>

// using PointT = pcl::PointXYZI;
// using PointCloudT = pcl::PointCloud<PointT>;
// using PointCloudPtr = pcl::PointCloud<PointT>::Ptr;
// using Vec3d = Eigen::Vector3d;
// using Vec6d = Eigen::Matrix<double, 6, 1>;
// using Mat3d = Eigen::Matrix3d;
// using Mat6d = Eigen::Matrix<double, 6, 6>;
// using KeyType = Eigen::Vector3i;

// inline KeyType CastToInt(const Vec3d& v) {
//     return KeyType((int)std::floor(v[0]), (int)std::floor(v[1]), (int)std::floor(v[2]));
// }

// inline Vec3d CastToVec3d(const KeyType& k) {
//     return Vec3d(static_cast<double>(k[0]), static_cast<double>(k[1]), static_cast<double>(k[2]));
// }

// class Ndt{
// public:
//     struct Options {

//         double global_voxel_size = 1.0;
//         int min_pts_in_voxel = 3;
//         int max_iterations = 20;
//         double eps = 1e-4;
//         bool nearby6 = true;
//         bool remove_centroid = false;   // 是否计算两个点云中心并移除中心？
//         double res_outlier_th_ = 20.0;  // 异常值拒绝阈值
//         int min_effective_pts = 10;     // 最近邻点数阈值

//     } options_;

//     PointCloudPtr target_;
//     PointCloudPtr source_;
//     Eigen::Matrix4d T_output_;
//     std::map<KeyType, std::vector<int>> nearby_grids_;  // 附近的网格
//     std::map<KeyType, std::vector<int>> voxel_map_;     // 网格数据

//     void SetTarget(PointCloudPtr target);
//     void SetSource(PointCloudPtr source);
//     bool AlignNdt(Eigen::Matrix4d& init_pose);
//     void GenerateNearbyGrids();
//     void BuildVoxels();
//     bool pclNdtAlign(const PointCloudPtr& source, const PointCloudPtr& target, Eigen::Matrix4d& T_output);
//     // bool GnAlignPoint2Plane_(const PointCloudPtr& source, const PointCloudPtr& target, Eigen::Matrix4d& T_output);

// private:


// };

// class Direct_Ndt{
// public:
//     struct Options {

//         bool use_pcl_ndt_ = false;
//         bool display_realtime_cloud_ = false;
//         int num_kfs_in_local_map_ = 10;
//         double kf_distance_ = 0.5;
//         double kf_angle_deg_ = 10.0;
//     } options_;

//     void CloudCallback(const sensor_msgs::PointCloud2::ConstPtr& msg);


// private:
//     pcl::NormalDistributionsTransform<PointT, PointT> ndt_pcl_;
//     Ndt ndt_;
//     std::deque<PointCloudPtr> scans_in_local_map_;
//     PointCloudPtr local_map_;
//     PointCloudPtr target_;
//     Eigen::Matrix4d current_pose_;
//     Eigen::Matrix4d last_kf_pose_;

// };

















// #endif