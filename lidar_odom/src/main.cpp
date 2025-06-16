// // main.cpp
// #include <ros/ros.h>
// #include <sensor_msgs/PointCloud2.h>
// #include <pcl_conversions/pcl_conversions.h>
// #include <pcl/point_cloud.h>
// #include <pcl/point_types.h>


// class LidarOdometry {
// public:
//     LidarOdometry(ros::NodeHandle& nh) {
//         // 订阅LiDAR数据
//         lidar_sub_ = nh.subscribe("/velodyne_points", 10, &LidarOdometry::lidarCallback, this);
        
//         // 初始化组件
//         data_processor_.reset(new DataProcessor);
//         feature_extractor_.reset(new FeatureExtractor);
//         pose_estimator_.reset(new PoseEstimator);
//     }
    
//     void lidarCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {
//         // 转换ROS消息到PCL点云
//         pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
//         pcl::fromROSMsg(*msg, *cloud);
        
//         // 1. 数据预处理
//         pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud = data_processor_->process(cloud);
        
//         // 2. 特征提取
//         Features features = feature_extractor_->extract(filtered_cloud);
        
//         // 3. 位姿估计
//         Eigen::Matrix4f transform = pose_estimator_->estimate(features);
        
//         // 4. 更新轨迹
//         updateTrajectory(transform);
        
//         // 5. 发布结果
//         publishOdometry(transform);
//     }
    
// private:
//     ros::Subscriber lidar_sub_;
//     std::unique_ptr<DataProcessor> data_processor_;
//     std::unique_ptr<FeatureExtractor> feature_extractor_;
//     std::unique_ptr<PoseEstimator> pose_estimator_;
// };

// int main(int argc, char** argv) {
//     ros::init(argc, argv, "lidar_odometry");
//     ros::NodeHandle nh;
//     LidarOdometry odom(nh);
//     ros::spin();
//     return 0;
// }