#include <iostream>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_broadcaster.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>

ros::Publisher pubLaserCloud;

void Callback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg) {
    
    // 转为 PCL 点云格式
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::fromROSMsg(*cloud_msg, *cloud_in);

    // 体素滤波降采样
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_down(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::VoxelGrid<pcl::PointXYZI> voxel;
    voxel.setInputCloud(cloud_in);
    voxel.setLeafSize(0.1f, 0.1f, 0.1f);  // 分辨率
    voxel.filter(*cloud_down);

    ROS_INFO("Before : %zu, After: %zu", cloud_in->size(), cloud_down->size());

    // 转回 ROS 消息
    sensor_msgs::PointCloud2 cloud_out;
    // pcl::toROSMsg(*cloud_in, cloud_out);
    pcl::toROSMsg(*cloud_down, cloud_out);
    cloud_out.header = cloud_msg->header;
    cloud_out.header.frame_id = "map";  
    pubLaserCloud.publish(cloud_out);
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "lidar_system_node");
    ros::NodeHandle nh;

    // ros::Subscriber subLaserCloud = nh.subscribe("/velodyne_points", 10, Callback); //vlp-16
    ros::Subscriber subLaserCloud = nh.subscribe("/velodyne_points_0", 10, Callback); //ulhk
    pubLaserCloud = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud", 10);

    ros::spin();
    return 0;
}
