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
    // 发布坐标变换
    // static tf::TransformBroadcaster br;
    // tf::Transform transform;
    // transform.setOrigin(tf::Vector3(0, 0, 0));
    // tf::Quaternion q;
    // q.setRPY(0, 0, 0);  // 无旋转
    // transform.setRotation(q);
    // br.sendTransform(tf::StampedTransform(transform, cloud_msg->header.stamp, "map", "velodyne"));
    
    // 转为 PCL 点云格式
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::fromROSMsg(*cloud_msg, *cloud_in);

    // 筛选低于某 z 值的点
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZI>());
    for (const auto& pt : cloud_in->points) {
        if (pt.z > -1.5 )  
            cloud_filtered->points.push_back(pt);
    }

    // // 体素滤波降采样
    // pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_down(new pcl::PointCloud<pcl::PointXYZI>());
    // pcl::VoxelGrid<pcl::PointXYZI> voxel;
    // voxel.setInputCloud(cloud_filtered);
    // voxel.setLeafSize(0.1f, 0.1f, 0.1f);  // 分辨率
    // voxel.filter(*cloud_down);

    // std::cout << "原始点数: " << cloud_in -> size() << "筛选后点数: " << cloud_filtered -> size() << "降采样后点数: " << cloud_down -> size() << std::endl;

    // 转回 ROS 消息
    sensor_msgs::PointCloud2 cloud_out;
    // pcl::toROSMsg(*cloud_in, cloud_out);
    pcl::toROSMsg(*cloud_filtered, cloud_out);
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
