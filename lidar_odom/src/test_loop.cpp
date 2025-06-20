// 该代码只用于在里程计节点中可视化关键帧图
#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/PoseStamped.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <nav_msgs/Path.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

// 订阅关键帧位姿话题
ros::Subscriber keyframe_pose;
ros::Subscriber keyframe_cloud;
ros::Publisher pub_marker;
ros::Publisher pub_path;

std::string result_path = "/home/syx/my_lio/lidar_odom_ws/src/lidar_odom/pcl_kf_distance_1/";

struct Keyframe {
    int id;
    double timestamp;
    Eigen::Vector3d t;
    Eigen::Quaterniond q;
};

std::vector<Keyframe> keyframes;


// 读取关键帧位姿
// bool LoadKeyframesFromFile(const std::string& path) {
//     std::ifstream fin(path);
//     if (!fin.is_open()) {
//         ROS_ERROR("Cannot open keyframe file: %s", path.c_str());
//         return false;
//     }

//     ROS_INFO("keyframe file is opened from: %s", path.c_str());

//     std::string line;
//     while (std::getline(fin, line)) {
//         if (line.empty() || line[0] == '#') continue;
//         std::istringstream iss(line);
//         Keyframe kf;
//         double qx, qy, qz, qw;
//         iss >> kf.id >> kf.timestamp >> kf.t.x() >> kf.t.y() >> kf.t.z() >> qx >> qy >> qz >> qw;
//         kf.q = Eigen::Quaterniond(qw, qx, qy, qz);
//         keyframes.push_back(kf);
//     }

//     return true;
// }

void KeyframeCallback(const geometry_msgs::PoseStamped::ConstPtr& msg) {
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
    // visualization_msgs::Marker clear_all;
    // clear_all.action = visualization_msgs::Marker::DELETEALL;
    // pub_marker.publish(clear_all);

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
   
    for (const auto& kf : keyframes){
        
        // 3. 发布最新关键帧小球        
        visualization_msgs::Marker sphere;
        sphere.header.frame_id = "map";
        sphere.header.stamp = stamp;
        sphere.ns = "keyframe_sphere";
        sphere.id = kf.id;
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
        text.id = kf.id;
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
    }


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


int main(int argc, char** argv) {
    ros::init(argc, argv, "keyframe_graph_node");
    ros::NodeHandle nh;

    // 订阅关键帧话题
    ros::Subscriber sub_kf = nh.subscribe("/keyframe_pose", 100, KeyframeCallback);

    // 发布轨迹 Marker 和 Path
    pub_marker = nh.advertise<visualization_msgs::Marker>("keyframe_graph", 10);
    pub_path   = nh.advertise<nav_msgs::Path>("keyframe_path", 1);

    ros::Rate rate(1.0);  // 1Hz 可视化频率
    while (ros::ok()) {
        ros::spinOnce();
        PublishVisualization();
        rate.sleep();
    }

    return 0;
}