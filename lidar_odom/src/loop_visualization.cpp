// 用于展示优化前后结果图，纯可视化，不做任何处理
#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/Point.h>
#include <nav_msgs/Path.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>

struct Keyframe {
    int id;
    double timestamp;
    Eigen::Vector3d t;
    Eigen::Quaterniond q;
};

std::vector<Keyframe> keyframes_raw;
std::vector<Keyframe> keyframes_opt;

ros::Publisher pub_marker;
ros::Publisher pub_path_raw;
ros::Publisher pub_path_opt;

std::string result_path = "/home/syx/my_lio/lidar_odom_ws/src/lidar_odom/tmp/pcl_kf_distance_1/";

bool LoadKeyframesFromFile(const std::string& path, std::vector<Keyframe>& kfs) {
    std::ifstream fin(path);
    if (!fin.is_open()) {
        ROS_ERROR("Cannot open keyframe file: %s", path.c_str());
        return false;
    }

    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        Keyframe kf;
        double qx, qy, qz, qw;
        if (!(iss >> kf.id >> kf.timestamp >> kf.t.x() >> kf.t.y() >> kf.t.z() >> qx >> qy >> qz >> qw)) {
            ROS_WARN("Invalid line in keyframe file: %s", line.c_str());
            continue;
        }
        kf.q = Eigen::Quaterniond(qw, qx, qy, qz);
        kfs.push_back(kf);
    }

    return true;
}

void PublishTrajectory(const std::vector<Keyframe>& kfs, const std_msgs::ColorRGBA& color, const std::string& ns, int id, ros::Publisher& pub_path, ros::Publisher& pub_marker) {
    visualization_msgs::Marker line;
    line.header.frame_id = "map";
    line.header.stamp = ros::Time::now();
    line.ns = ns;
    line.id = id;
    line.type = visualization_msgs::Marker::LINE_STRIP;
    line.action = visualization_msgs::Marker::ADD;
    line.scale.x = 0.05;
    line.color = color;

    nav_msgs::Path path;
    path.header.frame_id = "map";
    path.header.stamp = ros::Time::now();

    for (const auto& kf : kfs) {
        geometry_msgs::Point p;
        p.x = kf.t.x(); p.y = kf.t.y(); p.z = kf.t.z();
        line.points.push_back(p);

        geometry_msgs::PoseStamped pose;
        pose.header.frame_id = "map";
        pose.header.stamp = ros::Time::now();
        pose.pose.position.x = kf.t.x();
        pose.pose.position.y = kf.t.y();
        pose.pose.position.z = kf.t.z();
        pose.pose.orientation.x = kf.q.x();
        pose.pose.orientation.y = kf.q.y();
        pose.pose.orientation.z = kf.q.z();
        pose.pose.orientation.w = kf.q.w();
        path.poses.push_back(pose);
    }

    pub_marker.publish(line);
    pub_path.publish(path);
}

void PublishMarkers(const std::vector<Keyframe>& kfs, const std::string& ns_prefix, const std_msgs::ColorRGBA& color, ros::Publisher& pub_marker) {
    for (const auto& kf : kfs) {
        visualization_msgs::Marker sphere;
        sphere.header.frame_id = "map";
        sphere.header.stamp = ros::Time::now();
        sphere.ns = ns_prefix + "_sphere";
        sphere.id = kf.id;
        sphere.type = visualization_msgs::Marker::SPHERE;
        sphere.action = visualization_msgs::Marker::ADD;
        sphere.pose.position.x = kf.t.x();
        sphere.pose.position.y = kf.t.y();
        sphere.pose.position.z = kf.t.z();
        sphere.scale.x = 0.2;
        sphere.scale.y = 0.2;
        sphere.scale.z = 0.2;
        sphere.color = color;
        pub_marker.publish(sphere);

        visualization_msgs::Marker text;
        text.header.frame_id = "map";
        text.header.stamp = ros::Time::now();
        text.ns = ns_prefix + "_text";
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
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "keyframe_graph_visualizer");
    ros::NodeHandle nh;

    pub_marker    = nh.advertise<visualization_msgs::Marker>("keyframe_markers", 100);
    pub_path_raw  = nh.advertise<nav_msgs::Path>("raw_path", 1);
    pub_path_opt  = nh.advertise<nav_msgs::Path>("optimized_path", 1);

    std::string raw_file = result_path + "keyframes.txt";
    std::string opt_file = result_path + "optimized_keyframes.txt";
    if (!LoadKeyframesFromFile(raw_file, keyframes_raw)) return -1;
    if (!LoadKeyframesFromFile(opt_file, keyframes_opt)) ROS_WARN("optimized_keyframes.txt not found!");

    ros::Rate rate(1.0);  // 1Hz
    while (ros::ok()) {
        std_msgs::ColorRGBA yellow, green, red, blue;
        yellow.r = 1.0; yellow.g = 1.0; yellow.b = 0.0; yellow.a = 1.0;
        green.g = 1.0; green.a = 1.0;
        red.r = 1.0; red.a = 1.0;
        blue.b = 1.0; blue.a = 1.0;

        PublishTrajectory(keyframes_raw, yellow, "raw_traj", 0, pub_path_raw, pub_marker);
        PublishMarkers(keyframes_raw, "raw", red, pub_marker);

        if (!keyframes_opt.empty()) {
            PublishTrajectory(keyframes_opt, green, "opt_traj", 1, pub_path_opt, pub_marker);
            PublishMarkers(keyframes_opt, "opt", blue, pub_marker);
        }

        ros::spinOnce();
        rate.sleep();
    }

    return 0;
}
