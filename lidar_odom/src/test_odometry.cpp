#include <cmath>
#include <vector>
#include <string>
#include <time.h>
#include<iostream>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>

#include<ceres/ceres.h>
#include <ceres/rotation.h>
#include <eigen3/Eigen/Dense>

typedef pcl::PointXYZ PointType;

//上一帧plane点，所有帧地图点，后面寻最近用的KD树
pcl::PointCloud<PointType> laserCloudIn_plane_last;
pcl::PointCloud<PointType>::Ptr laserCloud_map(new pcl::PointCloud<PointType>());
pcl::KdTreeFLANN<PointType> kdtreePlaneLast;

//输出里程计，路径，当前全部点，地图点
ros::Publisher pubLaserOdometry;
ros::Publisher pubLaserPath;
ros::Publisher pubLaserCloudall_02;
ros::Publisher pubLaserCloud_map;
//定义路径，用于保存帧的位置，发布于pubLaserPath
nav_msgs::Path laserPath;

//当前帧到世界坐标系
Eigen::Quaterniond q_w_curr(1, 0, 0, 0);
Eigen::Vector3d t_w_curr(0, 0, 0);
 
//当前帧到上一帧
double para_q[4] = {0, 0, 0, 1};
double para_t[3] = {0, 0, 0};
//四元数Q，这帧到上帧
Eigen::Map<Eigen::Quaterniond> q_last_curr(para_q);
//这帧到上帧位移量t，配合四元数累加在一起就是当前帧到最开始帧也就是世界坐标系
Eigen::Map<Eigen::Vector3d> t_last_curr(para_t);
//后面要进行距离比较的参数,实际为两点相距3m为阈值
constexpr double DISTANCE_SQ_THRESHOLD = 9;
//找点进行匹配优化时的线数距离(13线-10线>2.5就break介样用)
constexpr double NEARBY_SCAN = 2.5;

//构建代价函数结构体，residual为残差。
//last_point_a_为这一帧中的点a，curr_point_b_为点a旋转后和上一帧里最近的点
//curr_point_c_为点b同线或上线号的点，curr_point_d_为点b下线号的点
//b，c，d与a点距离不超过3m
//plane_norm为根据向量bc和bd求出的法向量
struct POINT_TO_POINT_COST {
  POINT_TO_POINT_COST(Eigen::Vector3d _pt_a, Eigen::Vector3d _pt_b) : pt_a(_pt_a), pt_b(_pt_b) {}

  template <typename T>
  bool operator()(const T* q, const T* t, T* residual) const {
    Eigen::Matrix<T, 3, 1> p_a(T(pt_a.x()), T(pt_a.y()), T(pt_a.z()));
    Eigen::Matrix<T, 3, 1> p_b(T(pt_b.x()), T(pt_b.y()), T(pt_b.z()));
    Eigen::Quaternion<T> q_lw(q[3], q[0], q[1], q[2]);
    Eigen::Matrix<T, 3, 1> t_lw(t[0], t[1], t[2]);
    Eigen::Matrix<T, 3, 1> p_a_trans = q_lw * p_a + t_lw;
    residual[0] = p_a_trans.x() - p_b.x();
    residual[1] = p_a_trans.y() - p_b.y();
    residual[2] = p_a_trans.z() - p_b.z();
    return true;
  }

  const Eigen::Vector3d pt_a, pt_b;
};

//用于推算一个这一帧点在上一帧lidar坐标系下的位置
void TransformToStart(PointType const *const pi, PointType *const po)
{
    Eigen::Vector3d point(pi->x, pi->y, pi->z);
    Eigen::Vector3d un_point;
    
    un_point= q_last_curr * point + t_last_curr;
 
    //输出一下
    po->x = un_point.x();
    po->y = un_point.y();
    po->z = un_point.z();
}

//用于推算一个这一帧的点在世界坐标系下的位置
void TransformToMap(PointType const *const pi, PointType *const po)
{
    Eigen::Vector3d point(pi->x, pi->y, pi->z);
    Eigen::Vector3d un_point;
    Eigen::Quaterniond q_w_(q_w_curr.w(), q_w_curr.x(), q_w_curr.y(), q_w_curr.z());
    Eigen::Vector3d t_w_(t_w_curr.x(), t_w_curr.y(), t_w_curr.z());
    un_point= q_w_ * point + t_w_;
 
    //输出一下
    po->x = un_point.x();
    po->y = un_point.y();
    po->z = un_point.z();
}

void cloud_point_Callhandle(const sensor_msgs::PointCloud2 ros_cloud_plane)
{
    clock_t  clock_start, clock_end;
    clock_start=clock();
    pcl::PointCloud<PointType> laserCloudIn_plane;
    pcl::fromROSMsg(ros_cloud_plane, laserCloudIn_plane);
    //上一帧的点数和这一帧的点数,上一帧点数足够再进行位姿估计
    int laserCloudIn_plane_last_num=laserCloudIn_plane_last.points.size();
    int laserCloudIn_plane_num=laserCloudIn_plane.points.size();
    if(laserCloudIn_plane_last_num<10) laserCloudIn_plane_last=laserCloudIn_plane;
    else
    {
    // ICP 点到点配准主逻辑，并更新全局地图
    kdtreePlaneLast.setInputCloud(laserCloudIn_plane_last.makeShared());
    for (int optize_num = 0; optize_num <= 1; optize_num++) {
    ceres::LossFunction* loss_function = new ceres::HuberLoss(0.1);
    ceres::Manifold *q_parameterization = new ceres::EigenQuaternionManifold();
    ceres::Problem::Options problem_options;
    ceres::Problem problem(problem_options);
    problem.AddParameterBlock(para_q, 4, q_parameterization);
    problem.AddParameterBlock(para_t, 3);

    for (int i = 0; i < laserCloudIn_plane_num; ++i) {
        PointType pointseed;
        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;
        TransformToStart(&laserCloudIn_plane.points[i], &pointseed);
        kdtreePlaneLast.nearestKSearch(pointseed, 1, pointSearchInd, pointSearchSqDis);
        if (pointSearchSqDis[0] < DISTANCE_SQ_THRESHOLD) {
        Eigen::Vector3d pt_a(laserCloudIn_plane.points[i].x,
                            laserCloudIn_plane.points[i].y,
                            laserCloudIn_plane.points[i].z);
        Eigen::Vector3d pt_b(laserCloudIn_plane_last.points[pointSearchInd[0]].x,
                            laserCloudIn_plane_last.points[pointSearchInd[0]].y,
                            laserCloudIn_plane_last.points[pointSearchInd[0]].z);
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<POINT_TO_POINT_COST, 3, 4, 3>(
            new POINT_TO_POINT_COST(pt_a, pt_b)),
            loss_function, para_q, para_t);
        }
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = 5;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    }
    }
    //优化后，当前帧变前帧，位姿累积，并且根据位姿变换当前帧所有点到世界坐标系，放入地图中
    // laserCloudIn_plane_last=laserCloudIn_plane;
    // 更新全局位姿
    q_w_curr = q_w_curr * q_last_curr;
    t_w_curr = t_w_curr + q_w_curr * t_last_curr;

    // 当前帧变换到全局地图坐标系并累加
    pcl::PointCloud<PointType>::Ptr laserCloud_map_curr(new pcl::PointCloud<PointType>());
    for (int i = 0; i < laserCloudIn_plane_num; i += 10) {
    PointType point_in_map;
    TransformToMap(&laserCloudIn_plane.points[i], &point_in_map);
    laserCloud_map_curr->push_back(point_in_map);
    }
    
    //降采样，输出地图点
    pcl::PointCloud<PointType> laserCloud_map_filter;
    pcl::VoxelGrid<PointType> sor;
    sor.setInputCloud(laserCloud_map_curr);
    sor.setLeafSize(0.2f, 0.2f, 0.2f);
    sor.filter(laserCloud_map_filter);
    *laserCloud_map += laserCloud_map_filter;
    
    sensor_msgs::PointCloud2 laserCloudMapMsg;
    pcl::toROSMsg(*laserCloud_map, laserCloudMapMsg);
    laserCloudMapMsg.header.stamp = ros_cloud_plane.header.stamp;
    laserCloudMapMsg.header.frame_id = "map";
    pubLaserCloud_map.publish(laserCloudMapMsg);
    
    //输出位姿和path
    nav_msgs::Odometry laserOdometry;
    laserOdometry.header.frame_id = "map";
    laserOdometry.child_frame_id = "map_child";
    laserOdometry.header.stamp = ros_cloud_plane.header.stamp;
    laserOdometry.pose.pose.orientation.x = q_w_curr.x();
    laserOdometry.pose.pose.orientation.y = q_w_curr.y();
    laserOdometry.pose.pose.orientation.z = q_w_curr.z();
    laserOdometry.pose.pose.orientation.w = q_w_curr.w();
    laserOdometry.pose.pose.position.x = t_w_curr.x();
    laserOdometry.pose.pose.position.y = t_w_curr.y();
    laserOdometry.pose.pose.position.z = t_w_curr.z();
    pubLaserOdometry.publish(laserOdometry);

    geometry_msgs::PoseStamped laserPose;
    laserPose.header = laserOdometry.header;
    laserPose.pose = laserOdometry.pose.pose;
    laserPath.header.stamp = laserOdometry.header.stamp;
    laserPath.poses.push_back(laserPose);
    laserPath.header.frame_id = "map";
    pubLaserPath.publish(laserPath); 
    
    //输出世界参考系的坐标，这里transform为map_child的位姿，并在此位姿下显示当前所有点（通过 cloud_all_Callhandle函数）
    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion q;
    transform.setOrigin(tf::Vector3(t_w_curr.x(),t_w_curr.y(),t_w_curr.z()));
    q.setW(q_w_curr.w());
    q.setX(q_w_curr.x());
    q.setY(q_w_curr.y());
    q.setZ(q_w_curr.z());
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, ros_cloud_plane.header.stamp, "map", "map_child"));
    
    clock_end = clock();
    std::cout<<"odo_time:"<<double(clock_end-clock_start)/ CLOCKS_PER_SEC<<std::endl;
}

//显示当前所有点，注意这里使用的id为map_child，和rosbagregist.cpp里的输出基准不同了，
//表示我们的点跟着位姿走了，不和以前一样只在世界原点附近显示一圈就完了
void cloud_all_Callhandle(const sensor_msgs::PointCloud2 ros_cloud_all)
{
    pcl::PointCloud<PointType> laserCloudIn_all;
    pcl::fromROSMsg(ros_cloud_all, laserCloudIn_all);
    sensor_msgs::PointCloud2 laserCloudplaneMsg;
    pcl::toROSMsg(laserCloudIn_all, laserCloudplaneMsg);
    laserCloudplaneMsg.header.stamp = ros_cloud_all.header.stamp;
    laserCloudplaneMsg.header.frame_id = "map_child";
    pubLaserCloudall_02.publish(laserCloudplaneMsg);
}

int main(int argc, char **argv)
{ 
    ros::init(argc, argv, "odometry");
    ros::NodeHandle n;
    ros::Subscriber cloud_all = n.subscribe("/velodyne_cloud", 100, cloud_point_Callhandle);

    //pubLaserOdometry包括当前帧四元数Q和位置t,pubLaserPath包含当前帧的位置t
    pubLaserOdometry = n.advertise<nav_msgs::Odometry>("/laser_odom_to_init", 100);
    pubLaserPath = n.advertise<nav_msgs::Path>("/laser_odom_path", 100);
    pubLaserCloudall_02 = n.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_all", 100);
    pubLaserCloud_map = n.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_map", 100);
    ros::spin();
    return 0;
}