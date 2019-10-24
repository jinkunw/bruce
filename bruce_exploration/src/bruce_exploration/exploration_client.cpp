#include <actionlib/client/simple_action_client.h>
#include <angles/angles.h>
#include <costmap_2d/cost_values.h>

#include "bruce_exploration/exploration_client.h"

namespace bruce_exploration
{
ExplorationClient::ExplorationClient(ros::NodeHandle nh)
  : nh_(nh), move_base_client_("/move_base", true), grid_loaded_(false), replan_(true)
{
  nh_.param<double>("replan_distance", replan_distance_, 10.0);
  nh_.param<double>("replan_time", replan_time_, 60.0);
  nh_.param<bool>("force_quit", force_quit_, false);

  while (ros::ok() && !move_base_client_.waitForServer(ros::Duration(5.0)))
  {
    ROS_INFO("Waiting for the move_base action server to come up");
  }
  if (!move_base_client_.isServerConnected())
  {
    ROS_ERROR("Couldn't connect to move_base action server");
  }

  costmap_sub_ = nh_.subscribe<nav_msgs::OccupancyGrid>("costmap", 1, &ExplorationClient::costmapCallback, this);
  costmap_update_sub_ = nh_.subscribe<map_msgs::OccupancyGridUpdate>("costmap_updates", 1,
                                                                     &ExplorationClient::costmapUpdateCallback, this);

  global_plan_sub_ = nh_.subscribe<nav_msgs::Path>("global_plan", 10, &ExplorationClient::globalPlanCallback, this);

  if (force_quit_)
    query_result_sub_ = nh_.subscribe<visualization_msgs::MarkerArray>("query_result", 10,
                                                                       &ExplorationClient::queryResultcallback, this);
}

void ExplorationClient::inplaceRotate()
{
  ROS_INFO("Start rotating inplace...");
  ros::Rate r(10);
  ros::Publisher vel_pub = nh_.advertise<geometry_msgs::Twist>("/cmd_vel", 10);

  static tf::TransformListener listener;
  tf::StampedTransform transform;
  try
  {
    listener.waitForTransform("map", "base_link", ros::Time(0), ros::Duration(1.0));
    listener.lookupTransform("map", "base_link", ros::Time(0), transform);
  }
  catch (tf::TransformException& ex)
  {
    ROS_ERROR("%s", ex.what());
  }
  tf::Matrix3x3 mat(transform.getRotation());
  double roll, pitch, yaw;
  mat.getRPY(roll, pitch, yaw);

  double current_angle = yaw;
  double start_angle = current_angle;

  bool got_180 = false;
  double tolerance = 0.2;

  while (nh_.ok() && (!got_180 || std::fabs(angles::shortest_angular_distance(current_angle, start_angle)) > tolerance))
  {
    try
    {
      listener.waitForTransform("map", "base_link", ros::Time(0), ros::Duration(1.0));
      listener.lookupTransform("map", "base_link", ros::Time(0), transform);
    }
    catch (tf::TransformException& ex)
    {
      ROS_ERROR("%s", ex.what());
    }
    tf::Matrix3x3 mat(transform.getRotation());
    double roll, pitch, yaw;
    mat.getRPY(roll, pitch, yaw);
    current_angle = yaw;

    // compute the distance left to rotate
    double dist_left;
    if (!got_180)
    {
      // If we haven't hit 180 yet, we need to rotate a half circle plus the distance to the 180 point
      double distance_to_180 = std::fabs(angles::shortest_angular_distance(current_angle, start_angle + M_PI));
      dist_left = M_PI + distance_to_180;

      if (distance_to_180 < tolerance)
      {
        got_180 = true;
      }
    }
    else
    {
      // If we have hit the 180, we just have the distance back to the start
      dist_left = std::fabs(angles::shortest_angular_distance(current_angle, start_angle));
    }

    const double vel = 0.2;

    geometry_msgs::Twist cmd_vel;
    cmd_vel.linear.x = 0.0;
    cmd_vel.linear.y = 0.0;
    cmd_vel.angular.z = vel;

    vel_pub.publish(cmd_vel);

    r.sleep();
  }
  ROS_INFO("Done rotating inplace");
}

void ExplorationClient::run()
{
  inplaceRotate();
  ROS_INFO("Start exploration...");

  ros::Rate rate(1);
  while (ros::ok())
  {
    if (replan_)
    {
      // Let the robot stop completely
      move_base_client_.cancelAllGoals();
      ros::Duration(1.0).sleep();

      // Send an empty goal.
      move_base_msgs::MoveBaseGoal goal;
      // Provide a valid pose
      goal.target_pose.pose.orientation.w = 1.0;
      goal.target_pose.header.frame_id = "map";
      move_base_client_.sendGoal(goal, boost::bind(&ExplorationClient::doneCallback, this, _1, _2),
                                 MoveBaseClient::SimpleActiveCallback(),
                                 boost::bind(&ExplorationClient::feedbackCallback, this, _1));

      // Reset current plan pose/distance/time
      try
      {
        tf_.waitForTransform("map", "base_link", ros::Time(0), ros::Duration(2.0));
        tf_.lookupTransform("map", "base_link", ros::Time(0), plan_start_pose_);
      }
      catch (tf::TransformException ex)
      {
        ROS_ERROR("%s", ex.what());
        continue;
      }
      plan_distance_ = 0.0;
      replan_ = false;
    }
    ros::spinOnce();
    rate.sleep();
  }
}

void ExplorationClient::doneCallback(const actionlib::SimpleClientGoalState& state,
                                     const move_base_msgs::MoveBaseResultConstPtr& result)
{
  if (replan_)
    return;
  ROS_INFO("Replan: done with state %s", state.toString().c_str());
  replan_ = true;
}

void ExplorationClient::feedbackCallback(const move_base_msgs::MoveBaseFeedbackConstPtr& feedback)
{
  if (replan_)
    return;

  tf::Stamped<tf::Pose> stamped_pose;
  tf::poseStampedMsgToTF(feedback->base_position, stamped_pose);
  tf::StampedTransform plan_pose;
  plan_pose.setData(stamped_pose);
  plan_pose.stamp_ = stamped_pose.stamp_;

  double dt = (plan_pose.stamp_ - plan_start_pose_.stamp_).toSec();
  if (dt > replan_time_)
  {
    ROS_INFO("Replan: timeout %.2f > %.2f", dt, replan_time_);
    replan_ = true;
    return;
  }

  tf::Transform delta = plan_last_pose_.inverseTimes(plan_pose);
  plan_last_pose_ = plan_pose;
  plan_distance_ += delta.getOrigin().length();
  if (plan_distance_ > replan_distance_)
  {
    ROS_INFO("Replan: distance %.2f > %.2f", plan_distance_, replan_distance_);
    replan_ = true;
    return;
  }
}

void ExplorationClient::globalPlanCallback(const nav_msgs::PathConstPtr& path)
{
  if (replan_)
    return;

  for (const geometry_msgs::PoseStamped& pose : path->poses)
  {
    if (isOccupied(pose.pose.position.x, pose.pose.position.y))
    {
      ROS_INFO("Replan: path is blocked");
      replan_ = true;
      return;
    }
  }
}

bool ExplorationClient::isOccupied(double x, double y) const
{
  if (x < grid_.info.origin.position.x || y < grid_.info.origin.position.y)
    return false;

  int mx = (int)((x - grid_.info.origin.position.x) / grid_.info.resolution);
  int my = (int)((y - grid_.info.origin.position.y) / grid_.info.resolution);

  if (mx >= grid_.info.width || my >= grid_.info.height)
    return false;

  return grid_.data[my * grid_.info.width + mx] > 0;
}

/// https://github.com/ros-visualization/rviz/blob/4b6c0f447159044bfaa633e140e6094b19516b02/src/rviz/default_plugin/map_display.cpp#L550
void ExplorationClient::costmapCallback(const nav_msgs::OccupancyGridConstPtr& msg)
{
  grid_ = *msg;
  grid_loaded_ = true;
}

/// https://github.com/ros-visualization/rviz/blob/4b6c0f447159044bfaa633e140e6094b19516b02/src/rviz/default_plugin/map_display.cpp#L559
void ExplorationClient::costmapUpdateCallback(const map_msgs::OccupancyGridUpdateConstPtr& msg)
{
  if (!grid_loaded_)
    return;

  if (msg->x < 0 || msg->y < 0 || grid_.info.width < msg->x + msg->width || grid_.info.height < msg->y + msg->height)
    return;

  for (size_t y = 0; y < msg->height; y++)
  {
    memcpy(&grid_.data[(msg->y + y) * grid_.info.width + msg->x], &msg->data[y * msg->width], msg->width);
  }
}

void ExplorationClient::queryResultcallback(const visualization_msgs::MarkerArrayConstPtr& result)
{
  if (!force_quit_)
    return;

  for (const auto& markers : result->markers)
  {
    if (markers.ns == "frontiers" && !markers.points.empty())
    {
      return;
    }
  }
  system("killall roslaunch");
}

}  // namespace bruce_exploration

int main(int argc, char** argv)
{
  ros::init(argc, argv, "exploreration_client");
  ros::NodeHandle nh("~");

  bruce_exploration::ExplorationClient client(nh);
  ROS_INFO("Exploration client is initialized");
  client.run();

  return 0;
}