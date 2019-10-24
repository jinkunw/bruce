#include <actionlib/client/simple_action_client.h>
#include <nav_core/base_global_planner.h>
#include <pluginlib/class_list_macros.h>

#include <bruce_msgs/QueryExplorationPath.h>
#include "bruce_exploration/exploration_planner.h"

PLUGINLIB_EXPORT_CLASS(bruce_exploration::ExplorationPlanner, nav_core::BaseGlobalPlanner)

namespace bruce_exploration
{
ExplorationPlanner::ExplorationPlanner() : nav_core::BaseGlobalPlanner()
{
  ros::NodeHandle private_nh("~");
  query_path_client_ =
      private_nh.serviceClient<bruce_msgs::QueryExplorationPath>("/bruce/exploration/server/query_exploration_path");

  while (!query_path_client_.waitForExistence(ros::Duration(5.0)))
  {
    ROS_WARN("Waiting for the %s server to come up", query_path_client_.getService().c_str());
  }
}

bool ExplorationPlanner::makePlan(const geometry_msgs::PoseStamped &start, const geometry_msgs::PoseStamped &goal,
                                  std::vector<geometry_msgs::PoseStamped> &plan)
{
  bruce_msgs::QueryExplorationPath srv;
  if (query_path_client_.call(srv))
  {
    plan = srv.response.path.poses;
    return true;
  }
  else
  {
    ROS_ERROR("Failed to call service %s", query_path_client_.getService().c_str());
    // Avoid service call immediately if failed
    ros::Duration(1.0).sleep();

    return false;
  }
}

}  // namespace bruce_exploration
