#pragma once
#include <geometry_msgs/PoseStamped.h>
#include <move_base_msgs/MoveBaseAction.h>
#include <nav_core/base_global_planner.h>

namespace bruce_exploration
{
class ExplorationPlanner : public nav_core::BaseGlobalPlanner
{
public:
  ExplorationPlanner();

  void initialize(std::string name, costmap_2d::Costmap2DROS *costmap_ros) override
  {
  }

  bool makePlan(const geometry_msgs::PoseStamped &start, const geometry_msgs::PoseStamped &goal,
                std::vector<geometry_msgs::PoseStamped> &plan) override;

private:
  ros::ServiceClient query_path_client_;
};

}  // namespace bruce_exploration
