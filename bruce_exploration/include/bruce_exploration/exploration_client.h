#pragma once
#include <geometry_msgs/PoseStamped.h>
#include <map_msgs/OccupancyGridUpdate.h>
#include <move_base_msgs/MoveBaseAction.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>

namespace bruce_exploration
{
class ExplorationClient
{
  typedef actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction> MoveBaseClient;

public:
  ExplorationClient(ros::NodeHandle nh);

  void inplaceRotate();

  void run();

private:
  void costmapCallback(const nav_msgs::OccupancyGridConstPtr &msg);

  void costmapUpdateCallback(const map_msgs::OccupancyGridUpdateConstPtr &msg);

  bool isOccupied(double x, double y) const;

  /// Report failure with the global plan is blocked
  /// Local planner should be responsible to report block,
  /// but it's better to replan earlier.
  void globalPlanCallback(const nav_msgs::PathConstPtr &path);

  void doneCallback(const actionlib::SimpleClientGoalState &state,
                    const move_base_msgs::MoveBaseResultConstPtr &result);

  void feedbackCallback(const move_base_msgs::MoveBaseFeedbackConstPtr &feedback);

  void queryResultcallback(const visualization_msgs::MarkerArrayConstPtr &result);

private:
  ros::NodeHandle nh_;
  MoveBaseClient move_base_client_;
  tf::TransformListener tf_;
  bool grid_loaded_;
  bool path_blocked_;
  nav_msgs::OccupancyGrid grid_;
  ros::Subscriber costmap_sub_, costmap_update_sub_, global_plan_sub_;
  ros::Subscriber query_result_sub_;

  tf::StampedTransform plan_start_pose_;
  tf::StampedTransform plan_last_pose_;
  double plan_distance_;
  bool replan_;

  double replan_distance_;
  double replan_time_;

  bool force_quit_;
};
}  // namespace bruce_exploration
