#pragma once
#include <grid_map_msgs/GridMap.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <nav_msgs/OccupancyGrid.h>
#include <ros/ros.h>
#include <tf/transform_listener.h>

#include <bruce_exploration/base/targets.h>
#include <bruce_msgs/ISAM2Update.h>
#include <bruce_msgs/QueryExplorationPath.h>
#include "3rdparty/cubic_spline_interpolator.h"


namespace bruce_exploration
{
namespace base
{
template <class PATHLIBRARY, class UTILITYFUNCTION>
class ExplorationServer
{
public:
  typedef ExplorationServer<PATHLIBRARY, UTILITYFUNCTION> This;
  typedef std::shared_ptr<This> Ptr;

  ExplorationServer(ros::NodeHandle nh);

  void publishBBox() const;

  bool getRobotPose(double &x, double &y, double &theta) const;

  void publishAll() const;

protected:
  void callback(const bruce_msgs::ISAM2UpdateConstPtr &isam2_update_msg,
                const nav_msgs::OccupancyGridConstPtr &occ_grid_msg);
  bool queryExplorationPathSrv(bruce_msgs::QueryExplorationPathRequest &req,
                               bruce_msgs::QueryExplorationPathResponse &resp);

protected:
  ros::NodeHandle nh_;
  message_filters::Subscriber<bruce_msgs::ISAM2Update> isam2_sub_;
  message_filters::Subscriber<nav_msgs::OccupancyGrid> occ_sub_;
  message_filters::TimeSynchronizer<bruce_msgs::ISAM2Update, nav_msgs::OccupancyGrid> sync_;
  ros::ServiceServer query_exploration_path_srv_;
  ros::Publisher virtual_map_pub_, cost_map_pub_, query_result_pub_, path_pub_, bbox_pub_;
  tf::TransformListener listener_;

  path_smoothing::CubicSplineInterpolator csi_;

  bruce_msgs::ISAM2Update isam2_update_msg_cache_;
  nav_msgs::OccupancyGrid occ_grid_msg_cache_;

  PATHLIBRARY path_library_;
  UTILITYFUNCTION utility_function_;
};

}  // namespace base
}  // namespace bruce_exploration

#include "bruce_exploration/base/exploration_server.hpp"