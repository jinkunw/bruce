#include <nav_msgs/Path.h>
#include <grid_map_ros/GridMapRosConverter.hpp>

#define LAZY_EVAL

namespace bruce_exploration
{
namespace base
{
template <class PATHLIBRARY, class UTILITYFUNCTION>
ExplorationServer<PATHLIBRARY, UTILITYFUNCTION>::ExplorationServer(ros::NodeHandle nh)
  : nh_(nh)
  , isam2_sub_(nh_, "isam2", 3)
  , occ_sub_(nh_, "map", 3)
  , sync_(isam2_sub_, occ_sub_, 3)
  , csi_("smooth/")
  , path_library_(nh_)
  , utility_function_(nh_)
{
  sync_.registerCallback(boost::bind(&This::callback, this, _1, _2));

  // Publishers
  cost_map_pub_ = nh_.advertise<grid_map_msgs::GridMap>("cost_map", 1, true);
  query_result_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("query_result", 1, true);
  path_pub_ = nh_.advertise<nav_msgs::Path>("query_path", 1, true);
  bbox_pub_ = nh_.advertise<visualization_msgs::Marker>("bbox", 1, true);

  // query_exploration_path_srv_ = nh_.advertiseService("query_exploration_path",
  // &This::queryExplorationPathSrv, this);

  publishBBox();
}

template <class PATHLIBRARY, class UTILITYFUNCTION>
void ExplorationServer<PATHLIBRARY, UTILITYFUNCTION>::publishBBox() const
{
  auto min_lim = path_library_.getCostMap()->getLimitMin();
  auto max_lim = path_library_.getCostMap()->getLimitMax();

  visualization_msgs::Marker bbox;
  bbox.header.stamp = ros::Time::now();
  bbox.header.frame_id = "map";
  bbox.type = visualization_msgs::Marker::LINE_STRIP;
  bbox.scale.x = 0.5;
  bbox.color.r = 1.0;
  bbox.color.a = 1.0;

  geometry_msgs::Point point;
  point.x = min_lim(0);
  point.y = min_lim(1);
  bbox.points.push_back(point);
  point.x = min_lim(0);
  point.y = max_lim(1);
  bbox.points.push_back(point);
  point.x = max_lim(0);
  point.y = max_lim(1);
  bbox.points.push_back(point);
  point.x = max_lim(0);
  point.y = min_lim(1);
  bbox.points.push_back(point);
  point.x = min_lim(0);
  point.y = min_lim(1);
  bbox.points.push_back(point);
  bbox_pub_.publish(bbox);
}

template <class PATHLIBRARY, class UTILITYFUNCTION>
bool ExplorationServer<PATHLIBRARY, UTILITYFUNCTION>::getRobotPose(double &x, double &y, double &theta) const
{
  tf::StampedTransform transform;
  try
  {
    listener_.waitForTransform("map", "base_link", ros::Time(0), ros::Duration(2.0));
    listener_.lookupTransform("map", "base_link", ros::Time(0), transform);
  }
  catch (tf::TransformException ex)
  {
    ROS_ERROR("%s", ex.what());
    return false;
  }

  x = transform.getOrigin().x();
  y = transform.getOrigin().y();
  tf::Matrix3x3 R(transform.getRotation());
  double pitch, roll;
  R.getEulerYPR(theta, pitch, roll);

  return true;
}

template <class PATHLIBRARY, class UTILITYFUNCTION>
void ExplorationServer<PATHLIBRARY, UTILITYFUNCTION>::publishAll() const
{
  publishBBox();

  if (cost_map_pub_.getNumSubscribers())
  {
    grid_map_msgs::GridMapPtr cost_map_msg = boost::make_shared<grid_map_msgs::GridMap>();
    grid_map::GridMapRosConverter::toMessage(*path_library_.getCostMap(), *cost_map_msg);
    cost_map_msg->info.header.frame_id = "map";
    cost_map_msg->info.header.stamp = ros::Time::now();
    cost_map_pub_.publish(cost_map_msg);
  }

  // Publish if necessary
  utility_function_.publish();
}

template <class PATHLIBRARY, class UTILITYFUNCTION>
void ExplorationServer<PATHLIBRARY, UTILITYFUNCTION>::callback(const bruce_msgs::ISAM2UpdateConstPtr &isam2_update_msg,
                                                               const nav_msgs::OccupancyGridConstPtr &occ_grid_msg)
{
  if (query_exploration_path_srv_.getService().empty())
  {
    // Start server here
    ROS_INFO("slam update and occupancy map are received. Start query exploration server");
    query_exploration_path_srv_ = nh_.advertiseService("query_exploration_path", &This::queryExplorationPathSrv, this);
  }

#ifdef LAZY_EVAL
  isam2_update_msg_cache_ = *isam2_update_msg;
  occ_grid_msg_cache_ = *occ_grid_msg;
#else
  path_library_.setOccupancyMap(*occ_grid_msg);
  double x, y, theta;
  if (getRobotPose(x, y, theta))
  {
    path_library_.setOrigin(x, y, theta);
  }
  utility_function_.setOccupancyMap(*occ_grid_msg);
  utility_function_.setISAM2Update(*isam2_update_msg);

  publishAll();
#endif
}

template <class PATHLIBRARY, class UTILITYFUNCTION>
bool ExplorationServer<PATHLIBRARY, UTILITYFUNCTION>::queryExplorationPathSrv(
    bruce_msgs::QueryExplorationPathRequest &req, bruce_msgs::QueryExplorationPathResponse &resp)
{
  ROS_INFO("Query exploration path received");

#ifdef LAZY_EVAL
  path_library_.setOccupancyMap(occ_grid_msg_cache_);
  utility_function_.setOccupancyMap(occ_grid_msg_cache_);
  utility_function_.setISAM2Update(isam2_update_msg_cache_);
#endif

  double x, y, theta;
  if (getRobotPose(x, y, theta))
  {
    path_library_.setOrigin(x, y, theta);
  }
  else
  {
    ROS_ERROR("Failed to get robot pose");
    return false;
  }

  Targets::Ptr ret = path_library_.findAndPlanTargets();
  if (ret->targets.empty())
  {
    ROS_ERROR("No targets are detected");
    query_result_pub_.publish(ret->toMsg());
    return false;
  }

  std::vector<Path> smoothed;
  for (const auto &path : ret->paths)
  {
    nav_msgs::Path temp;
    csi_.interpolatePath(*path.toMsg(), temp);
    smoothed.emplace_back();
    smoothed.back().fromMsg(temp);
  }

  std::vector<Path> copy = ret->paths;
  ret->paths = smoothed;
  int best = utility_function_.evaluate(*ret);
  ret->paths = copy;

  if (best == -1)
  {
    ROS_ERROR("Failed to evaluate utility functions");
  }
  else
  {
    ret->best = ret->paths[best];
    resp.path = *smoothed[best].toMsg();
    resp.type = ret->targets[best].type;
  }

#ifdef LAZY_EVAL
  publishAll();
#endif

  query_result_pub_.publish(ret->toMsg());
  return best != -1;
}

}  // namespace base
}  // namespace bruce_exploration
