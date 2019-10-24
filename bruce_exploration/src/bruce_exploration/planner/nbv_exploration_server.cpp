#include <grid_map_ros/GridMapRosConverter.hpp>
#include "bruce_exploration/planner/nbv_exploration_server.h"

using namespace bruce_exploration::base;

namespace bruce_exploration
{
namespace planner
{
const std::string NBVUtilityFunction::NS = UtilityFunction::NS + "nbv/";

NBVUtilityFunction::NBVUtilityFunction(ros::NodeHandle nh) : UtilityFunction(nh), virtual_map_(nh)
{
  nh.getParam(NS + "ig_free", ig_free_);
  nh.getParam(NS + "ig_occupied", ig_occupied_);
  nh.getParam(NS + "ig_unknown", ig_unknown_);
  nh.getParam(NS + "degressive_coeff", degressive_coeff_);

  virtual_map_pub_ = nh.advertise<grid_map_msgs::GridMap>("virtual_map", 1, true);
}

void NBVUtilityFunction::setOccupancyMap(const nav_msgs::OccupancyGrid &occ_grid)
{
  std::unique_lock<std::mutex> lock(mutex_);

  virtual_map_.setOccupancyMap(occ_grid);
}

int NBVUtilityFunction::evaluate(const Targets &targets)
{
  std::unique_lock<std::mutex> lock(mutex_);

  ros::WallTime start_time = ros::WallTime::now();
  std::vector<Path> keyframes;
  predictSLAMUpdates(targets.paths, keyframes);

  std::vector<double> igs;
  for (int n = 0; n < keyframes.size(); ++n)
  {
    const auto &path = keyframes[n];
    double ig = 0.0;
    double distance = 0.0;
    for (int k = 0; k < path.size(); ++k)
    {
      const auto &node2 = path[k];
      const auto &pose2 = node2.pose;

      double step_ig = 0.0;
      for (auto iter = virtual_map_.iterateVisible(pose2); !iter.isPastEnd(); ++iter)
      {
        step_ig += virtual_map_.isOccupancyFree(*iter) ?
                       ig_free_ :
                       (virtual_map_.isOccupancyUnknown(*iter) ? ig_unknown_ : ig_occupied_);
      }

      if (k >= 1)
      {
        const auto &node1 = path[k - 1];
        distance += computePathDistance(node1, node2);
      }
      ig += step_ig * std::exp(-degressive_coeff_ * distance);
    }

    if (verbose_)
    {
      if (igs.empty() || ig > *std::max_element(igs.begin(), igs.end()))
        ROS_INFO("\033[1;31mpath %i/%i: ig=%.5f, distance=%.2f\033[0m", n + 1, (int)targets.paths.size(), ig, distance);
      else
        ROS_INFO("path %i/%i: ig=%.5f, distance=%.2f", n + 1, (int)targets.paths.size(), ig, distance);
    }

    igs.push_back(ig);
  }

  ROS_INFO("Evaluate NBV utility function (%d paths, %f sec)", (int)targets.paths.size(),
           (ros::WallTime::now() - start_time).toSec());
  auto it = std::max_element(igs.begin(), igs.end());
  return std::distance(igs.begin(), it);
}

void NBVUtilityFunction::publish() const
{
  if (virtual_map_pub_.getNumSubscribers())
  {
    grid_map_msgs::GridMapPtr virtual_map_msg = boost::make_shared<grid_map_msgs::GridMap>();
    grid_map::GridMapRosConverter::toMessage(virtual_map_, *virtual_map_msg);
    virtual_map_msg->info.header.frame_id = "map";
    virtual_map_msg->info.header.stamp = ros::Time::now();
    virtual_map_pub_.publish(virtual_map_msg);
  }
}

}  // namespace planner
}  // namespace bruce_exploration