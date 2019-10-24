#include <bruce_msgs/conversions.h>
#include <grid_map_ros/GridMapRosConverter.hpp>
#include "bruce_exploration/planner/heuristic_exploration_server.h"

using namespace bruce_exploration::base;

namespace bruce_exploration
{
namespace planner
{
const std::string HeuristicUtilityFunction::NS = UtilityFunction::NS + "heuristic/";

HeuristicUtilityFunction::HeuristicUtilityFunction(ros::NodeHandle nh) : NBVUtilityFunction(nh), belief_propagation_()
{
  nh.getParam(NS + "alpha", alpha_);
  nh.getParam(NS + "max_det", max_det_);
}

void HeuristicUtilityFunction::setISAM2Update(const bruce_msgs::ISAM2Update &isam2_update)
{
  std::unique_lock<std::mutex> lock(mutex_);

  UtilityFunction::setISAM2Update(isam2_update);

  gtsam::ISAM2 isam2;
  bruce_msgs::fromMsg(isam2_update, isam2);
  belief_propagation_ = BeliefPropagationPlus();
  belief_propagation_.setISAM2(isam2);
}

int HeuristicUtilityFunction::evaluate(const Targets &targets)
{
  std::unique_lock<std::mutex> lock(mutex_);

  ros::WallTime start_time = ros::WallTime::now();

  double current_det = belief_propagation_.propagate().back().cov.determinant();
  if (current_det < max_det_)
  {
    ROS_INFO("Current det(cov)=%.5f < %.5f. Switch to nbv", current_det, max_det_);
    // Use NBV on frontiers
    Targets frontier_targets;
    std::map<int, int> index_map;
    for (int n = 0; n < targets.paths.size(); ++n)
    {
      if (targets.targets[n].type == PathLibrary::TARGET_TYPE)
      {
        index_map[frontier_targets.targets.size()] = n;
        frontier_targets.targets.push_back(targets.targets[n]);
        frontier_targets.paths.push_back(targets.paths[n]);
      }
    }
    lock.unlock();
    int best = NBVUtilityFunction::evaluate(frontier_targets);
    return index_map[best];
  }
  ROS_INFO("Current det(cov)=%.5f > %.5f. Switch to heuristic", current_det, max_det_);

  double ig_max = 0.0;
  for (auto iter = virtual_map_.iterate(); !iter.isPastEnd(); ++iter)
  {
    ig_max += virtual_map_.isOccupancyFree(*iter) ? ig_free_ : (virtual_map_.isOccupancyUnknown(*iter) ? ig_unknown_ :
                                                                                                         ig_occupied_);
  }

  std::vector<Path> keyframes;
  std::vector<bruce_msgs::ISAM2Update> isam2_updates;
  predictSLAMUpdates(targets.paths, keyframes, isam2_updates);
  assert(isam2_updates.size() == targets.paths.size());

  std::vector<double> us;
  for (int n = 0; n < keyframes.size(); ++n)
  {
    const auto &isam2_update = isam2_updates[n];
    gtsam::NonlinearFactorGraph new_graph;
    gtsam::Values new_values;
    bruce_msgs::fromMsg(isam2_update, new_graph, new_values);

    const auto &beliefs = belief_propagation_.propagate(new_graph, new_values);
    double det = beliefs.back().cov.determinant();
    double u_robot = std::pow(det / max_det_, 1.0 / 3.0);

    const auto &path = keyframes[n];
    double ig = 0.0;
    double distance = 0.0;
    for (int k = 0; k < keyframes[n].size(); ++k)
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
    double u_ig = -ig / ig_max;
    double u = alpha_ * u_robot + (1.0 - alpha_) * u_ig;

    if (verbose_)
    {
      if (us.empty() || u < *std::min_element(us.begin(), us.end()))
        ROS_INFO("\033[1;31mpath %i/%i: u_robot=%.5f, alpha*u_robot=%.5f, -u_ig=%.5f, -(1-alpha)*u_ig=%.5f, "
                 "u=%.5f\033[0m",
                 n + 1, (int)targets.paths.size(), u_robot, alpha_ * u_robot, u_ig, (1.0 - alpha_) * u_ig, u);
      else
        ROS_INFO("path %i/%i: u_robot=%.5f, alpha*u_robot=%.5f, -u_ig=%.5f, -(1-alpha)*u_ig=%.5f, u=%.5f", n + 1,
                 (int)targets.paths.size(), u_robot, alpha_ * u_robot, u_ig, (1.0 - alpha_) * u_ig, u);
    }

    us.push_back(u);
  }

  ROS_INFO("Evaluate heuristic utility function (%d paths, %f sec)", (int)targets.paths.size(),
           (ros::WallTime::now() - start_time).toSec());
  auto it = std::min_element(us.begin(), us.end());
  return std::distance(us.begin(), it);
}

}  // namespace planner
}  // namespace bruce_exploration
