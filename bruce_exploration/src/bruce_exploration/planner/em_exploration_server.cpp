#include <bruce_msgs/conversions.h>
#include <grid_map_ros/GridMapRosConverter.hpp>
#include "bruce_exploration/planner/em_exploration_server.h"

// For logging
#include <gtsam/geometry/Pose2.h>
#include <fstream>

using namespace bruce_exploration::base;

namespace bruce_exploration
{
namespace planner
{
void saveAll(const VirtualMap &vm, const std::vector<BeliefState> &beliefs, const gtsam::NonlinearFactorGraph &graph,
             const gtsam::Values &values, const gtsam::NonlinearFactorGraph &new_graph, const gtsam::Values &new_values,
             std::string prefix, int n);

const std::string EMUtilityFunction::NS = UtilityFunction::NS + "em/";

EMUtilityFunction::EMUtilityFunction(ros::NodeHandle nh) : UtilityFunction(nh), virtual_map_(nh), belief_propagation_()
{
  nh.getParam(NS + "distance_weight", distance_weight_);

  virtual_map_pub_ = nh.advertise<grid_map_msgs::GridMap>("virtual_map", 1, true);
}

void EMUtilityFunction::publish() const
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

void EMUtilityFunction::setISAM2Update(const bruce_msgs::ISAM2Update &isam2_update)
{
  std::unique_lock<std::mutex> lock(mutex_);

  UtilityFunction::setISAM2Update(isam2_update);

  gtsam::ISAM2 isam2;
  bruce_msgs::fromMsg(isam2_update, isam2);
  belief_propagation_ = BeliefPropagationPlus();
  belief_propagation_.setISAM2(isam2);
}

void EMUtilityFunction::setOccupancyMap(const nav_msgs::OccupancyGrid &occ_grid)
{
  std::unique_lock<std::mutex> lock(mutex_);

  virtual_map_.setOccupancyMap(occ_grid);
}

int EMUtilityFunction::evaluate(const Targets &targets)
{
  std::unique_lock<std::mutex> lock(mutex_);

  ros::WallTime start_time = ros::WallTime::now();
  std::vector<Path> keyframes;
  std::vector<bruce_msgs::ISAM2Update> isam2_updates;
  predictSLAMUpdates(targets.paths, keyframes, isam2_updates);
  assert(isam2_updates.size() == targets.paths.size());

  auto beliefs0 = belief_propagation_.propagate();
  virtual_map_.setBeliefStates(beliefs0);

  gtsam::NonlinearFactorGraph graph0;
  gtsam::Values values0;
  if (log_)
  {
    graph0 = belief_propagation_.getISAM2().getFactorsUnsafe();
    values0 = belief_propagation_.getISAM2().calculateEstimate();
    saveAll(virtual_map_, beliefs0, graph0, values0, gtsam::NonlinearFactorGraph(), gtsam::Values(), log_prefix_, 0);
  }

  double ld0 = virtual_map_.getLogDOpt();

  std::vector<double> us;
  for (int n = 0; n < targets.paths.size(); ++n)
  {
    const auto &isam2_update = isam2_updates[n];
    gtsam::NonlinearFactorGraph new_graph;
    gtsam::Values new_values;
    bruce_msgs::fromMsg(isam2_update, new_graph, new_values);

    const auto &beliefs = belief_propagation_.propagate(new_graph, new_values);
    virtual_map_.setBeliefStates(beliefs);

    double ld = virtual_map_.getLogDOpt();
    double dist = computePathDistance(targets.paths[n]);
    double u = ld - ld0 + distance_weight_ * dist;

    if (verbose_)
    {
      if (us.empty() || u < *std::min_element(us.begin(), us.end()))
        ROS_INFO("\033[1;31mpath %i/%i: ld=%.2f, ld-ld0=%.2f, dist=%.2f, u=%.2f\033[0m", n + 1,
                 (int)targets.paths.size(), ld, ld - ld0, dist, u);
      else
        ROS_INFO("path %i/%i: ld=%.2f, ld-ld0=%.2f, dist=%.2f, u=%.2f", n + 1, (int)targets.paths.size(), ld, ld - ld0,
                 dist, u);
    }

    if (log_)
    {
      graph0 = belief_propagation_.getISAM2().getFactorsUnsafe();
      values0 = belief_propagation_.getISAM2().calculateEstimate();
      saveAll(virtual_map_, beliefs, graph0, values0, new_graph, new_values, log_prefix_, n + 1);
    }

    us.push_back(u);
  }

  ROS_INFO("Evaluate EM utility function (%d paths, %f sec)", (int)targets.paths.size(),
           (ros::WallTime::now() - start_time).toSec());
  auto it = std::min_element(us.begin(), us.end());
  return std::distance(us.begin(), it);
}

void saveBeliefStates(const std::vector<BeliefState> &beliefs, std::ostream &os)
{
  for (const auto &bs : beliefs)
  {
    os << bs.pose.x() << " " << bs.pose.y() << " " << bs.pose.theta() << " " << bs.cov(0, 0) << " " << bs.cov(0, 1)
       << " " << bs.cov(0, 2) << " " << bs.cov(1, 0) << " " << bs.cov(1, 1) << " " << bs.cov(1, 2) << " "
       << bs.cov(2, 0) << " " << bs.cov(2, 1) << " " << bs.cov(2, 2) << std::endl;
  }
}

void saveGraph(const gtsam::NonlinearFactorGraph &graph, const gtsam::Values &values, std::ostream &os)
{
  for (const auto &factor : graph)
  {
    if (factor->keys().size() != 2)
      continue;

    // key1 < key2
    gtsam::Key key1 = factor->keys()[0];
    gtsam::Key key2 = factor->keys()[1];

    // only save non sequential factors
    if (key2 > key1 + 1)
    {
      // Special case for BetweenFactor<Pose2> with Diagonal::Sigmas
      auto model = boost::static_pointer_cast<gtsam::NoiseModelFactor>(factor)->noiseModel();
      auto sigmas = boost::static_pointer_cast<gtsam::noiseModel::Diagonal>(model)->sigmas();
      gtsam::Pose2 pose1 = values.at<gtsam::Pose2>(key1);
      gtsam::Pose2 pose2 = values.at<gtsam::Pose2>(key2);
      os << pose1.x() << " " << pose1.y() << " " << pose1.theta() << " " << pose2.x() << " " << pose2.y() << " "
         << pose2.theta() << " " << sigmas(0) << " " << sigmas(1) << " " << sigmas(2) << std::endl;
    }
  }
}

void saveAll(const VirtualMap &vm, const std::vector<BeliefState> &beliefs, const gtsam::NonlinearFactorGraph &graph,
             const gtsam::Values &values, const gtsam::NonlinearFactorGraph &new_graph, const gtsam::Values &new_values,
             std::string prefix, int n)
{
  std::ofstream file1(prefix + "vm" + std::to_string(n) + ".csv");
  vm.save(file1);
  std::ofstream file2(prefix + "bs" + std::to_string(n) + ".csv");
  saveBeliefStates(beliefs, file2);
  gtsam::NonlinearFactorGraph gg(graph);
  gg += new_graph;
  gtsam::Values vv(values);
  vv.insert(new_values);
  std::ofstream file3(prefix + "graph" + std::to_string(n) + ".csv");
  saveGraph(gg, vv, file3);
}

}  // namespace planner
}  // namespace bruce_exploration
