#include "bruce_exploration/planner/nf_exploration_server.h"

using namespace bruce_exploration::base;

namespace bruce_exploration
{
namespace planner
{
const std::string NFUtilityFunction::NS = "utility_function/nf/";

NFUtilityFunction::NFUtilityFunction(ros::NodeHandle nh) : UtilityFunction(nh)
{
}

int NFUtilityFunction::evaluate(const Targets &targets)
{
  ros::WallTime start_time = ros::WallTime::now();
  if (targets.paths.empty())
    return -1;

  std::vector<double> distances;
  for (const auto &path : targets.paths)
    distances.push_back(computePathDistance(path));

  ROS_INFO("Evaluate NF utility function (%d paths, %f sec)", (int)targets.paths.size(), (ros::WallTime::now() - start_time).toSec());
  auto it = std::min_element(distances.begin(), distances.end());
  return std::distance(distances.begin(), it);
}

}  // namespace planner
}  // namespace bruce_exploration