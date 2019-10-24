#pragma once
#include "bruce_exploration/base/belief_propagation.h"
#include "bruce_exploration/base/exploration_server.h"
#include "bruce_exploration/base/path_library.h"
#include "bruce_exploration/base/utility_function.h"
#include "bruce_exploration/base/virtual_map.h"

namespace bruce_exploration
{
namespace planner
{
class EMUtilityFunction : public base::UtilityFunction
{
public:
  const static std::string NS;

  EMUtilityFunction(ros::NodeHandle nh);

  std::string getUtilityName() const override
  {
    return std::string("em");
  }

  void setISAM2Update(const bruce_msgs::ISAM2Update &isam2_update) override;

  void setOccupancyMap(const nav_msgs::OccupancyGrid &occ_grid) override;

  int evaluate(const base::Targets &targets) override;

  void publish() const override;

private:
  base::VirtualMap virtual_map_;
  base::BeliefPropagationPlus belief_propagation_;

  double distance_weight_;
  ros::Publisher virtual_map_pub_;
};

typedef base::ExplorationServer<base::RevisitPathLibrary, EMUtilityFunction> EMExplorationServer;

}  // namespace planner
}  // namespace bruce_exploration
