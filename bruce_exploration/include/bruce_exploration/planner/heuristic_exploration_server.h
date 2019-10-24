#pragma once
#include "bruce_exploration/base/belief_propagation.h"
#include "bruce_exploration/base/exploration_server.h"
#include "bruce_exploration/base/path_library.h"
#include "bruce_exploration/base/utility_function.h"
#include "bruce_exploration/base/virtual_map.h"
#include "bruce_exploration/planner/nbv_exploration_server.h"

namespace bruce_exploration
{
namespace planner
{
class HeuristicUtilityFunction : public NBVUtilityFunction
{
public:
  const static std::string NS;

  HeuristicUtilityFunction(ros::NodeHandle nh);

  std::string getUtilityName() const override
  {
    return std::string("heuristic");
  }

  void setISAM2Update(const bruce_msgs::ISAM2Update &isam2_update) override;

  int evaluate(const base::Targets &targets) override;

private:
  base::BeliefPropagationPlus belief_propagation_;

  double alpha_;
  double max_det_;
};

typedef base::ExplorationServer<base::RevisitPathLibrary, HeuristicUtilityFunction> HeuristicExplorationServer;

}  // namespace planner
}  // namespace bruce_exploration
