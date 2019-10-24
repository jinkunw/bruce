#pragma once
#include "bruce_exploration/base/exploration_server.h"
#include "bruce_exploration/base/path_library.h"
#include "bruce_exploration/base/utility_function.h"
#include "bruce_exploration/base/virtual_map.h"

namespace bruce_exploration
{
namespace planner
{
class NBVUtilityFunction : public base::UtilityFunction
{
public:
  const static std::string NS;

  NBVUtilityFunction(ros::NodeHandle nh);

  std::string getUtilityName() const override
  {
    return std::string("nbv");
  }

  void setOccupancyMap(const nav_msgs::OccupancyGrid &occ_grid) override;

  int evaluate(const base::Targets &targets) override;

  void publish() const override;

protected:
  base::VirtualMap virtual_map_;

  double ig_free_;
  double ig_occupied_;
  double ig_unknown_;
  double degressive_coeff_;

  ros::Publisher virtual_map_pub_;
};

typedef base::ExplorationServer<base::PathLibrary, NBVUtilityFunction> NBVExplorationServer;

}  // namespace planner
}  // namespace bruce_exploration