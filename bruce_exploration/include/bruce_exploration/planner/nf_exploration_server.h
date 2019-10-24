#pragma once
#include "bruce_exploration/base/exploration_server.h"
#include "bruce_exploration/base/path_library.h"
#include "bruce_exploration/base/utility_function.h"

namespace bruce_exploration
{
namespace planner
{
class NFUtilityFunction : public base::UtilityFunction
{
public:
  const static std::string NS;

  NFUtilityFunction(ros::NodeHandle nh);
  
  std::string getUtilityName() const
  {
    return std::string("nf");
  }

//   int evaluate(const std::vector<nav_msgs::Path> &paths) override;
  int evaluate(const base::Targets &targets) override;
};

typedef base::ExplorationServer<base::PathLibrary, NFUtilityFunction> NFExplorationServer;

}  // namespace planner
}  // namespace bruce_exploration