#pragma once
#include <bruce_msgs/ISAM2Update.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Path.h>
#include <mutex>
#include "bruce_exploration/base/math.h"
#include "bruce_exploration/base/targets.h"
#include "bruce_exploration/base/virtual_map.h"

namespace bruce_exploration
{
namespace base
{
class UtilityFunction
{
public:
  const static std::string NS;

  UtilityFunction(ros::NodeHandle nh);

  virtual std::string getUtilityName() const
  {
    return std::string();
  }

  virtual void setISAM2Update(const bruce_msgs::ISAM2Update &isam2_update);

  virtual void setOccupancyMap(const nav_msgs::OccupancyGrid &occ_grid);

  /*
   * Evaluate utilty function for potential paths.
   * The batch evaluation is necessary to ensure that
   * the SLAM node remains unchanged during service calling.
   */

  //   virtual int evaluate(const std::vector<nav_msgs::Path> &paths) = 0;
  virtual int evaluate(const Targets &targets) = 0;

  //   double computePathDistance(const nav_msgs::Path &path) const;
  double computePathDistance(const Path &path) const;

  //   double computePathDistance(const geometry_msgs::Pose &from, const geometry_msgs::Pose &to) const;
  double computePathDistance(const Node &from, const Node &to) const;

  virtual void publish() const
  {
  }

protected:
  //   void predictSLAMUpdates(const std::vector<nav_msgs::Path> &paths, std::vector<nav_msgs::Path> &keyframes);
  void predictSLAMUpdates(const std::vector<Path> &paths, std::vector<Path> &keyframes);

  //   void predictSLAMUpdates(const std::vector<nav_msgs::Path> &paths, std::vector<nav_msgs::Path> &keyframes,
  //                           std::vector<bruce_msgs::ISAM2Update> &isam2_updates);
  void predictSLAMUpdates(const std::vector<Path> &paths, std::vector<Path> &keyframes,
                          std::vector<bruce_msgs::ISAM2Update> &isam2_updates);

  ros::ServiceClient predict_slam_update_client_;
  int key_;

  bool verbose_;
  bool log_;
  std::string log_prefix_;

  std::mutex mutex_;
};

}  // namespace base
}  // namespace bruce_exploration