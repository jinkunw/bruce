#include <bruce_msgs/PredictSLAMUpdate.h>
#include <bruce_msgs/conversions.h>
#include "bruce_exploration/base/belief_propagation.h"
#include "bruce_exploration/base/math.h"
#include "bruce_exploration/base/utility_function.h"

namespace bruce_exploration
{
namespace base
{
const std::string UtilityFunction::NS = "utility_function/";

UtilityFunction::UtilityFunction(ros::NodeHandle nh)
{
  nh.getParam(NS + "verbose", verbose_);
  nh.getParam(NS + "log", log_);

  predict_slam_update_client_ = nh.serviceClient<bruce_msgs::PredictSLAMUpdate>("predict_slam_update");
}

void UtilityFunction::setISAM2Update(const bruce_msgs::ISAM2Update &isam2_update)
{
  key_ = isam2_update.key;
  log_prefix_ = getUtilityName() + "-step-" + std::to_string(key_) + "-";
}

void UtilityFunction::setOccupancyMap(const nav_msgs::OccupancyGrid &occ_grid)
{
}

// void UtilityFunction::predictSLAMUpdates(const std::vector<nav_msgs::Path> &paths,
//                                          std::vector<nav_msgs::Path> &keyframes)
// {
//   ros::WallTime start_time = ros::WallTime::now();
//   bruce_msgs::PredictSLAMUpdate srv;
//   srv.request.key = key_;
//   srv.request.paths = paths;
//   srv.request.return_isam2_update = false;
//   if (predict_slam_update_client_.call(srv))
//   {
//     keyframes = srv.response.keyframes;
//   }
//   else
//   {
//     ROS_ERROR_STREAM("Failed to call " << predict_slam_update_client_.getService());
//     return;
//   }
//   ROS_INFO("Predict SLAM update (%d paths, %f sec)", (int)paths.size(), (ros::WallTime::now() - start_time).toSec());
// }

void UtilityFunction::predictSLAMUpdates(const std::vector<Path> &paths, std::vector<Path> &keyframes)
{
  ros::WallTime start_time = ros::WallTime::now();

  std::vector<nav_msgs::Path> path_msgs;
  for (const auto &path : paths)
    path_msgs.push_back(*path.toMsg());

  bruce_msgs::PredictSLAMUpdate srv;
  srv.request.key = key_;
  srv.request.paths = path_msgs;
  srv.request.return_isam2_update = false;
  if (predict_slam_update_client_.call(srv))
  {
    for (const auto &path_msg : srv.response.keyframes)
    {
      keyframes.push_back(Path());
      keyframes.back().fromMsg(path_msg);
    }
  }
  else
  {
    ROS_ERROR_STREAM("Failed to call " << predict_slam_update_client_.getService());
    return;
  }
  ROS_INFO("Predict SLAM update (%d paths, %f sec)", (int)paths.size(), (ros::WallTime::now() - start_time).toSec());
}

void UtilityFunction::predictSLAMUpdates(const std::vector<Path> &paths, std::vector<Path> &keyframes,
                                         std::vector<bruce_msgs::ISAM2Update> &isam2_updates)
{
  ros::WallTime start_time = ros::WallTime::now();
  std::vector<nav_msgs::Path> path_msgs;
  for (const auto &path : paths)
    path_msgs.push_back(*path.toMsg());

  bruce_msgs::PredictSLAMUpdate srv;
  srv.request.key = key_;
  srv.request.paths = path_msgs;
  srv.request.return_isam2_update = true;
  if (predict_slam_update_client_.call(srv))
  {
    for (const auto &path_msg : srv.response.keyframes)
    {
      keyframes.push_back(Path());
      keyframes.back().fromMsg(path_msg);
    }
    isam2_updates = srv.response.isam2_updates;
  }
  else
  {
    ROS_ERROR_STREAM("Failed to call " << predict_slam_update_client_.getService());
    return;
  }
  ROS_INFO("Predict SLAM update with ISAM2 updates (%d paths, %f sec)", (int)paths.size(),
           (ros::WallTime::now() - start_time).toSec());
}

// double UtilityFunction::computePathDistance(const geometry_msgs::Pose &from, const geometry_msgs::Pose &to) const
// {
//   const double &x1 = from.position.x;
//   const double &y1 = from.position.y;
//   const auto &q1 = from.orientation;
//   const double &theta1 = yawFromQuaternion(q1.w, q1.x, q1.y, q1.z);

//   const double &x2 = to.position.x;
//   const double &y2 = to.position.y;
//   const auto &q2 = to.orientation;
//   const double &theta2 = yawFromQuaternion(q2.w, q2.x, q2.y, q2.z);

//   return std::sqrt(std::pow(x2 - x1, 2) + std::pow(y2 - y1, 2) + std::pow(ANGLE_WEIGHT * wrapToPi(theta2 - theta1),
//   2));
// }

double UtilityFunction::computePathDistance(const Node &from, const Node &to) const
{
  const double &x1 = from.pose.x();
  const double &y1 = from.pose.y();
  const auto &theta1 = from.pose.theta();

  const double &x2 = to.pose.x();
  const double &y2 = to.pose.y();
  const auto &theta2 = to.pose.theta();

  return std::sqrt(std::pow(x2 - x1, 2) + std::pow(y2 - y1, 2) + std::pow(ANGLE_WEIGHT * wrapToPi(theta2 - theta1), 2));
}

// double UtilityFunction::computePathDistance(const nav_msgs::Path &path) const
// {
//   double distance = 0.0;
//   for (int i = 1; i < path.poses.size(); ++i)
//   {
//     const auto &pose1 = path.poses[i - 1].pose;
//     const auto &pose2 = path.poses[i].pose;

//     distance += computePathDistance(pose1, pose2);
//   }
//   return distance;
// }

double UtilityFunction::computePathDistance(const Path &path) const
{
  double distance = 0.0;
  for (int i = 1; i < path.size(); ++i)
  {
    const auto &pose1 = path[i - 1];
    const auto &pose2 = path[i];

    distance += computePathDistance(pose1, pose2);
  }
  return distance;
}

}  // namespace base
}  // namespace bruce_exploration