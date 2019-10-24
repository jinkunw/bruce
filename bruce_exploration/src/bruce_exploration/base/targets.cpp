#include "bruce_exploration/base/math.h"
#include "bruce_exploration/base/targets.h"

namespace bruce_exploration
{
namespace base
{
const double Node::MAX = std::numeric_limits<double>::max();
std::vector<double> Targets::target_color = { 1.0, 0.0, 1.0, 1.0 };
double Targets::target_size = 2.0;
std::vector<double> Targets::tree_color = { 0.0, 1.0, 0.0, 0.5 };
double Targets::tree_size = 0.2;
std::vector<double> Targets::path_color = { 0.0, 0.0, 1.0, 1.0 };
double Targets::path_size = 0.5;
std::vector<double> Targets::best_path_color = { 1.0, 0.0, 0.0, 1.0 };
double Targets::best_path_size = 0.8;

nav_msgs::PathPtr Path::toMsg() const
{
  nav_msgs::PathPtr path_msg(new nav_msgs::Path);
  for (const auto &node : *this)
  {
    path_msg->header.frame_id = "map";
    geometry_msgs::PoseStamped pose_msg;
    pose_msg.header.frame_id = "map";
    pose_msg.pose.position.x = node.pose.x();
    pose_msg.pose.position.y = node.pose.y();
    auto q = quaternionFromYaw(node.pose.theta());
    pose_msg.pose.orientation.w = q.w();
    pose_msg.pose.orientation.x = q.x();
    pose_msg.pose.orientation.y = q.y();
    pose_msg.pose.orientation.z = q.z();
    path_msg->poses.push_back(pose_msg);
  }
  return path_msg;
}

void Path::fromMsg(const nav_msgs::Path &path)
{
  clear();
  for (const auto &pose : path.poses)
  {
    const double &x = pose.pose.position.x;
    const double &y = pose.pose.position.y;
    auto q = pose.pose.orientation;
    const double &th = yawFromQuaternion(q.w, q.x, q.y, q.z);
    emplace_back(0, x, y, th);
  }
}

visualization_msgs::MarkerArrayPtr Targets::toMsg() const
{
  visualization_msgs::MarkerArrayPtr markers(new visualization_msgs::MarkerArray);
  markers->markers.emplace_back();
  markers->markers.back().action = visualization_msgs::Marker::DELETEALL;

  /////////////////////////////////////////////////////////////
  // Targets without specific headings, e.g. frontiers
  /////////////////////////////////////////////////////////////
  markers->markers.emplace_back();
  visualization_msgs::Marker &points = markers->markers.back();
  points.header.stamp = ros::Time::now();
  points.header.frame_id = "map";
  points.ns = "frontiers";
  points.type = visualization_msgs::Marker::CUBE_LIST;
  points.id = 0;
  points.scale.x = target_size;
  points.scale.y = target_size;
  points.scale.z = target_size;
  points.color.r = target_color[0];
  points.color.g = target_color[1];
  points.color.b = target_color[2];
  points.color.a = target_color[3];

  for (const Node &node : targets)
  {
    if (!std::isnan(node.pose.theta()))
      continue;
    geometry_msgs::Point point;
    point.x = node.pose.x();
    point.y = node.pose.y();
    points.points.push_back(point);
  }

  /////////////////////////////////////////////////////////////
  // Targets with predefined headings, e.g. revisitations
  /////////////////////////////////////////////////////////////
  for (int i = 0; i < targets.size(); ++i)
  {
    if (std::isnan(targets[i].pose.theta()))
      continue;
    markers->markers.emplace_back();
    visualization_msgs::Marker &arrow = markers->markers.back();
    arrow.header.stamp = ros::Time::now();
    arrow.header.frame_id = "map";
    arrow.ns = "revisitations";
    arrow.type = visualization_msgs::Marker::ARROW;
    arrow.id = i;
    arrow.pose.position.x = targets[i].pose.x();
    arrow.pose.position.y = targets[i].pose.y();
    auto q = quaternionFromYaw(targets[i].pose.theta());
    arrow.pose.orientation.x = q.x();
    arrow.pose.orientation.y = q.y();
    arrow.pose.orientation.z = q.z();
    arrow.pose.orientation.w = q.w();
    arrow.scale.x = 2.0 * target_size;
    arrow.scale.y = 0.5 * target_size;
    arrow.scale.z = 0.5 * target_size;
    arrow.color.r = target_color[0];
    arrow.color.g = target_color[1];
    arrow.color.b = target_color[2];
    arrow.color.a = target_color[3];
  }

  /////////////////////////////////////////////////////////////
  // The entire search tree
  /////////////////////////////////////////////////////////////
  markers->markers.emplace_back();
  visualization_msgs::Marker &lines1 = markers->markers.back();
  lines1.header.stamp = ros::Time::now();
  lines1.header.frame_id = "map";
  lines1.ns = "trees";
  lines1.type = visualization_msgs::Marker::LINE_LIST;
  lines1.id = 0;
  lines1.scale.x = tree_size;
  lines1.color.r = tree_color[0];
  lines1.color.g = tree_color[1];
  lines1.color.b = tree_color[2];
  lines1.color.a = tree_color[3];

  for (int i = 0; i < edges.size(); ++i)
  {
    geometry_msgs::Point point1, point2;
    point1.x = edges[i][0].pose.x();
    point1.y = edges[i][0].pose.y();
    point1.z = -0.05;
    point2.x = edges[i][1].pose.x();
    point2.y = edges[i][1].pose.y();
    point2.z = -0.05;
    lines1.points.push_back(point1);
    lines1.points.push_back(point2);
  }

  /////////////////////////////////////////////////////////////
  // The candidate paths
  /////////////////////////////////////////////////////////////
  markers->markers.emplace_back();
  visualization_msgs::Marker &lines2 = markers->markers.back();
  lines2.ns = "paths";
  lines2.header.stamp = ros::Time::now();
  lines2.header.frame_id = "map";
  lines2.type = visualization_msgs::Marker::LINE_LIST;
  lines2.id = 0;
  lines2.scale.x = path_size;
  lines2.color.r = path_color[0];
  lines2.color.g = path_color[1];
  lines2.color.b = path_color[2];
  lines2.color.a = path_color[3];
  for (const auto &path : paths)
  {
    for (int i = 1; i < path.size(); ++i)
    {
      geometry_msgs::Point point1, point2;
      point1.x = path[i - 1].pose.x();
      point1.y = path[i - 1].pose.y();
      point1.z = -0.1;
      point2.x = path[i].pose.x();
      point2.y = path[i].pose.y();
      point2.z = -0.1;
      lines2.points.push_back(point1);
      lines2.points.push_back(point2);
    }
  }

  /////////////////////////////////////////////////////////////
  // The best path
  /////////////////////////////////////////////////////////////
  markers->markers.emplace_back();
  visualization_msgs::Marker &lines3 = markers->markers.back();
  lines3.ns = "best";
  lines3.header.stamp = ros::Time::now();
  lines3.header.frame_id = "map";
  lines3.type = visualization_msgs::Marker::LINE_STRIP;
  lines3.id = 0;
  lines3.scale.x = best_path_size;
  lines3.color.r = best_path_color[0];
  lines3.color.g = best_path_color[1];
  lines3.color.b = best_path_color[2];
  lines3.color.a = best_path_color[3];
  for (const Node &node : best)
  {
    geometry_msgs::Point point;
    point.x = node.pose.x();
    point.y = node.pose.y();
    point.z = -0.2;
    lines3.points.push_back(point);
  }

  return markers;
}

}  // namespace base

}  // namespace bruce_exploration