#pragma once
#include <nav_msgs/Path.h>
#include <visualization_msgs/MarkerArray.h>
#include "bruce_exploration/base/types.h"

namespace bruce_exploration
{
namespace base
{
struct Node
{
  static const double MAX;
  Node(size_t id, double x, double y, double theta = NAN) : id(id), pose(x, y, theta)
  {
  }

  /// root : 0,
  /// invalid node in graph : = -1
  int id = -1;
  Pose pose;

  /// Euclidean distance between two nodes
  double distance = MAX;
  /// The cost used for searching shortest path
  double cost = MAX;
  /// estimated cost to target
  double cost_to_go = MAX;

  // Cached grid_map index
  Index index;

  /// Target type
  std::string type;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct Path : std::vector<Node>
{
  using std::vector<Node>::vector;

  nav_msgs::PathPtr toMsg() const;

  void fromMsg(const nav_msgs::Path &path);
};

struct Targets
{
  typedef std::shared_ptr<Targets> Ptr;

  /// Frontiers
  std::vector<Node> targets;

  /// Best one to go
  Path best;

  /// Candidate paths to evaluate
  std::vector<Path> paths;

  /// The entire tree for visualization
  std::vector<Path> edges;

  /// Visualization params
  static std::vector<double> target_color;
  static double target_size;
  static std::vector<double> tree_color;
  static double tree_size;
  static std::vector<double> path_color;
  static double path_size;
  static std::vector<double> best_path_color;
  static double best_path_size;

  virtual visualization_msgs::MarkerArrayPtr toMsg() const;
};
}  // namespace base

}  // namespace bruce_exploration