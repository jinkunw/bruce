#pragma once
#include <nav_msgs/OccupancyGrid.h>
#include <ros/ros.h>

#include "3rdparty/nanoflann.hpp"
#include "bruce_exploration/base/cost_map.h"
#include "bruce_exploration/base/targets.h"

#define HEURISTIC

namespace bruce_exploration
{
namespace base
{
struct NodeLessThan
{
  bool operator()(const Node *node1, const Node *node2)
  {
#ifdef HEURISTIC
    return node1->cost + node1->cost_to_go < node2->cost + node2->cost_to_go;
#else
    return node1->cost < node2->cost;
#endif
  }
};

struct Graph
{
  /// Max degree of the graph
  int knn;
  /// index of node == node.id == index of points
  std::vector<Node> nodes;
  std::map<int, std::map<int, double>> nns;

  /// KDTree points
  MatrixXd points;
  typedef nanoflann::KDTreeEigenMatrixAdaptor<MatrixXd, 2, nanoflann::metric_L2_Simple> KDTree;
  std::shared_ptr<KDTree> tree = nullptr;

  std::vector<std::pair<double, int>> searchNodes(double x, double y, size_t knn, double radius = Node::MAX) const;
};

struct MotionModel
{
  static const std::string NS;
  double linear_velocity;
  double angular_velocity;

  MotionModel(ros::NodeHandle nh);
  bool check(const Node &from, const Node &to) const;
};

template <class MOTIONMODEL>
class PathLibraryBase
{
public:
  const static std::string NS;
  const static std::string TARGET_TYPE;
  typedef std::shared_ptr<PathLibraryBase<MOTIONMODEL>> Ptr;

  PathLibraryBase(ros::NodeHandle nh);

  CostMap::Ptr getCostMap() const
  {
    return cost_map_;
  }
  virtual void setOccupancyMap(const nav_msgs::OccupancyGrid &occ);

  virtual void setOrigin(double x0, double y0, double theta0);

  virtual Targets::Ptr findTargets();

  virtual Targets::Ptr planTargets(std::vector<Node> &goals);

  virtual Targets::Ptr findAndPlanTargets()
  {
    std::vector<Node> goals = findTargets()->targets;
    return planTargets(goals);
  }

protected:
  bool isNewFrontier(const Index &index) const;

  /// Frontier detection
  std::vector<Index> expandAndFilterFrontiers(const grid_map::Index &index);

  /// Search
  void initializeGraph();

  virtual void updateGraph(std::vector<Node> &goals);

  virtual bool isSteerable(const Node &from, const Node &to) const;

  double computeDistance(const Node &from, const Node &to, bool use_angle = true) const;

  virtual double computeCost(double distance, const Node &node) const;

  /// Heuristic cost-to-go
  /// Since we add clearance weight to Euclidean distance,
  /// the cost is not consistent and therefore the path is suboptimal.
  virtual double computeCostToGo(const Node &node, const std::map<int, Node> &not_reached_goals) const;

  double computeHeading(const Node &from, const Node &to) const;

protected:
  CostMap::Ptr cost_map_;
  double max_clearance_;
  double node_density_;
  double step_length_;
  double clearing_radius_;
  double frontier_resolution_;

  MOTIONMODEL motion_model_;

  Graph g_;

  // Collision free graph
  // But KDTree searching still happens in g_
  Graph working_g_;

private:
  const static std::string FRONTIER_LAYER;
  const static DataType FRONTIER_FALSE;
  const static DataType FRONTIER_TRUE;
  const static DataType FRONTIER_FILTERED;

  const static std::string VISITED_LAYER;
  const static DataType VISITED_FALSE;
  const static DataType VISITED_TRUE;
};

template <class MOTIONMODEL>
class RevisitPathLibraryBase : public PathLibraryBase<MOTIONMODEL>
{
public:
  const static std::string TARGET_TYPE;
  typedef PathLibraryBase<MOTIONMODEL> Base;
  typedef RevisitPathLibraryBase<MOTIONMODEL> This;

  RevisitPathLibraryBase(ros::NodeHandle nh);

  Targets::Ptr findTargets() override;

private:
  double revisitation_density_;
  double revisitation_separation_;
  double revisitation_radius_;
  double revisitation_max_num_;
};

typedef PathLibraryBase<MotionModel> PathLibrary;
typedef RevisitPathLibraryBase<MotionModel> RevisitPathLibrary;

}  // namespace base
}  // namespace bruce_exploration