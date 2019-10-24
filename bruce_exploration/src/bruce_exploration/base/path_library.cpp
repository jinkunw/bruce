#include <grid_map_core/grid_map_core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc.hpp>
#include <queue>

#include "3rdparty/BinaryHeap.h"
#include "3rdparty/halton.hpp"
#include "bruce_exploration/base/math.h"
#include "bruce_exploration/base/path_library.h"

namespace bruce_exploration
{
namespace base
{
const std::string MotionModel::NS = "motion_model/";
template <class MOTIONMODEL>
const std::string PathLibraryBase<MOTIONMODEL>::NS = "path_library/";
template <class MOTIONMODEL>
const std::string PathLibraryBase<MOTIONMODEL>::TARGET_TYPE = "frontier";
template <class MOTIONMODEL>
const std::string PathLibraryBase<MOTIONMODEL>::FRONTIER_LAYER = "frontier";
template <class MOTIONMODEL>
const DataType PathLibraryBase<MOTIONMODEL>::FRONTIER_FALSE = 0.0;
template <class MOTIONMODEL>
const DataType PathLibraryBase<MOTIONMODEL>::FRONTIER_TRUE = 1.0;
template <class MOTIONMODEL>
const DataType PathLibraryBase<MOTIONMODEL>::FRONTIER_FILTERED = 2.0;
template <class MOTIONMODEL>
const std::string PathLibraryBase<MOTIONMODEL>::VISITED_LAYER = "frontier_visited";
template <class MOTIONMODEL>
const DataType PathLibraryBase<MOTIONMODEL>::VISITED_FALSE = 0.0;
template <class MOTIONMODEL>
const DataType PathLibraryBase<MOTIONMODEL>::VISITED_TRUE = 1.0;
template <class MOTIONMODEL>
const std::string RevisitPathLibraryBase<MOTIONMODEL>::TARGET_TYPE = "revisit";

std::vector<std::pair<double, int>> Graph::searchNodes(double x, double y, size_t knn, double radius) const
{
  if (!tree)
  {
    ROS_ERROR("KDTree is not built");
    return {};
  }

  std::vector<size_t> indices(knn);
  std::vector<double> indices_dists(knn);
  nanoflann::KNNResultSet<double> result_set(knn);
  result_set.init(&indices[0], &indices_dists[0]);

  std::vector<std::pair<double, int>> ret;
  double query_pt[2] = { x, y };
  tree->index->findNeighbors(result_set, query_pt, nanoflann::SearchParams());
  for (size_t k = 0; k < knn; ++k)
  {
    size_t j = indices[k] + 1;
    double distance = std::sqrt(indices_dists[k]);
    if (distance < radius)
      ret.emplace_back(distance, j);
  }

  return ret;
}

MotionModel::MotionModel(ros::NodeHandle nh)
{
  nh.getParam(NS + "linear_velocity", linear_velocity);
  nh.getParam(NS + "angular_velocity", angular_velocity);
}

bool MotionModel::check(const Node &from, const Node &to) const
{
  double theta1 = from.pose.theta();
  double theta2 = to.pose.theta();
  if (std::isnan(theta2))
    theta2 = std::atan2(to.pose.y() - from.pose.y(), to.pose.x() - from.pose.x());

  double dl = (from.pose - to.pose).head(2).norm();
  double dt = dl / linear_velocity;
  double dy = wrapToPi(theta2 - theta1);
  return dy < dt * angular_velocity;
}

template <class MOTIONMODEL>
PathLibraryBase<MOTIONMODEL>::PathLibraryBase(ros::NodeHandle nh) : motion_model_(nh)
{
  double xmin, xmax, ymin, ymax, resolution, inflation_radius;
  nh.getParam(NS + "xmin", xmin);
  nh.getParam(NS + "ymin", ymin);
  nh.getParam(NS + "xmax", xmax);
  nh.getParam(NS + "ymax", ymax);
  nh.getParam(NS + "resolution", resolution);
  nh.getParam(NS + "inflation_radius", inflation_radius);
  cost_map_ = std::make_shared<CostMap>(xmin, xmax, ymin, ymax, resolution, inflation_radius);

  nh.getParam(NS + "node_density", node_density_);
  nh.getParam(NS + "max_clearance", max_clearance_);
  nh.getParam(NS + "step_length", step_length_);
  nh.getParam(NS + "clearing_radius", clearing_radius_);
  nh.getParam(NS + "frontier_resolution", frontier_resolution_);

  cost_map_->add(VISITED_LAYER, VISITED_FALSE);
  cost_map_->add(FRONTIER_LAYER, FRONTIER_FALSE);

  initializeGraph();
  ROS_INFO("Initialize graph with %d nodes.", (int)g_.points.rows());
}

template <class MOTIONMODEL>
void PathLibraryBase<MOTIONMODEL>::initializeGraph()
{
  double xmin = cost_map_->getLimitMin()(0);
  double ymin = cost_map_->getLimitMin()(1);
  double xmax = cost_map_->getLimitMax()(0);
  double ymax = cost_map_->getLimitMax()(1);
  int max_nodes = (ymax - ymin) * (xmax - xmin) * node_density_;
  g_.points.resize(max_nodes, 2);
  // Add a temporary root
  g_.nodes.emplace_back(0, 0, 0, 0);
  while (g_.nodes.size() - 1 < max_nodes)
  {
    int id = (int)g_.nodes.size();
    double *values = halton(id, 2);
    double x = xmin + (xmax - xmin) * values[0];
    double y = ymin + (ymax - ymin) * values[1];
    delete[] values;

    Index index;
    if (cost_map_->getIndex(Position(x, y), index))
    {
      g_.nodes.emplace_back(id, x, y);
      g_.nodes.back().index = index;
      g_.points(id - 1, 0) = x;
      g_.points(id - 1, 1) = y;
    }
  }

  g_.tree = std::make_shared<Graph::KDTree>(2, std::cref(g_.points));
  g_.knn = std::ceil(2.0 * M_E * std::log((double)g_.nodes.size()));
  ROS_INFO("Use knn = %d", g_.knn);
  for (const Node &node_i : g_.nodes)
  {
    const int &i = node_i.id;
    auto nn = g_.searchNodes(node_i.pose.x(), node_i.pose.y(), g_.knn, step_length_);
    for (const auto &pair_j : nn)
    {
      const int &j = pair_j.second;
      const double &distance = pair_j.first;
      if (i != j)
      {
        g_.nns[i][j] = distance;
      }
    }
  }
}

template <class MOTIONMODEL>
void PathLibraryBase<MOTIONMODEL>::updateGraph(std::vector<Node> &goals)
{
  working_g_ = Graph();
  working_g_.knn = g_.knn;
  working_g_.nodes = g_.nodes;

  // Remove nodes and edges that aren't collision-free.
  for (int i = 1; i < working_g_.nodes.size(); ++i)
  {
    if (!cost_map_->isCollisionFree(working_g_.nodes[i].index))
      working_g_.nodes[i].id = -1;
  }

  for (int i = 1; i < working_g_.nodes.size(); ++i)
  {
    const Node &node_i = working_g_.nodes[i];

    if (node_i.id < 0)
      continue;

    for (const auto &pair_j : g_.nns[i])
    {
      int j = pair_j.first;
      const Node &node_j = working_g_.nodes.at(j);

      if (node_j.id < 0)
        continue;

      // Skip collision checking if j -> i has been checked
      if (working_g_.nns.find(j) != working_g_.nns.end())
      {
        if (working_g_.nns[j].find(i) != working_g_.nns[j].end())
          working_g_.nns[i][j] = pair_j.second;
        else
          continue;
      }

      if (cost_map_->isCollisionFree(node_i.index, node_j.index))
        working_g_.nns[i][j] = pair_j.second;
    }
  }

  // Add start point
  Node &root = working_g_.nodes[0];
  root.pose = cost_map_->getOrigin();
  root.distance = 0;
  root.cost = 0;
  cost_map_->getIndex(Position(root.pose.x(), root.pose.y()), root.index);

  auto nn = g_.searchNodes(root.pose.x(), root.pose.y(), working_g_.knn, step_length_);
  for (const auto &pair : nn)
  {
    const Node &node_j = working_g_.nodes[pair.second];
    if (node_j.id > 0 && cost_map_->isCollisionFree(root.index, node_j.index))
      working_g_.nns[root.id][node_j.id] = pair.first;
  }

  // Add goals, ignore edges between goals
  // Update goal id if it's not set
  for (Node &goal : goals)
  {
    if (goal.id < 0)
      goal.id = (int)working_g_.nodes.size();
    goal.cost = goal.distance = Node::MAX;
    cost_map_->getIndex(Position(goal.pose.x(), goal.pose.y()), goal.index);
    working_g_.nodes.push_back(goal);

    auto nn = g_.searchNodes(goal.pose.x(), goal.pose.y(), working_g_.knn, step_length_);
    for (const auto &pair : nn)
    {
      const Node &node_i = working_g_.nodes[pair.second];
      if (node_i.id >= 0 && cost_map_->isCollisionFree(node_i.index, goal.index))
      {
        working_g_.nns[goal.id][node_i.id] = pair.first;
        working_g_.nns[node_i.id][goal.id] = pair.first;
      }
    }
  }
}

template <class MOTIONMODEL>
void PathLibraryBase<MOTIONMODEL>::setOccupancyMap(const nav_msgs::OccupancyGrid &occ)
{
  cost_map_->setOccupancyMap(occ);
}

template <class MOTIONMODEL>
void PathLibraryBase<MOTIONMODEL>::setOrigin(double x0, double y0, double theta0)
{
  if (!cost_map_->isInside(Position(x0, y0)))
  {
    ROS_ERROR("Set origin failed. Robot is out of grid map");
    return;
  }

  if (!cost_map_->isCollisionFree(x0, y0))
  {
    ROS_ERROR("Robot is close to obstacles. Clear robot cells within %.2f m", clearing_radius_);
    cost_map_->clearCells(x0, y0, clearing_radius_);
  }

  cost_map_->setOrigin(x0, y0, theta0);
}

template <class MOTIONMODEL>
Targets::Ptr PathLibraryBase<MOTIONMODEL>::findTargets()
{
  ros::WallTime start_time = ros::WallTime::now();
  Targets::Ptr ret = std::make_shared<Targets>();

  // Reset
  cost_map_->get(FRONTIER_LAYER).fill(FRONTIER_FALSE);
  cost_map_->get(VISITED_LAYER).fill(VISITED_FALSE);

  Index index0;
  cost_map_->getIndex(cost_map_->getOrigin().head(2), index0);

  std::queue<Index> q;
  q.push(index0);

  std::vector<Index> frontier_indices;
  while (!q.empty())
  {
    Index index = q.front();
    q.pop();

    for (const auto &nh : cost_map_->getNeighborIndex4(index))
    {
      if (cost_map_->at(VISITED_LAYER, nh) == VISITED_TRUE)
        continue;

      if (isNewFrontier(nh))
      {
        std::vector<Index> new_indices = expandAndFilterFrontiers(nh);
        frontier_indices.insert(frontier_indices.end(), new_indices.begin(), new_indices.end());
      }
      else if (cost_map_->isCollisionFree(nh) && cost_map_->isOccupancyFree(nh))
      {
        cost_map_->at(VISITED_LAYER, nh) = VISITED_TRUE;
        q.push(nh);
      }
    }
  }

  for (const auto &index : frontier_indices)
  {
    Position position;
    cost_map_->getPosition(index, position);
    ret->targets.emplace_back(-1, position(0), position(1));
    ret->targets.back().index = index;
    ret->targets.back().type = TARGET_TYPE;
  }

  ROS_INFO("Find frontiers (%d frontier cells, %f sec)", (int)frontier_indices.size(),
           (ros::WallTime::now() - start_time).toSec());

  return ret;
}

template <class MOTIONMODEL>
std::vector<Index> PathLibraryBase<MOTIONMODEL>::expandAndFilterFrontiers(const Index &index0)
{
  std::queue<Index> q;
  q.push(index0);
  typedef std::pair<Index, double> IndexClearance;
  std::vector<IndexClearance> frontiers;

  while (!q.empty())
  {
    Index index = q.front();
    q.pop();

    frontiers.emplace_back(std::make_pair(index, cost_map_->getClearance(index)));
    cost_map_->at(FRONTIER_LAYER, index) = FRONTIER_TRUE;

    for (const auto &nh : cost_map_->getNeighborIndex8(index))
    {
      if (isNewFrontier(nh))
      {
        cost_map_->at(FRONTIER_LAYER, nh) = FRONTIER_TRUE;
        q.push(nh);
      }
    }
  }

  std::vector<Index> filtered;
  if (frontiers.size() < 2)
    return filtered;

  double min_clr = cost_map_->getInflationRadius();
  while (!frontiers.empty())
  {
    IndexClearance best_index =
        *std::max_element(frontiers.begin(), frontiers.end(),
                          [](const IndexClearance &a, const IndexClearance &b) { return a.second < b.second; });
    if (!filtered.empty() && best_index.second < min_clr)
      break;

    filtered.push_back(best_index.first);

    cost_map_->at(FRONTIER_LAYER, best_index.first) = FRONTIER_FILTERED;

    auto erased = std::remove_if(frontiers.begin(), frontiers.end(), [this, best_index](const IndexClearance &a) {
      double dist = (best_index.first - a.first).matrix().norm() * cost_map_->getResolution();
      return dist < frontier_resolution_;
    });
    frontiers.erase(erased, frontiers.end());
  }

  return filtered;
}

template <class MOTIONMODEL>
bool PathLibraryBase<MOTIONMODEL>::isNewFrontier(const Index &index) const
{
  if (cost_map_->at(FRONTIER_LAYER, index) != FRONTIER_FALSE || !cost_map_->isOccupancyFree(index) ||
      !cost_map_->isCollisionFree(index))
    return false;

  for (const auto &nh : cost_map_->getNeighborIndex4(index))
  {
    if (cost_map_->isOccupancyUnknown(nh))
      return true;
  }
  return false;
}

template <class MOTIONMODEL>
Targets::Ptr PathLibraryBase<MOTIONMODEL>::planTargets(std::vector<Node> &goals)
{
  ros::WallTime start_time = ros::WallTime::now();

  std::vector<Node> filtered;
  const auto &limit_min = cost_map_->getLimitMin();
  const auto &limit_max = cost_map_->getLimitMax();
  const double &safe_dist = 1.0;
  for (const Node &goal : goals)
  {
    if (std::abs(goal.pose.x() - limit_min.x()) > safe_dist && std::abs(goal.pose.x() - limit_max.x()) > safe_dist &&
        std::abs(goal.pose.y() - limit_min.y()) > safe_dist && std::abs(goal.pose.y() - limit_max.y()) > safe_dist)
      filtered.push_back(goal);
  }
  goals = filtered;

  Targets::Ptr ret = std::make_shared<Targets>();

  if (goals.empty())
    return ret;

  updateGraph(goals);

  typedef ompl::BinaryHeap<Node *, NodeLessThan> PriorityQueue;
  PriorityQueue open;
  std::map<int, PriorityQueue::Element *> elements;
  elements[0] = open.insert(&working_g_.nodes[0]);
  std::set<int> visited;

  std::map<int, Node> not_reached;
  for (const Node &goal : goals)
    not_reached.insert({ goal.id, goal });

  std::map<int, int> tree;
  while (!open.empty())
  {
    if (!goals.empty() && not_reached.empty())
    {
      ROS_INFO("All goals are reached");
      break;
    }

    const Node *node_i = open.top()->data;
    open.remove(open.top());

    int i = node_i->id;
    const double &distance_i = node_i->distance;
    const double &cost_i = node_i->cost;

    if (working_g_.nns.find(i) == working_g_.nns.end())
    {
      visited.insert(i);
      continue;
    }

    for (const auto &pair : working_g_.nns.at(i))
    {
      const int &j = pair.first;
      if (visited.find(j) != visited.end())
        continue;

      Node &node_j = working_g_.nodes.at(j);

      // Ignore checking on root node
      if (node_i->id && !isSteerable(*node_i, node_j))
        continue;

      const double &cost_j = node_j.cost;
      const double &distance_ij = pair.second;
      double cost_ij = computeCost(distance_ij, node_j);

      if (cost_i + cost_ij < cost_j)
      {
        node_j.pose.theta() = computeHeading(*node_i, node_j);
        node_j.distance = distance_i + distance_ij;
        node_j.cost = cost_i + cost_ij;
        node_j.cost_to_go = computeCostToGo(node_j, not_reached);
        tree[j] = i;

        if (elements.find(node_j.id) == elements.end())
          elements[node_j.id] = open.insert(&working_g_.nodes[node_j.id]);
        else
          open.update(elements[node_j.id]);
      }
    }
    visited.insert(i);

    if (not_reached.find(i) != not_reached.end())
    {
      not_reached.erase(i);
    }
  }

  /// Return all edges for visualization
  for (const auto &pair : tree)
  {
    const Node &node1 = working_g_.nodes.at(pair.second);
    const Node &node2 = working_g_.nodes.at(pair.first);
    ret->edges.push_back({ node1, node2 });
  }

  /// Extract shortest paths to all goals
  for (const Node &goal : goals)
  {
    Node node = goal;

    if (tree.find(node.id) == tree.end())
    {
      auto nn = g_.searchNodes(node.pose.x(), node.pose.y(), working_g_.knn, step_length_);
      if (nn.empty())
        continue;

      std::sort(nn.begin(), nn.end());
      for (const auto &pair : nn)
      {
        if (tree.find(pair.second) != tree.end())
        {
          node.pose = working_g_.nodes[pair.second].pose;
          break;
        }
      }
    }
    ret->targets.push_back(node);

    if (!std::isnan(goal.pose.theta()))
    {
      node.pose.theta() = goal.pose.theta();
    }

    ret->paths.emplace_back();
    auto &path = ret->paths.back();
    while (true)
    {
      if (!path.empty())
      {
        Node &next = path.front();
        next.pose.theta() = computeHeading(node, next);
      }

      ret->paths.back().insert(ret->paths.back().begin(), node);
      auto parent = tree.find(node.id);
      if (parent == tree.end())
        break;
      else
        node = working_g_.nodes.at(parent->second);
    }

    if (path.size() <= 2 || path.front().id != 0)
    {
      ROS_INFO("Path to one goal not found");
      ret->paths.pop_back();
    }
    else
    {
      // Calculate the actual distance and cost in SE2
      for (int i = 1; i < path.size(); ++i)
      {
        double distance = computeDistance(path[i - 1], path[i]);
        path[i].distance = path[i - 1].distance + distance;
        path[i].cost += path[i - 1].cost + computeCost(distance, path[i]);
      }
    }
  }

  ROS_INFO("Build trees (%d paths, %f sec)", (int)ret->paths.size(), (ros::WallTime::now() - start_time).toSec());
  return ret;
}

template <class MOTIONMODEL>
double PathLibraryBase<MOTIONMODEL>::computeDistance(const Node &from, const Node &to, bool use_angle) const
{
  if (!use_angle)
    return (from.pose - from.pose).head(2).norm();
  else
  {
    Pose d = from.pose - to.pose;
    d(2) = ANGLE_WEIGHT * wrapToPi(d(2));
    return d.norm();
  }
}

template <class MOTIONMODEL>
double PathLibraryBase<MOTIONMODEL>::computeCost(double distance, const Node &node) const
{
  // double c = computeClearance(node.x, node.y);
  double c = cost_map_->getClearance(node.index);
  return distance * (2.0 - std::min(max_clearance_, c) / max_clearance_);
}

template <class MOTIONMODEL>
double PathLibraryBase<MOTIONMODEL>::computeCostToGo(const Node &node,
                                                     const std::map<int, Node> &not_reached_goals) const
{
  std::vector<double> distances;
  for (const auto &goal : not_reached_goals)
    distances.push_back(computeDistance(node, goal.second, false));

  return *std::min_element(distances.begin(), distances.end());
}

template <class MOTIONMODEL>
double PathLibraryBase<MOTIONMODEL>::computeHeading(const Node &from, const Node &to) const
{
  return std::atan2(to.pose.y() - from.pose.y(), to.pose.x() - from.pose.x());
}

template <class MOTIONMODEL>
bool PathLibraryBase<MOTIONMODEL>::isSteerable(const Node &from, const Node &to) const
{
  return motion_model_.check(from, to);
}

template <class MOTIONMODEL>
RevisitPathLibraryBase<MOTIONMODEL>::RevisitPathLibraryBase(ros::NodeHandle nh) : Base(nh)
{
  nh.getParam(Base::NS + "revisitation_radius", revisitation_radius_);
  nh.getParam(Base::NS + "revisitation_separation", revisitation_separation_);
  nh.getParam(Base::NS + "revisitation_density", revisitation_density_);
  nh.getParam(Base::NS + "revisitation_max_num", revisitation_max_num_);
}

template <class MOTIONMODEL>
Targets::Ptr RevisitPathLibraryBase<MOTIONMODEL>::findTargets()
{
  auto ret = Base::findTargets();

  ros::WallTime start_time = ros::WallTime::now();

  cv::Mat occ;
  cv::eigen2cv(this->cost_map_->get(CostMap::OCCUPANCY_LAYER), occ);
  cv::Mat obs(occ == CostMap::OCCUPANCY_TRUE);

  cv::Mat locations;
  cv::findNonZero(obs, locations);
  locations.convertTo(locations, CV_32FC1);

  int k = std::max(1, (int)ceil(locations.rows / revisitation_density_));
  cv::Mat labels, centers;
  cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0);
  cv::kmeans(locations, k, labels, criteria, 10, cv::KMEANS_PP_CENTERS, centers);

  std::vector<std::pair<int, int>> cluster_counts;
  for (int i = 0; i < centers.rows; ++i)
    cluster_counts.emplace_back(i, cv::countNonZero(labels == i));

  std::sort(cluster_counts.begin(), cluster_counts.end(),
            [](const std::pair<int, int> &a, const std::pair<int, int> &b) { return a.second > b.second; });

  std::vector<Node> revisitations;
  typedef std::tuple<Index, Pose, double> IndexPoseClearance;
  double angle_inc = this->cost_map_->getResolution() / revisitation_radius_;
  for (int i = 0; i < centers.rows; ++i)
  {
    if (revisitations.size() > revisitation_max_num_)
      break;

    int n = cluster_counts[i].first;
    int i0 = (int)std::round(centers.at<float>(n, 1));
    int j0 = (int)std::round(centers.at<float>(n, 0));
    Position center;
    this->cost_map_->getPosition(Index(i0, j0), center);

    std::vector<IndexPoseClearance> index_pose_clearance;
    for (double b = -M_PI; b < M_PI; b += angle_inc)
    {
      double x = center(0) + revisitation_radius_ * std::cos(b);
      double y = center(1) + revisitation_radius_ * std::sin(b);
      Index index;
      if (!this->cost_map_->getIndex(Position(x, y), index))
        continue;
      if (this->cost_map_->isReachable(index) && this->cost_map_->isCollisionFree(index) &&
          !this->cost_map_->isOccupancyUnknown(index))
      {
        index_pose_clearance.emplace_back(
            std::make_tuple(index, Pose(x, y, b + M_PI), this->cost_map_->getClearance(index)));
      }
    }

    if (index_pose_clearance.empty())
      continue;

    std::sort(index_pose_clearance.begin(), index_pose_clearance.end(),
              [](const IndexPoseClearance &a, const IndexPoseClearance &b) { return std::get<2>(a) > std::get<2>(b); });
    const Pose &pose = std::get<1>(index_pose_clearance.front());

    bool valid = true;
    for (const auto &node : revisitations)
    {
      double sep = (pose - node.pose).norm();
      if (sep < revisitation_separation_)
      {
        valid = false;
        break;
      }
    }

    if (valid)
    {
      revisitations.emplace_back(-1, pose(0), pose(1), pose(2));
      revisitations.back().type = This::TARGET_TYPE;
    }
  }

  ret->targets.insert(ret->targets.end(), revisitations.begin(), revisitations.end());

  ROS_INFO("Find revisitations (%d revisitations cells, %f sec)", (int)revisitations.size(),
           (ros::WallTime::now() - start_time).toSec());
  return ret;
}

template class PathLibraryBase<MotionModel>;
template class RevisitPathLibraryBase<MotionModel>;

}  // namespace base
}  // namespace bruce_exploration