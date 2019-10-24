#pragma once
#include <nav_msgs/OccupancyGrid.h>
#include <grid_map_core/grid_map_core.hpp>

#include "bruce_exploration/base/types.h"

// #define ALLOW_UNKNOWN

namespace bruce_exploration
{
namespace base
{
class CostMap : public grid_map::GridMap
{
public:
  typedef std::shared_ptr<CostMap> Ptr;

  CostMap();

  CostMap(double xmin, double xmax, double ymin, double ymax, double resolution, double inflation_radius);

  void setGeometry(double xmin, double xmax, double ymin, double ymax, double resolution, double inflation_radius);

  void setOccupancyMap(const nav_msgs::OccupancyGrid &occ_grid);

  void setOrigin(double x, double y, double theta);

  grid_map::GridMapIterator iterate() const;

  inline Position getLimitMin() const
  {
    return limit_min_;
  }

  inline Position getLimitMax() const
  {
    return limit_max_;
  }

  inline Pose getOrigin() const
  {
    return origin_;
  }

  inline double getInflationRadius() const
  {
    return inflation_radius_;
  }

  const Matrix &getOccupancyLayer() const
  {
    return get(OCCUPANCY_LAYER);
  }

  const Matrix &getObstacleLayer() const
  {
    return get(OBSTACLE_LAYER);
  }

  const Matrix &getClearanceLayer() const
  {
    return get(CLEARANCE_LAYER);
  }

  const Matrix &getReachabilityLayer() const
  {
    return get(REACHABILITY_LAYER);
  }

  void clearCells(DataType x, DataType y, double clearing_radius);

  std::vector<Index> getNeighborIndex4(const Index &index) const;

  std::vector<Index> getNeighborIndex8(const Index &index) const;

  bool isOccupancyFree(double x, double y) const;

  bool isOccupancyUnknown(double x, double y) const;

  bool isCollisionFree(double x, double y) const;

  bool isCollisionFree(double x1, double y1, double x2, double y2) const;

  bool isReachable(double x, double y) const;

  CostMap::DataType getClearance(double x, double y) const;

  /// Without validity check
  bool isOccupancyFree(const Index &index) const;

  bool isOccupancyUnknown(const Index &index) const;

  bool isCollisionFree(const Index &start, const Index &end) const;

  bool isCollisionFree(const Index &index) const;

  bool isReachable(const Index &index) const;

  double getClearance(const Index &index) const;

  void save(std::ostream &os) const;

public:
  const static std::string OCCUPANCY_LAYER;
  const static DataType OCCUPANCY_UNKNOWN;
  const static DataType OCCUPANCY_TRUE;
  const static DataType OCCUPANCY_FALSE;

  const static std::string OBSTACLE_LAYER;
  const static DataType OBSTACLE_FALSE;
  const static DataType OBSTACLE_TRUE;

  const static std::string CLEARANCE_LAYER;
  const static DataType CLEARANCE_MAX;

  const static std::string REACHABILITY_LAYER;
  const static DataType REACHABILITY_TRUE;
  const static DataType REACHABILITY_FALSE;

private:
  void updateCosts();

  Position limit_min_;
  Position limit_max_;

  double inflation_radius_;
  Pose origin_;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
}
}  // namespace bruce_exploration