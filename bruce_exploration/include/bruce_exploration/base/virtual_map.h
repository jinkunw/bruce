#pragma once
#include <ros/ros.h>
#include <grid_map_core/GridMap.hpp>
#include <grid_map_core/iterators/SubmapIterator.hpp>
#include "bruce_exploration/base/cost_map.h"

namespace bruce_exploration
{
namespace base
{
/*
 * Iterate a circular sector in map.
 * Assume all cells inside the sector are visible.
 */
class CircularSectorIterator
{
public:
  CircularSectorIterator(const grid_map::GridMap &gridMap, const Pose &center, const double radius,
                         const double half_angle);

  CircularSectorIterator &operator=(const CircularSectorIterator &other);

  bool operator!=(const CircularSectorIterator &other) const;
  const Index &operator*() const;

  CircularSectorIterator &operator++();

  bool isPastEnd() const;

private:
  bool isInside() const;

  void findSubmapParameters(const Position &center, const double radius, Index &startIndex,
                            grid_map::Size &bufferSize) const;

  Pose center_;
  double radius_;
  double half_angle_;
  double radiusSquare_;
  std::shared_ptr<grid_map::SubmapIterator> internalIterator_;
  grid_map::Length mapLength_;
  Position mapPosition_;
  double resolution_;
  grid_map::Size bufferSize_;
  Index bufferStartIndex_;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/*
 * Iterate a circular sector in map beam by beam.
 * Assume non-occupied cells after the first hit inside the sector are invisible.
 */
class BeamIterator
{
public:
  BeamIterator(const CostMap &cost_map, const Pose &center, const double radius, const double half_angle);

  BeamIterator &operator=(const BeamIterator &other);

  bool operator!=(const BeamIterator &other) const;

  const Index &operator*() const;

  BeamIterator &operator++();

  bool isPastEnd() const;

private:
  std::map<size_t, Index> indices_;
  std::map<size_t, Index>::iterator internel_iter_;

  bool getIndexLimitedToMapRange(const grid_map::GridMap &gridMap, const grid_map::Position &start,
                                 const grid_map::Position &end, grid_map::Index &index);

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct SonarModel
{
  static const std::string NS;
  // typedef PizzaIterator Iterator;
  typedef BeamIterator Iterator;

  /// Sensor FOV
  /// [-aperture / 2, aperture / 2]
  double aperture;
  /// [0, sensor_range]
  double range;

  double bearing_sigma;
  double range_sigma;

  SonarModel(ros::NodeHandle nh);

  Iterator iterate(const CostMap &map, const Pose &origin) const
  {
    return Iterator(map, origin, range, aperture / 2);
  }

  Vector2 measure(const BeliefState &state, const Position &point) const;

  Matrix2 propagate(const BeliefState &state, const Vector2 &z) const;

  std::pair<Matrix2, Matrix2> propagateSplit(const BeliefState &state, const Vector2 &z) const;
};

template <class SENSORMODEL>
class VirtualMapBase : public CostMap
{
public:
  static const std::string NS;

  VirtualMapBase(ros::NodeHandle nh);

  ~VirtualMapBase();

  typename SENSORMODEL::Iterator iterateVisible(const Pose &origin) const;

  double getLogDOpt(const Index &index) const;

  double getLogDOpt() const;

  virtual void setBeliefStates(const std::vector<BeliefState> &beliefs);

  Matrix2 getCovariance(const Index &index) const;

  Matrix getDeterminant() const;

  void save(std::ostream &os) const;

protected:
  virtual void reset();

  void mergeCovariance(const Index &index, const Matrix2 &new_cov);

  void mergeCovariance(const Index &index, const Matrix2 &new_cov1, const Matrix2 &new_cov2);

  SENSORMODEL sensor_model_;

  double sigma0_;

  const static std::string DET_LAYER;

  Matrix2 **cov1_;
  Matrix2 **cov2_;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

typedef VirtualMapBase<SonarModel> VirtualMap;

}  // namespace base
}  // namespace bruce_exploration