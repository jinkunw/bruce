#include <grid_map_core/GridMapMath.hpp>
#include "bruce_exploration/base/math.h"
#include "bruce_exploration/base/virtual_map.h"

#define SPLITCI

namespace bruce_exploration
{
namespace base
{
const std::string SonarModel::NS = "sonar_model/";

CircularSectorIterator::CircularSectorIterator(const grid_map::GridMap &gridMap, const Pose &center,
                                               const double radius, const double half_angle)
  : center_(center), radius_(radius), half_angle_(half_angle)
{
  radiusSquare_ = pow(radius_, 2);
  mapLength_ = gridMap.getLength();
  mapPosition_ = gridMap.getPosition();
  resolution_ = gridMap.getResolution();
  bufferSize_ = gridMap.getSize();
  bufferStartIndex_ = gridMap.getStartIndex();
  Index submapStartIndex;
  Index submapBufferSize;
  findSubmapParameters(center.head(2), radius, submapStartIndex, submapBufferSize);
  internalIterator_ = std::shared_ptr<grid_map::SubmapIterator>(
      new grid_map::SubmapIterator(gridMap, submapStartIndex, submapBufferSize));
  if (!isInside())
    ++(*this);
}

CircularSectorIterator &CircularSectorIterator::operator=(const CircularSectorIterator &other)
{
  center_ = other.center_;
  radius_ = other.radius_;
  half_angle_ = other.half_angle_;
  radiusSquare_ = other.radiusSquare_;
  internalIterator_ = other.internalIterator_;
  mapLength_ = other.mapLength_;
  mapPosition_ = other.mapPosition_;
  resolution_ = other.resolution_;
  bufferSize_ = other.bufferSize_;
  bufferStartIndex_ = other.bufferStartIndex_;
  return *this;
}

bool CircularSectorIterator::operator!=(const CircularSectorIterator &other) const
{
  return (internalIterator_ != other.internalIterator_);
}

const Index &CircularSectorIterator::operator*() const
{
  return *(*internalIterator_);
}

CircularSectorIterator &CircularSectorIterator::operator++()
{
  ++(*internalIterator_);
  if (internalIterator_->isPastEnd())
    return *this;

  for (; !internalIterator_->isPastEnd(); ++(*internalIterator_))
  {
    if (isInside())
      break;
  }

  return *this;
}

bool CircularSectorIterator::isPastEnd() const
{
  return internalIterator_->isPastEnd();
}

bool CircularSectorIterator::isInside() const
{
  Position position;
  grid_map::getPositionFromIndex(position, *(*internalIterator_), mapLength_, mapPosition_, resolution_, bufferSize_,
                                 bufferStartIndex_);
  double squareNorm = (position - center_.head(2)).array().square().sum();
  double angle = std::atan2(position(1) - center_(1), position(0) - center_(0));
  double half_angle = std::abs(wrapToPi(angle - center_(2)));
  return (squareNorm <= radiusSquare_) && (half_angle < half_angle_);
}

void CircularSectorIterator::findSubmapParameters(const Position &center, const double radius, Index &startIndex,
                                                  grid_map::Size &bufferSize) const
{
  Position topLeft = center.array() + radius;
  Position bottomRight = center.array() - radius;
  grid_map::boundPositionToRange(topLeft, mapLength_, mapPosition_);
  grid_map::boundPositionToRange(bottomRight, mapLength_, mapPosition_);
  grid_map::getIndexFromPosition(startIndex, topLeft, mapLength_, mapPosition_, resolution_, bufferSize_,
                                 bufferStartIndex_);
  Index endIndex;
  grid_map::getIndexFromPosition(endIndex, bottomRight, mapLength_, mapPosition_, resolution_, bufferSize_,
                                 bufferStartIndex_);
  bufferSize = grid_map::getSubmapSizeFromCornerIndeces(startIndex, endIndex, bufferSize_, bufferStartIndex_);
}

BeamIterator::BeamIterator(const CostMap &cost_map, const Pose &center, const double radius, const double half_angle)
{
  // Calculate beam angle such that all cells within sensor range are covered.
  // Still miss some cells!
  // double sep = cost_map.getResolution() / radius;
  static const double sep = M_PI / 180.0 * 130.0 / 512.0;
  std::set<size_t> ends;

  grid_map::Position start(center.x(), center.y());
  Index start_index;
  if (!cost_map.getIndex(start, start_index))
  {
    internel_iter_ = indices_.end();
    return;
  }

  size_t ncols = cost_map.getSize()(1);
  for (double angle = -half_angle; angle < half_angle; angle += sep)
  {
    grid_map::Position end;
    end(0) = start.x() + radius * std::cos(angle + center.theta());
    end(1) = start.y() + radius * std::sin(angle + center.theta());

    grid_map::Index end_index;
    if (!getIndexLimitedToMapRange(cost_map, end, start, end_index))
      continue;

    size_t i = end_index(0) * ncols + end_index(1);
    if (ends.find(i) != ends.end())
      continue;
    ends.insert(i);

    bool pass_first_hit = false;
    for (grid_map::LineIterator iter(cost_map, start_index, end_index); !iter.isPastEnd(); ++iter)
    {
      bool free = cost_map.isOccupancyFree(*iter);
      bool unknown = cost_map.isOccupancyUnknown(*iter);
      if (pass_first_hit && (free || unknown))
        continue;

      // Unique index in the map
      indices_[(*iter)(0) * ncols + (*iter)(1)] = *iter;

      if (!free && !unknown)
        pass_first_hit = true;
    }
  }
  internel_iter_ = indices_.begin();
}

BeamIterator &BeamIterator::operator=(const BeamIterator &other)
{
  indices_ = other.indices_;
  internel_iter_ = other.internel_iter_;
  return *this;
}

bool BeamIterator::operator!=(const BeamIterator &other) const
{
  return internel_iter_ != other.internel_iter_;
}

const Index &BeamIterator::operator*() const
{
  return internel_iter_->second;
}

BeamIterator &BeamIterator::operator++()
{
  ++internel_iter_;
  return *this;
}

bool BeamIterator::isPastEnd() const
{
  return internel_iter_ == indices_.end();
}

bool BeamIterator::getIndexLimitedToMapRange(const grid_map::GridMap &gridMap, const Position &start,
                                             const Position &end, Index &index)
{
  Position newStart = start;
  grid_map::Vector direction = (end - start).normalized();
  while (!gridMap.getIndex(newStart, index))
  {
    newStart += (gridMap.getResolution() - std::numeric_limits<double>::epsilon()) * direction;
    if ((end - newStart).norm() < gridMap.getResolution() - std::numeric_limits<double>::epsilon())
      return false;
  }
  return true;
}

SonarModel::SonarModel(ros::NodeHandle nh)
{
  nh.getParam(NS + "range", range);
  nh.getParam(NS + "aperture", aperture);
  nh.getParam(NS + "range_sigma", range_sigma);
  nh.getParam(NS + "bearing_sigma", bearing_sigma);
}

Vector2 SonarModel::measure(const BeliefState &state, const Position &point) const
{
  // assert(point in field of view)
  double bearing = wrapToPi(std::atan2(point.y() - state.pose.y(), point.x() - state.pose.x()) - state.pose.theta());
  double range = std::sqrt(std::pow(point.y() - state.pose.y(), 2) + std::pow(point.x() - state.pose.x(), 2));
  return (Vector2() << bearing, range).finished();
}

Matrix2 SonarModel::propagate(const BeliefState &state, const Vector2 &z) const
{
  const auto &pair = propagateSplit(state, z);
  return pair.first + pair.second;
}

std::pair<Matrix2, Matrix2> SonarModel::propagateSplit(const BeliefState &state, const Vector2 &z) const
{
  double s = std::sin(state.pose.theta() + z(0));
  double c = std::cos(state.pose.theta() + z(0));
  Eigen::Matrix<double, 2, 3> H1 = MatrixXd::Zero(2, 3);
  H1 << 1, 0, -z(1) * s, 0, 1, z(1) * c;
  Matrix2 H2 = MatrixXd::Zero(2, 2);
  H2 << -z(1) * s, c, z(1) * c, s;
  Matrix2 Q = MatrixXd::Zero(2, 2);
  Q << bearing_sigma * bearing_sigma, 0, 0, range_sigma * range_sigma;
  return { H1 * state.cov * H1.transpose(), H2 * Q * H2.transpose() };
}

template <class SENSORMODEL>
const std::string VirtualMapBase<SENSORMODEL>::NS = "virtual_map/";
template <class SENSORMODEL>
const std::string VirtualMapBase<SENSORMODEL>::DET_LAYER = "det";

template <class SENSORMODEL>
VirtualMapBase<SENSORMODEL>::VirtualMapBase(ros::NodeHandle nh) : CostMap(), sensor_model_(nh), cov1_(NULL), cov2_(NULL)
{
  double xmin, xmax, ymin, ymax, resolution;
  nh.getParam(NS + "xmin", xmin);
  nh.getParam(NS + "ymin", ymin);
  nh.getParam(NS + "xmax", xmax);
  nh.getParam(NS + "ymax", ymax);
  nh.getParam(NS + "resolution", resolution);
  setGeometry(xmin, xmax, ymin, ymax, resolution, 0.0);

  nh.getParam(NS + "sigma0", sigma0_);

  cov1_ = new Matrix2 *[getSize()(0)];
  for (size_t i = 0; i < getSize()(0); ++i)
    cov1_[i] = new Matrix2[getSize()(1)];
#ifdef SPLITCI
  cov2_ = new Matrix2 *[getSize()(0)];
  for (size_t i = 0; i < getSize()(0); ++i)
    cov2_[i] = new Matrix2[getSize()(1)];
#endif

  reset();
}

template <class SENSORMODEL>
VirtualMapBase<SENSORMODEL>::~VirtualMapBase()
{
  if (cov1_)
  {
    for (size_t i = 0; i < getSize()(0); ++i)
      delete[] cov1_[i];
    delete[] cov1_;
  }
  if (cov2_)
  {
    for (size_t i = 0; i < getSize()(0); ++i)
      delete[] cov2_[i];
    delete[] cov2_;
  }
}

template <class SENSORMODEL>
typename SENSORMODEL::Iterator VirtualMapBase<SENSORMODEL>::iterateVisible(const Pose &origin) const
{
  return sensor_model_.iterate(*this, origin);
}

template <class SENSORMODEL>
double VirtualMapBase<SENSORMODEL>::getLogDOpt(const Index &index) const
{
  return logDOpt(getCovariance(index));
}

template <class SENSORMODEL>
double VirtualMapBase<SENSORMODEL>::getLogDOpt() const
{
  double lp = 0.0;
  for (auto iter = iterate(); !iter.isPastEnd(); ++iter)
  {
    if (!isOccupancyFree(*iter))
      lp += getLogDOpt(*iter);
  }
  return lp;
}

template <class SENSORMODEL>
void VirtualMapBase<SENSORMODEL>::setBeliefStates(const std::vector<BeliefState> &beliefs)
{
  reset();

  for (const auto &bs : beliefs)
  {
    for (auto iter = iterateVisible(bs.pose); !iter.isPastEnd(); ++iter)
    {
      if (isOccupancyFree(*iter))
        continue;

      Position position;
      getPosition(*iter, position);

      const Vector2 &z = sensor_model_.measure(bs, position);
#ifdef SPLITCI
      Matrix2 cov1, cov2;
      std::tie(cov1, cov2) = sensor_model_.propagateSplit(bs, z);
      mergeCovariance(*iter, cov1, cov2);
#else
      const Matrix2 cov = sensor_model_.propagate(bs, z);
      mergeCovariance(*iter, cov);
#endif
    }
  }

  for (auto iter = iterate(); !iter.isPastEnd(); ++iter)
  {
    const int &i = (*iter)(0);
    const int &j = (*iter)(1);
    if (!cov2_ || std::isnan(cov2_[i][j](0, 0)))
      at(DET_LAYER, (*iter)) = cov1_[i][j].determinant();
    else
      at(DET_LAYER, (*iter)) = (cov1_[i][j] + cov2_[i][j]).determinant();
  }
}

template <class SENSORMODEL>
Matrix2 VirtualMapBase<SENSORMODEL>::getCovariance(const Index &index) const
{
  const int &i = index(0);
  const int &j = index(1);
  if (!cov2_ || std::isnan(cov2_[i][j](0, 0)))
    return cov1_[i][j];
  else
    return cov1_[i][j] + cov2_[i][j];
}

template <class SENSORMODEL>
CostMap::Matrix VirtualMapBase<SENSORMODEL>::getDeterminant() const
{
  return get(DET_LAYER);
}

template <class SENSORMODEL>
void VirtualMapBase<SENSORMODEL>::reset()
{
  for (size_t i = 0; i < getSize()(0); ++i)
  {
    for (size_t j = 0; j < getSize()(1); ++j)
    {
      cov1_[i][j] = Matrix2::Identity() * (sigma0_ * sigma0_);
#ifdef SPLITCI
      cov2_[i][j] = Matrix2::Zero() * NAN;
#endif
    }
  }
  add(DET_LAYER, std::pow(sigma0_, 4));
}

template <class SENSORMODEL>
void VirtualMapBase<SENSORMODEL>::mergeCovariance(const Index &index, const Matrix2 &new_cov)
{
  const int &i = index(0);
  const int &j = index(1);
  Matrix2 ci = covarianceIntersection(cov1_[i][j], new_cov);
  cov1_[i][j] = ci;
}

template <class SENSORMODEL>
void VirtualMapBase<SENSORMODEL>::mergeCovariance(const Index &index, const Matrix2 &new_cov1, const Matrix2 &new_cov2)
{
  const int &i = index(0);
  const int &j = index(1);
  if (std::isnan(cov2_[i][j](0, 0)))
  {
    cov1_[i][j] = new_cov1;
    cov2_[i][j] = new_cov2;
  }
  else
  {
    Matrix2 ci1, ci2;
    std::tie(ci1, ci2) = splitCovarianceIntersection(cov1_[i][j], cov2_[i][j], new_cov1, new_cov2);
    cov1_[i][j] = ci1;
    cov2_[i][j] = ci2;
  }
}

template <class SENSORMODEL>
void VirtualMapBase<SENSORMODEL>::save(std::ostream &os) const
{
  CostMap::save(os);

  Matrix cov11(getSize()(0), getSize()(1));
  Matrix cov12(getSize()(0), getSize()(1));
  Matrix cov22(getSize()(0), getSize()(1));
  for (size_t i = 0; i < getSize()(0); ++i)
  {
    for (size_t j = 0; j < getSize()(1); ++j)
    {
      MatrixXd cov = getCovariance(Index(i, j));
      cov11(i, j) = cov(0, 0);
      cov12(i, j) = cov(0, 1);
      cov22(i, j) = cov(1, 1);
    }
  }

  const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, " ", "\n");
  os << "cov11" << std::endl;
  os << cov11.format(CSVFormat) << std::endl;
  os << "cov12" << std::endl;
  os << cov12.format(CSVFormat) << std::endl;
  os << "cov22" << std::endl;
  os << cov22.format(CSVFormat) << std::endl;
}

template class VirtualMapBase<SonarModel>;

}  // namespace base
}  // namespace bruce_exploration
