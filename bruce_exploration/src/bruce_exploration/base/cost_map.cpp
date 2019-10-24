#include <ros/ros.h>
#include <Eigen/Core>
#include <grid_map_core/iterators/GridMapIterator.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc.hpp>

#include "bruce_exploration/base/cost_map.h"

namespace bruce_exploration
{
namespace base
{
const std::string CostMap::OCCUPANCY_LAYER = "occupancy";
// Don't compare it with NAN; use std::isnan
const DataType CostMap::OCCUPANCY_UNKNOWN = NAN;
const DataType CostMap::OCCUPANCY_TRUE = 100.0;
const DataType CostMap::OCCUPANCY_FALSE = 0.0;
const std::string CostMap::OBSTACLE_LAYER = "obstacle";
const DataType CostMap::OBSTACLE_FALSE = 0.0;
const DataType CostMap::OBSTACLE_TRUE = 255.0;
const std::string CostMap::CLEARANCE_LAYER = "clearance";
const DataType CostMap::CLEARANCE_MAX = std::numeric_limits<DataType>::max();
const std::string CostMap::REACHABILITY_LAYER = "reachability";
const DataType CostMap::REACHABILITY_TRUE = 255.0;
const DataType CostMap::REACHABILITY_FALSE = 0.0;

CostMap::CostMap() : grid_map::GridMap()
{
}

CostMap::CostMap(double xmin, double xmax, double ymin, double ymax, double resolution, double inflation_radius)
  : grid_map::GridMap()
{
  setGeometry(xmin, xmax, ymin, ymax, resolution, inflation_radius);
}

grid_map::GridMapIterator CostMap::iterate() const
{
  return grid_map::GridMapIterator(*this);
}

void CostMap::setGeometry(double xmin, double xmax, double ymin, double ymax, double resolution,
                          double inflation_radius)
{
  limit_min_ << xmin, ymin;
  limit_max_ << xmax, ymax;
  inflation_radius_ = inflation_radius;
  grid_map::Length length = limit_max_ - limit_min_;
  Position position = length * 0.5 + limit_min_.array();
  GridMap::setGeometry(length, resolution, position);
  add(OCCUPANCY_LAYER, OCCUPANCY_UNKNOWN);
  add(OBSTACLE_LAYER, OBSTACLE_FALSE);
  add(CLEARANCE_LAYER, CLEARANCE_MAX);
  add(REACHABILITY_LAYER, REACHABILITY_FALSE);
  setBasicLayers({ OCCUPANCY_LAYER });
  ROS_DEBUG("Created cost map with size %f x %f m (%i x %i cells) and origin at "
            "(%f m, %f m).",
            getLength().x(), getLength().y(), getSize()(0), getSize()(1), getPosition()(0), getPosition()(1));
}

void CostMap::setOccupancyMap(const nav_msgs::OccupancyGrid &occ_grid)
{
  get(OCCUPANCY_LAYER).fill(NAN);

  int width = occ_grid.info.width;
  int height = occ_grid.info.height;
  float resolution = occ_grid.info.resolution;
  float x0 = occ_grid.info.origin.position.x;
  float y0 = occ_grid.info.origin.position.y;

  const int8_t *data = static_cast<const int8_t *>(&occ_grid.data[0]);
  cv::Mat mat(height, width, CV_8SC1, (void *)data);

  if (resolution > getResolution())
  {
    ROS_WARN_ONCE("Resolution of occupancy grid %f > resolution of grid map %f. Upsampling is performed.", resolution,
                  getResolution());
    // Upsample; Otherwise there are gaps between grid map cells.
    int ratio = (int)std::ceil(resolution / getResolution());
    resolution /= ratio;
    width *= ratio;
    height *= ratio;

    cv::resize(mat, mat, cv::Size(height, width), 0, 0, cv::INTER_NEAREST);
    data = reinterpret_cast<const int8_t *>(mat.ptr());
  }

  for (size_t i = 0; i < height; ++i)
  {
    DataType y = y0 + (i + 0.5) * resolution;
    for (size_t j = 0; j < width; ++j)
    {
      size_t n = i * width + j;
      // if (data[n] == -1)
      //   continue;

      DataType x = x0 + (j + 0.5) * resolution;
      Position position(x, y);
      Index index;
      if (!getIndex(position, index))
        continue;

      /// Use the following for occupancy values = {-1, 0, 100}
      // at(CostMap::OCCUPANCY_LAYER, index) =
      //     data[n] == -1 ? OCCUPANCY_UNKNOWN : (data[n] > 0 ? OCCUPANCY_TRUE : OCCUPANCY_FALSE);

      static const int8_t FREE_THRESH = 45;
      static const int8_t OCCUPIED_THRESH = 55;
      DataType &value = at(CostMap::OCCUPANCY_LAYER, index);
      DataType new_value = data[n] >= OCCUPIED_THRESH ?
                               OCCUPANCY_TRUE :
                               (data[n] >= 0 && data[n] <= FREE_THRESH ? OCCUPANCY_FALSE : OCCUPANCY_UNKNOWN);
      // Use max occupancy when multiple cells are pointint to the same costmap cell
      value = (std::isnan(value) ? new_value : std::max(value, new_value));
    }
  }

  updateCosts();
}

void CostMap::setOrigin(double x, double y, double theta)
{
  origin_ << x, y, theta;
  Index index0;
  if (!getIndex(Position(x, y), index0))
  {
    ROS_ERROR("Origin (%f, %f, %f) is out of map", x, y, theta);
    return;
  }

  cv::Mat obs;
  cv::eigen2cv(get(OBSTACLE_LAYER), obs);
  cv::Mat rch(obs.rows, obs.cols, CV_8UC1, cv::Scalar(0));
#ifndef ALLOW_UNKNOWN
  cv::Mat occ;
  cv::eigen2cv(get(OCCUPANCY_LAYER), occ);
  cv::bitwise_or(rch, occ != occ, rch);
#endif
  cv::bitwise_or(rch, obs == OBSTACLE_TRUE, rch);
  cv::floodFill(rch, cv::Point(index0(1), index0(0)), cv::Scalar(100), 0, 0, 10, 8);

  Matrix rch_mat;
  cv::cv2eigen(rch == 100, rch_mat);
  get(REACHABILITY_LAYER) = rch_mat;
}

void CostMap::updateCosts()
{
  cv::Mat occ;
  cv::eigen2cv(get(OCCUPANCY_LAYER), occ);
  cv::Mat obs(occ == OCCUPANCY_TRUE);

  int kernel_size = static_cast<int>(std::ceil(inflation_radius_ / getResolution()));
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * kernel_size + 1, 2 * kernel_size + 1),
                                             cv::Point(kernel_size, kernel_size));
  cv::dilate(obs, obs, kernel);

  Matrix obs_mat;
  cv::cv2eigen(obs, obs_mat);
  get(OBSTACLE_LAYER) = obs_mat;

  cv::Mat clr;
  cv::bitwise_not(obs, clr);
  cv::distanceTransform(clr, clr, cv::DIST_L2, 3);

  Matrix clr_mat;
  cv::cv2eigen(clr, clr_mat);
  get(CLEARANCE_LAYER) = clr_mat * getResolution();
}

std::vector<Index> CostMap::getNeighborIndex4(const grid_map::Index &index) const
{
  static std::vector<std::pair<int, int>> inc4 = { { -1, 0 }, { 1, 0 }, { 0, -1 }, { 0, 1 } };

  std::vector<Index> ret;
  for (const auto &it : inc4)
  {
    Index nh(index(0) + it.first, index(1) + it.second);
    if (nh(0) >= 0 && nh(0) < getSize()(0) && nh(1) >= 0 && nh(1) < getSize()(1))
      ret.push_back(nh);
  }
  return ret;
}

std::vector<Index> CostMap::getNeighborIndex8(const grid_map::Index &index) const
{
  static std::vector<std::pair<int, int>> inc8 = { { -1, 1 }, { -1, 0 }, { -1, -1 }, { 0, 1 },
                                                   { 0, -1 }, { 1, 1 },  { 1, 0 },   { 1, -1 } };

  std::vector<Index> ret;
  for (const auto &it : inc8)
  {
    Index nh(index(0) + it.first, index(1) + it.second);
    if (nh(0) >= 0 && nh(0) < getSize()(0) && nh(1) >= 0 && nh(1) < getSize()(1))
      ret.push_back(nh);
  }
  return ret;
}

bool CostMap::isOccupancyFree(double x, double y) const
{
  Index index;
  if (getIndex(Position(x, y), index))
  {
    return isOccupancyFree(index);
  }
  return false;
}

bool CostMap::isOccupancyUnknown(double x, double y) const
{
  Index index;
  if (getIndex(Position(x, y), index))
  {
    return isOccupancyUnknown(index);
  }
  return true;
}

bool CostMap::isCollisionFree(double x, double y) const
{
  Index index;
  if (getIndex(Position(x, y), index))
  {
    return isCollisionFree(index);
  }
  return true;
}

bool CostMap::isCollisionFree(double x1, double y1, double x2, double y2) const
{
  for (grid_map::LineIterator iterator(*this, Position(x1, y1), Position(x2, y2)); !iterator.isPastEnd(); ++iterator)
  {
    if (!isCollisionFree(*iterator))
      return false;
  }
  return true;
}

bool CostMap::isReachable(double x, double y) const
{
  Index index;
  if (getIndex(Position(x, y), index))
  {
    return isReachable(index);
  }
  return false;
}

CostMap::DataType CostMap::getClearance(double x, double y) const
{
  Index index;
  if (getIndex(Position(x, y), index))
  {
    return getClearance(index);
  }
  return CLEARANCE_MAX;
}

/// Without validity check
bool CostMap::isOccupancyFree(const Index &index) const
{
  return at(OCCUPANCY_LAYER, index) == OCCUPANCY_FALSE;
}

bool CostMap::isOccupancyUnknown(const Index &index) const
{
  const DataType &v = at(OCCUPANCY_LAYER, index);
  return std::isnan(v);  // || v == OCCUPANCY_UNKNOWN;
}

bool CostMap::isCollisionFree(const Index &start, const Index &end) const
{
  for (grid_map::LineIterator iterator(*this, start, end); !iterator.isPastEnd(); ++iterator)
  {
    if (!isCollisionFree(*iterator))
      return false;
  }
  return true;
}

bool CostMap::isCollisionFree(const grid_map::Index &index) const
{
#ifdef ALLOW_UNKNOWN
  return at(OBSTACLE_LAYER, index) == OBSTACLE_FALSE;
#else
  return !isOccupancyUnknown(index) && at(OBSTACLE_LAYER, index) == OBSTACLE_FALSE;
#endif
}

bool CostMap::isReachable(const Index &index) const
{
  return at(REACHABILITY_LAYER, index) == REACHABILITY_TRUE;
}

double CostMap::getClearance(const grid_map::Index &index) const
{
  return at(CLEARANCE_LAYER, index);
}

void CostMap::clearCells(DataType x, DataType y, double clearing_radius)
{
  for (grid_map::CircleIterator iter(*this, grid_map::Position(x, y), clearing_radius); !iter.isPastEnd(); ++iter)
  {
    at(OCCUPANCY_LAYER, *iter) = OCCUPANCY_FALSE;
    at(OBSTACLE_LAYER, *iter) = OBSTACLE_FALSE;
  }
}

void CostMap::save(std::ostream &os) const
{
  const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, " ", "\n");
  // xmin, xmax, ymin, ymax, # rows, # cols
  os << limit_min_(0) << " " << limit_max_(0) << " " << limit_min_(1) << " " << limit_max_(1) << " " << getSize()(0)
     << " " << getSize()(1) << std::endl;
  for (const auto &layer : getLayers())
  {
    const auto &mat = get(layer);
    os << layer << std::endl;
    os << mat.format(CSVFormat) << std::endl;
  }
}

}  // namespace base
}  // namespace bruce_exploration
