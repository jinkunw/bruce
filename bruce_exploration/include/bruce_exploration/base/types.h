#pragma once
#include <Eigen/Geometry>
#include <Eigen/StdVector>
#include <unordered_map>
#include <unordered_set>

namespace bruce_exploration
{
namespace base
{
/// Special types for cost map for consistency
typedef float DataType;
typedef Eigen::Vector2d Position;
typedef Eigen::Array2i Index;
typedef Eigen::Quaternion<double> Quaternion;
typedef Eigen::Vector2d Vector2;
typedef Eigen::Vector3d Vector3;
typedef Eigen::VectorXd VectorXd;
typedef Eigen::MatrixXd MatrixXd;
typedef Eigen::Matrix2d Matrix2;
typedef Eigen::Matrix3d Matrix3;
struct Pose : public Vector3
{
  using Vector3::Vector3;

  double theta() const
  {
    return z();
  }
  double &theta()
  {
    return z();
  }
};

struct BeliefState
{
  Pose pose;
  Matrix3 cov;
};

}  // namespace base
}  // namespace bruce_exploration