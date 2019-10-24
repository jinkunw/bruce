#pragma once
#include "bruce_exploration/base/types.h"

namespace bruce_exploration
{
namespace base
{
// Weight of SO2 distance in SE2
const double ANGLE_WEIGHT = 1.0;

double wrapToPi(double angle);

double yawFromQuaternion(double w, double x, double y, double z);
double yawFromQuaternion(const Quaternion &q);

Quaternion quaternionFromYaw(double theta);

template <typename MatrixType>
MatrixType covarianceIntersection(const MatrixType &A, const MatrixType &B)
{
  double a = 1.0 / A.determinant();
  double b = 1.0 / B.determinant();
  Matrix2 IB = B.inverse();
  double c = a * (A * IB).trace();
  double d = a + b - c;
  double w = 0.5 * (2 * b - c) / d;

  if (w < 0.0 || w > 1.0)
  {
    return a > b ? A : B;
  }

  return (w * A.inverse() + (1.0 - w) * IB).inverse();
}

// Matrix2 covarianceIntersection(const Matrix2 &A, const Matrix2 &B);

double fmin(std::function<double(double)> f, double min, double max, double tolerance);

template <typename MatrixType>
std::pair<MatrixType, MatrixType> splitCovarianceIntersection(const MatrixType &A1, const MatrixType &A2,
                                                              const MatrixType &B1, const MatrixType &B2)
{
  auto f = [A1, A2, B1, B2](double w) {
    if (w < 1e-3)
      return (B1 + B2).determinant();
    else if (w > 1.0 - 1e-3)
      return (A1 + A2).determinant();
    else
      return 1.0 / ((1.0 / w * A1 + A2).inverse() + (1.0 / (1.0 - w) * B1 + B2).inverse()).determinant();
  };

  double w = fmin(f, 0.0, 1.0, 1e-3);
  if (w < 1e-3)
    return { B1, B2 };
  if (w > 1.0 - 1e-3)
    return { A1, A2 };

  MatrixType A = 1.0 / w * A1 + A2;
  MatrixType B = 1.0 / (1.0 - w) * B1 + B2;
  MatrixType IA = A.inverse();
  MatrixType IB = B.inverse();
  MatrixType C = (IA + IB).inverse();
  MatrixType C2 = C * (IA * A2 * IA + IB * B2 * IB) * C;
  MatrixType C1 = C - C2;
  return { C1, C2 };
}

// std::pair<Matrix2, Matrix2> splitCovarianceIntersection(const Matrix2 &A1, const Matrix2 &A2, const Matrix2 &B1,
//                                                         const Matrix2 &B2);

template <typename MatrixType>
inline typename MatrixType::Scalar entropy(const MatrixType &cov)
{
  return 0.5 * (std::log(cov.determinant()) + cov.rows() * std::log(2 * M_PI) + cov.rows());
}

template <typename MatrixType>
inline typename MatrixType::Scalar logDOpt(const MatrixType &cov)
{
  return std::log(cov.determinant());
}

}  // namespace base
}  // namespace bruce_exploration