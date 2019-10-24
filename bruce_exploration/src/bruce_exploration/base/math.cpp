#include <cmath>
#include "bruce_exploration/base/math.h"

namespace bruce_exploration
{
namespace base
{
double wrapToPi(double angle)
{
  double wrapped = std::fmod(angle + M_PI, M_PI * 2);
  if (wrapped < 0)
    wrapped += M_PI * 2;
  return wrapped - M_PI;
}

double yawFromQuaternion(double w, double x, double y, double z)
{
  Quaternion q(w, x, y, z);
  const auto R = q.toRotationMatrix();
  return std::atan2(R(1, 0), R(0, 0));
}

Quaternion quaternionFromYaw(double theta)
{
  return Quaternion(Eigen::AngleAxis<Quaternion::Scalar>(theta, Eigen::Matrix<Quaternion::Scalar, 3, 1>::UnitZ()));
}

// Matrix2 covarianceIntersection(const Matrix2 &A, const Matrix2 &B)
// {
//   double a = 1.0 / A.determinant();
//   double b = 1.0 / B.determinant();
//   Matrix2 IB = B.inverse();
//   double c = a * (A * IB).trace();
//   double d = a + b - c;
//   double w = 0.5 * (2 * b - c) / d;

//   if (w < 0.0 || w > 1.0)
//   {
//     return a > b ? A : B;
//   }

//   return (w * A.inverse() + (1.0 - w) * IB).inverse();
// }

// std::pair<Matrix2, Matrix2> splitCovarianceIntersection(const Matrix2 &A1, const Matrix2 &A2, const Matrix2 &B1,
//                                                         const Matrix2 &B2)
// {
//   auto f = [A1, A2, B1, B2](double w) {
//     if (w < 1e-3)
//       return (B1 + B2).determinant();
//     else if (w > 1.0 - 1e-3)
//       return (A1 + A2).determinant();
//     else
//       return 1.0 / ((1.0 / w * A1 + A2).inverse() + (1.0 / (1.0 - w) * B1 + B2).inverse()).determinant();
//   };

//   double w = fmin(f, 0.0, 1.0, 1e-2);
//   if (w < 1e-3)
//     return { B1, B2 };
//   if (w > 1.0 - 1e-3)
//     return { A1, A2 };

//   Matrix2 A = 1.0 / w * A1 + A2;
//   Matrix2 B = 1.0 / (1.0 - w) * B1 + B2;
//   Matrix2 IA = A.inverse();
//   Matrix2 IB = B.inverse();
//   Matrix2 C = (IA + IB).inverse();
//   Matrix2 C2 = C * (IA * A2 * IA + IB * B2 * IB) * C;
//   Matrix2 C1 = C - C2;
//   return { C1, C2 };
// }

//  Brent's method of minimizing scalar function with bounds
//  (C) Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
double fmin(std::function<double(double)> f, double min, double max, double tolerance)
{
  double x;               // minima so far
  double w;               // second best point
  double v;               // previous value of w
  double u;               // most recent evaluation point
  double delta;           // The distance moved in the last step
  double delta2;          // The distance moved in the step before last
  double fu, fv, fw, fx;  // function evaluations at u, v, w, x
  double mid;             // midpoint of min and max
  double fract1, fract2;  // minimal relative movement in x

  static const double golden = 0.3819660f;  // golden ratio, don't need too much precision here!

  x = w = v = max;
  fw = fv = fx = f(x);
  delta2 = delta = 0;

  uintmax_t count = std::numeric_limits<uintmax_t>::max();

  do
  {
    // get midpoint
    mid = (min + max) / 2;
    // work out if we're done already:
    fract1 = tolerance * std::fabs(x) + tolerance / 4;
    fract2 = 2 * fract1;
    if (std::fabs(x - mid) <= (fract2 - (max - min) / 2))
      break;

    if (std::fabs(delta2) > fract1)
    {
      // try and construct a parabolic fit:
      double r = (x - w) * (fx - fv);
      double q = (x - v) * (fx - fw);
      double p = (x - v) * q - (x - w) * r;
      q = 2 * (q - r);
      if (q > 0)
        p = -p;
      q = fabs(q);
      double td = delta2;
      delta2 = delta;
      // determine whether a parabolic step is acceptible or not:
      if ((std::fabs(p) >= std::fabs(q * td / 2)) || (p <= q * (min - x)) || (p >= q * (max - x)))
      {
        // nope, try golden section instead
        delta2 = (x >= mid) ? min - x : max - x;
        delta = golden * delta2;
      }
      else
      {
        // whew, parabolic fit:
        delta = p / q;
        u = x + delta;
        if (((u - min) < fract2) || ((max - u) < fract2))
          delta = (mid - x) < 0 ? -std::fabs(fract1) : std::fabs(fract1);
      }
    }
    else
    {
      // golden section:
      delta2 = (x >= mid) ? min - x : max - x;
      delta = golden * delta2;
    }
    // update current position:
    u = (fabs(delta) >= fract1) ? x + delta : (delta > 0 ? x + fabs(fract1) : x - fabs(fract1));
    fu = f(u);
    if (fu <= fx)
    {
      // good new point is an improvement!
      // update brackets:
      if (u >= x)
        min = x;
      else
        max = x;
      // update control points:
      v = w;
      w = x;
      x = u;
      fv = fw;
      fw = fx;
      fx = fu;
    }
    else
    {
      // Oh dear, point u is worse than what we have already,
      // even so it *must* be better than one of our endpoints:
      if (u < x)
        min = u;
      else
        max = u;
      if ((fu <= fw) || (w == x))
      {
        // however it is at least second best:
        v = w;
        w = u;
        fv = fw;
        fw = fu;
      }
      else if ((fu <= fv) || (v == x) || (v == w))
      {
        // third best:
        v = u;
        fv = fu;
      }
    }

  } while (--count);

  return x;
}

// double logProbability(const MatrixXd &cov)
// {
//   return 0.5 * (std::log(cov.determinant()) + cov.rows() * std::log(2 * M_PI) + cov.rows());
// }

// double logDopt(const MatrixXd &cov)
// {
//   return cov.determinant();
// }

}  // namespace base
}  // namespace bruce_exploration