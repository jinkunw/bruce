#pragma once
#include <gtsam/nonlinear/ISAM2.h>
#include <ros/ros.h>
#include <Eigen/Sparse>
#include "bruce_exploration/base/types.h"

namespace bruce_exploration
{
namespace base
{
class BeliefPropagation
{
public:
  BeliefPropagation();

  void setISAM2(const gtsam::ISAM2 &isam2);

  const gtsam::ISAM2 &getISAM2() const
  {
    return isam2_;
  }

  void split(const gtsam::NonlinearFactorGraph &graph, gtsam::NonlinearFactorGraph &odom_factors,
             gtsam::NonlinearFactorGraph &loop_factors) const;

  std::vector<BeliefState> propagate(const gtsam::NonlinearFactorGraph &graph = gtsam::NonlinearFactorGraph(),
                                     const gtsam::Values &values = gtsam::Values());

protected:
  virtual std::vector<BeliefState> propagateL(const gtsam::NonlinearFactorGraph &graph = gtsam::NonlinearFactorGraph(),
                                              const gtsam::Values &values = gtsam::Values());

  gtsam::ISAM2 isam2_;
  std::vector<BeliefState> cache_;
};

class BeliefPropagationPlus : public BeliefPropagation
{
public:
  BeliefPropagationPlus();

private:
  void initialize();

  std::vector<BeliefState> propagateL(const gtsam::NonlinearFactorGraph &odom_graph,
                                      const gtsam::NonlinearFactorGraph &loop_graph, const gtsam::Values &values);

  std::vector<BeliefState> propagateL(const gtsam::NonlinearFactorGraph &graph, const gtsam::Values &values) override;

  gtsam::Matrix recover(const size_t &i, const size_t &j);

  gtsam::Matrix recoverBlockColumn(const size_t &v);

  gtsam::Matrix jointMarginalCovariance(const gtsam::KeyVector &variables1, const gtsam::KeyVector &variables2);

  // ordering = [0 2 3 1 5 4]
  gtsam::KeyVector ordering_;
  // indices = {0: 0, 1: 3, 2: 1, 3: 2, 4: 5, 5: 4}
  gtsam::FastMap<gtsam::Key, size_t> indices_;
  // Upper triangular part of cov
  // which is retrieved by the covariance recovery algorithm in
  // - Covariance Recovery from a Square Root Information Matrix for Data Association
  //  C(0, 0)  C(0, 2)  C(0, 3)  C(0, 1)  C(0, 5)  C(0, 4)
  //           C(2, 2)  C(2, 3)  C(2, 1)  C(2, 5)  C(2, 4)
  //                    C(3, 3)  C(3, 1)  C(3, 5)  C(3, 4)
  //                             C(1, 1)  C(1, 5)  C(1, 4)
  //                                      C(5, 5)  C(4, 4)
  //                                               C(4, 4)
  gtsam::Matrix cov_;
  // flag for column in cov
  gtsam::FastSet<size_t> calculated_;
  // Upper triangular matrix R from QR/Cholesky in ISAM2
  Eigen::SparseMatrix<double> R_;

  // last pose x5
  gtsam::Key xn_;
  // extended ordering = [0 2 3 1 5 4 6 7 8]
  gtsam::KeyVector ordering_ext_;
  // extended indices = {0: 0, 1: 3, 2: 1, 3: 2, 4: 5, 5: 4, 6: 6, 7: 7, 8: 8}
  gtsam::FastMap<gtsam::Key, size_t> indices_ext_;
  // right cols of cov
  // which is calculated by
  // - C(i, j + 1) = C(i, j) H_j^T
  // - C(j + 1, j + 1) = H_j C(j, j) H_j^T + Q
  // C(0, 6)  C(0, 7)  C(0, 8)
  // C(2, 6)  C(2, 7)  C(2, 8)
  // C(3, 6)  C(3, 7)  C(3, 8)
  // C(1, 6)  C(1, 7)  C(1, 8)
  // C(5, 6)  C(5, 7)  C(5, 8)
  // C(4, 6)  C(4, 7)  C(4, 8)
  // C(6, 6)  C(6, 7)  C(6, 8)
  //          C(7, 7)  C(7, 8)
  //                   C(8, 8)
  gtsam::Matrix cov_ext_;

  bool initialized_;
};

}  // namespace base
}  // namespace bruce_exploration
