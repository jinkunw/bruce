#include <gtsam/geometry/Pose2.h>
#include <gtsam/inference/Symbol.h>
#include <queue>
#include "bruce_exploration/base/belief_propagation.h"

namespace bruce_exploration
{
namespace base
{
BeliefPropagation::BeliefPropagation()
{
}

void BeliefPropagation::setISAM2(const gtsam::ISAM2 &isam2)
{
  isam2_ = isam2;
  isam2_.disableRelinearization();

  cache_.clear();
  auto estimate = isam2_.calculateEstimate();
  for (gtsam::Key key : estimate.keys())
  {
    gtsam::Pose2 pose = estimate.at<gtsam::Pose2>(key);
    gtsam::Matrix cov = isam2_.marginalCovariance(key);
    cache_.push_back({ Pose(pose.x(), pose.y(), pose.theta()), cov });
  }
}

void BeliefPropagation::split(const gtsam::NonlinearFactorGraph &graph, gtsam::NonlinearFactorGraph &odom_factors,
                              gtsam::NonlinearFactorGraph &loop_factors) const
{
  for (const auto &factor : graph)
  {
    if (factor->size() != 2)
    {
      std::cout << "only binary factors are supported" << std::endl;
    }
    // if ((factor->keys()[0] <= last_key && factor->keys()[1] > last_key) ||
    //     (factor->keys()[0] > last_key && factor->keys()[1] <= last_key))
    if (factor->keys()[0] + 1 == factor->keys()[1])
      odom_factors.push_back(factor);
    else
      loop_factors.push_back(factor);
  }
}

std::vector<BeliefState> BeliefPropagation::propagate(const gtsam::NonlinearFactorGraph &graph,
                                                      const gtsam::Values &values)
{
  std::vector<BeliefState> beliefs = propagateL(graph, values);
  for (auto &belief : beliefs)
  {
    double theta = belief.pose.theta();
    gtsam::Matrix2 R;
    R << std::cos(theta), -std::sin(theta), std::sin(theta), std::cos(theta);
    belief.cov.block<2, 3>(0, 0) = R * belief.cov.block<2, 3>(0, 0);
    belief.cov.block<3, 2>(0, 0) = belief.cov.block<3, 2>(0, 0) * R.transpose();
  }
  return beliefs;
}

std::vector<BeliefState> BeliefPropagation::propagateL(const gtsam::NonlinearFactorGraph &graph,
                                                       const gtsam::Values &values)
{
  if (graph.empty())
    return cache_;
  else
  {
    gtsam::NonlinearFactorGraph odom_factors, loop_factors;
    split(graph, odom_factors, loop_factors);

    std::vector<BeliefState> beliefs = cache_;
    auto indices = isam2_.update(graph, values).newFactorsIndices;

    // keys in values are supposed to be sorted
    for (gtsam::Key key : values.keys())
    {
      // gtsam::Pose2 pose = isam2_.calculateEstimate<gtsam::Pose2>(key);
      gtsam::Pose2 pose = values.at<gtsam::Pose2>(key);
      gtsam::Matrix cov = isam2_.marginalCovariance(key);
      beliefs.push_back({ Pose(pose.x(), pose.y(), pose.theta()), cov });
    }

    if (!loop_factors.empty())
    {
      auto keys = isam2_.getLinearizationPoint().keys();
      for (size_t x = 0; x < keys.size(); ++x)
      {
        beliefs[x].cov = isam2_.marginalCovariance(keys[x]);
      }
    }

    isam2_.update(gtsam::NonlinearFactorGraph(), gtsam::Values(), indices);
    return beliefs;
  }
}

BeliefPropagationPlus::BeliefPropagationPlus() : BeliefPropagation(), initialized_(false)
{
}

void BeliefPropagationPlus::initialize()
{
  // Get the ordering
  std::queue<gtsam::ISAM2Clique::shared_ptr> q;
  q.push(isam2_.roots()[0]);
  while (!q.empty())
  {
    gtsam::ISAM2Clique::shared_ptr c = q.front();
    gtsam::ISAM2Clique::sharedConditional conditional = c->conditional();
    q.pop();
    for (auto child : c->children)
      q.push(child);
    ordering_.insert(ordering_.begin(), conditional->beginFrontals(), conditional->endFrontals());
  }

  for (size_t i = 0; i < ordering_.size(); ++i)
    indices_[ordering_[i]] = i;

  // Get the sparse R matrix
  R_ = Eigen::SparseMatrix<double>(3 * ordering_.size(), 3 * ordering_.size());
  std::vector<Eigen::Triplet<double>> coeffs;

  q.push(isam2_.roots()[0]);
  while (!q.empty())
  {
    gtsam::ISAM2Clique::shared_ptr c = q.front();
    gtsam::ISAM2Clique::sharedConditional conditional = c->conditional();
    q.pop();

    const auto &R = conditional->get_R();
    const auto &frontals = conditional->frontals();
    for (size_t r = 0; r < R.rows(); ++r)
    {
      size_t i = indices_[frontals[r / 3]];
      for (size_t c = r; c < R.cols(); ++c)
      {
        size_t j = indices_[frontals[c / 3]];
        coeffs.emplace_back(3 * i + r % 3, 3 * j + c % 3, R(r, c));
      }
    }
    const auto &S = conditional->get_S();
    const auto &parents = conditional->parents();
    for (size_t r = 0; r < S.rows(); ++r)
    {
      size_t i = indices_[frontals[r / 3]];
      for (size_t c = 0; c < S.cols(); ++c)
      {
        size_t j = indices_[parents[c / 3]];
        coeffs.emplace_back(3 * i + r % 3, 3 * j + c % 3, S(r, c));
      }
    }
    for (auto child : c->children)
      q.push(child);
  }

  R_.setFromTriplets(coeffs.begin(), coeffs.end());
  // std::cout << R_.toDense() << std::endl;

  // Cache cov blocks
  cov_ = gtsam::Matrix(R_.rows(), R_.cols());
  cov_.fill(NAN);

  // Get most recent pose
  xn_ = isam2_.getLinearizationPoint().keys().back();
  initialized_ = true;
}

gtsam::Matrix BeliefPropagationPlus::jointMarginalCovariance(const gtsam::KeyVector &variables1,
                                                             const gtsam::KeyVector &variables2)
{
  // Calculate column cov for each variable 2
  for (size_t vj = 0; vj < variables2.size(); ++vj)
  {
    size_t j = indices_ext_[variables2[vj]];
    gtsam::Matrix cov_col;
    if (j < ordering_.size())
      recoverBlockColumn(j);
  }

  gtsam::Matrix cov = gtsam::Matrix::Zero(3 * variables1.size(), 3 * variables2.size());
  for (size_t vj = 0; vj < variables2.size(); ++vj)
  {
    size_t j = indices_ext_[variables2[vj]];
    for (size_t vi = 0; vi < variables1.size(); ++vi)
    {
      size_t i = indices_ext_[variables1[vi]];
      cov.block<3, 3>(vi * 3, vj * 3) = recover(i, j);
    }
  }
  return cov;
}

gtsam::Matrix BeliefPropagationPlus::recover(const size_t &i, const size_t &j)
{
  if (i > j)
    return recover(j, i).transpose();
  if (j >= ordering_.size())
    return cov_ext_.block<3, 3>(i * 3, (j - ordering_.size()) * 3);
  else
    return cov_.block<3, 3>(i * 3, j * 3);
}

gtsam::Matrix BeliefPropagationPlus::recoverBlockColumn(const size_t &v)
{
  if (calculated_.find(v) != calculated_.end())
    return cov_.middleCols(v * 3, 3);

  // R * Sv = R^{-T} * Iv
  gtsam::Matrix Iv = gtsam::Matrix::Zero(ordering_.size() * 3, 3);
  Iv.block<3, 3>(3 * v, 0) = gtsam::I_3x3;

  // 1) R^T * Bv = Iv
  const gtsam::Matrix Bv = R_.transpose().triangularView<Eigen::Lower>().solve(Iv);
  // 2) R * Sv = Bv
  const gtsam::Matrix Sv = R_.triangularView<Eigen::Upper>().solve(Bv);

  cov_.middleCols(v * 3, 3) = Sv;
  cov_.middleRows(v * 3, 3) = Sv.transpose();
  calculated_.insert(v);
  return Sv;
}

std::vector<BeliefState> BeliefPropagationPlus::propagateL(const gtsam::NonlinearFactorGraph &odom_factors,
                                                           const gtsam::NonlinearFactorGraph &loop_factors,
                                                           const gtsam::Values &values)
{
  gtsam::Values initial(isam2_.getLinearizationPoint());
  initial.insert(values);

  gtsam::GaussianFactorGraph::shared_ptr linear_factors = odom_factors.linearize(initial);
  std::sort(linear_factors->begin(), linear_factors->end(),
            [](const gtsam::GaussianFactor::shared_ptr &f1, const gtsam::GaussianFactor::shared_ptr &f2) {
              return f1->keys()[0] < f2->keys()[0];
            });

  auto beliefs = cache_;
  for (size_t t = 0; t < linear_factors->size(); ++t)
  {
    gtsam::GaussianFactor::shared_ptr factor = linear_factors->at(t);
    // Propagate the diagonal term in cov
    // A1 * x1 + A2 * x2 = b
    // x2 = -A2^{-1} * A1 * x1 + A2^{-1} * u, u ~ N(0, I)
    // x2 = H1 * x1 + H2 * u
    const auto &A = factor->jacobian().first;
    gtsam::Matrix H2 = A.block<3, 3>(0, 3).inverse();
    gtsam::Matrix H1 = -H2 * A.block<3, 3>(0, 0);

    // cov from odometry
    gtsam::Pose2 pose = initial.at<gtsam::Pose2>(factor->keys()[1]);
    gtsam::Matrix cov = H1 * beliefs.back().cov * H1.transpose() + H2 * H2.transpose();
    beliefs.push_back({ Pose(pose.x(), pose.y(), pose.theta()), cov });
  }
  if (loop_factors.empty())
    return beliefs;

  if (!initialized_)
    initialize();

  gtsam::KeyVector keys = values.keys();
  ordering_ext_ = ordering_;
  ordering_ext_.insert(ordering_ext_.end(), keys.begin(), keys.end());
  indices_ext_.clear();
  for (size_t i = 0; i < ordering_ext_.size(); ++i)
    indices_ext_[ordering_ext_[i]] = i;

  cov_ext_ = gtsam::Matrix::Zero(ordering_ext_.size() * 3, values.size() * 3 + 3);
  cov_ext_.block(0, 0, ordering_.size() * 3, 3) = recoverBlockColumn(indices_[xn_]);
  for (size_t t = 0; t < linear_factors->size(); ++t)
  {
    // Propagate the off-diagonal term in cov if there exist loop closure factors
    const gtsam::GaussianFactor::shared_ptr factor = linear_factors->at(t);
    const auto &A = factor->jacobian().first;
    const gtsam::Matrix H2 = A.block<3, 3>(0, 3).inverse();
    const gtsam::Matrix H1 = -H2 * A.block<3, 3>(0, 0);

    size_t i = ordering_.size() + t;
    size_t j = t + 1;

    // x2 = H1 * x1 + H2 * u
    // S[:, x(j)] = S[:, x(j - 1)] * H1^T
    cov_ext_.block(0, j * 3, i * 3, 3) = cov_ext_.block(0, (j - 1) * 3, i * 3, 3) * H1.transpose();
    // S[x(j), x(j)] = H1 * S[x(j - 1), x(j - 1)] * H1^T + H2 * H2^T
    if (t == 0)
      // The index of last pose is not j - 1
      cov_ext_.block<3, 3>(i * 3, j * 3) =
          H1 * cov_ext_.block<3, 3>(indices_ext_[xn_] * 3, (j - 1) * 3) * H1.transpose() + H2 * H2.transpose();
    else
      cov_ext_.block<3, 3>(i * 3, j * 3) =
          H1 * cov_ext_.block<3, 3>((i - 1) * 3, (j - 1) * 3) * H1.transpose() + H2 * H2.transpose();
  }
  // https://stackoverflow.com/questions/30145771/shrink-matrix-with-eigen-using-block-in-assignment
  cov_ext_ = cov_ext_.rightCols(values.size() * 3).eval();

  // Fast Covariance Recovery in Incremental Nonlinear Least Square Solvers
  // Fig. 3: Sparsity patterns involved in covariance update calculation
  //   S_hat =  S - delta_S, where
  // delta_S =  B * U^{−1} * B^T
  //       B =  S * Au^T
  //       U =  I + Au * S * Au^T
  const auto &linear_loop_factors = loop_factors.linearize(initial);
  const gtsam::KeyVector loop_keys = linear_loop_factors->keyVector();
  const gtsam::Ordering loop_ordering(loop_keys);
  const gtsam::Matrix Au = linear_loop_factors->jacobian(loop_ordering).first;
  const gtsam::KeyVector all_keys = initial.keys();
  const gtsam::Matrix S = jointMarginalCovariance(all_keys, loop_keys);
  const gtsam::Matrix SS = jointMarginalCovariance(loop_keys, loop_keys);
  const gtsam::Matrix B = S * Au.transpose();
  const gtsam::Matrix U = gtsam::Matrix::Identity(Au.rows(), Au.rows()) + Au * SS * Au.transpose();
  const gtsam::Matrix LTI = U.llt().matrixU().solve(gtsam::Matrix::Identity(U.rows(), U.cols()));
  const gtsam::Matrix BLTI = B * LTI;
  for (size_t x = 0; x < beliefs.size(); ++x)
  {
    const gtsam::Matrix delta = BLTI.middleRows(x * 3, 3) * BLTI.middleRows(x * 3, 3).transpose();
    beliefs[x].cov -= delta;
  }
  return beliefs;
}

std::vector<BeliefState> BeliefPropagationPlus::propagateL(const gtsam::NonlinearFactorGraph &graph,
                                                           const gtsam::Values &values)
{
  if (graph.empty())
    return cache_;

  gtsam::NonlinearFactorGraph odom_factors;
  gtsam::NonlinearFactorGraph loop_factors;
  split(graph, odom_factors, loop_factors);
  return propagateL(odom_factors, loop_factors, values);
}

}  // namespace base
}  // namespace bruce_exploration
