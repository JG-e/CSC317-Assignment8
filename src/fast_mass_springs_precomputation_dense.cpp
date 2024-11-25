#include "fast_mass_springs_precomputation_dense.h"
#include "signed_incidence_matrix_dense.h"
#include <Eigen/Dense>
#define w 1e10
using namespace Eigen;

bool fast_mass_springs_precomputation_dense(
  const Eigen::MatrixXd & V,
  const Eigen::MatrixXi & E,
  const double k,
  const Eigen::VectorXd & m,
  const Eigen::VectorXi & b,
  const double delta_t,
  Eigen::VectorXd & r,
  Eigen::MatrixXd & M,
  Eigen::MatrixXd & A,
  Eigen::MatrixXd & C,
  Eigen::LLT<Eigen::MatrixXd> & prefactorization)
{
  /////////////////////////////////////////////////////////////////////////////

  Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(V.rows(),V.rows());

  signed_incidence_matrix_dense(V.rows(), E, A);

  M = Eigen::MatrixXd::Zero(V.rows(),V.rows());

  C = Eigen::MatrixXd::Zero(b.rows(), V.rows());
  for (int i = 0; i < b.rows(); i++) {
    C(i, b(i)) = 1;
  }

  r.resize(E.rows());
  for (int i = 0; i < E.rows(); i++) {
    auto V0 = V.row( E(i, 0) );
    auto V1 = V.row( E(i, 1) );
    r(i) = (V0 - V1).norm();
  }

  M.diagonal() += m;

  Q = k * A.transpose() * A + M / pow(delta_t,2);
  MatrixXd Qc = w * C.transpose() * C; // Qc = wCᵀC
  Q += Qc;
  /////////////////////////////////////////////////////////////////////////////
  prefactorization.compute(Q);
  return prefactorization.info() != Eigen::NumericalIssue;
}
