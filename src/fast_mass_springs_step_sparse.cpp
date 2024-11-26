#include "fast_mass_springs_step_sparse.h"
#include <igl/matlab_format.h>
#define w 1e10

void fast_mass_springs_step_sparse(
  const Eigen::MatrixXd & V,
  const Eigen::MatrixXi & E,
  const double k,
  const Eigen::VectorXi & b,
  const double delta_t,
  const Eigen::MatrixXd & fext,
  const Eigen::VectorXd & r,
  const Eigen::SparseMatrix<double>  & M,
  const Eigen::SparseMatrix<double>  & A,
  const Eigen::SparseMatrix<double>  & C,
  const Eigen::SimplicialLLT<Eigen::SparseMatrix<double> > & prefactorization,
  const Eigen::MatrixXd & Uprev,
  const Eigen::MatrixXd & Ucur,
  Eigen::MatrixXd & Unext)
{
  //////////////////////////////////////////////////////////////////////////////
  int satisfied = 50;
  Eigen::MatrixXd d = Eigen::MatrixXd::Zero(E.rows(), 3);
  Eigen::MatrixXd Y = M * (2 * Ucur - Uprev) / pow(delta_t, 2) + fext;
  Eigen::MatrixXd var_b = Ucur;
  Unext = Ucur;

  for(int iter = 0; iter < satisfied; iter++)
  {
    // Local step: Given current values of p, determine dij for each spring
    for (int i = 0; i < E.rows(); i++) {
      d.row(i) = r[i] * (Unext.row( E(i,0) ) - Unext.row( E(i,1) )).normalized();
    }

    var_b = k * A.transpose() * d + Y + w * C.transpose() * C * V;
    Unext = prefactorization.solve(var_b);
  }
  //////////////////////////////////////////////////////////////////////////////
}
