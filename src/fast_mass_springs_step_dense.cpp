#include "fast_mass_springs_step_dense.h"
#include <igl/matlab_format.h>

using namespace Eigen;
using namespace std;

#define w 1e10

void fast_mass_springs_step_dense(
  const Eigen::MatrixXd & V,
  const Eigen::MatrixXi & E,
  const double k,
  const Eigen::VectorXi & b,
  const double delta_t,
  const Eigen::MatrixXd & fext,
  const Eigen::VectorXd & r,
  const Eigen::MatrixXd & M,
  const Eigen::MatrixXd & A,
  const Eigen::MatrixXd & C,
  const Eigen::LLT<Eigen::MatrixXd> & prefactorization,
  const Eigen::MatrixXd & Uprev,
  const Eigen::MatrixXd & Ucur,
  Eigen::MatrixXd & Unext)
{
  //////////////////////////////////////////////////////////////////////////////
  int satisfied = 50;
  Eigen::MatrixXd d = Eigen::MatrixXd::Zero(E.rows(), 3);
  MatrixXd Y = M * (2 * Ucur - Uprev) / pow(delta_t, 2) + fext;
  MatrixXd var_b;

  Unext = Ucur;

  for(int iter = 0;iter < satisfied; iter++)
  {
    // Local step: Given current values of p, determine dij for each spring
    for (int i = 0; i < E.rows(); i++) {
      d.row(i) = r[i] * (Unext.row( E(i,0) ) - Unext.row( E(i,1) )).normalized();
    }
     
    var_b = k * A.transpose() * d + Y+ w * C.transpose() * C * V;
    Unext = prefactorization.solve(var_b);
  }
  //////////////////////////////////////////////////////////////////////////////
}
