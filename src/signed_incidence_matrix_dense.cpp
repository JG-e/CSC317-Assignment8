#include "signed_incidence_matrix_dense.h"
using namespace std;
using namespace Eigen;
void signed_incidence_matrix_dense(
  const int n,
  const Eigen::MatrixXi & E,
  Eigen::MatrixXd & A)
{
  //////////////////////////////////////////////////////////////////////////////
  A = Eigen::MatrixXd::Zero(E.rows(),n);
  for(int e = 0; e<E.rows(); e++)
  {
    A(e, E(e,0)) = 1;
    A(e, E(e,1)) = -1;
  }
  
  //////////////////////////////////////////////////////////////////////////////
}
