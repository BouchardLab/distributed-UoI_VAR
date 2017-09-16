#ifndef LASSO_H
#define LASSO_H

#include <eigen3/Eigen/Dense>
using namespace Eigen; 

#ifdef __cplusplus
	extern "C" {
#endif

float* lasso(float *A_in, int m, int n,  float *b_in, float lambda, MPI_Comm comm, double *time);	

//float* lasso_admm (MatrixXd  A_in, int m, int n,  VectorXd b_in, float lambda, MPI_Comm comm);
	
#ifdef __cplusplus
 }
#endif

#endif
