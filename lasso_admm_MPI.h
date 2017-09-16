#ifndef WRAPPER_H
#define WRAPPER_H

#ifdef __cplusplus
	extern "C" {
#endif

//double* Wrapper_lasso_admm (double *A_in, int m, int n,  double *b_in, double lambda, MPI_Comm comm);	

double* lasso_admm (double *A_in, int m, int n,  double *b_in, double lambda, MPI_Comm comm);
	
#ifdef __cplusplus
 }
#endif

#endif
