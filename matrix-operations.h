#ifndef MATRIX_OPERATIONS_H
#define MATRIX_OPERATIONS_H

#include <eigen3/Eigen/Dense>

using namespace Eigen; 

#ifdef __cplusplus
	extern "C" {
#endif
		//float *combine_matrix (float * Mat, float *Arr, int m, int n); 
		float * logspace (int start, int end, int size);

		float* linspace(float start, float end, int size);

		void combine_matrix (float *Mat, float *Vec, float *Out, int m, int n);

		void split_matrix (float *In, float *Mat, float *Vec, int m, int n); 

            	//void split_matrix (float *In, float *Mat, float *Vec, int m, int n, int this_rank);

		//void get_train (float *Mat, float *Vec, float *Mat_train, float *Vec_train, int m, int n, int train); 

		//void get_test (float *Mat, float *Vec, float *Out_hat, float *Out_mean, int m, int n, int train, float *B, int rank);

		//void get_train (float *Mat, float *Mat_train, float *Vec_train, int m, int n, int train); 

		//void get_test (float *Mat, float *Out_hat, float *Out_mean, int m, int n, int train, float *B, int rank); 				

		void get_train (float *Mat, float *Mat_train, float *Vec_train, float *Mat_test, float *Vec_test, int m, int n, int train);

		void get_test(float *Mat, float *Vec, int m, int n, float *B_in, float *Out_hat);

		//void get_estimate(float *Mat, float *Mat_T, float *Vec_T, int m, int n, int train); 

		void get_estimate(float *Mat, float *Mat_tr, float *Vec_tr, float *Mat_tst, float *Vec_tst, float *Mat_T, float *Vec_T,  int m, int n, int CV, int train_row); 

		void get_estimate2(float *Mat, float *Mat_L, float *Vec_L, float *Mat_T, float *Vec_T,  int m, int n, int CV);

		//float pearson (float *vec1, float *vec2, int m);

		//float dense_sweep (float *R2m, float *lamb, int maxBoot, int nMP, int m); 

		float dense_sweep (float *R2m, float *lamb, int Boot);

 		float get_dv (float s); 

		void get_support (MatrixXf B_eig, float *sprt_in, int nMP, int nbootS, int n);

		void average (float *In, int max, int nM, int y, int size);

		//void get_random_rows(float *In, float *Out1, float *Out2, int rows, int m, int n, MPI_Comm comm);  

		void get_random_rows(float *In, float *Out1, float *Out2, int rows, int m_train, int n, int L_g, int T_g, MPI_Comm comm); 
		
		//void average_v (float *In, int rnd, int y, int size);

		VectorXi setdiff1d (VectorXi vec1, VectorXi vec2);

		VectorXi where (VectorXf vec, float value);

		//double pearson1 (VectorXf vec1, VectorXf vec2);

		float pearson1 (const VectorXf& x, const VectorXf& y); 

		VectorXf median (MatrixXf mat);  	  
		
#ifdef __cplusplus
 }
#endif

#endif
