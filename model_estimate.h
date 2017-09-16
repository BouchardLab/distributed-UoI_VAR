#ifndef ESTIMATE_H
#define ESTIMATE_H

#ifdef __cplusplus
        extern "C" {
#endif

//void model_estimate(float *sprt_in, float *X_train_e, float *y_train_e, float *X_test_e, float *y_test_e, float *X_T_e, float *y_T_e, int L, int T, int train, int test,
//              float *Bgd, float *rsd, float *R2, float *bic, int m_node, int n, int nbootS, int nbootE, int nrnd, float *time, MPI_Comm comm_WORLD, MPI_Comm comm_NRND, MPI_Comm comm_EST);


//void model_estimate(float *A_in, int local,  float *sprt_in, int L, int train, int m, int n, int n_rows, int k_rows, int nbootS, int nbootE, int nrnd, float *time,
  //                      float *Bgd_out, float *rsd_out, float *R2_out, float *bic_out, MPI_Comm comm_WORLD, MPI_Comm comm_NRND); 

//void randomize (float *Mat_train, float *Vec_train, float *Mat_test, float *Vec_test, int rows, int cols, int l);

void model_estimate(float *X_L_e, float *y_L_e, float *X_T_e, float *y_T_e, float *sprt_in, int L, int train, int m, int n, int n_rows, int k_rows, int nbootS, int nbootE, int nrnd, double *time, 
                        float *Bgd_out, float *rsd_out, float *R2_out, float *bic_out, int *roots, MPI_Comm comm_WORLD, MPI_Comm comm_group, MPI_Comm comm_NRND);


#ifdef __cplusplus
 }
#endif

#endif
