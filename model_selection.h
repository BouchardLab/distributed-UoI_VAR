#ifndef SELECTION_H
#define SELECTION_H

#ifdef __cplusplus
        extern "C" {
#endif

void model_selection (float *X_train_e, float *y_train_e, float *X_test_e, float *y_test_e, float *lamb, float *B_out, float *R_out, double *lastime, double *las2time,  int nboot, int coarse, int train, int n, int q_rows, int nMP, MPI_Comm comm_world, MPI_Comm comm_group);

#ifdef __cplusplus
 }
#endif

#endif
