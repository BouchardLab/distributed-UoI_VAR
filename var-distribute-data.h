#ifndef VAR_DISTRIBUTE_H
#define VAR_DISTRIBUTE_H

#ifdef __cplusplus
        extern "C" {
#endif
	void var_distribute_data (float *d, int local, int q_rows, int n_rows, int n_cols, int k_rows, float *B_all, int L, int D, MPI_Comm comm_world, MPI_Comm comm_group);
	void var_vectorize_response(float *d, int local, int yrows, int n_rows, int n_cols, float *B_out, int L, int D, MPI_Comm comm_world, MPI_Comm comm_group);
	void var_generate_Z(float *d, int local, int yrows, int n_rows, int n_cols, float *B_out, int D, MPI_Comm comm_world, MPI_Comm comm_group);  



#ifdef __cplusplus
 }
#endif


#endif
