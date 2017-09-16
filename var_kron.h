#ifndef VAR_KRON_H
#define VAR_KRON_H

#ifdef __cplusplus
        extern "C" {
#endif
	float *var_kron (float *d, int local, int yrows, int n_rows, int n_cols, int D,  MPI_Comm comm_world, MPI_Comm comm_group); 



#ifdef __cplusplus
 }
#endif


#endif
