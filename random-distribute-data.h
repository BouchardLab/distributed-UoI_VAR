#ifndef RAND_DISTRIBUTE_H
#define RAND_DISTRIBUTE_H

#ifdef __cplusplus
        extern "C" {
#endif
	void random_distribute_data (float *d, int local, int q_rows, int n_rows, int n_cols, int k_rows, float *B_all, MPI_Comm comm_world, MPI_Comm comm_group); 



#ifdef __cplusplus
 }
#endif


#endif
