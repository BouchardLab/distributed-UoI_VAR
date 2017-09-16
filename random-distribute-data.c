#include <stdio.h>
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include "bins.h" 
#include "random-distribute-data.h"

/*  Only effective if N is much smaller than RAND_MAX */
/*void shuffle(int *array, size_t n) {
  if (n > 1) {
    size_t i;
    for (i = 0; i < n - 1; i++) {
      size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
      int t = array[j];
      array[j] = array[i];
      array[i] = t;
    }
  }
}*/

void random_distribute_data (float *d, int local, int q_rows, int n_rows, int n_cols, int k_rows, float *B_out,  MPI_Comm comm_world, MPI_Comm comm_group) {

  int i;
  size_t sized = (size_t) local * (size_t) (n_cols+1) * sizeof(float); 

  int rank_world, nprocs_world;
  MPI_Comm_rank(comm_world, &rank_world);
  MPI_Comm_size(comm_world, &nprocs_world);

  int size_group, rank_group;
  MPI_Comm_size(comm_group, &size_group);
  MPI_Comm_rank(comm_group, &rank_group); 


  MPI_Win win;
  MPI_Win_create(d, sized, sizeof(float), MPI_INFO_NULL, comm_world, &win); 

  //if (rank_world == 0)
  //	printf("passed win\n");

  int qcols = n_cols+1;


#ifndef SIMPLESAMPLE
  int *sample;
  if (rank_group == 0) {
    sample = (int *)malloc( n_rows * sizeof(int) );
    for (i=0; i<n_rows; i++) sample[i]=i;
    shuffle(sample, n_rows);
  } else {
    sample = NULL;
  }

  int s_rows[q_rows]; 
  
  {
    /*int sendcounts[size_group];
    int displs[size_group];

    for (i=0; i<size_group; i++) {
      int ubound;
      bin_range_1D(i, k_rows, size_group, &displs[i], &ubound);
      sendcounts[i] = bin_size_1D(i, k_rows, size_group);
    }

    MPI_Scatterv(sample, sendcounts, displs, MPI_INT, &s_rows, q_rows, MPI_INT, 0, comm_group);
    */ 

    MPI_Scatter(sample, q_rows, MPI_INT, &s_rows, q_rows, MPI_INT, 0, comm_group); 

    if (rank_group == 0) free(sample);
  }
  

#endif
 

  double t = MPI_Wtime();
  MPI_Win_fence(MPI_MODE_NOPUT | MPI_MODE_NOPRECEDE, win);

  for (i=0; i<q_rows; i++) {
#ifdef SIMPLESAMPLE
    int trow = (int) random_at_mostL( (long) n_rows);
#else
    int trow = s_rows[i];
#endif
    int target_rank = bin_coord_1D(trow, n_rows, nprocs_world);
    int target_disp = bin_index_1D(trow, n_rows, nprocs_world) * qcols;
    MPI_Get( &B_out[i*qcols], qcols, MPI_FLOAT, target_rank, target_disp, qcols, MPI_FLOAT, win);
  }

  MPI_Win_fence(MPI_MODE_NOSUCCEED, win);

  double tmax, tcomm = MPI_Wtime() - t;
  MPI_Reduce(&tcomm, &tmax, 1, MPI_DOUBLE, MPI_MAX, 0, comm_world);
  if (rank_world == 0) {
    printf("Comm time: %f (s)\n", tmax);
  }
  
  MPI_Win_free(&win);
}
