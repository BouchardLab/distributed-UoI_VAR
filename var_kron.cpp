#include <iostream>
#include <fstream>
#include <mpi.h>
#define EIGEN_USE_MKL_ALL
#include <eigen3/Eigen/Dense>
#include <unistd.h>
#include <string.h>
#include "bins.h"
#include "var_kron.h"

using namespace Eigen; 
using namespace std; 

float *var_kron (float *d, int local, int yrows, int n_rows, int n_cols, int D,  MPI_Comm comm_world, MPI_Comm comm_group)
{
  int i, j;
  size_t sized = (size_t) local * (size_t) (n_cols) * sizeof(float);

  int rank_world, nprocs_world;
  MPI_Comm_rank(comm_world, &rank_world);
  MPI_Comm_size(comm_world, &nprocs_world);

  int size_group, rank_group;
  MPI_Comm_size(comm_group, &size_group);
  MPI_Comm_rank(comm_group, &rank_group);


  MPI_Win win;
  MPI_Win_create(d, sized, sizeof(float), MPI_INFO_NULL, comm_world, &win);

 int *sample;
   if (rank_group==0) {
        sample = (int *)malloc((n_rows-D)* n_cols * sizeof(int));
        for (int i=0; i<(n_rows-D)*n_cols;i++) sample[i] = i;
   } else {
    sample = NULL;
  }

  int srows[yrows];

  {
    int sendcounts[size_group];
    int displs[size_group];

    for (i=0; i<size_group; i++) {
      int ubound;
      bin_range_1D(i, (n_rows-D)* n_cols, size_group, &displs[i], &ubound);
      sendcounts[i] = bin_size_1D(i, (n_rows-D), size_group) * n_cols;
    }

    MPI_Scatterv(sample, sendcounts, displs, MPI_INT, &srows, yrows, MPI_INT, 0, comm_group);

    if(rank_group==0) free(sample);
  }



  double t = MPI_Wtime();
  MPI_Win_fence(MPI_MODE_NOPUT | MPI_MODE_NOPRECEDE, win);
  
  Matrix<float, Dynamic, Dynamic, RowMajor>B_out(yrows, D*n_cols*n_cols);
  B_out.setZero();  

  for (i=0; i<yrows; i++) {
#ifdef SIMPLESAMPLE
    int trow = (int) random_at_mostL( (long) n_rows);
#else
    int trow = srows[i] % (n_rows-D);
#endif
    int target_rank = bin_coord_1D(trow, (n_rows-D), nprocs_world);
    int target_disp = bin_index_1D(trow, (n_rows-D), nprocs_world) * n_cols;
    int col_disp = srows[i] / (n_rows-D);  
    MPI_Get(B_out.row(i).segment((col_disp*n_cols), n_cols).data(), n_cols, MPI_FLOAT, target_rank, target_disp, n_cols, MPI_FLOAT, win);
  }

  MPI_Win_fence(MPI_MODE_NOSUCCEED, win);

  MPI_Win_free(&win);
	
  float *out;
  out = (float *)malloc(B_out.rows() * B_out.cols() * sizeof(float));
  Map<Matrix<float, Dynamic, Dynamic, RowMajor> >(out, B_out.rows(), B_out.cols()) = B_out; 
  return out;
}
