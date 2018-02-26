#include <stdio.h>
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include "bins.h" 
#include "var-distribute-data.h"

/*  Only effective if N is much smaller than RAND_MAX */
void shuffle_block(int *array, size_t n) {
  if (n > 1) {
    size_t i;
    for (i = 0; i < n - 1; i++) {
      size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
      int t = array[j];
      array[j] = array[i];
      array[i] = t;
    }
  }
}

void print_array (int *vec, int rows, char name[]) {

 int leni;
 FILE *fp;
 fp = fopen(name, "w");

 for (leni =0; leni < rows; leni++) {
      fprintf(fp, "%d\n", *(vec + leni));
   }

fclose (fp);

}


void var_distribute_data (float *d, int local, int q_rows, int n_rows, int n_cols, int k_rows, float *B_out, int L, int D, MPI_Comm comm_world, MPI_Comm comm_group) {

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

#ifndef SIMPLESAMPLE
  int *sample;
  if (rank_group == 0) {
    int row_ids[n_rows-L+1];
    for (i=0;i<(n_rows-L+1);i++) row_ids[i] = i;
    shuffle_block(row_ids, (n_rows-L+1)); //shuffle row_ids 

    int s = n_rows/L;
    int b = n_rows % L; 
    int t;
    sample = (int *)malloc(n_rows * sizeof(int) );
 
    for (i=0; i<s;i+=D){
       for(j=0;j<L;j++){  
          sample[(i*L)+j] = row_ids[i]+j; 
	  t = row_ids[i];
	}
    }
    if (b>0){
	int tmp = rand() % b;
	j=0;
	for(i=s*L;i<(s*L)+b;i++) {
	   sample[i]=tmp+t+j;
	   //printf("sample[%d]=%d\n", i, sample[i]);
	   j++;
	}
		

    }	
   print_array(sample, n_rows, "./data/sample_last.dat");
		 
  } else {
    sample = NULL;
  }

  //printf("q_rows = %d\n", q_rows); 

  int s_rows[q_rows]; 
  
  {
    int sendcounts[size_group];
    int displs[size_group];

    for (i=0; i<size_group; i++) {
      int ubound;
      bin_range_1D(i, n_rows, size_group, &displs[i], &ubound);
      sendcounts[i] = bin_size_1D(i, n_rows, size_group);
    }

    MPI_Scatterv(sample, sendcounts, displs, MPI_INT, &s_rows, q_rows, MPI_INT, 0, comm_group);

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

    //if (trow<0 || trow>n_rows)	
//	print_array(s_rows, q_rows, "./data/s_rows_dis.dat");

    int target_rank = bin_coord_1D(trow, n_rows, size_group); 

    int target_disp = bin_index_1D(trow, n_rows, size_group) * n_cols;

  //  if (target_disp < 0)
    //    printf("var_dis i: %d\t trow: %d\t n_rows: %d\t size_group: %d\t n_cols: %d\t target_disp: %d\t rank: %d\n", i, trow, n_rows-D, size_group, n_cols, target_disp, rank_group);
 

    MPI_Get( &B_out[i*n_cols], n_cols, MPI_FLOAT, target_rank, target_disp, n_cols, MPI_FLOAT, win);
  }

  MPI_Win_fence(MPI_MODE_NOSUCCEED, win);

  double tmax, tcomm = MPI_Wtime() - t;
  MPI_Reduce(&tcomm, &tmax, 1, MPI_DOUBLE, MPI_MAX, 0, comm_world);
  if (rank_world == 0) {
    printf("Comm time: %f (s)\n", tmax);
  }
  
  MPI_Win_free(&win);
}

void var_vectorize_response(float *d, int local, int yrows, int n_rows, int n_cols, float *B_out, int L, int D, MPI_Comm comm_world, MPI_Comm comm_group)
{

  int i;
  size_t sized = (size_t) local * (size_t) (n_cols) * sizeof(float);  

  int rank_world, nprocs_world;
  MPI_Comm_rank(comm_world, &rank_world);
  MPI_Comm_size(comm_world, &nprocs_world);

  int size_group, rank_group;
  MPI_Comm_size(comm_group, &size_group);
  MPI_Comm_rank(comm_group, &rank_group);


  MPI_Win win;
  MPI_Win_create(d, sized, sizeof(float), MPI_INFO_NULL, comm_world, &win);

  
#ifndef SIMPLESAMPLE
  int *sample;
  if (rank_group == 0) {
    sample = (int *)malloc((n_rows-D)*sizeof(int)); 
    for (i=0; i<n_rows-D; i++) sample[i]=i+D;
  } else {
    sample = NULL;
  }
	
  int s_rows[yrows];

  {
    int sendcounts[size_group];
    int displs[size_group];

    for (i=0; i<size_group; i++) {
      int ubound;
      bin_range_1D(i, n_rows-D, size_group, &displs[i], &ubound);
      sendcounts[i] = bin_size_1D(i, n_rows-D, size_group);
    }

    MPI_Scatterv(sample, sendcounts, displs, MPI_INT, &s_rows, yrows, MPI_INT, 0, comm_group);

    if(rank_group==0) free(sample); 

  } 

#endif

  double t = MPI_Wtime();
  MPI_Win_fence(MPI_MODE_NOPUT | MPI_MODE_NOPRECEDE, win);

  for (i=0; i<yrows; i++) {
#ifdef SIMPLESAMPLE
    int trow = (int) random_at_mostL( (long) n_rows);
#else
    int trow = s_rows[i]-1; //-1 because now bdata is indexed from 0:N-1 and not from 1:N-2
#endif
    int target_rank = bin_coord_1D(trow, n_rows, size_group);
 
    int target_disp = bin_index_1D(trow, n_rows-D, size_group) * n_cols;

   // if (target_disp < 0)
     //   printf("var_vec i: %d\t trow: %d\t n_rows: %d\t size_group: %d\t n_cols: %d\t target_disp: %d\t rank: %d\n", i, trow, n_rows-D, size_group, n_cols, target_disp, rank_group);


    MPI_Get( &B_out[i*n_cols], n_cols, MPI_FLOAT, target_rank, target_disp, n_cols, MPI_FLOAT, win);
  }

  MPI_Win_fence(MPI_MODE_NOSUCCEED, win);

  /*double tmax, tcomm = MPI_Wtime() - t;
  MPI_Reduce(&tcomm, &tmax, 1, MPI_DOUBLE, MPI_MAX, 0, comm_world);
  if (rank_world == 0) {
    printf("Comm time: %f (s)\n", tmax);
  }*/

  MPI_Win_free(&win);
 

}


void var_generate_Z(float *d, int local, int yrows, int n_rows, int n_cols, float *B_out, int D, MPI_Comm comm_world, MPI_Comm comm_group) 
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

#ifndef SIMPLESAMPLE
  int *sample;
  if (rank_group == 0) {
    sample = (int *)malloc((n_rows-D)*D*sizeof(int));
    for (i=0; i<n_rows-D; i++)
        for (j=0;j<D;j++)
                sample[(i*D)+j]=i+D+j;

  } else {
    sample = NULL;
  }

  //int yrows = bin_size_1D(rank_group, (n_rows-D)*D, nprocs_group)
  int s_rows[yrows];

  {
    int sendcounts[size_group];
    int displs[size_group];

    for (i=0; i<size_group; i++) {
      int ubound;
      bin_range_1D(i, (n_rows-D)*D, size_group, &displs[i], &ubound);
      sendcounts[i] = bin_size_1D(i, (n_rows-D)*D, size_group);
    }

    MPI_Scatterv(sample, sendcounts, displs, MPI_INT, &s_rows, yrows, MPI_INT, 0, comm_group);

    if(rank_group==0) free(sample);

  }

#endif 


 double t = MPI_Wtime();
  MPI_Win_fence(MPI_MODE_NOPUT | MPI_MODE_NOPRECEDE, win);

  for (i=0; i<yrows; i++) {
#ifdef SIMPLESAMPLE
    int trow = (int) random_at_mostL( (long) n_rows);
#else
    int trow = s_rows[i]-1; //-1 because now bdata is indexed from 0:N-1 and not from 1:N-2
#endif
    int target_rank = bin_coord_1D(trow, (n_rows-D), size_group);  
    int target_disp = bin_index_1D(trow, (n_rows-D)*D, size_group) * n_cols;
   
    //if (target_disp < 0)
      //  printf("var_gen i: %d\t trow: %d\t n_rows: %d\t size_group: %d\t n_cols: %d\t target_disp: %d\t rank: %d\n", i, trow, n_rows-D, size_group, n_cols, target_disp, rank_group);

    MPI_Get( &B_out[i*n_cols], n_cols, MPI_FLOAT, target_rank, target_disp, n_cols, MPI_FLOAT, win);
  }

  MPI_Win_fence(MPI_MODE_NOSUCCEED, win);

  MPI_Win_free(&win);
}
