#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <assert.h>
#include "bins.h"
#include "manage-data.h"
#include "matrix-operations.h"
#include "model_estimate.h"
#include "model_selection.h"
//#include "lasso_admm_MPI.h"
#include "distribute-data.h"
//#include "debug.h"
//#ifdef DEBUG
//#undef DEBUG
//#endif
//#define DEBUG 1

long random_at_mostL(long max) {
  unsigned long num_bins = (unsigned long) max + 1, num_rand = (unsigned long) RAND_MAX + 1, bin_size = num_rand / num_bins, defect = num_rand % num_bins;
  long x;
  do {
    x = random();
  }
  while (num_rand - defect <= (unsigned long)x);
  return x/bin_size;
}

char* readable_fs(float size/*in bytes*/, char *buf) {
  int i = 0;
  const char* units[] = {"B", "kB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"};
  while (size > 1024) {
    size /= 1024;
    i++;
  }
  sprintf(buf, "%.*f %s", i, size, units[i]);
  return buf;
}

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


/* Print Matrix */
void print_mat (float *Mat, int rows, int cols, char name[]) {

 int leni, lenj;
 FILE *fp;
 fp = fopen(name, "w");

 for (leni =0; leni < rows; leni++) {
   for (lenj =0; lenj < cols; lenj++) {
      fprintf(fp, "%lf ", *(Mat + leni*cols + lenj));
   }

  fprintf (fp, "\n");
 }

fclose (fp);

}


/* Print array */
void print_array (float *vec, int rows, char name[]) {

 int leni;
 FILE *fp;
 fp = fopen(name, "w");

 for (leni =0; leni < rows; leni++) {
      fprintf(fp, "%lf\n", *(vec + leni));
   }

fclose (fp);

}

void print_array1 (int vec[], int rows, char name[]) {

 int leni;
 FILE *fp;
 fp = fopen(name, "w");

 for (leni =0; leni < rows; leni++) {
      fprintf(fp, "%d\n", vec[leni]);
   }

fclose (fp);

}

int main(int argc, char** argv) {

  MPI_Init(&argc, &argv);
  int rank, nprocs; 
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  double start = MPI_Wtime(); 
 
  char *INPUTFILE, *OUTPUTFILE, *OUTPUTFILE1;
  INPUTFILE = argv[1];  
  OUTPUTFILE = argv[4]; 
  OUTPUTFILE1 = argv[5];  
  char *dataset_x = "/data/X";
  char *dataset_y = "/data/y"; 
  char buf[20];

  int nrows;  // A( nrows X ncols )
  int ncols;
  int krows;   // B( krows X ncols )i
  int ycols;
  int ngroups; // number of worker groups
  int nbootE, nbootS; 
  int nMP; 
  int i,j;
  float rndfrctL=0.8, rndfrct=0.8, cvlfrct=0.9;
  int nrnd;
  int bgdOpt = 1;
  int nboot;
 
  float *A, *B, *_X, *Y;

  if (rank == 0) {
 
    if (argc != 10) {
      printf("Usage: %s InputFile nbootS nbootE OutputFile LassoAttributesFile nrnd nMP ngroups testl\n", argv[0]);
      fflush(stdout);
      MPI_Abort(MPI_COMM_WORLD, 1);
    } else {
      nrows = get_rows(INPUTFILE, dataset_x);
      ncols = get_cols(INPUTFILE,dataset_x);
      krows = nrows; 
      ycols = get_cols (INPUTFILE, dataset_y); 
      nbootS = atoi(argv[2]);
      nbootE = atoi(argv[3]); 
      nrnd = atoi(argv[6]);
      nMP = atoi (argv[7]);
      ngroups = atoi(argv[8]); 
      nboot = atoi(argv[9]);
    }

   srand(rank);
 
    /*if (nbootS != nrnd*nbootE) {
        printf ("nrnd is 10! Selection bootstraps should be equal to nrnd*(Estimation bootstraps)");
         fflush(stdout);
         MPI_Abort(MPI_COMM_WORLD, 1);
    }*/
  

    size_t sizeX = (size_t) nrows * (size_t) ncols * sizeof(float);
    size_t sizeY = (size_t) ycols * sizeof(float);  

    size_t sizeA = (size_t) nrows * (size_t) (ncols+1) * sizeof(float);
    size_t sizeB = (size_t) ngroups * (size_t) krows * (size_t) (ncols+1) * sizeof(float);

    printf("Total A: %s\n", readable_fs((float) sizeA, buf));
    printf("Total B: %s\n", readable_fs((float) sizeB, buf));
    printf("Total:   %s\n", readable_fs((float) (sizeA+sizeB), buf));

    printf("A per rank: %s\n", readable_fs( (float) sizeA / (float) nprocs , buf));
    printf("B per rank: %s\n", readable_fs( (float) sizeB / (float) nprocs , buf));

    printf("Num procs: %i\n", nprocs);
    printf("B groups: %i\n\n", ngroups);
    printf("A dimensions: (%i, %i)\n", nrows, ncols+1);
    printf("B dimensions: (%i, %i)\n", krows, ncols+1);
      
    if ( nprocs > nrows ) {
      printf("must have nprocs < nrows \n");
      fflush(stdout);
      MPI_Abort(MPI_COMM_WORLD, 3);
    }

    if ( ngroups > nprocs ) {
      printf("must have ngroups < nprocs \n");
      fflush(stdout);
      MPI_Abort(MPI_COMM_WORLD, 4);
    }   

  }

  MPI_Bcast(&nrows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&ncols, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&krows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&ycols, 1, MPI_INT, 0, MPI_COMM_WORLD); 
  MPI_Bcast(&ngroups, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&nMP, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&nrnd, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&nbootS, 1, MPI_INT, 0, MPI_COMM_WORLD); 
  MPI_Bcast(&nbootE, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&nboot, 1, MPI_INT, 0, MPI_COMM_WORLD);

     
  int local_rows = bin_size_1D(rank, nrows, nprocs);
  double read_t = MPI_Wtime();
  
  size_t sizeX = (size_t) local_rows * (size_t) ncols * sizeof(float);  
  _X = (float *) malloc(sizeX);
  get_matrix (local_rows, ncols, nrows, MPI_COMM_WORLD, rank, _X, dataset_x, INPUTFILE); 


  size_t sizeY = (size_t) local_rows * sizeof(float); 
  Y = (float *)malloc(sizeY); 
  get_array (local_rows, nrows, MPI_COMM_WORLD, rank, Y, dataset_y, INPUTFILE);

  double end_read = MPI_Wtime() - read_t; 
  if (rank == 0) {
    printf("Read time: %f (s)\n", end_read);
  }

  double dist_t = MPI_Wtime(); 

  size_t sizeA = (size_t) local_rows * (size_t) (ncols+1) * sizeof(float);
  A = (float *) malloc(sizeA);
  combine_matrix (_X, Y, A, local_rows, ncols);

  double end_dist = MPI_Wtime() - dist_t; 
  if (rank == 0) {
    printf("Dist time: %f (s)\n", end_dist);
  }
  free(_X);
  free(Y);

  int color = bin_coord_1D(rank, nprocs, ngroups);  
  MPI_Comm comm_g;
  MPI_Comm_split(MPI_COMM_WORLD, color, rank, &comm_g);
  
  int nprocs_g, rank_g;
  MPI_Comm_size(comm_g, &nprocs_g);
  MPI_Comm_rank(comm_g, &rank_g); 

  //int qrows = bin_size_1D(rank_g, krows, nprocs_g);
  int qrows = floor(krows/nprocs_g);  
  int qcols = ncols+1;
  size_t sizeB = (size_t) (ncols+1) * (size_t) qrows * sizeof(float);    
  B = (float *) malloc(sizeB);

  distribute_data (A,  local_rows, qrows, nrows, ncols, krows, B, MPI_COMM_WORLD, comm_g);

  MPI_Barrier(MPI_COMM_WORLD); 
  /******************************************


                MODEL SELECTION


  *******************************************/

  /************************
  *** COARSE SWEEP ********
  ************************/

  float *B_select, *R2m; 
  
  if (rank == 0) {
	B_select = (float *) malloc((nMP*nbootS) * ncols * sizeof(float));
	R2m = (float *) malloc ((nMP*nbootS) * sizeof(float));
  }

  float *lamb;
  lamb = (float *)malloc (nbootS * sizeof(float)); 
  double lasso1_t, lasso1_end, lasso2_end;
  int train_rows = floor(rndfrctL * qrows);
  float *X_train, *y_train, *X_test, *ymean;

  X_train  =(float *) malloc (train_rows * ncols * sizeof(float));
  y_train = (float *) malloc (train_rows * sizeof(float));
  X_test = (float *) malloc ((qrows - train_rows) * ncols * sizeof(float));
  ymean = (float *)malloc((qrows - train_rows) * sizeof(float));
  get_train (B, X_train, y_train, X_test, ymean, qrows, ncols, train_rows);

  free(B);  

  double select_t = MPI_Wtime();

  model_selection (X_train, y_train, X_test, ymean, lamb, B_select, R2m, &lasso1_end, &lasso2_end, nbootS, nboot, train_rows, ncols, qrows, nMP, MPI_COMM_WORLD, comm_g);

  double sel_end = MPI_Wtime() - select_t;
   if (rank == 0) {
        printf("Lasso1 time : %f (s)\n", lasso1_end); 
	printf("Lasso2 time : %f (s)\n", lasso2_end);
	printf("Selection time : %f (s)\n", sel_end);
   } 

  free(X_train);
  free(y_train);
  free(X_test);
  free(ymean); 

/************************
******** SUPPORT ********
************************/

  float *sprt;
  sprt = (float *) malloc (nMP * ncols * sizeof(float));
  double saveTime, end_saveTime; 

  if (rank == 0)
        get_support (B_select, sprt, nMP, nbootS, ncols);

  MPI_Bcast(sprt, nMP * ncols, MPI_FLOAT, 0, MPI_COMM_WORLD);  

  ngroups = nrnd;


  int color1 = bin_coord_1D(rank, nprocs, ngroups); 
  MPI_Comm comm_nrnd;
  MPI_Comm_split(MPI_COMM_WORLD, color1, rank, &comm_nrnd);

  int nprocs_nrnd, rank_nrnd;
  MPI_Comm_size(comm_nrnd, &nprocs_nrnd);
  MPI_Comm_rank(comm_nrnd, &rank_nrnd); 
  

  /* cross validation root ranks for parallel writing */ 

  MPI_Group world_group;
  MPI_Comm_group (MPI_COMM_WORLD, &world_group);
 
  int root[nrnd];

  if (rank == 0) {
  	int ss=0;
  	for (i=0; i<nrnd; i++) {
        	root[i] = ss;
        	ss+=nprocs_nrnd;
  	}
  }

  MPI_Bcast(root, nrnd, MPI_INT, 0, MPI_COMM_WORLD);
 
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Group root_nrnd;
  MPI_Group_incl(world_group, nrnd, root, &root_nrnd);

  MPI_Comm comm_ROOT_NRND;
  MPI_Comm_create_group(MPI_COMM_WORLD, root_nrnd, 1, &comm_ROOT_NRND);
 
  int rank_roots;
  if (MPI_COMM_NULL != comm_ROOT_NRND) 
  	MPI_Comm_rank(comm_ROOT_NRND, &rank_roots); 
 
  MPI_Barrier (MPI_COMM_WORLD); 
  if (MPI_COMM_NULL != comm_ROOT_NRND) {
        if (rank_roots != 0) {
		B_select = (float *) malloc ((nbootS*nMP) * ncols * sizeof(float)); 
		R2m = (float *)malloc ((nbootS*nMP) * sizeof(float));
	}
 
	
	MPI_Bcast(B_select, nbootS * nMP * ncols, MPI_FLOAT, 0, comm_ROOT_NRND); 
  	MPI_Bcast(R2m, nbootS * nMP, MPI_FLOAT, 0, comm_ROOT_NRND);

	if (rank == 0)
		saveTime = MPI_Wtime();
  
	write_selections(OUTPUTFILE1, B_select, R2m, lamb, sprt, nbootS, nMP, ncols, comm_ROOT_NRND);

	if (rank == 0)
		end_saveTime = MPI_Wtime() - saveTime; 

  }

 /* free intermediate malloc's */ 

 if (MPI_COMM_NULL != comm_ROOT_NRND) {

	free(B_select);
	free(R2m); 
  }

  free(lamb);
 
 

  /* Redistribute data for Cross validation */ 
  
  double redis_s, redis_end;
  krows=nrows;
  //qrows = bin_size_1D(rank_nrnd, krows, nprocs_nrnd); 
  qrows = floor(krows/nprocs_g);

  int L = floor(cvlfrct * qrows);
  int train = floor(rndfrct * L);
  float *Bgd, *rsd, *R2, *bic;
  Bgd = (float *)malloc(ncols * sizeof(float));
  rsd = (float *)malloc((qrows - L) * sizeof(float));
  R2 = (float *) malloc (1 * sizeof(float));
  bic = (float *) malloc (1 * sizeof(float));
  double est_t, est_end, end_ols;  
  double t_start, t_end, tmax; 


 
/******************************************


        MODEL ESTIMATION


*******************************************/

  float *B_per_core;
  B_per_core = (float *) malloc (qrows * (ncols+1) * sizeof(float));
  t_start = MPI_Wtime();  
  distribute_data (A,  local_rows, qrows, nrows, ncols, krows, B_per_core, MPI_COMM_WORLD, comm_g);
  t_end += MPI_Wtime() - t_start;
  MPI_Reduce(&t_end, &tmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
 
  free(A);

  float *X_L, *y_L, *X_T, *y_T;
  X_L = (float *) malloc(L * ncols * sizeof(float));
  y_L = (float *) malloc (L *sizeof(float));
  X_T = (float *) malloc((qrows-L) * ncols * sizeof(float));
  y_T = (float *) malloc ((qrows-L) *sizeof(float));

  get_estimate2(B_per_core, X_L, y_L, X_T, y_T, qrows, ncols, L);
  free(B_per_core);    

  est_t = MPI_Wtime();
  model_estimate(X_L, y_L, X_T, y_T, sprt, L, train, qrows, ncols, nrows, krows, nMP, nbootE, nrnd, &end_ols, Bgd, rsd, R2, bic, root, MPI_COMM_WORLD, comm_g, comm_nrnd);
  est_end = MPI_Wtime() - est_t;

  if (rank ==0) {
  	printf("Estimation time : %f (s)\n", est_end);
  }

  MPI_Barrier (MPI_COMM_WORLD);
 
  free(X_L);
  free(y_L);
  free(X_T); 
  free(y_T); 
  free(sprt);


 /*write data into output file*/ 
   saveTime = MPI_Wtime();  
   if (MPI_COMM_NULL != comm_ROOT_NRND)
	write_output (OUTPUTFILE, nrnd, ncols, Bgd, R2, (qrows-L), rsd, bic, comm_ROOT_NRND ); 


  /*free all the variables now*/

   free(Bgd);
   free(rsd);
   free(R2);
   free(bic);


   if (rank == 0) {
  	end_saveTime += MPI_Wtime() - saveTime; 
     	printf ("Save time: %f (s)\n", end_saveTime);
	double end = MPI_Wtime() - start; 
	
        printf("\nUnion of Intersections analysis completed");
        printf("\n---------------------------------------");
	printf("\nLasso results stored in %s", OUTPUTFILE1);
        printf("\n\t-Bagging Option: 							%d", bgdOpt);
        printf("\n\t-data fraction for training during each bagged-OLS randomization: 	%.1f", cvlfrct);
        printf("\n\t-data fraction used for linear regression fitting: 			%.1f", rndfrct);
        printf("\n\t-fraction of data used for Lasso fitting:	 			%.1f", rndfrctL);
        printf("\n\t-Selection bootstraps: 							%d", nbootS);
        printf("\n\t-Evaluation bootstraps: 						%d", nbootE);
        printf("\n\t-number of processes to compute ADMM 					%d", nprocs_g);
	printf("\n\t-number of samples 							%d", nrows);
 	printf("\n\t-number of features: 							%d", ncols);

        printf("\nUOI Times (s) ");
    	printf("\n---------------------------------------");
    	printf("\nResults stored in %s", OUTPUTFILE);
	printf("\n\t-load time: %.4f", end_read);
    	printf("\n\t-comm time: %.4f", tmax);
    	printf("\n\t-dist time: %.4f", end_dist);
	printf("\n\t-redis time: %.4f", redis_end);
	printf("\n\t-selection time: %4f", sel_end); 
    	printf("\n\t-comp time: %.4f", sel_end+est_end);
    	printf("\n\t\t-las1 time: %.4f", lasso1_end);
    	printf("\n\t\t-las2 time: %.4f", lasso2_end);
    	printf("\n\t\t-ols  time: %.4f", end_ols);
    	printf("\n\t-est time: %.4f\n", est_end); 
    	printf("\n\t-save time: %.4f\n", end_saveTime);
	printf("\n----------------------------");
	printf("\n\t-Total time: %.4f\n", end);

 

  }

  MPI_Finalize();
  return 0;
}
