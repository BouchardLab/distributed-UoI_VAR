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
#include "lasso_admm_MPI.h"
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

char* readable_fs(double size/*in bytes*/, char *buf) {
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
void print_mat (double *Mat, int rows, int cols, char name[]) {

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
void print_array (double *vec, int rows, char name[]) {

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
  int nMP; // number of lambda parameters
  int nbootE, nbootS; 

  int i,j;
  float rndfrctL=0.8, rndfrct=0.8, cvlfrct=0.9;
  int nrnd;
  int bgdOpt = 1;
  int nEP=0; 
  double *A, *B, *_X, *Y;

  srand(rank);

  if (rank == 0) {
 
    if (argc != 7) {
      printf("Usage: %s InputFile nbootS nbootE OutputFile LassoAttributesFile nrnd\n", argv[0]);
      fflush(stdout);
      MPI_Abort(MPI_COMM_WORLD, 1);
    } else {
      nrows = get_rows(INPUTFILE, dataset_x);
      ncols = get_cols(INPUTFILE,dataset_x);
      krows = nrows; 
      ycols = get_cols (INPUTFILE, dataset_y); 
      nbootS = atoi(argv[2]);
      nbootE = atoi(argv[3]);
      ngroups = 1;  
      nrnd = atoi(argv[6]);
      nEP = nrnd * nbootE;
    }

 
    /*if (nbootS != nrnd*nbootE) {
        printf ("nrnd is 10! Selection bootstraps should be equal to nrnd*(Estimation bootstraps)");
         fflush(stdout);
         MPI_Abort(MPI_COMM_WORLD, 1);
    }*/
  

    size_t sizeX = (size_t) nrows * (size_t) ncols * sizeof(double);
    size_t sizeY = (size_t) ycols * sizeof(double);  

    size_t sizeA = (size_t) nrows * (size_t) (ncols+1) * sizeof(double);
    size_t sizeB = (size_t) ngroups * (size_t) krows * (size_t) (ncols+1) * sizeof(double);

    printf("Total A: %s\n", readable_fs((double) sizeA, buf));
    printf("Total B: %s\n", readable_fs((double) sizeB, buf));
    printf("Total:   %s\n", readable_fs((double) (sizeA+sizeB), buf));

    printf("A per rank: %s\n", readable_fs( (double) sizeA / (double) nprocs , buf));
    printf("B per rank: %s\n", readable_fs( (double) sizeB / (double) nprocs , buf));

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
  MPI_Bcast(&nEP, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&nrnd, 1, MPI_INT, 0, MPI_COMM_WORLD); 
  MPI_Bcast(&nbootE, 1, MPI_INT, 0, MPI_COMM_WORLD);

     
  int local_rows = bin_size_1D(rank, nrows, nprocs);
  double read_t = MPI_Wtime(); 
  
  size_t sizeX = (size_t) local_rows * (size_t) ncols * sizeof(double);  
  _X = (double *) malloc(sizeX);
  get_matrix (local_rows, ncols, nrows, MPI_COMM_WORLD, rank, _X, dataset_x, INPUTFILE); 


  size_t sizeY = (size_t) local_rows * sizeof(double); 
  Y = (double *)malloc(sizeY); 
  get_array (local_rows, nrows, MPI_COMM_WORLD, rank, Y, dataset_y, INPUTFILE);

  double end_read = MPI_Wtime() - read_t; 
  if (rank == 0) {
    printf("Read time: %f (s)\n", end_read);
  }

  double dist_t = MPI_Wtime(); 

  size_t sizeA = (size_t) local_rows * (size_t) (ncols+1) * sizeof(double);
  A = (double *) malloc(sizeA);
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

  int qrows = bin_size_1D(rank_g, krows, nprocs_g);
  int qcols = ncols+1;
  size_t sizeB = (size_t) (ncols+1) * (size_t) qrows * sizeof(double);    
  B = (double *) malloc(sizeB);

  distribute_data (A,  local_rows, qrows, nrows, ncols, krows, B, MPI_COMM_WORLD, comm_g);

  
  /******************************************


                MODEL SELECTION


  *******************************************/

  /************************
  *** COARSE SWEEP ********
  ************************/

  double *B_coarse, *R2m0; 
  
  if (rank == 0) {
	B_coarse = (double *) malloc(nbootS * ncols * sizeof(double));
	R2m0 = (double *) malloc (nbootS * sizeof(double));
  }

  if (rank == 0)
	printf("passed B_coarse malloc\n");

  double *lamb0;
  lamb0 = logspace (-3, 3, nbootS);
  double my_lamb0; 
  double lasso1_t, lasso1_end;
  int train_rows = round(rndfrctL * qrows);

  double *X_train, *y_train, *X_test, *ymean;

  X_train  =(double *) malloc (train_rows * ncols * sizeof(double));
  y_train = (double *) malloc (train_rows * sizeof(double));
  X_test = (double *) malloc ((qrows - train_rows) * ncols * sizeof(double));
  ymean = (double *)malloc((qrows - train_rows) * sizeof(double));
  get_train (B, X_train, y_train, X_test, ymean, qrows, ncols, train_rows);

  free(B); 

  double t_start, t_end, tmax;  
  int ci, di; 
  for (ci=0; ci<nbootS; ci++) { 

	my_lamb0 = lamb0[ci];

	if (ci != 0)
		randomize(X_train, y_train, X_test, ymean, train_rows, ncols, qrows);
 
  	double *my_B0; 
  	my_B0 = (double *)malloc(ncols * sizeof(double));
 
  	lasso1_t = MPI_Wtime();  
  	my_B0 = lasso_admm(X_train, train_rows, ncols, y_train, my_lamb0, comm_g); 

  	if (rank == 0) {
		double *yhat; 
  		yhat = (double *)malloc((qrows-train_rows) * sizeof(double));
		get_test(X_test, ymean, qrows-train_rows, ncols, my_B0, yhat); 
		double r;
		r = pearson (yhat, ymean, qrows-train_rows);
  		double my_R2m0;
		my_R2m0 = r*r;  
		printf("my_R2m0 = %f\n", my_R2m0);
 		R2m0[ci] = my_R2m0;
		memcpy(B_coarse + ci * ncols, my_B0, sizeof(double) * ncols); 
 	} 
  	lasso1_end += MPI_Wtime() - lasso1_t;
  }
  
  if (rank == 0) {
 	printf("Lasso1 time : %f (s)\n\n", lasso1_end);    
  } 

 
  /************************
  *** DENSE SWEEP ********
  ************************/

  double lasso2_t, lasso2_end; 
  double *lambL;
  float v,dv;

  if (rank == 0) { 
  	v = dense_sweep (R2m0, lamb0, nbootS); 
  	dv = get_dv (v); 
  }

  if (rank == 0)
  	printf("passed dense_sweep\n"); 
 
  MPI_Bcast(&v, 1, MPI_FLOAT, 0, MPI_COMM_WORLD); 
  MPI_Bcast(&dv, 1, MPI_FLOAT, 0, MPI_COMM_WORLD); 
  lambL = linspace(v-5*dv,v+5*dv,nbootS);  

  if (rank == 0)
  	printf("passed lambL\n"); 
 
  double my_lambL;
  double *B_dense, *R2m;

  if (rank == 0) {
        B_dense = (double *)malloc(nbootS * ncols * sizeof(double));
	R2m = (double *) malloc (nbootS * sizeof(double));
   }

  //abort(); 

  for (di=0; di<nbootS; di++) {
	my_lambL = lambL[di];  

	if (di != 0)
                randomize(X_train, y_train, X_test, ymean, train_rows, ncols, qrows);

  	double *my_B;
  	my_B = (double *)malloc(ncols * sizeof(double));
	
	lasso2_t = MPI_Wtime();
 	my_B = lasso_admm(X_train, train_rows, ncols,  y_train, my_lambL, comm_g);

	if (rank == 0)
		printf("passed lasso admm\n"); 

  	if (rank == 0) {
        	double *yhat_d; 
		yhat_d = (double *)malloc((qrows-train_rows) * sizeof(double));
		get_test(X_test, ymean, qrows-train_rows, ncols, my_B, yhat_d);
        	double r = pearson (yhat_d, ymean, qrows-train_rows);
       	 	double my_R2m;
		my_R2m  = r*r;
		printf("my_R2m = %f\n", my_R2m); 
		R2m[di] = my_R2m;
		memcpy(B_dense + di * ncols, my_B, sizeof(double) * ncols); 
 	}
  	lasso2_end += MPI_Wtime() - lasso2_t; 
  }

  if (rank == 0)
       	printf("Lasso2 time : %f (s)\n", lasso2_end);

/************************
******** SUPPORT ********
************************/

  double *sprt;
  sprt = (double *) malloc (nbootS * ncols * sizeof(double));
  double saveTime, end_saveTime; 

  if (rank == 0)
        get_support (B_dense, sprt, nbootS, nMP, ncols);


  MPI_Bcast(sprt, nbootS * ncols, MPI_DOUBLE, 0, MPI_COMM_WORLD); 
  
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
  int ss=0;
  for (i=0; i<nrnd; i++) {
        root[i] = ss;
        ss+=nprocs_nrnd;
  }
  

  MPI_Group root_nrnd;
  MPI_Group_incl(world_group, nrnd, root, &root_nrnd);

  MPI_Comm comm_ROOT_NRND;
  MPI_Comm_create_group(MPI_COMM_WORLD, root_nrnd, 1, &comm_ROOT_NRND);
 
  int rank_roots; 
  MPI_Comm_rank(comm_ROOT_NRND, &rank_roots); 
 
  MPI_Barrier (MPI_COMM_WORLD); 
  if (MPI_COMM_NULL != comm_ROOT_NRND) {
        if (rank_roots != 0) {
		B_coarse = (double *) malloc (nbootS * ncols * sizeof(double));
		B_dense = (double *) malloc (nbootS * ncols * sizeof(double)); 
		R2m0 = (double *)malloc (nbootS * sizeof(double));
		R2m = (double *)malloc (nbootS * sizeof(double));
	}
 
	MPI_Bcast(B_coarse, nbootS * ncols, MPI_DOUBLE, 0, comm_ROOT_NRND);
	MPI_Bcast(B_dense, nbootS * ncols, MPI_DOUBLE, 0, comm_ROOT_NRND); 
 	MPI_Bcast(R2m0, nbootS, MPI_DOUBLE, 0, comm_ROOT_NRND);
  	MPI_Bcast(R2m, nbootS, MPI_DOUBLE, 0, comm_ROOT_NRND);

	if (rank == 0)
		saveTime = MPI_Wtime();
  
	write_selections(OUTPUTFILE1, B_coarse, B_dense, R2m0, R2m, lamb0, lambL, sprt, nbootS, ncols, comm_ROOT_NRND);

	if (rank == 0)
		end_saveTime = MPI_Wtime() - saveTime; 

  }

 /* free intermediate malloc's */ 

 if (MPI_COMM_NULL != comm_ROOT_NRND) {

	free(B_coarse);
	free(B_dense);
	free(R2m0);
	free(R2m); 
  }

  free(lamb0);
  free(lambL); 


  /* Redistribute data for Cross validation */ 
  
  double redis_s, redis_end;
  krows=nrows;
  qrows = bin_size_1D(rank_nrnd, krows, nprocs_nrnd); 

  int L = floor(cvlfrct * qrows);
  int train = floor(rndfrct * L);
  double *Bgd, *rsd, *R2, *bic;
  Bgd = (double *)malloc(ncols * sizeof(double));
  rsd = (double *)malloc((qrows - L) * sizeof(double));
  R2 = (double *) malloc ( 1 * sizeof(double));
  bic = (double *) malloc (1 * sizeof(double));
  double est_t, est_end, end_ols;  
 
/******************************************


        MODEL ESTIMATION


*******************************************/

  double *B_per_core;
  B_per_core = (double *) malloc (qrows * (ncols+1) * sizeof(double));
  t_start = MPI_Wtime();  
  distribute_data (A,  local_rows, qrows, nrows, ncols, krows, B_per_core, MPI_COMM_WORLD, comm_nrnd);
  t_end += MPI_Wtime() - t_start;
  MPI_Reduce(&t_end, &tmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
 
  free(A);

  double *X_tr, *y_tr, *X_te, *y_te, *X_T, *y_T;
  X_tr =(double *) malloc (train * ncols * sizeof(double));
  y_tr = (double *) malloc ( train * sizeof(double));
  X_te = (double *) malloc ((L  - train) * ncols * sizeof(double));
  y_te = (double *)malloc((L - train) * sizeof(double));
  X_T = (double *) malloc((qrows-L) * ncols * sizeof(double));
  y_T = (double *) malloc ((qrows-L) *sizeof(double));


  get_estimate(B_per_core, X_tr, y_tr, X_te, y_te, X_T, y_T, qrows, ncols, L, train);
  free(B_per_core);   

  est_t = MPI_Wtime();
  model_estimate(X_tr, y_tr, X_te, y_te, X_T, y_T, sprt, L, train, qrows, ncols, nrows, krows, nbootS, nbootE, nrnd, &end_ols, Bgd, rsd, R2,bic, MPI_COMM_WORLD, comm_nrnd);
  est_end = MPI_Wtime() - est_t;

  if (rank ==0) {
  	printf("Estimation time : %f (s)\n", est_end);
  }

  MPI_Barrier (MPI_COMM_WORLD);
 
  free(X_tr);
  free(y_tr);
  free(X_te);
  free(y_te);
  free(X_T); 
  free(y_T); 


 /*write data into output file*/ 
   saveTime = MPI_Wtime();  
   if (MPI_COMM_NULL != comm_ROOT_NRND)
	write_output (OUTPUTFILE, nrnd, ncols, Bgd, R2, (qrows-L), rsd, bic, comm_ROOT_NRND ); 

   if (rank == 0) {
  	end_saveTime += MPI_Wtime() - saveTime; 
     	printf ("Save time: %f (s)\n", end_saveTime);

	
        printf("\nUnion of Intersections analysis completed");
        printf("\n---------------------------------------");
        printf("\n\t-Bagging Option: 							%d", bgdOpt);
        printf("\n\t-Number of randomizations for bagged estimates: 			%d", nrnd);
        printf("\n\t-data fraction for training during each bagged-OLS randomization: 	%.1f", cvlfrct);
        printf("\n\t-data fraction used for linear regression fitting: 			%.1f", rndfrct);
        printf("\n\t-fraction of data used for Lasso fitting:	 			%.1f", rndfrctL);
        printf("\n\t-Selection bootstraps: 							%d", nbootS);
        printf("\n\t-Evaluation bootstraps: 						%d", nbootE*nrnd);
        printf("\n\t-number of processes to compute ADMM 					%d", nprocs_g);
	printf("\n\t-number of samples 							%d", nrows);
 	printf("\n\t-number of features: 							%d", ncols);

        printf("\nUOI Times");
    	printf("\n---------------------------------------");
    	printf("\nResults stored in %s", OUTPUTFILE);
	printf("\n\t-load time: %.4f", end_read);
    	printf("\n\t-comm time: %.4f", tmax);
    	printf("\n\t-dist time: %.4f", end_dist);
	printf("\n\t-redis time: %.4f", redis_end); 
    	printf("\n\t-comp time: %.4f", lasso1_end+lasso2_end+est_end);
    	printf("\n\t\t-las1 time: %.4f", lasso1_end);
    	printf("\n\t\t-las2 time: %.4f", lasso2_end);
    	printf("\n\t\t-ols  time: %.4f", end_ols);
    	printf("\n\t-est time: %.4f\n", est_end); 
    	printf("\n\t-save time: %.4f\n", end_saveTime);
	printf("\n----------------------------");
	printf("\n\t-Total time: %.4f\n", end_saveTime+lasso1_end+lasso2_end+est_end+end_dist+tmax+end_read+redis_end);

 

  }

  MPI_Finalize();
  return 0;
}
