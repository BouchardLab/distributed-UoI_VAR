#include <iostream>
#include <mpi.h>
#include <fstream>
#include <vector>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <assert.h>
#include "bins.h"
#define EIGEN_USE_MKL_ALL
#include <eigen3/Eigen/Dense>
#include "manage-data.h"
#include "matrix-operations.h"
//#include "model_estimate.h"
//#include "model_selection.h"
#include "lasso.h"
#include "var-distribute-data.h"
#include "var_kron.h"

using namespace Eigen;
using namespace std; 

float not_NaN (float x) {if (!isnan(x)) return x; else return 0;}


template<typename M>
M load_csv (const std::string & path) {
    std::ifstream indata;
    indata.open(path);
    std::string line;
    std::vector<float> values;
    uint rows = 0;
    while (std::getline(indata, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
            values.push_back(std::stod(cell));
        }
        ++rows;
    }
    return Map<const Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime, RowMajor> >(values.data(), rows, values.size()/rows);
}

MatrixXf stack_Z (float *mat_f, int rows, int cols, int D, int rank) { 

     Map<Matrix<float, Dynamic, Dynamic, RowMajor> > mat(mat_f, rows, cols);
     
     //if (rank == 0) {
     // cout << "rows " << rows << "\tcols " << cols << "\tmat.rows " << mat.rows() << "\tmat.cols " << mat.cols() << endl;
     //}

  
     MatrixXf Out(mat.rows()-D, D*mat.cols());
     
    for(int i=0; i<mat.rows()-D; i++) {
	for(int j=0; j<D;j++) {
	   Out.row(i).segment((j*mat.cols()), mat.cols()) = mat.row((i+j)+D);
	}
    }
     	
   return Out; 	

}

int main(int argc, char** argv) {

  MPI_Init(&argc, &argv);
  int rank, nprocs;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  double start = MPI_Wtime();

  char *INPUTFILE, *OUTPUTFILE, *OUTPUTFILE1;
  INPUTFILE = argv[1];
  OUTPUTFILE = argv[7];
  OUTPUTFILE1 = argv[8];
  char *dataset_simdata = "/data/sim_data";
  //char *dataset_y = "/data/y";
  char buf[20];
  int L, N, p, D, nlamb, B1, B2, bgdopt=1;
  double start_time, end_time;  
  //atrixXf inputdata; 
  if (rank == 0) {
	start_time = MPI_Wtime(); 
	if (argc != 9) {
		printf("Usage: %s InputFile L D nlamb B1 B2 OutFile1 OutFile2\n", argv[0]);
        	fflush(stdout);
        	MPI_Abort(MPI_COMM_WORLD, 1);
   	} else {
		N = get_rows(INPUTFILE, dataset_simdata);
      		p = get_cols(INPUTFILE,dataset_simdata);
		
		//cout << "N = " << N << "\t p = " << p << endl ;	
	
      		L = atoi(argv[2]);
      		D = atoi(argv[3]); 
		nlamb = atoi(argv[4]);
		B1 = atoi(argv[5]);
  		B2 = atoi(argv[6]); 
   	}	

  	//inputdata = load_csv<MatrixXf>(INPUTFILE); /* substitue this with hdf5 reads parallel reads*/ 

	/*ofstream myfile121 ("data/Input.dat");
        if (myfile121.is_open())
        {
                myfile121 << inputdata;
                myfile121.close();
        }

	//MPI_Abort(MPI_COMM_WORLD, 2);

  	N = inputdata.rows();
  	p = inputdata.cols();*/ 
  
  	if ( nprocs > N ) {
     		printf("must have nprocs < nrows \n");
      		fflush(stdout);
      		MPI_Abort(MPI_COMM_WORLD, 1);
    	}

  }
  MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&L, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&D, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&nlamb, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&B1, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&B2, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&p, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int ngroups = 1; 

 /* create groups for bootstraps */
  int color = bin_coord_1D(rank, nprocs, ngroups);
  MPI_Comm comm_g;
  MPI_Comm_split(MPI_COMM_WORLD, color, rank, &comm_g);

  int nprocs_g, rank_g;
  MPI_Comm_size(comm_g, &nprocs_g);
  MPI_Comm_rank(comm_g, &rank_g);  

  /* Distribute the matrix across the cores from 0 : 
   * Remove the following step once the hdf5 input data 
   * is ready.
   */


  /*Create root ranks for writing*/
  MPI_Group world_group;
  MPI_Comm_group (MPI_COMM_WORLD, &world_group);

  int root[ngroups]; 

  if (rank == 0) {
        int ss=0;
        for (int i=0; i<ngroups; i++) {
                root[i] = ss;
                ss+=nprocs_g;
        }
  }

  MPI_Bcast(root, ngroups, MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Group root_ngroups;
  MPI_Group_incl(world_group, ngroups, root, &root_ngroups);

  MPI_Comm comm_ROOT;
  MPI_Comm_create_group(MPI_COMM_WORLD, root_ngroups, 1, &comm_ROOT);

  int rank_roots;
  if (MPI_COMM_NULL != comm_ROOT)
        MPI_Comm_rank(comm_ROOT, &rank_roots);

 
  int qrows = bin_size_1D(rank_g, N, nprocs_g);
  //MatrixXf data(qrows, p);

  double read_t = MPI_Wtime();

  size_t sizeX = (size_t) qrows * (size_t) p * sizeof(float);
  float *data_f; 
  data_f = (float *) malloc(sizeX);
  get_matrix (qrows, p, N, MPI_COMM_WORLD, rank, data_f, dataset_simdata, INPUTFILE);

  //Map<Matrix<float, Dynamic, Dynamic, RowMajor> > data(_temp, qrows, p); 
  //free(_temp); 

  double end_read = MPI_Wtime() - read_t;
  if (rank == 0) {
    cout << "Read time: " << end_read << " (s)\n"; 
  }

  /*{
    int sendcounts[nprocs_g];
    int displs[nprocs_g];

    for (int i=0; i<nprocs_g; i++) {
       int ubound;
       bin_range_1D(i, N, nprocs_g, &displs[i], &ubound);
       displs[i]*=p; 
       sendcounts[i] = bin_size_1D(i, N, nprocs_g) * p;
    } 
    MPI_Scatterv(inputdata.data(), sendcounts, displs, MPI_FLOAT, data.data(), qrows*p, MPI_FLOAT, 0, comm_g); 
  }*/

  int size_s = (N-L+2) * L; 
  //int q_rows = bin_size_1D(rank_g, p, nprocs_g);
 
  //float *data_f; 
  //data_f = (float *)malloc(data.rows() * data.cols()* sizeof(float));
  //Map<Matrix<float, Dynamic, Dynamic, RowMajor> >(data_f, data.rows(), data.cols()) = data;  

#ifdef DEBUG
  if (rank == 0) {
	Map<Matrix<float, Dynamic, Dynamic, RowMajor> > temp(data_f, qrows, p);   
	ofstream myfile1 ("data/data_f.dat");
        if (myfile1.is_open())
        {
                myfile1 << temp; 
                myfile1.close();
        }

	MPI_Abort(MPI_COMM_WORLD, 2);
  }
#endif

  MPI_Barrier(MPI_COMM_WORLD);

  float *lamb0;
  lamb0 = logspace(-2, 1, nlamb);

  float *bdata_f, *Y, *Z_mx_e, *Z_stacked, *X, *my_B0;  
  MatrixXf B0; 


  /* Model Selection -- Intersection step 
   * This implementation has only one sweep -- TODO: Trevor is still working on Coarse and Dense sweeps 
   */ 

  double allred_1, allred_2;

  double start_rand, end_rand, start_Zmx, end_Zmx, start_kron, end_kron; 
  for(int k=0; k<B1; k++) {  

	start_rand = MPI_Wtime(); 
  	//float *bdata_f;
  	bdata_f = (float *)malloc(qrows * p * sizeof(float));

	//if (rank == 0)
	 //	cout << "B1 data.rows() = " << data.rows() << " q_rows = " << qrows << "\t p = " << p << "\t N = " << N << "\t size_s = " << size_s <<  endl;     
 	var_distribute_data(data_f, qrows, qrows, N, p, size_s, bdata_f, L, D, MPI_COMM_WORLD, comm_g );

#ifdef DEBUG 
	if (rank == 0) {
        	ofstream myfile2("data/bdata_f.dat");
		Map<Matrix<float, Dynamic, Dynamic, RowMajor> >bdata(bdata_f, qrows, p); 
        	if (myfile2.is_open())
        	{
                	myfile2 << bdata;  
                	myfile2.close();
        	}

  	}
#endif

	end_rand += MPI_Wtime() - start_rand;
 	start_Zmx = MPI_Wtime(); 
  	//float *Y;
	int y_rows= bin_size_1D(rank_g, N-D, nprocs_g); 
  	Y = (float *)malloc(y_rows*p*sizeof(float)); 
  	var_vectorize_response(bdata_f, qrows, y_rows, N, p, Y, L, D, MPI_COMM_WORLD, comm_g); 

#ifdef DEBUG 
	if (rank == 1) {
                ofstream myfile3("data/Y.dat");
                Map<VectorXf> Y_print(Y, y_rows*p);
                if (myfile3.is_open())
                {
                        myfile3 << Y_print;
                        myfile3.close();
                }
        //MPI_Abort(MPI_COMM_WORLD, 2);

        }

#endif 
  	/* Generate Z_mx matrix */ 

  	int z_rows = bin_size_1D(rank_g, (N-D)*D, nprocs_g); 
  	//float *Z_mx_e;
  	Z_mx_e = (float *)malloc(z_rows * p * sizeof(float)); 

  	var_generate_Z(bdata_f, qrows, z_rows, N, p, Z_mx_e, D, MPI_COMM_WORLD, comm_g);
  	free(bdata_f); 
  	MatrixXf Z_mx;
  	Z_mx = stack_Z(Z_mx_e, z_rows, p, D, rank); 

#ifdef DEBUG

	if (rank == 0) {
                ofstream myfile4("data/Z_mx.dat"); 
                if (myfile4.is_open())
                {
                        myfile4 << Z_mx;
                        myfile4.close();
                }

        } 
#endif

  	free(Z_mx_e);    

 	end_Zmx += MPI_Wtime() - start_Zmx;
	start_kron = MPI_Wtime(); 

 	/* do a distributed kronecker product on Z_mx with I(p,p) to get X*/
  	/* this implementation of the Kronecker product is not an actual product
   	* This implementation is just a bookkeeping technique using MPI_Get 
   	* via one-sided communication 
   	*/
  
   	//float *Z_stacked;
   	Z_stacked = (float *)malloc(Z_mx.rows() * Z_mx.cols() * sizeof(float)); 
   	Map<Matrix<float, Dynamic, Dynamic, RowMajor> >(Z_stacked, Z_mx.rows(), Z_mx.cols()) = Z_mx; 

   	//int kron_rows = bin_size_1D(rank_g, (N-D)*p, nprocs_g);
	int kron_rows = y_rows * p; 
	//float *X; 
   	X = var_kron (Z_stacked, Z_mx.rows(), kron_rows, N, p, D, MPI_COMM_WORLD, comm_g);       
   	free(Z_stacked);
	end_kron += MPI_Wtime() - start_kron; 

	/*if (rank == 1) {
                ofstream myfile5("data/X.dat");
		Map<Matrix<float, Dynamic, Dynamic, RowMajor> >X_print(X, kron_rows, p*p);
                if (myfile5.is_open())
                {
                        myfile5 << X_print;
                        myfile5.close();
                }
        }*/

	MPI_Barrier(MPI_COMM_WORLD);

	//float *my_B0;
	//MatrixXf B0; 
	if (rank_g == 0) B0 = MatrixXf::Zero(B1*nlamb, D*p*p);  
  	for (int nMP = 0; nMP < nlamb; nMP++) {
   		double my_lamb = lamb0[nMP];  
		my_B0 = lasso(X, kron_rows, D*p*p , Y, my_lamb, comm_g, &allred_1);
 		if (rank_g == 0) {
		   Map<VectorXf> my_B(my_B0, D*p*p);
		   B0.row((k*nlamb)+nMP) = my_B;
		}


	 	/*if (rank == 0 && nMP == nlamb/2) {
                	ofstream myfile6("data/my_B0.dat");
               	 	Map<VectorXf> my_B(my_B0, D*p*p);
                	if (myfile6.is_open())
                	{
                        	myfile6 << my_B;
                        	myfile6.close();
                	}

        	}*/
		free(my_B0); 
	}

	/* free memory for the current bootstrap */ 
	free(Y);
	free(X); 	
   }

  /* Create family of supports */ 

  float *sprt;
  sprt = (float *) malloc (nlamb * D*p*p * sizeof(float));
  double saveTime, end_saveTime;

  MPI_Barrier(MPI_COMM_WORLD); 

#ifdef DEBUG

  if (rank == 0) {
  	ofstream myfile7("data/B0.dat"); 
        if (myfile7.is_open())
        {
        	myfile7 << B0;
                myfile7.close();
        }
	//MPI_Abort(MPI_COMM_WORLD, 2);

  }

#endif

  if (rank == 0) 
        get_support (B0, sprt, nlamb, B1, D*p*p);


  MPI_Bcast(sprt, nlamb * D*p*p, MPI_FLOAT, 0, MPI_COMM_WORLD);

  double start_savetime, end_savetime;

  if (rank == 0)
	start_savetime = MPI_Wtime(); 

  if (MPI_COMM_NULL != comm_ROOT) {
	float R2m[nlamb*B1];
    	float *B_select;
	B_select = (float *)malloc(B1*nlamb * D*p*p * sizeof(float));
	Map<Matrix<float, Dynamic, Dynamic, RowMajor> >(B_select, B0.rows(), B0.cols()) = B0; 
	
	write_selections(OUTPUTFILE1, B_select, R2m, lamb0, sprt, B1, nlamb, D*p*p, comm_ROOT);

       	free(B_select);
  }

  if (rank == 0)
	end_savetime = MPI_Wtime() - start_savetime; 

  if(rank==0) {
	cout << "Read time:\t" << end_read << endl;
	cout << "Save time:\t" <<  end_savetime << endl;
        cout << "Model Selecion stats:\t" << endl;
        cout << "\t-randomization time:\t" << end_rand << "(s)" << endl;
        cout << "\t-Zmx matrix creation time:\t" << end_Zmx << "(s)" << endl;
        cout << "\t-Distr. Kronecker Product time:\t"  << end_kron << "(s)" << endl;
        cout << "\t-Lasso1 time:\t" << allred_1 << "(s)" << endl;
   }



  float *bdata_es, *bdata_ts, *Y_train, *Y_test, *Z_mx_train, *Z_mx_test, *Z_stacked_train, *Z_stacked_test, *X_train, *X_test, *my_B;
  MatrixXf B;
  Map<MatrixXf> sprt_d (sprt, nlamb, D*p*p);
  MatrixXf Bgols_R2m(B2,nlamb);
  MatrixXf Bgols_B(B2*nlamb, D*p*p);

  /* Model Estimation -- Union step */  
  start_rand=0; end_rand=0; start_Zmx=0; end_Zmx=0; start_kron=0; end_kron=0;
  double start_rand_test, end_rand_test, start_Zmx_test, end_Zmx_test, start_kron_test, end_kron_testi, end_stack; 
  for(int k=0; k<B2; k++) {
	
	start_rand = MPI_Wtime(); 
        //float *bdata_f;
        bdata_es = (float *)malloc(qrows * p * sizeof(float));
	//if (rank == 0)
          //      cout << "B2 data.rows() = " << data.rows() << " q_rows = " << qrows << "\t p = " << p << "\t N = " << N << "\t size_s = " << size_s <<  endl;
        var_distribute_data(data_f, qrows, qrows, N, p, size_s, bdata_es, L, D, MPI_COMM_WORLD, comm_g );

	end_rand += MPI_Wtime() - start_rand; 
        //Map<Matrix<float, Dynamic, Dynamic, RowMajor> >bdata(bdata_f, q_rows, p);

	start_Zmx = MPI_Wtime(); 
        //float *Y; 
        //Y_train = (float *)malloc((N-D)*p*sizeof(float));
        int y_rows= bin_size_1D(rank_g, N-D, nprocs_g);
        Y_train = (float *)malloc(y_rows*p*sizeof(float));
        var_vectorize_response(bdata_es, qrows, y_rows, N, p, Y_train, L, D, MPI_COMM_WORLD, comm_g);

	//if (rank == 0)
	//	cout << "Passed var_vectorize" << endl ; 

        /* Generate Z_mx matrix for training data*/

        int z_rows = bin_size_1D(rank_g, (N-D)*D, nprocs_g);
        //float *Z_mx_e;
        Z_mx_train = (float *)malloc(z_rows * p * sizeof(float));

        var_generate_Z(bdata_es, qrows, z_rows, N, p, Z_mx_train, D, MPI_COMM_WORLD, comm_g);
        free(bdata_es);
	end_Zmx += MPI_Wtime() - start_Zmx; 
	//if (rank ==0)
	//	cout << "Passed Var_generate" << endl; 

	/* Generate Z_mx matrix for testing data*/

	start_rand_test = MPI_Wtime(); 
	bdata_ts = (float *)malloc(qrows * p * sizeof(float));
	var_distribute_data(data_f, qrows, qrows, N, p, size_s, bdata_ts, L, D, MPI_COMM_WORLD, comm_g );
	end_rand_test += MPI_Wtime() - start_rand_test;
	start_Zmx_test = MPI_Wtime(); 
	//Y_test = (float *)malloc((N-D)*p*sizeof(float));
	Y_test = (float *)malloc(y_rows*p*sizeof(float));
	var_vectorize_response(bdata_ts, qrows, y_rows, N, p, Y_test, L, D, MPI_COMM_WORLD, comm_g);

	Z_mx_test = (float *)malloc(z_rows * p * sizeof(float));
	var_generate_Z(bdata_ts, qrows, z_rows, N, p, Z_mx_test, D, MPI_COMM_WORLD, comm_g);
        free(bdata_ts);
 	
	end_Zmx_test += MPI_Wtime() - start_Zmx_test; 

	double start_stack = MPI_Wtime(); 
        MatrixXf Z_mx_tr;
        Z_mx_tr = stack_Z(Z_mx_train, z_rows, p, D, rank);
        free(Z_mx_train);
	MatrixXf Z_mx_ts;	
	Z_mx_tr = stack_Z(Z_mx_test, z_rows, p, D, rank);
	free(Z_mx_test);

	end_stack += MPI_Wtime() - start_stack; 
	
        /* do a distributed kronecker product on Z_mx with I(p,p) to get X*/
        /* this implementation of the Kronecker product is not an actual product
        * This implementation is just a bookkeeping technique using MPI_Get 
        * via one-sided communication 
        */

        //float *Z_stacked;
        Z_stacked_train = (float *)malloc(Z_mx_tr.rows() * Z_mx_tr.cols() * sizeof(float));
        Map<Matrix<float, Dynamic, Dynamic, RowMajor> >(Z_stacked_train, Z_mx_tr.rows(), Z_mx_tr.cols()) = Z_mx_tr;
	Z_stacked_test = (float *)malloc(Z_mx_ts.rows() * Z_mx_ts.cols() * sizeof(float));
	Map<Matrix<float, Dynamic, Dynamic, RowMajor> >(Z_stacked_test, Z_mx_ts.rows(), Z_mx_ts.cols()) = Z_mx_ts;

	start_kron = MPI_Wtime();

        //int kron_rows = bin_size_1D(rank_g, (N-D)*p, nprocs_g);
	int kron_rows = y_rows * p;
        //float *X; 
        X_train = var_kron (Z_stacked_train, Z_mx_tr.rows(), kron_rows, N, p, D, MPI_COMM_WORLD, comm_g);
        free(Z_stacked_train);

	X_test = var_kron (Z_stacked_test, Z_mx_ts.rows(), kron_rows, N, p, D, MPI_COMM_WORLD, comm_g);
	free(Z_stacked_test);

	end_kron += MPI_Wtime() - start_kron; 

	//float *my_B0;
        //MatrixXf B0; 
	VectorXi sprt_ids, arange, zdids;
        //if (rank_g == 0) B0 = MatrixXf::Zero(B1*nlamb, D*p*p);
        for (int nMP = 0; nMP < nlamb; nMP++) {
    
                my_B = lasso(X_train, kron_rows, D*p*p , Y_train, 0.0, comm_g, &allred_2);

		sprt_ids = sprt_d.row(nMP).unaryExpr(ptr_fun(not_NaN)).cast<int>();
		Map<VectorXf> rgstrct(my_B, D*p*p);
                arange.setLinSpaced(D*p*p, 0, D*p*p);
                zdids = setdiff1d(arange, sprt_ids);


                /*Apply support*/

                VectorXf my_Bgols_B(D*p*p);

                for (int i = 0; i < sprt_ids.size(); i++) {
                	my_Bgols_B(0) = 0;
                        if (sprt_ids(i) != 0)
                        	my_Bgols_B(sprt_ids(i)) = rgstrct(i);
                }

                for (int i=0; i<zdids.size(); i++)
                	my_Bgols_B(zdids(i)) = 0;

		free(my_B); 
		VectorXf yhat; 
                if (rank_g == 0) {
                	Map<Matrix<float, Dynamic, Dynamic, RowMajor> > X_test_eig (X_test, kron_rows, D*p*p);
                        Map<VectorXf> y_test (Y_test, y_rows*p);
                        yhat = X_test_eig * my_Bgols_B;
                        float r =  pearson1(yhat, y_test);
                        Bgols_R2m(k,nMP) = r*r;
                        Bgols_B.row((k*nlamb)+nMP) = my_Bgols_B;
                }
        }

        /* free memory for the current bootstrap */
        free(Y_train);
        free(X_train);
	free(Y_test);
	free(X_test); 
   }


  double est_time, est_end; 
  est_time = MPI_Wtime(); 
  VectorXf Bgd_r(D*p*p);
  MatrixXf btmp(B2, D*p*p);

  if (rank_g == 0) {
	float v;
	if (bgdopt == 1) {
	 	v = Bgols_R2m.unaryExpr(ptr_fun(not_NaN)).maxCoeff();
                VectorXi ids_kk;
                for (int kk=0; kk<B2; kk++) {
                	ids_kk = where(Bgols_R2m.row(kk), v);
                        if (!ids_kk.isZero())
                        	btmp.row(kk) = Bgols_B.row(ids_kk(floor(ids_kk.size()/2)));
                        else
                        	btmp.row(kk) = Bgols_B.row(kk % B2);	
		}

	}
	Bgd_r = median(btmp);
  }

  est_end = MPI_Wtime() - est_time ; 
  saveTime=0;

  if (rank == 0) saveTime = MPI_Wtime();
 
  if (MPI_COMM_NULL != comm_ROOT) {
	
	float *R2;
	R2 = (float *) malloc (Bgols_R2m.rows() * Bgols_R2m.cols() * sizeof(float));
	Map<Matrix<float, Dynamic, Dynamic, RowMajor> >(R2, Bgols_R2m.rows(), Bgols_R2m.cols()) = Bgols_R2m;

	float *Bgd; 
	Bgd = (float *) malloc (D*p*p * sizeof(float));
 	Map<VectorXf>(Bgd, Bgd_r.size()) = Bgd_r;
	 
        write_output (OUTPUTFILE, ngroups, D*p*p, B2, nlamb, Bgd, R2, comm_ROOT);
	
	free(R2);
	free(Bgd);
  }
  if(rank==0) {
	double end_saveTime = MPI_Wtime() - saveTime; 

        cout << "Model Estimation stats: " << endl;
        cout << "\t-randomization time train:\t" << end_rand << "(s)" << endl;
        cout << "\t-randomization time test:\t" << end_rand_test << "(s)" << endl;
        cout << "\t-Zmx matrix creation time train:\t" << end_Zmx << "(s)" << endl;
        cout << "\t-Zmx matrix creation time test:\t" << end_Zmx_test << "(s)" << endl;
        cout << "\t-stack Z time:\t" << end_stack << "(s)" << endl;
        cout << "\t-Distr. Kronecker Product time:\t"  << end_kron << "(s)" << endl;
        cout << "\t-Lasso2 time:\t" << allred_2 << "(s)" << endl;
	cout << "\t-Final Est time:\t" << est_end << "(s)" << endl; 
	cout << "\t-Save time:\t" << end_saveTime << "(s)" << endl; 
   }


 /* Write output to hdf5 file: use module from manage-data.c */ 

  if (rank == 0) {
	end_time = MPI_Wtime() - start_time; 
	 cout << "\nUnion of Intersections Vector Auto-Regressive model (UOI-VAR) analysis completed in :\t" << end_time << endl ; 
  }

  MPI_Finalize();
  return 0; 

}
