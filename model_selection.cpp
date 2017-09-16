#include <iostream>
#include <fstream>
#include <stdio.h>
#include <cmath>
#include <mpi.h>
#include <algorithm>
#include <vector>
#define EIGEN_DEFAULT_TO_ROW_MAJOR
#include <eigen3/Eigen/Dense>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_statistics.h>
#include "lasso_admm_MPI.h"
#include "lasso.h"
#include "matrix-operations.h"
#include "distribute-data.h"
#include "model_selection.h"


using namespace Eigen;
using namespace std;

float is_not_NaN2(float x) {if (!isnan(x)) return x; else return 0;}
pointer_to_unary_function <float,float> floorObject1 (floor) ;

double pearson2 (VectorXf vec1, VectorXf vec2) {


        VectorXd vec1_d = vec1.cast <double>();
        VectorXd vec2_d = vec2.cast <double>();

        gsl_vector_view gsl_x = gsl_vector_view_array( vec1_d.data(), vec1_d.size());
        gsl_vector_view gsl_y = gsl_vector_view_array( vec2_d.data(), vec2_d.size());

        gsl_vector *gsl_v1 =  &gsl_x.vector;
        gsl_vector *gsl_v2 = &gsl_y.vector;
        double r = gsl_stats_correlation (gsl_v1->data, 1, gsl_v2->data, 1, gsl_v1->size);
        return r;
}

/*void randomize1(float *Mat_train, float *Vec_train, int rows, int cols, int seed, int rank_s) {

	float map_m = MPI_Wtime();
        Map<Matrix<float, Dynamic, Dynamic, RowMajor> > M_train(Mat_train, rows, cols);
        Map<VectorXf> V_train (Vec_train, rows);

	if (rank_s == 0) {
	  	float map_end = MPI_Wtime() - map_m;	
	 	cout << "map_end = " << map_end << "(s)" << endl; 
	}

	float stack = MPI_Wtime(); 

        MatrixXf Big_M(rows, cols+1);

        Big_M << M_train,V_train; 

	if (rank_s == 0) {
                float stack_end = MPI_Wtime() - stack;
                cout << "stack_end = " << stack_end << "(s)" << endl;
        }
        //srand(seed);
        VectorXi ids;
        ids.setLinSpaced(rows, 0, rows);
        random_shuffle(ids.data(), ids.data()+ids.size());

	MatrixXf Big_out(rows, cols+1); 
        MatrixXf M_out1(rows, cols);
        VectorXf V_out1(rows); 

	float loop_h = MPI_Wtime();
        for (int i=0;i<rows; i++) 
                Big_out.row(i) = Big_M.row(ids(i));

	if (rank_s == 0) {
                float loop_h_end = MPI_Wtime() - loop_h;
                cout << "rand loop_end = " << loop_h_end << "(s)" << endl;
        }

	float block = MPI_Wtime(); 
	M_out1 = Big_out.topLeftCorner(rows, cols);
        V_out1 = Big_out.topLeftCorner(rows, cols+1).rightCols(1); 
	if (rank_s == 0) {
                float block_end = MPI_Wtime() - block;
                cout << "block_end = " << block_end << "(s)" << endl;
        }

        Map<MatrixXf> (Mat_train, M_out1.rows(), M_out1.cols()) = M_out1;
        Map<VectorXf> (Vec_train, V_out1.size()) = V_out1;

}*/

void randomize1(float *Mat_train, float *Vec_train, int rows, int cols, int seed, int rank) {

	//srand(seed);


	//if (seed == 2 && rank == 0) {cout << "inside randomize" << endl;}

	PermutationMatrix<Dynamic,Dynamic> perm(rows);
	perm.setIdentity();
	random_shuffle(perm.indices().data(), perm.indices().data()+perm.indices().size());

	//if (seed == 2 && rank == 0) {cout << "passed shuffle" << endl;}

	Map<Matrix<float, Dynamic, Dynamic, RowMajor> > M_train(Mat_train, rows, cols);
        Map<VectorXf> V_train(Vec_train, rows);

	M_train = perm * M_train;
	V_train = perm * V_train; 

	  //if (seed == 2 && rank == 0) {cout << "passed compute" << endl;}
	Map<MatrixXf> (Mat_train, M_train.rows(), M_train.cols()) = M_train;
        Map<VectorXf> (Vec_train, V_train.size()) = V_train; 
	  //if (seed == 2 && rank == 0) {cout << "passed map" << endl;}

}

VectorXf logspace2 (int start, int end, int size) {

    VectorXf vec;
    vec.setLinSpaced(size, start, end);

    for(int i=0; i<size; i++)
        vec(i) = pow(10,vec(i));

    return vec;
}

VectorXf linspace2 (float start, float end, int size) {

    VectorXf vec;
    vec.setLinSpaced(size, start, end);
    return vec;
}

VectorXi where_eq2(VectorXi vec, float value) {
        vector<int> v_in; 

        for (int i =0; i < vec.size(); i++) {
                if (vec(i) == value) {
			v_in.push_back(i);
		} 
	}

	//for (int i=0; i<v_in.size(); i++)
	//	cout << v_in[i] << endl;
	
	
         Map<VectorXi> vec1(v_in.data(),v_in.size());
        return vec1;
}

float dense_sweep2(VectorXf R2m_vec, VectorXf lamb0, int boot) {

        VectorXi Lids, Mt;

        MatrixXf R2(R2m_vec.size(), 1);
        R2 << R2m_vec.unaryExpr(ptr_fun(is_not_NaN2))*1e4;
	
	R2.resize(boot, lamb0.size()); 
	
	Mt = R2.colwise().mean().unaryExpr(floorObject1).cast<int>();
        float s;
        int tmp = Mt.maxCoeff(); 
        Lids = where_eq2(Mt, tmp);
        s = lamb0(Lids(floor(Lids.size()/2))); 
        return s;


}


void model_selection (float *X_train_e, float *y_train_e, float *X_test_e, float *y_test_e, float *lamb_out, float *B_out, float *R_out, double *las1time, double *las2time, int nboot, int coarse, int train, int n, int q_rows, int nMP, MPI_Comm comm_world, MPI_Comm comm_group) {
 

	int rank_world, rank_group; 
	MPI_Comm_rank(comm_world, &rank_world);
	MPI_Comm_rank(comm_group, &rank_group);

        VectorXf yhat, R2m0(coarse*nMP), R2m1(nboot*nMP);

	VectorXf lamb(nMP);
	if (rank_world==0) 
		lamb = logspace2(-3, 3, nMP);
	
	MPI_Bcast(lamb.data(), nMP, MPI_INT, 0, comm_world);  


	int count=0;
	float *my_B0, *my_B; 
	//my_B0 = (float *)malloc(n * sizeof(float)); 

 	for (int i=0; i<coarse; i++) {
		randomize1(X_train_e, y_train_e, train, n, i, rank_world);
		randomize1(X_test_e, y_test_e, q_rows-train, n, i, rank_world); 
	
		for (int j=0; j<nMP; j++) {
			count++;
			float my_lamb = lamb(j);	
			MPI_Barrier(comm_world);
			double las_start = MPI_Wtime();
			my_B0 = lasso(X_train_e, train, n, y_train_e, my_lamb, comm_group);

			if(i==0) {
				*las1time = MPI_Wtime() - las_start;
			}

        		if (rank_world == 0) {
				Map<Matrix<float, Dynamic, Dynamic, RowMajor> > X_test (X_test_e, q_rows-train, n);
                		Map<VectorXf> y_test (y_test_e, q_rows-train);
				Map<VectorXf> my_B (my_B0, n);
                		yhat = X_test * my_B;
                		double r = pearson2(yhat, y_test);
                		float my_R2m =(float) r*r;
				R2m0((i*nMP)+j) = my_R2m;
				//cout << "count " << count << endl;  
        		}
			free(my_B0);
		} 
	}

	float v, dv; 

	if (rank_world == 0) {  
		v = dense_sweep2(R2m0, lamb, nboot);
		dv = get_dv(v);
		lamb = linspace2(v-5*dv,v+5*dv,nMP);
	}

	MPI_Bcast(lamb.data(), nMP, MPI_INT, 0, comm_world);
 
	MatrixXf B0(nboot*nMP, n); 	
	count = 0; 
	for (int i=0; i<nboot; i++) {
                randomize1(X_train_e, y_train_e, train, n, i, rank_world);	
		randomize1(X_test_e, y_test_e, q_rows-train, n, i, rank_world);
	
		for (int j=0; j<nMP; j++) {
			count++;
                	float my_lamb = lamb(j); 

                	double lasso2_t = MPI_Wtime();
			my_B = lasso(X_train_e, train, n, y_train_e, my_lamb, comm_group);   
			 *las2time = MPI_Wtime() - lasso2_t; 
                	if (rank_world == 0) {
				Map<Matrix<float, Dynamic, Dynamic, RowMajor> > X_test (X_test_e, q_rows-train, n);
                		Map<VectorXf> y_test (y_test_e, q_rows-train);
                        	Map<VectorXf> my_B1(my_B, n);	
                        	yhat = X_test * my_B1;
                        	double r = pearson2(yhat, y_test);
                        	float my_R2m1 =(float) r*r;
                        	R2m1((i*nMP)+j) = my_R2m1;
                        	B0.row((i*nMP)+j) = my_B1; 
                	}
			free(my_B);	

        	}
	} 


	if (rank_world == 0) { 
        	Map<MatrixXf> (B_out, B0.rows(), B0.cols()) = B0;
                Map<VectorXf> (R_out, R2m1.size()) = R2m1;
		Map<VectorXf> (lamb_out, lamb.size()) = lamb;
       }
}
