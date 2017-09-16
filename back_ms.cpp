#include <iostream>
#include <fstream>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <algorithm>
#include <vector>
#define EIGEN_DEFAULT_TO_ROW_MAJOR
#include <eigen3/Eigen/Dense>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_statistics.h>
//#include "lasso_admm_MPI.h"
#include "lasso.h"
#include "matrix-operations.h"
#include "distribute-data.h"
#include "model_estimate.h"
//#include "debug.h"

#define DEBUGS 0

using namespace Eigen;
using namespace std; 


float not_NaN (float x) {if (!isnan(x)) return x; else return 0;}

VectorXi where (VectorXf vec, float value) {
        vector<int> v;

        for (int i =0; i < vec.size(); i++)
                if (vec(i) == value) {v.push_back(i);}

         Map<VectorXi> vec1(v.data(),v.size());
        return vec1;
}

/*int bin_coord(int i, int N, int M) {
  // range of size N divided in M bins
  // returns which bin owns i
  int j,k=0;
  int b = N / M;
  for (j=0; j<M; j++) {
    k += (j < N % M) ? b+1 : b;
    if (i < k) return j;
  }
}*/

/*VectorXi where (VectorXf vec, float value) {
        vector<int> v;

        for (int i =0; i < vec.size(); i++)
                if (vec(i) == value) {v.push_back(i);}

         Map<VectorXi> vec1(v.data(),v.size());
        return vec1;
}*/ 


VectorXi where(VectorXi vec, int value) {
        vector<int> v;

        for (int i=0; i<vec.size(); i++)
                if (vec(i) == value) {v.push_back(i);}

        Map<VectorXi> vec1(v.data(),v.size());
        return vec1;
}

VectorXi setdiff1d (VectorXi vec1, VectorXi vec2) {

        vector<int> v;

        sort(vec1.data(), vec1.data()+vec1.size());
        sort(vec2.data(), vec2.data()+vec2.size());

        set_difference (vec1.data(), vec1.data()+vec1.size(), vec2.data(), vec2.data()+vec2.size(), back_inserter(v));

        Map<VectorXi> vec3(v.data(),v.size());

        return vec3;
}


VectorXf shuffles (VectorXf vec1, VectorXf vec2) {
        VectorXf vec3(vec2.size());

        for (int i=0; i<vec2.size(); i++)
                vec3(i) = vec1(vec2(i));

        return vec3;
}

VectorXi shuffles (VectorXi vec1, VectorXi vec2) {
        VectorXi vec3(vec2.size());

        for (int i=0; i<vec2.size(); i++)
                vec3(i) = vec1(vec2(i));

        return vec3;
}

VectorXf shuffles (VectorXf vec1, VectorXi vec2) {
        VectorXf vec3(vec2.size());

        for (int i=0; i<vec2.size(); i++)
                vec3(i) = vec1(vec2(i));

        return vec3;
}


MatrixXf shuffles (MatrixXf mat, VectorXf vec) {
        MatrixXf mat1(vec.size(), mat.cols());

        for(int i=0; i<vec.size(); i++)
                mat1.row(i) = mat.row(vec(i));

        return mat1;
}

MatrixXf shuffles (MatrixXf mat, VectorXi vec) {
        MatrixXf mat1(vec.size(), mat.cols());

        for(int i=0; i<vec.size(); i++)
                mat1.row(i) = mat.row(vec(i));

        return mat1;
}

VectorXi select (VectorXi a, VectorXi b) {

        vector<int> v;

        for (int i =0; i<b.size(); i++) {
                for (int j=0; j< a.size(); j++) {
                        if (b(i) == a(j)) {v.push_back(i); break;}
                }
        }

        Map<VectorXi> vec (v.data(), v.size());

        return vec;
}


VectorXi select_not (VectorXi a, VectorXi b) {

        vector<int> v;

        for (int i =0; i<b.size(); i++) {
                for (int j=0; j< a.size(); j++) {
                        if (b(i) != a(j)) {v.push_back(i); break;}
                }
        }

        Map<VectorXi> vec (v.data(), v.size());

        return vec;
}


double pearson1 (VectorXf vec1, VectorXf vec2) {


        VectorXd vec1_d = vec1.cast <double>();
        VectorXd vec2_d = vec2.cast <double>();

        gsl_vector_view gsl_x = gsl_vector_view_array( vec1_d.data(), vec1_d.size());
        gsl_vector_view gsl_y = gsl_vector_view_array( vec2_d.data(), vec2_d.size());

        gsl_vector *gsl_v1 =  &gsl_x.vector;
        gsl_vector *gsl_v2 = &gsl_y.vector;
        double r = gsl_stats_correlation (gsl_v1->data, 1, gsl_v2->data, 1, gsl_v1->size);
        return r;
}


VectorXf median (MatrixXf mat) {

        VectorXf get_col;
        vector<float> v;

        for (int i=0; i<mat.cols(); i++) {
                get_col = mat.col(i);
                nth_element (get_col.data(), get_col.data()+ get_col.size()/2,
                                get_col.data()+get_col.size());

                v.push_back(get_col(get_col.size()/2));

        }

        Map<VectorXf> vec (v.data(), v.size());

        return vec;
} 


/*VectorXf median (VectorXf Vec_in) {
        vector<float> v;

        for (int i=0; i<Vec_in.size(); i++) {
            
                nth_element (Vec_in.data(), Vec_in.data()+ Vec_in.size()/2,
                                Vec_in.data()+Vec_in.size());

                v.push_back(get_col(get_col.size()/2));

        }

        Map<VectorXf> vec (v.data(), v.size());

        return vec;
}*/

void get_train_rows(float *Mat_in, float *Vec_in, float *Mat1, float *vec1, float *Mat2, float *Vec2, int rows, int train_row, int test_row, int cols, int L_h) {
	Map<MatrixXf> Mat_eig(Mat_in, rows, cols);
	Map<VectorXf> Vec_eig(Vec_in, rows); 

	VectorXi cv_ids(L_h), boot_ids(train_row);
	
         cv_ids.setLinSpaced(L_h, 0, L_h);
         random_shuffle (cv_ids.data(), cv_ids.data()+cv_ids.size());
         boot_ids.setLinSpaced(train_row, 0, train_row);
         random_shuffle (boot_ids.data(), boot_ids.data()+boot_ids.size());

	MatrixXf Mat_train(train_row, cols), Mat_test(test_row, cols); 
	VectorXf Vec_train(train_row), Vec_test(test_row); 

	for (int i=0; i<train_row; i++) {
		Mat_train.row(i) = Mat_eig.row(cv_ids(boot_ids(i))); 
		Vec_train(i) = Vec_eig(cv_ids(boot_ids(i))); 
	}

	int tt=0;
	for (int i=0; i<test_row; i++) {
		tt = i+train_row; 
                Mat_test.row(i) = Mat_eig.row(cv_ids(tt));
                Vec_test(i) = Vec_eig(cv_ids(tt));
        }

	Map<MatrixXf> (Mat1, Mat_train.rows(), Mat_train.cols()) = Mat_train; 
	Map<VectorXf> (vec1, Vec_train.size()) = Vec_train; 
	Map<MatrixXf> (Mat2, Mat_test.rows(), Mat_test.cols()) = Mat_test;
        Map<VectorXf> (Vec2, Vec_test.size()) = Vec_test;

	//cout << "Passed train" << endl;

}

/*get_test_rows(float *Mat_in, float *Vec_in, float *Mat, float *vec, int rows, int test_row, int cols, int L_h) {
        Map<MatrixXf> Mat_eig(Mat_in, rows, cols);
        Map<VectorXf> Vec_eig(VEc_in, rows);

         VectorXi cv_ids, boot_ids;
         cv_ids.setLinSpaced(L_h 0, L_h);
         random_shuffle (cv_ids.data(), cv_ids.data()+cv_ids.size());
         boot_ids.setLinSpaced(train_row, 0, train_row);
         random_shuffle (boot_ids.data(), boot_ids.data()+boot_ids.size());

        MatrixXf Mat_out;
        VectorXf Vec_out;

        for (int i=0; i<train_row; i++) {
                Mat_out.row(i) = Mat_eig(cv_ids(boot_ids(i)));
                Vec_eig(i) = Vec_eig(cv_ids(boot_ids(i)));
        }

        Map<MatrixXf> (Mat, Mat_out.rows(), Mat_out.cols()) = Mat_out;
        Map<VectorXf> (vec, Vec_out.size()) = Vec_out;

}*/


/*get_test_rows_mat(X_in, X_test_e, m_node, test, n, L)


get_test_rows_vec(y_in, y_test, m_node, test, L)*/


void get_T_rows(float *mat_in, float *vec_in, float *mat_out, float *vec_out, int rows, int cols, int T_h) {

	//cout << "In T" << endl;
	Map<MatrixXf> Mat_eig(mat_in, rows, cols);
        Map<VectorXf> Vec_eig(vec_in, rows);

	MatrixXf Mat_out; 
	VectorXf Vec_out;
		
	Mat_out = Mat_eig.bottomRows(T_h); 
	Vec_out = Vec_eig.tail(T_h);

	Map<MatrixXf> (mat_out, Mat_out.rows(), Mat_out.cols()) = Mat_out;
        Map<VectorXf> (vec_out, Vec_out.size()) = Vec_out;

	//cout << "Passed T" << endl; 

}


/*get_T_rows_mat(X_in, X_T_e, m_node, T, n)


get_T_rows_vec(y_in, y_T_e, m_node, T)*/



/*void randomize(float *Mat_train, float *Vec_train, int rows, int cols, int seed) {

        Map<Matrix<float, Dynamic, Dynamic, RowMajor> > M_train(Mat_train, rows, cols);
        Map<VectorXf> V_train (Vec_train, rows);

        MatrixXf Big_M(rows, cols+1);

        Big_M << M_train,V_train; 

        //srand(seed);
        VectorXi ids;
        ids.setLinSpaced(rows, 0, rows);
        random_shuffle(ids.data(), ids.data()+ids.size());

        MatrixXf Big_out(rows, cols+1);
        MatrixXf M_out1(rows, cols);
        VectorXf V_out1(rows);


        for (int i=0;i<rows; i++)
                Big_out.row(i) = Big_M.row(ids(i)); 
	
        M_out1 = Big_out.topLeftCorner(rows, cols);
        V_out1 = Big_out.topLeftCorner(rows, cols+1).rightCols(1);
	
        Map<MatrixXf> (Mat_train, M_out1.rows(), M_out1.cols()) = M_out1;
        Map<VectorXf> (Vec_train, V_out1.size()) = V_out1;

}*/

void randomize(float *Mat_train, float *Vec_train, int rows, int cols, int seed) {

        PermutationMatrix<Dynamic,Dynamic> perm(rows);
        perm.setIdentity();
        random_shuffle(perm.indices().data(), perm.indices().data()+perm.indices().size());

        Map<Matrix<float, Dynamic, Dynamic, RowMajor> > M_train(Mat_train, rows, cols);
        Map<VectorXf> V_train(Vec_train, rows);

        M_train = perm * M_train;
        V_train = perm * V_train;

        Map<MatrixXf> (Mat_train, M_train.rows(), M_train.cols()) = M_train;
        Map<VectorXf> (Vec_train, V_train.size()) = V_train;

}

//void model_estimate(float *sprt_in, float *X_train_e, float *y_train_e, float *X_test_e, float *y_test_e, float *X_T_e, float *y_T_e, int L, int T, int train, int test,  
//	float *Bgd_out, float *rsd_out, float *R2_out, float *bic_out, int m_node, int n, int nbootS, int nbootE, int nrnd, float *time, MPI_Comm comm_WORLD, MPI_Comm comm_NRND, MPI_Comm comm_EST) { 

void model_estimate(float *X_train_e, float *y_train_e, float *X_test_e, float *y_test_e, float *X_T_e, float *y_T_e, float *sprt_in, 
		    int L, int train, int m, int n, int n_rows, int k_rows, int nMP, int nbootE, int nrnd, double *time, 
			float *Bgd_out, float *rsd_out, float *R2_out, float *bic_out, MPI_Comm comm_WORLD, MPI_Comm comm_NRND) {

    //cout.precision(6);
 
    int T = m - L; 
    int test = L - train; 

    int rank_WORLD, nprocs_WORLD;
    MPI_Comm_rank (comm_WORLD, &rank_WORLD);
    MPI_Comm_size (comm_WORLD, &nprocs_WORLD); 

    
    int rank_NRND, nprocs_NRND;
    MPI_Comm_rank(comm_NRND, &rank_NRND);
    MPI_Comm_size(comm_NRND, &nprocs_NRND);
 
    MPI_Group WORLD_group;
    MPI_Comm_group (comm_WORLD, &WORLD_group);

    int root[nrnd];
    int tt=0;
    for (int i=0; i<nrnd; i++) {
        root[i] = tt;
        tt+=nprocs_NRND;
    }

    MPI_Group root_all;
    MPI_Group_incl(WORLD_group, nrnd, root, &root_all);

    MPI_Comm comm_root_nrnd;
    MPI_Comm_create_group(comm_WORLD, root_all, 1, &comm_root_nrnd);

    int bgdOpt = 1; 
    float start_time;

     float *_rgstrct;
     _rgstrct = (float *) malloc (n * sizeof(float));

     int mp_id = rank_WORLD % nMP;

    VectorXf  Bgd_r(n), R2_r(1), rsd_r(T), bic_r(1);
    MatrixXf btmp(nbootE, n);

    Map<MatrixXf> sprt_d (sprt_in, nMP, n);
    VectorXi sprt_ids(n);
	
    //cout << "RANK / RANK_NRND" << "(" << rank_WORLD << "/" << rank_NRND << ") " << "sprt.size"  << "(" << sprt_d.rows() << "," << sprt_d.cols() << ")" << endl; 

    for (int kk=0; kk<nbootE; kk++) {

	randomize(X_train_e, y_train_e, train, n, kk);
    	start_time = MPI_Wtime(); 	

    	//_rgstrct = lasso_admm(X_train_e, train, n,  y_train_e, 0.0, comm_NRND);
	_rgstrct = lasso(X_train_e, train, n, y_train_e, 0.0, comm_NRND);  
	Map<VectorXf> rgstrct(_rgstrct, n);
    
    	*time += MPI_Wtime() - start_time;

    	VectorXi  zdids, arange(n);

    	sprt_ids = sprt_d.row(mp_id).unaryExpr(ptr_fun(not_NaN)).cast<int>(); 

    	arange.setLinSpaced(n, 0, n);
    	zdids = setdiff1d(arange, sprt_ids);


	/*Apply support*/

    	VectorXf my_Bgols_B(n), my_Bgols_R2m(1); 

    	for (int i = 0; i < sprt_ids.size(); i++) {
		my_Bgols_B(0) = 0;  
    		if (sprt_ids(i) != 0) 
			my_Bgols_B(sprt_ids(i)) = rgstrct(i);
   	} 

   	for (int i=0; i<zdids.size(); i++)
    		my_Bgols_B(zdids(i)) = 0;


    	VectorXf yhat; 

    	if (rank_NRND == 0) {
		Map<MatrixXf> X_test (X_test_e, test, n);
		Map<VectorXf> y_test (y_test_e, test); 
		yhat = X_test * my_Bgols_B;
   		float r =(float) pearson1(yhat, y_test);
		my_Bgols_R2m << r*r;
    	}
    
    	VectorXf Bgols_R2m(nrnd); 
    	MatrixXf Bgols_B(nrnd, n);

   	MPI_Barrier(comm_WORLD);

   	if (MPI_COMM_NULL != comm_root_nrnd) {
   		MPI_Gather (my_Bgols_B.data(), n, MPI_FLOAT, Bgols_B.data(), n, MPI_FLOAT, 0, comm_root_nrnd);
   		MPI_Gather (my_Bgols_R2m.data(), 1, MPI_FLOAT, Bgols_R2m.data(), 1, MPI_FLOAT, 0, comm_root_nrnd); 
   	} 

   	/*** Bagging ******/

   	float v;
   	VectorXi  ids_r, ids_c;  

   	if (rank_WORLD == 0) { 
   		if (bgdOpt == 1) {
   			v = Bgols_R2m.unaryExpr(ptr_fun(not_NaN)).maxCoeff();

			VectorXi ids_kk;	
			ids_kk = where(Bgols_R2m, v);

			if (!ids_kk.isZero())
				btmp.row(kk) = Bgols_B.row(ids_kk(floor(ids_kk.size()/2)));
			else	
				btmp.row(kk) = Bgols_B.row(kk % nrnd);

			
		

		}

   	}

  }
  

  if (rank_WORLD == 0)
	 Bgd_r = median(btmp);


   /*section when bgdopt!=1*/ 
    /*else {
    	VectorXf mean_Bgols_R2m;
       	 MatrixXf Bgols_B_median;
        mean_Bgols_R2m = Bgols_R2m.colwise().mean();
        float vv = mean_Bgols_R2m.maxCoeff();
        ids_r = where(mean_Bgols_R2m, vv);

        for (int jj=0; jj<n; jj++)
        	Bgols_B_median.col(jj) = Bgols_B.col(ids(jj));
        Bgd_r = median(Bgols_B_median);
   }*/


   if(rank_WORLD==0) {
	Map<MatrixXf> X_T(X_T_e, T, n);
        Map<VectorXf> y_T(y_T_e, T);
	VectorXf yhat;         
	yhat = X_T * Bgd_r; 
	float r = pearson1(yhat, y_T); 
	R2_r << r*r;
	cout << "R2 = " << R2_r << endl; 
	rsd_r = y_T - yhat; 
	bic_r << (m-train) * log(rsd_r.squaredNorm()/(m-train)) + log(m-train) * n; 
  }


   if (MPI_COMM_NULL != comm_root_nrnd) {
  	MPI_Bcast(Bgd_r.data(), n, MPI_FLOAT, 0, comm_root_nrnd); 
	MPI_Bcast(rsd_r.data(), rsd_r.rows(), MPI_FLOAT, 0, comm_root_nrnd);
	MPI_Bcast(R2_r.data(), 1, MPI_FLOAT, 0, comm_root_nrnd);
	MPI_Bcast(bic_r.data(), 1, MPI_FLOAT, 0, comm_root_nrnd);
   }

  /*MatrixXf Bgd(nrnd, n), rsd(nrnd, T); 
  VectorXf R2(nrnd), bic(nrnd); */ 
   
  if (rank_NRND == 0) {
  	Map<MatrixXf> (Bgd_out, Bgd_r.rows(), Bgd_r.cols()) = Bgd_r; 
 	Map<MatrixXf> (rsd_out, rsd_r.rows(), rsd_r.cols()) = rsd_r;
  	Map<VectorXf> (R2_out, R2_r.size()) = R2_r; 
 	Map<VectorXf> (bic_out, bic_r.size()) = bic_r; 
  }

}
