#include <iostream>
#include <fstream>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <algorithm>
#include <vector>
#define EIGEN_USE_MKL_ALL
#define EIGEN_DEFAULT_TO_ROW_MAJOR
#include <eigen3/Eigen/Dense>
//#include <gsl/gsl_vector.h>
//#include <gsl/gsl_statistics.h>
#include "matrix-operations.h"
//#include "debug.h"

using namespace Eigen;
using namespace std;

float is_not_NaN (float x) {if (!isnan(x)) return x; else return 0;}
pointer_to_unary_function <float,float> floorObject (floor) ;

/*float* logspace (int start, int end, int size) {

    VectorXf vec; 
    vec.setLinSpaced(size, start, end);
    float * Out;
    Out = (float *)malloc(size*sizeof(float)); 

    for(int i=0; i<size; i++)
        vec(i) = pow(10,vec(i));	
	
    Map<VectorXf>(Out, vec.size()) = vec;

   return Out; 
}*/

float* logspace (int start, int end, int size) {

    VectorXf vec1, vec;
    vec1.setLinSpaced(size, start, end);
    VectorXf vec2(size);
    vec2.fill(10);
    vec = vec2.array().pow(vec1.array());
    float * Out;
    Out = (float *)malloc(size*sizeof(float));
    Map<VectorXf>(Out, vec.size()) = vec;

   return Out;
}

float* linspace (float start, float end, int size) {

    VectorXf vec;
    vec.setLinSpaced(size, start, end);
    float * Out;
    Out = (float *)malloc(size*sizeof(float));

    /*for(int i=0; i<size; i++)
        vec(i) = pow(10,vec(i));*/

    Map<VectorXf>(Out, vec.size()) = vec;

   return Out;
}


void combine_matrix (float *Mat, float *Vec, float *Out, int m, int n) {

   //cout << "m=" << m << endl;
   //cout << "n=" << n << endl;

   Map<Matrix<float, Dynamic, Dynamic, RowMajor> > In (Mat, m, n);
   Map<VectorXf> Vec_eig(Vec, m); 

   Matrix<float, Dynamic, Dynamic, RowMajor> Out_eig (m, n+1); 

   Out_eig << In,Vec_eig; 
 
   Map<Matrix<float, Dynamic, Dynamic, RowMajor> > (Out, Out_eig.rows(), Out_eig.cols() ) = Out_eig;

}


void split_matrix (float *In, float *Mat, float *Vec, int m, int n) {

	//cout << "m=" << m << endl; 
	//cout << "n=" << n << endl;

	Map<Matrix<float, Dynamic, Dynamic, RowMajor> > In_eig (In, m, n);
	//Map<Matrix<float, Dynamic, Dynamic, RowMajor> > Mat_eig (Mat, m, n-1);
	//Map<VectorXf> Vec_eig(Vec, m);
	
	Matrix<float, Dynamic, Dynamic, RowMajor> Mat_eig(m,n-1); 
	VectorXf Vec_eig (m);

	//cout << "before split" << endl; 
	Vec_eig << In_eig.rightCols(1);
	Mat_eig << In_eig.topLeftCorner(m, n-1);

	//cout << "after split" << endl; 
	Map<Matrix<float, Dynamic, Dynamic, RowMajor> > (Mat, Mat_eig.rows(), Mat_eig.cols() ) = Mat_eig; 
	Map<VectorXf> (Vec, Vec_eig.size()) = Vec_eig;	

}


/*void split_matrix (float *In, float *Mat, float *Vec, int m, int n, int this_rank) {

      //Map<Matrix<float, Dynamic, Dynamic, RowMajor> > In_eig (In, m, n);
      //Map<Matrix<float, Dynamic, Dynamic, RowMajor> > Mat_eig (Mat, m, n-1);	
      Map<MatrixXf> In_eig(In, m, n);
      Map<MatrixXf> Mat_eig(Mat, m, n-1);
      Map<VectorXf> Vec_eig(Vec, m); 

      //MatrixXf Mat_eig; 
      //VectorXf Vec_eig; 
    
      
      	 Vec_eig << In_eig.rightCols(1); 

      Mat_eig << In_eig.topLeftCorner(m, n-1);

      ofstream myfile2 ("y_eig.dat");

    if (myfile2.is_open())

    {

        myfile2 << Vec_eig;

        myfile2.close();

    }


	 ofstream myfile1 ("X_eig.dat");

    if (myfile1.is_open())

    {

        myfile1 << Mat_eig;

        myfile1.close();

    }


      Map<MatrixXf> (Vec, Vec_eig.rows(), Vec_eig.cols()) = Vec_eig;

	ofstream myfile ("y_this.txt");
   	if (myfile.is_open()) {
 	 for (int i=0; i<m; i++){
		myfile << *(Vec + i) << "\n"; 
	 }
	myfile.close();
	}
		
      Map<MatrixXf> (Mat, Mat_eig.rows(), Mat_eig.cols()) = Mat_eig;
}*/



/*void get_test (float *Mat, float *Out_hat, float *Out_mean, int m, int n, int train, float *B, int rank) {

     Map<MatrixXf> Mat_eig (Mat, m, n+1);
     //Map<VectorXf> Vec_eig(Vec, m);
     Map<VectorXf> B_eig(B, n);    

    MatrixXf X_test; 
    VectorXf y_test, y_mean, yhat;

    //X_test = Mat_eig.bottomRows(m - train);
    X_test = Mat_eig.block(train, 0, m-train, n); 
    //y_test = Vec_eig.tail(m - train);
    y_test = Mat_eig.block(train, 0, m-train, n+1).rightCols(1); 
    yhat = X_test * B_eig; 
    y_mean = y_test.array() - y_test.mean(); 

    Map<VectorXf> (Out_hat, yhat.size()) = yhat;
    Map<VectorXf> (Out_mean, y_mean.size()) = y_mean;
}*/

void get_test(float *Mat, float *Vec, int m, int n, float *B_in, float *Out_hat) {

	Map<MatrixXf> Mat_eig(Mat, m, n);
	Map<VectorXf> Vec_eig(Vec, m);
	Map<VectorXf> B(B_in, n);

	VectorXf y_hat; 
	y_hat = Mat_eig * B;
	//y_mean = Vec_eig.array() - Vec_eig.array().mean();

	Map<VectorXf> (Out_hat, y_hat.size()) = y_hat;
    	//Map<VectorXf> (Out_mean, y_mean.size()) = y_mean;

}



VectorXi setdiff1d (VectorXi vec1, VectorXi vec2) {

        vector<int> v;

        sort(vec1.data(), vec1.data()+vec1.size());
        sort(vec2.data(), vec2.data()+vec2.size());

        set_difference (vec1.data(), vec1.data()+vec1.size(), vec2.data(), vec2.data()+vec2.size(), back_inserter(v));

        Map<VectorXi> vec3(v.data(),v.size());

        return vec3;
}

/*double pearson1 (VectorXf vec1, VectorXf vec2) {


        VectorXd vec1_d = vec1.cast <double>();
        VectorXd vec2_d = vec2.cast <double>();

        gsl_vector_view gsl_x = gsl_vector_view_array( vec1_d.data(), vec1_d.size());
        gsl_vector_view gsl_y = gsl_vector_view_array( vec2_d.data(), vec2_d.size());

        gsl_vector *gsl_v1 =  &gsl_x.vector;
        gsl_vector *gsl_v2 = &gsl_y.vector;
        double r = gsl_stats_correlation (gsl_v1->data, 1, gsl_v2->data, 1, gsl_v1->size);
        return r;
}*/

float pearson1 (const VectorXf& x, const VectorXf& y)
{
  const float num_observations = static_cast<float>(x.size());
  float x_stddev = sqrt((x.array()-x.mean()).square().sum()/(num_observations-1));
  float y_stddev = sqrt((y.array()-y.mean()).square().sum()/(num_observations-1));
  float numerator = ((x.array() - x.mean() ) * (y.array() - y.mean())).sum() ;
  float denomerator = (num_observations-1)*(x_stddev * y_stddev);
  return numerator / denomerator;
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



/*float pearson (float *vec1, float *vec2, int m) {


	//Map<VectorXf> Vec1(vec1, m);
	//Map<VectorXf> Vec2(vec2, m);       

        //gsl_vector_view gsl_x = gsl_vector_view_array( Vec1.data(), Vec1.size());
        //gsl_vector_view gsl_y = gsl_vector_view_array( Vec2.data(), Vec2.size());
	
	gsl_vector_const_view gsl_x = gsl_vector_const_view_array( &vec1[0], m );
	gsl_vector_const_view gsl_y = gsl_vector_const_view_array( &vec2[0], m );
        //gsl_vector *gsl_v1 =  &gsl_x.vector;
        //gsl_vector *gsl_v2 = &gsl_y.vector;

        //float r = gsl_stats_correlation (gsl_v1->data, 1, gsl_v2->data, 1, gsl_v1->size);

	float r = gsl_stats_correlation (gsl_x.vector.data, 1, gsl_y.vector.data, 1, m);
        //cout << "r=" << r; 
        return r;
}*/


/*VectorXf where_neq (VectorXf vec, int zero) {
	vector<float> v;

	for (int i=0; i<vec.size(); i++)
		if (vec(i) != zero) {v.push_back(i);}

	Map<VectorXf> vec1(v.data(),v.size());
	return vec1;
}*/

VectorXi where (VectorXf vec, float value) {
        vector<int> v;

        for (int i =0; i < vec.size(); i++)
                if (vec(i) == value) {v.push_back(i);}

         Map<VectorXi> vec1(v.data(),v.size());
        return vec1;
}

VectorXf where_neq (VectorXf vec, int zero) {
        vector<float> v;

        for (int i=0; i<vec.size(); i++)
                if (vec(i) != zero) {v.push_back(i);}

        Map<VectorXf> vec1(v.data(),v.size());
        return vec1;
}

VectorXf intersect1d (VectorXf vec1, VectorXf vec2) {

        vector<float> v;

        sort(vec1.data(), vec1.data()+vec1.size());
        sort(vec2.data(), vec2.data()+vec2.size());

        set_intersection (vec1.data(), vec1.data()+vec1.size(), vec2.data(), vec2.data()+vec2.size(), back_inserter(v));

        Map<VectorXf> vec3(v.data(),v.size());

        return vec3;
}



/*VectorXf intersect1d (VectorXf vec1, VectorXf vec2) {

        vector<float> v;

        sort(vec1.data(), vec1.data()+vec1.size());
        sort(vec2.data(), vec2.data()+vec2.size());

        set_intersection (vec1.data(), vec1.data()+vec1.size(), vec2.data(), vec2.data()+vec2.size(), back_inserter(v));

        Map<VectorXf> vec3(v.data(),v.size());

        return vec3;
}*/


/*float dense_sweep (float *R2m, float *lamb, int maxBoot, int nMP, int m) {

	//cout << "IN dense sweep\n" << endl;

        Map<VectorXf> R2m_vec(R2m, maxBoot*nMP);
        Map<VectorXf> lamb0(lamb, nMP);
        VectorXf lamb_h(nMP);
	lamb_h = lamb0.cast<float>(); 
        MatrixXf R2m0_m(maxBoot * nMP, 1); 

      // cout << "Before resize\n" << endl;
   
	//R2m0_m << R2m_vec;

	VectorXf R2(R2m_vec.size()); 
	R2 = R2m_vec.unaryExpr(ptr_fun(is_not_NaN));


	//cout << "R2vec = \n" << R2m_vec << endl; 

	for (int i=0; i<maxBoot*nMP;i++)
		R2m0_m(i,0) = R2(i);
  
        R2m0_m.resize(maxBoot, nMP);
        //cout << "R2m0 = " << R2m0_m << endl;
   
        VectorXf Mt(nMP);
        vector <int> vec;
        //cout << "before MT\n" << endl;        
 
        Mt = R2m0_m.colwise().mean() * 1e4;
        transform (Mt.data(), Mt.data()+Mt.size(), Mt.data(), floorObject) ;
        //Lids = num.where(Mt, Mt.maxCoeff());

	//cout << "Mt = " << Mt << endl; 

       //cout << "Before for loop\n" << endl;
 
	//cout << "MT max coeff = " << Mt.maxCoeff()  << endl;
 
	for (int i=0;i<Mt.size(); i++) {
		if (Mt(i) == Mt.maxCoeff())
			vec.push_back(i);	
	}

	//for(int i=0; i<vec.size(); i++)
	//	cout << "vec = " << vec[i] << endl;
 
	Map<VectorXi> Lids(vec.data(),vec.size());

        //cout << "Before Result stage\n" << endl;
	float s = lamb_h(Lids(floor(Lids.size()/2)));
	//cout << "dv from dense = "  << s << endl; 
        return s;
}*/ 

VectorXi where_eq (VectorXf vec, float value) {
        vector<int> v;

        for (int i =0; i < vec.size(); i++)
                if (vec(i) == value) {v.push_back(i);}

         Map<VectorXi> vec1(v.data(),v.size());
        return vec1;
}

float dense_sweep (float *R2m, float *lamb, int Boot) {
	Map<VectorXf> R2m_vec(R2m, Boot);
        Map<VectorXf> lamb0(lamb, Boot);
	VectorXi Lids; 

	VectorXf R2(R2m_vec.size()); 
        R2 = R2m_vec.unaryExpr(ptr_fun(is_not_NaN))*1e4;
	
	float tmp;
	float s; 
	tmp = R2.maxCoeff(); 
	Lids = where_eq(R2, tmp);
	int si = Lids.size() / 2; 
	s = (float) lamb0(Lids(si)); 

	return s;


}


float get_dv (float s) { float ds; return (ds = pow(10, floor(log10(s)-1)));}

void get_support (MatrixXf B_eig, float *sprt_in, int nMP, int nbootS, int n) { 
  
        //Map<MatrixXf> B_eig(B_in, nMP*nbootS, n);
	MatrixXf sprt_h(nMP, n); 

        sprt_h.fill(NAN);
        VectorXf intv, tmp_ids; 

 	for (int i=0; i<nMP; i++) {		
        	for (int j=0; j<nbootS; j++) {
			tmp_ids = where_neq(B_eig.row(j),0);

                	if (j == 0)
                		intv = tmp_ids;

                	intv = intersect1d(intv,tmp_ids); 
		}

        	sprt_h.row(i).head(intv.size()) = intv;	 
   	}

	Map<MatrixXf> (sprt_in, sprt_h.rows(), sprt_h.cols()) = sprt_h;
}


void get_train (float *Mat, float *Mat_train, float *Vec_train, float *Mat_test, float *Vec_test, int m, int n, int train) {

	Map<MatrixXf> Mat_e(Mat, m, n+1);
	//Map<VectorXf> Vec_e(Vec,m);   

	MatrixXf Mat_t, Mat_tst; 
	VectorXf Vec_t, Vec_tst; 

	//Mat_t = Mat_e.topRows(train); 
	Mat_t = Mat_e.topLeftCorner(train, n); 
	//Vec_t = Vec_e.head(train);
	Vec_t = Mat_e.topLeftCorner(train, n+1).rightCols(1); 


	//X_test = Mat_eig.bottomRows(m - train);
    	Mat_tst = Mat_e.block(train, 0, m-train, n);
    	//y_test = Vec_eig.tail(m - train);
    	Vec_tst = Mat_e.block(train, 0, m-train, n+1).rightCols(1);
	
	Map<MatrixXf> (Mat_train, Mat_t.rows(), Mat_t.cols()) = Mat_t; 
	Map<VectorXf> (Vec_train, Vec_t.size()) = Vec_t.array()-Vec_t.array().mean();
	Map<MatrixXf> (Mat_test, Mat_tst.rows(), Mat_tst.cols()) = Mat_tst; 
        Map<VectorXf> (Vec_test, Vec_tst.size()) = Vec_tst.array()-Vec_tst.array().mean();

}

void get_estimate(float *Mat, float *Mat_tr, float *Vec_tr, float *Mat_tst, float *Vec_tst, float *Mat_T, float *Vec_T,  int m, int n, int CV, int train_row) {

	Map<MatrixXf> Mat_e(Mat, m, n+1);
	MatrixXf Mat_train, Mat_ts, Mat_CV;
	VectorXf Vec_train, Vec_ts, Vec_CV; 
	
        Mat_train = Mat_e.topLeftCorner(train_row, n);
        Vec_train = Mat_e.topLeftCorner(train_row, n+1).rightCols(1);
	
	Mat_ts = Mat_e.block(train_row, 0, CV-train_row, n);
        Vec_ts = Mat_e.block(train_row, 0, CV-train_row, n+1).rightCols(1);
	
	Mat_CV = Mat_e.block(CV, 0, m-CV, n);
	Vec_CV = Mat_e.block(CV, 0, m-CV, n+1).rightCols(1); 

        Map<MatrixXf> (Mat_tr, Mat_train.rows(), Mat_train.cols()) = Mat_train;
        Map<VectorXf> (Vec_tr, Vec_train.size()) = Vec_train.array()-Vec_train.array().mean();

	Map<MatrixXf> (Mat_tst, Mat_ts.rows(), Mat_ts.cols()) = Mat_ts;
        Map<VectorXf> (Vec_tst, Vec_ts.size()) = Vec_ts.array()-Vec_ts.array().mean();

	Map<MatrixXf> (Mat_T, Mat_CV.rows(), Mat_CV.cols()) = Mat_CV;
        Map<VectorXf> (Vec_T, Vec_CV.size()) = Vec_CV.array()-Vec_CV.array().mean();

}

void get_estimate2(float *Mat, float *Mat_L, float *Vec_L, float *Mat_T, float *Vec_T,  int m, int n, int CV) {

        Map<MatrixXf> Mat_e(Mat, m, n+1);
        MatrixXf Mat_T_,  Mat_L_;
        VectorXf Vec_T_, Vec_L_;

	Mat_L_ = Mat_e.topLeftCorner(CV, n);
        Vec_L_ = Mat_e.topLeftCorner(CV, n+1).rightCols(1);

	Mat_T_ = Mat_e.block(CV, 0, m-CV, n);
        Vec_T_ = Mat_e.block(CV, 0, m-CV, n+1).rightCols(1); 

        Map<MatrixXf> (Mat_L, Mat_L_.rows(), Mat_L_.cols()) = Mat_L_;
        Map<VectorXf> (Vec_L, Vec_L_.size()) = Vec_L_.array()-Vec_L_.array().mean();

        Map<MatrixXf> (Mat_T, Mat_T_.rows(), Mat_T_.cols()) = Mat_T_;
        Map<VectorXf> (Vec_T, Vec_T_.size()) = Vec_T_.array()-Vec_T_.array().mean();

}



void average (float *In, int max, int nM, int y, int size) {

	if (y==1) {
	    Map<VectorXf> R (In, max*nM);
	    R /= size; 
	    Map<VectorXf> (In, R.size()) = R; 	 
	}
 
	else {
	    Map<MatrixXf> B (In, max*nM, y); 
	    B /= size; 
	    Map<MatrixXf> (In, B.rows(), B.cols()) = B; 
	}

}

/*void average_v (float *In, int rnd, int y, int size) {
	
	if (y==1) {
            Map<VectorXf> R (In, rnd);
            R /= size;
            Map<VectorXf> (In, R.size()) = R;
        }

        else {
            Map<MatrixXf> B (In, rnd, y);
            B /= size;
            Map<MatrixXf> (In, B.rows(), B.cols()) = B;
        }

}*/ 


void get_random_rows(float *In, float *Out1, float *Out2, int rows, int m_train, int n, int L_g, int T_g, MPI_Comm comm) {

	int rank_h, size_h;
	MPI_Comm_rank(comm, &rank_h);
	MPI_Comm_size(comm, &size_h);
 
	Map<MatrixXf> In_eig(In, rows, n); 
	MatrixXf Out1_eig;
        MatrixXf Out2_eig;
#if DEBUG
	ofstream myfile121 ("data/B_In.dat");
        if (myfile121.is_open())
        {
                myfile121 << In_eig;
                myfile121.close();
        }
#endif
		


 	 VectorXi cv_ids, boot_ids;
        if (rank_h == 0) {
                cv_ids.setLinSpaced(L_g, 0, L_g);
                random_shuffle (cv_ids.data(), cv_ids.data()+cv_ids.size());
		boot_ids.setLinSpaced(m_train, 0, m_train);
		random_shuffle (boot_ids.data(), boot_ids.data()+boot_ids.size());
        }

	MPI_Bcast(cv_ids.data(), cv_ids.size(), MPI_INT, 0, comm); 
	MPI_Bcast(boot_ids.data(), boot_ids.size(), MPI_INT, 0, comm);
	
	for (int i=0; i<m_train; i++) {
		Out1_eig.row(i) = In_eig.row(cv_ids(boot_ids(i)));
	}

	for (int i=m_train; i<cv_ids.size()-m_train; i++) {
		Out2_eig.row(i) = In_eig.row(cv_ids(i));
	}
#if DEBUG
	ofstream myfile2 ("data/B_train.dat");
        if (myfile2.is_open())
        {
                myfile2 << Out1_eig;
                myfile2.close();
        }
	ofstream myfile12 ("data/B_test.dat");
        if (myfile12.is_open())
        {
                myfile12 << Out2_eig;
                myfile12.close();
        }
#endif


	Map<MatrixXf> (Out1, Out1_eig.rows(),Out1_eig.cols()) = Out1_eig;
	Map<MatrixXf> (Out2, Out2_eig.rows(),Out2_eig.cols()) = Out2_eig;

}
 
