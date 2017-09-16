#include <iostream>
#include <fstream>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <eigen3/Eigen/Dense>
#include "lasso_admm_MPI.h"
#define EIGEN_DEFAULT_TO_ROW_MAJOR


using namespace std; 
using namespace Eigen; 

/*template <typename Derived1, typename Derived2>
void product(MatrixBase<Derived1>& dst, const MatrixBase<Derived2>& src)
{
#ifndef EIGEN_DONT_VECTORIZE
#define EIGEN_DONT_VECTORIZE
        dst.template selfadjointView<Upper>().rankUpdate(src);
#undef EIGEN_DONT_VECTORIZE
#endif

}*/

/*template <typename Derived1>
void copyLowerTriangularPart(MatrixBase<Derived1>& dst)
{
  	dst.template triangularView<Lower>() = dst.transpose();
}
*/ 

template <typename Derived1, typename Derived2>
void product(MatrixBase<Derived1>& dst, const MatrixBase<Derived2>& src)
{
        dst.template triangularView<Upper>() = src.template triangularView<Upper>() * src.transpose();
}

double *lasso_admm (double *A_in, int m, int n,  double *b_in, double lambda, MPI_Comm comm)
 {
        const int MAX_ITER  = 50;
        const double RELTOL = 1e-2;
        const double ABSTOL = 1e-3;
        /*
         * Some bookkeeping variables for MPI. The 'rank' of a process is its numeric id
         * in the process pool. For example, if we run a program via `mpirun -np 4 foo', then
         * the process ranks are 0 through 3. Here, N and size are the total number of processes 
         * running (in this example, 4).
         */

        int rank;
        int size;

        MPI_Comm_rank(comm, &rank); // Determine current running process
        MPI_Comm_size(comm, &size); // Total number of processes
        double N = (double) size;             // Number of subsystems/slaves for ADMM

        /* Read in local data */

        int skinny;           // A flag indicating whether the matrix A is fat or skinny


	/*
	 * For simple data where m is not so greater than n, then the matrix is copied
	 * as such and estimates are calculated. This will not gain anything from mpi_lasso.
	 * But for very large dataset the input matrix is chunked and individual estimates 
	 * are calculated by the each core and communicating through Allreduce, thereby
	 * distributing the problem across admm_cores. This strategy will greatly boost the computation time. 
	 */

	Map<Matrix<double, Dynamic, Dynamic, RowMajor> > A (A_in, m, n);
	Map<VectorXd> b (b_in, m);

      
	skinny = (m >= n);

        /*
         * These are all variables related to ADMM itself. There are many
         * more variables than in the Matlab implementation because we also
         * require vectors and matrices to store various intermediate results.
         * The naming scheme follows the Matlab version of this solver.
         */

        double rho = 1.0;

         VectorXd   x   =  VectorXd::Zero(n);
         VectorXd   u   =  VectorXd::Zero(n);
         VectorXd   z   =  VectorXd::Zero(n);
         VectorXd   y   =  VectorXd::Zero(n);
         VectorXd   r   =  VectorXd::Zero(n);
      	 VectorXd   zprev =  VectorXd::Zero(n);
         VectorXd   zdiff =  VectorXd::Zero(n);
         VectorXd   q   =  VectorXd::Zero(n);
         VectorXd   w   =  VectorXd::Zero(n);
         VectorXd   Aq  =  VectorXd::Zero(m);
         VectorXd   p   =  VectorXd::Zero(m);
         VectorXd   Atb         =  VectorXd::Zero(n);

        double send[3]; // an array used to aggregate 3 scalars at once
        double recv[3]; // used to receive the results of these aggregations

        double nxstack  = 0;
        double nystack  = 0;
        double prires   = 0;
        double dualres  = 0;
        double eps_pri  = 0;
        double eps_dual = 0; 

        Atb = A.transpose()*b; // Atb = A^T b

         /*
         * The lasso regularization parameter here is just hardcoded
         * to 0.5 for simplicity. Using the lambda_max heuristic would require
         * network communication, since it requires looking at the *global* A^T b.
         */

        //double lambda = 0.5;
        /*if (rank == 0) {
                printf("using lambda: %.4f\n", lambda);
        }*/

         LLT< MatrixXd> chol;
         MatrixXd L;
         MatrixXd M, M_t;

	if (rank == 0)
		cout << "before the major if-else" << endl; 

        if(skinny) {
                MatrixXd AtA(n,n);
                AtA = A.transpose()*A;
                MatrixXd rhoI(n,n);
                rhoI.setIdentity(n,n);
 
                L = (AtA+rho*rhoI);
                chol = L.llt();		

        } else {
                // MatrixXd AAt(m,m); 
                //AAt = A*A.transpose();

                 MatrixXd eye(m,m);
                eye.setIdentity(m,m);
		double Attime = MPI_Wtime(); 
		//MatrixXd AtA(n,n);
		//AtA = A*A.transpose();
		MatrixXd AAt = MatrixXd::Zero(m,m);
		double Atatime = MPI_Wtime();
		 /* AAt = A * A.transpose(); */
		AAt.noalias() = A * A.transpose();	

		if (rank == 0)
			cout << "AtA optim time " << MPI_Wtime() - Atatime << endl ;
		double l_time = MPI_Wtime();
                L = (AAt + (1/rho)*eye);
		double l_end = MPI_Wtime() - l_time;
                if (rank==0)
                	cout << "L-else time " << l_end << endl;
                chol = L.llt();
        }	 

	//if(rank == 0) {cout << "passed skinny" << endl; }

	/* Main ADMM solver loop */

        int iter;
        /*if (rank == 0) {
                printf("%3s %10s %10s %10s %10s %10s\n", "#", "r norm", "eps_pri", "s norm", "eps_dual", "objective");
        }*/

        for (iter=49; iter--; ) {

                /* u-update: u = u + x - z */

                u += x - z;

                /* x-update: x = (A^T A + rho I) \ (A^T b + rho z - y) */
                q = Atb+rho*(z - u);

		//if(rank == 0) {cout << "before iter skinny" << endl; }

                if (skinny) {
                        /* x = U \ (L \ q) */
                        x = chol.solve(q);
                } else {
                        /* x = q/rho - 1/rho^2 * A^T * (U \ (L \ (A*q))) */
                        //Aq = A * q;
                        //p = chol.solve(Aq);

                        x = (A.transpose() * chol.solve(A*q)) * (-1/(rho*rho));
                        q = q * 1/rho;
                        x = x + q;
                }

		//if(rank == 0) {cout << "passed iter skinny" << endl; }

                /*
                 * Message-passing: compute the global sum over all processors of the
                 * contents of w and t. Also, update z.
                 */

                w = x+u;

                send[0] = r.transpose()*r; //r * r;
                send[1] = x.transpose()*x; //x * x;
                send[2] = u.transpose()*u;
                send[2] /= rho*rho;

		zprev = z;

                // could be reduced to a single Allreduce call by concatenating send to w
                MPI_Allreduce(w.data(), z.data(),  n, MPI_DOUBLE, MPI_SUM, comm);
                MPI_Allreduce(send,    recv,     3, MPI_DOUBLE, MPI_SUM, comm);

                prires  = sqrt(recv[0]);  /* sqrt(sum ||r_i||_2^2) */
                nxstack = sqrt(recv[1]);  /* sqrt(sum ||x_i||_2^2) */
                nystack = sqrt(recv[2]);  /* sqrt(sum ||y_i||_2^2) */

                z /= N;

		double k = lambda/(N*rho);
	
		/* soft thresholding inlined */ 
                for (int i=0; i < z.size(); i++) {
                if (z(i) > k)  	    { z(i) = z(i) - k; }
                else if (z(i) < -k) { z(i) = z(i) + k; }
                else                { z(i) = 0; }
                }

                 /* Termination checks */

                /* dual residual */
                zdiff = z - zprev;

                dualres = sqrt(N) * rho * zdiff.norm();

                /* compute primal and dual feasibility tolerances */
                eps_pri = sqrt(n*N)*ABSTOL + RELTOL * fmax(nxstack, sqrt(N)*z.norm());
                eps_dual = sqrt(n*N)*ABSTOL + RELTOL * nystack;

                /*if (rank == 0) {
                        printf("%3d %10.4f %10.4f %10.4f %10.4f %10.4f\n", iter,
                                        prires, eps_pri, dualres, eps_dual, objective(A, b, lambda, z));
                }*/

                if (prires <= eps_pri && dualres <= eps_dual) {
                        break;
                }

                 /* Compute residual: r = x - z */
                r = x-z;

        }

	double *z_out;
 	z_out = (double*)malloc(n * sizeof(double));	
	Map<VectorXd> (z_out, z.size()) = z;
	return z_out; 
}


