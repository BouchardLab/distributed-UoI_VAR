#!/usr/bin/env python
import os,pdb,h5py
import numpy as np
from optparse import OptionParser
import scipy.io as sio
from scipy.linalg import block_diag
from scipy.stats import uniform
import sys

"""
Authors : Mahesh Balasubramanian (adapted from Trevor Ruiz's matlab code)
Date    : 09/27/2017
"""

def main():
    usage = '%prog [options]'
    parser = OptionParser(usage)
    parser.add_option("--p",type="int",default=354,\
        help="dimension (think size of MEA array)")
    parser.add_option("--n_chunks",type="int",default=5,\
        help="Number of chunks")
    parser.add_option("--T",type="int",default=100000,\
        help="Chunk size")
    parser.add_option("--store",action="store_true",dest="store",\
        help="store results to file")
    parser.add_option("--saveAs",type="string",default='hdf5',\
        help="File format to store the data. Options: hdf5(default), mat, txt")
    parser.add_option("--path",type="string",default=os.getcwd(),\
        help="path to store the results (default: current directory)")
    parser.add_option("--seed",type="int",default=np.random.randint(9999),\
        help="seed for generating pseudo-random numbers (default: random seed)")

    (options, args) = parser.parse_args()

    if options.store:
        store=True
    else:
        store=False
    
    Model_VAR_Create(p=options.p,n_chunks=options.n_chunks,T=options.T,\
                    store=store,path=options.path,saveAs=options.saveAs,\
                    seed=options.seed)


def sim_var(p, T):

    """
    Matrix Construction
    -----------------------
    Input: 
    
    p	: dimension of the response vector
    T 	: length of the simulated series

    Output: 

    output is a cell array, with first element simulated data, second element
    coefficient matrix, third element gaussian noise covariance, last element
    indices of nonzero coefficients in vectorized coefficient matrix

    """
    pctNZ  = 1/p 		#percent nonzero entries in transition matrix
    numNZ = round(p*p*pctNZ) 	#number nonzero entries

    # fix nonzero terms in coefficient matrix

    u_smp = np.random.uniform(0.,1.,size=(numNZ,1)) 					#uniform sample
    tht = 1 										#rate parameter for coefficient distribution
    u_smp_trns = -np.log(1 - u_smp*(1 - np.exp(-8*tht)))					#inverse cdf transform uniform sample
    NZcoef = ((2*np.random.binomial(1., 0.5, size=(numNZ, 1))) - 1)*(8 - u_smp_trns)		#randomly choose sign
    	

    #randomly allocate nonzero coefficients to matrix.

    NZix = np.random.randint(p**2, size=(p,1)) 						#indices
    A_vec = np.zeros((p**2))								#storage vector
    A_vec[NZix] = NZcoef								#assign	
    A_mx = A_vec.reshape(p,p)								#arrange as matrix

    #recondition matrix for stability
    
    A_eigV = np.linalg.eig(A_mx)[0]
    coef_mx = A_mx / (np.max(np.abs(A_eigV)) + 0.1)				#coefficient matrix

    #simulate data recursively

    n_initialize = 500
    N = n_initialize + T

    #construct block-diagonal covariance matrix

    cov_parm = 0.7 									#value for off-diagonal covariance
    blk_sz = 3										#block size	
    nblk   = np.floor(p/blk_sz)								#number of blocks
    lastblk_sz = p - nblk*blk_sz							#'remainder block'
    blk = (1 - cov_parm) * np.eye(blk_sz) + cov_parm					#make block
    cov_mx = block_diag(np.kron(np.eye(nblk), blk), (1 - 0.7) * np.eye(lastblk_sz) + 0.7)  #put them together

    #simulate Gaussian innovations

    innov_mx = np.random.multivariate_normal(np.zeros((p)), cov_mx, N)
    
    #recursively construct series

    sim_series = np.zeros((N, p))
    sim_series[0,:] = innov_mx[0, :]
 
    for j in range(1, N):
	sim_series[j, :] = sim_series[j-1, :].dot(coef_mx.T) + innov_mx[j,:]    

    #subset by removing burn in

    data_mx = sim_series[(n_initialize):N, :]

    return data_mx, coef_mx, cov_mx, data_mx[T-1, :]



def continue_var(p, T, A_mx, cov_mx, init_val):

    """
    Matrix Construction
    -----------------------
    Input: 
    p		: int  dimension of the response vector
    T 		: int  Chunk size
    A_mx	: randomly allocated non-zero coefficient matrix 
    cov_mx 	: covariance matrix
    init_val	: data matrix from previous step

    """
   								
    d = int(np.log(A_mx.shape[1])/np.log(p))					#var order
    p = A_mx.shape[0]
    N = T+1 									#total length

    #simulate Gaussian innovations
    
    innov_mx = np.random.multivariate_normal(np.zeros((p)), cov_mx, N)

    #recursively construct series

    sim_series = np.zeros((N,p))
    sim_series[0, :]  = init_val
    
    for j in range(1,N):
	ar_lags = np.zeros((d,p))
	for k in range(1,d):
		ar_lags[k, :] = sim_series[j-k, :] * A_mx.T

	sim_series[j,:] = np.sum(ar_lags, 0) + innov_mx[j, :]

    #subset by removing burn in

    data_mx = sim_series[1:N, :]

    return data_mx, data_mx[T-1, :]




def Model_VAR_Create(p,n_chunks,T,store=False,path=os.getcwd(),saveAs='hdf5',seed=np.random.randint(9999)):

    """	
    Model_VAR_Create
    ______________________________
    
    Creates VAR data samples. 
    
    Input: 
    -p 		: Dimensions or the number of samples.
    -n_chunks	: Number of chunks 
    -T 		: Chunk size usually default to 100000
    -store      : store results to file
    -saveAs     : file format to store the data; options: hdf5(default),
                        mat, txt
    -path       : path to store the results (default: current directory)


    Output:
    - sim_data	: This is actually a p*N design matrix created
    - coef_mx	: Coefficient matrix
    - cov_mx	: Convariance matrix

    """

    np.random.seed(seed)
    N = n_chunks * T #total length of series to simulate
    A_mx = np.zeros((p,p))
    cov_mx = np.zeros((p,p))
    init_val = np.zeros((p)) 
    sim_data = np.zeros((N, p))
    
    for chunk_num in range(0,n_chunks):  #treat the first iteration differently.
	if chunk_num==0:
		sim_data[(chunk_num*T):(chunk_num*T)+T, :], A_mx, cov_mx, init_val = sim_var(p,T)
	else:
		sim_data[(chunk_num*T):(chunk_num*T)+T, :], init_val = continue_var(p, T, A_mx, cov_mx, init_val)

    	print(np.max(init_val))

    if store:
        name = '%s_%i_%i'%(T,p,n_chunks)
        if saveAs=='hdf5':
            with h5py.File('%s/Model_Data_%s.h5'%(path,name),'w') as f:
		g1 = f.create_group('Coef_Cov_mx')
                g1.create_dataset(name='coef_mx',data=A_mx,dtype=np.float64,\
                                shape=A_mx.shape,compression="gzip")
                g1.create_dataset(name='cov_mx',data=cov_mx,dtype=np.float64,\
                                shape=cov_mx.shape,compression="gzip")
		g = f.create_group('data')
                g.create_dataset(name='sim_data',data=sim_data,dtype=np.float64,\
                                shape=sim_data.shape,compression="gzip")

        elif saveAs=='txt':
            np.savetxt('%s/sim_data_%s.txt'%(path,name),sim_data.astype(np.float32))
            np.savetxt('%s/A_mx_%s.txt'%(path,name),A_mx.astype(np.float32))
            np.savetxt('%s/cov_mx_%s.txt'%(path,name),cov_mx.astype(np.float32))

        elif saveAs=='mat':
            sio.savemat('%s/Model_Data_%s.mat'%(path,name),\
                        {'sim_data'    : sim_data.astype(np.float32),\
                         'A_mx'    : A_mx.astype(np.float32),\
                         'cov_mx' : cov_mx.astype(np.float32)})

        print '\nData Model:'
        print '\t* No covariates:\t%i'%sim_data.shape[1]
        print '\t* No samples   :\t%i'%sim_data.shape[0]
        print 'Data stored in %s'%path

    else:
	return data, A_mx, cov_mx

if __name__=='__main__':
    main()
