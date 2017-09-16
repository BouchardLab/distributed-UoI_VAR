#include <math.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include "bins.h"

int bin_size_1D(int i, int N, int M) {
  // range of size N divided in M bins
  // return size of bin i
  int b = N / M;
  return (i < N % M) ? b+1 : b;
}

int bin_coord_1D(int i, int N, int M) {
  // range of size N divided in M bins
  // returns which bin owns i
  int j,k=0;
  int b = N / M;
  for (j=0; j<M; j++) {    
    k += (j < N % M) ? b+1 : b;
    if (i < k) return j;
  }
}

int bin_index_1D(int i, int N, int M) {
  // range of size N divided in M bins
  // returns index of i within the bin
  int j,s,k=0;
  int b = N / M;
  for (j=0; j<M; j++) {
    s = (j < N % M) ? b+1 : b;
    k += s;
    if (i < k) return i-k+s;
  }
}

void bin_range_1D(int i, int N,int M, int *start, int *end) {
  // range of size N divided in M bins
  // compute the start, end index of the range of bin i
  int j,s,k=0;
  int b = N / M;
  for (j=0; j<i+1; j++) {
    s = (j < N % M) ? b+1 : b;
    k += s;
    if (i < k) {
      *start = k-s;
      *end   = k;
    }
  }
}

/*  Only effective if N is much smaller than RAND_MAX */
void shuffle(int *array, size_t n) {
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
