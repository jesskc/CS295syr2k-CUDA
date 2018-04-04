// This is an incomplete implementation as I do not completely understand how to turn the given syr2k code into a CUDA compatible program.


#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include "timer.h"

// Defines
#define min(x,y) ( ((x) < (y))? (x) : (y))
#define GridWidth 60
#define BlockWidth 128

// Variables for host and device arrays.
double* h_A; 
double* h_B; 
double* h_C; 
double* d_A; 
double* d_B; 
double* d_C; 

// Utility Functions
void Cleanup(int b);
void checkCUDAError(const char *msg);

// Original kernel_syr2k 
/* void kernel_syr2k(int N, int M,
    double C[ N][N],
    double A[ N][M],
    double B[ N][M])
{
  int i, j, k;

  for (i = 0; i < N; i++) {
    for (k = 0; k < M; k++) {
      for (j = 0; j < N; j++) {
        C[i][j] += A[j][k] * B[i][k] + B[j][k] * A[i][k];
      }
    }
  }
}*/

__global__ void kernel_syr2k(int N, int M, double C[N][M],
	double A[N][M], double B[N][M], int values_per_thread) {
	
	int blockStartIndex = blockIdx.x * blockDim.x * values_per_thread;
	int threadStartIndex = blockStartIndex + (threadIdx.x * N);
	int threadEndIndex = threadStartIndex + N;
	int i, j, k, l;

    for (i = 0; i < N; i++) {
      for (k = 0; k < M; k++) {
        for (j = 0; j < N; j++) {
          for (l = threadStartIndex; l < threadEndIndex; l++) {
            C[i][j] += A[j][k] * B[i][k] + B[j][k] * A[i][k];
          } // end 4th for
        } // end 3rd for
      } // end 2nd for
    } // end 1st for 	 
} // end __global__ kernel_syr2k

///
/// vecAddKernel00.cu
/// For CSU CS575 Spring 2011
/// Instructor: Wim Bohm
/// Based on code from the CUDA Programming Guide
/// By David Newman
/// Created: 2011-02-16
/// Last Modified: 2011-02-16 DVN
///
/// This Kernel adds two Vectors A and B in C on GPU
/// without using coalesced memory access.
/// 

/*__global__ void AddVectors(const float* A, const float* B, float* C, int N)
{
    int blockStartIndex  = blockIdx.x * blockDim.x * N;
    int threadStartIndex = blockStartIndex + (threadIdx.x * N);
    int threadEndIndex   = threadStartIndex + N;
    int i;

    for( i=threadStartIndex; i<threadEndIndex; ++i ){
        C[i] = A[i] + B[i];
    }
}*/

// Initialize host arrays.
void init_array(int N, int M,
  double C[N][N],
  double A[N][M],
  double B[N][M])
{
  int i, j;

  for (i = 0; i < N; i++)
    for (j = 0; j < M; j++) {
      h_A[i][j] = (double) (i*j%N) / N;
      h_B[i][j] = (double) (i*j%M) / M;
    }
  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      h_C[i][j] = (double) (i*j%N) / M;
}



void print_array(int N,
   double C[N][N])
{
  int i, j;

  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "C");
  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++) {
 if ((i * N + j) % 20 == 0) fprintf (stderr, "\n");
 fprintf (stderr, "%0.2lf ", C[i][j]);
    }
  fprintf(stderr, "\nend   dump: %s\n", "C");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}


int main(int argc, char** argv)
{

  int N, M, values_per_thread;
  
  // Tell CUDA how big to make the grid and thread blocks.
  dim3 dimGrid(GridWidth);                    
  dim3 dimBlock(BlockWidth); 
     
  struct timeval t_start;
  struct timeval t_end;
  double etime;

  double* C;
  double* A;
  double* B;

  // Parse arguments.
  if (argc != 4) {
    printf("usage ./syr2k N M ValuesPerThread\n");
    printf("ValuesPerThread is the number of values added by each thread.\n");
    printf("Total array size is 128 * 60 * this value.\n");
    exit(0);
    } else {
      N = atoi(argv[1]);
      M = atoi(argv[2]);
      values_per_thread = atoi(argv[3]);
    }    
    // else {
    //  sscanf(argv[1], "%d", &ValuesPerThread);
    //}   
  

  // Allocate pointers to arrays in host memory.
  C = (double*)malloc(N*N * sizeof(double));
  A = (double*)malloc(N*M * sizeof(double));
  B = (double*)malloc(N*M * sizeof(double));

  // Initialize arrays in host memory.
  init_array (N, M, *((double(*)[N][N])C), *((double(*)[N][M])A), *((double(*)[N][M])B));

  // Allocate arrays in device memory.
  cudaError_t error;
  error = cudaMalloc((void**)&d_A, size);
  if (error != cudaSuccess) Cleanup(false);
  error = cudaMalloc((void**)&d_B, size);
  if (error != cudaSuccess) Cleanup(false);
  error = cudaMalloc((void**)&d_C, size);
  if (error != cudaSuccess) Cleanup(false);

  gettimeofday (&t_start, NULL);
  // Initialize timer  
  initialize_timer();
  start_timer();
  
  // Copy host arrays h_A and h_B to device vectores d_A and d_B
  error = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  if (error != cudaSuccess) Cleanup(false);
  error = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
  if (error != cudaSuccess) Cleanup(false);

  // Invoke kernel.
  kernel_syr2k<<<dimGrid, dimBlock>>>(N, M, d_A, d_B, d_C, values_per_thread);
  error = cudaGetLastError();
  if (error != cudaSuccess) Cleanup(false);
  cudaThreadSynchronize();
  
  // kernel_syr2k  (N, M, *((double(*)[N][N])C), *((double(*)[N][M])A), *((double(*)[N][M])B));

  gettimeofday (&t_end, NULL);

  etime = t_end.tv_sec - t_start.tv_sec + 
        (t_end.tv_usec - t_start.tv_usec) * 1.0e-6;

  print_array(N, *((double(*)[N][N])C));

  printf("execution time=%lf\n", etime);
  
  // Compute elapsed time 
  stop_timer();
  double time = elapsed_time();

  // Compute floating point operations per second.
  int nFlops = N;
  double nFlopsPerSec = nFlops/time;
  double nGFlopsPerSec = nFlopsPerSec*1e-9;

  // Compute transfer rates.
  int nBytes = 3*4*N; // 2N words in, 1N word out
  double nBytesPerSec = nBytes/time;
  double nGBytesPerSec = nBytesPerSec*1e-9;

  // Report timing data.
  printf( "Time: %lf (sec), GFlopsS: %lf, GBytesS: %lf\n", 
           time, nGFlopsPerSec, nGBytesPerSec);
     
  // Copy result from device memory to host memory
  error = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) Cleanup(false);

  // Clean up and exit.
  Cleanup(true);
}

void Cleanup(int noError) {  // simplified version from CUDA SDK
    cudaError_t error;
        
    // Free device vectors
    if (d_A)
        cudaFree(d_A);
    if (d_B)
        cudaFree(d_B);
    if (d_C)
        cudaFree(d_C);

    // Free host memory
    if (h_A)
        free(h_A);
    if (h_B)
        free(h_B);
    if (h_C)
        free(h_C);
        
    error = cudaThreadExit();
    
    if (!noError || error != cudaSuccess)
        printf("cuda malloc or cuda thread exit failed \n");
    
    fflush( stdout);
    fflush( stderr);

    exit(0);
}

void checkCUDAError(const char *msg)
{
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) 
    {
      fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err) );
      exit(-1);
    }                         
}
