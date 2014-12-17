/*==============================================================================
 * Program  : gpu_dgemm
 * Revision : 1.0 (2014-08-27)
 * Author   : Carlos Rosales Fernandez [carlos.rosales.fernandez(at)gmail.com]
 *==============================================================================
 * Copyright 2014 Carlos Rosales Fernandez and The University of Texas at Austin
 *
 * This code was originally written with support from the National Science 
 * Foundation under Grant #OCI-1134872
 *
 * This file is part of the Performance Assessment Workbench (PAW).
 * PAW is free software: you can redistribute it and/or modify it under the
 * terms of the GNU GPL version 2 or (at your option) any later version.
 *
 * PAW is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * PAW, in the file COPYING.txt. If not, see <http://www.gnu.org/licenses/>.
 *==============================================================================
 * Test sustained double precision matrix matrix multiplication performance 
 * on GPUs using the CUBLAS DGEMM call for a range of matrix sizes.
 *============================================================================*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "aux.h"
#include "constants.h"

int main( int argc, char **argv )
{
    cublasStatus_t status, statusB, statusC;
    cublasHandle_t handle;
    FILE   *fp, *fp2;
    char   testName[32] = "GPU_DGEMM";
    int    Nsq, count, i, j, deviceID, gpuNum;
    unsigned int size, localSize, NLOOP = NLOOP_MAX, smed = MED_GPU_BLAS_SIZE;
    unsigned int smin = MIN_GPU_BLAS_SIZE, smax = MAX_GPU_BLAS_SIZE;
    const double alpha = 1.0f;
    const double beta  = 1.0f;
    double  *A, *B, *C, *A_d, *B_d, *C_d;
    double tScale = USEC, fpScale = GFLOP;
    double timeMin, tStart, overhead, threshold_lo, threshold_hi;
    double matSize, usedMem, localMax, ops;
    double tElapsed[NREPS];

    // Make sure there is at least one GPU in the system
    cudaGetDeviceCount( &gpuNum );
    if( gpuNum == 0 )
        fatalError( "No GPU devices found. Test Aborted." );

    // Check for user defined limits
    checkEnvGPUBLAS( &NLOOP, &smin, &smed, &smax );
    usedMem = smax*smax*3.0*sizeof(double);
    
    // Initiallize cublas
    status = cublasCreate(&handle);
    if( status != CUBLAS_STATUS_SUCCESS )
        fatalError( "CUBLAS initialization failed.");

    // Select first GPU in the node
    cudaGetDevice( &deviceID );
    cudaSetDevice( deviceID );

    // Check timer overhead in seconds
    timerTest( &overhead, &threshold_lo, &threshold_hi );
    // Open output files and write headers
    fp  = fopen("gpu_dgemm_time.dat","a");
    fp2 = fopen("gpu_dgemm_flops.dat","a");
    printHeaders( fp, fp2, testName, usedMem, overhead, threshold_lo );

    /* Initialize variables */
    Nsq = smax * smax;

    /* Allocate and initialize host arrays */
    srand( SEED );
    A = doubleVector( Nsq );
    B = doubleVector( Nsq );
    C = doubleVector( Nsq );
    if( !A || !B || !C )
        fatalError( "Failed to allocate host memory in GPU SGEMM test." );

    /* Allocate device arrays */
    cudaMalloc( (void **) &A_d, Nsq*sizeof(double) );
    cudaMalloc( (void **) &B_d, Nsq*sizeof(double) );
    cudaMalloc( (void **) &C_d, Nsq*sizeof(double) );
    if( !A_d || !B_d || !C_d )
        fatalError( "Failed to allocate GPU memory in GPU SGEMM test." ); 

    /* Initialize the device matrices with the host matrices */
    status  = cublasSetVector( Nsq, sizeof(double), A, 1, A_d, 1 );
    statusB = cublasSetVector( Nsq, sizeof(double), B, 1, B_d, 1 );
    statusC = cublasSetVector( Nsq, sizeof(double), C, 1, C_d, 1 );
    if( status  != CUBLAS_STATUS_SUCCESS ||
        statusB != CUBLAS_STATUS_SUCCESS ||
        statusC != CUBLAS_STATUS_SUCCESS )
        fatalError( "CUBLAS array initialization failed in GPU SGEMM test.");

    //================================================================
    // Single loop with minimum size to verify that inner loop length  
    // is long enough for the timings to be accurate                     
    //================================================================
    // Warmup processor with a medium size DGEMM
    cublasDgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N, smed, smed, smed,
                  &alpha, A_d, smed, B_d, smed, &beta, C_d, smed );
    if( cublasGetError() != CUBLAS_STATUS_SUCCESS )
        fatalError( "Failed to run warmup cublasSGEMM" );
    // Test is current NLOOP is enough to capture fastest test cases
    cudaDeviceSynchronize();
    tStart = benchTimer();
    for(j = 0; j < NLOOP; j++){
        cublasDgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N, smin, smin, smin, 
                     &alpha, A_d, smin, B_d, smin, &beta, C_d, smin );
    }
    cudaDeviceSynchronize();
    timeMin = benchTimer() - tStart;
    resetInnerLoop( timeMin, threshold_lo, &NLOOP );

    //================================================================
    // Execute test for each requested size                  
    //================================================================
    localMax = 0.0;
    for( size = smin; size <= smax; size = size*2 ){

        // Warmup processor with a medium size DGEMM
        cublasDgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N, smed, smed, smed, 
                     &alpha, A_d, smed, B_d, smed, &beta, C_d, smed );
        cudaDeviceSynchronize();

        // Copy an array into another (read/write test)
        for( i = 0; i < NREPS; i++){
            tStart = benchTimer();
            for(j = 0; j < NLOOP; j++){
                cublasDgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N, size, size, size, 
                             &alpha, A_d, size, B_d, size, &beta, C_d, size );
            }
            cudaDeviceSynchronize();
            tElapsed[i] = benchTimer() - tStart;
        }

        // Get time and flops per DGEMM call
        // ops is total floating point operations per matrix matrix product
        // matSize is just the leading size of the matrices used
        ops     = (double)size*(double)size*2.0E0*( (double)size + 1.0E0 );
        matSize = (double)size;
        post_process( fp, fp2, threshold_hi, tElapsed, tScale, fpScale, size,
                      matSize, ops, &NLOOP, &localMax, &localSize );
    }
    // Print completion message                 
    printSummary( fp2, testName, localMax, localSize );
    fclose( fp2 ); 
    fclose( fp );

    // Free device allocated arrays and shut down cublas
    cudaFree( A_d );
    cudaFree( B_d );
    cudaFree( C_d );
    if( cublasShutdown()  != CUBLAS_STATUS_SUCCESS )
        fatalError( "CUBLAS failed to shut down cleanly." );

    // Free all host allocated memory
    free( A );
    free( B );
    free( C );

    return 0;
}
