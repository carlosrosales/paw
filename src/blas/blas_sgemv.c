/*==============================================================================
 * Program  : blas_sgemv
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
 * Test sustained single precision matrix vector multiplication performance 
 * using the BLAS SGEMV call for a range of matrix sizes.
 *============================================================================*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <math.h>
#include "aux.h"
#include "constants.h"

#ifdef MKL
    #include <mkl.h>
    #include <mkl_cblas.h>
#else
    #include <malloc.h>
    #include <cblas.h>
#endif

int main(int argc, char **argv)
{
    FILE   *fp, *fp2;
    char   testName[32] = "SGEMV", file1[64], file2[64];
    unsigned int i, j, size, localSize, nthreads;
    unsigned int NLOOP = NLOOP_MAX, MED_SIZE = MED_BLAS_SIZE;
    unsigned int MIN_SIZE = MIN_BLAS_SIZE, MAX_SIZE = MAX_BLAS_SIZE;
    double tScale = SEC, fpScale = GFLOP;
    double tStart, timeMin, overhead, threshold_lo, threshold_hi;
    double ops, matSize, usedMem, localMax;
    double tElapsed[NREPS];


    unsigned long NN;
    float alpha, beta;
    float *A, *B, *C;

    // Check for user defined limits
    if( getenv( "NLOOP_MAX" ) != NULL ) NLOOP = atoi( getenv( "NLOOP_MAX" ) );
    if( getenv( "MIN_BLAS_SIZE" ) != NULL ) MIN_SIZE = atoi( getenv( "MIN_BLAS_SIZE" ) );
    if( getenv( "MED_BLAS_SIZE" ) != NULL ) MED_SIZE = atoi( getenv( "MED_BLAS_SIZE" ) );
    if( getenv( "MAX_BLAS_SIZE" ) != NULL ) MAX_SIZE = atoi( getenv( "MAX_BLAS_SIZE" ) );

    // Initialize variables
    localMax = 0;
    alpha    = 1.0E0;
    beta     = 1.0E0;
    NN       = MAX_SIZE*MAX_SIZE;
    usedMem  = (double)MAX_SIZE*sizeof(float)*( 2.0 + (double)MAX_SIZE );

    // Allocate and initialize arrays
    // TODO: Consider Mersenne Twister to improve startup time
    srand( SEED );
    A = floatVector( NN );
    B = floatVector( MAX_SIZE );
    C = floatVector( MAX_SIZE );

    // Check timer overhead in seconds
    timerTest( &overhead, &threshold_lo, &threshold_hi );
    // Check how many threads are going to be used
    nthreads = threadCount();
    // Open output files and write headers
    sprintf( file1, "sgemv_time-np_%.4d.dat", nthreads );
    sprintf( file2, "sgemv_fp-np_%.4d.dat",   nthreads  );
    fp  = fopen( file1, "a" );
    fp2 = fopen( file2, "a" );
    printHeaders( fp, fp2, testName, usedMem, overhead, threshold_lo );

    //================================================================
    // Single loop with minimum size to verify that inner loop length  
    // is long enough for the timings to be accurate                     
    //================================================================
    // Warmup processor with a medium size SGEMV
    cblas_sgemv( CblasRowMajor, CblasNoTrans, MED_SIZE, MED_SIZE, alpha, A, 
                 MED_SIZE, B, 1, beta, C, 1 );
    // Test is current NLOOP is enough to capture fastest test cases
    tStart = benchTimer();
    for(j = 0; j < NLOOP; j++){
         cblas_sgemv( CblasRowMajor, CblasNoTrans, MIN_SIZE, MIN_SIZE, alpha, 
                      A, MIN_SIZE, B, 1, beta, C, 1 );
    }
    timeMin = benchTimer() - tStart;
    resetInnerLoop( timeMin, threshold_lo, &NLOOP );

    //================================================================
    // Execute test for each requested size                  
    //================================================================
    for( size = MIN_SIZE; size <= MAX_SIZE; size = size*2 ){

        // Warmup processor with a medium size DGEMV
        cblas_sgemv( CblasRowMajor, CblasNoTrans, MED_SIZE, MED_SIZE, alpha, A, 
                     MED_SIZE, B, 1, beta, C, 1 );

        // Call SGEMV to solve C = alpha*(A*B) + beta*C in each processor
        for( i = 0; i < NREPS; i++){
                tStart = benchTimer();
                for(j = 0; j < NLOOP; j++){
                    cblas_sgemv( CblasRowMajor, CblasNoTrans, size, size, 
                                 alpha, A, size, B, 1, beta, C, 1 );
                }
                tElapsed[i] = benchTimer() - tStart;
        }

        // Get time and flops per SGEMV call
        // ops is total floating point operations per matrix matrix product
        // matSize is just the leading size of the matrices used
        ops     = (double)size*2.0E0*( (double)size + 1.0E0 );
        matSize = (double)size;
        post_process( fp, fp2, threshold_hi, tElapsed, tScale, fpScale, size,
                      matSize, ops, &NLOOP, &localMax, &localSize );
    }
    // Print completion message                 
    printSummary( fp2, testName, localMax, localSize );
    fclose( fp2 ); 
    fclose( fp );

    // Free all allocated memory
    #ifdef MKL
        mkl_free( A ); mkl_free( B ); mkl_free( C );
    #else
        free( A ); free( B ); free( C );
    #endif

    return 0;
}
