/*==============================================================================
 * Program  : gpu_transfer_out_nopin
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
 * Test sustained data transfer from GPU to CPU for a range of message sizes.
 * In this test memory is not pinned.
 *============================================================================*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "aux.h"
#include "constants.h"

int main( int argc, char **argv )
{
    FILE   *fp, *fp2;
    char   testName[32] = "GPU_TRANSFER_OUT_NOPIN";
    int    i, j, deviceID, gpuNum;
    unsigned int size, localSize, NLOOP = NLOOP_MAX, MED_SIZE = MED_GPU_SIZE;
    unsigned int MIN_SIZE = MIN_GPU_SIZE, MAX_SIZE = MAX_GPU_SIZE;
    float  *A, *A_d;
    double tScale = USEC, bwScale = MB;
    double timeMin, tStart, overhead, threshold_lo, threshold_hi;
    double usedMem, localMax, msgBytes;
    double tElapsed[NREPS];

    // Make sure there is at least one GPU in the system
    cudaGetDeviceCount( &gpuNum );
    if( gpuNum == 0 )
        fatalError( "No GPU devices found. Test Aborted." );

    // Check for user defined limits
    if( getenv( "NLOOP_MAX" ) != NULL ) NLOOP = atoi( getenv( "NLOOP_MAX" ) );
    if( getenv( "MIN_GPU_SIZE" ) != NULL ) MIN_SIZE = atoi( getenv( "MIN_GPU_SIZE" ) );
    if( getenv( "MED_GPU_SIZE" ) != NULL ) MED_SIZE = atoi( getenv( "MED_GPU_SIZE" ) );
    if( getenv( "MAX_GPU_SIZE" ) != NULL ) MAX_SIZE = atoi( getenv( "MAX_GPU_SIZE" ) );
    usedMem = MAX_SIZE * sizeof(float);

    // Select first GPU in the node
    cudaGetDevice( &deviceID );
    cudaSetDevice( deviceID );

    // Check timer overhead in seconds
    timerTest( &overhead, &threshold_lo, &threshold_hi );
    // Open output files and write headers
    fp  = fopen("gpu_transfer_time_out_nopin.dat","a");
    fp2 = fopen("gpu_transfer_bw_out_nopin.dat","a");
    printHeaders( fp, fp2, testName, usedMem, overhead, threshold_lo );

    /* Allocate and initialize variables */
    srand( SEED );
    A = floatVector( MAX_SIZE );
    cudaMalloc( (void **) &A_d, MAX_SIZE*sizeof(float) );
    if( !A || !A_d )
       fatalError( "Failed to allocate memory in GPU bandwidth test." );
    cudaMemcpy( A_d, A, MAX_SIZE * sizeof(float), cudaMemcpyHostToDevice );

    //================================================================
    // Single loop with minimum size to verify that inner loop length  
    // is long enough for the timings to be accurate                     
    //================================================================
    // Warmup processor with a medium size DGEMM
    cudaMemcpy( A, A_d, MED_SIZE * sizeof(float), cudaMemcpyDeviceToHost );
    // Test is current NLOOP is enough to capture fastest test cases
    cudaDeviceSynchronize();
    tStart = benchTimer();
    for(j = 0; j < NLOOP; j++){
        cudaMemcpy( A, A_d, MIN_SIZE * sizeof(float), cudaMemcpyDeviceToHost );
    }
    cudaDeviceSynchronize();
    timeMin = benchTimer() - tStart;
    resetInnerLoop( timeMin, threshold_lo, &NLOOP );

    //================================================================
    // Execute test for each requested size                  
    //================================================================
    localMax = 0.0;
    for( size = MIN_SIZE; size <= MAX_SIZE; size = size*2 ){

        // Warmup processor with a medium size DGEMM
        cudaMemcpy( A, A_d, MED_SIZE * sizeof(float), cudaMemcpyDeviceToHost );
        cudaDeviceSynchronize();

        // Copy an array into another (read/write test)
        for( i = 0; i < NREPS; i++){
            tStart = benchTimer();
            for(j = 0; j < NLOOP; j++){
                cudaMemcpy( A, A_d, size * sizeof(float), cudaMemcpyDeviceToHost );
            }
            cudaDeviceSynchronize();
            tElapsed[i] = benchTimer() - tStart;
        }
        msgBytes  = (double)( size*sizeof(float));
        post_process( fp, fp2, threshold_hi, tElapsed, tScale, bwScale, size,
                      msgBytes, msgBytes, &NLOOP, &localMax, &localSize );
    }
    // Print completion message                 
    printSummary( fp2, testName, localMax, localSize );
    fclose( fp2 ); 
    fclose( fp );

    free( A ); 
    cudaFree( A_d );

    return 0;
}
