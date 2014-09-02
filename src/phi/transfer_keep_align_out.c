/*==============================================================================
 * Program  : transfer_keep_align_out
 * Revision : 1.0 (2014-08-27)
 * Author   : Carlos Rosales Fernandez [carlos.rosales.fernandez(at)gmail.com]
 *==============================================================================
 * Copyright 2014 The University of Texas at Austin
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
 * Test sustained data transfer rate from Phi coprocessor to CPU for a range 
 * of message sizes using the "offload_transfer" pragma. In this test arrays 
 * are preallocated on the Phi coprocessor and the data is moved between Phi and 
 * CPU without freeing and reallocating between exchanges. Data is aligned to 
 * a 64 byte boundary.
 *
 * If two Phi coporcessors are present three total tests are executed, on for 
 * each Xeon Phi (with the other coprocessor idle), and one where two threads
 * in the host CPU are used to offload simultaneously to both Phi coprocessors.
 *============================================================================*/

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <offload.h>
#include "aux.h"
#include "constants.h"

int main(int argc, char **argv)
{
    FILE    *fp, *fp2;
    char    testName[32] = "PHI_TRANSFER_KEEP_AL_OUT";
    int     micNum, tid;
    unsigned int i, j, size, localSize, NLOOP = NLOOP_PHI_MAX, NLOOP_PHI;
    unsigned int smin = MIN_PHI_SIZE, smed = MED_PHI_SIZE, smax = MAX_PHI_SIZE;
    double  *f0, *f1;
    double timeMin, tStart, tElapsed[NREPS];
    double tScale = USEC, bwScale = MB;
    double overhead, threshold_lo, threshold_hi;
    double tMin, tMax, tAvg, stdDev, bwMax, bwMin, bwAvg, bwDev;
    double UsedMem, localMax, msgBytes;
    double tMsg[NREPS], bwMsg[NREPS];

    // Identify number of MIC devices
    micNum = _Offload_number_of_devices();
    if( micNum == 0 ) fatalError( "No Xeon Phi devices found. Test Aborted." );

    // Check for user defined limits
    if( getenv( "NLOOP_PHI_MAX" ) != NULL ) NLOOP = atoi( getenv( "NLOOP_PHI_MAX" ) );
    if( getenv( "MIN_PHI_SIZE" ) != NULL ) smin = atoi( getenv( "MIN_PHI_SIZE" ) );
    if( getenv( "MED_PHI_SIZE" ) != NULL ) smed = atoi( getenv( "MED_PHI_SIZE" ) );
    if( getenv( "MAX_PHI_SIZE" ) != NULL ) smax = atoi( getenv( "MAX_PHI_SIZE" ) );
    if( micNum == 1 ) UsedMem = (double)smax*sizeof(double);
    if( micNum == 2 ) UsedMem = (double)smax*2.0*sizeof(double);

    // Allocate and initialize test array
    srand( SEED );
    f0 = doubleVector( smax );

    // Check timer overhead in seconds
    timerTest( &overhead, &threshold_lo, &threshold_hi );

    // Open output files and write headers
    fp  = fopen( "mic0_keep_al_time_out.dat", "a" );
    fp2 = fopen( "mic0_keep_al_bw_out.dat", "a" );
    printHeaders( fp, fp2, testName, UsedMem, overhead, threshold_lo );

    //================================================================
    // Single loop with minimum size to verify that inner loop length  
    // is long enough for the timings to be accurate                     
    //================================================================
    // Warmup processor with a large size exchange
    // Since we will be reusing we want to make sure this exchange uses smax
    #pragma offload_transfer target(mic:0) out( f0 : length(smax) ALLOC KEEP )
    // Test is current NLOOP is enough to capture fastest test cases
    tStart = benchTimer();
    for(j = 0; j < NLOOP; j++){
        #pragma offload_transfer target(mic:0) out( f0 : length(smin) REUSE KEEP )
    }
    timeMin = benchTimer() - tStart;
    resetInnerLoop( timeMin, threshold_lo, &NLOOP );
    // Let's save this info in case we have more than one Phi device
    NLOOP_PHI = NLOOP;

    //================================================================
    // Execute test for each requested size                  
    //================================================================
    localSize = smin;
    localMax  = 0.0;
    for( size = smin; size <= smax; size = size*2 ){

        // Copy array to Phi (read/write test)
        for( i = 0; i < NREPS; i++){
            tStart = benchTimer();
            for(j = 0; j < NLOOP; j++){
                #pragma offload_transfer target(mic:0) out( f0 : length(size) REUSE KEEP )
            }
            tElapsed[i] = benchTimer() - tStart;
        }
        msgBytes = (double)( size*sizeof(double));
        post_process( fp, fp2, threshold_hi, tElapsed, tScale, bwScale, size,
                      msgBytes, msgBytes, &NLOOP, &localMax, &localSize );
    }
    // Print completion message                 
    printSummary( fp2, testName, localMax, localSize );
    fclose( fp2 ); 
    fclose( fp );

    if( micNum == 2 ){

    // Allocate and initialize test array for second Phi coprocessor (mic:1)
    f1 = doubleVector(smax);

    // Open output files and write headers
    fp  = fopen( "mic1_keep_al_time_out.dat", "a" );
    fp2 = fopen( "mic1_keep_al_bw_out.dat", "a" );
    printHeaders( fp, fp2, testName, UsedMem, overhead, threshold_lo );

    //================================================================
    // Single loop with minimum size to verify that inner loop length  
    // is long enough for the timings to be accurate                     
    //================================================================
    // Warmup processor with a large size exchange
    // Since we will be reusing we want to make sure this exchanges uses smax
    #pragma offload_transfer target(mic:1) out( f1 : length(smax) ALLOC KEEP )
    // Reset innermost loop to safe value and local quantities to defaults
    NLOOP = NLOOP_PHI;
    localSize = smin;
    localMax  = 0.0;

   //================================================================
    // Execute test for each requested size                  
    //================================================================
    for( size = smin; size <= smax; size = size*2 ){

        // Copy array to Phi (read/write test)
        for( i = 0; i < NREPS; i++){
            tStart = benchTimer();
            for(j = 0; j < NLOOP; j++){
                #pragma offload_transfer target(mic:1) out( f1 : length(size) REUSE KEEP )
            }
            tElapsed[i] = benchTimer() - tStart;
        }
        msgBytes = (double)( size*sizeof(double));
        post_process( fp, fp2, threshold_hi, tElapsed, tScale, bwScale, size,
                      msgBytes, msgBytes, &NLOOP, &localMax, &localSize );
    }
    // Print completion message                 
    printSummary( fp2, testName, localMax, localSize );
    fclose( fp2 ); 
    fclose( fp );

    //------- TESTING SIMULTANEOUS DATA TRANSFER TO BOTH PHI DEVICES ------

    // Open output files and write headers
    fp  = fopen( "mic0+1_keep_al_time_out.dat", "a" );
    fp2 = fopen( "mic0+1_keep_al_bw_out.dat", "a" );
    printHeaders( fp, fp2, testName, UsedMem, overhead, threshold_lo );

    // Warmup processor with a medium size exchange
    #pragma offload_transfer target(mic:0) out( f0 : length(smed) REUSE KEEP )
    #pragma offload_transfer target(mic:1) out( f1 : length(smed) REUSE KEEP )
    // Reset innermost loop to safe value and local quantities to defaults
    NLOOP = NLOOP_PHI;
    localSize = smin;
    localMax  = 0.0;

    //================================================================
    // Execute test for each requested size                  
    //================================================================
    for( size = smin; size <= smax; size = size*2 ){

        for( i = 0; i < NREPS; i++){
            tStart = benchTimer();
            #pragma omp parallel private(j,tid) num_threads(2)
            {
                tid = omp_get_thread_num();
                if( tid == 0 ){
                    for(j = 0; j < NLOOP; j++){
                        #pragma offload_transfer target(mic:0)  out( f0 : length(size) REUSE KEEP )
                    }
                }
                if( tid == 1 ){
                    for(j = 0; j < NLOOP; j++){
                        #pragma offload_transfer target(mic:1)  out( f1 : length(size) REUSE KEEP )
                    }
                }
            }
            tElapsed[i] = 0.5*( benchTimer() - tStart );
        }
        msgBytes = (double)( size*sizeof(double));
        post_process( fp, fp2, threshold_hi, tElapsed, tScale, bwScale, size,
                      msgBytes, msgBytes, &NLOOP, &localMax, &localSize );
    }
    // Print completion message                 
    printSummary( fp2, testName, localMax, localSize );
    fclose( fp2 ); 
    fclose( fp );

    }

    free( f0 );
    if( micNum == 2 ) free( f1 );
    return 0;
}

