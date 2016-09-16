/*==============================================================================
 * Program  : p2p_rate
 * Revision : 1.6 (2016-09-16)
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
 * Test sustained MPI message rate using a ping-pong setup with non-blocking 
 * MPI_Isend / MPI_Irecv message pairs completed by MPI_Wait calls. This is 
 * an attempt at measuring the maximum sustained one-directional message rate 
 * of the system.
 *============================================================================*/

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <mpi.h>
#include "aux.h"
#include "aux_mpi.h"
#include "constants.h"

int main(int argc, char **argv)
{
    FILE   *fp, *fp2;
    char   testName[32] = "MPI_RATE", file1[64], file2[64];
    int    dblSize, proc, nprocs, npairs, partner, tag = 0;
    unsigned int i, j, k, size, localSize, NLOOP = NLOOP_MAX;
    unsigned int windowSize = DEFAULT_WINDOW_SIZE;
    unsigned int smin = MIN_P2P_SIZE, smed = MED_P2P_SIZE, smax = MAX_P2P_SIZE;
    double tScale = USEC, bwScale = MB_8;
    double tStart, timeMin, timeMinGlobal, overhead, threshold_lo, threshold_hi;
    double msgBytes, sizeBytes, localMax, UsedMem, work;
    double tElapsed[NREPS], tElapsedGlobal[NREPS];
    double *A, *B;
    MPI_Status  *stat1, *stat2;
    MPI_Request *req1,  *req2;


    // Initialize parallel environment
    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &nprocs );
    MPI_Comm_rank( MPI_COMM_WORLD, &proc );

    // Test input parameters
    if( nprocs%2 != 0 && proc == 0 )
        fatalError( "P2P test requires an even number of processors" );

    // Check for user defined limits
    checkEnvMRT( proc, &windowSize, &NLOOP, &smin, &smed, &smax );

    // Initialize local variables
    localMax = 0.0;
    dblSize  = sizeof(double);
    npairs   = nprocs/2;
    if( proc < npairs  ) partner = proc + npairs;
    if( proc >= npairs ) partner = proc - npairs;
    UsedMem = (double)smax*(double)dblSize*2.0;

    // Allocate and initialize arrays
    srand( SEED );
    A  = doubleVector( smax );
    B  = doubleVector( smax );
    stat1 = (MPI_Status *)malloc( 2*windowSize*sizeof(MPI_Status) );
    stat2 = (MPI_Status *)malloc( 2*windowSize*sizeof(MPI_Status) );
    req1  = (MPI_Request *)malloc( 2*windowSize*sizeof(MPI_Request) );
    req2  = (MPI_Request *)malloc( 2*windowSize*sizeof(MPI_Request) );

    // Open output file and write header
    if( proc == 0 ){
        // Check timer overhead in seconds
        timerTest( &overhead, &threshold_lo, &threshold_hi );
        // Open output files and write headers
        sprintf( file1, "mrate_bidir_time-ws_%.3d-np_%.4d.dat", windowSize, nprocs );
        sprintf( file2, "mrate_bidir_rate-ws_%.3d-np_%.4d.dat", windowSize, nprocs );
        fp  = fopen( file1, "a" );
        fp2 = fopen( file2, "a" );
        printHeaders( fp, fp2, testName, UsedMem, overhead, threshold_lo );
    }

    //================================================================
    // Single loop with minimum size to verify that inner loop length  
    // is long enough for the timings to be accurate                     
    //================================================================
    // Warmup with a medium size message
    if( proc < npairs ){
        MPI_Isend( A, smed, MPI_DOUBLE, partner, tag, MPI_COMM_WORLD, req1   );
        MPI_Irecv( B, smed, MPI_DOUBLE, partner, tag, MPI_COMM_WORLD, req1+1 );
        MPI_Waitall( 2, req1, stat1);
    }else{
        MPI_Irecv( B, smed, MPI_DOUBLE, partner, tag, MPI_COMM_WORLD, req2   );
        MPI_Isend( A, smed, MPI_DOUBLE, partner, tag, MPI_COMM_WORLD, req2+1 );
        MPI_Waitall( 2, req2, stat2 );
    }
    // Test if current NLOOP is enough to capture fastest test cases
    MPI_Barrier( MPI_COMM_WORLD );
    if( proc < npairs ){
        tStart = benchTimer();
        for(j = 0; j < NLOOP; j++){
            for( k = 0; k < windowSize; k++ ){
                MPI_Isend( A, smin, MPI_DOUBLE, partner, tag, MPI_COMM_WORLD, req1+k );
                MPI_Irecv( B, smin, MPI_DOUBLE, partner, tag, MPI_COMM_WORLD, req1+k+windowSize);
            }
            MPI_Waitall( 2*windowSize, req1, stat1 );
        }
        timeMin = benchTimer() - tStart;
    }else{
        tStart = benchTimer();
        for(j = 0; j < NLOOP; j++){
            for( k = 0; k < windowSize; k++ ){
                MPI_Irecv( B, smin, MPI_DOUBLE, partner, tag, MPI_COMM_WORLD, req2+k);
                MPI_Isend( A, smin, MPI_DOUBLE, partner, tag, MPI_COMM_WORLD, req2+k+windowSize );
            }
            MPI_Waitall( 2*windowSize, req2, stat2 );
        }
        timeMin = benchTimer() - tStart;
    }
    MPI_Reduce( &timeMin, &timeMinGlobal, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD );
    if( proc == 0 ) resetInnerLoop( timeMinGlobal, threshold_lo, &NLOOP );
    MPI_Bcast( &NLOOP, 1, MPI_INT, 0, MPI_COMM_WORLD );


    //================================================================
    // Execute test for each requested size                  
    //================================================================
    for( size = smin; size <= smax; size = size*2 ){

        // Warmup with a medium size message
        if( proc < npairs ){
            MPI_Isend( A, smed, MPI_DOUBLE, partner, tag, MPI_COMM_WORLD, req1   );
            MPI_Irecv( B, smed, MPI_DOUBLE, partner, tag, MPI_COMM_WORLD, req1+1 );
            MPI_Waitall( 2, req1, stat1);
        }else{
            MPI_Irecv( B, smed, MPI_DOUBLE, partner, tag, MPI_COMM_WORLD, req2   );
            MPI_Isend( A, smed, MPI_DOUBLE, partner, tag, MPI_COMM_WORLD, req2+1 );
            MPI_Waitall( 2, req2, stat2 );
        }

        // Repeat NREPS to collect statistics
        for(i = 0; i < NREPS; i++){
            MPI_Barrier( MPI_COMM_WORLD );
            if( proc < npairs ){
                tStart = benchTimer();
                for(j = 0; j < NLOOP; j++){
                    for( k = 0; k < windowSize; k++ ){
                        MPI_Isend( A, size, MPI_DOUBLE, partner, tag, MPI_COMM_WORLD, req1+k );
                        MPI_Irecv( B, size, MPI_DOUBLE, partner, tag, MPI_COMM_WORLD, req1+k+windowSize);
                    }
                    MPI_Waitall( 2*windowSize, req1, stat1 );
                }
                timeMin = benchTimer() - tStart;
            }else{
                tStart = benchTimer();
                for(j = 0; j < NLOOP; j++){
                    for( k = 0; k < windowSize; k++ ){
                        MPI_Irecv( B, size, MPI_DOUBLE, partner, tag, MPI_COMM_WORLD, req2+k);
                        MPI_Isend( A, size, MPI_DOUBLE, partner, tag, MPI_COMM_WORLD, req2+k+windowSize );
                    }
                    MPI_Waitall( 2*windowSize, req2, stat2 );
                }
                timeMin = benchTimer() - tStart;
            }
            // Factor 0.5 included because we go both ways
            tElapsed[i] = 0.5*( benchTimer() - tStart );
        } 
        MPI_Reduce( tElapsed, tElapsedGlobal, NREPS, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        // Only task 0 needs to do the analysis of the collected data
        if( proc == 0 ){
            // sizeBytes is size of the message in bytes
            // msgBytes is actual data exchanged on the wire
            // work is the total numbe or message exchagnes timed
            msgBytes  = (double)size*(double)npairs*(double)dblSize;
            sizeBytes = (double)size*(double)dblSize;
            work      = (double)npairs*(double)windowSize;
            post_process( fp, fp2, threshold_hi, tElapsedGlobal, tScale, 
                          bwScale, size*dblSize, sizeBytes, work, &NLOOP, 
                          &localMax, &localSize );
        }
        MPI_Bcast( &NLOOP, 1, MPI_INT, 0, MPI_COMM_WORLD );

    }
    //================================================================
    // Print completion message, close result files, and exit                  
    //================================================================
    if( proc == 0 ){
        printSummary( fp2, testName, localMax, localSize );
        fclose( fp2 ); 
        fclose( fp );
    }
    free( A );
    free( B );
    free( stat1 ); free( req1 );
    free( stat2 ); free( req2 );

    MPI_Finalize();
    return 0;
}

