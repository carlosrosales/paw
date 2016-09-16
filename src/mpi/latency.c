/*==============================================================================
 * Program  : latency
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
 * Test MPI latency using short blocking Send/Recv message pairs.
 *
 * This code measures latency by exchanging a message between two processors
 * using blocking Send/Recv message pairs. A range of small message sizes are 
 * exchanged and the actual latency is taken as that corresponding to the 
 * smallest message size (by default 1 byte). This assumes latency remains 
 * constant for very small MPI messages. All results are saved to disk in order
 * to be able to verify this assumption.
 *
 * In distributed systems it may be of interest to run this test inside a node, 
 * across nodes, and across racks in a controlled manner. This measures will 
 * give a better understanding of the network topology.
 *============================================================================*/

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <mpi.h>
#include "aux.h"
#include "aux_mpi.h"
#include "constants.h"

int main( int argc, char **argv )
{
    FILE    *fp;
    char    testName[32] = "MPI_Latency", file1[64], file2[64];
    int     dblSize, proc, nprocs, partner, tag = 0, NodeProcs;
    unsigned int i, j, size, localSize, NLOOP = NLOOP_MAX;
    unsigned int smin = MIN_P2P_SIZE, smed = MED_P2P_SIZE, smax = MAX_P2P_SIZE;
    double  tScale = USEC;
    double  overhead, threshold_lo, threshold_hi;
    double  tStart, timeMin, timeMinGlobal, msgBytes, localMax, UsedMem, ReqMem, NodeMem;
    double  tAvg, tMin, tMax, stdDev;
    double  tElapsed[NREPS], tElapsedGlobal[NREPS], tMsg[NREPS];
    char    sndBuffer = 'a', rcvBuffer = 'b';
    double  *A, *B;
    MPI_Status status;

    // Initialize parallel environment
    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &nprocs );
    MPI_Comm_rank( MPI_COMM_WORLD, &proc );

    // Test input parameters
    if( nprocs != 2 && proc == 0 )
        fatalError( "P2P latency will only run with 2 tasks" );

    // Check for user defined limits
    checkEnvP2P( proc, &NLOOP, &smin, &smed, &smax );

    // Initialize local variables
    partner = 1 - proc;
    dblSize  = sizeof(double);
    UsedMem = (double)smed*(double)dblSize*2.0;

    // Allocate and initialize arrays
    // TODO: Consider Mersenne Twister to improve startup time
    srand( SEED );
    A  = doubleVector( smed );
    B  = doubleVector( smed );

    // Open output file and write header
    if( proc == 0 ){
        // Check timer overhead in seconds
        timerTest( &overhead, &threshold_lo, &threshold_hi );
        // Open output files and write headers
        sprintf( file1, "latency.dat" );
        fp  = fopen( file1, "a" );
        printLatencyHeader( fp, testName, UsedMem, overhead, threshold_lo );
    }

    //================================================================
    // Single loop with minimum size to verify that inner loop length  
    // is long enough for the timings to be accurate                     
    //================================================================
    // Warmup with a medium size exchange
    if( proc == 0 ){
        MPI_Send( A, smed, MPI_DOUBLE, partner, tag, MPI_COMM_WORLD );
        MPI_Recv( B, smed, MPI_DOUBLE, partner, tag, MPI_COMM_WORLD, &status );
    }else{
        MPI_Recv( B, smed, MPI_DOUBLE, partner, tag, MPI_COMM_WORLD, &status );
        MPI_Send( A, smed, MPI_DOUBLE, partner, tag, MPI_COMM_WORLD );
    }
    // Test if current NLOOP is enough to capture fastest test cases
    MPI_Barrier( MPI_COMM_WORLD );
    tStart = benchTimer();
    if( proc == 0 ){
        for(j = 0; j < NLOOP; j++){
            MPI_Send( &sndBuffer, 1, MPI_CHAR, partner, tag, MPI_COMM_WORLD );
       	    MPI_Recv( &rcvBuffer, 1, MPI_CHAR, partner, tag, MPI_COMM_WORLD, &status );
        }
    }else{
        for(j = 0; j < NLOOP; j++){
       	    MPI_Recv( &rcvBuffer, 1, MPI_CHAR, partner, tag, MPI_COMM_WORLD, &status );
            MPI_Send( &sndBuffer, 1, MPI_CHAR, partner, tag, MPI_COMM_WORLD );
        }
    }
    timeMin = benchTimer() - tStart;
    MPI_Reduce( &timeMin, &timeMinGlobal, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD );
    if( proc == 0 ) resetInnerLoop( timeMinGlobal, threshold_lo, &NLOOP );
    MPI_Bcast( &NLOOP, 1, MPI_INT, 0, MPI_COMM_WORLD );

    //================================================================
    // Execute test
    //================================================================
    // Warmup with a medium size exchange
    if( proc == 0 ){
        MPI_Send( A, smed, MPI_DOUBLE, partner, tag, MPI_COMM_WORLD );
        MPI_Recv( B, smed, MPI_DOUBLE, partner, tag, MPI_COMM_WORLD, &status );
    }else{
        MPI_Recv( B, smed, MPI_DOUBLE, partner, tag, MPI_COMM_WORLD, &status );
        MPI_Send( A, smed, MPI_DOUBLE, partner, tag, MPI_COMM_WORLD );
    }

    // Repeat NREPS to collect statistics
    for(i = 0; i < NREPS; i++){
        MPI_Barrier(MPI_COMM_WORLD);
        tStart = benchTimer();
        if( proc == 0 ){
            for(j = 0; j < NLOOP; j++){
               MPI_Send( &sndBuffer, 1, MPI_CHAR, partner, tag, MPI_COMM_WORLD );
               MPI_Recv( &rcvBuffer, 1, MPI_CHAR, partner, tag, MPI_COMM_WORLD, &status );
            }
        }else{
            for(j = 0; j < NLOOP; j++){
                MPI_Recv( &rcvBuffer, 1, MPI_CHAR, partner, tag, MPI_COMM_WORLD, &status );
                MPI_Send( &sndBuffer, 1, MPI_CHAR, partner, tag, MPI_COMM_WORLD );
                }
        }
        tElapsed[i] = benchTimer() - tStart;
    } 
    MPI_Reduce( tElapsed, tElapsedGlobal, NREPS, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD );
    
    // Only task 0 needs to do the analysis of the collected data
    if( proc == 0 ){
        // Get the time per iteration
        for(i = 0; i < NREPS; i++){
            tMsg[i]  = 0.5*tElapsedGlobal[i] / ( (double)NLOOP );
        }
        // Calculate Average, Minimum and Maximum values
        stats( NREPS, tMsg,  &tAvg,  &tMax,  &tMin,  &stdDev, tScale );
        // Save these results to file
        saveData( fp,  sizeof(char), NLOOP, tAvg,  tMax,  tMin,  stdDev );
	    fprintf( stdout, "MPI latency is %6.1f usec\n\n", tMin );
        
    }
    //================================================================
    // Print completion message, free memory and exit                  
    //================================================================
    if( proc == 0 ) fclose( fp );
    free( A );
    free( B );

    MPI_Finalize();
    return 0;
}
