/*==============================================================================
 * Program  : col_allgather
 * Revision : 1.5 (2015-10-14)
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
 * Test the effective bandwidth of MPI_Allgather for a set of message sizes.
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
    FILE   *fp, *fp2, *pipe;
    char   testName[32] = "MPI_Allgather", file1[64], file2[64], pipeStr[8];
    int    dblSize, proc, nprocs, nodeCPUs, nodes;
    unsigned int i, j, size, localSize, NLOOP = NLOOP_MAX;
    unsigned int smin = MIN_COL_SIZE, smed = MED_COL_SIZE, smax = MAX_COL_SIZE;
    double tScale = USEC, bwScale = MB;
    double overhead, threshold_lo, threshold_hi, tStart, timeMin, timeMinGlobal;
    double sizeBytes, msgBytes, UsedMem, localMax;
    double tElapsed[NREPS], tElapsedGlobal[NREPS];
    double *A, *B;

    pipe = popen( "cat /proc/cpuinfo | grep processor | wc -l", "r" );
    fgets( pipeStr, 8, pipe ); pclose(pipe);
    nodeCPUs = atoi(pipeStr);

    // Initialize parallel environment
    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &nprocs );
    MPI_Comm_rank( MPI_COMM_WORLD, &proc );

    // Reset maximum message size to fit within node memory
    if( nprocs > nodeCPUs ){
        nodes = nprocs / nodeCPUs;
        if( smax > nodes ) smax = smax / nodes;
        if( smed > nodes ) smed = smed / nodes;
    }
    
    // Check for user defined limits
    checkEnvCOL( proc, &NLOOP, &smin, &smed, &smax );

    // Initialize local variables
    dblSize = sizeof(double);
    UsedMem = (double)smax*(double)dblSize*(double)( nprocs + 1 );

    // Allocate and initialize arrays
    srand( SEED );
    A = doubleVector( smax );
    B = doubleVector( smax*nprocs );

    // Open output file and write header
    if( proc == 0 ){
        // Check timer overhead in seconds
        timerTest( &overhead, &threshold_lo, &threshold_hi );
        // Open output files and write headers
        sprintf( file1, "allgather_time-np_%.4d.dat", nprocs );
        sprintf( file2, "allgather_bw-np_%.4d.dat",   nprocs );
        fp  = fopen( file1, "a" );
        fp2 = fopen( file2, "a" );
        printHeaders( fp, fp2, testName, UsedMem, overhead, threshold_lo );
    }

    //================================================================
    // Single loop with minimum size to verify that inner loop length  
    // is long enough for the timings to be accurate                     
    //================================================================
    // Warmup with a medium size message
    MPI_Allgather( A, smed, MPI_DOUBLE, 
                   B, smed, MPI_DOUBLE, MPI_COMM_WORLD );
    // Test is current NLOOP is enough to capture fastest test cases
    MPI_Barrier( MPI_COMM_WORLD );
    tStart = benchTimer();
    for(j = 0; j < NLOOP; j++){
        MPI_Allgather( A, smin, MPI_DOUBLE, 
                       B, smin, MPI_DOUBLE, MPI_COMM_WORLD );
    }
    timeMin = benchTimer() - tStart;
    MPI_Reduce( &timeMin, &timeMinGlobal, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD );
    if( proc == 0 ) resetInnerLoop( timeMinGlobal, threshold_lo, &NLOOP );
    MPI_Bcast( &NLOOP, 1, MPI_INT, 0, MPI_COMM_WORLD );

    //================================================================
    // Execute test for each requested size                  
    //================================================================
    localMax = 0.0;
    for( size = smin; size <= smax; size = size*2 ){

        // Warmup with a medium size message
        MPI_Allgather( A, smed, MPI_DOUBLE, 
                       B, smed, MPI_DOUBLE, MPI_COMM_WORLD );

        // Repeat NREPS to collect statistics
        for(i = 0; i < NREPS; i++){
            MPI_Barrier( MPI_COMM_WORLD );
            tStart = benchTimer();
            for(j = 0; j < NLOOP; j++){
                MPI_Allgather( A, size, MPI_DOUBLE, 
                               B, size, MPI_DOUBLE, MPI_COMM_WORLD );
            }
            tElapsed[i] = benchTimer() - tStart;
        }
        MPI_Reduce( tElapsed, tElapsedGlobal, NREPS, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);   
        // Only task 0 needs to do the analysis of the collected data
        if( proc == 0 ){
            // sizeBytes is size to write to file
            // msgBytes is actual data exchanged on the wire
            msgBytes  = (double)size*(double)nprocs*(double)dblSize;
            sizeBytes = (double)size*(double)dblSize;
            post_process( fp, fp2, threshold_hi, tElapsedGlobal, tScale, 
                          bwScale, size*dblSize, sizeBytes, msgBytes, &NLOOP, 
                          &localMax, &localSize );
        }
        MPI_Bcast( &NLOOP, 1, MPI_INT, 0, MPI_COMM_WORLD );
    }
    //================================================================
    // Print completion message, free memory and exit                  
    //================================================================
    if( proc == 0 ){
        fclose(fp);
        fclose(fp2);
        fprintf( stdout,"\n %s test completed.\n\n", testName );
    }
    free( A );
    free( B );

    MPI_Finalize();
    return 0;
}
