/*==============================================================================
 * Program  : col_accumulate_fence
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
 * Test sustained MPI bandwidth using MPI-2 Accumulate.
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
    FILE    *fp, *fp2;
    char    testName[32] = "MPI_Accumulate", file1[64], file2[64];
    int     dblSize, proc, nprocs;
    unsigned int i, j, k, size, localSize, origin, target, NLOOP = NLOOP_MAX;
    unsigned int smin = MIN_COL_SIZE, smed = MED_COL_SIZE, smax = MAX_COL_SIZE;
    double  tScale = USEC, bwScale = MB;
    double  tStart, timeMin, timeMinGlobal, overhead, threshold_lo, threshold_hi;
    double  msgBytes, sizeBytes, localMax, UsedMem;
    double  tElapsed[NREPS], tElapsedGlobal[NREPS];
    double  *A, *B;
    MPI_Win   win;

    // Initialize parallel environment
    MPI_Init(&argc, &argv);
    MPI_Comm_size( MPI_COMM_WORLD, &nprocs );
    MPI_Comm_rank( MPI_COMM_WORLD, &proc );

    // Test input parameters
    if( nprocs < 2 && proc == 0 )
        fatalError( "Accumulate test requires at leaast two processors" );

    // Check for user defined limits
    checkEnvCOL( proc, &NLOOP, &smin, &smed, &smax );

    // Initialize local variables
    localMax = 0.0;
    UsedMem = (double)smax*(double)sizeof(double)*2.0;

    // Allocate and initialize arrays
    srand( SEED );
    A = doubleVector( smax );
    B = doubleVector( smax );

    // Open output file and write header
    if( proc == 0 ){
        // Check timer overhead in seconds
        timerTest( &overhead, &threshold_lo, &threshold_hi );
        // Open output files and write headers
        sprintf( file1, "accfence_time-np_%.4d.dat", nprocs );
        sprintf( file2, "accfence_bw-np_%.4d.dat",   nprocs );
        fp  = fopen( file1, "a" );
        fp2 = fopen( file2, "a" );
        printHeaders( fp, fp2, testName, UsedMem, overhead, threshold_lo );
    }

    // Set up pair groups for exchange
    MPI_Type_size( MPI_DOUBLE, &dblSize );

    // Set up a win for RMA
    MPI_Win_create( A, smax*dblSize, dblSize, MPI_INFO_NULL, MPI_COMM_WORLD, &win );

    //================================================================
    // Single loop with minimum size to verify that inner loop length  
    // is long enough for the timings to be accurate                     
    //================================================================
    // Warmup with a medium size message
    if( proc > 0 ){
        MPI_Win_fence( 0, win );
        MPI_Accumulate( B, smed, MPI_DOUBLE, 0, 0, smed, MPI_DOUBLE, MPI_SUM, win );
        MPI_Win_fence( 0, win );
    }else{
        MPI_Win_fence( 0, win );
        MPI_Win_fence( 0, win );
    }
    // Test if current NLOOP is enough to capture fastest test cases
    MPI_Barrier( MPI_COMM_WORLD );
    tStart = benchTimer();
    if( proc > 0 ){
        for(j = 0; j < NLOOP; j++){
        MPI_Win_fence( 0, win );
        MPI_Accumulate( B, smin, MPI_DOUBLE, 0, 0, smin, MPI_DOUBLE, MPI_SUM, win );
        MPI_Win_fence( 0, win );
        }
    }else{
        for(j = 0; j < NLOOP; j++){
            MPI_Win_fence( 0, win );
            MPI_Win_fence( 0, win );
        }
    }
    timeMin = benchTimer() - tStart;
    MPI_Reduce( &timeMin, &timeMinGlobal, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD );
    if( proc == 0 ) resetInnerLoop( timeMinGlobal, threshold_lo, &NLOOP );
    MPI_Bcast( &NLOOP, 1, MPI_INT, 0, MPI_COMM_WORLD );


    //================================================================
    // Execute test for each requested size                  
    //================================================================
    for( size = smin; size <= smax; size = size*2 ){

        // Warmup with a medium size message
        if( proc > 0 ){
          MPI_Win_fence( 0, win );
          MPI_Accumulate( B, smed, MPI_DOUBLE, 0, 0, smed, MPI_DOUBLE, MPI_SUM, win );
          MPI_Win_fence( 0, win );
        }else{
            MPI_Win_fence( 0, win );
            MPI_Win_fence( 0, win );
        }

        // Repeat NREPS to collect statistics
        for(i = 0; i < NREPS; i++){
            MPI_Barrier( MPI_COMM_WORLD );
            tStart = benchTimer();
            if( proc > 0 ){
                for(j = 0; j < NLOOP; j++){
                  MPI_Win_fence( 0, win );
                  MPI_Accumulate( B, size, MPI_DOUBLE, 0, 0, size, MPI_DOUBLE, MPI_SUM, win );
                  MPI_Win_fence( 0, win );
                }
            }else{
                for(j = 0; j < NLOOP; j++){
                    MPI_Win_fence( 0, win );
                    MPI_Win_fence( 0, win );
                }
            }
        	tElapsed[i] = benchTimer() - tStart;
        }
        MPI_Reduce( tElapsed, tElapsedGlobal, NREPS, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD );
        // Only task 0 needs to do the analysis of the collected data
        if( proc == 0 ){
            // sizeBytes is size to write to file
            // msgBytes is actual data exchanged on the wire
            msgBytes  = (double)(size*(nprocs-1)*dblSize);
            sizeBytes = (double)(size*dblSize);
            post_process( fp, fp2, threshold_hi, tElapsedGlobal, tScale, 
                          bwScale, size*dblSize, sizeBytes, msgBytes, &NLOOP, 
                          &localMax, &localSize );
        }
        MPI_Bcast( &NLOOP, 1, MPI_INT, 0, MPI_COMM_WORLD );

    }
    MPI_Win_free( &win );
    MPI_Barrier( MPI_COMM_WORLD );
    free( A );
    free( B );
    MPI_Group_free( &group );

    //================================================================
    // Print completion message, free memory and exit                  
    //================================================================
    if( proc == 0 ){
        printSummary( fp2, testName, localMax, localSize );
        fclose( fp2 ); 
        fclose( fp );
    }

    MPI_Finalize();
    return 0;
}

