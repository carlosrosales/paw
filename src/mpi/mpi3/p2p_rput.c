/*==============================================================================
 * Program  : p2p_rput
 * Revision : 1.3 (2014-12-17)
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
 * Test sustained MPI bandwidth using MPI-3 Rput/Wait calls.
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
    char    testName[32] = "MPI_Rput", file1[64], file2[64];
    int     dblSize, proc, nprocs, npairs, partner;
    unsigned int i, j, k, size, localSize, NLOOP = NLOOP_MAX;
    unsigned int smin = MIN_P2P_SIZE, smed = MED_P2P_SIZE, smax = MAX_P2P_SIZE;
    double  tScale = USEC, bwScale = MB;
    double  tStart, timeMin, timeMinGlobal, overhead, threshold_lo, threshold_hi;
    double  msgBytes, sizeBytes, localMax, UsedMem;
    double  tElapsed[NREPS], tElapsedGlobal[NREPS];
    double  *A, *B;
    MPI_Win     win;
    MPI_Status  stat;
    MPI_Request req;

    // Initialize parallel environment
    MPI_Init(&argc, &argv);
    MPI_Comm_size( MPI_COMM_WORLD, &nprocs );
    MPI_Comm_rank( MPI_COMM_WORLD, &proc );

    // Test input parameters
    if( nprocs%2 != 0 && proc == 0 )
        fatalError( "P2P test requires an even number of processors" );

    // Check for user defined limits
    checkEnvP2P( proc, &NLOOP, &smin, &smed, &smax );

    // Initialize local variables
    localMax = 0.0;
    npairs   = nprocs/2;
    if( proc < npairs  ) partner = proc + npairs;
    if( proc >= npairs ) partner = proc - npairs;
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
        sprintf( file1, "rput_time-np_%.4d.dat", nprocs );
        sprintf( file2, "rput_bw-np_%.4d.dat",   nprocs );
        fp  = fopen( file1, "a" );
        fp2 = fopen( file2, "a" );
        printHeaders( fp, fp2, testName, UsedMem, overhead, threshold_lo );
    }

    // Get type size
    MPI_Type_size( MPI_DOUBLE, &dblSize );
    // Set up a window for RMA
    MPI_Win_create( A, smax*dblSize, dblSize, MPI_INFO_NULL, MPI_COMM_WORLD, &win );
    MPI_Win_lock_all( 0, win );
    
    //================================================================
    // Single loop with minimum size to verify that inner loop length  
    // is long enough for the timings to be accurate                     
    //================================================================
    // Warmup with a medium size message
    if( proc < npairs ){
        MPI_Rput( B, smed, MPI_DOUBLE, partner, 0, smed, MPI_DOUBLE, win, &req );
        MPI_Wait( &req, &stat );
        MPI_Win_flush_all( win );
    }
    
    // Test if current NLOOP is enough to capture fastest test cases
    MPI_Barrier( MPI_COMM_WORLD );
    tStart = benchTimer();
    if( proc < npairs ){
        for(j = 0; j < NLOOP; j++){
        	MPI_Rput( B, smin, MPI_DOUBLE, partner, 0, smin, MPI_DOUBLE, win, &req );
        	MPI_Wait( &req, &stat );
            MPI_Win_flush_all( win );
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
        if( proc < npairs ){
            MPI_Rput( B, smed, MPI_DOUBLE, partner, 0, smed, MPI_DOUBLE, win, &req );
            MPI_Wait( &req, &stat );
            MPI_Win_flush_all( win );
        }

        // Repeat NREPS to collect statistics
        for(i = 0; i < NREPS; i++){
            MPI_Barrier( MPI_COMM_WORLD );
            tStart = benchTimer();
            if( proc < npairs ){
                for(j = 0; j < NLOOP; j++){
        	        MPI_Rput( B, size, MPI_DOUBLE, partner, 0, size, MPI_DOUBLE, win, &req );
        	        MPI_Wait( &req, &stat );
        	        MPI_Win_flush_all( win );
                }
            }
        	tElapsed[i] = benchTimer() - tStart;
        }
        MPI_Reduce( tElapsed, tElapsedGlobal, NREPS, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD );
        // Only task 0 needs to do the analysis of the collected data
        if( proc == 0 ){
            // sizeBytes is size to write to file
            // msgBytes is actual data exchanged on the wire
            msgBytes  = (double)(size*npairs*dblSize);
            sizeBytes = (double)(size*dblSize);
            post_process( fp, fp2, threshold_hi, tElapsedGlobal, tScale, 
                          bwScale, size*dblSize, sizeBytes, msgBytes, &NLOOP, 
                          &localMax, &localSize );
        }
        MPI_Bcast( &NLOOP, 1, MPI_INT, 0, MPI_COMM_WORLD );

    }
    MPI_Win_unlock_all( win );
    
    MPI_Win_free( &win );
    MPI_Barrier( MPI_COMM_WORLD );
    free( A );
    free( B );

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

