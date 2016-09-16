/*==============================================================================
 * Program  : col_init_th
 * Revision : 1.6 (2016-09-16)
 * Author   : Carlos Rosales Fernandez [carlos.rosales.fernandez(at)gmail.com]
 *==============================================================================
 * Copyright 2016 Carlos Rosales Fernandez and The University of Texas at Austin
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
 * Test MPI initialization time. Output goes to stdout as <NumProcs> <Time (s)>
 * A warning is prepended to the output if timer overhead approaches the 
 * initialization times
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
    int    proc, nprocs, provided;
    double tStart, tElapsed, tMax, overhead, threshold_lo, threshold_hi;

    // Starting time
    tStart = benchTimer();
    // Initialize parallel environment
    MPI_Init_thread( &argc, &argv, MPI_THREAD_FUNNELED, &provided );
    // Time that it takes this task to be able to continue
    tElapsed = benchTimer() - tStart;
    // Synchronize and reduce result
    MPI_Barrier( MPI_COMM_WORLD );
    MPI_Reduce( &tElapsed, &tMax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD );

    // Get Comm size and rank for output
    MPI_Comm_size( MPI_COMM_WORLD, &nprocs );
    MPI_Comm_rank( MPI_COMM_WORLD, &proc );

    // Check for timer overhead
    // We should output warning if overhead > initialization
    if( proc == 0 ){
       timerTest( &overhead, &threshold_lo, &threshold_hi );
       if( overhead > 0.5*tMax ) fprintf( stdout, "# WARNING: Timer overhead approaches measurement: %e seconds", overhead ); 
       fprintf( stdout, "%d\t%e\n", nprocs, tMax );
    }

    MPI_Finalize();
    return 0;
}

