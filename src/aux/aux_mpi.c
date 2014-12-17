/*==============================================================================
 * File     : aux_mpi.c
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
 * Auxiliary MPI functions for PAW: error handling, input parsing
 *============================================================================*/

#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <mpi.h>
#include "aux.h"
#include "aux_mpi.h"
#include "constants.h"

// Check for environmental variables overrriding the default
// test settings for the MPI collectives tests
int checkEnvCOL( int proc, unsigned int *NLOOP, unsigned int *smin, 
 	              unsigned int *smed, unsigned int *smax  )
{
    if( proc == 0 ){
        if( getenv( "NLOOP_MAX" ) != NULL ){
            if( atoi( getenv( "NLOOP_MAX" ) ) <= 0 ) 
                fatalError("NLOOP_MAX must be a positive integer");
            else
                *NLOOP = atoi( getenv( "NLOOP_MAX" ) );
        }
        if( getenv( "MIN_COL_SIZE" ) != NULL ){
            if( atoi( getenv( "MIN_COL_SIZE" ) ) <= 0 ) 
                fatalError("MIN_COL_SIZE must be a positive integer");
            else
                *smin = atoi( getenv( "MIN_COL_SIZE" ) );
        }
        if( getenv( "MED_COL_SIZE" ) != NULL ){
            if( atoi( getenv( "MED_COL_SIZE" ) ) <= 0 ) 
                fatalError("MED_COL_SIZE must be a positive integer");
            else
                *smed = atoi( getenv( "MED_COL_SIZE" ) );
        }
        if( getenv( "MAX_COL_SIZE" ) != NULL ){
            if( atoi( getenv( "MAX_COL_SIZE" ) ) <= 0 ) 
                fatalError("MAX_COL_SIZE must be a positive integer");
            else
                *smax = atoi( getenv( "MAX_COL_SIZE" ) );
        }
    }
    MPI_Bcast( NLOOP, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD );
    MPI_Bcast( smin,  1, MPI_UNSIGNED, 0, MPI_COMM_WORLD );
    MPI_Bcast( smed,  1, MPI_UNSIGNED, 0, MPI_COMM_WORLD );
    MPI_Bcast( smax,  1, MPI_UNSIGNED, 0, MPI_COMM_WORLD );

    return 0;
}

// Check for environmental variables overrriding the default
// test settings for the MPI point to point tests
int checkEnvP2P( int proc, unsigned int *NLOOP, unsigned int *smin, 
 	              unsigned int *smed, unsigned int *smax  )
{
    if( proc == 0 ){
        if( getenv( "NLOOP_MAX" ) != NULL ){
            if( atoi( getenv( "NLOOP_MAX" ) ) <= 0 ) 
                fatalError("NLOOP_MAX must be a positive integer");
            else
                *NLOOP = atoi( getenv( "NLOOP_MAX" ) );
        }
        if( getenv( "MIN_P2P_SIZE" ) != NULL ){
            if( atoi( getenv( "MIN_P2P_SIZE" ) )  <= 0 ) 
                fatalError("MIN_P2P_SIZE must be a positive integer");
            else
                *smin = atoi( getenv( "MIN_P2P_SIZE" ) );
        }
        if( getenv( "MED_P2P_SIZE" ) != NULL ){
            if( atoi( getenv( "MED_P2P_SIZE" ) )  <= 0 ) 
                fatalError("MED_P2P_SIZE must be a positive integer");
            else
                *smed = atoi( getenv( "MED_P2P_SIZE" ) );
        }
        if( getenv( "MAX_P2P_SIZE" ) != NULL ){
            if( atoi( getenv( "MAX_P2P_SIZE" ) )  <= 0 ) 
                fatalError("MAX_P2P_SIZE must be a positive integer");
            else
                *smax = atoi( getenv( "MAX_P2P_SIZE" ) );
        }
    }
    MPI_Bcast( NLOOP, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD );
    MPI_Bcast( smin,  1, MPI_UNSIGNED, 0, MPI_COMM_WORLD );
    MPI_Bcast( smed,  1, MPI_UNSIGNED, 0, MPI_COMM_WORLD );
    MPI_Bcast( smax,  1, MPI_UNSIGNED, 0, MPI_COMM_WORLD );

    return 0;
}

// Check for environmental variables overrriding the default
// test settings for the MPI message rate tests
int checkEnvMRT( int proc, unsigned int *windowSize, unsigned int *NLOOP, 
	             unsigned int *smin, unsigned int *smed, unsigned int *smax  )
{
    if( proc == 0 ){
    	if( getenv( "WINDOW_SIZE"  ) != NULL ){
    	    if( atoi( getenv( "WINDOW_SIZE" ) ) <= 0 ) 
    	        fatalError("WINDOW_SIZE must be a positive integer");
    	    else
    	        *windowSize = atoi( getenv( "WINDOW_SIZE" ) );
    	}
        if( getenv( "NLOOP_MAX"    ) != NULL ){
            if( atoi( getenv( "NLOOP_MAX" ) ) <= 0 ) 
                fatalError("NLOOP_MAX must be a positive integer");
            else
                *NLOOP = atoi( getenv( "NLOOP_MAX" ) );
        }
        if( getenv( "MIN_P2P_SIZE" ) != NULL ){
            if( atoi( getenv( "MIN_P2P_SIZE" ) )  <= 0 ) 
                fatalError("MIN_P2P_SIZE must be a positive integer");
            else
                *smin = atoi( getenv( "MIN_P2P_SIZE" ) );
        }
        if( getenv( "MED_P2P_SIZE" ) != NULL ){
            if( atoi( getenv( "MED_P2P_SIZE" ) )  <= 0 ) 
                fatalError("MED_P2P_SIZE must be a positive integer");
            else
                *smed = atoi( getenv( "MED_P2P_SIZE" ) );
        }
        if( getenv( "MAX_P2P_SIZE" ) != NULL ){
            if( atoi( getenv( "MAX_P2P_SIZE" ) )  <= 0 ) 
                fatalError("MAX_P2P_SIZE must be a positive integer");
            else
                *smax = atoi( getenv( "MAX_P2P_SIZE" ) );
        }
    }
    MPI_Bcast( &windowSize,  1, MPI_UNSIGNED, 0, MPI_COMM_WORLD );
    MPI_Bcast( NLOOP, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD );
    MPI_Bcast( smin,  1, MPI_UNSIGNED, 0, MPI_COMM_WORLD );
    MPI_Bcast( smed,  1, MPI_UNSIGNED, 0, MPI_COMM_WORLD );
    MPI_Bcast( smax,  1, MPI_UNSIGNED, 0, MPI_COMM_WORLD );

    return 0;
}