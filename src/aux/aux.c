/*==============================================================================
 * File     : aux.c
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
 * Auxiliary functions for PAW: timer, formatted printing, error handling, 
 * memory allocation and statistics
 *============================================================================*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include "aux.h"
#include "constants.h"

#ifdef MKL
    #include <mkl.h>
#else
    #include <malloc.h>
#endif

double benchTimer(void)
{
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);

    return (double)time.tv_sec + 1.0E-09*(double)time.tv_nsec;
}

// Return the overhad of calling the benchmark timer
// This value is then used to determine the minimum outer loop length
int timerTest( double *overhead, double *threshold_lo, double *threshold_hi )
{
    int     i;
    double  tstart, tend, tmax, tdelta, resolution;

    struct timespec res;


    tstart = benchTimer();
    for( i = 0; i < TIMER_REPS; i++){
        tdelta = benchTimer();
    }
    tend = benchTimer();
    *overhead = (tend - tstart)/(double)TIMER_REPS;
    *threshold_lo = *overhead*TIMER_LO;
    *threshold_hi = *overhead*TIMER_HI;

    return 0;
}

int fatalError( char *errorText ){
    fprintf( stderr, "\n FATAL ERROR: %s\n\n", errorText );
    fflush( stderr );
    exit(1);
    return 1;
}

int userWarning( char *warningText ){
    fprintf( stdout, "\n WARNING: %s\n\n", warningText );
    fflush( stdout );

    return 0;
}

int stats( int N, double *x, double *xAvg, double *xMax, double *xMin, 
           double *xDev, double scale )
{
    int     i;
    double  yAvg, yMax, yMin, yDev;

    yAvg  = yDev = 0.0;
    yMin  = yMax = x[0];

    /* Calculate average, maximum and minimum values */
    for(i = 0; i < N; i++){
        yAvg = yAvg + x[i];
        if( x[i] < yMin ) yMin = x[i];
        if( x[i] > yMax ) yMax = x[i];
    }
    yAvg  = yAvg  / ( (double)N );

    /* Calculate the standard deviation over the N measurements */
    for( i = 0; i < N; i++ ){
        yDev = yDev + ( x[i] - yAvg )*( x[i]  - yAvg );
    }
    yDev = sqrt( yDev /( (double)N ) );

    /* Return scaled results */
    *xAvg = yAvg / scale;
    *xMax = yMax / scale;
    *xMin = yMin / scale;
    *xDev = yDev / scale;

    return 0;
}

int saveData( FILE *fp, double size, int NLOOP, double xAvg, 
              double xMax, double xMin, double xDev )
{
    fprintf( fp, "%E\t%E\t\t%E\t", size, xAvg, xMin );
    fprintf( fp, "\t%E\t%E\t\t%5d\t%5d\n", xMax, xDev, NLOOP, NREPS );
    fflush( fp );

    return 0;
}

int printHeaders( FILE *fp, FILE *fp2, char *testName, double UsedMem, 
                  double overhead, double threshold_lo )
{
    FILE   *pipe;
    time_t rawtime;
    struct tm *timeinfo;
    char   tmpStr[64];

    time( &rawtime );
    timeinfo = localtime( &rawtime );
    fprintf(stdout,"\n\n PAW v%s\n %s", version, bar );
    fprintf(stdout," %s", asctime(timeinfo) );

    fprintf(stdout,"\n %s test in progress ... \n", testName );
    fprintf(stdout,"\n\tMemory Used Per Task : %6.1f MB", UsedMem / MB );
    fprintf(stdout,"\n\tTimer Overhead       : %6.1f ns", overhead / NSEC );
    fprintf(stdout,"\n\tTimer Threshold      : %6.1f ns\n", threshold_lo / NSEC );
    fflush(stdout);

    fprintf( fp,"%s%s%s%s%s", sbar, sver, sauth, smail, sbar );
    fprintf( fp,"# Date : %s", asctime(timeinfo) );
    fprintf( fp,"# Test : %s", testName );
    fprintf( fp2,"%s%s%s%s%s", sbar, sver, sauth, smail, sbar );
    fprintf( fp2,"# Date : %s", asctime(timeinfo) );
    fprintf( fp2,"# Test : %s", testName );

    if( strcmp( testName, "MPI_RATE" ) == 0 ){
        fprintf( fp,  "\n%s%s", sbar, smpi );
        fprintf( fp2, "\n%s%s", sbar, srate );
    }else if( strncmp( testName, "MPI", 3 ) == 0 || 
              strcmp( testName, "GPU_TRANSFER" ) == 0 ||
              strncmp( testName, "PHI", 3 ) == 0 ){
        fprintf( fp, "\n%s%s", sbar, smpi );
        fprintf( fp2,"\n%s%s", sbar, sbw );
    }else{
        fprintf( fp, "\n%s%s", sbar, slat );
        fprintf( fp2,"\n%s%s", sbar, sfp );
    }

    return 0;
}

int printLatencyHeader( FILE *fp, char *testName, double UsedMem, 
                        double overhead, double threshold_lo )
{
    FILE   *pipe;
    time_t rawtime;
    struct tm *timeinfo;
    char   tmpStr[64];

    time( &rawtime );
    timeinfo = localtime( &rawtime );
    fprintf(stdout,"\n\n PAW v%s\n %s", version, bar );
    fprintf(stdout," %s", asctime(timeinfo) );

    fprintf(stdout,"\n %s test in progress ... \n", testName );
    fprintf(stdout,"\n\tMemory Used Per Node : %6.1f MB", UsedMem / MB );
    fprintf(stdout,"\n\tTimer Overhead       : %6.1f ns", overhead / NSEC );
    fprintf(stdout,"\n\tTimer Threshold      : %6.1f ns\n", threshold_lo / NSEC );
    fflush(stdout);

    fprintf( fp,"%s%s%s%s%s", sbar, sver, sauth, smail, sbar );
    fprintf( fp,"# Date : %s", asctime(timeinfo) );
    fprintf( fp,"# Test : %s", testName );

    if( strncmp( testName, "MPI", 3 ) == 0 ){
        fprintf( fp, "\n%s%s", sbar, smpi );
    }else{
        fprintf( fp, "\n%s%s", sbar, slat );
    }

    return 0;
}

int printSummary( FILE *fp, char *testName, double localMax, 
                  unsigned int localSize )
{

	if( strcmp( testName, "MPI_RATE" ) == 0 ){
        fprintf( stdout,"\tMaximum Rate         : %6.1f MIO/s (N=%d)\n", localMax, 
                 localSize);
        fprintf( stdout,"\n %s test completed.\n\n", testName );
        fflush( stdout );

        fprintf( fp,"\n# Maximum Rate : %6.1f MIO/s (N=%d)\n", localMax, localSize );

    }else if( strncmp( testName, "MPI", 3 ) == 0 || 
        strcmp( testName, "GPU_TRANSFER" ) == 0 ||
        strncmp( testName, "PHI", 3 ) == 0 ){
        fprintf( stdout,"\tBW Maximum            : %6.1f MB/s (N=%d)\n", localMax, 
                 localSize);
        fprintf( stdout,"\n %s test completed.\n\n", testName );
        fflush( stdout );

        fprintf( fp,"\n# BW Maximum : %6.1f MB/s (N=%d)\n", localMax, localSize );

    }else{
        fprintf( stdout,"\tFastest %s        : %6.1f GFLOPS (N=%d)\n", testName, 
                 localMax, localSize );
        fprintf( stdout,"\n %s test completed.\n\n", testName );
        fflush( stdout );

        fprintf( fp, "\n# Fastest %s         : %6.1f GFLOPS (N=%d)\n", testName, 
                localMax, localSize );
    }

    return 0;
}

// Note that srand must be called from the parent program to get
// reproducible results 
double *doubleVector( unsigned long size )
{
    unsigned long i;
    double        invRND;
    double        *X;

    #ifdef MKL
        X = (double *)mkl_malloc( size*sizeof(double), ALIGNMENT );
    #else
        X = (double *)memalign( ALIGNMENT, size*sizeof(double) );
    #endif
    if( !X ) fatalError( "Failed to allocate memory for test." );

    invRND = 1.0E0 / (double)RAND_MAX;
    for( i = 0; i < size; i++ ) X[i] = invRND*rand();

    return X;
}

// Note that srand must be called from the parent program to get
// reproducible results 
float *floatVector( unsigned long size )
{
    unsigned long i;
    double        invRND;
    float        *X;

    #ifdef MKL
        X = (float *)mkl_malloc( size*sizeof(float), ALIGNMENT );
    #else
        X = (float *)memalign( ALIGNMENT, size*sizeof(float) );
    #endif
    if( !X ) fatalError( "Failed to allocate memory for test." );

    invRND = 1.0E0 / (double)RAND_MAX;
    for( i = 0; i < size; i++ ) X[i] = invRND*rand();

    return X;
}

int setLoopIters( double tMin, double tScale, double threshold_hi, 
                  unsigned int *NLOOP )
{
    double timeMin;

    timeMin = tMin*tScale*(*NLOOP);
    if( timeMin > threshold_hi ){
        (*NLOOP) = (*NLOOP)*threshold_hi / timeMin;
        if( (*NLOOP) < NLOOP_MIN ) *NLOOP = NLOOP_MIN;             
    }

    return 0;
}

int resetInnerLoop( double timeMin, double threshold_lo, unsigned int *NLOOP )
{

    if( timeMin < threshold_lo ){
        *NLOOP = (*NLOOP)*threshold_lo / timeMin;
        fprintf(stdout,"\n\tWarning : Internal loop too short for timer overhead.");
        fprintf(stdout,"\n\tWarning : Internal loop reset to NLOOP = %d\n", *NLOOP );
        fflush(stdout);
    }

    return 0;
}

// All quantities with the prefix "w" correspond to a transformation of the 
// timing values. In some cases this will be bandwidth and in others flops
// depending on the calling function.
int post_process( FILE *fp, FILE *fp2, double threshold_hi, double *tElapsed, 
                  double tScale, double wScale, unsigned int size, 
                  double sizeBytes, double work,
                  int *NLOOP, double *localMax, unsigned int *localSize )
{
    int    i;
    double tMin, tMax, tAvg, tDev, wMax, wMin, wAvg, wDev;
    double tMsg[NREPS], wMsg[NREPS];

    // Get the time and bw (or flops) per iteration
    for(i = 0; i < NREPS; i++){
        tMsg[i]  = tElapsed[i] / ( (double)(*NLOOP) );
        wMsg[i] = work /  tMsg[i];
    }

    // Calculate Average, Minimum, Maximum and standard deviation values
    stats( NREPS, tMsg, &tAvg, &tMax, &tMin, &tDev, tScale );
    stats( NREPS, wMsg, &wAvg, &wMax, &wMin, &wDev, wScale );

    // Save these results to file
    saveData( fp,  sizeBytes, *NLOOP, tAvg, tMax, tMin, tDev );
    saveData( fp2, sizeBytes, *NLOOP, wAvg, wMax, wMin, wDev );

    // Check if the bandwith is maximum for this size
    if( wMax > *localMax ){
        *localMax  = wMax;
        *localSize = size;
    }

    // Check for excessive iterations and reset NLOOP if needed
    setLoopIters( tMin, tScale, threshold_hi, NLOOP );

    return 0;
}


// Check for number of threads set in environment
// Use order preference used by MKL, default to 1 if not set
// Give warning if multiple variables set to different numbers
int threadCount( void )
{
    char *thChar;
    int  thNum[3], choice = 2;

    thNum[0] = 0; thNum[1] = 0; thNum[2] = 1;
    if( (thChar = getenv( "OMP_NUM_THREADS" )) != NULL ){
        thNum[0] = atoi( thChar );
        choice   = 0;
    }
    if( (thChar = getenv( "MKL_NUM_THREADS" )) != NULL ){
        thNum[1] = atoi( thChar );
        choice   = 1;
    }
    if( thNum[0] != 0 && thNum[1] != 0 && thNum[0] != thNum[1] )
        userWarning( "OMP and MKL NUM_THREADS differ" );
    if( choice == 2 )
        userWarning( "The number of threads has not been set for this test" );

    return thNum[choice];
}

// Check for environmental variables overrriding the default
// test settings for the BLAS tests
int checkEnvBLAS( unsigned int *NLOOP, unsigned int *smin, 
                  unsigned int *smed, unsigned int *smax  )
{
    if( getenv( "NLOOP_MAX" ) != NULL ){
        if( atoi( getenv( "NLOOP_MAX" ) ) <= 0 ) 
            fatalError("NLOOP_MAX must be a positive integer");
        else
            *NLOOP = atoi( getenv( "NLOOP_MAX" ) );
    }
    if( getenv( "MIN_P2P_SIZE" ) != NULL ){
        if( atoi( getenv( "MIN_P2P_SIZE" ) ) <= 0 ) 
            fatalError("MIN_P2P_SIZE must be a positive integer");
        else
            *smin = atoi( getenv( "MIN_P2P_SIZE" ) );
    }
    if( getenv( "MED_P2P_SIZE" ) != NULL ){
        if( atoi( getenv( "MED_P2P_SIZE" ) ) <= 0 ) 
            fatalError("MED_P2P_SIZE must be a positive integer");
        else
            *smed = atoi( getenv( "MED_P2P_SIZE" ) );
    }
    if( getenv( "MAX_P2P_SIZE" ) != NULL ){
        if( atoi( getenv( "MAX_P2P_SIZE" ) ) <= 0 ) 
            fatalError("MAX_P2P_SIZE must be a positive integer");
        else
            *smax = atoi( getenv( "MAX_P2P_SIZE" ) );
    }

    return 0;
}

// Check for environmental variables overrriding the default
// test settings for the GPU data transfer tests
int checkEnvGPU( unsigned int *NLOOP, unsigned int *smin, 
                     unsigned int *smed, unsigned int *smax  )
{
    if( getenv( "NLOOP_MAX" ) != NULL ){
        if( atoi( getenv( "NLOOP_MAX" ) ) <= 0 )
            fatalError("NLOOP_MAX must be a positive integer");
        else
            *NLOOP = atoi( getenv( "NLOOP_MAX" ) );
    }
    if( getenv( "MIN_GPU_SIZE" ) != NULL ){
    	if( atoi( getenv( "MIN_GPU_SIZE" ) ) <= 0 )
    		fatalError("MIN_GPU_BLAS_SIZE must be a positive integer");
    	else
            *smin = atoi( getenv( "MIN_GPU_SIZE" ) );
    }
    if( getenv( "MED_GPU_SIZE" ) != NULL ){
        if( atoi( getenv( "MED_GPU_SIZE" ) ) <= 0 ) 
            fatalError("MED_GPU_BLAS_SIZE must be a positive integer");
        else
            *smed = atoi( getenv( "MED_GPU_SIZE" ) );
    }
    if( getenv( "MAX_GPU_SIZE" ) != NULL ){
        if( atoi( getenv( "MAX_GPU_SIZE" ) ) <= 0 ) 
            fatalError("MAX_GPU_BLAS_SIZE must be a positive integer");
        else
            *smax = atoi( getenv( "MAX_GPU_SIZE" ) );
    }

    return 0;
}

// Check for environmental variables overrriding the default
// test settings for the GPU BLAS tests
int checkEnvGPUBLAS( unsigned int *NLOOP, unsigned int *smin, 
                     unsigned int *smed, unsigned int *smax  )
{
    if( getenv( "NLOOP_MAX" ) != NULL ){
        if( atoi( getenv( "NLOOP_MAX" ) ) <= 0 ) 
            fatalError("NLOOP_MAX must be a positive integer");
        else
            *NLOOP = atoi( getenv( "NLOOP_MAX" ) );
    }
    if( getenv( "MIN_GPU_BLAS_SIZE" ) != NULL ){
        if( atoi( getenv( "MIN_GPU_BLAS_SIZE" ) ) <= 0 ) 
            fatalError("MIN_GPU_BLAS_SIZE must be a positive integer");
        else
            *smin = atoi( getenv( "MIN_GPU_BLAS_SIZE" ) );
    }
    if( getenv( "MED_GPU_BLAS_SIZE" ) != NULL ){
        if( atoi( getenv( "MED_GPU_BLAS_SIZE" ) ) <= 0 ) 
            fatalError("MED_GPU_BLAS_SIZE must be a positive integer");
        else
            *smed = atoi( getenv( "MED_GPU_BLAS_SIZE" ) );
    }
    if( getenv( "MAX_GPU_BLAS_SIZE" ) != NULL ){
        if( atoi( getenv( "MAX_GPU_BLAS_SIZE" ) ) <= 0 ) 
            fatalError("MAX_GPU_BLAS_SIZE must be a positive integer");
        else
            *smax = atoi( getenv( "MAX_GPU_BLAS_SIZE" ) );
    }
    
    return 0;
}

// Check for environmental variables overrriding the default
// test settings for the PHI data transfer tests
int checkEnvPHI( unsigned int *NLOOP, unsigned int *smin, 
                  unsigned int *smed, unsigned int *smax  )
{
    if( getenv( "NLOOP_PHI_MAX" ) != NULL ){
        if( atoi( getenv( "NLOOP_PHI_MAX" ) ) <= 0 )
            fatalError("NLOOP_PHI_MAX must be a positive integer");
        else
            *NLOOP = atoi( getenv( "NLOOP_PHI_MAX" ) );
    }
    if( getenv( "MIN_PHI_SIZE" ) != NULL  ){
        if( atoi( getenv( "MIN_PHI_SIZE" ) ) <= 0 ) 
            fatalError("MIN_P2P_SIZE must be a positive integer");
        else
            *smin = atoi( getenv( "MIN_PHI_SIZE" ) );
    }
    if( getenv( "MED_PHI_SIZE" ) != NULL  ){
        if( atoi( getenv( "MED_PHI_SIZE" ) ) <= 0 ) 
            fatalError("MED_P2P_SIZE must be a positive integer");
        else
            *smed = atoi( getenv( "MED_PHI_SIZE" ) );
    }
    if( getenv( "MAX_PHI_SIZE" ) != NULL  ){
        if( atoi( getenv( "MAX_PHI_SIZE" ) ) <= 0 ) 
            fatalError("MAX_P2P_SIZE must be a positive integer");
        else
            *smax = atoi( getenv( "MAX_PHI_SIZE" ) );
    }

    return 0;
}

