/*==============================================================================
 * File     : aux.h
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
 * Auxiliary function declarations.
 *============================================================================*/

double benchTimer( void );

int post_process( FILE *fp, FILE *fp2, double threshold_hi, double *tElapsed, 
                  double tScale, double wScale, unsigned int size, 
                  double sizeBytes, double work,
                  int *NLOOP, double *localMax, unsigned int *localSize );

#ifdef CUDABUILD
extern "C" int fatalError( char *errorText );
extern "C" int stats(  int N, double *x, double *xAvg, double *xMax, double *xMin, 
                       double *xDev, double scale );
extern "C" int saveData( FILE *fp, double size, int NLOOP, double xAvg, 
                         double xMax, double xMin, double xDev );
extern "C" int printHeaders( FILE *fp, FILE *fp2, char *testName, double usedMem, 
                             double overhead, double threshold_lo );
extern "C" int printSummary( FILE *fp, char *testName, double localMax, 
                             unsigned int localSize );
extern "C" float *floatVector( unsigned long size );
extern "C" int setLoopIters( double tMin, double tScale, double threshold_hi, 
                             unsigned int *NLOOP );
extern "C" int resetInnerLoop( double timeMin, double threshold_lo, unsigned int *NLOOP );
#else
int fatalError( char *errorText );
int stats(  int N, double *x, double *xAvg, double *xMax, double *xMin, 
            double *xDev, double scale );
int saveData( FILE *fp, double size, int NLOOP, double xAvg, 
              double xMax, double xMin, double xDev );
int printHeaders( FILE *fp, FILE *fp2, char *testName, double usedMem, 
                  double overhead, double threshold_lo );
int printSummary( FILE *fp, char *testName, double localMax, 
                  unsigned int localSize );
float *floatVector( unsigned long size );
int setLoopIters( double tMin, double tScale, double threshold_hi, 
                  unsigned int *NLOOP );
int resetInnerLoop( double timeMin, double threshold_lo, unsigned int *NLOOP );
#endif

int userWarning( char *warningText );

int printLatencyHeader( FILE *fp, char *testName, double usedMem, 
                        double overhead, double threshold_lo );

double *doubleVector( unsigned long size );

int timerTest( double *overhead, double *threshold_lo, double *threshold_hi );

int threadCount( void );

