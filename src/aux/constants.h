/*==============================================================================
 * File     : aux.h
 * Revision : 1.5 (2015-10-14)
 * Author   : Carlos Rosales Fernandez [carlos.rosales.fernandez(at)gmail.com]
 *==============================================================================
 * Copyright 2015 Carlos Rosales Fernandez and The University of Texas at Austin
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
 * Constants used in PAW.
 *============================================================================*/

// PAW version number
#define version "1.5"

// Some useful strings
#define bar   "=================================\n"
#define sbar  "#=======================================================\n"
#define slat  "#\n# SIZE\t\tAVG(s)\t\t\t\tMIN(s)\t\t\t\tMAX(s)\t\t\tstdDev(s)\t\t\tNLOOP\tNREPS\n"
#define sfp   "#\n# SIZE\t\t\tAVG(GFP/s)\t\t\tMIN(GFP/s)\t\t\tMAX(GFP/s)\t\tstdDev(GFP/s)\t\tNLOOP\tNREPS\n"
#define sbw   "#\n# SIZE(B)\t\tAVG(MB/s)\t\t\tMIN(MB/s)\t\t\tMAX(MB/s)\t\tstdDev(MB/s)\t\tNLOOP\tNREPS\n"
#define smpi  "#\n# SIZE(B)\t\tAVG(us)\t\t\t\tMIN(us)\t\t\t\tMAX(us)\t\t\tstdDev(us)\t\t\tNLOOP\tNREPS\n"
#define srate "#\n# SIZE(B)\t\tAVG(MIO/s)\t\t\tMIN(MIO/s)\t\t\tMAX(MIO/s)\t\tstdDev(MIO/s)\t\tNLOOP\tNREPS\n"
#define sver  "# PAW - Performance Assessment Workbench v1.5\n#\n"
#define sauth "# Carlos Rosales-Fernandez\n"
#define smail "# carlos.rosales.fernandez@gmail.com\n"

#define KB_8 1024.0
#define MB_8 1048576.0
#define GB_8 1073741820.0

// A bunch of useful constants
#define KB  1.0E3
#define MB  1.0E6
#define GB  1.0E9
#define SEC   1.0
#define USEC  1.0E-6
#define NSEC  1.0E-9
#define GFLOP 1.0E9

// Shorthand for Xeon Phi offload pragma options
#define ALLOC  alloc_if(1)
#define FREE   free_if(1)
#define KEEP   free_if(0)
#define REUSE  alloc_if(0)

// Array alignment in bytes
#define ALIGNMENT 64

// Seed for random number generator
#define SEED 8713U

// Timer safety margin
// We will use the measured timer overhead and multiply it by these two numbers
// to control the number of inner loop iterations needed to achieve accurate results
#define TIMER_REPS 1000000UL
#define TIMER_LO 100.0
#define TIMER_HI 1000.0

// Number of internal loop iterations for tests
#define NLOOP_MIN 2
#define NLOOP_MAX 1000
#define NLOOP_PHI_MAX 100

// Number of times a measurement is repeated
#define NREPS 10

// Linear size of matrix used in the CPU floating point calculation (DGEMM)
// Note that the memory required by that test is ( 8 * 3 * NDGEMM * NDGEMM )
#define MIN_BLAS_SIZE 8
#define MED_BLAS_SIZE 1024
#define MAX_BLAS_SIZE 10000

// Message sizes for Point To Point Bandwidth calculation
// This is a linear size, it will by X(sizeofDouble) in bytes
//  64 KBytes = 65536   B, 128 KBytes = 131072  B
// 256 KBytes = 262144  B, 512 KBytes = 524288  B 
// 768 KBytes = 786432  B,   1 MByte  = 1048576 B
//   2 MBytes = 2097152 B,   4 MBytes = 4194304 B
//   6 MBytes = 6291456 B,   8 MBytes = 8388608 B
#define MIN_P2P_SIZE 1
#define MED_P2P_SIZE 20000
#define MAX_P2P_SIZE 1000000

// Message sizes for MPI collective tests 
// This is reset dynamically for MPI_Alltoall to avoid excessive memory usage
#define MIN_COL_SIZE 1
#define MED_COL_SIZE 10000
#define MAX_COL_SIZE 100000

// Default window size for MPI message rate benchmarks
#define DEFAULT_WINDOW_SIZE 128

// Sizes for GPU data exchange
#define MIN_GPU_SIZE 128
#define MED_GPU_SIZE 20000
#define MAX_GPU_SIZE 200000000

// Sizes for GPU BLAS test
#define MIN_GPU_BLAS_SIZE 8
#define MED_GPU_BLAS_SIZE 1024
#define MAX_GPU_BLAS_SIZE 9192

// Sizes for Xeon Phi data exchange
#define MIN_PHI_SIZE 256
#define MED_PHI_SIZE 8192
#define MAX_PHI_SIZE 614400000



