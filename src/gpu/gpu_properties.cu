/*==============================================================================
 * Program  : gpu_properties
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
 * Test CUDA GPU properties.
 *============================================================================*/

#include <stdio.h>
#include <assert.h>
#include <cuda.h>

int main(void)
{
	cudaDeviceProp deviceProp;
	int nDevCount = 0;
    FILE *fp;

	cudaGetDeviceCount( &nDevCount );
    fp = fopen( "gpuinfo.dat", "w" );
	fprintf( fp, "\nNumber of Devices found: %d", nDevCount );
	for (int nDeviceIdx = 0; nDeviceIdx < nDevCount; ++nDeviceIdx ){
		memset( &deviceProp, 0, sizeof(deviceProp));
		if( cudaSuccess == cudaGetDeviceProperties(&deviceProp, nDeviceIdx) ){
	        fprintf( fp, "\nDevice Name : %s ", deviceProp.name );
	        fprintf( fp, "\n=====================================");
	        fprintf( fp, "\nClock rate                    : %4.2f GHz", (double)deviceProp.clockRate / 1.0E6 );
	        fprintf( fp, "\nGlobal memory                 : %4.2f GB", (double)deviceProp.totalGlobalMem / 1.0E9 );
	        fprintf( fp, "\nCUDA ver                      : %d.%d", deviceProp.major, deviceProp.minor );
	        fprintf( fp, "\nNumber of Multi processors    : %d", deviceProp.multiProcessorCount );
	        fprintf( fp, "\nShared memory per block       : %d KB", deviceProp.sharedMemPerBlock/1024 );
	        fprintf( fp, "\nConstant memory               : %d bytes", deviceProp.totalConstMem );
	        fprintf( fp, "\nRegisters per thread block    : %d", deviceProp.regsPerBlock );
            fprintf( fp, "\nECC Enabled                   : %d", deviceProp.ECCEnabled );
            fprintf( fp, "\n" );
        }
		else
			printf( "\n%s", cudaGetErrorString(cudaGetLastError()));
	}
    fclose( fp );

  return 0;

}

