/*==============================================================================
 * File     : aux_mpi.h
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


int checkEnvCOL( int proc, unsigned int *NLOOP, unsigned int *smin, 
                  unsigned int *smed, unsigned int *smax  );


int checkEnvP2P( int proc, unsigned int *NLOOP, unsigned int *smin, 
                  unsigned int *smed, unsigned int *smax  );


int checkEnvP2P( int proc, unsigned int *windowSize, unsigned int *NLOOP, 
	             unsigned int *smin, unsigned int *smed, unsigned int *smax  );