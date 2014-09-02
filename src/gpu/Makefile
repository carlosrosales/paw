# Makefile for PAW GPU tests
# v1.0 (2014-08-27) Carlos Rosales-Fernandez
#------------------------------------------------------------------------------
# Copyright 2014 Carlos Rosales Fernandez and The University of Texas at Austin
#
# This file is part of the Performance Assessment Workbench (PAW).
# PAW is free software: you can redistribute it and/or modify it under the
# terms of the GNU GPL version 2 or (at your option) any later version.
#
# PAW is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# PAW, in the file COPYING.txt. If not, see <http://www.gnu.org/licenses/>.
#------------------------------------------------------------------------------
#
#  Basic definitions
SHELL  = /bin/sh

CFLAGS += $(ARCH_FLAGS)
CFLAGS += $(CUDA_CFLAGS)

LFLAGS += $(ARCH_FLAGS)
LFLAGS += $(CUDA_LFLAGS)
LFLAGS_BLAS = $(LFLAGS) $(CUDA_LFLAGS_BLAS)

SRC       = ../aux/aux.c ../aux/aux.h ../aux/constants.h
OBJ_BW1   = gpu_transfer_in_pinned.o aux.o 
OBJ_BW2   = gpu_transfer_out_pinned.o aux.o
OBJ_BW3   = gpu_transfer_inout_pinned.o aux.o
OBJ_BW4   = gpu_transfer_in_nopin.o aux.o
OBJ_BW5   = gpu_transfer_out_nopin.o aux.o
OBJ_BW6   = gpu_transfer_inout_nopin.o aux.o
OBJ_SGEMM = gpu_sgemm.o aux.o 
OBJ_DGEMM = gpu_dgemm.o aux.o 

# Build options
bw    : gpu_in_pinned gpu_out_pinned gpu_inout_pinned gpu_in_nopin gpu_out_nopin gpu_inout_nopin 
blas  : gpu_sgemm gpu_dgemm
all   : gpu_info gpu_in_pinned gpu_out_pinned gpu_inout_pinned gpu_in_nopin gpu_out_nopin gpu_inout_nopin gpu_sgemm gpu_dgemm

# Linking Stage
gpu_info         : gpu_properties.cu; $(NVCC) gpu_properties.cu -o gpu_info
gpu_in_pinned    : $(OBJ_BW1);   $(CC) $(OBJ_BW1)   $(LFLAGS) -o gpu_in_pinned
gpu_out_pinned   : $(OBJ_BW2);   $(CC) $(OBJ_BW2)   $(LFLAGS) -o gpu_out_pinned
gpu_inout_pinned : $(OBJ_BW3);   $(CC) $(OBJ_BW3)   $(LFLAGS) -o gpu_inout_pinned
gpu_in_nopin     : $(OBJ_BW4);   $(CC) $(OBJ_BW4)   $(LFLAGS) -o gpu_in_nopin
gpu_out_nopin    : $(OBJ_BW5);   $(CC) $(OBJ_BW5)   $(LFLAGS) -o gpu_out_nopin
gpu_inout_nopin  : $(OBJ_BW6);   $(CC) $(OBJ_BW6)   $(LFLAGS) -o gpu_inout_nopin
gpu_sgemm        : $(OBJ_SGEMM); $(CC) $(OBJ_SGEMM) $(LFLAGS_BLAS) -o gpu_sgemm
gpu_dgemm        : $(OBJ_DGEMM); $(CC) $(OBJ_DGEMM) $(LFLAGS_BLAS) -o gpu_dgemm

# Compilation Stage
gpu_transfer_in_pinned.o    : $(SRC) gpu_transfer_in_pinned.c;    $(CC) $(CFLAGS) gpu_transfer_in_pinned.c
gpu_transfer_out_pinned.o   : $(SRC) gpu_transfer_out_pinned.c;   $(CC) $(CFLAGS) gpu_transfer_out_pinned.c
gpu_transfer_inout_pinned.o : $(SRC) gpu_transfer_inout_pinned.c; $(CC) $(CFLAGS) gpu_transfer_inout_pinned.c
gpu_transfer_in_nopin.o     : $(SRC) gpu_transfer_in_nopin.c;     $(CC) $(CFLAGS) gpu_transfer_in_nopin.c
gpu_transfer_out_nopin.o    : $(SRC) gpu_transfer_out_nopin.c;    $(CC) $(CFLAGS) gpu_transfer_out_nopin.c
gpu_transfer_inout_nopin.o  : $(SRC) gpu_transfer_inout_nopin.c;  $(CC) $(CFLAGS) gpu_transfer_inout_nopin.c
gpu_sgemm.o                 : $(SRC) gpu_sgemm.c;                 $(CC) $(CFLAGS) gpu_sgemm.c
gpu_dgemm.o                 : $(SRC) gpu_dgemm.c;                 $(CC) $(CFLAGS) gpu_dgemm.c
aux.o                       : $(SRC);                             $(CC) $(CFLAGS) ../aux/aux.c

# Installation
install:
	mkdir -p $(INSTALL_DIR)
	mv -v ./gpu_in* ./gpu_out* ./gpu_sgemm ./gpu_dgemm $(INSTALL_DIR)

# Clean
clean:
	rm -f ./*.o

clean-all:
	rm -f ./*.o ./gpu_in* ./gpu_out* ./gpu_sgemm ./gpu_dgemm

