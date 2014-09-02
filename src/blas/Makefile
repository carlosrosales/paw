# Makefile for PAW BLAS tests
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

CFLAGS += $(OMP_FLAGS)
CFLAGS += $(ARCH_FLAGS)
CFLAGS += $(BLAS_CFLAGS)

LFLAGS += $(OMP_FLAGS)
LFLAGS += $(ARCH_FLAGS)
LFLAGS += $(BLAS_LFLAGS)

SRC       = ../aux/aux.c ../aux/aux.h ../aux/constants.h
OBJ_DGEMM = blas_dgemm.o aux.o 
OBJ_DGEMV = blas_dgemv.o aux.o 
OBJ_SGEMM = blas_sgemm.o aux.o 
OBJ_SGEMV = blas_sgemv.o aux.o 

# Build options
all: blas_dgemm blas_sgemm blas_dgemv blas_sgemv

# Linking Stage
blas_dgemm  : $(OBJ_DGEMM); $(CC) $(OBJ_DGEMM) $(LFLAGS) -o blas_dgemm$(BIN_SUFFIX)
blas_dgemv  : $(OBJ_DGEMV); $(CC) $(OBJ_DGEMV) $(LFLAGS) -o blas_dgemv$(BIN_SUFFIX)
blas_sgemm  : $(OBJ_SGEMM); $(CC) $(OBJ_SGEMM) $(LFLAGS) -o blas_sgemm$(BIN_SUFFIX)
blas_sgemv  : $(OBJ_SGEMV); $(CC) $(OBJ_SGEMV) $(LFLAGS) -o blas_sgemv$(BIN_SUFFIX)

# Compilation Stage
blas_dgemm.o  : $(SRC) blas_dgemm.c;  $(CC) $(CFLAGS) blas_dgemm.c
blas_dgemv.o  : $(SRC) blas_dgemv.c;  $(CC) $(CFLAGS) blas_dgemv.c
blas_sgemm.o  : $(SRC) blas_sgemm.c;  $(CC) $(CFLAGS) blas_sgemm.c
blas_sgemv.o  : $(SRC) blas_sgemv.c;  $(CC) $(CFLAGS) blas_sgemv.c
aux.o         : $(SRC);               $(CC) $(CFLAGS) ../aux/aux.c

# Installation
install:
	mkdir -p $(INSTALL_DIR)
	mv -v ./blas_dgemm$(BIN_SUFFIX) ./blas_dgemv$(BIN_SUFFIX) ./blas_sgemm$(BIN_SUFFIX) ./blas_sgemv$(BIN_SUFFIX) $(INSTALL_DIR)

# Clean
clean:
	rm -f ./*.o

clean-all:
	rm -f ./*.o ./blas_dgemm$(BIN_SUFFIX) ./blas_dgemv$(BIN_SUFFIX) ./blas_sgemm$(BIN_SUFFIX) ./blas_sgemv$(BIN_SUFFIX)


