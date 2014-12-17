# Makefile for PAW MPI tests
# v1.3 (2014-12-17) Carlos Rosales-Fernandez
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

CFLAGS += $(ARCH_FLAGS) -I../../aux
LFLAGS += $(ARCH_FLAGS)

# Object files
OBJS = aux.o aux_mpi.o
SRCS = ../../aux/aux.c ../../aux/aux.h ../../aux/constants.h ../../aux/aux_mpi.c ../../aux/aux_mpi.h

# Build options
all: col_racc p2p_rget p2p_rput

# Linking Stage
col_racc : $(OBJS) col_raccumulate.o; $(MPICC) $(LFLAGS) $(OBJS) col_raccumulate.o -o col_racc$(BIN_SUFFIX)
p2p_rget : $(OBJS) p2p_rget.o;        $(MPICC) $(LFLAGS) $(OBJS) p2p_rget.o        -o p2p_rget$(BIN_SUFFIX)
p2p_rput : $(OBJS) p2p_rput.o;        $(MPICC) $(LFLAGS) $(OBJS) p2p_rput.o        -o p2p_rput$(BIN_SUFFIX)



# Compilation Stage
aux.o             : $(SRCS);                   $(MPICC) $(CFLAGS) ../../aux/aux.c
aux_mpi.o         : $(SRCS);                   $(MPICC) $(CFLAGS) ../../aux/aux_mpi.c
col_raccumulate.o : $(SRCS) col_raccumulate.c; $(MPICC) $(CFLAGS) col_raccumulate.c
p2p_rget.o        : $(SRCS) p2p_rget.c;        $(MPICC) $(CFLAGS) p2p_rget.c
p2p_rput.o        : $(SRCS) p2p_rput.c;        $(MPICC) $(CFLAGS) p2p_rput.c

#Installation
install:
	mkdir -p ../$(INSTALL_DIR)
	mv -v ./col_racc$(BIN_SUFFIX) ./p2p_rget$(BIN_SUFFIX) ./p2p_rput$(BIN_SUFFIX)  ../$(INSTALL_DIR)

# Clean
clean:
	rm -f ./*.o

clean-all:
	rm -f ./*.o
	rm -f ./col_raccumulate$(BIN_SUFFIX) ./p2p_rget$(BIN_SUFFIX) ./p2p_rput$(BIN_SUFFIX) 




