# Makefile for PAW MPI tests
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
LFLAGS += $(ARCH_FLAGS)

# Object files
OBJS = aux.o aux_mpi.o
SRCS = ../aux/aux.c ../aux/aux.h ../aux/constants.h ../aux/aux_mpi.c ../aux/aux_mpi.h

# Build options
all: col_agtr col_a2a col_ardc col_bct col_gtr col_rdc col_sct col_acc col_acc_fence  \
     p2p_get p2p_isend p2p_put p2p_send p2p_isend_bidir p2p_isend_ping p2p_get_fence  \
     p2p_put_fence p2p_mrate p2p_mrate_bidir latency

# Linking Stage
latency         : $(OBJS) latency.o;              $(MPICC) $(LFLAGS) $(OBJS) latency.o               -o latency$(BIN_SUFFIX)
col_acc         : $(OBJS) col_accumulate.o;       $(MPICC) $(LFLAGS) $(OBJS) col_accumulate.o        -o col_acc$(BIN_SUFFIX)
col_acc_fence   : $(OBJS) col_accumulate_fence.o; $(MPICC) $(LFLAGS) $(OBJS) col_accumulate_fence.o  -o col_acc_fence$(BIN_SUFFIX)
col_agtr        : $(OBJS) col_allgather.o;   $(MPICC) $(LFLAGS) $(OBJS) col_allgather.o   -o col_agtr$(BIN_SUFFIX)
col_a2a         : $(OBJS) col_alltoall.o;    $(MPICC) $(LFLAGS) $(OBJS) col_alltoall.o    -o col_a2a$(BIN_SUFFIX)
col_ardc        : $(OBJS) col_allreduce.o;   $(MPICC) $(LFLAGS) $(OBJS) col_allreduce.o   -o col_ardc$(BIN_SUFFIX)
col_bct         : $(OBJS) col_broadcast.o;   $(MPICC) $(LFLAGS) $(OBJS) col_broadcast.o   -o col_bct$(BIN_SUFFIX)
col_gtr         : $(OBJS) col_gather.o;      $(MPICC) $(LFLAGS) $(OBJS) col_gather.o      -o col_gtr$(BIN_SUFFIX)
col_rdc         : $(OBJS) col_reduce.o;      $(MPICC) $(LFLAGS) $(OBJS) col_reduce.o      -o col_rdc$(BIN_SUFFIX)
col_sct         : $(OBJS) col_scatter.o;     $(MPICC) $(LFLAGS) $(OBJS) col_scatter.o     -o col_sct$(BIN_SUFFIX)
p2p_get         : $(OBJS) p2p_get.o;         $(MPICC) $(LFLAGS) $(OBJS) p2p_get.o         -o p2p_get$(BIN_SUFFIX)
p2p_get_fence   : $(OBJS) p2p_get_fence.o;   $(MPICC) $(LFLAGS) $(OBJS) p2p_get_fence.o   -o p2p_get_fence$(BIN_SUFFIX)
p2p_isend       : $(OBJS) p2p_isend.o;       $(MPICC) $(LFLAGS) $(OBJS) p2p_isend.o       -o p2p_isend$(BIN_SUFFIX)
p2p_put         : $(OBJS) p2p_put.o;         $(MPICC) $(LFLAGS) $(OBJS) p2p_put.o         -o p2p_put$(BIN_SUFFIX)
p2p_put_fence   : $(OBJS) p2p_put_fence.o;   $(MPICC) $(LFLAGS) $(OBJS) p2p_put_fence.o   -o p2p_put_fence$(BIN_SUFFIX)
p2p_send        : $(OBJS) p2p_send.o;        $(MPICC) $(LFLAGS) $(OBJS) p2p_send.o        -o p2p_send$(BIN_SUFFIX)
p2p_mrate       : $(OBJS) p2p_mrate.o;       $(MPICC) $(LFLAGS) $(OBJS) p2p_mrate.o       -o p2p_mrate$(BIN_SUFFIX)
p2p_mrate_bidir : $(OBJS) p2p_mrate_bidir.o; $(MPICC) $(LFLAGS) $(OBJS) p2p_mrate_bidir.o -o p2p_mrate_bidir$(BIN_SUFFIX)
p2p_isend_bidir : $(OBJS) p2p_isend_bidir.o; $(MPICC) $(LFLAGS) $(OBJS) p2p_isend_bidir.o -o p2p_isend_bidir$(BIN_SUFFIX)
p2p_isend_ping  : $(OBJS) p2p_isend_ping.o;  $(MPICC) $(LFLAGS) $(OBJS) p2p_isend_ping.o  -o p2p_isend_ping$(BIN_SUFFIX)


# Compilation Stage
aux.o            : $(SRCS);                  $(MPICC) $(CFLAGS) ../aux/aux.c
aux_mpi.o        : $(SRCS);                  $(MPICC) $(CFLAGS) ../aux/aux_mpi.c
latency.o        : $(SRCS) latency.c;        $(MPICC) $(CFLAGS) latency.c
col_accumulate.o       : $(SRCS) col_accumulate.c;       $(MPICC) $(CFLAGS) col_accumulate.c
col_accumulate_fence.o : $(SRCS) col_accumulate_fence.c; $(MPICC) $(CFLAGS) col_accumulate_fence.c
col_allgather.o  : $(SRCS) col_allgather.c;  $(MPICC) $(CFLAGS) col_allgather.c
col_alltoall.o   : $(SRCS) col_alltoall.c;   $(MPICC) $(CFLAGS) col_alltoall.c
col_allreduce.o  : $(SRCS) col_allreduce.c;  $(MPICC) $(CFLAGS) col_allreduce.c
col_broadcast.o  : $(SRCS) col_broadcast.c;  $(MPICC) $(CFLAGS) col_broadcast.c
col_gather.o     : $(SRCS) col_gather.c;     $(MPICC) $(CFLAGS) col_gather.c
col_reduce.o     : $(SRCS) col_reduce.c;     $(MPICC) $(CFLAGS) col_reduce.c
col_scatter.o    : $(SRCS) col_scatter.c;    $(MPICC) $(CFLAGS) col_scatter.c
p2p_get.o        : $(SRCS) p2p_get.c;        $(MPICC) $(CFLAGS) p2p_get.c
p2p_get_fence.o  : $(SRCS) p2p_get_fence.c;  $(MPICC) $(CFLAGS) p2p_get_fence.c
p2p_isend.o      : $(SRCS) p2p_isend.c;      $(MPICC) $(CFLAGS) p2p_isend.c
p2p_put.o        : $(SRCS) p2p_put.c;        $(MPICC) $(CFLAGS) p2p_put.c
p2p_put_fence.o  : $(SRCS) p2p_put_fence.c;  $(MPICC) $(CFLAGS) p2p_put_fence.c
p2p_send.o       : $(SRCS) p2p_send.c;       $(MPICC) $(CFLAGS) p2p_send.c
p2p_mrate.o      : $(SRCS) p2p_mrate.c;      $(MPICC) $(CFLAGS) p2p_mrate.c
p2p_mrate_bidir.o : $(SRCS) p2p_mrate_bidir.c; $(MPICC) $(CFLAGS) p2p_mrate_bidir.c
p2p_isend_bidir.o : $(SRCS) p2p_isend_bidir.c; $(MPICC) $(CFLAGS) p2p_isend_bidir.c
p2p_isend_ping.o  : $(SRCS) p2p_isend_ping.c;  $(MPICC) $(CFLAGS) p2p_isend_ping.c

#Installation
install:
	mkdir -p $(INSTALL_DIR)
	mv -v ./latency$(BIN_SUFFIX) ./col_agtr$(BIN_SUFFIX) ./col_a2a$(BIN_SUFFIX) ./col_ardc$(BIN_SUFFIX) $(INSTALL_DIR)
	mv -v ./col_bct$(BIN_SUFFIX) ./col_gtr$(BIN_SUFFIX) ./col_rdc$(BIN_SUFFIX) ./col_sct$(BIN_SUFFIX) $(INSTALL_DIR)
	mv -v ./col_acc$(BIN_SUFFIX) ./col_acc_fence$(BIN_SUFFIX) ./p2p_get_fence$(BIN_SUFFIX) ./p2p_put_fence$(BIN_SUFFIX) $(INSTALL_DIR)
	mv -v ./p2p_get$(BIN_SUFFIX) ./p2p_isend$(BIN_SUFFIX) ./p2p_put$(BIN_SUFFIX) ./p2p_send$(BIN_SUFFIX) $(INSTALL_DIR)
	mv -v ./p2p_isend_bidir$(BIN_SUFFIX) ./p2p_isend_ping$(BIN_SUFFIX) $(INSTALL_DIR)
	mv -v ./p2p_mrate$(BIN_SUFFIX) ./p2p_mrate_bidir$(BIN_SUFFIX) $(INSTALL_DIR)

# Clean
clean:
	rm -f ./*.o

clean-all:
	rm -f ./*.o
	rm -f ./latency$(BIN_SUFFIX) ./col_agtr$(BIN_SUFFIX) ./col_a2a$(BIN_SUFFIX) ./col_ardc$(BIN_SUFFIX)
	rm -f ./col_bct$(BIN_SUFFIX) ./col_gtr$(BIN_SUFFIX) ./col_rdc$(BIN_SUFFIX) ./col_sct$(BIN_SUFFIX)
	rm -f ./col_acc$(BIN_SUFFIX) ./col_acc_fence$(BIN_SUFFIX) ./p2p_get_fence$(BIN_SUFFIX) ./p2p_put_fence$(BIN_SUFFIX)
	rm -f ./p2p_get$(BIN_SUFFIX) ./p2p_isend$(BIN_SUFFIX) ./p2p_put$(BIN_SUFFIX) ./p2p_send$(BIN_SUFFIX)
	rm -f ./p2p_isend_bidir$(BIN_SUFFIX) ./p2p_isend_ping$(BIN_SUFFIX)
	rm -f ./p2p_mrate$(BIN_SUFFIX) ./p2p_mrate_bidir$(BIN_SUFFIX) 




