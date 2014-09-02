# Makefile for PAW Xeon Phi tests
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
# Basic definitions

SHELL = /bin/sh

CFLAGS += $(ARCH_FLAGS)
CFLAGS += $(OMP_FLAGS)
CFLAGS += $(PHI_CFLAGS)

LFLAGS += $(ARCH_FLAGS)
LFLAGS += $(OMP_FLAGS)
LFLAGS += $(PHI_LFLAGS)

# Object files
OBJS = aux.o
SRCS = ../aux/aux.c ../aux/aux.h ../aux/constants.h

#Link into an excutable
all : keep_al_in keep_al_out keep_al_inout keep_noal_in keep_noal_out \
      keep_noal_inout alloc_al_in alloc_al_out alloc_al_inout         \
      alloc_noal_in alloc_noal_out alloc_noal_inout

keep_al_in       : $(OBJS) transfer_keep_align_in.o;       $(CC) $(LFLAGS) $(OBJS) transfer_keep_align_in.o       -o phi_keep_al_in
keep_al_out      : $(OBJS) transfer_keep_align_out.o;      $(CC) $(LFLAGS) $(OBJS) transfer_keep_align_out.o      -o phi_keep_al_out
keep_al_inout    : $(OBJS) transfer_keep_align_inout.o;    $(CC) $(LFLAGS) $(OBJS) transfer_keep_align_inout.o    -o phi_keep_al_inout
keep_noal_in     : $(OBJS) transfer_keep_noalign_in.o;     $(CC) $(LFLAGS) $(OBJS) transfer_keep_noalign_in.o     -o phi_keep_noal_in
keep_noal_out    : $(OBJS) transfer_keep_noalign_out.o;    $(CC) $(LFLAGS) $(OBJS) transfer_keep_noalign_out.o    -o phi_keep_noal_out
keep_noal_inout  : $(OBJS) transfer_keep_noalign_inout.o;  $(CC) $(LFLAGS) $(OBJS) transfer_keep_noalign_inout.o  -o phi_keep_noal_inout
alloc_al_in      : $(OBJS) transfer_alloc_align_in.o;      $(CC) $(LFLAGS) $(OBJS) transfer_alloc_align_in.o      -o phi_alloc_al_in
alloc_al_out     : $(OBJS) transfer_alloc_align_out.o;     $(CC) $(LFLAGS) $(OBJS) transfer_alloc_align_out.o     -o phi_alloc_al_out
alloc_al_inout   : $(OBJS) transfer_alloc_align_inout.o;   $(CC) $(LFLAGS) $(OBJS) transfer_alloc_align_inout.o   -o phi_alloc_al_inout
alloc_noal_in    : $(OBJS) transfer_alloc_noalign_in.o;    $(CC) $(LFLAGS) $(OBJS) transfer_alloc_noalign_in.o    -o phi_alloc_noal_in
alloc_noal_out   : $(OBJS) transfer_alloc_noalign_out.o;   $(CC) $(LFLAGS) $(OBJS) transfer_alloc_noalign_out.o   -o phi_alloc_noal_out
alloc_noal_inout : $(OBJS) transfer_alloc_noalign_inout.o; $(CC) $(LFLAGS) $(OBJS) transfer_alloc_noalign_inout.o -o phi_alloc_noal_inout

#Compile the subroutines
aux.o                          : $(SRCS);                                 $(CC) $(CFLAGS) ../aux/aux.c
transfer_keep_align_in.o       : $(SRCS) transfer_keep_align_in.c;        $(CC) $(CFLAGS) transfer_keep_align_in.c
transfer_keep_align_out.o      : $(SRCS) transfer_keep_align_out.c;       $(CC) $(CFLAGS) transfer_keep_align_out.c
transfer_keep_align_inout.o    : $(SRCS) transfer_keep_align_inout.c;     $(CC) $(CFLAGS) transfer_keep_align_inout.c
transfer_keep_noalign_in.o     : $(SRCS) transfer_keep_noalign_in.c;      $(CC) $(CFLAGS) transfer_keep_noalign_in.c
transfer_keep_noalign_out.o    : $(SRCS) transfer_keep_noalign_out.c;     $(CC) $(CFLAGS) transfer_keep_noalign_out.c
transfer_keep_noalign_inout.o  : $(SRCS) transfer_keep_noalign_inout.c;   $(CC) $(CFLAGS) transfer_keep_noalign_inout.c
transfer_alloc_align_in.o      : $(SRCS) transfer_alloc_align_in.c;       $(CC) $(CFLAGS) transfer_alloc_align_in.c
transfer_alloc_align_out.o     : $(SRCS) transfer_alloc_align_out.c;      $(CC) $(CFLAGS) transfer_alloc_align_out.c
transfer_alloc_align_inout.o   : $(SRCS) transfer_alloc_align_inout.c;    $(CC) $(CFLAGS) transfer_alloc_align_inout.c
transfer_alloc_noalign_in.o    : $(SRCS) transfer_alloc_noalign_in.c;     $(CC) $(CFLAGS) transfer_alloc_noalign_in.c
transfer_alloc_noalign_out.o   : $(SRCS) transfer_alloc_noalign_out.c;    $(CC) $(CFLAGS) transfer_alloc_noalign_out.c
transfer_alloc_noalign_inout.o : $(SRCS) transfer_alloc_noalign_inout.c;  $(CC) $(CFLAGS) transfer_alloc_noalign_inout.c

# Installation
install:
	mkdir -p $(INSTALL_DIR)
	mv -v ./phi_* $(INSTALL_DIR)

# Clean
clean:
	rm -f ./*.o

clean-all:
	rm -f ./*.o ./phi_*

