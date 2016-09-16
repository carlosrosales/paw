#!/bin/sh
#------------------------------------------------------------------------------
# Copyright 2016 Carlos Rosales Fernandez and The University of Texas at Austin
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
# Makefile for PAW
#
# Change CC, MPICC and the corresponding flags to match your own compiler in
# file "Makefile.in". You should not have to edit this file at all.
#
# v1.6 (2016-09-16)  Carlos Rosales Fernandez

include ./Makefile.in

COPYRIGHT1="Copyright 2016 The University of Texas at Austin."
COPYRIGHT2="License: GNU GPL version 2 <http://gnu.org/licenses/gpl.html>"
COPYRIGHT3="This is free software: you are free to change and redistribute it."
COPYRIGHT4="There is NO WARRANTY, to the extent permitted by law."

BUILD_LOG="`pwd`/paw_build.log"
INSTALL_LOG="`pwd`/paw_install.log"

SEPARATOR="======================================================================"
PKG      ="Package  : PAW"
VER      ="Version  : 1.6"
DATE     ="Date     : `date +%Y.%m.%d`"
SYSTEM   ="System   : `uname -sr`"
COMPILER ="Compiler : `$(CC) --version | head -n 1`"

core: logs blas-core-build mpi-core-build stream-core-build
paw:  logs blas-build mpi-build check-build
all:  logs blas-build mpi-build gpu-build phi-build check-build
blas: logs blas-build check-build
mpi:  logs mpi-build check-build
mpi3: logs mpi3-build check-build
gpu:  logs gpu-build check-build
phi:  logs phi-build check-build

install:      blas-raw-install mpi-raw-install check-install
core-install: blas-core-install mpi-core-install stream-core-install check-install
all-install:  blas-raw-install mpi-raw-install gpu-raw-install phi-raw-install check-install
blas-install: blas-raw-install check-install
mpi-install:  mpi-raw-install check-install
mpi3-install: mpi3-raw-install check-install
gpu-install:  gpu-raw-install check-install
phi-install:  phi-raw-install check-install

logs:
# Initialize the log files
	@touch $(BUILD_LOG)
	@touch $(INSTALL_LOG)

# Record the local conditions for the compilation
	@echo
	@echo $(SEPARATOR)  | tee $(BUILD_LOG)
	@echo $(PKG)        | tee -a $(BUILD_LOG)
	@echo $(VER)        | tee -a $(BUILD_LOG)
	@echo $(DATE)       | tee -a $(BUILD_LOG)
	@echo $(SYSTEM)     | tee -a $(BUILD_LOG)
	@echo $(COMPILER)   | tee -a $(BUILD_LOG)
	@echo $(SEPARATOR)  | tee -a $(BUILD_LOG)
	@echo $(COPYRIGHT1) | tee -a $(BUILD_LOG)
	@echo $(COPYRIGHT2) | tee -a $(BUILD_LOG)
	@echo $(COPYRIGHT3) | tee -a $(BUILD_LOG)
	@echo $(COPYRIGHT4) | tee -a $(BUILD_LOG)
	@echo $(SEPARATOR)  | tee -a $(BUILD_LOG)
	@echo               | tee -a $(BUILD_LOG)

	@echo "Starting build..."                   | tee -a $(BUILD_LOG)
	@echo "Working Directory : `pwd`"           | tee -a $(BUILD_LOG)
	@echo                                       | tee -a $(BUILD_LOG)

blas-core-build:
# Core Floating Point Test Codes
	@echo "Generating core BLAS test..."            | tee -a $(BUILD_LOG)
	@$(MAKE) --directory=`pwd`/src/blas core   2>&1 | tee -a $(BUILD_LOG)
	@echo

blas-core-install:
# Core Floating Point Test Install
	@echo "Installing core BLAS test executable..."       |  tee -a $(INSTALL_LOG)
	@$(MAKE) --directory=`pwd`/src/blas core-install 2>&1 | tee -a $(INSTALL_LOG)
	@echo   

blas-build:
# Floating Point Test Codes
	@echo "Generating BLAS tests..."               |  tee -a $(BUILD_LOG)
	@$(MAKE) --directory=`pwd`/src/blas all   2>&1 | tee -a $(BUILD_LOG)
	@echo                                          |  tee -a $(BUILD_LOG)

blas-raw-install:
# Floating Point Test Install
	@echo "Installing BLAS test executables..."      |  tee -a $(INSTALL_LOG)
	@$(MAKE) --directory=`pwd`/src/blas install 2>&1 | tee -a $(INSTALL_LOG)
	@echo                                            |  tee -a $(INSTALL_LOG)

mpi-build:
# MPI Test Codes
	@echo "Generating MPI tests..."                |  tee -a $(BUILD_LOG)
	@$(MAKE) --directory=`pwd`/src/mpi all    2>&1 | tee -a $(BUILD_LOG)
	@echo                                          |  tee -a $(BUILD_LOG)

mpi-raw-install:
# MPI Test Install
	@echo "Installing MPI test executables..."      |  tee -a $(INSTALL_LOG)
	@$(MAKE) --directory=`pwd`/src/mpi install 2>&1 | tee -a $(INSTALL_LOG)
	@echo                                           |  tee -a $(INSTALL_LOG)

mpi-core-build:
# Core MPI Test Codes
	@echo "Generating core MPI tests..."            |  tee -a $(BUILD_LOG)
	@$(MAKE) --directory=`pwd`/src/mpi core    2>&1 | tee -a $(BUILD_LOG)
	@echo                                           |  tee -a $(INSTALL_LOG)

mpi-core-install:
# Core MPI Test Install
	@echo "Installing core MPI test executables..."      |  tee -a $(INSTALL_LOG)
	@$(MAKE) --directory=`pwd`/src/mpi core-install 2>&1 | tee -a $(INSTALL_LOG)
	@echo                                                |  tee -a $(INSTALL_LOG)

mpi3-build:
# MPI-3.0 Test Codes
	@echo "Generating MPI-3.0 tests..."              |  tee -a $(BUILD_LOG)
	@$(MAKE) --directory=`pwd`/src/mpi/mpi3 all 2>&1 | tee -a $(BUILD_LOG)
	@echo                                            |  tee -a $(BUILD_LOG)

mpi3-raw-install:
# MPI Test Install
	@echo "Installing MPI-3.0 test executables..."       |  tee -a $(INSTALL_LOG)
	@$(MAKE) --directory=`pwd`/src/mpi/mpi3 install 2>&1 | tee -a $(INSTALL_LOG)
	@echo                                                |  tee -a $(INSTALL_LOG)

gpu-build:
# GPU Test Codes
	@echo "Generating GPU specific tests..."      |  tee -a $(BUILD_LOG)
	@$(MAKE) --directory=`pwd`/src/gpu all   2>&1 | tee -a $(BUILD_LOG)
	@echo                                         |  tee -a $(BUILD_LOG)

gpu-raw-install:
# GPU Test Install
	@echo "Installing GPU specific test executables..."      |  tee -a $(INSTALL_LOG)
	@$(MAKE) --directory=`pwd`/src/gpu install          2>&1 | tee -a $(INSTALL_LOG)
	@echo  

phi-build:
# Phi Test Codes
	@echo "Generating Phi specific tests..."      |  tee -a $(BUILD_LOG)
	@$(MAKE) --directory=`pwd`/src/phi all   2>&1 | tee -a $(BUILD_LOG)
	@echo                                         |  tee -a $(BUILD_LOG)

phi-raw-install:
# Phi Test Install
	@echo "Installing Phi specific test executables..."      |  tee -a $(INSTALL_LOG)
	@$(MAKE) --directory=`pwd`/src/phi install          2>&1 | tee -a $(INSTALL_LOG)
	@echo                                                    |  tee -a $(INSTALL_LOG)

stream-core-build:
	@echo "Generating core STREAM test..."                           |  tee -a $(BUILD_LOG)
	@$(MAKE) --directory=`pwd`/extras -f Makefile.stream core   2>&1 | tee -a $(BUILD_LOG)
	@echo                                                            |  tee -a $(BUILD_LOG)

stream-core-install:
	@echo "Installing core STREAM test executable..."                        |  tee -a $(INSTALL_LOG)
	@$(MAKE) --directory=`pwd`/extras -f Makefile.stream core-install   2>&1 |  tee -a $(INSTALL_LOG)
	@echo                                                                    |  tee -a $(INSTALL_LOG)

clean:
# Cleanup all directories
	@$(MAKE) --directory=`pwd`/src/blas clean
	@$(MAKE) --directory=`pwd`/src/mpi clean
	@$(MAKE) --directory=`pwd`/src/mpi/mpi3 clean
	@$(MAKE) --directory=`pwd`/src/gpu clean
	@$(MAKE) --directory=`pwd`/src/phi clean
	@rm -f ./extras/*.o


distclean:
# Cleanup all directories and binaries
	@$(MAKE) --directory=`pwd`/src/blas clean
	@$(MAKE) --directory=`pwd`/src/mpi clean
	@$(MAKE) --directory=`pwd`/src/mpi/mpi3 clean
	@$(MAKE) --directory=`pwd`/src/gpu clean
	@$(MAKE) --directory=`pwd`/src/phi clean
	rm ./bin/*
	rm ./*.log

check-build:
	@echo "Build completed. Check $(BUILD_LOG) for details."

check-install:
	@echo "Installation completed. Check $(INSTALL_LOG) for details."

