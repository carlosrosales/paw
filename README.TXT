=== PAW v1.3 ===

The Performance Assessment Workbench (PAW) is a suite of simple codes designed 
to test baseline performance in high performance computing systems.

PAW provides codes to test floating point performance usign BLAS routines, MPI 
performance of both point to point and collective operations, and also specific 
codes to test NVIDIA GPU and Intel Xeon Phi coprocessor performance.

There are some areas of performance where excellent benchmarks already exist, 
and this project aims to complement those rather than substitute them. Examples 
of recommended benchmarks that complement PAW well are:

1. EPCC OMP benchmark,           https://www.epcc.ed.ac.uk/research/software/
2. STREAM (memory bandwidth),    http://www.cs.virginia.edu/stream/
3. IOR (parallel filesystem IO), https://github.com/chaos/ior

For detailed functionality and implementation read the User Guide associated to 
this release.

The Performance Assessment Workbench code is provided as Open Source under the 
GPLv2 license (See COPYING.TXT).

=== INSTALLATION ===

Modify compilation and linking flags in Makefile.in, then simply build all 
benchmark executables using:

make all
make all install

You may also choose to build only a subset of test codes:

make blas && make blas-install
make mpi && make mpi-install
make gnu && make gnu-install
make phi && make phi-install

The executables will be placed in the /bin directory unless a PREFIX
is passed to Make in the installation step.

=== CHANGELOG ===

Revision 1.3: (2014-12-17) Added MPI-3.0 point to point tests
Revision 1.2: (2014-11-19) Minor update to output text banners to show correct version
Revision 1.1: (2014-09-08) Bugfix - corrected error in p2p_isend_ping.c
Revision 1.0: (2014-08-27) First public release of PAW

2014-12-17 \ Carlos Rosales Fernandez \ carlos.rosales.fernandez@gmail.com





