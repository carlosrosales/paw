# Script for system information collection
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
mkdir ./info

# This is the kernel we are running
uname -r > ./info/kernel.txt

# This is the node memory in GigaBytes
head -n 1 /proc/meminfo | awk '{print $2/1000/1024}' > ./info/total_mem.txt

# This is the name of the processor
grep "model name" /proc/cpuinfo | uniq | awk -F: '{print $2}' > ./info/cpu_model.txt

# This is the number of sockets in the node
grep "physical id" /proc/cpuinfo | uniq -c | wc -l > ./info/total_sockets.txt

# This is the number of cores in the node
grep -c "cpu cores" /proc/cpuinfo > ./info/total_cores.txt


