#!/bin/bash
#PBS -l select=1:ncpus=1:mem=2gb
#PBS -l walltime=0:20:00
#PBS -q short_cpuQ

module load mpich-3.2
mpirun.actual -n 1 ./gwo/gwo_serial 10000 512 1000 -10 10
# mpirun.actual -n 1 ./gwo/gwo_serial DIM N_WOLVES MAX_ITER LB UB