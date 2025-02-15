#!/bin/bash
#PBS -l select=1:ncpus=4:mem=2gb -l place=pack:excl
#PBS -l walltime=0:30:00
#PBS -q short_cpuQ

module load mpich-3.2
mpirun.actual -n 4 ./gwo/gwo_parallel_distributed 10000 1024 1000 -10 10
# mpirun.actual -n 4 ./gwo/gwo_parallel_distributed DIM N_WOLVES MAX_ITER LB UB