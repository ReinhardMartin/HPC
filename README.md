# High-Performance Computing Implementation of Grey Wolf Optimizer (GWO)

## Overview

This project provides different implementations of the Grey Wolf Optimizer (GWO) algorithm, focusing on high-performance computing (HPC) approaches. It includes:

- A **serial implementation** of GWO.
- Two **parallel implementations** where the population is split across multiple processors.
- A **hybrid parallel implementation** where both the population and dimensions are split across processors.
- **Bash scripts** to execute each implementation using the Portable Batch System (PBS) for job scheduling on an HPC cluster.

## File Structure

```
📂 project_directory
│-- gwo_serial.c                    # Serial implementation of GWO
│-- gwo_parallel_master-slave.c     # Parallel implementation with master-slave communication pattern
│-- gwo_parallel_distributed.c		# Parallel implementation with fully-distributed communication pattern
│-- gwo_parallel_hybrid.c           # Parallel implementation splitting both population and dimensions
│-- gwo_serial.sh             # PBS script for serial implementation
│-- gwo_parallel_master-slave.sh # PBS script for master-slave implementation
│-- gwo_parallel_distributed.sh  # PBS script for fully-distributed implementation
│-- gwo_parallel_hybrid.sh       # PBS script for hybrid implementation
│-- README.md                       # This documentation file
```

## Dependencies

To compile and run the parallel implementations, ensure you have:

- A C compiler (e.g., `gcc`)
- MPI (Message Passing Interface) installed (`mpicc` for compilation)
- Access to an HPC cluster with PBS job scheduling

## Compilation

Compile the implementations using the following commands:

```bash
# Serial version
gcc -o gwo_serial gwo_serial.c -lm

# Parallel versions (MPI required)
mpicc -o gwo_parallel_master-slave gwo_parallel_master-slave.c -lm
mpicc -o gwo_parallel_population2 gwo_parallel_distributed.c -lm
mpicc -o gwo_parallel_hybrid gwo_parallel_hybrid.c -lm
```

## Running the Implementations

### Running Locally

For the serial implementation:

```bash
./gwo_serial
```

For the parallel implementations:

```bash
mpirun -np <num_processes> ./gwo_parallel_master-slave
mpirun -np <num_processes> ./gwo_parallel_distributed
mpirun -np <num_processes> ./gwo_parallel_hybrid
```

### Running on an HPC Cluster (PBS)

Use the bash script associated with your desired version to submit a job:

```bash
qsub gwo_serial.sh
qsub gwo_parallel_master-slave.sh
qsub gwo_parallel_distributed.sh
qsub gwo_parallel_hybrid.sh
```

