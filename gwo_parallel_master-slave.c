/* BASELINE VERSION */
/* TWO ARRAYS */
/* PROCESS 0 ACTS AS COORDINATOR */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

#define PI 3.14159265358979323846
#define DBL_MAX 1.7976931348623157e+308

/* 
 * Computes the Rastrigin function
 */
double fitness_function(double *position, int dim) {
    int i;
    double f = 10 * dim;
    for (i = 0; i < dim; i++) {
        f += position[i] * position[i] - 10 * cos(2 * PI * position[i]);
    }
    return f;
}

/* 
 * Wolf position update guided by alpha, beta, and delta 
 */
void update_wolf(double *wolf, double *alpha, double *beta, double *delta, int DIM, double LB, double UB, double a) {
    int j;
    double r1, r2, A1, A2, A3, C1, C2, C3;
    double D_alpha, D_beta, D_delta;
    double X1, X2, X3, new_pos;

    for (j = 0; j < DIM; j++) {
        r1 = ((double) rand()) / RAND_MAX;
        r2 = ((double) rand()) / RAND_MAX;
        A1 = 2.0 * a * r1 - a;
        C1 = 2.0 * r2;
        D_alpha = fabs(C1 * alpha[j] - wolf[j]);
        X1 = alpha[j] - A1 * D_alpha;

        r1 = ((double) rand()) / RAND_MAX;
        r2 = ((double) rand()) / RAND_MAX;
        A2 = 2.0 * a * r1 - a;
        C2 = 2.0 * r2;
        D_beta = fabs(C2 * beta[j] - wolf[j]);
        X2 = beta[j] - A2 * D_beta;

        r1 = ((double) rand()) / RAND_MAX;
        r2 = ((double) rand()) / RAND_MAX;
        A3 = 2.0 * a * r1 - a;
        C3 = 2.0 * r2;
        D_delta = fabs(C3 * delta[j] - wolf[j]);
        X3 = delta[j] - A3 * D_delta;

        new_pos = (X1 + X2 + X3) / 3.0;
        if (new_pos < LB) {
            new_pos = LB;
        }
        if (new_pos > UB) {
            new_pos = UB;
        }
        wolf[j] = new_pos;
    }
}

void parallel_gwo(int DIM, int N_WOLVES, int MAX_ITER, double LB, double UB, int rank, int size) {

    // Declare all variables at the beginning
    int wolves_per_proc, rem, iter, i, j;
    double *local_positions, *local_fitness;
    double *global_positions = NULL, *global_fitness = NULL;
    double global_alpha_fitness, global_beta_fitness, global_delta_fitness;
    double *global_alpha_position, *global_beta_position, *global_delta_position;
    double a;

    // Calculate local population size
    wolves_per_proc = N_WOLVES / size;
    rem = N_WOLVES % size;
    if (rank < rem) {
        wolves_per_proc++;
    }

    // Allocate memory for local wolves and fitness
    local_positions = (double *)malloc(wolves_per_proc * DIM * sizeof(double));
    local_fitness = (double *)malloc(wolves_per_proc * sizeof(double));

    if (local_positions == NULL || local_fitness == NULL) {
        fprintf(stderr, "Memory allocation failed for local arrays.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Initialize wolf positions and fitness
    for (i = 0; i < wolves_per_proc; i++) {
        for (j = 0; j < DIM; j++) {
            double r = ((double)rand()) / RAND_MAX;
            local_positions[i * DIM + j] = LB + r * (UB - LB);
        }
        local_fitness[i] = fitness_function(&local_positions[i * DIM], DIM);
    }

    // Allocate memory for global positions and fitness (root process only)
    if (rank == 0) {
        global_positions = (double *)malloc(N_WOLVES * DIM * sizeof(double));
        global_fitness = (double *)malloc(N_WOLVES * sizeof(double));

        if (global_positions == NULL || global_fitness == NULL) {
            fprintf(stderr, "Memory allocation failed for global arrays.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Allocate memory for global best positions
    global_alpha_position = (double *)malloc(DIM * sizeof(double));
    global_beta_position = (double *)malloc(DIM * sizeof(double));
    global_delta_position = (double *)malloc(DIM * sizeof(double));

    if (global_alpha_position == NULL || global_beta_position == NULL || global_delta_position == NULL) {
        fprintf(stderr, "Memory allocation failed for global best positions.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Initialize global best fitness values
    global_alpha_fitness = DBL_MAX;
    global_beta_fitness = DBL_MAX;
    global_delta_fitness = DBL_MAX;

    // Main GWO loop
    for (iter = 0; iter < MAX_ITER; iter++) {

        // Gather all positions and fitness values to the root process
        MPI_Gather(local_positions, wolves_per_proc * DIM, MPI_DOUBLE,
                   global_positions, wolves_per_proc * DIM, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(local_fitness, wolves_per_proc, MPI_DOUBLE,
                   global_fitness, wolves_per_proc, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Find global alpha, beta, and delta (root process only)
        if (rank == 0) {
            int alpha_index = 0, beta_index = 0, delta_index = 0;

            for (i = 0; i < N_WOLVES; i++) {
                if (global_fitness[i] < global_alpha_fitness) {
                    global_delta_fitness = global_beta_fitness;
                    global_beta_fitness = global_alpha_fitness;
                    global_alpha_fitness = global_fitness[i];

                    delta_index = beta_index;
                    beta_index = alpha_index;
                    alpha_index = i;
                } else if (global_fitness[i] < global_beta_fitness) {
                    global_delta_fitness = global_beta_fitness;
                    global_beta_fitness = global_fitness[i];

                    delta_index = beta_index;
                    beta_index = i;
                } else if (global_fitness[i] < global_delta_fitness) {
                    global_delta_fitness = global_fitness[i];
                    delta_index = i;
                }
            }

            // Copy the best positions
            for (j = 0; j < DIM; j++) {
                global_alpha_position[j] = global_positions[alpha_index * DIM + j];
                global_beta_position[j] = global_positions[beta_index * DIM + j];
                global_delta_position[j] = global_positions[delta_index * DIM + j];
            }
        }

        // Broadcast global best positions to all processes
        MPI_Bcast(global_alpha_position, DIM, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(global_beta_position, DIM, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(global_delta_position, DIM, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Update local wolves
        a = 2.0 * (1.0 - (double)iter / MAX_ITER);
        for (i = 0; i < wolves_per_proc; i++) {
            update_wolf(&local_positions[i * DIM], global_alpha_position, global_beta_position, global_delta_position, DIM, LB, UB, a);
            local_fitness[i] = fitness_function(&local_positions[i * DIM], DIM);
        }

        // Print progress (root process only)
        if (iter==MAX_ITER-1 && rank == 0) {
            printf("Iter %d: Best Fitness = %f\n", iter, global_alpha_fitness);
        }
    }

    // Free memory
    free(local_positions);
    free(local_fitness);
    free(global_alpha_position);
    free(global_beta_position);
    free(global_delta_position);

    if (rank == 0) {
        free(global_positions);
        free(global_fitness);
    }
}

int main(int argc, char **argv) {
    // Declare all variables at the beginning
    double start, end;
    double local_elapsed_usec, global_elapsed_usec;
    int rank, size;
    int DIM, N_WOLVES, MAX_ITER;
    double LB, UB;
    int run;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Get attributes from the command line
    if (rank == 0) {
        if (argc < 4) {
            fprintf(stderr, "Usage: %s DIM N_WOLVES MAX_ITER [LB UB]\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        DIM = atoi(argv[1]);
        N_WOLVES = atoi(argv[2]);
        MAX_ITER = atoi(argv[3]);
        if (argc >= 6) {
            LB = atof(argv[4]);
            UB = atof(argv[5]);
        } else {
            LB = -100.0;
            UB = 100.0;
        }

        printf("Running: %s with %d processes.\n", argv[0], size);
        printf("Simulation Parameters:\n");
        printf("  DIM        = %d\n", DIM);
        printf("  N_WOLVES   = %d\n", N_WOLVES);
        printf("  MAX_ITER   = %d\n", MAX_ITER);
        printf("  LB         = %f\n", LB);
        printf("  UB         = %f\n", UB);
    }

    // Broadcast parameters to all processes
    MPI_Bcast(&DIM, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N_WOLVES, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&MAX_ITER, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&LB, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&UB, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Warm-up run
    srand((unsigned int) (time(NULL) + (rank) * 100));
    if (rank == 0) {
        printf("Warmup\n");
    }
    parallel_gwo(DIM, N_WOLVES, MAX_ITER, LB, UB, rank, size);
    if (rank == 0) {
        printf("\n");
    }

    // Main timing loop
    for (run = 0; run < 10; run++) {
        // Seed the random number generator
        srand((unsigned int) (time(NULL) + (rank + run) * 100));

        // Synchronize all processes before starting the run
        MPI_Barrier(MPI_COMM_WORLD);

        // Start timing
        start = MPI_Wtime();

        parallel_gwo(DIM, N_WOLVES, MAX_ITER, LB, UB, rank, size);

        // End timing
        end = MPI_Wtime();  

        // Calculate the elapsed time in microseconds
        local_elapsed_usec = (end - start);

        // Use MPI_Reduce to find the maximum time across all processes
        MPI_Reduce(&local_elapsed_usec, &global_elapsed_usec, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        // Print the elapsed time (root process only)
        if (rank == 0) {
            printf("elapsed %f\n", global_elapsed_usec);
        }
    }

    // Finalize MPI
    MPI_Finalize();

    return 0;
}