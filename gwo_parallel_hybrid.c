#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <time.h>

#define PI 3.14159265358979323846

// Return a random double between 0 and 1.
double rand_double() {
    return (double)rand() / (double)RAND_MAX;
}

/* 
 * Computes the Rastrigin function (minimization).
 */
double fitness_function(double *position, int dim) {
    double f = 10 * dim;
    for (int i = 0; i < dim; i++) {
        f += position[i] * position[i] - 10 * cos(2 * PI * position[i]);
    }
    return f;
}

/* 
 * Updates a wolf’s position based on the three best wolves:
 * alpha, beta, and delta.
 * The update is performed for each dimension in the subcomponent.
 */
void update_wolf(double *wolf, double *alpha, double *beta, double *delta,
                 int dim, double LB, double UB, double a) {
    for (int j = 0; j < dim; j++) {
        double r1 = rand_double();
        double r2 = rand_double();
        double A1 = 2.0 * a * r1 - a;
        double C1 = 2.0 * r2;
        double D_alpha = fabs(C1 * alpha[j] - wolf[j]);
        double X1 = alpha[j] - A1 * D_alpha;

        r1 = rand_double();
        r2 = rand_double();
        double A2 = 2.0 * a * r1 - a;
        double C2 = 2.0 * r2;
        double D_beta = fabs(C2 * beta[j] - wolf[j]);
        double X2 = beta[j] - A2 * D_beta;

        r1 = rand_double();
        r2 = rand_double();
        double A3 = 2.0 * a * r1 - a;
        double C3 = 2.0 * r2;
        double D_delta = fabs(C3 * delta[j] - wolf[j]);
        double X3 = delta[j] - A3 * D_delta;

        double new_pos = (X1 + X2 + X3) / 3.0;

        // Enforce the bounds.
        if (new_pos < LB)
            new_pos = LB;
        if (new_pos > UB)
            new_pos = UB;
        wolf[j] = new_pos;
    }
}

/*
 * The parallel version of the Grey Wolf Optimizer.
 *
 * The full solution of dimension DIM is split evenly among the processes.
 * Each process holds a subcomponent of dimension sub_dim = DIM/size.
 */
void parallel_gwo(int DIM, int N_WOLVES, int MAX_ITER, double LB, double UB,
                  int rank, int size) {

    N_WOLVES = N_WOLVES / size;
    // Each process handles a subcomponent of dimension sub_dim.
    int sub_dim = DIM / size;

    // Allocate local population. Each wolf (candidate) is represented by an array of sub_dim values.
    double **wolves = (double **)malloc(N_WOLVES * sizeof(double *));
    for (int i = 0; i < N_WOLVES; i++) {
        wolves[i] = (double *)malloc(sub_dim * sizeof(double));
    }

    // Allocate array to store the fitness values of the local candidates.
    double *wolf_fitness = (double *)malloc(N_WOLVES * sizeof(double));

    // Each process keeps track of its local best candidate (its “alpha” wolf for its subcomponent).
    double *sub_best = (double *)malloc(sub_dim * sizeof(double));
    double sub_best_fit = DBL_MAX;

    // global_best will hold the full solution (the concatenation of the best subcomponents from all processes).
    double *global_best = (double *)malloc(DIM * sizeof(double));

    // -------------------------
    // Initialization: Randomly initialize the local population.
    for (int i = 0; i < N_WOLVES; i++) {
        for (int j = 0; j < sub_dim; j++) {
            wolves[i][j] = LB + (UB - LB) * rand_double();
        }
    }

    // Initialize local best using the first candidate.
    memcpy(sub_best, wolves[0], sub_dim * sizeof(double));

    // Build an initial full solution by gathering each process’s sub_best.
    MPI_Allgather(sub_best, sub_dim, MPI_DOUBLE, global_best, sub_dim, MPI_DOUBLE, MPI_COMM_WORLD);

    // Evaluate each candidate in the local population.
    // The full solution is built by taking:
    //  - the candidate’s own subcomponent for this process, and
    //  - the best known subcomponents from other processes (from global_best).
    for (int i = 0; i < N_WOLVES; i++) {
        double *full_sol = (double *)malloc(DIM * sizeof(double));
        for (int p = 0; p < size; p++) {
            if (p == rank) {
                for (int j = 0; j < sub_dim; j++) {
                    full_sol[p * sub_dim + j] = wolves[i][j];
                }
            } else {
                for (int j = 0; j < sub_dim; j++) {
                    full_sol[p * sub_dim + j] = global_best[p * sub_dim + j];
                }
            }
        }
        double fit = fitness_function(full_sol, DIM);
        wolf_fitness[i] = fit;
        if (fit < sub_best_fit) {
            sub_best_fit = fit;
            memcpy(sub_best, wolves[i], sub_dim * sizeof(double));
        }
        free(full_sol);
    }

    // -------------------------
    // Main optimization loop.
    double a; // Parameter that decreases linearly from 2 to 0.
    int idx_alpha, idx_beta, idx_delta;  // Declare indexes outside the update loop.
    for (int iter = 0; iter < MAX_ITER; iter++) {
        a = 2.0 - 2.0 * ((double)iter / MAX_ITER);

        // Share the current best subcomponents with all processes.
        MPI_Allgather(sub_best, sub_dim, MPI_DOUBLE, global_best, sub_dim, MPI_DOUBLE, MPI_COMM_WORLD);

        // Re-evaluate all candidates using the current global best information.
        for (int i = 0; i < N_WOLVES; i++) {
            double *full_sol = (double *)malloc(DIM * sizeof(double));
            for (int p = 0; p < size; p++) {
                if (p == rank) {
                    for (int j = 0; j < sub_dim; j++) {
                        full_sol[p * sub_dim + j] = wolves[i][j];
                    }
                } else {
                    for (int j = 0; j < sub_dim; j++) {
                        full_sol[p * sub_dim + j] = global_best[p * sub_dim + j];
                    }
                }
            }
            double fit = fitness_function(full_sol, DIM);
            wolf_fitness[i] = fit;
            if (fit < sub_best_fit) {
                sub_best_fit = fit;
                memcpy(sub_best, wolves[i], sub_dim * sizeof(double));
            }
            free(full_sol);
        }

        // Identify the top three wolves (alpha, beta, delta) in the local population.
        idx_alpha = 0;
        idx_beta  = 0;
        idx_delta = 0;
        double best1 = DBL_MAX, best2 = DBL_MAX, best3 = DBL_MAX;
        for (int i = 0; i < N_WOLVES; i++) {
            if (wolf_fitness[i] < best1) {
                best3 = best2;
                idx_delta = idx_beta;
                best2 = best1;
                idx_beta = idx_alpha;
                best1 = wolf_fitness[i];
                idx_alpha = i;
            } else if (wolf_fitness[i] < best2) {
                best3 = best2;
                idx_delta = idx_beta;
                best2 = wolf_fitness[i];
                idx_beta = i;
            } else if (wolf_fitness[i] < best3) {
                best3 = wolf_fitness[i];
                idx_delta = i;
            }
        }

        // Update positions (subcomponents) of all wolves using the update_wolf function.
        // Each wolf (candidate) is updated using the best three (alpha, beta, delta) found.
        for (int i = 0; i < N_WOLVES; i++) {
            update_wolf(wolves[i],
                        wolves[idx_alpha],
                        wolves[idx_beta],
                        wolves[idx_delta],
                        sub_dim, LB, UB, a);
        }

        // Optionally, print progress (only from rank 0).
        if (rank == 0 && iter==MAX_ITER-1) {
            printf("%f,\n", sub_best_fit);
        }
    }

    // Gather the final best subcomponents from all processes to form the full solution.
    MPI_Allgather(sub_best, sub_dim, MPI_DOUBLE, global_best, sub_dim, MPI_DOUBLE, MPI_COMM_WORLD);
    double final_fitness = fitness_function(global_best, DIM);

    if (rank == 0) {
        printf("%f,", final_fitness);
        // printf("Best solution:\n");
        // for (int i = 0; i < DIM; i++) {
        //     printf("%f ", global_best[i]);
        // }
        // printf("\n");
    }

    // Free allocated memory.
    for (int i = 0; i < N_WOLVES; i++) {
        free(wolves[i]);
    }
    free(wolves);
    free(wolf_fitness);
    free(sub_best);
    free(global_best);
}

int main(int argc, char **argv) {
    double start, end;
    double local_elapsed, global_elapsed;

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // GWO parameters.
    int DIM, N_WOLVES, MAX_ITER;
    double LB, UB; 

    // Get attributes from the command line.
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
    }

    // Broadcast parameters to all processes.
    MPI_Bcast(&DIM, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N_WOLVES, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&MAX_ITER, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&LB, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&UB, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Run multiple independent runs.
    for (int run = 0; run < 30; run++) {
        // Seed the random number generator (varying the seed per run and rank).
        srand((unsigned int)(time(NULL) + (rank + run) * 100));

        // Synchronize all processes before starting the run.
        MPI_Barrier(MPI_COMM_WORLD);
        start = MPI_Wtime();

        parallel_gwo(DIM, N_WOLVES, MAX_ITER, LB, UB, rank, size);

        end = MPI_Wtime();
        local_elapsed = end - start;

        // Use MPI_Reduce to get the maximum elapsed time across all processes.
        MPI_Reduce(&local_elapsed, &global_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Run %d: %f seconds\n", run, global_elapsed);
        }
    }

    MPI_Finalize();
    return 0;
}
