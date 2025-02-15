#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <time.h>

#define PI 3.14159265358979323846

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
    for (j = 0; j < DIM; j++) {
        double r1, r2, A1, A2, A3, C1, C2, C3;
        double D_alpha, D_beta, D_delta;
        double X1, X2, X3, new_pos;

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

    // Declare indexes
    int i, j, iter;

    // Calculate local population size
    int local_population_size = N_WOLVES / size;

    // For better load balancing distribute the remainder among the first few processes.
    int rem = N_WOLVES % size;
    if (rank < rem) {
        local_population_size++;
    }

    // Candidate count (alpha, beta, delta)
    // Each process finds its 3 local candidates,
    // then gathers the candidates from the others,
    // and finally selects the 3 global best.
    int candidate_count = 3;
    
    // Dynamically allocate memory for local wolves and their fitnesses
    double *local_wolves = malloc(local_population_size * DIM * sizeof(double));
    double *local_fitness = malloc(local_population_size * sizeof(double));
    if (local_wolves == NULL || local_fitness == NULL) {
        fprintf(stderr, "Memory allocation failed.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Initialize wolf positions
    for (i = 0; i < local_population_size; i++) {
        for (j = 0; j < DIM; j++) {
            double r = ((double) rand()) / RAND_MAX;
            local_wolves[i * DIM + j] = LB + r * (UB - LB);
        }
        local_fitness[i] = fitness_function(&local_wolves[i * DIM], DIM);
    }
    
    // Each candidate is represented by (DIM+1) doubles:
    // 1 for fitness and DIM for position
    int candidate_data_size = DIM + 1;  

    // Allocate buffer for local candidate data (for alpha, beta, delta)
    double *local_candidates_buffer = malloc(candidate_count * candidate_data_size * sizeof(double));
    if (local_candidates_buffer == NULL) {
        fprintf(stderr, "Memory allocation failed for local candidates buffer.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Allocate buffer for global candidates gathered from all processes
    double *global_candidates_buffer = malloc(size * candidate_count * candidate_data_size * sizeof(double));
    if (global_candidates_buffer == NULL) {
        fprintf(stderr, "Memory allocation failed for global candidates buffer.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // GWO Main Loop
    for (iter = 0; iter < MAX_ITER; iter++) {    

        // Find the three best wolves (alpha, beta, delta)
        double alpha_fitness = DBL_MAX, beta_fitness = DBL_MAX, delta_fitness = DBL_MAX;
        int alpha_index = -1, beta_index = -1, delta_index = -1;
        
        for (i = 0; i < local_population_size; i++) {
            double fit = local_fitness[i];
            if (fit < alpha_fitness) {
                delta_fitness = beta_fitness;
                delta_index = beta_index;

                beta_fitness = alpha_fitness;
                beta_index = alpha_index;

                alpha_fitness = fit;
                alpha_index = i;
            } else if (fit < beta_fitness) {
                delta_fitness = beta_fitness;
                delta_index = beta_index;

                beta_fitness = fit;
                beta_index = i;
            } else if (fit < delta_fitness) {
                delta_fitness = fit;
                delta_index = i;
            }
        }

        // Pack the local top candidates into the local_candidates_buffer.
        // For each candidate:
        //   - The first double stores the fitness.
        //   - The next DIM doubles store the position.
        // Candidate 0: alpha
        local_candidates_buffer[0 * candidate_data_size] = alpha_fitness;
        for (j = 0; j < DIM; j++) {
            local_candidates_buffer[0 * candidate_data_size + 1 + j] = local_wolves[alpha_index * DIM + j];
        }
        // Candidate 1: beta
        local_candidates_buffer[1 * candidate_data_size] = beta_fitness;
        for (j = 0; j < DIM; j++) {
            local_candidates_buffer[1 * candidate_data_size + 1 + j] = local_wolves[beta_index * DIM + j];
        }
        // Candidate 2: delta
        local_candidates_buffer[2 * candidate_data_size] = delta_fitness;
        for (j = 0; j < DIM; j++) {
            local_candidates_buffer[2 * candidate_data_size + 1 + j] = local_wolves[delta_index * DIM + j];
        }

        // All processes share their local top candidates
        MPI_Allgather(local_candidates_buffer, candidate_count * candidate_data_size, MPI_DOUBLE,
                      global_candidates_buffer, candidate_count * candidate_data_size, MPI_DOUBLE,
                      MPI_COMM_WORLD);

        // Similarly to local search of alpha, beta and delta
        // Determine the global alpha, beta, and delta wolves
        alpha_fitness = DBL_MAX, beta_fitness = DBL_MAX, delta_fitness = DBL_MAX;
        int global_alpha_index = 0, global_beta_index = 0, global_delta_index = 0;
        int total_candidates = candidate_count * size;

        for (i = 0; i < total_candidates; i++) {
            double fit = global_candidates_buffer[i * candidate_data_size];
            if (fit < alpha_fitness) {
                delta_fitness = beta_fitness;
                delta_index = beta_index;

                beta_fitness = alpha_fitness;
                beta_index = alpha_index;

                alpha_fitness = fit;
                alpha_index = i;
            } else if (fit < beta_fitness) {
                delta_fitness = beta_fitness;
                delta_index = beta_index;

                beta_fitness = fit;
                beta_index = i;
            } else if (fit < delta_fitness) {
                delta_fitness = fit;
                delta_index = i;
            }
        }

        // Print progress
        if (iter==MAX_ITER-1 && rank == 0) {
            printf("%f,", alpha_fitness);
        }

        // a decreases linearly from 2 to 0
        double a = 2.0 * (1.0 - ((double) iter / MAX_ITER));

        // Get pointers to the global best positions.
        double *global_alpha_position = &global_candidates_buffer[global_alpha_index * candidate_data_size + 1];
        double *global_beta_position  = &global_candidates_buffer[global_beta_index  * candidate_data_size + 1];
        double *global_delta_position = &global_candidates_buffer[global_delta_index * candidate_data_size + 1];

        // Update wolf positions and fitnesses
        for (i = 0; i < local_population_size; i++) {
            update_wolf(&local_wolves[i * DIM],
                        global_alpha_position,
                        global_beta_position,
                        global_delta_position,
                        DIM, LB, UB, a);
            local_fitness[i] = fitness_function(&local_wolves[i * DIM], DIM);
        }

    }
    
    // Clean up
    free(local_wolves);
    free(local_fitness);
    free(local_candidates_buffer);
    free(global_candidates_buffer);
}
    

 
int main(int argc, char **argv) {

    double start, end;
    double local_elapsed_usec, global_elapsed_usec;

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // gwo parameters
    int DIM, N_WOLVES, MAX_ITER;
    double LB, UB; 

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

    int run;
    for (run = 0; run < 10; run++) {

        // Seed the random number generator
        // srand(time(NULL)); // Use current time as seed
        srand((unsigned int) (time(NULL) + (rank+run) * 100));

        // Synchronize all processes before starting the run
        MPI_Barrier(MPI_COMM_WORLD);

        // Start timing
        start = MPI_Wtime();

        parallel_gwo(DIM, N_WOLVES, MAX_ITER, LB, UB, rank, size);

        // End timing
        end = MPI_Wtime();  
        
        // Calculate the elapsed time in microseconds
        local_elapsed_usec = (end - start);

        // Use MPI_Reduce to find the maximum time across all processes.
        MPI_Reduce(&local_elapsed_usec, &global_elapsed_usec, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            printf("%f\n", global_elapsed_usec);
        }
    }

    MPI_Finalize();

    return 0;
}