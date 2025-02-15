/*
* Serial Grey Wolf Optimization 
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <sys/time.h>


#define PI 3.14159265358979323846

/* 
* Computes the Rastrign function
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
void update_wolf(double *wolf, double *alpha, double *beta, double *delta, int DIM, double LB, double UB, double a){
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


void gwo(int DIM, int N_WOLVES, int MAX_ITER, double LB, double UB){

    // Declare indexes
    int i, j, iter;
    
    // Dynamically allocate memory
    double *wolves = (double*) malloc(sizeof(double) * N_WOLVES * DIM);
    double *fitness = (double*) malloc(sizeof(double) * N_WOLVES);
    if (wolves == NULL || fitness == NULL) {
        fprintf(stderr, "Memory allocation failed.\n");
        return ;
    }

    // Initialize wolf positions
    for (i = 0; i < N_WOLVES; i++) {
        for (j = 0; j < DIM; j++) {
            double r = ((double) rand()) / RAND_MAX;
            wolves[i * DIM + j] = LB + r * (UB - LB);
        }
        fitness[i] = fitness_function(&wolves[i * DIM], DIM);
    }

    // GWO Main Loop
    for (iter = 0; iter < MAX_ITER; iter++) {

        // Find the three best wolves (alpha, beta, delta)
        double alpha_fitness = DBL_MAX, beta_fitness = DBL_MAX, delta_fitness = DBL_MAX;
        int alpha_index, beta_index, delta_index = 0;

        // Find alpha, beta, delta
        for (i = 0; i < N_WOLVES; i++) {
            double fit = fitness[i];
            if (fit < alpha_fitness) {
                delta_fitness = beta_fitness;
                beta_fitness = alpha_fitness;
                alpha_fitness = fit;

                delta_index = beta_index;
                beta_index = alpha_index;
                alpha_index = i;

            } else if (fit < beta_fitness) {
                delta_fitness = beta_fitness;
                beta_fitness = fit;

                delta_index = beta_index;
                beta_index = i;
            } else if (fit < delta_fitness) {
                delta_fitness = fit;

                delta_index = i;
            }
        }
        
        // a decreases linearly from 2 to 0
        double a = 2.0 * (1.0 - ((double) iter / MAX_ITER));

        // Update wolf positions and fitnesses
        for(i = 0; i < N_WOLVES; i++) {
            update_wolf(
                &wolves[i * DIM],
                &wolves[alpha_index * DIM],
                &wolves[beta_index * DIM],
                &wolves[delta_index * DIM],
                DIM, LB, UB, a
            );
            fitness[i] = fitness_function(&wolves[i * DIM], DIM);
        }

        if(iter==MAX_ITER-1){
            printf("%f,", alpha_fitness);
        }

        //printf("Iteration %d, Best fitness so far: %f\n", iter, alpha_fitness);
    }


    // //Print the final best wolf
    // double final_best_score = DBL_MAX;
    // int best_index = -1;
    // for (i = 0; i < N_WOLVES; i++) {
    //     if (fitness[i] < final_best_score) {
    //         final_best_score = fitness[i];
    //         best_index = i;
    //     }
    // }

    // printf("\n==============================\n");
    // printf("Global Best Fitness: %f\n", final_best_score);
    // printf("Global Best Position:\n");
    // for (j = 0; j < DIM; j++) {
    //     printf("%f ", wolves[best_index * DIM + j]);
    // }

    free(wolves);
    free(fitness);

}



int main(int argc, char **argv) {

    struct timeval start, end;
    long long elapsed_usec;

    int DIM, N_WOLVES, MAX_ITER;
    double LB, UB; 
    

    // Get attributes from the command line
    if (argc < 4) {
        fprintf(stderr, "Usage: %s DIM N_WOLVES MAX_ITER [LB UB]\n", argv[0]);
        return 1;
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

    printf("Running: %s\n", argv[0]);
    printf("Simulation Parameters:\n");
    printf("  DIM        = %d\n", DIM);
    printf("  N_WOLVES   = %d\n", N_WOLVES);
    printf("  MAX_ITER   = %d\n", MAX_ITER);
    printf("  LB         = %f\n", LB);
    printf("  UB         = %f\n", UB);
 
    int run;
    for (run = 0; run < 1; run++) {

        // Seed the random number generator
        srand((unsigned int) time(NULL));    // Use current time as seed
                
        // Start timing
        gettimeofday(&start, NULL);
    
        gwo(DIM, N_WOLVES, MAX_ITER, LB, UB);
    
        // End timing
        gettimeofday(&end, NULL);

        // Calculate the elapsed time in microseconds
        elapsed_usec = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);


        printf("%lld\n", elapsed_usec);
    }
 
    return 0;
}