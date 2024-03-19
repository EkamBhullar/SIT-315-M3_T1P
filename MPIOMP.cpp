#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <omp.h>

#define DIM 1000 // Define the size of the matrices
#define MAX_RAND 10 // Define the maximum value for the random generator

/* Matrix containers */
int matrix1[DIM][DIM];
int matrix2[DIM][DIM];
int result[DIM][DIM];

/* Function declarations */
void headProcess(int proc, int start, int end); // Head process 
void nodeProcess(int proc, int start, int end); // Node processes
void createMatrix(int matrix[DIM][DIM]); // Function to create a matrix
void outputMatrix(int matrix[DIM][DIM]); // Function to output a matrix

/* Struct to hold various values */
struct Variables
{
    int proc_rank, proc, start, end, num_threads;
    double t_start, t_stop;
};

Variables vars = {0, 0, 0, 0, 2, 0.0, 0.0}; // Initializing variables with default values

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv); // Initialize MPI environment
    MPI_Comm_rank(MPI_COMM_WORLD, &vars.proc_rank); // Get current process ID
    MPI_Comm_size(MPI_COMM_WORLD, &vars.proc); // Get number of processes 

    /* Assigning start and end indices for computation */
    vars.start = vars.proc_rank * DIM / vars.proc; 
    vars.end = ((vars.proc_rank + 1) * DIM / vars.proc);

    if (vars.proc_rank == 0){
        vars.t_start = omp_get_wtime(); // Start time for the master process
    }

    if (vars.proc_rank == 0)
    {
        headProcess(vars.proc, vars.start, vars.end);
    }
    else
    {
        nodeProcess(vars.proc, vars.start, vars.end);
    }

    /* Print performance metrics and results */
    if (vars.proc_rank == 0){
        vars.t_stop = omp_get_wtime(); // Stop time for the master process
        printf("MPI Matrix Multiplication Performance with OpenMP \n");
        printf("Dimension: %d \n", DIM);
        printf("Processes: %d \n", vars.proc);
        printf("Threads: %d \n", vars.num_threads);
        printf("Run time: %f \n", vars.t_stop - vars.t_start);
        if (DIM <= 10){
            printf("First matrix: \n");
            outputMatrix(matrix1);
            printf("Second matrix: \n"); 
            outputMatrix(matrix2);
            printf("Result: \n");  
            outputMatrix(result);
        }
    }

    MPI_Finalize(); // Shutdown MPI environment
    return 0;
}

/* Head process */
void headProcess(int proc, int start, int end){
    createMatrix(matrix1); // Create matrix1
    createMatrix(matrix2); // Create matrix2

    MPI_Bcast(matrix2, DIM * DIM, MPI_INT, 0, MPI_COMM_WORLD); // Broadcast matrix2 to all processes
    MPI_Scatter(&matrix1[0][0], DIM * DIM / proc, MPI_INT, MPI_IN_PLACE, DIM * DIM / proc, MPI_INT, 0, MPI_COMM_WORLD); // Scatter matrix1 data

    #pragma omp parallel num_threads(vars.num_threads)
    {
        #pragma omp for
        for (int i = start; i < end; i++)
            for (int j = 0; j < DIM; j++)
            {
                result[i][j] = 0;
                for (int k = 0; k < DIM; k++)
                    result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
    }

    MPI_Gather(MPI_IN_PLACE, DIM*DIM/proc, MPI_INT, &result[0][0], DIM*DIM/proc, MPI_INT, 0, MPI_COMM_WORLD); // Gather result data
}

/* Node processes */
void nodeProcess(int proc, int start, int end){
    MPI_Bcast(matrix2, DIM * DIM, MPI_INT, 0, MPI_COMM_WORLD); // Broadcast matrix2 to all processes
    MPI_Scatter(&matrix1[0][0], DIM * DIM / proc, MPI_INT, &matrix1[start], DIM * DIM / proc, MPI_INT, 0, MPI_COMM_WORLD); // Scatter matrix1 data

    #pragma omp parallel num_threads(vars.num_threads)
    {
        #pragma omp for
        for (int i = start; i < end; i++)
            for (int j = 0; j < DIM; j++)
            {
                result[i][j] = 0;
                for (int k = 0; k < DIM; k++)
                    result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
    }

    MPI_Gather(&result[start], DIM*DIM/proc, MPI_INT, &result, DIM*DIM/proc, MPI_INT, 0, MPI_COMM_WORLD); // Gather result data
}

/* Function to create a matrix */
void createMatrix(int matrix[DIM][DIM])
{
    for (int i = 0; i < DIM; i++)
        for (int j = 0; j < DIM; j++)
            matrix[i][j] = rand() % MAX_RAND;
}

/* Function to output a matrix */
void outputMatrix(int matrix[DIM][DIM])
{
    for (int i = 0; i < DIM; i++) {
        for (int j = 0; j < DIM; j++)
            std::cout << matrix[i][j] << "\t";
        std::cout << "\n";
    }
    std::cout << "\n";
}
