#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <omp.h>
#include <CL/cl.h>

#define DIM 1000 // Define matrix size
#define MAX_RAND 5 // Maximum value for random generator

/* Matrix containers */
int matrix1[DIM][DIM];
int matrix2[DIM][DIM];
int result[DIM][DIM];
cl_mem bufferA, bufferB, bufferC;

cl_device_id device_id;
cl_context context;
cl_program program;
cl_kernel kernel;
cl_command_queue queue;
cl_event event = NULL;

int err;

const int max_size = DIM;
const int thread_size = 4;
const size_t local_size[2] = { (size_t)thread_size, (size_t)thread_size };
const size_t global_size[2] = { (size_t)max_size, (size_t)max_size }; 

cl_device_id create_device();
void setup_openCL_device_context_queue_kernel(char* filename, char* kernelname);
cl_program build_program(cl_context ctx, cl_device_id dev, const char* filename);
void setup_kernel_memory();
void copy_kernel_arguments();
void free_memory();

/* Function declarations */
void headProcess(int proc, int start, int end);
void nodeProcess(int proc, int start, int end);
void createMatrix(int matrix[DIM][DIM]);
void outputMatrix(int matrix[DIM][DIM]);

/* Struct to hold various values */
struct Variables
{
    int proc_rank, proc, start, end, num_threads;
    double t_start, t_stop;
};

Variables vars = {0, 0, 0, 0, 2, 0.0, 0.0}; // Initialize variables with default values

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv); // Initialize MPI environment
    MPI_Comm_rank(MPI_COMM_WORLD, &vars.proc_rank); // Get current process ID
    MPI_Comm_size(MPI_COMM_WORLD, &vars.proc); // Get number of processes 

    vars.num_threads = 1;
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

    if (vars.proc_rank == 0)
    {

    }

    if (vars.proc_rank == 0){
        vars.t_stop = omp_get_wtime(); // Stop time for the master process
        printf("MPI Matrix Multiplication Performance with OpenCL\n");
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

        free_memory(); // Free allocated memory

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

    for (int i = start; i < end; i++)
        for (int j = 0; j < DIM; j++)
        {
            result[i][j] = 0;
            for (int k = 0; k < DIM; k++)
                result[i][j] += matrix1[i][k] * matrix2[k][j];
        }

    MPI_Gather(MPI_IN_PLACE, DIM*DIM/proc, MPI_INT, &result[0][0], DIM*DIM/proc, MPI_INT, 0, MPI_COMM_WORLD); // Gather result data

    // Setup OpenCL device, context, queue, program, and kernel
    setup_openCL_device_context_queue_kernel( (char*) "./matrix_ops.cl" , (char*) "multiply_matrices");

    // Setup kernel memory
    setup_kernel_memory();

    // Copy kernel arguments
    copy_kernel_arguments();

    // Submit the kernel for execution 
    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size, local_size, 0, NULL, &event);
    clWaitForEvents(1, &event);

    // Copy data from device back to host result matrix
    clEnqueueReadBuffer(queue, bufferC, CL_TRUE, 0, DIM * DIM *sizeof(int), &result[0][0], 0, NULL, NULL);
}

/* Node processes */
void nodeProcess(int proc, int start, int end){
    MPI_Bcast(matrix2, DIM * DIM, MPI_INT, 0, MPI_COMM_WORLD); // Broadcast matrix2 to all processes
    MPI_Scatter(&matrix1[0][0], DIM * DIM / proc, MPI_INT, &matrix1[start], DIM * DIM / proc, MPI_INT, 0, MPI_COMM_WORLD); // Scatter matrix1 data

    for (int i = start; i < end; i++)
        for (int j = 0; j < DIM; j++)
        {
            result[i][j] = 0;
            for (int k = 0; k < DIM; k++)
                result[i][j] += matrix1[i][k] * matrix2[k][j];
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

/* Function to free allocated memory */
void free_memory() {
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferC);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseContext(context);
}

/* Function to copy kernel arguments */
void copy_kernel_arguments() {
    clSetKernelArg(kernel, 0, sizeof(int), (void*)&max_size);
    clSetKernelArg(kernel, 1, sizeof(int), (void*)&max_size);
    clSetKernelArg(kernel, 2, sizeof(int), (void*)&max_size);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&bufferA);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&bufferB);
    clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&bufferC);
    if(err < 0) {
        perror("Couldn't create a kernel argument");
        printf("error = %d", err);
        exit(1);
    }
}

/* Function to setup kernel memory */
void setup_kernel_memory() {
    bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY,  DIM*DIM*sizeof(int), NULL, NULL);
    bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY,  DIM*DIM*sizeof(int), NULL, NULL);
    bufferC = clCreateBuffer(context, CL_MEM_READ_WRITE, DIM*DIM*sizeof(int), NULL, NULL);

    // Copy matrices to the GPU
    clEnqueueWriteBuffer(queue, bufferA, CL_TRUE, 0, DIM*DIM*sizeof(int), &matrix1[0][0], 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, bufferB, CL_TRUE, 0, DIM*DIM*sizeof(int), &matrix2[0][0], 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, bufferC, CL_TRUE, 0, DIM*DIM*sizeof(int), &result[0][0], 0, NULL, NULL);
}

/* Function to setup OpenCL device, context, queue, program, and kernel */
void setup_openCL_device_context_queue_kernel(char* filename, char* kernelname) {
    device_id = create_device();
    cl_int err;
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    if(err < 0) {
        perror("Couldn't create a context");
        exit(1);   
    }

    program = build_program(context, device_id, filename );
    queue = clCreateCommandQueueWithProperties(context, device_id, 0, &err);
    if(err < 0) {
        perror("Couldn't create a command queue");
        exit(1);   
    };

    kernel = clCreateKernel(program, kernelname, &err);
    if(err < 0) {
        perror("Couldn't create a kernel");
        printf("error =%d", err);
        exit(1);
    };
}

/* Function to create OpenCL device */
cl_device_id create_device() {
    cl_platform_id platform;
    cl_device_id dev;
    int err;

    // Identify a platform
    err = clGetPlatformIDs(1, &platform, NULL);
    if(err < 0) {
        perror("Couldn't identify a platform");
        exit(1);
    } 

    // Access a device (GPU or CPU)
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
    if(err == CL_DEVICE_NOT_FOUND) {
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
    }
    if(err < 0) {
        perror("Couldn't access any devices");
        exit(1);   
    }

    return dev;
}

/* Function to build OpenCL program */
cl_program build_program(cl_context ctx, cl_device_id dev, const char* filename) {
    cl_program program;
    FILE *program_handle;
    char *program_buffer, *program_log;
    size_t program_size, log_size;

    // Read program file and place content into buffer
    program_handle = fopen(filename, "r");
    if(program_handle == NULL) {
        perror("Couldn't find the program file");
        exit(1);
    }
    fseek(program_handle, 0, SEEK_END);
    program_size = ftell(program_handle);
    rewind(program_handle);
    program_buffer = (char*)malloc(program_size + 1);
    program_buffer[program_size] = '\0';
    fread(program_buffer, sizeof(char), program_size, program_handle);
    fclose(program_handle);

    // Create program from file
    program = clCreateProgramWithSource(ctx, 1, (const char**)&program_buffer, &program_size, &err);
    if(err < 0) {
        perror("Couldn't create the program");
        exit(1);
    }
    free(program_buffer);

    // Build program
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if(err < 0) {
        // Find size of log and print to std output
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        program_log = (char*) malloc(log_size + 1);
        program_log[log_size] = '\0';
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, log_size + 1, program_log, NULL);
        printf("%s\n", program_log);
        free(program_log);
        exit(1);
    }

    return program;
}

