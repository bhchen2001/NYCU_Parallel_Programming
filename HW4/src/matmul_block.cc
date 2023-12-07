#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

// Read size of matrix_a and matrix_b (n, m, l) and whole data of matrixes from stdin
//
// n_ptr:     pointer to n
// m_ptr:     pointer to m
// l_ptr:     pointer to l
// a_mat_ptr: pointer to matrix a (a should be a continuous memory space for placing n * m elements of int)
// b_mat_ptr: pointer to matrix b (b should be a continuous memory space for placing m * l elements of int)
void construct_matrices(int *n_ptr, int *m_ptr, int *l_ptr, int **a_mat_ptr, int **b_mat_ptr){
    int size, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    /* read inputs from master */
    if(rank == 0){
        scanf("%d %d %d", n_ptr, m_ptr, l_ptr);
        int n = *n_ptr, m = *m_ptr, l = *l_ptr;
        *a_mat_ptr = (int *)malloc(sizeof(int) * n * m);
        *b_mat_ptr = (int *)malloc(sizeof(int) * m * l);
        for (int i = 0; i < n; i++){
            for (int j = 0; j < m; j++){
                scanf("%d", (*a_mat_ptr) + i * m + j);
            }
        }
        for (int i = 0; i < m; i++){
            for (int j = 0; j < l; j++){
                scanf("%d", (*b_mat_ptr) + i * l + j);
            }
        }

    //     print out for debug
    //     printf("rank %d: n = %d, m = %d, l = %d\n", rank, *n_ptr, *m_ptr, *l_ptr);
    //     for (int i = 0; i < n; i++){
    //         for (int j = 0; j < m; j++){
    //             printf("%d ", (*a_mat_ptr)[i * m + j]);
    //         }
    //         printf("\n");
    //     }
    //     for (int i = 0; i < m; i++){
    //         for (int j = 0; j < l; j++){
    //             printf("%d ", (*b_mat_ptr)[i * l + j]);
    //         }
    //         printf("\n");
    //     }
    }
}

// Just matrix multiplication (your should output the result in this function)
// 
// n:     row number of matrix a
// m:     col number of matrix a / row number of matrix b
// l:     col number of matrix b
// a_mat: a continuous memory placing n * m elements of int
// b_mat: a continuous memory placing m * l elements of int
void matrix_multiply(const int n, const int m, const int l, const int *a_mat, const int *b_mat){
    int size, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N, M, L, work_load;

    if(rank == 0){
        work_load = n / (size - 1);
        int remain = n % (size - 1);
        /* split a_matrix into rows with same size and send to workers */
        for(int i = 1; i < size; i++){
            /* send the setting arguments to the workers */
            MPI_Send(&n, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&m, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&l, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&work_load, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(a_mat + (work_load * m) * (i - 1), work_load * m, MPI_INT, i, 0, MPI_COMM_WORLD);
            /* send the whole b_matrix */
            MPI_Send(b_mat, m * l, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
        /* master will calculate the remaining rows */
        int *c = (int *)malloc(sizeof(int) * n * l);
        // printf("remain = %d\n", remain);  
        for(int i =0; i < remain; i++){
            for(int j = 0; j < l; j++){
                c[(work_load * (size - 1) + i) * l + j] = 0;
                int mat_sum = 0;
                for(int k = 0; k < m; k++){
                    mat_sum += a_mat[(work_load * (size - 1) + i) * m + k] * b_mat[k * l + j];
                }
                c[(work_load * (size - 1) + i) * l + j] = mat_sum;
            }
        }

        // /* receive the results from the workers */
        for(int i = 1; i < size; i++){
            MPI_Recv(c + (i - 1) * work_load * l, work_load * l, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        /* write the result into file */
        // FILE *fp = fopen("output.txt", "w");
        // for(int i = 0; i < n; i++){
        //     for(int j = 0; j < l; j++){
        //         fprintf(fp, "%d ", c[i * l + j]);
        //     }
        //     fprintf(fp, "\n");
        // }

        /* print out the result */
        for(int i = 0; i < n; i++){
            for(int j = 0; j < l; j++){
                printf("%d ", c[i * l + j]);
            }
            printf("\n");
        }
        free(c);
    }
    else if(rank > 0){
        MPI_Recv(&N, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&M, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&L, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&work_load, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        int *a = (int *)malloc(sizeof(int) * work_load * M);
        int *b = (int *)malloc(sizeof(int) * M * L);
        MPI_Recv(a, work_load * M, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(b, M * L, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        /* calculate the result */
        int *c = (int *)malloc(sizeof(int) * work_load * L);
        for(int i = 0; i < work_load; i++){
            for(int j = 0; j < L; j++){
                c[i * L + j] = 0;
                for(int k = 0; k < M; k++){
                    c[i * L + j] += a[i * M + k] * b[k * L + j];
                }
            }
        }
        MPI_Send(c, work_load * L, MPI_INT, 0, 0, MPI_COMM_WORLD);
        free(a);
        free(b);
        free(c);
    }
}

// Remember to release your allocated memory
void destruct_matrices(int *a_mat, int *b_mat){
    int size, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if(rank == 0){
        free(a_mat);
        free(b_mat);
    }
}