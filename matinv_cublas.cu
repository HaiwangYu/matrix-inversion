#include <string>
#include <cuda_runtime.h>
#include <cublas_v2.h>
//#include <conio.h>

#define CUDA_CALL(res, str) { if (res != cudaSuccess) { printf("CUDA Error : %s : %s %d : ERR %s\n", str, __FILE__, __LINE__, cudaGetErrorName(res)); } }
#define CUBLAS_CALL(res, str) { if (res != CUBLAS_STATUS_SUCCESS) { printf("CUBLAS Error : %s : %s %d : ERR %d\n", str, __FILE__, __LINE__, int(res)); } }

static cudaEvent_t cu_TimerStart;
static cudaEvent_t cu_TimerStop;

void d_CUDATimerStart(void)
{
    CUDA_CALL(cudaEventCreate(&cu_TimerStart), "Failed to create start event!");
    CUDA_CALL(cudaEventCreate(&cu_TimerStop), "Failed to create stop event!");

    CUDA_CALL(cudaEventRecord(cu_TimerStart), "Failed to record start event!");
}

float d_CUDATimerStop(void)
{
    CUDA_CALL(cudaEventRecord(cu_TimerStop), "Failed to record stop event!");

    CUDA_CALL(cudaEventSynchronize(cu_TimerStop), "Failed to synch stop event!");

    float ms;

    CUDA_CALL(cudaEventElapsedTime(&ms, cu_TimerStart, cu_TimerStop), "Failed to elapse events!");

    CUDA_CALL(cudaEventDestroy(cu_TimerStart), "Failed to destroy start event!");
    CUDA_CALL(cudaEventDestroy(cu_TimerStop), "Failed to destroy stop event!");

    return ms;
}

// L point to the first element of first array
float* d_GetInv(float* L, int n, int b)
{
    cublasHandle_t cu_cublasHandle;
    CUBLAS_CALL(cublasCreate(&cu_cublasHandle), "Failed to initialize cuBLAS!");

    float** Lp = (float**)malloc(b * sizeof(float*));
    float** Cp = (float**)malloc(b * sizeof(float*));
    float** adL;
    float** adC;
    float* dL;
    float* dC;
    int* dLUPivots;
    int* dLUInfo;

    size_t szA = n * n * sizeof(float);

    CUDA_CALL(cudaMalloc(&adL, b * sizeof(float*)), "Failed to allocate adL!");
    CUDA_CALL(cudaMalloc(&adC, b * sizeof(float*)), "Failed to allocate adC!");
    CUDA_CALL(cudaMalloc(&dL, b * szA), "Failed to allocate dL!");
    CUDA_CALL(cudaMalloc(&dC, b * szA), "Failed to allocate dC!");
    CUDA_CALL(cudaMalloc(&dLUPivots, b * n * sizeof(int)), "Failed to allocate dLUPivots!");
    CUDA_CALL(cudaMalloc(&dLUInfo, b * sizeof(int)), "Failed to allocate dLUInfo!");

    CUDA_CALL(cudaMemcpy(dL, L, b * szA, cudaMemcpyHostToDevice), "Failed to copy to dL!");
    // create pointer array for matrices
    for (int i = 0; i < b; i++) {
        Lp[i] = dL + (i*n*n);
    }
    CUDA_CALL(cudaMemcpy(adL, Lp, b * sizeof(float*), cudaMemcpyHostToDevice), "Failed to copy to adL!");

    for (int i = 0; i < b; i++) {
        Cp[i] = dC + (i*n*n);
    }
    CUDA_CALL(cudaMemcpy(adC, Cp, b * sizeof(float*), cudaMemcpyHostToDevice), "Failed to copy to adC!");

    d_CUDATimerStart();

    CUBLAS_CALL(cublasSgetrfBatched(cu_cublasHandle, n, adL, n, dLUPivots, dLUInfo, b), "Failed to perform LU decomp operation!");
    CUDA_CALL(cudaDeviceSynchronize(), "Failed to synchronize after kernel call!");

    CUBLAS_CALL(cublasSgetriBatched(cu_cublasHandle, n, (const float **)adL, n, dLUPivots, adC, n, dLUInfo, b), "Failed to perform Inverse operation!");
    CUDA_CALL(cudaDeviceSynchronize(), "Failed to synchronize after kernel call!");

    float timed = d_CUDATimerStop();

    printf("cublas inverse in: %.5f ms.\n", timed);

    float* res = (float*)malloc(b * szA);

    CUDA_CALL(cudaMemcpy(res, dC, b * szA, cudaMemcpyDeviceToHost), "Failed to copy to res!");

    CUDA_CALL(cudaFree(adL), "Failed to free adL!");
    CUDA_CALL(cudaFree(adC), "Failed to free adC!");
    CUDA_CALL(cudaFree(dL), "Failed to free dL!");
    CUDA_CALL(cudaFree(dC), "Failed to free dC!");
    CUDA_CALL(cudaFree(dLUPivots), "Failed to free dLUPivots!");
    CUDA_CALL(cudaFree(dLUInfo), "Failed to free dLUInfo!");

    CUBLAS_CALL(cublasDestroy(cu_cublasHandle), "Failed to destroy cuBLAS!");

    free(Lp);
    free(Cp);
    return res;
}

void print(float *m, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%f ", m[i+j*n]);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[])
{
    int b = 1;
    int nloop = 100;
    int n = 364;
    if (argc > 1) {
        b = atoi(argv[1]);
    }

    float* L = (float*)malloc(b * n * n * sizeof(float));
    float* inv = 0;
    for(int iloop=0; iloop<nloop; ++iloop) {
        for(int i = 0; i < b * n * n; i++) {
            L[i] = ((float)rand()/(float)(RAND_MAX));
        }
        inv = d_GetInv(L, n, b);
        free(inv);
    }

    //for(int i = 0; i < b * n * n; i++) {
    //    L[i] = 0;
    //}
    //for (int i = 0; i < b; ++i) {
    //    L[0+0*n+i*n*n] = 1;
    //    L[1+1*n+i*n*n] = 10;
    //}
    //inv = d_GetInv(L, n, b);
    //for (int i = 0; i < b; ++i) {
    //    print(L+i*n*n, n);
    //    print(inv+i*n*n, n);
    //    printf("\n");
    //}

    free(L);

    printf("done.");
    //_getch();

    return 0;
}
