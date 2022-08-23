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

float* d_GetInv(float* L, int n)
{
    cublasHandle_t cu_cublasHandle;
    CUBLAS_CALL(cublasCreate(&cu_cublasHandle), "Failed to initialize cuBLAS!");

    float** adL;
    float** adC;
    float* dL;
    float* dC;
    int* dLUPivots;
    int* dLUInfo;

    size_t szA = n * n * sizeof(float);

    CUDA_CALL(cudaMalloc(&adL, sizeof(float*)), "Failed to allocate adL!");
    CUDA_CALL(cudaMalloc(&adC, sizeof(float*)), "Failed to allocate adC!");
    CUDA_CALL(cudaMalloc(&dL, szA), "Failed to allocate dL!");
    CUDA_CALL(cudaMalloc(&dC, szA), "Failed to allocate dC!");
    CUDA_CALL(cudaMalloc(&dLUPivots, n * sizeof(int)), "Failed to allocate dLUPivots!");
    CUDA_CALL(cudaMalloc(&dLUInfo, sizeof(int)), "Failed to allocate dLUInfo!");

    CUDA_CALL(cudaMemcpy(dL, L, szA, cudaMemcpyHostToDevice), "Failed to copy to dL!");
    CUDA_CALL(cudaMemcpy(adL, &dL, sizeof(float*), cudaMemcpyHostToDevice), "Failed to copy to adL!");
    CUDA_CALL(cudaMemcpy(adC, &dC, sizeof(float*), cudaMemcpyHostToDevice), "Failed to copy to adC!");

    d_CUDATimerStart();

    CUBLAS_CALL(cublasSgetrfBatched(cu_cublasHandle, n, adL, n, dLUPivots, dLUInfo, 1), "Failed to perform LU decomp operation!");
    CUDA_CALL(cudaDeviceSynchronize(), "Failed to synchronize after kernel call!");

    CUBLAS_CALL(cublasSgetriBatched(cu_cublasHandle, n, (const float **)adL, n, dLUPivots, adC, n, dLUInfo, 1), "Failed to perform Inverse operation!");
    CUDA_CALL(cudaDeviceSynchronize(), "Failed to synchronize after kernel call!");

    float timed = d_CUDATimerStop();

    printf("cublas inverse in: %.5f ms.\n", timed);

    float* res = (float*)malloc(szA);

    CUDA_CALL(cudaMemcpy(res, dC, szA, cudaMemcpyDeviceToHost), "Failed to copy to res!");

    CUDA_CALL(cudaFree(adL), "Failed to free adL!");
    CUDA_CALL(cudaFree(adC), "Failed to free adC!");
    CUDA_CALL(cudaFree(dL), "Failed to free dL!");
    CUDA_CALL(cudaFree(dC), "Failed to free dC!");
    CUDA_CALL(cudaFree(dLUPivots), "Failed to free dLUPivots!");
    CUDA_CALL(cudaFree(dLUInfo), "Failed to free dLUInfo!");

    CUBLAS_CALL(cublasDestroy(cu_cublasHandle), "Failed to destroy cuBLAS!");

    return res;
}

int main()
{
    int n = 364;
    float* L = (float*)malloc(n * n * sizeof(float));
    for(int iloop=0; iloop<10; ++iloop) {
        for(int i = 0; i < n * n; i++)
            L[i] = ((float)rand()/(float)(RAND_MAX));
        float* inv = d_GetInv(L, n);
    }

    printf("done.");
    //_getch();

    return 0;
}
