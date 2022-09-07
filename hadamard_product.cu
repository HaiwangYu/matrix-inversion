#include "TFile.h"
#include "TMatrixD.h"

#include <iostream>

#include <Eigen/Dense>
using Eigen::MatrixXd;
using Eigen::MatrixXf;

// CUDA runtime
#include <cuda_runtime.h>

#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 128 // number of threads in each block
#endif

#define CUDA_CALL(res, str) { if (res != cudaSuccess) { printf("CUDA Error : %s : %s %d : ERR %s\n", str, __FILE__, __LINE__, cudaGetErrorName(res)); } }

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


// https://stackoverflow.com/questions/20481390/3d-elementwise-matrix-multiplication-in-cuda
// https://web.engr.oregonstate.edu/~mjb/cs575/Handouts/cudaArrayMult.1pp.pdf
__global__ void ArrayMul(float *dA, float *dB, float *dC, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        dC[gid] = dA[gid] * dB[gid];
    }
}

int main(int argc, char *argv[])
{
    int nrep = 1;
    if (argc > 1) {
        nrep = atoi(argv[1]);
    }

    size_t nbatch = 100;
    if (argc > 2) {
        nbatch = atoi(argv[2]);
    }

    // READ in ROOT TMatrixD
    auto *fin = TFile::Open("mat.root", "read");
    TMatrixD *mat_d_r = (TMatrixD*)fin->Get("mat");
    std::cout << mat_d_r->GetNrows() << ", " << mat_d_r->GetNcols() << std::endl;
    auto nrows = mat_d_r->GetNrows();
    auto ncols = mat_d_r->GetNcols();
    TMatrixD had_d_r(nrows, ncols);

    // prepare Eigen Matrix
    MatrixXd mat_d_e(nrows, ncols);
    MatrixXd had_d_e(nrows, ncols);
    for (int irow = 0; irow < nrows; ++irow) {
        for (int icol = 0; icol < ncols; ++icol) {
            mat_d_e(irow, icol) = (*mat_d_r)(irow, icol);
        }
    }
    MatrixXf mat_f_e(nrows, ncols);
    MatrixXf had_f_e(nrows, ncols);
    for (int irow = 0; irow < nrows; ++irow) {
        for (int icol = 0; icol < ncols; ++icol) {
            mat_f_e(irow, icol) = (float)(*mat_d_r)(irow, icol);
        }
    }

    {
        auto time_start = std::chrono::high_resolution_clock::now();
        for (int irep = 0; irep < nrep; ++irep) {
            for (int irow = 0; irow < nrows; ++irow) {
                for (int icol = 0; icol < ncols; ++icol) {
                    had_d_r(irow, icol) = (*mat_d_r)(irow, icol) * (*mat_d_r)(irow, icol);
                }
            }
        }
        auto time_stop = std::chrono::high_resolution_clock::now();
        auto time_duration = std::chrono::duration_cast<std::chrono::milliseconds>(time_stop - time_start);
        std::cout<<"TMatrixD: nrep: "<< nrep << " time: " <<time_duration.count() << " ms" <<std::endl;
    }

    {
        auto time_start = std::chrono::high_resolution_clock::now();
        for (int irep = 0; irep < nrep; ++irep) {
            had_d_e = mat_d_e.cwiseProduct(mat_d_e);
        }

        auto time_stop = std::chrono::high_resolution_clock::now();
        auto time_duration = std::chrono::duration_cast<std::chrono::milliseconds>(time_stop - time_start);
        std::cout<<"Eigen::MatrixXd: nrep: "<< nrep << " time: " <<time_duration.count() << " ms" <<std::endl;
    }

    {
        auto time_start = std::chrono::high_resolution_clock::now();
        for (int irep = 0; irep < nrep; ++irep) {
            had_f_e = mat_f_e.cwiseProduct(mat_f_e);
        }

        auto time_stop = std::chrono::high_resolution_clock::now();
        auto time_duration = std::chrono::duration_cast<std::chrono::milliseconds>(time_stop - time_start);
        std::cout<<"Eigen::MatrixXf: nrep: "<< nrep << " time: " <<time_duration.count() << " ms" <<std::endl;
    }

    // prepare cuda arrays
    size_t mat_zize = nrows * ncols;
    float *mat_f_c = (float*)malloc(nbatch * mat_zize * sizeof(float));
    float *had_f_c = (float*)malloc(nbatch * mat_zize * sizeof(float));
    for (size_t ibatch=0; ibatch<nbatch; ++ibatch) {
        memcpy(mat_f_c+ibatch*mat_zize, mat_f_e.data(), mat_zize * sizeof(float));
    }
    float *mat_f_d;
    float *had_f_d;
    CUDA_CALL(cudaMalloc(&mat_f_d, nbatch * mat_zize * sizeof(float)), "Failed to allocate mat_f_d!");
    CUDA_CALL(cudaMalloc(&had_f_d, nbatch * mat_zize * sizeof(float)), "Failed to allocate had_f_d!");
    dim3 grid( nbatch * mat_zize / THREADS_PER_BLOCK, 1, 1 );
    dim3 threads( THREADS_PER_BLOCK, 1, 1 );
    d_CUDATimerStart();
    for (int irep = 0; irep < nrep; ++irep)
    {
        CUDA_CALL(cudaMemcpy(mat_f_d, mat_f_c, nbatch * mat_zize * sizeof(float), cudaMemcpyHostToDevice), "Failed to copy to mat_f_d!");
        ArrayMul<<< grid, threads >>>( mat_f_d, mat_f_d, had_f_d,  nbatch * mat_zize);
        CUDA_CALL(cudaMemcpy(had_f_c, had_f_d, nbatch * mat_zize * sizeof(float), cudaMemcpyDeviceToHost), "Failed to copy to had_f_c!");
        CUDA_CALL(cudaDeviceSynchronize(), "Failed to synchronize after kernel call!");
    }
    float timed = d_CUDATimerStop();
    printf("CUDA: %.5f ms.\n", timed);

    fin->Close();
    return 0;
}
