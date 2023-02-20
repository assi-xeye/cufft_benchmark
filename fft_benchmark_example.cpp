#include <complex>
#include <iostream>
#include <vector>
#include <chrono>

#include <cuda_runtime.h>
#include <cufftXt.h>
#include "cufft_utils.h"

using namespace std;

int main(int argc, char *argv[]) {
    
    cufftHandle plan;
    cudaStream_t stream = NULL;

    int n = 4096;
    int batch_size = 704 * 54;  // 54 = 432 /8 , 1/8 of frame
    int fft_size = batch_size * n;

    using scalar_type = float;
    using input_type = scalar_type;
    using output_type = std::complex<scalar_type>;

    std::vector<input_type> input(fft_size, 0);
    std::vector<output_type> output(static_cast<int>((fft_size / 2 + 1)));

    for (int i = 0; i < fft_size; i++) {
        input[i] = static_cast<input_type>(i);
    }

    input_type *d_input = nullptr;
    cufftComplex *d_output = nullptr;

    size_t workSize;
    CUFFT_CALL(cufftCreate(&plan));
  
    //https://stackoverflow.com/questions/25603394/1d-batched-ffts-of-real-arrays%5B/url%5D
    int rank = 1;                           // --- 1D FFTs
    int nv[] = { n };                       // --- Size of the Fourier transform
    int istride = 1, ostride = 1;           // --- Distance between two successive input/output elements
    int idist = n, odist = (n / 2 + 1);     // --- Distance between batches
    int inembed[] = { 0 };                  // --- Input size with pitch (ignored for 1D transforms)
    int onembed[] = { 0 };                  // --- Output size with pitch (ignored for 1D transforms)
    // batch_size                           // --- Number of batched executions
    CUFFT_CALL(cufftPlanMany(&plan, rank, nv, 
                inembed, istride, idist,
                onembed, ostride, odist, CUFFT_R2C, batch_size));   

    CUDA_RT_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUFFT_CALL(cufftSetStream(plan, stream));

    // Create device arrays
    CUDA_RT_CALL(
        cudaMalloc(reinterpret_cast<void **>(&d_input), sizeof(input_type) * input.size()));
    CUDA_RT_CALL(
        cudaMalloc(reinterpret_cast<void **>(&d_output), sizeof(output_type) * output.size()));
    CUDA_RT_CALL(cudaMemcpyAsync(d_input, input.data(), sizeof(input_type) * input.size(),
                                 cudaMemcpyHostToDevice, stream));

    auto start = chrono::steady_clock::now();
    CUFFT_CALL(cufftExecR2C(plan, d_input, d_output));

    // No need to measure copy to host so commented out
    //CUDA_RT_CALL(cudaMemcpyAsync(output.data(), d_output, sizeof(output_type) * output.size(),
    //                             cudaMemcpyDeviceToHost, stream));

    CUDA_RT_CALL(cudaStreamSynchronize(stream));

    auto end = chrono::steady_clock::now();

   cout << "Elapsed time in microseconds: " 
        << chrono::duration_cast<chrono::microseconds>(end - start).count()
        << " µs" << endl;

     /* free resources */
    CUDA_RT_CALL(cudaFree(d_input));
    CUDA_RT_CALL(cudaFree(d_output));

    CUFFT_CALL(cufftDestroy(plan));

    CUDA_RT_CALL(cudaStreamDestroy(stream));

    CUDA_RT_CALL(cudaDeviceReset());

    return EXIT_SUCCESS;
}