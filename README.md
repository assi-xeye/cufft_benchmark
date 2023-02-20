# Scope
Minimal benchmark for cuFFT 
Based on [Nvidia cuFFT 1D R2C exmaple](https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuFFT/1d_r2c)
The source example does not support multiple batches beyond 8. Changed the code to use ```cufftPlanMany``` based on rececommendations in answer [here](https://stackoverflow.com/questions/25603394/1d-batched-ffts-of-real-arrays%5B/url%5D)

# Benchmark parameters
 We perform batch_size ffts of n each, using 1D, R2C.  Uses single precision (not double, not half).
Currently hard coded to sizes:
 ```
 int n = 4096;               // This is the fft_size. Used 4096 as close to real size (~4500?) , must be power of 2
 int batch_size = 704 * 54;  // Full frame size is 704 * 432, we use 1/8 as plan is to processes parts of frame in streams
  ```
 
# Results
We do not measure time to copy from host to device or back. Tested with repetition to see if warmup has affect, no affect seen.
For Nvidia Quadro P6000 , this takes 6.7 ms. 

# Next step
- Will be tested on latest GPU for reference
- Nvidia recently released mathdx which includes cuFFTdx, which is a C++ interface to creake FFT kernels directly. This shows [improvement](https://docs.nvidia.com/cuda/cufftdx/examples.html) mostly for small FFT size. Can be considered in the future
- TODO: Test on latest GPU
- TODO: Discuss using Powers of 2 to avoid padding. 

