#include "definitions.cuh"
__global__ void windowData(U16 *data, float *toFFT, float *window, int windowSize, int fftSize, int numPixels);
__global__ void maskData(float *mag, float *summedPixels, int *maskPoints, int fftSize, int numPixels, int numMasks, int ptsPerMask);
__global__ void sumConfocal(U16 *dataIn, float *dataOut, int cSamplesPerPixel, int numPixels);

__global__ void takeAbs(Complex *fftOut, float *dataOut, int fftSize, int numPixels);

__device__ int getTid();
__device__ int getTid2D2D();
__device__ int getTid2D1D();
__device__ int getTid3D2D();
__device__ int getTid3D1D();