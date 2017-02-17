#include "kernel.cuh"

/**
Thread Id Functions
Return the correct thread Id based on the dimensionality of the function call
sadly this is not automatic, but instead must be selected based on how you
plan to call the device
*/
__device__ int getTid()
{
	return blockIdx.x * blockDim.x + threadIdx.x;
}
__device__ int getTid2D2D()
{
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.y) + threadIdx.x;
	return threadId;
}

__device__ int getTid2D1D()
{
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int threadId = blockId * blockDim.x + threadIdx.x;
	return threadId;
}

__device__ int getTid3D2D()
{
	int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
	int threadId = blockId * (blockDim.x *blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	return threadId;
}

__device__ int getTid3D1D()
{
	int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
	int threadId = blockId * blockDim.x + threadIdx.x;
	return threadId;
}

/*Global Functions*/
__global__ void windowData(U16 *data, float *toFFT, float *window, int windowSize, int fftSize, int numPixels) {
	int tid = getTid2D2D();
	if (tid < windowSize * numPixels) {
		int winIdx = tid % windowSize;
		int currentFFT = tid / windowSize;
		int outIdx = currentFFT * fftSize + winIdx;
		float temp = float(data[tid]);
		temp = temp - 32768;
		temp = temp * window[winIdx];
		toFFT[outIdx] = temp;
	}
}

__global__ void maskData(float *mag, float *summedPixels, int *maskPoints, int fftSize, int numPixels, int numMasks, int ptsPerMask) {
	int tid = getTid2D2D();
	if (tid < numPixels * numMasks) {
		int currentPixel = tid / numMasks;
		int currentMask = tid % numMasks;
		int outId = currentMask * numPixels + currentPixel;
		float sum = 0;
		float temp = 0;
		int maskId = 0;
		int readId = 0;
		int halfFFT = fftSize / 2;
		for (int i = 0; i < ptsPerMask; ++i) {
			maskId = currentMask * ptsPerMask + i;
			readId = currentPixel * halfFFT + maskPoints[maskId];
			temp = mag[readId];
			sum = sum + temp;
		}
		summedPixels[outId] = sum;
	}
}

__global__ void takeAbs(Complex *fftOut, float *dataOut, int fftSize, int numPixels) {
	int tid = getTid2D2D();
	int halfFFT = fftSize / 2;
	if (tid < halfFFT * numPixels) {
		int currentPixel = tid / halfFFT;
		int currentSpot = tid % halfFFT;
		int readIdx = currentPixel * fftSize + currentSpot;
		Complex temp = fftOut[readIdx];
		dataOut[tid] = cuCabsf(temp);
	}
}

__global__ void sumConfocal(U16 *dataIn, float *dataOut, int cSamplesPerPixel, int numPixels) {
	int tid = getTid2D2D();
	if (tid < numPixels) {
		float sum = 0;
		U16 temp = 0;
		for (int i = 0; i < cSamplesPerPixel; ++i) {
			int inIdx = tid * cSamplesPerPixel + i;
			temp = dataIn[inIdx];
			sum = sum + temp;
		}
		dataOut[tid] = sum;
	}
}