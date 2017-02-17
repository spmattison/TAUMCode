#include "TAUMCode.cuh"
#include "kernel.cuh"

#define CUDA_FREE_NN(x) if (x != nullptr) { cudaFree(x); x = nullptr; }

static U16 *d_data = nullptr;
static float *d_window = nullptr;
static float *d_fftIn = nullptr;
static Complex *d_fftOut = nullptr;
static float *d_magOut = nullptr;
static float *d_TAUMSignals = nullptr;
static int *d_binaryIndices = nullptr;

static U16 *d_confocalRaw = nullptr;
static float *d_confocalOut = nullptr;

static int numPixels;
static int samplesPerPixel;
static int fftSize;
static int pointsPerMask;
static int numBinaryMasks;
static int confocalSamplesPerPixel;

static bool isSetup = false;
static bool withConfocal = false;
dim3 blocks;
dim3 threadsPerBlock(32, 32);
dim3 sumBlocks(4,4);
dim3 threadsPerSumBlock(16,16);

cufftHandle handle = NULL;

int calculateFFTSize() {
	float size = powf(2, ceilf(logf(samplesPerPixel) / logf(2) + 1));
	//cout << "FFT Size "<< size << endl;
	return (int)size;
}

int calculateBlocks() {
	int size = 0;
	if (samplesPerPixel > (numBinaryMasks*pointsPerMask))
		size = samplesPerPixel * numPixels;
	else
		size = numPixels * numBinaryMasks*pointsPerMask;
	float blocksNeeded = size / (threadsPerBlock.x * threadsPerBlock.y);
	blocksNeeded = powf(blocksNeeded, 0.5);
	blocksNeeded = powf(2, ceilf(logf(blocksNeeded) / logf(2)));
	int blockSize = int(blocksNeeded);

	blocks.x = blockSize;
	blocks.y = blockSize;
	cout << "Block Size " << blockSize << endl;
	if (blockSize > 2048)
		return _KERNEL_TOO_LARGE_F;

	int sumSize = numPixels * numBinaryMasks;
	if (sumSize > (sumBlocks.x * sumBlocks.y * threadsPerSumBlock.x * threadsPerSumBlock.y)) {
		float resizeTo = sumSize / (threadsPerSumBlock.x * threadsPerSumBlock.y);
		resizeTo = powf(resizeTo, 0.5);
		resizeTo = powf(2, ceilf((logf(resizeTo) / logf(2))));
		int sumBlockSize = int(resizeTo);
		sumBlocks.x = sumBlockSize;
		sumBlocks.y = sumBlockSize;
		cout << "Test: " << sumBlockSize << endl;
		if (sumBlockSize > 2048)
			return _KERNEL_TOO_LARGE_F;
	}
	return _SUCCESS;
}

EXTERN int clearGPU() {
	CUDA_FREE_NN(d_binaryIndices);
	CUDA_FREE_NN(d_data);
	CUDA_FREE_NN(d_fftIn);
	CUDA_FREE_NN(d_fftOut);
	CUDA_FREE_NN(d_magOut);
	CUDA_FREE_NN(d_TAUMSignals);
	CUDA_FREE_NN(d_confocalOut);
	CUDA_FREE_NN(d_confocalRaw);
	CUDA_FREE_NN(d_window);

	if (handle != NULL)
		cufftDestroy(handle);
	return _SUCCESS;
}

EXTERN int updateNumPixels(int nPixels) {
	if (!isSetup)
		return _NOT_SETUP_F;
	if (nPixels > numPixels) {
		numPixels = nPixels;
		CUDA_FREE_NN(d_data);
		CUDA_FREE_NN(d_fftIn);
		CUDA_FREE_NN(d_fftOut);
		CUDA_FREE_NN(d_magOut);
		CUDA_FREE_NN(d_TAUMSignals);
		CUDA_FREE_NN(d_confocalOut);
		CUDA_FREE_NN(d_confocalRaw);
		isSetup = false;
		int error = cudaSuccess;
		error = cudaMalloc(&d_data, sizeof(U16)*samplesPerPixel*numPixels);
		if (error != cudaSuccess)
			return _MEM_ALLOCATION_F;
		error = cudaMalloc(&d_fftIn, sizeof(float)*fftSize*numPixels);
		if (error != cudaSuccess)
			return _MEM_ALLOCATION_F;
		error = cudaMalloc(&d_fftOut, sizeof(Complex)*fftSize*numPixels);
		if (error != cudaSuccess)
			return _MEM_ALLOCATION_F;
		error = cudaMalloc(&d_magOut, sizeof(float)*pointsPerMask*numBinaryMasks*numPixels);
		if (error != cudaSuccess)
			return _MEM_ALLOCATION_F;
		error = cudaMalloc(&d_TAUMSignals, sizeof(float)*numBinaryMasks*numPixels);
		if (error != cudaSuccess)
			return _MEM_ALLOCATION_F;
		if (withConfocal) {
			error = cudaMalloc(&d_confocalOut, sizeof(float)*numPixels);
			if (error != cudaSuccess)
				return _MEM_ALLOCATION_F;
			error = cudaMalloc(&d_confocalRaw, sizeof(U16)*numPixels*confocalSamplesPerPixel);
			if (error != cudaSuccess)
				return _MEM_ALLOCATION_F;
		}
	}
	else {
		numPixels = nPixels;
	}
	isSetup = true;
	
	return calculateBlocks();
}

EXTERN int updateConfocalSampling(int samples) {
	if (!isSetup)
		return _NOT_SETUP_F;
	if (samples == 0) {
		withConfocal = false;
		return _SUCCESS;
	}
	if (samples < confocalSamplesPerPixel) {
		confocalSamplesPerPixel = samples;
		return _SUCCESS;
	}
	confocalSamplesPerPixel = samples;
	withConfocal = false;
	CUDA_FREE_NN(d_confocalRaw);
	int error = cudaMalloc(&d_confocalRaw, sizeof(U16)*confocalSamplesPerPixel*numPixels);
	if (error != cudaSuccess) {
		return _MEM_ALLOCATION_F;
	}
	withConfocal = true;
	return _SUCCESS;
}

EXTERN int updateMasks(int ptsPerMask, int numMasks, int *maskIndices) {
	if (!isSetup)
		return _NOT_SETUP_F;
	isSetup = false;
	int error = cudaSuccess;
	//Remove all old data
	CUDA_FREE_NN(d_magOut);
	CUDA_FREE_NN(d_TAUMSignals);
	CUDA_FREE_NN(d_binaryIndices);

	//Update globals
	pointsPerMask = ptsPerMask;
	numBinaryMasks = numMasks;

	//Reallocate memory for new masks
	error = cudaMalloc(&d_magOut, sizeof(float)*pointsPerMask*numBinaryMasks*numPixels);
	if (error != cudaSuccess)
		return _MEM_ALLOCATION_F;
	error = cudaMalloc(&d_TAUMSignals, sizeof(float)*numBinaryMasks*numPixels);
	if (error != cudaSuccess)
		return _MEM_ALLOCATION_F;
	error = cudaMalloc(&d_binaryIndices, sizeof(int)*numBinaryMasks*pointsPerMask);
	if (error != cudaSuccess)
		return _MEM_ALLOCATION_F;

	//Copy new masks to memory
	error = cudaMemcpy(d_binaryIndices, maskIndices, sizeof(int)*pointsPerMask*numBinaryMasks, cudaMemcpyHostToDevice);
	if (error != cudaSuccess)
		return _MEMCPY_TO_DEVICE_F;

	isSetup = true;
	return _SUCCESS;
}



EXTERN int setup(int numPxl, int sampsPerPxl, int numMasks, int ptsPerMask, int* binaryMasks, float *window, int confocalSamps){

	numPixels = numPxl;
	samplesPerPixel = sampsPerPxl;
	fftSize = calculateFFTSize();
	pointsPerMask = ptsPerMask;
	numBinaryMasks = numMasks;
	confocalSamplesPerPixel = confocalSamps;
	withConfocal = confocalSamplesPerPixel > 0;
	clearGPU();
	/*
	Allocate Memory
	*/
	int error = cudaSuccess;
	error = calculateBlocks();
	if (error != _SUCCESS)
		return _KERNEL_TOO_LARGE_F;
	error = cudaMalloc(&d_data, sizeof(U16)*samplesPerPixel*numPixels);
	if (error != cudaSuccess)
		return _MEM_ALLOCATION_F;
	error = cudaMalloc(&d_window, sizeof(float)*samplesPerPixel);
	if (error != cudaSuccess)
		return _MEM_ALLOCATION_F;
	error = cudaMalloc(&d_fftIn, sizeof(float)*fftSize*numPixels);
	if (error != cudaSuccess)
		return _MEM_ALLOCATION_F;
	error = cudaMalloc(&d_fftOut, sizeof(Complex)*fftSize*numPixels);
	if (error != cudaSuccess)
		return _MEM_ALLOCATION_F;
	error = cudaMalloc(&d_magOut, sizeof(float)*fftSize/2*numPixels);
	if (error != cudaSuccess)
		return _MEM_ALLOCATION_F;
	error = cudaMalloc(&d_TAUMSignals, sizeof(float)*numMasks*numPixels);
	if (error != cudaSuccess)
		return _MEM_ALLOCATION_F;
	error = cudaMalloc(&d_binaryIndices, sizeof(int)*numMasks*ptsPerMask);
	if (error != cudaSuccess)
		return _MEM_ALLOCATION_F;
	if (withConfocal) {
		error = cudaMalloc(&d_confocalOut, sizeof(float)*numPixels);
		if (error != cudaSuccess)
			return _MEM_ALLOCATION_F;
		error = cudaMalloc(&d_confocalRaw, sizeof(U16)*numPixels*confocalSamplesPerPixel);
		if (error != cudaSuccess)
			return _MEM_ALLOCATION_F;
	}

	/*
	Copy Preloaded data
	*/
	error = cudaMemcpy(d_binaryIndices, binaryMasks, sizeof(int)*numBinaryMasks*pointsPerMask, cudaMemcpyHostToDevice);
	if (error != cudaSuccess)
		return _MEMCPY_TO_DEVICE_F;

	error = cudaMemcpy(d_window, window, sizeof(float)*samplesPerPixel, cudaMemcpyHostToDevice);
	if (error != cudaSuccess)
		return _MEMCPY_TO_DEVICE_F;

	/*
	Create FFT Plan
	*/
	error = cufftPlan1d(&handle, fftSize, CUFFT_R2C, numPixels);
	if (error != CUFFT_SUCCESS)
		return _FFT_PLAN_F;

	isSetup = true;
	return _SUCCESS;
}

EXTERN int copyDataToDevice(U16 *data) {
	if (!isSetup)
		return _NOT_SETUP_F;

	int error = cudaMemcpy(d_data, data, sizeof(U16)*samplesPerPixel*numPixels, cudaMemcpyHostToDevice);
	if (error != cudaSuccess)
		return _MEMCPY_TO_DEVICE_F;
	return _SUCCESS;
}

EXTERN int copyConfocalToDevice(U16 *confocal) {
	if (!isSetup)
		return _NOT_SETUP_F;
	if (!withConfocal) {
		return _NO_CONFOCAL_ALLOCATED_F;
	}
	int error = cudaMemcpy(d_confocalRaw, confocal,sizeof(U16)*confocalSamplesPerPixel*numPixels, cudaMemcpyHostToDevice);
	if (error != cudaSuccess)
		return _MEMCPY_TO_DEVICE_F;
	return _SUCCESS;
}

EXTERN int getResults(float *TAUMOutput) {
	if (!isSetup)
		return _NOT_SETUP_F;
	int error = cudaMemcpy(TAUMOutput, d_TAUMSignals, sizeof(float)*numPixels*numBinaryMasks, cudaMemcpyDeviceToHost);
	if (error != cudaSuccess)
		return _MEMCPY_TO_HOST_F;
	return _SUCCESS;
}

EXTERN int getConfocalResults(float *confocalOutput) {
	if (!isSetup)
		return _NOT_SETUP_F;
	if (!withConfocal)
		return _NO_CONFOCAL_ALLOCATED_F;
	int error = cudaMemcpy(confocalOutput, d_confocalOut, sizeof(float)*numPixels, cudaMemcpyDeviceToHost);
	if (error != cudaSuccess)
		return _MEMCPY_TO_HOST_F;
	return _SUCCESS;
}

EXTERN int processData() {
	if (!isSetup)
		return _NOT_SETUP_F;

	int error = cudaSuccess;
	windowData<<<blocks,threadsPerBlock>>>(d_data, d_fftIn, d_window, samplesPerPixel, fftSize, numPixels);
	error = cudaDeviceSynchronize();
	if (error != cudaSuccess)
		return _WINDOW_DATA_F;

	error = cufftExecR2C(handle, d_fftIn, d_fftOut);
	if (error != CUFFT_SUCCESS)
		return _FFT_F;
	
	//maskData<<<blocks, threadsPerBlock>>>(d_fftOut, d_maskedData, d_binaryIndices, fftSize, numPixels, pointsPerMask, numBinaryMasks);
	takeAbs<<<blocks, threadsPerBlock>>>(d_fftOut, d_magOut, fftSize, numPixels);
	error = cudaDeviceSynchronize();
	if (error != cudaSuccess)
		return _ABS_F;

	maskData << <blocks, threadsPerBlock >> > (d_magOut, d_TAUMSignals, d_binaryIndices, fftSize, numPixels, pointsPerMask, numBinaryMasks);
	error = cudaDeviceSynchronize();
	if (error != cudaSuccess)
		return _MASK_SUM_F;

	if (withConfocal) {
		sumConfocal << <sumBlocks, threadsPerSumBlock >> > (d_confocalRaw, d_confocalOut, confocalSamplesPerPixel, numPixels);
		error = cudaDeviceSynchronize();
		if (error != cudaSuccess)
			return _CONFOCAL_SUM_F;
	}
	return _SUCCESS;
}
