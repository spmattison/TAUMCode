#include "definitions.cuh"

int calculateFFTSize();
int calculateBlocks();

EXTERN int clearGPU();

EXTERN int copyConfocalToDevice(U16 *confocal);
EXTERN int copyDataToDevice(U16 *data);

EXTERN int getResults(float *TAUMOutput);
EXTERN int getConfocalResults(float *confocalOutput);


EXTERN int processData();

EXTERN int setup(int numPxl, int sampsPerPxl, int numMasks, int ptsPerMask, int* binaryMasks, float *window, int confocalSamps);

EXTERN int updateConfocalSampling(int samples);
EXTERN int updateMasks(int ptsPerMask, int numMasks, int *maskIndices);
EXTERN int updateNumPixels(int nPixels);
