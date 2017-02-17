#ifndef __KERNEL__
#define __KERNEL__
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cufft.h>
#include <cuComplex.h>
#include <stdio.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cstdlib>
#include <math.h>
#define PI 3.14159

#define EXTERN extern "C"
#define GPUFUNCSDLL extern "C" __declspec(dllexport)


//Error Codes
#define _MEM_ALLOCATION_F -225
#define _MEMCPY_TO_DEVICE_F -226
#define _FFT_PLAN_F -227
#define _MEMCPY_TO_HOST_F -228

#define _NOT_SETUP_F -404
#define _NOT_INTERFEROGRAM_SETUP_F -405
#define _NOT_M_SCAN_SETUP_F -406
#define _INVALID_FUNCTION -408
#define _INVALID_DATA_TYPE_F -409
#define _GPU_IN_ERROR_STATE_F -410
#define _KERNEL_TOO_LARGE_F -411
#define _INVALID_MINMAX_F -412
#define _INVALID_CROP_RANGE -413
#define _NO_CONFOCAL_ALLOCATED_F -414

#define _SUCCESS 0
#define _WINDOW_DATA_F -101
#define _FFT_F -102
#define _ABS_F -103
#define _MASK_SUM_F -104
#define _CONFOCAL_SUM_F -105

using namespace std;

typedef uint16_t U16;
typedef cuFloatComplex Complex;
typedef uint64_t U64;
typedef int16_t I16;
typedef int64_t I64;


#endif
