#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <complex>
#include "cuda.h"
#include "cufft.h"

// include types.
#ifndef Included_Types
#define Included_Types
#include "cuTypes.h"
#endif

//
//	HOST & DEVICE FUNCTIONS
//

	__host__ __device__ cufftComplex conjugate( cufftComplex pIn );
	__host__ __device__ cufftDoubleComplex divideComplex( cufftDoubleComplex pOne, double pTwo );
	__host__ __device__ cufftComplex divideComplex( cufftComplex pOne, float pTwo );
	__host__ __device__ cufftComplex divideComplex( cufftComplex pOne, cufftComplex pTwo );
	__host__ __device__ cufftDoubleComplex divideComplex( cufftDoubleComplex pOne, cufftDoubleComplex pTwo );
	__host__ __device__ cufftDoubleComplex divideComplex( cufftDoubleComplex pOne, cufftComplex pTwo );
	__host__ __device__ cufftDoubleComplex divideComplex( double pOne, cufftDoubleComplex pTwo );
	__host__ __device__ double gaussian2D( double pNormalisation, double pX, double pY, double pAngle, double pR1, double pR2 );
	__host__ __device__ int intFloor( int pValue1, int pValue2 );
	__host__ __device__ float interpolateBeam( float * pBeam, int pBeamSize, int pI, int pJ, double pFracI, double pFracJ );
	__host__ __device__ cufftComplex interpolateBeam( cufftComplex * pBeam, int pBeamSize, int pI, int pJ, double pFracI, double pFracJ );
	__host__ __device__ int mod( int pValue1, int pValue2 );
	__host__ __device__ cufftDoubleComplex multComplex( cufftDoubleComplex pOne, double pTwo );
	__host__ __device__ cufftComplex multComplex( cufftComplex pOne, float pTwo );
	__host__ __device__ cufftComplex multComplex( cufftComplex pOne, cufftComplex pTwo );
	__host__ __device__ cufftDoubleComplex multComplex( cufftDoubleComplex pOne, cufftComplex pTwo );
	__host__ __device__ cufftDoubleComplex multComplex( cufftDoubleComplex pOne, cufftDoubleComplex pTwo );
	__host__ __device__ cufftDoubleComplex multComplex( cufftComplex pOne, cufftDoubleComplex pTwo );
	__host__ __device__ void swap( VectorI & pOne, VectorI & pTwo );
	__host__ __device__ void swap( int & pOne, int & pTwo );
	__host__ __device__ void swap( cufftComplex & pOne, cufftComplex & pTwo );
	__host__ __device__ void swap( bool & pOne, bool & pTwo );
	__host__ __device__ void swap( float & pOne, float & pTwo );

//
//	KERNEL FUNCTIONS
//

	__global__ void devAddArrays( cufftComplex * pOne, cufftComplex * pTwo, int pSize );
	__global__ void devAddArrays( cufftDoubleComplex * pOne, cufftDoubleComplex * pTwo, int pSize );
	__global__ void devAddArrays( float * pOne, float * pTwo, int pSize );
	__global__ void devConvertImage( cufftComplex * pOut, cufftDoubleComplex * pIn, int pSize );
	__global__ void devConvertImage( cufftComplex * pOut, float * pIn, int pSize );
	__global__ void devConvertImage( float * pOut, cufftDoubleComplex * pIn, int pSize );
	__global__ void devDivideImages( float * pOne, float * pTwo, bool * pMask, int pSizeOne, int pSizeTwo, bool pInterpolate );
	__global__ void devDivideImages( cufftComplex * pOne, cufftComplex * pTwo, bool * pMask, int pSizeOne, int pSizeTwo );
	__global__ void devDivideImages( cufftDoubleComplex * pOne, float * pTwo, bool * pMask, int pSizeOne, int pSizeTwo );
	__global__ void devDivideImages( cufftComplex * pOne, float * pTwo, bool * pMask, int pSizeOne, int pSizeTwo );
	__global__ void devDivideImages( cufftDoubleComplex * pOne, cufftDoubleComplex * pTwo, bool * pMask, int pSizeOne, int pSizeTwo );
	__global__ void devDivideImages( float * pOne, cufftDoubleComplex * pTwo, bool * pMask, int pSizeOne, int pSizeTwo );
	__global__ void devDivideImages( cufftDoubleComplex * pOne, cufftComplex * pTwo, bool * pMask, int pSizeOne, int pSizeTwo );
	__global__ void devFFTShift( cufftComplex * pDestination, cufftComplex * pSource, fftdirection pFFTDirection, int pSize );
	__global__ void devFFTShift( double * pDestination, cufftComplex * pSource, fftdirection pFFTDirection, int pSize );
	__global__ void devFFTShift( cufftComplex * pDestination, float * pSource, fftdirection pFFTDirection, int pSize );
	__global__ void devFFTShift( float * pDestination, cufftComplex * pSource, fftdirection pFFTDirection, int pSize );
	__global__ void devFindCutoffPixel( int * pTmpResults, int * pSupport, int pElements, findpixel pFindType );
	__global__ void devFindCutoffPixelParallel( cufftComplex * pKernel, int pSize, double * pMaxValue, int pCellsPerThread, int * pTmpResults, double pCutoffFraction, findpixel pFindType );
	__global__ void devFindCutoffPixelParallel( float * pKernel, int pSize, double * pMaxValue, int pCellsPerThread, int * pTmpResults, double pCutoffFraction, findpixel pFindType );
	__global__ void devGetMaxValue( double * pArray, double * pMaxValue, bool pUseAbsolute, int pElements );
	__global__ void devGetMaxValueParallel( cufftComplex * pArray, int pWidth, int pHeight, int pCellsPerThread, double * pBlockMax, bool pIncludeComplexComponent, bool pMultiplyByConjugate, bool * pMask );
	__global__ void devGetMaxValueParallel( float * pArray, int pWidth, int pHeight, int pNumImages, int pCellsPerThread, double * pBlockMax, bool pUseAbsolute, bool * pMask );
	__global__ void devGetPrimaryBeam( float * pPrimaryBeam, cufftDoubleComplex ** pMueller, int pImageSize, int pStokes );
	__global__ void devMakeBeam( float * pBeam, double pAngle, double pR1, double pR2, double pX, double pY, int pSize );
	__global__ void devMakeBeam( cufftComplex * pBeam, double pAngle, double pR1, double pR2, double pX, double pY, int pSize );
	__global__ void devMoveImages( cufftDoubleComplex * pOne, cufftComplex * pTwo, int pSize );
	__global__ void devMultiplyArrays( cufftComplex * pOne, int * pTwo, int pSize );
	__global__ void devMultiplyArrays( cufftComplex * pOne, float * pTwo, int pSize );
	__global__ void devMultiplyArrays( cufftComplex * pOne, cufftComplex * pTwo, int pSize, bool pConjugate );
	__global__ void devMultiplyArrays( float * pOne, float * pTwo, int pSize );
	__global__ void devMultiplyArrays( float * pOne, float * pTwo, bool * pMask, int pSizeOne, int pSizeTwo );
	__global__ void devMultiplyImages( float * pOne, float * pTwo, bool * pMask, int pSizeOne, int pSizeTwo, bool pInterpolate );
	__global__ void devMultiplyImages( cufftComplex * pOne, cufftComplex * pTwo, bool * pMask, int pSizeOne, int pSizeTwo );
	__global__ void devMultiplyImages( cufftComplex * pOne, float * pTwo, bool * pMask, int pSizeOne, int pSizeTwo );
	__global__ void devMultiplyImages( cufftDoubleComplex * pOne, cufftComplex * pTwo, bool * pMask, int pSizeOne, int pSizeTwo );
	__global__ void devMultiplyImages( cufftDoubleComplex * pOne, cufftDoubleComplex * pTwo, bool * pMask, int pSizeOne, int pSizeTwo );
	__global__ void devMultiplyImages( cufftComplex * pOne, float pScalar, bool * pMask, int pSizeOne );
	__global__ void devMultiplyImages( cufftDoubleComplex * pOne, float pScalar, bool * pMask, int pSizeOne );
	__global__ void devNormalise( cufftComplex * pArray, double pConstant, int pItems );
	__global__ void devNormalise( double * pArray, double * pConstant, int pItems );
	__global__ void devNormalise( double * pArray, double pConstant, int pItems );
	__global__ void devNormalise( float * pArray, double pConstant, int pItems );
	__global__ void devNormalise( float * pArray, double * pConstant, int pItems );
	__global__ void devResizeImage( cufftComplex * pNewImage, cufftComplex * pOldImage, int pNewSize, int pOldSize );
	__global__ void devResizeImage( float * pNewImage, float * pOldImage, int pNewSize, int pOldSize );
	__global__ void devReverseXDirection( float * pGrid, int pSize );
	__global__ void devReverseXDirection( cufftComplex * pGrid, int pSize );
	__global__ void devReverseYDirection( float * pGrid, int pSize );
	__global__ void devReverseYDirection( cufftComplex * pGrid, int pSize );
	__global__ void devScaleImage( cufftComplex * pNewImage, float * pOldImage, int pNewSize, int pOldSize, double pScale );
	__global__ void devScaleImage( cufftComplex * pNewImage, cufftComplex * pOldImage, int pNewSize, int pOldSize, double pScale );
	__global__ void devSquareRoot( cufftComplex * pArray, int pSize );
	__global__ void devSubtractArrays( cufftComplex * pOne, cufftComplex * pTwo, int pSize );
	__global__ void devSubtractArrays( cufftDoubleComplex * pOne, cufftDoubleComplex * pTwo, int pSize );
	__global__ void devTakeConjugateImage( cufftComplex * pImage, int pSize );
	__global__ void devUpdateComplexArray( cufftComplex * pArray, int pElements, float pReal, float pImaginary );
	__global__ void devUpperThreshold( cufftComplex * pImage, float pThreshold, int pSize );

//
//	HOST FUNCTIONS
//

	double deg( double pIn );
	bool moveDeviceToHost( void * pToPtr, void * pFromPtr, long int pSize, const char * pTask, int pLineNumber );
	bool moveHostToDevice( void * pToPtr, void * pFromPtr, long int pSize, const char * pTask, int pLineNumber );
	void finaliseFFT( cufftHandle pFFTPlan );
	int findCutoffPixel( cufftComplex * pdevKernel, double * pdevMaxValue, int pSize, double pCutoffFraction, findpixel pFindType );
	int findCutoffPixel( float * pdevKernel, double * pdevMaxValue, int pSize, double pCutoffFraction, findpixel pFindType );
	bool getMaxValue( cufftComplex * pdevImage, double * pdevMaxValue, int pWidth, int pHeight, bool pIncludeComplexComponent, bool pMultiplyByConjugate, bool * pdevMask );
	bool getMaxValue( float * pdevImage, double * pdevMaxValue, int pWidth, int pHeight, bool pUseAbsolute, bool * pdevMask, int pNumImages );
	cufftHandle initialiseFFT( int pSize );
	bool performFFT( cufftComplex ** pdevGrid, int pSize, fftdirection pFFTDirection, cufftHandle pFFTPlan, ffttype pFFTType, bool pResizeArray );
	double rad( double pIn );
	bool reserveGPUMemory( void ** pMemPtr, long int pSize, const char * pTask, int pLineNumber );
	void setThreadBlockSize1D( int * pThreads, int * pBlocks );
	void setThreadBlockSize2D( int pThreadsX, int pThreadsY, dim3 & pGridSize2D, dim3 & pBlockSize2D );
	bool zeroGPUMemory( void * pMemPtr, long int pSize, const char * pTask, int pLineNumber );
