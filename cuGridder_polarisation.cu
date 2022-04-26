#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <complex>
#include "cuda.h"
#include <math.h>
#include <string.h>
#include "cufft.h"
#include <time.h>
#include <complex.h>
#include <fftw3.h>
#include <sys/sysinfo.h>

// include types.
#ifndef Included_Types
#define Included_Types
#include "cuTypes.h"
#endif

// include functions.
#include "cuFunctions.h"

// my phase correction code.
#ifndef Included_PhaseCorrection
#define Included_PhaseCorrection
#include "cppPhaseCorrection.h"
#endif

// my casacore interface.
#ifndef Included_Casacore
#define Included_Casacore
#include "cppCasacoreInterface.h"
#endif

// my image-plane reproject code.
#ifndef Included_Reprojection
#define Included_Reprojection
#include "cuReprojectionNotComplex.h"
#endif

// the data-processing class.
#ifndef Included_Data
#define Included_Data
#include "cuData_polarisation.h"
#endif

// the kernel set class.
#ifndef Included_KernelSet
#define Included_KernelSet
#include "cuKernelSet.h"
#endif

// the kernel set class.
#ifndef Included_KernelCache
#define Included_KernelCache
#include "cuKernelCache.h"
#endif

// the parameters class.
#ifndef Included_Parameters
#define Included_Parameters
#include "cuParameters.h"
#endif

using namespace std;

//
//	CONSTANTS
//

// cuda constants.
const int MAXIMUM_BLOCKS_PER_DIMENSION = 65535;
//const int MAX_THREADS = 33554432;		// maximum number of total threads per cuda call (32768 x 1024). We can actually have 65535 x 1024, but we set the limit lower.

// other constants.
const int MAX_SIZE_FOR_PSF_FITTING = 60;

// cleaning
const int CYCLE_NITER = -1;

//
//	ENUMERATED TYPES
//

//
//	STRUCTURES
//

//
//	FORWARDS DECLARATIONS
//

//
//	GLOBAL VARIABLES
//

// the clean parameters class.
Parameters * _param = Parameters::getInstance();

// timing variables - to be deleted.
double _generateKernel = 0.0, _uploadData = 0.0, _setup = 0.0, _gridding = 0.0, _otherGridding = 0.0;

// array of data objects - one for each mosaic componet.
vector<Data *> _hstData;

float * _hstPrimaryBeamPattern = NULL;
float * _hstNormalisationPattern = NULL;
float * _hstPrimaryBeamRatioPattern = NULL;

int _hstNumDirtyBeams = 1;

// the FFT on the host: only used for images too large to fit on the GPU.
fftwf_complex * _hstFFTGrid = NULL;
fftwf_plan _fftPlanForward;
fftwf_plan _fftPlanInverse;
bool _fftForwardActive = false;
bool _fftInverseActive = false;

// kerkel cache.
vector<KernelCache *> _griddingKernelCache;
vector<KernelCache *> _degriddingKernelCache;
vector<KernelCache *> _psfKernelCache;
KernelCache _psKernelCache;

// psf
int _hstPsfX = 0, _hstPsfY = 0;
	
// clean beam.
int _hstCleanBeamSize = 0;	// holds the size of the non-zero portion of the clean beam.

// deconvolution image.
float * _devImageDomainPSFunction = NULL;

// device properties.
int _maxThreadsPerBlock = 0;
int _warpSize = 0;
long int _gpuMemory = 0;

// kernel calls.
dim3 _gridSize2D( 1, 1 );
dim3 _gridSize3D( 1, 1, 1 );
dim3 _blockSize2D( 1, 1 );
int _itemsPerBlock;		// the number of items (i.e. visibilities) along the x-axis of a block. will be 1 if
				//	the kernel is larger than one block.
int _blocksPerItem;		// the number of blocks used for each visibility. will be equal to 1 if the kernel is
				// smaller than one block.

int _gridSize1D;
int _blockSize1D;
	
// create a casacore interface.
CasacoreInterface _hstCasacoreInterface;

//
//	CONSTANT MEMORY
//

__constant__ int _devVisibilityBatchSize;
__constant__ int _devPsfX;
__constant__ int _devPsfY;

//
//	GENERAL FUNCTIONS
//

//
//	atomicAddDouble()
//
//	CJS: 09/10/2018
//
//	Fudges an atomic add for doubles, which CUDA cannot do by itself.
//

__device__ double atomicAddDouble( double * pAddress, double pVal )
{

	unsigned long long int * address_as_ull = (unsigned long long int *)pAddress;
	unsigned long long int old = *address_as_ull, assumed;

	do
	{
		assumed = old;
		old = atomicCAS( address_as_ull, assumed, __double_as_longlong( pVal + __longlong_as_double( assumed ) ) );
	} while (assumed != old);

	return __longlong_as_double( old );

} // atomicAddDouble

//
//	addComplex()
//
//	CJS: 28/09/2015
//
//	Add two complex numbers (using atomic add since multiple threads may be trying to write the the same address).
//

__device__ void addComplex( cufftComplex * pOne, cufftComplex pTwo )
{
	
	atomicAdd( /* pAddress = */ &(pOne->x), /* pVal = */ pTwo.x );
	atomicAdd( /* pAddress = */ &(pOne->y), /* pVal = */ pTwo.y );
	
} // addComplex

//
//	DEVICE FUNCTIONS
//

//
//	devCalculateGaussianError()
//
//	CJS: 30/10/2018
//
//	Calculate the error squared between an image and a Gaussian fit. The errors are
//	stored in the pError array.
//

__global__ void devCalculateGaussianError
			(
			float * pImage,				// the image to process
			double * pError,				// the returned error
			int pSizeOfFittingRegion,			// we only fit a small region of our image; this is the size
			double pCentreX,				// }- the centre of the Gaussian
			double pCentreY,				// }
			double pAngle,					// the angle of rotation of the Gaussian
			double pR1,					// the length of axis 1
			double pR2,					// the length of axis 2
			int pImageSize,				// the image size in pixels
			double pNormalisation				// a scalar for the generated Gaussian
			)
{
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;

	// ensure we are within the fitting region.
	if (i < pSizeOfFittingRegion && j < pSizeOfFittingRegion)
	{
		
		int index = (j * pSizeOfFittingRegion) + i;
		int posX = (int) round( pCentreX ) - (pSizeOfFittingRegion / 2) + i;
		int posY = (int) round( pCentreY ) - (pSizeOfFittingRegion / 2) + j;

		// calculate the Gaussian error for these fitting parameters.
		double error = 0;
		if (posX >= 0 && posX < pImageSize && posY >= 0 && posY < pImageSize)
			if (pImage[ (posY * pImageSize) + posX ] >= 0.5)
				error = pow( pImage[ (posY * pImageSize) + posX ] - gaussian2D(	/* pNormalisation = */ pNormalisation,
													/* pX = */ (double) posX - pCentreX,
													/* pY = */ (double) posY - pCentreY,
													/* pAngle = */ pAngle,
													/* pR1 = */ pR1,
													/* pR2 = */ pR2 ), 2 );
		pError[ index ] = error;

	}

} // devCalculateGaussianError

//
//	devSumDouble()
//
//	CJS: 30/10/2018
//
//	Adds up the double values of an array, 10 items per thread, and stores the sums in another array.
//

__global__ void devSumDouble( double * pValue, double * pSum, int pItems )
{

	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	double sum = 0;
	for ( int i = index * 10; i < (index + 1) * 10; i++ )

		// ensure we are within the bounds of the array.
		if (i < pItems)
			sum = sum + pValue[ i ];

	// store the sum.
	pSum[ index ] = sum;

} // devSumDouble

//
//	devGridVisibilities()
//
//	CJS: 24/09/2015
//
//	Produce an image of gridding visibilities by convolving the complex visibilities with the kernel function.
//
//	visibilities are arranged in one of two ways: either there are multiple visibilities per block (for small kernels), or we have multiple blocks
//	per visibility (for large kernels). there is only one block in the x-direction, unless we have exceeded the allowable limit (usually 65536, held
//	in a constant at the top of this code) - in which case we add additional columns of blocks, which are traversed in the order
//	column 0 (row 0 - max), column 1 (row 0 - max), etc.
//
//	multiple visibilities per block:
//
//		visibility 0-   visibility 1-   visibility 2-   visibility 3-
//		bbbbbbbbbbbbbbbbbbbbbbbbbbbbb   bbbbbbbbbbbbbbbbbbbbbbbbbbbbb
//		t t t t t t t   t t t t t t t   t t t t t t t   t t t t t t t
//
//	multiple blocks per visibility:
//
//		-------------------------visibility 0------------------------   -------------------------visibility 1------------------------
//		bbbbbbbbbbbbbbbbbbbbbbbbbbbbb   bbbbbbbbbbbbbbbbbbbbbbbbbbbbb   bbbbbbbbbbbbbbbbbbbbbbbbbbbbb   bbbbbbbbbbbbbbbbbbbbbbbbbbbbb
//		t t t t t t t t t t t t t t t   t t t t t t t t t t t t t t t   t t t t t t t t t t t t t t t   t t t t t t t t t t t t t t t
//

__global__ void devGridVisibilities
			(
			cufftComplex * pGrid,				// the grid (OUTPUT)
			cufftComplex * pVisibility,			// the list of visibilities (INPUT)
			int pVisibilitiesPerBlock,			// > 1 if the kernel is smaller than one block
			int pBlocksPerVisibility,			// > 1 if the kernel is larger than one block
			VectorI * pGridPosition,			// list of u,v,w coordinates to place on the grid (INPUT)
			cufftComplex * pKernel,			// data area holding the kernels (INPUT)
			int * pKernelIndex,				// list of kernel indexes to apply to each visibility (INPUT)
			float * pWeight,				// list of weights to apply to each visibility (INPUT)
			int pNumVisibilities,				// the number of visibilities to grid (INPUT)
			int pSize,					// the size of the image
			bool pComplex,					// are we gridding complex or non-complex visibilities?
			int pSupport					// the support size of the kernel
			)
{

	// the dynamic memory area stores arrays of visibilities, samples and frequencies.
	extern __shared__ char shrDynamic[];
	
	// pointers to dynamic shared memory.
	//	shrVisibility - holds all the visibilities in this thread block.
	//	shrGridPosition - holds all the grid positions (3 x int) for the visibilities in this thread block.
	//	shrKernelIndex - holds the kernel indexes for the visibilities in this thread block.
	//	shrWeight - holds the weight for this visibility.
	cufftComplex * shrVisibility = (cufftComplex *) &shrDynamic[ 0 ];
	VectorI * shrGridPosition = (VectorI *) &shrDynamic[ pVisibilitiesPerBlock * sizeof( cufftComplex ) ];
	int * shrKernelIndex = (int *) &shrDynamic[ pVisibilitiesPerBlock * (sizeof( cufftComplex ) + sizeof( VectorI )) ];
	float * shrWeight = (float *) &shrDynamic[ pVisibilitiesPerBlock * (sizeof( cufftComplex ) + sizeof( VectorI ) + sizeof( int )) ];

	int visibilityArrayIndex = 0;
	int visibilityIndex = 0;
	int kernelX = 0;
	int kernelY = 0;
	int kernelSize = (pSupport * 2) + 1;
	bool firstThread = false;	// true if a) this is the first thread for this visibility, or b) this is the first
					//	thread for this block.
	bool applyWeighting = (pWeight != NULL);
		
	// flatten the 2D grid to get the block number.
	int blockIndex = (blockIdx.x * gridDim.y) + blockIdx.y;

	// have we got multiple visibilities per block, or multiple blocks per visibility? we'll never have both at the same time.
	if (pVisibilitiesPerBlock > 1)
	{
		
		// visibilities are only stacked in the y-direction - never the x-direction. the x-direction
		// gives only the kernel pixels in the x-direction for a single visibility.

		// we have multiple visibilities per block. calculate the array index within the block.
		visibilityArrayIndex = threadIdx.y / kernelSize;
		visibilityIndex = (blockIndex * pVisibilitiesPerBlock) + visibilityArrayIndex;
		
		// calculate the kernel position.
		kernelX = threadIdx.x;
		kernelY = (threadIdx.y % kernelSize);
		firstThread = (kernelX == 0 && kernelY == 0);
		
	}
	else
	{
		
		// calculate the block index WITHIN the visibility.
		int blockIndexInVisibility = blockIndex % pBlocksPerVisibility;
		
		// we have multiple blocks per visibility. set the array index to 0 because we only have one array element for this block.
		visibilityArrayIndex = 0;
		visibilityIndex = blockIndex / pBlocksPerVisibility;
		
		// calculate the kernel position.
		kernelX = threadIdx.x;
		kernelY = (blockIndexInVisibility * blockDim.y) + threadIdx.y;
		firstThread = (threadIdx.x == 0 && threadIdx.y == 0);
		
	}

	// if this is the first thread for a visibility we need to get the visibility, grid position and kernel index and store them in shared memory.
	if (firstThread == true && visibilityIndex < pNumVisibilities)
	{
		
		// get grid position, kernel index and visibility.
		// the visibility is either a single-precision complex number, or else a double-precision scalar.
		if (pComplex == true)
			shrVisibility[ visibilityArrayIndex ] = pVisibility[ visibilityIndex ];
		else
		{
			shrVisibility[ visibilityArrayIndex ].x = (float) ((double *) pVisibility)[ visibilityIndex ];
			shrVisibility[ visibilityArrayIndex ].y = 0.0;
		}
		shrGridPosition[ visibilityArrayIndex ] = pGridPosition[ visibilityIndex ];

		if (pKernelIndex != NULL)
			shrKernelIndex[ visibilityArrayIndex ] = pKernelIndex[ visibilityIndex ];
		else
			shrKernelIndex[ visibilityArrayIndex ] = 0;
		if (applyWeighting == true)
			shrWeight[ visibilityArrayIndex ] = pWeight[ visibilityIndex ];
		
	}
	
	__syncthreads();
	
	// check that we haven't gone outside the kernel dimensions (some threads will), and that we haven't gone past the last item in the visibility list.
	if ( kernelX < kernelSize && kernelY < kernelSize && visibilityIndex < pNumVisibilities )
	{
		
		// get exact grid coordinates,
		VectorI grid = shrGridPosition[ visibilityArrayIndex ];

		// if we are gridding then we want to pick a grid offset that matches the kernel offset.
		grid.u += (kernelX - pSupport);
		grid.v += (kernelY - pSupport);
					
		// get kernel value.
		cufftComplex kernel = { .x = 0.0, .y = 0.0 };
		int kernelIndex = (kernelY * kernelSize) + kernelX;
		kernelIndex += shrKernelIndex[ visibilityArrayIndex ] * kernelSize * kernelSize;

		// the kernel is either a single-precision complex value, or else a single-precision scalar value.
		if (pComplex == true)
			kernel = pKernel[ kernelIndex ];
		else
			kernel.x = ((float *) pKernel)[ kernelIndex ];

		// is this pixel within the grid range?
		if ((grid.u >= 0) && (grid.u < pSize) && (grid.v >= 0) && (grid.v < pSize))
		{
							
			// get pointer to grid.
			cufftComplex * gridPtr = NULL;
			if (pComplex == true)
				gridPtr = &pGrid[ (grid.v * pSize) + grid.u ];
			else
				gridPtr = (cufftComplex *) &((float *) pGrid)[ (grid.v * pSize) + grid.u ];
					
			// update the grid using an atomic add (passing a pointer and the value to add).
			// add complex and real numbers differently.
			if (pComplex == true)
			{

				// with or without weighting.
				if (applyWeighting == true)
					addComplex( gridPtr, multComplex( multComplex( shrVisibility[ visibilityArrayIndex ],
														shrWeight[ visibilityArrayIndex ] ), kernel ) );
				else
					addComplex( gridPtr, multComplex( shrVisibility[ visibilityArrayIndex ], kernel ) );

			}
			else
			{

				// with or without weighting.
				if (applyWeighting == true)
					atomicAdd( (float *) gridPtr, shrVisibility[ visibilityArrayIndex ].x * kernel.x * shrWeight[ visibilityArrayIndex ] ); 
				else
					atomicAdd( (float *) gridPtr, shrVisibility[ visibilityArrayIndex ].x * kernel.x ); 

			}

		}
	
	}
	
} // devGridVisibilities

//
//	devDegridVisibilities()
//
//	CJS: 16/10/2020
//
//	Degrid a list of visibilities. The grid is divided into X (kernel.x), Y (kernel.y), and Z (page, since there are far more visibilities than blocks).
//

__global__ void devDegridVisibilities
			(
			cufftComplex * pGrid,
			cufftComplex * pVisibility,
			VectorI * pGridPosition,
			cufftComplex * pKernel,
			int * pKernelIndex,
			int pNumVisibilities,
			int pSize,
			int pVisibilitiesPerPage,
			int pSupport
			)
{

	int kernelX = blockIdx.x;
	int kernelY = blockIdx.y;
	int page = blockIdx.z;
	int kernelSize = (2 * pSupport) + 1;
	int visibilityIndex = (page * pVisibilitiesPerPage) + threadIdx.x;
	
	// check that we haven't gone outside the kernel dimensions (some threads will), and that we haven't gone past the last item in the visibility list.
	if ( kernelX < kernelSize && kernelY < kernelSize && visibilityIndex < pNumVisibilities )
	{
		
		// get exact grid coordinates,
		VectorI grid = pGridPosition[ visibilityIndex ];

		// since we are degridding then the grid offset should be opposite to the kernel offset. the reason for this is that we are actually convolving the kernel
		// with the grid, and reading off the pixel value from the grid position.
		grid.u -= (kernelX - pSupport);
		grid.v -= (kernelY - pSupport);
					
		// get kernel value.
		cufftComplex kernel = pKernel[ (pKernelIndex[ visibilityIndex ] * kernelSize * kernelSize) + (kernelY * kernelSize) + kernelX ];

		// is this pixel within the grid range? If so, add up visibility.
		if ((grid.u >= 0) && (grid.u < pSize) && (grid.v >= 0) && (grid.v < pSize))
			addComplex( &pVisibility[ visibilityIndex ], multComplex( pGrid[ (grid.v * pSize) + grid.u ], kernel ) );
	
	}

} // devDegridVisibilities

//
//	devBuildWeightedDirtyBeam()
//
//	CJS: 01/09/2021
//
//	Build a dirty beam from a series of dirty beams, weighted by the primary beams.
//

__global__ void devBuildWeightedDirtyBeam
			(
			float * pOutput,				// output psf
			float * pInput,				// input psf for a single a-plane and/or mosaic component
			float * pPrimaryBeam,				// the primary beam pattern at the mean frequency (used for mosaicing)
			float * pNormalisationPattern,		// the normalisation pattern
			float * pPrimaryBeamPattern,			// the primary beam pattern
			int pPsfSize,					// the size of the psf image
			int pBeamSize,					// the size of the primary beam image
			int pImageSize,				// the size of the dirty image
			int pX,					// the X-position of the component
			int pY						// the Y-position of the component
			)
{

	__shared__ float shrPrimaryBeamComponent;
//	__shared__ float shrPrimaryBeamPatternComponent;

	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	int psfSupport = pPsfSize / 2;

	// calculate the component position in the beam image, and get the primary beam value for this component position.
	if (threadIdx.x == 0 && threadIdx.y == 0)
	{

		shrPrimaryBeamComponent = 0.0;
//		shrPrimaryBeamPatternComponent = 0.0;
		int xBeamComponent = (int) ((double) pX * (double) pBeamSize / (double) pImageSize);
		int yBeamComponent = (int) ((double) pY * (double) pBeamSize / (double) pImageSize);
		if (xBeamComponent >= 0 && xBeamComponent < pBeamSize && yBeamComponent >= 0 && yBeamComponent < pBeamSize)
		{
			if (pPrimaryBeam != NULL)
				shrPrimaryBeamComponent = pPrimaryBeam[ (yBeamComponent * pBeamSize) + xBeamComponent ];
//			shrPrimaryBeamPatternComponent = pPrimaryBeamPattern[ (yBeamComponent * pBeamSize) + xBeamComponent ];
		}

	}

	__syncthreads();

	// check we're within the range of the psf image.
	if (i >= 0 && i < pPsfSize && j >= 0 && j < pPsfSize)
	{

		int psfIndex = (j * pPsfSize) + i;

		// get the position of this pixel within the beam image.
		int xBeamPixel = (int) ((double) (pX + i - psfSupport) * (double) pBeamSize / (double) pImageSize);
		int yBeamPixel = (int) ((double) (pY + j - psfSupport) * (double) pBeamSize / (double) pImageSize);
		if (xBeamPixel >= 0 && xBeamPixel < pBeamSize && yBeamPixel >= 0 && yBeamPixel < pBeamSize)
		{

			int beamPixelIndex = (yBeamPixel * pBeamSize) + xBeamPixel;

			// get the normalisation pattern for this pixel.
			float normalisationPattern = 0.0;
			if (pNormalisationPattern != NULL)
				normalisationPattern = pNormalisationPattern[ beamPixelIndex ];

			// add the value to the output pixel.
			pOutput[ psfIndex ] += pInput[ psfIndex ] * pPrimaryBeam[ beamPixelIndex ] * shrPrimaryBeamComponent / normalisationPattern;
//			pOutput[ psfIndex ] += pInput[ psfIndex ] * pPrimaryBeam[ beamPixelIndex ] / normalisationPattern;

		}

	}

} // devBuildWeightedDirtyBeam

//
//	devBuildMask()
//
//	CJS: 15/05/2020
//
//	Builds a boolean mask based upon a threshold value.
//

__global__ void devBuildMask
			(
			bool * pMask,					// the output mask
			float * pArray,				// the input image
			int pSize,					// the size of the image
			double pValue,					// the threshold value
			masktype pMaxMin				// MASK_MIN if we want TRUE to mean > threshold, or MASK_MAX if we want TRUE to mean < threshold
			)
{
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	// check that we haven't gone outside the grid dimensions, and set the mask.
	if (i < pSize && j < pSize)
	{
		if (pMaxMin == MASK_MIN)
			pMask[ (j * pSize) + i ] = (pArray[ (j * pSize) + i ] >= pValue);
		else
			pMask[ (j * pSize) + i ] = (pArray[ (j * pSize) + i ] < pValue);
	}

} // devBuildMask

//
//	devGetMaxMfsResidualParallel()
//
//	CJS: 21/06/2021
//
//	Get the maximum MFS residual value:  R0[j]^2.A11[0] + R1[j]^2.A00[0] - 2.R0[j].R1[j].A01[0]. Sault et al 1994, A&ASS, 108, 585-594, Eq.22
//

__global__ void devGetMaxMfsResidualParallel
			(
			float * pR0,					// the zeroth order image
			float * pR1,					// the first order image
			int pWidth,					// the array width
			int pHeight,					// the array height
			int pCellsPerThread,				// the number of array cells we should search per thread
			double * pBlockMax,				// an array that stores the maximum values per block
			bool * pMask,					// an optional mask that restricts which pixels we search
			double pA00,					//
			double pA11,					//
			double pA01					//
			)
{
	
	// the dynamic memory area stores arrays of visibilities, samples and frequencies.
	extern __shared__ double shrMaxValue[];
	
	double maxValue = 0;
	double maxI = 0;
	double maxJ = 0;
	
	// get the starting cell index.
	int cell = ((blockIdx.x * blockDim.x) + threadIdx.x) * pCellsPerThread;
	
	for ( int i = cell; i < cell + pCellsPerThread; i++ )
	{
		
		// ensure we are within bounds.
		if (i < pWidth * pHeight)
		{
		
			double value = (pow( (double) pR0[ i ], 2 ) * pA11) + (pow( (double) pR1[ i ], 2 ) * pA00) -
						(2 * (double) pR0[ i ] * (double) pR1[ i ] * pA01);

			// has this cell been masked? we can include it if there is no mask provided, or if the mask is TRUE (i.e. cell is good).
			bool includeCell = (pMask == NULL);
			if (pMask != NULL)
				includeCell = (pMask[ i ] == true);
			
			// is this value greater than the previous greatest?
			if (value > maxValue && includeCell == true)
			{
				maxValue = (double) value;
				maxI = (double) (i % pWidth);
				maxJ = i / pWidth;
			}
		
		}
		
	}
		
	// update maximum values.
	shrMaxValue[ (threadIdx.x * MAX_PIXEL_DATA_AREA_SIZE) + MAX_PIXEL_VALUE ] = maxValue;
	shrMaxValue[ (threadIdx.x * MAX_PIXEL_DATA_AREA_SIZE) + MAX_PIXEL_X ] = maxI;
	shrMaxValue[ (threadIdx.x * MAX_PIXEL_DATA_AREA_SIZE) + MAX_PIXEL_Y ] = maxJ;
	shrMaxValue[ (threadIdx.x * MAX_PIXEL_DATA_AREA_SIZE) + MAX_PIXEL_REAL ] = maxValue;
	shrMaxValue[ (threadIdx.x * MAX_PIXEL_DATA_AREA_SIZE) + MAX_PIXEL_IMAG ] = 0;
	
	__syncthreads();
	
	// now, get the maximum value from the shared array.
	if (threadIdx.x == 0)
	{
	
		double maxValue = 0;
		double maxI = 0;
		double maxJ = 0;
		double maxValueReal = 0;
		double maxValueImag = 0;
	
		for ( int i = 0; i < blockDim.x; i++ )
		{
		
			double value = shrMaxValue[ (i * MAX_PIXEL_DATA_AREA_SIZE) + MAX_PIXEL_VALUE ];
			
			// is this value greater than the previous greatest?
			if (value > maxValue)
			{
				maxValue = value;
				maxI = shrMaxValue[ (i * MAX_PIXEL_DATA_AREA_SIZE) + MAX_PIXEL_X ];
				maxJ = shrMaxValue[ (i * MAX_PIXEL_DATA_AREA_SIZE) + MAX_PIXEL_Y ];
				maxValueReal = shrMaxValue[ (i * MAX_PIXEL_DATA_AREA_SIZE) + MAX_PIXEL_REAL ];
				maxValueImag = shrMaxValue[ (i * MAX_PIXEL_DATA_AREA_SIZE) + MAX_PIXEL_IMAG ];
			}
			
		}
		
		// update global memory with these values.
		pBlockMax[ (blockIdx.x * MAX_PIXEL_DATA_AREA_SIZE) + MAX_PIXEL_VALUE ] = maxValue;
		pBlockMax[ (blockIdx.x * MAX_PIXEL_DATA_AREA_SIZE) + MAX_PIXEL_X ] = maxI;
		pBlockMax[ (blockIdx.x * MAX_PIXEL_DATA_AREA_SIZE) + MAX_PIXEL_Y ] = maxJ;
		pBlockMax[ (blockIdx.x * MAX_PIXEL_DATA_AREA_SIZE) + MAX_PIXEL_REAL ] = maxValueReal;
		pBlockMax[ (blockIdx.x * MAX_PIXEL_DATA_AREA_SIZE) + MAX_PIXEL_IMAG ] = maxValueImag;
	
	}
	
} // devGetMaxMfsResidualParallel

//
//	devConvertFloatToComplex()
//
//	CJS: 21/06/2021
//
//	Copy an array of floats into an array of complex numbers, setting the imaginary part to zero.
//

__global__ void devConvertFloatToComplex
			(
			cufftComplex * pOut,				// output array
			float * pIn,					// input array
			int pItems					// number of items
			)
{

	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	// move this value into the complex array.
	if (index < pItems)
	{
		pOut[ index ].x = pIn[ index ];
		pOut[ index ].y = 0.0;
	}

} // devConvertFloatToComplex

//
//	devRearrangeKernel()
//
//	CJS: 22/01/2016
//
//	Rearrange a kernel so that the real numbers are at the start and the imaginary numbers are at the end.
//

__global__ void devRearrangeKernel
			(
			float * pTarget,				// the output array
			float * pSource,				// the input array
			long int pElements				// number of items
			)
{
	
	long int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	// update the kernels, but only if we are within the bounds of the array.
	if (index < pElements)
	{
		pTarget[ index ] = pSource[ index * 2 ];
		pTarget[ index + pElements ] = pSource[ (index * 2) + 1 ];
	}

} // devRearrangeKernel

//
//	devAddSubtractBeam()
//
//	CJS: 06/11/2015
//
//	Add or subtracts the clean beam/dirty beam from the clean image/dirty image.
//
//	The window size is the support size of the region of the beam that is to be added or subtracted. the rest of the beam outside this region is ignored.
//

__global__ void devAddSubtractBeam(	float * pImage, float * pBeam, double * pMaxValue, double pLoopGain,
						int pImageWidth, int pImageHeight, int pBeamSize, addsubtract pAddSubtract	)
{
	
	__shared__ double maxValue;
	__shared__ int maxX;
	__shared__ int maxY;
	
	// retrieve the maximum value and pixel position.
	if ( threadIdx.x == 0 && threadIdx.y == 0 )
	{
		
		maxValue = pMaxValue[ MAX_PIXEL_VALUE ];
		maxX = (int) round( pMaxValue[ MAX_PIXEL_X ] );
		maxY = (int) round( pMaxValue[ MAX_PIXEL_Y ] );

	}
	
	__syncthreads();
	
	// calculate position in clean/dirty beam image (i,j).
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	// check that we haven't gone outside the grid dimensions (some threads will).
	if (i >= 0 && i < pBeamSize && j >= 0 && j < pBeamSize)
	{
				
		// calculate position in clean image (x,y).
		int x = maxX + i - _devPsfX;
		int y = maxY + j - _devPsfY;
				
		// are we within the image bounds ?
		if (x >= 0 && x < pImageWidth && y >= 0 && y < pImageHeight)
		{
					
			// get some pointers to the image, psf, and primary beam pattern.
			float * tmpImage = &pImage[ (y * pImageWidth) + x ];
			float tmpPSF = pBeam[ (j * pBeamSize) + i ];
						
			// subtract the psf (scaled).
			if (pAddSubtract == ADD)
				*tmpImage += maxValue * tmpPSF * pLoopGain;
			else
				*tmpImage -= maxValue * tmpPSF * pLoopGain;
					
		}
		
	}
	
} // devAddSubtractBeam

//
//	devAddPixelToModelImage()
//
//	CJS: 17/08/2018
//
//	Builds the model image by adding a single pixel.
//	We only have one thread because there is only one pixel to add with each iteration.
//

__global__ void devAddPixelToModelImage( float * pModelImage, double * pMaxValue, double pLoopGain, int pSize )
{
	
	int i = (int) round( pMaxValue[ MAX_PIXEL_X ] );
	int j = (int) round( pMaxValue[ MAX_PIXEL_Y ] );
	double maxValue = pMaxValue[ MAX_PIXEL_VALUE ];
	
	// double check that our maximum pixel position is within the bounds of the image.
	if (i >= 0 && j >= 0 && i < pSize && j < pSize)
					
		// add the pixel.
		pModelImage[ (j * pSize) + i ] += (maxValue * pLoopGain);
	
} // devAddPixelToModelImage

//
//	HOST FUNCTIONS
//

//
//	setThreadBlockSizeForGridding()
//
//	CJS:	10/11/2015
//
//	Determine a suitable thread and block size for the current GPU.
//	The number of threads must be less than the maximum number allowed by the current GPU.
//
//	For gridding the x-axis of the kernel size must be less than the block size limit (usually 1024). Rows
//	from the kernel are moved one at a time to the next block until the block size is within the limit. Blocks are tiled
//	in the y-direction first, until we hit the block dimension limit (usually 65536) and then we move to the next column in the
//	x-direction.
//
//	This subroutine is used when we have N number of small XxY grids (i.e. N visibilities, each involving a convolution of
//	the XxY kernel).
//

void setThreadBlockSizeForGridding( int pThreadsX, int pThreadsY, int pItems )
{
	
	_gridSize2D.x = 1;
	_gridSize2D.y = 1;
	
	// store the total number of X and Y threads - these threads represent the X and Y axes of the kernel.
	_blockSize2D.x = pThreadsX;
	_blockSize2D.y = pThreadsY;
	
	// how many items can we process in each block? if we have more than 1, then we tile them in the x-direction - there is only ever one item in the y-direction.
	_itemsPerBlock = _maxThreadsPerBlock / (pThreadsX * pThreadsY);
	
	// ensure _itemsPerBlock is not larger than the required number of items.
	if (_itemsPerBlock > pItems)
		_itemsPerBlock = pItems;

	// if we have more than one item then increase the block size accordingly.
	if (_itemsPerBlock > 0)
		_blockSize2D.y *= _itemsPerBlock;
	else
	{

		// _itemsPerBlock is zero. this means we need to split each kernel over a number of blocks, which are tiled in the y direction.
		// do we still have too many threads for one block?
		while ((_blockSize2D.x * _blockSize2D.y) > _maxThreadsPerBlock && _gridSize2D.y < 905)
		{
		
			// increment the number of Y blocks, and divide the required number of y threads between each block.
			_gridSize2D.y++;
			_blockSize2D.y = pThreadsY / _gridSize2D.y;
			if (pThreadsY % _gridSize2D.y != 0)
				_blockSize2D.y++;
//			_blockSize2D.y = (int) ceil( (double) pThreadsY / (double) _gridSize2D.y );
		
		}

		// we have now split our threads over multiple blocks, set items-per-block to 1.
		_itemsPerBlock = 1;
	
	}

	// set the number of blocks per item.
	_blocksPerItem = _gridSize2D.y;
	
	// divide the number of items by the number per block to get the total required number of blocks. They will initially ALL be tiled in the y direction, but
	// we will shortly increment the x size in order to bring the y size below the maximum.
	long int requiredBlocks = pItems / _itemsPerBlock;
	if (pItems % _itemsPerBlock != 0)
		requiredBlocks++;
	requiredBlocks *= (long int) _blocksPerItem;
//	int requiredBlocks = (int) ceil( (double) pItems / (double) _itemsPerBlock ) * _blocksPerItem;

	// ensure the number of y blocks is less than the maximum allowed. the x-size should still be 1 at this point, so
	// we keep incrementing the x-size until the y-size is within the required limit.
	long int yBlocks = requiredBlocks;
	while (yBlocks > MAXIMUM_BLOCKS_PER_DIMENSION)
	{
		_gridSize2D.x++;
		yBlocks = requiredBlocks / _gridSize2D.x;
		if (requiredBlocks % _gridSize2D.x != 0)
			yBlocks++;
//		_gridSize2D.y = (int) ceil( (double) requiredBlocks / (double) _gridSize2D.x );
	}
	_gridSize2D.y = (int) yBlocks;

} // setThreadBlockSizeForGridding

//
//	setThreadBlockSizeForDegridding()
//
//	CJS:	10/11/2015
//
//	Determine a suitable thread and block size for the current GPU.
//	The number of threads must be less than the maximum number allowed by the current GPU.
//
//	For gridding the x-axis of the kernel size must be less than the block size limit (usually 512). Rows
//	from the kernel are moved one at a time to the next block until the block size is within the limit. Blocks are tiled
//	in the y-direction first, until we hit the block dimension limit (usually 65536) and then we move to the next column in the
//	x-direction.
//
//	This subroutine is used when we have N number of small XxY grids (i.e. N visibilities, each involving a convolution of
//	the XxY kernel).
//

void setThreadBlockSizeForDegridding( int pKernelSizeX, int pKernelSizeY, int pNumVisibilities )
{

	_gridSize3D.x = pKernelSizeX;
	_gridSize3D.y = pKernelSizeY;
	_gridSize3D.z = (pNumVisibilities / _maxThreadsPerBlock);
	if (pNumVisibilities % _maxThreadsPerBlock != 0)
		_gridSize3D.z += 1;

} // setThreadBlockSizeForDegridding

//
//	minimum()
//
//	CJS: 27/06/2020
//
//	Take the minimum of two integers.
//

int minimum( int pOne, int pTwo )
{

	int returnValue = pOne;
	if (pTwo < pOne)
		returnValue = pTwo;

	// return something.
	return returnValue;

} // minimum

//
//	quickSortComponents()
//
//	CJS: 02/06/2020
//
//	Sort a list of clean image components into order of y position.
//

void quickSortComponents( VectorI * phstComponentListPos, double ** phstComponentListValue, int pLeft, int pRight, int pNumTaylorTerms )
{

	long int i = pLeft, j = pRight;
	int pivot = phstComponentListPos[ (pLeft + pRight) / 2 ].v;

	// partition.
	while (i <= j)
	{

		while (phstComponentListPos[ i ].v < pivot)
			i = i + 1;
		while (phstComponentListPos[ j ].v > pivot)
			j = j - 1;
		if (i <= j)
		{

			swap( phstComponentListPos[ i ], phstComponentListPos[ j ] );
			for ( int taylorTerm = 0; taylorTerm < pNumTaylorTerms; taylorTerm++ )
				swap( phstComponentListValue[ taylorTerm ][ i ], phstComponentListValue[ taylorTerm ][ j ] );

			i = i + 1;
			j = j - 1;

		}

	}
	
	// recursion.
	if (pLeft < j)
		quickSortComponents( phstComponentListPos, phstComponentListValue, pLeft, j, pNumTaylorTerms );
	if (i < pRight)
		quickSortComponents( phstComponentListPos, phstComponentListValue, i, pRight, pNumTaylorTerms );

} // quickSortComponents

//
//	divideImage()
//
//	CJS: 22/05/2020
//
//	Divides one image by another of potentially a different size.
//

void divideImage( double * pImageOne, float * pImageTwo, int pSizeOne, int pSizeTwo )
{

	long int index = 0;
	for ( int j = 0; j < pSizeOne; j++ )
	{
		long int jTwo = (long int) ((double) j * (double) pSizeTwo / (double) pSizeOne) * (long int) pSizeTwo;
		for ( int i = 0; i < pSizeOne; i++, index++ )
		{
			int iTwo = (int) ((double) i * (double) pSizeTwo / (double) pSizeOne);
			pImageOne[ index ] = pImageOne[ index ] / (double) (pImageTwo[ jTwo + iTwo ]);
		}
	}

} // divideImage

void divideImage( float * pImageOne, float * pImageTwo, int pSizeOne, int pSizeTwo )
{

	long int index = 0;
	for ( int j = 0; j < pSizeOne; j++ )
	{
		long int jTwo = (long int) ((double) j * (double) pSizeTwo / (double) pSizeOne) * (long int) pSizeTwo;
		for ( int i = 0; i < pSizeOne; i++, index++ )
		{
			int iTwo = (int) ((double) i * (double) pSizeTwo / (double) pSizeOne);
			pImageOne[ index ] /= pImageTwo[ jTwo + iTwo ];
		}
	}

} // divideImage

//
//	multiplyImage()
//
//	CJS: 04/08/2020
//
//	Multiplies one image by another of potentially a different size.
//

void multiplyImage( float * pImageOne, float * pImageTwo, int pSizeOne, int pSizeTwo )
{

	long int index = 0;
	for ( int j = 0; j < pSizeOne; j++ )
	{
		long int jTwo = (long int) ((double) j * (double) pSizeTwo / (double) pSizeOne) * (long int) pSizeTwo;
		for ( int i = 0; i < pSizeOne; i++, index++ )
		{
			int iTwo = (int) ((double) i * (double) pSizeTwo / (double) pSizeOne);
			pImageOne[ index ] *= pImageTwo[ jTwo + iTwo ];
		}
	}

} // multiplyImage

//
//	getTime()
//
//	CJS: 16/11/2015
//
//	Get the elapsed time.
//

double getTime( struct timespec start, struct timespec end )
{

	return ((double)(end.tv_sec - start.tv_sec) * 1000.0) + ((double)(end.tv_nsec - start.tv_nsec) / 1000000.0);

} // getTime

//
//	getMaxMfsResidual()
//
//	CJS: 24/06/2021
//
//	Get the maximum value of the MFS computed image: R0[j]^2.A11[0] + R1[j]^2.A00[0] - 2.R0[j].R1[j].A01[0]. Sault et al 1994, A&ASS, 108, 585-594, Eq.22
//

bool getMaxMfsResidual( float * pdevR0, float * pdevR1, double * pdevMaxValue, int pWidth, int pHeight, bool * pdevMask, double pA00, double pA11, double pA01 )
{
	
	const int PIXELS_PER_THREAD = 32;
	
	bool ok = true;
	cudaError_t err;
		
	// find a suitable thread/block size for finding the maximum pixel value. each thread block will find the max
	// over N pixels (held in the above constant), and write the result to a shared memory array. a final loop will then
	// get the max from the array.
	int threads = (pWidth * pHeight / PIXELS_PER_THREAD) + 1, blocks = 1;
	setThreadBlockSize1D( &threads, &blocks );
		
	// declare global memory for writing the result of each block.
	double * devTmpResults = NULL;
	ok = reserveGPUMemory( (void **) &devTmpResults, MAX_PIXEL_DATA_AREA_SIZE * blocks * sizeof( double ), "declaring device memory for max mfs block value", __LINE__ );
		
	if (ok == true)
	{
		
		// get maximum pixel value.
		devGetMaxMfsResidualParallel<<< blocks, threads, MAX_PIXEL_DATA_AREA_SIZE * threads * sizeof( double ) >>>
					(	/* pR0 = */ pdevR0,
						/* pR1 = */ pdevR1,
						/* pWidth = */ pWidth,
						/* pHeight = */ pHeight,
						/* pCellsPerThread = */ PIXELS_PER_THREAD,
						/* pBlockMax = */ devTmpResults,
						/* pMask = */ pdevMask,
						/* pA00 = */ pA00,
						/* pA11 = */ pA11,
						/* pA01 = */ pA01 );
		err = cudaGetLastError();
		if (err != cudaSuccess)
		{
			printf( "error getting max Mfs pixel value (%s)\n", cudaGetErrorString( err ) );
			ok = false;
		}
		
	}
	
	if (ok == true)
	{
			
		// get maximum pixel value from the block list.
		devGetMaxValue<<< 1, 1 >>>(	/* pArray = */ devTmpResults,
						/* pMaxValue = */ pdevMaxValue,
						/* pUseAbsolute = */ false,
						/* pElements = */ blocks );
		err = cudaGetLastError();
		if (err != cudaSuccess)
		{
			printf( "error getting final max value (%s)\n", cudaGetErrorString( err ) );
			ok = false;
		}
			
	}
		
	// free memory.
	if (devTmpResults != NULL)
		cudaFree( (void *) devTmpResults );
	
	// return success flag.
	return ok;
	
} // getMaxMfsResidual

//
//	gridVisibilities()
//
//	CJS: 10/08/2015
//
//	Produce an image of gridded visibilities by convolving the complex visibilities with the kernel function.
//
//	The grid positions, and kernel indexes, have already been calculated for each visibility and stored on the device. Only the
//	complex visibilities, the kernel size and the kernel indexes assigned to each visibility are uploaded to the device here,
//	because these values will change depending upon whether we are gridding the observed data or gridding the uv coverage to
//	generate the psf.
//

void gridVisibilities
			(
			cufftComplex ** pdevGrid,			// data area (device) holding the grid
			int pStageID,					// the CPU stage ID
			int pBatchID,					// the GPU batch ID
			cufftComplex ** pdevVisibility,		// data area (device) holding the visibilities
			int ** pdevKernelIndex,			// an array of kernel indexes assigned to each visibility
			VectorI ** pdevGridPositions,			// a list of integer grid positions for each visibility
			float ** pdevWeight,				// a list of weights for each visibility
			int pSize,					// the size of the image
			int pNumGPUs,					// the number of GPUs to use for gridding,
			int pStokesTo,					// the stokes product (used by a-projection only).
			int pStokesFrom,				// the stokes leakage term
			KernelCache & phstKernelCache			// the kernel cache; indexes are pb-correction channel, wPlane
			)
{
	
	cudaError_t err;

struct timespec otherStart, otherEnd;

	err = cudaGetLastError();
	if (err != cudaSuccess)
		printf( "unknown CUDA error in gridVisibilities() [i] (%s)\n", cudaGetErrorString( err ) );

	// get some gridding details from the kernel cache.
	int oversample = phstKernelCache.oversample;
	bool wProjection = phstKernelCache.wProjection;
	int wPlanes = phstKernelCache.wPlanes;
	int pbChannels = phstKernelCache.pbChannels;
	griddegrid gridDegrid = phstKernelCache.gridDegrid;

	// find the number of kernels sets in this batch.
	int numKernelSets = 0;
	for ( int wPlane = 0; wPlane < wPlanes; wPlane++ )
		for ( int pbChannel = 0; pbChannel < pbChannels; pbChannel++ )
		{
			bool foundVis = false;
			for ( int gpu = 0; gpu < pNumGPUs; gpu++ )
				if (phstKernelCache( pbChannel, pStokesTo, pStokesFrom, wPlane ).visibilities[ pStageID ][ pBatchID ][ gpu ] > 0)
					foundVis = true;
			if (foundVis == true)
				numKernelSets++;
		}
	printf( "                found visibilities for %i kernel set(s)\n", numKernelSets );

	// declare array of device kernels.
	cufftComplex ** devKernel = (cufftComplex **) malloc( pNumGPUs * sizeof( cufftComplex * ) );
	for ( int gpu = 0; gpu < pNumGPUs; gpu++ )
		devKernel[ gpu ] = NULL;
		
	// maintain pointers to the next visibilities for each GPU.
	int * hstNextVisibility = (int *) malloc( pNumGPUs * sizeof( int ) );
	for ( int gpu = 0; gpu < pNumGPUs; gpu++ )
		hstNextVisibility[ gpu ] = 0;

	// initialise the GPU to 0. we will increment this number for each kernel set so that each kernel set is passed to a different gpu.
	int cudaDeviceIndex = 0;

	// grid the visibilities one kernel set at a time.
	for ( int wPlane = 0; wPlane < wPlanes; wPlane++ )
		for ( int pbChannel = 0; pbChannel < pbChannels; pbChannel++ )
		{
						
			// get kernel set from cache.
			KernelSet & hstKernelSet = phstKernelCache( pbChannel, pStokesTo, pStokesFrom, wPlane );

			int firstGPU = cudaDeviceIndex;
			int latestGPU = cudaDeviceIndex;
			do
			{

				int numVisibilities = hstKernelSet.visibilities[ pStageID ][ pBatchID ][ cudaDeviceIndex ];
				if (numVisibilities > 0)
				{

struct timespec setupStart, setupEnd;
clock_gettime( CLOCK_REALTIME, &setupStart );

					// set the cuda device, and wait for whatever is running to finish.
					if (pNumGPUs > 1)
						cudaSetDevice( _param->GPU[ cudaDeviceIndex ] );

					// free the kernels on this device if they exist.
					if (devKernel[ cudaDeviceIndex ] != NULL)
					{
						cudaFree( (void *) devKernel[ cudaDeviceIndex ] );
						devKernel[ cudaDeviceIndex ] = NULL;
					}

					if (gridDegrid == GRID)
						printf( "                        gridding " );
					else
						printf( "                        degridding " );
					printf( "%i visibilities", numVisibilities );
					if (wProjection == true || pbChannels > 1)
						printf( " for " );
					if (wProjection == true)
						printf( "w-plane %i ", wPlane );
					if (wProjection == true && pbChannels > 1)
						printf( "and " );
					if (pbChannels > 1)
						printf( "pb-correction channel %i", pbChannel );
					if (pNumGPUs > 1)
						printf( " on GPU %i", _param->GPU[ cudaDeviceIndex ] );

clock_gettime( CLOCK_REALTIME, &setupEnd );
_setup += getTime( setupStart, setupEnd );

struct timespec kernelStart, kernelEnd;
clock_gettime( CLOCK_REALTIME, &kernelStart );
					
					// get the kernel size from the cache.
					int kernelSize = hstKernelSet.kernelSize;
					int supportSize = (kernelSize - 1) / 2;
					
					// copy the kernel to the device.
					reserveGPUMemory( (void **) &devKernel[ cudaDeviceIndex ],
							kernelSize * kernelSize * oversample * oversample * sizeof( cufftComplex ),
							"reserving GPU memory for the kernel from the cache", __LINE__ );
					moveHostToDevice( (void *) devKernel[ cudaDeviceIndex ], (void *) hstKernelSet.kernel, 
								hstKernelSet.kernelSize * hstKernelSet.kernelSize * hstKernelSet.oversample *
								hstKernelSet.oversample * sizeof( cufftComplex ),
							"copying kernel from cache to device", __LINE__ );

					printf( "\n" );

clock_gettime( CLOCK_REALTIME, &kernelEnd );
_generateKernel += getTime( kernelStart, kernelEnd );

					// ensure we don't report old errors.
					err = cudaGetLastError();
					if (err != cudaSuccess)
						printf( "unknown CUDA error in gridVisibilities() [ii] (%s)\n", cudaGetErrorString( err ) );

					if (gridDegrid == GRID)
					{

						// define the block/thread dimensions.
						setThreadBlockSizeForGridding(	/* pThreadsX = */ kernelSize,
											/* pThreadsY = */ kernelSize,
											/* pItems = */ numVisibilities );
				
						// work out how much shared memory we need to store items-per-block visibilities.
						int sharedMemSize = _itemsPerBlock * (sizeof( cufftComplex ) + sizeof( VectorI ) + sizeof( int ) + sizeof( float ));

						// do the 2-d convolution loop on the device.
						devGridVisibilities<<< _gridSize2D, _blockSize2D, sharedMemSize >>>
								(	/* pGrid = */ pdevGrid[ cudaDeviceIndex ],
									/* pVisibility = */ &pdevVisibility[ cudaDeviceIndex ][ hstNextVisibility[ cudaDeviceIndex ] ],
									/* pVisibilitiesPerBlock = */ _itemsPerBlock,
									/* pBlocksPerVisibility = */ _blocksPerItem,
									/* pGridPosition = */ &pdevGridPositions[ cudaDeviceIndex ][ hstNextVisibility[ cudaDeviceIndex ] ],
									/* pKernel = */ devKernel[ cudaDeviceIndex ],
									/* pKernelIndex = */ (pdevKernelIndex != NULL ?
											&pdevKernelIndex[ cudaDeviceIndex ][ hstNextVisibility[ cudaDeviceIndex ] ] : NULL),
									/* pWeight = */ (pdevWeight != NULL ?
											&pdevWeight[ cudaDeviceIndex ][ hstNextVisibility[ cudaDeviceIndex ] ] : NULL),
									/* pNumVisibilities = */ numVisibilities,
									/* pSize = */ pSize,
									/* pComplex = */ true,
									/* pSupport = */ supportSize );

					}
					else
					{

						// define the block/thread dimensions.
						setThreadBlockSizeForDegridding(	/* pKernelSizeX = */ kernelSize,
											/* pKernelSizeY = */ kernelSize,
											/* pNumVisibilities = */ numVisibilities );

						// do the degridding on the device.
						devDegridVisibilities<<< _gridSize3D, _maxThreadsPerBlock >>>
								(	/* pGrid = */ pdevGrid[ cudaDeviceIndex ],
									/* pVisibility = */ &pdevVisibility[ cudaDeviceIndex ][ hstNextVisibility[ cudaDeviceIndex ] ],
									/* pGridPosition = */ &pdevGridPositions[ cudaDeviceIndex ][ hstNextVisibility[ cudaDeviceIndex ] ],
									/* pKernel = */ devKernel[ cudaDeviceIndex ],
									/* pKernelIndex = */ (pdevKernelIndex != NULL ?
											&pdevKernelIndex[ cudaDeviceIndex ][ hstNextVisibility[ cudaDeviceIndex ] ] : NULL),
									/* pNumVisibilities = */ numVisibilities,
									/* pSize = */ pSize,
									/* pVisibilitiesPerPage = */ _maxThreadsPerBlock,
									/* pSupport = */ supportSize );

					}

					// use the next lot of visibilities.
					hstNextVisibility[ cudaDeviceIndex ] += numVisibilities;
					latestGPU = cudaDeviceIndex;

				} // numVisibilities > 0

				// check if there's anything to grid for the next GPU.
				cudaDeviceIndex++;
				if (cudaDeviceIndex == pNumGPUs)
					cudaDeviceIndex = 0;

			} while (cudaDeviceIndex != firstGPU);

			// for the next kernel set we'll want to start off using the next available GPU.
			cudaDeviceIndex = latestGPU + 1;
			if (cudaDeviceIndex == pNumGPUs)
				cudaDeviceIndex = 0;

		} // LOOP: wPlane, pbChannel

clock_gettime( CLOCK_REALTIME, &otherStart );

	// clear the kernels if they exist.
	for ( int gpu = 0; gpu < pNumGPUs; gpu++ )
	{
		if (devKernel[ gpu ] != NULL)
		{
			if (pNumGPUs > 1)
				cudaSetDevice( _param->GPU[ gpu ] );
			cudaFree( (void *) devKernel[ gpu ] );
			devKernel[ gpu ] = NULL;
		}
	} // LOOP: gpu

clock_gettime( CLOCK_REALTIME, &otherEnd );
_otherGridding += getTime( otherStart, otherEnd );

	// reset the GPU to the first device.
	if (pNumGPUs > 1)
		cudaSetDevice( _param->GPU[ 0 ] );
		
	printf( "\n" );

	// free memory.
	if (hstNextVisibility != NULL)
		free( (void *) hstNextVisibility );
	if (devKernel != NULL)
		free( (void *) devKernel );
	
} // gridVisibilities

//
//	gridComponents()
//
//	CJS: 22/05/2020
//
//	Grids a list of clean components to the clean image.
//

void gridComponents(	float * pdevGrid,				// data area (device) holding the grid
			double * pdevComponentValue,			// data area (device) holding the visibilities
			int phstSupportSize,				// kernel and gridding parameters
			float * pdevKernel,				// the kernel array.
			VectorI * pdevGridPositions,			// a list of integer grid positions for each visibility
			int pComponents,				// the number of components to grid.
			int pSize )					// the size of the image
{
	
	cudaError_t err;
		
	err = cudaGetLastError();
	if (err != cudaSuccess)
		printf( "unknown CUDA error in gridComponents() [i] (%s)\n", cudaGetErrorString( err ) );

	// calculate kernel size.
	int hstKernelSize = (phstSupportSize * 2) + 1;

	// generate the texture maps for these kernel sets.
	if (pComponents > 0)
	{

		// ensure we don't report old errors.
		err = cudaGetLastError();
		if (err != cudaSuccess)
			printf( "unknown CUDA error in gridComponents() [ii] (%s)\n", cudaGetErrorString( err ) );

		// define the block/thread dimensions.
		setThreadBlockSizeForGridding(	/* pThreadsX = */ hstKernelSize,
							/* pThreadsY = */ hstKernelSize,
							/* pItems = */ pComponents );

		// work out how much shared memory we need to store items-per-block visibilities.
		int sharedMemSize = _itemsPerBlock * (sizeof( cufftComplex ) + sizeof( VectorI ) + sizeof( int ) + sizeof( float ));

		// do the 2-d convolution loop on the device.
		devGridVisibilities<<< _gridSize2D, _blockSize2D, sharedMemSize >>>
				(	/* pGrid = */ (cufftComplex *) pdevGrid,
					/* pVisibility = */ (cufftComplex *) pdevComponentValue,
					/* pVisibilitiesPerBlock = */ _itemsPerBlock,
					/* pBlocksPerVisibility = */ _blocksPerItem,
					/* pGridPosition = */ pdevGridPositions,
					/* pKernel = */ (cufftComplex *) pdevKernel,
					/* pKernelIndex = */ NULL,
					/* pWeight = */ NULL,
					/* pNumVisibilities = */ pComponents,
					/* pSize = */ pSize,
					/* pComplex = */ false,
					/* pSupport = */ phstSupportSize );

		err = cudaGetLastError();
		if (err != cudaSuccess)
			printf( "error gridding components on device (%s)\n", cudaGetErrorString( err ) );

	}

} // gridComponents

//
//	displaySupportSizes()
//
//	CJS: 25/02/2022
//
//	Display the support sizes being used with gridding.
//

void displaySupportSizes
			(
			KernelCache & pKernelCache,
			int pStokes
			)
{

	printf( "	applying kernel cache with %i channel(s)", pKernelCache.pbChannels );
	if (pKernelCache.wProjection == true)
		printf( " and %i w-planes", pKernelCache.wPlanes );
	printf( ". support sizes are:\n\n" );

	for ( int stokesTo = 0; stokesTo < pKernelCache.stokesProducts; stokesTo++ )
		for ( int stokesFrom = 0; stokesFrom < pKernelCache.stokesProducts; stokesFrom++ )
		{
		
			bool displaySupport = false;
			if (pKernelCache.stokesProducts == 1 && pStokes == -1)
			{
				printf( "		Stokes " );
				if (_param->Stokes == STOKES_I || _param->Stokes == STOKES_ALL) printf( "I: " );
				else if (_param->Stokes == STOKES_Q) printf( "Q: " );
				else if (_param->Stokes == STOKES_U) printf( "U: " );
				else if (_param->Stokes == STOKES_V) printf( "V: " );
				displaySupport = true;
			}
			else if (pKernelCache.stokesProducts == 1 && pStokes > -1)
			{
				printf( "		Stokes " );
				if (pStokes == STOKES_I) printf( "I: " );
				else if (pStokes == STOKES_Q) printf( "Q: " );
				else if (pStokes == STOKES_U) printf( "U: " );
				else if (pStokes == STOKES_V) printf( "V: " );
				displaySupport = true;
			}
			else if (pKernelCache.stokesProducts > 1 && stokesTo == stokesFrom && (stokesTo == pStokes || pStokes == -1))
			{
				printf( "		Stokes " );
				if (stokesTo == STOKES_I) printf( "I: " );
				else if (stokesTo == STOKES_Q) printf( "Q: " );
				else if (stokesTo == STOKES_U) printf( "U: " );
				else if (stokesTo == STOKES_V) printf( "V: " );
				displaySupport = true;
			}
			else if (pKernelCache.stokesProducts > 1 && stokesTo != stokesFrom && pKernelCache.stokesFlag[ (stokesTo * 4) + stokesFrom ] == true && (stokesTo == pStokes || pStokes == -1))
			{
				printf( "		Stokes " );
				if (stokesFrom == STOKES_I) printf( "I" );
				else if (stokesFrom == STOKES_Q) printf( "Q" );
				else if (stokesFrom == STOKES_U) printf( "U" );
				else if (stokesFrom == STOKES_V) printf( "V" );
				printf( " into Stokes " );
				if (stokesTo == STOKES_I) printf( "I" );
				else if (stokesTo == STOKES_Q) printf( "Q" );
				else if (stokesTo == STOKES_U) printf( "U" );
				else if (stokesTo == STOKES_V) printf( "V" );
				if (pKernelCache.gridDegrid == GRID)
					printf( " leakage correction: " );
				else
					printf( " leakage term: " );
				displaySupport = true;
			}
			
			// display support sizes unless this is an ignored leakage term.
			if (displaySupport == true)
			{
				printf( "s = [" );
				for ( int wPlane = 0; wPlane < pKernelCache.wPlanes; wPlane++ )
					for ( int pbChannel = 0; pbChannel < pKernelCache.pbChannels; pbChannel++ )
					{
						printf( "%i", pKernelCache( pbChannel, stokesTo, stokesFrom, wPlane ).supportSize );
						if (wPlane < pKernelCache.wPlanes - 1 || pbChannel < pKernelCache.pbChannels - 1)
							printf( ", " );
					}
				printf( "]\n\n" );
			}
			
		} // LOOP: stokesTo, stokesFrom

} // displaySupportSizes

void displaySupportSizes
			(
			KernelCache & pKernelCache
			)
{

	displaySupportSizes(	/* pKernelCache = */ pKernelCache,
				/* pStokes = */ -1 );

} // displaySupportSizes

//
//	generateImageOfConvolutionFunction()
//
//	CJS: 18/01/2016
//
//	Generate the deconvolution function by gridding a single source at u = 0, v = 0, and FFT'ing.
//

void generateImageOfConvolutionFunction( char * pDeconvolutionFilename )
{

	// create a new data object to hold a single visibility.
	Data * tmpData = new Data(	/* pTaylorTerms = */ 1,
					/* pMosaicID = */ 0,
					/* pWProjection = */ false,
					/* pAProjection = */ false,
					/* pWPlanes = */ 1,
					/* pPBChannels = */ 1,
					/* pCacheData = */ false,
					/* pStokes = */ STOKES_I,
					/* pStokesImages = */ 1 );

	// set up data.
	tmpData->Stages = 1;
	tmpData->Batches = (int *) realloc( tmpData->Batches, sizeof( int ) );
	tmpData->Batches[ /* STAGE = */ 0 ] = 1;
	tmpData->NumVisibilities = (long *) realloc( tmpData->NumVisibilities, sizeof( long ) );
	tmpData->NumVisibilities[ /* STAGE = */ 0 ] = 1;
	tmpData->AverageWeight = (double *) malloc( sizeof( double ) );
	tmpData->AverageWeight[ 0 ] = 1.0;

	// create the deconvolution image, and clear it.
	cufftComplex * devDeconvolutionImageGrid = NULL;
	reserveGPUMemory( (void **) &devDeconvolutionImageGrid, _param->PsfSize * _param->PsfSize * sizeof( cufftComplex ),
				"declaring memory for deconvolution image", __LINE__ );
	zeroGPUMemory( (void *) devDeconvolutionImageGrid, _param->PsfSize * _param->PsfSize * sizeof( cufftComplex ), "zeroing the grid on the device", __LINE__ );

	// create space for a single visibility on the device.
	cufftComplex * tmpdevVisibility;
	reserveGPUMemory( (void **) &tmpdevVisibility, 1 * sizeof( cufftComplex ), "declaring device memory for visibility", __LINE__ );
	{
		cufftComplex tmphstVisibility;
		tmphstVisibility.x = 1;
		tmphstVisibility.y = 0;
		moveHostToDevice( (void *) tmpdevVisibility, (void *) &tmphstVisibility, sizeof( cufftComplex ), "copying visibility to device", __LINE__ );
	}

	// create space for a single weight on the device.
	float * tmpdevWeight = NULL;
	reserveGPUMemory( (void **) &tmpdevWeight, 1 * sizeof( float ), "declaring device memory for weights", __LINE__ );

	float tmpWeight = 1.0;
	moveHostToDevice( (void *) tmpdevWeight, (void *) &tmpWeight, sizeof( float ), "copying weight to device", __LINE__ );

	// create a single int pointer to use as a kernel pointer.
	tmpData->KernelIndex = (int *) malloc( sizeof( int ) );
	tmpData->KernelIndex[ /* ITEM = */ 0 ] = 0;
	int * tmpdevKernelIndex = NULL;
	reserveGPUMemory( (void **) &tmpdevKernelIndex, 1 * sizeof( int ), "declaring device memory for kernel index", __LINE__ );
	moveHostToDevice( (void *) tmpdevKernelIndex, (void *) &tmpData->KernelIndex[ /* ITEM = */ 0 ], sizeof( int ), "copying kernel index to device", __LINE__ );

	// create a single vector to hold the grid positions on the device.
	tmpData->GridPosition = (VectorI *) malloc( sizeof( VectorI ) );
	tmpData->GridPosition[ /* ITEM = */ 0 ].u = (_param->PsfSize / 2.0);
	tmpData->GridPosition[ /* ITEM = */ 0 ].v = (_param->PsfSize / 2.0);
	tmpData->GridPosition[ /* ITEM = */ 0 ].w = 0;
	VectorI * tmpdevGridPositions = NULL;
	reserveGPUMemory( (void **) &tmpdevGridPositions, 1 * sizeof( VectorI ), "declaring device memory for grid positions", __LINE__ );
	moveHostToDevice( (void *) tmpdevGridPositions, (void *) &tmpData->GridPosition[ /* ITEM = */ 0 ], sizeof( VectorI ),
														"copying grid positions to device", __LINE__ );

	// work out how the visibilities are going to be split between GPUs and GPU batches.
	_psKernelCache.CountVisibilities(	/* pData = */ tmpData,
						/* pMaxBatchSize = */ _param->PREFERRED_VISIBILITY_BATCH_SIZE,
						/* pNumGPUs = */ _param->NumGPUs );
						
	printf( "gridding visibilities for anti-aliasing correction.....\n\n" );

	// print the support sizes.
	displaySupportSizes( /* pKernelCache = */ _psKernelCache );

	// generate the deconvolution function by gridding a single visibility without w-projection
	gridVisibilities(	/* pdevGrid = */ &devDeconvolutionImageGrid,
				/* pStageID = */ 0,
				/* pBatchID = */ 0,
				/* pdevVisibility = */ &tmpdevVisibility,
				/* pdevKernelIndex = */ &tmpdevKernelIndex,
				/* pdevGridPositions = */ &tmpdevGridPositions,
				/* pdevWeight = */ &tmpdevWeight,
				/* pSize = */ _param->PsfSize,
				/* pNumGPUs = */ 1,
				/* pStokesTo = */ STOKES_I,
				/* pStokesFrom = */ STOKES_I,
				/* phstKernelCache = */ _psKernelCache );

	// FFT the gridded data to get the deconvolution map.
	performFFT(	/* pdevGrid = */ &devDeconvolutionImageGrid,
			/* pSize = */ _param->PsfSize,
			/* pFFTDirection = */ INVERSE,
			/* pFFTPlan = */ -1,
			/* pFFTType = */ C2F,
			/* pResizeArray = */ true );

	// create memory for the deconvolution image, and copy the image from the device.
	float * hstDeconvolutionImage = (float *) malloc( _param->PsfSize * _param->PsfSize * sizeof( float ) );
	moveDeviceToHost( (void *) hstDeconvolutionImage, (void *) devDeconvolutionImageGrid, _param->PsfSize * _param->PsfSize * sizeof( float ),
					"copying deconvolution image from device", __LINE__ );

	// re-cast the deconvolution image pointer from a complex to a float.
	_devImageDomainPSFunction = (float *) devDeconvolutionImageGrid;

	// save the deconvolution image.
	_hstCasacoreInterface.WriteCasaImage(	/* pFilename = */ pDeconvolutionFilename,
						/* pWidth = */ _param->PsfSize,
						/* pHeight = */ _param->PsfSize,
						/* pRA = */ _param->OutputRA,
						/* pDec = */ _param->OutputDEC,
						/* pPixelSize = */ _param->CellSize * (double) _param->ImageSize / (double) _param->PsfSize, 
						/* pImage = */ hstDeconvolutionImage,
						/* pFrequency = */ CONST_C / _hstData[ /* MOSAIC_COMPONENT = */ 0 ]->AverageWavelength,
						/* pMask = */ NULL,
						/* pDirectionType = */ CasacoreInterface::J2000,
						/* pStokesImages = */ 1 );

	// clean up memory.
	if (tmpdevVisibility != NULL)
		cudaFree( (void *) tmpdevVisibility );
	if (tmpdevKernelIndex != NULL)
		cudaFree( (void *) tmpdevKernelIndex );
	if (tmpdevGridPositions != NULL)
		cudaFree( (void *) tmpdevGridPositions );
	if (tmpdevWeight != NULL)
		cudaFree( (void *) tmpdevWeight );
	if (hstDeconvolutionImage != NULL)
		free( (void *) hstDeconvolutionImage );

	if (tmpData != NULL)
		delete tmpData;

	printf( "\n" );
	
} // generateImageOfConvolutionFunction

//
//	extractFromMosaic()
//
//	CJS: 19/03/2019
//
//	Extracts an image from the mosaic.
//

void extractFromMosaic( float * pdevImage, float * phstMosaic, bool * phstMask, int pImageID )
{

	// create memory for mosaic, mask and beam on the device.
	bool * devMask = NULL;
	float * devBeam = NULL;

	// copy the mask into device memory.
	if (phstMask != NULL)
	{
		reserveGPUMemory( (void **) &devMask, (long int) _param->ImageSize * (long int) _param->ImageSize * (long int) sizeof( bool ),
					"creating device memory for the image mask", __LINE__ );
		cudaMemcpy( devMask, phstMask, (long int) _param->ImageSize * (long int) _param->ImageSize * (long int) sizeof( bool ), cudaMemcpyHostToDevice );
	}

	// upload the primary beam to the device.
	reserveGPUMemory( (void **) &devBeam, _param->BeamSize * _param->BeamSize * sizeof( float ), "creating device memory for the primary beam", __LINE__ );
	cudaMemcpy( devBeam, _hstData[ pImageID ]->PrimaryBeamInFrame, _param->BeamSize * _param->BeamSize * sizeof( float ), cudaMemcpyHostToDevice );

	// create a reprojection object.
	Reprojection imagePlaneReprojection;

	// set up pixel vector and CD matrix.
	Reprojection::rpVectD tmpPixel = { /* x = */ _param->ImageSize / 2, /* y = */ _param->ImageSize / 2, /* z = */ 0 };
	Reprojection::rpMatr2x2 tmpCD = { /* a11 = */ -sin( rad( _param->CellSize / 3600.0 ) ), /* a12 = */ 0.0, /* a21 = */ 0.0, /* a22 = */ sin( rad( _param->CellSize / 3600.0 ) ) };

	// build input and output size.
	Reprojection::rpVectI size = { /* x = */ _param->ImageSize, /* y = */ _param->ImageSize };

	// build beam size.
	Reprojection::rpVectI beamSize = { /* x = */ _param->BeamSize, /* y = */ _param->BeamSize };

	// build in coordinate system.
	Reprojection::rpCoordSys inCoordSystem;
	inCoordSystem.crVAL.x = _param->OutputRA;
	inCoordSystem.crVAL.y = _param->OutputDEC;
	inCoordSystem.crPIX = tmpPixel;
	inCoordSystem.cd = tmpCD;
	inCoordSystem.epoch = Reprojection::EPOCH_J2000;

	// build out coordinate system.
	Reprojection::rpCoordSys outCoordSystem;
	outCoordSystem.crPIX = tmpPixel;
	outCoordSystem.cd = tmpCD;
	outCoordSystem.epoch = Reprojection::EPOCH_J2000;

	// create the device memory required by the reprojection code.
	imagePlaneReprojection.CreateDeviceMemory( size );

	// clear the grid.
	zeroGPUMemory( (void *) pdevImage, (long int) _param->ImageSize * (long int) _param->ImageSize * (long int) sizeof( float ), "zeroing grid on the device", __LINE__ );

	// copy the mosaic to the device.
	float * devMosaic = NULL;
	reserveGPUMemory( (void **) &devMosaic, _param->ImageSize * _param->ImageSize * sizeof( float ), "reserving device memory for the mosaic", __LINE__ );
	moveHostToDevice( (void *) devMosaic, (void *) phstMosaic, _param->ImageSize * _param->ImageSize * sizeof( float ), "copying mosaic to the device", __LINE__ );

	// set out coordinate system RA and DEC.
	outCoordSystem.crVAL.x = _hstData[ pImageID ]->ImagePlaneRA;
	outCoordSystem.crVAL.y = _hstData[ pImageID ]->ImagePlaneDEC;

	// reproject this image in order to construct this part of the mosaic.
	imagePlaneReprojection.ReprojectImage(	/* pdevInImage = */ devMosaic,
							/* pdevOutImage = */ pdevImage,
							/* pInCoordinateSystem = */ inCoordSystem,
							/* pOutCoordinateSystem = */ outCoordSystem,
							/* pInSize = */ size,
							/* pOutSize = */ size,
							/* pdevInMask = */ devMask,
							/* pdevBeamIn = */ NULL,
							/* pdevBeamOut = */ devBeam,
							/* pBeamSize = */ beamSize,
							/* pProjectionDirection = */ Reprojection::INPUT_TO_OUTPUT,
							/* pAProjection = */ _param->AProjection,
							/* pVerbose = */ false );

	// free memory.
	if (devMask != NULL)
		cudaFree( (void *) devMask );
	if (devBeam != NULL)
		cudaFree( (void *) devBeam );
	if (devMosaic != NULL)
		cudaFree( (void *) devMosaic );

} // extractFromMosaic

//
//	addToMosaic()
//
//	CJS: 09/07/2021
//
//	Adds an image to a mosaic.
//

void addToMosaic( float * phstMosaic, float * pdevImage, int pImageID )
{

	// create memory for mosaic, pixel weights, mask and primary beam on the device.
	float * devMosaic = NULL;
	float * devBeam = NULL;
	reserveGPUMemory( (void **) &devMosaic, _param->ImageSize * _param->ImageSize * sizeof( float ), "creating device memory for the mosaic", __LINE__ );
	reserveGPUMemory( (void **) &devBeam, _param->BeamSize * _param->BeamSize * sizeof( float ), "creating device memory for the primary beam", __LINE__ );

	// store the image and primary beam on the device.
	moveHostToDevice( (void *) devMosaic, (void *) phstMosaic, _param->ImageSize * _param->ImageSize * sizeof( float ), "copying mosaic image to device", __LINE__ );
	moveHostToDevice( (void *) devBeam, (void *) _hstData[ pImageID ]->PrimaryBeamInFrame, _param->BeamSize * _param->BeamSize * sizeof( float ),
					"copying primary beam to device", __LINE__ );

	// create a reprojection object.
	Reprojection imagePlaneReprojection;

	// set up pixel vector and CD matrix.
	Reprojection::rpVectD tmpPixel = { /* x = */ _param->ImageSize / 2, /* y = */ _param->ImageSize / 2, /*z = */ 0 };
	Reprojection::rpMatr2x2 tmpCD = { /* a11 = */ -sin( rad( _param->CellSize / 3600.0 ) ), /* a12 = */ 0.0, /* a21 = */ 0.0, /* a22 = */ sin( rad( _param->CellSize / 3600.0 ) ) };

	// build input and output size, and beam size.
	Reprojection::rpVectI size = { /* x = */ _param->ImageSize, /* y = */ _param->ImageSize };
	Reprojection::rpVectI beamSize = { /* x = */ _param->BeamSize, /* y = */ _param->BeamSize };

	// build in coordinate system.
	Reprojection::rpCoordSys inCoordSystem;
	inCoordSystem.crVAL.x = _hstData[ pImageID ]->ImagePlaneRA;
	inCoordSystem.crVAL.y = _hstData[ pImageID ]->ImagePlaneDEC;;
	inCoordSystem.crPIX = tmpPixel;
	inCoordSystem.cd = tmpCD;
	inCoordSystem.epoch = Reprojection::EPOCH_J2000;

	// build out coordinate system.
	Reprojection::rpCoordSys outCoordSystem;
	outCoordSystem.crVAL.x = _param->OutputRA;
	outCoordSystem.crVAL.y = _param->OutputDEC;
	outCoordSystem.crPIX = tmpPixel;
	outCoordSystem.cd = tmpCD;
	outCoordSystem.epoch = Reprojection::EPOCH_J2000;

	// create the device memory for the reprojection code.
	imagePlaneReprojection.CreateDeviceMemory( size );

	// reproject this image in order to construct this part of the mosaic.
	imagePlaneReprojection.ReprojectImage(	/* pdevInImage = */ pdevImage,
						/* pdevOutImage = */ devMosaic,
						/* pInCoordinateSystem = */ inCoordSystem,
						/* pOutCoordinateSystem = */ outCoordSystem,
						/* pInSize = */ size,
						/* pOutSize = */ size,
						/* pdevInMask = */ NULL,
						/* pdevBeamIn = */ devBeam,
						/* pdevBeamOut = */ NULL,
						/* pBeamSize = */ beamSize,
						/* pProjectionDirection = */ Reprojection::OUTPUT_TO_INPUT,
						/* pAProjection = */ _param->AProjection,
						/* pVerbose = */ false );

	// store the image on the host.
	moveDeviceToHost( (void *) phstMosaic, (void *) devMosaic, _param->ImageSize * _param->ImageSize * sizeof( float ), "copying mosaic image to host", __LINE__ );

	// free memory.
	if (devMosaic != NULL)
		cudaFree( (void *) devMosaic );
	if (devBeam != NULL)
		cudaFree( (void *) devBeam );

} // addToMosaic

//
//	hogbomClean()
//
//	CJS: 05/11/2015
//
//	Perform a Hogbom clean on our dirty image.
//

void hogbomClean( int * pMinorCycle, double pHogbomLimit, float ** phstDirtyBeam, float * pdevDirtyImage, VectorI ** phstComponentListPos,
			double *** phstComponentListValue, int * pComponentListItems, float * phstPrimaryBeam )
{
	
	cudaError_t err;
		
	printf( "\n                minor cycles: " );
	fflush( stdout );
	
	double ** devMaxValue = (double **) malloc( _param->NumStokesImages * sizeof( double * ) );
	for ( int stokes = 0; stokes < _param->NumStokesImages; stokes++ )
		devMaxValue[ stokes ] = NULL;
		
	// reserve host memory for the maximum pixel value.
	double * hstMaxValue = (double *) malloc( MAX_PIXEL_DATA_AREA_SIZE * sizeof( double ) );
	hstMaxValue[ MAX_PIXEL_VALUE ] = 0.0;
	hstMaxValue[ MAX_PIXEL_X ] = _param->ImageSize / 2;
	hstMaxValue[ MAX_PIXEL_Y ] = _param->ImageSize / 2;
	hstMaxValue[ MAX_PIXEL_IMAGE ] = 0.0;

	// keep a record of the minimum value. if it goes up by a certain factor then we need to stop cleaning.
	double minimumValue = -1.0;
	
	cudaDeviceSynchronize();

	err = cudaGetLastError();
	if (err != cudaSuccess)
		printf( "unknown CUDA error in hogbomClean() [i] (%s)\n", cudaGetErrorString( err ) );

	// create and upload the normalisation pattern to the device.
	float * devNormalisationPattern = NULL;
	if (_param->UseMosaicing == true)
	{
		reserveGPUMemory( (void **) &devNormalisationPattern, _param->BeamSize * _param->BeamSize * sizeof( float ),
						"declaring device memory for normalisation pattern", __LINE__ );
		moveHostToDevice( (void *) devNormalisationPattern, (void *) _hstNormalisationPattern, _param->BeamSize * _param->BeamSize * sizeof( float ),
						"copying normalisation pattern to the device", __LINE__ );
	}

	// create and upload the primary beam pattern to the device.
	float * devPrimaryBeamPattern = NULL;
	if (_param->UseMosaicing == true || _param->AProjection == true)
	{
		reserveGPUMemory( (void **) &devPrimaryBeamPattern, _param->BeamSize * _param->BeamSize * sizeof( float ),
						"declaring device memory for primary beam pattern", __LINE__ );
		moveHostToDevice( (void *) devPrimaryBeamPattern, (void *) _hstPrimaryBeamPattern, _param->BeamSize * _param->BeamSize * sizeof( float ),
						"copying primary beam pattern to the device", __LINE__ );
	}
	
	// create memory for psf cache.
	const int NUM_PSFS_IN_CACHE = 40;
	const int MAX_DISTANCE_IN_PIXELS = pow( 400, 2 );
	
	int * hstPsfX = NULL;
	int * hstPsfY = NULL;
	float ** hstPsfCache = NULL;
	if ((_param->UseMosaicing == true || _param->AProjection == true))
	{
		hstPsfX = (int *) malloc( NUM_PSFS_IN_CACHE * sizeof( int ) );
		hstPsfY = (int *) malloc( NUM_PSFS_IN_CACHE * sizeof( int ) );
		hstPsfCache = (float **) malloc( NUM_PSFS_IN_CACHE * sizeof( float * ) );
		for ( int i = 0; i < NUM_PSFS_IN_CACHE; i++ )
			hstPsfCache[ i ] = NULL;
	}
	
	int maxMinorCyclesBeforeFFT = -1;
	if (CYCLE_NITER > 0)
		maxMinorCyclesBeforeFFT = *pMinorCycle + CYCLE_NITER;

	// loop over each minor cycle.
bool tmp = false;
	while (*pMinorCycle < _param->MinorCycles && tmp == false)
	{
//tmp = true; // cjs-mod
		
		printf( "." );
		fflush( stdout );
		
		// get maximum pixel value.
		for ( int stokes = 0; stokes < _param->NumStokesImages; stokes++ )
			if (devMaxValue[ stokes ] == NULL)
			{
			
				reserveGPUMemory( (void **) &devMaxValue[ stokes ], MAX_PIXEL_DATA_AREA_SIZE * sizeof( double ),
							"declaring device memory for max pixel value", __LINE__ );
				getMaxValue(	/* pdevImage = */ pdevDirtyImage + (stokes * _param->ImageSize * _param->ImageSize),
						/* pdevMaxValue = */ devMaxValue[ stokes ],
						/* pWidth = */ _param->ImageSize,
						/* pHeight = */ _param->ImageSize,
						/* pUseAbsolute = */ true,
						/* pdevMask = */ NULL,
						/* pNumImages = */ 1 );

			} // LOOP: stokes
			
		// check each image to find the maximum pixel value.
		for ( int stokes = 0; stokes < _param->NumStokesImages; stokes++ )
		{
	
			// get details back from the device.
			double * tmpMaxValue = (double *) malloc( MAX_PIXEL_DATA_AREA_SIZE * sizeof( double ) );
			moveDeviceToHost( (void *) tmpMaxValue, (void *) devMaxValue[ stokes ], MAX_PIXEL_DATA_AREA_SIZE * sizeof( double ),
									"copying max pixel data to host", __LINE__ );

			// check if this Stokes image has the largest value.							
			if (abs( tmpMaxValue[ MAX_PIXEL_VALUE ] ) > abs( hstMaxValue[ MAX_PIXEL_VALUE ] ) || stokes == 0)
			{
				memcpy( hstMaxValue, tmpMaxValue, MAX_PIXEL_DATA_AREA_SIZE * sizeof( double ) );
				hstMaxValue[ MAX_PIXEL_IMAGE ] = stokes;
			}
		
			// free memory.
			if (tmpMaxValue != NULL)
				free( (void *) tmpMaxValue );
		
		}

		// has the peak value fallen within a specified number of S.D. of the mean? If so, cleaning must stop.
		if (abs( hstMaxValue[ MAX_PIXEL_VALUE ] ) < pHogbomLimit)
		{
			printf( "\n                reached threshold of %6.4e Jy", pHogbomLimit );
			break;
		}

		// check if the peak value is rising rather than falling.
		if (minimumValue >= 0 && abs( hstMaxValue[ MAX_PIXEL_VALUE ] ) >= (minimumValue * 1.1))
		{
			printf( "\n                clean not converging on threshold %6.4e Jy", pHogbomLimit );
			break;
		}
		
		if (*pMinorCycle >= maxMinorCyclesBeforeFFT && maxMinorCyclesBeforeFFT > -1)
		{
			printf( "\n                %i minor cycles performed in this major cycle. max residual is %6.4e Jy", CYCLE_NITER,
																abs( hstMaxValue[ MAX_PIXEL_VALUE ] ) );
			break;
		}
		
		// have we reached the required number of iterations without hitting the stopping limit?
		if (abs( hstMaxValue[ MAX_PIXEL_VALUE ] ) > pHogbomLimit && *pMinorCycle == _param->MinorCycles - 1)
			printf( "\n                clean stopped at %6.4e Jy", abs( hstMaxValue[ MAX_PIXEL_VALUE ] ) );

		// update the minimum value.
		if (minimumValue < 0.0 || abs( hstMaxValue[ MAX_PIXEL_VALUE ] ) < minimumValue)
			minimumValue = abs( hstMaxValue[ MAX_PIXEL_VALUE ] );

		// get the component values.
		int x = (int) round( hstMaxValue[ MAX_PIXEL_X ] );
		int y = (int) round( hstMaxValue[ MAX_PIXEL_Y ] );
		double value = hstMaxValue[ MAX_PIXEL_VALUE ];
		int image = (int) round( hstMaxValue[ MAX_PIXEL_IMAGE ] );

		// define the block/thread dimensions.
		setThreadBlockSize2D( _param->PsfSize, _param->PsfSize, _gridSize2D, _blockSize2D );

		// if we are creating a mosaic then we need to do some manipulation on the psf.
		float * devDirtyBeam = NULL;
		reserveGPUMemory( (void **) &devDirtyBeam, _param->PsfSize * _param->PsfSize * sizeof( float ), "reserving device memory for the dirty beam", __LINE__ );
		if (_param->UseMosaicing == true)
		{
		
			// see if we've got a suitable PSF in the PSF cache.
			int cacheID = -1;
			int bestDist = -1;
			for ( int i = 0; i < NUM_PSFS_IN_CACHE; i++ )
				if (hstPsfCache[ i ] != NULL)
				{
					int dist = (int) floor( pow( hstPsfX[ i ] - x, 2 ) + pow( hstPsfY[ i ] - y, 2 ) );
					if (dist <= MAX_DISTANCE_IN_PIXELS && (dist < bestDist || cacheID == -1))
					{
						cacheID = i;
						bestDist = dist;
					}
				}
			
			// did we find one to use?
			if (cacheID > -1)
			{
			
				// upload PSF to device, and move this psf to the start of the cache.
				moveHostToDevice( (void *) devDirtyBeam, (void *) hstPsfCache[ cacheID ], _param->PsfSize * _param->PsfSize * sizeof( float ),
							"copying dirty beam to the device", __LINE__ );
							
				float * tmpPsf = hstPsfCache[ cacheID ];
				int tmpX = hstPsfX[ cacheID ], tmpY = hstPsfY[ cacheID ];
				for ( int i = cacheID; i > 0; i-- )
				{
					hstPsfX[ i ] = hstPsfX[ i - 1 ];
					hstPsfY[ i ] = hstPsfY[ i - 1 ];
					hstPsfCache[ i ] = hstPsfCache[ i - 1 ];
				}
				hstPsfX[ 0 ] = tmpX; hstPsfY[ 0 ] = tmpY; hstPsfCache[ 0 ] = tmpPsf;
							
			}
			else
			{

				// create some memory for an input dirty beam, primary beam, and normalisation pattern.
				float * devDirtyBeamIn = NULL, * devPrimaryBeam = NULL;
				reserveGPUMemory( (void **) &devDirtyBeamIn, _param->PsfSize * _param->PsfSize * sizeof( float ), "reserving device memory for the input dirty beam", __LINE__ );
				reserveGPUMemory( (void **) &devPrimaryBeam, _param->BeamSize * _param->BeamSize * sizeof( float ),
							"reserving device memory for the primary beam", __LINE__ );

				zeroGPUMemory( (void *) devDirtyBeam, _param->PsfSize * _param->PsfSize * sizeof( float ), "zeroing dirty beam on the device", __LINE__ );
				for ( int mosaicComponent = 0; mosaicComponent < _hstData.size(); mosaicComponent++ )
				{

					// upload the primary beam for this mosaic component.
					moveHostToDevice( (void *) devPrimaryBeam, (void *) _hstData[ mosaicComponent ]->PrimaryBeam,
								_param->BeamSize * _param->BeamSize * sizeof( float ), "copying primary beam to the device", __LINE__ );

					// upload the dirty beam for this mosaic component.
					moveHostToDevice( (void *) devDirtyBeamIn, (void *) phstDirtyBeam[ mosaicComponent ],
									_param->PsfSize * _param->PsfSize * sizeof( float ), "copying dirty beam to the device", __LINE__ );

					// build the output beam.
					devBuildWeightedDirtyBeam<<< _gridSize2D, _blockSize2D >>>(	/* pOutput = */ devDirtyBeam,
													/* pInput = */ devDirtyBeamIn,
													/* pPrimaryBeam = */ devPrimaryBeam,
													/* pNormalisationPattern = */ devNormalisationPattern,
													/* pPrimaryBeamPattern = */ devPrimaryBeamPattern,
													/* pPsfSize = */ _param->PsfSize,
													/* pBeamSize = */ _param->BeamSize,
													/* pImageSize = */ _param->ImageSize,
													/* pX = */ x,
													/* pY = */ y );

				} // LOOP: mosaicComponent
				
				// normalise the dirty beam to 1. get the maximum value from the beam. create a new memory area to hold the maximum pixel value.
				double * devMaxBeam;
				reserveGPUMemory( (void **) &devMaxBeam, MAX_PIXEL_DATA_AREA_SIZE * sizeof( double ), "declaring device memory for psf max pixel value", __LINE__ );
					
				// get the maximum value from this image.
				getMaxValue(	/* pdevImage = */ devDirtyBeam,
						/* pdevMaxValue = */ devMaxBeam,
						/* pWidth = */ _param->PsfSize,
						/* pHeight = */ _param->PsfSize,
						/* pUseAbsolute = */ false,
						/* pdevMask = */ NULL,
						/* pNumImages = */ 1 );

				// define the block/thread dimensions.
				int threads = _param->PsfSize * _param->PsfSize;
				int blocks;
				setThreadBlockSize1D( &threads, &blocks );

				// normalise the image
				devNormalise<<< blocks, threads >>>(	/* pArray = */ devDirtyBeam,
									/* pConstant = */ &devMaxBeam[ MAX_PIXEL_VALUE ],
									/* pItems = */ _param->PsfSize * _param->PsfSize );

				// free the max value memory area.
				if (devMaxBeam != NULL)
					cudaFree( (void *) devMaxBeam );

				// free memory.
				if (devDirtyBeamIn != NULL)
					cudaFree( (void *) devDirtyBeamIn );
				if (devPrimaryBeam != NULL)
					cudaFree( (void *) devPrimaryBeam );
					
				// save this item to the PSF cache. we need to delete the last item in the cache if the cache is full.
				if (hstPsfCache[ NUM_PSFS_IN_CACHE - 1 ] != NULL)
					free( (void *) hstPsfCache[ NUM_PSFS_IN_CACHE - 1 ] );
				for ( int i = NUM_PSFS_IN_CACHE - 1; i > 0; i-- )
				{
					hstPsfX[ i ] = hstPsfX[ i - 1 ];
					hstPsfY[ i ] = hstPsfY[ i - 1 ]; 
					hstPsfCache[ i ] = hstPsfCache[ i - 1 ];
				}
				hstPsfX[ 0 ] = x;
				hstPsfY[ 0 ] = y;
				hstPsfCache[ 0 ] = (float *) malloc( _param->PsfSize * _param->PsfSize * sizeof( float ) );
				moveDeviceToHost( (void *) hstPsfCache[ 0 ], (void *) devDirtyBeam, _param->PsfSize * _param->PsfSize * sizeof( float ),
							"copying dirty beam to the host cache", __LINE__ );

//{

//float * hstDirtyBeam = (float *) malloc( _param->PsfSize * _param->PsfSize * sizeof( float ) );
//moveDeviceToHost( (void *) hstDirtyBeam, (void *) devDirtyBeam, _param->PsfSize * _param->PsfSize * sizeof( float ), "copying dirty beam to the host", __LINE__ );
//char psfFilename[100];
//sprintf( psfFilename, "psf-%i-%i-%i", *pMinorCycle, x, y );
//_hstCasacoreInterface.WriteCasaImage( psfFilename, _param->PsfSize, _param->PsfSize, .parameters.OutputRA,
//					_param->OutputDEC, _param->CellSize, hstDirtyBeam, CONST_C / _hstData[ 0 ]->AverageWavelength, NULL );
//if (hstDirtyBeam != NULL)
//	free( (void *) hstDirtyBeam );
//}

			} // generate beam to add to cache

//			moveHostToDevice( (void *) devDirtyBeam, (void *) phstDirtyBeam[ _hstData.size() ], _param->PsfSize * _param->PsfSize * sizeof( float ),
//						"copying dirty beam to the device", __LINE__ );

		}
		else
			moveHostToDevice( (void *) devDirtyBeam, (void *) phstDirtyBeam[ /* MOSAIC_COMPONENT = */ 0 ], _param->PsfSize * _param->PsfSize * sizeof( float ),
						"copying dirty beam to the device", __LINE__ );
				
		// subtract dirty beam.
		devAddSubtractBeam<<< _gridSize2D, _blockSize2D >>>(	/* pImage = */ pdevDirtyImage + (image * _param->ImageSize * _param->ImageSize),
									/* pBeam = */ devDirtyBeam,
									/* pMaxValue = */ devMaxValue[ image ],
									/* pLoopGain = */ _param->LoopGain,
									/* pImageWidth = */ _param->ImageSize,
									/* pImageHeight = */ _param->ImageSize,
									/* pBeamSize = */ _param->PsfSize,
									/* pAddSubtract = */ SUBTRACT );
		err = cudaGetLastError();
		if (err != cudaSuccess)
			printf( "error subtracting beams (%i: %s).\n", __LINE__, cudaGetErrorString( err ) );

//float * tmp = (float *) malloc( _param->PsfSize * _param->PsfSize * sizeof( float ) );
//cudaMemcpy( tmp, devDirtyBeam, _param->PsfSize * _param->PsfSize * sizeof( float ), cudaMemcpyDeviceToHost );
//char filename[100];
//sprintf( filename, "psf-%i", *pMinorCycle );
//_hstCasacoreInterface.WriteCasaImage( filename, _param->PsfSize, _param->PsfSize, .parameters.OutputRA, _param->OutputDEC, _param->CellSize, tmp, CONST_C / _hstData[ 0 ]->AverageWavelength, NULL );
//free( (void *) tmp );

		// free memory. we free the maximum value for the image we've just updated so that it will be re-populated during the next loop cycle. the maximum value
		// data areas for the other Stokes products do not need to be rebuilt.
		if (devDirtyBeam != NULL)
			cudaFree( (void *) devDirtyBeam );
		if (devMaxValue[ image ] != NULL)
		{
			cudaFree( (void *) devMaxValue[ image ] );
			devMaxValue[ image ] = NULL;
		}

		// add item to component list.
		int found = -1;
		for ( int i = 0; i < pComponentListItems[ image ]; i++ )
			if (phstComponentListPos[ image ][ i ].u == x && phstComponentListPos[ image ][ i ].v == y)
			{
				found = i;
				break;
			}

		if (found == -1)
		{
			phstComponentListPos[ image ][ pComponentListItems[ image ] ].u = x;
			phstComponentListPos[ image ][ pComponentListItems[ image ] ].v = y;
			phstComponentListValue[ image ][ /* TAYLOR_TERM = */ 0 ][ pComponentListItems[ image ] ] = value * _param->LoopGain;
			pComponentListItems[ image ]++;
		}
		else
			phstComponentListValue[ image ][ /* TAYLOR_TERM = */ 0 ][ found ] += value * _param->LoopGain;

		// next minor cycle.
		*pMinorCycle = *pMinorCycle + 1;

	} // (*pMinorCycle < _param->MinorCycles)
	printf( "\n" );
		
	// free memory.
	if (devMaxValue != NULL)
	{
		for ( int stokes = 0; stokes < _param->NumStokesImages; stokes++ )
			if (devMaxValue[ stokes ] != NULL)
				cudaFree( (void *) devMaxValue[ stokes ] );
		free( (void *) devMaxValue );
	}
	if (hstMaxValue != NULL)
		free( (void *) hstMaxValue );
	if (devNormalisationPattern != NULL)
		cudaFree( (void *) devNormalisationPattern );
	if (devPrimaryBeamPattern != NULL)
		cudaFree( (void *) devPrimaryBeamPattern );
	if (hstPsfX != NULL)
		free( (void *) hstPsfX );
	if (hstPsfY != NULL)
		free( (void *) hstPsfY );
	if (hstPsfCache != NULL)
	{
		for ( int i = 0; i < NUM_PSFS_IN_CACHE; i++ )
			if (hstPsfCache[ i ] != NULL)
				free( (void *) hstPsfCache[ i ] );
		free( (void *) hstPsfCache );
	}
	
} // hogbomClean

//
//	mfsDeconvolve()
//
//	CJS: 24/06/2021
//
//	Perform a multi-frequency synthesis version of Hogbom clean (minor cycles).
//

void mfsDeconvolve( int * pMinorCycle, bool * phstMask, double pHogbomLimit, float ** pdevDirtyImage, float ** pdevDirtyBeam, VectorI * phstComponentListPos,
				 double ** phstComponentListValue, int * pComponentListItems )
{

	// calculate delta: (A00[0].A11[0] - A01[0].A01[0]). Sault et al 1994, A&ASS, 108, 585-594, Eq.15
	float a00 = 0.0, a11 = 0.0, a01 = 0.0;
	moveDeviceToHost( (void *) &a00, (void *) &pdevDirtyBeam[ 0 ][ ((_param->PsfSize / 2) * _param->PsfSize) + (_param->PsfSize / 2) ], sizeof( float ),
					"copying MFS A00 to host", __LINE__ );
	moveDeviceToHost( (void *) &a01, (void *) &pdevDirtyBeam[ 1 ][ ((_param->PsfSize / 2) * _param->PsfSize) + (_param->PsfSize / 2) ], sizeof( float ),
					"copying MFS A01 to host", __LINE__ );
	moveDeviceToHost( (void *) &a11, (void *) &pdevDirtyBeam[ 2 ][ ((_param->PsfSize / 2) * _param->PsfSize) + (_param->PsfSize / 2) ], sizeof( float ),
					"copying MFS A11 to host", __LINE__ );
	double delta = (a00 * a11) - (a01 * a01);

printf( "\nHpeak:\n\n" );
printf( "( %6.3f %6.3f )\n", a00, a01 );
printf( "( %6.3f %6.3f )\n\n", a01, a11 );
printf( "determinant: %6.3f\n", delta );
printf( "\nHpeak^-1:\n\n" );
printf( "( %6.3f %6.3f )\n", a11 / delta, -a01 / delta );
printf( "( %6.3f %6.3f )\n\n", -a01 / delta, a00 / delta );
	
	// hold the maximum value in the residual images.
	double * devMaxValueR;
	reserveGPUMemory( (void **) &devMaxValueR, MAX_PIXEL_DATA_AREA_SIZE * sizeof( double ),
						"declaring device memory for residual max pixel value", __LINE__ );
		
	// reserve host memory for the maximum pixel value.
	double * hstMaxValueR = (double *) malloc( MAX_PIXEL_DATA_AREA_SIZE * sizeof( double ) );

	// keep a record of the minimum value. if it does up by a certain factor then we need to stop cleaning.
	double minimumValue = -1.0;

	// create device memory for the mask. we only get the mask from the cache if we're using one.
	bool * devMask = NULL;
	if (phstMask != NULL)
	{
		reserveGPUMemory( (void **) &devMask, _param->ImageSize * _param->ImageSize * sizeof( bool ), "creating device memory for the mask", __LINE__ );
		cudaMemcpy( (void *) devMask, phstMask, _param->ImageSize * _param->ImageSize * sizeof( bool ), cudaMemcpyHostToDevice );
	}
	
	// loop over each minor cycle.
	while (*pMinorCycle < _param->MinorCycles)
	{
		
		printf( "." );
		fflush( stdout );

		// get the maximum value from the residual image.
		getMaxValue(	/* pdevImage = */ pdevDirtyImage[ 0 ],
				/* pdevMaxValue = */ devMaxValueR,
				/* pWidth = */ _param->ImageSize,
				/* pHeight = */ _param->ImageSize,
				/* pUseAbsolute = */ true,
				/* pdevMask = */ devMask,
				/* pNumImages = */ 1 );
	
		// get details back from the device.
		moveDeviceToHost( (void *) hstMaxValueR, (void *) devMaxValueR, MAX_PIXEL_DATA_AREA_SIZE * sizeof( double ),
					"copying max pixel data from R to host", __LINE__ );

		// has the peak value fallen within a specified number of S.D. of the mean? If so, cleaning must stop.
		if (hstMaxValueR[ MAX_PIXEL_VALUE ] < pHogbomLimit)
		{
			printf( "\n                reached threshold of %6.4e Jy", pHogbomLimit );
			break;
		}

		// check if the peak value is rising rather than falling.
		if (minimumValue >= 0 && hstMaxValueR[ MAX_PIXEL_VALUE ] >= (minimumValue * 1.1))
		{
			printf( "\n                clean not converging on threshold %6.4e Jy", pHogbomLimit );
			break;
		}

		// update the minimum value.
		if (minimumValue < 0 || hstMaxValueR[ MAX_PIXEL_VALUE ] < minimumValue)
			minimumValue = hstMaxValueR[ MAX_PIXEL_VALUE ];

		// compute the image: R0[j]^2.A11[0] + R1[j]^2.A00[0] - 2.R0[j].R1[j].A01[0]. Sault et al 1994, A&ASS, 108, 585-594, Eq.22
		getMaxMfsResidual(	/* pdevR0 = */ pdevDirtyImage[ 0 ],
					/* pdevR1 = */ pdevDirtyImage[ 1 ],
					/* pdevMaxValue = */ devMaxValueR,
					/* pWidth = */ _param->ImageSize,
					/* pHeight = */ _param->ImageSize,
					/* pMask = */ devMask,
					/* pA00 = */ a00,
					/* pA11 = */ a11,
					/* pA01 = */ a01 );
	
		// get details back from the device.
		moveDeviceToHost( (void *) hstMaxValueR, (void *) devMaxValueR, MAX_PIXEL_DATA_AREA_SIZE * sizeof( double ),
					"copying max pixel data from R0 to host", __LINE__ );

		// get the x and y coordinates of this component.
		int x = (int) round( hstMaxValueR[ MAX_PIXEL_X ] );
		int y = (int) round( hstMaxValueR[ MAX_PIXEL_Y ] );

		// get the pixel values from R0 and R1 for this pixel position.
		float r0Value = 0.0, r1Value = 0.0;
		moveDeviceToHost( (void *) &r0Value, (void *) &pdevDirtyImage[ 0 ][ (y * _param->ImageSize) + x ], sizeof( float ), "copying R0 pixel value to host", __LINE__ );
		moveDeviceToHost( (void *) &r1Value, (void *) &pdevDirtyImage[ 1 ][ (y * _param->ImageSize) + x ], sizeof( float ), "copying R1 pixel value to host", __LINE__ );
printf( " [%i] Res-0 %f, Res-1 %f, Max %f, ", *pMinorCycle, r0Value, r1Value, hstMaxValueR[ MAX_PIXEL_VALUE ] );
		// calculate the zeroth and first-order components a0 and a1.: Sault et al 1994, A&ASS, 108, 585-594, Eq.14. These equations are also
		// the solution to equation 21 in Rau & Cornwell, 2011, after the both sides are multiplied by H_peak^-1.
		// H_peak and H_peak^-1 are given by:
		//	H_peak = ( a00	a01 )		H_peak^-1 = 1/delta ( a11  -a01 ), where the determinant delta = a00.a11 - a01.a10.
		//		 ( a10	a11 )				    ( -a10  a00 )
		double a0 = (((double) a11 * (double) r0Value) - ((double) a01 * (double) r1Value)) / delta;
		double a1 = (((double) a00 * (double) r1Value) - ((double) a01 * (double) r0Value)) / delta;
printf( "Coeffs: %f %f\n", a0, a1 );
		// define the block/thread dimensions.
		setThreadBlockSize2D( _param->PsfSize, _param->PsfSize, _gridSize2D, _blockSize2D );

		// update the max value storage with a0.
		moveHostToDevice( (void *) &devMaxValueR[ MAX_PIXEL_VALUE ], &a0, sizeof( double ), "moving a0 value to device", __LINE__ );

		// subtract A00, and A01 from the R0, and R1, respectively.
		for ( int i = 0; i < _param->TaylorTerms; i++ )
			devAddSubtractBeam<<< _gridSize2D, _blockSize2D >>>(	/* pImage = */ pdevDirtyImage[ i ],
										/* pBeam = */ pdevDirtyBeam[ i ],
										/* pMaxValue = */ devMaxValueR,
										/* pLoopGain = */ _param->LoopGain,
										/* pImageWidth = */ _param->ImageSize,
										/* pImageHeight = */ _param->ImageSize,
										/* pBeamSize = */ _param->PsfSize,
										/* pAddSubtract = */ SUBTRACT );

		// update the max value storage with a1.
		moveHostToDevice( (void *) &devMaxValueR[ MAX_PIXEL_VALUE ], &a1, sizeof( double ), "moving a1 value to device", __LINE__ );

		// add A10, and A11 to R0, and R1, respectively.
		for ( int i = 0; i < _param->TaylorTerms; i++ )
			devAddSubtractBeam<<< _gridSize2D, _blockSize2D >>>(	/* pImage = */ pdevDirtyImage[ i ],
										/* pBeam = */ pdevDirtyBeam[ i + 1 ],
										/* pMaxValue = */ devMaxValueR,
										/* pLoopGain = */ _param->LoopGain,
										/* pImageWidth = */ _param->ImageSize,
										/* pImageHeight = */ _param->ImageSize,
										/* pBeamSize = */ _param->PsfSize,
										/* pAddSubtract = */ SUBTRACT );		// NOTE: ADD in Sault

		// add item to component list.
		int found = -1;
		for ( int i = 0; i < *pComponentListItems; i++ )
			if (phstComponentListPos[ i ].u == x && phstComponentListPos[ i ].v == y)
			{
				found = i;
				break;
			}

		if (found == -1)
		{
			phstComponentListPos[ *pComponentListItems ].u = x;
			phstComponentListPos[ *pComponentListItems ].v = y;
			phstComponentListValue[ 0 ][ *pComponentListItems ] = a0 * _param->LoopGain;
			phstComponentListValue[ 1 ][ *pComponentListItems ] = a1 * _param->LoopGain;
			(*pComponentListItems)++;
		}
		else
		{
			phstComponentListValue[ 0 ][ found ] += a0 * _param->LoopGain;
			phstComponentListValue[ 1 ][ found ] += a1 * _param->LoopGain;
		}

		// next minor cycle.
		*pMinorCycle = *pMinorCycle + 1;

	}
	printf( "\n" );

	// free memory.
	if (devMaxValueR != NULL)
		cudaFree( (void *) devMaxValueR );
	if (hstMaxValueR != NULL)
		free( (void *) hstMaxValueR );

} // mfsDeconvolve

//
//	getMaxValueFromMaskedImage()
//
//	CJS: 21/06/2021
//
//	Construct a mask based upon an image (when we only want values less than the threshold), and then look for the maximum value in another image of the same size
//	based upon the mask.
//

double getMaxValueFromMaskedImage( float * pdevImage, float * pdevMaskingImage, double pMaskingValue, int pSize )
{

	// create device memory for the psf mask, and the data area.
	bool * devMask = NULL;
	double * devMaxValue;
	reserveGPUMemory( (void **) &devMask, pSize * pSize * sizeof( bool ), "reserving device memory for the ask", __LINE__ );
	reserveGPUMemory( (void **) &devMaxValue, MAX_PIXEL_DATA_AREA_SIZE * sizeof( double ), "declaring device memory for max pixel value", __LINE__ );

	// build a mask based upon the clean beam where TRUE means the value is less than 0.1.
	setThreadBlockSize2D( _param->PsfSize, _param->PsfSize, _gridSize2D, _blockSize2D );
	devBuildMask<<< _gridSize2D, _blockSize2D >>>(	/* pMask = */ devMask,
								/* pArray = */ pdevMaskingImage,
								/* pSize = */ pSize,
								/* pValue = */ pMaskingValue,
								/* pMaxMin = */ MASK_MAX );

	// get the peak value from the psf sidelobes.
	getMaxValue(	/* pdevImage = */ pdevImage,
			/* pdevMaxValue = */ devMaxValue,
			/* pWidth = */ pSize,
			/* pHeight = */ pSize,
			/* pUseAbsolute = */ false,
			/* pdevMask = */ devMask,
			/* pNumImages = */ 1 );
	double hstMaxValue = 0.0;
	moveDeviceToHost( (void *) &hstMaxValue, (void *) &devMaxValue[ MAX_PIXEL_VALUE ], sizeof( double ), "moving maximum value to the host", __LINE__ );
	
	// free memory.
	if (devMaxValue != NULL)
		cudaFree( (void *) devMaxValue );
	if (devMask != NULL)
		cudaFree( (void *) devMask );

	// return something.
	return hstMaxValue;

} // getMaxValueFromMaskedImage

//
//	degridVisibilitiesForImage()
//
//	CJS: 07/07/2021
//
//	Degrids a single image, and subtracts the model visibilities from the residual visibilities.
//

void degridVisibilitiesForImage
			(
			int pMosaicID,					// the index of the image being processed.
			cufftComplex *** phstModelImage,		// an array of model images - one for each Taylor term of MFS.
			long int pTotalVisibilities,			// the total number of visibilities being processed (required for updating the percentage bar).
			bool pFinalPass
			)
{

	// process all the stages.
	long int visibilitiesProcessed = 0;
	for ( int stageID = 0; stageID < _hstData[ pMosaicID ]->Stages; stageID++ )
	{

		// uncache the data for this mosaic.
		if (_param->CacheData == true)
			_hstData[ pMosaicID ]->UncacheData(	/* pBatchID = */ stageID,
								/* pTaylorTerm = */ -1,
								/* pOffset = */ 0,
								/* pWhatData = */ DATA_GRID_POSITIONS | DATA_KERNEL_INDEXES,
								/* pStokes = */ -1 );

		// calculate the batch size.
		int hstVisibilityBatchSize = 0;
		{
			long int nextBatchSize = _hstData[ pMosaicID ]->NumVisibilities[ stageID ];
			if (nextBatchSize > _param->PREFERRED_VISIBILITY_BATCH_SIZE)
				nextBatchSize = _param->PREFERRED_VISIBILITY_BATCH_SIZE;
			hstVisibilityBatchSize = (int) nextBatchSize;
		}

		// create some memory to store the model visibilities, and clear it.
		cufftComplex *** hstModelVisibility = (cufftComplex ***) malloc( _param->NumStokesImages * sizeof( cufftComplex ** ) );
		for ( int s = 0; s < _param->NumStokesImages; s++ )
			hstModelVisibility[ s ] = (cufftComplex **) malloc( _param->TaylorTerms * sizeof( cufftComplex * ) );

		// we loop over the Taylor terms, degridding each model image.
		for ( int taylorTerm = 0; taylorTerm < _param->TaylorTerms; taylorTerm++ )
		{

			// create memory for the model visibilities.
			for ( int s = 0; s < _param->NumStokesImages; s++ )
				hstModelVisibility[ s ][ taylorTerm ] = (cufftComplex *) malloc( _hstData[ pMosaicID ]->NumVisibilities[ stageID ] * sizeof( cufftComplex ) );

			// variables for device memory.
			cufftComplex ** devModelImageUV = (cufftComplex **) malloc( _param->NumGPUs * sizeof( cufftComplex * ) );
			VectorI ** devGridPosition = (VectorI **) malloc( _param->NumGPUs * sizeof( VectorI ) );
			int ** devKernelIndex = (int **) malloc( _param->NumGPUs * sizeof( int * ) );
			cufftComplex ** devModelVisibilities = (cufftComplex **) malloc( _param->NumGPUs * sizeof( cufftComplex * ) );
			for ( int gpu = 0; gpu < _param->NumGPUs; gpu++ )
			{
				if (_param->NumGPUs > 1)
					cudaSetDevice( _param->GPU[ gpu ] );
				reserveGPUMemory( (void **) &devModelImageUV[ gpu ], _param->ImageSize * _param->ImageSize * sizeof( cufftComplex ),
							"reserving device memory for the model image", __LINE__ );
				reserveGPUMemory( (void **) &devGridPosition[ gpu ], hstVisibilityBatchSize * sizeof( VectorI ),
							"reserving device memory for grid positions", __LINE__ );
				reserveGPUMemory( (void **) &devKernelIndex[ gpu ], hstVisibilityBatchSize * sizeof( int ),
							"reserving device memory for kernel indexes", __LINE__ );
				reserveGPUMemory( (void **) &devModelVisibilities[ gpu ], hstVisibilityBatchSize * sizeof( cufftComplex ),
							"creating device memory for model visibilities", __LINE__ );
			}
			if (_param->NumGPUs > 1)
				cudaSetDevice( _param->GPU[ 0 ] );

			// here is the start of the visibility batch loop.
			int batch = 0;
			int hstCurrentVisibility = 0;
			while (hstCurrentVisibility < _hstData[ pMosaicID ]->NumVisibilities[ stageID ])
			{

				KernelCache & kernelCache = *_degriddingKernelCache[ pMosaicID ];
				int wPlanes = kernelCache.wPlanes;
				int pbChannels = kernelCache.pbChannels;

				// count the number of visibilities in this batch.
				int visibilitiesInThisBatch = 0;
				for ( int wPlane = 0; wPlane < wPlanes; wPlane++ )
					for ( int pbChannel = 0; pbChannel < pbChannels; pbChannel++ )
						for ( int gpu = 0; gpu < _param->NumGPUs; gpu++ )
							visibilitiesInThisBatch += kernelCache( pbChannel, /* STOKES = */ 0, /* STOKES = */ 0, wPlane ).
																	visibilities[ stageID ][ batch ][ gpu ];

				printf( "	degridding" );
				if (_param->Deconvolver == MFS)
					printf( " (Taylor term %i)", taylorTerm );
				if (_hstData[ pMosaicID ]->Stages > 1 || _hstData[ pMosaicID ]->Batches[ stageID ] > 1)
					printf( " " );
				else
					printf( " visibilities\n\n" );
				if (_hstData[ pMosaicID ]->Stages > 1)
					printf( "host batch %i of %i", stageID + 1, _hstData[ pMosaicID ]->Stages );
				if (_hstData[ pMosaicID ]->Stages > 1 && _hstData[ pMosaicID ]->Batches[ stageID ] > 1)
					printf( ", " );
				if (_hstData[ pMosaicID ]->Batches[ stageID ] > 1)
					printf( "gpu batch %i of %i", batch + 1, _hstData[ pMosaicID ]->Batches[ stageID ] );
				if (_hstData[ pMosaicID ]->Stages > 1 || _hstData[ pMosaicID ]->Batches[ stageID ] > 1)
				{
					int fractionDone = (int) round( (double) visibilitiesProcessed * 50.0 / (double) pTotalVisibilities );
					int fractionDoing = (int) round( (double) (visibilitiesProcessed + visibilitiesInThisBatch) * 50.0 /
											(double) pTotalVisibilities );
					printf( " [" );
					for ( int i = 0; i < fractionDone; i++ )
						printf( "*" );
					for ( int i = 0; i < (fractionDoing - fractionDone); i++ )
						printf( "+" );
					for ( int i = 0; i < (50 - fractionDoing); i++ )
						printf( "." );
					printf( "]\n\n" );
					visibilitiesProcessed += visibilitiesInThisBatch;
				}

				// maintain pointers to the next visibilities for each GPU.
				int * hstNextVisibility = (int *) malloc( _param->NumGPUs * sizeof( int ) );
				for ( int gpu = 0; gpu < _param->NumGPUs; gpu++ )
					hstNextVisibility[ gpu ] = 0;

				int visibilityPointer = hstCurrentVisibility;
				for ( int wPlane = 0; wPlane < wPlanes; wPlane++ )
					for ( int pbChannel = 0; pbChannel < pbChannels; pbChannel++ )
						for ( int cudaDeviceIndex = 0; cudaDeviceIndex < _param->NumGPUs; cudaDeviceIndex++ )
						{
					
							int visibilitiesInKernelSet = kernelCache( pbChannel, /* STOKES = */ 0, /* STOKES = */ 0, wPlane ).
															visibilities[ stageID ][ batch ][ cudaDeviceIndex ];
							if (visibilitiesInKernelSet > 0)
							{

								// set the cuda device.
								if (_param->NumGPUs > 1)
									cudaSetDevice( _param->GPU[ cudaDeviceIndex ] );

								// upload grid positions, kernel indexes, density map, and original visibilities to the device.
								moveHostToDevice( (void *) &devGridPosition[ cudaDeviceIndex ][ hstNextVisibility[ cudaDeviceIndex ] ],
											(void *) &_hstData[ pMosaicID ]->GridPosition[ visibilityPointer ],
											visibilitiesInKernelSet * sizeof( VectorI ),
											"copying grid positions to the device", __LINE__ );
								moveHostToDevice( (void *) &devKernelIndex[ cudaDeviceIndex ][ hstNextVisibility[ cudaDeviceIndex ] ],
											(void *) &_hstData[ pMosaicID ]->KernelIndex[ visibilityPointer ],
											visibilitiesInKernelSet * sizeof( int ),
											"copying kernel indexes to the device", __LINE__ );

								// get the next set of visibilities.
								visibilityPointer += visibilitiesInKernelSet;
								hstNextVisibility[ cudaDeviceIndex ] += visibilitiesInKernelSet;

							} // visibilitiesInKernelSet > 0

						}
				if (_param->NumGPUs > 1)
					cudaSetDevice( _param->GPU[ 0 ] );

				// loop over the Stokes products and leakage patterns. we need to degrid each product, and create model visibilities.
				for ( int stokesProduct = 0; stokesProduct < _param->NumStokesImages; stokesProduct++ )
				{

					// loop over the GPUs again, setting the model visibilities to zero.
					for ( int gpu = 0; gpu < _param->NumGPUs; gpu++ )
						hstNextVisibility[ gpu ] = 0;

					int visibilityPointer = hstCurrentVisibility;
					for ( int wPlane = 0; wPlane < wPlanes; wPlane++ )
						for ( int pbChannel = 0; pbChannel < pbChannels; pbChannel++ )
							for ( int cudaDeviceIndex = 0; cudaDeviceIndex < _param->NumGPUs; cudaDeviceIndex++ )
							{
				
								int visibilitiesInKernelSet =
											kernelCache( pbChannel, stokesProduct, /* STOKES = */ 0, wPlane ).
															visibilities[ stageID ][ batch ][ cudaDeviceIndex ];
								if (visibilitiesInKernelSet > 0)
								{

									// set the cuda device.
									if (_param->NumGPUs > 1)
										cudaSetDevice( _param->GPU[ cudaDeviceIndex ] );

									// set the model visibilities to zero.
									zeroGPUMemory( (void *) &devModelVisibilities[ cudaDeviceIndex ]
															[ hstNextVisibility[ cudaDeviceIndex ] ],
												visibilitiesInKernelSet * sizeof( cufftComplex ),
												"clearing the model visibilities on the device", __LINE__ );

									// get the next set of visibilities.
									visibilityPointer += visibilitiesInKernelSet;
									hstNextVisibility[ cudaDeviceIndex ] += visibilitiesInKernelSet;

								} // visibilitiesInKernelSet > 0

							}
					if (_param->NumGPUs > 1)
						cudaSetDevice( _param->GPU[ 0 ] );
								
					for ( int stokesLeakage = 0; stokesLeakage < _param->NumStokesImages; stokesLeakage++ )
// cjs-mod					for ( int stokesLeakage = stokesProduct; stokesLeakage <= stokesProduct; stokesLeakage++ )
//if (pFinalPass == false || stokesLeakage == 1) // cjs-mod
						if (stokesLeakage == stokesProduct || (_param->AProjection == true && _param->LeakageCorrection == true))
						{

							printf( "		degridding Stokes " );
							if (_param->NumStokesImages == 1)
							{
								if (_param->Stokes == STOKES_I) printf( "I" );
								else if (_param->Stokes == STOKES_Q) printf( "Q" );
								else if (_param->Stokes == STOKES_U) printf( "U" );
								else if (_param->Stokes == STOKES_V) printf( "V" );
							}
							else if (_param->NumStokesImages > 1 && stokesProduct == stokesLeakage)
							{
								if (stokesProduct == STOKES_I) printf( "I" );
								else if (stokesProduct == STOKES_Q) printf( "Q" );
								else if (stokesProduct == STOKES_U) printf( "U" );
								else if (stokesProduct == STOKES_V) printf( "V" );
							}
							else
							{
								if (stokesLeakage == STOKES_I) printf( "I" );
								else if (stokesLeakage == STOKES_Q) printf( "Q" );
								else if (stokesLeakage == STOKES_U) printf( "U" );
								else if (stokesLeakage == STOKES_V) printf( "V" );
								printf( " leakage into Stokes " );
								if (stokesProduct == STOKES_I) printf( "I" );
								else if (stokesProduct == STOKES_Q) printf( "Q" );
								else if (stokesProduct == STOKES_U) printf( "U" );
								else if (stokesProduct == STOKES_V) printf( "V" );
							}
							printf( " to model visibilities" );
							
							// move the model image for the Stokes leakage product to the GPU.
							for ( int gpu = 0; gpu < _param->NumGPUs; gpu++ )
							{
								if (_param->NumGPUs > 1)
									cudaSetDevice( _param->GPU[ gpu ] );
								moveHostToDevice( (void *) devModelImageUV[ gpu ], (void *) phstModelImage[ stokesLeakage ][ taylorTerm ],
									_param->ImageSize * _param->ImageSize * sizeof( cufftComplex ), "moving model image to the device",
									__LINE__ );
							}
							if (_param->NumGPUs > 1)
								cudaSetDevice( _param->GPU[ 0 ] );
						
							// ensure the beam pattern for A-projection is not empty.
							if (_param->AProjection == true && _param->LeakageCorrection == true &&
								_hstData[ pMosaicID ]->MuellerMatrixFlag[ (stokesProduct * _param->NumStokesImages) + stokesLeakage ] == false)
								printf( ": skipping due to missing beam pattern\n\n" );
							else
							{
							
								printf( "\n\n" );

								// degridding with w-projection and oversampling.
								gridVisibilities(	/* pdevGrid = */ devModelImageUV,
											/* pStageID = */ stageID,
											/* pBatchID = */ batch,
											/* pdevVisibility = */ devModelVisibilities,
											/* pdevKernelIndex = */ devKernelIndex,
											/* pdevGridPositions = */ devGridPosition,
											/* pdevWeight = */ NULL,
											/* pSize = */ _param->ImageSize,
											/* pNumGPUs = */ _param->NumGPUs,
											/* pStokesTo = */ stokesProduct,
											/* pStokesFrom = */ stokesLeakage,
											/* phstKernelCache = */ kernelCache );

							} // (beam is not missing)
							
						} // LOOP: stokesLeakage

					// reset next visibility counters.
					for ( int gpu = 0; gpu < _param->NumGPUs; gpu++ )
						hstNextVisibility[ gpu ] = 0;

					visibilityPointer = hstCurrentVisibility;
					for ( int wPlane = 0; wPlane < wPlanes; wPlane++ )
						for ( int pbChannel = 0; pbChannel < pbChannels; pbChannel++ )
							for ( int cudaDeviceIndex = 0; cudaDeviceIndex < _param->NumGPUs; cudaDeviceIndex++ )
							{
				
								int visibilitiesInKernelSet = kernelCache( pbChannel, stokesProduct, /* STOKES = */ 0, wPlane ).
															visibilities[ stageID ][ batch ][ cudaDeviceIndex ];
								if (visibilitiesInKernelSet > 0)
								{

									// set the cuda device.
									if (_param->NumGPUs > 1)
										cudaSetDevice( _param->GPU[ cudaDeviceIndex ] );

									// download model visibilities to the host.
									moveDeviceToHost( (void *) &hstModelVisibility[ stokesProduct ][ taylorTerm ][ visibilityPointer ],
												(void *) &devModelVisibilities[ cudaDeviceIndex ]
																[ hstNextVisibility[ cudaDeviceIndex ] ],
												visibilitiesInKernelSet * sizeof( cufftComplex ),
												"copying model visibilities to the host", __LINE__ );

									// get the next set of visibilities.
									visibilityPointer += visibilitiesInKernelSet;
									hstNextVisibility[ cudaDeviceIndex ] += visibilitiesInKernelSet;

								} // visibilitiesInKernelSet > 0

							} // LOOP: kernelSet
					if (_param->NumGPUs > 1)
						cudaSetDevice( _param->GPU[ 0 ] );
					
				} // LOOP: stokesProduct

				// free memory.
				if (hstNextVisibility != NULL)
					free( (void *) hstNextVisibility );

				// move to the next set of batch of data.
				hstCurrentVisibility += visibilitiesInThisBatch;
				batch = batch + 1;

			} // (current-visibility < num-visibilities)

			// free memory.
			for ( int gpu = 0; gpu < _param->NumGPUs; gpu++ )
			{
				if (_param->NumGPUs > 1)
					cudaSetDevice( _param->GPU[ gpu ] );
				if (devModelImageUV[ gpu ] != NULL)
					cudaFree( (void *) devModelImageUV[ gpu ] );
				if (devModelVisibilities[ gpu ] != NULL)
					cudaFree( (void *) devModelVisibilities[ gpu ] );
				if (devGridPosition[ gpu ] != NULL)
					cudaFree( (void *) devGridPosition[ gpu ] );
				if (devKernelIndex[ gpu ] != NULL)
					cudaFree( (void *) devKernelIndex[ gpu ] );
			}
			if (_param->NumGPUs > 1)
				cudaSetDevice( _param->GPU[ 0 ] );
			if (devModelImageUV != NULL)
				free( (void *) devModelImageUV );
			if (devModelVisibilities != NULL)
				free( (void *) devModelVisibilities );
			if (devGridPosition != NULL)
				free( (void *) devGridPosition );
			if (devKernelIndex != NULL)
				free( (void *) devKernelIndex );

		} // LOOP: taylorTerm

		// free grid positions and kernel indexes.
		if (_param->CacheData == true)
			_hstData[ pMosaicID ]->FreeData( /* pWhatData = */ DATA_GRID_POSITIONS | DATA_KERNEL_INDEXES );

		// get the visibilities, density map, and mfs weights.
		if (_param->CacheData == true)
			_hstData[ pMosaicID ]->UncacheData(	/* pBatchID = */ stageID,
								/* pTaylorTerm = */ -1,
								/* pOffset = */ 0,
								/* pWhatData = */ DATA_VISIBILITIES | DATA_DENSITIES | DATA_MFS_WEIGHTS,
								/* pStokes = */ -1 );

		// loop over Taylor terms again, calculating the model visibilities for each Taylor term, and then the residual visibilities.
		for ( int taylorTerm = 0; taylorTerm < _param->TaylorTerms; taylorTerm++ )
		{

			// create some memory to store the residual visibilities.
			for ( int s = 0; s < _param->NumStokesImages; s++ )
				_hstData[ pMosaicID ]->ResidualVisibility[ /* STOKES = */ s ][ taylorTerm ] =
									(cufftComplex *) malloc( _hstData[ pMosaicID ]->NumVisibilities[ stageID ] * sizeof( cufftComplex ) );

			// calculate the batch size.
			int hstVisibilityBatchSize = 0;
			{
				long int nextBatchSize = _hstData[ pMosaicID ]->NumVisibilities[ stageID ];
				if (nextBatchSize > _param->PREFERRED_VISIBILITY_BATCH_SIZE)
					nextBatchSize = _param->PREFERRED_VISIBILITY_BATCH_SIZE;
				hstVisibilityBatchSize = (int) nextBatchSize;
			}

			// variables for device memory.
			int ** devDensityMap = (int **) malloc( _param->NumGPUs * sizeof( int * ) );
			cufftComplex ** devModelVisibility_0 = (cufftComplex **) malloc( _param->NumGPUs * sizeof( cufftComplex * ) );
			cufftComplex ** devModelVisibility_1 = (cufftComplex **) malloc( _param->NumGPUs * sizeof( cufftComplex * ) );
			cufftComplex ** devOriginalVisibilities = (cufftComplex **) malloc( _param->NumGPUs * sizeof( cufftComplex * ) );
			float ** devMfsWeight_1 = (float **) malloc( _param->NumGPUs * sizeof( float * ) );
			float ** devMfsWeight_2 = (float **) malloc( _param->NumGPUs * sizeof( float * ) );
			for ( int gpu = 0; gpu < _param->NumGPUs; gpu++ )
			{
				if (_param->NumGPUs > 1)
					cudaSetDevice( _param->GPU[ gpu ] );
				reserveGPUMemory( (void **) &devDensityMap[ gpu ], hstVisibilityBatchSize * sizeof( int ),
							"declaring device memory for density map", __LINE__ );
				reserveGPUMemory( (void **) &devModelVisibility_0[ gpu ], hstVisibilityBatchSize * sizeof( cufftComplex ),
							"creating device memory for model visibilities", __LINE__ );
				if (_param->Deconvolver == MFS)
					reserveGPUMemory( (void **) &devModelVisibility_1[ gpu ], hstVisibilityBatchSize * sizeof( cufftComplex ),
								"creating device memory for model visibilities", __LINE__ );
				else
					devModelVisibility_1[ gpu ] = NULL;
				reserveGPUMemory( (void **) &devOriginalVisibilities[ gpu ], hstVisibilityBatchSize * sizeof( cufftComplex ),
							"creating memory for original visibilities", __LINE__ );
				if (_param->Deconvolver == MFS)
					reserveGPUMemory( (void **) &devMfsWeight_1[ gpu ], hstVisibilityBatchSize * sizeof( float ),
								"creating memory for the mfs weights (t = 1)", __LINE__ );
				else
					devMfsWeight_1[ gpu ] = NULL;
				if (_param->Deconvolver == MFS && taylorTerm > 0)
					reserveGPUMemory( (void **) &devMfsWeight_2[ gpu ], hstVisibilityBatchSize * sizeof( float ),
								"creating memory for the mfs weights (t = 2)", __LINE__ );
				else
					devMfsWeight_2[ gpu ] = NULL;
			}
			if (_param->NumGPUs > 1)
				cudaSetDevice( _param->GPU[ 0 ] );

			// here is the start of the visibility batch loop.
			int batch = 0;
			int hstCurrentVisibility = 0;
			while (hstCurrentVisibility < _hstData[ pMosaicID ]->NumVisibilities[ stageID ])
			{

				KernelCache & kernelCache = *_degriddingKernelCache[ pMosaicID ];
				int wPlanes = kernelCache.wPlanes;
				int pbChannels = kernelCache.pbChannels;

				// count the number of visibilities in this batch.
				int visibilitiesInThisBatch = 0;
				for ( int wPlane = 0; wPlane < wPlanes; wPlane++ )
					for ( int pbChannel = 0; pbChannel < pbChannels; pbChannel++ )
						for ( int gpu = 0; gpu < _param->NumGPUs; gpu++ )
							visibilitiesInThisBatch += kernelCache( pbChannel, /* STOKES = */ 0, /* STOKES = */ 0, wPlane ).
																	visibilities[ stageID ][ batch ][ gpu ];

				// maintain pointers to the next visibilities for each GPU.
				int * hstNextVisibility = (int *) malloc( _param->NumGPUs * sizeof( int ) );
				for ( int gpu = 0; gpu < _param->NumGPUs; gpu++ )
					hstNextVisibility[ gpu ] = 0;

				int visibilityPointer = hstCurrentVisibility;
				for ( int wPlane = 0; wPlane < wPlanes; wPlane++ )
					for ( int pbChannel = 0; pbChannel < pbChannels; pbChannel++ )
						for ( int cudaDeviceIndex = 0; cudaDeviceIndex < _param->NumGPUs; cudaDeviceIndex++ )
						{
			
							int visibilitiesInKernelSet = kernelCache( pbChannel, /* STOKES = */ 0, /* STOKES = */ 0, wPlane ).
															visibilities[ stageID ][ batch ][ cudaDeviceIndex ];
							if (visibilitiesInKernelSet > 0)
							{

								// set the cuda device.
								if (_param->NumGPUs > 1)
									cudaSetDevice( _param->GPU[ cudaDeviceIndex ] );

								// upload density map to the device.
								moveHostToDevice( (void *) &devDensityMap[ cudaDeviceIndex ][ hstNextVisibility[ cudaDeviceIndex ] ],
											(void *) &_hstData[ pMosaicID ]->DensityMap[ visibilityPointer ],
											visibilitiesInKernelSet * sizeof( int ),
											"copying density map to the device", __LINE__ );

								// get the next set of visibilities.
								visibilityPointer += visibilitiesInKernelSet;
								hstNextVisibility[ cudaDeviceIndex ] += visibilitiesInKernelSet;

							} // visibilitiesInKernelSet > 0

						} // LOOP: kernelSet
				if (_param->NumGPUs > 1)
					cudaSetDevice( _param->GPU[ 0 ] );

				// loop over all the Stokes products, calculating residual visibilities for each.
				for ( int stokesProduct = 0; stokesProduct < _param->NumStokesImages; stokesProduct++ )
				{
				
					// upload the original visibilities and the model visibilities.
					for ( int gpu = 0; gpu < _param->NumGPUs; gpu++ )
						hstNextVisibility[ gpu ] = 0;

					visibilityPointer = hstCurrentVisibility;
					for ( int wPlane = 0; wPlane < wPlanes; wPlane++ )
						for ( int pbChannel = 0; pbChannel < pbChannels; pbChannel++ )
							for ( int cudaDeviceIndex = 0; cudaDeviceIndex < _param->NumGPUs; cudaDeviceIndex++ )
							{

								int visibilitiesInKernelSet = kernelCache( pbChannel, stokesProduct, /* STOKES = */ 0, wPlane ).
															visibilities[ stageID ][ batch ][ cudaDeviceIndex ];
								if (visibilitiesInKernelSet > 0)
								{

									// set the cuda device.
									if (_param->NumGPUs > 1)
										cudaSetDevice( _param->GPU[ cudaDeviceIndex ] );

									// upload original visibilities to the device.
									moveHostToDevice( (void *) &devOriginalVisibilities[ cudaDeviceIndex ]
																[ hstNextVisibility[ cudaDeviceIndex ] ],
												(void *) &_hstData[ pMosaicID ]->Visibility[ stokesProduct ]
																[ taylorTerm ][ visibilityPointer ],
												visibilitiesInKernelSet * sizeof( cufftComplex ),
												"copying original visibilities to the device", __LINE__ );

									moveHostToDevice( (void *) &devModelVisibility_0[ cudaDeviceIndex ]
																[ hstNextVisibility[ cudaDeviceIndex ] ],
												(void *) &hstModelVisibility[ stokesProduct ][ /* TAYLOR_TERM = */ 0 ]
																[ visibilityPointer ],
												visibilitiesInKernelSet * sizeof( cufftComplex ),
												"copying model visibilities to the device", __LINE__ );

									if (_param->Deconvolver == MFS)
									{
										moveHostToDevice( (void *) &devModelVisibility_1[ cudaDeviceIndex ]
															[ hstNextVisibility[ cudaDeviceIndex ] ],
													(void *) &hstModelVisibility[ stokesProduct ]
															[ /* TAYLOR_TERM = */ 1 ][ visibilityPointer ],
													visibilitiesInKernelSet * sizeof( cufftComplex ),
													"copying model visibilities for Taylor term 1 to the device",
													__LINE__ );
										moveHostToDevice( (void *) &devMfsWeight_1[ cudaDeviceIndex ]
															[ hstNextVisibility[ cudaDeviceIndex ] ],
												(void *) &_hstData[ pMosaicID ]->MfsWeight[ stokesProduct ]
															[ /* TAYLOR_TERM = */ 0 ][ visibilityPointer ],
													visibilitiesInKernelSet * sizeof( float ),
													"copying MFS weights for Taylor term 1 to the device", __LINE__ );
										if (taylorTerm > 0)
											moveHostToDevice( (void *) &devMfsWeight_2[ cudaDeviceIndex ]
																[ hstNextVisibility[ cudaDeviceIndex ] ],
														(void *) &_hstData[ pMosaicID ]->
																MfsWeight[ stokesProduct ]
																[ /* TAYLOR_TERM = */ 1 ]
																[ visibilityPointer ],
														visibilitiesInKernelSet * sizeof( float ),
														"copying MFS weights for Taylor term 2 to the device",
														__LINE__ );
									}

									// get the next set of visibilities.
									visibilityPointer += visibilitiesInKernelSet;
									hstNextVisibility[ cudaDeviceIndex ] += visibilitiesInKernelSet;

								} // visibilitiesInKernelSet > 0

							} // LOOP: kernelSet
					if (_param->NumGPUs > 1)
						cudaSetDevice( _param->GPU[ 0 ] );

					// apply density map, and subtract from the real visibilities:
					for ( int gpu = 0; gpu < _param->NumGPUs; gpu++ )
						if (hstNextVisibility[ gpu ] > 0)
						{

							// set the cuda device.
							if (_param->NumGPUs > 1)
								cudaSetDevice( _param->GPU[ gpu ] );

							// define the block/thread dimensions.
							int threads = hstNextVisibility[ gpu ];
							int blocks;
							setThreadBlockSize1D( &threads, &blocks );

							// for Taylor term 0, we calculate N x V_m, where N is the density map.
							if (taylorTerm == 0)
							{
								devMultiplyArrays<<< blocks, threads >>>(	/* pOne = */ devModelVisibility_0[ gpu ],
														/* pTwo = */ devDensityMap[ gpu ],
														/* pSize = */ hstNextVisibility[ gpu ] );
								if (_param->Deconvolver == MFS)
									devMultiplyArrays<<< blocks, threads >>>(	/* pOne = */ devModelVisibility_1[ gpu ],
															/* pTwo = */ devMfsWeight_1[ gpu ],
															/* pSize = */ hstNextVisibility[ gpu ] );
							}

							// for Taylor term 1, we calculate (SUM m_i) x V_m, where m_i is the mfs weight of this visibility. we have
							// SUM m_i stored.
							if (taylorTerm == 1)
							{
								devMultiplyArrays<<< blocks, threads >>>(	/* pOne = */ devModelVisibility_0[ gpu ],
														/* pTwo = */ devMfsWeight_1[ gpu ],
														/* pSize = */ hstNextVisibility[ gpu ] );
								devMultiplyArrays<<< blocks, threads >>>(	/* pOne = */ devModelVisibility_1[ gpu ],
														/* pTwo = */ devMfsWeight_2[ gpu ],
														/* pSize = */ hstNextVisibility[ gpu ] );
							}

							// subtract the model visibilities from the real visibilities to get a new set of (dirty) visibilities.
if (pFinalPass == false) // cjs-mod
							devSubtractArrays<<< blocks, threads >>>(	/* pOne = */ devOriginalVisibilities[ gpu ],
													/* pTwo = */ devModelVisibility_0[ gpu ],
													/* pSize = */ hstNextVisibility[ gpu ] );
else
cudaMemcpy( devOriginalVisibilities[ gpu ], devModelVisibility_0[ gpu ], hstNextVisibility[ gpu ] * sizeof( cufftComplex ), cudaMemcpyDeviceToDevice );
//cudaMemcpy( devOriginalVisibilities[ gpu ], devModelVisibility_0[ gpu ], hstNextVisibility[ gpu ] * sizeof( cufftComplex ), cudaMemcpyDeviceToDevice );
							if (_param->Deconvolver == MFS)
								devSubtractArrays<<< blocks, threads >>>(	/* pOne = */ devOriginalVisibilities[ gpu ],
														/* pTwo = */ devModelVisibility_1[ gpu ],
														/* pSize = */ hstNextVisibility[ gpu ] );
//if (taylorTerm == 0)
//	cudaMemcpy( (void *) devOriginalVisibilities[ gpu ], (void *) devModelVisibility_0[ gpu ],
//				hstNextVisibility[ gpu ] * sizeof( cufftComplex ), cudaMemcpyDeviceToDevice );
//else
//	cudaMemcpy( (void *) devOriginalVisibilities[ gpu ], (void *) devModelVisibility_1[ gpu ],
//				hstNextVisibility[ gpu ] * sizeof( cufftComplex ), cudaMemcpyDeviceToDevice );

						}
					if (_param->NumGPUs > 1)
						cudaSetDevice( _param->GPU[ 0 ] );

					// reset next visibility counters.
					for ( int gpu = 0; gpu < _param->NumGPUs; gpu++ )
						hstNextVisibility[ gpu ] = 0;

					visibilityPointer = hstCurrentVisibility;
					for ( int wPlane = 0; wPlane < wPlanes; wPlane++ )
						for ( int pbChannel = 0; pbChannel < pbChannels; pbChannel++ )
							for ( int cudaDeviceIndex = 0; cudaDeviceIndex < _param->NumGPUs; cudaDeviceIndex++ )
							{
				
								int visibilitiesInKernelSet = kernelCache( pbChannel, /* STOKES = */ 0, /* STOKES = */ 0, wPlane ).
														visibilities[ stageID ][ batch ][ cudaDeviceIndex ];
								if (visibilitiesInKernelSet > 0)
								{

									// set the cuda device.
									if (_param->NumGPUs > 1)
										cudaSetDevice( _param->GPU[ cudaDeviceIndex ] );

									// download residual visibilities to the host.
									moveDeviceToHost( (void *) &_hstData[ pMosaicID ]->ResidualVisibility[ stokesProduct ]
													[ taylorTerm ][ visibilityPointer ],
												(void *) &devOriginalVisibilities[ cudaDeviceIndex ]
													[ hstNextVisibility[ cudaDeviceIndex ] ],
												visibilitiesInKernelSet * sizeof( cufftComplex ),
												"copying model visibilities to the host", __LINE__ );

									// get the next set of visibilities.
									visibilityPointer += visibilitiesInKernelSet;
									hstNextVisibility[ cudaDeviceIndex ] += visibilitiesInKernelSet;

								} // visibilitiesInKernelSet > 0

							} // LOOP: kernelSet
					if (_param->NumGPUs > 1)
						cudaSetDevice( _param->GPU[ 0 ] );

				} // LOOP: stokesProduct

				// free memory.
				if (hstNextVisibility != NULL)
					free( (void *) hstNextVisibility );

				// move to the next set of batch of data.
				hstCurrentVisibility += visibilitiesInThisBatch;
				batch = batch + 1;

			} // (current-visibility < num-visibilities)

			// free memory.
			for ( int gpu = 0; gpu < _param->NumGPUs; gpu++ )
			{
				if (_param->NumGPUs > 1)
					cudaSetDevice( _param->GPU[ gpu ] );
				if (devModelVisibility_0[ gpu ] != NULL)
					cudaFree( (void *) devModelVisibility_0[ gpu ] );
				if (devModelVisibility_1[ gpu ] != NULL)
					cudaFree( (void *) devModelVisibility_1[ gpu ] );
				if (devDensityMap[ gpu ] != NULL)
					cudaFree( (void *) devDensityMap[ gpu ] );
				if (devMfsWeight_1[ gpu ] != NULL)
					cudaFree( (void *) devMfsWeight_1[ gpu ] );
				if (devMfsWeight_2[ gpu ] != NULL)
					cudaFree( (void *) devMfsWeight_2[ gpu ] );
				if (devOriginalVisibilities[ gpu ] != NULL)
					cudaFree( (void *) devOriginalVisibilities[ gpu ] );
			}
			if (_param->NumGPUs > 1)
				cudaSetDevice( _param->GPU[ 0 ] );
			if (devModelVisibility_0 != NULL)
				free( (void *) devModelVisibility_0 );
			if (devModelVisibility_1 != NULL)
				free( (void *) devModelVisibility_1 );
			if (devDensityMap != NULL)
				free( (void *) devDensityMap );
			if (devOriginalVisibilities != NULL)
				free( (void *) devOriginalVisibilities );
			if (devMfsWeight_1 != NULL)
				free( (void *) devMfsWeight_1 );
			if (devMfsWeight_2 != NULL)
				free( (void *) devMfsWeight_2 );

			// cache the residual visibilities.
			if (_param->CacheData == true)
				_hstData[ pMosaicID ]->CacheData(	/* pBatchID = */ stageID,
									/* pTaylorTerm = */ taylorTerm,
									/* pWhatData = */ DATA_RESIDUAL_VISIBILITIES );
			
			// free more memory.
			for ( int s = 0; s < _param->NumStokesImages; s++ )
				if (hstModelVisibility[ /* STOKES = */ s ][ taylorTerm ] != NULL)
					free( (void *) hstModelVisibility[ /* STOKES = */ s ][ taylorTerm ] );

		} // LOOP: taylorTerm

		// free the data for this mosaic.
		if (_param->CacheData == true)
			_hstData[ pMosaicID ]->FreeData(/* pWhatData = */ DATA_ALL );
			
		// free memory.
		if (hstModelVisibility != NULL)
		{
			for ( int s = 0; s < _param->NumStokesImages; s++ )
				free( (void *) hstModelVisibility[ s ] );
			free( (void *) hstModelVisibility );
		}

	} // LOOP: stageID

} // degridVisibilitiesForImage

//
//	gridVisibilitiesForImage()
//
//	CJS: 07/07/2021
//
//	Grids a single image. that is, one component of the mosaic if we have a mosaic.
//

void gridVisibilitiesForImage
			(
			int pMosaicID,					// the index of the image being processed.
			cufftComplex ** pdevDirtyImageGrid,		// an array of grids, one for each GPU
			long int pTotalVisibilities,			// the total number of visibilities being processed (required for updating the percentage bar).
			int pTaylorTerm,				// either 0 or 1 since we are currently using two terms.
			visibilitytype pVisibilityType,		// either RESIDUAL or OBSERVED
			int pStokes					// the Stokes image being gridded
			)
{

	// process all the stages.
	long int visibilitiesProcessed = 0;
	for ( int stageID = 0; stageID < _hstData[ pMosaicID ]->Stages; stageID++ )
	{

		// uncache the data for this mosaic.
		if (_param->CacheData == true)
			_hstData[ pMosaicID ]->UncacheData(	/* pBatchID = */ stageID,
								/* pTaylorTerm = */ pTaylorTerm,
								/* pOffset = */ 0,
								/* pWhatData = */ DATA_GRID_POSITIONS | DATA_KERNEL_INDEXES | DATA_DENSITIES | DATA_WEIGHTS |
											(pVisibilityType == OBSERVED ? DATA_VISIBILITIES : DATA_RESIDUAL_VISIBILITIES),
								/* pStokes = */ pStokes );
								
		// calculate the batch size.
		int hstVisibilityBatchSize = 0;
		{
			long int nextBatchSize = _hstData[ pMosaicID ]->NumVisibilities[ stageID ];
			if (nextBatchSize > _param->PREFERRED_VISIBILITY_BATCH_SIZE)
				nextBatchSize = _param->PREFERRED_VISIBILITY_BATCH_SIZE;
			hstVisibilityBatchSize = (int) nextBatchSize;
		}

		// variables for device memory.
		VectorI ** devGridPosition = (VectorI **) malloc( _param->NumGPUs * sizeof( VectorI ) );
		int ** devKernelIndex = (int **) malloc( _param->NumGPUs * sizeof( int * ) );
		cufftComplex ** devVisibility = (cufftComplex **) malloc( _param->NumGPUs * sizeof( cufftComplex * ) );
		float ** devWeight = NULL;
		if (_param->Weighting != NONE)
			devWeight = (float **) malloc( _param->NumGPUs * sizeof( float * ) );
		for ( int gpu = 0; gpu < _param->NumGPUs; gpu++ )
		{
			if (_param->NumGPUs > 1)
				cudaSetDevice( _param->GPU[ gpu ] );
			reserveGPUMemory( (void **) &devGridPosition[ gpu ], hstVisibilityBatchSize * sizeof( VectorI ),
						"reserving device memory for grid positions", __LINE__ );
			reserveGPUMemory( (void **) &devKernelIndex[ gpu ], hstVisibilityBatchSize * sizeof( int ),
						"reserving device memory for kernel indexes", __LINE__ );
			reserveGPUMemory( (void **) &devVisibility[ gpu ], hstVisibilityBatchSize * sizeof( cufftComplex ),
						"creating device memory for visibilities", __LINE__ );
			if (_param->Weighting != NONE)
				reserveGPUMemory( (void **) &devWeight[ gpu ], hstVisibilityBatchSize * sizeof( float ),
						"creating device memory for weights", __LINE__ );
		}
		if (_param->NumGPUs > 1)
			cudaSetDevice( _param->GPU[ 0 ] );

		// here is the start of the visibility batch loop.
		int hstCurrentVisibility = 0;
		int batch = 0;
		while (hstCurrentVisibility < _hstData[ pMosaicID ]->NumVisibilities[ stageID ])
		{

			KernelCache & kernelCache = *_griddingKernelCache[ pMosaicID ];
			int wPlanes = kernelCache.wPlanes;
			int pbChannels = kernelCache.pbChannels;
			
			// count the number of visibilities in this batch.
			int visibilitiesInThisBatch = 0;
			for ( int wPlane = 0; wPlane < wPlanes; wPlane++ )
				for ( int pbChannel = 0; pbChannel < pbChannels; pbChannel++ )
					for ( int gpu = 0; gpu < _param->NumGPUs; gpu++ )
						visibilitiesInThisBatch += kernelCache( pbChannel, /* STOKES = */ 0, /* STOKES = */ 0, wPlane ).
																	visibilities[ stageID ][ batch ][ gpu ];

			printf( "        gridding " );
			if (_hstData[ pMosaicID ]->Stages == 1 && _hstData[ pMosaicID ]->Batches[ stageID ] == 1)
				printf( "visibilities\n\n" );
			if (_hstData[ pMosaicID ]->Stages > 1)
				printf( "host batch %i of %i", stageID + 1, _hstData[ pMosaicID ]->Stages );
			if (_hstData[ pMosaicID ]->Stages > 1 && _hstData[ pMosaicID ]->Batches[ stageID ] > 1)
				printf( ", " );
			if (_hstData[ pMosaicID ]->Batches[ stageID ] > 1)
				printf( "gpu batch %i of %i", batch + 1, _hstData[ pMosaicID ]->Batches[ stageID ] );
			if (_hstData[ pMosaicID ]->Stages > 1 || _hstData[ pMosaicID ]->Batches[ stageID ] > 1)
			{
				int fractionDone = (int) round( (double) visibilitiesProcessed * 50.0 / (double) pTotalVisibilities );
				int fractionDoing = (int) round( (double) (visibilitiesProcessed + visibilitiesInThisBatch) * 50.0 /
										(double) pTotalVisibilities );
				printf( " [" );
				for ( int i = 0; i < fractionDone; i++ )
					printf( "*" );
				for ( int i = 0; i < (fractionDoing - fractionDone); i++ )
					printf( "+" );
				for ( int i = 0; i < (50 - fractionDoing); i++ )
					printf( "." );
				printf( "]\n\n" );
				visibilitiesProcessed += visibilitiesInThisBatch;
			}

			// maintain pointers to the next visibilities for each GPU.
			int * hstNextVisibility = (int *) malloc( _param->NumGPUs * sizeof( int ) );
			for ( int gpu = 0; gpu < _param->NumGPUs; gpu++ )
				hstNextVisibility[ gpu ] = 0;

			int arrayIndex = hstCurrentVisibility;
			for ( int wPlane = 0; wPlane < wPlanes; wPlane++ )
				for ( int pbChannel = 0; pbChannel < pbChannels; pbChannel++ )
					for ( int cudaDeviceIndex = 0; cudaDeviceIndex < _param->NumGPUs; cudaDeviceIndex++ )
					{
			
						int visibilitiesInKernelSet = kernelCache( pbChannel, /* STOKES = */ 0, /* STOKES = */ 0, wPlane ).
															visibilities[ stageID ][ batch ][ cudaDeviceIndex ];
						if (visibilitiesInKernelSet > 0)
						{

							// set the cuda device, and make sure nothing is running there already.
							if (_param->NumGPUs > 1)
								cudaSetDevice( _param->GPU[ cudaDeviceIndex ] );

							// upload grid positions and kernel indexes to the device.
							moveHostToDevice( (void *) &devGridPosition[ cudaDeviceIndex ][ hstNextVisibility[ cudaDeviceIndex ] ],
										(void *) &_hstData[ pMosaicID ]->GridPosition[ arrayIndex ],
										visibilitiesInKernelSet * sizeof( VectorI ),
										"copying grid positions to the device", __LINE__ );
							moveHostToDevice( (void *) &devKernelIndex[ cudaDeviceIndex ][ hstNextVisibility[ cudaDeviceIndex ] ],
										(void *) &_hstData[ pMosaicID ]->KernelIndex[ arrayIndex ],
										visibilitiesInKernelSet * sizeof( int ),
										"copying kernel indexes to the device", __LINE__ );

							// get the next set of visibilities.
							arrayIndex += visibilitiesInKernelSet;
							hstNextVisibility[ cudaDeviceIndex ] += visibilitiesInKernelSet;

						}

					} // LOOP: kernelSet
			if (_param->NumGPUs > 1)
				cudaSetDevice( _param->GPU[ 0 ] );

			// we have to grid up to four sets of visibilities - one for each Stokes product.
//			for ( int stokesLeakage = 0; stokesLeakage < _param->NumStokesImages; stokesLeakage++ )
			{
			
//				printf( "                gridding Stokes " );
//				else if (stokesLeakage == pStokes)
//				{
//					if (pStokes == STOKES_I) printf( "I" );
//					else if (pStokes == STOKES_Q) printf( "Q" );
//					else if (pStokes == STOKES_U) printf( "U" );
//					else if (pStokes == STOKES_V) printf( "V" );
//				}
//				else
//				{
//					if (stokesLeakage == STOKES_I) printf( "I" );
//					else if (stokesLeakage == STOKES_Q) printf( "Q" );
//					else if (stokesLeakage == STOKES_U) printf( "U" );
//					else if (stokesLeakage == STOKES_V) printf( "V" );
//					printf( " to Stokes " );
//					if (pStokes == STOKES_I) printf( "I" );
//					else if (pStokes == STOKES_Q) printf( "Q" );
//					else if (pStokes == STOKES_U) printf( "U" );
//					else if (pStokes == STOKES_V) printf( "V" );
//					printf( " leakage correction" );
//				}
				
				// ensure the beam pattern for A-projection is not empty.
//				if (_param->AProjection == true && _hstData[ pMosaicID ]->InverseMuellerMatrixFlag[ (pStokes * 4) + stokesLeakage ] == false)
//				if (_param->AProjection == true && _hstData[ pMosaicID ]->InverseMuellerMatrixFlag[ (pStokes * 4) + pStokes ] == false)
//					printf( ": skipping due to missing beam\n\n" );
//				else
				{
					
//					printf( "\n\n" );

					// upload the visibilities for this Stokes product.
					for ( int gpu = 0; gpu < _param->NumGPUs; gpu++ )
						hstNextVisibility[ gpu ] = 0;
				
					// reset the array index to the current visibility. we do this because we may need to grid these visibilities 4 times - once for each
					// Stokes product.
					arrayIndex = hstCurrentVisibility;

					for ( int wPlane = 0; wPlane < wPlanes; wPlane++ )
						for ( int pbChannel = 0; pbChannel < pbChannels; pbChannel++ )
							for ( int cudaDeviceIndex = 0; cudaDeviceIndex < _param->NumGPUs; cudaDeviceIndex++ )
							{

//								int visibilitiesInKernelSet = kernelCache( pbChannel, pStokes, stokesLeakage, wPlane ).
//															visibilities[ stageID ][ batch ][ cudaDeviceIndex ];
								int visibilitiesInKernelSet = kernelCache( pbChannel, pStokes, pStokes, wPlane ).
															visibilities[ stageID ][ batch ][ cudaDeviceIndex ];

								if (visibilitiesInKernelSet > 0)
								{

									// set the cuda device, and make sure nothing is running there already.
									if (_param->NumGPUs > 1)
										cudaSetDevice( _param->GPU[ cudaDeviceIndex ] );

									// upload weights, visibilities to the device.
									if (_param->Weighting != NONE)
										moveHostToDevice( (void *) &devWeight[ cudaDeviceIndex ]
															[ hstNextVisibility[ cudaDeviceIndex ] ],
//													(void *) &_hstData[ pMosaicID ]->Weight[ stokesLeakage ][ arrayIndex ],
													(void *) &_hstData[ pMosaicID ]->Weight[ pStokes ][ arrayIndex ],
													visibilitiesInKernelSet * sizeof( float ),
													"copying weights to the device", __LINE__ );

									if (pVisibilityType == OBSERVED)
										moveHostToDevice( (void *) &devVisibility[ cudaDeviceIndex ]
																	[ hstNextVisibility[ cudaDeviceIndex ] ],
//													(void *) &_hstData[ pMosaicID ]->Visibility[ stokesLeakage ]
													(void *) &_hstData[ pMosaicID ]->Visibility[ pStokes ]
																	[ pTaylorTerm ][ arrayIndex ],
													visibilitiesInKernelSet * sizeof( cufftComplex ),
													"copying visibilities to the device", __LINE__ );

									if (pVisibilityType == RESIDUAL)
										moveHostToDevice( (void *) &devVisibility[ cudaDeviceIndex ]
															[ hstNextVisibility[ cudaDeviceIndex ] ],
//												(void *) &_hstData[ pMosaicID ]->ResidualVisibility[ stokesLeakage ]
												(void *) &_hstData[ pMosaicID ]->ResidualVisibility[ pStokes ]
															[ pTaylorTerm ][ arrayIndex ],
												visibilitiesInKernelSet * sizeof( cufftComplex ),
												"copying residual visibilities to the device", __LINE__ );

									// get the next set of visibilities.
									arrayIndex += visibilitiesInKernelSet;
									hstNextVisibility[ cudaDeviceIndex ] += visibilitiesInKernelSet;

								}

							} // LOOP: kernelSet
					if (_param->NumGPUs > 1)
						cudaSetDevice( _param->GPU[ 0 ] );

					// grid the new set of dirty visibilities.
					gridVisibilities(	/* pdevGrid = */ pdevDirtyImageGrid,
								/* pStageID = */ stageID,
								/* pBatchID = */ batch,
								/* pdevVisibility = */ devVisibility,
								/* pdevKernelIndexes = */ devKernelIndex,
								/* pdevGridPositions = */ devGridPosition,
								/* pdevWeight = */ devWeight,
								/* pSize = */ _param->ImageSize,
								/* pNumGPUs = */ _param->NumGPUs,
								/* pStokesTo = */ pStokes,
//								/* pStokesFrom = */ stokesLeakage,
								/* pStokesFrom = */ pStokes,
								/* phstKernelCache = */ kernelCache );

				} // (beam is not missing)
							
			} // LOOP: stokesLeakage

			// free memory.
			if (hstNextVisibility != NULL)
				free( (void *) hstNextVisibility );

			// move to the next set of batch of data.
			batch = batch + 1;
			hstCurrentVisibility = arrayIndex;

		} // WHILE: (current-visibility < num-visibilities)

		// free memory.
		for ( int gpu = 0; gpu < _param->NumGPUs; gpu++ )
		{
			if (_param->NumGPUs > 1)
				cudaSetDevice( _param->GPU[ gpu ] );
			if (devVisibility[ gpu ] != NULL)
				cudaFree( (void *) devVisibility[ gpu ] );
			if (_param->Weighting != NONE)
				if (devWeight[ gpu ] != NULL)
					cudaFree( (void *) devWeight[ gpu ] );
			if (devGridPosition[ gpu ] != NULL)
				cudaFree( (void *) devGridPosition[ gpu ] );
			if (devKernelIndex[ gpu ] != NULL)
				cudaFree( (void *) devKernelIndex[ gpu ] );
		}
		if (devVisibility != NULL)
			free( (void *) devVisibility );
		if (devWeight != NULL)
			free( (void *) devWeight );
		if (devGridPosition != NULL)
			free( (void *) devGridPosition );
		if (devKernelIndex != NULL)
			free( (void *) devKernelIndex );

		if (_param->NumGPUs > 1)
			cudaSetDevice( _param->GPU[ 0 ] );

		// free the data.
		if (_param->CacheData == true)
			_hstData[ pMosaicID ]->FreeData( /* pWhatData = */ DATA_GRID_POSITIONS | DATA_KERNEL_INDEXES | DATA_DENSITIES | DATA_WEIGHTS |
										DATA_RESIDUAL_VISIBILITIES );

	} // LOOP: stageID

} // gridVisibilitiesForImage

//
//	generateDirtyImages()
//
//	CJS: 13/07/2021
//
//	Generate the dirty images by FFT'ing the gridded data. We generate one image for each Taylor term.
//
//	According to Rau & Cornwell, 1990, we need to generate T spectral dirty images for MFS. The function we generate for the psf is:
//
//		I_t^dirty = SUM_nu omega_nu^t.I_nu^dirty
//
//	where omega_nu^t = [ (nu - vu_ref) / nu ]^t. So on the first iteration of the loop (t = 0) we don't apply the MFS weights at all.
//

void generateDirtyImages
			(
			float ** phstDirtyImage,			// an array of images, one for each Taylor term
			bool * phstMask,				//
			visibilitytype pVisibilityType,		// OBSERVED or RESIDUAL
			int pStokes					// the Stokes parameter to image. this will be zero if we're not using A-projection.
			)
{

	// copy the normalisation pattern, and primary beam patterns, over.
	float * devNormalisationPattern = NULL, * devPrimaryBeamPattern = NULL;
	if (_hstPrimaryBeamPattern != NULL && _param->UseMosaicing == true)
	{
		reserveGPUMemory( (void **) &devPrimaryBeamPattern, _param->BeamSize * _param->BeamSize * sizeof( float ), "reserving GPU memory for primary beam pattern",
																					__LINE__ );
		moveHostToDevice( (void *) devPrimaryBeamPattern, (void *) _hstPrimaryBeamPattern, _param->BeamSize * _param->BeamSize * sizeof( float ),
					"copying primary beam pattern to the device", __LINE__ );
	}
	if (_hstNormalisationPattern != NULL && _param->UseMosaicing == true)
	{
		reserveGPUMemory( (void **) &devNormalisationPattern, _param->BeamSize * _param->BeamSize * sizeof( float ),
					"reserving GPU memory for normalisation pattern", __LINE__ );
		moveHostToDevice( (void *) devNormalisationPattern, (void *) _hstNormalisationPattern, _param->BeamSize * _param->BeamSize * sizeof( float ),
					"copying normalisation pattern to the device", __LINE__ );
	}

	// declare device memory for the dirty image grid, and zero this memory. we need to do this on ALL gpus.
	cufftComplex ** devDirtyImageGrid = (cufftComplex **) malloc( _param->NumGPUs * sizeof( cufftComplex * ) );
	for ( int gpu = 0; gpu < _param->NumGPUs; gpu++ )
	{
		if (_param->NumGPUs > 1)
			cudaSetDevice( _param->GPU[ gpu ] );
		reserveGPUMemory( (void **) &devDirtyImageGrid[ gpu ], _param->ImageSize * _param->ImageSize * sizeof( cufftComplex ),
					"declaring device memory for grid", __LINE__ );
		zeroGPUMemory( (void *) devDirtyImageGrid[ gpu ], _param->ImageSize * _param->ImageSize * sizeof( cufftComplex ), "zeroing the grid on the device", __LINE__ );
	}
	if (_param->NumGPUs > 1)
		cudaSetDevice( _param->GPU[ 0 ] );

	int stokes = pStokes;
	if (_param->NumStokesImages == 1)
		stokes = _param->Stokes;
		
	printf( "\ngridding visibilities for " );
	if (stokes == STOKES_I) printf( "Stokes I " );
	else if (stokes == STOKES_Q) printf( "Stokes Q " );
	else if (stokes == STOKES_U) printf( "Stokes U " );
	else if (stokes == STOKES_V) printf( "Stokes V " );
	printf( "dirty image" );

	printf( ".....\n\n" );
	for ( int image = 0; image < _hstData.size(); image++ )
	{

		// count the total number of visibilities.
		long int totalVisibilities = 0;
		for ( int stageID = 0; stageID < _hstData[ image ]->Stages; stageID++ )
			totalVisibilities += _hstData[ image ]->NumVisibilities[ stageID ];

		if (_hstData.size() > 1)
			printf( "	processing mosaic component %i of %i.....", image + 1, (int) _hstData.size() );
		else
			printf( "	processing visibilities....." );
		printf( "(stages: %i, visibilities: %li)\n\n", _hstData[ image ]->Stages, totalVisibilities );
		
		// print the support sizes.
		displaySupportSizes(	/* pKernelCache = */ *_griddingKernelCache[ image ],
					/* pStokes = */ stokes );

		for ( int taylorTerm = 0; taylorTerm < _param->TaylorTerms; taylorTerm++ )
		{

			// if we're doing an image-plane mosaic then clear the grids now.
			if (_param->ImagePlaneMosaic == true)
			{
			
				for ( int gpu = 0; gpu < _param->NumGPUs; gpu++ )
				{
					if (_param->NumGPUs > 1)
						cudaSetDevice( _param->GPU[ gpu ] );
					zeroGPUMemory( (void *) devDirtyImageGrid[ gpu ], _param->ImageSize * _param->ImageSize * sizeof( cufftComplex ),
								"zeroing the grid on the device", __LINE__ );
				}
				if (_param->NumGPUs > 1)
					cudaSetDevice( _param->GPU[ 0 ] );

			} // (_param->ImagePlaneMosaic == true)

			// grid the visibilities from this image.
			gridVisibilitiesForImage(	/* pMosaicID = */ image,
							/* pdevDirtyImageGrid = */ devDirtyImageGrid,
							/* pTotalVisibilities = */ totalVisibilities,
							/* pTaylorTerm = */ taylorTerm,
							/* pVisibilityType = */ pVisibilityType,
							/* pStokes = */ pStokes );

			// we only do FFT and normalisation if we've finished gridding. if we are making an UV mosaic then we only do this on the last pass.
			if (_param->UvPlaneMosaic == false || image == _hstData.size() - 1)
			{

				double normalisation = 1.0;

				// we normalise the image by the number of gridded visibilities, but only if we're not using UV-plane mosaicing. UV-plane mosaicing will do the
				// normalisation using the kernel.
				if (_param->UvPlaneMosaic == true)
					normalisation *= (double) _hstData[ image ]->MinimumVisibilitiesInMosaic;
				else
					normalisation *= (double) _hstData[ image ]->GriddedVisibilities;

				// move all images to the same GPU and add them together.
				if (_param->NumGPUs > 1)
				{

					cufftComplex * hstTmpImage = (cufftComplex *) malloc( _param->ImageSize * _param->ImageSize * sizeof( cufftComplex ) );
					cufftComplex * devTmpImage = NULL;
					reserveGPUMemory( (void **) &devTmpImage, _param->ImageSize * _param->ImageSize * sizeof( cufftComplex ),
									"reserving GPU memory for the temporary gridded data", __LINE__ );

					int items = _param->ImageSize * _param->ImageSize;
					int stages = items / MAX_THREADS;
					if (items % MAX_THREADS != 0)
						stages++;

					for ( int gpu = 1; gpu < _param->NumGPUs; gpu++ )
					{

						// set gpu device, and move image to the host.
						cudaSetDevice( _param->GPU[ gpu ] );
						moveDeviceToHost( (void *) hstTmpImage, devDirtyImageGrid[ gpu ],
								_param->ImageSize * _param->ImageSize * sizeof( cufftComplex ), "moving gridded data to the host", __LINE__ );
						cudaDeviceSynchronize();

						// set gpu device, and move image to the device.
						cudaSetDevice( _param->GPU[ 0 ] );
						moveHostToDevice( (void *) devTmpImage, (void *) hstTmpImage, _param->ImageSize * _param->ImageSize * sizeof( cufftComplex ),
									"moving gridded data to the device", __LINE__ );

						for ( int i = 0; i < stages; i++ )
						{

							// define the block/thread dimensions.
							int itemsThisStage = items - (i * MAX_THREADS);
							if (itemsThisStage > MAX_THREADS)
								itemsThisStage = MAX_THREADS;
							int threads = itemsThisStage;
							int blocks;
							setThreadBlockSize1D( &threads, &blocks );

							// add images together.
							devAddArrays<<< blocks, threads >>>(	/* pOne = */ &devDirtyImageGrid[ /* GPU = */ 0 ][ /* CELL = */ i * MAX_THREADS ],
												/* pTwo = */ &devTmpImage[ /* CELL = */ i * MAX_THREADS ],
												/* pSize = */ itemsThisStage );

						}

					} // LOOP: gpu

					// free memory.
					if (hstTmpImage != NULL)
						free( (void *) hstTmpImage );
					if (devTmpImage != NULL)
						cudaFree( (void *) devTmpImage );

				}

				printf( "\n        performing fft on dirty image grid.....\n" );
	
				// make dirty image on the device.
				performFFT(	/* pdevGrid = */ &devDirtyImageGrid[ /* GPU = */ 0 ],
						/* pSize = */ _param->ImageSize,
						/* pFFTDirection = */ INVERSE,
						/* pFFTPlan = */ -1,
						/* pFFTType = */ C2F,
						/* pResizeArray = */ false );

//{
//float * tmp = (float *) malloc( _param->ImageSize * _param->ImageSize * sizeof( cufftComplex ) );
//cudaMemcpy( tmp, devDirtyImageGrid[ 0 ], _param->ImageSize * _param->ImageSize * sizeof( cufftComplex ), cudaMemcpyDeviceToHost );
//for ( int i = 0; i < _param->ImageSize * _param->ImageSize; i++ )
//	tmp[ i ] = tmp[ i * 2 ];
//cudaMemcpy( devDirtyImageGrid[ 0 ], tmp, _param->ImageSize * _param->ImageSize * sizeof( cufftComplex ), cudaMemcpyHostToDevice );
//free( tmp );
//}
					
				// define the block/thread dimensions.
				int items = _param->ImageSize * _param->ImageSize;
				int stages = items / MAX_THREADS;
				if (items % MAX_THREADS != 0)
					stages++;

				for ( int i = 0; i < stages; i++ )
				{

					// define the block/thread dimensions.
					int itemsThisStage = items - (i * MAX_THREADS);
					if (itemsThisStage > MAX_THREADS)
						itemsThisStage = MAX_THREADS;
					int threads = itemsThisStage;
					int blocks;
					setThreadBlockSize1D( &threads, &blocks );

					// normalise the image by the normalisation factor.
					float * devDirtyImageDbl = (float *) devDirtyImageGrid[ /* GPU = */ 0 ];
					devNormalise<<< blocks, threads >>>( &devDirtyImageDbl[ i * MAX_THREADS ], normalisation, itemsThisStage );

				}

				// define the block/thread dimensions.
				setThreadBlockSize2D( _param->ImageSize, _param->ImageSize, _gridSize2D, _blockSize2D );

				// divide the dirty image by the prolate-spheroidal correction.
				devDivideImages<<< _gridSize2D, _blockSize2D >>>(	/* pOne = */ (float *) devDirtyImageGrid[ /* GPU = */ 0 ],
											/* pTwo = */ _devImageDomainPSFunction,
											/* pMask = */ NULL,
											/* pSizeOne = */ _param->ImageSize,
											/* pSizeTwo = */ _param->PsfSize,
											/* pInterpolate = */ true );

				// if we're not mosaicing then we need to re-scale the flux in our image so that the primary beam appears to be that of the longest wavelength.
				if (_param->UseMosaicing == false)
				{
					
					// multiply the dirty image by the primary beam pattern at the longest wavelength, and divide the dirty image by the primary beam
					// pattern.
					float * devPrimaryBeamRatio = NULL;
					reserveGPUMemory( (void **) &devPrimaryBeamRatio, _param->BeamSize * _param->BeamSize * sizeof( float ),
									"reserving memory for the primary beam ratio on the device", __LINE__ );
					moveHostToDevice( (void *) devPrimaryBeamRatio, (void *) _hstPrimaryBeamRatioPattern,
									_param->BeamSize * _param->BeamSize * sizeof( float ),
									"copying primary beam ratio to the device", __LINE__ );

					// divide the dirty image by the primary beam ratio.
					devDivideImages<<< _gridSize2D, _blockSize2D >>>(	/* pOne = */ (float *) devDirtyImageGrid[ /* GPU = */ 0 ],
												/* pTwo = */ devPrimaryBeamRatio,
												/* pMask = */ NULL,
												/* pSizeOne = */ _param->ImageSize,
												/* pSizeTwo = */ _param->BeamSize,
												/* pInterpolate = */ true );
									
					// free memory.
					if (devPrimaryBeamRatio != NULL)
						cudaFree( (void *) devPrimaryBeamRatio );
				
				}

				// if we're using a-projection then we need to divide by the determinant of the Mueller matrix.
//				if (_param->AProjection == true)
//				{
						
//					float * devMuellerDeterminant = NULL;
//					reserveGPUMemory( (void **) &devMuellerDeterminant, _param->BeamSize * _param->BeamSize * sizeof( float ),
//									"reserving memory for the primary beam on the device", __LINE__ );
//					moveHostToDevice( (void *) devMuellerDeterminant, (void *) _hstData[ /* IMAGE = */ 0 ]->MuellerDeterminant,
//									_param->BeamSize * _param->BeamSize * sizeof( float ),
//									"copying primary beam to the device", __LINE__ );
									
//					bool * devMask = NULL;
//					if (phstMask != NULL)
//					{
//						reserveGPUMemory( (void **) &devMask, _param->ImageSize * _param->ImageSize * sizeof( bool ),
//										"reserving memory for the primary beam mask on the device", __LINE__ );
//						moveHostToDevice( (void *) devMask, (void *) phstMask, _param->ImageSize * _param->ImageSize * sizeof( bool ),
//										"copying primary-beam mask to the device", __LINE__ );
//					}

					// divide the dirty image by the determinant.
//					devDivideImages<<< _gridSize2D, _blockSize2D >>>(	/* pOne = */ (float *) devDirtyImageGrid[ /* GPU = */ 0 ],
//												/* pTwo = */ devMuellerDeterminant,
//												/* pMask = */ devMask,
//												/* pSizeOne = */ _param->ImageSize,
//												/* pSizeTwo = */ _param->BeamSize,
//												/* pInterpolate = */ true );
									
					// free memory.
//					if (devMuellerDeterminant != NULL)
//						cudaFree( (void *) devMuellerDeterminant );
//					if (devMask != NULL)
//						cudaFree( (void *) devMask );
				
//				}

				// create the dirty image.
				if (phstDirtyImage[ taylorTerm ] == NULL)
				{
					phstDirtyImage[ taylorTerm ] = (float *) malloc( _param->ImageSize * _param->ImageSize * sizeof( float ) );
					memset( phstDirtyImage[ taylorTerm ], 0, _param->ImageSize * _param->ImageSize * sizeof( float ) );
				}

				// copy the residual image into the mosaic (image-plane mosaicing), or the dirty image if we're not mosaicing.
				if (_param->ImagePlaneMosaic == true)
					addToMosaic(	/* phstMosaic = */ phstDirtyImage[ taylorTerm ],
							/* pdevImage = */ (float *) devDirtyImageGrid[ /* GPU = */ 0 ],
							/* phstImageID = */ image );
				else
					moveDeviceToHost( (void *) phstDirtyImage[ taylorTerm ], (void *) devDirtyImageGrid[ /* GPU = */ 0 ],
										_param->ImageSize * _param->ImageSize * sizeof( float ), "copying dirty image from device", __LINE__ );

				printf( "\n" );

			} // (_param->UvPlaneMosaic == false || image == _hstData.size() - 1)

		} // LOOP: taylorTerm

	} // LOOP: image

	// free memory.
	if (devDirtyImageGrid != NULL)
		for ( int gpu = 0; gpu < _param->NumGPUs; gpu++ )
			if (devDirtyImageGrid[ gpu ] != NULL)
			{
				if (_param->NumGPUs > 1)
					cudaSetDevice( _param->GPU[ gpu ] );
				cudaFree( (void *) devDirtyImageGrid[ gpu ] );
			}
	if (_param->NumGPUs > 1)
		cudaSetDevice( _param->GPU[ 0 ] );

	// reweight the mosaic if we're using mosaicing, and multiply by the primary beam pattern (for mosaicing and A-projection).
	if (devNormalisationPattern != NULL || devPrimaryBeamPattern != NULL)
		for ( int taylorTerm = 0; taylorTerm < _param->TaylorTerms; taylorTerm++ )
		{
			
			// create memory for the mosaic on the device.
			float * devDirtyImage = NULL;
			reserveGPUMemory( (void **) &devDirtyImage, _param->ImageSize * _param->ImageSize * sizeof( float ),
							"reserving memory for the mosaic on the device", __LINE__ );
			moveHostToDevice( (void *) devDirtyImage, (void *) phstDirtyImage[ taylorTerm ], _param->ImageSize * _param->ImageSize * sizeof( float ),
							"copying mosaic to the device", __LINE__ );
	
			// divide the mosaic by the normalisation pattern.
			setThreadBlockSize2D( _param->ImageSize, _param->ImageSize, _gridSize2D, _blockSize2D );
			if (devNormalisationPattern != NULL)				
				devDivideImages<<< _gridSize2D, _blockSize2D >>>(	/* pOne = */ devDirtyImage,
											/* pTwo = */ devNormalisationPattern,
											/* pMask = */ NULL,
											/* pSizeOne = */ _param->ImageSize,
											/* pSizeTwo = */ _param->BeamSize,
											/* pInterpolate = */ true );
			if (devPrimaryBeamPattern != NULL)
				devMultiplyImages<<< _gridSize2D, _blockSize2D >>>(	/* pOne = */ devDirtyImage,
											/* pTwo = */ devPrimaryBeamPattern,
											/* pMask = */ NULL,
											/* pSizeOne = */ _param->ImageSize,
											/* pSizeTwo = */ _param->BeamSize,
											/* pInterpolate = */ true );

			// move image back to the host.
			moveDeviceToHost( (void *) phstDirtyImage[ taylorTerm ], (void *) devDirtyImage, _param->ImageSize * _param->ImageSize * sizeof( float ),
							"copying mosaic to the host", __LINE__ );
						
			// free memory.	
			if (devDirtyImage != NULL)
				cudaFree( (void *) devDirtyImage );

		} // LOOP: taylorTerm

	// free memory.
	if (devNormalisationPattern != NULL)
		cudaFree( (void *) devNormalisationPattern );
	if (devPrimaryBeamPattern != NULL)
		cudaFree( (void *) devPrimaryBeamPattern );

} // generateDirtyImages

//
//	cottonSchwabClean()
//
//	CJS: 13/08/2018.
//
//	Perform a major/minor cycle clean.
//

bool cottonSchwabClean
			(						//
			float * pdevCleanBeam,				//
			float *** phstDirtyBeam,			//
			float *** phstDirtyImage,			// array of dirty images, indexed by Taylor term.
			bool * phstMask,				// the 20% beam-level mask
			char ** pCleanImageFilename,			//
			char ** pResidualImageFilename,		//
			char * pAlphaImageFilename			//
			)
{

	bool ok = true;

	// create memory to hold the current number of minor cycles.
	int numMinorCycles = 0;

	// copy the dirty beam to the device (taylor term 0, all visibilities).
	float * devDirtyBeam = NULL;
	reserveGPUMemory( (void **) &devDirtyBeam, _param->PsfSize * _param->PsfSize * sizeof( float ), "reserving device memory for dirty beam", __LINE__ );
	moveHostToDevice( (void *) devDirtyBeam, (void *) phstDirtyBeam[ /* TAYLOR TERM = */ 0 ][ /* MOSAIC COMPONENT = */ _hstNumDirtyBeams - 1 ],
				_param->PsfSize * _param->PsfSize * sizeof( float ), "copy dirty beam to the device", __LINE__ );

	// get the maximum sidelobe from the dirty beam, by constructing a mask from the clean beam.
	double maxSidelobe = getMaxValueFromMaskedImage(	/* pdevImage = */ devDirtyBeam,
								/* pdevMaskingImage = */ pdevCleanBeam,
								/* pMaskingValue = */ 0.1,
								/* pSize = */ _param->PsfSize );

	// free memory.
	if (devDirtyBeam != NULL)
		cudaFree( (void *) devDirtyBeam );

	// if we're using MFS deconvolution then we compute a mask based upon the peak residual / 10.
	bool * hstMfsMask = NULL;
	if (_param->Deconvolver == MFS)
	{

		// get maximum residual.
		double maxResidual = 0.0;
		for ( long int i = 0; i < (long int) _param->ImageSize * (long int) _param->ImageSize; i++ )
		{
			bool pixelOK = true;
			if (phstMask != NULL)
				pixelOK = (phstMask[ i ] == true);
			if (phstDirtyImage[ /* STOKES = */ 0 ][ /* TAYLOR_TERM = */ 0 ][ i ] > maxResidual && pixelOK == true)
				maxResidual = abs( phstDirtyImage[ /* STOKES = */ 0 ][ /* TAYLOR_TERM = */ 0 ][ i ] );
		}

		// create the MFS mask.
		hstMfsMask = (bool *) malloc( (long int) _param->ImageSize * (long int) _param->ImageSize * (long int) sizeof( bool ) );
		for ( long int i = 0; i < (long int) _param->ImageSize * (long int) _param->ImageSize; i++ )
			hstMfsMask[ i ] = phstMask[ i ] && (phstDirtyImage[ /* STOKES = */ 0 ][ /* TAYLOR_TERM = */ 0 ][ i ] >= maxResidual / 10.0);

	}

	// create a list of dirty image components.
//	VectorI * hstComponentListPos = (VectorI *) malloc( _param->MinorCycles * sizeof( VectorI ) );
//	double ** hstComponentListValue = (double **) malloc( _param->TaylorTerms * sizeof( double * ) );
	VectorI ** hstComponentListPos = (VectorI **) malloc( _param->NumStokesImages * sizeof( VectorI * ) );
	double *** hstComponentListValue = (double ***) malloc( _param->NumStokesImages * sizeof( double ** ) );
	for ( int stokes = 0; stokes < _param->NumStokesImages; stokes++ )
	{
		hstComponentListPos[ stokes ] = (VectorI *) malloc( _param->MinorCycles * sizeof( VectorI ) );
		hstComponentListValue[ stokes ] = (double **) malloc( _param->TaylorTerms * sizeof( double * ) );
		for ( int taylorTerm = 0; taylorTerm < _param->TaylorTerms; taylorTerm++ )
			hstComponentListValue[ stokes ][ taylorTerm ] = (double *) malloc( _param->MinorCycles * sizeof( double ) );
	}

	int * numComponents = (int *) malloc( _param->NumStokesImages * sizeof( int ) );
	for ( int stokes = 0; stokes < _param->NumStokesImages; stokes++ )
		numComponents[ stokes ] = 0;
		
	printf( "\nPerforming Cotton-Schwab Clean.....\n" );
	printf( "-----------------------------------\n\n" );
				
	char cleanType[ 10 ];
	if (_param->Deconvolver == MFS)
		strcpy( cleanType, "MFS" );
	else
		strcpy( cleanType, "Högbom" );

	printf( "Deconvolver: %s\n", cleanType );
	printf( "Threshold = %6.4e Jy\n", _param->Threshold );
	printf( "Cycle factor = %f\n", _param->CycleFactor );
	printf( "Minor cycles = %i\n\n", _param->MinorCycles );

	double bestResidual = 0.0;
	bool allCleaningStopped = false;
	int majorCycle = 0;
	while (allCleaningStopped == false)
	{
//if (majorCycle == 1)
//break;
		printf( "	Major cycle %i\n\n", majorCycle );

		// -------------------------------------------------------------------
		//
		// S T E P   1 :   H O G B O M   C L E A N
		//
		// -------------------------------------------------------------------

		// perform Hogbom cleaning on each mosaic image.
		int currentMinorCycles = numMinorCycles;
	
		double hogbomLimit = 0.0;
		double maxResidual = 0.0;
		if (allCleaningStopped == false)
		{
			
			printf( "		performing %s clean\n\n", cleanType );

			// get maximum residual.
			for ( int stokes = 0; stokes < _param->NumStokesImages; stokes++ )
				for ( long int i = 0; i < (long int) _param->ImageSize * (long int) _param->ImageSize; i++ )
					if (abs( phstDirtyImage[ stokes ][ /* TAYLOR_TERM = */ 0 ][ i ] ) > maxResidual)
						maxResidual = abs( phstDirtyImage[ stokes ][ /* TAYLOR_TERM = */ 0 ][ i ] );

			// check if the residuals are getting worse. we stop cleaning if we reach 1.5x the best residual.
//			if (maxResidual > bestResidual * 1.2 && majorCycle > 0)
//			{
//				printf( "		the maximum residual (%6.4e Jy) is getting worse", maxResidual );
//				printf( " (was previously %6.4e Jy). clean is not converging for %s and will stop\n", bestResidual, stokesImageName );
//				allCleaningStopped = true;
//			}
			
			// calculate Hogbom limit.
			hogbomLimit = _param->CycleFactor * maxSidelobe * maxResidual;
			
		} // (allCleaningStopped == false)
	
		if (allCleaningStopped == false)
		{

			// have we reached the required number of minor cycles ?
			if (numMinorCycles >= _param->MinorCycles)
			{
				printf( "		no available clean cycles. %s clean will stop\n", cleanType );
				allCleaningStopped = true;
			}
			
		} // (allCleaningStopped == false)
			
		if (allCleaningStopped == false)
		{
		
			if (maxResidual < bestResidual || majorCycle == 0)
				bestResidual = maxResidual;

			if (_param->Threshold < hogbomLimit)
				printf( "                cleaning down to %6.4e Jy (max sidelobe: %8.6f mJy, max residual: %6.4e Jy)\n", hogbomLimit,
															maxSidelobe * 1000.0, maxResidual );
			else
			{
				hogbomLimit = _param->Threshold;
				printf( "                cleaning down to required stopping threshold of %6.4e Jy\n", hogbomLimit );
			}

			// and create device memory for these images.
//			if (_param->Deconvolver == MFS)
//			{
//{

//char r0Filename[100];
//sprintf( r0Filename, "r0" );
//_hstCasacoreInterface.WriteCasaImage( r0Filename, _param->ImageSize, _param->ImageSize, _param->OutputRA, _param->OutputDEC, _param->CellSize, hstR0, CONST_C / _hstAverageWavelength[ 0 ], NULL );

//}

//			}

			// do Hogbom clean.
			if (_param->Deconvolver == HOGBOM)
			{

				// create memory for the dirty image on the device, but only if we're able to clean it on the device.
				float * devDirtyImage = NULL;
				reserveGPUMemory( (void **) &devDirtyImage, _param->NumStokesImages * _param->ImageSize * _param->ImageSize * sizeof( float ),
							"reserving device memory for dirty image (cleaning)", __LINE__ );
				for ( int stokes = 0; stokes < _param->NumStokesImages; stokes++ )
					moveHostToDevice( (void *) (devDirtyImage + (stokes * _param->ImageSize * _param->ImageSize)),
								(void *) phstDirtyImage[ stokes ][ /* TAYLOR_TERM = */ 0 ],
								_param->ImageSize * _param->ImageSize * sizeof( float ), "copying dirty image to the device", __LINE__ );

				// perform a Hogbom clean until we reach the noise limit.
				hogbomClean(	/* pMinorCycle = */ &numMinorCycles,
						/* pHogbomLimit = */ hogbomLimit,
						/* phstDirtyBeam = */ phstDirtyBeam[ /* TAYLOR_TERM = */ 0 ],
						/* pdevDirtyImage = */ devDirtyImage,
						/* phstComponentListPos = */ hstComponentListPos,
						/* phstComponentListValue = */ hstComponentListValue,
						/* pComponentListItems = */ numComponents,
						/* phstPrimaryBeam = */ _hstPrimaryBeamPattern );

				// free memory.
				if (devDirtyImage != NULL)
					cudaFree( (void *) devDirtyImage );

			} // (_param->Deconvolver == HOGBOM)

			// do MFS clean.
			if (_param->Deconvolver == MFS)
			{

				// create memory for the dirty image on the device.
				float ** devDirtyImage = (float **) malloc( _param->TaylorTerms * sizeof( float * ) );
				for ( int t = 0; t < _param->TaylorTerms; t++ )
				{
					devDirtyImage[ t ] = NULL;
					reserveGPUMemory( (void **) &devDirtyImage[ t ], _param->NumStokesImages * _param->ImageSize * _param->ImageSize * sizeof( float ),
								"reserving device memory for dirty image (cleaning)", __LINE__ );
					for ( int stokes = 0; stokes < _param->NumStokesImages; stokes++ )
						moveHostToDevice( (void *) (devDirtyImage[ t ] + (stokes * _param->ImageSize * _param->ImageSize)),
									(void *) phstDirtyImage[ stokes ][ /* TAYLOR_TERM = */ t ],
									_param->ImageSize * _param->ImageSize * sizeof( float ),
														"copying dirty image to the device", __LINE__ );
				}

					// perform a MFS deconvolution until we reach the noise level.
//					mfsDeconvolve(	/* pMinorCycle = */ &numMinorCycles,
//  need to convert					/* phstMask = */ phstMask,
//  pdevDirtyBeam to phstDirtyBeam			/* pHogbomLimit = */ hogbomLimit,
//							/* pdevDirtyImage = */ devDirtyImage,
//							/* pdevDirtyBeam = */ pdevDirtyBeam,
//							/* phstComponentListPos = */ hstComponentListPos[ stokes ],
//							/* phstComponentListValue = */ hstComponentListValue[ stokes ],
//							/* pComponentListItems = */ &numComponents[ stokes ] );

				// free memory.
				if (devDirtyImage != NULL)
				{
					for ( int t = 0; t < _param->TaylorTerms; t++ )
						if (devDirtyImage[ t ] != NULL)
							cudaFree( (void *) devDirtyImage[ t ] );
					free( (void *) devDirtyImage );
				}

			} // (_param->Deconvolver == MFS)

			printf( "                %i %s clean iterations performed up to Major cycle %i\n", numMinorCycles, cleanType, majorCycle );

		} // (allCleaningStopped == false)

		// if there were no additional minor cycles then stop cleaning.
		if (numMinorCycles == currentMinorCycles && allCleaningStopped == false)
		{
			printf( "		no new minor cycles performed. %s clean will stop.\n\n", cleanType );
			allCleaningStopped = true;
		}
		
		if (hogbomLimit <= _param->Threshold && allCleaningStopped == false)
		{
			printf( "		reached required stopping threshold. %s clean will stop\n\n", cleanType );
			allCleaningStopped = true;
		}

		// have we reached the required number of minor cycles ?
		if (numMinorCycles >= _param->MinorCycles && allCleaningStopped == false)
		{
			printf( "		reached maximum clean cycles (%i). %s clean will stop\n\n", _param->MinorCycles, cleanType );
			allCleaningStopped = true;
		}
		
		if (allCleaningStopped == false)
			printf( "\n" );

		// free data.
		for ( int stokes = 0; stokes < _param->NumStokesImages; stokes++ )
			for ( int t = 0; t < _param->TaylorTerms; t++ )
			{
				if (phstDirtyImage[ stokes ][ /* TAYLOR_TERM = */ t ] != NULL)
					free( (void *) phstDirtyImage[ stokes ][ /* TAYLOR_TERM = */ t ] );
				phstDirtyImage[ stokes ][ /* TAYLOR_TERM = */ t ] = NULL;
			}
		// LOOP: stokes

		// -------------------------------------------------------------------
		//
		// S T E P   2 :   C O N T R U C T   M O D E L   I M A G E   F O R
		//		   E A C H   T A Y L O R   T E R M   A N D
		//		   S T O K E S   I M A G E
		//
		// -------------------------------------------------------------------

		// hold model images and mosaics.
//		float ** hstMosaicModel = (float **) malloc( _param->TaylorTerms * sizeof( float * ) );
//		cufftComplex ** hstModelImageUV = (cufftComplex **) malloc( _param->TaylorTerms * sizeof( cufftComplex * ) );
		float *** hstMosaicModel = (float ***) malloc( _param->NumStokesImages * sizeof( float ** ) );
		cufftComplex *** hstModelImageUV = (cufftComplex ***) malloc( _param->NumStokesImages * sizeof( cufftComplex ** ) );
		for ( int stokes = 0; stokes < _param->NumStokesImages; stokes++ )
		{
		
			// create arrays for model images and model mosaics.
			hstMosaicModel[ stokes ] = (float **) malloc( _param->TaylorTerms * sizeof( float * ) );
			hstModelImageUV[ stokes ] = (cufftComplex **) malloc( _param->TaylorTerms * sizeof( cufftComplex * ) );

			for ( int taylorTerm = 0; taylorTerm < _param->TaylorTerms; taylorTerm++ )
			{

				// upload the component values to the device.
				double * devComponentValue = NULL;
				reserveGPUMemory( (void **) &devComponentValue, numComponents[ stokes ] * sizeof( double ),
															"reserving device memory for clean components", __LINE__ );
				moveHostToDevice( (void *) devComponentValue, (void *) hstComponentListValue[ stokes ][ taylorTerm ],
								numComponents[ stokes ] * sizeof( double ), "moving component list values to the device", __LINE__ );

				// upload the component positions to the device.
				VectorI * devComponentPos = NULL;
				reserveGPUMemory( (void **) &devComponentPos, numComponents[ stokes ] * sizeof( VectorI ),
													"reserving device memory for clean component positions", __LINE__ );
				moveHostToDevice( (void *) devComponentPos, (void *) hstComponentListPos[ stokes ], numComponents[ stokes ] * sizeof( VectorI ),
							"moving component list positions to the device", __LINE__ );

				// upload a single pixel as a gridding kernel.
				float * devKernel = NULL;
				float kernel = 1.0;
				reserveGPUMemory( (void **) &devKernel, 1 * sizeof( float ), "reserving device memory for the model image gridding kernel", __LINE__ );
				cudaMemcpy( devKernel, &kernel, sizeof( float ), cudaMemcpyHostToDevice );

				// create the model image on the device.
				float * devModelImage = NULL;
				reserveGPUMemory( (void **) &devModelImage, _param->ImageSize * _param->ImageSize * sizeof( float ),
																"creating memory for the model image", __LINE__ );
				zeroGPUMemory( devModelImage, _param->ImageSize * _param->ImageSize * sizeof( float ), "zeroing the model image on the device", __LINE__ );

				// grid the clean components to make a model image.
				gridComponents(	/* pdevGrid = */ devModelImage,
							/* pdevComponentValue = */ devComponentValue,
							/* phstSupportSize = */ 0,
							/* pdevKernel = */ devKernel,
							/* pdevGridPositions = */ devComponentPos,
							/* pComponents = */ numComponents[ stokes ],
							/* pSize = */ _param->ImageSize );

				// free memory.
				if (devKernel != NULL)
					cudaFree( (void *) devKernel );
				if (devComponentValue != NULL)
					cudaFree( (void *) devComponentValue );
				if (devComponentPos != NULL)
					cudaFree( (void *) devComponentPos );
				
//{

//	float * tmp = (float *) malloc( _param->ImageSize * _param->ImageSize * sizeof( float ) );
//	moveDeviceToHost( (void *) tmp, (void *) devModelImage, _param->ImageSize * _param->ImageSize * sizeof( float ), "copying model image to the host", __LINE__ );
	
	// save the model image.
//	char filename[ 100 ];
//	sprintf( filename, "%s-model-image-%i", _param->OutputPrefix, majorCycle );
//	_hstCasacoreInterface.WriteCasaImage(	/* pFilename = */ filename,
//						/* pWidth = */ _param->ImageSize,
//						/* pHeight = */ _param->ImageSize,
//						/* pRA = */ _param->OutputRA,
//						/* pDec = */ _param->OutputDEC,
//						/* pPixelSize = */ _param->CellSize,
//						/* pImage = */ tmp,
//						/* pFrequency = */ CONST_C / _hstData[ /* MOSAIC_COMPONENT = */ 0 ]->AverageWavelength,
//						/* pMask = */ phstMask,
//						/* pDirectionType = */ CasacoreInterface::J2000,
//						/* pStokesImages = */ 1 );
						
//	free( (void *) tmp );

//}

				// if we are image-plane mosaicing then store the mosaic.
				hstMosaicModel[ stokes ][ taylorTerm ] = NULL;
				hstModelImageUV[ stokes ][ taylorTerm ] = (cufftComplex *) malloc( _param->ImageSize * _param->ImageSize * sizeof( cufftComplex ) );
				if (_param->ImagePlaneMosaic == true)
				{
					hstMosaicModel[ stokes ][ taylorTerm ] = (float *) malloc( _param->ImageSize * _param->ImageSize * sizeof( float ) );
					moveDeviceToHost( (void *) hstMosaicModel[ stokes ][ taylorTerm ], (void *) devModelImage,
									_param->ImageSize * _param->ImageSize * sizeof( float ), "copying mosaic model to the host", __LINE__ );
				}	
				else
				{

					// since we are not doing an image-plane mosaic then we can normalise and FFT the model image here.
					// divide the model image by the image-domain PS-function (which will be reversed during degridding).
					setThreadBlockSize2D( _param->ImageSize, _param->ImageSize, _gridSize2D, _blockSize2D );
					devDivideImages<<< _gridSize2D, _blockSize2D >>>(	/* pOne = */ devModelImage,
												/* pTwo = */ _devImageDomainPSFunction,
												/* pMask = */ NULL,
												/* pSizeOne = */ _param->ImageSize,
												/* pSizeTwo = */ _param->PsfSize,
												/* pInterpolate = */ true );

					// FFT the model image into the UV domain.
					performFFT(	/* pdevGrid = */ (cufftComplex **) &devModelImage,
							/* pSize = */ _param->ImageSize,
							/* pFFTDirection = */ FORWARD,
							/* pFFTPlan = */ -1,
							/* pFFTType = */ F2C,
							/* pResizeArray = */ false );

					// copy this image into the model image array.
					moveDeviceToHost( (void *) hstModelImageUV[ stokes ][ taylorTerm ], (void *) devModelImage, 
								_param->ImageSize * _param->ImageSize * sizeof( cufftComplex ), "copying model image to the host", __LINE__ );

				} // (_param->ImagePlaneMosaic == false)

				// free memory.
				if (devModelImage != NULL)
					cudaFree( (void *) devModelImage );

			} // LOOP: taylorTerm
		
		} // LOOP: stokes

		// -------------------------------------------------------------------
		//
		// S T E P   3 :   D E G R I D   M O D E L   I M A G E S   T O   G E T
		//                 A   S E T   O F   M O D E L
		//                 V I S I B I L I T I E S
		//
		// -------------------------------------------------------------------

		for ( int image = 0; image < _hstData.size(); image++ )
		{

			// count total visibilities.
			long int totalVisibilities = 0;
			for ( int stageID = 0; stageID < _hstData[ image ]->Stages; stageID++ )
				totalVisibilities += _hstData[ image ]->NumVisibilities[ stageID ];

			if (_hstData.size() > 1)
				printf( "        processing mosaic component %i of %i.....", image + 1, (int) _hstData.size() );
			else
				printf( "        processing visibilities....." );
			printf( "(stages: %i, visibilities: %li)\n\n", _hstData[ image ]->Stages, totalVisibilities );
		
			for ( int stokes = 0; stokes < _param->NumStokesImages; stokes++ )
			{

				// if we are using image-plane mosaicing then copy the model image from where it is temporarily stored in the model image cache.
				if (_param->ImagePlaneMosaic == true)
				{

					// create the model image on the device.
					float * devModelImage = NULL;
					reserveGPUMemory( (void **) &devModelImage, _param->ImageSize * _param->ImageSize * sizeof( float ), "creating memory for the model image", __LINE__ );

					for ( int taylorTerm = 0; taylorTerm < _param->TaylorTerms; taylorTerm++ )
					{
		
						// clear the model image.
						zeroGPUMemory( devModelImage, _param->ImageSize * _param->ImageSize * sizeof( float ), "zeroing the model image on the device", __LINE__ );

						// get the image from the mosaic.
						extractFromMosaic(	/* pdevImage = */ devModelImage,
									/* phstMosaic = */ hstMosaicModel[ stokes ][ taylorTerm ],
									/* phstMask = */ phstMask,
									/* pImageID = */ image );

						// divide the model image by the deconvolution image.
						setThreadBlockSize2D( _param->ImageSize, _param->ImageSize, _gridSize2D, _blockSize2D );
						devDivideImages<<< _gridSize2D, _blockSize2D >>>(	/* pOne = */ devModelImage,
													/* pTwo = */ _devImageDomainPSFunction,
													/* pMask = */ NULL,
													/* pSizeOne = */ _param->ImageSize,
													/* pSizeTwo = */ _param->PsfSize,
													/* pInterpolate = */ true );

						// FFT the model image into the UV domain.
						performFFT(	/* pdevGrid = */ (cufftComplex **) &devModelImage,
								/* pSize = */ _param->ImageSize,
								/* pFFTDirection = */ FORWARD,
								/* pFFTPlan = */ -1,
								/* pFFTType = */ F2C,
								/* pResizeArray = */ false );

						// copy the model image to the host.
						moveDeviceToHost( (void *) hstModelImageUV[ stokes ][ taylorTerm ], (void *) devModelImage,
								_param->ImageSize * _param->ImageSize * sizeof( cufftComplex ), "copying model image to the host", __LINE__ );

					} // LOOP: taylorTerm

					// free memory.
					if (devModelImage != NULL)
						cudaFree( (void *) devModelImage );

				} // (_param->ImagePlaneMosaic == true)
			
			} // LOOP: stokes
				
			// print the support sizes.
			displaySupportSizes( /* pKernelCache = */ *_degriddingKernelCache[ image ] );

			// degrid the visibilities from these images, and subtract the model visibilities from the original visibilities.
			degridVisibilitiesForImage(	/* pMosaicID = */ image,
							/* phstModelImage = */ hstModelImageUV,
							/* pTotalVisibilities = */ totalVisibilities,
/* pFinalPass = */ false ); // allCleaningStopped );

		} // LOOP: image

		// free memory.
		if (hstModelImageUV != NULL)
		{
			for ( int stokes = 0; stokes < _param->NumStokesImages; stokes++ )
				if (hstModelImageUV[ stokes ] != NULL)
				{
					for ( int t = 0; t < _param->TaylorTerms; t++ )
						if (hstModelImageUV[ stokes ][ t ] != NULL)
							free( (void *) hstModelImageUV[ stokes ][ t ] );
					free( (void *) hstModelImageUV[ stokes ] );
				}
			free( (void *) hstModelImageUV );
		}

		// -------------------------------------------------------------------
		//
		// S T E P   4 :   C O N S T R U C T   N E W   D I R T Y   I M A G E
		//                 ( O R   I M A G E S ,   I F   W E ' R E
		//                 I M A G E - P L A N E   M O S A I C I N G )
		//
		// -------------------------------------------------------------------
		
		for ( int stokes = 0; stokes < _param->NumStokesImages; stokes++ )
			generateDirtyImages(	/* phstDirtyImage = */ phstDirtyImage[ stokes ],
						/* phstMask = */ phstMask,
						/* pVisibilityType = */ RESIDUAL,
						/* pStokes = */ stokes );

		// free memory.
		if (hstMosaicModel != NULL)
		{
			for ( int stokes = 0; stokes < _param->NumStokesImages; stokes++ )
				if (hstMosaicModel[ stokes ] != NULL)
				{
					for ( int t = 0; t < _param->TaylorTerms; t++ )
						if (hstMosaicModel[ stokes ][ t ] != NULL)
							free( (void *) hstMosaicModel[ stokes ][ t ] );
					free( (void *) hstMosaicModel[ stokes ] );
				}
			free( (void *) hstMosaicModel );
		}
		for ( int image = 0; image < _hstData.size(); image++ )
			_hstData[ image ]->FreeData( /* pWhatData = */ DATA_RESIDUAL_VISIBILITIES );

		// increment major cycle.
		majorCycle++;

	} // WHILE: allCleaningStopped == false

	// write residual images.
	for ( int taylorTerm = 0; taylorTerm < _param->TaylorTerms; taylorTerm++ )
	{
	
		float * tmpImage = NULL;
		if (_param->NumStokesImages == 1)
			tmpImage = phstDirtyImage[ /* STOKES = */ 0 ][ taylorTerm ];
		else
		{
			tmpImage = (float *) malloc( _param->NumStokesImages * _param->ImageSize * _param->ImageSize * sizeof( float ) );
			for ( int stokes = 0; stokes < _param->NumStokesImages; stokes++ )
				memcpy( &tmpImage[ stokes * _param->ImageSize * _param->ImageSize ], phstDirtyImage[ stokes ][ taylorTerm ], 
						_param->ImageSize * _param->ImageSize * sizeof( float ) );
		}
		
		// save the residual image (the tt0 image for MFS).
		_hstCasacoreInterface.WriteCasaImage(	/* pFilename = */ pResidualImageFilename[ taylorTerm ],
							/* pWidth = */ _param->ImageSize,
							/* pHeight = */ _param->ImageSize,
							/* pRA = */ _param->OutputRA,
							/* pDec = */ _param->OutputDEC,
							/* pPixelSize = */ _param->CellSize,
							/* pImage = */ tmpImage,
							/* pFrequency = */ CONST_C / _hstData[ /* MOSAIC_COMPONENT = */ 0 ]->AverageWavelength,
							/* pMask = */ phstMask,
							/* pDirectionType = */ CasacoreInterface::J2000,
							/* pStokesImages = */ _param->NumStokesImages );
							

		// free memory
		if (_param->NumStokesImages > 1 && tmpImage != NULL)
			free( (void *) tmpImage );
			
	}

	// -------------------------------------------------------------------
	//
	// S T E P   5 :   C O N S T R U C T   C L E A N   I M A G E S   B Y
	//                 G R I D D I N G   T H E   C L E A N
	//                 C O M P O N E N T S   O V E R   T H E
	//                 R E S I D U A L   I M A G E S
	//
	// -------------------------------------------------------------------


	for ( int stokes = 0; stokes < _param->NumStokesImages; stokes++ )
	{

		// upload the component values to the device.
		double ** devComponentValue = (double **) malloc( _param->TaylorTerms * sizeof( double ) );
		for ( int taylorTerm = 0; taylorTerm < _param->TaylorTerms; taylorTerm++ )
		{
			reserveGPUMemory( (void **) &devComponentValue[ taylorTerm ], numComponents[ stokes ] * sizeof( double ),
															"reserving device memory for clean components", __LINE__ );
			moveHostToDevice( (void *) devComponentValue[ taylorTerm ], (void *) hstComponentListValue[ stokes ][ taylorTerm ],
									numComponents[ stokes ] * sizeof( double ), "moving component list values to the device", __LINE__ );
		}

		// upload the grid positions to the device.
		VectorI * devComponentPos = NULL;
		reserveGPUMemory( (void **) &devComponentPos, numComponents[ stokes ] * sizeof( VectorI ),
														"reserving device memory for clean component positions", __LINE__ );
		moveHostToDevice( (void *) devComponentPos, (void *) hstComponentListPos[ stokes ], numComponents[ stokes ] * sizeof( VectorI ),
															"moving component list positions to the device", __LINE__ );

		// upload the clean beam as a gridding kernel. _hstCleanBeamSize holds the support of the non-zero portion of the clean beam.
		int cleanBeamSize = (_hstCleanBeamSize * 2) + 1;
		float * devKernel = NULL;
		reserveGPUMemory( (void **) &devKernel, cleanBeamSize * cleanBeamSize * sizeof( float ),
					"reserving device memory for the clean component gridding kernel", __LINE__ );

		// cut out the centre portion of the kernel.
		for ( int i = 0; i < cleanBeamSize; i++ )
			cudaMemcpy(	&devKernel[ i * cleanBeamSize ],
					&pdevCleanBeam[ ((i + _hstPsfY - _hstCleanBeamSize) * _param->PsfSize) + _hstPsfX - _hstCleanBeamSize ],
					cleanBeamSize * sizeof( float ),
					cudaMemcpyDeviceToDevice );

		//
		// We need to calculate clean images by moving the residual image into memory, and gridding the components for each Taylor term (in turn) on top.
		//
		for ( int taylorTerm = 0; taylorTerm < _param->TaylorTerms; taylorTerm++ )
		{

			// create memory for the clean image on the device.
			float * devCleanImage = NULL;
			reserveGPUMemory( (void **) &devCleanImage, _param->ImageSize * _param->ImageSize * sizeof( float ), "reserving device memory for the clean image", __LINE__ );

			// copy the dirty image to the device so that we include our residuals.
			moveHostToDevice( (void *) devCleanImage, (void *) phstDirtyImage[ stokes ][ taylorTerm ], _param->ImageSize * _param->ImageSize * sizeof( float ),
					"moving residual image to device", __LINE__ );

			// grid the clean components to make a clean image.
			gridComponents(	/* pdevGrid = */ devCleanImage,
					/* pdevComponentValue = */ devComponentValue[ taylorTerm ],
					/* phstSupportSize = */ _hstCleanBeamSize,
					/* pdevKernel = */ devKernel,
					/* pdevGridPositions = */ devComponentPos,
					/* pComponents = */ numComponents[ stokes ],
					/* pSize = */ _param->ImageSize );

			// get the image from the device.
			moveDeviceToHost( (void *) phstDirtyImage[ stokes ][ taylorTerm ], (void *) devCleanImage, _param->ImageSize * _param->ImageSize * sizeof( float ),
						"moving clean image to the host", __LINE__ );

			// free memory
			if (devCleanImage != NULL)
				cudaFree( (void *) devCleanImage );

		} // LOOP: taylorTerm

		// free memory.
		if (devKernel != NULL)
			cudaFree( (void *) devKernel );
		if (devComponentValue != NULL)
		{
			for ( int i = 0; i < _param->TaylorTerms; i++ )
				if (devComponentValue[ i ] != NULL)
					cudaFree( (void *) devComponentValue[ i ] );
			free( (void *) devComponentValue );
		}
		if (devComponentPos != NULL)
			cudaFree( (void *) devComponentPos );

	} // LOOP: stokes

	// free memory.
	if (hstComponentListPos != NULL)
	{
		for ( int stokes = 0; stokes < _param->NumStokesImages; stokes++ )
			if (hstComponentListPos[ stokes ] != NULL)
				free( (void *) hstComponentListPos[ stokes ] );
		free( (void *) hstComponentListPos );
	}
	if (hstComponentListValue != NULL)
	{
		for ( int stokes = 0; stokes < _param->NumStokesImages; stokes++ )
			if (hstComponentListValue[ stokes ] != NULL)
			{
				for ( int i = 0; i < _param->TaylorTerms; i++ )
					if (hstComponentListValue[ stokes ][ i ] != NULL)
						free( (void *) hstComponentListValue[ stokes ][ i ] );
			}
		free( (void *) hstComponentListValue );
	}

	// write clean images.
	for ( int taylorTerm = 0; taylorTerm < _param->TaylorTerms; taylorTerm++ )
	{
	
		float * tmpImage = NULL;
		if (_param->NumStokesImages == 1)
			tmpImage = phstDirtyImage[ /* STOKES = */ 0 ][ taylorTerm ];
		else
		{
			tmpImage = (float *) malloc( _param->NumStokesImages * _param->ImageSize * _param->ImageSize * sizeof( float ) );
			for ( int stokes = 0; stokes < _param->NumStokesImages; stokes++ )
				memcpy( &tmpImage[ stokes * _param->ImageSize * _param->ImageSize ], phstDirtyImage[ stokes ][ taylorTerm ], 
						_param->ImageSize * _param->ImageSize * sizeof( float ) );
		}
	
		// save the clean image.
		_hstCasacoreInterface.WriteCasaImage(	/* pFilename = */ pCleanImageFilename[ taylorTerm ],
							/* pWidth = */ _param->ImageSize,
							/* pHeight = */ _param->ImageSize,
							/* pRA = */ _param->OutputRA,
							/* pDec = */ _param->OutputDEC,
							/* pPixelSize = */ _param->CellSize,
							/* pImage = */ tmpImage,
							/* pFrequency = */ CONST_C / _hstData[ /* MOSAIC_COMPONENT = */ 0 ]->AverageWavelength,
							/* pMask = */ phstMask,
							/* pDirectionType = */ CasacoreInterface::J2000,
							/* pStokesImages = */ _param->NumStokesImages );
						

		// free memory
		if (_param->NumStokesImages > 1 && tmpImage != NULL)
			free( (void *) tmpImage );

	} // LOOP: taylorTerm

	//
	// We need to calculate the spectral index (alpha) image, which is tt1 / tt0.
	//
	if (_param->Deconvolver == MFS && _param->TaylorTerms > 1)
	{

		// divide tt1 by tt0.
		for ( int i = 0; i < _param->ImageSize * _param->ImageSize; i++ )
			if (phstDirtyImage[ /* STOKES = */ 0 ][ /* TAYLOR_TERM = */ 0 ][ i ] != 0.0)
				phstDirtyImage[ /* STOKES = */ 0 ][ /* TAYLOR_TERM = */ 0 ][ i ] =
					phstDirtyImage[ /* STOKES = */ 0 ][ /* TAYLOR_TERM = */ 1 ][ i ] / phstDirtyImage[ /* STOKES = */ 0 ][ /* TAYLOR_TERM = */ 0 ][ i ];

		// save the alpa image.
		_hstCasacoreInterface.WriteCasaImage(	/* pFilename = */ pAlphaImageFilename,
							/* pWidth = */ _param->ImageSize,
							/* pHeight = */ _param->ImageSize,
							/* pRA = */ _param->OutputRA,
							/* pDec = */ _param->OutputDEC,
							/* pPixelSize = */ _param->CellSize,
							/* pImage = */ phstDirtyImage[ /* STOKES = */ 0 ][ /* TAYLOR_TERM = */ 0 ],
							/* pFrequency = */ CONST_C / _hstData[ /* MOSAIC_COMPONENT = */ 0 ]->AverageWavelength,
							/* pMask = */ hstMfsMask,
							/* pDirectionType = */ CasacoreInterface::J2000,
							/* pStokesImages = */ 1 );

	}

	// free memory.
	if (hstMfsMask != NULL)
		free( (void *) hstMfsMask );
	if (numComponents != NULL)
		free( (void *) numComponents );

	// return something.
	return ok;

} // cottonSchwabClean

//
//	getErrorBetweenImageAndGaussianFit()
//
//	CJS: 09/04/2019
//
//	Measures the summed square error between an image and a 2D Gaussian fit.
//

double getErrorBetweenImageAndGaussianFit( float * pdevImage, double * pdevError, int pSizeOfFittingRegion, double pX, double pY,
						double pAngle, double pR1, double pR2, double pSize, double pNormalisation )
{
		
	// define the block/thread dimensions.
	setThreadBlockSize2D( pSizeOfFittingRegion, pSizeOfFittingRegion, _gridSize2D, _blockSize2D );
		
	// sum the error between the Gaussian fit and the actual data.
	devCalculateGaussianError<<< _gridSize2D, _blockSize2D >>>(	/* pImage = */ pdevImage,
									/* pError = */ pdevError,
									/* pSizeOfFittingRegion = */ pSizeOfFittingRegion,
									/* pCentreX = */ pX,
									/* pCentreY = */ pY,
									/* pAngle = */ pAngle,
									/* pR1 = */ pR1,
									/* pR2 = */ pR2,
									/* pImageSize = */ pSize,
									/* pNormalisation = */ pNormalisation );
					
	int itemsToAdd = pSizeOfFittingRegion * pSizeOfFittingRegion;
	int destinationPtr = itemsToAdd, sourcePtr = 0;
	while (itemsToAdd > 1)
	{
	
		// set a suitable thread and block size.
		int threads = (int) ceil( (double) itemsToAdd / 10.0 );
		int blocks = 1;
		setThreadBlockSize1D( &threads, &blocks );
					
		// sum the error between the Gaussian fit and the actual data.
		devSumDouble<<< blocks, threads >>>(	/* pValue = */ &pdevError[ sourcePtr ],
							/* pSum = */ &pdevError[ destinationPtr ],
							/* pItems = */ itemsToAdd );

		// decrease the items to add by a factor of 10.
		itemsToAdd = (int) ceil( (double) itemsToAdd / 10.0 );

		// swap the pointers.
		int tmp = sourcePtr;
		sourcePtr = destinationPtr;
		destinationPtr = tmp;

	}

	// get the error from the device.
	double error = 0;
	cudaMemcpy( &error, &pdevError[ sourcePtr ], sizeof( double ), cudaMemcpyDeviceToHost );

	// return something.
	return error;

} // getErrorBetweenImageAndGaussianFit

//
//	generateDirtyBeams()
//
//	CJS: 05/11/2015
//
//	Generate the dirty beams by FFT'ing the gridded data. We generate 2T - 1 beams for each MFS, where T is the number of Taylor terms.
//
//	According to Rau & Cornwell, 1990, we need to generate 2T - 1 spectral PSFs for MFS. this is because the psf is a function of t + q, where 0 <= t,q < T. This is not
//	the case for gridding the dirty image: there, the dirty image is a function of t alone (there's no q), so the loop is over the number of Taylor terms.
//	The function we generate for the psf is:
//
//		I_t,q^psf = SUM_nu omega_nu^(t+q).I_nu^psf
//
//	where omega_nu^t = [ (nu - vu_ref) / nu ]^t. So on the first iteration of the loop (t = 0) we don't apply the MFS weights at all.
//

void generateDirtyBeams( float *** phstDirtyBeam, char ** pFilename )
{

	cudaError_t err;

	cufftComplex ** devDirtyBeamGrid = (cufftComplex **) malloc( _param->NumGPUs * sizeof( cufftComplex * ) );

	// create memory for the psf on the device, and clear it.
	for ( int gpu = 0; gpu < _param->NumGPUs; gpu++ )
	{
		if (_param->NumGPUs > 1)
			cudaSetDevice( _param->GPU[ gpu ] );
		reserveGPUMemory( (void **) &devDirtyBeamGrid[ gpu ], _param->ImageSize * _param->ImageSize * sizeof( cufftComplex ), "declaring device memory for psf", __LINE__ );
	}
	if (_param->NumGPUs > 1)
		cudaSetDevice( _param->GPU[ 0 ] );

	// how many dirty beams are we generating? we need one for each mosaic componant, plus one extra for the 'summed' dirty beam.
	_hstNumDirtyBeams = _hstData.size();
	if (_param->UseMosaicing == true)
		_hstNumDirtyBeams++;

	for ( int taylorTerm = 0; taylorTerm < (_param->TaylorTerms * 2) - 1; taylorTerm++ )
	{

		printf( "gridding visibilities for psf" );
		if (_param->Deconvolver == MFS)
			printf( " (cross Taylor terms t+v=%i)", taylorTerm );
		printf( ".....\n\n" );

		// create some space for the images.		
		phstDirtyBeam[ taylorTerm ] = (float **) malloc( _hstNumDirtyBeams * sizeof( float * ) );

		// create the psf for ALL the mosaic components, and clear it.
		if (_param->UseMosaicing == true)
		{
			phstDirtyBeam[ taylorTerm ][ _hstNumDirtyBeams - 1 ] = (float *) malloc( _param->PsfSize * _param->PsfSize * sizeof( float ) );
			memset( phstDirtyBeam[ taylorTerm ][ _hstNumDirtyBeams - 1 ], 0, _param->PsfSize * _param->PsfSize * sizeof( float ) );
		}

		// loop over all the mosaic components.
		for ( int mosaicID = 0; mosaicID < _hstData.size(); mosaicID++ )
		{

			// count the total number of visibilities.
			long int totalVisibilities = 0;
			for ( int stageID = 0; stageID < _hstData[ mosaicID ]->Stages; stageID++ )
				totalVisibilities += _hstData[ mosaicID ]->NumVisibilities[ stageID ];

			if (_hstData.size() > 1)
				printf( "        processing mosaic component %i of %i.....", mosaicID + 1, (int) _hstData.size() );
			else
				printf( "        processing visibilities....." );
			printf( "(stages: %i, visibilities: %li)\n", _hstData[ mosaicID ]->Stages, totalVisibilities );

			long int visibilitiesProcessed = 0;

			// clear the grids on each GPU.
			for ( int gpu = 0; gpu < _param->NumGPUs; gpu++ )
			{
				if (_param->NumGPUs > 1)
					cudaSetDevice( _param->GPU[ gpu ] );
				zeroGPUMemory( (void *) devDirtyBeamGrid[ gpu ], _param->ImageSize * _param->ImageSize * sizeof( cufftComplex ),
								"zeroing the dirty beam on the device", __LINE__ );
			}
			if (_param->NumGPUs > 1)
				cudaSetDevice( _param->GPU[ 0 ] );
					
			// print the support sizes.
			displaySupportSizes( /* pKernelCache = */ *_psfKernelCache[ mosaicID ] );

			// uncache the data for this mosaic image (if we have more than one mosaic component).
			for ( int stageID = 0; stageID < _hstData[ mosaicID ]->Stages; stageID++ )
			{

				// which Stokes image are we making ?
//				int stokesIdx = _param->Stokes;
//				if (_param->NumStokesImages == 1)
//					stokesIdx = 0;
				int stokesIdx = STOKES_I;

				// get the data from the file.
				if (_param->CacheData == true)
					_hstData[ mosaicID ]->UncacheData(	/* pStageID = */ stageID,
										/* pTaylorTerm = */ -1,
										/* pOffset = */ 0,
										/* pWhatData = */ DATA_GRID_POSITIONS | DATA_KERNEL_INDEXES | DATA_DENSITIES |
														 DATA_WEIGHTS,
										/* pStokes = */ stokesIdx );

				// if the number of visibilities is greater than some maximum then we are going to set a smaller batch size, and load these
				// visibilities in batches.
				int hstVisibilityBatchSize = 0;
				{
					long int nextBatchSize = _hstData[ mosaicID ]->NumVisibilities[ stageID ];
					if (nextBatchSize > _param->PREFERRED_VISIBILITY_BATCH_SIZE)
						nextBatchSize = _param->PREFERRED_VISIBILITY_BATCH_SIZE;
					hstVisibilityBatchSize = (int) nextBatchSize;
				}

				// create space for the unity (psf) visibilities, the density map, and the weights on the device.
				cufftComplex ** devBeamVisibility = (cufftComplex **) malloc( _param->NumGPUs * sizeof( cufftComplex * ) );
				int ** devDensityMap = (int **) malloc( _param->NumGPUs * sizeof( int * ) );
				float ** devWeight = NULL;
				if (_param->Weighting != NONE)
					devWeight = (float **) malloc( _param->NumGPUs * sizeof( float * ) );
				VectorI ** devGridPosition = (VectorI **) malloc( _param->NumGPUs * sizeof( VectorI * ) );
				int ** devKernelIndex = (int **) malloc( _param->NumGPUs * sizeof( int * ) );
				float ** devMfsWeight = (float **) malloc( _param->NumGPUs * sizeof( float * ) );
				for ( int gpu = 0; gpu < _param->NumGPUs; gpu++ )
				{
					if (_param->NumGPUs > 1)
						cudaSetDevice( _param->GPU[ gpu ] );
					reserveGPUMemory( (void **) &devBeamVisibility[ gpu ], hstVisibilityBatchSize * sizeof( cufftComplex ),
								"creating device memory for the psf visibilities", __LINE__ );
					reserveGPUMemory( (void **) &devDensityMap[ gpu ], hstVisibilityBatchSize * sizeof( int ),
								"declaring device memory for the density map", __LINE__ );
					reserveGPUMemory( (void **) &devGridPosition[ gpu ], hstVisibilityBatchSize * sizeof( VectorI ),
								"reserving device memory for grid positions", __LINE__ );
					reserveGPUMemory( (void **) &devKernelIndex[ gpu ], hstVisibilityBatchSize * sizeof( int ),
								"reserving device memory for kernel indexes", __LINE__ );
					if (_param->Weighting != NONE)
						reserveGPUMemory( (void **) &devWeight[ gpu ], hstVisibilityBatchSize * sizeof( float ),
									"declaring device memory for the weights", __LINE__ );
					if (taylorTerm > 0)
						reserveGPUMemory( (void **) &devMfsWeight[ gpu ], hstVisibilityBatchSize * sizeof( float ),
									"reserving device memory for mfs weights", __LINE__ );
					else
						devMfsWeight[ gpu ] = NULL;
				}
				if (_param->NumGPUs > 1)
					cudaSetDevice( _param->GPU[ 0 ] );

				// keep looping until we have loaded and gridded all visibilities.
				int batch = 0;
				long int hstCurrentVisibility = 0;

				while (hstCurrentVisibility < _hstData[ mosaicID ]->NumVisibilities[ stageID ])
				{
				
					KernelCache & kernelCache = *_psfKernelCache[ mosaicID ];
					int wPlanes = kernelCache.wPlanes;

					// count the number of visibilities in this batch.
					int visibilitiesInThisBatch = 0;
					for ( int wPlane = 0; wPlane < wPlanes; wPlane++ )
							for ( int gpu = 0; gpu < _param->NumGPUs; gpu++ )
								visibilitiesInThisBatch += kernelCache( /* CHANNEL = */ 0, /* STOKES = */ 0, /* STOKES = */ 0, wPlane ).
																	visibilities[ stageID ][ batch ][ gpu ];

					if (_hstData[ mosaicID ]->Stages > 1 || _hstData[ mosaicID ]->Batches[ stageID ] > 1)
						printf( "        gridding " );
					if (_hstData[ mosaicID ]->Stages > 1)
						printf( "host batch %i of %i", stageID + 1, _hstData[ mosaicID ]->Stages );
					if (_hstData[ mosaicID ]->Stages > 1 && _hstData[ mosaicID ]->Batches[ stageID ] > 1)
						printf( ", " );
					if (_hstData[ mosaicID ]->Batches[ stageID ] > 1)
						printf( "gpu batch %i of %i", batch + 1, _hstData[ mosaicID ]->Batches[ stageID ] );
					if (_hstData[ mosaicID ]->Stages > 1 || _hstData[ mosaicID ]->Batches[ stageID ] > 1)
					{
						int fractionDone = (int) round( (double) visibilitiesProcessed * 30.0 / (double) totalVisibilities );
						int fractionDoing = (int) round( (double) (visibilitiesProcessed + visibilitiesInThisBatch) * 30.0 /
													(double) totalVisibilities );
						printf( " [" );
						for ( int i = 0; i < fractionDone; i++ )
							printf( "*" );
						for ( int i = 0; i < (fractionDoing - fractionDone); i++ )
							printf( "+" );
						for ( int i = 0; i < (30 - fractionDoing); i++ )
							printf( "." );
						printf( "]\n\n" );
						visibilitiesProcessed += visibilitiesInThisBatch;
					}

					// maintain pointers to the next visibilities for each GPU.
					int * hstNumVisibilities = (int *) malloc( _param->NumGPUs * sizeof( int ) );
					for ( int gpu = 0; gpu < _param->NumGPUs; gpu++ )
						hstNumVisibilities[ gpu ] = 0;

					int cudaDeviceIndex = 0;
					for ( int wPlane = 0; wPlane < _param->WPlanes; wPlane++ )
					{

						int visibilitiesInKernelSet = kernelCache( /* CHANNEL = */ 0, /* STOKES = */ 0, /* STOKES = */ 0, wPlane ).
															visibilities[ stageID ][ batch ][ cudaDeviceIndex ];

						int lastGPU = cudaDeviceIndex;
						do
						{

							if (visibilitiesInKernelSet > 0)
							{

								// set the cuda device, and make sure nothing is running there already.
								if (_param->NumGPUs > 1)
									cudaSetDevice( _param->GPU[ cudaDeviceIndex ] );

								// upload the grid positions, kernel indexes, and density map to the device.
								moveHostToDevice( (void *) &devGridPosition[ cudaDeviceIndex ][ hstNumVisibilities[ cudaDeviceIndex ] ],
											(void *) &_hstData[ mosaicID ]->GridPosition[ hstCurrentVisibility ],
											visibilitiesInKernelSet * sizeof( VectorI ),
											"copying grid positions to the device", __LINE__ );
								moveHostToDevice( (void *) &devKernelIndex[ cudaDeviceIndex ][ hstNumVisibilities[ cudaDeviceIndex ] ],
											(void *) &_hstData[ mosaicID ]->KernelIndex[ hstCurrentVisibility ],
											visibilitiesInKernelSet * sizeof( int ),
											"copying kernel indexes to the device", __LINE__ );
								moveHostToDevice( (void *) &devDensityMap[ cudaDeviceIndex ][ hstNumVisibilities[ cudaDeviceIndex ] ],
											(void *) &_hstData[ mosaicID ]->DensityMap[ hstCurrentVisibility ],
											visibilitiesInKernelSet * sizeof( int ),
											"copying density map to the device", __LINE__ );

								// upload weights and Mfs weights to the device.
								if (_param->Weighting != NONE)
									moveHostToDevice( (void *) &devWeight[ cudaDeviceIndex ]
															[ hstNumVisibilities[ cudaDeviceIndex ] ],
												(void *) &_hstData[ mosaicID ]->Weight[ stokesIdx ]
																	[ hstCurrentVisibility ],
												visibilitiesInKernelSet * sizeof( float ),
												"copying weights to the device", __LINE__ );
								if (taylorTerm > 0)
									moveHostToDevice( (void *) &devMfsWeight[ cudaDeviceIndex ]
																[ hstNumVisibilities[ cudaDeviceIndex ] ],
												(void *) &_hstData[ mosaicID ]->MfsWeight[ taylorTerm - 1 ]
																		[ hstCurrentVisibility ],
												visibilitiesInKernelSet * sizeof( float ),
												"copying spectral beam weights to the device", __LINE__ );

								// get the next set of visibilities.
								hstCurrentVisibility += visibilitiesInKernelSet;
								hstNumVisibilities[ cudaDeviceIndex ] += visibilitiesInKernelSet;

							} // (visibilitiesInKernelSet > 0)
							cudaDeviceIndex++;
							if (cudaDeviceIndex == _param->NumGPUs)
								cudaDeviceIndex = 0;

						} while (cudaDeviceIndex != lastGPU);

					} // LOOP: wPlane
					if (_param->NumGPUs > 1)
						cudaSetDevice( _param->GPU[ 0 ] );

					// process the visibilities and apply the density map.
					for ( int gpu = 0; gpu < _param->NumGPUs; gpu++ )
						if (hstNumVisibilities[ gpu ] > 0)
						{

							if (_param->NumGPUs > 1)
								cudaSetDevice( _param->GPU[ gpu ] );

							// set all the visibilities to (1, 0). these are the visibilities used for generating the dirty beam.
							int threads = hstNumVisibilities[ gpu ];
							int blocks = 1;
							setThreadBlockSize1D( &threads, &blocks );

							// update the real part of each visibility to 1.
							devUpdateComplexArray<<< blocks, threads>>>( devBeamVisibility[ gpu ], hstNumVisibilities[ gpu ], 1, 0 );
							err = cudaGetLastError();
							if (err != cudaSuccess)
								printf( "error building visibilities for psf (%s)\n", cudaGetErrorString( err ) );

							// if this is the zeroth-order PSF then we need to multiply each visibility by the density map because
							// there are many visibilities with the same grid location and kernel. if this is not the zeroth-order psf then
							// we multiply each visibility by the sum of the mfs weights (i.e. [(lambda_ref / lambda) - 1]^t).
							if (taylorTerm == 0)
							{

								// apply the density map - multiply all the visibilities by the value of the density map at that
								// position.
								devMultiplyArrays<<< blocks, threads >>>(	/* pOne = */ devBeamVisibility[ gpu ],
														/* pTwo = */ devDensityMap[ gpu ],
														/* pSize = */ hstNumVisibilities[ gpu ] );
								err = cudaGetLastError();
								if (err != cudaSuccess)
									printf( "error applying the density map to the beam visibilities (%s)\n",
												cudaGetErrorString( err ) );

							}
							else
							{

								// apply the mfs weights - multiply all the visibilities by the mfs weights.
								devMultiplyArrays<<< blocks, threads >>>(	/* pOne = */ devBeamVisibility[ gpu ],
														/* pTwo = */ devMfsWeight[ gpu ],
														/* pSize = */ hstNumVisibilities[ gpu ] );
								err = cudaGetLastError();
								if (err != cudaSuccess)
									printf( "error building visibilities for psf (Taylor term %i) (%s)\n", taylorTerm,
																	cudaGetErrorString( err ) );

							}

						}
					if (_param->NumGPUs > 1)
						cudaSetDevice( _param->GPU[ 0 ] );

					// free memory.
					if (hstNumVisibilities != NULL)
						free( (void *) hstNumVisibilities );

					// generate the uv coverage using the gridder.
					gridVisibilities(	/* pdevGrid = */ devDirtyBeamGrid,
								/* pStageID = */ stageID,
								/* pBatchID = */ batch,
								/* pdevVisibility = */ devBeamVisibility,
								/* pdevKernelIndex = */ devKernelIndex,
								/* pdevGridPositions = */ devGridPosition,
								/* pdevWeight = */ devWeight,
								/* pSize = */ _param->ImageSize,
								/* pNumGPUs = */ _param->NumGPUs,
								/* pStokesTo = */ STOKES_I,
								/* pStokesFrom = */ STOKES_I,
								/* phstKernelCache = */ kernelCache );

					// get the next batch of data.
					batch = batch + 1;

				} // (hstCurrentVisibility < visibilitiesInAPlane)

				// free memory.
				for ( int gpu = 0; gpu < _param->NumGPUs; gpu++ )
				{
					if (_param->NumGPUs > 1)
						cudaSetDevice( _param->GPU[ gpu ] );
					if (devBeamVisibility[ gpu ] != NULL)
						cudaFree( (void *) devBeamVisibility[ gpu ] );
					if (devDensityMap[ gpu ] != NULL)
						cudaFree( (void *) devDensityMap[ gpu ] );
					if (_param->Weighting != NONE)
						if (devWeight[ gpu ] != NULL)
							cudaFree( (void *) devWeight[ gpu ] );
					if (devGridPosition[ gpu ] != NULL)
						cudaFree( (void *) devGridPosition[ gpu ] );
					if (devKernelIndex[ gpu ] != NULL)
						cudaFree( (void *) devKernelIndex[ gpu ] );
					if (devMfsWeight[ gpu ] != NULL)
						cudaFree( (void *) devMfsWeight[ gpu ] );
				}
				if (_param->NumGPUs > 1)
					cudaSetDevice( _param->GPU[ 0 ] );
				if (devBeamVisibility != NULL)
					free( (void *) devBeamVisibility );
				if (devDensityMap != NULL)
					free( (void *) devDensityMap );
				if (devWeight != NULL)
					free( (void *) devWeight );
				if (devGridPosition != NULL)
					free( (void *) devGridPosition );
				if (devKernelIndex != NULL)
					free( (void *) devKernelIndex );
				if (devMfsWeight != NULL)
					free( (void *) devMfsWeight );

				// free the data.
				if (_param->CacheData == true)
					_hstData[ mosaicID ]->FreeData( /* pWhatData = */ DATA_GRID_POSITIONS | DATA_KERNEL_INDEXES | DATA_DENSITIES |
													DATA_WEIGHTS );

			} // LOOP: stageID

			// move all images to the same GPU and add them together.
			if (_param->NumGPUs > 1)
			{

				cufftComplex * hstTmpImage = (cufftComplex *) malloc( _param->ImageSize * _param->ImageSize * sizeof( cufftComplex ) );
				cufftComplex * devTmpImage = NULL;
				reserveGPUMemory( (void **) &devTmpImage, _param->ImageSize * _param->ImageSize * sizeof( cufftComplex ),
							"reserving GPU memory for the temporary gridded data", __LINE__ );

				int items = _param->ImageSize * _param->ImageSize;
				int stages = items / MAX_THREADS;
				if (items % MAX_THREADS != 0)
					stages++;

				for ( int gpu = 1; gpu < _param->NumGPUs; gpu++ )
				{

					// set gpu device, and move image to the host.
					cudaSetDevice( _param->GPU[ gpu ] );
					moveDeviceToHost( (void *) hstTmpImage, devDirtyBeamGrid[ gpu ], _param->ImageSize * _param->ImageSize * sizeof( cufftComplex ),
								"moving gridded data to the host", __LINE__ );
					cudaDeviceSynchronize();

					// set gpu device, and move image to the device.
					cudaSetDevice( _param->GPU[ 0 ] );
					moveHostToDevice( (void *) devTmpImage, (void *) hstTmpImage, _param->ImageSize * _param->ImageSize * sizeof( cufftComplex ),
								"moving gridded data to the device", __LINE__ );

					for ( int i = 0; i < stages; i++ )
					{

						// define the block/thread dimensions.
						int itemsThisStage = items - (i * MAX_THREADS);
						if (itemsThisStage > MAX_THREADS)
							itemsThisStage = MAX_THREADS;
						int threads = itemsThisStage;
						int blocks;
						setThreadBlockSize1D( &threads, &blocks );

						// add images together.
						devAddArrays<<< blocks, threads >>>(	/* pOne = */ &devDirtyBeamGrid[ /* GPU = */ 0 ][ /* CELL = */ i * MAX_THREADS ],
											/* pTwo = */ &devTmpImage[ /* CELL = */ i * MAX_THREADS ],
											/* pSize = */ itemsThisStage );

					}

				} // LOOP: gpu

				// free memory.
				if (hstTmpImage != NULL)
					free( (void *) hstTmpImage );
				if (devTmpImage != NULL)
					cudaFree( (void *) devTmpImage );

			} // (_param->NumGPUs > 1)

			// FFT the uv coverage to get the psf.
			performFFT(	/* pdevGrid = */ &devDirtyBeamGrid[ /* GPU = */ 0 ],
					/* pSize = */ _param->ImageSize,
					/* pFFTDirection = */ INVERSE,
					/* pFFTPlan = */ -1,
					/* pFFTType = */ C2F,
					/* pResizeArray = */ false );
	
			// define the block/thread dimensions.
			setThreadBlockSize2D( _param->ImageSize, _param->ImageSize, _gridSize2D, _blockSize2D );

			// divide the dirty beam by the image-domain prolate-spheroidal function to remove the anti-aliasing kernel.
			devDivideImages<<< _gridSize2D, _blockSize2D >>>(	/* pOne = */ (float *) devDirtyBeamGrid[ /* GPU = */ 0 ],
										/* pTwo = */ _devImageDomainPSFunction,
										/* pMask = */ NULL,
										/* pSizeOne = */ _param->ImageSize,
										/* pSizeTwo = */ _param->PsfSize,
										/* pInterpolate = */ true );

			// chop out the central portion of the image.
			if (_param->PsfSize < _param->ImageSize)
			{

				float * devtmpDirtyBeam = NULL;
				reserveGPUMemory( (void **) &devtmpDirtyBeam, _param->PsfSize * _param->PsfSize * sizeof( cufftComplex ),
								"reserving device memory for temporary psf", __LINE__ );

				// define the block/thread dimensions.
				setThreadBlockSize2D( _param->PsfSize, _param->PsfSize, _gridSize2D, _blockSize2D );

				// chop out the centre of the psf.
				devResizeImage<<< _gridSize2D, _blockSize2D >>>(	/* pNewImage = */ devtmpDirtyBeam,
											/* pOldImage = */ (float *) devDirtyBeamGrid[ /* GPU = */ 0 ],
											/* pNewSize = */ _param->PsfSize,
											/* pOldSize = */ _param->ImageSize );

				// copy back into dirty beam grid.
				cudaMemcpy( (void *) devDirtyBeamGrid[ /* GPU = */ 0 ], (void *) devtmpDirtyBeam, _param->PsfSize * _param->PsfSize * sizeof( float ),
							cudaMemcpyDeviceToDevice );
							
				if (devtmpDirtyBeam != NULL)
					cudaFree( (void *) devtmpDirtyBeam );

			}

			// store this psf.
			phstDirtyBeam[ taylorTerm ][ mosaicID ] = (float *) malloc( _param->PsfSize * _param->PsfSize * sizeof( float ) );
			moveDeviceToHost( (void *) phstDirtyBeam[ taylorTerm ][ mosaicID ], (void *) devDirtyBeamGrid[ /* GPU = */ 0 ],
						_param->PsfSize * _param->PsfSize * sizeof( float ), "moving psf to the host", __LINE__ );

			// add up the psf for ALL the mosaic components.
			if (_param->UseMosaicing == true)
				for ( int i = 0; i < _param->PsfSize * _param->PsfSize; i++ )
					phstDirtyBeam[ taylorTerm ][ _hstNumDirtyBeams - 1 ][ i ] +=
											phstDirtyBeam[ taylorTerm ][ mosaicID ][ i ];

		} // LOOP: mosaicID

	} // LOOP: taylorTerm

	// normalise the psf for each mosaic component.
	for ( int dirtyBeam = 0; dirtyBeam < _hstNumDirtyBeams; dirtyBeam++ )
	{

		// move the dirty beam to the device.
		moveHostToDevice( (void *) devDirtyBeamGrid[ /* GPU = */ 0 ], (void *) phstDirtyBeam[ /* TAYLOR_TERM = */ 0 ][ dirtyBeam ],
						_param->PsfSize * _param->PsfSize * sizeof( float ), "copying dirty beam to the device", __LINE__ );
				
		// get maximum pixel value.
		double * devMaxValue;
		reserveGPUMemory( (void **) &devMaxValue, MAX_PIXEL_DATA_AREA_SIZE * sizeof( double ), "declaring device memory for psf max pixel value", __LINE__ );
		
		// get the maximum value from this image.
		getMaxValue(	/* pdevImage = */ (float *) devDirtyBeamGrid[ /* GPU = */ 0 ],
				/* pdevMaxValue = */ devMaxValue,
				/* pWidth = */ _param->PsfSize,
				/* pHeight = */ _param->PsfSize,
				/* pUseAbsolute = */ false,
				/* pdevMask = */ NULL,
				/* pNumImages = */ 1 );

		// store the normalisation.
		double beamNormalisation = 1.0;
		moveDeviceToHost( (void *) &beamNormalisation, (void *) devMaxValue, sizeof( double ), "copying dirty beam normalisation from device", __LINE__ );

		// free memory.
		if (devMaxValue != NULL)
			cudaFree( devMaxValue );
	
		// set a suitable thread and block size.
		int threads = _param->PsfSize * _param->PsfSize;
		int blocks;
		setThreadBlockSize1D( &threads, &blocks );

		// loop over all the Taylor terms.
		for ( int taylorTerm = 0; taylorTerm < (_param->TaylorTerms * 2) - 1; taylorTerm++ )
		{

			// copy the psf to the device (not for Taylor term 0, because it's already there.
			if (taylorTerm > 0)
				moveHostToDevice( (void *) devDirtyBeamGrid[ /* GPU = */ 0 ], (void *) phstDirtyBeam[ taylorTerm ][ dirtyBeam ],
							_param->PsfSize * _param->PsfSize * sizeof( float ), "copying dirty beam to the device", __LINE__ );
	
			// normalise the psf so that the maximum value of the zeroth-order Taylor-term image is 1.
			devNormalise<<< blocks, threads >>>( (float *) devDirtyBeamGrid[ /* GPU = */ 0 ], beamNormalisation, _param->PsfSize * _param->PsfSize );

			// move the psf back to the host.
			moveDeviceToHost( (void *) phstDirtyBeam[ taylorTerm ][ dirtyBeam ], (void *) devDirtyBeamGrid[ /* GPU = */ 0 ],
						_param->PsfSize * _param->PsfSize * sizeof( float ), "copying dirty beam to the host", __LINE__ );

		} // LOOP: taylorTerm

	} // LOOP: dirtyBeam

	// free memory
	for ( int gpu = 0; gpu < _param->NumGPUs; gpu++ )
		if (devDirtyBeamGrid[ gpu ] != NULL)
		{
			cudaSetDevice( _param->GPU[ gpu ] );
			cudaFree( (void *) devDirtyBeamGrid[ gpu ] );
		}
	cudaSetDevice( _param->GPU[ 0 ] );
	if (devDirtyBeamGrid != NULL)
		free( (void *) devDirtyBeamGrid );

	// save the dirty beam.
	for ( int taylorTerm = 0; taylorTerm < (_param->TaylorTerms * 2) - 1; taylorTerm++ )
		_hstCasacoreInterface.WriteCasaImage(	/* pFilename = */ pFilename[ taylorTerm ],
							/* pWidth = */ _param->PsfSize,
							/* pHeight = */ _param->PsfSize,
							/* pRA = */ _param->OutputRA,
							/* pDec = */ _param->OutputDEC,
							/* pPixelSize = */ _param->CellSize,
							/* pImage = */ phstDirtyBeam[ taylorTerm ][ _hstNumDirtyBeams - 1 ],
							/* pFrequency = */ CONST_C / _hstData[ /* MOSAIC_COMPONENT = */ 0 ]->AverageWavelength,
							/* pMask = */ NULL,
							/* pDirectionType = */ CasacoreInterface::J2000,
							/* pStokesImages = */ 1 );

//for ( int beam = 0; beam < _hstNumDirtyBeams; beam++ )
//{
//char beamFilename[ 100 ];
//sprintf( beamFilename, "%s-tmp-dirty-beam-%i", _param->OutputPrefix, beam );
//_hstCasacoreInterface.WriteCasaImage(	/* pFilename = */ beamFilename,
//					/* pWidth = */ _param->PsfSize,
//					/* pHeight = */ _param->PsfSize,
//					/* pRA = */ _param->OutputRA,
//					/* pDec = */ _param->OutputDEC,
//					/* pPixelSize = */ _param->CellSize,
//					/* pImage = */ phstDirtyBeam[ 0 ][ beam ],
//					/* pFrequency = */ CONST_C / _hstData[ 0 ]->AverageWavelength,
//					/* pMask = */ NULL,
//					/* pDirectionType = */ CasacoreInterface::J2000,
//					/* pStokesImages = */ 1 );
//}

} // generateDirtyBeams

//
//	generateCleanBeam()
//
//	CJS: 05/11/2015
//
//	Generate the clean beam by fitting elliptical Gaussian to the dirty beam.
//

bool generateCleanBeam( float * pdevCleanBeam, float * pdevDirtyBeam, char * pFilename )
{
	
	struct Params
	{
		double angle;
		double r1;
		double r2;
		double x;
		double y;
	};
	
	bool ok = true;
	struct Params bestFit;
	struct Params testFit;
	double bestError = -1;
	cudaError_t err;
		
	printf( "\nconstructing clean beam.....\n" );
	printf( "----------------------------\n\n" );

	err = cudaGetLastError();
	if (err != cudaSuccess)
		printf( "unknown CUDA error in generateCleanBeam() [i] (%s)\n", cudaGetErrorString( err ) );
	
	// NOTE: the psf should already be in the host grid, so we're not going to bother getting it again.

	// initialise random seed.
	srand( time( NULL ) );

	// create a new memory area to hold the maximum pixel value.
	double * devMaxValue;
	reserveGPUMemory( (void **) &devMaxValue, MAX_PIXEL_DATA_AREA_SIZE * sizeof( double ), "declaring device memory for kernel max pixel value", __LINE__ );

	// get the peak value from the kernel.
	getMaxValue(	/* pdevImage = */ pdevDirtyBeam,
			/* pdevMaxValue = */ devMaxValue,
			/* pWidth = */ _param->PsfSize,
			/* pHeight = */ _param->PsfSize,
			/* pUseAbsolute = */ false,
			/* pdevMask = */ NULL,
			/* pNumImages = */ 1 );

	err = cudaGetLastError();
	if (err != cudaSuccess)
		printf( "unknown CUDA error on line %d (%s)\n", __LINE__, cudaGetErrorString( err ) );

	// find the closest pixel from the centre of the kernel that is negative.
	int closestNegativePixel = findCutoffPixel(	/* pdevKernel = */ pdevDirtyBeam,
							/* pdevMaxValue = */ devMaxValue,
							/* pSize = */ _param->PsfSize,
							/* pCutoffFraction = */ -1,
							/* pFindType = */ CLOSEST );

	// get the X and Y positions of the maximum pixel.
	double x = 0, y = 0;
	cudaMemcpy( &x, &devMaxValue[ MAX_PIXEL_X ], sizeof( double ), cudaMemcpyDeviceToHost );
	cudaMemcpy( &y, &devMaxValue[ MAX_PIXEL_Y ], sizeof( double ), cudaMemcpyDeviceToHost );
	
	// free memory.
	if (devMaxValue != NULL)
		cudaFree( (void *) devMaxValue );

	// set the fitting region size.
	int sizeOfFittingRegion = closestNegativePixel * 4;
	if (sizeOfFittingRegion > MAX_SIZE_FOR_PSF_FITTING)
		sizeOfFittingRegion = MAX_SIZE_FOR_PSF_FITTING;

	// set the initial seed for fitting.
	bestFit.r1 = closestNegativePixel / 2.0;
	bestFit.r2 = closestNegativePixel / 2.0;
	bestFit.x = x;
	bestFit.y = y;
	bestFit.angle = 0.0;

	// create some memory on the device for calculating the error on the Gaussian fit.
	double * devError = NULL;
	ok = ok && reserveGPUMemory( (void **) &devError, sizeOfFittingRegion * sizeOfFittingRegion * 2 * sizeof( double ),
						"creating device memory for Gaussian fit error", __LINE__ );
			
	printf( "        size of fitting region - %i, initial seed [x: %f, y: %f, r: %f]\n", sizeOfFittingRegion, bestFit.x, bestFit.y, bestFit.r1 );

	// measure the error for the initial seed.
	bestError = getErrorBetweenImageAndGaussianFit(	/* pdevImage = */ pdevDirtyBeam,
								/* pdevError = */ devError,
								/* pSizeOfFittingRegion = */ sizeOfFittingRegion,
								/* pX = */ bestFit.x,
								/* pY = */ bestFit.y,
								/* pAngle = */ bestFit.angle,
								/* pR1 = */ bestFit.r1,
								/* pR2 = */ bestFit.r2,
								/* pSize = */ _param->PsfSize,
								/* pNormalisation = */ 1.0 );

	const int NUM_SCALES = 6;
	const int NUM_OUTER_TESTS = 30;
	const int NUM_PARAMS = 3;
	const int NUM_INNER_TESTS = 10;
			
	bool updatedAnything;
	do
	{
		
		// reset flag.
		updatedAnything = false;
		
		// the scale loop will attempt to fit the parameters at six different scales. the scale of each change
		// divides by two with each iteration of the loop.
		struct Params scaleFit = { .angle = 0.5, .r1 = (double) closestNegativePixel, .r2 = (double) closestNegativePixel, .x = 0.0, .y = 0.0 };
		for ( int scale = 0; scale < NUM_SCALES; scale++ )
		{

			// we have N goes where we only change R, and then 15 goes where we change R1, R2 and angle.
			for ( int outerTest = 0; outerTest < NUM_OUTER_TESTS; outerTest++ )
		
				// the param loop will change each of the N fit parameters in turn.
				// param 0 = r1, param 1 = r2, param 2 = angle.
				for ( int param = 0; param < NUM_PARAMS; param++ )

					// some parameters are only fitted if the outerTest loop is in the correct range.
					if ((param == 0) || (param == 1 && outerTest >= (NUM_OUTER_TESTS / 2)) || (param == 2 && outerTest >= (NUM_OUTER_TESTS / 2)))
			
						// attempt N random changes to this parameter.
						for ( int test = 0; test < NUM_INNER_TESTS; test++ )
						{
			
							// randomly add or subtract some amount from this parameter, and re-assess the error.
							// here we produce a random double between -0.5 and 0.5.
							double change = (static_cast <double> (rand()) / static_cast <double> (RAND_MAX)) - 0.5;
							double change2 = (static_cast <double> (rand()) / static_cast <double> (RAND_MAX)) - 0.5;
				
							testFit = bestFit;
							switch (param)
							{
								case 0:
									testFit.r1 = bestFit.r1 + (change * scaleFit.r1);
									if (testFit.r1 < 0)
										testFit.r1 = 0;
									if (outerTest < (NUM_OUTER_TESTS / 2))
										testFit.r2 = testFit.r1 * (bestFit.r2 / bestFit.r1);
									break;
								case 1:
									testFit.r2 = bestFit.r2 + (change * scaleFit.r2);
									if (testFit.r2 < 0)
										testFit.r2 = 0;
									break;
								case 2:
									testFit.angle = bestFit.angle + (change * scaleFit.angle);
									break;
							}

							// calculate the error between the psf and the Gaussian fit.
							double error = getErrorBetweenImageAndGaussianFit(	/* pdevImage = */ pdevDirtyBeam,
														/* pdevError = */ devError,
														/* pSizeOfFittingRegion = */ sizeOfFittingRegion,
														/* pX = */ testFit.x,
														/* pY = */ testFit.y,
														/* pAngle = */ testFit.angle,
														/* pR1 = */ testFit.r1,
														/* pR2 = */ testFit.r2,
														/* pSize = */ _param->PsfSize,
														/* pNormalisation = */ 1.0 );

							// are these parameters an improvement?
							if (error < bestError)
							{
								bestFit = testFit;
								bestError = error;
								updatedAnything = true;
							}
			
						}
				
			// divide all the scales by 2 so that we are fitting finer values.
			scaleFit.angle = scaleFit.angle / 2.0;
			scaleFit.r1 = scaleFit.r1 / 2.0;
			scaleFit.r2 = scaleFit.r2 / 2.0;
			
		}
		
	} while (updatedAnything == true);

	printf( "        parameters fitted to psf: angle %f, r1 %f, r2 %f, x %f, y %f\n\n", bestFit.angle, bestFit.r1, bestFit.r2, bestFit.x, bestFit.y );
	
	if (ok == true)
	{
		
		// define the block/thread dimensions.
		setThreadBlockSize2D( _param->PsfSize, _param->PsfSize, _gridSize2D, _blockSize2D );

		// construct the clean beam on the device.
		devMakeBeam<<< _gridSize2D, _blockSize2D >>>(	/* pBeam = */ pdevCleanBeam,
								/* pAngle = */ (double) bestFit.angle,
								/* pR1 = */ (double) bestFit.r1,
								/* pR2 = */ (double) bestFit.r2,
								/* pX = */ (double) bestFit.x,
								/* pY = */ (double) bestFit.y,
								/* pSize = */ _param->PsfSize );
		err = cudaGetLastError();
		if (err != cudaSuccess)
		{
			printf( "error making the clean beam (%s)\n", cudaGetErrorString( err ) );
			ok = false;
		}

	}

	// make a note of the centre position of the psf - we'll need to for cleaning.
//	_hstPsfX = (int) floor( bestFit.x + 0.5 ); _hstPsfY = (int) floor( bestFit.y + 0.5 );
	_hstPsfX = (int) round( bestFit.x ); _hstPsfY = (int) round( bestFit.y );
	
	// copy psf parameters to constant memory.
	err = cudaMemcpyToSymbol( _devPsfX, &_hstPsfX, sizeof( int ) );
	if (err != cudaSuccess)
	{
		printf( "error copying psf x to device (%s)\n", cudaGetErrorString( err ) );
		ok = false;
	}
	err = cudaMemcpyToSymbol( _devPsfY, &_hstPsfY, sizeof( int ) );
	if (err != cudaSuccess)
	{
		printf( "error copying psf y to device (%s)\n", cudaGetErrorString( err ) );
		ok = false;
	}

	// clear memory.
	if (devError != NULL)
		cudaFree( (void *) devError );

	// crate some host memory for the clean beam.
	float * hstCleanBeam = (float *) malloc( (long int) _param->PsfSize * (long int) _param->PsfSize * sizeof( float ) );

	// copy the clean beam to the host.
	ok = ok && moveDeviceToHost( (void *) hstCleanBeam, (void *) pdevCleanBeam, _param->PsfSize * _param->PsfSize * sizeof( float ), "copying clean beam from device", __LINE__ );

	// work out the size of the clean beam by finding the furthest pixel above a threshold from the centre of the psf.
	_hstCleanBeamSize = 0;
	int pixel = 0;
	for ( int j = 0; j < _param->PsfSize; j++ )
		for ( int i = 0; i < _param->PsfSize; i++, pixel++ )
			if (hstCleanBeam[ pixel ] >= 0.000001)
			{
				if (abs( i - _hstPsfX ) > _hstCleanBeamSize)
					_hstCleanBeamSize = abs( i - _hstPsfX );
				if (abs( j - _hstPsfY ) > _hstCleanBeamSize)
					_hstCleanBeamSize = abs( j - _hstPsfY );
			}

	// ensure clean beam size is not too large. we set a limit because currently it's more complication to grid the clean beam because of the thread limit of 1024.
	if (_hstCleanBeamSize > 250)
		_hstCleanBeamSize = 250;

	// save the clean beam.
	_hstCasacoreInterface.WriteCasaImage(	/* pFilename = */ pFilename,
						/* pWidth = */ _param->PsfSize,
						/* pHeight = */ _param->PsfSize,
						/* pRA = */ _param->OutputRA,
						/* pDec = */ _param->OutputDEC,
						/* pPixelSize = */ _param->CellSize,
						/* pImage = */ hstCleanBeam,
						/* pFrequency = */ CONST_C / _hstData[ /* MOSAIC_COMPONENT = */ 0 ]->AverageWavelength,
						/* pMask = */ NULL,
						/* pDirectionType = */ CasacoreInterface::J2000,
						/* pStokesImages = */ 1 );

	// free memory.
	if (hstCleanBeam != NULL)
		free( (void *) hstCleanBeam );
	
	// return success flag.
	return ok;
	
} // generateCleanBeam

//
//	performUniformWeighting()
//
//	CJS: 27/11/2020
//
//	Looks through all the data we've got, calculates the weight of each visibility, and the average weight.
//	This routine can act on a single mosaic ID, or on all of them.
//
//	We are implementing the equation:
//
//		w_i =    omega_i
//			----------
//			W(u_i,v_i)
//
//	where omega_i is the variance = 1/sigma_i^2, and is also the weight from the measurement set. W(u,v) is the gridded weights.
//

double * performUniformWeighting( double ** phstTotalWeightPerCell )
{

	double * averageWeight = (double *) malloc( _param->NumStokesImages * sizeof( double ) );
	for ( int s = 0; s < _param->NumStokesImages; s++ )
		averageWeight[ s ] = 0.0;
	long int griddedVisibilities = 0;

	//
	// update the summed cell weighting using:
	//
	// 	total_weight[ cell ] =          total_weight[ cell ]
	//				---------------------------------------
	//				SUM( weight[ cell ] ) / number_of_cells
	//
	for ( int s = 0; s < _param->NumStokesImages; s++ )
	{
	
		for ( long int i = 0; i < (long int) _param->ImageSize * (long int) _param->ImageSize; i++ )
			averageWeight[ s ] += phstTotalWeightPerCell[ s ][ i ];
		averageWeight[ s ] /= (double) pow( (long int) _param->ImageSize, 2 );

		// normalise the gridded weights using the average weight.
		if (averageWeight[ s ] > 0.0)
			for ( long int i = 0; i < (long int) _param->ImageSize * (long int) _param->ImageSize; i++ )
				phstTotalWeightPerCell[ s ][ i ] /= averageWeight[ s ];

		// reset the average weight. we will compute it per visibility now.
		averageWeight[ s ] = 0.0;
	
	} // LOOP: s

	//
	// update the weight in each cell using:
	//
	//	weight[ cell ] = weight[ cell ] / total_weight[ cell ]
	//
	for ( int mosaicID = 0; mosaicID < _hstData.size(); mosaicID++ )
	{
		griddedVisibilities += _hstData[ mosaicID ]->GriddedVisibilities;
		for ( int stageID = 0; stageID < _hstData[ mosaicID ]->Stages; stageID++ )
		{

			// get the weights, densities and grid positions from the file for this stage.
			if (_param->CacheData == true)
				_hstData[ mosaicID ]->UncacheData(	/* pStageID = */ stageID,
									/* pTaylorTerm = */ -1,
									/* pOffset = */ 0,
									/* pWhatData = */ DATA_WEIGHTS | DATA_GRID_POSITIONS | DATA_DENSITIES,
									/* pStokes = */ -1 );

			// divide each weight by the total weight in that cell, and add up the weights so we can make an average.
			for ( long int i = 0; i < _hstData[ mosaicID ]->NumVisibilities[ stageID ]; i++ )
			{
				VectorI grid = _hstData[ mosaicID ]->GridPosition[ i ];
				if (grid.u >= 0 && grid.u < _param->ImageSize && grid.v >= 0 && grid.v < _param->ImageSize)
					for ( int s = 0; s < _param->NumStokesImages; s++ )
					{			
						_hstData[ mosaicID ]->Weight[ s ][ i ] /= phstTotalWeightPerCell[ s ][ (grid.v * _param->ImageSize) + grid.u ];
						averageWeight[ s ] += (double) _hstData[ mosaicID ]->Weight[ s ][ i ] * (double) _hstData[ mosaicID ]->DensityMap[ i ];
					}
			}

			// re-cache the weights and free the densities and grid positions for this stage.
			if (_param->CacheData == true)
			{
				_hstData[ mosaicID ]->CacheData(	/* pStageID = */ stageID,
									/* pTaylorTerm = */ -1,
									/* pWhatData = */ DATA_WEIGHTS );
				_hstData[ mosaicID ]->FreeData( /* pWhatData = */ DATA_DENSITIES | DATA_GRID_POSITIONS );
			}

		}
	}
	
	// calculate average weights by dividing by the number of gridded visibilities.
	for ( int s = 0; s < _param->NumStokesImages; s++ )
		averageWeight[ s ] /= (double) griddedVisibilities;

	// return something.
	return averageWeight;

} // performUniformWeighting

//
//	performRobustWeighting()
//
//	CJS: 27/11/2020
//
//	Looks through all the data we've got, calculates the weight of each visibility, and the average weight.
//	This routine can act on a single mosaic ID, or on all of them.
//
//	We are implementing the equation:
//
//		w_i =          omega_i
//			-----------------------
//			1 + W(u_i,v_i).f^2
//			    ----------
//			    meanWeight
//
//	where f^2 = (5.10^(-R))^2, meanWeight = SUM_i( W(u_i,v_i) ^ 2 ) / SUM_i( W(u_i,v_i) ), omega_i is the variance = 1/sigma_i^2, and is also the weight from the
//	measurement set. W(u,v) is the gridded weights.
//

double * performRobustWeighting( double ** phstTotalWeightPerCell )
{

	double * averageWeight = (double *) malloc( _param->NumStokesImages * sizeof( double ) );
	for ( int s = 0; s < _param->NumStokesImages; s++ )
		averageWeight[ s ] = 0.0;
	long int griddedVisibilities = 0;

	// we might be dealing with HUGE numbers, so we normalise our gridded weights using the maximum weight. we're happy to work with numbers between 0 and 1.
	// first, find the maximum gridded cell weight.
	double * maxWeight = (double *) malloc( _param->NumStokesImages * sizeof( double ) );
	for ( int s = 0; s < _param->NumStokesImages; s++ )
	{
	
		double totalWeightSquared = 0.0;
		double totalWeight = 0.0;
	
		maxWeight[ s ] = 0.0;
		for ( long int i = 0; i < (long int) _param->ImageSize * (long int) _param->ImageSize; i++ )
			if (phstTotalWeightPerCell[ s ][ i ] > maxWeight[ s ])
				maxWeight[ s ] = phstTotalWeightPerCell[ s ][ i ];

		// compute the average weight per cell.
		if (maxWeight[ s ] > 0.0)
			for ( long int i = 0; i < (long int) _param->ImageSize * (long int) _param->ImageSize; i++ )
			{
		
				// normalise by the maximum weight found.
				phstTotalWeightPerCell[ s ][ i ] /= maxWeight[ s ];

				// sum the weight and the weight squared.
				totalWeightSquared += pow( phstTotalWeightPerCell[ s ][ i ], 2 );
				totalWeight += phstTotalWeightPerCell[ s ][ i ];

			}

		// calculate average and f^2 parameter.
		if (totalWeight != 0.0)
		{
			double meanWeight = totalWeightSquared / totalWeight;
			for ( long int i = 0; i < (long int) _param->ImageSize * (long int) _param->ImageSize; i++ )
				phstTotalWeightPerCell[ s ][ i ] /= meanWeight;
		}
		
	} // LOOP: s

	double fSquared = pow( 5.0 * pow( 10.0, -_param->Robust ), 2 );
	for ( int mosaicID = 0; mosaicID < _hstData.size(); mosaicID++ )
	{

		griddedVisibilities += _hstData[ mosaicID ]->GriddedVisibilities;
		for ( int stageID = 0; stageID < _hstData[ mosaicID ]->Stages; stageID++ )
		{

			// get the grid positions, densities and weights.
			if (_param->CacheData == true)
				_hstData[ mosaicID ]->UncacheData(	/* pStageID = */ stageID,
									/* pTaylorTerm = */ -1,
									/* pOffset = */ 0,
									/* pWhatData = */ DATA_GRID_POSITIONS | DATA_WEIGHTS | DATA_DENSITIES,
									/* pStokes = */ -1 );

			// update the weight of each visibility using the original weight, the sum of weights in the cell, and the f^2 parameter. also, add
			// up the weights so we can construct an average.
			for ( long int i = 0; i < _hstData[ mosaicID ]->NumVisibilities[ stageID ]; i++ )
				if (	_hstData[ mosaicID ]->GridPosition[ i ].u >= 0 && _hstData[ mosaicID ]->GridPosition[ i ].u < _param->ImageSize &&
					_hstData[ mosaicID ]->GridPosition[ i ].v >= 0 && _hstData[ mosaicID ]->GridPosition[ i ].v < _param->ImageSize)
					for ( int s = 0; s < _param->NumStokesImages; s++ )
					{
						_hstData[ mosaicID ]->Weight[ s ][ i ] /=
									(1.0 + (phstTotalWeightPerCell[ s ][ (_hstData[ mosaicID ]->GridPosition[ i ].v * _param->ImageSize) +
														_hstData[ mosaicID ]->GridPosition[ i ].u ] * fSquared));
						averageWeight[ s ] += (double) _hstData[ mosaicID ]->Weight[ s ][ i ] * (double) _hstData[ mosaicID ]->DensityMap[ i ];
					}

			// re-cache the weights and free the densities and grid positions for this stage.
			if (_param->CacheData == true)
			{
				_hstData[ mosaicID ]->CacheData(	/* pStageID = */ stageID,
									/* pTaylorTerm = */ -1,
									/* pWhatData = */ DATA_WEIGHTS );
				_hstData[ mosaicID ]->FreeData( /* pWhatData = */ DATA_DENSITIES | DATA_GRID_POSITIONS );
			}

		}
	}
	
	// calculate average weights by dividing by the number of gridded visibilities.
	for ( int s = 0; s < _param->NumStokesImages; s++ )
		averageWeight[ s ] /= (double) griddedVisibilities;

	// return something.
	return averageWeight;

} // performRobustWeighting

//
//	performNaturalWeighting()
//
//	CJS: 16/12/2020
//
//	calculates the average weight.
//	This routine can act on a single mosaic ID, or on all of them.
//

double * performNaturalWeighting()
{

	double * averageWeight = (double *) malloc( _param->NumStokesImages * sizeof( double ) );
	for ( int s = 0; s < _param->NumStokesImages; s++ )
		averageWeight[ s ] = 0.0;
	long int griddedVisibilities = 0;

	// calculate the average cell weighting.
	for ( int mosaicID = 0; mosaicID < _hstData.size(); mosaicID++ )
	{
		griddedVisibilities += _hstData[ mosaicID ]->GriddedVisibilities;
		for ( int s = 0; s < _param->NumStokesImages; s++ )
			averageWeight[ s ] += (double) _hstData[ mosaicID ]->AverageWeight[ s ] * (double) _hstData[ mosaicID ]->GriddedVisibilities;
	}
	
	// calculate average weights by dividing by the number of gridded visibilities.
	for ( int s = 0; s < _param->NumStokesImages; s++ )
		averageWeight[ s ] /= (double) griddedVisibilities;

	// return something.
	return averageWeight;

} // performNaturalWeighting

//
//	buildMaskAndPrimaryBeamPattern()
//
//	CJS: 13/07/2021
//
//	Assemble an image mask, and for mosaics also a primary-beam pattern.
//

void buildMaskAndPrimaryBeamPattern( bool * phstMask, char * pPrimaryBeamPatternFilename )
{

	const double PB_MASK_THRESHOLD = 0.2;

	// hold the primary beam pattern mask at the size of the primary beam.
	bool * hstPrimaryBeamPatternMask = NULL;

	// set the mask to false.
	memset( phstMask, 0, (long int) _param->ImageSize * (long int) _param->ImageSize * sizeof( bool ) );
	
	double imageScale = (double) _param->ImageSize / (double) _param->BeamSize;

	// if we are not using mosaicing then generate a mask for the image based upon the primary beam.
	if (_param->UseMosaicing == false)
	{

		long int index = 0;
		for ( long j = 0; j < _param->ImageSize; j++ )
		{
			int beamJ = (int) floor( (double) j / imageScale );
			double fracJ = ((double) j / imageScale) - (double) beamJ;
			for ( int i = 0; i < _param->ImageSize; i++, index++ )
			{
				int beamI = (int) floor( (double) i / imageScale );
				double fracI = ((double) i / imageScale) - (double) beamI;
				float beam = interpolateBeam(	/* pBeam = */ _hstData[ /* MOSAIC_ID = */ 0 ]->PrimaryBeam,
								/* pBeamSize = */ _param->BeamSize,
								/* pI = */ beamI,
								/* pJ = */ beamJ,
								/* pFracI = */ fracI,
								/* pFracJ = */ fracJ );
				phstMask[ index ] = (beam >= PB_MASK_THRESHOLD);
			}
		}

	} // (_param->UseMosaicing == false)

	// if we are using mosaicing then we need to assemble a weighted image of the primary beam patterns, and build a mask from it.
	if (_param->UseMosaicing == true || _param->AProjection == true)
	{

		// the primary beam pattern.
		_hstPrimaryBeamPattern = (float *) malloc( (long int) _param->BeamSize * (long int) _param->BeamSize * sizeof( float ) );
		memset( (void *) _hstPrimaryBeamPattern, 0, (long int) _param->BeamSize * (long int) _param->BeamSize * sizeof( float ) );

//		// calculate the sum of the primary beam squared.
//		double maxValue = 0.0;
//		for ( int index = 0; index < _param->BeamSize * _param->BeamSize; index++ )
//		{

//			for ( int beam = 0; beam < _hstData.size(); beam++ )
//				_hstPrimaryBeamPattern[ index ] += pow( _hstData[ beam ]->PrimaryBeam[ index ], 2 );
//			_hstPrimaryBeamPattern[ index ] = sqrt( _hstPrimaryBeamPattern[ index ] );

//			// record the maximum pixel value for normalisation.
//			if (_hstPrimaryBeamPattern[ index ] > maxValue || index == 0)
//				maxValue = _hstPrimaryBeamPattern[ index ];

//		}
		
		// calculate the sum of the primary beam squared.
		double maxValue = 0.0;
		for ( int beam = 0; beam < _hstData.size(); beam++ )
		{
		
			// reproject the average primary beam to the maximum wavelength.
			float * reprojectedBeam = _hstData[ beam ]->ReprojectPrimaryBeam(	/* pBeamOutSize = */ _param->BeamSize,
												/* pBeamOutCellSize = */ _param->CellSize,
												/* pToRA = */ _param->OutputRA,
												/* pToDEC = */ _param->OutputDEC,
												/* pToWavelength = */ _hstData[ beam ]->MaximumWavelength,
												/* pVerbose = */ false );
												
			for ( int index = 0; index < _param->BeamSize * _param->BeamSize; index++ )
				_hstPrimaryBeamPattern[ index ] += pow( reprojectedBeam[ index ], 2 );
												
			// free memory.
			if (reprojectedBeam != NULL)
				free( (void *) reprojectedBeam );
		
		}
		for ( int index = 0; index < _param->BeamSize * _param->BeamSize; index++ )
		{
		
			_hstPrimaryBeamPattern[ index ] = sqrt( _hstPrimaryBeamPattern[ index ] );

			// record the maximum pixel value for normalisation.
			if (_hstPrimaryBeamPattern[ index ] > maxValue || index == 0)
				maxValue = _hstPrimaryBeamPattern[ index ];
			
		}

		// normalise the primary beam pattern.
		if (maxValue > 0.0)
			for ( long int i = 0; i < (long int) _param->BeamSize * (long int) _param->BeamSize; i++ )
				_hstPrimaryBeamPattern[ i ] /= maxValue;

		// build the primary-beam pattern mask at the size of the primary beam..
		hstPrimaryBeamPatternMask = (bool *) malloc( _param->BeamSize * _param->BeamSize * sizeof( bool ) );
		for ( int i = 0; i < _param->BeamSize * _param->BeamSize; i++ )
			hstPrimaryBeamPatternMask[ i ] = (_hstPrimaryBeamPattern[ i ] >= PB_MASK_THRESHOLD);
			
		if (_param->UseMosaicing == true)
		{

			// and now build the mask at the size of the output image.
			long int index = 0;
			for ( int j = 0; j < _param->ImageSize; j++ )
			{
				int beamJ = (int) floor( (double) j / imageScale );
				double fracJ = ((double) j / imageScale) - (double) beamJ;
				for ( int i = 0; i < _param->ImageSize; i++, index++ )
				{
					int beamI = (int) floor( (double) i / imageScale );
					double fracI = ((double) i / imageScale) - (double) beamI;
					float beam = interpolateBeam(	/* pBeam = */ _hstPrimaryBeamPattern,
									/* pBeamSize = */ _param->BeamSize,
									/* pI = */ beamI,
									/* pJ = */ beamJ,
									/* pFracI = */ fracI,
									/* pFracJ = */ fracJ );
					phstMask[ index ] = (beam >= PB_MASK_THRESHOLD);
				}
			}
		
		}

		// save the primary beam pattern.
		_hstCasacoreInterface.WriteCasaImage(	/* pFilename = */ pPrimaryBeamPatternFilename,
							/* pWidth = */ _param->BeamSize,
							/* pHeight = */ _param->BeamSize,
							/* pRA = */ _param->OutputRA,
							/* pDec = */ _param->OutputDEC,
							/* pPixelSize = */ _param->CellSize * (double) _param->ImageSize / (double) _param->BeamSize,
							/* pImage = */ _hstPrimaryBeamPattern,
							/* pFrequency = */ CONST_C / _hstData[ /* MOSAIC_COMPONENT = */ 0 ]->AverageWavelength,
							/* pMask = */ hstPrimaryBeamPatternMask,
							/* pDirectionType = */ CasacoreInterface::J2000,
							/* pStokesImages = */ 1 );
			
	} // (_param->UseMosaicing == true || _param->AProjection == true)

	// assemble a weighted image of the primary-beam ratios. this pattern is used to correct the dirty-image fluxes so that they are consistent
	// with a primary-beam pattern at the longest wavelength in the measurement set.
	{

		// the primary beam ratio.
		_hstPrimaryBeamRatioPattern = (float *) malloc( _param->BeamSize * _param->BeamSize * sizeof( float ) );
		memset( (void *) _hstPrimaryBeamRatioPattern, 0, _param->BeamSize * _param->BeamSize * sizeof( float ) );
		
		float * summedPrimaryBeamMaxWavelength = (float *) malloc( _param->BeamSize * _param->BeamSize * sizeof( float ) );
		memset( (void *) summedPrimaryBeamMaxWavelength, 0, _param->BeamSize * _param->BeamSize * sizeof( float ) );
		
		for ( int beam = 0; beam < _hstData.size(); beam++ )
		{
									
			// we need a reprojected primary beam and a reprojected primary beam at the maximum wavelength.
			float * tmp = NULL;
			tmp = _hstData[ beam ]->ReprojectPrimaryBeam(	/* pBeamOutSize = */ _param->BeamSize,
									/* pBeamOutCellSize = */ _param->CellSize,
									/* pToRA = */ _param->OutputRA,
									/* pToDEC = */ _param->OutputDEC,
									/* pToWavelength = */ _hstData[ beam ]->AverageWavelength,
									/* pVerbose = */ false );
			for ( int index = 0; index < _param->BeamSize * _param->BeamSize; index++ )
				_hstPrimaryBeamRatioPattern[ index ] += pow( tmp[ index ], 2 );
			tmp = _hstData[ beam ]->ReprojectPrimaryBeam(	/* pBeamOutSize = */ _param->BeamSize,
									/* pBeamOutCellSize = */ _param->CellSize,
									/* pToRA = */ _param->OutputRA,
									/* pToDEC = */ _param->OutputDEC,
									/* pToWavelength = */ _hstData[ beam ]->MaximumWavelength,
									/* pVerbose = */ false );
			for ( int index = 0; index < _param->BeamSize * _param->BeamSize; index++ )
				summedPrimaryBeamMaxWavelength[ index ] += pow( tmp[ index ], 2 );
				
			// free memory.
			free( (void *) tmp );
		
		}

		// calculate the primary-beam ratio pattern.
		for ( int index = 0; index < _param->BeamSize * _param->BeamSize; index++ )
			if (summedPrimaryBeamMaxWavelength[ index ] != 0.0)
			{
			
				_hstPrimaryBeamRatioPattern[ index ] = sqrt( _hstPrimaryBeamRatioPattern[ index ] ) / sqrt( summedPrimaryBeamMaxWavelength[ index ] );
				
				// as we'll be dividing the dirty image by this pattern we need to take out the small values. we do this by having the function
				// flatten out at a value of 0.2 instead of falling to lower values.
				_hstPrimaryBeamRatioPattern[ index ] = pow( pow( 0.2, 4 ) + pow( _hstPrimaryBeamRatioPattern[ index ], 4 ), 0.25 );
				
			}
			else
				_hstPrimaryBeamRatioPattern[ index ] = 0.0;
			
		// free memory.	
		free( (void *) summedPrimaryBeamMaxWavelength );
			
	}

	// if we are using mosaicing then we need to assemble a weighted image of the primary beam patterns in order that we can correct the final mosaic and remove
	// the weighting functions.
	if (_param->UseMosaicing == true)
	{

		// we need to assemble a weighted image of the primary beam patterns (the normalisation pattern), which corrects for the fact each image has been
		// weighted by its dirty beam.
		_hstNormalisationPattern = (float *) malloc( (long int) _param->BeamSize * (long int) _param->BeamSize * (long int) sizeof( float ) );
		memset( (void *) _hstNormalisationPattern, 0, (long int) _param->BeamSize * (long int) _param->BeamSize * (long int) sizeof( float ) );

		for ( int index = 0; index < _param->BeamSize * _param->BeamSize; index++ )
		{
		
			// add up normalisation factor. the normalisation factor is PB^2 because we need to correct for
			// the use of the PB as a weighting function whilst gridding, and also remove the effect of the PB which will naturally be in
			// our image.
			for ( int beam = 0; beam < _hstData.size(); beam++ )
				_hstNormalisationPattern[ index ] += pow( _hstData[ /* MOSAIC_ID = */ beam ]->PrimaryBeam[ index ], 2 );
				
			// since we'll be dividing the image by the normalisation pattern, we ensure that it doesn't fall below 0.2. this step makes sure that the masked areas
			// output the primary-beam cutoff aren't very large, and we can clean the whole image.
			_hstNormalisationPattern[ index ] = pow( pow( 0.2, 4 ) + pow( _hstNormalisationPattern[ index ], 4 ), 0.25 );
			
		}

	} // (_param->UseMosaicing == true)

	// free memory.
	if (hstPrimaryBeamPatternMask != NULL)
		free( (void *) hstPrimaryBeamPatternMask );

} // buildMaskAndPrimaryBeamPattern

//
//	addMosaicComponent()
//
//	CJS: 20/08/2021
//
//	Adds a new mosaic component object.
//

void addMosaicComponent()
{

	Data * newComponent = new Data(	/* pTaylorTerms = */ _param->TaylorTerms,
						/* pMosaicID = */ _hstData.size(),
						/* pWProjection = */ _param->WProjection,
						/* pAProjection = */ _param->AProjection,
						/* pWPlanes = */ _param->WPlanes,
						/* pPBChannels = */ _param->PBChannels,
						/* pCacheData = */ _param->CacheData,
						/* pStokes = */ _param->Stokes,
						/* pStokesImages = */ _param->NumStokesImages );
	_hstData.push_back( newComponent );
//	_hstData[ _hstData.size() - 1 ]->Create(	/* pTaylorTerms = */ _param->TaylorTerms,
//							/* pMosaicID = */ _hstData.size() - 1,
//							/* pWProjection = */ _param->WProjection,
//							/* pAProjection = */ _param->AProjection,
//							/* pWPlanes = */ _param->WPlanes,
//							/* pPBChannels = */ _param->PBChannels,
//							/* pCacheData = */ _param->CacheData,
//							/* pStokes = */ _param->Stokes,
//							/* pStokesImages = */ _param->NumStokesImages );
						
	// set up the pointers.
	if (_hstData.size() > 1)
		_hstData[ _hstData.size() - 2 ]->NextComponent = _hstData[ _hstData.size() - 1 ];

} // addMosaicComponent

//
//	generateKernelCache()
//
//	CJS: 21/02/2022
//
//	Generates a cache of kernels for gridding and degridding.
//

void generateKernelCache()
{

	const int OVERSAMPLED_BEAM_SIZE = 2048;

	printf( "\nGenerating kernel caches..............\n" );
	printf( "--------------------------------------\n\n" );
		
	for ( int mosaicID = 0; mosaicID < _hstData.size(); mosaicID++ )
	{
	
		printf( "mosaic component %i:\n\n", mosaicID );
		printf( "        get required beams:\n\n" );
									
		// we also need a reprojected primary beam (for mosaicing), and a reprojected primary beam at the maximum wavelength (for degridding without
		// A-projection). we reduce the output cell size by the oversampling factor because these beams will be used in kernel generation.
		float * hstPrimaryBeamAtMaxWavelength = NULL;
		if (_param->UseMosaicing == false && _param->AProjection == false)
		{
			printf( "                reprojecting primary beam at maximum wavelength of %6.4f mm\n        ", _hstData[ mosaicID ]->MaximumWavelength * 1000.0 );
			hstPrimaryBeamAtMaxWavelength = _hstData[ mosaicID ]->ReprojectPrimaryBeam(	/* pBeamOutSize = */ OVERSAMPLED_BEAM_SIZE,
													/* pBeamOutCellSize = */ _param->CellSize * (double) _param->Oversample,
													/* pToRA = */ _param->OutputRA,
													/* pToDEC = */ _param->OutputDEC,
													/* pToWavelength = */ _hstData[ mosaicID ]->MaximumWavelength,
													/* pVerbose = */ true );
			printf( "\n" );
		}
		
		// and reproject the primary beam for mosaicing.
		float * hstPrimaryBeamAverageWavelength = NULL;
		if (_param->UvPlaneMosaic == true)
		{
			printf( "                reprojecting primary beam at average wavelength of %6.4f mm\n        ", _hstData[ mosaicID ]->AverageWavelength * 1000.0 );
			hstPrimaryBeamAverageWavelength = _hstData[ mosaicID ]->ReprojectPrimaryBeam(	/* pBeamOutSize = */ OVERSAMPLED_BEAM_SIZE,
														/* pBeamOutCellSize = */ _param->CellSize *
																			(double) _param->Oversample,
														/* pToRA = */ _param->OutputRA,
														/* pToDEC = */ _param->OutputDEC,
														/* pToWavelength = */ _hstData[ mosaicID ]->AverageWavelength,
														/* pVerbose = */ true );
			printf( "\n" );
		}
	
		// build gridding kernel cache.
		printf( "        generating gridding " );
		if (_param->MinorCycles > 0)
			printf( "and degridding " );
		printf( "cache(s):\n" );

		// construct empty kernel cache.
		{
			KernelCache * newKernelCache = new KernelCache(	/* pPBChannels = */ 1, // cjs-mod (_param->AProjection == true ? _param->PBChannels : 1),
										/* pWPlanes = */ _param->WPlanes,
										/* pWProjection = */ _param->WProjection,
										/* pAProjection = */ false, // cjs-mod _param->AProjection,
										/* pUseMosaicing = */ _param->UseMosaicing,
										/* pUVPlaneMosaic = */ _param->UvPlaneMosaic,
										/* pData = */ _hstData[ mosaicID ],
										/* pBeamSize = */ OVERSAMPLED_BEAM_SIZE,
										/* pStokesProducts = */ _param->NumStokesImages,
										/* pStokes = */ _param->Stokes,
										/* pLeakageCorrection = */ false,
										/* pOversample = */ _param->Oversample,
										/* pGridDegrid = */ GRID,
										/* phstPrimaryBeamAtMaxWavelength = */ NULL,
										/* phstPrimaryBeamMosaicing = */ hstPrimaryBeamAverageWavelength,	// only used by mosaicing
										/* pCacheData = */ _param->CacheData );
			_griddingKernelCache.push_back( newKernelCache );
		}

//		float pbMaxValue = 0.0;

		// move primary beam at the average wavelength to the device.
//		float * devPrimaryBeamAverageWavelength = NULL;
//		if (_param->AProjection == true)
//		{					
					
			// get the maximum absolute value of the primary beam.
//			for ( int i = 0; i < OVERSAMPLED_BEAM_SIZE * OVERSAMPLED_BEAM_SIZE; i++ )
//				if (abs( hstPrimaryBeamAverageWavelength[ i ] ) > pbMaxValue)
//					pbMaxValue = abs( hstPrimaryBeamAverageWavelength[ i ] );

//			reserveGPUMemory( (void **) &devPrimaryBeamAverageWavelength, OVERSAMPLED_BEAM_SIZE * OVERSAMPLED_BEAM_SIZE * sizeof( float ),
//														"reserving device memory for primary beam", __LINE__ );
//			moveHostToDevice( (void *) devPrimaryBeamAverageWavelength, (void *) hstPrimaryBeamAverageWavelength,
//							OVERSAMPLED_BEAM_SIZE * OVERSAMPLED_BEAM_SIZE * sizeof( float ), "copying primary beam to the device", __LINE__ );
							
//		}

		// build degridding kernel cache.
		if (_param->MinorCycles > 0)
		{
			KernelCache * newKernelCache = new KernelCache(	/* pPBChannels = */ _param->PBChannels,
										/* pWPlanes = */ _param->WPlanes,
										/* pWProjection = */ _param->WProjection,
										/* pAProjection = */ _param->AProjection,
										/* pUseMosaicing = */ _param->UseMosaicing,
										/* pUVPlaneMosaic = */ _param->UvPlaneMosaic,
										/* pData = */ _hstData[ mosaicID ],
										/* pBeamSize = */ OVERSAMPLED_BEAM_SIZE,
										/* pStokesProducts = */ _param->NumStokesImages,
										/* pStokes = */ _param->Stokes,
										/* pLeakageCorrection = */ _param->LeakageCorrection,
										/* pOversample = */ _param->Oversample,
										/* pGridDegrid = */ DEGRID,
										/* phstPrimaryBeamAtMaxWavelength = */ hstPrimaryBeamAtMaxWavelength,
										/* phstPrimaryBeamMosaicing = */ NULL, // cjs-mod hstPrimaryBeamAverageWavelength,
										/* pCacheData = */ _param->CacheData );
			_degriddingKernelCache.push_back( newKernelCache );
		}

		// free memory.
		if (hstPrimaryBeamAtMaxWavelength != NULL)
			free( (void *) hstPrimaryBeamAtMaxWavelength );
		if (hstPrimaryBeamAverageWavelength != NULL)
			free( (void *) hstPrimaryBeamAverageWavelength );

		// loop over all the channels. we generate the gridding kernels only if A-projection is being used (because we don't need channel-specific gridding
		// kernels without A-projection), and we generate the degridding kernels if cleaning is required.
		for ( int pbChannel = 0; pbChannel < _param->PBChannels; pbChannel++ )
		{

//			if (_param->AProjection == true && _param->MinorCycles > 0)
			if (_param->MinorCycles > 0)
			{

				// reproject the Jones matrix to the right size and place for this channel.
				_hstData[ mosaicID ]->ReprojectJonesMatrix(	/* pPBChannel = */ pbChannel,
										/* pBeamOutSize = */ OVERSAMPLED_BEAM_SIZE,
										/* pBeamOutCellSize = */ _param->CellSize * (double) _param->Oversample );

				// generate the Mueller and inverse-Mueller matrices for this channel.
				if (_param->AProjection == true)
				{
				
					_hstData[ mosaicID ]->GenerateMuellerMatrix(	/* pPBChannel = */ pbChannel,
											/* pImageSize = */ OVERSAMPLED_BEAM_SIZE );

					// free the kernels we don't need. for Stokes I, for instance, we only need the first row of the Inverse Mueller matrix,
					// and the first column of the Mueller matrix.
//					_hstData[ mosaicID ]->FreeUnwantedMuellerMatrices( /* pStokes = */ _param->Stokes );

					// if we're not using leakage correction, then we need to free all the off-axis Mueller cells.
					if (_param->LeakageCorrection == false)
						_hstData[ mosaicID ]->FreeOffAxisMuellerMatrices();
					
				}

//if (pbChannel == 0)
//for ( int cell = 0; cell < 16; cell++ )
//{
//float * mueller = (float *) malloc( OVERSAMPLED_BEAM_SIZE * OVERSAMPLED_BEAM_SIZE * sizeof( float ) );
//for ( int i = 0; i < OVERSAMPLED_BEAM_SIZE * OVERSAMPLED_BEAM_SIZE; i++ )
//	mueller[ i ] = _hstData[ mosaicID ]->MuellerMatrix[ cell ][ i ].x;
//char filename[100];
//sprintf( filename, "mueller-%i", cell );
//_hstCasacoreInterface.WriteCasaImage( filename, OVERSAMPLED_BEAM_SIZE, OVERSAMPLED_BEAM_SIZE, _param->OutputRA,
//					_param->OutputDEC, _param->ImageSize * _param->CellSize * (double) _param->Oversample / (double) OVERSAMPLED_BEAM_SIZE, mueller,
//					CONST_C / _hstData[ 0 ]->AverageWavelength, NULL, CasacoreInterface::J2000, 1 );
//for ( int i = 0; i < OVERSAMPLED_BEAM_SIZE * OVERSAMPLED_BEAM_SIZE; i++ )
//	mueller[ i ] = _hstData[ mosaicID ]->InverseMuellerMatrix[ 0 ][ i ].x;
//sprintf( filename, "inv-mueller-0" );
//_hstCasacoreInterface.WriteCasaImage( filename, OVERSAMPLED_BEAM_SIZE, OVERSAMPLED_BEAM_SIZE, _param->OutputRA,
//					_param->OutputDEC, _param->CellSize, mueller, CONST_C / _hstData[ 0 ]->AverageWavelength, NULL, CasacoreInterface::J2000, 1 );
//if (mueller != NULL)
//	free( (void *) mueller );
//}

				// generate degridding kernels
				_degriddingKernelCache[ mosaicID ]->GenerateKernelCache( /* pPBChannel = */ pbChannel );
				
			} // (_param->MinorCycles > 0)

		} // LOOP: channel

		// free the Jones matrices and Mueller matrices if they haven't already been released.
		_hstData[ mosaicID ]->FreeJonesMatrices();
		_hstData[ mosaicID ]->FreeMuellerMatrices();

		// free device memory.
//		if (devPrimaryBeamAverageWavelength != NULL)
//			cudaFree( (void *) devPrimaryBeamAverageWavelength );

		// if we're not using A-projection then the gridding kernel is constructed separately.
// cjs-mod 		if (_param->AProjection == false)
			_griddingKernelCache[ mosaicID ]->GenerateKernelCache( /* pPBChannel = */ 0 );

		printf( "\n\n" );

		// and work out how the visibilities are going to be split between GPUs and GPU batches.
		_griddingKernelCache[ mosaicID ]->CountVisibilities(	/* pMaxBatchSize = */ _param->PREFERRED_VISIBILITY_BATCH_SIZE,
									/* pNumGPUs = */ _param->NumGPUs );

		// build gridding kernel cache.
		if (_param->MinorCycles > 0)
			_degriddingKernelCache[ mosaicID ]->CountVisibilities(	/* pMaxBatchSize = */ _param->PREFERRED_VISIBILITY_BATCH_SIZE,
											/* pNumGPUs = */ _param->NumGPUs );

		// construct kernels for psf gridding.
		if (_param->MinorCycles > 0)
		{

			printf( "        generating uv coverage:\n" );
			KernelCache * newKernelCache = new KernelCache(	/* pPBChannels = */ 1,
										/* pWPlanes = */ _param->WPlanes,
										/* pWProjection = */ _param->WProjection,
										/* pAProjection = */ false,
										/* pUseMosaicing = */ false,
										/* pUVPlaneMosaic = */ false,
										/* pData = */ _hstData[ mosaicID ],
										/* pBeamSize = */ -1,
										/* pStokesProducts = */ 1,
										/* pStokes = */ STOKES_I,
										/* pLeakageCorrection = */ false,
										/* pOversample = */ _param->Oversample,
										/* pGridDegrid = */ GRID,
										/* phstPrimaryBeamAtMaxWavelength = */ NULL,
										/* phstPrimaryBeamMosaicing = */ NULL,
										/* pCacheData = */ _param->CacheData );
			_psfKernelCache.push_back( newKernelCache );

			_psfKernelCache[ mosaicID ]->GenerateKernelCache( /* pPBChannel = */ 0 );

			// and work out how the visibilities are going to be split between GPUs and GPU batches.
			_psfKernelCache[ mosaicID ]->CountVisibilities(	/* pMaxBatchSize = */ _param->PREFERRED_VISIBILITY_BATCH_SIZE,
										/* pNumGPUs = */ _param->NumGPUs );
			
			printf( "\n\n" );

		} // (_param->MinorCycles > 0)

	} // LOOP: mosaicID

	// construct prolate-spheroidal kernel.
	printf( "        generating prolate-spheroidal function:\n" );
	_psKernelCache.Create(	/* pPBChannels = */ 1,
				/* pWPlanes = */ 1,
				/* pWProjection = */ false,
				/* pAProjection = */ false,
				/* pUseMosaicing = */ false,
				/* pUVPlaneMosaic = */ false,
				/* pData = */ NULL,
				/* pBeamSize = */ -1,
				/* pStokesProducts = */ 1,
				/* pStokes = */ STOKES_I,
				/* pLeakageCorrection = */ false,
				/* pOversample = */ 1,
				/* pGridDegrid = */ GRID,
				/* phstPrimaryBeamAtMaxWavelength = */ NULL,
				/* phstPrimaryBeamMosaicing = */ NULL,
				/* pCacheData = */ false );

	_psKernelCache.GenerateKernelCache( /* pPBChannel = */ 0 );
			
	printf( "\n" );

} // generateKernelCache

//
//	main()
//
//	CJS: 07/08/2015
//
//	Main processing.
//

int main( int pArgc, char ** pArgv )
{
	
	// read program arguments. we expect to see the program call (0), the input filename (1) and the output filename (2).
	if (pArgc != 2)
	{
		printf("Wrong number of arguments. I require the parameter filename.\n");
		return 1;
	}
	
	char * parameterFile = pArgv[ 1 ];

	// ----------------------------------------------------
	//
	// Get parameters and perform validation
	//
	// ----------------------------------------------------
	
	// get the parameters from file.
	_param->GetParameters( parameterFile );

	// check path name is valid.
	int length = strlen( _param->CacheLocation );
	if (_param->CacheLocation[ length - 1 ] != '/' && length > 0)
	{
		_param->CacheLocation[ length ] = '/';
		_param->CacheLocation[ length + 1 ] = '\0';
	}

	// check that a beam size and beam cell size have been specified.
	for ( int measurementSet = 0; measurementSet < _param->MeasurementSets; measurementSet++ )
	{
		if (_param->BeamPattern[ measurementSet ][0] != '\0' && _param->BeamInSize <= 0)
		{
			printf( "ERROR: A beam pattern file has been supplied, but the size of the beam is not specified\n" );
			break;
		}
		if (_param->BeamPattern[ measurementSet ][0] != '\0' && _param->BeamInCellSize < 0.0)
		{
			printf( "ERROR: A beam pattern file has been supplied, but the beam cell size has not been specified\n" );
			break;
		}
	}
	
	// check that we have a minimum of 1 primary-beam channel.
	if (_param->PBChannels < 1)
		_param->PBChannels = 1;
		
	// check that we haven't turned leakage correction on without A-projection.
	if (_param->AProjection == false && _param->LeakageCorrection == true)
		_param->LeakageCorrection = false;
		
	// turn on Stokes I, Q, U and V imaging if we're using A-projection and leakage correction.
	if (_param->AProjection == true && _param->LeakageCorrection == true)
		_param->Stokes = STOKES_ALL;
					
	// is this primary beam large enough? if not we need to scale it up.
	_param->BeamSize = _param->BeamInSize;
	if (_param->BeamSize < Parameters::BEAM_SIZE)
		_param->BeamSize = Parameters::BEAM_SIZE;

	// ----------------------------------------------------
	
	char ** outputDirtyBeamFilename = (char **) malloc( ((_param->TaylorTerms * 2) - 1) * sizeof( char * ) );
	char ** outputDirtyImageFilename = (char **) malloc( _param->TaylorTerms * sizeof( char * ) );
	char ** outputCleanImageFilename = (char **) malloc( _param->TaylorTerms * sizeof( char * ) );
	char ** outputResidualImageFilename = (char **) malloc( _param->TaylorTerms * sizeof( char * ) );
	char ** mfsExtension = (char **) malloc( ((_param->TaylorTerms * 2) - 1) * sizeof( char * ) );
	char outputCleanBeamFilename[ 100 ];
	char outputGriddedFilename[ 100 ];
	char outputDeconvolutionFilename[ 100 ];
	char outputPrimaryBeamPatternFilename[ 100 ];
	char outputAlphaImageFilename[ 100 ];
	for ( int t = 0; t < ((_param->TaylorTerms * 2) - 1); t++ )
	{

		mfsExtension[ t ] = (char *) malloc( 25 * sizeof( char ) );
		sprintf( mfsExtension[ t ], "-tt%i", t );

		outputDirtyBeamFilename[ t ] = (char *) malloc( 100 * sizeof( char ) );
		strcpy( outputDirtyBeamFilename[ t ], _param->OutputPrefix ); strcat( outputDirtyBeamFilename[ t ], Parameters::DIRTY_BEAM_EXTENSION );
		if (_param->Deconvolver == MFS)
			strcat( outputDirtyBeamFilename[ t ], mfsExtension[ t ] );

	}
	for ( int t = 0; t < _param->TaylorTerms; t++ )
	{

		outputDirtyImageFilename[ t ] = (char *) malloc( 100 * sizeof( char ) );
		strcpy( outputDirtyImageFilename[ t ], _param->OutputPrefix ); strcat( outputDirtyImageFilename[ t ], Parameters::DIRTY_IMAGE_EXTENSION );
		if (_param->Deconvolver == MFS)
			strcat( outputDirtyImageFilename[ t ], mfsExtension[ t ] );

		outputCleanImageFilename[ t ] = (char *) malloc( 100 * sizeof( char ) );
		strcpy( outputCleanImageFilename[ t ], _param->OutputPrefix ); strcat( outputCleanImageFilename[ t ], Parameters::CLEAN_IMAGE_EXTENSION );
		if (_param->Deconvolver == MFS)
			strcat( outputCleanImageFilename[ t ], mfsExtension[ t ] );

		outputResidualImageFilename[ t ] = (char *) malloc( 100 * sizeof( char ) );
		strcpy( outputResidualImageFilename[ t ], _param->OutputPrefix ); strcat( outputResidualImageFilename[ t ], Parameters::RESIDUAL_IMAGE_EXTENSION );
		if (_param->Deconvolver == MFS)
			strcat( outputResidualImageFilename[ t ], mfsExtension[ t ] );

	}
	strcpy( outputCleanBeamFilename, _param->OutputPrefix ); strcat( outputCleanBeamFilename, Parameters::CLEAN_BEAM_EXTENSION );
	strcpy( outputGriddedFilename, _param->OutputPrefix ); strcat( outputGriddedFilename, Parameters::GRIDDED_EXTENSION );
	strcpy( outputAlphaImageFilename, _param->OutputPrefix ); strcat( outputAlphaImageFilename, Parameters::ALPHA_EXTENSION );

	// free memory.
	if (mfsExtension != NULL)
	{
		for ( int t = 0; t < ((_param->TaylorTerms * 2) - 1); t++ )
			if (mfsExtension[ t ] != NULL)
				free( (void *) mfsExtension[ t ] );
		free( (void *) mfsExtension );
	}
	
	// how many Stokes images do we need to generate? this would normally be one (whichever one the user has asked for), but if we are using A-projection, and we
	// will also need to clean images, then we need to generate ALL Stokes images.
	_param->NumStokesImages = (_param->Stokes == STOKES_ALL ? 4 : 1);

	// count the GPUs.
	char * gpuString;
	char tmp[ 512 ];
	strcpy( tmp, _param->GPUParam );
	_param->NumGPUs = 0;
	while ((gpuString = strtok( (_param->NumGPUs > 0 ? NULL : tmp), "," )) != NULL)
		_param->NumGPUs++;

	// build a list of GPUs.
//	_param->GPU = (int *) malloc( 1 * sizeof( int ) );
//	_param->GPU[ 0 ] = atoi( _param->GPUParam );
	if (_param->NumGPUs == 0)
	{
		_param->NumGPUs = 1;
		_param->GPU = (int *) malloc( sizeof( int ) );
		_param->GPU[ 0 ] = 0;
	}
	else
	{
		_param->GPU = (int *) malloc( _param->NumGPUs * sizeof( int ) );
		int i = 0;
		while ((gpuString = strtok( (i > 0 ? NULL : _param->GPUParam), "," )) != NULL)
		{
			_param->GPU[ i ] = atoi( gpuString );
			i++;
		}
	}
	
	// get some properties from the device.
	cudaDeviceProp gpuProperties;
	cudaGetDeviceProperties( &gpuProperties, _param->GPU[ 0 ] );
	_maxThreadsPerBlock = gpuProperties.maxThreadsPerBlock;
	_warpSize = gpuProperties.warpSize;
	int * maxGridSize = gpuProperties.maxGridSize;
	_gpuMemory = (long int) gpuProperties.totalGlobalMem;

	printf( "\nGIMAGE\n" );
	printf( "======\n\n" );

	printf( "GPU properties:\n" );
	printf( "---------------\n\n" );

	printf( "using %i GPU(s): ", _param->NumGPUs );
	for ( int i = 0; i < _param->NumGPUs; i++ )
	{
		if (i > 0)
			printf( "," );
		printf( "%i", _param->GPU[ i ] );
	}
	printf( "\n\n" );

	// set the device.
	cudaSetDevice( _param->GPU[ 0 ] );

	printf( "Device #: %i\n", _param->GPU[ 0 ] );
	printf( "Name: %s\n", gpuProperties.name );
	printf( "Compute capability: %i.%i\n", gpuProperties.major, gpuProperties.minor );
	printf( "Memory: %li bytes\n", _gpuMemory );
	printf( "Max grid size: <%i, %i, %i>\n", maxGridSize[ 0 ], maxGridSize[ 1 ], maxGridSize[ 2 ] );
	printf( "Max threads per block: %i\n", _maxThreadsPerBlock );
	printf( "Threads per warp: %i\n\n", _warpSize );

	printf( "Processing parameter file: %s.....\n\n", parameterFile );

	printf( "Gridding properties:\n" );
	printf( "--------------------\n\n" );
	printf( "weighting: " );

	if (_param->Weighting == NATURAL)
		printf( "NATURAL\n" );
	else if (_param->Weighting == UNIFORM)
		printf( "UNIFORM\n" );
	else if (_param->Weighting == ROBUST)
		printf( "ROBUST (R = %4.2f)\n", _param->Robust );
	else
		printf( "NONE\n" );

	if (_param->WProjection == true)
		printf( "w-projection: ON, %i W planes\n", _param->WPlanes );
	else
		printf( "w-projection: OFF\n" );

	if (_param->AProjection == true)
	{
		printf( "a-projection: ON" );
		if (_param->LeakageCorrection == true)
			printf( " (with leakage correction)\n" );
		else
			printf( " (without leakage correction)\n" );
	}
	else
		printf( "a-projection: OFF\n" );
	printf( "primary-beam correction channels: %i\n", _param->PBChannels );
	printf( "oversampling factor: %i\n", _param->Oversample );
	printf( "polarisation: " );
	if (_param->Stokes == STOKES_I)
		printf( "STOKES_I" );
	else if (_param->Stokes == STOKES_Q)
		printf( "STOKES_Q" );
	else if (_param->Stokes == STOKES_U)
		printf( "STOKES_U" );
	else if (_param->Stokes == STOKES_V)
		printf( "STOKES_V" );
	else if (_param->Stokes == STOKES_ALL)
		printf( "STOKES_I, STOKES_Q, STOKES_U, STOKES_V" );
	printf( "\n" );
	printf( "kernel cutoff fraction: %f\n\n", _param->KernelCutoffFraction );
	printf( "phase position: <%f, %f>\n\n", _param->OutputRA, _param->OutputDEC );

	printf( "Cleaning properties:\n" );
	printf( "--------------------\n\n" );
	printf( "deconvolver: " );
	if (_param->Deconvolver == MFS)
		printf( "MFS (%i Taylor terms)\n", _param->TaylorTerms );
	else
		printf( "HÖGBOM\n" );
	printf( "clean cycles: %i\n", _param->MinorCycles );
	printf( "gain factor: %f\n", _param->LoopGain );
	printf( "threshold: %f μJy\n\n", _param->Threshold * 1000000 );

	// initialise the number of measurement sets to 1.
	if (_param->MeasurementSets == 0)
		_param->MeasurementSets = 1;

	// if we are using multiple measurement sets then we need to cache our data.
	if (_param->MeasurementSets > 1)
		_param->CacheData = true;

	// calculate the size of the dirty beam. this must be the largest even-sized image up to the dirty image size.
	_param->PsfSize = _param->ImageSize;

	// we don't want the psf to be larger than 2048 x 2048.
	if (_param->PsfSize > 2048)
		_param->PsfSize = 2048;

	printf( "the size of the psf will be %i x %i pixels\n", _param->PsfSize, _param->PsfSize );
	printf( "the size of the primary beam will be %i x %i pixels\n\n", Parameters::BEAM_SIZE, Parameters::BEAM_SIZE );

	// turn on mosaicing if we are using multi files. we currently restrict this software to EITHER assembling a mosaic from the various files of a single
	// measurement set, OR assembling the images from multi files, with the same FOV and phase centre, into a single image. The latter is used for the multi-beams
	// of a PAF.
	_param->UseMosaicing = (_param->MeasurementSets > 1);
	_param->UvPlaneMosaic = (_param->MeasurementSets > 1 && _param->MosaicDomain == UV);
	_param->ImagePlaneMosaic = (_param->MeasurementSets > 1 && _param->MosaicDomain == IMAGE);

	printf( "Image properties:\n" );
	printf( "-----------------\n\n" );
	printf( "Telescope: " );
	switch (_param->Telescope)
	{
		case UNKNOWN_TELESCOPE:	{ printf( "Unknown" ); break; }
		case ALMA:
		case ALMA_7M:
		case ALMA_12M:			{ printf( "ALMA" ); break; }
		case ASKAP:			{ printf( "ASKAP" ); break; }
		case EMERLIN:			{ printf( "E-MERLIN" ); break; }
		case VLA:			{ printf( "VLA" ); break; }
		case MEERKAT:			{ printf( "MEERKAT" ); break; }
	}
	printf( "\n" );

	// get the total system memory.
	struct sysinfo memInfo;
	sysinfo( &memInfo );

	printf( "\nHost properties:\n" );
	printf( "----------------\n\n" );
	printf( "total physical memory: %4.2f GB\n", (double) memInfo.totalram / 1073741824.0 );
	
	// calculate the size of each uv pixel.
	_param->UvCellSize = (1.0 / ((double) _param->ImageSize * (_param->CellSize / 3600.0) * (PI / 180.0)));
	
	printf( "\nKernel properties:\n" );
	printf( "------------------\n\n" );
	printf( "cell size = %f arcsec, %1.12f rad\n", _param->CellSize, (_param->CellSize / 3600) * (PI / 180) );
	printf( "uv cell size = %f\n\n", _param->UvCellSize );

	// build some of the output filenames.
	strcpy( outputDeconvolutionFilename, _param->OutputPrefix ); strcat( outputDeconvolutionFilename, Parameters::DECONVOLUTION_EXTENSION );
	strcpy( outputPrimaryBeamPatternFilename, _param->OutputPrefix ); strcat( outputPrimaryBeamPatternFilename, Parameters::PRIMARY_BEAM_PATTERN_EXTENSION );

	// for uniform or robust weighting we need to store the sum of weights in each cell.
	double ** hstTotalWeightPerCell = NULL;
	if (_param->Weighting == ROBUST || _param->Weighting == UNIFORM)
	{
		hstTotalWeightPerCell = (double **) malloc( _param->NumStokesImages * sizeof( double * ) );
		for ( int s = 0; s < _param->NumStokesImages; s++ )
		{
			hstTotalWeightPerCell[ s ] = (double *) malloc( (long int) _param->ImageSize * (long int) _param->ImageSize * (long int) sizeof( double ) );
			memset( hstTotalWeightPerCell[ s ], 0, (long int) _param->ImageSize * (long int) _param->ImageSize * (long int) sizeof( double ) );
		}
	}

	// --------------------------------------------------------------------------------------------
	//
	// p r o c e s s   m e a s u r e m e n t   s e t s
	//
	// --------------------------------------------------------------------------------------------

	// process the input measurement sets by loading the data and caching it.
	for ( int measurementSet = 0; measurementSet < _param->MeasurementSets; measurementSet++ )
	{

		// add a new data component.
		addMosaicComponent();

		// load this measurement set, creating one or more data objects (one for each mosaic component found).
		_hstData[ _hstData.size() - 1 ]->ProcessMeasurementSet(	/* pFileIndex = */ measurementSet,
										/* phstTotalWeightPerCell = */ hstTotalWeightPerCell,
										/* pData = */ _hstData );

	}
	
	// if the primary beam for any mosaic component has been generated then turn off leakage correction.
	if (_param->LeakageCorrection == true)
	{
		for ( int image = 0; image < _hstData.size(); image++ )
			if (_hstData[ image ]->LoadedPrimaryBeam == false)
				_param->LeakageCorrection = false;
		if (_param->LeakageCorrection == false)
			printf( "WARNING: the primary beam for one or more mosaic components has been generated, and therefore has no polarisation leakage patterns. turning off leakage correction\n\n" );
	}

	// update the weighting for uv mosaics.
	if (_param->UvPlaneMosaic == true)
	{
	
		double * hstTotalAverageWeight = (double *) malloc( _param->NumStokesImages * sizeof( double ) );
		
		if (_param->Weighting == UNIFORM)
			hstTotalAverageWeight = performUniformWeighting( /* phstTotalWeightPerCell = */ hstTotalWeightPerCell );

		if (_param->Weighting == ROBUST)
			hstTotalAverageWeight = performRobustWeighting( /* phstTotalWeightPerCell = */ hstTotalWeightPerCell );

		if (_param->Weighting == NATURAL)
			hstTotalAverageWeight = performNaturalWeighting();
			
		// update the average weight in all the mosaics.
		for ( int i = 0; i < _hstData.size(); i++ )
			memcpy( (void *) _hstData[ i ]->AverageWeight, (void *) hstTotalAverageWeight, _param->NumStokesImages * sizeof( double ) );
			
		// free memory.
		if (hstTotalAverageWeight != NULL)
			free( (void *) hstTotalAverageWeight );
			
	}

	// free memory.
	if (hstTotalWeightPerCell != NULL)
	{
		for ( int s = 0; s < _param->NumStokesImages; s++ )
			free( (void *) hstTotalWeightPerCell[ s ] );
		free( (void *) hstTotalWeightPerCell );
	}

	printf( "\nMosaicing..................\n" );
	printf( "---------------------------\n\n" );
	if (_param->UvPlaneMosaic == true)
		printf( "a UV-plane mosaic will be generated based upon %i mosaic components from %i measurement sets\n\n", (int) _hstData.size(), _param->MeasurementSets );
	else if (_param->ImagePlaneMosaic == true)
		printf( "an image-plane mosaic will be generated based upon %i mosaic components from %i measurement sets\n\n", (int) _hstData.size(), _param->MeasurementSets );
	else
		printf( "no mosaicing will be used. we will grid only one field from one measurement set\n\n" );

	// we need to count how many visibilities are being gridded with each mosaic component.
	if (_param->UseMosaicing == true)
	{

		// find the minimum number of gridded visibilities. we will use this figure for our normalisation pattern. Our beams with higher numbers of gridded
		// visibilities will be corrected for in the gridding kernel.
		long int minimumVisibilitiesInMosaic = 0;
		for ( int i = 0; i < _hstData.size(); i++ )
			if (i == 0 || _hstData[ i ]->GriddedVisibilities < minimumVisibilitiesInMosaic)
				minimumVisibilitiesInMosaic = _hstData[ i ]->GriddedVisibilities;

		// ensure the number of gridded visibilities is not zero.
		if (minimumVisibilitiesInMosaic == 0)
			minimumVisibilitiesInMosaic = 1;
			
		// update each data item with the minimum number of gridded visibilities per mosaic component.
		for ( int i = 0; i < _hstData.size(); i++ )
			_hstData[ i ]->MinimumVisibilitiesInMosaic = minimumVisibilitiesInMosaic;

	}

	// build an image mask, and for mosaicing also a primary beam pattern.
	bool * hstMask = (bool *) malloc( (long int) _param->ImageSize * (long int) _param->ImageSize * sizeof( bool ) );
	buildMaskAndPrimaryBeamPattern(	/* phstMask = */ hstMask,
						/* pPrimaryBeamPattern = */ outputPrimaryBeamPatternFilename );

	// if we're making an image-plane mosaic then we need to build a mask for each mosaic component.
	if (_param->ImagePlaneMosaic == true)
		for ( int image = 0; image < _hstData.size(); image++ )
			_hstData[ image ]->BuildComponentMask(	/* pPrimaryBeamPattern = */ _hstPrimaryBeamPattern,
									/* pCellSize = */ _param->CellSize,
									/* pOutputRA = */ _param->OutputRA,
									/* pOutputDEC = */ _param->OutputDEC,
									/* pBeamSize = */ _param->BeamSize );

//for (int cell = 0; cell < 16; cell++)
//{

//cufftComplex * tmp = (cufftComplex *) malloc( _param->BeamSize * _param->BeamSize * sizeof( cufftComplex ) );
//char filename[100];

//if (_hstData[ 0 ]->InverseMuellerMatrix != NULL)
//if (_hstData[ 0 ]->InverseMuellerMatrix[ cell ] != NULL)
//if (_hstData[ 0 ]->InverseMuellerMatrix[ cell ][ 0 ] != NULL)
//{
//	memcpy( tmp, _hstData[ 0 ]->InverseMuellerMatrix[ cell ][ 0 ], _param->BeamSize * _param->BeamSize * sizeof( cufftComplex ) );
//	for ( long int i = 0; i < _param->BeamSize * _param->BeamSize; i++ )
//		((float *) tmp)[ i ] = ((float *) tmp)[ i * 2 ];
//	sprintf( filename, "inv-mueller-%i", cell );
//	_hstCasacoreInterface.WriteCasaImage( filename, _param->BeamSize, _param->BeamSize, 0.0, 0.0, 1.0, (float *) tmp, 1.0, NULL, CasacoreInterface::J2000, 1 );
//}
//if (_hstData[ 0 ]->MuellerMatrix != NULL)
//if (_hstData[ 0 ]->MuellerMatrix[ cell ] != NULL)
//if (_hstData[ 0 ]->MuellerMatrix[ cell ][ 0 ] != NULL)
//{
//	memcpy( tmp, _hstData[ 0 ]->MuellerMatrix[ cell ][ 0 ], _param->BeamSize * _param->BeamSize * sizeof( cufftComplex ) );
//	for ( long int i = 0; i < _param->BeamSize * _param->BeamSize; i++ )
//		((float *) tmp)[ i ] = ((float *) tmp)[ i * 2 ];
//	sprintf( filename, "mueller-%i", cell );
//	_hstCasacoreInterface.WriteCasaImage( filename, _param->BeamSize, _param->BeamSize, 0.0, 0.0, 1.0, (float *) tmp, 1.0, NULL, CasacoreInterface::J2000, 1 );
//}
//free( tmp );
//}

	// --------------------------------------------------------------------------------------------
	//
	// g e n e r a t e   k e r n e l   c a c h e
	//
	// --------------------------------------------------------------------------------------------

	// generate a cache of kernels for gridding and degridding.
	generateKernelCache();

	struct timespec time1, time2;
	clock_gettime( CLOCK_REALTIME, &time1 );

	// --------------------------------------------------------------------------------------------
	//
	// g r i d   v i s i b i l i t i e s   f o r   d i r t y   i m a g e
	//
	// --------------------------------------------------------------------------------------------

	// according to Rau & Cornwell, 1990, we need to generate T spectral dirty images for MFS. the function we generate for the psf is:
	//
	//	I_t^dirty = SUM_nu omega_nu^t.I_nu^dirty
	//
	// where omega_nu^t = [ (nu - vu_ref) / nu ]^t. So on the first iteration of the loop (t = 0) we don't apply the MFS weights at all.

	printf( "\nGridding...................\n" );
	printf( "---------------------------\n\n" );
	
	// generate image of convolution function.
	generateImageOfConvolutionFunction( outputDeconvolutionFilename );

	// create the pointers to the dirty image(s).
//	float ** hstDirtyImage = (float **) malloc( _param->TaylorTerms * sizeof( float * ) );
	float *** hstDirtyImage = (float ***) malloc( _param->NumStokesImages * sizeof( float ** ) );
	for ( int stokes = 0; stokes < _param->NumStokesImages; stokes++ )
	{
		hstDirtyImage[ stokes ] = (float **) malloc( _param->TaylorTerms * sizeof( float * ) );
		for ( int taylorTerm = 0; taylorTerm < _param->TaylorTerms; taylorTerm++ )
			hstDirtyImage[ stokes ][ taylorTerm ] = NULL;
	}

	// generate dirty images for this Stokes parameter (one for each Taylor term).
	for ( int stokes = 0; stokes < _param->NumStokesImages; stokes++ )
		generateDirtyImages(	/* phstDirtyImage = */ hstDirtyImage[ stokes ],
					/* phstMask = */ hstMask,
					/* pVisibilityType = */ OBSERVED,
					/* pStokes = */ stokes );

	// save the dirty images.
	for ( int taylorTerm = 0; taylorTerm < _param->TaylorTerms; taylorTerm++ )
	{
	
		float * tmpImage = NULL;
		if (_param->NumStokesImages == 1)
			tmpImage = hstDirtyImage[ /* STOKES = */ 0 ][ taylorTerm ];
		else
		{
			tmpImage = (float *) malloc( _param->NumStokesImages * _param->ImageSize * _param->ImageSize * sizeof( float ) );
			for ( int stokes = 0; stokes < _param->NumStokesImages; stokes++ )
				memcpy( &tmpImage[ stokes * _param->ImageSize * _param->ImageSize ], hstDirtyImage[ stokes ][ taylorTerm ], 
						_param->ImageSize * _param->ImageSize * sizeof( float ) );
		}
		
		_hstCasacoreInterface.WriteCasaImage(	/* pFilename = */ outputDirtyImageFilename[ taylorTerm ],
							/* pWidth = */ _param->ImageSize,
							/* pHeight = */ _param->ImageSize,
							/* pRA = */ _param->OutputRA,
							/* pDec = */ _param->OutputDEC,
							/* pPixelSize = */ _param->CellSize,
							/* pImage = */ tmpImage,
							/* pFrequency = */ CONST_C / _hstData[ /* MOSAIC_COMPONENT = */ 0 ]->AverageWavelength,
							/* pMask = */ hstMask,
							/* pDirectionType = */ CasacoreInterface::J2000,
							/* pStokesImages = */ _param->NumStokesImages );
							
		if (_param->NumStokesImages > 1 && tmpImage != NULL)
			free( (void *) tmpImage );
			
	} // LOOP: taylorTerm

	clock_gettime( CLOCK_REALTIME, &time2 );
	printf( "--- time (setup gridding): (%f ms) ---\n\n", _setup );
	printf( "--- time (generate kernel): (%f ms) ---\n\n", _generateKernel );
	printf( "--- time (upload data): (%f ms) ---\n\n", _uploadData );
	printf( "--- time (other gridding1): (%f ms) ---\n\n", _otherGridding );
	printf( "--- time (gridding): (%f ms) ---\n\n", _gridding );
	printf( "--- time (total, inc fft): (%f ms) ---\n\n", getTime( time1, time2 ) );

	// --------------------------------------------------------------------------------------------
	//
	// c o t t o n - s c h w a b   c l e a n i n g
	//
	// --------------------------------------------------------------------------------------------

	// are we doing cleaning ?
	if (_param->MinorCycles > 0)
	{
	
		// create a list of pointers to the dirty beam(s) on the device.
		float *** hstDirtyBeam = (float ***) malloc( ((_param->TaylorTerms * 2) - 1) * sizeof( float ** ) );
		
		// generate the dirty beam(s) by FFTing the gridded data.
		generateDirtyBeams(	/* phstDirtyBeam = */ hstDirtyBeam,
					/* pFilename = */ outputDirtyBeamFilename );

		// create memory for the clean and dirty beams.
		float * devCleanBeam = NULL, * devDirtyBeam = NULL;
		reserveGPUMemory( (void **) &devCleanBeam, _param->PsfSize * _param->PsfSize * sizeof( float ), "declaring device memory for clean beam", __LINE__ );
		reserveGPUMemory( (void **) &devDirtyBeam, _param->PsfSize * _param->PsfSize * sizeof( float ), "declaring device memory for dirty beam", __LINE__ );
		moveHostToDevice( (void *) devDirtyBeam, (void *) hstDirtyBeam[ /* TAYLOR_TERM = */ 0 ][ /* IMAGE = */ _hstNumDirtyBeams - 1 ],
					_param->PsfSize * _param->PsfSize * sizeof( float ), "moving dirty beam to the device", __LINE__ );
		
		// generate the clean beam (relies on the dirty beam already being within device memory). for mfs, we generate the clean beam from the zeroth-order dirty
		// beam.
		generateCleanBeam(	/* pdevCleanBeam = */ devCleanBeam,
					/* pdevDirtyBeam = */ devDirtyBeam,
					/* pFilename = */ outputCleanBeamFilename );

		// do a Cotton-Schwab clean.
		cottonSchwabClean(	/* pdevCleanBeam = */ devCleanBeam,
					/* phstDirtyBeam = */ hstDirtyBeam,
					/* phstDirtyImage = */ hstDirtyImage,
					/* phstMask = */ hstMask,
					/* pCleanImageFilename = */ outputCleanImageFilename,
					/* pResidualImageFilename = */ outputResidualImageFilename,
					/* pAlphaImageFilename = */ outputAlphaImageFilename );

		// free memory.
		if (devCleanBeam != NULL)
			cudaFree( (void *) devCleanBeam );
		if (devDirtyBeam != NULL)
			cudaFree( (void *) devDirtyBeam );

		if (hstDirtyBeam != NULL)
		{
			for ( int i = 0; i < (_param->TaylorTerms * 2) - 1; i++ )
				if (hstDirtyBeam[ i ] != NULL)
				{
					for ( int j = 0; j < _hstNumDirtyBeams; j++ )
						if (hstDirtyBeam[ i ][ j ] != NULL)
							free( (void *) hstDirtyBeam[ i ][ j ] );
					free( (void *) hstDirtyBeam[ i ] );
				}
			free( (void *) hstDirtyBeam );
		}

	} // (_param->MinorCycles > 0)

	// make sure our data is freed, and delete the cache.
	for ( int mosaicID = 0; mosaicID < _hstData.size(); mosaicID++ )
	{
		_hstData[ mosaicID ]->FreeData( /* pWhatData = */ DATA_ALL );
		if (_param->CacheData == true)
			_hstData[ mosaicID ]->DeleteCache();
	}

	// --------------------------------------------------------------------------------------------
	//
	// c l e a n   u p   m e m o r y
	//
	// --------------------------------------------------------------------------------------------

	if (_devImageDomainPSFunction != NULL)
		cudaFree( (void *) _devImageDomainPSFunction );
	if (_hstPrimaryBeamPattern != NULL)
		free( (void *) _hstPrimaryBeamPattern );
	if (_hstPrimaryBeamRatioPattern != NULL)
		free( (void *) _hstPrimaryBeamRatioPattern );
	if (_hstNormalisationPattern != NULL)
		free( (void *) _hstNormalisationPattern );
	if (hstDirtyImage != NULL)
	{
		for ( int s = 0; s < _param->NumStokesImages; s++ )
			if (hstDirtyImage[ s ] != NULL)
			{
				for ( int t = 0; t < _param->TaylorTerms; t++ )
					if (hstDirtyImage[ s ][ t ] != NULL)
						free( (void *) hstDirtyImage[ s ][ t ] );
				free( (void *) hstDirtyImage[ s ] );
			}
		free( (void *) hstDirtyImage );
	}
	if (hstMask != NULL)
		free( (void *) hstMask );
	if (outputDirtyBeamFilename != NULL)
	{
		for ( int t = 0; t < _param->TaylorTerms; t++ )
			if (outputDirtyBeamFilename[ t ] != NULL)
				free( (void *) outputDirtyBeamFilename[ t ] );
		free( (void *) outputDirtyBeamFilename );
	}
	if (outputDirtyImageFilename != NULL)
	{
		for ( int t = 0; t < _param->TaylorTerms; t++ )
			if (outputDirtyImageFilename[ t ] != NULL)
				free( (void *) outputDirtyImageFilename[ t ] );
		free( (void *) outputDirtyImageFilename );
	}
	if (outputCleanImageFilename != NULL)
	{
		for ( int t = 0; t < _param->TaylorTerms; t++ )
			if (outputCleanImageFilename[ t ] != NULL)
				free( (void *) outputCleanImageFilename[ t ] );
		free( (void *) outputCleanImageFilename );
	}
	if (outputResidualImageFilename != NULL)
	{
		for ( int t = 0; t < _param->TaylorTerms; t++ )
			if (outputResidualImageFilename[ t ] != NULL)
				free( (void *) outputResidualImageFilename[ t ] );
		free( (void *) outputResidualImageFilename );
	}
	for ( int mosaicID = 0; mosaicID < _hstData.size(); mosaicID++ )
		if (_hstData[ mosaicID ] != NULL)
			delete _hstData[ mosaicID ];
	for ( int kernelCache = 0; kernelCache < _griddingKernelCache.size(); kernelCache++ )
		if (_griddingKernelCache[ kernelCache ] != NULL)
			delete _griddingKernelCache[ kernelCache ];
	for ( int kernelCache = 0; kernelCache < _degriddingKernelCache.size(); kernelCache++ )
		if (_degriddingKernelCache[ kernelCache ] != NULL)
			delete _degriddingKernelCache[ kernelCache ];
	for ( int kernelCache = 0; kernelCache < _psfKernelCache.size(); kernelCache++ )
		if (_psfKernelCache[ kernelCache ] != NULL)
			delete _psfKernelCache[ kernelCache ];

	return true;
	
} // main
