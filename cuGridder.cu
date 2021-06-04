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
#include "cppTypes.h"
#endif

// my phase correction code.
#include "cppPhaseCorrection.h"

// my casacore interface.
#include "cppCasacoreInterface.h"

// my image-plane reproject code.
#include "cuReprojectionNotComplex.h"

using namespace std;

//
//	CONSTANTS
//

// the input parameters from file gridder-params.
const char MEASUREMENT_SET[] = "measurement-set:";
const char FIELD_ID[] = "field-id:";
const char BEAM_ID[] = "beam-id:";
const char SPW[] = "spw:";
const char DATA_FIELD[] = "data-field:";
const char TABLE_DATA[] = "table-data:";
const char OUTPUT_PREFIX[] = "output-prefix:";
const char CELL_SIZE[] = "cell-size:";
const char PIXELS_UV[] = "pixels-uv:";
const char W_PLANES[] = "w-planes:";
const char OVERSAMPLE[] = "oversample:";
const char KERNEL_CUTOFF_FRACTION[] = "kernel-cutoff-fraction:";
const char KERNEL_CUTOFF_SUPPORT[] = "kernel-cutoff-support:";
const char MINOR_CYCLES[] = "minor-cycles:";
const char LOOP_GAIN[] = "loop-gain:";
const char CYCLEFACTOR[] = "cyclefactor:";
const char THRESHOLD[] = "threshold:";
const char OUTPUT_RA[] = "output-ra:";
const char OUTPUT_DEC[] = "output-dec:";
const char VISIBILITY_BATCH_SIZE[] = "visibility-batch-size:";
const char WEIGHTING[] = "weighting:";
const char ROBUST_PARAMETER[] = "robust:";
const char A_PLANES[] = "a-planes:";
const char MOSAIC[] = "mosaic:";
const char MOSAIC_DOMAIN[] = "mosaic-domain:";
const char AIRY_DISK_DIAMETER[] = "airy-disk-diameter:";
const char AIRY_DISK_BLOCKAGE[] = "airy-disk-blockage:";
const char BEAM_PATTERN[] = "beam-pattern:";
const char BEAM_SIZE_PIXELS[] = "beam-size:";
const char BEAM_CELL_SIZE[] = "beam-cell-size:";
const char BEAM_FREQUENCY[] = "beam-frequency:";
const char STOKES[] = "stokes:";
const char TELESCOPE[] = "telescope:";
const char CACHE_LOCATION[] = "cache-location:";
const char GPU[] = "gpu:";

// the input parameters for debugging.
const char SAVE_MOSAIC_DIRTY_IMAGE[] = "SAVE_MOSAIC_DIRTY_IMAGES";

// speed of light.
const double CONST_C = 299792458.0;
const double PI = 3.141592654;

// cuda constants.
const int MAXIMUM_BLOCKS_PER_DIMENSION = 65535;
const int MAX_THREADS = 33554432;		// maximum number of total threads per cuda call (32768 x 1024). We can actually have 65535 x 1024, but we set the limit lower.

// other constants.
const int MAX_SIZE_FOR_PSF_FITTING = 60;

// the size of the data area required by the routine to find the maximum pixel value.
const int MAX_PIXEL_DATA_AREA_SIZE = 5;
const int MAX_PIXEL_VALUE = 0;
const int MAX_PIXEL_X = 1;
const int MAX_PIXEL_Y = 2;
const int MAX_PIXEL_REAL = 3;
const int MAX_PIXEL_IMAG = 4;

// the type of data to cache and uncache.
const int DATA_VISIBILITIES = 0x01;
const int DATA_GRID_POSITIONS = 0x02;
const int DATA_KERNEL_INDEXES = 0x04;
const int DATA_DENSITIES = 0x08;
const int DATA_WEIGHTS = 0x10;
const int DATA_RESIDUAL_VISIBILITIES = 0x20;
const int DATA_ALL = DATA_VISIBILITIES | DATA_GRID_POSITIONS | DATA_KERNEL_INDEXES | DATA_DENSITIES | DATA_WEIGHTS;

// the size of the primary beam and deconvolution image.
const int BEAM_SIZE = 1024;

//
//	ENUMERATED TYPES
//

enum fftdirection { FORWARD, INVERSE };
enum griddegrid { GRID, DEGRID };
enum weighting { NONE, NATURAL, UNIFORM, ROBUST };
enum findpixel { CLOSEST, FURTHEST };
enum stokes { STOKES_I, STOKES_Q, STOKES_U, STOKES_V, STOKES_NONE };
enum telescope { UNKNOWN_TELESCOPE, ASKAP, EMERLIN, ALMA, ALMA_7M, ALMA_12M, VLA, MEERKAT };
enum ffttype { C2C, F2C, C2F, F2F };
enum masktype { MASK_MAX, MASK_MIN };
enum mosaicdomain { IMAGE, UV };

//
//	STRUCTURES
//

// vector with double. Can be used either as a 2 or 3 element vector.
struct VectorD
{
	double u;
	double v;
	double w;
};
typedef struct VectorD VectorD;

// vector with integers. Can be used either as a 2 or 3 element vector.
struct VectorI
{
	int u;
	int v;
	int w;
};
typedef struct VectorI VectorI;

//
//	FORWARDS DECLARATIONS
//

//
//	GLOBAL VARIABLES
//

// telescope.
telescope _hstTelescope = UNKNOWN_TELESCOPE;

// debugging parameters.
bool _hstSaveMosaicDirtyImages = false;	// save the dirty image components of a file mosaic.

// grid parameters.
bool _hstWProjection = false;
double _hstCellSize = 0;		// the angular size of each output pixel
double _hstUvCellSize = 0;	// in units of lambda
int _hstUvPixels = 0;
int _hstWPlanes = 1;
double _hstOutputRA = 0;
double _hstOutputDEC = 0;

// GPU info.
int _hstNumGPUs = 0;
int * _hstGPU = NULL;
char _hstGPUParam[1024] = "\0";

int _hstPsfSize = 0;		// the psf size may be smaller than the grid if we're using a grid that is too large to fit on the gpu. a smaller psf size
					//will be selected.

// data parameters.
char ** _hstFieldID = NULL;
char ** _hstDataField = NULL;
char ** _hstSpwRestriction = NULL;
char ** _hstMeasurementSetPath = NULL;
char _hstOutputPrefix[1024] = "output";
char _hstCacheLocation[1024] = "\0";
char ** _hstTableData = NULL;
	
// samples.
int * _hstNumSamples = NULL;
int _hstSampleBatchSize = 0;

// channels.
double * _hstAverageWavelength = NULL;
double _hstMinWavelength = 0.0, _hstMaxWavelength = 0.0;

// weighting.
weighting _hstWeighting = NONE;
float * _hstWeight = NULL;
double * _hstAverageWeight = NULL;
double _hstTotalAverageWeight = 0.0;		// the average weight if we're creating a UV mosaic across multiple files.
double _hstRobust = 0.0;

// A-projection
bool _hstAProjection = false;
int _hstAPlanes = 1;
double _hstAiryDiskDiameter = 25.0;		// the diameter of the Airy disk.
double _hstAiryDiskBlockage = 0.0;		// the width of the blockage at the centre of the Airy disk.
bool _hstDiskDiameterSupplied = false;
bool _hstDiskBlockageSupplied = false;

// primary beams.
int _hstBeamMosaicComponents = 0;			// holds the number of fields contained in a beam mosaic
float ** _hstPrimaryBeam = NULL;			// this primary beam is used for mosaicing, and also for setting the image mask.
float ** _hstPrimaryBeamAProjection = NULL;	// this primary beam is used for a-projection, so it is resized depending upon the wavelength for this A-plane.
int _hstBeamSize = -1;
double _hstBeamCellSize = -1.0;
double _hstBeamWidth = 0;
char ** _hstBeamPattern = NULL;			// the file or files with the primary beam patterns.
double _hstBeamFrequency = -1;			// the frequency for the primary beam patterns.

// field id for each sample.
int * _hstFieldIDArray = NULL;

// file id if we have multiple files.
int _hstMeasurementSets = 0;
int * _hstBeamID = NULL;
bool _hstCacheData = false;			// we set this flag to true if we need to cache and uncache our data.

// mosaic?
bool _hstUseMosaicing = false;
bool _hstFileMosaic = false;
bool _hstBeamMosaic = false;
bool _hstUVMosaic = false;
mosaicdomain _hstMosaicDomain = IMAGE;
int _numMosaicImages = 0;
float * _hstPrimaryBeamPattern = NULL;
float * _hstNormalisationPattern = NULL;

// w-plane details.
double ** _hstWPlaneMean = NULL;
double ** _hstWPlaneMax = NULL;
int ***** _hstVisibilitiesInKernelSet = NULL;
int _hstKernelSets = 1;

// kernel parameters.
double _hstKernelCutoffFraction = 0.01;
int _hstKernelCutoffSupport = 125;
	
// hogbom parameters.
int _hstMinorCycles = 10;
double _hstLoopGain = 0.1;
double _hstCycleFactor = 1.5;
double _hstThreshold = 0.0;
	
// visibilities.
cufftComplex * _hstVisibility = NULL;
cufftComplex * _hstResidualVisibility = NULL;
bool * _hstFlag = NULL;
int * _hstSampleID = NULL;
int * _hstChannelID = NULL;
int _hstPreferredVisibilityBatchSize = 4000000;
long int * _hstGriddedVisibilities = NULL, * _hstGriddedVisibilitiesPerField = NULL;
long int _griddedVisibilitiesForBeamMosaic = 0;

// the batches of data in the file.
int * _hstNumberOfStages = NULL;
int ** _hstNumberOfBatches = NULL;
long int ** _hstNumVisibilities = NULL;

// the FFT on the host: only used for images too large to fit on the GPU.
fftwf_complex * _hstFFTGrid = NULL;
fftwf_plan _fftPlanForward;
fftwf_plan _fftPlanInverse;
bool _fftForwardActive = false;
bool _fftInverseActive = false;

// XX = I + Q	I = (XX + YY) / 2	RR = R + V	I = (LL + RR) / 2
// YY = I - Q	Q = (XX - YY) / 2	LL = I - V	V = (RR - LL) / 2
// XY = U + iV	U = (XY + YX) / 2	RL = Q + iU	Q = (RL + LR) / 2
// YX = U - iV	V = i(YX - XY) / 2	LR = Q - iU	U = i(LR - RL) / 2
stokes _hstStokes = STOKES_I;

// the grid positions and kernel indexes of each visibility.
VectorI * _hstGridPosition = NULL;
int * _hstKernelIndex = NULL;

// density map of visibilities. this map is populated when the data is compacted and visibilities are summed.
int * _hstDensityMap = NULL;
	
// kernel parameters.
int _hstOversample = 0;
int * _hstSupportSize = NULL;
int * _hstKernelSize = NULL;

// anti-aliasing kernel parameters.
int _hstAASupport = 0;
int _hstAAKernelSize = 0;

// psf
int _hstPsfX = 0, _hstPsfY = 0;
	
// clean beam.
int _hstCleanBeamSize = 0;	// holds the size of the non-zero portion of the clean beam.

// deconvolution image.
float * _devDeconvolutionImage = NULL;
float * _hstDeconvolutionImage = NULL;

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

__constant__ int _devUvPixels;
__constant__ int _devVisibilityBatchSize;
__constant__ int _devNumSpws;
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
//	multComplex()
//
//	CJS: 28/09/2015
//
//	Multiply two complex numbers.
//

__host__ __device__ cufftDoubleComplex multComplex( cufftDoubleComplex pOne, double pTwo )
{
	
	cufftDoubleComplex newValue;
	
	newValue.x = (pOne.x * pTwo);
	newValue.y = (pOne.y * pTwo);
	
	// return something.
	return newValue;
	
} // multComplex

__host__ __device__ cufftComplex multComplex( cufftComplex pOne, float pTwo )
{
	
	cufftComplex newValue;
	
	newValue.x = (pOne.x * pTwo);
	newValue.y = (pOne.y * pTwo);
	
	// return something.
	return newValue;
	
} // multComplex

__host__ __device__ cufftComplex multComplex( cufftComplex pOne, cufftComplex pTwo )
{
	
	cufftComplex newValue;
	
	newValue.x = (pOne.x * pTwo.x) - (pOne.y * pTwo.y);
	newValue.y = (pOne.x * pTwo.y) + (pOne.y * pTwo.x);
	
	// return something.
	return newValue;
	
} // multComplex

__host__ __device__ cufftDoubleComplex multComplex( cufftComplex pOne, cufftDoubleComplex pTwo )
{
	
	cufftDoubleComplex newValue;
	
	newValue.x = ((double) pOne.x * pTwo.x) - ((double) pOne.y * pTwo.y);
	newValue.y = ((double) pOne.x * pTwo.y) + ((double) pOne.y * pTwo.x);
	
	// return something.
	return newValue;
	
} // multComplex

//
//	divideComplex()
//
//	CJS: 28/09/2015
//
//	Divide one complex number by another.
//

__host__ __device__ cufftDoubleComplex divideComplex( cufftDoubleComplex pOne, double pTwo )
{
	
	cufftDoubleComplex newValue;
	
	newValue.x = (pOne.x / pTwo);
	newValue.y = (pOne.y / pTwo);
	
	// return something.
	return newValue;
	
} // divideComplex

__host__ __device__ int intFloor( int pValue1, int pValue2 )
{

	// return something.
	return (int) floor( (double) pValue1 / (double) pValue2 );

} // intFloor

__host__ __device__ int mod( int pValue1, int pValue2 )
{

	int value = intFloor( /* pValue1 = */ pValue1, /* pValue2 = */ pValue2 );

	// return something.
	return (pValue1 - (value * pValue2));

} // mod

//
//	rad()
//
//	CJS: 01/04/2019
//
//	Convert degrees to radians
//

double rad( double pIn )
{
	
	return ( pIn * PI / 180.0 );

} // rad

//
//	deg()
//
//	CJS: 08/04/2019
//
//	Convert radians to degrees
//

double deg( double pIn )
{
	
	return ( pIn * 180.0 / PI );

} // deg

//
//	gaussian2D()
//
//	CJS: 05/11/2015
//
//	Create an elliptical 2D Gaussian at position (pX, pY), with long and short axes pR1 and pR2, rotated at pAngle.
//

__host__ __device__ double gaussian2D( double pNormalisation, double pX, double pY, double pAngle, double pR1, double pR2 )
{
	
	// calculate the distance along the long and short axes.
	double rOne = ((pY * cos( pAngle )) + (pX * sin( pAngle )));
	double rTwo = ((pX * cos( pAngle )) - (pY * sin( pAngle )));
	
	// we want the axis-one distance as a multiple of the Gaussian width in this direction, and then squared.
	if (pR1 != 0)
		rOne = pow( rOne / pR1, 2 );
	else
		
		// the Gaussian has no width along axis one. we use a flag of -1 to indicate that we need to return a zero value.
		if (rOne != 0)
			rOne = -1;
	
	// we want the axis-two distance as a multiple of the Gaussian width in this direction, and then squared.
	if (pR2 != 0)
		rTwo = pow( rTwo / pR2, 2 );
	else
		
		// the Gaussian has no width along axis two. we use a flag of -1 to indicate that we need to return a zero value.
		if (rTwo != 0)
			rTwo = -1;
	
	// calculate the return value.
	double returnValue = pNormalisation;
	if (rOne >= 0)
		returnValue = returnValue * exp( -rOne );
	if (rTwo >= 0)
		returnValue = returnValue * exp( -rTwo );
	
	// if either of the distances is < 1 then this means our Gaussian has no width in this direction, and our return
	// value should be 0.
	if (rOne < 0 || rTwo < 0)
		returnValue = 0;
		
	// return something.
	return returnValue;
	
} // gaussian2D

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

__global__ void devCalculateGaussianError( float * pImage, double * pError, int pSizeOfFittingRegion, double pCentreX, double pCentreY,
						double pAngle, double pR1, double pR2, int pImageSize, double pNormalisation )
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
		if (posX >= 0 && posX < pImageSize && posY >= 0 && posY < pImageSize)
			pError[ index ] = pow(	pImage[ (posY * pImageSize) + posX ] -
						gaussian2D(	/* pNormalisation = */ pNormalisation,
								/* pX = */ (double) posX - pCentreX,
								/* pY = */ (double) posY - pCentreY,
								/* pAngle = */ pAngle,
								/* pR1 = */ pR1,
								/* pR2 = */ pR2 ), 2 );
		else
			pError[ index ] = 0;

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
//	devFindCutoffPixelParallel()
//
//	CJS: 03/04/2019
//
//	Finds the furthest pixel from the centre of the kernel that is at least 1% of the maximum kernel value.
//

__global__ void devFindCutoffPixelParallel( cufftComplex * pKernel, int pSize, double * pMaxValue, int pCellsPerThread, int * pTmpResults,
						double pCutoffFraction, findpixel pFindType )
{
	
	// the dynamic memory area stores arrays of visibilities, samples and frequencies.
	extern __shared__ int shrSupport[];
	
	// get the starting cell index.
	long int cell = ((blockIdx.x * blockDim.x) + threadIdx.x) * (long int) pCellsPerThread;
	
	int bestSupport = -1;
	for ( long int index = cell; index < cell + pCellsPerThread; index++ )
	{
		
		// ensure we are within bounds.
		if (index < (long int) (pSize * pSize))
		{

			// set our search value to N% of the kernel maximum.
			double searchValue = pMaxValue[ MAX_PIXEL_VALUE ] * pCutoffFraction;

			// get coordinates and pixel value.
			int i = index % pSize;
			int j = index / pSize;
			float pixelValue = pKernel[ index ].x;
			if ((pixelValue >= searchValue && pCutoffFraction > -1) || (pixelValue < 0.0 && pCutoffFraction == -1))
			{

				// get the size of the kernel to this point.
				int supportX = abs( i - (pSize / 2) );
				int supportY = abs( j - (pSize / 2) );
				int support = supportX;
				if (supportY > supportX)
					support = supportY;

				// update the maximum value.
				if (((support > bestSupport) && pFindType == FURTHEST) || ((support < bestSupport) && pFindType == CLOSEST) || bestSupport == -1)
					bestSupport = support;

			}
		
		}
		
	}
		
	// update maximum values.
	shrSupport[ threadIdx.x ] = bestSupport;
	
	__syncthreads();
	
	// now, get the maximum/minimum value from the shared array.
	if (threadIdx.x == 0)
	{
	
		int bestSupport = -1;
	
		for ( int i = 0; i < blockDim.x; i++ )
		{
		
			int support = shrSupport[ i ];
			if (support == -1 && pFindType == CLOSEST)
				support = pSize / 2;
			if (support == -1 && pFindType == FURTHEST)
				support = 0;
			
			// is this value greater than the previous greatest?
			if ((support > bestSupport && pFindType == FURTHEST) || (support < bestSupport && pFindType == CLOSEST) || bestSupport == -1)
				bestSupport = support;
			
		}
		
		// update global memory with these values.
		pTmpResults[ blockIdx.x ] = bestSupport;

	}

} // devFindCutoffPixelParallel

__global__ void devFindCutoffPixelParallel( float * pKernel, int pSize, double * pMaxValue, int pCellsPerThread, int * pTmpResults, double pCutoffFraction,
						findpixel pFindType )
{
	
	// the dynamic memory area stores arrays of visibilities, samples and frequencies.
	extern __shared__ int shrSupport[];
	
	// get the starting cell index.
	long int cell = ((blockIdx.x * blockDim.x) + threadIdx.x) * (long int) pCellsPerThread;
	
	int bestSupport = -1;
	for ( long int index = cell; index < cell + pCellsPerThread; index++ )
	{
		
		// ensure we are within bounds.
		if (index < (long int) (pSize * pSize))
		{

			// set our search value to N% of the kernel maximum.
			double searchValue = pMaxValue[ MAX_PIXEL_VALUE ] * pCutoffFraction;

			// get coordinates and pixel value.
			int i = index % pSize;
			int j = index / pSize;
			float pixelValue = pKernel[ index ];
			if ((pixelValue >= searchValue && pCutoffFraction > -1) || (pixelValue < 0.0 && pCutoffFraction == -1))
			{

				// get the size of the kernel to this point.
				int supportX = abs( i - (pSize / 2) );
				int supportY = abs( j - (pSize / 2) );
				int support = supportX;
				if (supportY > supportX)
					support = supportY;

				// update the maximum value.
				if (((support > bestSupport) && pFindType == FURTHEST) || ((support < bestSupport) && pFindType == CLOSEST) || bestSupport == -1)
					bestSupport = support;

			}
		
		}
		
	}
		
	// update maximum values.
	shrSupport[ threadIdx.x ] = bestSupport;
	
	__syncthreads();
	
	// now, get the maximum/minimum value from the shared array.
	if (threadIdx.x == 0)
	{
	
		int bestSupport = -1;
	
		for ( int i = 0; i < blockDim.x; i++ )
		{
		
			int support = shrSupport[ i ];
			if (support == -1 && pFindType == CLOSEST)
				support = pSize / 2;
			if (support == -1 && pFindType == FURTHEST)
				support = 0;
			
			// is this value greater than the previous greatest?
			if ((support > bestSupport && pFindType == FURTHEST) || (support < bestSupport && pFindType == CLOSEST) || bestSupport == -1)
				bestSupport = support;
			
		}
		
		// update global memory with these values.
		pTmpResults[ blockIdx.x ] = bestSupport;

	}

} // devFindCutoffPixelParallel

//
//	devFindCutoffPixel()
//
//	CJS: 02/04/2019
//
//	Finds the furthest pixel from the centre of the kernel that is at least 1% of the maximum kernel value.
//

__global__ void devFindCutoffPixel( int * pTmpResults, int * pSupport, int pElements, findpixel pFindType )
{
	
	int bestSupport = -1;
	
	// get maximum value.
	for ( int i = 0; i < pElements; i++ )
	{
			
		// get the support.
		double support = *pTmpResults;
			
		// is this value greater than the previous greatest?
		if (((support > bestSupport) && pFindType == FURTHEST) || ((support < bestSupport) && pFindType == CLOSEST) || bestSupport == -1)
			bestSupport = support;

		pTmpResults++;
			
	}
		
	// update maximum support.
	*pSupport = bestSupport;

} // devFindCutoffPixel

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
//	pGrid				the grid (OUTPUT)
//	pVisibility			the list of visibilities (INPUT)
//	pVisibilitiesPerBlock		> 1 if the kernel is smaller than one block
//	pBlocksPerVisibility		> 1 if the kernel is larger than one block
//	pGridPosition			list of u,v,w coordinates to place on the grid (INPUT)
//	pKernelIndex			list of kernel indexes to apply to each visibility (INPUT)
//	pNumVisibilities		the number of visibilities to grid (INPUT)
//	pGridDegrid			either GRID or DEGRID
//		pFirstKernel			the first kernel set index in this batch of visibilities.
//	pNumKernels			the number of kernels we are gridding with. if the kernel index is > than this number than we take kernelIndex % pNumKernels.
//	pSize				the size of the image
//	pComplex			are we gridding complex or non-complex visibilities?
//
//	shared memory:
//	shrVisibility - holds all the visibilities in this thread block.
//	shrGridPosition - holds all the grid positions (3 x int) for the visibilities in this thread block.
//	shrKernelIndex - holds the kernel indexes for the visibilities in this thread block.
//	shrWeight - holds the weight for this visibility.
//

__global__ void devGridVisibilities( cufftComplex * pGrid, cufftComplex * pVisibility, int pVisibilitiesPerBlock, int pBlocksPerVisibility,
					VectorI * pGridPosition, cufftComplex * pKernel, int * pKernelIndex, float * pWeight, int pNumVisibilities,
					int pNumKernels, int pSize, bool pComplex, int pSupport )
{

	// the dynamic memory area stores arrays of visibilities, samples and frequencies.
	extern __shared__ char shrDynamic[];
	
	// pointers to dynamic shared memory.
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
		if (pComplex == true)
			shrVisibility[ visibilityArrayIndex ] = pVisibility[ visibilityIndex ];
		else
		{
			shrVisibility[ visibilityArrayIndex ].x = ((double *) pVisibility)[ visibilityIndex ];
			shrVisibility[ visibilityArrayIndex ].y = 0.0;
		}
		shrGridPosition[ visibilityArrayIndex ] = pGridPosition[ visibilityIndex ];

		if (pKernelIndex != NULL)
			shrKernelIndex[ visibilityArrayIndex ] = pKernelIndex[ visibilityIndex ];
		else
			shrKernelIndex[ visibilityArrayIndex ] = 0;
		if (applyWeighting == true)
			shrWeight[ visibilityArrayIndex ] = pWeight[ visibilityIndex ];

		// if the kernel index is > the number of kernels then we have multiple fields and we're using the primary beam for each field as the gridding kernel.
		// however, when gridding the psf we don't need to do this so we will be using the same kernel for each field.
		if (shrKernelIndex[ visibilityArrayIndex ] > pNumKernels)
			shrKernelIndex[ visibilityArrayIndex ] = shrKernelIndex[ visibilityArrayIndex ] % pNumKernels;
		
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
		int kernelIndex = (shrKernelIndex[ visibilityArrayIndex ] * kernelSize * kernelSize) + (kernelY * kernelSize) + (kernelX);
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
					addComplex( gridPtr, multComplex( multComplex( shrVisibility[ visibilityArrayIndex ], shrWeight[ visibilityArrayIndex ] ),
										kernel ) );
				else
					addComplex( gridPtr, multComplex( shrVisibility[ visibilityArrayIndex ], kernel ) );

			}
			else
			{

				// with or without weighting.
				if (applyWeighting == true)
					atomicAdd( (float *) gridPtr, shrVisibility[ visibilityArrayIndex ].x * kernel.x *
								shrWeight[ visibilityArrayIndex ] ); 
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

__global__ void devDegridVisibilities( cufftComplex * pGrid, cufftComplex * pVisibility, VectorI * pGridPosition, cufftComplex * pKernel, int * pKernelIndex,
					int pNumVisibilities, int pSize, int pVisibilitiesPerPage, int pSupport )
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

		// since we are degridding then the grid offset should be opposite to the kernel offset.
		grid.u -= (kernelX - pSupport);
		grid.v -= (kernelY - pSupport);
					
		// get kernel value.
		cufftComplex kernel = pKernel[ (pKernelIndex[ visibilityIndex ] * kernelSize * kernelSize) + (kernelY * kernelSize) + (kernelX) ];

		// is this pixel within the grid range? If so, add up visibility.
		if ((grid.u >= 0) && (grid.u < pSize) && (grid.v >= 0) && (grid.v < pSize))
			addComplex( &pVisibility[ visibilityIndex ], multComplex( pGrid[ (grid.v * pSize) + grid.u ], kernel ) );
	
	}

} // devDegridVisibilities

//
//	devTakeConjugate()
//
//	CJS: 18/10/2018
//
//	Take the conjugate values of a complex visibility data set, but only for the second half of the data.
//

__global__ void devTakeConjugate( cufftComplex * pVisibility, long int pCurrentVisibility, long int pNumVisibilities )
{
	
	// calculate visibility index and grid position index. we have twice as many grid positions as visibilities because
	// each visibility is gridded twice - once at B and the complex conjugate at -B.
	int visibilityIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	// ensure we haven't gone out of bounds.
	if (visibilityIndex < _devVisibilityBatchSize)
	{

		// ensure we are in the second half of the visibility data set.
		long int thisVisibility = pCurrentVisibility + (long int) visibilityIndex;
		if (thisVisibility >= (pNumVisibilities / 2))
			pVisibility[ visibilityIndex ].y *= -1;

	}

} // devTakeConjugate

//
//	devPhaseCorrection()
//
//	CJS: 17/10/2018
//
//	Phase correct all the visibilities.
//

__global__ void devPhaseCorrection( cufftComplex * pVisibility, double * pPhase, double ** pWavelength, int * pSpw, int * pSampleID, int * pChannelID )
{
	
	// calculate visibility index and grid position index. we have twice as many grid positions as visibilities because
	// each visibility is gridded twice - once at B and the complex conjugate at -B.
	long int visibilityIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	// ensure we haven't gone out of bounds.
	if (visibilityIndex < _devVisibilityBatchSize)
	{

		// get the sample id and channel id.
		int sample = pSampleID[ visibilityIndex ];
		int channel = pChannelID[ visibilityIndex ];
		int spw = pSpw[ sample ];

		// only proceed if the spw is within the expected range. this visibility will already be flagged if this is not the case.
		cufftDoubleComplex newVis = { .x = 0, .y = 0 };
		if (spw >= 0 && spw < _devNumSpws)
		{

			// get the wavelength.
			double wavelength = pWavelength[ spw ][ channel ];

			cufftDoubleComplex phasor;
			sincos( 2.0 * PI * pPhase[ sample ] / wavelength, &phasor.y, &phasor.x );

			// multiply phasor by visibility.
			newVis = multComplex( /* pOne = */ pVisibility[ visibilityIndex ], /* pTwo = */ phasor );

		}
		pVisibility[ visibilityIndex ].x = (float) newVis.x;
		pVisibility[ visibilityIndex ].y = (float) newVis.y;

	}
	
} // devPhaseCorrection

//
//	devCalculateGridPositions()
//
//	CJS: 11/11/2015
//
//	Calculate a grid position and a kernel index for each visibility.
//
//	pGridPosition		a list of integer u,v,w grid coordinates - one per visibility (OUTPUT)
//	pKernelIndex		a list of kernel indexes - one per visibility (OUTPUT)
//	pUvCellSize		}
//	pOversample		}- gridding parameters
//	pWPlanes		}
//	pSample			a list of UVW coordinates for all the samples
//	pWavelength		a list of wavelengths for all the channels and spws
//	pWPlaneMax		a list of W limits for all the W planes
//	pSampleID		the sample ID for each visibility
//	pChannelID		the channel ID for each visibility
//	pSize			the image size
//

__global__ void devCalculateGridPositions(	VectorI * pGridPosition, int * pKernelIndex, double pUvCellSize, int pOversample,
						int pWPlanes, int pAPlanes, VectorD * pSample, double ** pWavelength, int * pField, int * pSpw,
						double * pWPlaneMax, int * pAPlane, int * pSampleID, int * pChannelID, int pSize	)
{
	
	// calculate visibility index and grid position index. we have twice as many grid positions as visibilities because
	// each visibility is gridded twice - once at B and the complex conjugate at -B.
	long int visibilityIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	// ensure we haven't gone out of bounds.
	if (visibilityIndex < _devVisibilityBatchSize)
	{

		// get the sample id and channel id.
		int sample = pSampleID[ visibilityIndex ];
		int channel = pChannelID[ visibilityIndex ];
		int spw = pSpw[ sample ];
		
		// declare grid position and kernel index.
		VectorI grid = { .u = 0, .v = 0, .w = 0 };
		int kernelIdx = 0;

		// only calculate a grid position if the spw is within the expected range. this visibility will already be flagged if this is not the case.
		if (spw >= 0 && spw < _devNumSpws)
		{

			// get the sample UVW and the wavelength.
			VectorD uvw = pSample[ sample ];
			double wavelength = pWavelength[ spw ][ channel ];

			// get the field ID.
			int fieldID = -1;
			if (pField != NULL)
				fieldID = pField[ sample ];
		
			// declare (integer) oversample vector.
			VectorI oversample, oversampleIndex;

			// calculate the uvw coordinate by dividing the sample UVW by the wavelength.
			if (wavelength != 0)
			{
				uvw.u = uvw.u / wavelength;
				uvw.v = uvw.v / wavelength;
				uvw.w = uvw.w / wavelength;
			}
			else
			{
				uvw.u = 0;
				uvw.v = 0;
				uvw.w = 0;
			}

			// divide the uvw position by the uv cell size to get the exactly (floating) pixel position.
			VectorD exact = { .u = uvw.u / pUvCellSize, .v = uvw.v / pUvCellSize, .w = uvw.w };

			oversample.u = (int) round( exact.u * (double) pOversample);
			oversampleIndex.u = mod( oversample.u, pOversample );
			grid.u = intFloor( oversample.u, pOversample ) + (pSize / 2);
		
			oversample.v = (int) round( exact.v * (double) pOversample);
			oversampleIndex.v = mod( oversample.v, pOversample );
			grid.v = intFloor( oversample.v, pOversample ) + (pSize / 2);
				
			// calculate the kernel offset using the uOversample and vOversample.
			// no need to add the index of the w-plane. we will be gridding one w-plane at a time.
			kernelIdx = (oversampleIndex.u) + (oversampleIndex.v * pOversample);

			// if we have separate kernels for each field, then add an offset here.
			if (fieldID > -1)
				kernelIdx += (fieldID * pOversample * pOversample);

			// calculate which w plane we are in.
			if (pWPlanes > 1)
				for ( int i = pWPlanes - 1; i >= 0; i-- )
					if (exact.w <= pWPlaneMax[ i ])
						grid.w = i;

			// replace the w-value with the kernel set index. This is calculated using (w * num_a_planes) + a
			if (pAPlane != NULL)
				grid.w = (grid.w * pAPlanes) + pAPlane[ visibilityIndex ];

		}

		// update the arrays.
		pGridPosition[ visibilityIndex ] = grid;
		pKernelIndex[ visibilityIndex ] = kernelIdx;
	
	}
	
} // devCalculateGridPositions

//
//	devFFTShift()
//
//	CJS: 24/09/2015
//
//	Perform an FFT shift on the data following the FFT operation.
//

__global__ void devFFTShift( cufftComplex * pDestination, cufftComplex * pSource, fftdirection pFFTDirection, int pSize )
{
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	int pivotOne = (int) floor( (double) pSize / 2.0 );
	int pivotTwo = (int) ceil( (double) pSize / 2.0 );
	int iFrom, jFrom;
	
	// if we are doing an inverse FFT shift then swap the pivots.
	if (pFFTDirection == INVERSE)
	{
		int tmp = pivotOne;
		pivotOne = pivotTwo;
		pivotTwo = tmp;
	}
	
	// calculate which pixel we should take our value from.
	if (i < pivotTwo)
		iFrom = i + pivotOne;
	else
		iFrom = i - pivotTwo;
	if (j < pivotTwo)
		jFrom = j + pivotOne;
	else
		jFrom = j - pivotTwo;
	
	// if we are within the bounds of the array, do FFT shift.
	if (i >= 0 && i < pSize && j >= 0 && j < pSize)
		memcpy( &pDestination[ (j * pSize) + i ], &pSource[ (jFrom * pSize) + iFrom ], sizeof( cufftComplex ) );
	
} // devFFTShift

__global__ void devFFTShift( double * pDestination, cufftComplex * pSource, fftdirection pFFTDirection, int pSize )
{
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	int pivotOne = (int) floor( (double) pSize / 2.0 );
	int pivotTwo = (int) ceil( (double) pSize / 2.0 );
	int iFrom, jFrom;
	
	// if we are doing an inverse FFT shift then swap the pivots.
	if (pFFTDirection == INVERSE)
	{
		int tmp = pivotOne;
		pivotOne = pivotTwo;
		pivotTwo = tmp;
	}
	
	// calculate which pixel we should take our value from.
	if (i < pivotTwo)
		iFrom = i + pivotOne;
	else
		iFrom = i - pivotTwo;
	if (j < pivotTwo)
		jFrom = j + pivotOne;
	else
		jFrom = j - pivotTwo;
	
	// if we are within the bounds of the array, do FFT shift.
	if (i >= 0 && i < pSize && j >= 0 && j < pSize)
		pDestination[ (j * pSize) + i ] = pSource[ (jFrom * pSize) + iFrom ].x;
	
} // devFFTShift

__global__ void devFFTShift( float * pDestination, cufftComplex * pSource, fftdirection pFFTDirection, int pSize )
{
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	int pivotOne = (int) floor( (double) pSize / 2.0 );
	int pivotTwo = (int) ceil( (double) pSize / 2.0 );
	int iFrom, jFrom;
	
	// if we are doing an inverse FFT shift then swap the pivots.
	if (pFFTDirection == INVERSE)
	{
		int tmp = pivotOne;
		pivotOne = pivotTwo;
		pivotTwo = tmp;
	}
	
	// calculate which pixel we should take our value from.
	if (i < pivotTwo)
		iFrom = i + pivotOne;
	else
		iFrom = i - pivotTwo;
	if (j < pivotTwo)
		jFrom = j + pivotOne;
	else
		jFrom = j - pivotTwo;
	
	// if we are within the bounds of the array, do FFT shift.
	if (i >= 0 && i < pSize && j >= 0 && j < pSize)
		pDestination[ (j * pSize) + i ] = pSource[ (jFrom * pSize) + iFrom ].x;
	
} // devFFTShift

__global__ void devFFTShift( cufftComplex * pDestination, float * pSource, fftdirection pFFTDirection, int pSize )
{
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	int pivotOne = (int) floor( (double) pSize / 2.0 );
	int pivotTwo = (int) ceil( (double) pSize / 2.0 );
	int iFrom, jFrom;
	
	// if we are doing an inverse FFT shift then swap the pivots.
	if (pFFTDirection == INVERSE)
	{
		int tmp = pivotOne;
		pivotOne = pivotTwo;
		pivotTwo = tmp;
	}
	
	// calculate which pixel we should take our value from.
	if (i < pivotTwo)
		iFrom = i + pivotOne;
	else
		iFrom = i - pivotTwo;
	if (j < pivotTwo)
		jFrom = j + pivotOne;
	else
		jFrom = j - pivotTwo;
	
	// if we are within the bounds of the array, do FFT shift.
	if (i >= 0 && i < pSize && j >= 0 && j < pSize)
	{
		pDestination[ (j * pSize) + i ].x = pSource[ (jFrom * pSize) + iFrom ];
		pDestination[ (j * pSize) + i ].y = 0.0;
	}
	
} // devFFTShift

//
//	devReverseYDirection()
//
//	CJS: 04/10/2018
//
//	Reverse the y-direction of the image.
//

__global__ void devReverseYDirection( cufftComplex * pGrid, int pSize )
{
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	// check that we haven't gone outside the grid dimensions (some threads will).
	if ( i < pSize && j < (pSize / 2) )
	{
	
		// calculate cell indexes for swapping.
		long int indexOne = (j * pSize) + i;
		long int indexTwo = ((pSize - j - 1) * pSize) + i;
		
		// reverse the x-axis (for some reason).
		cufftComplex tmp = pGrid[ indexOne ];
		pGrid[ indexOne ] = pGrid[ indexTwo ];
		pGrid[ indexTwo ] = tmp;
		
	}
	
} // devReverseYDirection

__global__ void devReverseYDirection( double * pGrid, int pSize )
{
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	// check that we haven't gone outside the grid dimensions (some threads will).
	if (i < pSize && j < (pSize / 2))
	{
	
		// calculate cell indexes for swapping.
		long int indexOne = (j * pSize) + i;
		long int indexTwo = ((pSize - j - 1) * pSize) + i;
		
		// reverse the x-axis (for some reason).
		double tmp = pGrid[ indexOne ];
		pGrid[ indexOne ] = pGrid[ indexTwo ];
		pGrid[ indexTwo ] = tmp;
		
	}
	
} // devReverseYDirection

__global__ void devReverseYDirection( float * pGrid, int pSize )
{
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	// check that we haven't gone outside the grid dimensions (some threads will).
	if (i < pSize && j < (pSize / 2))
	{
	
		// calculate cell indexes for swapping.
		long int indexOne = (j * pSize) + i;
		long int indexTwo = ((pSize - j - 1) * pSize) + i;
		
		// reverse the x-axis (for some reason).
		double tmp = pGrid[ indexOne ];
		pGrid[ indexOne ] = pGrid[ indexTwo ];
		pGrid[ indexTwo ] = tmp;
		
	}
	
} // devReverseYDirection

//
//	devBuildMask()
//
//	CJS: 15/05/2020
//
//	Builds a boolean mask based upon a threshold value.
//

__global__ void devBuildMask( float * pArray, int pSize, double pValue, masktype pMaxMin, bool * pMask )
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
//	devGetMaxValue()
//
//	CJS: 06/11/2015
//
//	Get the maximum value from a 1d array (containing value, x, y), and store this number along with the x and y coordinates.
//

__global__ void devGetMaxValue( double * pArray, double * pMaxValue, int pElements )
{
	
	double maxValue = 0;
	double maxI = 0;
	double maxJ = 0;
	double maxValueReal = 0;
	double maxValueImag = 0;
	
	// get maximum value.
	for ( int i = 0; i < pElements; i++ )
	{
			
		// get the value.
		double value = pArray[ MAX_PIXEL_VALUE ];
			
		// is this value greater than the previous greatest?
		if (value > maxValue)
		{
			maxValue = value;
			maxI = pArray[ MAX_PIXEL_X ];
			maxJ = pArray[ MAX_PIXEL_Y ];
			maxValueReal = pArray[ MAX_PIXEL_REAL ];
			maxValueImag = pArray[ MAX_PIXEL_IMAG ];
		}
		pArray = pArray + MAX_PIXEL_DATA_AREA_SIZE;
			
	}
		
	// update maximum values.
	pMaxValue[ MAX_PIXEL_VALUE ] = maxValue;
	pMaxValue[ MAX_PIXEL_X ] = maxI;
	pMaxValue[ MAX_PIXEL_Y ] = maxJ;
	pMaxValue[ MAX_PIXEL_REAL ] = maxValueReal;
	pMaxValue[ MAX_PIXEL_IMAG ] = maxValueImag;
	
} // devGetMaxValue

//
//	devGetMaxValueParallel()
//
//	CJS: 06/11/2015
//
//	Get the maximum complex number from an 2-d array, and store this number along with the x and y coordinates.
//
//	Uses the absolute value if pIncludeComplexComponent == true, else only uses the real value.
//

__global__ void devGetMaxValueParallel( double * pArray, int pWidth, int pHeight, int pCellsPerThread, double * pBlockMax, bool * pMask )
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
		
			double value = pArray[ i ];

			// has this cell been masked? we can include it if there is no mask provided, or if the mask is TRUE (i.e. cell is good).
			bool includeCell = (pMask == NULL);
			if (pMask != NULL)
				includeCell = (pMask[ i ] == true);
			
			// is this value greater than the previous greatest?
			if (value > maxValue && includeCell == true)
			{
				maxValue = value;
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
	
} // devGetMaxValueParallel

__global__ void devGetMaxValueParallel( cufftComplex * pArray, int pWidth, int pHeight, int pCellsPerThread, double * pBlockMax,
							bool pIncludeComplexComponent, bool * pMask )
{
	
	// the dynamic memory area stores arrays of visibilities, samples and frequencies.
	extern __shared__ double shrMaxValue[];
	
	double maxValue = 0;
	double maxI = 0;
	double maxJ = 0;
	double maxValueReal = 0;
	double maxValueImag = 0;
	
	// get the starting cell index.
	int cell = ((blockIdx.x * blockDim.x) + threadIdx.x) * pCellsPerThread;
	
	for ( int i = cell; i < cell + pCellsPerThread; i++ )
	{
		
		// ensure we are within bounds.
		if (i < pWidth * pHeight)
		{
		
			float value = 0;
			if (pIncludeComplexComponent == true)
				value = cuCabsf( pArray[ i ] );
			else
				value = pArray[ i ].x;

			// has this cell been masked? we can include it if there is no mask provided, or if the mask is TRUE (i.e. cell is good).
			bool includeCell = (pMask == NULL);
			if (pMask != NULL)
				includeCell = (pMask[ i ] == true);
			
			// is this value greater than the previous greatest?
			if (value > maxValue && includeCell == true)
			{
				maxValue = value;
				maxI = (double) (i % pWidth);
				maxJ = i / pWidth;
				maxValueReal = (double) pArray[ i ].x;
				maxValueImag = (double) pArray[ i ].y;
			}
		
		}
		
	}
		
	// update maximum values.
	shrMaxValue[ (threadIdx.x * MAX_PIXEL_DATA_AREA_SIZE) + MAX_PIXEL_VALUE ] = maxValue;
	shrMaxValue[ (threadIdx.x * MAX_PIXEL_DATA_AREA_SIZE) + MAX_PIXEL_X ] = maxI;
	shrMaxValue[ (threadIdx.x * MAX_PIXEL_DATA_AREA_SIZE) + MAX_PIXEL_Y ] = maxJ;
	shrMaxValue[ (threadIdx.x * MAX_PIXEL_DATA_AREA_SIZE) + MAX_PIXEL_REAL ] = maxValueReal;
	shrMaxValue[ (threadIdx.x * MAX_PIXEL_DATA_AREA_SIZE) + MAX_PIXEL_IMAG ] = maxValueImag;
	
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
		
			double value = shrMaxValue[ i * MAX_PIXEL_DATA_AREA_SIZE ];
			
			// is this value greater than the previous greatest?
			if (value > maxValue)
			{
				maxValue = value;
				maxI = shrMaxValue[ (i * MAX_PIXEL_DATA_AREA_SIZE) + 1 ];
				maxJ = shrMaxValue[ (i * MAX_PIXEL_DATA_AREA_SIZE) + 2 ];
				maxValueReal = shrMaxValue[ (i * MAX_PIXEL_DATA_AREA_SIZE) + 3 ];
				maxValueImag = shrMaxValue[ (i * MAX_PIXEL_DATA_AREA_SIZE) + 4 ];
			}
			
		}
		
		// update global memory with these values.
		pBlockMax[ (blockIdx.x * MAX_PIXEL_DATA_AREA_SIZE) + MAX_PIXEL_VALUE ] = maxValue;
		pBlockMax[ (blockIdx.x * MAX_PIXEL_DATA_AREA_SIZE) + MAX_PIXEL_X ] = maxI;
		pBlockMax[ (blockIdx.x * MAX_PIXEL_DATA_AREA_SIZE) + MAX_PIXEL_Y ] = maxJ;
		pBlockMax[ (blockIdx.x * MAX_PIXEL_DATA_AREA_SIZE) + MAX_PIXEL_REAL ] = maxValueReal;
		pBlockMax[ (blockIdx.x * MAX_PIXEL_DATA_AREA_SIZE) + MAX_PIXEL_IMAG ] = maxValueImag;
	
	}
	
} // devGetMaxValueParallel

__global__ void devGetMaxValueParallel( float * pArray, int pWidth, int pHeight, int pCellsPerThread, double * pBlockMax, bool * pMask )
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
		
			float value = pArray[ i ];

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
	
} // devGetMaxValueParallel

//
//	devMoveToStartOfImage()
//
//	CJS: 12/11/2019
//
//	Move the central part of an image to the start of the image. The initial size must be at least 2x the final size or else we could be overwriting pixels before we
//	move them.
//

__global__ void devMoveToStartOfImage( float * pImage, int pInitialSize, int pFinalSize )
{
	
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	// check we're in the bounds of the array.
	if (index < (pFinalSize * pFinalSize))
	{

		// get the final coordinates.
		int i = index % pFinalSize;
		int j = index / pFinalSize;

		// get the initial coordinates.
		int iI = i + ((pInitialSize - pFinalSize) / 2);
		int jI = j + ((pInitialSize - pFinalSize) / 2);

		pImage[ index ] = pImage[ (jI * pInitialSize) + iI ];

	}

} // devMoveToStartOfImage

//
//	devNormalise()
//
//	CJS: 06/11/2015
//
//	Divide an array of complex numbers by a constant.
//

__global__ void devNormalise( cufftComplex * pArray, double pConstant, int pItems )
{
	
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	// update the value, but only if we are within the bounds of the array.
	if (index < pItems)
	{
		pArray[ index ].x /= pConstant;
		pArray[ index ].y /= pConstant;
	}
	
} // devNormalise

__global__ void devNormalise( double * pArray, double * pConstant, int pItems )
{
	
	// store the constant in shared memory.
	__shared__ double constant;
	if (threadIdx.x == 0)
		constant = *pConstant;
	
	__syncthreads();
	
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	// update the value, but only if we are within the bounds of the array.
	if (index < pItems)
		pArray[ index ] /= constant;
	
} // devNormalise

__global__ void devNormalise( double * pArray, double pConstant, int pItems )
{
	
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	// update the value, but only if we are within the bounds of the array.
	if (index < pItems)
		pArray[ index ] /= pConstant;
	
} // devNormalise

__global__ void devNormalise( float * pArray, double pConstant, int pItems )
{
	
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	// update the value, but only if we are within the bounds of the array.
	if (index < pItems)
		pArray[ index ] /= pConstant;
	
} // devNormalise

__global__ void devNormalise( float * pArray, double * pConstant, int pItems )
{
	
	// store the constant in shared memory.
	__shared__ double constant;
	if (threadIdx.x == 0)
		constant = *pConstant;
	
	__syncthreads();
	
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	// update the value, but only if we are within the bounds of the array.
	if (index < pItems)
		pArray[ index ] /= constant;
	
} // devNormalise

//
//	devCalculateVisibilityAndFlag()
//
//	CJS: 31/10/2018
//
//	Calculate the visibility as either (LL + RR) / 2 or (XX + YY) / 2, and sets the flag is either of the polarisations are flagged.
//

__global__ void devCalculateVisibilityAndFlag( cufftComplex * pVisibilityIn, cufftComplex * pVisibilityOut, bool * pFlagIn, bool * pFlagOut,
						int pNumPolarisations, double * pMultiplier, int * pPolarisationConfig, int * pSampleID )
{

	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	// ensure we are not out of bounds.
	if (index < _devVisibilityBatchSize)
	{

		bool flag = false;

		// get sample ID and the polarisation config for this sample.
		int polarisationConfig = pPolarisationConfig[ pSampleID[ index ] ];

		// calculate visibility and flag.
		cufftComplex value; value.x = 0.0; value.y = 0.0;
		for ( int polarisation = 0; polarisation < pNumPolarisations; polarisation++ )
		{
			double tmpMultiplier = pMultiplier[ (polarisationConfig * pNumPolarisations) + polarisation ];
			if (tmpMultiplier != 0.0)
			{
				cufftComplex tmp = pVisibilityIn[ (index * pNumPolarisations) + polarisation ];
				value.x += (tmp.x * tmpMultiplier);
				value.y += (tmp.y * tmpMultiplier);
				flag = flag || (pFlagIn[ (index * pNumPolarisations) + polarisation ] == true && tmpMultiplier > 0.0);
			}
		}
		pVisibilityOut[ index ] = value;
		pFlagOut[ index ] = flag;

	}

} // devCalculateVisibilityAndFlag

//
//	devApplyDensityMap()
//
//	CJS: 16/10/2018
//
//	Loop through a list of visibilities, and multiply each visibility by the density map at the position of the visibility.
//

__global__ void devApplyDensityMap( cufftComplex * pVisibilities, int * pDensityMap, int pItems )
{

	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	// ensure we are within the visibility batch limits.
	if (index < pItems)
	{

		// get the density at this grid position.
		double density = (double) pDensityMap[ index ];

		// multiply the visibility by the density at this position.
		pVisibilities[ index ].x *= density;
		pVisibilities[ index ].y *= density;

	}

} // devApplyDensityMap

//
//	devRearrangeKernel()
//
//	CJS: 22/01/2016
//
//	Rearrange a kernel so that the real numbers are at the start and the imaginary numbers are at the end.
//

__global__ void devRearrangeKernel( float * pTarget, float * pSource, long int pElements )
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
//	devMakeCleanBeam()
//
//	CJS: 06/11/2015
//
//	Constructs the clean beam from a set of parameters.
//

__global__ void devMakeCleanBeam( float * pCleanBeam, double pAngle, double pR1, double pR2, double pX, double pY, int pSize )
{
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	// check that we haven't gone outside the grid dimensions (some threads will).
	if ( i < pSize && j < pSize )
		
		// populate Gaussian image.
		pCleanBeam[ (j * pSize) + i ] = (float) gaussian2D(	/* pNormalisation = */ 1.0,
									/* pX = */ (double) i - pX,
									/* pY = */ (double) j - pY,
									/* pAngle = */ pAngle,
									/* pR1 = */ pR1,
									/* pR2 = */ pR2 );
	
} // devMakeCleanBeam

//
//	devMultiplyImages()
//
//	CJS: 18/01/2016
//
//	Multiply two complex images together.
//

__global__ void devMultiplyImages( cufftComplex * pOne, cufftComplex * pTwo, int pSize )
{
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	// check that we haven't gone outside the grid dimensions (some threads will).
	if ( i < pSize && j < pSize )
	{
		
		// multiply images together.
		pOne[ (j * pSize) + i ] = multComplex( /* pOne = */ pOne[ (j * pSize) + i ], /* pTwo = */ pTwo[ (j * pSize) + i ] );
		
	}
	
} // devMultiplyImages

__global__ void devMultiplyImages( float * pOne, float * pTwo, int pSize )
{
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	// check that we haven't gone outside the grid dimensions (some threads will).
	if ( i < pSize && j < pSize )
	{
		
		// multiply images together.
		pOne[ (j * pSize) + i ] *= pTwo[ (j * pSize) + i ];
		
	}
	
} // devMultiplyImages

__global__ void devMultiplyImages( float * pOne, float * pTwo, bool * pMask, int pSizeOne, int pSizeTwo )
{
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	// check that we haven't gone outside the grid dimensions of image one (some threads will).
	if ( i < pSizeOne && j < pSizeOne )
	{

		// ensure this pixel isn't masked.
		bool masked = false;
		if (pMask != NULL)
			masked = (pMask[ (j * pSizeOne) + i ] == false);
		if (masked == false)
		{

			// calculate position in image two.
			int iTwo = (int) ((double) i * (double) pSizeTwo / (double) pSizeOne);
			int jTwo = (int) ((double) j * (double) pSizeTwo / (double) pSizeOne);
		
			// multiply
			pOne[ (j * pSizeOne) + i ] *= pTwo[ (jTwo * pSizeTwo) + iTwo ];

		}
		else
			pOne[ (j * pSizeOne) + i ] = 0.0;
		
	}
	
} // devMultiplyImages

//
//	devDivideImages()
//
//	CJS: 18/01/2016
//
//	Divide one complex image by the magnitude of another.
//

__global__ void devDivideImages( double * pOne, float * pTwo, bool * pMask, int pSizeOne, int pSizeTwo )
{
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	// check that we haven't gone outside the grid dimensions of image one (some threads will).
	if ( i < pSizeOne && j < pSizeOne )
	{

		// ensure this pixel isn't masked.
		bool masked = false;
		if (pMask != NULL)
			masked = (pMask[ (j * pSizeOne) + i ] == false);
		if (masked == false)
		{

			// calculate position in image two.
			int iTwo = (int) ((double) i * (double) pSizeTwo / (double) pSizeOne);
			int jTwo = (int) ((double) j * (double) pSizeTwo / (double) pSizeOne);
		
			// divide images.
			if (pTwo[ (jTwo * pSizeTwo) + iTwo ] == 0.0)
				pOne[ (j * pSizeOne) + i ] = 0.0;
			else
				pOne[ (j * pSizeOne) + i ] /= (double) (pTwo[ (jTwo * pSizeTwo) + iTwo ]);

		}
		else
			pOne[ (j * pSizeOne) + i ] = 0.0;
		
	}
	
} // devDivideImages

__global__ void devDivideImages( float * pOne, float * pTwo, bool * pMask, int pSizeOne, int pSizeTwo )
{
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	// check that we haven't gone outside the grid dimensions of image one (some threads will).
	if ( i < pSizeOne && j < pSizeOne )
	{

		// ensure this pixel isn't masked.
		bool masked = false;
		if (pMask != NULL)
			masked = (pMask[ (j * pSizeOne) + i ] == false);
		if (masked == false)
		{

			// calculate position in image two.
			int iTwo = (int) ((double) i * (double) pSizeTwo / (double) pSizeOne);
			int jTwo = (int) ((double) j * (double) pSizeTwo / (double) pSizeOne);
		
			// divide images.
			if (pTwo[ (jTwo * pSizeTwo) + iTwo ] == 0.0)
				pOne[ (j * pSizeOne) + i ] = 0.0;
			else
				pOne[ (j * pSizeOne) + i ] /= pTwo[ (jTwo * pSizeTwo) + iTwo ];

		}
		else
			pOne[ (j * pSizeOne) + i ] = 0.0;
		
	}
	
} // devDivideImages

//
//	devAddComplexData()
//
//	CJS: 22/06/2020
//
//	Adds two sets of complex data using non-atomic additions.
//

__global__ void devAddComplexData( cufftComplex * pOne, cufftComplex * pTwo, int pElements )
{
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	// check that we are within the array bounds.
	if (i < pElements)
	{
		cufftComplex addition = pTwo[ i ];
		pOne[ i ].x += addition.x;
		pOne[ i ].y += addition.y;
	}

} // devAddComplexData

//
//	devUpdateKernel()
//
//	CJS: 18/01/2016
//
//	Updates the kernel function by extracting the centre part of an image.
//

__global__ void devUpdateKernel( cufftComplex * pKernel, cufftComplex * pImage, int pSupport, int pOversample, int pOversampleI, int pOversampleJ,
					int pWorkspaceSize, griddegrid pGridDegrid )
{
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	// get the sizes of the kernel and the oversampled kernel.
	int kernelSize = (pSupport * 2) + 1;

	// get the workspace support.
	int workspaceSupport = pWorkspaceSize / 2;
	
	// check that we haven't gone outside the kernel dimensions (some threads will).
	if (i < kernelSize && j < kernelSize)
	{

		// get the coordinates within the workspace.
		int imageI = workspaceSupport + ((i - pSupport) * pOversample);
		int imageJ = workspaceSupport + ((j - pSupport) * pOversample);
		if (pGridDegrid == GRID)
		{
			imageI -= pOversampleI;
			imageJ -= pOversampleJ;
		}
		else
		{
			imageI += pOversampleI;
			imageJ += pOversampleJ;
		}
				
		// check that we haven't gone outside the workspace dimensions.
		cufftComplex value;
		value.x = 0; value.y = 0;
		if (imageI >= 0 && imageI < pWorkspaceSize && imageJ >= 0 && imageJ < pWorkspaceSize)
			value = pImage[ (imageJ * pWorkspaceSize) + imageI ];

		// update the kernel.
		pKernel[ (j * kernelSize) + i ] = value;
		
	}
	
} // devUpdateKernel

//
//	devSubtractBeam()
//
//	CJS: 06/11/2015
//
//	Add or subtracts the clean beam/dirty beam from the clean image/dirty image.
//
//	The window size is the support size of the region of the beam that is to be added or subtracted. the rest of the beam outside this region is ignored.
//

__global__ void devSubtractBeam(	float * pImage, float * pBeam, double * pMaxValue, int pWindowSize, double pLoopGain,
						int pImageWidth, int pImageHeight, int pBeamSize	)
{
	
	__shared__ double maxValue;
	__shared__ double maxX;
	__shared__ double maxY;
	
	// retrieve the maximum value and pixel position.
	if ( threadIdx.x == 0 && threadIdx.y == 0 )
	{
		
		maxValue = pMaxValue[ MAX_PIXEL_VALUE ];
		maxX = pMaxValue[ MAX_PIXEL_X ];
		maxY = pMaxValue[ MAX_PIXEL_Y ];
		
	}
	
	__syncthreads();
	
	// calculate position in clean beam image (i,j).
	int i = (blockIdx.x * blockDim.x) + threadIdx.x + (_devPsfX - pWindowSize);
	int j = (blockIdx.y * blockDim.y) + threadIdx.y + (_devPsfY - pWindowSize);
	
	// check that we haven't gone outside the grid dimensions (some threads will).
	if (i >= 0 && i < pBeamSize && j >= 0 && j < pBeamSize)
	{
				
		// calculate position in clean image (x,y).
		int x = (int) round( maxX ) + (i - _devPsfX);
		int y = (int) round( maxY ) + (j - _devPsfY);
				
		// are we within the image bounds ?
		if (x >= 0 && x < pImageWidth && y >= 0 && y < pImageHeight)
		{
					
			// get some pointers to the image, psf, and primary beam pattern.
			float * tmpImage = &pImage[ (y * pImageWidth) + x ];
			float * tmpPSF = &pBeam[ (j * pBeamSize) + i ];
						
			// subtract the psf (scaled).
			*tmpImage -= maxValue * *tmpPSF * pLoopGain;
					
		}
		
	}
	
} // devSubtractBeam

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
//	devUpdateComplexArray()
//
//	CJS: 23/11/2015
//
//	Update the elements of a complex array.
//

__global__ void devUpdateComplexArray( cufftComplex * pArray, int pElements, float pReal, float pImaginary )
{
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	// check that we are within the array bounds.
	if (i < pElements)
	{
		pArray[ i ].x = pReal;
		pArray[ i ].y = pImaginary;
	}
	
} // devUpdateComplexArray

//
//	spheroidalWaveFunction()
//
//	CJS: 09/04/2019
//
//	Returns the prolate spheroidal wave function at a distance pR from the centre.
//

__host__ __device__ double spheroidalWaveFunction( double pR )
{
	
	// constants.
	const int NP = 4;
	const int NQ = 2;
	
	// data for spheroidal gridding function.
	double dataP[2][5] = {	{8.203343e-2, -3.644705e-1, 6.278660e-1, -5.335581e-1, 2.312756e-1},
				{4.028559e-3, -3.697768e-2, 1.021332e-1, -1.201436e-1, 6.412774e-2}	};
	double dataQ[2][3] = {	{1.0, 8.212018e-1, 2.078043e-1},
				{1.0, 9.599102e-1, 2.918724e-1} };
					
	// now, calculate the anti-aliasing kernel.
	double val = 0;
						
	// only calculate a kernel value if this pixel is within the required radius from the centre.
	if (pR <= 1)
	{
						
		int part = 1;
		double radiusEnd = 1.0;
		if (abs( pR ) < 0.75)
		{
			part = 0;
			radiusEnd = 0.75;
		}
							
		double delRadiusSq = (pR * pR) - (radiusEnd * radiusEnd);
							
		double top = 0;
		for ( int k = 0; k < NP; k++ )
			top = top + (dataP[ part ][ k ] * pow( delRadiusSq, k ));
			
		double bottom = 0;
		for ( int k = 0; k < NQ; k++ )
			bottom = bottom + (dataQ[ part ][ k ] * pow( delRadiusSq, k ));
					
		if (bottom != 0)
			val = top / bottom;
							
		// the gridding function is (1 - spheroidRadius^2) x gridsf
		val = val * (1 - (pR * pR));

	}

	// return something.
	return val;

} // spheroidalWaveFunction

//
//	devGenerateAAKernel()
//
//	CJS: 11/03/2016
//
//	Generate the anti-aliasing kernel in parallel.
//

__global__ void devGenerateAAKernel( cufftComplex * pAAKernel, int pKernelSize, int pWorkspaceSize )
{

	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;

	// ensure we're within the bounds of the workspace.
	if (i < pWorkspaceSize && j < pWorkspaceSize)
	{
		
		int supportSize = (pKernelSize - 1) / 2;
		int workspaceSupport = pWorkspaceSize / 2;

		// i and j are the coordinates within the aa-kernel. get the equivalent coordinates within the whole image.
		int imageI = i - workspaceSupport + supportSize;
		int imageJ = j - workspaceSupport + supportSize;
		
		// ensure we're within the bounds of the kernel.
		if (imageI >= 0 && imageI < pKernelSize && imageJ >= 0 && imageJ < pKernelSize)
		{

			// get kernel pointer.
			cufftComplex * aaKernelPtr = &pAAKernel[ (j * pWorkspaceSize) + i ];
					
			// calculate the x-offset from the centre of the kernel.
			double x = (double) (imageI - supportSize);
			double y = (double) (imageJ - supportSize);
					
			// now, calculate the anti-aliasing kernel.
			double val = 0;
//			val = spheroidalWaveFunction( x / ((double) supportSize + 0.5) );
//			val *= spheroidalWaveFunction( y / ((double) supportSize + 0.5) );
			val = spheroidalWaveFunction( x / (double) supportSize );
			val *= spheroidalWaveFunction( y / (double) supportSize );

			// update the appropriate pixel in the anti-aliasing kernel.
			aaKernelPtr->x = (float) val;
			aaKernelPtr->y = 0;
		
		}

	}

} // devGenerateAAKernel

//
//	devGenerateWKernel()
//
//	CJS: 14/03/2016
//
//	Generate the W-kernel in parallel.
//

__global__ void devGenerateWKernel( cufftComplex * pWKernel, double pW, int pWorkspaceSize, double pCellSizeDirectionalCosine, griddegrid pGridDegrid, int pSize )
{

	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;

	// ensure we're within the bounds of the kernel.
	if (i < pWorkspaceSize && j < pWorkspaceSize)
	{

		int workspaceSupport = pWorkspaceSize / 2;

		// l and m are the directional cosines in the x and y directions respectively. they are given by:
		//
		// 	l = x.sin( cell_size ), m = y.sin( cell_size )
		//
		// sin( cell_size ) is found in pCellSizeDirectionalCosine. we need to scale the directional cosine by the ratio of the image size to the workspace size
		// in order that the image we construct is an exact scaled version of what we would have if we were using the whole image size.
		double workspaceCellSize = pCellSizeDirectionalCosine * (double) pSize / (double) pWorkspaceSize;
					
		// calculate the offset in radians from the centre of the image.
		double l = (double) (i - workspaceSupport) * workspaceCellSize;
		double m = (double) (j - workspaceSupport) * workspaceCellSize;

		// calculate r^2 (dist from centre of image squared).
		double rSquared = pow( l, 2 ) + pow( m, 2 );

		// calculate kernel value. if gridding then the sine term should be negative; if degridding it should be positive.
		cufftComplex kernelValue = { .x = 0.0, .y = 0.0 };
		if (rSquared <= 1.0)
		{

			//
			// visibility equation is:
			//
			// 	Vj = int{ I(l,m) / sqrt(1-l^2-m^2) x exp[ i.2.PI( uj.l + vj.m + wj.(sqrt(1-l^2-m^2)-1) ) ] } dl dm
			//
			// if we're gridding we want to remove:
			//
			//	exp[ i.2.PI.wj.(sqrt(1-l^2-m^2)-1) ] / sqrt(1-l^2-m^2)
			//
			// from the image domain using a multiplication, so that we're left with the Fourier transform:
			//
			// 	Vj = int{ I(l,m} x exp[ i.2.PI( uj.l + vj.m ) ] } dl dm.
			//
			// if we're degridding then we want to add these components back in.
			//
			double exponent = (pGridDegrid == GRID ? -1 : +1) * 2.0 * PI * pW * (sqrt( 1.0 - rSquared ) - 1.0);
			sincosf( exponent, &(kernelValue.y), &(kernelValue.x) );

			// if we are gridding then we multiply the kernel by sqrt( 1.0 - l^2 - m^2 ). If degridding then this is a division.
			// NOTE: Neither Tim Corwell's paper, or ASKAPsoft, includes the bits below.
			if (pGridDegrid == GRID)
			{
				kernelValue.x *= sqrt( 1.0 - rSquared );
				kernelValue.y *= sqrt( 1.0 - rSquared );
			}
			else
			{
				kernelValue.x /= sqrt( 1.0 - rSquared );
				kernelValue.y /= sqrt( 1.0 - rSquared );
			}

		}

		// update kernel.
		pWKernel[ (j * pWorkspaceSize) + i ] = kernelValue;

	}

} // devGenerateWKernel

//
//	devGenerateAKernel()
//
//	CJS: 08/11/2018
//
//	Generate the A-kernel in parallel.
//

__global__ void devGenerateAKernel( cufftComplex * pAKernel, float * pPrimaryBeam, int pPrimaryBeamSupport, int pWorkspaceSize, griddegrid pGridDegrid )
{

	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;

	// ensure we're within the bounds of the kernel.
	if (i < pWorkspaceSize && j < pWorkspaceSize)
	{

		// initialise the kernel value to zero.
		double kernelValue = 0;

		// calculate the primary beam size, and the workspace support.
		int beamSize = pPrimaryBeamSupport * 2;

		// calculate the required position within the primary beam.
		int iBeam = (int) ((double) i * (double) beamSize / (double) pWorkspaceSize);
		int jBeam = (int) ((double) j * (double) beamSize / (double) pWorkspaceSize);

		// ensure we're within the bounds of the beam.
		if (iBeam >= 0 && iBeam < beamSize && jBeam >= 0 && jBeam < beamSize)
		{

			float primaryBeam = pPrimaryBeam[ (jBeam * beamSize) + iBeam ];
			if (pGridDegrid == GRID)
			{

				// set the kernel to the inverse of the primary beam.
				if (primaryBeam != 0)
					kernelValue = 1.0 / (double) primaryBeam;

			}
			else

				// we are degridding. set the kernel to the primary beam so that we ignore sources outside our field of view.
				kernelValue = (double) primaryBeam;

		}

		// update kernel.
		pAKernel[ (j * pWorkspaceSize) + i ].x = (float) kernelValue;
		pAKernel[ (j * pWorkspaceSize) + i ].y = 0.0;

	}

} // devGenerateAKernel

//
//	devCopyImage()
//
//	CJS: 29/03/2019
//
//	Copy one image into another image.
//
//	Preconditions:	1. The size of the new image is always >= the size of the old image.
//			2. The scale is always >= 1, which means objects in the old image will appear smaller in the new image.
//

__global__ void devCopyImage( cufftComplex * pNewImage, cufftComplex * pOldImage, int pNewSize, int pOldSize, double pScale, int pThreadOffset )
{

	// calculate support.
	int oldSupport = pOldSize / 2;
	int newSupport = pNewSize / 2;

	// the thread indexes correspond to pixels in the new image, but we add an offset because we may only be interested in updating a small portion of the new image.
	int iNew = (blockIdx.x * blockDim.x) + threadIdx.x + pThreadOffset;
	int jNew = (blockIdx.y * blockDim.y) + threadIdx.y + pThreadOffset;

	// ensure we're within the bounds of the new image.
	if (iNew >= 0 && iNew < pNewSize && jNew >= 0 && jNew < pNewSize)
	{

		// calculate old pixel position.
		int iOld = (int) ((double) (iNew - newSupport) * pScale) + oldSupport;
		int jOld = (int) ((double) (jNew - newSupport) * pScale) + oldSupport;

		// copy pixel from old image to new image.
		if (iOld >= 0 && iOld < pOldSize && jOld >= 0 && jOld < pOldSize )
			pNewImage[ (jNew * pNewSize) + iNew ] = pOldImage[ (jOld * pOldSize) + iOld ];

	}

} // devCopyImage

__global__ void devCopyImage( cufftComplex * pNewImage, float * pOldImage, int pNewSize, int pOldSize, double pScale, int pThreadOffset )
{

	// calculate support.
	int oldSupport = pOldSize / 2;
	int newSupport = pNewSize / 2;

	// the thread indexes correspond to pixels in the new image, but we add an offset because we may only be interested in updating a small portion of the new image.
	int iNew = (blockIdx.x * blockDim.x) + threadIdx.x + pThreadOffset;
	int jNew = (blockIdx.y * blockDim.y) + threadIdx.y + pThreadOffset;

	// ensure we're within the bounds of the new image.
	if (iNew >= 0 && iNew < pNewSize && jNew >= 0 && jNew < pNewSize)
	{

		// calculate old pixel position.
		int iOld = (int) ((double) (iNew - newSupport) * pScale) + oldSupport;
		int jOld = (int) ((double) (jNew - newSupport) * pScale) + oldSupport;

		// copy pixel from old image to new image.
		if (iOld >= 0 && iOld < pOldSize && jOld >= 0 && jOld < pOldSize )
		{
			pNewImage[ (jNew * pNewSize) + iNew ].x = pOldImage[ (jOld * pOldSize) + iOld ];
			pNewImage[ (jNew * pNewSize) + iNew ].y = 0.0;
		}

	}

} // devCopyImage

__global__ void devCopyImage( double * pNewImage, double * pOldImage, int pNewSize, int pOldSize, double pScale, int pThreadOffset )
{

	// calculate support.
	int oldSupport = pOldSize / 2;
	int newSupport = pNewSize / 2;

	// the thread indexes correspond to pixels in the new image, but we add an offset because we may only be interested in updating a small portion of the new image.
	int iNew = (blockIdx.x * blockDim.x) + threadIdx.x + pThreadOffset;
	int jNew = (blockIdx.y * blockDim.y) + threadIdx.y + pThreadOffset;

	// ensure we're within the bounds of the new image.
	if (iNew >= 0 && iNew < pNewSize && jNew >= 0 && jNew < pNewSize)
	{

		// calculate old pixel position.
		int iOld = (int) ((double) (iNew - newSupport) * pScale) + oldSupport;
		int jOld = (int) ((double) (jNew - newSupport) * pScale) + oldSupport;

		// copy pixel from old image to new image.
		if (iOld >= 0 && iOld < pOldSize && jOld >= 0 && jOld < pOldSize )
			pNewImage[ (jNew * pNewSize) + iNew ] = pOldImage[ (jOld * pOldSize) + iOld ];

	}

} // devCopyImage

__global__ void devCopyImage( float * pNewImage, float * pOldImage, int pNewSize, int pOldSize, double pScale, int pThreadOffset )
{

	// calculate support.
	int oldSupport = pOldSize / 2;
	int newSupport = pNewSize / 2;

	// the thread indexes correspond to pixels in the new image, but we add an offset because we may only be interested in updating a small portion of the new image.
	int iNew = (blockIdx.x * blockDim.x) + threadIdx.x + pThreadOffset;
	int jNew = (blockIdx.y * blockDim.y) + threadIdx.y + pThreadOffset;

	// ensure we're within the bounds of the new image.
	if (iNew >= 0 && iNew < pNewSize && jNew >= 0 && jNew < pNewSize)
	{

		// calculate old pixel position.
		int iOld = (int) ((double) (iNew - newSupport) * pScale) + oldSupport;
		int jOld = (int) ((double) (jNew - newSupport) * pScale) + oldSupport;

		// copy pixel from old image to new image.
		if (iOld >= 0 && iOld < pOldSize && jOld >= 0 && jOld < pOldSize )
			pNewImage[ (jNew * pNewSize) + iNew ] = pOldImage[ (jOld * pOldSize) + iOld ];

	}

} // devCopyImage

//
//	devSubtractVisibilities()
//
//	CJS: 15/08/2018
//
//	Subtract the model visibilities from the original visibilities.
//

__global__ void devSubtractVisibilities( cufftComplex * pOriginalVisibility, cufftComplex * pModelVisibility, int pItems )
{

	int i = (blockIdx.x * blockDim.x) + threadIdx.x;

	// ensure we're within the bounds of the kernel.
	if (i >= 0 && i < pItems)
	{
		pModelVisibility[ i ].x = pOriginalVisibility[ i ].x - pModelVisibility[ i ].x;
		pModelVisibility[ i ].y = pOriginalVisibility[ i ].y - pModelVisibility[ i ].y;
	}

} // devSubtractVisibilities

//
//	HOST FUNCTIONS
//

//
//	setThreadBlockSize1D()
//
//	CJS:	06/11/2015
//
//	Determine a suitable thread and block size for the current GPU.
//	The number of threads must be less than the maximum number allowed by the current GPU.
//

void setThreadBlockSize1D( int * pThreads, int * pBlocks )
{
	
	// store the total number of threads.
	int totalThreads = *pThreads;
	
	*pBlocks = 1;
	if ( *pThreads > _maxThreadsPerBlock )
	{
		*pThreads = _maxThreadsPerBlock;
		*pBlocks = (totalThreads / _maxThreadsPerBlock);
		if (totalThreads % _maxThreadsPerBlock != 0)
			(*pBlocks)++;
	}
	
} // setThreadBlockSize1D

//
//	setThreadBlockSize2D()
//
//	CJS:	10/11/2015
//
//	Determine a suitable thread and block size for the current GPU.
//	The number of threads must be less than the maximum number allowed by the current GPU.
//
//	This subroutine is used when we have a single, large XxY grid (i.e. when we are processing an image).
//

void setThreadBlockSize2D( int pThreadsX, int pThreadsY )
{
	
	// store the total number of X and Y threads.
	int totalThreadsX = pThreadsX;
	int totalThreadsY = pThreadsY;
	
	_blockSize2D.x = pThreadsX;
	_blockSize2D.y = pThreadsY;
	_gridSize2D.x = 1;
	_gridSize2D.y = 1;
	
	// do we have too many threads?
	while ( (_blockSize2D.x * _blockSize2D.y) > _maxThreadsPerBlock )
	{
		
		// increment the number of Y blocks.
		_gridSize2D.y = _gridSize2D.y + 1;
		_blockSize2D.y = (int) ceil( (double) totalThreadsY / (double) _gridSize2D.y );
		
		// if this doesn't help, increment the number of X blocks. if we have multiple iterations of this loop then
		// we will be incrementing Y, X, Y, X, Y, X, Y, .... etc.
		if ( (_blockSize2D.x * _blockSize2D.y) > _maxThreadsPerBlock )
		{
			_gridSize2D.x = _gridSize2D.x + 1;
			_blockSize2D.x = (int) ceil( (double) totalThreadsX / (double) _gridSize2D.x );
		}
		
	}
	
} // setThreadBlockSize2D

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
		while ((_blockSize2D.x * _blockSize2D.y) > _maxThreadsPerBlock)
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
//	freeData()
//
//	CJS: 27/03/2019
//
//	Free the memory used to store the data for a mosaic image. If an offset is supplied then we don't free the array, we only free the cells after the offset.
//

void freeData( int pWhatData )
{

	// free the memory.
	if ((pWhatData & DATA_VISIBILITIES) == DATA_VISIBILITIES && _hstVisibility != NULL)
	{
		free( (void *) _hstVisibility );
		_hstVisibility = NULL;
	}
	if ((pWhatData & DATA_GRID_POSITIONS) == DATA_GRID_POSITIONS && _hstGridPosition != NULL)
	{
		free( (void *) _hstGridPosition );
		_hstGridPosition = NULL;
	}
	if ((pWhatData & DATA_KERNEL_INDEXES) == DATA_KERNEL_INDEXES && _hstKernelIndex != NULL)
	{
		free( (void *) _hstKernelIndex );
		_hstKernelIndex = NULL;
	}
	if ((pWhatData & DATA_DENSITIES) == DATA_DENSITIES && _hstDensityMap != NULL)
	{
		free( (void *) _hstDensityMap );
		_hstDensityMap = NULL;
	}
	if ((pWhatData & DATA_WEIGHTS) == DATA_WEIGHTS && _hstWeight != NULL)
	{
		free( (void *) _hstWeight );
		_hstWeight = NULL;
	}
	if ((pWhatData & DATA_RESIDUAL_VISIBILITIES) == DATA_RESIDUAL_VISIBILITIES && _hstResidualVisibility != NULL)
	{
		free( (void *) _hstResidualVisibility );
		_hstResidualVisibility = NULL;
	}

} // freeData

//
//	cacheData()
//
//	CJS: 25/03/2019
//
//	Store a whole set of visibilities, grid positions, kernel indexes, etc to disk, and free the memory. We use the offset parameter in the rare cases
//	where the data does not start at the beginning of the array. Most of the time the offset will be zero.
//

void cacheData( char * pMeasurementSetFilename, int pMosaicID, int pBatchID, int pWhatData )
{

	// build filename.
	char filename[ 255 ];

	// build the full filename.
	if (_hstCacheLocation[0] != '\0')
		sprintf( filename, "%s%s-%02i-%i-cache.dat", _hstCacheLocation, pMeasurementSetFilename, pMosaicID, pBatchID );
	else
		sprintf( filename, "%s-%02i-%i-cache.dat", pMeasurementSetFilename, pMosaicID, pBatchID );

	// open the file for writing.
	FILE * fr;
	if (pWhatData == DATA_ALL)
		fr = fopen( filename, "wb" );
	else
	{
		fr = fopen( filename, "r+b" );
		fseek( fr, 0, SEEK_SET );
	}
	
	// write the data.
	if ((pWhatData & DATA_VISIBILITIES) == DATA_VISIBILITIES)
		fwrite( (void *) _hstVisibility, sizeof( cufftComplex ), _hstNumVisibilities[ pMosaicID ][ pBatchID ], fr );
	else
		fseek( fr, _hstNumVisibilities[ pMosaicID ][ pBatchID ] * sizeof( cufftComplex ), SEEK_CUR );
	if ((pWhatData & DATA_GRID_POSITIONS) == DATA_GRID_POSITIONS)
		fwrite( (void *) _hstGridPosition, sizeof( VectorI ), _hstNumVisibilities[ pMosaicID ][ pBatchID ], fr );
	else
		fseek( fr, _hstNumVisibilities[ pMosaicID ][ pBatchID ] * sizeof( VectorI ), SEEK_CUR );
	if ((pWhatData & DATA_KERNEL_INDEXES) == DATA_KERNEL_INDEXES)
		fwrite( (void *) _hstKernelIndex, sizeof( int ), _hstNumVisibilities[ pMosaicID ][ pBatchID ], fr );
	else
		fseek( fr, _hstNumVisibilities[ pMosaicID ][ pBatchID ] * sizeof( int ), SEEK_CUR );
	if ((pWhatData & DATA_DENSITIES) == DATA_DENSITIES)
		fwrite( (void *) _hstDensityMap, sizeof( int ), _hstNumVisibilities[ pMosaicID ][ pBatchID ], fr );
	else
		fseek( fr, _hstNumVisibilities[ pMosaicID ][ pBatchID ] * sizeof( int ), SEEK_CUR );
	if ((pWhatData & DATA_WEIGHTS) == DATA_WEIGHTS)
		fwrite( (void *) _hstWeight, sizeof( float ), _hstNumVisibilities[ pMosaicID ][ pBatchID ], fr );
	else
		fseek( fr, _hstNumVisibilities[ pMosaicID ][ pBatchID ] * sizeof( float ), SEEK_CUR );
	if ((pWhatData & DATA_RESIDUAL_VISIBILITIES) == DATA_RESIDUAL_VISIBILITIES)
		fwrite( (void *) _hstResidualVisibility, sizeof( cufftComplex ), _hstNumVisibilities[ pMosaicID ][ pBatchID ], fr );

	// close the file.
	fclose( fr );

	// free the memory. if there's an offset then we don't free the array - we only free the cells after the offset.
	freeData( /* pWhatData = */ pWhatData );

} // cacheData

//
//	uncacheData()
//
//	CJS: 25/03/2019
//
//	Retrieve a whole set of visibilities, grid positions, kernel indexes, etc from disk. if an offset is supplied then the data should be loaded
//	into the arrays at this position, so we need to expand rather than initialise the arrays.
//

void uncacheData( char * pFilenamePrefix, int pMosaicID, int pBatchID, int pWhatData, long int pOffset )
{

	// build filename.
	char filename[ 255 ];

	// build the full filename.
	if (_hstCacheLocation[0] != '\0')
		sprintf( filename, "%s%s-%02i-%i-cache.dat", _hstCacheLocation, pFilenamePrefix, pMosaicID, pBatchID );
	else
		sprintf( filename, "%s-%02i-%i-cache.dat", pFilenamePrefix, pMosaicID, pBatchID );

	// open the file for reading.
	FILE * fr = fopen( filename, "rb" );

	// create the required memory, and read the data.
	if ((pWhatData & DATA_VISIBILITIES) == DATA_VISIBILITIES)
	{
		if (pOffset == 0)
			_hstVisibility = (cufftComplex *) malloc( _hstNumVisibilities[ pMosaicID ][ pBatchID ] * sizeof( cufftComplex ) );
		else
			_hstVisibility = (cufftComplex *) realloc( _hstVisibility, (pOffset + _hstNumVisibilities[ pMosaicID ][ pBatchID ]) * sizeof( cufftComplex ) );
		fread( (void *) &_hstVisibility[ pOffset ], sizeof( cufftComplex ), _hstNumVisibilities[ pMosaicID ][ pBatchID ], fr );
	}
	else
		fseek( fr, _hstNumVisibilities[ pMosaicID ][ pBatchID ] * sizeof( cufftComplex ), SEEK_CUR );
	if ((pWhatData & DATA_GRID_POSITIONS) == DATA_GRID_POSITIONS)
	{
		if (pOffset == 0)
			_hstGridPosition = (VectorI *) malloc( _hstNumVisibilities[ pMosaicID ][ pBatchID ] * sizeof( VectorI ) );
		else
			_hstGridPosition = (VectorI *) realloc( _hstGridPosition, (pOffset + _hstNumVisibilities[ pMosaicID ][ pBatchID ]) * sizeof( VectorI ) );
		fread( (void *) &_hstGridPosition[ pOffset ], sizeof( VectorI ), _hstNumVisibilities[ pMosaicID ][ pBatchID ], fr );
	}
	else
		fseek( fr, _hstNumVisibilities[ pMosaicID ][ pBatchID ] * sizeof( VectorI ), SEEK_CUR );
	if ((pWhatData & DATA_KERNEL_INDEXES) == DATA_KERNEL_INDEXES)
	{
		if (pOffset == 0)
			_hstKernelIndex = (int *) malloc( _hstNumVisibilities[ pMosaicID ][ pBatchID ] * sizeof( int ) );
		else
			_hstKernelIndex = (int *) realloc( _hstKernelIndex, (pOffset + _hstNumVisibilities[ pMosaicID ][ pBatchID ]) * sizeof( int ) );
		fread( (void *) &_hstKernelIndex[ pOffset ], sizeof( int ), _hstNumVisibilities[ pMosaicID ][ pBatchID ], fr );
	}
	else
		fseek( fr, _hstNumVisibilities[ pMosaicID ][ pBatchID ] * sizeof( int ), SEEK_CUR );
	if ((pWhatData & DATA_DENSITIES) == DATA_DENSITIES)
	{
		if (pOffset == 0)
			_hstDensityMap = (int *) malloc( _hstNumVisibilities[ pMosaicID ][ pBatchID ] * sizeof( int ) );
		else
			_hstDensityMap = (int *) realloc( _hstDensityMap, (pOffset + _hstNumVisibilities[ pMosaicID ][ pBatchID ]) * sizeof( int ) );
		fread( (void *) &_hstDensityMap[ pOffset ], sizeof( int ), _hstNumVisibilities[ pMosaicID ][ pBatchID ], fr );
	}
	else
		fseek( fr, _hstNumVisibilities[ pMosaicID ][ pBatchID ] * sizeof( int ), SEEK_CUR );
	if ((pWhatData & DATA_WEIGHTS) == DATA_WEIGHTS)
	{
		if (pOffset == 0)
			_hstWeight = (float *) malloc( _hstNumVisibilities[ pMosaicID ][ pBatchID ] * sizeof( float ) );
		else
			_hstWeight = (float *) realloc( _hstWeight, (pOffset + _hstNumVisibilities[ pMosaicID ][ pBatchID ]) * sizeof( float ) );
		fread( (void *) &_hstWeight[ pOffset ], sizeof( float ), _hstNumVisibilities[ pMosaicID ][ pBatchID ], fr );
	}
	else
		fseek( fr, _hstNumVisibilities[ pMosaicID ][ pBatchID ] * sizeof( float ), SEEK_CUR );
	if ((pWhatData & DATA_RESIDUAL_VISIBILITIES) == DATA_RESIDUAL_VISIBILITIES)
	{
		if (pOffset == 0)
			_hstResidualVisibility = (cufftComplex *) malloc( _hstNumVisibilities[ pMosaicID ][ pBatchID ] * sizeof( cufftComplex ) );
		else
			_hstResidualVisibility = (cufftComplex *) realloc( _hstResidualVisibility, (pOffset + _hstNumVisibilities[ pMosaicID ][ pBatchID ]) *
										sizeof( cufftComplex ) );
		fread( (void *) &_hstResidualVisibility[ pOffset ], sizeof( cufftComplex ), _hstNumVisibilities[ pMosaicID ][ pBatchID ], fr );
	}

	// close the file.
	fclose( fr );

} // uncacheData

//
//	saveComplexData()
//
//	CJS: 16/06/2020
//
//	Cache some complex data to disk.
//

void saveComplexData( char * pFilename, cufftComplex * pData, long int pOffset, long int pSize )
{

	// open the file for writing.
	FILE * fr;
	if (pOffset == 0)
		fr = fopen( pFilename, "wb" );
	else
	{
		fr = fopen( pFilename, "r+b" );
		fseek( fr, pOffset, SEEK_SET );
	}

	// write the data.
	fwrite( (void *) pData, sizeof( cufftComplex ), pSize, fr );

	// close the file.
	fclose( fr );

} // saveComplexData

void saveComplexData( char * pFilename, float * pData, long int pOffset, long int pSize )
{

	// open the file for writing.
	FILE * fr;
	if (pOffset == 0)
		fr = fopen( pFilename, "wb" );
	else
	{
		fr = fopen( pFilename, "r+b" );
		fseek( fr, pOffset, SEEK_SET );
	}

	// write the data.
	fwrite( (void *) pData, sizeof( float ), pSize, fr );

	// close the file.
	fclose( fr );

} // saveComplexData

//
//	getComplexData()
//
//	CJS: 16/06/2020
//
//	Retrieve some complex data from the disk cache.
//

void getComplexData( char * pFilename, cufftComplex * pData, long int pOffset, long int pSize )
{

	// open the file for reading.
	FILE * fr = fopen( pFilename, "rb" );

	// seek the required part of the file.
	fseek( fr, pOffset, SEEK_CUR );

	// read the data.
	fread( (void *) pData, sizeof( cufftComplex ), pSize, fr );

	// close the file.
	fclose( fr );

} // getComplexData

void getComplexData( char * pFilename, float * pData, long int pOffset, long int pSize )
{

	// open the file for reading.
	FILE * fr = fopen( pFilename, "rb" );

	// seek the required part of the file.
	fseek( fr, pOffset, SEEK_CUR );

	// read the data.
	fread( (void *) pData, sizeof( float ), pSize, fr );

	// close the file.
	fclose( fr );

} // getComplexData

//
//	moveDeviceToHost()
//
//	CJS: 29/04/2020
//
//	Copy some memory from the device to the host, and display an error if it failed.
//

bool moveDeviceToHost( void * pToPtr, void * pFromPtr, long int pSize, const char * pTask )
{

	cudaError_t err = cudaMemcpy( pToPtr, pFromPtr, pSize, cudaMemcpyDeviceToHost );
	if (err != cudaSuccess)
	{
		printf( "Error %s (%s)\n", pTask, cudaGetErrorString( err ) );
		exit( 1 );
	}

	// return something.
	return (err == cudaSuccess);

} // moveDeviceToHost

//
//	moveDeviceToHostAsync()
//
//	CJS: 01/06/2020
//
//	Copy some memory from the device to the host asynchronously.
//

void moveDeviceToHostAsync( void * pToPtr, void * pFromPtr, long int pSize, const char * pTask, cudaStream_t pStream )
{

	cudaError_t err = cudaMemcpyAsync( pToPtr, pFromPtr, pSize, cudaMemcpyDeviceToHost, pStream );
	if (err != cudaSuccess)
		printf( "Error %s (%s)\n", pTask, cudaGetErrorString( err ) );

} // moveDeviceToHostAsync

//
//	moveHostToDevice()
//
//	CJS: 29/04/2020
//
//	Copy some memory from the host to the device, and display an error if it failed.
//

bool moveHostToDevice( void * pToPtr, void * pFromPtr, long int pSize, const char * pTask )
{

	cudaError_t err = cudaMemcpy( pToPtr, pFromPtr, pSize, cudaMemcpyHostToDevice );
	if (err != cudaSuccess)
	{
		printf( "Error %s (%s)\n", pTask, cudaGetErrorString( err ) );
		exit( 1 );
	}

	// return something.
	return (err == cudaSuccess);

} // moveHostToDevice

//
//	moveHostToDeviceAsync()
//
//	CJS: 01/06/2020
//
//	Copy some memory from the host to the device asynchronously.
//

void moveHostToDeviceAsync( void * pToPtr, void * pFromPtr, long int pSize, const char * pTask, cudaStream_t pStream )
{

	cudaError_t err = cudaMemcpyAsync( pToPtr, pFromPtr, pSize, cudaMemcpyHostToDevice, pStream );
	if (err != cudaSuccess)
		printf( "Error %s (%s)\n", pTask, cudaGetErrorString( err ) );

} // moveHostToDeviceAsync

//
//	reserveGPUMemory()
//
//	CJS: 15/10/2019
//
//	Reserve some memory on the GPU, and display an error if this fails.
//

bool reserveGPUMemory( void ** pMemPtr, long int pSize, const char * pTask )
{

	// determine how much free memory is available.
//	size_t freeMem = 0, totalMem = 0;
//	cudaError_t errMem = cudaMemGetInfo( &freeMem, &totalMem );
//	if (errMem == cudaSuccess)
//		printf( "reserving %li MB. %li MB from %li MB available (%s)\n", pSize / (1024 * 1024), freeMem / (1024 * 1024), totalMem / (1024 * 1024), pTask );
//	else
//		printf( "cudaMemGetInfo failed in reserveGPUMemory() (%s)\n", cudaGetErrorString( errMem ) );

	cudaError_t err = cudaMalloc( pMemPtr, pSize );
	if (err != cudaSuccess)
	{
		printf( "Error %s (%s)\n", pTask, cudaGetErrorString( err ) );
		exit( 1 );
	}

	// return something.
	return (err == cudaSuccess);

} // reserveGPUMemory

//
//	zeroGPUMemory()
//
//	CJS: 15/10/2019
//
//	Zero some memory on the GPU, and display an error if this fails.
//

bool zeroGPUMemory( void * pMemPtr, long int pSize, const char * pTask )
{

	cudaError_t err = cudaMemset( pMemPtr, 0, pSize );
	if (err != cudaSuccess)
	{
		printf( "Error %s (%s)\n", pTask, cudaGetErrorString( err ) );
		exit( 1 );
	}

	// return something.
	return (err == cudaSuccess);

} // zeroGPUMemory

//
//	quickSortData()
//
//	CJS: 15/10/2018
//
//	Sort a list of kernel indexes and grid positions, and swap the corresponding visibilities (if required).
//

void quickSortData( long int pLeft, long int pRight )
{

	if (pLeft <= pRight)
	{
	
		long int i = pLeft, j = pRight;

		// use temporary values for swapping.
		VectorI tmpGridPosition;
		int tmpKernelIndex, tmpDensityMap, tmpFieldID;
		cufftComplex tmpComplex;
		float tmpWeight;
		bool tmpFlag;

		// we need to preserve and sort by field id for beam mosaicing.
		bool preserveField = (_hstBeamMosaic == true && _hstFieldIDArray != NULL);

		VectorI pivot = _hstGridPosition[ (pLeft + pRight) / 2 ];
		int pivotKernel = _hstKernelIndex[ (pLeft + pRight) / 2 ];
		int pivotFieldID = -1;
		if (preserveField == true)
			pivotFieldID = _hstFieldIDArray[ (pLeft + pRight) / 2 ];
		
		// partition, and sort by W plane, A plane, V position, U position, kernel index, and field ID (if we're preserving field id).
		while (i <= j)
		{

			while (true)
			{
				if (_hstGridPosition[ i ].w > pivot.w)
					break;
				if (_hstGridPosition[ i ].w == pivot.w)
				{
					if (_hstGridPosition[ i ].v > pivot.v)
						break;
					if (_hstGridPosition[ i ].v == pivot.v)
					{
						if (_hstGridPosition[ i ].u >= pivot.u)
							break;
						if (_hstGridPosition[ i ].u == pivot.u)
						{
							if (_hstKernelIndex[ i ] > pivotKernel)
								break;
							if (_hstKernelIndex[ i ] == pivotKernel && preserveField == true)
								if (_hstFieldIDArray[ i ] > pivotFieldID)
									break;
						}
					}
				}
				i = i + 1;
			}

			while (true)
			{
				if (_hstGridPosition[ j ].w < pivot.w)
					break;
				if (_hstGridPosition[ j ].w == pivot.w)
				{
					if (_hstGridPosition[ j ].v < pivot.v)
						break;
					if (_hstGridPosition[ j ].v == pivot.v)
					{
						if (_hstGridPosition[ j ].u <= pivot.u)
							break;
						if (_hstGridPosition[ j ].u == pivot.u)
						{
							if (_hstKernelIndex[ j ] < pivotKernel)
								break;
							if (_hstKernelIndex[ j ] == pivotKernel && preserveField == true)
								if (_hstFieldIDArray[ j ] < pivotFieldID)
									break;
						}
					}
				}
				j = j - 1;
			}

			if (i <= j)
			{

				// swap the grid positions, kernel indexes, visibilities, flags, field IDs and densities.
				tmpGridPosition = _hstGridPosition[ i ]; _hstGridPosition[ i ] = _hstGridPosition[ j ]; _hstGridPosition[ j ] = tmpGridPosition;
				tmpKernelIndex = _hstKernelIndex[ i ]; _hstKernelIndex[ i ] = _hstKernelIndex[ j ]; _hstKernelIndex[ j ] = tmpKernelIndex;
				tmpComplex = _hstVisibility[ i ]; _hstVisibility[ i ] = _hstVisibility[ j ]; _hstVisibility[ j ] = tmpComplex;
				if (_hstFlag != NULL)
				{
					tmpFlag = _hstFlag[ i ]; _hstFlag[ i ] = _hstFlag[ j ]; _hstFlag[ j ] = tmpFlag;
				}
				if (preserveField == true)
				{
					tmpFieldID = _hstFieldIDArray[ i ]; _hstFieldIDArray[ i ] = _hstFieldIDArray[ j ]; _hstFieldIDArray[ j ] = tmpFieldID;
				}
				tmpDensityMap = _hstDensityMap[ i ]; _hstDensityMap[ i ] = _hstDensityMap[ j ]; _hstDensityMap[ j ] = tmpDensityMap;
				if (_hstWeighting != NONE)
				{
					tmpWeight = _hstWeight[ i ]; _hstWeight[ i ] = _hstWeight[ j ]; _hstWeight[ j ] = tmpWeight;
				}

				i = i + 1;
				j = j - 1;

			}
		}
	
		// recursion.
		if (pLeft < j)
			quickSortData( pLeft, j );
		if (i < pRight)
			quickSortData( i, pRight );

	}
	
} // quickSortData

//
//	quickSortComponents()
//
//	CJS: 02/06/2020
//
//	Sort a list of clean image components into order of y position.
//

void quickSortComponents( VectorI * phstComponentListPos, double * phstComponentListValue, int pLeft, int pRight )
{

	long int i = pLeft, j = pRight;

	// use temporary values for swapping.
	VectorI tmpPos;
	double tmpValue;
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

			tmpPos = phstComponentListPos[ i ];
			phstComponentListPos[ i ] = phstComponentListPos[ j ];
			phstComponentListPos[ j ] = tmpPos;

			tmpValue = phstComponentListValue[ i ];
			phstComponentListValue[ i ] = phstComponentListValue[ j ];
			phstComponentListValue[ j ] = tmpValue;

			i = i + 1;
			j = j - 1;

		}

	}
	
	// recursion.
	if (pLeft < j)
		quickSortComponents( phstComponentListPos, phstComponentListValue, pLeft, j );
	if (i < pRight)
		quickSortComponents( phstComponentListPos, phstComponentListValue, i, pRight );

} // quickSortComponents

//
//	quickSortFieldIDs()
//
//	CJS: 19/11/2018
//
//	Sort a list of field IDs.
//

void quickSortFieldIDs( int * pFieldID, int pLeft, int pRight )
{
	
	long int i = pLeft, j = pRight;

	// use temporary values for swapping.
	int tmpFieldID;
	int pivot = pFieldID[ (pLeft + pRight) / 2 ];
		
	// partition.
	while (i <= j)
	{
		while (pFieldID[ i ] < pivot)
			i = i + 1;
		while (pFieldID[ j ] > pivot)
			j = j - 1;
		if (i <= j)
		{

			tmpFieldID = pFieldID[ i ];
			pFieldID[ i ] = pFieldID[ j ];
			pFieldID[ j ] = tmpFieldID;

			i = i + 1;
			j = j - 1;

		}
	}
	
	// recursion.
	if (pLeft < j)
		quickSortFieldIDs( pFieldID, pLeft, j );
	if (i < pRight)
		quickSortFieldIDs( pFieldID, i, pRight );
	
} // quickSortFieldIDs

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
//	initialiseFFT()
//
//	CJS: 15/03/2016
//
//	Create an FFT plan.
//

cufftHandle initialiseFFT( int pSize )
{

	// generate the FFT plan.
	cufftHandle fftPlan;
	cufftPlan2d( &fftPlan, pSize, pSize, CUFFT_C2C );
	
	// return the plan.
	return fftPlan;
	
} // initialiseFFT

//
//	finaliseFFT()
//
//	CJS: 15/03/2016
//
//	Delete an FFT plan.
//

void finaliseFFT( cufftHandle pFFTPlan )
{

	// destroy the FFT plan.
	cufftDestroy( pFFTPlan );
	pFFTPlan = -1;
	
} // finaliseFFT

//
//	performFFT()
//
//	CJS: 11/08/2015
//
//	Make a dirty image by inverse FFTing the gridded visibilites.
//

bool performFFT( cufftComplex ** pdevGrid, int pSize, fftdirection pFFTDirection, cufftHandle pFFTPlan, ffttype pFFTType )
{
	
	bool ok = true;
	cudaError_t err;

	// reserve some memory to hold a temporary image, which allows us to do an FFT shift.
	cufftComplex * devTmpImage = NULL;
	reserveGPUMemory( (void **) &devTmpImage, sizeof( cufftComplex ) * pSize * pSize, "reserving device memory for the FFT temporary image" );

	// reverse the y-direction because images produced from inverse FFTs will be upside-down, so we need our image to also be upside-down before
	// switching to the uv domain.
	if (pFFTDirection == FORWARD)
	{
		setThreadBlockSize2D( pSize, pSize / 2 );
		if (pFFTType == F2F || pFFTType == F2C)
			devReverseYDirection<<< _gridSize2D, _blockSize2D >>>(	/* pGrid = */ (float *) *pdevGrid,
										/* pSize = */ pSize );
		else
			devReverseYDirection<<< _gridSize2D, _blockSize2D >>>(	/* pGrid = */ *pdevGrid,
										/* pSize = */ pSize );
		err = cudaGetLastError();
		if (err != cudaSuccess)
		{
			printf( "error swapping image y-coordinates (%s)\n", cudaGetErrorString( err ) );
			ok = false;
		}
	}

	// do forward FFT shift.
	setThreadBlockSize2D( pSize, pSize );
	if (pFFTType == F2F || pFFTType == F2C)
		devFFTShift<<< _gridSize2D, _blockSize2D >>>(	/* pDestination = */ devTmpImage,
								/* pSource = */ (float *) *pdevGrid,
								/* pFFTDirection = */ FORWARD,
								/* pSize = */ pSize );
	else
		devFFTShift<<< _gridSize2D, _blockSize2D >>>(	/* pDestination = */ devTmpImage,
								/* pSource = */ *pdevGrid,
								/* pFFTDirection = */ FORWARD,
								/* pSize = */ pSize );

	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		printf( "error doing FFT shift prior to FFT (%s)\n", cudaGetErrorString( err ) );
		ok = false;
	}

	// if we're doing a 'from real' transform then free the grid and recreate it.
	if (pFFTType == F2C || pFFTType == F2F)
	{
		cudaFree( (void *) *pdevGrid );
		reserveGPUMemory( (void **) pdevGrid, pSize * pSize * sizeof( cufftComplex ), "creating device memory for enlarged grid following FFT" );
	}

	// move image from temporary memory.
	cudaMemcpy( (void *) *pdevGrid, (void *) devTmpImage, pSize * pSize * sizeof( cufftComplex ), cudaMemcpyDeviceToDevice );

	// free memory.
	if (devTmpImage != NULL)
		cudaFree( (void *) devTmpImage );

	// create the plan if it doesn't exist.
	bool destroyPlan = false;
	if (pFFTPlan == -1)
	{
		pFFTPlan = initialiseFFT( pSize );
		destroyPlan = true;
	}

	// execute the fft.
	cufftExecC2C( pFFTPlan, *pdevGrid, *pdevGrid, (pFFTDirection == INVERSE ? CUFFT_INVERSE : CUFFT_FORWARD) );

	// destroy the plan if we need to.
	if (destroyPlan == true)
		finaliseFFT( pFFTPlan );

	// reserve some memory to hold a temporary image, which allows us to do an FFT shift.
	reserveGPUMemory( (void **) &devTmpImage, pSize * pSize * sizeof( cufftComplex ), "reserving device memory for the FFT temporary image" );

	// move image to temporary memory.
	cudaMemcpy( (void *) devTmpImage, (void *) *pdevGrid, pSize * pSize * sizeof( cufftComplex ), cudaMemcpyDeviceToDevice );

	// if we are doing a 'to real' FFT then destroy the original data area and redimension.
	if (pFFTType == C2F || pFFTType == F2F)
	{
		cudaFree( (void *) *pdevGrid );
		reserveGPUMemory( (void **) pdevGrid, sizeof( float ) * pSize * pSize, "reserving device memory for smaller FFT image" );	
	}

	// do inverse FFT shift.
	setThreadBlockSize2D( pSize, pSize );
	if (pFFTType == C2F || pFFTType == F2F)
		devFFTShift<<< _gridSize2D, _blockSize2D >>>(	/* pDestination = */ (float *) *pdevGrid,
								/* pSource = */ devTmpImage,
								/* pFFTDirection = */ INVERSE,
								/* pSize = */ pSize );
	else
		devFFTShift<<< _gridSize2D, _blockSize2D >>>(	/* pDestination = */ *pdevGrid,
								/* pSource = */ devTmpImage,
								/* pFFTDirection = */ INVERSE,
								/* pSize = */ pSize );

	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		printf( "error doing FFT shift following FFT (%s)\n", cudaGetErrorString( err ) );
		ok = false;
	}

	// free memory.
	if (devTmpImage != NULL)
		cudaFree( (void *) devTmpImage );

	// reverse the y-direction because images produces here from inverse FFTs will be naturally upside-down.
	if (pFFTDirection == INVERSE)
	{
		setThreadBlockSize2D( pSize, pSize / 2 );
		if (pFFTType == C2F || pFFTType == F2F)
			devReverseYDirection<<< _gridSize2D, _blockSize2D >>>(	/* pGrid = */ (float *) *pdevGrid,
										/* pSize = */ pSize );
		else
			devReverseYDirection<<< _gridSize2D, _blockSize2D >>>(	/* pGrid = */ *pdevGrid,
										/* pSize = */ pSize );
		err = cudaGetLastError();
		if (err != cudaSuccess)
		{
			printf( "error swapping image y-coordinates (%s)\n", cudaGetErrorString( err ) );
			ok = false;
		}
	}
		
	// return success/failure.
	return ok;
	
} // performFFT

//
//	calculateAPlanes()
//
//	CJS: 03/04/2020
//
//	Calculate which channels/spws are in which A plane.
//

void calculateAPlanes( int *** pWhichAPlane, double * phstAPlaneWavelength, double ** phstWavelength, int pNumSpws, int * phstNumChannels, bool ** phstSpwChannelFlag )
{

	printf( "Preparing for A projection.....\n" );

	printf( "        setting %i a-plane(s) for %i SPW(s):\n", _hstAPlanes, pNumSpws );

	// make an array of all the wavelengths we're using.
	int numWavelengths = 0;
	for ( int spw = 0; spw < pNumSpws; spw++ )
		numWavelengths += phstNumChannels[ spw ];
	double * hstWavelength = (double *) malloc( numWavelengths * sizeof( double ) );
	numWavelengths = 0;
	for ( int spw = 0; spw < pNumSpws; spw++ )
		for ( int channel = 0; channel < phstNumChannels[ spw ]; channel++ )
			if (phstSpwChannelFlag[ spw ][ channel ] == false)
			{

				// check for duplicates.
				bool wavelengthFound = false;
				for ( int checkWavelength = 0; checkWavelength < numWavelengths; checkWavelength++ )
					wavelengthFound = (wavelengthFound || phstWavelength[ spw ][ channel ] == hstWavelength[ checkWavelength ]);

				// add this new wavelength.
				if (wavelengthFound == false)
				{
					hstWavelength[ numWavelengths ] = phstWavelength[ spw ][ channel ];
					numWavelengths++;
				}

			}

	// sort the array of wavelengths.
	for ( int wavelength1 = 0; wavelength1 < numWavelengths - 1; wavelength1++ )
		for ( int wavelength2 = wavelength1 + 1; wavelength2 < numWavelengths; wavelength2++ )
			if ( hstWavelength[ wavelength2 ] < hstWavelength[ wavelength1 ] )
			{
				double tmp = hstWavelength[ wavelength1 ];
				hstWavelength[ wavelength1 ] = hstWavelength[ wavelength2 ];
				hstWavelength[ wavelength2 ] = tmp;
			}

	// calculate the wavelengths.
	for ( int i = 0; i < _hstAPlanes; i++ )
		phstAPlaneWavelength[ i ] = hstWavelength[ ((2 * i) + 1) * numWavelengths / (2 * _hstAPlanes) ];

	// free data.
	if (hstWavelength != NULL)
		free( (void *) hstWavelength );

	printf( "                a-planes based upon wavelengths [" );

	// display the wavelengths.
	for ( int i = 0; i < _hstAPlanes; i++ )
	{
		if (i > 0)
			printf( ", " );
		printf( "%5.4f", phstAPlaneWavelength[ i ] * 1000.0 );
	}
	printf( "] mm\n\n" );

	// update each spw and channel with the appropriate A-plane.
	(*pWhichAPlane) = (int **) malloc( pNumSpws * sizeof( int * ) );
	int ** whichAPlane = (*pWhichAPlane);
	for ( int spw = 0; spw < pNumSpws; spw++ )
	{

		// create array for this spw, and find the appropriate A-plane.
		whichAPlane[ spw ] = (int *) malloc( phstNumChannels[ spw ] * sizeof( int ) );
		for ( int channel = 0; channel < phstNumChannels[ spw ]; channel++ )
		{

			// find the closest A-plane.
			double bestError = 0.0;
			for ( int aPlane = 0; aPlane < _hstAPlanes; aPlane++ )
				if (abs( phstWavelength[ spw ][ channel ] - phstAPlaneWavelength[ aPlane ] ) < bestError || aPlane == 0)
				{
					whichAPlane[ spw ][ channel ] = aPlane;
					bestError = abs( phstWavelength[ spw ][ channel ] - phstAPlaneWavelength[ aPlane ] );
				}

		}
	}

} // calculateAPlanes

//
//	calculateWPlanes()
//
//	CJS: 07/08/2015
//
//	Calculate some values such as support, kernel size and w-cell size. We need to work out the maximum baseline length first. The support is given
//	by the square root of the number of uv cells that fit into the maximum baseline, multiplied by 1.5. The w-cell size is given by the maximum
//	baseline length divided by the number of w-planes, multiplied by two.
//

void calculateWPlanes( int pMosaicIndex, int pNumSamples, VectorD * phstSample )
{

	// calculate the maximum absolute w value [in metres].
	double maxW = 0.0;
	for ( int i = 0; i < pNumSamples; i++ )
		if (abs( phstSample[ i ].w ) > maxW)
			maxW = abs( phstSample[ i ].w );

	// calculate the maximum and minimum w [in lambda].
	maxW = maxW / _hstMinWavelength;
	double minW = -maxW;
			
	// create some memory for the mean and maximum W.
	_hstWPlaneMean[ pMosaicIndex ] = (double *) malloc( _hstWPlanes * sizeof( double ) );
	_hstWPlaneMax[ pMosaicIndex ] = (double *) malloc( _hstWPlanes * sizeof( double ) );

	if (_hstWPlanes > 1)
	{

		double B = maxW / pow( (double) _hstWPlanes / 2.0, 1.5 );
		for ( int i = (_hstWPlanes / 2) - 1; i < _hstWPlanes; i++ )
			_hstWPlaneMax[ pMosaicIndex ][ i ] = B * pow( (double) (i - ((_hstWPlanes / 2) - 1)), 1.5 );
		for ( int i = 0; i < (_hstWPlanes / 2) - 1; i++ )
			_hstWPlaneMax[ pMosaicIndex ][ i ] = -_hstWPlaneMax[ pMosaicIndex ][ _hstWPlanes - 2 - i ];

		printf( "upper limits of w-planes: [" );
		for ( int i = 0; i < _hstWPlanes; i++ )
		{

			// set the mean and maximum w values for this plane.
			if (i == 0)
				_hstWPlaneMean[ pMosaicIndex ][ i ] = ((_hstWPlaneMax[ pMosaicIndex ][ i ] - minW) / 2.0) + minW;
			else
				_hstWPlaneMean[ pMosaicIndex ][ i ] = ((_hstWPlaneMax[ pMosaicIndex ][ i ] - _hstWPlaneMax[ pMosaicIndex ][ i - 1 ]) / 2.0) +
											_hstWPlaneMax[ pMosaicIndex ][ i - 1 ];

			if (i > 0)
				printf( ", " );
			printf( "%5.4f", _hstWPlaneMax[ pMosaicIndex ][ i ] );
			
		}
		printf( "] lambda\n\n" );
			
		printf( "mean values of w-planes: [" );
		for ( int i = 0; i < _hstWPlanes; i++ )
		{
			if (i > 0)
				printf( ", " );
			printf( "%5.4f", _hstWPlaneMean[ pMosaicIndex ][ i ] );
		}
		printf( "] lambda\n\n" );

	}
	else if (_hstWPlanes == 1)
	{
		_hstWPlaneMean[ pMosaicIndex][ 0 ] = 0;
		_hstWPlaneMax[ pMosaicIndex][ 0 ] = maxW;
	}
	
} // calculateWPlanes

//
//	rotateX(), rotateY()
//
//	CJS: 13/04/2021
//
//	Rotate a vector about the x and y axes respectively.
//

VectorD rotateX( VectorD pIn, double pAngle )
{

	VectorD out = { .u = pIn.u, .v = (pIn.v * cos( pAngle )) + (pIn.w * sin( pAngle )), .w = (pIn.w * cos( pAngle )) - (pIn.v * sin( pAngle )) };

	// return something.
	return out;

} // rotateX

VectorD rotateY( VectorD pIn, double pAngle )
{

	VectorD out = { .u = (pIn.u * cos( pAngle )) + (pIn.w * sin( pAngle )), .v = pIn.v, .w = (pIn.w * cos( pAngle )) - (pIn.u * sin( pAngle )) };

	// return something.
	return out;

} // rotateY

//
//	getASKAPBeamPosition()
//
//	CJS: 20/12/2019
//
//	Calculates the position of each ASKAP beam relative to the phase centre.
//

void getASKAPBeamPosition( double * pRA, double * pDEC, double pXOffset, double pYOffset, double pCentreRA, double pCentreDEC )
{

	// initialise vector.
	VectorD vect = { .u = 0.0, .v = 0.0, .w = 1.0 };

	// rotate the vector about the x axis by the y offset.
	vect = rotateX( /* pIn = */ vect, /* pAngle = */ pYOffset );

	// rotate the vector about the y axis by the x offset.
	vect = rotateY( /* pIn = */ vect, /* pAngle = */ pXOffset );

	// now we rotate to the dish pointing position. rotate the vector about the x axis by the declination.
	vect = rotateX( /* pIn = */ vect, /* pAngle = */ pCentreDEC * PI / 180.0 );

	// rotate the vector about the y axis by the right ascension.
	vect = rotateY( /* pIn = */ vect, /* pAngle = */ -pCentreRA * PI / 180.0 );

	// get the right ascension and declination of the beam.
	*pDEC = asin( vect.v ) * 180.0 / PI;
	*pRA = asin( -vect.u / cos( asin( vect.v ) ) ) * 180.0 / PI;

} // getASKAPBeamPosition

//
//	doPhaseCorrectionSamples()
//
//	CJS: 12/08/2015
//
//	Phase correct the samples using the PhaseCorrection class.
//

void doPhaseCorrectionSamples( PhaseCorrection * pPhaseCorrection, int pNumSamples, double * pPhaseCentreIn, double * pPhaseCentreOut, VectorD * pSample, 
				int * pFieldID, double * pPhase )
{
	
	const char J2000[] = "J2000";
	
	// set up the coordinate system.
	strcpy( pPhaseCorrection->inCoords.epoch, J2000 );
	strcpy( pPhaseCorrection->outCoords.epoch, J2000 );
	pPhaseCorrection->uvProjection = false;

	// current field - initialise to -1, the data should be ordered by Field ID, so each time the field id changes we need to regenerate the phase rotation matrices.
	int currentField = -1;
	
	// loop through all the samples.
	for ( int i = 0; i < pNumSamples; i++ )
	{

		// get the field for this sample. has it changed from the last time ?
		int thisField = pFieldID[ i ];
		if (thisField != currentField)
		{

			// set up the input coordinate system.
			pPhaseCorrection->inCoords.longitude = pPhaseCentreIn[ thisField * 2 ];
			pPhaseCorrection->inCoords.latitude = pPhaseCentreIn[ (thisField * 2) + 1 ];

			// set up the output coordinate system.
			pPhaseCorrection->outCoords.longitude = pPhaseCentreOut[ thisField * 2 ];
			pPhaseCorrection->outCoords.latitude = pPhaseCentreOut[ (thisField * 2) + 1 ];

			// initialise phase rotation matrices.
			pPhaseCorrection->init();

			currentField = thisField;

		}
		
		// get the rotated uvw coordinate.
		VectorXYZ uvwIn = { .x = pSample[ i ].u, .y = pSample[ i ].v, .z = pSample[ i ].w };
		pPhaseCorrection->uvwIn = uvwIn;
		pPhaseCorrection->rotate();
		if (pPhase != NULL)
			pPhase[ i ] = pPhaseCorrection->phase;
		VectorD uvwOut = { .u = pPhaseCorrection->uvwOut.x, .v = pPhaseCorrection->uvwOut.y, .w = pPhaseCorrection->uvwOut.z };
		pSample[ i ] = uvwOut;
		
		
	}
	
} // doPhaseCorrectionSamples

//
//	getMaxValue()
//
//	CJS: 23/11/2015
//
//	Gets the maximum complex value from a 2D image array, and writes it (along with the x and y coordinates)
//	to a 3 double area in global memory.
//

bool getMaxValue( cufftComplex * pdevImage, double * pdevMaxValue, int pWidth, int pHeight, bool pIncludeComplexComponent, bool * pdevMask )
{
	
	const int PIXELS_PER_THREAD = 32;
	
	bool ok = true;
	cudaError_t err;
		
	// find a suitable thread/block size for finding the maximum pixel value. each thread block will find the max
	// over N pixels (held in the above constant), and write the result to a shared memory array. a final loop will then
	// get the max from the array.
	int threads = (pWidth * pHeight / PIXELS_PER_THREAD) + 1, blocks = 1;
	setThreadBlockSize1D( &threads, &blocks);
		
	// declare global memory for writing the result of each block.
	double * devTmpResults = NULL;
	ok = reserveGPUMemory( (void **) &devTmpResults, MAX_PIXEL_DATA_AREA_SIZE * blocks * sizeof( double ), "declaring device memory for psf max block value" );
		
	if (ok == true)
	{
		
		// get maximum pixel value.
		devGetMaxValueParallel<<< blocks, threads, MAX_PIXEL_DATA_AREA_SIZE * threads * sizeof( double ) >>>
					(	/* pArray = */ pdevImage,
						/* pWidth = */ pWidth,
						/* pHeight = */ pHeight,
						/* pCellsPerThread = */ PIXELS_PER_THREAD,
						/* pBlockMax = */ devTmpResults,
						/* pIncludeComplexComponent = */ pIncludeComplexComponent,
						/* pMask = */ pdevMask );
		err = cudaGetLastError();
		if (err != cudaSuccess)
		{
			printf( "error getting psf max pixel value (%s)\n", cudaGetErrorString( err ) );
			ok = false;
		}
		
	}
	
	if (ok == true)
	{
			
		// get maximum pixel value from the block list.
		devGetMaxValue<<< 1, 1 >>>(	/* pArray = */ devTmpResults,
						/* pMaxValue = */ pdevMaxValue,
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
	
} // getMaxValue

bool getMaxValue( double * pdevImage, double * pdevMaxValue, int pWidth, int pHeight, bool * pdevMask, cudaStream_t pStream )
{
	
	const int PIXELS_PER_THREAD = 32;
	
	bool ok = true;
	cudaError_t err;
		
	// find a suitable thread/block size for finding the maximum pixel value. each thread block will find the max
	// over N pixels (held in the above constant), and write the result to a shared memory array. a final loop will then
	// get the max from the array.
	int threads = (pWidth * pHeight / PIXELS_PER_THREAD) + 1, blocks = 1;
	setThreadBlockSize1D( &threads, &blocks);
		
	// declare global memory for writing the result of each block.
	double * devTmpResults = NULL;
	ok = reserveGPUMemory( (void **) &devTmpResults, MAX_PIXEL_DATA_AREA_SIZE * blocks * sizeof( double ), "declaring device memory for psf max block value" );
		
	if (ok == true)
	{
		
		// get maximum pixel value.
		devGetMaxValueParallel<<< blocks, threads, MAX_PIXEL_DATA_AREA_SIZE * threads * sizeof( double ), pStream >>>
					(	/* pArray = */ pdevImage,
						/* pWidth = */ pWidth,
						/* pHeight = */ pHeight,
						/* pCellsPerThread = */ PIXELS_PER_THREAD,
						/* pBlockMax = */ devTmpResults,
						/* pMask = */ pdevMask );
		err = cudaGetLastError();
		if (err != cudaSuccess)
		{
			printf( "error getting psf max pixel value (%s)\n", cudaGetErrorString( err ) );
			ok = false;
		}
		
	}
	
	if (ok == true)
	{
			
		// get maximum pixel value from the block list.
		devGetMaxValue<<< 1, 1, 0, pStream >>>(	/* pArray = */ devTmpResults,
							/* pMaxValue = */ pdevMaxValue,
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
	
} // getMaxValue

bool getMaxValue( double * pdevImage, double * pdevMaxValue, int pWidth, int pHeight, bool * pdevMask )
{
	
	const int PIXELS_PER_THREAD = 32;
	
	bool ok = true;
	cudaError_t err;

	err = cudaGetLastError();
	if (err != cudaSuccess)
		printf( "unknown CUDA error in getMaxValue() [i, double] (%s)\n", cudaGetErrorString( err ) );
		
	// find a suitable thread/block size for finding the maximum pixel value. each thread block will find the max
	// over N pixels (held in the above constant), and write the result to a shared memory array. a final loop will then
	// get the max from the array.
	int threads = (pWidth * pHeight / PIXELS_PER_THREAD) + 1, blocks = 1;
	setThreadBlockSize1D( &threads, &blocks);
		
	// declare global memory for writing the result of each block.
	double * devTmpResults = NULL;
	reserveGPUMemory( (void **) &devTmpResults, MAX_PIXEL_DATA_AREA_SIZE * blocks * sizeof( double ), "declaring device memory for psf max block value" );
		
	// get maximum pixel value.
	devGetMaxValueParallel<<< blocks, threads, MAX_PIXEL_DATA_AREA_SIZE * threads * sizeof( double ) >>>
				(	/* pArray = */ pdevImage,
					/* pWidth = */ pWidth,
					/* pHeight = */ pHeight,
					/* pCellsPerThread = */ PIXELS_PER_THREAD,
					/* pBlockMax = */ devTmpResults,
					/* pMask = */ pdevMask );
	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		printf( "error getting psf max pixel value (%s)\n", cudaGetErrorString( err ) );
		ok = false;
	}
			
	// get maximum pixel value from the block list.
	devGetMaxValue<<< 1, 1 >>>(	/* pArray = */ devTmpResults,
					/* pMaxValue = */ pdevMaxValue,
					/* pElements = */ blocks );
	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		printf( "error getting final max value (%s)\n", cudaGetErrorString( err ) );
		ok = false;
	}
		
	// free memory.
	if (devTmpResults != NULL)
		cudaFree( (void *) devTmpResults );
	
	// return success flag.
	return ok;
	
} // getMaxValue

bool getMaxValue( float * pdevImage, double * pdevMaxValue, int pWidth, int pHeight, bool * pdevMask )
{
	
	const int PIXELS_PER_THREAD = 32;
	
	bool ok = true;
	cudaError_t err;
		
	// find a suitable thread/block size for finding the maximum pixel value. each thread block will find the max
	// over N pixels (held in the above constant), and write the result to a shared memory array. a final loop will then
	// get the max from the array.
	int threads = (pWidth * pHeight / PIXELS_PER_THREAD) + 1, blocks = 1;
	setThreadBlockSize1D( &threads, &blocks);
		
	// declare global memory for writing the result of each block.
	double * devTmpResults = NULL;
	ok = reserveGPUMemory( (void **) &devTmpResults, MAX_PIXEL_DATA_AREA_SIZE * blocks * sizeof( double ), "declaring device memory for psf max block value" );
		
	if (ok == true)
	{
		
		// get maximum pixel value.
		devGetMaxValueParallel<<< blocks, threads, MAX_PIXEL_DATA_AREA_SIZE * threads * sizeof( double ) >>>
					(	/* pArray = */ pdevImage,
						/* pWidth = */ pWidth,
						/* pHeight = */ pHeight,
						/* pCellsPerThread = */ PIXELS_PER_THREAD,
						/* pBlockMax = */ devTmpResults,
						/* pMask = */ pdevMask );
		err = cudaGetLastError();
		if (err != cudaSuccess)
		{
			printf( "error getting psf max pixel value (%s)\n", cudaGetErrorString( err ) );
			ok = false;
		}
		
	}
	
	if (ok == true)
	{
			
		// get maximum pixel value from the block list.
		devGetMaxValue<<< 1, 1 >>>(	/* pArray = */ devTmpResults,
						/* pMaxValue = */ pdevMaxValue,
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
	
} // getMaxValue

bool getMaxValue( float * pdevImage, double * pdevMaxValue, int pWidth, int pHeight, bool * pdevMask, cudaStream_t pStream )
{
	
	const int PIXELS_PER_THREAD = 32;
	
	bool ok = true;
	cudaError_t err;
		
	// find a suitable thread/block size for finding the maximum pixel value. each thread block will find the max
	// over N pixels (held in the above constant), and write the result to a shared memory array. a final loop will then
	// get the max from the array.
	int threads = (pWidth * pHeight / PIXELS_PER_THREAD) + 1, blocks = 1;
	setThreadBlockSize1D( &threads, &blocks);
		
	// declare global memory for writing the result of each block.
	double * devTmpResults = NULL;
	ok = reserveGPUMemory( (void **) &devTmpResults, MAX_PIXEL_DATA_AREA_SIZE * blocks * sizeof( double ), "declaring device memory for psf max block value" );
		
	if (ok == true)
	{
		
		// get maximum pixel value.
		devGetMaxValueParallel<<< blocks, threads, MAX_PIXEL_DATA_AREA_SIZE * threads * sizeof( double ), pStream >>>
					(	/* pArray = */ pdevImage,
						/* pWidth = */ pWidth,
						/* pHeight = */ pHeight,
						/* pCellsPerThread = */ PIXELS_PER_THREAD,
						/* pBlockMax = */ devTmpResults,
						/* pMask = */ pdevMask );
		err = cudaGetLastError();
		if (err != cudaSuccess)
		{
			printf( "error getting psf max pixel value (%s)\n", cudaGetErrorString( err ) );
			ok = false;
		}
		
	}
	
	if (ok == true)
	{
			
		// get maximum pixel value from the block list.
		devGetMaxValue<<< 1, 1, 0, pStream >>>(	/* pArray = */ devTmpResults,
							/* pMaxValue = */ pdevMaxValue,
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
	
} // getMaxValue

//
//	findCutoffPixel()
//
//	CJS: 03/04/2019
//
//	Find the furthest pixel from the centre of the kernel which is >= pCutoffFraction % of the peak kernel value. this value is used as the support size
//	when the kernel is cropped.
//

int findCutoffPixel( cufftComplex * pdevKernel, double * pdevMaxValue, int pSize, double pCutoffFraction, findpixel pFindType )
{
	
	const int PIXELS_PER_THREAD = 32;
	
	bool ok = true;
	int support = 0;
	cudaError_t err;

	// create memory to hold the support on the device.
	int * devSupport = NULL;
	reserveGPUMemory( (void **) &devSupport, sizeof( int ), "reserving device memory for the cutoff support" );
	zeroGPUMemory( (void *) devSupport, sizeof( int ), "zeroing device memory for the cutoff support" );
		
	// find a suitable thread/block size for finding the cutoff pixel. each thread block will find the largest support needed for
	// N pixels (held in the above constant), and write the result to a shared memory array. a final loop will then
	// get the max from the array.
	int threads = (pSize * pSize / PIXELS_PER_THREAD) + 1, blocks = 1;
	setThreadBlockSize1D( &threads, &blocks);
		
	// declare global memory for writing the result of each block.
	int * devTmpResults = NULL;
	ok = reserveGPUMemory( (void **) &devTmpResults, blocks * sizeof( int ), "declaring device memory for max support" );
	if (ok == true)
	{
		
		// get maximum support in parallel.
		devFindCutoffPixelParallel<<< blocks, threads, threads * sizeof( int ) >>>(	/* pKernel = */ pdevKernel,
												/* pSize = */ pSize,
												/* pMaxValue = */ pdevMaxValue,
												/* pCellsPerThread = */ PIXELS_PER_THREAD,
												/* pTmpResults = */ devTmpResults,
												/* pCutoffFraction = */ pCutoffFraction,
												/* pFindType = */ pFindType );
		err = cudaGetLastError();
		if (err != cudaSuccess)
		{
			printf( "error finding max support (parallel) (%s)\n", cudaGetErrorString( err ) );
			ok = false;
		}
		
	}
	
	if (ok == true)
	{
			
		// get support from the block list.
		devFindCutoffPixel<<< 1, 1 >>>(	/* pTmpResults = */ devTmpResults,
						/* pSupport = */ devSupport,
						/* pElements = */ blocks,
						/* pFindType = */ pFindType );
		err = cudaGetLastError();
		if (err != cudaSuccess)
		{
			printf( "error getting final support value (%s)\n", cudaGetErrorString( err ) );
			ok = false;
		}
			
	}

	// get the support from the storage area.
	moveDeviceToHost( (void *) &support, (void *) devSupport, sizeof( int ), "copying support from the host" );
		
	// free memory.
	if (devTmpResults != NULL)
		cudaFree( (void *) devTmpResults );
	if (devSupport != NULL)
		cudaFree( (void *) devSupport );
	
	// return the support
	return support;

} // findCutoffPixel

int findCutoffPixel( float * pdevKernel, double * pdevMaxValue, int pSize, double pCutoffFraction, findpixel pFindType )
{
	
	const int PIXELS_PER_THREAD = 32;
	
	bool ok = true;
	int support = 0;
	cudaError_t err;

	// create memory to hold the support on the device.
	int * devSupport = NULL;
	reserveGPUMemory( (void **) &devSupport, sizeof( int ), "reserving device memory for the cutoff support" );
	zeroGPUMemory( (void *) devSupport, sizeof( int ), "zeroing device memory for the cutoff support" );
		
	// find a suitable thread/block size for finding the cutoff pixel. each thread block will find the largest support needed for
	// N pixels (held in the above constant), and write the result to a shared memory array. a final loop will then
	// get the max from the array.
	int threads = (pSize * pSize / PIXELS_PER_THREAD) + 1, blocks = 1;
	setThreadBlockSize1D( &threads, &blocks);
		
	// declare global memory for writing the result of each block.
	int * devTmpResults = NULL;
	ok = reserveGPUMemory( (void **) &devTmpResults, blocks * sizeof( int ), "declaring device memory for max support" );
	if (ok == true)
	{
		
		// get maximum support in parallel.
		devFindCutoffPixelParallel<<< blocks, threads, threads * sizeof( int ) >>>(	/* pKernel = */ pdevKernel,
												/* pSize = */ pSize,
												/* pMaxValue = */ pdevMaxValue,
												/* pCellsPerThread = */ PIXELS_PER_THREAD,
												/* pTmpResults = */ devTmpResults,
												/* pCutoffFraction = */ pCutoffFraction,
												/* pFindType = */ pFindType );
		err = cudaGetLastError();
		if (err != cudaSuccess)
		{
			printf( "error finding max support (parallel) (%s)\n", cudaGetErrorString( err ) );
			ok = false;
		}
		
	}
	
	if (ok == true)
	{
			
		// get support from the block list.
		devFindCutoffPixel<<< 1, 1 >>>(	/* pTmpResults = */ devTmpResults,
						/* pSupport = */ devSupport,
						/* pElements = */ blocks,
						/* pFindType = */ pFindType );
		err = cudaGetLastError();
		if (err != cudaSuccess)
		{
			printf( "error getting final support value (%s)\n", cudaGetErrorString( err ) );
			ok = false;
		}
			
	}

	// get the support from the storage area.
	moveDeviceToHost( (void *) &support, (void *) devSupport, sizeof( int ), "copying support from the host" );
		
	// free memory.
	if (devTmpResults != NULL)
		cudaFree( (void *) devTmpResults );
	if (devSupport != NULL)
		cudaFree( (void *) devSupport );
	
	// return the support
	return support;

} // findCutoffPixel

//
//	generateKernel()
//
//	CJS: 10/08/2015
//
//	Generates the convolution function. A separate kernel is generated for each w-plane, and for each oversampled intermediate
//	grid position. Here we generate all the kernels for a specific W plane and A plane.
//
//	parameters:
//
//	pW - the W plane to generate kernels for (will be 0 if we're not using W projection).
//	pA - the A plane to generate kernels for (will be 0 if we're not using A projection).
//	pWProjection - true if we're using W projection (DON'T use the global parameter).
//	pAProjection - true if we're using A projection (DON'T use the global parameter).
//	phstKernelSize - the kernel size for this kernel set. we calculate this value here and return it.
//	pOversample - the oversample parameter (DON'T use the global parameter).
//	pPrimaryBeamMosaicing - the primary beam for mosaicing (if needed)
//	pPrimaryBeamAProjection - the primary beam for A projection (if needed)
//	pFieldID - the ID of the field we are generating a kernel for. If we aren't doing beam corrections then this value is -1.
//	pMosaicIndex - the index of the mosaic currently being processed.
//	pGridDegrid - either GRID or DEGRID
//	pdevKernelCachePtr - a location on the host where we store the kernel we generate.
//	pKernelOverflow - we set this parameter to true if the kernel needs to be truncated due to size.
//

bool generateKernel( int pW, int pA, bool pWProjection, bool pAProjection, int * phstKernelSize, int pOversample,
			float * phstPrimaryBeamMosaicing, float * phstPrimaryBeamAProjection, int pFieldID,
			int pMosaicIndex, griddegrid pGridDegrid, cufftComplex ** pdevKernelPtr, bool * pKernelOverflow )
{

	// we only use a maximum fraction of the available memory to create a workspace.
	const double MAX_MEM_TO_USE = 0.8;
	const int MAX_WORKSPACE_SIZE = 2048;
	
	bool ok = true;
	cudaError_t err;

	// for beam mosaicing we need to do some primary beam correction using the average (mosaic) primary beam. we only have to do this for degridding if we're not
	// using A-projection.
	bool doBeamCorrection = (phstPrimaryBeamMosaicing != NULL && ((pGridDegrid == GRID && pAProjection == true) || (pGridDegrid == DEGRID && pAProjection == false)));

	int numberOfWorkspacesRequired = 1;
	if (doBeamCorrection == true)
		numberOfWorkspacesRequired++;
	if (pWProjection == true)
		numberOfWorkspacesRequired++;
	if (pAProjection == true)
		numberOfWorkspacesRequired++;

	// find how much GPU memory is available.
	// NOTE: There has been a bug on the GeForce RTX 2070 on Lofar7 that causes this instruction to fail. This bug seems to occur when the kernel size is too large. If
	// it does fail, we assume that there is 1 GB of free memory.
	size_t freeMem = 1073741824, totalMem = 0;
	err = cudaMemGetInfo( &freeMem, &totalMem );
	if (err != cudaSuccess)
	{
		printf( "error reading free memory from the device (%s)\n", cudaGetErrorString( err ) );
		freeMem = 1073741824;
	}

	// calculate a pixel size from the free memory, and reduce the free memory to the next lowest power of 2.
	int maximumWorkspaceSize = (int) sqrt( double( freeMem ) * MAX_MEM_TO_USE / double( numberOfWorkspacesRequired * sizeof( cufftComplex ) ) );
	int oversampledWorkspaceSize = 1;
	while (oversampledWorkspaceSize <= maximumWorkspaceSize)
		oversampledWorkspaceSize = oversampledWorkspaceSize * 2;
	oversampledWorkspaceSize = oversampledWorkspaceSize / 2;

	// set the workspace size and the maximum kernel size.
	int workspaceSize = oversampledWorkspaceSize / pOversample;
	if (workspaceSize > MAX_WORKSPACE_SIZE)
		workspaceSize = MAX_WORKSPACE_SIZE;
	oversampledWorkspaceSize = workspaceSize * pOversample;

	// if we are using beam mosaicing then we need to correct the image for the primary beam, and weight the image for mosaicing.
	cufftComplex * devBeamCorrection = NULL;
	if (doBeamCorrection == true)
	{

		// create space for the primary beam on the device, and copy it across.
		float * devPrimaryBeam = NULL;
		reserveGPUMemory( (void **) &devPrimaryBeam, _hstBeamSize * _hstBeamSize * sizeof( float ),
					"declaring device memory for primary beam" );
		moveHostToDevice( (void *) devPrimaryBeam, (void *) phstPrimaryBeamMosaicing, _hstBeamSize * _hstBeamSize * sizeof( float ),
					"copying primary beam to the device" );

		// create the kernel and clear it.
		reserveGPUMemory( (void **) &devBeamCorrection, workspaceSize * workspaceSize * sizeof( cufftComplex ),
					"declaring device memory for the beam-correction kernel" );
		zeroGPUMemory( (void *) devBeamCorrection, workspaceSize * workspaceSize * sizeof( cufftComplex ), "clearing beam-correction kernel on the device" );

		// copy the kernel into the workspace.
		setThreadBlockSize2D( workspaceSize, workspaceSize );
		devCopyImage<<< _gridSize2D, _blockSize2D >>>(	/* pNewImage = */ devBeamCorrection,
								/* pOldImage = */ devPrimaryBeam,
								/* pNewSize = */ workspaceSize,
								/* pOldSize = */ _hstBeamSize,
								/* pScale = */ (double) _hstBeamSize / (double) workspaceSize,
								/* pThreadOffset = */ 0 );
		
		// free the primary beam.
		if (devPrimaryBeam != NULL)
			cudaFree( (void *) devPrimaryBeam );

	}

	// are we using W-projection ?
	cufftComplex * devWKernel = NULL;
	if (pWProjection == true)
	{

		// create w-kernel.
		reserveGPUMemory( (void **) &devWKernel, workspaceSize * workspaceSize * sizeof( cufftComplex ), "declaring device memory for w-kernel" );

		// convert the cell size from arcseconds to radians, and take the sine to get the directional cosine.
		double cellSizeDirectionalCosine = sin( (_hstCellSize / 3600.0) * (PI / 180.0) );

		// generate the W-kernel on the GPU.
		setThreadBlockSize2D( workspaceSize, workspaceSize );
		devGenerateWKernel<<< _gridSize2D, _blockSize2D >>>(	/* pWKernel = */ devWKernel,
									/* pW = */ _hstWPlaneMean[ pMosaicIndex ][ pW ],
									/* pWorkspaceSize = */ workspaceSize,
									/* pCellSizeDirectionalCosine = */ cellSizeDirectionalCosine,
									/* pGridDegrid = */ pGridDegrid,
									/* pSize = */ _hstUvPixels );

	}

	// are we using A-projection ?
	cufftComplex * devAKernel = NULL;
	if (pAProjection == true)
	{

		// reserve some memory for the A kernel and primary beam.
		float * devPrimaryBeam = NULL;
		reserveGPUMemory( (void **) &devAKernel, workspaceSize * workspaceSize * sizeof( cufftComplex ), "declaring device memory for a-kernel" );
		reserveGPUMemory( (void **) &devPrimaryBeam, _hstBeamSize * _hstBeamSize * sizeof( float ), "declaring device memory for the primary beam" );

		// upload the primary beam to the device.
		moveHostToDevice( (void *) devPrimaryBeam, (void *) phstPrimaryBeamAProjection, _hstBeamSize * _hstBeamSize * sizeof( float ),
						"uploading primary beam to the device" );

		// generate the A kernel on the device. The A kernel should be the inverse of the primary beam (1 / pbeam) for gridding, and the primary beam for degridding.
		setThreadBlockSize2D( workspaceSize, workspaceSize );
		devGenerateAKernel<<< _gridSize2D, _blockSize2D >>>(	/* pAKernel = */ devAKernel,
									/* pPrimaryBeam = */ devPrimaryBeam,
									/* pPrimaryBeamSupport = */ (_hstBeamSize / 2),
									/* pWorkspaceSize = */ workspaceSize,
									/* pGridDegrid = */ pGridDegrid );
		err = cudaGetLastError();
		if (err != cudaSuccess)
			printf( "error generating A kernel (%s)\n", cudaGetErrorString( err ) );

		// free the primary beam.
		if (devPrimaryBeam != NULL)
			cudaFree( (void *) devPrimaryBeam );

	}
	
	// reserve some memory for the AA kernel and clear it.
	cufftComplex  * devAAKernel = NULL;
	reserveGPUMemory( (void **) &devAAKernel, workspaceSize * workspaceSize * sizeof( cufftComplex ), "declaring device memory for the aa-kernel" );
	zeroGPUMemory( (void *) devAAKernel, workspaceSize * workspaceSize * sizeof( cufftComplex ), "zeroing aa-kernel on the device" );

	err = cudaGetLastError();
	if (err != cudaSuccess)
		printf( "Unknown CUDA error in generateKernel() (%s)\n", cudaGetErrorString( err ) );

	// generate the AA-kernel on the GPU.
	int aaKernelSize = (int) ceil( sqrt( 2 * workspaceSize * workspaceSize ) );
	if (aaKernelSize % 2 == 0)
		aaKernelSize++;
	setThreadBlockSize2D( workspaceSize, workspaceSize );
	devGenerateAAKernel<<< _gridSize2D, _blockSize2D >>>(	/* pAAKernel = */ devAAKernel,
								/* pKernelSize = */ aaKernelSize,
								/* pWorkspaceSize = */ workspaceSize );
	err = cudaGetLastError();
	if (err != cudaSuccess)
		printf( "error generating AA kernel (%s)\n", cudaGetErrorString( err ) );

	// create a kernel workspace to store the image-plane kernel.
	cufftComplex * devImagePlaneKernel = NULL;
	reserveGPUMemory( (void **) &devImagePlaneKernel, workspaceSize * workspaceSize * sizeof( cufftComplex ), "declaring device memory for image-plane kernel" );

	// we now work with the whole workspace.
	setThreadBlockSize2D( workspaceSize, workspaceSize );

	// we start off with the anti-aliasing kernel.
	cudaMemcpy( (void *) devImagePlaneKernel, (void *) devAAKernel, workspaceSize * workspaceSize * sizeof( cufftComplex ), cudaMemcpyDeviceToDevice );

	// are we using W-projection ? Convolve with the kernel.
	if (pWProjection == true)
		devMultiplyImages<<< _gridSize2D, _blockSize2D >>>( devImagePlaneKernel, devWKernel, workspaceSize );

	// are we using A-projection ? Convolve with the kernel.
	if (pAProjection == true)
		devMultiplyImages<<< _gridSize2D, _blockSize2D >>>( devImagePlaneKernel, devAKernel, workspaceSize );

	// are we using beam correction ? Convolve with the kernel.
	if (doBeamCorrection == true)
		devMultiplyImages<<< _gridSize2D, _blockSize2D >>>( devImagePlaneKernel, devBeamCorrection, workspaceSize );

	// reserve some memory for the combined kernel, and clear it.
	cufftComplex * devCombinedKernel = NULL;
	reserveGPUMemory( (void **) &devCombinedKernel, oversampledWorkspaceSize * oversampledWorkspaceSize * sizeof( cufftComplex ),
				"declaring device memory for the combined kernel" );
	zeroGPUMemory( (void *) devCombinedKernel, oversampledWorkspaceSize * oversampledWorkspaceSize * sizeof( cufftComplex ),
				"zeroing device memory for the combined kernel" );

	// copy the image-domain kernel into the centre of the combined kernel. i.e. we are padding our kernel by the oversampling factor.
	devCopyImage<<< _gridSize2D, _blockSize2D >>>(	/* pNewImage = */ devCombinedKernel,
							/* pOldImage = */ devImagePlaneKernel,
							/* pNewSize = */ oversampledWorkspaceSize,
							/* pOldSize = */ workspaceSize,
							/* pScale = */ 1.0,
							/* pThreadOffset = */ (oversampledWorkspaceSize - workspaceSize) / 2 );

	// free the image-plane kernel.
	if (devImagePlaneKernel != NULL)
		cudaFree( (void *) devImagePlaneKernel );

	// FFT the convolved kernel back into the UV domain.
	performFFT(	/* pdevGrid = */ &devCombinedKernel,
			/* pSize = */ oversampledWorkspaceSize,
			/* pFFTDirection = */ FORWARD,
			/* pFFTPlan = */ -1,
			/* pFFTType = */ C2C );

	// initialise support size.
	int supportSize = 3;

	// create a new memory area to hold the maximum pixel value.
	double * devMaxValue;
	reserveGPUMemory( (void **) &devMaxValue, MAX_PIXEL_DATA_AREA_SIZE * sizeof( double ), "declaring device memory for kernel max pixel value" );

	// get the peak value from the kernel.
	getMaxValue(	/* pdevImage = */ devCombinedKernel,
			/* pdevMaxValue = */ devMaxValue,
			/* pWidth = */ oversampledWorkspaceSize,
			/* pHeight = */ oversampledWorkspaceSize,
			/* pIncludeComplexComponent = */ true,
			/* pdevMask = */ NULL );

	// find the furthest pixel from the centre of the kernel that is >= 1% of the kernel max.
	// For w-projection I think CASA uses something like 0.001, but this requires HUGE kernels.
	supportSize = findCutoffPixel(	/* pdevKernel = */ devCombinedKernel,
					/* pdevMaxValue = */ devMaxValue,
					/* pSize = */ oversampledWorkspaceSize,
					/* pCutoffFraction = */ _hstKernelCutoffFraction,
					/* pFindType = */ FURTHEST );

	// free memory.
	if (devMaxValue != NULL)
		cudaFree( (void *) devMaxValue );

	// divide the support size by the oversampling factor, and round up.
	supportSize = (int) ceil( (double) supportSize / (double) pOversample );

	// ensure the support is at least 3.
	if (supportSize < 3)
		supportSize = 3;

	// calculate the maximum support size, and make sure it's not larger than a pre-fixed size for performance reasons.
	// NOTE: There has been a bug on the GeForce RTX 2070 on Lofar7 that causes this instruction to fail. If it does fail, we assume that there is 1 GB of free memory.
	err = cudaMemGetInfo( &freeMem, &totalMem );
	if (err != cudaSuccess)
	{
		printf( "error reading free memory from the device (%s)\n", cudaGetErrorString( err ) );
		freeMem = 1073741824;
	}
	int maximumSupportSize = (int) (sqrt( freeMem ) / 2.0);
	if (maximumSupportSize > _hstKernelCutoffSupport)
		maximumSupportSize = _hstKernelCutoffSupport;

	// restrict support based upon the workspace size.
	if (supportSize > maximumSupportSize)
	{
		supportSize = maximumSupportSize;
		*pKernelOverflow = true;
	}

	// calculate kernel size.
	*phstKernelSize = (supportSize * 2) + 1;

	// kernel data on the device.
	reserveGPUMemory( (void **) pdevKernelPtr, *phstKernelSize * *phstKernelSize * pOversample * pOversample * sizeof( cufftComplex ),
				"declaring device memory for kernel" );

	// get a shortcut to the kernel.
	cufftComplex * devKernel = *pdevKernelPtr;

	// define the block/thread dimensions.
	setThreadBlockSize2D( *phstKernelSize, *phstKernelSize );
					
	// calculate separate kernels for each (oversampled) intermediate grid position.
	for ( int oversampleI = 0; oversampleI < pOversample; oversampleI++ )
		for ( int oversampleJ = 0; oversampleJ < pOversample; oversampleJ++ )
		{
				
			// get the index of the oversampled kernels. no need to add the index of the w-kernel because
			// we're putting them in separate arrays.
			long int kernelIdx = ((long int) oversampleI * (long int) *phstKernelSize * (long int) *phstKernelSize) +
						((long int) oversampleJ * (long int) *phstKernelSize * (long int) *phstKernelSize * (long int) pOversample);
				
			// copy the kernel from the anti-aliasing kernel into the actual kernel.
			devUpdateKernel<<< _gridSize2D, _blockSize2D >>>(	/* pKernel = */ &devKernel[ kernelIdx ],
										/* pImage = */ devCombinedKernel,
										/* pSupport = */ supportSize,
										/* pOversample = */ pOversample,
										/* pOversampleI = */ oversampleI,
										/* pOversampleJ = */ oversampleJ,
										/* pWorkspaceSize = */ oversampledWorkspaceSize,
										/* pGridDegrid = */ pGridDegrid );
			
		}
			
	// define the block/thread dimensions.
	int threads = *phstKernelSize * *phstKernelSize * pOversample * pOversample, blocks = 1;
	setThreadBlockSize1D( &threads, &blocks );
					
	// normalise the image following the FFT by dividing by the number of pixels.
	devNormalise<<< blocks, threads >>>(	devKernel,
						(double) (workspaceSize * workspaceSize),
						*phstKernelSize * *phstKernelSize * pOversample * pOversample );

	// for beam mosaics, normalise the image following the FFT by dividing by the number of gridded visibilities. We actually do most of this normalisation in the image
	// domain:
	// The normalisation factor, N_i, will be different for each field, and we find the lowest N_i, which we call N_min and hold in _griddedVisibilitiesForBeamMosaic,
	// and take it out as a factor. We correct for N_min in the image domain, and here we will correct for N_i / N_min.
	if (phstPrimaryBeamMosaicing != NULL && pGridDegrid == GRID && _hstBeamMosaic == true)
		devNormalise<<< blocks, threads>>>(	devKernel,
							(double) _hstGriddedVisibilitiesPerField[ pFieldID ] / (double) _griddedVisibilitiesForBeamMosaic,
							*phstKernelSize * *phstKernelSize * pOversample * pOversample );
	if (phstPrimaryBeamMosaicing != NULL && pGridDegrid == GRID && _hstUVMosaic == true)
		devNormalise<<< blocks, threads>>>(	devKernel,
							(double) _hstGriddedVisibilities[ pMosaicIndex ] / (double) _griddedVisibilitiesForBeamMosaic,
							*phstKernelSize * *phstKernelSize * pOversample * pOversample );
	
	// cleanup memory.
	if (devWKernel != NULL)
		cudaFree( (void *) devWKernel );
	if (devAKernel != NULL)
		cudaFree( (void *) devAKernel );
	if (devAAKernel != NULL)
		cudaFree( (void *) devAAKernel );
	if (devCombinedKernel != NULL)
		cudaFree( (void *) devCombinedKernel );
	if (devBeamCorrection != NULL)
		cudaFree( (void *) devBeamCorrection );
	
	// return success/failure.
	return ok;
	
} // generateKernel

//
//	getParameters()
//
//	CJS: 07/08/2015
//
//	Load the following parameters from the parameter file gridder-params: uv cell size, uv grid size, # w-planes, oversample.
//

void getParameters( char * pParameterFile )
{

	char params[1024], line[2048], par[1024];

	// initialise arrays.
	_hstMeasurementSetPath = (char **) malloc( sizeof( char * ) );
	_hstMeasurementSetPath[ 0 ] = NULL;
	_hstBeamID = (int *) malloc( sizeof( int ) );
	_hstBeamID[ 0 ] = -1;
	_hstFieldID = (char **) malloc( sizeof( char * ) );
	_hstFieldID[ 0 ] = NULL;
	_hstSpwRestriction = (char **) malloc( sizeof( char * ) );
	_hstSpwRestriction[ 0 ] = NULL;
	_hstDataField = (char **) malloc( sizeof( char * ) );
	_hstDataField[ 0 ] = NULL;
	_hstTableData = (char **) malloc( sizeof( char * ) );
	_hstTableData[ 0 ] = NULL;
	_hstBeamPattern = (char **) malloc( sizeof( char * ) );
	_hstBeamPattern[ 0 ] = NULL;

	// we keep track of the range of measurement set ids for the current path name. normally we would only have one, but if we use a wildcard then we could
	// have lots more.
	int minMeasurementSet = -1;
	int maxMeasurementSet = -1;
 
	// Open the parameter file and get all lines.
	FILE *fr = fopen( pParameterFile, "rt" );
	while (fgets( line, 1024, fr ) != NULL)
	{

		params[0] = '\0';
		sscanf( line, "%s %s", par, params );
		if (strcmp( par, MEASUREMENT_SET ) == 0)
		{

			// pathname.
			_hstMeasurementSets++;
			_hstMeasurementSetPath = (char **) realloc( _hstMeasurementSetPath, _hstMeasurementSets * sizeof( char * ) );
			_hstMeasurementSetPath[ _hstMeasurementSets - 1 ] = (char *) malloc( 1024 * sizeof( char ) );
			strcpy( _hstMeasurementSetPath[ _hstMeasurementSets - 1 ], params );

			// the mosaic, or beam, id. initialised to -1 and will be updated if we specify a wildcard.
			_hstBeamID = (int *) realloc( _hstBeamID, _hstMeasurementSets * sizeof( int ) );
			_hstBeamID[ _hstMeasurementSets - 1 ] = -1;

			// field id.
			_hstFieldID = (char **) realloc( _hstFieldID, _hstMeasurementSets * sizeof( char * ) );
			_hstFieldID[ _hstMeasurementSets - 1 ] = (char *) malloc( 1024 * sizeof( char ) );
			strcpy( _hstFieldID[ _hstMeasurementSets - 1 ], "\0" );

			// spw.
			_hstSpwRestriction = (char **) realloc( _hstSpwRestriction, _hstMeasurementSets * sizeof( char * ) );
			_hstSpwRestriction[ _hstMeasurementSets - 1 ] = (char *) malloc( 1024 * sizeof( char ) );
			strcpy( _hstSpwRestriction[ _hstMeasurementSets - 1 ], "\0" );

			// data field.
			_hstDataField = (char **) realloc( _hstDataField, _hstMeasurementSets * sizeof( char * ) );
			_hstDataField[ _hstMeasurementSets - 1 ] = (char *) malloc( 1024 * sizeof( char ) );
			strcpy( _hstDataField[ _hstMeasurementSets - 1 ], "CORRECTED_DATA" );

			// table data.
			_hstTableData = (char **) realloc( _hstTableData, _hstMeasurementSets * sizeof( char * ) );
			_hstTableData[ _hstMeasurementSets - 1 ] = (char *) malloc( 1024 * sizeof( char ) );
			strcpy( _hstTableData[ _hstMeasurementSets - 1 ], "\0" );

			// beam pattern.
			_hstBeamPattern = (char **) realloc( _hstBeamPattern, _hstMeasurementSets * sizeof( char * ) );
			_hstBeamPattern[ _hstMeasurementSets - 1 ] = (char *) malloc( 1024 * sizeof( char ) );
			strcpy( _hstBeamPattern[ _hstMeasurementSets - 1 ], "\0" );

			minMeasurementSet = _hstMeasurementSets - 1;
			maxMeasurementSet = _hstMeasurementSets - 1;

		}
		else if (strcmp( par, FIELD_ID ) == 0 && _hstMeasurementSets > 0)
		{

			// ensure we've read a path name already.
			if (minMeasurementSet >= 0 && maxMeasurementSet >= 0)
				for ( int i = minMeasurementSet; i <= maxMeasurementSet; i++ )
					strcpy( _hstFieldID[ i ], params );

		}
		else if (strcmp( par, SPW ) == 0 && _hstMeasurementSets > 0)
		{

			// ensure we've read a path name already.
			if (minMeasurementSet >= 0 && maxMeasurementSet >= 0)
				for ( int i = minMeasurementSet; i <= maxMeasurementSet; i++ )
					strcpy( _hstSpwRestriction[ i ], params );

		}
		else if (strcmp( par, DATA_FIELD ) == 0 && _hstMeasurementSets > 0)
		{

			// ensure we've read a path name already.
			if (minMeasurementSet >= 0 && maxMeasurementSet >= 0)
				for ( int i = minMeasurementSet; i <= maxMeasurementSet; i++ )
					strcpy( _hstDataField[ i ], params );

		}
		else if (strcmp( par, TABLE_DATA ) == 0 && _hstMeasurementSets > 0)
		{

			// ensure we've read a path name already.
			if (minMeasurementSet >= 0 && maxMeasurementSet >= 0)
				for ( int i = minMeasurementSet; i <= maxMeasurementSet; i++ )
					strcpy( _hstTableData[ i ], params );

		}
		else if (strcmp( par, BEAM_PATTERN ) == 0)
		{

			// ensure we've read a path name already.
			if (minMeasurementSet >= 0 && maxMeasurementSet >= 0)
				for ( int i = minMeasurementSet; i <= maxMeasurementSet; i++ )
					strcpy( _hstBeamPattern[ i ], params );

		}
		else if (strcmp( par, BEAM_ID ) == 0)
		{

			// ensure we've read a path name already.
			if (minMeasurementSet >= 0 && maxMeasurementSet >= 0)
				for ( int i = minMeasurementSet; i <= maxMeasurementSet; i++ )
					_hstBeamID[ i ] = atoi( params );

		}
		else if (strcmp( par, OUTPUT_PREFIX ) == 0)
			strcpy( _hstOutputPrefix, params );
		else if (strcmp( par, CELL_SIZE ) == 0)
			_hstCellSize = atof( params );
		else if (strcmp( par, PIXELS_UV ) == 0)
			_hstUvPixels = atoi( params );
		else if (strcmp( par, W_PLANES ) == 0)
		{
			_hstWPlanes = atoi( params );
			_hstWProjection = (_hstWPlanes > 0);
			if (_hstWProjection == false)
				_hstWPlanes = 1;
		}
		else if (strcmp( par, OVERSAMPLE ) == 0)
			_hstOversample = atof( params );
		else if (strcmp( par, KERNEL_CUTOFF_FRACTION ) == 0)
			_hstKernelCutoffFraction = atof( params );
		else if (strcmp( par, KERNEL_CUTOFF_SUPPORT ) == 0)
			_hstKernelCutoffSupport = atoi( params );
		else if (strcmp( par, CACHE_LOCATION ) == 0)
			strcpy( _hstCacheLocation, params );
		else if (strcmp( par, MINOR_CYCLES ) == 0)
			_hstMinorCycles = atoi( params );
		else if (strcmp( par, LOOP_GAIN ) == 0)
			_hstLoopGain = atof( params );
		else if (strcmp( par, CYCLEFACTOR ) == 0)
			_hstCycleFactor = atof( params );
		else if (strcmp( par, THRESHOLD ) == 0)
			_hstThreshold = atof( params );
		else if (strcmp( par, OUTPUT_RA ) == 0)
			_hstOutputRA = atof( params );
		else if (strcmp( par, OUTPUT_DEC ) == 0)
			_hstOutputDEC = atof( params );
		else if (strcmp( par, WEIGHTING ) == 0)
		{
			if (strcmp( params, "NATURAL" ) == 0)
				_hstWeighting = NATURAL;
			if (strcmp( params, "UNIFORM" ) == 0)
				_hstWeighting = UNIFORM;
			if (strcmp( params, "ROBUST" ) == 0)
				_hstWeighting = ROBUST;
		}
		else if (strcmp( par, ROBUST_PARAMETER ) == 0)
			_hstRobust = atof( params );
		else if (strcmp( par, A_PLANES ) == 0)
		{
			_hstAPlanes = atoi( params );
			_hstAProjection = (_hstAPlanes > 0);
			if (_hstAProjection == false)
				_hstAPlanes = 1;
		}
		else if (strcmp( par, MOSAIC ) == 0)
			_hstUseMosaicing = (strcmp( params, "Y" ) == 0 || strcmp( params, "YES" ) == 0 || strcmp( params, "y" ) == 0 || strcmp( params, "yes" ) == 0);
		else if (strcmp( par, MOSAIC_DOMAIN ) == 0)
			_hstMosaicDomain = (strcmp( params, "IMAGE" ) == 0 || strcmp( params, "image" ) == 0 ? IMAGE : UV );
		else if (strcmp( par, VISIBILITY_BATCH_SIZE ) == 0)
			_hstPreferredVisibilityBatchSize = atof( params );
		else if (strcmp( par, AIRY_DISK_DIAMETER ) == 0)
		{
			_hstAiryDiskDiameter = atof( params );
			_hstDiskDiameterSupplied = true;
		}
		else if (strcmp( par, AIRY_DISK_BLOCKAGE ) == 0)
		{
			_hstAiryDiskBlockage = atof( params );
			_hstDiskBlockageSupplied = true;
		}
		else if (strcmp( par, BEAM_SIZE_PIXELS ) == 0)
			_hstBeamSize = atoi( params );
		else if (strcmp( par, BEAM_CELL_SIZE ) == 0)
			_hstBeamCellSize = atof( params );
		else if (strcmp( par, BEAM_FREQUENCY ) == 0)
			_hstBeamFrequency = atof( params );
		else if (strcmp( par, STOKES ) == 0)
		{
			if (strcmp( params, "I" ) == 0 || strcmp( params, "i" ) == 0)
				_hstStokes = STOKES_I;
			else if (strcmp( params, "Q" ) == 0 || strcmp( params, "q" ) == 0)
				_hstStokes = STOKES_Q;
			else if (strcmp( params, "U" ) == 0 || strcmp( params, "u" ) == 0)
				_hstStokes = STOKES_U;
			else if (strcmp( params, "V" ) == 0 || strcmp( params, "v" ) == 0)
				_hstStokes = STOKES_V;
		}
		else if (strcmp( par, TELESCOPE ) == 0)
		{
			if (strcmp( params, "ASKAP" ) == 0)
				_hstTelescope = ASKAP;
			else if (strcmp( params, "ALMA" ) == 0)
				_hstTelescope = ALMA;
			else if (strcmp( params, "EMERLIN" ) == 0)
				_hstTelescope = EMERLIN;
			else if (strcmp( params, "VLA" ) == 0)
				_hstTelescope = VLA;
			else if (strcmp( params, "MEERKAT" ) == 0)
				_hstTelescope = MEERKAT;
		}
		else if (strcmp( par, GPU ) == 0)
			strcpy( _hstGPUParam, params );

		// debugging options.
		else if (strcmp( par, SAVE_MOSAIC_DIRTY_IMAGE ) == 0)
			_hstSaveMosaicDirtyImages = true;
            
	}
	fclose( fr );
	
} // getParameters

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

void gridVisibilities(	cufftComplex ** pdevGrid,		// data area (device) holding the grid
			cufftComplex ** pdevVisibility,		// data area (device) holding the visibilities
			int pOversample,				// }
			int * phstKernelSize,				// }- kernel and gridding parameters
			int * phstSupportSize,				// }
			int ** pdevKernelIndex,			// an array of kernel indexes assigned to each visibility
			bool pWProjection,				// true if w-projection is being used
			bool pAProjection,				// true if a-projection is being used
			int pWPlanes,					// the number of w-planes to use
			int pAPlanes,					// the number of a-planes to use
			VectorI ** pdevGridPositions,			// a list of integer grid positions for each visibility
			float ** pdevWeight,				// a list of weights for each visibility
			int ** pVisibilitiesInKernelSet,		// an array holding the number of visibilities in each kernel set and each GPU, for
									//		this batch of visibilities
			griddegrid pGridDegrid,				// GRID or DEGRID
			float * phstPrimaryBeamMosaicing,		// the primary beam for mosaicing
			float * phstPrimaryBeamAProjection,		// the primary beam for A-projection
			int pNumFields,				// the number of fields we have in our data.
			int pMosaicIndex,				// the index of the mosaic image currently being processed
			int pSize,					// the size of the image
			int pNumGPUs )					// the number of GPUs to use for gridding
{
	
	cudaError_t err;

	err = cudaGetLastError();
	if (err != cudaSuccess)
		printf( "unknown CUDA error in gridVisibilities() [i] (%s)\n", cudaGetErrorString( err ) );

	int numKernelSets = pWPlanes * pAPlanes;

	// find the range of kernels sets in this data.
	int minKernelSet = numKernelSets;
	int maxKernelSet = -1;
	for ( int i = 0; i < numKernelSets; i++ )
		if (pVisibilitiesInKernelSet[ i ] > 0)
		{
			if (i < minKernelSet)
				minKernelSet = i;
			if (i > maxKernelSet)
				maxKernelSet = i;
		}
	int numKernelSetsInData = (maxKernelSet + 1 - minKernelSet );
	printf( "                found visibilities for %i kernel set(s) (min: %i, max: %i)\n", numKernelSetsInData, minKernelSet, maxKernelSet );

	// calculate how many kernels we have per kernel set. this will be the number of oversampled kernels X by the number of
	// beam-correction kernels.
	int kernelsPerSet = pOversample * pOversample;
	if (pNumFields > -1)
		kernelsPerSet *= pNumFields;

	// declare array of device kernels.
	cufftComplex ** devKernel = (cufftComplex **) malloc( pNumGPUs * sizeof( cufftComplex * ) );
	for ( int gpu = 0; gpu < pNumGPUs; gpu++ )
		devKernel[ gpu ] = NULL;

	if (numKernelSetsInData > 0)
	{
		
		// maintain pointers to the next visibilities for each GPU.
		int * hstNextVisibility = (int *) malloc( pNumGPUs * sizeof( int ) );
		for ( int gpu = 0; gpu < pNumGPUs; gpu++ )
			hstNextVisibility[ gpu ] = 0;

		// initialise the GPU to 0. we will increment this number for each kernel set so that each kernel set is passed to a different gpu.
		int cudaDeviceIndex = 0;

		// grid the visibilities one kernel set at a time.
		for ( int kernelSet = minKernelSet; kernelSet <= maxKernelSet; kernelSet++ )
		{

			int firstGPU = cudaDeviceIndex;
			int latestGPU = cudaDeviceIndex;
			do
			{

				int numVisibilities = pVisibilitiesInKernelSet[ kernelSet ][ cudaDeviceIndex ];
				if (numVisibilities > 0)
				{

					// set the cuda device, and wait for whatever is running to finish.
					if (pNumGPUs > 1)
						cudaSetDevice( _hstGPU[ cudaDeviceIndex ] );

					// free the kernels on this device if they exist.
					if (devKernel[ cudaDeviceIndex ] != NULL)
					{
						cudaFree( (void *) devKernel[ cudaDeviceIndex ] );
						devKernel[ cudaDeviceIndex ] = NULL;
					}

					// calculate W- and A-plane.
					int wPlane = kernelSet / pAPlanes;
					int aPlane = kernelSet % pAPlanes;

					if (pGridDegrid == GRID)
						printf( "                        gridding " );
					else
						printf( "                        degridding " );
					printf( "%i visibilities", numVisibilities );
					if (pWProjection == true || pAProjection == true)
						printf( " for " );
					if (pWProjection == true)
						printf( "w-plane %i", wPlane );
					if (pWProjection == true && pAProjection == true)
						printf( " and " );
					if (pAProjection == true)
						printf( "a-plane %i", aPlane );
					if (pNumGPUs > 1)
						printf( " on GPU %i", _hstGPU[ cudaDeviceIndex ] );

					// how many fields are there? If pNumFields is -1 then we only have one field, otherwise we are using beam mosaicing and may
					// have lots.
					int numFields = (pNumFields == -1 ? 1 : pNumFields);

					// create an array of kernel pointers for each field, and an array to store the sizes.
					cufftComplex ** devFieldKernelPtr = (cufftComplex **) malloc( numFields * sizeof( cufftComplex * ) );
					int * hstFieldKernelSize = (int *) malloc( numFields * sizeof( int ) );

					cufftComplex * devWorkspace = NULL;
					bool kernelOverflow = false;
					for ( int field = 0; field < numFields; field++ )
					{

						// get primary beam pointer (for mosaicing).
						float * primaryBeamMosaicing = phstPrimaryBeamMosaicing;
						if (primaryBeamMosaicing != NULL)
							primaryBeamMosaicing = &primaryBeamMosaicing[ field * _hstBeamSize * _hstBeamSize ];

						// get primary beam pointer (for A-projection).
						float * primaryBeamAProjection = phstPrimaryBeamAProjection;
						if (primaryBeamAProjection != NULL)
							primaryBeamAProjection = &primaryBeamAProjection[ ((field * pAPlanes) + aPlane) * _hstBeamSize * _hstBeamSize ];

						// generate kernel.
						generateKernel(	/* pW = */ wPlane,
								/* pA = */ aPlane,
								/* pWProjection = */ pWProjection,
								/* pAProjection = */ pAProjection,
								/* phstKernelSize = */ &hstFieldKernelSize[ field ],
								/* pOversample = */ pOversample,
								/* phstPrimaryBeamMosaicing = */ primaryBeamMosaicing,
								/* phstPrimaryBeamAProjection = */ primaryBeamAProjection,
								/* pFieldID = */ (pNumFields == -1 ? -1 : field),
								/* pMosaicIndex = */ pMosaicIndex,
								/* pGridDegrid = */ pGridDegrid,
								/* pdevKernelPtr = */ &devWorkspace,
								/* pKernelOverflow = */ &kernelOverflow );

						// create space for the kernel.
						reserveGPUMemory( (void **) &devFieldKernelPtr[ field ], hstFieldKernelSize[ field ] * hstFieldKernelSize[ field ] *
									pOversample * pOversample * sizeof( cufftComplex ), "creating device memory for field kernel" );

						// copy the kernels for this field into the kernel area on the device.
						cudaMemcpy(	(void *) devFieldKernelPtr[ field ],
								(void *) devWorkspace,
								hstFieldKernelSize[ field ] * hstFieldKernelSize[ field ] * pOversample * pOversample *
										sizeof( cufftComplex ),
								cudaMemcpyDeviceToDevice );

						// get rid of the workspace.
						cudaFree( (void *) devWorkspace );

					} // LOOP: field

					// display a warning if any kernel has been truncated in size.
					if (kernelOverflow == true)
						printf( " - WARNING: one or more kernel(s) are truncated." );

					printf( "\n" );

					// set kernel size to the maximum of the field kernel sizes.
					for ( int field = 0; field < numFields; field++ )
						if (hstFieldKernelSize[ field ] > phstKernelSize[ kernelSet ] || field == 0)
							phstKernelSize[ kernelSet ] = hstFieldKernelSize[ field ];
					phstSupportSize[ kernelSet ] = (phstKernelSize[ kernelSet ] - 1) / 2;

					// create space for the kernels on the device, and clear this memory.
					reserveGPUMemory( (void **) &devKernel[ cudaDeviceIndex ], phstKernelSize[ kernelSet ] * phstKernelSize[ kernelSet ] *
								kernelsPerSet * sizeof( cufftComplex ), "creating device memory for the kernels" );
					zeroGPUMemory( (void *) devKernel[ cudaDeviceIndex ], phstKernelSize[ kernelSet ] * phstKernelSize[ kernelSet ] * kernelsPerSet *
								sizeof( cufftComplex ), "zeroing device memory for the kernels" );

					// copy the kernels for all fields into the kernel area on the device.
					for ( int field = 0; field < numFields; field++ )
					{
						setThreadBlockSize2D( hstFieldKernelSize[ field ], hstFieldKernelSize[ field ] );
						for ( int i = 0; i < pOversample * pOversample; i++ )					
							devCopyImage<<< _gridSize2D, _blockSize2D >>>
								(	/* pNewImage = */ &devKernel[ cudaDeviceIndex ][ ((field * pOversample * pOversample) + i) *
															(phstKernelSize[ kernelSet ] *
																phstKernelSize[ kernelSet ]) ],
									/* pOldImage = */ &devFieldKernelPtr[ field ][ i * hstFieldKernelSize[ field ] *
															hstFieldKernelSize[ field ] ],
									/* pNewSize = */ phstKernelSize[ kernelSet ],
									/* pOldSize = */ hstFieldKernelSize[ field ],
									/* pScale = */ 1.0,
									/* pThreadOffset = */ (phstKernelSize[ kernelSet ] - hstFieldKernelSize[ field ]) / 2 );
					}

					// free field kernel pointers.
					if (devFieldKernelPtr != NULL)
					{
						for ( int i = 0; i < numFields; i++ )
							if (devFieldKernelPtr[ i ] != NULL)
								cudaFree( (void *) devFieldKernelPtr[ i ] );
						free( (void *) devFieldKernelPtr );
					}
					if (hstFieldKernelSize != NULL)
						free( (void *) hstFieldKernelSize );

					// ensure we don't report old errors.
					err = cudaGetLastError();
					if (err != cudaSuccess)
						printf( "unknown CUDA error in gridVisibilities() [ii] (%s)\n", cudaGetErrorString( err ) );

					if (pGridDegrid == GRID)
					{

						// define the block/thread dimensions.
						setThreadBlockSizeForGridding(	/* pThreadsX = */ phstKernelSize[ kernelSet ],
										/* pThreadsY = */ phstKernelSize[ kernelSet ],
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
									/* pNumKernels = */ kernelsPerSet,
									/* pSize = */ pSize,
									/* pComplex = */ true,
									/* pSupport = */ phstSupportSize[ kernelSet ] );

					}
					else
					{

						// define the block/thread dimensions.
						setThreadBlockSizeForDegridding(	/* pKernelSizeX = */ phstKernelSize[ kernelSet ],
											/* pKernelSizeY = */ phstKernelSize[ kernelSet ],
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
									/* pSupport = */ phstSupportSize[ kernelSet ] );

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
		
		} // LOOP: kernelSet

		// clear the kernels if they exist.
		for ( int gpu = 0; gpu < pNumGPUs; gpu++ )
		{
			if (devKernel[ gpu ] != NULL)
			{
				if (pNumGPUs > 1)
					cudaSetDevice( _hstGPU[ gpu ] );
				cudaFree( (void *) devKernel[ gpu ] );
				devKernel[ gpu ] = NULL;
			}
		} // LOOP: gpu

		// reset the GPU to the first device.
		if (pNumGPUs > 1)
			cudaSetDevice( _hstGPU[ 0 ] );

		// display the support sizes.
		bool first = true;
		printf( "                --> done. support size used in " );
		if (pGridDegrid == GRID)
			printf( "gridding = [" );
		else
			printf( "degridding = [" );
		for ( int i = 0; i < numKernelSets; i++ )
			if (pVisibilitiesInKernelSet[ i ] > 0)
			{
				if (first == false)
					printf( ", " );
				printf( "%i", phstSupportSize[ i ] );
				first = false;
			}
			else
			{
				if (first == false)
					printf( ", " );
				printf( "-" );
				first = false;
			}
		printf( "]\n\n" );

		// free memory.
		if (hstNextVisibility != NULL)
			free( (void *) hstNextVisibility );
		if (devKernel != NULL)
			free( (void *) devKernel );

	}
	
} // gridVisibilities

//
//	gridComponents()
//
//	CJS: 22/05/2020
//
//	Grids a list of clean components to the clean image.
//

void gridComponents(	float * pdevGrid,			// data area (device) holding the grid
			double * pdevComponentValue,			// data area (device) holding the visibilities
			int phstSupportSize,				// kernel and gridding parameters
			float * pdevKernel,			// the kernel array.
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
					/* pNumKernels = */ 1,
					/* pSize = */ pSize,
					/* pComplex = */ false,
					/* pSupport = */ phstSupportSize );

		err = cudaGetLastError();
		if (err != cudaSuccess)
			printf( "error gridding components on device (%s)\n", cudaGetErrorString( err ) );

	}
	
} // gridComponents

//
//	generateImageOfConvolutionFunction()
//
//	CJS: 18/01/2016
//
//	Generate the deconvolution function by gridding a single source at u = 0, v = 0, and FFT'ing.
//

bool generateImageOfConvolutionFunction( char * pDeconvolutionFilename )
{
	
	bool ok = true;
	cudaError_t err;
	
	// create the deconvolution image, and clear it.
	cufftComplex * devDeconvolutionImageGrid = NULL;
	reserveGPUMemory( (void **) &devDeconvolutionImageGrid, _hstPsfSize * _hstPsfSize * sizeof( cufftComplex ),
				"declaring memory for deconvolution image" );
	zeroGPUMemory( (void *) devDeconvolutionImageGrid, _hstPsfSize * _hstPsfSize * sizeof( cufftComplex ), "zeroing the grid on the device" );

	// create space for a single visibility on the device.
	cufftComplex * tmpdevVisibility;
	ok = reserveGPUMemory( (void **) &tmpdevVisibility, 1 * sizeof( cufftComplex ), "declaring device memory for visibility" );
	if (ok == true)
	{
		cufftComplex tmphstVisibility;
		tmphstVisibility.x = 1;
		tmphstVisibility.y = 0;
		ok = ok && moveHostToDevice( (void *) tmpdevVisibility, (void *) &tmphstVisibility, sizeof( cufftComplex ), "copying visibility to device" );
	}

	// create space for a single weight on the device.
	float * tmpdevWeight = NULL;
	reserveGPUMemory( (void **) &tmpdevWeight, 1 * sizeof( float ), "declaring device memory for weights" );

	float tmpWeight = 1.0;
	moveHostToDevice( (void *) tmpdevWeight, (void *) &tmpWeight, sizeof( float ), "copying weight to device" );
	
	// create a single int pointer to use as a kernel pointer.
	int * tmpdevKernelIndex = NULL;
	ok = reserveGPUMemory( (void **) &tmpdevKernelIndex, 1 * sizeof( int ), "declaring device memory for kernel index" );
	if (ok == true)
	{
		int tmphstKernelIndex = 0;
		ok = ok && moveHostToDevice( (void *) tmpdevKernelIndex, (void *) &tmphstKernelIndex, sizeof( int ), "copying kernel index to device" );
	}
	
	// create a single vector to hold the grid positions on the device.
	VectorI * tmpdevGridPositions = NULL;
	ok = reserveGPUMemory( (void **) &tmpdevGridPositions, 1 * sizeof( VectorI ), "declaring device memory for grid positions" );
	if (ok == true)
	{
		VectorI tmphstGridPositions;
		tmphstGridPositions.u = (_hstPsfSize / 2.0);
		tmphstGridPositions.v = (_hstPsfSize / 2.0);
		tmphstGridPositions.w = 0;
		ok = ok && moveHostToDevice( (void *) tmpdevGridPositions, (void *) &tmphstGridPositions, sizeof( VectorI ), "copying grid positions to device" );
	}
	
	// create a single integer to hold the number of visibilities per kernel set.
	int * tmphstVisibilitiesPerKernelSet = (int *) malloc( sizeof( int ) );
	tmphstVisibilitiesPerKernelSet[ 0 ] = 1;

	err = cudaGetLastError();
	if (err != cudaSuccess)
		printf( "unknown CUDA error on line %d (%s)\n", __LINE__, cudaGetErrorString( err ) );
		
	// generate the deconvolution function by gridding a single visibility without w-projection
	// (mirror visibilities = false, use conjugate values = false).
	printf( "gridding visibilities for deconvolution function.....\n\n" );
	gridVisibilities(	/* pdevGrid = */ &devDeconvolutionImageGrid,
				/* pdevVisibility = */ &tmpdevVisibility,
				/* pOversample = */ 1,
				/* pKernelSize = */ &_hstAAKernelSize,
				/* pSupport = */ &_hstAASupport,
				/* pdevKernelIndex = */ &tmpdevKernelIndex,
				/* pWProjection = */ false,
				/* pAProjection = */ false,
				/* pWPlanes = */ 1,
				/* pAPlanes = */ 1,
				/* pdevGridPositions = */ &tmpdevGridPositions,
				/* pdevWeight = */ &tmpdevWeight,
				/* pVisibilitiesInKernelSet = */ &tmphstVisibilitiesPerKernelSet,
				/* pGridDegrid = */ GRID,
				/* phstPrimaryBeamMosaicing = */ NULL,
				/* phstPrimaryBeamAProjection = */ NULL,
				/* pNumFields = */ -1,
				/* pMosaicIndex = */ -1,
				/* pSize = */ _hstPsfSize,
				/* pNumGPUs = */ 1 );

	// free memory.
	if (tmphstVisibilitiesPerKernelSet != NULL)
		free( (void *) tmphstVisibilitiesPerKernelSet );

	err = cudaGetLastError();
	if (err != cudaSuccess)
		printf( "unknown CUDA error on line %d (%s)\n", __LINE__, cudaGetErrorString( err ) );

	// FFT the gridded data to get the deconvolution map.
	performFFT(	/* pdevGrid = */ &devDeconvolutionImageGrid,
			/* pSize = */ _hstPsfSize,
			/* pFFTDirection = */ INVERSE,
			/* pFFTPlan = */ -1,
			/* pFFTType = */ C2F );

	// create memory for the deconvolution image, and copy the image from the device.
	_hstDeconvolutionImage = (float *) malloc( _hstPsfSize * _hstPsfSize * sizeof( float ) );
	moveDeviceToHost( (void *) _hstDeconvolutionImage, (void *) devDeconvolutionImageGrid, _hstPsfSize * _hstPsfSize * sizeof( float ),
					"copying deconvolution image from device" );

	// re-cast the deconvolution image pointer from a complex to a float.
	_devDeconvolutionImage = (float *) devDeconvolutionImageGrid;

	// save the deconvolution image.
	_hstCasacoreInterface.WriteCasaImage(	/* pFilename = */ pDeconvolutionFilename,
						/* pWidth = */ _hstPsfSize,
						/* pHeight = */ _hstPsfSize,
						/* pRA = */ _hstOutputRA,
						/* pDec = */ _hstOutputDEC,
						/* pPixelSize = */ _hstCellSize * (double) _hstUvPixels / (double) _hstPsfSize, 
						/* pImage = */ _hstDeconvolutionImage,
						/* pFrequency = */ CONST_C / _hstAverageWavelength[ 0 ],
						/* pMask = */ NULL );
	
	// clean up memory.
	cudaFree( (void *) tmpdevVisibility );
	cudaFree( (void *) tmpdevKernelIndex );
	cudaFree( (void *) tmpdevGridPositions );
	cudaFree( (void *) tmpdevWeight );

	// we don't need the host deconvolution image anymore. it's still on the device.
	if (_hstDeconvolutionImage != NULL)
		free( (void *) _hstDeconvolutionImage );
	_hstDeconvolutionImage = NULL;

	printf( "\n" );
	
	// return success flag.
	return ok;
	
} // generateImageOfConvolutionFunction

//
//	extractFromMosaic()
//
//	CJS: 19/03/2019
//
//	Extracts an image from the mosaic.
//

void extractFromMosaic( float * phstImageArray, float * pdevMosaic, bool * phstMask, double * phstPhaseCentre, float * phstPrimaryBeamPattern )
{

	// create memory for mosaic, mask and beam on the device.
	bool * devMask = NULL;
	float * devBeam = NULL;

	// copy the mask into device memory.
	if (phstMask != NULL)
	{
		reserveGPUMemory( (void **) &devMask, (long int) _hstUvPixels * (long int) _hstUvPixels * (long int) sizeof( bool ),
					"creating device memory for the image mask" );
		cudaMemcpy( devMask, phstMask, (long int) _hstUvPixels * (long int) _hstUvPixels * (long int) sizeof( bool ), cudaMemcpyHostToDevice );
	}
	reserveGPUMemory( (void **) &devBeam, _hstBeamSize * _hstBeamSize * sizeof( float ), "creating device memory for the primary beam" );

	// do we need to reintroduce the primary beam pattern ? the reason this may be needed is because we have multiplied our mosaic by the primary beam pattern to
	// suppress flux around the edges that is only covered by one beam. but we need the model visibilities to model the flux PRIOR to the primary-beam-pattern
	// correction or we will keep cleaning out the same flux with every major cycle.
	if (phstPrimaryBeamPattern != NULL)
	{

		// create some device memory for the primary beam pattern, and copy the pattern from the host.
		float * devPrimaryBeamPattern = NULL;
		reserveGPUMemory( (void **) &devPrimaryBeamPattern, _hstBeamSize * _hstBeamSize * sizeof( float ), "creating device memory for the primary beam pattern" );
		cudaMemcpy( (void *) devPrimaryBeamPattern, phstPrimaryBeamPattern, _hstBeamSize * _hstBeamSize * sizeof( float ), cudaMemcpyHostToDevice );

		// divide the mosaic by the primary beam pattern.
		setThreadBlockSize2D( _hstUvPixels, _hstUvPixels );
		devDivideImages<<< _gridSize2D, _blockSize2D >>>(	/* pOne = */ pdevMosaic,
									/* pTwo = */ devPrimaryBeamPattern,
									/* pMask = */ devMask,
									/* pSizeOne = */ _hstUvPixels,
									/* pSizeTwo = */ _hstBeamSize );

		// free the primary beam pattern on the device.
		cudaFree( (void *) devPrimaryBeamPattern );

	}

	printf( "extracting from mosaic using image-plane reprojection.....\n\n" );

	// create a reprojection object.
	Reprojection imagePlaneReprojection;

	// set up pixel vector and CD matrix.
	Reprojection::rpVectD tmpPixel = { /* x = */ _hstUvPixels / 2, /* y = */ _hstUvPixels / 2, /* z = */ 0 };
	Reprojection::rpMatr2x2 tmpCD = { /* a11 = */ -sin( rad( _hstCellSize / 3600.0 ) ), /* a12 = */ 0.0, /* a21 = */ 0.0, /* a22 = */ sin( rad( _hstCellSize / 3600.0 ) ) };

	// build input and output size.
	Reprojection::rpVectI size = { /* x = */ _hstUvPixels, /* y = */ _hstUvPixels };

	// build beam size.
	Reprojection::rpVectI beamSize = { /* x = */ _hstBeamSize, /* y = */ _hstBeamSize };

	// build in coordinate system.
	Reprojection::rpCoordSys inCoordSystem;
	inCoordSystem.crVAL.x = _hstOutputRA;
	inCoordSystem.crVAL.y = _hstOutputDEC;
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

	// create device memory for the output image.
	float * devOutImage = NULL;
	reserveGPUMemory( (void **) &devOutImage, (long int) _hstUvPixels * (long int) _hstUvPixels * (long int) sizeof( float ),
				"creating device memory for image extracted from mosaic" );

	// perform an image-plane reprojection of each image from the phase position of the field to the phase position of the mosaic.
	for ( int image = 0; image < _numMosaicImages; image++ )
	{

		// clear the grid.
		zeroGPUMemory( (void *) devOutImage, (long int) _hstUvPixels * (long int) _hstUvPixels * (long int) sizeof( float ),
					"zeroing grid on the device" );

		// set out coordinate system RA and DEC.
		outCoordSystem.crVAL.x = phstPhaseCentre[ image * 2 ];
		outCoordSystem.crVAL.y = phstPhaseCentre[ (image * 2) + 1 ];

		// upload the primary beam to the device.
		cudaMemcpy( devBeam, _hstPrimaryBeam[ image ], _hstBeamSize * _hstBeamSize * sizeof( float ), cudaMemcpyHostToDevice );

		// reproject this image in order to construct this part of the mosaic.
		imagePlaneReprojection.ReprojectImage(	/* pdevInImage = */ pdevMosaic,
							/* pdevOutImage = */ devOutImage,
							/* pdevNormalisationPattern = */ NULL,
							/* pdevPrimaryBeamPattern = */ NULL,
							/* pInCoordinateSystem = */ inCoordSystem,
							/* pOutCoordinateSystem = */ outCoordSystem,
							/* pInSize = */ size,
							/* pOutSize = */ size,
							/* pdevInMask = */ devMask,
							/* pdevBeamIn = */ NULL,
							/* pdevBeamOut = */ devBeam,
							/* pBeamSize = */ beamSize,
							/* pProjectionDirection = */ Reprojection::INPUT_TO_OUTPUT,
							/* pAProjection = */ _hstAProjection,
							/* pVerbose = */ false );

		// store the image on the host.
		moveDeviceToHost( (void *) &phstImageArray[ (long int) _hstUvPixels * (long int) _hstUvPixels * (long int) image ], (void *) devOutImage,
					_hstUvPixels * _hstUvPixels * sizeof( float ), "copying mosaic component to host" );

	}
	printf( "\n" );

	// free memory.
	if (devMask != NULL)
		cudaFree( (void *) devMask );
	if (devBeam != NULL)
		cudaFree( (void *) devBeam );
	if (devOutImage != NULL)
		cudaFree( (void *) devOutImage );

} // extractFromMosaic

//
//	createMosaic()
//
//	CJS: 19/03/2019
//
//	Create a mosaic image using an array of images.
//

void createMosaic( float * phstMosaic, float * phstImageArray, bool * phstMask, double * phstPhaseCentre, float ** phstPrimaryBeamPatternPtr )
{

	// create memory for mosaic, pixel weights, mask and primary beam on the device.
	float * devMosaic = NULL;
	float * devNormalisationPattern = NULL;
	float * devPrimaryBeamPattern = NULL;
	bool * devMask = NULL;
	float * devBeam = NULL;
	reserveGPUMemory( (void **) &devMosaic, _hstUvPixels * _hstUvPixels * sizeof( float ), "creating device memory for the mosaic" );
	reserveGPUMemory( (void **) &devNormalisationPattern, _hstBeamSize * _hstBeamSize * sizeof( float ),
				"creating device memory for the mosaic normalisation pattern" );
	if (phstPrimaryBeamPatternPtr != NULL)
		reserveGPUMemory( (void **) &devPrimaryBeamPattern, _hstBeamSize * _hstBeamSize * sizeof( float ),
					"creating device memory for the mosaic primary beam pattern" );
	reserveGPUMemory( (void **) &devMask, _hstUvPixels * _hstUvPixels * sizeof( bool ), "creating device memory for the image mask" );
	reserveGPUMemory( (void **) &devBeam, _hstBeamSize * _hstBeamSize * sizeof( float ), "creating device memory for the primary beam" );

	printf( "\nconstructing mosaic using image-plane reprojection.....\n\n" );

	// create a reprojection object.
	Reprojection imagePlaneReprojection;

	// set up pixel vector and CD matrix.
	Reprojection::rpVectD tmpPixel = { /* x = */ _hstUvPixels / 2, /* y = */ _hstUvPixels / 2, /*z = */ 0 };
	Reprojection::rpMatr2x2 tmpCD = { /* a11 = */ -sin( rad( _hstCellSize / 3600.0 ) ), /* a12 = */ 0.0, /* a21 = */ 0.0, /* a22 = */ sin( rad( _hstCellSize / 3600.0 ) ) };

	// build input and output size.
	Reprojection::rpVectI size = { /* x = */ _hstUvPixels, /* y = */ _hstUvPixels };

	// build beam size.
	Reprojection::rpVectI beamSize = { /* x = */ _hstBeamSize, /* y = */ _hstBeamSize };

	// build in coordinate system.
	Reprojection::rpCoordSys inCoordSystem;
	inCoordSystem.crPIX = tmpPixel;
	inCoordSystem.cd = tmpCD;
	inCoordSystem.epoch = Reprojection::EPOCH_J2000;

	// build out coordinate system.
	Reprojection::rpCoordSys outCoordSystem;
	outCoordSystem.crVAL.x = _hstOutputRA;
	outCoordSystem.crVAL.y = _hstOutputDEC;
	outCoordSystem.crPIX = tmpPixel;
	outCoordSystem.cd = tmpCD;
	outCoordSystem.epoch = Reprojection::EPOCH_J2000;

	// clear the output image, the normalisation pattern, the primary beam pattern, and the mask.
	zeroGPUMemory( (void *) devMosaic, _hstUvPixels * _hstUvPixels * sizeof( float ), "zeroing mosaic on the device" );
	zeroGPUMemory( (void *) devNormalisationPattern, _hstBeamSize * _hstBeamSize * sizeof( float ), "zeroing normalisation pattern on the device" );
	if (phstPrimaryBeamPatternPtr != NULL)
		zeroGPUMemory( (void *) devPrimaryBeamPattern, _hstBeamSize * _hstBeamSize * sizeof( float ), "zeroing primary beam pattern on the device" );
	zeroGPUMemory( (void *) devMask, _hstUvPixels * _hstUvPixels * sizeof( bool ), "zeroing mask on the device" ); // set mask to false (zero).

	// create the device memory for the reprojection code.
	imagePlaneReprojection.CreateDeviceMemory( size );

	// create device memory for the input image.
	float * devInImage = NULL;
	reserveGPUMemory( (void **) &devInImage, _hstUvPixels * _hstUvPixels * sizeof( double ), "creating device memory for image to be added to the mosaic" );

	// perform an image-plane reprojection of each image from the phase position of the field to the phase position of the mosaic.
	for ( int image = 0; image < _numMosaicImages; image++ )
	{

		// copy the image into a temporary work location on the device.
		cudaMemcpy( devInImage, &phstImageArray[ image * _hstUvPixels * _hstUvPixels ], _hstUvPixels * _hstUvPixels * sizeof( float ),
				cudaMemcpyHostToDevice );

		// set in coordinate system RA and DEC.
		inCoordSystem.crVAL.x = phstPhaseCentre[ image * 2 ];
		inCoordSystem.crVAL.y = phstPhaseCentre[ (image * 2) + 1 ];

		// upload the primary beam to the device.
		if (image == 0 || _hstFileMosaic == true)
			cudaMemcpy( devBeam, _hstPrimaryBeam[ image ], _hstBeamSize * _hstBeamSize * sizeof( float ), cudaMemcpyHostToDevice );

		// reproject this image in order to construct this part of the mosaic.
		imagePlaneReprojection.ReprojectImage(	/* pdevInImage = */ devInImage,
							/* pdevOutImage = */ devMosaic,
							/* pdevNormalisationPattern = */ devNormalisationPattern,
							/* pdevPrimaryBeamPattern = */ devPrimaryBeamPattern,
							/* pInCoordinateSystem = */ inCoordSystem,
							/* pOutCoordinateSystem = */ outCoordSystem,
							/* pInSize = */ size,
							/* pOutSize = */ size,
							/* pdevInMask = */ NULL,
							/* pdevBeamIn = */ devBeam,
							/* pdevBeamOut = */ NULL,
							/* pBeamSize = */ beamSize,
							/* pProjectionDirection = */ Reprojection::OUTPUT_TO_INPUT,
							/* pAProjection = */ _hstAProjection,
							/* pVerbose = */ false );

	}
	printf( "\n" );

	// free memory.
	if (devInImage != NULL)
		cudaFree( (void *) devInImage );

	// update the image from its weight.
	imagePlaneReprojection.ReweightImage(	/* pdevOutImage = */ devMosaic,
						/* pdevNormalisationPattern = */ devNormalisationPattern,
						/* pdevPrimaryBeamPattern = */ devPrimaryBeamPattern,
						/* pOutSize = */ size,
						/* pdevOutMask = */ devMask,
						/* pBeamSize = */ beamSize );

	// store the image and the mask on the host.
	moveDeviceToHost( (void *) phstMosaic, (void *) devMosaic, _hstUvPixels * _hstUvPixels * sizeof( float ), "copying mosaic image to host" );
	if (phstMask != NULL && devMask != NULL)
		moveDeviceToHost( (void *) phstMask, (void *) devMask, _hstUvPixels * _hstUvPixels * sizeof( bool ), "copying mosaic mask to host" );

	// are we dealing with a primary beam pattern ?
	if (phstPrimaryBeamPatternPtr != NULL)
	{

		// do we need to return the primary beam pattern to the host ? create host storage area, and copy from the device.
		if (*phstPrimaryBeamPatternPtr == NULL)
		{
			*phstPrimaryBeamPatternPtr = (float *) malloc( _hstBeamSize * _hstBeamSize * sizeof( float ) );
			moveDeviceToHost( (void *) *phstPrimaryBeamPatternPtr, (void *) devPrimaryBeamPattern, _hstBeamSize * _hstBeamSize * sizeof( float ),
						"copying primary beam pattern to host" );
		}

	}

	// free memory.
	if (devMosaic != NULL)
		cudaFree( (void *) devMosaic );
	if (devNormalisationPattern != NULL)
		cudaFree( (void *) devNormalisationPattern );
	if (devPrimaryBeamPattern != NULL)
		cudaFree( (void *) devPrimaryBeamPattern );
	if (devMask != NULL)
		cudaFree( (void *) devMask );
	if (devBeam != NULL)
		cudaFree( (void *) devBeam );

} // createMosaic

//
//	hogbomClean()
//
//	CJS: 05/11/2015
//
//	Perform a Hogbom clean on our dirty image.
//

void hogbomClean( int * pMinorCycle, bool * phstMask, double pHogbomLimit, float * pdevCleanBeam, float * pdevDirtyBeam,
			float * pdevDirtyImage, float * phstDirtyImage, VectorI * phstComponentListPos, double * phstComponentListValue,
			int * pComponentListItems )
{
	
	cudaError_t err;
		
	printf( "\n                minor cycles: " );
	fflush( stdout );
	
	double * devMaxValue;
	reserveGPUMemory( (void **) &devMaxValue, MAX_PIXEL_DATA_AREA_SIZE * sizeof( double ),
						"declaring device memory for psf max pixel value" );
	double * devMaxValueParallel;
	reserveGPUMemory( (void **) &devMaxValueParallel, MAX_PIXEL_DATA_AREA_SIZE * sizeof( double ),
						"declaring device memory for psf max pixel value" );
		
	// reserve host memory for the maximum pixel value.
	double * tmpMaxValue = (double *) malloc( MAX_PIXEL_DATA_AREA_SIZE * sizeof( double ) );

	// keep a record of the minimum value. if it does up by a certain factor then we need to stop cleaning.
	double minimumValue = -1.0;
	
	cudaDeviceSynchronize();

	err = cudaGetLastError();
	if (err != cudaSuccess)
		printf( "unknown CUDA error in hogbomClean() [i] (%s)\n", cudaGetErrorString( err ) );

	// create device memory for the mask. we only get the mask from the cache if we're using one.
	bool * devMask = NULL;
	if (phstMask != NULL)
	{
		reserveGPUMemory( (void **) &devMask, _hstUvPixels * _hstUvPixels * sizeof( bool ), "creating device memory for the mask" );
		cudaMemcpy( (void *) devMask, phstMask, _hstUvPixels * _hstUvPixels * sizeof( bool ), cudaMemcpyHostToDevice );
	}
	
	// loop over each minor cycle.
	while (*pMinorCycle < _hstMinorCycles)
	{
		
		printf( "." );
		fflush( stdout );
		
		// get maximum pixel value.
		getMaxValue(	/* pdevImage = */ pdevDirtyImage,
				/* pdevMaxValue = */ devMaxValue,
				/* pWidth = */ _hstUvPixels,
				/* pHeight = */ _hstUvPixels,
				/* pdevMask = */ devMask );
	
		// get details back from the device.
		moveDeviceToHost( (void *) tmpMaxValue, (void *) devMaxValue, MAX_PIXEL_DATA_AREA_SIZE * sizeof( double ),
					"copying max pixel data to host" );

		// has the peak value fallen within a specified number of S.D. of the mean? If so, cleaning must stop.
		if (tmpMaxValue[ MAX_PIXEL_VALUE ] < pHogbomLimit)
		{
			printf( "\n                reached threshold of %6.4e Jy", pHogbomLimit );
			break;
		}

		// check if the peak value is rising rather than falling.
		if (minimumValue >= 0 && tmpMaxValue[ MAX_PIXEL_VALUE ] >= (minimumValue * 1.1))
		{
			printf( "\n                clean not converging on threshold %6.4e Jy", pHogbomLimit );
			break;
		}

		// update the minimum value.
		if (minimumValue < 0 || tmpMaxValue[ MAX_PIXEL_VALUE ] < minimumValue)
			minimumValue = tmpMaxValue[ MAX_PIXEL_VALUE ];
		
		// define the block/thread dimensions.
		setThreadBlockSize2D( _hstPsfSize, _hstPsfSize );
				
		// subtract dirty beam.
		devSubtractBeam<<< _gridSize2D, _blockSize2D >>>(	/* pImage = */ pdevDirtyImage,
									/* pBeam = */ pdevDirtyBeam,
									/* pMaxValue = */ devMaxValue,
									/* pWindowSize = */ _hstPsfSize / 2,
									/* pLoopGain = */ _hstLoopGain,
									/* pImageWidth = */ _hstUvPixels,
									/* pImageHeight = */ _hstUvPixels,
									/* pBeamSize = */ _hstPsfSize );
		err = cudaGetLastError();
		if (err != cudaSuccess)
			printf( "error subtracting beams (%s).\n", cudaGetErrorString( err ) );

		// add item to component list.
		int found = -1;
		int x = (int) round( tmpMaxValue[ MAX_PIXEL_X ] );
		int y = (int) round( tmpMaxValue[ MAX_PIXEL_Y ] );
		double value = tmpMaxValue[ MAX_PIXEL_VALUE ] * _hstLoopGain;
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
			phstComponentListValue[ *pComponentListItems ] = value;
			(*pComponentListItems)++;
		}
		else
			phstComponentListValue[ found ] += value;

		// next minor cycle.
		*pMinorCycle = *pMinorCycle + 1;

	}
	printf( "\n" );
		
	// free memory.
	if (devMaxValue != NULL)
		cudaFree( (void *) devMaxValue );
	if (devMaxValueParallel != NULL)
		cudaFree( (void *) devMaxValueParallel );
	if (tmpMaxValue != NULL)
		free( tmpMaxValue );
	if (devMask != NULL)
		cudaFree( (void *) devMask );
	
} // hogbomClean

//
//	cottonSchwabClean()
//
//	CJS: 13/08/2018.
//
//	Perform a major/minor cycle clean.
//

bool cottonSchwabClean( double * phstPhaseCentre, char * pFilenamePrefix, float * pdevCleanBeam, float * pdevDirtyBeam,
				float ** phstDirtyImage, float * phstDirtyImageCache, bool * phstMask,
				char * pCleanImageFilename, char * pResidualImageFilename )
{

	bool ok = true;

	// create memory to hold the current number of minor cycles.
	int numMinorCycles = 0;

	// create device memory for the psf mask, and the data area.
	bool * devPsfMask = NULL;
	double * devMaxValue;
	reserveGPUMemory( (void **) &devPsfMask, _hstPsfSize * _hstPsfSize * sizeof( bool ), "reserving device memory for the psf mask" );
	reserveGPUMemory( (void **) &devMaxValue, MAX_PIXEL_DATA_AREA_SIZE * sizeof( double ), "declaring device memory for kernel max pixel value" );

	// build a mask based upon the clean beam where TRUE means the value is less than 0.1.
	setThreadBlockSize2D( _hstPsfSize, _hstPsfSize );
	devBuildMask<<< _gridSize2D, _blockSize2D >>>(	/* pArray = */ pdevCleanBeam,
							/* pSize = */ _hstPsfSize,
							/* pValue = */ 0.1,
							/* pMaxMin = */ MASK_MAX,
							/* pMask = */ devPsfMask );

	// get the peak value from the psf sidelobes.
	getMaxValue(	/* pdevImage = */ pdevDirtyBeam,
			/* pdevMaxValue = */ devMaxValue,
			/* pWidth = */ _hstPsfSize,
			/* pHeight = */ _hstPsfSize,
			/* pdevMask = */ devPsfMask );
	double maxSidelobe = 0.0;
	moveDeviceToHost( (void *) &maxSidelobe, (void *) &devMaxValue[ MAX_PIXEL_VALUE ], sizeof( double ), "moving maximum sidelobe value to the host" );
	
	// free memory.
	if (devMaxValue != NULL)
		cudaFree( (void *) devMaxValue );
	if (devPsfMask != NULL)
		cudaFree( (void *) devPsfMask );

	// create a list of dirty image components.
	VectorI * hstComponentListPos = (VectorI *) malloc( _hstMinorCycles * sizeof( VectorI ) );
	double * hstComponentListValue = (double *) malloc( _hstMinorCycles * sizeof( double ) );
	int numComponents = 0;
		
	printf( "\nPerforming Cotton-Schwab Clean.....\n" );
	printf( "-----------------------------------\n" );

	printf( "\nThreshold = %6.4e Jy\n", _hstThreshold );
	printf( "Cycle factor = %f\n", _hstCycleFactor );
	printf( "Minor cycles = %i\n\n", _hstMinorCycles );

	bool reachedLimit = false;
	int majorCycle = 0;
	double bestResidual = 0.0;
	while (reachedLimit == false)
	{

		printf( "        Major cycle %i - ", majorCycle );

		// -------------------------------------------------------------------
		//
		// S T E P   1 :   H O G B O M   C L E A N
		//
		// -------------------------------------------------------------------

		float * hstMosaic = NULL;

		// if we are mosaicing, create a mosaic from the dirty images.
		if (_hstFileMosaic == true)
		{

			// create memory for mosaic on the host.
			hstMosaic = (float *) malloc( (long int) _hstUvPixels * (long int) _hstUvPixels * (long int) sizeof( float ) );

			// construct the mosaic.
			createMosaic(	/* phstMosaic = */ hstMosaic,
					/* phstImageArray = */ phstDirtyImageCache,
					/* phstMask = */ NULL,
					/* phstPhaseCentre = */ phstPhaseCentre,
					/* phstPrimaryBeamPatternPtr = */ &_hstPrimaryBeamPattern );

		}

		// get maximum residual.
		double maxResidual = 0.0;
		for ( long int i = 0; i < (long int) _hstUvPixels * (long int) _hstUvPixels; i++ )
		{

			// ... for mosaicing.
			if (hstMosaic != NULL)
				if (hstMosaic[ i ] > maxResidual && phstMask[ i ] == true)
					maxResidual = hstMosaic[ i ];

			// ... for not mosaicing.
			if (hstMosaic == NULL)
			{
				bool pixelOK = true;
				if (phstMask != NULL)
					pixelOK = (phstMask[ i ] == true);
				if ((*phstDirtyImage)[ i ] > maxResidual && pixelOK == true)
					maxResidual = (*phstDirtyImage)[ i ];
			}

		}

		// check if the residuals are getting worse. we stop cleaning if we reach 1.5x the best residual.
		if (maxResidual > bestResidual * 1.2 && majorCycle > 0)
		{
			printf( "the maximum residual (%6.4e Jy) is getting worse (was previously %6.4e Jy). clean is not converging and will stop\n", maxResidual,
					bestResidual );
			break;
		}
		if (maxResidual < bestResidual || majorCycle == 0)
			bestResidual = maxResidual;

		// calculate Hogbom limit.
		double hogbomLimit = _hstCycleFactor * maxSidelobe * maxResidual;
		if (_hstThreshold < hogbomLimit)
			printf( "cleaning down to %6.4e Jy (based upon maximum sidelobe of %8.6f mJy and maximum residual of %6.4e Jy)\n", hogbomLimit, maxSidelobe * 1000.0,
					maxResidual );
		else
		{
			hogbomLimit = _hstThreshold;
			printf( "cleaning down to required stopping threshold of %6.4e Jy\n", hogbomLimit );
		}

		// perform Hogbom cleaning on each mosaic image.
		int currentMinorCycles = numMinorCycles;

		reachedLimit = true; // this flag will be reset if we need to do another major cycle.
		if (numMinorCycles >= _hstMinorCycles)
			printf( "		reached maximum clean cycles (%i)\n", _hstMinorCycles );
		else
		{

			// create memory for the dirty image on the device, but only if we're able to clean it on the device.
			float * devDirtyImage = NULL;
			reserveGPUMemory( (void **) &devDirtyImage, _hstUvPixels * _hstUvPixels * sizeof( float ),
						"reserving device memory for dirty image (cleaning)" );

			// get the dirty image, clean image and model images from the cache.
			if (hstMosaic != NULL)
				cudaMemcpy( (void *) devDirtyImage, hstMosaic, _hstUvPixels * _hstUvPixels * sizeof( float ), cudaMemcpyHostToDevice );
			else
				cudaMemcpy( (void *) devDirtyImage, *phstDirtyImage, _hstUvPixels * _hstUvPixels * sizeof( float ), cudaMemcpyHostToDevice );

			// perform a Hogbom clean until we reach the noise limit.
			hogbomClean(	/* pMinorCycle = */ &numMinorCycles,
					/* phstMask = */ phstMask,
					/* pHogbomLimit = */ hogbomLimit,
					/* pdevCleanBeam = */ pdevCleanBeam,
					/* pdevDirtyBeam = */ pdevDirtyBeam,
					/* pdevDirtyImage = */ devDirtyImage,
					/* phstDirtyImage = */ (hstMosaic != NULL ? hstMosaic : *phstDirtyImage),
					/* phstComponentListPos = */ hstComponentListPos,
					/* phstComponentListValue = */ hstComponentListValue,
					/* pComponentListItems = */ &numComponents );

			// free memory.
			if (devDirtyImage != NULL)
				cudaFree( (void *) devDirtyImage );

			printf( "                %i clean iterations performed up to Major cycle %i\n", numMinorCycles, majorCycle );

			// if we haven't reached the required number of minor cycles, and we haven't reached the required stopping threshold, then we need to clean
			// some more.
			if (numMinorCycles < _hstMinorCycles && hogbomLimit != _hstThreshold)
				reachedLimit = false;

		}

		// free data.
		if (hstMosaic != NULL)
			free( (void *) hstMosaic );
		hstMosaic = NULL;
		if (*phstDirtyImage != NULL)
			free( (void *) *phstDirtyImage );
		*phstDirtyImage = NULL;

		// if there were no additional minor cycles then stop cleaning.
		if (numMinorCycles == currentMinorCycles)
		{
			printf( "no new minor cycles performed. cleaning will stop.\n" );
			reachedLimit = true;
		}

		// sort the components into order of y-value.
		quickSortComponents(	/* phstComponentListPos = */ hstComponentListPos,
					/* phstComponentListValue = */ hstComponentListValue,
					/* pLeft = */ 0,
					/* pRight = */ numComponents - 1 );

		// -------------------------------------------------------------------
		//
		// S T E P   2 :   C O N T R U C T   M O D E L   I M A G E S
		//
		// -------------------------------------------------------------------

		// upload the component values to the device.
		double * devComponentValue = NULL;
		reserveGPUMemory( (void **) &devComponentValue, numComponents * sizeof( double ), "reserving device memory for clean components" );
		moveHostToDevice( (void *) devComponentValue, hstComponentListValue, numComponents * sizeof( double ), "moving component list values to the device" );

		// upload the grid positions to the device.
		VectorI * devComponentPos = NULL;
		reserveGPUMemory( (void **) &devComponentPos, numComponents * sizeof( VectorI ), "reserving device memory for clean component positions" );
		moveHostToDevice( (void *) devComponentPos, hstComponentListPos, numComponents * sizeof( VectorI ), "moving component list positions to the device" );

		// upload a single pixel as a gridding kernel.
		float * devKernel = NULL;
		float kernel = 1.0;
		reserveGPUMemory( (void **) &devKernel, 1 * sizeof( float ), "reserving device memory for the model image gridding kernel" );
		cudaMemcpy( devKernel, &kernel, sizeof( float ), cudaMemcpyHostToDevice );

		// create the model image on the device.
		float ** devModelImage = (float **) malloc( _hstNumGPUs * sizeof( float * ) );
		reserveGPUMemory( (void **) &devModelImage[ 0 ], _hstUvPixels * _hstUvPixels * sizeof( float ), "creating memory for the model image" );
		zeroGPUMemory( devModelImage[ 0 ], _hstUvPixels * _hstUvPixels * sizeof( float ), "zeroing the model image on the device" );

		// grid the clean components to make a model image.
		gridComponents(	/* pdevGrid = */ devModelImage[ 0 ],
				/* pdevComponentValue = */ devComponentValue,
				/* phstSupportSize = */ 0,
				/* pdevKernel = */ devKernel,
				/* pdevGridPositions = */ devComponentPos,
				/* pComponents = */ numComponents,
				/* pSize = */ _hstUvPixels );

		// free memory.
		if (devKernel != NULL)
			cudaFree( (void *) devKernel );
		if (devComponentValue != NULL)
			cudaFree( (void *) devComponentValue );
		if (devComponentPos != NULL)
			cudaFree( (void *) devComponentPos );

		// do we need to reintroduce the primary beam pattern ? the reason this may be needed is because we have multiplied our mosaic by the primary beam pattern to
		// suppress flux around the edges that is only covered by one beam. but we need the model visibilities to model the flux PRIOR to the primary-beam-pattern
		// correction or we will keep cleaning out the same flux with every major cycle.
		if (_hstPrimaryBeamPattern != NULL && (_hstUVMosaic == true || _hstBeamMosaic == true))
		{

			// create some device memory for the primary beam pattern, and copy the pattern from the host.
			float * devPrimaryBeamPattern = NULL;
			reserveGPUMemory( (void **) &devPrimaryBeamPattern, _hstBeamSize * _hstBeamSize * sizeof( float ),
							"creating device memory for the primary beam pattern" );
			cudaMemcpy( (void *) devPrimaryBeamPattern, _hstPrimaryBeamPattern, _hstBeamSize * _hstBeamSize * sizeof( float ), cudaMemcpyHostToDevice );

			// divide the mosaic by the primary beam pattern.
			setThreadBlockSize2D( _hstUvPixels, _hstUvPixels );
			devDivideImages<<< _gridSize2D, _blockSize2D >>>(	/* pOne = */ devModelImage[ 0 ],
										/* pTwo = */ devPrimaryBeamPattern,
										/* pMask = */ NULL,
										/* pSizeOne = */ _hstUvPixels,
										/* pSizeTwo = */ _hstBeamSize );

			// free memory.
			cudaFree( (void *) devPrimaryBeamPattern );

		}

		// now extract model images from this mosaic. we will extract them into the dirty image array since we don't need these any more (they will be
		// rebuilt following degridding).
		if (_hstFileMosaic == true)
			extractFromMosaic(	/* phstImageArray = */ phstDirtyImageCache,
						/* pdevMosaic = */ devModelImage[ 0 ],
						/* phstMask = */ phstMask,
						/* phstPhaseCentre = */ phstPhaseCentre,
						/* phstPrimaryBeamPattern = */ _hstPrimaryBeamPattern );
		else
		{

			// since we are not doing an image-plane mosaic then we can normalise and FFT the model image here.
			// divide the model image by the deconvolution image.
			setThreadBlockSize2D( _hstUvPixels, _hstUvPixels );
			devDivideImages<<< _gridSize2D, _blockSize2D >>>(	/* pOne = */ devModelImage[ 0 ],
										/* pTwo = */ _devDeconvolutionImage,
										/* pMask = */ NULL,
										/* pSizeOne = */ _hstUvPixels,
										/* pSizeTwo = */ _hstPsfSize );

			// FFT the model image into the UV domain.
			performFFT(	/* pdevGrid = */ (cufftComplex **) &devModelImage[ 0 ],
					/* pSize = */ _hstUvPixels,
					/* pFFTDirection = */ FORWARD,
					/* pFFTPlan = */ -1,
					/* pFFTType = */ F2C );

		}

		// create the model image on the other devices.
		if (_hstNumGPUs > 1)
		{

			for ( int gpu = 1; gpu < _hstNumGPUs; gpu++ )
			{
				cudaSetDevice( _hstGPU[ gpu ] );
				reserveGPUMemory( (void **) &devModelImage[ gpu ], _hstUvPixels * _hstUvPixels * sizeof( cufftComplex ),
							"creating memory for the model image" );
			}
			cudaSetDevice( _hstGPU[ 0 ] );

			// copy the model image to all the gpus.
			cufftComplex * hsttmpModelImage = (cufftComplex *) malloc( _hstUvPixels * _hstUvPixels * sizeof( cufftComplex ) );

			moveDeviceToHost( (void *) hsttmpModelImage, (void *) devModelImage[ 0 ], _hstUvPixels * _hstUvPixels * sizeof( cufftComplex ),
						"moving model image to the host" );
			for ( int gpu = 1; gpu < _hstNumGPUs; gpu++ )
			{
				cudaSetDevice( _hstGPU[ gpu ] );
				moveHostToDevice( (void *) devModelImage[ gpu ], hsttmpModelImage, _hstUvPixels * _hstUvPixels * sizeof( cufftComplex ),
						"moving model image to the device" );
			}
			cudaSetDevice( _hstGPU[ 0 ] );

			// free memory.
			if (hsttmpModelImage != NULL)
				free( (void *) hsttmpModelImage );
			
		}

		// -------------------------------------------------------------------
		//
		// S T E P   3 :   C O N S T R U C T   N E W   D I R T Y   I M A G E S
		//
		// -------------------------------------------------------------------

		// create memory for the dirty image grid, and clear it.
		cufftComplex ** devDirtyImageGrid = (cufftComplex **) malloc( _hstNumGPUs * sizeof( cufftComplex * ) );
		for ( int gpu = 0; gpu < _hstNumGPUs; gpu++ )
		{
			if (_hstNumGPUs > 1)
				cudaSetDevice( _hstGPU[ gpu ] );
			reserveGPUMemory( (void **) &devDirtyImageGrid[ gpu ], _hstUvPixels * _hstUvPixels * sizeof( cufftComplex ),
						"reserving device memory for dirty image grid (cleaning)" );
			zeroGPUMemory( devDirtyImageGrid[ gpu ], _hstUvPixels * _hstUvPixels * sizeof( cufftComplex ),
						"zeroing the dirty image grid on the device" );
		}
		if (_hstNumGPUs > 1)
			cudaSetDevice( _hstGPU[ 0 ] );

		for ( int image = 0; image < _numMosaicImages; image++ )
		{

			// count total visibilities.
			long int totalVisibilities = 0;
			for ( int stageID = 0; stageID < _hstNumberOfStages[ image ]; stageID++ )
				totalVisibilities += _hstNumVisibilities[ image ][ stageID ];

			if (_numMosaicImages > 1)
				printf( "\n        processing mosaic component %i of %i.....\n\n", image + 1, _numMosaicImages );
			else
				printf( "\n        processing visibilities.....\n\n" );
			printf( "                stages: %i\n", _hstNumberOfStages[ image ] );
			printf( "                visibilities: %li\n", totalVisibilities );

			// if we are using file mosaicing then copy the model image from where it is temporarily stored in the dirty image cache.
			if (_hstFileMosaic == true)
			{

				cudaMemcpy(	/* dst = */ devModelImage[ 0 ],
						/* src = */ &phstDirtyImageCache[ _hstUvPixels * _hstUvPixels * image ],
						/* count = */ _hstUvPixels * _hstUvPixels * sizeof( float ),
						/* kind = */ cudaMemcpyHostToDevice );

				// divide the model image by the deconvolution image.
				setThreadBlockSize2D( _hstUvPixels, _hstUvPixels );
				devDivideImages<<< _gridSize2D, _blockSize2D >>>(	/* pOne = */ devModelImage[ 0 ],
											/* pTwo = */ _devDeconvolutionImage,
											/* pMask = */ NULL,
											/* pSizeOne = */ _hstUvPixels,
											/* pSizeTwo = */ _hstPsfSize );

				// FFT the model image into the UV domain.
				performFFT(	/* pdevGrid = */ (cufftComplex **) &devModelImage[ 0 ],
						/* pSize = */ _hstUvPixels,
						/* pFFTDirection = */ FORWARD,
						/* pFFTPlan = */ -1,
						/* pFFTType = */ F2C );

				// copy the model image to all the gpus.
				if (_hstNumGPUs > 1)
				{

					cufftComplex * hsttmpModelImage = (cufftComplex *) malloc( _hstUvPixels * _hstUvPixels * sizeof( cufftComplex ) );

					moveDeviceToHost( (void *) hsttmpModelImage, (void *) devModelImage[ 0 ], _hstUvPixels * _hstUvPixels * sizeof( cufftComplex ),
								"moving model image to the host" );
					for ( int gpu = 1; gpu < _hstNumGPUs; gpu++ )
					{
						cudaSetDevice( _hstGPU[ gpu ] );
						moveHostToDevice( (void *) devModelImage[ gpu ], hsttmpModelImage, _hstUvPixels * _hstUvPixels * sizeof( cufftComplex ),
								"moving model image to the device" );
					}
					cudaSetDevice( _hstGPU[ 0 ] );

					// free memory.
					if (hsttmpModelImage != NULL)
						free( (void *) hsttmpModelImage );
			
				}

			}

			printf( "\n" );

			// process all the stages.
			long int visibilitiesProcessed = 0;
			for ( int stageID = 0; stageID < _hstNumberOfStages[ image ]; stageID++ )
			{

				// uncache the data for this mosaic.
				if (_hstCacheData == true)
				{
					uncacheData(	/* pFilenamePrefix = */ pFilenamePrefix,
							/* pMosaicID = */ image,
							/* pBatchID = */ stageID,
							/* pWhatData = */ DATA_ALL,
							/* pOffset = */ 0 );
				}

				// create some memory to store the residual visibilities.
				_hstResidualVisibility = (cufftComplex *) malloc( _hstNumVisibilities[ image ][ stageID ] * sizeof( cufftComplex ) );

				// calculate the batch size.
				int hstVisibilityBatchSize = 0;
				{
					long int nextBatchSize = _hstNumVisibilities[ image ][ stageID ];
					if (nextBatchSize > _hstPreferredVisibilityBatchSize)
						nextBatchSize = _hstPreferredVisibilityBatchSize;
					hstVisibilityBatchSize = (int) nextBatchSize;
				}

				// variables for device memory.
				VectorI ** devGridPosition = (VectorI **) malloc( _hstNumGPUs * sizeof( VectorI ) );
				int ** devKernelIndex = (int **) malloc( _hstNumGPUs * sizeof( int * ) );
				int ** devDensityMap = (int **) malloc( _hstNumGPUs * sizeof( int * ) );
				cufftComplex ** devModelVisibilities = (cufftComplex **) malloc( _hstNumGPUs * sizeof( cufftComplex * ) );
				float ** devWeight = (float **) malloc( _hstNumGPUs * sizeof( float * ) );
				cufftComplex ** devOriginalVisibilities = (cufftComplex **) malloc( _hstNumGPUs * sizeof( cufftComplex * ) );
				for ( int gpu = 0; gpu < _hstNumGPUs; gpu++ )
				{
					if (_hstNumGPUs > 1)
						cudaSetDevice( _hstGPU[ gpu ] );
					reserveGPUMemory( (void **) &devGridPosition[ gpu ], hstVisibilityBatchSize * sizeof( VectorI ),
								"reserving device memory for grid positions" );
					reserveGPUMemory( (void **) &devKernelIndex[ gpu ], hstVisibilityBatchSize * sizeof( int ),
								"reserving device memory for kernel indexes" );
					reserveGPUMemory( (void **) &devDensityMap[ gpu ], hstVisibilityBatchSize * sizeof( int ),
								"declaring device memory for density map" );
					reserveGPUMemory( (void **) &devModelVisibilities[ gpu ], hstVisibilityBatchSize * sizeof( cufftComplex ),
								"creating device memory for model visibilities" );
					if (_hstWeighting != NONE)
						reserveGPUMemory( (void **) &devWeight[ gpu ], hstVisibilityBatchSize * sizeof( float ),
								"creating device memory for weights" );
					reserveGPUMemory( (void **) &devOriginalVisibilities[ gpu ], hstVisibilityBatchSize * sizeof( cufftComplex ),
								"creating memory for original visibilities" );
				}
				if (_hstNumGPUs > 1)
					cudaSetDevice( _hstGPU[ 0 ] );

				// here is the start of the visibility batch loop.
				int batch = 0;
				int hstCurrentVisibility = 0;
				while (hstCurrentVisibility < _hstNumVisibilities[ image ][ stageID ])
				{

					int ** hstVisibilitiesInKernelSet = _hstVisibilitiesInKernelSet[ image ][ stageID ][ batch ];

					// count the number of visibilities in this batch.
					int visibilitiesInThisBatch = 0;
					for ( int kernelSet = 0; kernelSet < _hstKernelSets; kernelSet++ )
						for ( int gpu = 0; gpu < _hstNumGPUs; gpu++ )
							visibilitiesInThisBatch += hstVisibilitiesInKernelSet[ kernelSet ][ gpu ];

					printf( "        degridding" );
					if (_hstNumberOfStages[ image ] > 1 || _hstNumberOfBatches[ image ][ stageID ] > 1)
						printf( " " );
					else
						printf( " visibilities\n\n" );
					if (_hstNumberOfStages[ image ] > 1)
						printf( "host batch %i of %i", stageID + 1, _hstNumberOfStages[ image ] );
					if (_hstNumberOfStages[ image ] > 1 && _hstNumberOfBatches[ image ][ stageID ] > 1)
						printf( ", " );
					if (_hstNumberOfBatches[ image ][ stageID ] > 1)
						printf( "gpu batch %i of %i", batch + 1, _hstNumberOfBatches[ image ][ stageID ] );
					if (_hstNumberOfStages[ image ] > 1 || _hstNumberOfBatches[ image ][ stageID ] > 1)
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
					int * hstNextVisibility = (int *) malloc( _hstNumGPUs * sizeof( int ) );
					for ( int gpu = 0; gpu < _hstNumGPUs; gpu++ )
						hstNextVisibility[ gpu ] = 0;

					int cudaDeviceIndex = 0;
					int visibilityPointer = hstCurrentVisibility;
					for ( int kernelSet = 0; kernelSet < _hstKernelSets; kernelSet++ )
					{

						int lastGPU = cudaDeviceIndex;
						do
						{

							if (hstVisibilitiesInKernelSet[ kernelSet ][ cudaDeviceIndex ] > 0)
							{

								// set the cuda device.
								if (_hstNumGPUs > 1)
									cudaSetDevice( _hstGPU[ cudaDeviceIndex ] );

								// upload grid positions, kernel indexes, density map, weights, and original visibilities to the device.
								moveHostToDevice( (void *) &devGridPosition[ cudaDeviceIndex ][ hstNextVisibility[ cudaDeviceIndex ] ],
											(void *) &_hstGridPosition[ visibilityPointer ],
											hstVisibilitiesInKernelSet[ kernelSet ][ cudaDeviceIndex ] * sizeof( VectorI ),
											"copying grid positions to the device" );
								moveHostToDevice( (void *) &devKernelIndex[ cudaDeviceIndex ][ hstNextVisibility[ cudaDeviceIndex ] ],
											(void *) &_hstKernelIndex[ visibilityPointer ],
											hstVisibilitiesInKernelSet[ kernelSet ][ cudaDeviceIndex ] * sizeof( int ),
											"copying kernel indexes to the device" );
								moveHostToDevice( (void *) &devDensityMap[ cudaDeviceIndex ][ hstNextVisibility[ cudaDeviceIndex ] ],
											(void *) &_hstDensityMap[ visibilityPointer ],
											hstVisibilitiesInKernelSet[ kernelSet ][ cudaDeviceIndex ] * sizeof( int ),
											"copying density map to the device" );
								if (_hstWeighting != NONE)
									moveHostToDevice( (void *) &devWeight[ cudaDeviceIndex ][ hstNextVisibility[ cudaDeviceIndex ] ],
												(void *) &_hstWeight[ visibilityPointer ],
												hstVisibilitiesInKernelSet[ kernelSet ][ cudaDeviceIndex ] * sizeof( float ),
												"copying weights to the device" );
								moveHostToDevice( (void *) &devOriginalVisibilities[ cudaDeviceIndex ]
																[ hstNextVisibility[ cudaDeviceIndex ] ],
											(void *) &_hstVisibility[ visibilityPointer ],
											hstVisibilitiesInKernelSet[ kernelSet ][ cudaDeviceIndex ] * sizeof( cufftComplex ),
											"copying original visibilities to the device" );

								// set the model visibilities to zero.
								zeroGPUMemory( (void *) &devModelVisibilities[ cudaDeviceIndex ][ hstNextVisibility[ cudaDeviceIndex ] ],
											hstVisibilitiesInKernelSet[ kernelSet ][ cudaDeviceIndex ] * sizeof( cufftComplex ),
											"clearing the model visibilities on the device" );

								// get the next set of visibilities.
								visibilityPointer += hstVisibilitiesInKernelSet[ kernelSet ][ cudaDeviceIndex ];
								hstNextVisibility[ cudaDeviceIndex ] += hstVisibilitiesInKernelSet[ kernelSet ][ cudaDeviceIndex ];

							} // hstVisibilitiesInKernelSet[ kernelSet ][ cudaDeviceIndex ] > 0
							cudaDeviceIndex++;
							if (cudaDeviceIndex == _hstNumGPUs)
								cudaDeviceIndex = 0;

						} while (cudaDeviceIndex != lastGPU);

					}
					if (_hstNumGPUs > 1)
						cudaSetDevice( _hstGPU[ 0 ] );

					// degridding with w-projection and oversampling.
					gridVisibilities(	/* pdevGrid = */ (cufftComplex **) devModelImage,
								/* pdevVisibility = */ devModelVisibilities,
								/* pOversample = */ _hstOversample,
								/* pKernelSize = */ _hstKernelSize,
								/* pSupport = */ _hstSupportSize,
								/* pdevKernelIndex = */ devKernelIndex,
								/* pWProjection = */ _hstWProjection,
								/* pAProjection = */ _hstAProjection,
								/* pWPlanes = */ _hstWPlanes,
								/* pAPlanes = */ _hstAPlanes,
								/* pdevGridPositions = */ devGridPosition,
								/* pdevWeight = */ NULL,
								/* pVisibilitiesInKernelSet = */ hstVisibilitiesInKernelSet,
								/* pGridDegrid = */ DEGRID,
								/* phstPrimaryBeamMosaicing = */
											(_hstBeamMosaic == true || _hstUVMosaic == true ? _hstPrimaryBeam[ image ] : NULL),
								/* phstPrimaryBeamAProjection = */ _hstPrimaryBeamAProjection[ image ],
								/* pNumFields = */ (_hstBeamMosaic == true ? _hstBeamMosaicComponents : -1),
								/* pMosaicIndex = */ image,
								/* pSize = */ _hstUvPixels,
								/* pNumGPUs = */ _hstNumGPUs );

					// apply density map, and subtract from the real visibilities:
					for ( int gpu = 0; gpu < _hstNumGPUs; gpu++ )
						if (hstNextVisibility[ gpu ] > 0)
						{

							// set the cuda device.
							if (_hstNumGPUs > 1)
								cudaSetDevice( _hstGPU[ gpu ] );

							// define the block/thread dimensions.
							int threads = hstNextVisibility[ gpu ];
							int blocks;
							setThreadBlockSize1D( &threads, &blocks );
	
							// multiply all the visibilities by the value of the density map at that position.
							devApplyDensityMap<<< blocks, threads >>>(	/* pVisibilities = */ devModelVisibilities[ gpu ],
													/* pDensityMap = */ devDensityMap[ gpu ],
													/* pItems = */ hstNextVisibility[ gpu ] );

							// subtract the model visibilities from the real visibilities to get a new set of (dirty) visibilities.
							devSubtractVisibilities<<< blocks, threads >>>(	/* pOriginalVisibility = */ devOriginalVisibilities[ gpu ],
													/* pModelVisibility = */ devModelVisibilities[ gpu ],
													/* pItems = */ hstNextVisibility[ gpu ] );

						}
					if (_hstNumGPUs > 1)
						cudaSetDevice( _hstGPU[ 0 ] );

					// reset next visibility counters.
					for ( int gpu = 0; gpu < _hstNumGPUs; gpu++ )
						hstNextVisibility[ gpu ] = 0;
					
					// apply density map, and subtract from the real visibilities:
					cudaDeviceIndex = 0;
					for ( int kernelSet = 0; kernelSet < _hstKernelSets; kernelSet++ )
					{

						int lastGPU = cudaDeviceIndex;
						do
						{

							if (hstVisibilitiesInKernelSet[ kernelSet ][ cudaDeviceIndex ] > 0)
							{

								// set the cuda device.
								if (_hstNumGPUs > 1)
									cudaSetDevice( _hstGPU[ cudaDeviceIndex ] );

								// download model visibilities to the host.
								moveDeviceToHost( (void *) &_hstResidualVisibility[ hstCurrentVisibility ],
											(void *) &devModelVisibilities[ cudaDeviceIndex ]
																[ hstNextVisibility[ cudaDeviceIndex ] ],
											hstVisibilitiesInKernelSet[ kernelSet ][ cudaDeviceIndex ] * sizeof( cufftComplex ),
											"copying model visibilities to the host" );

								// get the next set of visibilities.
								hstCurrentVisibility += hstVisibilitiesInKernelSet[ kernelSet ][ cudaDeviceIndex ];
								hstNextVisibility[ cudaDeviceIndex ] += hstVisibilitiesInKernelSet[ kernelSet ][ cudaDeviceIndex ];

							} // hstVisibilitiesInKernelSet[ kernelSet ][ cudaDeviceIndex ] > 0
							cudaDeviceIndex++;
							if (cudaDeviceIndex == _hstNumGPUs)
								cudaDeviceIndex = 0;

						} while (cudaDeviceIndex != lastGPU);

					} // LOOP: kernelSet
					if (_hstNumGPUs > 1)
						cudaSetDevice( _hstGPU[ 0 ] );

					// free memory.
					if (hstNextVisibility != NULL)
						free( (void *) hstNextVisibility );

					// move to the next set of batch of data.
					batch = batch + 1;

				}

				// free memory.
				for ( int gpu = 0; gpu < _hstNumGPUs; gpu++ )
				{
					if (_hstNumGPUs > 1)
						cudaSetDevice( _hstGPU[ gpu ] );
					if (devModelVisibilities[ gpu ] != NULL)
						cudaFree( (void *) devModelVisibilities[ gpu ] );
					if (devDensityMap[ gpu ] != NULL)
						cudaFree( (void *) devDensityMap[ gpu ] );
					if (devWeight[ gpu ] != NULL)
						cudaFree( (void *) devWeight[ gpu ] );
					if (devOriginalVisibilities[ gpu ] != NULL)
						cudaFree( (void *) devOriginalVisibilities[ gpu ] );
					if (devGridPosition[ gpu ] != NULL)
						cudaFree( (void *) devGridPosition[ gpu ] );
					if (devKernelIndex[ gpu ] != NULL)
						cudaFree( (void *) devKernelIndex[ gpu ] );
				}
				if (_hstNumGPUs > 1)
					cudaSetDevice( _hstGPU[ 0 ] );
				if (devModelVisibilities != NULL)
					free( (void *) devModelVisibilities );
				if (devDensityMap != NULL)
					free( (void *) devDensityMap );
				if (devWeight != NULL)
					free( (void *) devWeight );
				if (devOriginalVisibilities != NULL)
					free( (void *) devOriginalVisibilities );
				if (devGridPosition != NULL)
					free( (void *) devGridPosition );
				if (devKernelIndex != NULL)
					free( (void *) devKernelIndex );

				// uncache the data for this mosaic.
				if (_hstCacheData == true)
				{
					cacheData(	/* pFilenamePrefix = */ pFilenamePrefix,
							/* pMosaicID = */ image,
							/* pBatchID = */ stageID,
							/* pWhatData = */ DATA_RESIDUAL_VISIBILITIES );
					freeData( /* pWhatData = */ DATA_ALL );
				}

			} // LOOP: stageID

			// re-create the dirty-image grid, if we're using image-plane mosaicing.
			if (_hstFileMosaic == true && image > 0)
				for ( int gpu = 0; gpu < _hstNumGPUs; gpu++ )
				{
				
					if (_hstNumGPUs > 1)
						cudaSetDevice( _hstGPU[ gpu ] );
					reserveGPUMemory( (void **) &devDirtyImageGrid[ gpu ], _hstUvPixels * _hstUvPixels * sizeof( cufftComplex ),
							"reserving device memory for dirty image grid (cleaning)" );
					zeroGPUMemory( devDirtyImageGrid[ gpu ], _hstUvPixels * _hstUvPixels * sizeof( cufftComplex ),
							"zeroing the dirty image grid on the device" );
				}
			if (_hstNumGPUs > 1)
				cudaSetDevice( _hstGPU[ 0 ] );

			// process all the stages.
			visibilitiesProcessed = 0;
			for ( int stageID = 0; stageID < _hstNumberOfStages[ image ]; stageID++ )
			{

				// uncache the data for this mosaic.
				if (_hstCacheData == true)
					uncacheData(	/* pFilenamePrefix = */ pFilenamePrefix,
							/* pMosaicID = */ image,
							/* pBatchID = */ stageID,
							/* pWhatData = */ DATA_GRID_POSITIONS | DATA_KERNEL_INDEXES | DATA_DENSITIES | DATA_WEIGHTS |
										DATA_RESIDUAL_VISIBILITIES,
							/* pOffset = */ 0 );

				// calculate the batch size.
				int hstVisibilityBatchSize = 0;
				{
					long int nextBatchSize = _hstNumVisibilities[ image ][ stageID ];
					if (nextBatchSize > _hstPreferredVisibilityBatchSize)
						nextBatchSize = _hstPreferredVisibilityBatchSize;
					hstVisibilityBatchSize = (int) nextBatchSize;
				}

				// variables for device memory.
				VectorI ** devGridPosition = (VectorI **) malloc( _hstNumGPUs * sizeof( VectorI ) );
				int ** devKernelIndex = (int **) malloc( _hstNumGPUs * sizeof( int * ) );
				int ** devDensityMap = (int **) malloc( _hstNumGPUs * sizeof( int * ) );
				cufftComplex ** devModelVisibilities = (cufftComplex **) malloc( _hstNumGPUs * sizeof( cufftComplex * ) );
				float ** devWeight = (float **) malloc( _hstNumGPUs * sizeof( float * ) );
				for ( int gpu = 0; gpu < _hstNumGPUs; gpu++ )
				{
					if (_hstNumGPUs > 1)
						cudaSetDevice( _hstGPU[ gpu ] );
					reserveGPUMemory( (void **) &devGridPosition[ gpu ], hstVisibilityBatchSize * sizeof( VectorI ),
								"reserving device memory for grid positions" );
					reserveGPUMemory( (void **) &devKernelIndex[ gpu ], hstVisibilityBatchSize * sizeof( int ),
								"reserving device memory for kernel indexes" );
					reserveGPUMemory( (void **) &devDensityMap[ gpu ], hstVisibilityBatchSize * sizeof( int ),
								"declaring device memory for density map" );
					reserveGPUMemory( (void **) &devModelVisibilities[ gpu ], hstVisibilityBatchSize * sizeof( cufftComplex ),
								"creating device memory for model visibilities" );
					if (_hstWeighting != NONE)
						reserveGPUMemory( (void **) &devWeight[ gpu ], hstVisibilityBatchSize * sizeof( float ),
								"creating device memory for weights" );
				}
				if (_hstNumGPUs > 1)
					cudaSetDevice( _hstGPU[ 0 ] );

				// here is the start of the visibility batch loop.
				int hstCurrentVisibility = 0;
				int batch = 0;
				while (hstCurrentVisibility < _hstNumVisibilities[ image ][ stageID ])
				{

					int ** hstVisibilitiesInKernelSet = _hstVisibilitiesInKernelSet[ image ][ stageID ][ batch ];

					// count the number of visibilities in this batch.
					int visibilitiesInThisBatch = 0;
					for ( int kernelSet = 0; kernelSet < _hstKernelSets; kernelSet++ )
						for ( int gpu = 0; gpu < _hstNumGPUs; gpu++ )
							visibilitiesInThisBatch += hstVisibilitiesInKernelSet[ kernelSet ][ gpu ];

					printf( "        gridding " );
					if (_hstNumberOfStages[ image ] == 1 && _hstNumberOfBatches[ image ][ stageID ] == 1)
						printf( "visibilities\n\n" );
					if (_hstNumberOfStages[ image ] > 1)
						printf( "host batch %i of %i", stageID + 1, _hstNumberOfStages[ image ] );
					if (_hstNumberOfStages[ image ] > 1 && _hstNumberOfBatches[ image ][ stageID ] > 1)
						printf( ", " );
					if (_hstNumberOfBatches[ image ][ stageID ] > 1)
						printf( "gpu batch %i of %i", batch + 1, _hstNumberOfBatches[ image ][ stageID ] );
					if (_hstNumberOfStages[ image ] > 1 || _hstNumberOfBatches[ image ][ stageID ] > 1)
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
					int * hstNextVisibility = (int *) malloc( _hstNumGPUs * sizeof( int ) );
					for ( int gpu = 0; gpu < _hstNumGPUs; gpu++ )
						hstNextVisibility[ gpu ] = 0;

					int cudaDeviceIndex = 0;
					for ( int kernelSet = 0; kernelSet < _hstKernelSets; kernelSet++ )
					{

						int lastGPU = cudaDeviceIndex;
						do
						{

							if (hstVisibilitiesInKernelSet[ kernelSet ][ cudaDeviceIndex ] > 0)
							{

								// set the cuda device, and make sure nothing is running there already.
								if (_hstNumGPUs > 1)
									cudaSetDevice( _hstGPU[ cudaDeviceIndex ] );

								// upload grid positions, kernel indexes, density map, weights, and original visibilities to the device.
								moveHostToDevice( (void *) &devGridPosition[ cudaDeviceIndex ][ hstNextVisibility[ cudaDeviceIndex ] ],
											(void *) &_hstGridPosition[ hstCurrentVisibility ],
											hstVisibilitiesInKernelSet[ kernelSet ][ cudaDeviceIndex ] * sizeof( VectorI ),
											"copying grid positions to the device" );
								moveHostToDevice( (void *) &devKernelIndex[ cudaDeviceIndex ][ hstNextVisibility[ cudaDeviceIndex ] ],
											(void *) &_hstKernelIndex[ hstCurrentVisibility ],
											hstVisibilitiesInKernelSet[ kernelSet ][ cudaDeviceIndex ] * sizeof( int ),
											"copying kernel indexes to the device" );
								moveHostToDevice( (void *) &devDensityMap[ cudaDeviceIndex ][ hstNextVisibility[ cudaDeviceIndex ] ],
											(void *) &_hstDensityMap[ hstCurrentVisibility ],
											hstVisibilitiesInKernelSet[ kernelSet ][ cudaDeviceIndex ] * sizeof( int ),
											"copying density map to the device" );
								if (_hstWeighting != NONE)
									moveHostToDevice( (void *) &devWeight[ cudaDeviceIndex ][ hstNextVisibility[ cudaDeviceIndex ] ],
												(void *) &_hstWeight[ hstCurrentVisibility ],
												hstVisibilitiesInKernelSet[ kernelSet ][ cudaDeviceIndex ] * sizeof( float ),
												"copying weights to the device" );
								moveHostToDevice( (void *) &devModelVisibilities[ cudaDeviceIndex ][ hstNextVisibility[ cudaDeviceIndex ] ],
											(void *) &_hstResidualVisibility[ hstCurrentVisibility ],
											hstVisibilitiesInKernelSet[ kernelSet ][ cudaDeviceIndex ] * sizeof( cufftComplex ),
											"copying original visibilities to the device" );

								// get the next set of visibilities.
								hstCurrentVisibility += hstVisibilitiesInKernelSet[ kernelSet ][ cudaDeviceIndex ];
								hstNextVisibility[ cudaDeviceIndex ] += hstVisibilitiesInKernelSet[ kernelSet ][ cudaDeviceIndex ];

							} // hstVisibilitiesInKernelSet[ kernelSet ][ cudaDeviceIndex ] > 0
							cudaDeviceIndex++;
							if (cudaDeviceIndex == _hstNumGPUs)
								cudaDeviceIndex = 0;

						} while (cudaDeviceIndex != lastGPU);

					} // LOOP: kernelSet
					if (_hstNumGPUs > 1)
						cudaSetDevice( _hstGPU[ 0 ] );

					// grid the new set of dirty visibilities.
					gridVisibilities(	/* pdevGrid = */ devDirtyImageGrid,
								/* pdevVisibility = */ devModelVisibilities,
								/* pOversample = */ _hstOversample,
								/* pKernelSize = */ _hstKernelSize,
								/* pSupport = */ _hstSupportSize,
								/* pdevKernelIndexes = */ devKernelIndex,
								/* pWProjection = */ _hstWProjection,
								/* pAProjection = */ _hstAProjection,
								/* pWPlanes = */ _hstWPlanes,
								/* pAPlanes = */ _hstAPlanes,
								/* pdevGridPositions = */ devGridPosition,
								/* pdevWeight = */ devWeight,
								/* pVisibilitiesInKernelSet = */ _hstVisibilitiesInKernelSet[ image ][ stageID ][ batch ],
								/* pGridDegrid = */ GRID,
								/* phstPrimaryBeamMosaicing = */
											(_hstBeamMosaic == true || _hstUVMosaic == true ? _hstPrimaryBeam[ image ] : NULL),
								/* phstPrimaryBeamAProjection = */ _hstPrimaryBeamAProjection[ image ],
								/* pNumFields = */ (_hstBeamMosaic == true ? _hstBeamMosaicComponents : -1),
								/* pMosaicIndex = */ image,
								/* pSize = */ _hstUvPixels,
								/* pNumGPUs = */ _hstNumGPUs );

					// move to the next set of batch of data.
					batch = batch + 1;

				} // WHILE: current-visibility < num-visibilities

				// free memory.
				for ( int gpu = 0; gpu < _hstNumGPUs; gpu++ )
				{
					if (_hstNumGPUs > 1)
						cudaSetDevice( _hstGPU[ gpu ] );
					if (devModelVisibilities[ gpu ] != NULL)
						cudaFree( (void *) devModelVisibilities[ gpu ] );
					if (devDensityMap[ gpu ] != NULL)
						cudaFree( (void *) devDensityMap[ gpu ] );
					if (devWeight[ gpu ] != NULL)
						cudaFree( (void *) devWeight[ gpu ] );
					if (devGridPosition[ gpu ] != NULL)
						cudaFree( (void *) devGridPosition[ gpu ] );
					if (devKernelIndex[ gpu ] != NULL)
						cudaFree( (void *) devKernelIndex[ gpu ] );
				}
				if (devModelVisibilities != NULL)
					free( (void *) devModelVisibilities );
				if (devDensityMap != NULL)
					free( (void *) devDensityMap );
				if (devWeight != NULL)
					free( (void *) devWeight );
				if (devGridPosition != NULL)
					free( (void *) devGridPosition );
				if (devKernelIndex != NULL)
					free( (void *) devKernelIndex );

				if (_hstNumGPUs > 1)
					cudaSetDevice( _hstGPU[ 0 ] );

				// free the data.
				if (_hstCacheData == true)
					freeData( /* pWhatData = */ DATA_GRID_POSITIONS | DATA_KERNEL_INDEXES | DATA_DENSITIES | DATA_WEIGHTS |
										DATA_RESIDUAL_VISIBILITIES );

			} // LOOP: stageID

			// free residual visibilities and model image if they are still there.
			if (_hstResidualVisibility != NULL)
			{
				free( (void *) _hstResidualVisibility );
				_hstResidualVisibility = NULL;
			}

			// we don't want to do normalisation if we're gridding multiple files as a UV mosaic, unless this is the last pass.
			if (_hstUVMosaic == false || image == _numMosaicImages - 1)
			{

				double normalisation = 1.0;
			
				// normalise by the number of visibilities, but only if we're not beam mosaicing. If we're using beam mosaicing then
				// the normalisation will have been done using the normalisation pattern.
				if (_hstBeamMosaic == true || _hstUVMosaic == true)
					normalisation *= (double) _griddedVisibilitiesForBeamMosaic;
				else
					normalisation *= (double) _hstGriddedVisibilities[ image ];

				if (_hstWeighting != NONE && _hstUVMosaic == false)
					normalisation *= _hstAverageWeight[ image ];
				if (_hstWeighting != NONE && _hstUVMosaic == true)
					normalisation *= _hstTotalAverageWeight;

				// move all images to the same GPU and add them together.
				if (_hstNumGPUs > 1)
				{

					cufftComplex * hstTmpImage = (cufftComplex *) malloc( _hstUvPixels * _hstUvPixels * sizeof( cufftComplex ) );
					cufftComplex * devTmpImage = NULL;
					reserveGPUMemory( (void **) &devTmpImage, _hstUvPixels * _hstUvPixels * sizeof( cufftComplex ),
									"reserving GPU memory for the temporary gridded data" );
				
					int threads = _hstUvPixels * _hstUvPixels, blocks = 1;
					setThreadBlockSize1D( &threads, &blocks );

					for ( int gpu = 1; gpu < _hstNumGPUs; gpu++ )
					{

						// set gpu device, and move image to the host.
						cudaSetDevice( _hstGPU[ gpu ] );
						moveDeviceToHost( (void *) hstTmpImage, devDirtyImageGrid[ gpu ], _hstUvPixels * _hstUvPixels * sizeof( cufftComplex ),
									"moving gridded data to the host" );
						cudaDeviceSynchronize();

						// set gpu device, and move image to the device.
						cudaSetDevice( _hstGPU[ 0 ] );
						moveHostToDevice( (void *) devTmpImage, (void *) hstTmpImage, _hstUvPixels * _hstUvPixels * sizeof( cufftComplex ),
									"moving gridded data to the device" );

						// add images together.
						devAddComplexData<<< blocks, threads >>>(	/* pOne = */ devDirtyImageGrid[ 0 ],
												/* pTwo = */ devTmpImage,
												/* pElements = */ _hstUvPixels * _hstUvPixels );

					}

					// free memory.
					if (hstTmpImage != NULL)
						free( (void *) hstTmpImage );
					if (devTmpImage != NULL)
						cudaFree( (void *) devTmpImage );

				}

				printf( "                performing FFT on gridded visibilities" );
				if (_hstFileMosaic == true)
					printf( " for mosaic component %i", image );
				printf( ".....\n" );

				// FFT the dirty visibilities into the image domain.
				performFFT(	/* pdevGrid = */ &devDirtyImageGrid[ 0 ],
						/* pSize = */ _hstUvPixels,
						/* pFFTDirection = */ INVERSE,
						/* pFFTPlan = */ -1,
						/* pFFTType = */ C2F );

				// recast the dirty image as a float.
				float * devDirtyImage = (float *) devDirtyImageGrid[ 0 ];

				// define the block/thread dimensions.
				setThreadBlockSize2D( _hstUvPixels, _hstUvPixels );

				// divide the dirty image by the deconvolution image.
				devDivideImages<<< _gridSize2D, _blockSize2D >>>(	/* pOne = */ devDirtyImage,
											/* pTwo = */ _devDeconvolutionImage,
											/* pMask = */ NULL,
											/* pSizeOne = */ _hstUvPixels,
											/* pSizeTwo = */ _hstPsfSize );

				// declare some device memory to hold the normalisation pattern.
				if (_hstNormalisationPattern != NULL && (_hstUVMosaic == true || _hstBeamMosaic == true))
				{

					float * devNormalisationPattern = NULL;
					reserveGPUMemory( (void **) &devNormalisationPattern, _hstBeamSize * _hstBeamSize * sizeof( float ),
								"declaring device memory for primary beam pattern" );

					// upload the primary beam pattern to the device.
					moveHostToDevice( (void *) devNormalisationPattern, (void *) _hstNormalisationPattern,
								_hstBeamSize * _hstBeamSize * sizeof( float ), "copying primary beam pattern to the host" );

					devDivideImages<<< _gridSize2D, _blockSize2D >>>(	/* pOne = */ devDirtyImage,
												/* pTwo = */ devNormalisationPattern,
												/* pMask = */ NULL,
												/* pSizeOne = */ _hstUvPixels,
												/* pSizeTwo = */ _hstBeamSize );

					// free memory.
					if (devNormalisationPattern != NULL)
						cudaFree( (void *) devNormalisationPattern );

				}

				int items = _hstUvPixels * _hstUvPixels;
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

					// normalise the image
					devNormalise<<< blocks, threads >>>( &devDirtyImage[ i * MAX_THREADS ], normalisation, itemsThisStage );

				}

				// create the dirty image.
				*phstDirtyImage = (float *) malloc( _hstUvPixels * _hstUvPixels * sizeof( float ) );

				// copy the residual image into the dirty image cache, or the dirty image if we're not mosaicing.
				if (_hstFileMosaic == true)
					moveDeviceToHost( (void *) &phstDirtyImageCache[ _hstUvPixels * _hstUvPixels * image ], (void *) devDirtyImage,
								_hstUvPixels * _hstUvPixels * sizeof( float ), "copying residual image to host" );
				else
					moveDeviceToHost( (void *) *phstDirtyImage, (void *) devDirtyImage, _hstUvPixels * _hstUvPixels * sizeof( float ),
								"copying residual image to host" );

				printf( "\n" );

				// free memory.
				if (devDirtyImageGrid != NULL)
				{
					for ( int gpu = 0; gpu < _hstNumGPUs; gpu++ )
						if (devDirtyImageGrid[ gpu ] != NULL)
						{
							if (_hstNumGPUs > 1)
								cudaSetDevice( _hstGPU[ gpu ] );
							cudaFree( (void *) devDirtyImageGrid[ gpu ] );
						}
					if (_hstNumGPUs > 1)
						cudaSetDevice( _hstGPU[ 0 ] );
					free( (void *) devDirtyImageGrid );
					devDirtyImageGrid = NULL;
				}

			} // (_hstUVMosaic == false || image == _numMosaicImages - 1)

		} // LOOP: image

		// free memory.
		if (devModelImage != NULL)
		{
			for ( int gpu = 0; gpu < _hstNumGPUs; gpu++ )
			{
				if (_hstNumGPUs > 1)
					cudaSetDevice( _hstGPU[ gpu ] );
				cudaFree( (void *) devModelImage[ gpu ] );
			}
			if (_hstNumGPUs > 1)
				cudaSetDevice( _hstGPU[ 0 ] );
			free( (void *) devModelImage );
			devModelImage = NULL;
		}

		// increment major cycle.
		majorCycle++;

	} // WHILE: reachedLimit == false

	// if we are making an image-plane mosaic, create a mosaic from the dirty (residual) images.
	if (_hstFileMosaic == true)
		createMosaic(	/* phstMosaic = */ *phstDirtyImage,
				/* phstImageArray = */ phstDirtyImageCache,
				/* phstMask = */ NULL,
				/* phstPhaseCentre = */ phstPhaseCentre,
				/* phstPrimaryBeamPatternPtr = */ &_hstPrimaryBeamPattern );
		
	// save the residual image.
	_hstCasacoreInterface.WriteCasaImage(	/* pFilename = */ pResidualImageFilename,
						/* pWidth = */ _hstUvPixels,
						/* pHeight = */ _hstUvPixels,
						/* pRA = */ _hstOutputRA,
						/* pDec = */ _hstOutputDEC,
						/* pPixelSize = */ _hstCellSize,
						/* pImage = */ *phstDirtyImage,
						/* pFrequency = */ CONST_C / _hstAverageWavelength[ 0 ],
						/* pMask = */ phstMask );

	// create memory for the clean image on the device.
	float * devCleanImage = NULL;
	reserveGPUMemory( (void **) &devCleanImage, _hstUvPixels * _hstUvPixels * sizeof( float ), "reserving device memory for the clean image" );

	// upload the component values to the device.
	double * devComponentValue = NULL;
	reserveGPUMemory( (void **) &devComponentValue, numComponents * sizeof( double ), "reserving device memory for clean components" );
	moveHostToDevice( (void *) devComponentValue, hstComponentListValue, numComponents * sizeof( double ), "moving component list values to the device" );

	// upload the grid positions to the device.
	VectorI * devComponentPos = NULL;
	reserveGPUMemory( (void **) &devComponentPos, numComponents * sizeof( VectorI ), "reserving device memory for clean component positions" );
	moveHostToDevice( (void *) devComponentPos, hstComponentListPos, numComponents * sizeof( VectorI ), "moving component list positions to the device" );

	// free memory.
	if (hstComponentListPos != NULL)
		free( (void *) hstComponentListPos );
	if (hstComponentListValue != NULL)
		free( (void *) hstComponentListValue );

	// upload the clean beam as a gridding kernel.
	int cleanBeamSize = (_hstCleanBeamSize * 2) + 1;
	float * devKernel = NULL;
	reserveGPUMemory( (void **) &devKernel, cleanBeamSize * cleanBeamSize * sizeof( float ),
				"reserving device memory for the clean component gridding kernel" );

	// cut out the centre portion of the kernel.
	for ( int i = 0; i < cleanBeamSize; i++ )
		cudaMemcpy(	&devKernel[ i * cleanBeamSize ],
				&pdevCleanBeam[ ((i + _hstPsfY - _hstCleanBeamSize) * _hstPsfSize) + _hstPsfX - _hstCleanBeamSize ],
				cleanBeamSize * sizeof( float ),
				cudaMemcpyDeviceToDevice );

//memset( *phstDirtyImage, 0, (long int) _hstUvPixels * (long int) _hstUvPixels * (long int) sizeof( float ) ); // cjs-mod

	// copy the dirty image to the device so that we include our residuals.
	moveHostToDevice( (void *) devCleanImage, (void *) *phstDirtyImage, _hstUvPixels * _hstUvPixels * sizeof( float ),
				"moving residual image to device" );

	// grid the clean components to make a clean image.
	gridComponents(	/* pdevGrid = */ devCleanImage,
			/* pdevComponentValue = */ devComponentValue,
			/* phstSupportSize = */ _hstCleanBeamSize,
			/* pdevKernel = */ devKernel,
			/* pdevGridPositions = */ devComponentPos,
			/* pComponents = */ numComponents,
			/* pSize = */ _hstUvPixels );

	// create host memory for the clean image and get the image from the device.
	moveDeviceToHost( (void *) *phstDirtyImage, (void *) devCleanImage, _hstUvPixels * _hstUvPixels * sizeof( float ),
				"moving clean image to the host" );

	// free memory.
	if (devKernel != NULL)
		cudaFree( (void *) devKernel );
	if (devCleanImage != NULL)
		cudaFree( (void *) devCleanImage );
	if (devComponentValue != NULL)
		cudaFree( (void *) devComponentValue );
	if (devComponentPos != NULL)
		cudaFree( (void *) devComponentPos );

	// save the clean image.
	_hstCasacoreInterface.WriteCasaImage(	/* pFilename = */ pCleanImageFilename,
						/* pWidth = */ _hstUvPixels,
						/* pHeight = */ _hstUvPixels,
						/* pRA = */ _hstOutputRA,
						/* pDec = */ _hstOutputDEC,
						/* pPixelSize = */ _hstCellSize,
						/* pImage = */ *phstDirtyImage,
						/* pFrequency = */ CONST_C / _hstAverageWavelength[ 0 ],
						/* pMask = */ phstMask );

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
	setThreadBlockSize2D( pSizeOfFittingRegion, pSizeOfFittingRegion );
		
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
		itemsToAdd = (int)ceil( (double) itemsToAdd / 10.0 );

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
//	generateDirtyBeam()
//
//	CJS: 05/11/2015
//
//	Generate the dirty beam by then FFT'ing the gridded data.
//

void generateDirtyBeam( cufftComplex ** pdevDirtyBeam, char * pFilename )
{

	printf( "        performing fft on psf grid.....\n" );
		
	// FFT the uv coverage to get the psf.
	performFFT(	/* pdevGrid = */ pdevDirtyBeam,
			/* pSize = */ _hstUvPixels,
			/* pFFTDirection = */ INVERSE,
			/* pFFTPlan = */ -1,
			/* pFFTType = */ C2F );
		
	// define the block/thread dimensions.
	setThreadBlockSize2D( _hstUvPixels, _hstUvPixels );
	
	// divide the dirty beam by the deconvolution image.
	devDivideImages<<< _gridSize2D, _blockSize2D >>>(	/* pOne = */ (float *) *pdevDirtyBeam,
								/* pTwo = */ _devDeconvolutionImage,
								/* pMask = */ NULL,
								/* pSizeOne = */ _hstUvPixels,
								/* pSizeTwo = */ _hstPsfSize );

	// chop out the central portion of the image.
	if (_hstPsfSize < _hstUvPixels)
	{

		float * devtmpDirtyBeam = NULL;
		reserveGPUMemory( (void **) &devtmpDirtyBeam, _hstPsfSize * _hstPsfSize * sizeof( float ), "reserving device memory for temporary psf" );

		// define the block/thread dimensions.
		setThreadBlockSize2D( _hstPsfSize, _hstPsfSize );

		// chop out the centre of the psf.
		devCopyImage<<< _gridSize2D, _blockSize2D >>>(	/* pNewImage = */ devtmpDirtyBeam,
								/* pOldImage = */ (float *) *pdevDirtyBeam,
								/* pNewSize = */ _hstPsfSize,
								/* pOldSize = */ _hstUvPixels,
								/* pScale = */ 1.0,
								/* pThreadOffset = */ 0 );

		// reassign the dirty beam to the new memory area.
		cudaFree( (void *) *pdevDirtyBeam );
		*pdevDirtyBeam = (cufftComplex *) devtmpDirtyBeam;

	}
			
	
	// get maximum pixel value.
	double * devMaxValue;
	reserveGPUMemory( (void **) &devMaxValue, MAX_PIXEL_DATA_AREA_SIZE * sizeof( double ), "declaring device memory for psf max pixel value" );
		
	// get the maximum complex value from this image.
	getMaxValue(	/* pdevImage = */ (float *) *pdevDirtyBeam,
			/* pdevMaxValue = */ devMaxValue,
			/* pWidth = */ _hstPsfSize,
			/* pHeight = */ _hstPsfSize,
			/* pdevMask = */ NULL );
	
	// set a suitable thread and block size.
	int threads = _hstPsfSize * _hstPsfSize;
	int blocks;
	setThreadBlockSize1D( &threads, &blocks );
		
	// normalise the psf so that the maximum value is 1.
	devNormalise<<< blocks, threads >>>( (float *) *pdevDirtyBeam, devMaxValue, _hstPsfSize * _hstPsfSize );

	// free memory.
	if (devMaxValue != NULL)
		cudaFree( devMaxValue );

	// create the dirty beam on the host, and copy to the host.
	cufftComplex * hstDirtyBeam = (cufftComplex *) malloc( _hstPsfSize * _hstPsfSize * sizeof( float ) );
	moveDeviceToHost( (void *) hstDirtyBeam, (void *) *pdevDirtyBeam, _hstPsfSize * _hstPsfSize * sizeof( float ),
				"copying dirty beam from device" );

	printf( "\n" );
	
	// save the dirty beam.
	_hstCasacoreInterface.WriteCasaImage(	/* pFilename = */ pFilename,
						/* pWidth = */ _hstPsfSize,
						/* pHeight = */ _hstPsfSize,
						/* pRA = */ _hstOutputRA,
						/* pDec = */ _hstOutputDEC,
						/* pPixelSize = */ _hstCellSize,
						/* pImage = */ (float *) hstDirtyBeam,
						/* pFrequency = */ CONST_C / _hstAverageWavelength[ 0 ],
						/* pMask = */ NULL );

	// free memory
	if (hstDirtyBeam != NULL)
		free( (void *) hstDirtyBeam );

} // generateDirtyBeam

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
	reserveGPUMemory( (void **) &devMaxValue, MAX_PIXEL_DATA_AREA_SIZE * sizeof( double ), "declaring device memory for kernel max pixel value" );

	// get the peak value from the kernel.
	getMaxValue(	/* pdevImage = */ pdevDirtyBeam,
			/* pdevMaxValue = */ devMaxValue,
			/* pWidth = */ _hstPsfSize,
			/* pHeight = */ _hstPsfSize,
			/* pdevMask = */ NULL );

	err = cudaGetLastError();
	if (err != cudaSuccess)
		printf( "unknown CUDA error on line %d (%s)\n", __LINE__, cudaGetErrorString( err ) );

	// find the closest pixel from the centre of the kernel that is negative.
	int closestNegativePixel = findCutoffPixel(	/* pdevKernel = */ pdevDirtyBeam,
							/* pdevMaxValue = */ devMaxValue,
							/* pSize = */ _hstPsfSize,
							/* pCutoffFraction = */ -1,
							/* pFindType = */ CLOSEST );

	// get the X and Y positions.of the maximum pixel.
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
	ok = ok && reserveGPUMemory( (void **) &devError, sizeOfFittingRegion * sizeOfFittingRegion * 2 * sizeof( double ), "creating device memory for Gaussian fit error" );
			
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
							/* pSize = */ _hstPsfSize,
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
														/* pSize = */ _hstPsfSize,
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
		setThreadBlockSize2D( _hstPsfSize, _hstPsfSize );

		// construct the clean beam on the device.
		devMakeCleanBeam<<< _gridSize2D, _blockSize2D >>>(	/* pCleanBeam = */ pdevCleanBeam,
									/* pAngle = */ (double) bestFit.angle,
									/* pR1 = */ (double) bestFit.r1,
									/* pR2 = */ (double) bestFit.r2,
									/* pX = */ (double) bestFit.x,
									/* pY = */ (double) bestFit.y,
									/* pSize = */ _hstPsfSize );
		err = cudaGetLastError();
		if (err != cudaSuccess)
		{
			printf( "error making the clean beam (%s)\n", cudaGetErrorString( err ) );
			ok = false;
		}

	}

	// make a note of the centre position of the psf - we'll need to for cleaning.
	_hstPsfX = (int) floor( bestFit.x + 0.5 ); _hstPsfY = (int) floor( bestFit.y + 0.5 );

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
		cudaFree( (void *)devError );

	// crate some host memory for the clean beam.
	float * hstCleanBeam = (float *) malloc( (long int) _hstPsfSize * (long int) _hstPsfSize * sizeof( float ) );

	// copy the clean beam to the host.
	ok = ok && moveDeviceToHost( (void *) hstCleanBeam, (void *) pdevCleanBeam, _hstPsfSize * _hstPsfSize * sizeof( float ),
					"copying clean beam from device" );

	// work out the size of the clean beam by finding the furthest pixel from the centre of the psf.
	_hstCleanBeamSize = 0;
	int pixel = 0;
	for ( int j = 0; j < _hstPsfSize; j++ )
		for ( int i = 0; i < _hstPsfSize; i++, pixel++ )
			if (hstCleanBeam[ pixel ] >= 0.000001)
			{
				if (abs( i - _hstPsfX ) > _hstCleanBeamSize)
					_hstCleanBeamSize = abs( i - _hstPsfX );
				if (abs( j - _hstPsfY ) > _hstCleanBeamSize)
					_hstCleanBeamSize = abs( j - _hstPsfY );
			}

	// save the clean beam.
	_hstCasacoreInterface.WriteCasaImage(	/* pFilename = */ pFilename,
						/* pWidth = */ _hstPsfSize,
						/* pHeight = */ _hstPsfSize,
						/* pRA = */ _hstOutputRA,
						/* pDec = */ _hstOutputDEC,
						/* pPixelSize = */ _hstCellSize,
						/* pImage = */ hstCleanBeam,
						/* pFrequency = */ CONST_C / _hstAverageWavelength[ 0 ],
						/* pMask = */ NULL );

	// free memory.
	if (hstCleanBeam != NULL)
		free( (void *) hstCleanBeam );
	
	// return success flag.
	return ok;
	
} // generateCleanBeam

//
//	compactData()
//
//	CJS: 15/10/2018
//
//	Compacts the visibility data so that items with a duplicate kernel index and grid position are only gridded once.
//

long int compactData( long int * pTotalVisibilities, long int pFirstVisibility, long int pNumVisibilitiesToProcess )
{

	// store the current visibility and weight in double format. the arrays are single precision, so we must store them separately.
	cufftDoubleComplex currentVis = { .x = 0.0, .y = 0.0 };
	double currentWeight = 0.0;

	// we ignore the field id column unless we're mosaicing, and then we need to preserve it.
	bool preserveFieldID = (_hstBeamMosaic == true && _hstFieldIDArray != NULL);

	// compact the array by looping through and adding up visibilities with the same kernel index and grid position.
	long int oldIndex = pFirstVisibility, newIndex = pFirstVisibility - 1;
	int lastKernelIndex = -1, lastFieldID = -1;
	VectorI lastPos = {-1, -1, -1};
	bool firstVis = true;
	while (oldIndex < pFirstVisibility + pNumVisibilitiesToProcess)
	{

		// ensure this visibility isn't flagged.
		bool flagged = false;
		if (_hstFlag != NULL)
			flagged = _hstFlag[ oldIndex ];
		if (flagged == false)
		{

			bool merge = false;
			if (_hstGridPosition[ oldIndex ].u == lastPos.u)
				if (_hstGridPosition[ oldIndex ].v == lastPos.v)
					if (_hstGridPosition[ oldIndex ].w == lastPos.w)
						if (_hstKernelIndex[ oldIndex ] == lastKernelIndex)
						{
							if (preserveFieldID == true)
								merge = (_hstFieldIDArray[ oldIndex ] == lastFieldID);
							else
								merge = true;
						}

			if (merge == false)
			{

				// store the last visibility and weight, if needed.
				if (firstVis == false)
				{
					if (_hstWeighting != NONE)
					{
						currentWeight /= (double) _hstDensityMap[ newIndex ];
						_hstWeight[ newIndex ] = (float) currentWeight;
						if (currentWeight != 0.0)
							currentVis = divideComplex( /* pOne = */ currentVis, /* pTwo = */ currentWeight );
						else
						{
							currentVis.x = 0.0;
							currentVis.y = 0.0;
						}
					}
					_hstVisibility[ newIndex ].x = (float) currentVis.x;
					_hstVisibility[ newIndex ].y = (float) currentVis.y;
				}
				firstVis = false;

				newIndex = newIndex + 1;

				// copy the data into the new array elements.
				_hstKernelIndex[ newIndex ] = _hstKernelIndex[ oldIndex ];
				_hstGridPosition[ newIndex ] = _hstGridPosition[ oldIndex ];
				currentVis.x = (double) _hstVisibility[ oldIndex ].x;
				currentVis.y = (double) _hstVisibility[ oldIndex ].y;
				if (_hstWeighting != NONE)
				{
					currentVis = multComplex( /* pOne = */ currentVis, /* pTwo = */ (double) _hstWeight[ oldIndex ] );
					currentWeight = ((double) _hstWeight[ oldIndex ] * (double) _hstDensityMap[ oldIndex ]);
				}
				_hstDensityMap[ newIndex ] = _hstDensityMap[ oldIndex ];
				if (_hstFlag != NULL)
					_hstFlag[ newIndex ] = false;
				if (preserveFieldID == true)
					_hstFieldIDArray[ newIndex ] = _hstFieldIDArray[ oldIndex ];

				// store the current kernel indexes, field ID and grid positions
				lastKernelIndex = _hstKernelIndex[ oldIndex ];
				lastPos = _hstGridPosition[ oldIndex ];
				if (preserveFieldID == true)
					lastFieldID = _hstFieldIDArray[ oldIndex ];

			}
			else
			{

				// add this visibility to the new index, along with the weight if required.
				if (_hstWeighting != NONE)
				{
					currentVis.x += ((double) _hstVisibility[ oldIndex ].x * (double) _hstWeight[ oldIndex ]);
					currentVis.y += ((double) _hstVisibility[ oldIndex ].y * (double) _hstWeight[ oldIndex ]);
					currentWeight += ((double) _hstWeight[ oldIndex ] * (double) _hstDensityMap[ oldIndex ] );
				}
				else
				{
					currentVis.x += (double) _hstVisibility[ oldIndex ].x;
					currentVis.y += (double) _hstVisibility[ oldIndex ].y;
				}

				// add up the density map at this position.
				_hstDensityMap[ newIndex ] += _hstDensityMap[ oldIndex ];

			}

		}

		// get next item in array.
		oldIndex = oldIndex + 1;

	}

	// store the last visibility and weight, if needed.
	if (firstVis == false)
	{
		if (_hstWeighting != NONE)
		{
			currentWeight /= (double) _hstDensityMap[ newIndex ];
			_hstWeight[ newIndex ] = (float) currentWeight;
			if (currentWeight != 0.0)
				currentVis = divideComplex( /* pOne = */ currentVis, /* pTwo = */ currentWeight );
			else
			{
				currentVis.x = 0.0;
				currentVis.y = 0.0;
			}
		}
		_hstVisibility[ newIndex ].x = (float) currentVis.x;
		_hstVisibility[ newIndex ].y = (float) currentVis.y;
	}

	int compactedVisibilities = newIndex - pFirstVisibility + 1;

	// have we shrunk the array ?
	if (compactedVisibilities < pNumVisibilitiesToProcess)
	{

		// move visibilities, A-planes, flags, samples, and channels to earlier in the memory.
		if (*pTotalVisibilities > pFirstVisibility + pNumVisibilitiesToProcess)
		{
			long int moveFrom = pFirstVisibility + pNumVisibilitiesToProcess;
			long int moveTo = newIndex + 1;
			long int moveNumber = *pTotalVisibilities - moveFrom;
			memmove( (void *) &_hstVisibility[ moveTo ], (void *) &_hstVisibility[ moveFrom ], moveNumber * (long int) sizeof( cufftComplex ) );
			if (_hstFlag != NULL)
				memmove( (void *) &_hstFlag[ moveTo ], (void *) &_hstFlag[ moveFrom ], moveNumber * (long int) sizeof( bool ) );
			if (_hstSampleID != NULL)
				memmove( (void *) &_hstSampleID[ moveTo ], (void *) &_hstSampleID[ moveFrom ], moveNumber * (long int) sizeof( int ) );
			if (_hstChannelID != NULL)
				memmove( (void *) &_hstChannelID[ moveTo ], (void *) &_hstChannelID[ moveFrom ], moveNumber * (long int) sizeof( int ) );
		}

		// update the total number of visibilities.
		*pTotalVisibilities -= (pNumVisibilitiesToProcess - compactedVisibilities);

		// compact the arrays.
		_hstVisibility = (cufftComplex *) realloc( _hstVisibility, *pTotalVisibilities * sizeof( cufftComplex ) );
		if (_hstFlag != NULL)
			_hstFlag = (bool *) realloc( _hstFlag, *pTotalVisibilities * sizeof( bool ) );
		if (_hstSampleID != NULL)
			_hstSampleID = (int *) realloc( _hstSampleID, *pTotalVisibilities * sizeof( int ) );
		if (_hstChannelID != NULL)
			_hstChannelID = (int *) realloc( _hstChannelID, *pTotalVisibilities * sizeof( int ) );
		_hstGridPosition = (VectorI *) realloc( _hstGridPosition, *pTotalVisibilities * sizeof( VectorI ) );
		_hstKernelIndex = (int *) realloc( _hstKernelIndex, *pTotalVisibilities * sizeof( int ) );
		_hstDensityMap = (int *) realloc( _hstDensityMap, *pTotalVisibilities * sizeof( int ) );
		_hstWeight = (float *) realloc( _hstWeight, *pTotalVisibilities * sizeof( float ) );
		if (_hstFieldIDArray != NULL)
			_hstFieldIDArray = (int *) realloc( _hstFieldIDArray, *pTotalVisibilities * sizeof( int ) );

	}

	// return the index of the next batch of data.
	return newIndex + 1;

} // compactData

//
//	generatePrimaryBeamAiry()
//
//	CJS: 01/011/2019
//
//	Generates a primary beam from an Airy disk with a blockage at the centre.
//

void generatePrimaryBeamAiry( float ** phstPrimaryBeamIn, double pWidth, double pCutout, double pWavelength )
{

	const double PIXELS_PER_METRE = 8.0;

	printf( "generating primary beam using an Airy disk from a uniformly illuminated %4.2f m aperture dish with a %4.2f m blockage.....", pWidth, pCutout );
	fflush( stdout );

	// simulate an airy disk primary beam. we use 6x the required beam size to generate the beam, but then chop it down by a factor of 6 later.
	int hstPrimaryBeamSupport = BEAM_SIZE * 3;
	_hstBeamSize = (hstPrimaryBeamSupport * 2);				// in pixels

	// calculate the image-plane pixel size of the primary beam.
	double uvPixelSize = 1.0 / PIXELS_PER_METRE; 					// in metres
	double uvPixelSizeInLambda = uvPixelSize / pWavelength;			// in units of lambda
	double imFieldOfView = (180.0 * 3600.0 / PI) * (1.0 / uvPixelSizeInLambda);	// in arcsec
	_hstBeamCellSize = imFieldOfView / _hstBeamSize;

	printf( "pixel size %6.4f arcsec.....", _hstBeamCellSize );
	fflush( stdout );

	*phstPrimaryBeamIn = (float *) malloc( _hstBeamSize * _hstBeamSize * sizeof( float ) );

	// calculate the max and min radius using the width and blockage. we divide by two to get the radius, and square (so we don't need to keep taking square roots).
	double maxRadius = pow( pWidth * PIXELS_PER_METRE / 2.0, 2 );
	double minRadius = pow( pCutout * PIXELS_PER_METRE / 2.0, 2 );

	long int beamPtr = 0;
	for ( int j = 0; j < _hstBeamSize; j++ )
		for ( int i = 0; i < _hstBeamSize; i++, beamPtr++ )
		{
			double r = (double) (pow( i - hstPrimaryBeamSupport, 2 ) + pow( j - hstPrimaryBeamSupport, 2 ));
			(*phstPrimaryBeamIn)[ beamPtr ] = ( r >= minRadius && r <= maxRadius ? 1.0 : 0.0 );
		}

	printf( "done\n\n" );

	// copy beam to device.
	float * devBeam = NULL;
	reserveGPUMemory( (void **) &devBeam, _hstBeamSize * _hstBeamSize * sizeof( float ), "reserving device memory for primary beam" );
	moveHostToDevice( (void *) devBeam, (void *) *phstPrimaryBeamIn, _hstBeamSize * _hstBeamSize * sizeof( float ),
				"copying primary beam to device" );

	// FFT the primary beam into the uv domain.
	performFFT(	/* pdevGrid = */ (cufftComplex **) &devBeam,
			/* pSize = */ _hstBeamSize,
			/* pFFTDirection = */ FORWARD,
			/* pFFTPlan = */ -1,
			/* pFFTType = */ F2F );

	// define the block/thread dimensions.
	setThreadBlockSize2D( _hstBeamSize, _hstBeamSize );

	// get the intensity of the primary beam by taking the square.
	devMultiplyImages<<< _gridSize2D, _blockSize2D >>>(	/* pOne = */ devBeam,
								/* pTwo = */ devBeam,
								/* pSize = */ _hstBeamSize );

	// get the maximum value from the beam. create a new memory area to hold the maximum pixel value.
	double * devMaxValue;
	reserveGPUMemory( (void **) &devMaxValue, MAX_PIXEL_DATA_AREA_SIZE * sizeof( double ), "declaring device memory for kernel max pixel value" );

	// get the peak value from the kernel.
	getMaxValue(	/* pdevImage = */ devBeam,
			/* pdevMaxValue = */ devMaxValue,
			/* pWidth = */ _hstBeamSize,
			/* pHeight = */ _hstBeamSize,
			/* pdevMask = */ NULL );

	// get the maximum value, and its position.
	double maxValue = 0.0;
	cudaMemcpy( &maxValue, &devMaxValue[ MAX_PIXEL_VALUE ], sizeof( double ), cudaMemcpyDeviceToHost );

	// free the max value memory area.
	if (devMaxValue != NULL)
		cudaFree( (void *) devMaxValue );

	// define the block/thread dimensions.
	int threads = _hstBeamSize * _hstBeamSize;
	int blocks;
	setThreadBlockSize1D( &threads, &blocks );

	// normalise the image
	devNormalise<<< blocks, threads >>>(	/* pArray = */ devBeam,
						/* pConstant = */ maxValue,
						/* pItems = */ _hstBeamSize * _hstBeamSize );

	// reduce the size of the primary beam.
	_hstBeamSize = _hstBeamSize / 6;

	// reduce the size by a factor of 6.
	threads = _hstBeamSize * _hstBeamSize;
	setThreadBlockSize1D( &threads, &blocks );

	// move the centre of the image into the first part of the image.
	devMoveToStartOfImage<<< blocks, threads >>>(	/* pImage = */ devBeam,
							/* pInitialSize = */ _hstBeamSize * 6,
							/* pFinalSize = */ _hstBeamSize );

	// copy primary beam back to host.
	cudaMemcpy( (void *) *phstPrimaryBeamIn, (void *) devBeam, _hstBeamSize * _hstBeamSize * sizeof( float ), cudaMemcpyDeviceToHost );

	// free device memory.
	cudaFree( (void *) devBeam );

} // generatePrimaryBeamAiry

//
//	loadPrimaryBeam()
//
//	CJS: 26/10/2020
//
//	Loads the primary beam from a csv file.
//

void loadPrimaryBeam( char * pBeamFilename, float ** phstPrimaryBeamIn, int pSize )
{

	char * fileData = NULL;

	printf( "loading primary beam %s\n\n", pBeamFilename );
	printf( "        beam size %i x %i pixels, beam cell size %6.2f arcsec/pixel\n\n", _hstBeamSize, _hstBeamSize, _hstBeamCellSize );

	// create some memory for the beam, and set it to zero.
	(*phstPrimaryBeamIn) = (float *) malloc( pSize * pSize * sizeof( float ) );
	memset( (*phstPrimaryBeamIn), 0.0, pSize * pSize * sizeof( float ) );

	// find the maximum pixel value for this beam.
	double maxValue = 0.0;

	// open the file.
	FILE * beamFile = fopen( pBeamFilename, "rt" );

	// get the file size.
	fseek( beamFile, 0L, SEEK_END );
	int fileSize = ftell( beamFile );

	// rewind the file.
	rewind( beamFile );

	if (fileSize > 0)
	{

		// create some memory to hold the contents of this file.
		fileData = (char *) malloc( fileSize );

		// get the contents of the file.
		fgets( fileData, fileSize, beamFile );

		// loop over the expected number of pixels.
		char * current = fileData, * next;
		for ( int j = 0; j < pSize; j++ )
			for ( int i = 0; i < pSize; i++ )
			{
				(*phstPrimaryBeamIn)[ (j * pSize) + i ] = strtod( current, &next );
				if ((*phstPrimaryBeamIn)[ (j * pSize) + i ] > maxValue)
					maxValue = (*phstPrimaryBeamIn)[ (j * pSize) + i ];
				current = next;
			}
			
		// free the memory.
		free( fileData );

	}

	// close the file
	fclose( beamFile );

	// normalise the beam.
	if (maxValue > 0.0)
		for ( int arrayIndex = 0; arrayIndex < pSize * pSize; arrayIndex++ )
			(*phstPrimaryBeamIn)[ arrayIndex ] /= maxValue;

} // loadPrimaryBeam

//
//	imagePlaneReprojectPrimaryBeam()
//
//	CJS: 30/11/2018
//
//	Reproject a primary beam to a different phase position.
//

void imagePlaneReprojectPrimaryBeam( float * phstPrimaryBeamIn, float * phstPrimaryBeamOut, int pBeam, double pInRA, double pInDec,
					double pOutRA, double pOutDec, float * pdevInBeam, float * pdevOutBeam, double pBeamCellSize,
					Reprojection & pImagePlaneReprojection, bool pVerbose )
{

	// set up input pixel vector and CD matrix.
	Reprojection::rpVectD tmpPixelIn = { /* x = */ _hstBeamSize / 2, /* y = */ _hstBeamSize / 2, /*z = */ 0 };
	Reprojection::rpMatr2x2 tmpCDIn = {	/* a11 = */ -sin( rad( pBeamCellSize / 3600.0 ) ),
						/* a12 = */ 0.0,
						/* a21 = */ 0.0,
						/* a22 = */ sin( rad( pBeamCellSize / 3600.0 ) ) };

	// set up output pixel vector and CD matrix.
	Reprojection::rpVectD tmpPixelOut = { /* x = */ _hstBeamSize / 2, /* y = */ _hstBeamSize / 2, /*z = */ 0 };
	Reprojection::rpMatr2x2 tmpCDOut = {	/* a11 = */ -sin( rad( (_hstCellSize / 3600.0) * (double) _hstUvPixels / (double) _hstBeamSize ) ),
						/* a12 = */ 0.0,
						/* a21 = */ 0.0,
						/* a22 = */ sin( rad( (_hstCellSize / 3600.0) * (double) _hstUvPixels / (double) _hstBeamSize ) ) };

	// build input and output size.
	Reprojection::rpVectI inSize = { /* x = */ _hstBeamSize, /* y = */ _hstBeamSize };
	Reprojection::rpVectI outSize = { /* x = */ _hstBeamSize, /* y = */ _hstBeamSize };

	// build beam size.
	Reprojection::rpVectI beamSize = { /* x = */ _hstBeamSize, /* y = */ _hstBeamSize };

	// build in coordinate system.
	Reprojection::rpCoordSys inCoordSystem;
	inCoordSystem.crVAL.x = pInRA;
	inCoordSystem.crVAL.y = pInDec;
	inCoordSystem.crPIX = tmpPixelIn;
	inCoordSystem.cd = tmpCDIn;
	inCoordSystem.epoch = Reprojection::EPOCH_J2000;

	// build out coordinate system.
	Reprojection::rpCoordSys outCoordSystem;
	outCoordSystem.crVAL.x = pOutRA;
	outCoordSystem.crVAL.y = pOutDec;
	outCoordSystem.crPIX = tmpPixelOut;
	outCoordSystem.cd = tmpCDOut;
	outCoordSystem.epoch = Reprojection::EPOCH_J2000;

	// copy the primary beam into a temporary work location.
	cudaMemcpy( pdevInBeam, phstPrimaryBeamIn, _hstBeamSize * _hstBeamSize * sizeof( float ), cudaMemcpyHostToDevice );

	// clear the output image.
	zeroGPUMemory( (void *) pdevOutBeam, _hstBeamSize * _hstBeamSize * sizeof( float ),
				"zeroing the reprojected output image on the device" );

	// reproject this image.
	pImagePlaneReprojection.ReprojectImage(	/* pdevInImage = */ pdevInBeam,
						/* pdevOutImage = */ pdevOutBeam,
						/* pdevNormalisationPattern = */ NULL,
						/* pdevPrimaryBeamPattern = */ NULL,
						/* pInCoordinateSystem = */ inCoordSystem,
						/* pOutCoordinateSystem = */ outCoordSystem,
						/* pInSize = */ inSize,
						/* pOutSize = */ outSize,
						/* pdevInMask = */ NULL,
						/* pdevBeamIn = */ NULL,
						/* pdevBeamOut = */ NULL,
						/* pBeamSize = */ beamSize,
						/* pProjectionDirection = */ Reprojection::OUTPUT_TO_INPUT,
						/* pAProjection = */ false,
						/* pVerbose = */ pVerbose );

	// store the beam on the host.
	cudaMemcpy( phstPrimaryBeamOut, pdevOutBeam, _hstBeamSize * _hstBeamSize * sizeof( float ), cudaMemcpyDeviceToHost );

	// cut off any value less than 0.1% to zero.
	for ( int i = 0; i < _hstBeamSize * _hstBeamSize; i++ )
		if (abs( phstPrimaryBeamOut[ i ] ) < 0.001)
			phstPrimaryBeamOut[ i ] = 0.0;

} // imagePlaneReprojectPrimaryBeam

//
//	compactFieldIDs()
//
//	CJS: 30/11/2018
//
//	We may only be using a few fields amongst the many in the MS, so we want to renumber them as 0,1,2,3,etc, instead of e.g. 3,7,9,11,23,etc.
//

void compactFieldIDs( double ** pPhaseCentrePtr, double ** pPhaseCentreImagePtr, int * pNumFields, int * pFieldID, int ** pFieldIDMap, int pNumSamples )
{

	// create an array of field IDs, and copy the real field IDs.
	int * tmpFieldID = (int *) malloc( pNumSamples * sizeof( int ) );
	memcpy( tmpFieldID, pFieldID, pNumSamples * sizeof( int ) );

	// sort the field ids.
	quickSortFieldIDs(	/* pFieldID = */ tmpFieldID,
				/* pLeft = */ 0,
				/* pRight = */ pNumSamples - 1 );

	// reduce the field ids to their unique values.
	int currentIndex = 0, newIndex = -1;
	int lastValue = -1;
	while (currentIndex < pNumSamples)
	{
		if (tmpFieldID[ currentIndex ] != lastValue)
		{
			newIndex = newIndex + 1;
			tmpFieldID[ newIndex ] = tmpFieldID[ currentIndex ];
			lastValue = tmpFieldID[ currentIndex ];
		}
		currentIndex = currentIndex + 1;
	}

	// update the number of fields to reflect the number that are actually found in our data.
	*pNumFields = newIndex + 1;

	// create the map.
	(*pFieldIDMap) = (int *) malloc( (*pNumFields) * sizeof( int ) );
	memcpy( (*pFieldIDMap), tmpFieldID, (*pNumFields) * sizeof( int ) );

	printf( "%i unique field ID(s) found in these data:\n        ", (*pNumFields) );
	for ( int i = 0; i < (*pNumFields); i++ )
	{
		if (i > 0)
			printf( "," );
		printf( "%i", tmpFieldID[ i ] );
	}
	printf( "\n\n" );

	// update the phase centres so that they correspond to the fields in the reduced list.
	double * tmpPhaseCentre = (double *) malloc( 2 * (*pNumFields) * sizeof( double ) );

	// update the phase centre in the temporary array, and then copy the temporary array into the original array.
	for ( int i = 0; i < (*pNumFields); i++ )
	{
		tmpPhaseCentre[ 2 * i ] = (*pPhaseCentrePtr)[ 2 * tmpFieldID[ i ] ];
		tmpPhaseCentre[ (2 * i) + 1 ] = (*pPhaseCentrePtr)[ (2 * tmpFieldID[ i ]) + 1 ];
	}
	*pPhaseCentrePtr = (double *) realloc( (*pPhaseCentrePtr), 2 * (*pNumFields) * sizeof( double ) );
	memcpy( *pPhaseCentrePtr, tmpPhaseCentre, 2 * (*pNumFields) * sizeof( double ) );

	// update the image phase centre in the temporary array, and then copy the temporary array into the original array.
	for ( int i = 0; i < (*pNumFields); i++ )
	{
		tmpPhaseCentre[ 2 * i ] = (*pPhaseCentreImagePtr)[ 2 * tmpFieldID[ i ] ];
		tmpPhaseCentre[ (2 * i) + 1 ] = (*pPhaseCentreImagePtr)[ (2 * tmpFieldID[ i ]) + 1 ];
	}
	*pPhaseCentreImagePtr = (double *) realloc( (*pPhaseCentreImagePtr), 2 * (*pNumFields) * sizeof( double ) );
	memcpy( *pPhaseCentreImagePtr, tmpPhaseCentre, 2 * (*pNumFields) * sizeof( double ) );

	// free data.
	if (tmpPhaseCentre != NULL)
		free( (void *) tmpPhaseCentre );
	if (tmpFieldID != NULL)
		free( (void *) tmpFieldID );

} // compactFieldIDs

//
//	parseChannelRange()
//
//	CJS: 26/11/2019
//
//	Parses a range of channels and works out which channels should be included.
//

void parseChannelRange( char * pChannelRange, int pNumChannels, bool * phstSpwChannelFlag )
{

	int posCharIn = -1;
	int channelFrom = -1, channelTo = -1;

	// look for a twiddle (~) which separates the from channel from the to channel.
	char * twiddle = strchr( pChannelRange, (int) '~' );
	if (twiddle != NULL)
		posCharIn = twiddle - pChannelRange;
	else
	{

		// get the channel index and update this channel to unflagged.
		channelFrom = atoi( pChannelRange );
		channelTo = channelFrom;

	}

	// get the from channel index.
	char channelChar[ 16 ] = "\0";
	if (posCharIn > 0)
	{

		// copy channel text into a new string, and covert to an integer.
		strncpy( channelChar, pChannelRange, posCharIn );
		channelFrom = atoi( channelChar );

	}

	// get the to channel index.
	posCharIn++;
	if (posCharIn >= 1 && posCharIn < strlen( pChannelRange ))
	{

		// copy channel text into a new string, and covert to an integer.
		strncpy( channelChar, &pChannelRange[ posCharIn ], strlen( pChannelRange ) - posCharIn );
		channelTo = atoi( channelChar );

	}

	// ensure channel from and to are within the range for this spw.
	if (channelFrom < 0)
		channelFrom = 0;
	if (channelFrom >= pNumChannels)
		channelFrom = pNumChannels - 1;
	if (channelTo < 0)
		channelTo = 0;
	if (channelTo >= pNumChannels)
		channelTo = pNumChannels - 1;

	// update flags to false for this channel range.
	if (channelTo >= channelFrom)
		for ( int channel = channelFrom; channel <= channelTo; channel++ )
			phstSpwChannelFlag[ channel ] = false;

} // parseChannelRange

//
//	parseSpwSpecifier()
//
//	CJS: 26/11/2019
//
//	Parses a single SPW specifier and works out which channels should be included.
//

void parseSpwSpecifier( char * pSpwSpecifier, int pNumSpws, int * phstNumChannels, bool ** phstSpwChannelFlag )
{

	int spw = -1;
	int posCharIn = 0;

	// look for a colon which separates the spw index from the channel list.
	char * colon = strchr( pSpwSpecifier, (int) ':' );
	if (colon != NULL)
		posCharIn = colon - pSpwSpecifier;
	else
	{

		// get the spw index and update all these channels to unflagged.
		spw = atoi( pSpwSpecifier );
		if (spw >= 0 && spw < pNumSpws)
			for ( int channel = 0; channel < phstNumChannels[ spw ]; channel++ )
				phstSpwChannelFlag[ spw ][ channel ] = false;

	}

	// we need to update the flag only for specific channels. get the spw index.
	char spwChar[ 16 ] = "\0";
	if (posCharIn > 0)
	{

		// copy spw text into a new string, and covert to an integer.
		strncpy( spwChar, pSpwSpecifier, posCharIn );
		spw = atoi( spwChar );

		// now parse the channel list.
		posCharIn++;

		// initialise an empty string.
		char channelRange[ 1024 ];
		int numCharOut = 0;

		// loop over all the characters in the spw array.
		while (numCharOut < strlen( pSpwSpecifier ))
		{

			// check for a channel list separator (i.e. semicolon).
			if (pSpwSpecifier[ posCharIn ] == ';')
			{
			
				// add a '\0' to the end of this string.
				channelRange[ numCharOut ] = '\0';

				// parse the channel range.
				if (numCharOut > 0)
					parseChannelRange(	/* pChannelRange = */ channelRange,
								/* pNumChannels = */ phstNumChannels[ spw ],
								/* phstSpwChannelFlag = */ phstSpwChannelFlag[ spw ] );

				// reset length of output string.
				numCharOut = 0;

			}
			else if (pSpwSpecifier[ posCharIn ] != ' ')
			{
				channelRange[ numCharOut ] = pSpwSpecifier[ posCharIn ];
				numCharOut++;
			}

			// increment character count.
			posCharIn++;

		}

		// parse the current channel range.
		if (numCharOut > 0)
		{
			
			// add a '\0' to the end of this string.
			channelRange[ numCharOut ] = '\0';

			// parse the spw specifier.
			if (numCharOut > 0)
				parseChannelRange(	/* pChannelRange = */ channelRange,
							/* pNumChannels = */ phstNumChannels[ spw ],
							/* phstSpwChannelFlag = */ phstSpwChannelFlag[ spw ] );

		}

	}

} // parseSpwSpecifier

//
//	setSpwAndChannelFlags()
//
//	CJS: 26/11/2019
//
//	Parses the SPW parameter and sets a flag to determine which spws and channels should be included in the image.
//

void setSpwAndChannelFlags( int pNumSpws, int * phstNumChannels, bool *** phstSpwChannelFlag, char * phstSpwRestriction )
{

	// create memory for the flags and set it all to true.
	*phstSpwChannelFlag = (bool **) malloc( pNumSpws * sizeof( bool * ) );
	for ( int spw = 0; spw < pNumSpws; spw++ )
	{
		(*phstSpwChannelFlag)[ spw ] = (bool *) malloc( phstNumChannels[ spw ] * sizeof( bool ) );
		for ( int channel = 0; channel < phstNumChannels[ spw ]; channel++ )
			if (phstSpwRestriction[ 0 ] == '\0')
				(*phstSpwChannelFlag)[ spw ][ channel ] = false;
			else
				(*phstSpwChannelFlag)[ spw ][ channel ] = true;
	}

	// initialise an empty string.
	char singleSpw[ 1024 ];
	int posCharIn = 0, numCharOut = 0;

	// loop over all the characters in the spw array.
	while (numCharOut < strlen( phstSpwRestriction ))
	{

		// check for a spw separator (i.e. comma).
		if (phstSpwRestriction[ posCharIn ] == ',')
		{
			
			// add a '\0' to the end of this string.
			singleSpw[ numCharOut ] = '\0';

			// parse the spw specifier.
			if (numCharOut > 0)
				parseSpwSpecifier(	/* pSpwSpecifier = */ singleSpw,
							/* pNumSpws = */ pNumSpws,
							/* phstNumChannels = */ phstNumChannels,
							/* phstSpwChannelFlag = */ *phstSpwChannelFlag );

			// reset length of output string.
			numCharOut = 0;

		}
		else if (phstSpwRestriction[ posCharIn ] != ' ')
		{
			singleSpw[ numCharOut ] = phstSpwRestriction[ posCharIn ];
			numCharOut++;
		}

		// increment character count.
		posCharIn++;

	}

	// parse the current spw specifier.
	if (numCharOut > 0)
		if (singleSpw[ 0 ] != '\0')
		{
			
			// add a '\0' to the end of this string.
			singleSpw[ numCharOut ] = '\0';

			// parse the spw specifier.
			if (numCharOut > 0)
				parseSpwSpecifier(	/* pSpwSpecifier = */ singleSpw,
							/* pNumSpws = */ pNumSpws,
							/* phstNumChannels = */ phstNumChannels,
							/* phstSpwChannelFlag = */ *phstSpwChannelFlag );

		}

} // setSpwAndChannelFlags

//
//	getSuitablePhasePositionForBeam()
//
//	CJS: 03/02/2020
//
//	Works out a suitable phase position for gridding based upon a primary beam at one position and a required image at another position.
//

void getSuitablePhasePositionForBeam( double * pBeamIn, double * pPhase, int pNumBeams )
{

	Reprojection imagePlaneReprojection;

	// set up input pixel vector and CD matrix.
	Reprojection::rpVectD tmpPixelIn = { /* x = */ _hstBeamSize / 2, /* y = */ _hstBeamSize / 2, /*z = */ 0 };
	Reprojection::rpMatr2x2 tmpCDIn = { /* a11 = */ -sin( rad( _hstBeamCellSize / 3600.0 ) ), /* a12 = */ 0.0, /* a21 = */ 0.0, /* a22 = */ sin( rad( _hstBeamCellSize / 3600.0 ) ) };

	// set up output pixel vector and CD matrix.
	Reprojection::rpVectD tmpPixelOut = { /* x = */ _hstUvPixels / 2, /* y = */ _hstUvPixels / 2, /*z = */ 0 };
	Reprojection::rpMatr2x2 tmpCDOut = { /* a11 = */ -sin( rad( _hstCellSize / 3600.0 ) ), /* a12 = */ 0.0, /* a21 = */ 0.0, /* a22 = */ sin( rad( _hstCellSize / 3600.0 ) ) };

	// build input and output size.
	Reprojection::rpVectI inSize = { /* x = */ _hstBeamSize, /* y = */ _hstBeamSize };
	Reprojection::rpVectI outSize = { /* x = */ _hstUvPixels, /* y = */ _hstUvPixels };

	// build in coordinate system.
	Reprojection::rpCoordSys inCoordSystem;
	inCoordSystem.crPIX = tmpPixelIn;
	inCoordSystem.cd = tmpCDIn;
	inCoordSystem.epoch = Reprojection::EPOCH_J2000;

	// build out coordinate system.
	Reprojection::rpCoordSys outCoordSystem;
	outCoordSystem.crVAL.x = _hstOutputRA;
	outCoordSystem.crVAL.y = _hstOutputDEC;
	outCoordSystem.crPIX = tmpPixelOut;
	outCoordSystem.cd = tmpCDOut;
	outCoordSystem.epoch = Reprojection::EPOCH_J2000;

	const int TOP_LEFT = 0;
	const int TOP_RIGHT = 2;
	const int BOTTOM_LEFT = 4;
	const int BOTTOM_RIGHT = 6;
	const int X = 0;
	const int Y = 1;

	double top = 0.0, bottom = (double) (_hstUvPixels - 1), left = (double) (_hstUvPixels - 1), right = 0.0;

	// loop over all the beams.
	for ( int beam = 0; beam < pNumBeams; beam++ )
	{

		inCoordSystem.crVAL.x = pBeamIn[ beam * 2 ];
		inCoordSystem.crVAL.y = pBeamIn[ (beam * 2) + 1 ];

		// build a list of pixels to be reprojected.
		double * pixel = (double *) malloc( 8 * sizeof( double ) );
		pixel[ TOP_LEFT + X ] = (double) ((_hstBeamSize / 2) - _hstBeamWidth); // top-left
		pixel[ TOP_LEFT + Y ] = (double) ((_hstBeamSize / 2) + _hstBeamWidth);
		pixel[ TOP_RIGHT + X ] = (double) ((_hstBeamSize / 2) + _hstBeamWidth); // top-right
		pixel[ TOP_RIGHT + Y ] = (double) ((_hstBeamSize / 2) + _hstBeamWidth);
		pixel[ BOTTOM_LEFT + X ] = (double) ((_hstBeamSize / 2) - _hstBeamWidth); // bottom-left
		pixel[ BOTTOM_LEFT + Y ] = (double) ((_hstBeamSize / 2) - _hstBeamWidth);
		pixel[ BOTTOM_RIGHT + X ] = (double) ((_hstBeamSize / 2) + _hstBeamWidth); // bottom-right
		pixel[ BOTTOM_RIGHT + Y ] = (double) ((_hstBeamSize / 2) - _hstBeamWidth);

		// convert the pixel positions into mosaic image coordinates.
		imagePlaneReprojection.ReprojectPixel(	/* pPixel = */ pixel,
							/* pNumPixels = */ 4,
							/* pInCoordinateSystem = */ inCoordSystem,
							/* pOutCoordinateSystem = */ outCoordSystem,
							/* pInSize = */ inSize,
							/* pOutSize = */ outSize );

		// calculate right-hand pixel.
		if (pixel[ BOTTOM_RIGHT + X ] > right)
			right = pixel[ BOTTOM_RIGHT + X ];
		if (pixel[ TOP_RIGHT + X ] > right)
			right = pixel[ TOP_RIGHT + X ];

		// calculate left-hand pixel.
		if (pixel[ BOTTOM_LEFT + X ] < left)
			left = pixel[ BOTTOM_LEFT + X ];
		if (pixel[ TOP_LEFT + X ] < left)
			left = pixel[ TOP_LEFT + X ];

		// calculate top pixel.
		if (pixel[ TOP_LEFT + Y ] > top)
			top = pixel[ TOP_LEFT + Y ];
		if (pixel[ TOP_RIGHT + Y ] > top)
			top = pixel[ TOP_RIGHT + Y ];

		// calculate bottom pixel.
		if (pixel[ BOTTOM_LEFT + Y ] < bottom)
			bottom = pixel[ BOTTOM_LEFT + Y ];
		if (pixel[ BOTTOM_RIGHT + Y ] < bottom)
			bottom = pixel[ BOTTOM_RIGHT + Y ];

		// free pixel memory.
		if (pixel != NULL)
			free( (void *) pixel );

	}
	
	// ensure we're within the image boundaries.
	if (top >= (_hstUvPixels - 1))
		top = (double) (_hstUvPixels - 1);
	if (right >= (_hstUvPixels - 1))
		right = (double) (_hstUvPixels - 1);
	if (bottom < 0.0)
		bottom = 0.0;
	if (left < 0.0)
		left = 0.0;

	// get the pixel in the middle of this region.
	double x = ((right + left) / 2.0);
	double y = ((top + bottom) / 2.0);

	// convert this pixel into a phase position.
	double ra = 0, dec = 0;
	imagePlaneReprojection.GetCoordinates(	/* pX = */ x,
						/* pY = */ y,
						/* pCoordinateSystem = */ outCoordSystem,
						/* pSize = */ outSize,
						/* pRA = */ &ra,
						/* pDEC = */ &dec );

	// update the phase position for ALL fields to be the same position.
	for ( int beam = 0; beam < pNumBeams; beam++ )
	{
		pPhase[ beam * 2 ] = ra;
		pPhase[ (beam * 2) + 1 ] = dec;
	}

} // getSuitablePhasePositionForBeam

//
//	getPrimaryBeamWidth()
//
//	CJS: 04/02/2020
//
//	Calculate the width of the primary beam in pixels, at the 1% level.
//

double getPrimaryBeamWidth( float * phstBeam, int pBeamSize )
{

	double maxWidth = 0.0;

	// search for the furthest pixel that has at least 1% of the peak value.
	int support = pBeamSize / 2;
	for ( int i = 0; i < pBeamSize; i++ )
		for ( int j = 0; j < pBeamSize; j++ )
			if (phstBeam[ (j * pBeamSize) + i ] >= 0.01)
			{
				double width = pow( (double) (i - support), 2 ) + pow( (double) (j - support), 2 );
				if (width > maxWidth)
					maxWidth = width;
			}

	// return the beam width.
	return sqrt( maxWidth );

} // getPrimaryBeamWidth

//
//	getPolarisationMultiplier()
//
//	CJS: 07/04/2020
//
//	Gets an array of multiplier that describe how the polarisation products should be handled.
//

double * getPolarisationMultiplier( char * pMeasurementSetFilename, int * pNumPolarisations, int * pNumPolarisationConfigurations, char * pTableData )
{

	// return value.	
	double * hstMultiplier = NULL;

	// get a list of polarisations.
	int * hstPolarisation = NULL;
	_hstCasacoreInterface.GetPolarisations(	/* pMeasurementSet = */ (pTableData[ 0 ] == '\0' ? pMeasurementSetFilename : pTableData),
						/* pNumPolarisations = */ pNumPolarisations,
						/* pNumPolarisationConfigurations = */ pNumPolarisationConfigurations,
						/* pPolarisation = */ &hstPolarisation );

	// create a list of multipliers for constructing visibilities from the Stokes parameters.
	if (*pNumPolarisationConfigurations > 0 && *pNumPolarisations > 0)
	{
		hstMultiplier = (double *) malloc( (*pNumPolarisationConfigurations) * (*pNumPolarisations) * sizeof( double * ) );
		memset( hstMultiplier, 0, (*pNumPolarisationConfigurations) * (*pNumPolarisations) * sizeof( double ) );
	}

	// if we have at least one polarisation product then we will check if we have the right one(s).
	if (*pNumPolarisations >= 1)
		for ( int config = 0; config < (*pNumPolarisationConfigurations); config++ )
		{

			const int UNDEF_CONST = 0, I_CONST = 1, Q_CONST = 2, U_CONST = 3, V_CONST = 4, RR_CONST = 5, RL_CONST = 6, LR_CONST = 7, LL_CONST = 8,
					XX_CONST = 9, XY_CONST = 10, YX_CONST = 11, YY_CONST = 12;

			// get pointer to these polarisations and multipliers.
			int * polarisationPtr = &hstPolarisation[ (config * (*pNumPolarisations)) ];
			double * multiplierPtr = &hstMultiplier[ (config * (*pNumPolarisations)) ];

			// check what we've got.
			bool undef = false, stokesI = false, stokesQ = false, stokesU = false, stokesV = false, rr = false, rl = false, lr = false, ll = false,
				xx = false, xy = false, yx = false, yy = false;
			for ( int i = 0; i < *pNumPolarisations; i++ )
			{
				undef = undef || (polarisationPtr[ i ] == UNDEF_CONST);
				stokesI = stokesI || (polarisationPtr[ i ] == I_CONST);
				stokesQ = stokesQ || (polarisationPtr[ i ] == Q_CONST);
				stokesU = stokesU || (polarisationPtr[ i ] == U_CONST);
				stokesV = stokesV || (polarisationPtr[ i ] == V_CONST);
				rr = rr || (polarisationPtr[ i ] == RR_CONST);
				rl = rl || (polarisationPtr[ i ] == RL_CONST);
				lr = lr || (polarisationPtr[ i ] == LR_CONST);
				ll = ll || (polarisationPtr[ i ] == LL_CONST);
				xx = xx || (polarisationPtr[ i ] == XX_CONST);
				xy = xy || (polarisationPtr[ i ] == XY_CONST);
				yx = yx || (polarisationPtr[ i ] == YX_CONST);
				yy = yy || (polarisationPtr[ i ] == YY_CONST);
			}

			// make sure we've got what we need.
			stokes whichStokes = _hstStokes;
			if (whichStokes == STOKES_Q && (stokesQ == false) && (xx == false || yy == false) && (rl == false || lr == false))
				whichStokes = STOKES_NONE;
			if (whichStokes == STOKES_U && (stokesU == false) && (xy == false || yx == false) && (rl == false || lr == false))
				whichStokes = STOKES_NONE;
			if (whichStokes == STOKES_V && (stokesV == false) && (xy == false || yx == false) && (rr == false || ll == false))
				whichStokes = STOKES_NONE;
			if (whichStokes == STOKES_I && (stokesI == false) && (xx == false || yy == false) && (rr == false || ll == false))
				whichStokes = STOKES_NONE;

			// display a warning if we can't do the requested Stokes imaging.
			if (whichStokes == STOKES_NONE)
			{
	
				printf( "WARNING: Polarisation configuration %i does have the correct polarisation products to image Stokes ", config );
				switch (_hstStokes)
				{
					case (STOKES_I):	{ printf( "I" ); break; }
					case (STOKES_Q):	{ printf( "Q" ); break; }
					case (STOKES_U):	{ printf( "U" ); break; }
					case (STOKES_V):	{ printf( "V" ); break; }
				}
				printf( ". We will use the first polarisation product (%i).\n\n", polarisationPtr[ 0 ] );

			}

			// set the multiplier.
			for ( int i = 0; i < *pNumPolarisations; i++ )
			{
				if (polarisationPtr[ i ] == XX_CONST && (whichStokes == STOKES_I || whichStokes == STOKES_Q))
					multiplierPtr[ i ] = 0.5;
				if (polarisationPtr[ i ] == XY_CONST && whichStokes == STOKES_U)
					multiplierPtr[ i ] = 0.5;
				if (polarisationPtr[ i ] == XY_CONST && whichStokes == STOKES_V)
					multiplierPtr[ i ] = -0.5;
				if (polarisationPtr[ i ] == YX_CONST && (whichStokes == STOKES_U || whichStokes == STOKES_V))
					multiplierPtr[ i ] = 0.5;
				if (polarisationPtr[ i ] == YY_CONST && whichStokes == STOKES_I)
					multiplierPtr[ i ] = 0.5;
				if (polarisationPtr[ i ] == YY_CONST && whichStokes == STOKES_Q)
					multiplierPtr[ i ] = -0.5;
				if (polarisationPtr[ i ] == RR_CONST && (whichStokes == STOKES_I || whichStokes == STOKES_V))
					multiplierPtr[ i ] = 0.5;
				if (polarisationPtr[ i ] == LL_CONST && (whichStokes == STOKES_I))
					multiplierPtr[ i ] = 0.5;
				if (polarisationPtr[ i ] == LL_CONST && (whichStokes == STOKES_V))
					multiplierPtr[ i ] = -0.5;
				if (polarisationPtr[ i ] == RL_CONST && (whichStokes == STOKES_Q))
					multiplierPtr[ i ] = 0.5;
				if (polarisationPtr[ i ] == RL_CONST && (whichStokes == STOKES_U))
					multiplierPtr[ i ] = -0.5;
				if (polarisationPtr[ i ] == LR_CONST && (whichStokes == STOKES_Q || whichStokes == STOKES_U))
					multiplierPtr[ i ] = 0.5;
				if (polarisationPtr[ i ] == I_CONST && (whichStokes == STOKES_I))
					multiplierPtr[ i ] = 1.0;
				if (polarisationPtr[ i ] == Q_CONST && (whichStokes == STOKES_Q))
					multiplierPtr[ i ] = 1.0;
				if (polarisationPtr[ i ] == U_CONST && (whichStokes == STOKES_U))
					multiplierPtr[ i ] = 1.0;
				if (polarisationPtr[ i ] == V_CONST && (whichStokes == STOKES_V))
					multiplierPtr[ i ] = 1.0;
			}
			if (whichStokes == STOKES_NONE)
				multiplierPtr[ 0 ] = 1.0;

	}

	// free the polarisations and multiplier.
	if (hstPolarisation != NULL)
		free( (void *) hstPolarisation );

	// return something.
	return hstMultiplier;

} // getPolarisationMultiplier

//
//	calculateVisibilitiesPerKernelSet()
//
//	CJS: 22/06/2020
//
//	Calculate the number of visibilities per kernel set for this image, stage and batch.
//

void calculateVisibilitiesPerKernelSet( int pNumVisibilities, int pBatchSize, VectorI * phstGridPosition, int **** phstVisibilitiesInKernelSet, int pNumGPUs,
						int * pNumBatches, int pNumKernelSets )
{

	// create a single batch, and set the number of visibilities to zero for all kernel sets.
	*pNumBatches = 1;
	*phstVisibilitiesInKernelSet = (int ***) malloc( sizeof( int ** ) );
	(*phstVisibilitiesInKernelSet)[ 0 ] = (int **) malloc( pNumKernelSets * sizeof( int * ) );
	for ( int kernelSet = 0; kernelSet < pNumKernelSets; kernelSet++ )
	{
		(*phstVisibilitiesInKernelSet)[ 0 ][ kernelSet ] = (int *) malloc( pNumGPUs * sizeof( int ) );
		for ( int gpu = 0; gpu < pNumGPUs; gpu++ )
			(*phstVisibilitiesInKernelSet)[ (*pNumBatches) - 1 ][ kernelSet ][ gpu ] = 0;
	}

	// store how many visibilities we're giving to each GPU, and default to zero.
	int * hstGPUVisibilities = (int *) malloc( pNumGPUs * sizeof( int ) );
	for ( int gpu = 0; gpu < pNumGPUs; gpu++ )
		hstGPUVisibilities[ gpu ] = 0;

	// calculate how many visibilities we have in each kernel set (A plane and W plane combination).
	int kernelSet = phstGridPosition[ 0 ].w;
	int currentGPU = 0;
	bool gpusFull = false;
	for ( int visibility = 0; visibility < pNumVisibilities; visibility++ )
	{

		// if we reach a new kernel set then we need to switch GPUs.
		if (phstGridPosition[ visibility ].w > kernelSet)
		{

			// find a GPU that still has space on it.
			currentGPU++;
			if (currentGPU == pNumGPUs)
				currentGPU = 0;
			while (hstGPUVisibilities[ currentGPU ] == pBatchSize)
			{
				currentGPU++;
				if (currentGPU == pNumGPUs)
					currentGPU = 0;
			}
			kernelSet = phstGridPosition[ visibility ].w;

		}

		// increment the visibility count.
		(*phstVisibilitiesInKernelSet)[ (*pNumBatches) - 1 ][ kernelSet ][ currentGPU ]++;
		hstGPUVisibilities[ currentGPU ]++;

		// have the number of visibilities for this gpu reached the batch size? if so, find a gpu with some space.
		if (hstGPUVisibilities[ currentGPU ] == pBatchSize)
		{
			gpusFull = true;
			int nextGPU = currentGPU + 1;
			if (nextGPU == pNumGPUs)
				nextGPU = 0;
			while (nextGPU != currentGPU)
			{
				if (hstGPUVisibilities[ nextGPU ] < pBatchSize)
				{
					currentGPU = nextGPU;
					gpusFull = false;
					break;
				}
				nextGPU++;
				if (nextGPU == pNumGPUs)
					nextGPU = 0;
			}
		}

		// if the number of visibilities for all gpus have reached the batch size then we need to make a new batch.
		if (gpusFull == true && visibility < pNumVisibilities - 1)
		{

			(*pNumBatches)++;
			*phstVisibilitiesInKernelSet = (int ***) realloc( *phstVisibilitiesInKernelSet, *pNumBatches * sizeof( int ** ) );
			(*phstVisibilitiesInKernelSet)[ (*pNumBatches) - 1 ] = (int **) malloc( pNumKernelSets * sizeof( int * ) );
			for ( int kernelSet = 0; kernelSet < pNumKernelSets; kernelSet++ )
			{
				(*phstVisibilitiesInKernelSet)[ (*pNumBatches) - 1 ][ kernelSet ] = (int *) malloc( pNumGPUs * sizeof( int ) );
				for ( int gpu = 0; gpu < pNumGPUs; gpu++ )
					(*phstVisibilitiesInKernelSet)[ (*pNumBatches) - 1 ][ kernelSet ][ gpu ] = 0;
			}

			// each batch starts with gpu 0.
			currentGPU = 0;

			// reset count of visibilities per gpu.
			for ( int gpu = 0; gpu < pNumGPUs; gpu++ )
				hstGPUVisibilities[ gpu ] = 0;
			gpusFull = false;

		}

	}

	// free memory.
	if (hstGPUVisibilities != NULL)
		free( (void *) hstGPUVisibilities );

} // calculateVisibilitiesPerKernelSet

//
//	mergeData()
//
//	CJS: 22/07/2020
//
//	Merge two data caches together.
//

void mergeData( char * pFilenamePrefix, int pMosaicID, int pStageID_one, int pStageID_two, bool pLoadAllData, int pWhatData )
{

	// load the data from stage two into the start of the array.
	if (pLoadAllData == true)
		uncacheData(	/* pFilenamePrefix = */ pFilenamePrefix,
				/* pMosaicID = */ pMosaicID,
				/* pBatchID = */ pStageID_two,
				/* pWhatData = */ pWhatData,
				/* pOffset = */ 0 );

	// load the records from stage one into the end of the arrays.
	uncacheData(	/* pFilenamePrefix = */ pFilenamePrefix,
			/* pMosaicID = */ pMosaicID,
			/* pBatchID = */ pStageID_one,
			/* pWhatData = */ pWhatData,
			/* pOffset = */ _hstNumVisibilities[ pMosaicID ][ pStageID_two ] );

	// update the number of stages and the number of visibilities.
	_hstNumberOfStages[ pMosaicID ] -= 1;
	_hstNumVisibilities[ pMosaicID ][ pStageID_one ] += _hstNumVisibilities[ pMosaicID ][ pStageID_two ];

	// if this is not the last stage then rename the subsequent files.
	if (pStageID_two < _hstNumberOfStages[ pMosaicID ])
		for ( int stageID = pStageID_two; stageID < _hstNumberOfStages[ pMosaicID ]; stageID++ )
		{
	
			// build filename.
			char filenameOld[ 255 ], filenameNew[ 255 ];
			if (_hstCacheLocation[0] != '\0')
			{
				sprintf( filenameNew, "%s%s-%02i-%i-cache.dat", _hstCacheLocation, pFilenamePrefix, pMosaicID, stageID );
				sprintf( filenameOld, "%s%s-%02i-%i-cache.dat", _hstCacheLocation, pFilenamePrefix, pMosaicID, stageID + 1 );
			}
			else
			{
				sprintf( filenameNew, "%s-%02i-%i-cache.dat", pFilenamePrefix, pMosaicID, stageID );
				sprintf( filenameOld, "%s-%02i-%i-cache.dat", pFilenamePrefix, pMosaicID, stageID + 1 );
			}

			// rename file.
			rename( filenameOld, filenameNew );

			// update number of visibilities.
			_hstNumVisibilities[ pMosaicID ][ stageID ] = _hstNumVisibilities[ pMosaicID ][ stageID + 1 ];

		}
	else
	{

		// build filename.
		char filename[ 255 ];
		if (_hstCacheLocation[0] != '\0')
			sprintf( filename, "%s%s-%02i-%i-cache.dat", _hstCacheLocation, pFilenamePrefix, pMosaicID, pStageID_two );
		else
			sprintf( filename, "%s-%02i-%i-cache.dat", pFilenamePrefix, pMosaicID, pStageID_two );

		// remove file.
		remove( filename );

	}
	_hstNumVisibilities[ pMosaicID ] = (long int *) realloc( _hstNumVisibilities[ pMosaicID ], _hstNumberOfStages[ pMosaicID ] * sizeof( long int * ) );

} // mergeData

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

double performUniformWeighting( char * pFilenamePrefix, int pMosaicID, double * phstTotalWeightPerCell )
{

	double averageWeight = 0.0;
	long int griddedVisibilities = 0;

	// we might be dealing with HUGE numbers, so we normalise our gridded weights using the maximum weight. we're happy to work with numbers between 0 and 1.
	// first, find the average gridded cell weight.
	for ( long int i = 0; i < (long int) _hstUvPixels * (long int) _hstUvPixels; i++ )
		averageWeight += phstTotalWeightPerCell[ i ];
	averageWeight /= (double) pow( (long int) _hstUvPixels, 2 );

	// normalise the gridded weights using the average weight.
	if (averageWeight > 0.0)
		for ( long int i = 0; i < (long int) _hstUvPixels * (long int) _hstUvPixels; i++ )
			phstTotalWeightPerCell[ i ] /= averageWeight;

	// reset the average weight. we will compute it per visibility now.
	averageWeight = 0.0;

	int mosaicMin = (pMosaicID == -1 ? 0 : pMosaicID );
	int mosaicMax = (pMosaicID == -1 ? _hstMeasurementSets - 1 : pMosaicID );
	for ( int mosaicID = mosaicMin; mosaicID <= mosaicMax; mosaicID++ )
	{
		griddedVisibilities += _hstGriddedVisibilities[ mosaicID ];
		for ( int stageID = 0; stageID < _hstNumberOfStages[ mosaicID ]; stageID++ )
		{

			// get the weights, densities and grid positions from the file for this stage.
			if (_hstCacheData == true)
				uncacheData(	/* pFilenamePrefix = */ pFilenamePrefix,
						/* pMosaicID = */ mosaicID,
						/* pStageID = */ stageID,
						/* pWhatData = */ DATA_WEIGHTS | DATA_GRID_POSITIONS | DATA_DENSITIES,
						/* pOffset = */ 0 );

			// divide each weight by the total weight in that cell, and add up the weights so we can make an average.
			for ( long int i = 0; i < _hstNumVisibilities[ mosaicID ][ stageID ]; i++ )
				if (	_hstGridPosition[ i ].u >= 0 && _hstGridPosition[ i ].u < _hstUvPixels &&
					_hstGridPosition[ i ].v >= 0 && _hstGridPosition[ i ].v < _hstUvPixels)
				{
					_hstWeight[ i ] /= phstTotalWeightPerCell[ (_hstGridPosition[ i ].v * _hstUvPixels) + _hstGridPosition[ i ].u ];
					averageWeight += (double) _hstWeight[ i ] * (double) _hstDensityMap[ i ];
				}

			// re-cache the weights and free the densities and grid positions for this stage.
			if (_hstCacheData == true)
			{
				cacheData(	/* pFilenamePrefix = */ pFilenamePrefix,
						/* pMosaicID = */ mosaicID,
						/* pStageID = */ stageID,
						/* pWhatData = */ DATA_WEIGHTS );
				freeData( /* pWhatData = */ DATA_DENSITIES | DATA_GRID_POSITIONS );
			}

		}
	}

	// calculate and return the average weight.
	return averageWeight / (double) griddedVisibilities;

} // processUniformWeighting

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

double performRobustWeighting( char * pFilenamePrefix, int pMosaicID, double * phstTotalWeightPerCell )
{

	double totalWeightSquared = 0.0;
	double totalWeight = 0.0;
	double averageWeight = 0.0;
	long int griddedVisibilities = 0;

	// calculate the average cell weighting.
	int mosaicMin = (pMosaicID == -1 ? 0 : pMosaicID );
	int mosaicMax = (pMosaicID == -1 ? _hstMeasurementSets - 1 : pMosaicID );

	// we might be dealing with HUGE numbers, so we normalise our gridded weights using the maximum weight. we're happy to work with numbers between 0 and 1.
	// first, find the maximum gridded cell weight.
	double maxWeight = 0.0;
	for ( long int i = 0; i < (long int) _hstUvPixels * (long int) _hstUvPixels; i++ )
		if (phstTotalWeightPerCell[ i ] > maxWeight)
			maxWeight = phstTotalWeightPerCell[ i ];

	// compute the average weight per cell.
	if (maxWeight > 0.0)
		for ( long int i = 0; i < (long int) _hstUvPixels * (long int) _hstUvPixels; i++ )
		{
		
			// normalise by the maximum weight found.
			phstTotalWeightPerCell[ i ] /= maxWeight;

			// sum the weight and the weight squared.
			totalWeightSquared += pow( phstTotalWeightPerCell[ i ], 2 );
			totalWeight += phstTotalWeightPerCell[ i ];

		}

	// calculate average and f^2 parameter.
	double fSquared = pow( 5.0 * pow( 10.0, -_hstRobust ), 2 );
	if (totalWeight != 0.0)
	{
		double meanWeight = totalWeightSquared / totalWeight;
		for ( long int i = 0; i < (long int) _hstUvPixels * (long int) _hstUvPixels; i++ )
			phstTotalWeightPerCell[ i ] /= meanWeight;
	}

	for ( int mosaicID = mosaicMin; mosaicID <= mosaicMax; mosaicID++ )
	{
		griddedVisibilities += _hstGriddedVisibilities[ mosaicID ];
		for ( int stageID = 0; stageID < _hstNumberOfStages[ mosaicID ]; stageID++ )
		{

			// get the grid positions, densities and weights.
			if (_hstCacheData == true)
				uncacheData(	/* pFilenamePrefix = */ pFilenamePrefix,
						/* pMosaicID = */ mosaicID,
						/* pStageID = */ stageID,
						/* pWhatData = */ DATA_GRID_POSITIONS | DATA_WEIGHTS | DATA_DENSITIES,
						/* pOffset = */ 0 );

			// update the weight of each visibility using the original weight, the sum of weights in the cell, and the f^2 parameter. also, add
			// up the weights so we can construct an average.
			for ( long int i = 0; i < _hstNumVisibilities[ mosaicID ][ stageID ]; i++ )
				if (	_hstGridPosition[ i ].u >= 0 && _hstGridPosition[ i ].u < _hstUvPixels &&
					_hstGridPosition[ i ].v >= 0 && _hstGridPosition[ i ].v < _hstUvPixels)
				{
					_hstWeight[ i ] /= (1.0 + (phstTotalWeightPerCell[ (_hstGridPosition[ i ].v * _hstUvPixels) + _hstGridPosition[ i ].u ] * fSquared));
					averageWeight += (double) _hstWeight[ i ] * (double) _hstDensityMap[ i ];
				}

			// re-cache the weights and free the densities and grid positions for this stage.
			if (_hstCacheData == true)
			{
				cacheData(	/* pFilenamePrefix = */ pFilenamePrefix,
						/* pMosaicID = */ mosaicID,
						/* pStageID = */ stageID,
						/* pWhatData = */ DATA_WEIGHTS );
				freeData( /* pWhatData = */ DATA_DENSITIES | DATA_GRID_POSITIONS );
			}

		}
	}

	// calculate and return average weight.
	return averageWeight / (double) griddedVisibilities;

} // performRobustWeighting

//
//	performNaturalWeighting()
//
//	CJS: 16/12/2020
//
//	calculates the average weight.
//	This routine can act on a single mosaic ID, or on all of them.
//

double performNaturalWeighting( int pMosaicID )
{

	double averageWeight = 0.0;
	long int griddedVisibilities = 0;

	// calculate the average cell weighting.
	int mosaicMin = (pMosaicID == -1 ? 0 : pMosaicID );
	int mosaicMax = (pMosaicID == -1 ? _hstMeasurementSets - 1 : pMosaicID );
	for ( int mosaicID = mosaicMin; mosaicID <= mosaicMax; mosaicID++ )
	{
		averageWeight += (double) _hstAverageWeight[ mosaicID ] * (double) _hstGriddedVisibilities[ mosaicID ];
		griddedVisibilities += _hstGriddedVisibilities[ mosaicID ];
	}

	// calculate and return the average weight.
	return averageWeight / (double) griddedVisibilities;

} // performNaturalWeighting

//
//	processMeasurementSet()
//
//	CJS: 05/07/2019
//
//	Loads the measurement sets into memory, and caches them so we don't have to store them all at the same time.
//
//	Notes:
//		The pTableData parameter holds the location of the tables, such as FIELD, ANTENNA, etc. If this parameter is empty then we use the measurement
//			set filename. The only times where we want to use a different location is when we are processing a MMS file, where all the tables will
//			be held in a different location.
//

void processMeasurementSet( char * pFilenamePrefix, char * pMeasurementSetFilename, char * pFieldID, double * phstImagePhasePosition, int pFileIndex,
				char * pTableData, double * phstTotalWeightPerCell )
{
	
	cudaError_t err;
	struct timespec time1, time2;

	// channels.
	int * hstNumChannels = NULL;
	int numSpws = 0;
	bool ** hstSpwChannelFlag = NULL;

	// get the total system memory.
	struct sysinfo memInfo;
	sysinfo( &memInfo );
	long int hstMaxMemory = memInfo.totalram / 2;

	printf( "\nProcessing measurement set: %s\n", pMeasurementSetFilename );
	printf( "------------------------------------------------------------------------------\n\n" );

	// declare array for the generated primary beam.
	float * hstPrimaryBeamIn = NULL;
	
	// load channel wavelengths.
	double ** hstWavelength = NULL;
	_hstCasacoreInterface.GetWavelengths(	/* pMeasurementSet = */ (pTableData[ 0 ] == '\0' ? pMeasurementSetFilename : pTableData),
						/* pNumSpws = */ &numSpws,
						/* pNumChannels = */ &hstNumChannels,
						/* pWavelength = */ &hstWavelength );

	// get the polarisation multiplier (and the number of polarisations) which describes how the polarisation products should be handled.
	int hstNumPolarisations = -1, hstNumPolarisationConfigurations = -1;
	double * hstMultiplier = getPolarisationMultiplier(	/* pMeasurementSetFilename = */ pMeasurementSetFilename,
								/* pNumPolarisations = */ &hstNumPolarisations,
								/* pNumPolarisationConfigurations = */ &hstNumPolarisationConfigurations,
								/* pTableData = */ pTableData );

	// upload the polarisation multipliers to the device.
	double * devMultiplier = NULL;
	if (hstNumPolarisations > 0 && hstNumPolarisationConfigurations > 0)
	{
		reserveGPUMemory( (void **) &devMultiplier, hstNumPolarisationConfigurations * hstNumPolarisations * sizeof( double ),
											"declaring device memory for polarisation multipliers" );
		cudaMemcpy( devMultiplier, hstMultiplier, hstNumPolarisationConfigurations * hstNumPolarisations * sizeof( double ), cudaMemcpyHostToDevice );
	}

	// get the data description info, which is the polarisation configuration id and spectral window id.
	int numDataDescItems = -1;
	int * hstDataDescPolarisationConfig = NULL, * hstDataDescSpw = NULL;
	_hstCasacoreInterface.GetDataDesc(	/* pMeasurementSet = */ (pTableData[ 0 ] == '\0' ? pMeasurementSetFilename : pTableData),
						/* pNumDataDescItems = */ &numDataDescItems,
						/* pPolarisationConfig = */ &hstDataDescPolarisationConfig,
						/* pSpw = */ &hstDataDescSpw );
	if (numDataDescItems < 1)
	{
		printf( "ERROR: No data found in data description table.\n" );
		abort();
	}

	// set the spw and channels flags based upon the supplied parameters.
	setSpwAndChannelFlags(	/* pNumSpws = */ numSpws,
				/* phstNumChannels = */ hstNumChannels,
				/* phstSpwChannelFlag = */ &hstSpwChannelFlag,
				/* phstSpwRestriction = */ _hstSpwRestriction[ pFileIndex ] );

	// count the number of unflagged channels.
	int * hstUnflaggedChannels = (int *) malloc( numSpws * sizeof( int ) );
	for ( int spw = 0; spw < numSpws; spw++ )
	{
		hstUnflaggedChannels[ spw ] = 0;
		for ( int channel = 0; channel < hstNumChannels[ spw ]; channel++ )
			if (hstSpwChannelFlag[ spw ][ channel ] == false)
				hstUnflaggedChannels[ spw ]++;
	}

	// flag certain data description IDs if we are not using any of these channels.
	bool * hstDataDescFlag = (bool *) malloc( numDataDescItems * sizeof( bool ) );
	for ( int i = 0; i < numDataDescItems; i++ )
		hstDataDescFlag[ i ] = true;
	for ( int dataDesc = 0; dataDesc < numDataDescItems; dataDesc++ )
		for ( int channel = 0; channel < hstNumChannels[ hstDataDescSpw[ dataDesc ] ]; channel++ )
			if (hstSpwChannelFlag[ hstDataDescSpw[ dataDesc ] ][ channel ] == false)
				hstDataDescFlag[ dataDesc ] = false;

	// declare array for samples, field ids, and data description ids.
	int hstNumSamples = 0;
	VectorD * hstSample = NULL;
	int * hstSampleFieldID = NULL;
	int * hstDataDescID = NULL;
	int * hstAntenna1 = NULL, * hstAntenna2 = NULL;

	long int numVisibilities = 0;
	{

		// load samples.
		_hstCasacoreInterface.GetSamples(	/* IN: pMeasurementSet = */ pMeasurementSetFilename,
							/* OUT: pNumSamples = */ &hstNumSamples,
							/* OUT: pSample = */ (double **) &hstSample,
							/* IN: pFieldID = */ pFieldID,
							/* IN: pDataDescFlag = */ hstDataDescFlag,
							/* IN: pDataDescItems = */ numDataDescItems,
							/* OUT: pFieldIDArray = */ &hstSampleFieldID,
							/* OUT: pDataDescID = */ &hstDataDescID,
							/* OUT: pAntenna1 = */ &hstAntenna1,
							/* OUT: pAntenna2 = */ &hstAntenna2 );

		// calculate number of visibilities, and store this number for later.
		for ( int sample = 0; sample < hstNumSamples; sample++ )
			numVisibilities += (long int) hstUnflaggedChannels[ hstDataDescSpw[ hstDataDescID[ sample ] ] ];
		numVisibilities *= 2;

		// we actually have twice as many samples as this because they are mirrored.
		hstNumSamples = hstNumSamples * 2;

		// create all the arrays we need to store the samples. The sample array already exists, so are resized to the new larger size.
		hstSample = (VectorD *) realloc( hstSample, hstNumSamples * sizeof( VectorD ) );
		hstSampleFieldID = (int *) realloc( hstSampleFieldID, hstNumSamples * sizeof( int ) );
		

		// create the array to hold the batches of data. we start of with 1 batch and will increase this if required.
		_hstNumVisibilities[ pFileIndex ] = (long int *) malloc( 1 * sizeof( long int ) );
		_hstNumberOfStages[ pFileIndex ] = 0;
		_hstNumberOfBatches[ pFileIndex ] = (int *) malloc( 1 * sizeof( int ) );

	}

	// get the antennae from the file.
	bool * hstAntennaFlag = NULL;
	double * hstDishDiameter = NULL;
	int numberOfAntennae = _hstCasacoreInterface.GetAntennae(	/* pMeasurementSet = */ (pTableData[ 0 ] == '\0' ? pMeasurementSetFilename : pTableData),
									/* pDishDiameter = */ &hstDishDiameter,
									/* pFlagged = */ &hstAntennaFlag );
	printf( "found %i antennae, ", numberOfAntennae );

	// count the number of unflagged antennae.
	int unflaggedAntennae = 0;
	if (numberOfAntennae > 0)
		for ( int i = 0; i < numberOfAntennae; i++ )
			unflaggedAntennae += (hstAntennaFlag[ i ] == false ? 1 : 0);
	printf( "and %i of them are unflagged\n", unflaggedAntennae );

	// get the antennae ID of all the unflagged antennae.
	int * hstAntenna = NULL;
	if (unflaggedAntennae > 0)
	{
		hstAntenna = (int *) malloc( unflaggedAntennae * sizeof( int ) );
		int index = 0;
		for ( int i = 0; i < numberOfAntennae; i++ )
			if (hstAntennaFlag[ i ] == false)
			{
				hstAntenna[ index ] = i;
				index++;
			}
	}

	printf( "antenna indexes in table are [" );
	for ( int i = 0; i < unflaggedAntennae; i++ )
	{
		if (i != 0)
			printf( ", " );
		printf( "%i", hstAntenna[ i ] );
	}
	printf( "]\n\n" );

	// calculate the minimum dish diameter from these data.
	double minimumDishDiameter = -1.0;
	for ( int sample = 0; sample < hstNumSamples / 2; sample++ )
	{
		if (hstDishDiameter[ hstAntenna1[ sample ] ] < minimumDishDiameter || sample == 0)
			minimumDishDiameter = hstDishDiameter[ hstAntenna1[ sample ] ];
		if (hstDishDiameter[ hstAntenna2[ sample ] ] < minimumDishDiameter)
			minimumDishDiameter = hstDishDiameter[ hstAntenna2[ sample ] ];
	}

	// free memory.
	if (hstAntennaFlag != NULL)
		free( (void *) hstAntennaFlag );
	if (hstDishDiameter != NULL)
		free( (void *) hstDishDiameter );
	if (hstAntenna1 != NULL)
		free( (void *) hstAntenna1 );
	if (hstAntenna2 != NULL)
		free( (void *) hstAntenna2 );

	// if we are gridding ALMA data then check what size dishes we're using.
	if (_hstTelescope == ALMA && (int) round( minimumDishDiameter ) == 7)
		_hstTelescope = ALMA_7M;
	if (_hstTelescope == ALMA && (int) round( minimumDishDiameter ) == 12)
		_hstTelescope = ALMA_12M;

	printf( "Properties:\n" );
	printf( "---------------------------\n\n" );
	printf( "num spectral windows %i\n", numSpws );
	printf( "num channels [" );
	for ( int spw = 0; spw < numSpws; spw++ )
	{
		printf( "%i", hstNumChannels[ spw ] );
		if (spw < numSpws - 1)
			printf( ", " );
	}
	printf( "]\n" );
	printf( "num visibilities %li\n", numVisibilities );
	printf( "num samples %i\n", hstNumSamples );
	printf( "num polarisations %i\n", hstNumPolarisations );
	printf( "minimum dish diameter %4.2f m\n", minimumDishDiameter );

	// copy constants to constant memory.
	err = cudaMemcpyToSymbol( _devUvPixels, &_hstUvPixels, sizeof( _hstUvPixels ) );
	if (err != cudaSuccess)
		printf( "error copying grid size to device (%s)\n", cudaGetErrorString( err ) );

	// initialise the kernel sizes to 0.
	for ( int i = 0; i < _hstKernelSets; i++ )
	{
		_hstSupportSize[ i ] = 0;
		_hstKernelSize[ i ] = 0;
	}

	// flag any spws that are not actually used in our data.
	for ( int spw = 0; spw < numSpws; spw++ )
	{
		bool found = false;
		for ( int sample = 0; sample < hstNumSamples / 2; sample++ )
			if (hstDataDescSpw[ hstDataDescID[ sample ] ] == spw)
				found = true;
		if (found == false)
			for ( int channel = 0; channel < hstNumChannels[ spw ]; channel++ )
				hstSpwChannelFlag[ spw ][ channel ] = true;
	}

	// get the minimum wavelength.
	_hstMinWavelength = -1.0;
	_hstMaxWavelength = -1.0;

	// store the total, minimum, and maximum wavelength, and the number of valid channels per spw.
	double * hstTmpTotal = (double *) malloc( numSpws * sizeof( double ) );
	double * hstTmpMax = (double *) malloc( numSpws * sizeof( double ) );
	double * hstTmpMin = (double *) malloc( numSpws * sizeof( double ) );
	long int * hstTmpValid = (long int *) malloc( numSpws * sizeof( long int ) );

	// calculate the total, minimum, and maximum wavelength, and the number of valid channels per spw.
	for ( int spw = 0; spw < numSpws; spw++ )
	{
		hstTmpTotal[ spw ] = 0.0;
		hstTmpMax[ spw ] = -1.0;
		hstTmpMin[ spw ] = -1.0;
		hstTmpValid[ spw ] = 0;
		for ( int channel = 0; channel < hstNumChannels[ spw ]; channel++ )
			if (hstSpwChannelFlag[ spw ][ channel ] == false)
			{
				hstTmpTotal[ spw ] += hstWavelength[ spw ][ channel ];
				if (hstWavelength[ spw ][ channel ] < hstTmpMin[ spw ] || hstTmpMin[ spw ] == -1.0)
					hstTmpMin[ spw ] = hstWavelength[ spw ][ channel ];
				if (hstWavelength[ spw ][ channel ] > hstTmpMax[ spw ] || hstTmpMax[ spw ] == -1.0)
					hstTmpMax[ spw ] = hstWavelength[ spw ][ channel ];
				hstTmpValid[ spw ]++;
			}
	}
	
	// get the average wavelength by summing over all the data. we're not just summing over the spws and channels, we're summing over the actual visibility data.
	_hstAverageWavelength[ pFileIndex ] = 0.0;
	long int hstValidChannels = 0;
	for ( int sample = 0; sample < hstNumSamples / 2; sample++ )
	{
		int spw = hstDataDescSpw[ hstDataDescID[ sample ] ];
		_hstAverageWavelength[ pFileIndex ] += hstTmpTotal[ spw ];
		if (hstTmpMin[ spw ] < _hstMinWavelength || _hstMinWavelength == -1.0)
			_hstMinWavelength = hstTmpMin[ spw ];
		if (hstTmpMax[ spw ] > _hstMaxWavelength || _hstMaxWavelength == -1.0)
			_hstMaxWavelength = hstTmpMin[ spw ];
		hstValidChannels += hstTmpValid[ spw ];
	}

	if (hstValidChannels > 0)
		_hstAverageWavelength[ pFileIndex ] /= (double) hstValidChannels;
	else
		_hstAverageWavelength[ pFileIndex ] = 1.0;

	// free data.
	if (hstTmpTotal != NULL)
		free( (void *) hstTmpTotal );
	if (hstTmpMax != NULL )
		free( (void *) hstTmpMax );
	if (hstTmpMin != NULL )
		free( (void *) hstTmpMin );
	if (hstTmpValid != NULL )
		free( (void *) hstTmpValid );
	if (hstDataDescID != NULL)
		free( (void *) hstDataDescID );

	// calculate a-planes for this mosaic field.
	int ** whichAPlane = NULL;
	double * hstAPlaneWavelength = NULL;
	hstAPlaneWavelength = (double *) malloc( _hstAPlanes * sizeof( double ) );
	if (_hstAProjection == true)
		calculateAPlanes(	/* OUT: pWhichAPlane = */ &whichAPlane,
					/* OUT: phstAPlaneWavelength = */ hstAPlaneWavelength,
					/* IN: phstWavelength = */ hstWavelength,
					/* IN: pNumSpws = */ numSpws,
					/* IN: phstNumChannels = */ hstNumChannels,
					/* IN: phstSpwChannelFlag = */ hstSpwChannelFlag );

	// choose which wavelength to use for the primary beam.
	double hstWavelengthForBeam = _hstAverageWavelength[ pFileIndex ];
	if (_hstAProjection == true)
		hstWavelengthForBeam = _hstMinWavelength;

	printf( "average wavelength %8.6f m (min: %8.6f m, max %8.6f m)\n\n", _hstAverageWavelength[ pFileIndex ], _hstMinWavelength, _hstMaxWavelength );

	// if a beam filename was provided in the settings file then we load the beam here.
	if (_hstBeamPattern[ pFileIndex ] != '\0')
		loadPrimaryBeam(	/* pBeamFilename = */ _hstBeamPattern[ pFileIndex ],
					/* phstPrimaryBeamIn = */ &hstPrimaryBeamIn,
					/* pSize = */ _hstBeamSize );
	else
	{

		// set primary beam parameters depending upon telescope.
		if (_hstDiskDiameterSupplied == false)
			switch (_hstTelescope)
			{
				case ALMA:
				case ALMA_7M:		{ _hstAiryDiskDiameter = 6.25; break; }
				case ALMA_12M:		{ _hstAiryDiskDiameter = 10.70; break; }
				case ASKAP:		{ _hstAiryDiskDiameter = 12.00; break; }
				case VLA:		{ _hstAiryDiskDiameter = 25.0; break; }
				case MEERKAT:		{ _hstAiryDiskDiameter = 13.5; break; }
				case EMERLIN:		{ _hstAiryDiskDiameter = minimumDishDiameter; break; }
			}
		if (_hstDiskBlockageSupplied == false)
			switch (_hstTelescope)
			{
				case ALMA:
				case ALMA_7M:		{ _hstAiryDiskBlockage = 0.75; break; }
				case ALMA_12M:		{ _hstAiryDiskBlockage = 0.75; break; }
				case ASKAP:		{ _hstAiryDiskBlockage = 0.75; break; }
				case VLA:		{ _hstAiryDiskBlockage = 0.75; break; }
				case MEERKAT:		{ _hstAiryDiskBlockage = 0.75; break; }
				case EMERLIN:		{ _hstAiryDiskBlockage = 0.75; break; }
			}

		// generate primary beams.
		generatePrimaryBeamAiry(	/* phstPrimaryBeamIn = */ &hstPrimaryBeamIn,
						/* pWidth = */ _hstAiryDiskDiameter,
						/* pCutout = */ _hstAiryDiskBlockage,
						/* pWavelength = */ hstWavelengthForBeam );

		_hstBeamFrequency = CONST_C / hstWavelengthForBeam;

	}

	// if we are file mosaicing then we need to measure the radius of the beam at the 1% level [in pixels].
	if (_hstFileMosaic == true)
		_hstBeamWidth = getPrimaryBeamWidth(	/* phstBeam = */ hstPrimaryBeamIn,
							/* pBeamSize = */ _hstBeamSize );

	// save primary beam.
	if (pFileIndex == 0)
	{

		// save primary beam.
		char beamFilename[ 100 ];
		sprintf( beamFilename, "%s-generated-primary-beam.image", _hstOutputPrefix );
		_hstCasacoreInterface.WriteCasaImage(	/* pFilename = */ beamFilename,
							/* pWidth = */ _hstBeamSize,
							/* pHeight = */ _hstBeamSize,
							/* pRA = */ _hstOutputRA,
							/* pDec = */ _hstOutputDEC,
							/* pPixelSize = */ _hstBeamCellSize,
							/* pImage = */ hstPrimaryBeamIn,
							/* pFrequency = */ CONST_C / _hstAverageWavelength[ pFileIndex ],
							/* pMask = */ NULL );

	}

	// hold the number of fields found in each file, and the phase centre of each field, along with the phase centre of each image and the phase centre
	// of each beam.
	int hstNumFields = 0;
	double * hstFieldPhaseFrom = NULL;
	double * hstFieldPhaseTo = NULL;

	// load the phase centres for each field.
	_hstCasacoreInterface.GetPhaseCentres(	/* pMeasurementSet = */ (pTableData[ 0 ] == '\0' ? pMeasurementSetFilename : pTableData),
						/* pNumFields = */ &hstNumFields,
						/* pPhaseCentre = */ &hstFieldPhaseFrom );

	// convert from radians to degrees.
	for ( int i = 0; i < hstNumFields * 2; i++ )
		hstFieldPhaseFrom[ i ] = hstFieldPhaseFrom[ i ] * 180.0 / PI;

	// ensure ra is in range 0 <= ra < 360.
	for ( int i = 0; i < hstNumFields * 2; i = i + 2 )
		if (hstFieldPhaseFrom[ i ] < 0.0)
			hstFieldPhaseFrom[ i ] += 360.0;

	// create new image phase centre array.
	hstFieldPhaseTo = (double *) malloc( hstNumFields * 2 * sizeof( double ) );

	// create a map between the field id and the 'reduced' field id (which only contains the field ids actually used in our data).
	int * fieldIDMap = NULL;

	// compress the field ids that have been used into the range 0...N.
	compactFieldIDs(	/* pPhaseCentrePtr = */ &hstFieldPhaseFrom,
				/* pPhaseCentreImagePtr = */ &hstFieldPhaseTo,
				/* pNumFields = */ &hstNumFields,
				/* pFieldID = */ hstSampleFieldID,
				/* pFieldIDMap = */ &fieldIDMap,
				/* pNumSamples = */ hstNumSamples / 2 );

	// update field id using the map.
	for ( int i = 0; i < hstNumSamples / 2; i++ )
		for ( int j = 0; j < hstNumFields; j++ )
			if (fieldIDMap[ j ] == hstSampleFieldID[ i ])
			{
				hstSampleFieldID[ i ] = j;
				break;
			}

	// mirror samples and field ids.
	long int sampleDestination = hstNumSamples / 2;
	for ( long int i = 0; i < hstNumSamples / 2; i++, sampleDestination++ )
	{
		hstSample[ sampleDestination ].u = -hstSample[ i ].u;
		hstSample[ sampleDestination ].v = -hstSample[ i ].v;
		hstSample[ sampleDestination ].w = -hstSample[ i ].w;
	}
	memcpy( &hstSampleFieldID[ hstNumSamples / 2 ], hstSampleFieldID, hstNumSamples * sizeof( int ) / 2 );

	// we get the position of the ASKAP PAF beam, based upon the pointing position of the dish (for which we use the phase position).
	if (_hstTelescope == ASKAP && _hstBeamID[ pFileIndex ] != -1)
	{
		double xOffset = 0.0, yOffset = 0.0;
		_hstCasacoreInterface.GetASKAPBeamOffset(	/* pMeasurementSet = */ pMeasurementSetFilename,
								/* pBeamID = */ _hstBeamID[ pFileIndex ],
								/* pXOffset = */ &xOffset,
								/* pYOffset = */ &yOffset );
		for ( int field = 0; field < hstNumFields; field++ )
			getASKAPBeamPosition(	/* pRA = */ &hstFieldPhaseFrom[ field * 2 ],
						/* pDEC = */ &hstFieldPhaseFrom[ (field * 2) + 1 ],
						/* pXOffset = */ xOffset,
						/* pYOffset = */ yOffset,
						/* pCentreRA = */ hstFieldPhaseFrom[ field * 2 ],
						/* pCentreDEC = */ hstFieldPhaseFrom[ (field * 2) + 1 ] );
	}

	// get suitable phase positions for gridding.
	if (_hstFileMosaic == true)
		getSuitablePhasePositionForBeam(	/* pBeamIn = */ hstFieldPhaseFrom,
							/* pPhase = */ hstFieldPhaseTo,
							/* pNumBeams = */ hstNumFields );

	// otherwise set the phase position of each field to the required output phase position.
	else
		for ( int field = 0; field < hstNumFields; field++ )
		{
			hstFieldPhaseTo[ field * 2 ] = _hstOutputRA;
			hstFieldPhaseTo[ (field * 2) + 1 ] = _hstOutputDEC;
		}

	// create a phase correction object, and do phase correction on the samples.
	PhaseCorrection phaseCorrection;
	doPhaseCorrectionSamples(	/* pPhaseCorrection = */ &phaseCorrection,
					/* pNumSamples = */ hstNumSamples,
					/* pPhaseCentreIn = */ hstFieldPhaseFrom,
					/* pPhaseCentreOut = */ hstFieldPhaseTo,
					/* pSample = */ hstSample,
					/* pFieldID = */ hstSampleFieldID,
					/* pPhase = */ NULL );

	// calculate w-planes for this mosaic field.
	calculateWPlanes(	/* pMosaicIndex = */ pFileIndex,
				/* pNumSamples = */ hstNumSamples,
				/* phstSample = */ hstSample );

	// free the samples and field ids.
	if (hstSample != NULL)
		free( (void *) hstSample );
	if (hstSampleFieldID != NULL)
		free( (void *) hstSampleFieldID );

	// now we know the number of fields we can check if we need to reduce the oversampling parameter. There is a limit for texture mapping of 16384 kernels, and
	// we will need (oversampling x oversampling) kernels, or (oversampling x oversampling x # fields) for beam mosaicing.
	if (_hstBeamMosaic == true && (_hstOversample * _hstOversample * hstNumFields > 16384))
	{
		_hstOversample = (int) floor( sqrt( 16384.0 / hstNumFields ) );
		printf( "Oversampling factor has been reduced to %i because of the array size limit when using texture mapping\n\n", _hstOversample );
	}
	if (_hstBeamMosaic == false && (_hstOversample * _hstOversample > 16384))
	{
		_hstOversample = (int) floor( sqrt( 16384.0 ) );
		printf( "Oversampling factor has been reduced to %i because of the array size limit when using texture mapping\n\n", _hstOversample );
	}

	// if we're file mosaicing then copy the field phase position into the mosaic phase position so we can construct the mosaic at the end. we currently copy
	// the phase position of the first field since for file mosaicing all the fields in this file must be phase rotated to the same position.
	if (_hstFileMosaic == true)
	{
		phstImagePhasePosition[ 0 ] = hstFieldPhaseTo[ 0 ];
		phstImagePhasePosition[ 1 ] = hstFieldPhaseTo[ 1 ];
	}

	// upload the number of pws to the device.
	err = cudaMemcpyToSymbol( _devNumSpws, &numSpws, sizeof( numSpws ) );
	if (err != cudaSuccess)
		printf( "error copying number of spws to device (%s)\n", cudaGetErrorString( err ) );

	// ----------------------------------------------------
	//
	// l o a d   v i s i b i l i t i e s
	//
	// ----------------------------------------------------

	// count the number of antennae, and work out the number of baselines.
	int numberOfBaselines = (unflaggedAntennae * (unflaggedAntennae - 1)) / 2;

	// work out how much memory we need to load all the visibilities, flags, sample IDs and channel IDs.
	int memoryNeededPerVis = sizeof( cufftComplex ) + sizeof( bool ) + sizeof( int ) + sizeof( int );

	// ensure we're not using more than 10 GB.
	const long int MEMORY_LIMIT = (long int) 10 * (long int) 1073741824;

	if (hstMaxMemory > MEMORY_LIMIT)
		hstMaxMemory = MEMORY_LIMIT;
	printf( "we will load a maximum of %4.2f GB of data in each batch\n", (double) hstMaxMemory / (double) 1073741824 );

	// work out how many visibilities we can load before we go over the available memory, and how many baselines should be loaded with each stage.
	long int stagedVisibilities = hstMaxMemory / memoryNeededPerVis;
	int baselinesPerStage = (int) ((double) numberOfBaselines * ((double) stagedVisibilities / (double) numVisibilities));
	if (baselinesPerStage == 0)
		baselinesPerStage = 1;
	if (baselinesPerStage > numberOfBaselines)
		baselinesPerStage = numberOfBaselines;

	printf( "we will load data for %i of our %i baselines per batch\n\n", baselinesPerStage, numberOfBaselines );

	// set a flag which determines if we need to cache and uncache our data. if we have only one stage (and no multiple measurement sets) then we don't
	// need caching.
	_hstCacheData = _hstCacheData || (baselinesPerStage < numberOfBaselines);
	if (_hstCacheData == true)
		printf( "the data will be loaded in %i batches, and cached to disk\n\n", (int) ceil( (double) numberOfBaselines / (double) baselinesPerStage) );
	else
		printf( "the data will be loaded in one batch, and no disk caching will be used\n\n" );

	// we will count the number of visibilities that need to be gridded.
	_hstGriddedVisibilities[ pFileIndex ] = 0;

	// for beam mosaicing we will need to count the number of gridded visibilities per field, so we need to redimension the array.
	if (_hstBeamMosaic == true)
	{
		_hstGriddedVisibilitiesPerField = (long int *) malloc( hstNumFields * sizeof( long int ) );
		memset( _hstGriddedVisibilitiesPerField, 0, hstNumFields * sizeof( long int ) );
	}

	// we will need to work out the average weight in the gridded cells.
	_hstAverageWeight[ pFileIndex ] = 0.0;

	// clear the total weight per cell, but only if we're not making a UV mosaic. for uv mosaics this total will be added up over all mosaic components.
	if (_hstUVMosaic == false && (_hstWeighting == ROBUST || _hstWeighting == UNIFORM))
		memset( phstTotalWeightPerCell, 0, (long int) _hstUvPixels * (long int) _hstUvPixels * (long int) sizeof( double ) );

	printf( "Retrieving visibilities.....\n" );
	printf( "----------------------------\n\n" );
	clock_gettime( CLOCK_REALTIME, &time2 );

	// staging loop starts here.
	int sampleStageID = 0;
	int startAnt1 = 0, startAnt2 = 0, endAnt1 = 0, endAnt2 = 0;
	int stageID = 0;

	while (endAnt1 < unflaggedAntennae - 2 || endAnt2 < unflaggedAntennae - 1)
	{

		// calculate the start and end antennae for retrieving visibilities.
		startAnt1 = endAnt1;
		startAnt2 = endAnt2 + 1;
		if (startAnt2 == unflaggedAntennae)
		{
			startAnt1 += 1;
			startAnt2 = startAnt1 + 1;
		}
		endAnt1 = startAnt1; endAnt2 = startAnt2;
		for ( int i = 0; i < baselinesPerStage - 1; i++ )
		{

			// don't go beyond the last antenna.
			if (endAnt1 == unflaggedAntennae - 2 && endAnt2 == unflaggedAntennae - 1)
				break;

			// increment the antennae.
			endAnt2 = endAnt2 + 1;
			if (endAnt2 == unflaggedAntennae)
			{
				endAnt1 += 1;
				endAnt2 = endAnt1 + 1;
			}

		}

		printf( "\rloading visibilities.....%3d%%, processing visibilities.....%3d%%          currently: getting visibilities for antennae %i,%i to %i,%i",
				(int) ((double) sampleStageID * 100.0 / (double) hstNumSamples), (int) ((double) sampleStageID * 100.0 / (double) hstNumSamples),
				hstAntenna[ startAnt1 ], hstAntenna[ startAnt2 ], hstAntenna[ endAnt1 ], hstAntenna[ endAnt2 ] );
		fflush( stdout );

		// resize the array holding the number of batch records.
		_hstNumVisibilities[ pFileIndex ] = (long int *) realloc( _hstNumVisibilities[ pFileIndex ], (stageID + 1) * sizeof( long int ) );
		_hstNumberOfStages[ pFileIndex ] = stageID + 1;
		_hstNumberOfBatches[ pFileIndex ] = (int *) realloc( _hstNumberOfBatches[ pFileIndex ], _hstNumberOfStages[ pFileIndex ] * sizeof( int ) );

		// create temporary arrays for the visibilities, flags, weights and field ids.
		cufftComplex * tmpVisibility = NULL;
		bool * tmpFlag = NULL;
		float * tmpWeight = NULL;
		int * tmpDataDescID = NULL;

		// load visibilities. we load ALL the visibilities to the host, and these are then processed on the device in batches.
		int numSamplesInStage = 0;
		_hstCasacoreInterface.GetVisibilities(	/* IN: pFilename = */ pMeasurementSetFilename,
							/* IN: pFieldID = */ pFieldID,
							/* OUT: pNumSamples = */ &numSamplesInStage,
							/* IN: pNumChannels = */ hstNumChannels,
							/* IN: pDataField = */ _hstDataField[ pFileIndex ],
							/* OUT: pVisibility = */ (complex<float> **) &tmpVisibility,
							/* OUT: pFlag = */ &tmpFlag,
							/* OUT: pSample = */ (double **) &hstSample,
							/* OUT: pWeight = */ &tmpWeight,
							/* OUT: pFieldIDArray = */ &hstSampleFieldID,
							/* OUT: pDataDescID = */ &tmpDataDescID,
							/* IN: pNumPolarisations = */ hstNumPolarisations,
							/* IN: pStartAnt1 = */ hstAntenna[ startAnt1 ],
							/* IN: pStartAnt2 = */ hstAntenna[ startAnt2 ],
							/* IN: pEndAnt1 = */ hstAntenna[ endAnt1 ],
							/* IN: pEndAnt2 = */ hstAntenna[ endAnt2 ],
							/* IN: pNumberOfAntennae = */ numberOfAntennae,
							/* IN: pCurrentSample = */ sampleStageID / 2,
							/* IN: pTotalSamples = */ hstNumSamples / 2,
							/* IN: pNumSpws = */ numSpws,
							/* IN: pDataDescSPW = */ hstDataDescSpw,
							/* IN: pDataDescFlag = */ hstDataDescFlag,
							/* IN: pNumDataDesc = */ numDataDescItems,
							/* IN: pSpwChannelFlag = */ hstSpwChannelFlag );

		// set the polarisation config and spw from the data description id.
		int * hstSpw = (int *) malloc( numSamplesInStage * 2 * sizeof( int ) );
		int * hstPolarisationConfig = (int *) malloc( numSamplesInStage * 2 * sizeof( int ) );
		for ( int sample = 0; sample < numSamplesInStage; sample++ )
		{
			hstPolarisationConfig[ sample ] = hstDataDescPolarisationConfig[ tmpDataDescID[ sample ] ];
			hstSpw[ sample ] = hstDataDescSpw[ tmpDataDescID[ sample ] ];
		}

		// free data desc id.
		if (tmpDataDescID != NULL)
			free( (void *) tmpDataDescID );

		// process weights.
		float * hstSampleWeight = (float *) malloc( numSamplesInStage * 2 * sizeof( float ) );
		for ( int sample = 0; sample < numSamplesInStage; sample++ )
		{
			double weight = 0.0;
			for ( int polarisation = 0; polarisation < hstNumPolarisations; polarisation++ )
				weight += abs( hstMultiplier[ (hstPolarisationConfig[ sample ] * hstNumPolarisations) + polarisation ] ) *
								tmpWeight[ (sample * hstNumPolarisations) + polarisation ];
			hstSampleWeight[ sample ] = weight;
		}

		// free memory.
		if (tmpWeight != NULL)
			free( (void *) tmpWeight );

		// change size of sample and field ID arrays.
		hstSample = (VectorD *) realloc( hstSample, numSamplesInStage * 2 * sizeof( VectorD ) );
		hstSampleFieldID = (int *) realloc( hstSampleFieldID, numSamplesInStage * 2 * sizeof( int ) );

		// process field ID. we reassign the field ID value from the file to our new field ID (which runs 0, 1, 2, 3).
		for ( int sample = 0; sample < numSamplesInStage; sample++ )
			for ( int j = 0; j < hstNumFields; j++ )
				if (fieldIDMap[ j ] == hstSampleFieldID[ sample ])
				{
					hstSampleFieldID[ sample ] = j;
					break;
				}

		// mirror samples, weights, field ids and Spws.
		long int sampleDestination = numSamplesInStage;
		for ( long int i = 0; i < numSamplesInStage; i++, sampleDestination++ )
		{
			hstSample[ sampleDestination ].u = -hstSample[ i ].u;
			hstSample[ sampleDestination ].v = -hstSample[ i ].v;
			hstSample[ sampleDestination ].w = -hstSample[ i ].w;
		}
		memcpy( &hstSampleWeight[ numSamplesInStage ], hstSampleWeight, numSamplesInStage * sizeof( float ) );
		memcpy( &hstSampleFieldID[ numSamplesInStage ], hstSampleFieldID, numSamplesInStage * sizeof( int ) );
		memcpy( &hstSpw[ numSamplesInStage ], hstSpw, numSamplesInStage * sizeof( int ) );
		memcpy( &hstPolarisationConfig[ numSamplesInStage ], hstPolarisationConfig, numSamplesInStage * sizeof( int ) );

		// we have now got twice as many samples in the stage.
		numSamplesInStage *= 2;

		// calculate how many visibilities we have to process here. this total will be reduced later once we have compacted our data.
		_hstNumVisibilities[ pFileIndex ][ stageID ] = 0;
		for ( int sample = 0; sample < numSamplesInStage; sample++ )
			_hstNumVisibilities[ pFileIndex ][ stageID ] += (long int) hstUnflaggedChannels[ hstSpw[ sample ] ];

		// declare some memory for the sample ID and channel ID for these visibilities.
		_hstSampleID = (int *) malloc( _hstNumVisibilities[ pFileIndex ][ stageID ] * sizeof( int ) / 2 );
		_hstChannelID = (int *) malloc( _hstNumVisibilities[ pFileIndex ][ stageID ] * sizeof( int ) / 2 );

		// set the sample id and channel id for each visibility.
		long int visibilityID = 0;
		for ( int sample = 0; sample < numSamplesInStage / 2; sample++ )
			for ( int channel = 0; channel < hstNumChannels[ hstSpw[ sample ] ]; channel++ )
				if (hstSpwChannelFlag[ hstSpw[ sample ] ][ channel ] == false)
				{
					_hstSampleID[ visibilityID ] = sample;
					_hstChannelID[ visibilityID ] = channel;
					visibilityID++;
				}

		// redimension arrays.
		_hstSampleID = (int *) realloc( _hstSampleID, _hstNumVisibilities[ pFileIndex ][ stageID ] * sizeof( int ) );
		_hstChannelID = (int *) realloc( _hstChannelID, _hstNumVisibilities[ pFileIndex ][ stageID ] * sizeof( int ) );
		tmpVisibility = (cufftComplex *) realloc( tmpVisibility, _hstNumVisibilities[ pFileIndex ][ stageID ] * hstNumPolarisations * sizeof( cufftComplex ) / 2 );
		tmpFlag = (bool *) realloc( tmpFlag, _hstNumVisibilities[ pFileIndex ][ stageID ] * hstNumPolarisations * sizeof( bool ) / 2 );

		// duplicate samples and channels, and then update the sample ID for the second half of the array.
		memcpy( &_hstSampleID[ _hstNumVisibilities[ pFileIndex ][ stageID ] / 2 ], _hstSampleID,
				_hstNumVisibilities[ pFileIndex ][ stageID ] * sizeof( int ) / 2 );
		memcpy( &_hstChannelID[ _hstNumVisibilities[ pFileIndex ][ stageID ] / 2 ], _hstChannelID,
				_hstNumVisibilities[ pFileIndex ][ stageID ] * sizeof( int ) / 2 );
		for ( long int visibility = _hstNumVisibilities[ pFileIndex ][ stageID ] / 2; visibility < _hstNumVisibilities[ pFileIndex ][ stageID ]; visibility++ )
			_hstSampleID[ visibility ] += numSamplesInStage / 2;

		// perform phase rotation on samples.
		double * hstPhase = (double *) malloc( numSamplesInStage * sizeof( double ) );
		{
			PhaseCorrection phaseCorrection;
			doPhaseCorrectionSamples(	/* pPhaseCorrection = */ &phaseCorrection,
							/* pNumSamples = */ numSamplesInStage,
							/* pPhaseCentreIn = */ hstFieldPhaseFrom,
							/* pPhaseCentreOut = */ hstFieldPhaseTo,
							/* pSample = */ hstSample,
							/* pFieldID = */ hstSampleFieldID,
							/* pPhase = */ hstPhase );
		}

		// if the number of visibilities is greater than the maximum number then we are going to set a smaller batch size, and load these
		// visibilities in batches.
		int hstVisibilityBatchSize = 0;
		{
			long int nextBatchSize = _hstNumVisibilities[ pFileIndex ][ stageID ] / 2;
			if (nextBatchSize > _hstPreferredVisibilityBatchSize)
				nextBatchSize = _hstPreferredVisibilityBatchSize;
			hstVisibilityBatchSize = (int) nextBatchSize;
		}

		cufftComplex * devVisibilityIn = NULL;
		cufftComplex * devVisibilityOut = NULL;
		bool * devFlagIn = NULL, * devFlagOut = NULL;
		int * devPolarisationConfig = NULL;
		int * devSampleID = NULL;

		reserveGPUMemory( (void **) &devVisibilityIn, hstNumPolarisations * hstVisibilityBatchSize * sizeof( cufftComplex ),
					"reserving device memory for loading visibilities" );
		reserveGPUMemory( (void **) &devVisibilityOut, hstVisibilityBatchSize * sizeof( cufftComplex ), "reserving device memory for the processed visibilities" );
		reserveGPUMemory( (void **) &devFlagIn, hstNumPolarisations * hstVisibilityBatchSize * sizeof( bool ), "reserving device memory for loading flags" );
		reserveGPUMemory( (void **) &devFlagOut, hstVisibilityBatchSize * sizeof( bool ), "reserving device memory for the processed flags" );
		reserveGPUMemory( (void **) &devPolarisationConfig, numSamplesInStage * sizeof( int ), "reserving device memory for the polarisation config id" );
		reserveGPUMemory( (void **) &devSampleID, hstVisibilityBatchSize * sizeof( int ), "creating device memory for the sample ID" );

		// upload the polarisation config ids to the device.
		err = cudaMemcpy( devPolarisationConfig, hstPolarisationConfig, numSamplesInStage * sizeof( int ), cudaMemcpyHostToDevice );
		if (hstPolarisationConfig != NULL)
			free( (void *) hstPolarisationConfig );

		// create the visibility and flag arrays to store the data.
		_hstVisibility = (cufftComplex *) malloc( _hstNumVisibilities[ pFileIndex ][ stageID ] * sizeof( cufftComplex ) );
		_hstFlag = (bool *) malloc( _hstNumVisibilities[ pFileIndex ][ stageID ] * sizeof( bool ) );

		// keep looping until we have calculated all the visibilities and flags.
		long int hstCurrentVisibility = 0;
		while (hstCurrentVisibility < (_hstNumVisibilities[ pFileIndex ][ stageID ] / 2))
		{

			// if the number of remaining visibilities is lower than the visibility batch size, then reduce the visibility batch size accordingly.
			if ((_hstNumVisibilities[ pFileIndex ][ stageID ] / 2) - hstCurrentVisibility < hstVisibilityBatchSize)
				hstVisibilityBatchSize = (_hstNumVisibilities[ pFileIndex ][ stageID ] / 2) - hstCurrentVisibility;

			// upload the visibility batch size to the device.
			err = cudaMemcpyToSymbol( _devVisibilityBatchSize, &hstVisibilityBatchSize, sizeof( hstVisibilityBatchSize ) );
			if (err != cudaSuccess)
				printf( "error copying visibility batch size to device (%s)\n", cudaGetErrorString( err ) );

			// upload the visibilities to the device.
			moveHostToDevice( (void *) devVisibilityIn, (void *) &tmpVisibility[ hstNumPolarisations * hstCurrentVisibility ],
						hstNumPolarisations * hstVisibilityBatchSize * sizeof( cufftComplex ), "copying loaded visibilities to device" );

			// upload the flags to the device.
			moveHostToDevice( (void *) devFlagIn, (void *) &tmpFlag[ hstNumPolarisations * hstCurrentVisibility ],
						hstNumPolarisations * hstVisibilityBatchSize * sizeof( bool ), "copying loaded flags to device" );

			// upload the sample IDs to the device.
			moveHostToDevice( (void *) devSampleID, (void *) &_hstSampleID[ hstCurrentVisibility ],
						hstVisibilityBatchSize * sizeof( int ), "copying sample ID to device" );

			int threads = hstVisibilityBatchSize;
			int blocks = 1;
			setThreadBlockSize1D( &threads, &blocks );

			// calculate the visibilities and their flags.
			devCalculateVisibilityAndFlag<<< blocks, threads >>>(	/* IN: pVisibilityIn = */ devVisibilityIn,
										/* OUT: pVisibilityOut = */ devVisibilityOut,
										/* IN: pFlagIn = */ devFlagIn,
										/* OUT: pFlagOut = */ devFlagOut,
										/* IN: pNumPolarisations = */ hstNumPolarisations,
										/* IN: pMultiplier = */ devMultiplier,
										/* IN: pPolarisationConfig = */ devPolarisationConfig,
										/* IN: pSampleID = */ devSampleID );

			// copy visibilities and flags out.
			moveDeviceToHost( (void *) &_hstVisibility[ hstCurrentVisibility ], (void *) devVisibilityOut,
						hstVisibilityBatchSize * sizeof( cufftComplex ), "copying calculated visibility from device" );
			moveDeviceToHost( (void *) &_hstFlag[ hstCurrentVisibility ], (void *) devFlagOut,
						hstVisibilityBatchSize * sizeof( bool ), "copying calculated flags from device" );
		
			// get the next batch of data.
			hstCurrentVisibility = hstCurrentVisibility + hstVisibilityBatchSize;

		}

		// free workspace memory.
		if (devVisibilityIn != NULL)
			cudaFree( (void *) devVisibilityIn );
		if (devVisibilityOut != NULL)
			cudaFree( (void *) devVisibilityOut );
		if (devFlagIn != NULL)
			cudaFree( (void *) devFlagIn );
		if (devFlagOut != NULL)
			cudaFree( (void *) devFlagOut );
		if (devPolarisationConfig != NULL)
			cudaFree( (void *) devPolarisationConfig );
		if (devSampleID != NULL)
			cudaFree( (void *) devSampleID );
		if (tmpVisibility != NULL)
			free( (void *) tmpVisibility );
		if (tmpFlag != NULL)
			free( (void *) tmpFlag );

		// duplicate visibilities. the conjugates will be calculated on the GPU.
		memcpy( &_hstVisibility[ _hstNumVisibilities[ pFileIndex ][ stageID ] / 2 ], _hstVisibility, (_hstNumVisibilities[ pFileIndex ][ stageID ] / 2) * sizeof( cufftComplex ) );
		memcpy( &_hstFlag[ _hstNumVisibilities[ pFileIndex ][ stageID ] / 2 ], _hstFlag, (_hstNumVisibilities[ pFileIndex ][ stageID ] / 2) * sizeof( bool ) );

		// ----------------------------------------------------
		//
		// c a l c u l a t e   g r i d   p o s i t i o n s
		//
		// ----------------------------------------------------

		// if the number of visibilities is greater than the maximum number then we are going to set a smaller batch size, and load these
		// visibilities in batches.
		{
			long int nextBatchSize = _hstNumVisibilities[ pFileIndex ][ stageID ];
			if (nextBatchSize > _hstPreferredVisibilityBatchSize)
				nextBatchSize = _hstPreferredVisibilityBatchSize;
			hstVisibilityBatchSize = (int) nextBatchSize;
		}

		// create space for some samples on the device.
		VectorD * devSample = NULL;
		reserveGPUMemory( (void **) &devSample, numSamplesInStage * sizeof( VectorD ), "creating device memory for the samples" );

		// create a host array pointing to wavelengths on the device.
		double ** hstdevWavelength = (double **) malloc( numSpws * sizeof( double * ) );
		for ( int spw = 0; spw < numSpws; spw++ )
			reserveGPUMemory( (void **) &hstdevWavelength[ spw ], hstNumChannels[ spw ] * sizeof( double ), "creating device memory for spw" );

		// create space for the wavelengths, spw ids,  w-plane limits, phases, visibilities, field ids and a-planes on the device.
		double ** devWavelength = NULL;
		int * devSpw = NULL;
		double * devWPlaneMax = NULL;
		double * devPhase = NULL;
		cufftComplex * devVisibility = NULL;
		int * devField = NULL;
		int * devAPlane = NULL;
		int * hstAPlane = NULL;
		int * devChannelID = NULL;
		reserveGPUMemory( (void **) &devWavelength, numSpws * sizeof( double * ), "creating device memory for the wavelengths" );
		reserveGPUMemory( (void **) &devSpw, numSamplesInStage * sizeof( int ), "creating device memory for the spw ids" );
		reserveGPUMemory( (void **) &devWPlaneMax, _hstWPlanes * sizeof( double ), "creating device memory for the w-plane limits" );
		reserveGPUMemory( (void **) &devPhase, numSamplesInStage * sizeof( double ), "creating device memory for the phases" );
		reserveGPUMemory( (void **) &devVisibility, hstVisibilityBatchSize * sizeof( cufftComplex ), "creating device memory for the visibilities" );
		if (_hstBeamMosaic == true)
			reserveGPUMemory( (void **) &devField, numSamplesInStage * sizeof( int ), "creating device memory for the field IDs" );
		if (_hstAProjection == true)
		{
			reserveGPUMemory( (void **) &devAPlane, hstVisibilityBatchSize * sizeof( int ), "reserving device memory for the a-planes" );
			hstAPlane = (int *) malloc( hstVisibilityBatchSize * sizeof( int ) );
		}
		reserveGPUMemory( (void **) &devSampleID, hstVisibilityBatchSize * sizeof( int ), "creating device memory for the sample ID" );
		reserveGPUMemory( (void **) &devChannelID, hstVisibilityBatchSize * sizeof( int ), "creating device memory for the channel ID" );

		// upload the samples to the device.
		moveHostToDevice( (void *) devSample, (void *) hstSample, numSamplesInStage * sizeof( VectorD ), "copying samples to the device" );

		// upload the wavelengths to the device.
		for ( int spw = 0; spw < numSpws; spw++ )
			moveHostToDevice( (void *) hstdevWavelength[ spw ], (void *) hstWavelength[ spw ], hstNumChannels[ spw ] * sizeof( double ),
						"copying wavelengths to device" );

		// upload the spw ids, wavelength pointers, w-plane limits and phases to the device.
		moveHostToDevice( (void *) devSpw, (void *) hstSpw, numSamplesInStage * sizeof( int ), "copying spw ids to the device" );
		moveHostToDevice( (void *) devWavelength, (void *) hstdevWavelength, numSpws * sizeof( double * ), "copying wavelength pointers to the device" );
		moveHostToDevice( (void *) devWPlaneMax, (void *) _hstWPlaneMax[ pFileIndex ], _hstWPlanes * sizeof( double ), "copying w-plane limits to the device" );
		moveHostToDevice( (void *) devPhase, (void *) hstPhase, numSamplesInStage * sizeof( double ), "copying phases to device" );

		// upload the field IDs to the device.
		if (_hstBeamMosaic == true)
			moveHostToDevice( (void *) devField, (void *) hstSampleFieldID, numSamplesInStage * sizeof( int ), "copying field IDs to the device" );

		// save the original number of visibilities.
		long int originalVisibilities = _hstNumVisibilities[ pFileIndex ][ stageID ];

		// create arrays for the grid positions, kernel indexes, density maps, weights, and field IDs.
		_hstGridPosition = (VectorI *) malloc( _hstNumVisibilities[ pFileIndex ][ stageID ] * sizeof( VectorI ) );
		_hstKernelIndex = (int *) malloc( _hstNumVisibilities[ pFileIndex ][ stageID ] * sizeof( int ) );
		_hstDensityMap = (int *) malloc( _hstNumVisibilities[ pFileIndex ][ stageID ] * sizeof( int ) );
		_hstWeight = (float *) malloc( _hstNumVisibilities[ pFileIndex ][ stageID ] * sizeof( float ) );
		_hstFieldIDArray = (int *) malloc( _hstNumVisibilities[ pFileIndex ][ stageID ] * sizeof( int ) );

		// keep looping until we have calculated the grid positions for all the UVWs.
		hstCurrentVisibility = 0;
		long int uncompactedVisibilityID = 0;
		while (hstCurrentVisibility < _hstNumVisibilities[ pFileIndex ][ stageID ])
		{
		
			printf( "\rloading visibilities.....%3d%%, processing visibilities.....%3d%%          currently: processing (calculating grid positions)             ",
				(int) ((double) (sampleStageID + numSamplesInStage) * 100.0 / (double) hstNumSamples),
				(int) ((double) (sampleStageID + _hstSampleID[ hstCurrentVisibility ]) * 100.0 / (double) hstNumSamples) );
			fflush( stdout );

			// if the number of remaining visibilities is lower than the visibility batch size, then reduce the visibility batch size accordingly.
			if (_hstNumVisibilities[ pFileIndex ][ stageID ] - hstCurrentVisibility < (long int) hstVisibilityBatchSize)
				hstVisibilityBatchSize = (int) (_hstNumVisibilities[ pFileIndex ][ stageID ] - hstCurrentVisibility);

			// update the weights and field ids of each visibility, and initialise density map to 1.
			for ( long int i = 0; i < hstVisibilityBatchSize; i++ )
			{

				_hstWeight[ hstCurrentVisibility + i ] = hstSampleWeight[ _hstSampleID[ hstCurrentVisibility + i ] ];
				_hstFieldIDArray[ hstCurrentVisibility + i ] = hstSampleFieldID[ _hstSampleID[ hstCurrentVisibility + i ] ];
				_hstDensityMap[ hstCurrentVisibility + i ] = 1;

				// set the a-planes if we're using a-projection.	
				if (_hstAProjection == true)
					hstAPlane[ i ] = whichAPlane[ hstSpw[ _hstSampleID[ hstCurrentVisibility + i ] ] ][ _hstChannelID[ hstCurrentVisibility + i ] ];

			}

			// upload the visibility batch size to the device.
			err = cudaMemcpyToSymbol( _devVisibilityBatchSize, &hstVisibilityBatchSize, sizeof( hstVisibilityBatchSize ) );
			if (err != cudaSuccess)
				printf( "error copying visibility batch size to device (%s)\n", cudaGetErrorString( err ) );

			// upload the visibilities, sample IDs and channel IDs to the device.
			moveHostToDevice( (void *) devVisibility, (void *) &_hstVisibility[ hstCurrentVisibility ],
						hstVisibilityBatchSize * sizeof( cufftComplex ), "copying visibilities to device" );
			moveHostToDevice( (void *) devSampleID, (void *) &_hstSampleID[ hstCurrentVisibility ],
						hstVisibilityBatchSize * sizeof( int ), "copying sample ID to device" );
			moveHostToDevice( (void *) devChannelID, (void *) &_hstChannelID[ hstCurrentVisibility ],
						hstVisibilityBatchSize * sizeof( int ), "copying channel ID to device" );

			// set a suitable thread and block size.
			int threads = hstVisibilityBatchSize;
			int blocks = 1;
			setThreadBlockSize1D( &threads, &blocks );

			// mirror the visibilities by taking the conjugate values for all the second half of the data set.
			devTakeConjugate<<< blocks, threads >>>(	/* IN/OUT: pVisibility = */ devVisibility,
									/* IN: pCurrentVisibility = */ uncompactedVisibilityID,
									/* IN: pNumVisibilities = */ originalVisibilities );
			err = cudaGetLastError();
			if (err != cudaSuccess)
				printf( "error taking conjugate values (%s)\n", cudaGetErrorString( err ) );

			// do phase correction on the visibilities.
			devPhaseCorrection<<< blocks, threads >>>(	/* IN/OUT: pVisibility = */ devVisibility,
									/* IN: pPhase = */ devPhase,
									/* IN: pWavelength = */ devWavelength,
									/* IN: pSpw = */ devSpw,
									/* IN: pSampleID = */ devSampleID,
									/* IN: pChannelID = */ devChannelID );
			err = cudaGetLastError();
			if (err != cudaSuccess)
				printf( "error doing phase correction (%s)\n", cudaGetErrorString( err ) );

			// download visibilities from the device.
			moveDeviceToHost( (void *) &_hstVisibility[ hstCurrentVisibility ], (void *) devVisibility,
						hstVisibilityBatchSize * sizeof( cufftComplex ), "copying visibilities from the device" );

			// copy the a-planes to the device.
			if (_hstAProjection == true)
				moveHostToDevice( (void *) devAPlane, (void *) hstAPlane, hstVisibilityBatchSize * sizeof( int ),
							"copying a-planes to the device" );

			// create memory for grid positions, and kernel indexes.
			VectorI * devGridPosition = NULL;
			reserveGPUMemory( (void **) &devGridPosition, hstVisibilityBatchSize * sizeof( VectorI ), "reserving device memory for grid positions" );
			int * devKernelIndex = NULL;
			reserveGPUMemory( (void **) &devKernelIndex, hstVisibilityBatchSize * sizeof( int ), "reserving device memory for kernel indexes" );

			// calculate the grid positions and kernel indexes.
			devCalculateGridPositions<<< blocks, threads >>>(	/* OUT: pGridPosition = */ devGridPosition,
										/* OUT: pKernelIndex = */ devKernelIndex,
										/* IN: pUvCellSize = */ _hstUvCellSize,
										/* IN: pOversample = */ _hstOversample,
										/* IN: pWPlanes = */ _hstWPlanes,
										/* IN: pAPlanes = */ _hstAPlanes,
										/* IN: pSample = */ devSample,
										/* IN: pWavelength = */ devWavelength,
										/* IN: pField = */ devField,
										/* IN: pSpw = */ devSpw,
										/* IN: pWPlaneMax = */ devWPlaneMax,
										/* IN: pAPlane = */ devAPlane,
										/* IN: pSampleID = */ devSampleID,
										/* IN: pChannelID = */ devChannelID,
										/* IN: pSize = */ _hstUvPixels );
			if (cudaGetLastError() != cudaSuccess)
				printf( "error calculating grid positions (%s)\n", cudaGetErrorString( err ) );

			// download these grid positions and kernel indexes to host memory.
			moveDeviceToHost( (void *) &_hstGridPosition[ hstCurrentVisibility ], (void *) devGridPosition,
						hstVisibilityBatchSize * sizeof( VectorI ), "copying grid positions to host" );
			moveDeviceToHost( (void *) &_hstKernelIndex[ hstCurrentVisibility ], (void *) devKernelIndex,
						hstVisibilityBatchSize * sizeof( int ), "copying kernel indexes to host" );

			// free memory.
			if (devGridPosition != NULL)
				cudaFree( (void *) devGridPosition );
			if (devKernelIndex != NULL)
				cudaFree( (void *) devKernelIndex );

			// compact the data so that items with a duplicate grid position are only gridded once.
			hstCurrentVisibility = compactData(	/* pTotalVisibilities = */ &_hstNumVisibilities[ pFileIndex ][ stageID ],
								/* pFirstVisibility = */ hstCurrentVisibility,
								/* pNumVisibilities = */ hstVisibilityBatchSize );

			// get the next batch of data.
			uncompactedVisibilityID += hstVisibilityBatchSize;

		}

		printf( "\rloading visibilities.....%3d%%, processing visibilities.....%3d%%          currently: processing (sorting)                         ",
			(int) ((double)(sampleStageID + numSamplesInStage) * 100.0 / (double) hstNumSamples),
			(int) ((double)(sampleStageID + numSamplesInStage) * 100.0 / (double) hstNumSamples) );
		fflush( stdout );

		// we need to sort data into order of W plane, kernel index, U value and V value.
		quickSortData(	/* pLeft = */ 0,
				/* pRight = */ _hstNumVisibilities[ pFileIndex ][ stageID ] - 1 );

		printf( "\rloading visibilities.....%3d%%, processing visibilities.....%3d%%          currently: processing (compacting)                      ",
			(int) ((double)( sampleStageID + numSamplesInStage) * 100.0 / (double) hstNumSamples),
			(int) ((double)( sampleStageID + numSamplesInStage) * 100.0 / (double) hstNumSamples) );
		fflush( stdout );

		// compact the data again so that items with a duplicate grid position are only gridded once.
		compactData(	/* pTotalVisibilities = */ &_hstNumVisibilities[ pFileIndex ][ stageID ],
				/* pFirstVisibility = */ 0,
				/* pNumVisibilities = */ _hstNumVisibilities[ pFileIndex ][ stageID ] );

		// count the number of gridded visibilities (using the density map). We use this figure for normalising our
		// images. for beam mosaics we do this later on a per-field basis.
		for ( long int visibilityIndex = 0; visibilityIndex < _hstNumVisibilities[ pFileIndex ][ stageID ]; visibilityIndex++ )
			if (	_hstGridPosition[ visibilityIndex ].u >= 0 && _hstGridPosition[ visibilityIndex ].u < _hstUvPixels &&
				_hstGridPosition[ visibilityIndex ].v >= 0 && _hstGridPosition[ visibilityIndex ].v < _hstUvPixels)
				_hstGriddedVisibilities[ pFileIndex ] += _hstDensityMap[ visibilityIndex ];

		// for beam mosaicing we count the number of gridded visibilities per field.
		if (_hstBeamMosaic == true)
			for ( long int visibilityIndex = 0; visibilityIndex < _hstNumVisibilities[ pFileIndex ][ stageID ]; visibilityIndex++ )
				if (	_hstGridPosition[ visibilityIndex ].u >= 0 && _hstGridPosition[ visibilityIndex ].u < _hstUvPixels &&
					_hstGridPosition[ visibilityIndex ].v >= 0 && _hstGridPosition[ visibilityIndex ].v < _hstUvPixels)
					_hstGriddedVisibilitiesPerField[ _hstFieldIDArray[ visibilityIndex ] ] += _hstDensityMap[ visibilityIndex ];

		// if we're using uniform or robust weighting then add up the total weight in each grid cell (if we're using weighting).
		if (_hstWeighting == UNIFORM || _hstWeighting == ROBUST)
			for ( long int i = 0; i < _hstNumVisibilities[ pFileIndex ][ stageID ]; i++ )
				if (	_hstGridPosition[ i ].u >= 0 && _hstGridPosition[ i ].u < _hstUvPixels &&
					_hstGridPosition[ i ].v >= 0 && _hstGridPosition[ i ].v < _hstUvPixels)
					phstTotalWeightPerCell[ (_hstGridPosition[ i ].v * _hstUvPixels) + _hstGridPosition[ i ].u ] +=
											((double) _hstWeight[ i ] * (double) _hstDensityMap[ i ]);

		// if we're using natural weighting then add up the total weight in the whole grid.
		if (_hstWeighting == NATURAL)
			for ( long int i = 0; i < _hstNumVisibilities[ pFileIndex ][ stageID ]; i++ )
				if (	_hstGridPosition[ i ].u >= 0 && _hstGridPosition[ i ].u < _hstUvPixels &&
					_hstGridPosition[ i ].v >= 0 && _hstGridPosition[ i ].v < _hstUvPixels)
					_hstAverageWeight[ pFileIndex ] += ((double) _hstWeight[ i ] * (double) _hstDensityMap[ i ]);

		// save the visibility, grid position, kernel index, density map and weight.
		if (_hstCacheData == true)
			cacheData(	/* pFilenamePrefix = */ pFilenamePrefix,
					/* pMosaicID = */ pFileIndex,
					/* pBatchID = */ stageID,
					/* pWhatData = */ DATA_ALL );

		// free memory.
		if (devSample != NULL)
			cudaFree( (void *) devSample );
		if (devField != NULL)
			cudaFree( (void *) devField );
		if (devWavelength != NULL)
			cudaFree( (void *) devWavelength );
		if (devSpw != NULL)
			cudaFree( (void *) devSpw );
		if (hstdevWavelength != NULL)
		{
			for ( int spw = 0; spw < numSpws; spw++ )
				cudaFree( (void *) hstdevWavelength[ spw ] );
			free( (void *) hstdevWavelength );
		}
		if (devWPlaneMax != NULL)
			cudaFree( (void *) devWPlaneMax );
		if (devPhase != NULL)
			cudaFree( (void *) devPhase );
		if (devVisibility != NULL)
			cudaFree( (void *) devVisibility );
		if (devSampleID != NULL)
			cudaFree( (void *) devSampleID );
		if (devChannelID != NULL)
			cudaFree( (void *) devChannelID );
		if (devAPlane != NULL)
			cudaFree( (void *) devAPlane );
		if (hstAPlane != NULL)
			free( (void *) hstAPlane );
		if (hstSpw != NULL)
			free( (void *) hstSpw );
		if (hstSample != NULL)
			free( (void *) hstSample );
		if (hstSampleWeight != NULL)
			free( (void *) hstSampleWeight );
		if (hstSampleFieldID != NULL)
			free( (void *) hstSampleFieldID );
		if (hstPhase != NULL)
			free( (void *) hstPhase );
		if (_hstFlag != NULL)
		{
			free( (void *) _hstFlag );
			_hstFlag = NULL;
		}
		if (_hstFieldIDArray != NULL)
		{
			free( (void *) _hstFieldIDArray );
			_hstFieldIDArray = NULL;
		}
		if (_hstSampleID != NULL)
		{
			free( (void *) _hstSampleID );
			_hstSampleID = NULL;
		}
		if (_hstChannelID != NULL)
		{
			free( (void *) _hstChannelID );
			_hstChannelID = NULL;
		}

		// staging loop ends here.
		sampleStageID += numSamplesInStage;
		stageID++;

	}
	printf( "\rloading visibilities.....100%%, processing visibilities.....100%%          DONE                                                            \n\n" );

	// free memory.
	if (hstMultiplier != NULL)
		free( (void *) hstMultiplier );
	if (hstDataDescPolarisationConfig != NULL)
		free( (void *) hstDataDescPolarisationConfig );
	if (hstDataDescSpw != NULL)
		free( (void *) hstDataDescSpw );
	if (hstDataDescFlag != NULL)
		free( (void *) hstDataDescFlag );
	if (whichAPlane != NULL)
	{
		for ( int spw = 0; spw < numSpws; spw++ )
			if (whichAPlane[ spw ] != NULL)
				free( (void *) whichAPlane[ spw ] );
		free( (void *) whichAPlane );
	}
	if (devMultiplier != NULL)
		cudaFree( (void *) devMultiplier );
	if (fieldIDMap != NULL)
		free( (void *) fieldIDMap );
	if (hstWavelength != NULL)
	{
		for ( int spw = 0; spw < numSpws; spw++ )
			free( hstWavelength[ spw ] );
		free( (void *) hstWavelength );
	}
	if (hstAntenna != NULL)
		free( (void *) hstAntenna );

	// see if we can merge files together.
	if (_hstNumberOfStages[ pFileIndex ] > 1)
	{

		int recordSize = sizeof( cufftComplex ) + sizeof( VectorI ) + sizeof( int ) + sizeof( int ) + sizeof( float ) + sizeof( cufftComplex );

		// calculate maximum number of visibilities to hold in ram.
		long int maxVis = hstMaxMemory / (long int) recordSize;

		// see if each stage can be merged with the stage that came before it.
		int numberMerged = 1;
		for ( int stage = _hstNumberOfStages[ pFileIndex ] - 1; stage >= 1; stage-- )
			if (_hstNumVisibilities[ pFileIndex ][ stage ] + _hstNumVisibilities[ pFileIndex ][ stage - 1 ] <= maxVis)
			{

				// this stage can be merged with the previous stage.
				mergeData(	/* pFilenamePrefix = */ pFilenamePrefix,
						/* pMosaicID = */ pFileIndex,
						/* pStageID_one = */ stage - 1,
						/* pStageID_two = */ stage,
						/* pLoadAllData = */ (numberMerged == 1),
						/* pWhatData = */ DATA_VISIBILITIES | DATA_GRID_POSITIONS | DATA_KERNEL_INDEXES | DATA_DENSITIES | DATA_WEIGHTS );
				numberMerged++;

			}
			else if (numberMerged > 1)
			{

				printf( "merging %i data caches\n", numberMerged );

				// sort the data.
				quickSortData(	/* pLeft = */ 0,
						/* pRight = */ _hstNumVisibilities[ pFileIndex ][ stage ] - 1 );

				// compact the data.
				compactData(	/* pTotalVisibilities = */ &_hstNumVisibilities[ pFileIndex ][ stage ],
						/* pFirstVisibility = */ 0,
						/* pNumVisibilities = */ _hstNumVisibilities[ pFileIndex ][ stage ] );

				// we have some data open which we need to save.
				cacheData(	/* pFilenamePrefix = */ pFilenamePrefix,
						/* pMosaicID = */ pFileIndex,
						/* pStageID = */ stage,
						/* pWhatData = */ DATA_VISIBILITIES | DATA_GRID_POSITIONS | DATA_KERNEL_INDEXES | DATA_DENSITIES | DATA_WEIGHTS );
				numberMerged = 1;

			}
		if (numberMerged > 1)
		{

			printf( "merging %i data caches\n", numberMerged );

			// sort the data.
			quickSortData(	/* pLeft = */ 0,
					/* pRight = */ _hstNumVisibilities[ pFileIndex ][ 0 ] - 1 );

			// compact the data.
			compactData(	/* pTotalVisibilities = */ &_hstNumVisibilities[ pFileIndex ][ 0 ],
					/* pFirstVisibility = */ 0,
					/* pNumVisibilities = */ _hstNumVisibilities[ pFileIndex ][ 0 ] );

			_hstCacheData = (_hstNumberOfStages[ pFileIndex ] > 1 || _hstMeasurementSets > 1);

			// we have some data open which we need to save.
			if (_hstCacheData == true)
				cacheData(	/* pFilenamePrefix = */ pFilenamePrefix,
						/* pMosaicID = */ pFileIndex,
						/* pStageID = */ 0,
						/* pWhatData = */ DATA_VISIBILITIES | DATA_GRID_POSITIONS | DATA_KERNEL_INDEXES | DATA_DENSITIES | DATA_WEIGHTS );

		}

	}

	clock_gettime( CLOCK_REALTIME, &time1 );
	fprintf( stderr, "--- time (load and process visibilities): (%f ms) ---\n\n", getTime( time2, time1 ) );

	// for uniform weighting update the weight using the density map.
	if (_hstWeighting == UNIFORM && _hstUVMosaic == false)
		_hstAverageWeight[ pFileIndex ] = performUniformWeighting(	/* pFilenamePrefix = */ pFilenamePrefix,
										/* pMosaicID = */ pFileIndex,
										/* phstTotalWeightPerCell = */ phstTotalWeightPerCell );

	// for robust weighting we need to calculate the average cell weighting, and then the parameter f^2.
	if (_hstWeighting == ROBUST && _hstUVMosaic == false)
		_hstAverageWeight[ pFileIndex ] = performRobustWeighting(	/* pFilenamePrefix = */ pFilenamePrefix,
										/* pMosaicID = */ pFileIndex,
										/* phstTotalWeightPerCell = */ phstTotalWeightPerCell );

	// for natural weighting, we have already summed the weight and now need to calculate the average weight.
	if (_hstWeighting == NATURAL)
		_hstAverageWeight[ pFileIndex ] /= (double) _hstGriddedVisibilities[ pFileIndex ];

	// how many primary beams are needed for this file ?
	int hstPrimaryBeamsForFile = 1;
	if (_hstAProjection == true || _hstBeamMosaic == true)
		hstPrimaryBeamsForFile = hstNumFields;

	// create some more primary beam space.
	_hstPrimaryBeam[ pFileIndex ] = (float *) malloc( hstPrimaryBeamsForFile * _hstBeamSize * _hstBeamSize * sizeof( float ) );

	// create a reprojection object.
	Reprojection imagePlaneReprojection;

	// create two workspace primary beams on the device.
	float * devInBeam = NULL;
	float * devOutBeam = NULL;
	reserveGPUMemory( (void **) &devInBeam, _hstBeamSize * _hstBeamSize * sizeof( float ), "reserving memory for the input primary beam on the device" );
	reserveGPUMemory( (void **) &devOutBeam, _hstBeamSize * _hstBeamSize * sizeof( float ), "reserving memory for the output primary beam on the device" );

	// create the device memory needed by the reprojection code.
	Reprojection::rpVectI outSize = { /* x = */ _hstBeamSize, /* y = */ _hstBeamSize };
	imagePlaneReprojection.CreateDeviceMemory( outSize );

	// we need to do an image-plane reprojection of the beam to the common phase position, scaling the beams in the process so that they are
	// the same size as our images.
	// the beamScalingFactor is used for when the beam is loaded from file, and needs to be scaled to the required size. if we generated the beam then it will
	// already be the required size.
	for ( int beam = 0; beam < hstPrimaryBeamsForFile; beam++ )
	{ 

		printf( "Reprojecting beam %i to the new phase position\n", beam );
		imagePlaneReprojectPrimaryBeam(	/* pPrimaryBeamIn = */ hstPrimaryBeamIn,
						/* pPrimaryBeamOut = */ &_hstPrimaryBeam[ pFileIndex ][ beam * _hstBeamSize * _hstBeamSize ],
						/* pBeam = */ beam,
						/* pInRA = */ hstFieldPhaseFrom[ beam * 2 ],
						/* pInDec = */ hstFieldPhaseFrom[ (beam * 2) + 1 ],
						/* pOutRA = */ hstFieldPhaseTo[ beam * 2 ],
						/* pOutDec = */ hstFieldPhaseTo[ (beam * 2) + 1 ],
						/* pdevInBeam = */ devInBeam,
						/* pdevOutBeam = */ devOutBeam,
						/* pBeamCellSize = */ _hstBeamCellSize * hstWavelengthForBeam * _hstBeamFrequency / CONST_C,
						/* pImagePlaneReprojection = */ imagePlaneReprojection,
						/* pVerbose = */ true );

	}

	// create some more space for the A-projection primary beams.
	if (_hstAProjection == true)
		_hstPrimaryBeamAProjection[ pFileIndex ] = (float *) malloc( hstPrimaryBeamsForFile * _hstAPlanes * _hstBeamSize * _hstBeamSize * sizeof( float ) );

	// if we're doing A-projection then we need to reproject the primary beam for size as well as possibly position.
	if (_hstAProjection == true)
	{

		printf( "\n\rReprojecting beams for %i a-planes.....", _hstAPlanes );
		fflush( stdout );

		for ( int beam = 0; beam < hstPrimaryBeamsForFile; beam++ )
		{

			printf( "\rReprojecting beams for %i a-planes.....%i%%", _hstAPlanes, beam * 100 / hstPrimaryBeamsForFile );
			fflush( stdout );
			for ( int aPlane = 0; aPlane < _hstAPlanes; aPlane++ )
				imagePlaneReprojectPrimaryBeam(	/* pPrimaryBeamIn = */ hstPrimaryBeamIn,
								/* pPrimaryBeamOut = */ &_hstPrimaryBeamAProjection[ pFileIndex ][ ((beam * _hstAPlanes) + aPlane) *
															_hstBeamSize * _hstBeamSize ],
								/* pBeam = */ beam,
								/* pInRA = */ hstFieldPhaseFrom[ beam * 2 ],
								/* pInDec = */ hstFieldPhaseFrom[ (beam * 2) + 1 ],
								/* pOutRA = */ hstFieldPhaseTo[ beam * 2 ],
								/* pOutDec = */ hstFieldPhaseTo[ (beam * 2) + 1 ],
								/* pdevInBeam = */ devInBeam,
								/* pdevOutBeam = */ devOutBeam,
								/* pBeamCellSize = */ _hstBeamCellSize * hstAPlaneWavelength[ aPlane ] * _hstBeamFrequency / CONST_C,
								/* pImagePlaneReprojection = */ imagePlaneReprojection,
								/* pVerbose = */ false );

		}

		printf( "\rReprojecting beams for %i a-planes.....100%%\n", _hstAPlanes );

	}

	// update the number of primary beams with the extra ones we generated for this measurement set.
	if (_hstBeamMosaic == true)
		_hstBeamMosaicComponents = hstPrimaryBeamsForFile;

	// free memory.
	if (devInBeam != NULL)
		cudaFree( (void *) devInBeam );
	if (devOutBeam != NULL)
		cudaFree( (void *) devOutBeam );
	if (hstSpwChannelFlag != NULL)
	{
		for ( int spw = 0; spw < numSpws; spw++ )
			free( (void *) hstSpwChannelFlag[ spw ] );
		free( (void *) hstSpwChannelFlag );
	}
	if (hstPrimaryBeamIn != NULL)
		free( (void *) hstPrimaryBeamIn );
	if (hstFieldPhaseFrom != NULL)
		free( (void *) hstFieldPhaseFrom );
	if (hstFieldPhaseTo != NULL)
		free( (void *) hstFieldPhaseTo );
	if (hstNumChannels != NULL)
		free( (void *) hstNumChannels );
	if (hstUnflaggedChannels != NULL)
		free( (void *) hstUnflaggedChannels );
	if (hstAPlaneWavelength != NULL)
		free( (void *) hstAPlaneWavelength );

} // processMeasurementSet

//
//	main()
//
//	CJS: 07/08/2015
//
//	Main processing.
//

int main( int pArgc, char ** pArgv )
{
	
	char DIRTY_BEAM_EXTENSION[] = "-dirty-beam";
	char CLEAN_BEAM_EXTENSION[] = "-clean-beam";
	char GRIDDED_EXTENSION[] = "-gridded";
	char DIRTY_IMAGE_EXTENSION[] = "-dirty-image";
	char CLEAN_IMAGE_EXTENSION[] = "-clean-image";
	char RESIDUAL_IMAGE_EXTENSION[] = "-residual-image";
	char DECONVOLUTION_EXTENSION[] = "-deconvolution";
	char PRIMARY_BEAM_PATTERN_EXTENSION[] = "-primarybeampattern";
	char MOSAIC_EXTENSION[] = "-mosaic";
	char FILE_EXTENSION[] = ".image";
	
	char outputDirtyBeamFilename[ 100 ];
	char outputCleanBeamFilename[ 100 ];
	char outputGriddedFilename[ 100 ];
	char outputDirtyImageFilename[ 100 ];
	char outputCleanImageFilename[ 100 ];
	char outputResidualImageFilename[ 100 ];
	char outputDeconvolutionFilename[ 100 ];
	char outputPrimaryBeamPatternFilename[ 100 ];
	char outputMosaicFilename[ 100 ];

	// timings. only used for development and debugging.
	cudaError_t err;
	
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
	getParameters( parameterFile );

	// check path name is valid.
	int length = strlen( _hstCacheLocation );
	if (_hstCacheLocation[ length - 1 ] != '/' && length > 0)
	{
		_hstCacheLocation[ length ] = '/';
		_hstCacheLocation[ length + 1 ] = '\0';
	}

	// check that a beam size and beam cell size have been specified.
	for ( int measurementSet = 0; measurementSet < _hstMeasurementSets; measurementSet++ )
	{
		if (_hstBeamPattern[ measurementSet ][0] != '\0' && _hstBeamSize <= 0)
		{
			printf( "ERROR: A beam pattern file has been supplied, but the size of the beam is not specified\n" );
			break;
		}
		if (_hstBeamPattern[ measurementSet ][0] != '\0' && _hstBeamFrequency < 0.0)
		{
			printf( "ERROR: A beam pattern file has been supplied, but the beam frequency has not been specified\n" );
			break;
		}
		if (_hstBeamPattern[ measurementSet ][0] != '\0' && _hstBeamCellSize < 0.0)
		{
			printf( "ERROR: A beam pattern file has been supplied, but the beam cell size has not been specified\n" );
			break;
		}
	}

	// ----------------------------------------------------
	
	// build output filenames.
	strcpy( outputCleanImageFilename, _hstOutputPrefix ); strcat( outputCleanImageFilename, CLEAN_IMAGE_EXTENSION ); strcat( outputCleanImageFilename, FILE_EXTENSION );
	strcpy( outputResidualImageFilename, _hstOutputPrefix ); strcat( outputResidualImageFilename, RESIDUAL_IMAGE_EXTENSION ); strcat( outputResidualImageFilename, FILE_EXTENSION );
	strcpy( outputDirtyBeamFilename, _hstOutputPrefix ); strcat( outputDirtyBeamFilename, DIRTY_BEAM_EXTENSION ); strcat( outputDirtyBeamFilename, FILE_EXTENSION );
	strcpy( outputCleanBeamFilename, _hstOutputPrefix ); strcat( outputCleanBeamFilename, CLEAN_BEAM_EXTENSION ); strcat( outputCleanBeamFilename, FILE_EXTENSION );
	strcpy( outputGriddedFilename, _hstOutputPrefix ); strcat( outputGriddedFilename, GRIDDED_EXTENSION ); strcat( outputGriddedFilename, FILE_EXTENSION );
	strcpy( outputDirtyImageFilename, _hstOutputPrefix ); strcat( outputDirtyImageFilename, DIRTY_IMAGE_EXTENSION ); strcat( outputDirtyImageFilename, FILE_EXTENSION );

	// count the GPUs.
	char * gpuString;
	char tmp[ 512 ];
	strcpy( tmp, _hstGPUParam );
	_hstNumGPUs = 0;
	while ((gpuString = strtok( (_hstNumGPUs > 0 ? NULL : tmp), "," )) != NULL)
		_hstNumGPUs++;

	// build a list of GPUs.
	if (_hstNumGPUs == 0)
	{
		_hstNumGPUs = 1;
		_hstGPU = (int *) malloc( sizeof( int ) );
		_hstGPU[ 0 ] = 0;
	}
	else
	{
		_hstGPU = (int *) malloc( _hstNumGPUs * sizeof( int ) );
		int i = 0;
		while ((gpuString = strtok( (i > 0 ? NULL : _hstGPUParam), "," )) != NULL)
		{
			_hstGPU[ i ] = atoi( gpuString );
			i++;
		}
	}
	
	// get some properties from the device.
	cudaDeviceProp gpuProperties;
	cudaGetDeviceProperties( &gpuProperties, _hstGPU[ 0 ] );
	_maxThreadsPerBlock = gpuProperties.maxThreadsPerBlock;
	_warpSize = gpuProperties.warpSize;
	int * maxGridSize = gpuProperties.maxGridSize;
	_gpuMemory = (long int) gpuProperties.totalGlobalMem;

	printf( "\nGIMAGE\n" );
	printf( "======\n\n" );

	printf( "GPU properties:\n" );
	printf( "---------------\n\n" );

	printf( "using %i GPU(s): ", _hstNumGPUs );
	for ( int i = 0; i < _hstNumGPUs; i++ )
	{
		if (i > 0)
			printf( "," );
		printf( "%i", _hstGPU[ i ] );
	}
	printf( "\n\n" );

	// set the device.
	cudaSetDevice( _hstGPU[ 0 ] );

	printf( "Device #: %i\n", _hstGPU[ 0 ] );
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
	if (_hstWeighting == NATURAL)
		printf( "NATURAL\n" );
	else if (_hstWeighting == UNIFORM)
		printf( "UNIFORM\n" );
	else if (_hstWeighting == ROBUST)
		printf( "ROBUST (R = %4.2f)\n", _hstRobust );
	else
		printf( "NONE\n" );
	if (_hstWProjection == true)
		printf( "w-projection: ON, %i W planes\n", _hstWPlanes );
	else
		printf( "w-projection: OFF\n" );
	if (_hstAProjection == true )
		printf( "a-projection: ON, %i A planes\n", _hstAPlanes );
	else
		printf( "a-projection: OFF\n" );
	printf( "oversampling factor: %i\n", _hstOversample );
	if (_hstStokes == STOKES_I)
		printf( "polarisation: STOKES_I\n" );
	else if (_hstStokes == STOKES_Q)
		printf( "polarisation: STOKES_Q\n" );
	else if (_hstStokes == STOKES_U)
		printf( "polarisation: STOKES_U\n" );
	else if (_hstStokes == STOKES_V)
		printf( "polarisation: STOKES_V\n" );
	printf( "kernel cutoff fraction: %f\n\n", _hstKernelCutoffFraction );

	// set a flag to determine if we're using multiple file mosaic (i.e. PAFs).
	bool useMultiFiles = (_hstMeasurementSets > 1);
	if (_hstMeasurementSets == 0)
		_hstMeasurementSets = 1;

	// if we are using multiple measurement sets then we need to cache our data.
	if (_hstMeasurementSets > 1)
		_hstCacheData = true;

	// calculate the size of the dirty beam. this must be the largest even-sized image up to the dirty image size.
	_hstPsfSize = _hstUvPixels;

	// we don't want the psf to be larger than 2048 x 2048.
	if (_hstPsfSize > 2048)
		_hstPsfSize = 2048;

	printf( "the size of the psf will be %i x %i pixels\n", _hstPsfSize, _hstPsfSize );
	printf( "the size of the primary beam will be %i x %i pixels\n\n", BEAM_SIZE, BEAM_SIZE );

	// turn on mosaicing if we are using multi files. we currently restrict this software to EITHER assembling a mosaic from the various files of a single
	// measurement set, OR assembling the images from multi files, with the same FOV and phase centre, into a single image. The latter is used for the multi-beams
	// of a PAF.
	_hstBeamMosaic = (_hstUseMosaicing == true && useMultiFiles == false && _hstMosaicDomain == UV);
	_hstUVMosaic = (_hstUseMosaicing == true && useMultiFiles == true && _hstMosaicDomain == UV);
	_hstFileMosaic = (_hstUseMosaicing == true && useMultiFiles == true && _hstMosaicDomain == IMAGE);
	if (_hstUseMosaicing == false && useMultiFiles == true)
	{
		printf( "loading multiple measurement sets, so I am turning mosaicing on.\n\n" );
		_hstUseMosaicing = true;
		if (_hstMosaicDomain == UV)
			_hstUVMosaic = true;
		else
			_hstFileMosaic = true;
	}

	printf( "Image properties:\n" );
	printf( "-----------------\n\n" );
	printf( "Telescope: " );
	switch (_hstTelescope)
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
	if (_hstBeamMosaic == true)
	{
		printf( "mosaic: Y (UV-plane mosaic based upon multiple fields)\n" );
		printf( "        fields - %s\n", _hstFieldID[ 0 ] );
	}
	else if (_hstFileMosaic == true)
		printf( "mosaic: Y (image-plane mosaic based upon multiple files)\n" );
	else if (_hstUVMosaic == true)
		printf( "mosaic: Y (UV-plane mosaic based upon multiple files)\n" );
	else
		printf( "mosaic: N\n" );
	printf( "phase position: <%f, %f>\n", _hstOutputRA, _hstOutputDEC );

	// get the total system memory.
	struct sysinfo memInfo;
	sysinfo( &memInfo );

	printf( "\nHost properties:\n" );
	printf( "----------------\n\n" );
	printf( "total physical memory: %4.2f GB\n", (double) memInfo.totalram / 1073741824.0 );

	// hold the required phase centre of each image. this array is only used for file mosaics.
	double * hstImagePhaseCentre = (double *) malloc( _hstMeasurementSets * 2 * sizeof( double ) );

	// set the number of mosaic images to the number of measurement sets (file mosaicing) or 1 (if not).
	_numMosaicImages = (_hstFileMosaic == true || _hstUVMosaic == true ? _hstMeasurementSets : 1);

	// create array to hold the batch info for each file.
	_hstNumVisibilities = (long int **) malloc( _numMosaicImages * sizeof( long int * ) );
	_hstNumberOfStages = (int *) malloc( _numMosaicImages * sizeof( int ) );
	_hstNumberOfBatches = (int **) malloc( _numMosaicImages * sizeof( int * ) );

	// store the number of gridded visibilities, and the average weights.
	_hstGriddedVisibilities = (long int *) malloc( _numMosaicImages * sizeof( long int ) );
	_hstAverageWeight = (double *) malloc( _numMosaicImages * sizeof( double ) );

	// create arrays to hold the W-plane parameters.
	_hstWPlaneMax = (double **) malloc( _numMosaicImages * sizeof( double * ) );
	_hstWPlaneMean = (double **) malloc( _numMosaicImages * sizeof( double * ) );

	// create array to hold the average wavelengths for the MS.
	_hstAverageWavelength = (double *) malloc( _numMosaicImages * sizeof( double ) );

	// initialise data arrays to null.
	for ( int i = 0; i < _numMosaicImages; i++ )
	{
		_hstWPlaneMax[ i ] = NULL;
		_hstWPlaneMean[ i ] = NULL;
	}

	// create some primary beam arrays.
	_hstPrimaryBeam = (float **) malloc( _hstMeasurementSets * sizeof( float * ) );
	_hstPrimaryBeamAProjection = (float **) malloc( _hstMeasurementSets * sizeof( float * ) );
	for ( int i = 0; i < _hstMeasurementSets; i++ )
	{
		_hstPrimaryBeam[ i ] = NULL;
		_hstPrimaryBeamAProjection[ i ] = NULL;
	}
	
	// calculate the size of each uv pixel.
	_hstUvCellSize = (1.0 / ((double) _hstUvPixels * (_hstCellSize / 3600.0) * (PI / 180.0)));
	
	// set the properties of the anti-aliasing kernel. we need a spare pixel on either side for oversampling.
	_hstAASupport = 3;
	_hstAAKernelSize = (2 * _hstAASupport) + 1;

	// calculate the number of kernel sets. each kernel set covers all the oversampled positions for a single W plane and A plane.
	_hstKernelSets = 1;
	if (_hstWProjection == true)
		_hstKernelSets = _hstKernelSets * _hstWPlanes;
	if (_hstAProjection == true)
		_hstKernelSets = _hstKernelSets * _hstAPlanes;

	// create memory to hold the kernel and support sizes.
	_hstKernelSize = (int *) malloc( _hstKernelSets * sizeof( int ) );
	_hstSupportSize = (int *) malloc( _hstKernelSets * sizeof( int ) );
	
	printf( "\nKernel properties:\n" );
	printf( "------------------\n\n" );
	printf( "aa-kernel support = %i\n", _hstAASupport );
	printf( "cell size = %f arcsec, %1.12f rad\n", _hstCellSize, (_hstCellSize / 3600) * (PI / 180) );
	printf( "uv cell size = %f\n\n", _hstUvCellSize );

	// build some of the output filenames.
	strcpy( outputMosaicFilename, _hstOutputPrefix ); strcat( outputMosaicFilename, MOSAIC_EXTENSION ); strcat( outputMosaicFilename, FILE_EXTENSION );
	strcpy( outputDeconvolutionFilename, _hstOutputPrefix ); strcat( outputDeconvolutionFilename, DECONVOLUTION_EXTENSION ); strcat( outputDeconvolutionFilename, FILE_EXTENSION );
	strcpy( outputPrimaryBeamPatternFilename, _hstOutputPrefix ); strcat( outputPrimaryBeamPatternFilename, PRIMARY_BEAM_PATTERN_EXTENSION ); strcat( outputPrimaryBeamPatternFilename, FILE_EXTENSION );

	// for uniform or robust weighting we need to store the sum of weights in each cell.
	double * hstTotalWeightPerCell = NULL;
	if (_hstWeighting == ROBUST || _hstWeighting == UNIFORM)
	{
		hstTotalWeightPerCell = (double *) malloc( (long int) _hstUvPixels * (long int) _hstUvPixels * (long int) sizeof( double ) );
		memset( hstTotalWeightPerCell, 0, (long int) _hstUvPixels * (long int) _hstUvPixels * (long int) sizeof( double ) );
	}

	// process the input measurement sets by loading the data and caching it.
	for ( int measurementSet = 0; measurementSet < _hstMeasurementSets; measurementSet++ )
		processMeasurementSet(	/* pFilenamePrefix = */ _hstOutputPrefix,
					/* pMeasurementSetFilename = */ _hstMeasurementSetPath[ measurementSet ],
					/* pFieldID = */ _hstFieldID[ measurementSet ],
					/* phstImagePhasePosition = */ &hstImagePhaseCentre[ measurementSet * 2 ],
					/* pFileIndex = */ measurementSet,
					/* pTableData = */ _hstTableData[ measurementSet ],
					/* phstTotalWeightPerCell = */ hstTotalWeightPerCell );

	// free data.
	if (_hstMeasurementSetPath != NULL)
	{
		for ( int i = 0; i < _hstMeasurementSets; i++ )
			if (_hstMeasurementSetPath[ i ] != NULL)
				free( (void *) _hstMeasurementSetPath[ i ] );
		free( (void *) _hstMeasurementSetPath );
	}
	if (_hstFieldID != NULL)
	{
		for ( int i = 0; i < _hstMeasurementSets; i++ )
			if (_hstFieldID[ i ] != NULL)
				free( (void *) _hstFieldID[ i ] );
		free( (void *) _hstFieldID );
	}
	if (_hstSpwRestriction != NULL)
	{
		for ( int i = 0; i < _hstMeasurementSets; i++ )
			if (_hstSpwRestriction[ i ] != NULL)
				free( (void *) _hstSpwRestriction[ i ] );
		free( (void *) _hstSpwRestriction );
	}
	if (_hstDataField != NULL)
	{
		for ( int i = 0; i < _hstMeasurementSets; i++ )
			if (_hstDataField[ i ] != NULL)
				free( (void *) _hstDataField[ i ] );
		free( (void *) _hstDataField );
	}
	if (_hstTableData != NULL)
	{
		for ( int i = 0; i < _hstMeasurementSets; i++ )
			if (_hstTableData[ i ] != NULL)
				free( (void *) _hstTableData[ i ] );
		free( (void *) _hstTableData );
	}
	if (_hstBeamPattern != NULL)
	{
		for ( int i = 0; i < _hstMeasurementSets; i++ )
			if (_hstBeamPattern[ i ] != NULL)
				free( (void *) _hstBeamPattern[ i ] );
		free( (void *) _hstBeamPattern );
	}

	// update the weighting for uv mosaics.
	if (_hstUVMosaic == true)
	{
		if (_hstWeighting == UNIFORM)
			_hstTotalAverageWeight = performUniformWeighting(	/* pFilenamePrefix = */ _hstOutputPrefix,
										/* pMosaicID = */ -1,
										/* phstTotalWeightPerCell = */ hstTotalWeightPerCell );
		if (_hstWeighting == ROBUST)
			_hstTotalAverageWeight = performRobustWeighting(	/* pFilenamePrefix = */ _hstOutputPrefix,
										/* pMosaicID = */ -1,
										/* phstTotalWeightPerCell = */ hstTotalWeightPerCell );
		if (_hstWeighting == NATURAL)
			_hstTotalAverageWeight = performNaturalWeighting(	/* pMosaicID = */ -1 );
	}

	// free memory.
	if (hstTotalWeightPerCell != NULL)
		free( (void *) hstTotalWeightPerCell );
	
	// generate image of convolution function.
	generateImageOfConvolutionFunction( outputDeconvolutionFilename );

	// we need to count how many visibilities are being gridded with each image/beam.
	if (_hstBeamMosaic == true)
	{

		// find the minimum number of gridded visibilities. we will use this figure for our normalisation pattern. Our beams with higher numbers of gridded
		// visibilities will be corrected for in the gridding kernel.
		_griddedVisibilitiesForBeamMosaic = 0;
		for ( int i = 0; i < _hstBeamMosaicComponents; i++ )
			if (i == 0 || _hstGriddedVisibilitiesPerField[ i ] < _griddedVisibilitiesForBeamMosaic)
				_griddedVisibilitiesForBeamMosaic = _hstGriddedVisibilitiesPerField[ i ];

		// ensure the number of gridded visibilities is not zero.
		if (_griddedVisibilitiesForBeamMosaic == 0)
			_griddedVisibilitiesForBeamMosaic = 1;

	}
	if (_hstUVMosaic == true)
	{

		// find the minimum number of gridded visibilities. we will use this figure for our normalisation pattern. Our beams with higher numbers of gridded
		// visibilities will be corrected for in the gridding kernel.
		_griddedVisibilitiesForBeamMosaic = 0;
		for ( int i = 0; i < _hstMeasurementSets; i++ )
			if (i == 0 || _hstGriddedVisibilities[ i ] < _griddedVisibilitiesForBeamMosaic)
				_griddedVisibilitiesForBeamMosaic = _hstGriddedVisibilities[ i ];

		// ensure the number of gridded visibilities is not zero.
		if (_griddedVisibilitiesForBeamMosaic == 0)
			_griddedVisibilitiesForBeamMosaic = 1;

		// set the number of beams.
		_hstBeamMosaicComponents = _hstMeasurementSets;

	}

	// we need to make an image mask.
	bool * hstMask = (bool *) malloc( (long int) _hstUvPixels * (long int) _hstUvPixels * sizeof( bool ) );

	// reserve some memory for the number of visibilities per kernel set.
	_hstVisibilitiesInKernelSet = (int *****) malloc( _numMosaicImages * sizeof( int **** ) );
	for ( int i = 0; i < _numMosaicImages; i++ )
		_hstVisibilitiesInKernelSet[ i ] = NULL;

	// ----------------------------------------------------
	//
	// c a l c u l a t e   m a s k   a n d   p r i m a r y
	// b e a m   p  a t t e r n s
	//
	// ----------------------------------------------------

	// if we are not using mosaicing then generate a mask for the image based upon the primary beam.
	if (_hstUseMosaicing == false)
	{

		// set the mask to false.
		memset( hstMask, 0, (long int) _hstUvPixels * (long int) _hstUvPixels * sizeof( bool ) );
		double imageScale = (double) _hstUvPixels / (double) _hstBeamSize;
		long int index = 0;
		for ( long j = 0; j < _hstUvPixels; j++ )
		{
			int beamJ = (int) ((double) j / imageScale);
			for ( int i = 0; i < _hstUvPixels; i++, index++ )
			{
				int beamI = (int) ((double) i / imageScale);
				hstMask[ index ] = (_hstPrimaryBeam[ /* mosaic index = */ 0 ][ (beamJ * _hstBeamSize) + beamI ] >= 0.2);
			}
		}

	}

	// if we are using UV-plane mosaicing then we need to assemble a weighted image of the primary beam patterns, and build a mask from it.
	if (_hstBeamMosaic == true || _hstUVMosaic == true)
	{

		// the primary beam pattern.
		_hstPrimaryBeamPattern = (float *) malloc( (long int) _hstBeamSize * (long int) _hstBeamSize * sizeof( float ) );
		memset( (void *) _hstPrimaryBeamPattern, 0, (long int) _hstBeamSize * (long int) _hstBeamSize * sizeof( float ) );

		// set the mask to false.
		memset( hstMask, 0, (long int) _hstUvPixels * (long int) _hstUvPixels * sizeof( bool ) );

		// calculate the sum of the primary beam squared.
		double maxValue = 0.0;
		for ( int index = 0; index < _hstBeamSize * _hstBeamSize; index++ )
		{

			for ( int beam = 0; beam < _hstBeamMosaicComponents; beam++ )
				if (_hstBeamMosaic == true)
					_hstPrimaryBeamPattern[ index ] += pow( _hstPrimaryBeam[ /* mosaic index = */ 0 ][ ((long int) beam *
											(long int) _hstBeamSize * (long int) _hstBeamSize) + (long int) index ], 2 );
				else
					_hstPrimaryBeamPattern[ index ] += pow( _hstPrimaryBeam[ /* mosaic index = */ beam ][ (long int) index ], 2 );
			_hstPrimaryBeamPattern[ index ] = sqrt( _hstPrimaryBeamPattern[ index ] );

			// record the maximum pixel value for normalisation.
			if (_hstPrimaryBeamPattern[ index ] > maxValue || index == 0)
				maxValue = _hstPrimaryBeamPattern[ index ];

		}

		// normalise the primary beam pattern.
		if (maxValue > 0.0)
			for ( long int i = 0; i < (long int) _hstBeamSize * (long int) _hstBeamSize; i++ )
				_hstPrimaryBeamPattern[ i ] /= maxValue;

		// set the mask.
		long int index = 0;
		for ( int j = 0; j < _hstUvPixels; j++ )
		{
			int beamJ = (int) ((double) j * (double) _hstBeamSize / (double) _hstUvPixels);
			for ( int i = 0; i < _hstUvPixels; i++, index++ )
			{
				int beamI = (int) ((double) i * (double) _hstBeamSize / (double) _hstUvPixels);
				hstMask[ index ] = (_hstPrimaryBeamPattern[ (beamJ * _hstBeamSize) + beamI ] >= 0.2);
			}
		}

		// we need to assemble a weighted image of the primary beam patterns (the normalisation pattern), which corrects for the fact each image has been
		// weighted by its dirty beam.
		_hstNormalisationPattern = (float *) malloc( (long int) _hstBeamSize * (long int) _hstBeamSize * (long int) sizeof( float ) );
		memset( (void *) _hstNormalisationPattern, 0, (long int) _hstBeamSize * (long int) _hstBeamSize * (long int) sizeof( float ) );

		for ( int index = 0; index < _hstBeamSize * _hstBeamSize; index++ )
		{
		
			// add up primary beam patterns and the normalisation factor. the normalisation factor is PB^2 because we need to correct for
			// the use of the PB as a weighting function whilst gridding, and also remove the effect of the PB which will naturally be in
			// our image.
			for ( int beam = 0; beam < _hstBeamMosaicComponents; beam++ )
			{
				float * beamPtr;
				if (_hstBeamMosaic == true)
					beamPtr = &_hstPrimaryBeam[ /* mosaic index = */ 0 ][ (long int) beam * (long int) _hstBeamSize * (long int) _hstBeamSize ];
				else
					beamPtr = _hstPrimaryBeam[ /* mosaic index = */ beam ];
				_hstNormalisationPattern[ index ] += pow( beamPtr[ index ], 1 );
			}

			// divide the normalisation pattern by the primary beam pattern in order than we smooth out the noise near the edges of the image.
			_hstNormalisationPattern[ index ] /= _hstPrimaryBeamPattern[ index ];

		}

	}

	// prepare an array to store the number of visibilities in each kernel set.
	for ( int image = 0; image < _numMosaicImages; image++ )
		_hstVisibilitiesInKernelSet[ image ] = (int ****) malloc( _hstNumberOfStages[ image ] * sizeof( int *** ) );

	// ----------------------------------------------------
	//
	// g r i d   v i s i b i l i t i e s
	//
	// ----------------------------------------------------

	cufftComplex ** devDirtyBeamGrid = (cufftComplex **) malloc( _hstNumGPUs * sizeof( cufftComplex * ) );
	if (_hstMinorCycles > 0)
	{

		// create memory for the psf on the device, and clear it.
		for ( int gpu = 0; gpu < _hstNumGPUs; gpu++ )
		{
			if (_hstNumGPUs > 1)
				cudaSetDevice( _hstGPU[ gpu ] );
			reserveGPUMemory( (void **) &devDirtyBeamGrid[ gpu ], _hstUvPixels * _hstUvPixels * sizeof( cufftComplex ), "declaring device memory for psf" );
			zeroGPUMemory( (void *) devDirtyBeamGrid[ gpu ], _hstUvPixels * _hstUvPixels * sizeof( cufftComplex ), "zeroing the dirty beam on the device" );
		}
		if (_hstNumGPUs > 1)
			cudaSetDevice( _hstGPU[ 0 ] );

		printf( "gridding visibilities for psf.....\n\n" );
		for ( int image = 0; image < _numMosaicImages; image++ )
		{

			// count the total number of visibilities.
			long int totalVisibilities = 0;
			for ( int stageID = 0; stageID < _hstNumberOfStages[ image ]; stageID++ )
				totalVisibilities += _hstNumVisibilities[ image ][ stageID ];

			if (_numMosaicImages > 1)
				printf( "        processing mosaic component %i of %i.....\n\n", image + 1, _numMosaicImages );
			printf( "                stages: %i\n", _hstNumberOfStages[ image ] );
			printf( "                visibilities: %li\n\n", totalVisibilities );

			// uncache the data for this mosaic image (if we have more than one field).
			long int visibilitiesProcessed = 0;
			for ( int stageID = 0; stageID < _hstNumberOfStages[ image ]; stageID++ )
			{

				// get the data from the file.
				if (_hstCacheData == true)
					uncacheData(	/* pFilenamePrefix = */ _hstOutputPrefix,
							/* pMosaicID = */ image,
							/* pStageID = */ stageID,
							/* pWhatData = */ DATA_GRID_POSITIONS | DATA_KERNEL_INDEXES | DATA_DENSITIES | DATA_WEIGHTS,
							/* pOffset = */ 0 );
	
				// if the number of visibilities is greater than some maximum then we are going to set a smaller batch size, and load these
				// visibilities in batches.
				int hstVisibilityBatchSize = 0;
				{
					long int nextBatchSize = _hstNumVisibilities[ image ][ stageID ];
					if (nextBatchSize > _hstPreferredVisibilityBatchSize)
						nextBatchSize = _hstPreferredVisibilityBatchSize;
					hstVisibilityBatchSize = (int) nextBatchSize;
				}

				// count the number of visibilities per kernel set for the whole stage.
				calculateVisibilitiesPerKernelSet(	/* pNumVisibilities = */_hstNumVisibilities[ image ][ stageID ],
									/* pBatchSize = */ hstVisibilityBatchSize,
									/* phstGridPosition = */ _hstGridPosition,
									/* phstVisibilitiesInKernelSet = */ &_hstVisibilitiesInKernelSet[ image ][ stageID ],
									/* pNumGPUs = */ _hstNumGPUs,
									/* pNumBatches = */ &_hstNumberOfBatches[ image ][ stageID ],
									/* pNumKernelSets = */ _hstKernelSets );

				// create space for the unity (psf) visibilities, the density map, and the weights on the device.
				cufftComplex ** devBeamVisibility = (cufftComplex **) malloc( _hstNumGPUs * sizeof( cufftComplex * ) );
				int ** devDensityMap = (int **) malloc( _hstNumGPUs * sizeof( int * ) );
				float ** devWeight = (float **) malloc( _hstNumGPUs * sizeof( float * ) );
				VectorI ** devGridPosition = (VectorI **) malloc( _hstNumGPUs * sizeof( VectorI * ) );
				int ** devKernelIndex = (int **) malloc( _hstNumGPUs * sizeof( int * ) );
				for ( int gpu = 0; gpu < _hstNumGPUs; gpu++ )
				{
					if (_hstNumGPUs > 1)
						cudaSetDevice( _hstGPU[ gpu ] );
					reserveGPUMemory( (void **) &devBeamVisibility[ gpu ], hstVisibilityBatchSize * sizeof( cufftComplex ),
								"creating device memory for the psf visibilities" );
					reserveGPUMemory( (void **) &devDensityMap[ gpu ], hstVisibilityBatchSize * sizeof( int ),
								"declaring device memory for the density map" );
					reserveGPUMemory( (void **) &devGridPosition[ gpu ], hstVisibilityBatchSize * sizeof( VectorI ),
								"reserving device memory for grid positions" );
					reserveGPUMemory( (void **) &devKernelIndex[ gpu ], hstVisibilityBatchSize * sizeof( int ),
								"reserving device memory for kernel indexes" );
					if (_hstWeighting != NONE)
						reserveGPUMemory( (void **) &devWeight[ gpu ], hstVisibilityBatchSize * sizeof( float ),
									"declaring device memory for the weights" );

				}
				if (_hstNumGPUs > 1)
					cudaSetDevice( _hstGPU[ 0 ] );

				// create some memory for storing the number of visibilities per W plane.
				int ** hstVisibilitiesInWPlane = (int **) malloc( _hstWPlanes * sizeof( int * ) );
				for ( int wPlane = 0; wPlane < _hstWPlanes; wPlane++ )
					hstVisibilitiesInWPlane[ wPlane ] = (int *) malloc( _hstNumGPUs * sizeof( int ) );

				// keep looping until we have loaded and gridded all visibilities.
				int batch = 0;
				long int hstCurrentVisibility = 0;
				while (hstCurrentVisibility < _hstNumVisibilities[ image ][ stageID ])
				{

					int ** hstVisibilitiesInKernelSet = _hstVisibilitiesInKernelSet[ image ][ stageID ][ batch ];

					// count the number of visibilities in this batch.
					int visibilitiesInThisBatch = 0;
					for ( int kernelSet = 0; kernelSet < _hstKernelSets; kernelSet++ )
						for ( int gpu = 0; gpu < _hstNumGPUs; gpu++ )
							visibilitiesInThisBatch += hstVisibilitiesInKernelSet[ kernelSet ][ gpu ];

					if (_hstNumberOfStages[ image ] > 1 || _hstNumberOfBatches[ image ][ stageID ] > 1)
						printf( "        gridding " );
					if (_hstNumberOfStages[ image ] > 1)
						printf( "host batch %i of %i", stageID + 1, _hstNumberOfStages[ image ] );
					if (_hstNumberOfStages[ image ] > 1 && _hstNumberOfBatches[ image ][ stageID ] > 1)
						printf( ", " );
					if (_hstNumberOfBatches[ image ][ stageID ] > 1)
						printf( "gpu batch %i of %i", batch + 1, _hstNumberOfBatches[ image ][ stageID ] );
					if (_hstNumberOfStages[ image ] > 1 || _hstNumberOfBatches[ image ][ stageID ] > 1)
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

					// calculate the number of visibilities per W plane.
					for ( int wPlane = 0; wPlane < _hstWPlanes; wPlane++ )
						for ( int gpu = 0; gpu < _hstNumGPUs; gpu++ )
						{
							hstVisibilitiesInWPlane[ wPlane ][ gpu ] = 0;
							for ( int aPlane = 0; aPlane < _hstAPlanes; aPlane++ )
								hstVisibilitiesInWPlane[ wPlane ][ gpu ] +=
											hstVisibilitiesInKernelSet[ (wPlane * _hstAPlanes) + aPlane ][ gpu ];
						}

					// maintain pointers to the next visibilities for each GPU.
					int * hstNextVisibility = (int *) malloc( _hstNumGPUs * sizeof( int ) );
					for ( int gpu = 0; gpu < _hstNumGPUs; gpu++ )
						hstNextVisibility[ gpu ] = 0;

					int cudaDeviceIndex = 0;
					for ( int wPlane = 0; wPlane < _hstWPlanes; wPlane++ )
					{

						int lastGPU = cudaDeviceIndex;
						do
						{

							if (hstVisibilitiesInWPlane[ wPlane ][ cudaDeviceIndex ] > 0)
							{

								// set the cuda device, and make sure nothing is running there already.
								if (_hstNumGPUs > 1)
									cudaSetDevice( _hstGPU[ cudaDeviceIndex ] );

								// upload the grid positions, kernel indexes, and density map to the device.
								moveHostToDevice( (void *) &devGridPosition[ cudaDeviceIndex ][ hstNextVisibility[ cudaDeviceIndex ] ],
											(void *) &_hstGridPosition[ hstCurrentVisibility ],
											hstVisibilitiesInWPlane[ wPlane ][ cudaDeviceIndex ] * sizeof( VectorI ),
											"copying grid positions to the device" );
								moveHostToDevice( (void *) &devKernelIndex[ cudaDeviceIndex ][ hstNextVisibility[ cudaDeviceIndex ] ],
											(void *) &_hstKernelIndex[ hstCurrentVisibility ],
											hstVisibilitiesInWPlane[ wPlane ][ cudaDeviceIndex ] * sizeof( int ),
											"copying kernel indexes to the device" );
								moveHostToDevice( (void *) &devDensityMap[ cudaDeviceIndex ][ hstNextVisibility[ cudaDeviceIndex ] ],
											(void *) &_hstDensityMap[ hstCurrentVisibility ],
											hstVisibilitiesInWPlane[ wPlane ][ cudaDeviceIndex ] * sizeof( int ),
											"copying density map to the device" );

								// upload weights to the device.
								if (_hstWeighting != NONE)
									moveHostToDevice( (void *) &devWeight[ cudaDeviceIndex ][ hstNextVisibility[ cudaDeviceIndex ] ],
												(void *) &_hstWeight[ hstCurrentVisibility ],
												hstVisibilitiesInWPlane[ wPlane ][ cudaDeviceIndex ] * sizeof( float ),
												"copying weights to the device" );

								// get the next set of visibilities.
								hstCurrentVisibility += hstVisibilitiesInWPlane[ wPlane ][ cudaDeviceIndex ];
								hstNextVisibility[ cudaDeviceIndex ] += hstVisibilitiesInWPlane[ wPlane ][ cudaDeviceIndex ];

							} // hstVisibilitiesInWPlane[ wPlane ][ cudaDeviceIndex ] > 0
							cudaDeviceIndex++;
							if (cudaDeviceIndex == _hstNumGPUs)
								cudaDeviceIndex = 0;

						} while (cudaDeviceIndex != lastGPU);

					} // LOOP: wPlane
					if (_hstNumGPUs > 1)
						cudaSetDevice( _hstGPU[ 0 ] );

					// process the visibilities and apply the density map.
					for ( int gpu = 0; gpu < _hstNumGPUs; gpu++ )
						if (hstNextVisibility[ gpu ] > 0)
						{

							if (_hstNumGPUs > 1)
								cudaSetDevice( _hstGPU[ gpu ] );

							// set all the visibilities to (1, 0). these are the visibilities used for generating the dirty beam.
							int threads = hstNextVisibility[ gpu ];
							int blocks = 1;
							setThreadBlockSize1D( &threads, &blocks );
		
							// update the real part of each visibility to 1.
							devUpdateComplexArray<<< blocks, threads>>>( devBeamVisibility[ gpu ], hstNextVisibility[ gpu ], 1, 0 );
							err = cudaGetLastError();
							if (err != cudaSuccess)
								printf( "error building visibilities for psf (%s)\n", cudaGetErrorString( err ) );

							// apply the density map - multiply all the visibilities by the value of the density map at that position.
							devApplyDensityMap<<< blocks, threads >>>(	/* pVisibilities = */ devBeamVisibility[ gpu ],
													/* pDensityMap = */ devDensityMap[ gpu ],
													/* pItems = */ hstNextVisibility[ gpu ] );
							err = cudaGetLastError();
							if (err != cudaSuccess)
								printf( "error applying the density map to the beam visibilities (%s)\n", cudaGetErrorString( err ) );

						}
					if (_hstNumGPUs > 1)
						cudaSetDevice( _hstGPU[ 0 ] );

					// free memory.
					if (hstNextVisibility != NULL)
						free( (void *) hstNextVisibility );

					// generate the uv coverage using the gridder.
					gridVisibilities(	/* pdevGrid = */ devDirtyBeamGrid,
								/* pdevVisibility = */ devBeamVisibility,
								/* pOversample = */ _hstOversample,
								/* pKernelSize = */ _hstKernelSize,
								/* pSupport = */ _hstSupportSize,
								/* pdevKernelIndex = */ devKernelIndex,
								/* pWProjection = */ _hstWProjection,
								/* pAProjection = */ false,
								/* pWPlanes = */ _hstWPlanes,
								/* pAPlanes = */ 1,
								/* pdevGridPositions = */ devGridPosition,
								/* pdevWeight = */ devWeight,
								/* pVisibilitiesInKernelSet = */ hstVisibilitiesInWPlane,
								/* pGridDegrid = */ GRID,
								/* phstPrimaryBeamMosaicing = */ NULL,
								/* phstPrimaryBeamAProjection = */ NULL,
								/* pNumFields = */ -1,
								/* pMosaicIndex = */ image,
								/* pSize = */ _hstUvPixels,
								/* pNumGPUs = */ _hstNumGPUs );
		
					// get the next batch of data.
					batch = batch + 1;
		
				}

				// free memory.
				for ( int gpu = 0; gpu < _hstNumGPUs; gpu++ )
				{
					if (_hstNumGPUs > 1)
						cudaSetDevice( _hstGPU[ gpu ] );
					if (devBeamVisibility[ gpu ] != NULL)
						cudaFree( (void *) devBeamVisibility[ gpu ] );
					if (devDensityMap[ gpu ] != NULL)
						cudaFree( (void *) devDensityMap[ gpu ] );
					if (devWeight[ gpu ] != NULL)
						cudaFree( (void *) devWeight[ gpu ] );
					if (devGridPosition[ gpu ] != NULL)
						cudaFree( (void *) devGridPosition[ gpu ] );
					if (devKernelIndex[ gpu ] != NULL)
						cudaFree( (void *) devKernelIndex[ gpu ] );
				}
				if (_hstNumGPUs > 1)
					cudaSetDevice( _hstGPU[ 0 ] );
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
				if (hstVisibilitiesInWPlane != NULL)
				{
					for ( int wPlane = 0; wPlane < _hstWPlanes; wPlane++ )
						if (hstVisibilitiesInWPlane[ wPlane ] != NULL)
							free( (void *) hstVisibilitiesInWPlane[ wPlane ] );
					free( (void *) hstVisibilitiesInWPlane );
				}

				// free the data.
				if (_hstCacheData == true)
					freeData( /* pWhatData = */ DATA_GRID_POSITIONS | DATA_KERNEL_INDEXES | DATA_DENSITIES | DATA_WEIGHTS );

			}

		}

		// move all images to the same GPU and add them together.
		if (_hstNumGPUs > 1)
		{

			cufftComplex * hstTmpImage = (cufftComplex *) malloc( _hstUvPixels * _hstUvPixels * sizeof( cufftComplex ) );
			cufftComplex * devTmpImage = NULL;
			reserveGPUMemory( (void **) &devTmpImage, _hstUvPixels * _hstUvPixels * sizeof( cufftComplex ),
							"reserving GPU memory for the temporary gridded data" );
				
			int threads = _hstUvPixels * _hstUvPixels, blocks = 1;
			setThreadBlockSize1D( &threads, &blocks );

			for ( int gpu = 1; gpu < _hstNumGPUs; gpu++ )
			{

				// set gpu device, and move image to the host.
				cudaSetDevice( _hstGPU[ gpu ] );
				moveDeviceToHost( (void *) hstTmpImage, devDirtyBeamGrid[ gpu ], _hstUvPixels * _hstUvPixels * sizeof( cufftComplex ),
							"moving gridded data to the host" );
				cudaDeviceSynchronize();

				// set gpu device, and move image to the device.
				cudaSetDevice( _hstGPU[ 0 ] );
				moveHostToDevice( (void *) devTmpImage, (void *) hstTmpImage, _hstUvPixels * _hstUvPixels * sizeof( cufftComplex ),
							"moving gridded data to the device" );

				// add images together.
				devAddComplexData<<< blocks, threads >>>(	/* pOne = */ devDirtyBeamGrid[ 0 ],
										/* pTwo = */ devTmpImage,
										/* pElements = */ _hstUvPixels * _hstUvPixels );

			}

			// free memory.
			if (hstTmpImage != NULL)
				free( (void *) hstTmpImage );
			if (devTmpImage != NULL)
				cudaFree( (void *) devTmpImage );

		}

		// generate the dirty beam by FFTing the gridded data.
		generateDirtyBeam(	/* pdevDirtyBeam = */ &devDirtyBeamGrid[ 0 ],
					/* pFilename = */ outputDirtyBeamFilename );

	}

	// re-cast the dirty beam from complex to doubles.
	float * devDirtyBeam = (float *) devDirtyBeamGrid[ 0 ];
	devDirtyBeamGrid = NULL;

	// create the dirty image, and, for file mosaicing only, a cache of dirty images.
	float * hstDirtyImage = NULL;
	float * hstDirtyImageCache = NULL;
	if (_hstFileMosaic == true)
	{
		hstDirtyImageCache = (float *) malloc( (long int) _hstUvPixels * (long int) _hstUvPixels * (long int) _numMosaicImages *
									(long int) sizeof( float ) );
		memset( hstDirtyImageCache, 0, (long int) _hstUvPixels * (long int) _hstUvPixels * (long int) _numMosaicImages * (long int) sizeof( float ) );
	}

	// copy the normalisation pattern over.
	float * devNormalisationPattern = NULL;
	if (_hstBeamMosaic == true || _hstUVMosaic == true)
	{
		reserveGPUMemory( (void **) &devNormalisationPattern, _hstBeamSize * _hstBeamSize * sizeof( float ), "reserving GPU memory for normalisation pattern" );
		moveHostToDevice( (void *) devNormalisationPattern, (void *) _hstNormalisationPattern, _hstBeamSize * _hstBeamSize * sizeof( float ),
					"copying normalisation pattern to the device" );
	}

	// declare device memory for the dirty image grid, and zero this memory. we need to do this on ALL gpus.
	cufftComplex ** devDirtyImageGrid = (cufftComplex **) malloc( _hstNumGPUs * sizeof( cufftComplex * ) );
	for ( int gpu = 0; gpu < _hstNumGPUs; gpu++ )
	{
		if (_hstNumGPUs > 1)
			cudaSetDevice( _hstGPU[ gpu ] );
		reserveGPUMemory( (void **) &devDirtyImageGrid[ gpu ], _hstUvPixels * _hstUvPixels * sizeof( cufftComplex ), "declaring device memory for grid" );
		zeroGPUMemory( (void *) devDirtyImageGrid[ gpu ], _hstUvPixels * _hstUvPixels * sizeof( cufftComplex ), "zeroing the grid on the device" );
	}
	if (_hstNumGPUs > 1)
		cudaSetDevice( _hstGPU[ 0 ] );

	struct timespec time1, time2;
	clock_gettime( CLOCK_REALTIME, &time1 );

	printf( "\ngridding visibilities for dirty image.....\n\n" );
	for ( int image = 0; image < _numMosaicImages; image++ )
	{

		// count the total number of visibilities.
		long int totalVisibilities = 0;
		for ( int stageID = 0; stageID < _hstNumberOfStages[ image ]; stageID++ )
			totalVisibilities += _hstNumVisibilities[ image ][ stageID ];

		if (_numMosaicImages > 1)
			printf( "        processing mosaic component %i of %i.....\n\n", image + 1, _numMosaicImages );
		printf( "                stages: %i\n", _hstNumberOfStages[ image ] );
		printf( "                visibilities: %li\n\n", totalVisibilities );

		// re-create the dirty-image grid, and zero grid memory.
		if (_hstFileMosaic == true && image > 0)
			for ( int gpu = 0; gpu < _hstNumGPUs; gpu++ )
			{
				if (_hstNumGPUs > 1)
					cudaSetDevice( _hstGPU[ gpu ] );
				reserveGPUMemory( (void **) &devDirtyImageGrid[ gpu ], _hstUvPixels * _hstUvPixels * sizeof( cufftComplex ),
							"declaring device memory for grid" );
				zeroGPUMemory( (void *) devDirtyImageGrid[ gpu ], _hstUvPixels * _hstUvPixels * sizeof( cufftComplex ), "zeroing the grid on the device" );
			}
		if (_hstNumGPUs > 1)
			cudaSetDevice( _hstGPU[ 0 ] );

		// free the dirty grid on the host, if it already exists.
		if (hstDirtyImage != NULL)
		{
			free( (void *) hstDirtyImage );
			hstDirtyImage = NULL;
		}

		// uncache the data for this mosaic image.
		long int visibilitiesProcessed = 0;
		for ( int stageID = 0; stageID < _hstNumberOfStages[ image ]; stageID++ )
		{

			// get the data from the file.
			if (_hstCacheData == true)
				uncacheData(	/* pFilenamePrefix = */ _hstOutputPrefix,
						/* pMosaicID = */ image,
						/* pStageID = */ stageID,
						/* pWhatData = */ DATA_ALL,
						/* pOffset = */ 0 );

			// if the number of visibilities is greater than the maximum number then we are going to set a smaller batch size, and load these
			// visibilities in batches.
			int hstVisibilityBatchSize = 0;
			{
				long int nextBatchSize = _hstNumVisibilities[ image ][ stageID ];
				if (nextBatchSize > _hstPreferredVisibilityBatchSize)
					nextBatchSize = _hstPreferredVisibilityBatchSize;
				hstVisibilityBatchSize = (int) nextBatchSize;
			}

			// if the number of minor cycles is zero then we haven't generated a dirty beam, or calculated the number of visibilities per kernel set.
			// we need to do that here, starting with reserving some memory.
			if (_hstMinorCycles == 0)

				// if the number of minor cycles is zero then we haven't generated a dirty beam, so we haven't counted the number of visibilities
				// per kernel set. we must do that here.
				calculateVisibilitiesPerKernelSet(	/* pNumVisibilities = */_hstNumVisibilities[ image ][ stageID ],
									/* pBatchSize = */ hstVisibilityBatchSize,
									/* phstGridPosition = */ _hstGridPosition,
									/* phstVisibilitiesInKernelSet = */ &_hstVisibilitiesInKernelSet[ image ][ stageID ],
									/* pNumGPUs = */ _hstNumGPUs,
									/* pNumBatches = */ &_hstNumberOfBatches[ image ][ stageID ],
									/* pNumKernelSets = */ _hstKernelSets );

			// create space for the visibilities, the unity (psf) visibilities, the density map, and the weights on the device.
			cufftComplex ** devVisibility = (cufftComplex **) malloc( _hstNumGPUs * sizeof( cufftComplex * ) );
			float ** devWeight = (float **) malloc( _hstNumGPUs * sizeof( float *) );
			VectorI ** devGridPosition = (VectorI **) malloc( _hstNumGPUs * sizeof( VectorI * ) );
			int ** devKernelIndex = (int **) malloc( _hstNumGPUs * sizeof( int * ) );
			for ( int gpu = 0; gpu < _hstNumGPUs; gpu++ )
			{
				if (_hstNumGPUs > 1)
					cudaSetDevice( _hstGPU[ gpu ] );
				reserveGPUMemory( (void **) &devVisibility[ gpu ], hstVisibilityBatchSize * sizeof( cufftComplex ),
							"creating device memory for the visibilities" );
				reserveGPUMemory( (void **) &devGridPosition[ gpu ], hstVisibilityBatchSize * sizeof( VectorI ),
							"declaring device memory for the grid positions" );
				reserveGPUMemory( (void **) &devKernelIndex[ gpu ], hstVisibilityBatchSize * sizeof( int ),
							"declaring device memory for the kernel indexes" );
				if (_hstWeighting != NONE)
					reserveGPUMemory( (void **) &devWeight[ gpu ], hstVisibilityBatchSize * sizeof( float ),
							"declaring device memory for the weights" );
			}
			if (_hstNumGPUs > 1)
				cudaSetDevice( _hstGPU[ 0 ] );

			// keep looping until we have loaded and gridded all visibilities.
			int batch = 0;
			long int hstCurrentVisibility = 0;
			while (hstCurrentVisibility < _hstNumVisibilities[ image ][ stageID ])
			{

				int ** hstVisibilitiesInKernelSet = _hstVisibilitiesInKernelSet[ image ][ stageID ][ batch ];

				// count the number of visibilities in this batch.
				int visibilitiesInThisBatch = 0;
				for ( int kernelSet = 0; kernelSet < _hstKernelSets; kernelSet++ )
					for ( int gpu = 0; gpu < _hstNumGPUs; gpu++ )
						visibilitiesInThisBatch += hstVisibilitiesInKernelSet[ kernelSet ][ gpu ];

				if (_hstNumberOfStages[ image ] > 1 || _hstNumberOfBatches[ image ][ stageID ] > 1)
					printf( "        gridding " );
				if (_hstNumberOfStages[ image ] > 1)
					printf( "host batch %i of %i", stageID + 1, _hstNumberOfStages[ image ] );
				if (_hstNumberOfStages[ image ] > 1 && _hstNumberOfBatches[ image ][ stageID ] > 1)
					printf( ", " );
				if (_hstNumberOfBatches[ image ][ stageID ] > 1)
					printf( "gpu batch %i of %i", batch + 1, _hstNumberOfBatches[ image ][ stageID ] );
				if (_hstNumberOfStages[ image ] > 1 || _hstNumberOfBatches[ image ][ stageID ] > 1)
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
				int * hstNextVisibility = (int *) malloc( _hstNumGPUs * sizeof( int ) );
				for ( int gpu = 0; gpu < _hstNumGPUs; gpu++ )
					hstNextVisibility[ gpu ] = 0;

				int cudaDeviceIndex = 0;
				for ( int kernelSet = 0; kernelSet < _hstKernelSets; kernelSet++ )
				{

					int lastGPU = cudaDeviceIndex;
					do
					{

						if (hstVisibilitiesInKernelSet[ kernelSet ][ cudaDeviceIndex ] > 0)
						{

							// set the cuda device.
							if (_hstNumGPUs > 1)
								cudaSetDevice( _hstGPU[ cudaDeviceIndex ] );

							// upload the visibilities, grid positions, kernel indexes, and density map to the device.
							moveHostToDevice( (void *) &devVisibility[ cudaDeviceIndex ][ hstNextVisibility[ cudaDeviceIndex ] ],
										(void *) &_hstVisibility[ hstCurrentVisibility ],
										hstVisibilitiesInKernelSet[ kernelSet ][ cudaDeviceIndex ] * sizeof( cufftComplex ),
										"copying visibilities to the device" );
							moveHostToDevice( (void *) &devGridPosition[ cudaDeviceIndex ][ hstNextVisibility[ cudaDeviceIndex ] ],
										(void *) &_hstGridPosition[ hstCurrentVisibility ],
										hstVisibilitiesInKernelSet[ kernelSet ][ cudaDeviceIndex ] * sizeof( VectorI ),
										"copying grid positions to the device" );
							moveHostToDevice( (void *) &devKernelIndex[ cudaDeviceIndex ][ hstNextVisibility[ cudaDeviceIndex ] ],
										(void *) &_hstKernelIndex[ hstCurrentVisibility ],
										hstVisibilitiesInKernelSet[ kernelSet ][ cudaDeviceIndex ] * sizeof( int ),
										"copying kernel indexes to the device" );

							// upload weights to the device.
							if (_hstWeighting != NONE)
								moveHostToDevice( (void *) &devWeight[ cudaDeviceIndex ][ hstNextVisibility[ cudaDeviceIndex ] ],
											(void *) &_hstWeight[ hstCurrentVisibility ],
											hstVisibilitiesInKernelSet[ kernelSet ][ cudaDeviceIndex ] * sizeof( float ),
											"copying weights to the device" );

							// get the next set of visibilities.
							hstCurrentVisibility += hstVisibilitiesInKernelSet[ kernelSet ][ cudaDeviceIndex ];
							hstNextVisibility[ cudaDeviceIndex ] += hstVisibilitiesInKernelSet[ kernelSet ][ cudaDeviceIndex ];

						} // hstVisibilitiesInKernelSet[ kernelSet ][ cudaDeviceIndex ] > 0
						cudaDeviceIndex++;
						if (cudaDeviceIndex == _hstNumGPUs)
							cudaDeviceIndex = 0;

					} while (cudaDeviceIndex != lastGPU);
			
				} // LOOP: kernelSet
				if (_hstNumGPUs > 1)
					cudaSetDevice( _hstGPU[ 0 ] );

				// free data.
				free( (void *) hstNextVisibility );

				err = cudaGetLastError();
				if (err != cudaSuccess)
					printf( "unknown CUDA error on line %d (%s)\n", __LINE__, cudaGetErrorString( err ) );

				// grid visibilities.
				gridVisibilities(	/* pdevGrid = */ devDirtyImageGrid,
							/* pdevVisibility = */ devVisibility,
							/* pOversample = */ _hstOversample,
							/* pKernelSize = */ _hstKernelSize,
							/* pSupport = */ _hstSupportSize,
							/* pdevKernelIndex = */ devKernelIndex,
							/* pWProjection = */ _hstWProjection,
							/* pAProjection = */ _hstAProjection,
							/* pWPlanes = */ _hstWPlanes,
							/* pAPlanes = */ _hstAPlanes,
							/* pdevGridPositions = */ devGridPosition,
							/* pdevWeight = */ devWeight,
							/* pVisibilitiesInKernelSet = */ hstVisibilitiesInKernelSet,
							/* pGridDegrid = */ GRID,
							/* phstPrimaryBeamMosaicing = */ (_hstBeamMosaic == true || _hstUVMosaic == true ? _hstPrimaryBeam[ image ] : NULL),
							/* phstPrimaryBeamAProjection = */ _hstPrimaryBeamAProjection[ image ],
							/* pNumFields = */ (_hstBeamMosaic == true ? _hstBeamMosaicComponents : -1),
							/* pMosaicIndex = */ image,
							/* pSize = */ _hstUvPixels,
							/* pNumGPUs = */ _hstNumGPUs );

				// get the next batch of data.
				batch = batch + 1;
		
			}

			// free memory.
			for ( int gpu = 0; gpu < _hstNumGPUs; gpu++ )
			{
				if (_hstNumGPUs > 1)
					cudaSetDevice( _hstGPU[ gpu ] );
				if (devVisibility[ gpu ] != NULL)
					cudaFree( (void *) devVisibility[ gpu ] );
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
			if (_hstNumGPUs > 1)
				cudaSetDevice( _hstGPU[ 0 ] );

			// free the data.
			if (_hstCacheData == true)
				freeData( /* pWhatData = */ DATA_ALL );

		}

		// we only do FFT and normalisation if we've finished gridding. if we are making an UV mosaic from multiple files then we only do this on the last file.
		if (_hstUVMosaic == false || image == _numMosaicImages - 1)
		{

			double normalisation = 1.0;

			// we normalise the image by the number of gridded visibilities, but only if we're not using beam mosaicing. Beam mosaicing will do the normalisation
			// using the kernel.
			if (_hstBeamMosaic == true || _hstUVMosaic == true)
				normalisation *= (double) _griddedVisibilitiesForBeamMosaic;
			else
				normalisation *= (double) _hstGriddedVisibilities[ image ];

			if (_hstWeighting != NONE && _hstUVMosaic == false)
				normalisation *= _hstAverageWeight[ image ];
			if (_hstWeighting != NONE && _hstUVMosaic == true)
				normalisation *= _hstTotalAverageWeight;

			// move all images to the same GPU and add them together.
			if (_hstNumGPUs > 1)
			{

				cufftComplex * hstTmpImage = (cufftComplex *) malloc( _hstUvPixels * _hstUvPixels * sizeof( cufftComplex ) );
				cufftComplex * devTmpImage = NULL;
				reserveGPUMemory( (void **) &devTmpImage, _hstUvPixels * _hstUvPixels * sizeof( cufftComplex ),
								"reserving GPU memory for the temporary gridded data" );
				
				int threads = _hstUvPixels * _hstUvPixels, blocks = 1;
				setThreadBlockSize1D( &threads, &blocks );

				for ( int gpu = 1; gpu < _hstNumGPUs; gpu++ )
				{

					// set gpu device, and move image to the host.
					cudaSetDevice( _hstGPU[ gpu ] );
					moveDeviceToHost( (void *) hstTmpImage, devDirtyImageGrid[ gpu ], _hstUvPixels * _hstUvPixels * sizeof( cufftComplex ),
								"moving gridded data to the host" );
					cudaDeviceSynchronize();

					// set gpu device, and move image to the device.
					cudaSetDevice( _hstGPU[ 0 ] );
					moveHostToDevice( (void *) devTmpImage, (void *) hstTmpImage, _hstUvPixels * _hstUvPixels * sizeof( cufftComplex ),
								"moving gridded data to the device" );

					// add images together.
					devAddComplexData<<< blocks, threads >>>(	/* pOne = */ devDirtyImageGrid[ 0 ],
											/* pTwo = */ devTmpImage,
											/* pElements = */ _hstUvPixels * _hstUvPixels );

				}

				// free memory.
				if (hstTmpImage != NULL)
					free( (void *) hstTmpImage );
				if (devTmpImage != NULL)
					cudaFree( (void *) devTmpImage );

			}

			printf( "\n        performing fft on dirty image grid.....\n" );
	
			// make dirty image on the device.
			performFFT(	/* pdevGrid = */ &devDirtyImageGrid[ 0 ],
					/* pSize = */ _hstUvPixels,
					/* pFFTDirection = */ INVERSE,
					/* pFFTPlan = */ -1,
					/* pFFTType = */ C2F );
					
			// define the block/thread dimensions.
			int items = _hstUvPixels * _hstUvPixels;
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
				float * devDirtyImageDbl = (float *) devDirtyImageGrid[ 0 ];
				devNormalise<<< blocks, threads >>>( &devDirtyImageDbl[ i * MAX_THREADS ], normalisation, itemsThisStage );

			}
		
			// define the block/thread dimensions.
			setThreadBlockSize2D( _hstUvPixels, _hstUvPixels );

			// divide the dirty image by the deconvolution image.
			devDivideImages<<< _gridSize2D, _blockSize2D >>>(	/* pOne = */ (float *) devDirtyImageGrid[ 0 ],
										/* pTwo = */ _devDeconvolutionImage,
										/* pMask = */ NULL,
										/* pSizeOne = */ _hstUvPixels,
										/* pSizeTwo = */ _hstPsfSize );

			// for beam mosaicing, divide the dirty image by the normalisation pattern.
			if (_hstBeamMosaic == true || _hstUVMosaic == true)
				devDivideImages<<< _gridSize2D, _blockSize2D >>>(	/* pOne = */ (float *) devDirtyImageGrid[ 0 ],
											/* pTwo = */ devNormalisationPattern,
											/* pMask = */ NULL,
											/* pSizeOne = */ _hstUvPixels,
											/* pSizeTwo = */ _hstBeamSize );

			// copy the dirty image to the host.
			if (image == _numMosaicImages - 1)
			{
				hstDirtyImage = (float *) malloc( _hstUvPixels * _hstUvPixels * sizeof( float ) );
				moveDeviceToHost( (void *) hstDirtyImage, (void *) devDirtyImageGrid[ 0 ], _hstUvPixels * _hstUvPixels * sizeof( float ),
							"copying dirty image from device" );
			}

			printf( "\n" );

			// if we're file mosaicing then copy the dirty image into the cache.
			if (_hstFileMosaic == true)
				moveDeviceToHost( (void *) &hstDirtyImageCache[ (long int) _hstUvPixels * (long int) _hstUvPixels * (long int) image ],
							(void *) devDirtyImageGrid[ 0 ], _hstUvPixels * _hstUvPixels * sizeof( float ),
							"copying dirty image from device" );

			// free memory.
			if (devDirtyImageGrid != NULL)
				for ( int gpu = 0; gpu < _hstNumGPUs; gpu++ )
					if (devDirtyImageGrid[ gpu ] != NULL)
					{
						if (_hstNumGPUs > 1)
							cudaSetDevice( _hstGPU[ gpu ] );
						cudaFree( (void *) devDirtyImageGrid[ gpu ] );
					}
			if (_hstNumGPUs > 1)
				cudaSetDevice( _hstGPU[ 0 ] );

		}

	}

	clock_gettime( CLOCK_REALTIME, &time2 );
	printf( "--- time (total, inc fft): (%f ms) ---\n\n", getTime( time1, time2 ) );

	// free memory.
	if (devNormalisationPattern != NULL)
		cudaFree( (void *) devNormalisationPattern );

	// if we are mosaicing, create a mosaic from the dirty images.
	if (_hstFileMosaic == true)
	{

		// construct the mosaic.
		createMosaic(	/* phstMosaic = */ hstDirtyImage,
				/* phstImageArray = */ hstDirtyImageCache,
				/* phstMask = */ hstMask,
				/* phstPhaseCentre = */ hstImagePhaseCentre,
				/* phstPrimaryBeamPatternPtr = */ &_hstPrimaryBeamPattern );

		// save the dirty image components of the mosaic (debugging option).
		if (_hstSaveMosaicDirtyImages == true)
			for ( int image = 0; image < _numMosaicImages; image++ )
			{
			        char filename[ 100 ];
			        sprintf( filename, "%s-dirty-image-%i.image", _hstOutputPrefix, image );
			        _hstCasacoreInterface.WriteCasaImage(	/* pFilename = */ filename,
									/* pWidth = */ _hstUvPixels,
									/* pHeight = */ _hstUvPixels,
									/* pRA = */ hstImagePhaseCentre[ image * 2 ],
									/* pDec = */ hstImagePhaseCentre[ (image * 2) + 1 ],
			                                                /* pPixelSize = */ _hstCellSize,
									/* pImage = */ &hstDirtyImageCache[ (long int) image * (long int) _hstUvPixels *
																(long int) _hstUvPixels ],
									/* pFrequency = */ CONST_C / _hstAverageWavelength[ image ],
									/* pMask = */ NULL );
			}

	}

	err = cudaGetLastError();
	if (err != cudaSuccess)
		printf( "unknown CUDA error on line %d (%s)\n", __LINE__, cudaGetErrorString( err ) );

	// save the dirty image.
	_hstCasacoreInterface.WriteCasaImage(	/* pFilename = */ outputDirtyImageFilename,
						/* pWidth = */ _hstUvPixels,
						/* pHeight = */ _hstUvPixels,
						/* pRA = */ _hstOutputRA,
						/* pDec = */ _hstOutputDEC,
						/* pPixelSize = */ _hstCellSize,
						/* pImage = */ hstDirtyImage,
						/* pFrequency = */ CONST_C / _hstAverageWavelength[ 0 ],
						/* pMask = */ hstMask );

	err = cudaGetLastError();
	if (err != cudaSuccess)
		printf( "unknown CUDA error on line %d (%s)\n", __LINE__, cudaGetErrorString( err ) );

	// save the primary beam pattern.
	if (_hstUseMosaicing == true)
	{

		// build the primary beam pattern mask.
		bool * hstPrimaryBeamPatternMask = (bool *) malloc( _hstBeamSize * _hstBeamSize * sizeof( bool ) );
		for ( int i = 0; i < _hstBeamSize * _hstBeamSize; i++ )
			hstPrimaryBeamPatternMask[ i ] = (_hstPrimaryBeamPattern[ i ] >= 0.2);

		// save the primary beam pattern.
		_hstCasacoreInterface.WriteCasaImage(	/* pFilename = */ outputPrimaryBeamPatternFilename,
							/* pWidth = */ _hstBeamSize,
							/* pHeight = */ _hstBeamSize,
							/* pRA = */ _hstOutputRA,
							/* pDec = */ _hstOutputDEC,
							/* pPixelSize = */ _hstCellSize * (double) _hstUvPixels / (double) _hstBeamSize,
							/* pImage = */ _hstPrimaryBeamPattern,
							/* pFrequency = */ CONST_C / _hstAverageWavelength[ 0 ],
							/* pMask = */ hstPrimaryBeamPatternMask );

		// free memory.
		free( (void *) hstPrimaryBeamPatternMask );

	}

	// are we doing cleaning ?
	if (_hstMinorCycles > 0)
	{

		err = cudaGetLastError();
		if (err != cudaSuccess)
			printf( "unknown CUDA error on line %d (%s)\n", __LINE__, cudaGetErrorString( err ) );

		// create memory for the clean beam.
		float * devCleanBeam = NULL;
		reserveGPUMemory( (void **) &devCleanBeam, _hstPsfSize * _hstPsfSize * sizeof( float ), "declaring device memory for clean beam" );

		err = cudaGetLastError();
		if (err != cudaSuccess)
			printf( "unknown CUDA error on line %d (%s)\n", __LINE__, cudaGetErrorString( err ) );
		
		// generate the clean beam (relies on the dirty beam already being within device memory).
		generateCleanBeam(	/* pdevCleanBeam = */ devCleanBeam,
					/* pdevDirtyBeam = */ devDirtyBeam,
					/* pFilename = */ outputCleanBeamFilename );
		
		// do a Cotton-Schwab clean.
		cottonSchwabClean(	/* phstPhaseCentre = */ hstImagePhaseCentre,
					/* pFilenamePrefix = */ _hstOutputPrefix,
					/* pdevCleanBeam = */ devCleanBeam,
					/* pdevDirtyBeam = */ devDirtyBeam,
					/* phstDirtyImage = */ &hstDirtyImage,
					/* phstDirtyImageCache = */ hstDirtyImageCache,
					/* phstMask = */ hstMask,
					/* pCleanImageFilename = */ outputCleanImageFilename,
					/* pResidualImageFilename = */ outputResidualImageFilename );

		// free memory.
		if (devCleanBeam != NULL)
			cudaFree( (void *) devCleanBeam );

	}

	// make sure our data is freed.
	freeData( /* pWhatData = */ DATA_ALL );

	// stuff to be released after we have cleaned our images.
	if (devDirtyBeam != NULL)
		cudaFree( (void *) devDirtyBeam );
	if (_devDeconvolutionImage != NULL)
		cudaFree( (void *) _devDeconvolutionImage );
	if (_hstWPlaneMean != NULL)
	{
		for ( int i = 0; i < _numMosaicImages; i++ )
			if (_hstWPlaneMean[ i ] != NULL)
				free( (void *) _hstWPlaneMean[ i ] );
		free( (void *) _hstWPlaneMean );
	}
	if (_hstWPlaneMax != NULL)
	{
		for ( int i = 0; i < _numMosaicImages; i++ )
			if (_hstWPlaneMax[ i ] != NULL)
				free( (void *) _hstWPlaneMax[ i ] );
		free( (void *) _hstWPlaneMax );
	}
	if (hstDirtyImageCache != NULL)
		free( (void *) hstDirtyImageCache );
	if (_hstNumVisibilities != NULL)
	{
		for ( int i = 0; i < _numMosaicImages; i++ )
			if (_hstNumVisibilities[ i ] != NULL)
				free( (void *) _hstNumVisibilities[ i ] );
		free( (void *) _hstNumVisibilities );
	}
	if (_hstVisibilitiesInKernelSet != NULL)
	{
		for ( int image = 0; image < _numMosaicImages; image++ )
			if (_hstVisibilitiesInKernelSet[ image ] != NULL)
			{
				for ( int stageID = 0; stageID < _hstNumberOfStages[ image ]; stageID++ )
					if (_hstVisibilitiesInKernelSet[ image ][ stageID ] != NULL)
					{
						for ( int batch = 0; batch < _hstNumberOfBatches[ image ][ stageID ]; batch++ )
							if (_hstVisibilitiesInKernelSet[ image ][ stageID ][ batch ] != NULL)
							{
								for ( int kernelSet = 0; kernelSet < _hstKernelSets; kernelSet++ )
									if (_hstVisibilitiesInKernelSet[ image ][ stageID ][ batch ][ kernelSet ] != NULL)
										free( (void *) _hstVisibilitiesInKernelSet[ image ][ stageID ][ batch ][ kernelSet ] );
								free( (void *) _hstVisibilitiesInKernelSet[ image ][ stageID ][ batch ] );
							}
						free( (void *) _hstVisibilitiesInKernelSet[ image ][ stageID ] );
					}
				free( (void *) _hstVisibilitiesInKernelSet[ image ] );
			}
		free( (void *) _hstVisibilitiesInKernelSet );
	}
	if (_hstKernelSize != NULL)
		free( (void *) _hstKernelSize );
	if (_hstSupportSize != NULL)
		free( (void *) _hstSupportSize );
	if (_hstAverageWeight != NULL)
		free( (void *) _hstAverageWeight );
	if (_hstNumberOfBatches != NULL)
	{
		for ( int image = 0; image < _numMosaicImages; image++ )
			if (_hstNumberOfBatches[ image ] != NULL)
				free( (void *) _hstNumberOfBatches[ image ] );
		free( (void *) _hstNumberOfBatches );
	}
	if (_hstNumberOfStages != NULL)
		free( (void *) _hstNumberOfStages );
	if (_hstDeconvolutionImage != NULL)
		free( (void *) _hstDeconvolutionImage );
	if (hstImagePhaseCentre != NULL)
		free( (void *) hstImagePhaseCentre );
	if (_hstPrimaryBeam != NULL)
	{
		for ( int i = 0; i < _numMosaicImages; i++ )
			if (_hstPrimaryBeam[ i ] != NULL)
				free( (void *) _hstPrimaryBeam[ i ] );
		free( (void *) _hstPrimaryBeam );
	}
	if (_hstPrimaryBeamAProjection != NULL)
	{
		for ( int i = 0; i < _numMosaicImages; i++ )
			if (_hstPrimaryBeamAProjection[ i ] != NULL)
				free( (void *) _hstPrimaryBeamAProjection[ i ] );
		free( (void *) _hstPrimaryBeamAProjection );
	}
	if (_hstBeamID != NULL)
		free( (void *) _hstBeamID );
	if (_hstPrimaryBeamPattern != NULL)
		free( (void *) _hstPrimaryBeamPattern );
	if (_hstNormalisationPattern != NULL)
		free( (void *) _hstNormalisationPattern );
	if (_hstGriddedVisibilities != NULL)
		free( (void *) _hstGriddedVisibilities );
	if (_hstGriddedVisibilitiesPerField != NULL)
		free( (void *) _hstGriddedVisibilitiesPerField );
	if (_hstAverageWavelength != NULL)
		free( (void *) _hstAverageWavelength );
	if (_hstGPU != NULL)
		free( (void *) _hstGPU );
	if (hstDirtyImage != NULL)
		free( (void *) hstDirtyImage );
	if (hstMask != NULL)
		free( (void *) hstMask );

	return true;
	
} // main

