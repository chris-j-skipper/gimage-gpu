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
const char MEASUREMENT_SET[] = "measurement_set:";
const char OUTPUT_PREFIX[] = "output_prefix:";
const char CELL_SIZE[] = "cell_size:";
const char PIXELS_UV[] = "pixels_uv:";
const char W_PLANES[] = "w_planes:";
const char OVERSAMPLE[] = "oversample:";
const char FIELD_ID[] = "field-id:";
const char FILE_ID[] = "file-id:";
const char SPW[] = "spw:";
const char MINOR_CYCLES[] = "minor_cycles:";
const char LOOP_GAIN[] = "loop_gain:";
const char CYCLEFACTOR[] = "cyclefactor:";
const char THRESHOLD[] = "threshold:";
const char OUTPUT_RA[] = "output-ra:";
const char OUTPUT_DEC[] = "output-dec:";
const char DATA_FIELD[] = "data-field:";
const char VISIBILITY_BATCH_SIZE[] = "visibility-batch-size:";
const char WEIGHTING[] = "weighting:";
const char ROBUST_PARAMETER[] = "robust:";
const char A_PLANES[] = "a_planes:";
const char MOSAIC[] = "mosaic:";
const char AIRY_DISK_DIAMETER[] = "airy-disk-diameter:";
const char AIRY_DISK_BLOCKAGE[] = "airy-disk-blockage:";
const char STOKES[] = "stokes:";
const char TELESCOPE[] = "telescope:";
const char CACHE_LOCATION[] = "cache-location:";

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
enum addsubtract { ADD, SUBTRACT };
enum griddegrid { GRID, DEGRID };
enum weighting { NONE, NATURAL, UNIFORM, ROBUST };
enum findpixel { CLOSEST, FURTHEST };
enum beamtype { GAUSSIAN, AIRY };
enum stokes { STOKES_I, STOKES_Q, STOKES_U, STOKES_V, STOKES_NONE };
enum telescope { UNKNOWN_TELESCOPE, ASKAP, EMERLIN, ALMA, ALMA_7M, ALMA_12M, VLA };
enum firstorlast { FIRST, LAST };
enum ffttype { C2C, F2C, C2F, F2F };
enum masktype { MASK_MAX, MASK_MIN };

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

// variables needed for batch gridding on the GPU.
int _hstGPURows = 0;
int _hstImageBatches = 1;

int _hstPsfSize = 0;		// the psf size may be smaller than the grid if we're using a grid that is too large to fit on the gpu. a smaller psf size
					//will be selected.

// data parameters.
char _hstFieldID[1024] = "\0";
char _hstDataField[100] = "CORRECTED_DATA";
char _hstSpwRestriction[1024] = "\0";
char _hstMeasurementSetPath[1024] = "\0";
char _hstOutputPrefix[1024] = "output";
char _hstCacheLocation[1024] = "\0";
	
// samples.
int * _hstNumSamples = NULL;
int _hstSampleBatchSize = 0;

// channels.
double * _hstAverageWavelength = NULL;
double _hstMinWavelength = 0.0, _hstMaxWavelength = 0.0;

// fields used in beam mosaicing.
int _hstNumFieldsForBeamMosaic = -1;

// weighting.
weighting _hstWeighting = NONE;
float * _hstWeight = NULL;
double * _hstAverageWeight = NULL;
double _hstRobust = 0.0;

// A-projection
bool _hstAProjection = false;
int _hstAPlanes = 1;
double _hstAiryDiskDiameter = 25.0;	// the diameter of the Airy disk.
double _hstAiryDiskBlockage = 0.0;	// the width of the blockage at the centre of the Airy disk.
bool _hstDiskDiameterSupplied = false;
bool _hstDiskBlockageSupplied = false;

// primary beams.
int _hstNumPrimaryBeams = 0;
float * _hstPrimaryBeam = NULL;			// this primary beam is used for mosaicing, and is in the reference frame of the mosaic position.
float * _hstPrimaryBeamAProjection = NULL;		// this primary beam is used for a-projection, and is in the reference frame of the mosaic component position.
int _hstBeamSize = -1;
double _hstBeamCellSize = 1;
double _hstBeamWidth = 0;
float ** _hstPrimaryBeamPtr = NULL;		// we define pointers to the primary beam arrays, which will be indexes using the mosaic image index. 
float ** _hstPrimaryBeamMosaicingPtr = NULL;	//
float ** _hstPrimaryBeamAProjectionPtr = NULL;	//

// field id for each sample.
int * _hstFieldIDArray = NULL;

// file id if we have multiple files.
int _hstMeasurementSets = 0;
int * _hstMosaicID = NULL;
bool _hstCacheData = false;				// we set this flag to true if we need to cache and uncache our data.

// mosaic?
bool _hstUseMosaicing = false;
bool _hstFileMosaic = false;
bool _hstBeamMosaic = false;
int _numMosaicImages = 0;
float * _hstPrimaryBeamPattern = NULL;
float * _hstNormalisationPattern = NULL;

// w-plane details.
double ** _hstWPlaneMean = NULL;
double ** _hstWPlaneMax = NULL;
int *** _hstVisibilitiesInKernelSet = NULL;
int _hstKernelSets = 1;
	
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
int _hstPreferredVisibilityBatchSize = 20000000;
long int * _hstGriddedVisibilities = NULL, * _hstGriddedVisibilitiesPerField = NULL;
long int _griddedVisibilitiesForBeamMosaic = 0;

// the batches of data in the file.
int * _hstNumberOfStages = NULL;
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

// texture maps.
cudaChannelFormatDesc _channelDesc = cudaCreateChannelDesc( 32, 0/*grid*/, 0, 0, cudaChannelFormatKindFloat );
texture< float, cudaTextureType3D, cudaReadModeElementType > _kernelTextureReal;
texture< float, cudaTextureType3D, cudaReadModeElementType > _kernelTextureImag;

// cuda arrays for texture maps.
cudaArray * _devKernelArrayRealPtr = NULL;
cudaArray * _devKernelArrayImagPtr = NULL;

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
__constant__ int _devSupport;
__constant__ int _devKernelSize;
__constant__ int _devAASupport;
__constant__ int _devAAKernelSize;
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

__global__ void devCalculateGaussianError( float/*grid*/ * pImage, double * pError, int pSizeOfFittingRegion, double pCentreX, double pCentreY,
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
//	devMoveCellsForFFT()
//
//	CJS: 10/06/2020
//
//	Moves cells from the batch staging area into the FFT area.
//

__global__ void devMoveCellsForFFT( cufftComplex * pTo, cufftComplex * pFrom, int pFFTSize, int pRowBatchSize, int pSize )
{
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;

	// ensure we are within the FFT region.
	if (i < pSize && j < pRowBatchSize)
	{

		// the output position.
		int iFFT = (i / pFFTSize);

		// the position within the input fft.
		int iColumn = (i % pFFTSize);

		// the input column.
		int iColumnIn = (iColumn * (pSize / pFFTSize)) + iFFT;

		// do the x-axis part of the FFT shift.
		iColumnIn = (iColumnIn + (pSize / 2)) % pSize;

		// output position.
		int outputPosition = (iFFT * pFFTSize * pRowBatchSize) + (j * pFFTSize) + iColumn;

		// update to grid.
		pTo[ outputPosition ] = pFrom[ (j * pSize) + iColumnIn ];

	}

} // devMoveCellsForFFT

__global__ void devMoveCellsForFFT( float * pTo, float * pFrom, int pFFTSize, int pRowBatchSize, int pSize )
{
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;

	// ensure we are within the FFT region.
	if (i < pSize && j < pRowBatchSize)
	{

		// the output position.
		int iFFT = (i / pFFTSize);

		// the position within the input fft.
		int iColumn = (i % pFFTSize);

		// the input column.
		int iColumnIn = (iColumn * (pSize / pFFTSize)) + iFFT;

		// do the x-axis part of the FFT shift.
		iColumnIn = (iColumnIn + (pSize / 2)) % pSize;

		// output position.
		int outputPosition = (iFFT * pFFTSize * pRowBatchSize) + (j * pFFTSize) + iColumn;

		// update to grid.
		pTo[ outputPosition ] = pFrom[ (j * pSize) + iColumnIn ];

	}

} // devMoveCellsForFFT

//
//	devConvertFloatToComplex()
//
//	CJS: 03/07/2020
//
//	Converts a floating point array into a complex array.
//

__global__ void devConvertFloatToComplex( cufftComplex * pTo, float * pFrom, int pSizeX, int pSizeY )
{
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;

	// ensure we are within the bounds of the array.
	if (i < pSizeX && j < pSizeY)
	{
		cufftComplex newValue = { .x = pFrom[ (j * pSizeX) + i ], .y = 0.0 };
		pTo[ (j * pSizeX) + i ] = newValue;
	}

} // devConvertFloatToComplex

//
//	devCalculateExponentialsForFFT()
//
//	CJS: 19/06/2020
//
//	Calculate the exponential values for a grid of FFTs. We calculate one value for each row and each column, and these can be multiplied together
//	to recover the unique FFT for the grid cell.
//

__global__ void devCalculateExponentialsForFFT( cufftComplex * pData, int pSize, int pFFTXIndex, int pFFTYIndex, int pMultiplier )
{

	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	// ensure we are within the region of interest.
	if (index < (pSize * 2))
	{

		// and which grid row/column ?
		double exponent = 0;
		if (index >= pSize)

			// we are processing a row. calculate the exponent.
			exponent = pMultiplier * 2.0 * PI * (double) (index - pSize) * (double) pFFTYIndex / (double) pSize;
			
		else

			// we are processing a column. calculate the exponent. Also, we need to do an x-axis FFT shift here.
//			exponent = pMultiplier * 2.0 * PI * (double) ((index + (pSize / 2)) % pSize) * (double) pFFTXIndex / (double) pSize;
			exponent = pMultiplier * 2.0 * PI * (double) index * (double) pFFTXIndex / (double) pSize;

		// calculate the exponential function.
		cufftComplex exponential;
		sincosf( exponent, &(exponential.y), &(exponential.x) );

		// update the array.
		pData[ index ] = exponential;


	}

} // devCalculateExponentialsForFFT

//
//	devApplyFFTExponential()
//
//	CJS: 11/06/2020
//
//	Apply an exponential function to the result of the FFT. This function is required when performing the FFT in chunks.
//

//__global__ void devApplyFFTExponential( cufftComplex * pToPtr, cufftComplex * pFromPtr, cufftComplex * pExponentialData, int pYOffset, int pSize, int pFFTSize,
//						bool pNeedImaginary )
//{

//	extern __shared__ cufftComplex shrRowExponential[];

//	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
//	int j = (blockIdx.y * blockDim.y) + threadIdx.y;

	// get the coordinates within the overall grid.
//	int iGrid = i;
//	int jGrid = j + pYOffset;
	
	// ensure we are within the grid dimensions.
//	if (jGrid < pSize && threadIdx.x == 0 && threadIdx.y == 0)
//		shrRowExponential[ 0 ] = pExponentialData[ pSize + jGrid ];
	
//	__syncthreads();
	
	// ensure we are within the grid dimensions.
//	if (iGrid < pSize && jGrid < pSize)
//	{

		// calculate the exponential function by multiplying the column and row exponentials.
//		cufftComplex exponential = multComplex( pExponentialData[ iGrid ], shrRowExponential[ 0 ] );

		// get the coordinates within the fft.
//		int iFFT = iGrid % pFFTSize;
//		int jFFT = jGrid % pFFTSize;

		// we do the x-axis FFT shift here.
//		i = (i + (pSize / 2)) % pSize;

		// multiply the fft by the exponential, and add to the staging area.
//		cufftComplex toAdd = multComplex(	/* pOne = */ pFromPtr[ (jFFT * pFFTSize) + iFFT ],
//							/* pTwo = */ exponential );
//		pToPtr[ (j * pSize) + i ].x += toAdd.x;
//		if (pNeedImaginary == true)
//			pToPtr[ (j * pSize) + i ].y += toAdd.y;

//	}

//} // devApplyFFTExponential

__global__ void devApplyFFTExponential( cufftComplex * pToPtr, cufftComplex * pFromPtr, cufftComplex * pExponentialData, int pFirstFFT, int pNumFFTs, int pSize,
						int pFFTSize, bool pNeedImaginary )
{

	extern __shared__ cufftComplex shrRowExponential[];

	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;

	// ensure the FFT and j coordinate are within the required bounds.
	bool okToContinue = false;
	int iGrid = -1, jGrid = -1, iFFT = -1, jFFT = -1;
	if (i < (pFFTSize * pNumFFTs) && j < pFFTSize)
	{

		// get the FFT index.
		int fft = (i / pFFTSize) + pFirstFFT;

		// get the fft coordinates.
		int fftX = fft % (pSize / pFFTSize);
		int fftY = fft / (pSize / pFFTSize);

		// calculate the coordinates within the fft.
		iFFT = (i % pFFTSize);
		jFFT = j;

		// get the coordinates within the overall grid.
		iGrid = (fftX * pFFTSize) + iFFT;
		jGrid = (fftY * pFFTSize) + jFFT;
	
		// ensure we are within the grid dimensions.
		if (iGrid < pSize && jGrid < pSize)
		{
			okToContinue = true;
			if (threadIdx.x == 0 && threadIdx.y == 0)
				shrRowExponential[ 0 ] = pExponentialData[ pSize + jGrid ];
		}

	}
	
	__syncthreads();
	
	// ensure we are within the grid dimensions.
	if (okToContinue == true)
	{

		// calculate the exponential function by multiplying the column and row exponentials.
		cufftComplex exponential = multComplex( pExponentialData[ iGrid ], shrRowExponential[ 0 ] );

		// multiply the fft by the exponential, and add to the staging area.
		cufftComplex toAdd = multComplex(	/* pOne = */ pFromPtr[ (jFFT * pFFTSize) + iFFT ],
							/* pTwo = */ exponential );
		pToPtr[ (j * pFFTSize * pNumFFTs) + i ].x += toAdd.x;
		if (pNeedImaginary == true)
			pToPtr[ (j * pFFTSize * pNumFFTs) + i ].y += toAdd.y;

	}
	

} // devApplyFFTExponential

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
//	devSearchForVis()
//
//	CJS: 11/05/2020
//
//	Find the first or last visibility with a particular row ID from a sorted list of grid positions.
//

__global__ void devSearchForVis( VectorI * pGridPositions, int pRowID, int pNumVisibilities, firstorlast pFirstOrLast, int * pVisibilityID )
{

	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	// ensure index is within the number of visibilities.
	if (index < pNumVisibilities)
	{

		bool foundItem = false;

		// are we looking for the first item with this row number ?
		if (pFirstOrLast == FIRST)

			// check if this item has the required row id.
			if (pGridPositions[ index ].v >= pRowID)
			{

				// is this the first item in the list ?
				if (index == 0)
					foundItem = true;
				else

					// does the previous visibility have a different row number ?
					foundItem = (pGridPositions[ index - 1 ].v < pRowID);

			}

		// are we looking for the last item with this row number ?
		if (pFirstOrLast == LAST)

			// check if this item has the required row id.
			if (pGridPositions[ index ].v <= pRowID)
			{


				// is this the last item in the list ?
				if (index == pNumVisibilities - 1)
					foundItem = true;
				else

					// does the next visibility have a different row number ?
					foundItem = (pGridPositions[ index + 1 ].v > pRowID);

			}

		// did we find the row we were looking for ?
		if (foundItem == true)
			*pVisibilityID = index;

	}

} // devSearchForVis

//
//	devFindCutoffPixelParallel()
//
//	CJS: 03/04/2019
//
//	Finds the furthest pixel from the centre of the kernel that is at least 1% of the maximum kernel value.
//

__global__ void devFindCutoffPixelParallel( cufftComplex/*grid*/ * pKernel, int pSize, double * pMaxValue, int pCellsPerThread, int * pTmpResults,
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
			float/*grid*/ pixelValue = pKernel[ index ].x;
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

__global__ void devFindCutoffPixelParallel( float/*grid*/ * pKernel, int pSize, double * pMaxValue, int pCellsPerThread, int * pTmpResults, double pCutoffFraction,
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
			float/*grid*/ pixelValue = pKernel[ index ];
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
//	pYOffset			we may be gridding visibilities in batches, so we need to subtract this offset from the y value to get the true grid pixel.
//	pNumberOfRows			the number of rows in our grid. this may be smaller than the number of columns.
//	pSize				the size of the image
//	pComplex			are we gridding complex or non-complex visibilities? (0 - N, 1 - Y)
//
//	shared memory:
//	shrVisibility - holds all the visibilities in this thread block.
//	shrGridPosition - holds all the grid positions (3 x int) for the visibilities in this thread block.
//	shrKernelIndex - holds the kernel indexes for the visibilities in this thread block.
//	shrWeight - holds the weight for this visibility.
//

__global__ void devGridVisibilities( cufftComplex/*grid*/ * pGrid, cufftComplex * pVisibility, int pVisibilitiesPerBlock, int pBlocksPerVisibility,
					VectorI * pGridPosition, int * pKernelIndex, float * pWeight, int pNumVisibilities, griddegrid pGridDegrid,
					int pNumKernels, int pYOffset, int pNumberOfRows, int pSize, int pComplex )
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
		visibilityArrayIndex = threadIdx.y / _devKernelSize;
		visibilityIndex = (blockIndex * pVisibilitiesPerBlock) + visibilityArrayIndex;
		
		// calculate the kernel position.
		kernelX = threadIdx.x;
		kernelY = (threadIdx.y % _devKernelSize);
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

	// check if this is the first thread for an item. if so, we need to get the visibility, grid position and kernel index
	// and store them in shared memory.
	if (firstThread == true && visibilityIndex < pNumVisibilities)
	{
		
		// get grid position, kernel index and visibility.
		if (pGridDegrid == GRID)
		{
			if (pComplex == 1)
				shrVisibility[ visibilityArrayIndex ] = pVisibility[ visibilityIndex ];
			else
			{
				shrVisibility[ visibilityArrayIndex ].x = ((double *) pVisibility)[ visibilityIndex ];
				shrVisibility[ visibilityArrayIndex ].y = 0.0;
			}
		}
		else
		{
			shrVisibility[ visibilityArrayIndex ].x = 0.0;
			shrVisibility[ visibilityArrayIndex ].y = 0.0;
		}
		shrGridPosition[ visibilityArrayIndex ] = pGridPosition[ visibilityIndex ];

		// subtract the y offset. this is because we may only be gridding a small region of our full grid, and the regions are full width with
		// restricted rows (pNumberOfRows). The y offset gives the row number of the start of this grid portion.
		shrGridPosition[ visibilityArrayIndex ].v -= pYOffset;

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
	if ( kernelX < _devKernelSize && kernelY < _devKernelSize && visibilityIndex < pNumVisibilities )
	{
		
		// get exact grid coordinates,
		VectorI grid = shrGridPosition[ visibilityArrayIndex ];

		// if we are gridding then we want to pick a grid offset that matches the kernel offset. if we are degridding then the grid offset should be opposite
		// to the kernel offset.
		if (pGridDegrid == GRID)
		{
			grid.u += kernelX - _devSupport;
			grid.v += kernelY - _devSupport;
		}
		else
		{
			grid.u -= kernelX - _devSupport;
			grid.v -= kernelY - _devSupport;
		}
					
		// get kernel value.
		cufftComplex kernel;
		kernel.x = tex3D( _kernelTextureReal, kernelX, kernelY, shrKernelIndex[ visibilityArrayIndex ] );
		if (pComplex == 1)
			kernel.y = tex3D( _kernelTextureImag, kernelX, kernelY, shrKernelIndex[ visibilityArrayIndex ] );
		else
			kernel.y = 0.0;
	
		// is this pixel within the grid range?
		if ((grid.u >= 0) && (grid.u < pSize) && (grid.v >= 0) && (grid.v < pNumberOfRows))
		{
						
			// get pointer to grid.
			cufftComplex/*grid*/ * gridPtr = NULL;
			if (pComplex == 1)
				gridPtr = &pGrid[ (grid.v * pSize) + grid.u ];
			else
				gridPtr = (cufftComplex/*grid*/ *) &((float/*grid*/ *) pGrid)[ (grid.v * pSize) + grid.u ];
						
			// update the grid using an atomic add (passing a pointer and the value to add).
			if (pGridDegrid == GRID)
			{

				// add complex and real numbers differently.
				if (pComplex == 1)
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
//						atomicAddDouble( (double *) gridPtr, shrVisibility[ visibilityArrayIndex ].x * kernel.x *
//									shrWeight[ visibilityArrayIndex ] ); /*grid*/
						atomicAdd( (float *) gridPtr, shrVisibility[ visibilityArrayIndex ].x * kernel.x *
									shrWeight[ visibilityArrayIndex ] ); /*grid*/
					else
//						atomicAddDouble( (double *) gridPtr, shrVisibility[ visibilityArrayIndex ].x * kernel.x ); /*grid*/
						atomicAdd( (float *) gridPtr, shrVisibility[ visibilityArrayIndex ].x * kernel.x ); /*grid*/


				}

			}
			else
			{

				// add complex and real numbers differently.
				if (pComplex == 1)
					addComplex( &shrVisibility[ visibilityArrayIndex ], multComplex( *gridPtr, kernel ) );
				else
					atomicAdd( &shrVisibility[ visibilityArrayIndex ].x, gridPtr->x * kernel.x );

			}
	
		}
	
	}

	__syncthreads();

	// if this is the first thread for an item, and we are degridding, we need to copy the visibility out of shared memory by atomic adding it to the required
	// memory address.
	if (firstThread == true && visibilityIndex < pNumVisibilities && pGridDegrid == DEGRID)
		addComplex( /* pOne = */ &pVisibility[ visibilityIndex ], /* pTwo = */ shrVisibility[ visibilityArrayIndex ] );
	
} // devGridVisibilities

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
//			phasor.x = cos( 2.0 * PI * pPhase[ sample ] / wavelength );
//			phasor.y = sin( 2.0 * PI * pPhase[ sample ] / wavelength );

			// multiply phasor by visibility.
			newVis = multComplex( /* pOne = */ pVisibility[ visibilityIndex ], /* pTwo = */ phasor );

		}
		pVisibility[ visibilityIndex ].x = (float/*vis*/) newVis.x;
		pVisibility[ visibilityIndex ].y = (float/*vis*/) newVis.y;

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

			oversample.u = (int)round( exact.u * (double)pOversample);
			oversampleIndex.u = mod( oversample.u, pOversample );
			grid.u = intFloor( oversample.u, pOversample ) + (pSize / 2);
		
			oversample.v = (int)round( exact.v * (double)pOversample);
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

__global__ void devApplyDensityMap( cufftComplex * pVisibilities, int * pDensityMap )
{

	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	// ensure we are within the visibility batch limits.
	if (index < _devVisibilityBatchSize)
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

__global__ void devRearrangeKernel( float/*grid*/ * pTarget, float/*grid*/ * pSource, long int pElements )
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

__global__ void devMakeCleanBeam( float/*grid*/ * pCleanBeam, double pAngle, double pR1, double pR2, double pX, double pY, int pSize )
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

__global__ void devUpdateKernel( cufftComplex/*grid*/ * pKernel, cufftComplex/*grid*/ * pImage, int pSupport, int pOversample, int pOversampleI, int pOversampleJ,
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
		cufftComplex/*grid*/ value;
		value.x = 0; value.y = 0;
		if (imageI >= 0 && imageI < pWorkspaceSize && imageJ >= 0 && imageJ < pWorkspaceSize)
		{
			value.x = pImage[ (imageJ * pWorkspaceSize) + imageI ].x;
			value.y = pImage[ (imageJ * pWorkspaceSize) + imageI ].y;
		}
			
		// update the kernel.
		pKernel[ (j * kernelSize) + i ] = value;
		
	}
	
} // devUpdateKernel

//
//	devAddSubtractBeams()
//
//	CJS: 06/11/2015
//
//	Add or subtracts the clean beam/dirty beam from the clean image/dirty image.
//
//	The window size is the support size of the region of the beam that is to be added or subtracted. the rest of the beam outside this region is ignored.
//

__global__ void devAddSubtractBeams(	float/*grid*/ * pImage, float/*grid*/ * pBeam, double * pMaxValue, int pWindowSize, double pLoopGain,
						int pImageWidth, int pImageHeight, int pBeamSize, addsubtract pAddSubtract	)
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
			float/*grid*/ * tmpImage = &pImage[ (y * pImageWidth) + x ];
			float/*grid*/ * tmpPSF = &pBeam[ (j * pBeamSize) + i ];
						
			// add/subtract the psf (scaled).
			float/*grid*/ value = maxValue * *tmpPSF * pLoopGain;
			if (pAddSubtract == ADD)
				*tmpImage += value;
			else
				*tmpImage -= value;
					
		}
		
	}
	
} // devAddSubtractBeams

//
//	devAddPixelToModelImage()
//
//	CJS: 17/08/2018
//
//	Builds the model image by adding a single pixel.
//	We only have one thread because there is only one pixel to add with each iteration.
//

__global__ void devAddPixelToModelImage( float/*grid*/ * pModelImage, double * pMaxValue, double pLoopGain, int pSize )
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

__global__ void devUpdateComplexArray( cufftComplex * pArray, int pElements, float/*vis*/ pReal, float/*vis*/ pImaginary )
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

__global__ void devGenerateAAKernel( cufftComplex/*grid*/ * pAAKernel, int pKernelSize, int pWorkspaceSize )
{

	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;

	// ensure we're within the bounds of the kernel.
	if (i < pKernelSize && j < pKernelSize)
	{
		
		int supportSize = (pKernelSize - 1) / 2;
		int workspaceSupport = pWorkspaceSize / 2;

		// i and j are the coordinates within the aa-kernel. get the equivalent coordinates within the whole image.
		int imageI = i + workspaceSupport - supportSize;
		int imageJ = j + workspaceSupport - supportSize;
		
		// ensure we're within the bounds of the whole image.
		if (imageI >= 0 && imageI < pWorkspaceSize && imageJ >= 0 && imageJ < pWorkspaceSize)
		{

			// get kernel pointer.
			cufftComplex/*grid*/ * aaKernelPtr = &pAAKernel[ (imageJ * pWorkspaceSize) + imageI ];
					
			// calculate the x-offset from the centre of the kernel.
			double x = (double) (i - supportSize);
			double y = (double) (j - supportSize);
					
			// now, calculate the anti-aliasing kernel.
			double val = 0;
			val = spheroidalWaveFunction( x / ((double) supportSize + 0.5) );
			val *= spheroidalWaveFunction( y / ((double) supportSize + 0.5) );

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

__global__ void devGenerateWKernel( cufftComplex/*grid*/ * pWKernel, double pW, int pWorkspaceSize, double pCellSizeRadians, griddegrid pGridDegrid, int pSize )
{

	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;

	// ensure we're within the bounds of the kernel.
	if (i < pWorkspaceSize && j < pWorkspaceSize)
	{

		int workspaceSupport = pWorkspaceSize / 2;
					
		// calculate the x-offset from the centre of the image.
		double x = (double) (i - workspaceSupport);
		double y = (double) (j - workspaceSupport);

		// use the cell size to convert x and y into radians, and correct the size of the image-plane kernel for the size of the workspace.
		x *= pCellSizeRadians * (double) pSize / (double) pWorkspaceSize;
		y *= pCellSizeRadians * (double) pSize / (double) pWorkspaceSize;

		// calculate r^2 (dist from centre of image squared).
		double rSquared = pow( x, 2 ) + pow( y, 2 );

		// calculate kernel value. if gridding then the sine term should be negative; if degridding it should be positive.
		cufftComplex kernelValue = { .x = 0.0, .y = 0.0 };
		if (rSquared <= 1.0)
		{

			//
			// visibility equation is:
			//
			// 	Vj = int{ I(l,m) / sqrt(1-l^2-m^2) x exp[ i.2.PI( uj.l + vj.m + wj.(sqrt(1-l^2-m^2)-1) ) ] }
			//
			// if we're gridding we want to remove:
			//
			//	exp[ i.2.PI.wj.(sqrt(1-l^2-m^2)-1) ] / sqrt(1-l^2-m^2)
			//
			// from the image domain using a multiplication, so that we're left with the Fourier transform:
			//
			// 	Vj = int{ I(l,m} x exp[ i.2.PI( uj.l + vj.m ) ] }.
			//
			// if we're degridding then we want to add these components back in.
			//
			double exponent = (pGridDegrid == GRID ? -1 : +1) * 2.0 * PI * pW * (sqrt( 1.0 - rSquared ) - 1.0);
			sincosf( exponent, &(kernelValue.y), &(kernelValue.x) );
//			kernelValue.x = cos( 2.0 * PI * pW * (sqrt( 1.0 - rSquared ) - 1.0) ) * sqrt( 1.0 - rSquared );
//			kernelValue.y = (pGridDegrid == GRID ? -1 : +1) * sin( 2.0 * PI * pW * (sqrt( 1.0 - rSquared ) - 1.0) ) * sqrt( 1.0 - rSquared );

			// if we are gridding then we multiply the kernel by sqrt( 1.0 - r^2 ). If degridding then this is a division.
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

__global__ void devGenerateAKernel( cufftComplex/*grid*/ * pAKernel, float * pPrimaryBeam, int pPrimaryBeamSupport, int pWorkspaceSize, griddegrid pGridDegrid )
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
		int workspaceSupport = pWorkspaceSize / 2;

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
//	devSetPrimaryBeamForGriddingAndDegridding()
//
//	CJS: 06/09/2019
//
//	Sets the primary beam for gridding and degridding.
//

__global__ void devSetPrimaryBeamForGriddingAndDegridding( float * pImage, int pSize, griddegrid pGridDegrid, bool pAProjection )
{

	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;

	// ensure we're within the bounds of the image.
	if (i < pSize && j < pSize)
	{

		float value = pImage[ (j * pSize) + i ];
		if (value != 0.0)
		{

			// if we are gridding then we use the primary beam pattern as the kernel. this is so that we adding each field such that it is weighted by its
			// primary beam.
			// we then divide the final image by the sum of primary beams squared (beam1^2 + beam2^2 + ... ), which we hold in _hstNormalisationPattern, in order
			// to correct for our weighting function and also to remove the effect of the primary beam which will naturally be in our image.
			// if we are degridding then we simply want to reintroduce the primary beam, so we use: val = beam<i>.
			if (pGridDegrid == GRID)
			{
				if (pAProjection == true)
					value = pow( value, 2 );
				else
					value = pow( value, 1 );
			}
			else
				value = pow( value, 1 );

		}

		// update the beam.
		pImage[ (j * pSize) + i ] = value;

	}

} // devSetPrimaryBeamForGriddingAndDegridding

//
//	devSubtractVisibilities()
//
//	CJS: 15/08/2018
//
//	Subtract the model visibilities from the original visibilities.
//

__global__ void devSubtractVisibilities( cufftComplex * pOriginalVisibility, cufftComplex * pModelVisibility )
{

	int i = (blockIdx.x * blockDim.x) + threadIdx.x;

	// ensure we're within the bounds of the kernel.
	if (i >= 0 && i < _devVisibilityBatchSize)
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
//	For gridding the x-axis of the kernel size must be less than the block size limit (usually 512). Rows
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
	
	// store the total number of X and Y threads.
	_blockSize2D.x = pThreadsX;
	_blockSize2D.y = pThreadsY;
	
	// how many items can we process in each block? if we have more than 1, then we tile then in the x-direction - there is only ever one item in the y-direction.
	_itemsPerBlock = _maxThreadsPerBlock / (_blockSize2D.x * _blockSize2D.y);
	
	// ensure _itemsPerBlock is not larger than the required number of items.
	if (_itemsPerBlock > pItems)
		_itemsPerBlock = pItems;
	
	// if we have more than one item then increase the block size.
	if (_itemsPerBlock > 0)
		_blockSize2D.y = _itemsPerBlock * _blockSize2D.y;
	else
	{

		// _itemsPerBlock is zero. this means we need to split each kernel over a number of blocks, which are tiled in the y direction.
		// do we still have too many threads for one block?
		while ((_blockSize2D.x * _blockSize2D.y) > _maxThreadsPerBlock)
		{
		
			// increment the number of Y blocks.
			_gridSize2D.y = _gridSize2D.y + 1;
			_blockSize2D.y = (int)ceil( (double)pThreadsY / (double)_gridSize2D.y );
		
		}
	
		// we have now split our threads over multiple blocks, set items-per-block to 1.
		_itemsPerBlock = 1;
	
	}
	
	// set the number of blocks per item.
	_blocksPerItem = _gridSize2D.y;
	
	// divide the number of items by the number per block to get the total required number of blocks..
	int requiredBlocks = (int)ceil( (double)pItems / (double)_itemsPerBlock ) * _blocksPerItem;
	
	// ensure the grid size y-axis is less than the maximum allowed. the x-size should still be 1 at this point, so
	// we keep incrementing the x-size until the y-size is within the required limit.
	_gridSize2D.y = requiredBlocks;
	while (_gridSize2D.y > MAXIMUM_BLOCKS_PER_DIMENSION)
	{
		_gridSize2D.x = _gridSize2D.x + 1;
		_gridSize2D.y = (int)ceil( (double)requiredBlocks / (double)_gridSize2D.x );
	}
	
} // setThreadBlockSizeForGridding

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
//	Free the memory used to store the data for a mosaic image.
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
//	Store a whole set of visibilities, grid positions, kernel indexes, etc to disk, and free the memory.
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

	// free the memory.
	freeData( /* pWhatData = */ pWhatData );

} // cacheData

//
//	uncacheData()
//
//	CJS: 25/03/2019
//
//	Retrieve a whole set of visibilities, grid positions, kernel indexes, etc from disk.
//

void uncacheData( char * pFilenamePrefix, int pMosaicID, int pBatchID, int pWhatData )
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
		_hstVisibility = (cufftComplex *) malloc( _hstNumVisibilities[ pMosaicID ][ pBatchID ] * sizeof( cufftComplex ) );
		fread( (void *) _hstVisibility, sizeof( cufftComplex ), _hstNumVisibilities[ pMosaicID ][ pBatchID ], fr );
	}
	else
		fseek( fr, _hstNumVisibilities[ pMosaicID ][ pBatchID ] * sizeof( cufftComplex ), SEEK_CUR );
	if ((pWhatData & DATA_GRID_POSITIONS) == DATA_GRID_POSITIONS)
	{
		_hstGridPosition = (VectorI *) malloc( _hstNumVisibilities[ pMosaicID ][ pBatchID ] * sizeof( VectorI ) );
		fread( (void *) _hstGridPosition, sizeof( VectorI ), _hstNumVisibilities[ pMosaicID ][ pBatchID ], fr );
	}
	else
		fseek( fr, _hstNumVisibilities[ pMosaicID ][ pBatchID ] * sizeof( VectorI ), SEEK_CUR );
	if ((pWhatData & DATA_KERNEL_INDEXES) == DATA_KERNEL_INDEXES)
	{
		_hstKernelIndex = (int *) malloc( _hstNumVisibilities[ pMosaicID ][ pBatchID ] * sizeof( int ) );
		fread( (void *) _hstKernelIndex, sizeof( int ), _hstNumVisibilities[ pMosaicID ][ pBatchID ], fr );
	}
	else
		fseek( fr, _hstNumVisibilities[ pMosaicID ][ pBatchID ] * sizeof( int ), SEEK_CUR );
	if ((pWhatData & DATA_DENSITIES) == DATA_DENSITIES)
	{
		_hstDensityMap = (int *) malloc( _hstNumVisibilities[ pMosaicID ][ pBatchID ] * sizeof( int ) );
		fread( (void *) _hstDensityMap, sizeof( int ), _hstNumVisibilities[ pMosaicID ][ pBatchID ], fr );
	}
	else
		fseek( fr, _hstNumVisibilities[ pMosaicID ][ pBatchID ] * sizeof( int ), SEEK_CUR );
	if ((pWhatData & DATA_WEIGHTS) == DATA_WEIGHTS)
	{
		_hstWeight = (float *) malloc( _hstNumVisibilities[ pMosaicID ][ pBatchID ] * sizeof( float ) );
		fread( (void *) _hstWeight, sizeof( float ), _hstNumVisibilities[ pMosaicID ][ pBatchID ], fr );
	}
	else
		fseek( fr, _hstNumVisibilities[ pMosaicID ][ pBatchID ] * sizeof( float ), SEEK_CUR );
	if ((pWhatData & DATA_RESIDUAL_VISIBILITIES) == DATA_RESIDUAL_VISIBILITIES)
	{
		_hstResidualVisibility = (cufftComplex *) malloc( _hstNumVisibilities[ pMosaicID ][ pBatchID ] * sizeof( cufftComplex ) );
		fread( (void *) _hstResidualVisibility, sizeof( cufftComplex ), _hstNumVisibilities[ pMosaicID ][ pBatchID ], fr );
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

bool performFFT( cufftComplex/*grid*/ ** pdevGrid, int pSize, fftdirection pFFTDirection, cufftHandle pFFTPlan, ffttype pFFTType )
{
	
	bool ok = true;
	cudaError_t err;
	
	// reserve some memory to hold a temporary image, which allows us to do an FFT shift.
	cufftComplex/*grid*/ * devTmpImage = NULL;
	reserveGPUMemory( (void **) &devTmpImage, sizeof( cufftComplex/*grid*/ ) * pSize * pSize, "reserving device memory for the FFT temporary image" );

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
		reserveGPUMemory( (void **) pdevGrid, pSize * pSize * sizeof( cufftComplex/*grid*/ ), "creating device memory for enlarged grid following FFT" );
	}

	// move image from temporary memory.
	cudaMemcpy( (void *) *pdevGrid, (void *) devTmpImage, pSize * pSize * sizeof( cufftComplex/*grid*/ ), cudaMemcpyDeviceToDevice );

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
	reserveGPUMemory( (void **) &devTmpImage, pSize * pSize * sizeof( cufftComplex/*grid*/ ), "reserving device memory for the FFT temporary image" );

	// move image to temporary memory.
	cudaMemcpy( (void *) devTmpImage, (void *) *pdevGrid, pSize * pSize * sizeof( cufftComplex/*grid*/ ), cudaMemcpyDeviceToDevice );

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
//	doFFTShift_host()
//
//	CJS: 11/05/2020
//
//	Do a forward or inverse FFT shift on the host.
//

void doFFTShift_host( cufftComplex * phstGrid, long int pSize, bool pSwapX, bool pSwapY )
{
	
	long int pivot = pSize / 2;

	if (pSwapX == true || pSwapY == true)
	{

		// swap the rows.
		cufftComplex * tmp = (cufftComplex *) malloc( pSize * sizeof( cufftComplex ) );
		if (pSwapX == true && pSwapY == true)
			for ( long int j = 0; j < pivot; j++ )
			{
				memcpy( tmp, &phstGrid[ j * pSize ], pSize * sizeof( cufftComplex ) );
				memcpy( &phstGrid[ j * pSize ], &phstGrid[ ((j + pivot) * pSize) + pivot ], pivot * sizeof( cufftComplex ) );
				memcpy( &phstGrid[ (j * pSize) + pivot ], &phstGrid[ (j + pivot) * pSize ], pivot * sizeof( cufftComplex ) );
				memcpy( &phstGrid[ (j + pivot) * pSize ], &tmp[ pivot ], pivot * sizeof( cufftComplex ) );
				memcpy( &phstGrid[ ((j + pivot) * pSize) + pivot ], tmp, pivot * sizeof( cufftComplex ) );
			}
		else if (pSwapX == true && pSwapY == false)
			for ( long int j = 0; j < pSize; j++ )
			{
				memcpy( tmp, &phstGrid[ j * pSize ], pSize * sizeof( cufftComplex ) );
				memcpy( &phstGrid[ j * pSize ],  &tmp[ pivot ], pivot * sizeof( cufftComplex ) );
				memcpy( &phstGrid[ (j * pSize) + pivot ], tmp, pivot * sizeof( cufftComplex ) );
			}
		else if (pSwapX == false && pSwapY == true)
			for ( long int j = 0; j < pivot; j++ )
			{
				memcpy( tmp, &phstGrid[ j * pSize ], pSize * sizeof( cufftComplex ) );
				memcpy( &phstGrid[ j * pSize ], &phstGrid[ (j + pivot) * pSize ], pSize * sizeof( cufftComplex ) );
				memcpy( &phstGrid[ (j + pivot) * pSize ], tmp, pSize * sizeof( cufftComplex ) );
			}
		free( (void *) tmp );

	}

} // doFFTShift_host

void doFFTShift_host( float * phstGrid, long int pSize, bool pSwapX, bool pSwapY )
{
	
	long int pivot = pSize / 2;

	if (pSwapX == true || pSwapY == true)
	{

		// swap the rows.
		float * tmp = (float *) malloc( pSize * sizeof( float ) );
		if (pSwapX == true && pSwapY == true)
			for ( long int j = 0; j < pivot; j++ )
			{
				memcpy( tmp, &phstGrid[ j * pSize ], pSize * sizeof( float ) );
				memcpy( &phstGrid[ j * pSize ], &phstGrid[ ((j + pivot) * pSize) + pivot ], pivot * sizeof( float ) );
				memcpy( &phstGrid[ (j * pSize) + pivot ], &phstGrid[ (j + pivot) * pSize ], pivot * sizeof( float ) );
				memcpy( &phstGrid[ (j + pivot) * pSize ], &tmp[ pivot ], pivot * sizeof( float ) );
				memcpy( &phstGrid[ ((j + pivot) * pSize) + pivot ], tmp, pivot * sizeof( float ) );
			}
		else if (pSwapX == true && pSwapY == false)
			for ( long int j = 0; j < pSize; j++ )
			{
				memcpy( tmp, &phstGrid[ j * pSize ], pSize * sizeof( float ) );
				memcpy( &phstGrid[ j * pSize ],  &tmp[ pivot ], pivot * sizeof( float ) );
				memcpy( &phstGrid[ (j * pSize) + pivot ], tmp, pivot * sizeof( float ) );
			}
		else if (pSwapX == false && pSwapY == true)
			for ( long int j = 0; j < pivot; j++ )
			{
				memcpy( tmp, &phstGrid[ j * pSize ], pSize * sizeof( float ) );
				memcpy( &phstGrid[ j * pSize ], &phstGrid[ (j + pivot) * pSize ], pSize * sizeof( float ) );
				memcpy( &phstGrid[ (j + pivot) * pSize ], tmp, pSize * sizeof( float ) );
			}
		free( (void *) tmp );

	}

} // doFFTShift_host

//
//	rearrangeImageForFFT()
//
//	CJS: 26/06/2020
//
//	Rearranges an image portion ready for caching to disk or processing the FFTs.
//

void rearrangeImageForFFT( cufftComplex * phstOutput, cufftComplex * phstGrid, cufftComplex * pdevGrid, cufftComplex * pdevStagingArea, int pRowFFT, int pFFTSize,
				int pRowBatchSize, int pSize )
{

//struct timespec time1, time2;
//double hostToDevice = 0.0;
//double rearrange = 0.0;
//double deviceToHost = 0.0;

	int numFFTsPerAxis = pSize / pFFTSize;

	// process the rows in this row of FFTs in batches.
	for ( int rowBatch = 0; rowBatch < pFFTSize / pRowBatchSize; rowBatch++ )
	{

//clock_gettime( CLOCK_REALTIME, &time1 );

		// move some rows to the device.
		for ( int row = 0; row < pRowBatchSize; row++ )
		{

			// do a y-axis FFT shift.
			int inputRow = (((row + (rowBatch * pRowBatchSize)) * numFFTsPerAxis) + pRowFFT + (pSize / 2)) % pSize;

			moveHostToDevice(	(void *) &pdevStagingArea[ row * pSize ],
						(void *) &phstGrid[ (long int) inputRow * (long int) pSize ],
						pSize * sizeof( cufftComplex ),
						"moving fft row to the device" );

		}

//clock_gettime( CLOCK_REALTIME, &time2 );
//hostToDevice += getTime( time1, time2 );

		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess)
			printf( "unknown CUDA error on line %d (%s)\n", __LINE__, cudaGetErrorString( err ) );

//clock_gettime( CLOCK_REALTIME, &time1 );

		// rearrange the columns so that they're suitable for fft batching.
		setThreadBlockSize2D( pSize, pRowBatchSize );
		devMoveCellsForFFT<<< _gridSize2D, _blockSize2D >>>(	/* pTo = */ pdevGrid,
									/* pFrom = */ pdevStagingArea,
									/* pFFTSize = */ pFFTSize,
									/* pRowBatchSize = */ pRowBatchSize,
									/* pSize = */ pSize );

		cudaDeviceSynchronize();

//clock_gettime( CLOCK_REALTIME, &time2 );
//rearrange += getTime( time1, time2 );

		err = cudaGetLastError();
		if (err != cudaSuccess)
			printf( "unknown CUDA error on line %d (%s)\n", __LINE__, cudaGetErrorString( err ) );

//clock_gettime( CLOCK_REALTIME, &time1 );

		// move the row of FFTs to the original image.
		cufftComplex * devGrid = pdevGrid;
		cufftComplex * hstOutput = &phstOutput[ (rowBatch * pRowBatchSize) * pFFTSize ];
		for ( int fft = 0; fft < numFFTsPerAxis; fft++, devGrid += (pFFTSize * pRowBatchSize), hstOutput += (pFFTSize * pFFTSize) )
			moveDeviceToHost(	(void *) hstOutput,
						(void *) devGrid,
						pFFTSize * pRowBatchSize * sizeof( cufftComplex ),
						"moving part of a row of FFTs to the host" );

//clock_gettime( CLOCK_REALTIME, &time2 );
//deviceToHost += getTime( time1, time2 );

	}

//printf( "hostToDevice %f\n", hostToDevice );
//printf( "rearrange %f\n", rearrange );
//printf( "deviceToHost %f\n", deviceToHost );

} // rearrangeImageForFFT

void rearrangeImageForFFT( float * phstOutput, float * phstGrid, float * pdevGrid, float * pdevStagingArea, int pRowFFT, int pFFTSize,
				int pRowBatchSize, int pSize )
{

	int numFFTsPerAxis = pSize / pFFTSize;

	// process the rows in this row of FFTs in batches.
	for ( int rowBatch = 0; rowBatch < pFFTSize / pRowBatchSize; rowBatch++ )
	{

		// move some rows to the device.
		for ( int row = 0; row < pRowBatchSize; row++ )
		{

			// do a y-axis FFT shift.
			int inputRow = (((row + (rowBatch * pRowBatchSize)) * numFFTsPerAxis) + pRowFFT + (pSize / 2)) % pSize;

			moveHostToDevice(	(void *) &pdevStagingArea[ row * pSize ],
						(void *) &phstGrid[ (long int) inputRow * (long int) pSize ],
						pSize * sizeof( float ),
						"moving fft row to the device" );

		}

		// rearrange the columns so that they're suitable for fft batching.
		setThreadBlockSize2D( pSize, pRowBatchSize );
		devMoveCellsForFFT<<< _gridSize2D, _blockSize2D >>>(	/* pTo = */ pdevGrid,
									/* pFrom = */ pdevStagingArea,
									/* pFFTSize = */ pFFTSize,
									/* pRowBatchSize = */ pRowBatchSize,
									/* pSize = */ pSize );

		// move the row of FFTs to the original image.
		float * devGrid = pdevGrid;
		float * hstOutput = &phstOutput[ (rowBatch * pRowBatchSize) * pFFTSize ];
		for ( int fft = 0; fft < numFFTsPerAxis; fft++, devGrid += (pFFTSize * pRowBatchSize), hstOutput += (pFFTSize * pFFTSize) )
			moveDeviceToHost(	(void *) hstOutput,
						(void *) devGrid,
						pFFTSize * pRowBatchSize * sizeof( float ),
						"moving part of a row of FFTs to the host" );

	}

} // rearrangeImageForFFT

//
//	performFFT_host()
//
//	CJS: 11/05/2020
//
//	FFT the gridded visibilites on the host using FFTW.
//

void performFFT_host( cufftComplex/*grid*/ ** phstGrid, long int pSize, fftdirection pFFTDirection, ffttype pFFTType )
{

	const int FFT_SIZE = 4096;
	const int ROW_BATCH_SIZE = 1024;
	const int NUM_FFTS_TO_PROCESS = 12;

	const char EXTENSION[] = "-image-cache.dat";

	int numFFTsPerAxis = pSize / FFT_SIZE;

struct timespec time1, time2, startTime;
double saveData = 0.0;
double loadData = 0.0;
double processImageForFFT = 0.0;
double calculateExponentials = 0.0;
double executeFFT = 0.0;
double applyExponentials = 0.0;
double cpuMaths = 0.0;
double synchronise1 = 0.0;
double finalFFTShift = 0.0;
double reverseY = 0.0;
double saveOutputCache = 0.0;
double loadOutputCache = 0.0;
double totalTime = 0.0;

	// check how much memory is available.
	struct sysinfo memInfo;
	sysinfo( &memInfo );

	// build cache filename.
	char cachedFilename[ 100 ];
	if (_hstCacheLocation[0] != '\0')
		sprintf( cachedFilename, "%s%s%s", _hstCacheLocation, _hstOutputPrefix, EXTENSION );
	else
		sprintf( cachedFilename, "%s%s", _hstOutputPrefix, EXTENSION );

	// how many bytes per pixel in the input and output images ?
	int inputSizePerPixel = ((pFFTType == F2F || pFFTType == F2C) ? 1 : 2 );
	int outputSizePerPixel = ((pFFTType == F2F || pFFTType == C2F) ? 1 : 2 );

	// calculate the size of the input and output images, and work out what we can have in memory at any one time.
	long int inputImageSize = (long int) pSize * (long int) pSize * (long int) sizeof( float ) * (long int) inputSizePerPixel;
	long int outputImageSize = (long int) pSize * (long int) pSize * (long int) sizeof( float ) * (long int) outputSizePerPixel;

	// do we need to cache the input image ?
	bool cacheInputImage = ((inputImageSize + outputImageSize) > (long int) ((double) memInfo.totalram * 0.9));

	// how many chunks should we use to process the output image ?
	int outputChunks = 1;
	if (outputImageSize > ((double) memInfo.totalram * 0.9) && outputImageSize <= ((double) memInfo.totalram * 1.8))
		outputChunks = 2;
	if (outputImageSize > ((double) memInfo.totalram * 1.8))
		outputChunks = 4;

printf( "cacheInputImage %i, outputChunks %i\n", (cacheInputImage ? 1 : 0), outputChunks ); // cjs-mod
clock_gettime( CLOCK_REALTIME, &startTime );

	// declare memory for holding a single row.
	cufftComplex/*grid*/ * hstTmp = (cufftComplex/*grid*/ *) malloc( pSize * sizeof( cufftComplex/*grid*/ ) );

	// reverse the y-axis (for some reason).
	if (pFFTDirection == FORWARD)
	{
		if (pFFTType == C2C || pFFTType == C2F)
			for ( int j = 0; j < pSize / 2; j++ )
			{
				memcpy( hstTmp, &(*phstGrid)[ j * pSize ], pSize * sizeof( cufftComplex/*grid*/ ) );
				memmove( &(*phstGrid)[ j * pSize ], &(*phstGrid)[ (pSize - j - 1) * pSize ], pSize * sizeof( cufftComplex/*grid*/ ) );
				memcpy( &(*phstGrid)[ (pSize - j - 1) * pSize ], hstTmp, pSize * sizeof( cufftComplex/*grid*/ ) );
			}
		if (pFFTType == F2F || pFFTType == F2C)
		{
			float * hstGrid = (float *) *phstGrid;
			for ( int j = 0; j < pSize / 2; j++ )
			{
				memcpy( hstTmp, &hstGrid[ j * pSize ], pSize * sizeof( float ) );
				memmove( &hstGrid[ j * pSize ], &hstGrid[ (pSize - j - 1) * pSize ], pSize * sizeof( float ) );
				memcpy( &hstGrid[ (pSize - j - 1) * pSize ], hstTmp, pSize * sizeof( float ) );
			}
		}
	}

	// get a pointer to the grid.
	float * hstGrid = (float *) *phstGrid;

	// create a grid section on the host.
	cufftComplex * hstGridSection = NULL;

	// if we're not caching the original image then we need somewhere to put it.
	cufftComplex * hstOriginalImage = NULL;

	// create some memory on the device.
	cufftComplex * devGrid = NULL;
	reserveGPUMemory( (void **) &devGrid, ROW_BATCH_SIZE * pSize * sizeof( cufftComplex ), "reserving device memory for the fft grid" );
	cufftComplex * devStagingArea = NULL;
	reserveGPUMemory( (void **) &devStagingArea, ROW_BATCH_SIZE * pSize * sizeof( float ) * inputSizePerPixel , "reserving device memory for the fft staging area" );

	if (cacheInputImage == true)
	{

		printf( "\r                caching input image.....       " );
		fflush( stdout );

		// create some temporary memory for the grid section.
		hstGridSection = (cufftComplex *) malloc( FFT_SIZE * pSize * sizeof( float ) * inputSizePerPixel );
		if (hstGridSection == NULL)
		{
			printf( "failed to allocate memory for grid section on line %d. Aborting.\n", __LINE__ );
			abort();
		}

clock_gettime( CLOCK_REALTIME, &time1 );

		// save the image to disk. we will load it in stages.
		for ( int rowFFT = 0; rowFFT < numFFTsPerAxis; rowFFT++ )
		{

			printf( "\r                caching input image.....%i%%       ", rowFFT * 100 / numFFTsPerAxis );
			fflush( stdout );

			// rearrange the data in this row of FFTs ready for performing the FFTs.
			if (pFFTType == F2F || pFFTType == F2C)
				rearrangeImageForFFT(	/* phstOutput = */ (float *) hstGridSection,
							/* phstGrid = */ (float *) hstGrid,
							/* pdevGrid = */ (float *) devGrid,
							/* pdevStagingArea = */ (float *) devStagingArea,
							/* pRowFFT = */ rowFFT,
							/* pFFTSize = */ FFT_SIZE,
							/* pRowBatchSize = */ ROW_BATCH_SIZE,
							/* pSize = */ pSize );
			else
				rearrangeImageForFFT(	/* phstOutput = */ (cufftComplex *) hstGridSection,
							/* phstGrid = */ (cufftComplex *) hstGrid,
							/* pdevGrid = */ (cufftComplex *) devGrid,
							/* pdevStagingArea = */ (cufftComplex *) devStagingArea,
							/* pRowFFT = */ rowFFT,
							/* pFFTSize = */ FFT_SIZE,
							/* pRowBatchSize = */ ROW_BATCH_SIZE,
							/* pSize = */ pSize );

			// save data.
			if (pFFTType == F2F || pFFTType == F2C)
				saveComplexData(	/* pFilename = */ cachedFilename,
							/* pData = */ (float *) hstGridSection,
							/* pOffset = */ (long int) (rowFFT * FFT_SIZE) * (long int) pSize * (long int) sizeof( float ),
							/* pSize = */ FFT_SIZE * pSize );
			else
				saveComplexData(	/* pFilename = */ cachedFilename,
							/* pData = */ (cufftComplex *) hstGridSection,
							/* pOffset = */ (long int) (rowFFT * FFT_SIZE) * (long int) pSize * (long int) sizeof( cufftComplex ),
							/* pSize = */ FFT_SIZE * pSize );

		}

clock_gettime( CLOCK_REALTIME, &time2 );
saveData += getTime( time1, time2 );

		// free the grid section.
		if (hstGridSection != NULL)
			free( (void *) hstGridSection );

	}
	else
	{

clock_gettime( CLOCK_REALTIME, &time1 );

		// we need to re-order the input rows.
		hstOriginalImage = (cufftComplex *) malloc( inputImageSize );
		float * hstOriginalImageFloat = (float *) hstOriginalImage;

		for ( int rowFFT = 0; rowFFT < numFFTsPerAxis; rowFFT++ )

			// rearrange the data in this row of FFTs ready for performing the FFTs.
			if (pFFTType == F2F || pFFTType == F2C)
				rearrangeImageForFFT(	/* phstOutput = */ (float *) &hstOriginalImageFloat[ ((long int) rowFFT * (long int) FFT_SIZE *
														(long int) pSize) ],
							/* phstGrid = */ (float *) hstGrid,
							/* pdevGrid = */ (float *) devGrid,
							/* pdevStagingArea = */ (float *) devStagingArea,
							/* pRowFFT = */ rowFFT,
							/* pFFTSize = */ FFT_SIZE,
							/* pRowBatchSize = */ ROW_BATCH_SIZE,
							/* pSize = */ pSize );
			else
				rearrangeImageForFFT(	/* phstOutput = */ &hstOriginalImage[ ((long int) rowFFT * (long int) FFT_SIZE * (long int) pSize) ],
							/* phstGrid = */ (cufftComplex *) hstGrid,
							/* pdevGrid = */ devGrid,
							/* pdevStagingArea = */ (cufftComplex *) devStagingArea,
							/* pRowFFT = */ rowFFT,
							/* pFFTSize = */ FFT_SIZE,
							/* pRowBatchSize = */ ROW_BATCH_SIZE,
							/* pSize = */ pSize );

clock_gettime( CLOCK_REALTIME, &time2 );
processImageForFFT += getTime( time1, time2 );

	}

	// free devGrid and devStagingArea, and re-dimension them for the FFT size.
	cudaFree( (void *) devGrid );
	cudaFree( (void *) devStagingArea );
	reserveGPUMemory( (void **) &devGrid, FFT_SIZE * FFT_SIZE * NUM_FFTS_TO_PROCESS * sizeof( cufftComplex ), "reserving device memory for the fft grid" );
	reserveGPUMemory( (void **) &devStagingArea, FFT_SIZE * FFT_SIZE * NUM_FFTS_TO_PROCESS * sizeof( cufftComplex ) ,
					"reserving device memory for the fft staging area" );

	// create the grid section for the required number of FFTs.
	hstGridSection = (cufftComplex *) malloc( FFT_SIZE * FFT_SIZE * NUM_FFTS_TO_PROCESS * sizeof( float ) * inputSizePerPixel );

	// how many fft chunks are we populating ?
	int fftChunksToRead = (numFFTsPerAxis * numFFTsPerAxis);
	int fftChunksToWrite = (fftChunksToRead / outputChunks);

	// how many FFTs should we process on the first iteration ?
	int fftsToProcess = minimum( NUM_FFTS_TO_PROCESS, fftChunksToRead );

	// either load the first N ffts, or copy from the original image.
	if (cacheInputImage == true)
	{

clock_gettime( CLOCK_REALTIME, &time1 );

		// load the data for the first batch of FFTs.
		if (pFFTType == F2F || pFFTType == F2C)
			getComplexData(	/* pFilename = */ cachedFilename,
					/* pData = */ (float *) hstGridSection,
					/* pOffset = */ 0,
					/* pSize = */ FFT_SIZE * FFT_SIZE * fftsToProcess );
		else
			getComplexData(	/* pFilename = */ cachedFilename,
					/* pData = */ (cufftComplex *) hstGridSection,
					/* pOffset = */ 0,
					/* pSize = */ FFT_SIZE * FFT_SIZE * fftsToProcess );

clock_gettime( CLOCK_REALTIME, &time2 );
loadData += getTime( time1, time2 );

	}
	else

		// copy the first section ready for use.
		memcpy( hstGridSection, hstOriginalImage, FFT_SIZE * FFT_SIZE * fftsToProcess * sizeof( float ) * inputSizePerPixel );

	// create two staging areas on the host.
	cufftComplex * hstStagingArea[ 2 ];
	hstStagingArea[ 0 ] = (cufftComplex *) malloc( FFT_SIZE * FFT_SIZE * NUM_FFTS_TO_PROCESS * sizeof( cufftComplex ) );
	hstStagingArea[ 1 ] = (cufftComplex *) malloc( FFT_SIZE * FFT_SIZE * NUM_FFTS_TO_PROCESS * sizeof( cufftComplex ) );

	// create memory to store the exponential functions.
	cufftComplex * devExponentials;
	reserveGPUMemory( (void **) &devExponentials, NUM_FFTS_TO_PROCESS * pSize * 2 * sizeof( cufftComplex ), "reserving device memory for the exponential values" );

	// free the grid and recreate it with some fraction of the memory.
	if (outputChunks > 1 || (outputImageSize != inputImageSize))
	{
		free( (void *) *phstGrid );
		*phstGrid = (cufftComplex *) malloc( (long int) outputImageSize / (long int) outputChunks );
		hstGrid = (float *) *phstGrid;
	}

	// clear the output grid.
	memset( (void *) hstGrid, 0, (long int) outputImageSize / (long int) outputChunks );

	// pin the host memory.
	cudaHostRegister( hstStagingArea[ 0 ], FFT_SIZE * FFT_SIZE * NUM_FFTS_TO_PROCESS * sizeof( cufftComplex ), 0 );
	cudaHostRegister( hstStagingArea[ 1 ], FFT_SIZE * FFT_SIZE * NUM_FFTS_TO_PROCESS * sizeof( cufftComplex ), 0 );

	// create a new cuda stream.
	cudaStream_t cudaStream;
	cudaStreamCreate( &cudaStream );

	// create some cuda events so we can schedule the cpu work.
	cudaEvent_t event[ 2 ];
	cudaEventCreate( &event[ 0 ] );
	cudaEventCreate( &event[ 1 ] );

	// initialise the FFT.
	cufftHandle pFFTPlan = initialiseFFT( FFT_SIZE );

	// process the output image in chunks.
	for ( int whichSection = 0; whichSection < outputChunks; whichSection++ )
	{

		// how many input batches are there ?
		int inputBatches = fftChunksToRead / NUM_FFTS_TO_PROCESS;
		if (fftChunksToRead % NUM_FFTS_TO_PROCESS > 0)
			inputBatches++;
	
		for ( int inputBatch = 0; inputBatch < inputBatches; inputBatch++ )
		{

			printf( "\r                performing fft.....%i%%         ",
					(whichSection * 100 / outputChunks) + (inputBatch * 100 / (inputBatches * outputChunks)) );
			fflush( stdout );

			// how many FFTs should we process in this batch?
			int inputFFTsToProcess = minimum( NUM_FFTS_TO_PROCESS, fftChunksToRead - (inputBatch * NUM_FFTS_TO_PROCESS) );

			if (pFFTType == F2F || pFFTType == F2C)
			{

				// upload the FFTs to the FFT staging area.
				moveHostToDevice(	/* pToPtr = */ (void *) devStagingArea,
							/* pFromPtr = */ (void *) hstGridSection,
							/* pSize = */ FFT_SIZE * FFT_SIZE * inputFFTsToProcess * sizeof( float ),
							/* pTask = */ "moving FFT batch to device" );

				// since our data is not complex, then we need to add the imaginary parts.
				setThreadBlockSize2D( FFT_SIZE * inputFFTsToProcess, FFT_SIZE );
				devConvertFloatToComplex<<< _gridSize2D, _blockSize2D >>>(	/* pTo = */ devGrid,
												/* pFrom = */ (float *) devStagingArea,
												/* pSizeX = */ FFT_SIZE * inputFFTsToProcess,
												/* pSizeY = */ FFT_SIZE );

			}
			else

				// upload the FFTs to the FFT staging area.
				moveHostToDevice(	/* pToPtr = */ (void *) devGrid,
							/* pFromPtr = */ (void *) hstGridSection,
							/* pSize = */ FFT_SIZE * FFT_SIZE * inputFFTsToProcess * sizeof( cufftComplex ),
							/* pTask = */ "moving FFT batch to device" );

clock_gettime( CLOCK_REALTIME, &time1 );

			// execute all the FFTs.
			for ( int fft = 0; fft < inputFFTsToProcess; fft++ )
				cufftExecC2C(	pFFTPlan, &devGrid[ fft * FFT_SIZE * FFT_SIZE ], &devGrid[ fft * FFT_SIZE * FFT_SIZE], 
						(pFFTDirection == INVERSE ? CUFFT_INVERSE : CUFFT_FORWARD) );

clock_gettime( CLOCK_REALTIME, &time2 );
executeFFT += getTime( time1, time2 );

clock_gettime( CLOCK_REALTIME, &time1 );

			// generate the exponential values for these FFTs. the x-axis FFT shift is done in this process.
			int threads = pSize * 2, blocks;
			setThreadBlockSize1D( &threads, &blocks );
			for ( int fft = 0; fft < inputFFTsToProcess; fft++ )
				devCalculateExponentialsForFFT<<< blocks, threads >>>
							(	/* pData = */ &devExponentials[ fft * pSize * 2 ],
								/* pSize = */ pSize,
								/* pFFTXIndex = */ (((inputBatch * NUM_FFTS_TO_PROCESS) + fft) % numFFTsPerAxis),
								/* pFFTYIndex = */ (((inputBatch * NUM_FFTS_TO_PROCESS) + fft) / numFFTsPerAxis),
								/* pMultiplier = */ (pFFTDirection == INVERSE ? 1.0 : -1.0) );

clock_gettime( CLOCK_REALTIME, &time2 );
calculateExponentials += getTime( time1, time2 );

			// how many output batches are there ?
			int outputBatches = fftChunksToWrite / NUM_FFTS_TO_PROCESS;
			if (fftChunksToWrite % NUM_FFTS_TO_PROCESS > 0)
				outputBatches++;

			for ( int outputBatch = 0; outputBatch < outputBatches + 1; outputBatch++ )
			{

				// ignore the final loop iteration - it's for CPU work only.
				if (outputBatch < outputBatches)
				{

					int outputFFTsToProcess = minimum( NUM_FFTS_TO_PROCESS, fftChunksToWrite - (outputBatch * NUM_FFTS_TO_PROCESS) );

					// clear the staging area.
					cudaMemsetAsync( devStagingArea, 0, FFT_SIZE * FFT_SIZE * outputFFTsToProcess * sizeof( cufftComplex ), cudaStream );

					// process each of the FFTs in this batch of FFTs.
					for ( int fft = 0; fft < inputFFTsToProcess; fft++ )
					{

//cudaDeviceSynchronize();

//clock_gettime( CLOCK_REALTIME, &time1 );

						// populate the staging area with a whole batch of ffts multiplied by the exponential. we could submit all these ffts in
						// one, but that would mean using atomic adds, so we choose to submit them separately.
						_blockSize2D.x = minimum( _maxThreadsPerBlock, FFT_SIZE ); _blockSize2D.y = 1;
						_gridSize2D.x = FFT_SIZE * outputFFTsToProcess / _blockSize2D.x; _gridSize2D.y = FFT_SIZE;
						devApplyFFTExponential<<< _gridSize2D, _blockSize2D, sizeof( cufftComplex ), cudaStream >>>
								(	/* pToPtr = */ devStagingArea,
									/* pFromPtr = */ &devGrid[ fft * FFT_SIZE * FFT_SIZE ],
									/* pExponentialData = */ &devExponentials[ fft * pSize * 2 ],
									/* pFirstFFT = */ (whichSection * fftChunksToWrite) + (outputBatch * NUM_FFTS_TO_PROCESS),
									/* pNumFFTs = */ outputFFTsToProcess,
									/* pSize = */ pSize,
									/* pFFTSize = */ FFT_SIZE,
									/* pNeedImaginary = */ (pFFTType == C2C || pFFTType == F2C) );

//cudaDeviceSynchronize();

//clock_gettime( CLOCK_REALTIME, &time2 );
//applyExponentials += getTime( time1, time2 );

					}

					// copy data from devGrid to the host.
					moveDeviceToHostAsync(	/* pToPtr = */ (void *) hstStagingArea[ outputBatch % 2 ],
								/* pFromPtr = */ (void *) devStagingArea,
								/* pSize = */ FFT_SIZE * FFT_SIZE * outputFFTsToProcess * sizeof( cufftComplex ),
								/* pTask = */ "moving fft grid to staging area on the host",
								/* pStream = */ cudaStream );

					// record an event so we know when this is finished.
					cudaEventRecord( event[ outputBatch % 2 ], cudaStream );

				}

				// before the first cpu operation we expect to wait a while for the gpu stuff to finish. we can save time by loading the next grid section
				// from the disk cache.
				if (outputBatch == 1 && inputBatch < inputBatches - 1)
				{

					// how many FFTs should we process in this batch ?
					int nextBatchSize = minimum( NUM_FFTS_TO_PROCESS, fftChunksToRead - ((inputBatch + 1) * NUM_FFTS_TO_PROCESS) );

					if (cacheInputImage == true)
					{

clock_gettime( CLOCK_REALTIME, &time1 );

						// get the next grid section from the disk cache.
						if (pFFTType == F2F || pFFTType == F2C)
							getComplexData(	/* pFilename = */ cachedFilename,
									/* pData = */ (float *) hstGridSection,
									/* pOffset = */ (long int) ((inputBatch + 1) * NUM_FFTS_TO_PROCESS) *
											(long int) (FFT_SIZE * FFT_SIZE * sizeof( float )),
									/* pSize = */ FFT_SIZE * FFT_SIZE * nextBatchSize );
						else
							getComplexData(	/* pFilename = */ cachedFilename,
									/* pData = */ (cufftComplex *) hstGridSection,
									/* pOffset = */ (long int) ((inputBatch + 1) * NUM_FFTS_TO_PROCESS) *
											(long int) (FFT_SIZE * FFT_SIZE * sizeof( cufftComplex )),
									/* pSize = */ FFT_SIZE * FFT_SIZE * nextBatchSize );

clock_gettime( CLOCK_REALTIME, &time2 );
loadData += getTime( time1, time2 );

					}
					else
					{
						if (pFFTType == F2F || pFFTType == F2C)
						{
							float * hstOriginalImageFloat = (float *) hstOriginalImage;
							memcpy( hstGridSection, &hstOriginalImageFloat[ (long int) ((inputBatch + 1) * NUM_FFTS_TO_PROCESS) *
													(long int) (FFT_SIZE * FFT_SIZE) ],
									FFT_SIZE * FFT_SIZE * nextBatchSize * sizeof( float ) );
						}
						else
							memcpy( hstGridSection, &hstOriginalImage[ (long int) ((inputBatch + 1) * NUM_FFTS_TO_PROCESS) *
													(long int) (FFT_SIZE * FFT_SIZE) ],
									FFT_SIZE * FFT_SIZE * nextBatchSize * sizeof( cufftComplex ) );
					}

				}

				// load the image for the next output chunk.
				if (outputBatch == 1 && inputBatch == inputBatches - 1 && cacheInputImage == true && whichSection < outputChunks - 1)
				{

					// how many FFTs should we process in this batch?
					int nextBatchSize = minimum( NUM_FFTS_TO_PROCESS, fftChunksToRead );

clock_gettime( CLOCK_REALTIME, &time1 );

					// get the next grid section from the disk cache.
					if (pFFTType == F2F || pFFTType == F2C)
						getComplexData(	/* pFilename = */ cachedFilename,
								/* pData = */ (float *) hstGridSection,
								/* pOffset = */ 0,
								/* pSize = */ FFT_SIZE * FFT_SIZE * nextBatchSize );
					else
						getComplexData(	/* pFilename = */ cachedFilename,
								/* pData = */ (cufftComplex *) hstGridSection,
								/* pOffset = */ 0,
								/* pSize = */ FFT_SIZE * FFT_SIZE * nextBatchSize );

clock_gettime( CLOCK_REALTIME, &time2 );
loadData += getTime( time1, time2 );

				}

				// the cpu step lags the gpu step by one loop iteration.
				if (outputBatch > 0)
				{

					int outputFFTsToProcess = minimum( NUM_FFTS_TO_PROCESS, fftChunksToWrite - ((outputBatch - 1) * NUM_FFTS_TO_PROCESS) );

					// calculate the start FFT coordinates.
					int iFFT = (((outputBatch - 1) * NUM_FFTS_TO_PROCESS)) % numFFTsPerAxis;
					int jFFT = (((outputBatch - 1) * NUM_FFTS_TO_PROCESS)) / numFFTsPerAxis;

clock_gettime( CLOCK_REALTIME, &time1 );

					// wait for the last download to complete.
					cudaEventSynchronize( event[ (outputBatch - 1) % 2 ] );

clock_gettime( CLOCK_REALTIME, &time2 );
synchronise1 += getTime( time1, time2 );

clock_gettime( CLOCK_REALTIME, &time1 );

					int currentFFT = 0;
					while (currentFFT < outputFFTsToProcess)
					{

						// how many FFTs are we updating on this row ?
						int maxFFT = minimum( iFFT + (outputFFTsToProcess - currentFFT) - 1, numFFTsPerAxis - 1 );
						int numFFTsOnRow = maxFFT - iFFT + 1;

						// do x-axis FFT shift.
						int start1 = ((iFFT * FFT_SIZE) + (pSize / 2)) % pSize;
						int size1 = 0, size2 = 0;
						if (start1 < (pSize / 2))
							size1 = minimum( (pSize / 2) - start1, numFFTsOnRow * FFT_SIZE );
						else
						{
							size1 = minimum( pSize - start1, numFFTsOnRow * FFT_SIZE );
							size2 = (numFFTsOnRow * FFT_SIZE) - size1;
						}
//printf( "%d: start1 %i, size1 %i, size2 %i, maxFFT %i, numFFTsOnRow %i, iFFT %i, jFFT %i, outputBatch %i, outputFFTsToProcess %i\n", __LINE__,
//		start1, size1, size2, maxFFT, numFFTsOnRow, iFFT, jFFT, outputBatch, outputFFTsToProcess );

						// for each row we make either one or two memory copies, depending upon whether our destination memory overlaps
						// the centre of the row. we do this because we are doing the x-axis FFT shift as part of our processing.
						for ( int row = 0; row < FFT_SIZE; row++ )
						{

							float * gridPtr = &hstGrid[ (((long int) ((jFFT * FFT_SIZE) + row) * (long int) pSize) +
											(long int) start1) * (long int) outputSizePerPixel ];
							float * stagingAreaPtr = (float *) &hstStagingArea[ (outputBatch - 1) % 2 ]
													[ (row * FFT_SIZE * outputFFTsToProcess) + (currentFFT * FFT_SIZE) ];

							// move the first half of the row.
							if (pFFTType == C2F || pFFTType == F2F)
								for ( int i = 0; i < size1; i++, gridPtr++, stagingAreaPtr += 2 )
									(*gridPtr) += (*stagingAreaPtr);
							else
								for ( int i = 0; i < size1 * 2; i++, gridPtr++, stagingAreaPtr++ )
									(*gridPtr) += (*stagingAreaPtr);

							// if the data crosses the half-way point of the row then move the second half of the row.
							if (size2 > 0)
							{

								float * gridPtr = &hstGrid[ (long int) ((jFFT * FFT_SIZE) + row) * (long int) pSize *
												(long int) outputSizePerPixel ];
								float * stagingAreaPtr = (float *) &hstStagingArea[ (outputBatch - 1) % 2 ]
														[ (row * FFT_SIZE * outputFFTsToProcess) +
															(currentFFT * FFT_SIZE) + size1 ];
								if (pFFTType == C2F || pFFTType == F2F)
									for ( int i = 0; i < size2; i++, gridPtr++, stagingAreaPtr += 2 )
										(*gridPtr) += (*stagingAreaPtr);
								else
									for ( int i = 0; i < size2 * 2; i++, gridPtr++, stagingAreaPtr++ )
										(*gridPtr) += (*stagingAreaPtr);

							}

						}

						// process the next row of FFTs.
						currentFFT += numFFTsOnRow;
						iFFT = 0;
						jFFT++;

					}

clock_gettime( CLOCK_REALTIME, &time2 );
cpuMaths += getTime( time1, time2 );

				}

			}

		}

		// save this data to the cache.
		if (whichSection < outputChunks - 1)
		{

			char saveFilename[ 100 ];

			// build filename.
			if (_hstCacheLocation[0] != '\0')
				sprintf( saveFilename, "%s%s-out-%i%s", _hstCacheLocation, _hstOutputPrefix, whichSection, EXTENSION );
			else
				sprintf( saveFilename, "%s-out-%i%s", _hstOutputPrefix, whichSection, EXTENSION );

clock_gettime( CLOCK_REALTIME, &time1 );

			// save data.
			saveComplexData(	/* pFilename = */ saveFilename,
						/* pData = */ hstGrid,
						/* pOffset = */ 0,
						/* pSize = */ (long int) pSize * (long int) pSize * (long int) outputSizePerPixel / (long int) outputChunks );

clock_gettime( CLOCK_REALTIME, &time2 );
saveOutputCache += getTime( time1, time2 );

			// clear the output grid.
			memset( (void *) hstGrid, 0, outputImageSize / (long int) outputChunks );

		}

	}
	printf( "\r                performing fft.....100%%         " );
	fflush( stdout );

	// finalise the FFT.
	finaliseFFT( pFFTPlan );

	// destroy cuda events.
	cudaEventDestroy( event[ 0 ] );
	cudaEventDestroy( event[ 1 ] );

	// destroy cuda stream.
	cudaStreamDestroy( cudaStream );

	// unpin the host memory.
	cudaHostUnregister( hstStagingArea[ 0 ] );
	cudaHostUnregister( hstStagingArea[ 1 ] );

	// free memory.
	if (devGrid != NULL)
		cudaFree( (void *) devGrid );
	if (hstStagingArea[ 0 ] != NULL)
		free( (void *) hstStagingArea[ 0 ] );
	if (hstStagingArea[ 1 ] != NULL)
		free( (void *) hstStagingArea[ 1 ] );
	if (devStagingArea != NULL)
		cudaFree( (void *) devStagingArea );
	if (hstGridSection != NULL)
		free( (void *) hstGridSection );
	if (devExponentials != NULL)
		cudaFree( (void *) devExponentials );

	// if we're not caching the input image, then we can free it now.
	if (cacheInputImage == false)
		free( (void *) hstOriginalImage );

	// expand the image.
	if (outputChunks > 1)
	{
		*phstGrid = (cufftComplex *) realloc( *phstGrid, outputImageSize );
		hstGrid = (float *) *phstGrid;
	}

	// move the data from the first part of the image to the appropriate position in the array.
	if (outputChunks == 4)
		memcpy( &hstGrid[ (long int) pSize * (long int) pSize * (long int) outputSizePerPixel / (long int) outputChunks ], hstGrid,
				outputImageSize / (long int) outputChunks );

clock_gettime( CLOCK_REALTIME, &time1 );

	// load the cached data into the remainder of the array.
	if (outputChunks > 1)
		for ( int i = 0; i < outputChunks; i++ )
		{

			// decide which section needs to be loaded in order to do a y-axis FFT shift.
			int getSection = -1;
			if (i == 0 && outputChunks == 4)
				getSection = 2;
			if (i == 1 && outputChunks == 2)
				getSection = 0;
			if (i == 2 || i == 3)
				getSection = i - 2;
			if (getSection > -1)
			{

				char saveFilename[ 100 ];

				// load the cached data.
				if (_hstCacheLocation[0] != '\0')
					sprintf( saveFilename, "%s%s-out-%i%s", _hstCacheLocation, _hstOutputPrefix, getSection, EXTENSION );
				else
					sprintf( saveFilename, "%s-out-%i%s", _hstOutputPrefix, getSection, EXTENSION );

				// get the next grid section from the disk cache.
				getComplexData(	/* pFilename = */ saveFilename,
						/* pData = */ &hstGrid[ (long int) i * (long int) pSize * (long int) pSize * (long int) outputSizePerPixel /
										(long int) outputChunks ],
						/* pOffset = */ 0,
						/* pSize = */ (long int) pSize * (long int) pSize * (long int) outputSizePerPixel / (long int) outputChunks );

			}

		}

clock_gettime( CLOCK_REALTIME, &time2 );
loadOutputCache += getTime( time1, time2 );

	// only do a Y-axis FFT shift it we're not using output chunks.
	if (outputChunks == 1)
	{

		printf( "\r                doing FFT shift.....           " );
		fflush( stdout );

clock_gettime( CLOCK_REALTIME, &time1 );

		// do inverse FFT shift.
		if (pFFTType == F2F || pFFTType == C2F)
			doFFTShift_host(	/* phstGrid = */ (float *) *phstGrid,
						/* pSize = */ pSize,
						/* pSwapX = */ false,
						/* pSwapY = */ true );
		else if (pFFTType == F2C || pFFTType == C2C)
			doFFTShift_host(	/* phstGrid = */ (cufftComplex *) *phstGrid,
						/* pSize = */ pSize,
						/* pSwapX = */ false,
						/* pSwapY = */ true );

clock_gettime( CLOCK_REALTIME, &time2 );
finalFFTShift += getTime( time1, time2 );

	}

clock_gettime( CLOCK_REALTIME, &time1 );

	// reverse the y-axis (for some reason).
	if (pFFTDirection == INVERSE)
	{
		if (pFFTType == C2C || pFFTType == F2C)
			for ( long int j = 0; j < pSize / 2; j++ )
			{
				memcpy( hstTmp, &(*phstGrid)[ j * pSize ], pSize * sizeof( cufftComplex/*grid*/ ) );
				memmove( &(*phstGrid)[ j * pSize ], &(*phstGrid)[ (pSize - j - 1) * pSize ], pSize * sizeof( cufftComplex/*grid*/ ) );
				memcpy( &(*phstGrid)[ (pSize - j - 1) * pSize ], hstTmp, pSize * sizeof( cufftComplex/*grid*/ ) );
			}
		if (pFFTType == C2F || pFFTType == F2F)
		{
			float * hstGrid = (float *) *phstGrid;
			for ( long int j = 0; j < pSize / 2; j++ )
			{
				memcpy( (float *) hstTmp, &hstGrid[ j * pSize ], pSize * sizeof( float ) );
				memmove( &hstGrid[ j * pSize ], &hstGrid[ (pSize - j - 1) * pSize ], pSize * sizeof( float ) );
				memcpy( &hstGrid[ (pSize - j - 1) * pSize ], (float *) hstTmp, pSize * sizeof( float ) );
			}
		}
	}

clock_gettime( CLOCK_REALTIME, &time2 );
reverseY += getTime( time1, time2 );

totalTime = getTime( startTime, time2 );

	printf( "\r                done                            \n" );
	fflush( stdout );

	// free memory.
	if (hstTmp != NULL)
		free( (void *) hstTmp );

	printf( "saveData %f\n", saveData );
	printf( "processImageForFFT %f\n", processImageForFFT );
	printf( "executeFFT %f\n", executeFFT );
	printf( "calculateExponentials %f\n", calculateExponentials );
//	printf( "applyExponentials %f\n", applyExponentials );
	printf( "{\n" );
	printf( "---- loadData %f\n", loadData );
	printf( "---- cpuMaths %f\n", cpuMaths );
	printf( "<parallel>\n" );
	printf( "---- synchronise1 %f\n", synchronise1 );
	printf( "}\n" );
	printf( "saveOutputCache %f\n", saveOutputCache );
	printf( "loadOutputCache %f\n", loadOutputCache );
	printf( "finalFFTShift %f\n", finalFFTShift );
	printf( "reverseY %f\n", reverseY );
	printf( "\n" );
	printf( "TOTAL TIME %f\n", totalTime );

} // performFFT_host

//
//	calculateAPlanes()
//
//	CJS: 03/04/2020
//
//	Calculate which channels/spws are in which A plane.
//

//void calculateAPlanes( int *** pWhichAPlane, double * phstAPlaneWavelength, double ** phstWavelength, int pNumSpws, int * phstNumChannels )
//{

//	printf( "Preparing for A projection.....\n" );

//	printf( "        setting %i a-plane(s) for %i SPW(s):\n", _hstAPlanes, pNumSpws );

	// calculate the minimum and maximum wavelengths.
//	double * minWavelength = (double *) malloc( pNumSpws * sizeof( double ) );
//	double * maxWavelength = (double *) malloc( pNumSpws * sizeof( double ) );
//	for ( int spw = 0; spw < pNumSpws; spw++ )
//		for ( int channel = 0; channel < phstNumChannels[ spw ]; channel++ )
//		{
//			if (phstWavelength[ spw ][ channel ] < minWavelength[ spw ] || channel == 0)
//				minWavelength[ spw ] = phstWavelength[ spw ][ channel ];
//			if (phstWavelength[ spw ][ channel ] > maxWavelength[ spw ] || channel == 0)
//				maxWavelength[ spw ] = phstWavelength[ spw ][ channel ];
//		}

	// merge any spws that overlap.				
//	int mergedSpws = pNumSpws;

//	int spw1 = 0, spw2 = 1;
//	while (spw2 < mergedSpws)
//		if (minWavelength[ spw1 ] <= maxWavelength[ spw2 ] && maxWavelength[ spw1 ] >= minWavelength[ spw2 ])
//		{

			// merge these spws.
//			if (minWavelength[ spw2 ] < minWavelength[ spw1 ])
//				minWavelength[ spw1 ] = minWavelength[ spw2 ];
//			if (maxWavelength[ spw2 ] > maxWavelength[ spw1 ])
//				maxWavelength[ spw1 ] = maxWavelength[ spw2 ];

			// shuffle the mins and maxes down.
//			int spw3 = spw2;
//			while (spw3 < mergedSpws - 1)
//			{
//				minWavelength[ spw3 ] = minWavelength[ spw3 + 1 ];
//				maxWavelength[ spw3 ] = maxWavelength[ spw3 + 1 ];
//				spw3++;
//			}

			// decrease the number of spws.
//			mergedSpws = mergedSpws - 1;
//			spw1 = 0;
//			spw2 = 1;

//		}
//		else
//		{

			// process next spw.
//			spw2 = spw2 + 1;
//			if (spw2 == mergedSpws)
//			{
//				spw1 = spw1 + 1;
//				spw2 = spw1 + 1;
//			}

//		}

	// sort the spws.
//	if (mergedSpws > 1)
//		for ( int spw1 = 0; spw1 < mergedSpws - 1; spw1++ )
//			for ( int spw2 = spw1 + 1; spw2 < mergedSpws; spw2++ )
//				if ( minWavelength[ spw2 ] < minWavelength[ spw1 ] )
//				{
//					double tmp = minWavelength[ spw1 ];
//					minWavelength[ spw1 ] = minWavelength[ spw2 ];
//					minWavelength[ spw2 ] = tmp;
//					tmp = maxWavelength[ spw1 ];
//					maxWavelength[ spw1 ] = maxWavelength[ spw2 ];
//					maxWavelength[ spw2 ] = tmp;
//				}

	// sum the wavelength range, and divide by the number of A-planes.
//	double range = 0.0;
//	for ( int i = 0; i < mergedSpws; i++ )
//		range = range + (maxWavelength[ i ] - minWavelength[ i ]);
//	range = range / (double) _hstAPlanes;

	// set the initial wavelength.
//	int currentSpw = 0;
//	double currentWavelength = minWavelength[ currentSpw ] - (range / 2.0);

	// calculate the wavelengths.
//	for ( int i = 0; i < _hstAPlanes; i++ )
//	{

		// increase the wavelength by the range between A-planes.
//		currentWavelength = currentWavelength + range;

		// have we gone outside the range of this spw ?
//		while (currentWavelength > maxWavelength[ currentSpw ])
//		{
//			currentWavelength = currentWavelength - maxWavelength[ currentSpw ] + minWavelength[ currentSpw + 1 ];
//			currentSpw = currentSpw + 1;
//		}

		// set the wavelength of this A-plane.
//		phstAPlaneWavelength[ i ] = currentWavelength;
				
//	}

//	printf( "                a-planes based upon wavelengths [" );

	// display the wavelengths.
//	for ( int i = 0; i < _hstAPlanes; i++ )
//	{
//		if (i > 0)
//			printf( ", " );
//		printf( "%5.4f", phstAPlaneWavelength[ i ] * 1000.0 );
//	}
//	printf( "] mm\n\n" );

	// update each spw and channel with the appropriate A-plane.
//	(*pWhichAPlane) = (int **) malloc( pNumSpws * sizeof( int * ) );
//	int ** whichAPlane = (*pWhichAPlane);
//	for ( int spw = 0; spw < pNumSpws; spw++ )
//	{

		// create array for this spw, and find the appropriate A-plane.
//		whichAPlane[ spw ] = (int *) malloc( phstNumChannels[ spw ] * sizeof( int ) );
//		for ( int channel = 0; channel < phstNumChannels[ spw ]; channel++ )
//		{

			// find the closest A-plane.
//			double bestError = 0.0;
//			for ( int aPlane = 0; aPlane < _hstAPlanes; aPlane++ )
//				if (abs( phstWavelength[ spw ][ channel ] - phstAPlaneWavelength[ aPlane ] ) < bestError || aPlane == 0)
//				{
//					whichAPlane[ spw ][ channel ] = aPlane;
//					bestError = abs( phstWavelength[ spw ][ channel ] - phstAPlaneWavelength[ aPlane ] );
//				}

//		}
//	}

	// free data.
//	if (minWavelength != NULL)
//		free( (void *) minWavelength );
//	if (maxWavelength != NULL)
//		free( (void *) maxWavelength );

//} // calculateAPlanes

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
			
		printf( "upper limits of w-planes: [" );
		for ( int i = 0; i < _hstWPlanes; i++ )
		{

			// set the mean and maximum w values for this plane.
			_hstWPlaneMax[ pMosaicIndex ][ i ] = ((maxW - minW) * (i + 1) / _hstWPlanes) + minW;
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

	}
	else if (_hstWPlanes == 1)
	{
		_hstWPlaneMean[ pMosaicIndex][ 0 ] = 0;
		_hstWPlaneMax[ pMosaicIndex][ 0 ] = maxW;
	}
	
} // calculateWPlanes

//
//	getASKAPBeamPosition()
//
//	CJS: 20/12/2019
//
//	Calculates the position of each ASKAP beam relative to the phase centre.
//

void getASKAPBeamPosition( double * pRA, double * pDEC, int pBeamIndex, double pCentreRA, double pCentreDEC )
{

	// get the ra and dec offset depending upon which beam this is.
	double raOffset = 0, decOffset = 0;
	switch (pBeamIndex)
	{
		case 0:	raOffset = 0.0432086; decOffset = -0.0340456; break; // raOffset = -0.0096172; decOffset = 0.0019379; break;
		case 1:	raOffset = 0.0274913; decOffset = -0.0340267; break; // raOffset = 0.0028024; decOffset = 0.0088222; break;
		case 2:	raOffset = 0.0117808; decOffset = -0.0340162; break; // raOffset = -0.0023984; decOffset = -0.0110852; break;
		case 3:	raOffset = -0.00392684; decOffset = -0.0340141; break; // raOffset = 0.0100212; decOffset = -0.0042009; break;
		case 4:	raOffset = -0.0196354; decOffset = -0.0340204; break; // raOffset = -0.0292557; decOffset = 0.0080766; break;
		case 5:	raOffset = -0.0353488; decOffset = -0.0340351; break; // raOffset = -0.0168361; decOffset = 0.0149609; break;
		case 6:	raOffset = 0.0353488; decOffset = -0.0204186; break; // raOffset = -0.0044165; decOffset = 0.0218452; break;
		case 7:	raOffset = 0.0196354; decOffset = -0.0204098; break; // raOffset = 0.0080031; decOffset = 0.0287295; break;
		case 8:	raOffset = 0.00392684; decOffset = -0.020406; break; // raOffset = 0.0152220; decOffset = 0.0157064; break;
		case 9:	raOffset = -0.0117808; decOffset = -0.0204073; break; // raOffset = 0.0224408; decOffset = 0.0026834; break;
		case 10:	raOffset = -0.0274913; decOffset = -0.0204136; break; // raOffset = 0.0296596; decOffset = -0.0103397; break;
		case 11:	raOffset = -0.0432086; decOffset = -0.0204249; break; // raOffset = 0.0172400; decOffset = -0.0172240; break;
		case 12:	raOffset = 0.0432086; decOffset = -0.00680788; break; // raOffset = 0.0048204; decOffset = -0.0241083; break;
		case 13:	raOffset = 0.0274913; decOffset = -0.0068041; break; // raOffset = -0.0075992; decOffset = -0.0309926; break;
		case 14:	raOffset = 0.0117808; decOffset = -0.006802; break; // raOffset = -0.0148180; decOffset = -0.0179695; break;
		case 15:	raOffset = -0.00392684; decOffset = -0.00680158; break; // raOffset = -0.0220368; decOffset = -0.0049464; break;
		case 16:	raOffset = -0.0196354; decOffset = -0.00680284; break; // raOffset = -0.0488941; decOffset = 0.0142154; break;
		case 17:	raOffset = -0.0353488; decOffset = -0.00680578; break; // raOffset = -0.0364745; decOffset = 0.0210997; break;
		case 18:	raOffset = 0.0353488; decOffset = 0.00680578; break; // raOffset = -0.0240549; decOffset = 0.0279840; break;
		case 19:	raOffset = 0.0196354; decOffset = 0.00680284; break; // raOffset = -0.0116353; decOffset = 0.0348683; break;
		case 20:	raOffset = 0.00392684; decOffset = 0.00680158; break; // raOffset = 0.0007843; decOffset = 0.0417526; break;
		case 21:	raOffset = -0.0117808; decOffset = 0.006802; break; // raOffset = 0.0132039; decOffset = 0.0486369; break;
		case 22:	raOffset = -0.0274913; decOffset = 0.0068041; break; // raOffset = 0.0204227; decOffset = 0.0356138; break;
		case 23:	raOffset = -0.0432086; decOffset = 0.00680788; break; // raOffset = 0.0276416; decOffset = 0.0225907; break;
		case 24:	raOffset = 0.0432086; decOffset = 0.0204249; break; // raOffset = 0.0348604; decOffset = 0.0095677; break;
		case 25:	raOffset = 0.0274913; decOffset = 0.0204136; break; // raOffset = 0.0420792; decOffset = -0.0034554; break;
		case 26:	raOffset = 0.0117808; decOffset = 0.0204073; break; // raOffset = 0.0492980; decOffset = -0.0164785; break;
		case 27:	raOffset = -0.00392684; decOffset = 0.020406; break; // raOffset = 0.0368784; decOffset = -0.0233628; break;
		case 28:	raOffset = -0.0196354; decOffset = 0.0204098; break; // raOffset = 0.0244588; decOffset = -0.0302471; break;
		case 29:	raOffset = -0.0353488; decOffset = 0.0204186; break; // raOffset = 0.0120392; decOffset = -0.0371314; break;
		case 30:	raOffset = 0.0353488; decOffset = 0.0340351; break; // raOffset = -0.0003804; decOffset = -0.0440157; break;
		case 31:	raOffset = 0.0196354; decOffset = 0.0340204; break; // raOffset = -0.0128000; decOffset = -0.0509000; break;
		case 32:	raOffset = 0.00392684; decOffset = 0.0340141; break; // raOffset = -0.0200188; decOffset = -0.0378769; break;
		case 33:	raOffset = -0.0117808; decOffset = 0.0340162; break; // raOffset = -0.0272376; decOffset = -0.0248538; break;
		case 34:	raOffset = -0.0274913; decOffset = 0.0340267; break; // raOffset = -0.0344564; decOffset = -0.0118307; break;
		case 35:	raOffset = -0.0432086; decOffset = 0.0340456; break; // raOffset = -0.0416753; decOffset = 0.0011923; break;
	}

	// 6:  - 0.001385355, - 0.000621774
	// 12: - 0.002170753, - 0.000119991

	*pRA = -(deg( raOffset ) / cos( rad( pCentreDEC ) )) + pCentreRA;
	*pDEC = deg( decOffset ) + pCentreDEC;

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
	
	cudaDeviceSynchronize();
	
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
	
	cudaDeviceSynchronize();
	
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

int findCutoffPixel( cufftComplex/*grid*/ * pdevKernel, double * pdevMaxValue, int pSize, double pCutoffFraction, findpixel pFindType )
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
	
	cudaDeviceSynchronize();
	
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

int findCutoffPixel( float/*grid*/ * pdevKernel, double * pdevMaxValue, int pSize, double pCutoffFraction, findpixel pFindType )
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
	
	cudaDeviceSynchronize();
	
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
//	createCUDAArrayForKernels()
//
//	CJS: 20/11/2018
//
//	Takes a complex array on the device and uploads the data into CUDA arrays for a complete kernel set.
//	the resulting CUDA arrays are held in _devKernelArrayRealPtr and _devKernelArrayImagPtr.
//

void createCUDAArrayForKernels( cufftComplex/*grid*/ * pdevKernel, int pKernelSize, int pNumKernels )
{

	cudaError_t err;
	
	// set up texture mapping. we can only create double-type texture maps, so
	// we put the real and imaginary components in separate texture maps.
	cudaExtent arraySize = make_cudaExtent( pKernelSize, pKernelSize, pNumKernels );
		
	// set some basic properties of the textures.
	_kernelTextureReal.addressMode[0] = cudaAddressModeWrap;
	_kernelTextureReal.addressMode[1] = cudaAddressModeWrap;
	_kernelTextureReal.filterMode = cudaFilterModePoint;
	_kernelTextureImag.addressMode[0] = cudaAddressModeWrap;
	_kernelTextureImag.addressMode[1] = cudaAddressModeWrap;
	_kernelTextureImag.filterMode = cudaFilterModePoint;
		
	// create arrays to store these textures.
	err = cudaMalloc3DArray( &_devKernelArrayRealPtr, &_channelDesc, arraySize );
	if (err != cudaSuccess)
		printf( "Error creating 3D array (%s)\n", cudaGetErrorString( err ) );
	err = cudaMalloc3DArray( &_devKernelArrayImagPtr, &_channelDesc, arraySize );
	if (err != cudaSuccess)
		printf( "Error creating 3D array (%s)\n", cudaGetErrorString( err ) );
		
	// construct a float/*grid*/ pointer to the kernel.
	float/*grid*/ * devKernelPtr = (float/*grid*/ *) pdevKernel;
	
	// set up texture mapping. we can only create double-type texture maps, so
	// we put the real and imaginary components in separate texture maps.
	// create the structures for copying data to 3D arrays.
	cudaMemcpy3DParms copyParams = {0};
	copyParams.srcPtr = make_cudaPitchedPtr( (void *) devKernelPtr, pKernelSize * sizeof( float/*grid*/ ), pKernelSize, pKernelSize );
	copyParams.dstArray = _devKernelArrayRealPtr;
	copyParams.extent = arraySize;
	copyParams.kind = cudaMemcpyDeviceToDevice;

	// create texture maps by copying real data from the kernel memory.
	err = cudaMemcpy3D( &copyParams );
	if (err != cudaSuccess)
		printf( "Error copying real data to cuda array (%s)\n", cudaGetErrorString( err ) );
		
	// create the structures for copying data to 3D arrays.
	copyParams.srcPtr = make_cudaPitchedPtr( (void *) &devKernelPtr[ pKernelSize * pKernelSize * pNumKernels ],
							pKernelSize * sizeof( float/*grid*/ ), pKernelSize, pKernelSize );
	copyParams.dstArray = _devKernelArrayImagPtr;

	// create texture maps by copying imaginary data from the kernel memory.
	err = cudaMemcpy3D( &copyParams );
	if (err != cudaSuccess)
		printf( "Error copying imaginary data to cuda array (%s)\n", cudaGetErrorString( err ) );

} // createCUDAArrayForKernels

void createCUDAArrayForKernels( float/*grid*/ * pdevKernel, int pKernelSize, int pNumKernels )
{

	cudaError_t err;
	
	// set up texture mapping. we can only create double-type texture maps, so
	// we put the real and imaginary components in separate texture maps.
	cudaExtent arraySize = make_cudaExtent( pKernelSize, pKernelSize, pNumKernels );
		
	// set some basic properties of the textures.
	_kernelTextureReal.addressMode[0] = cudaAddressModeWrap;
	_kernelTextureReal.addressMode[1] = cudaAddressModeWrap;
	_kernelTextureReal.filterMode = cudaFilterModePoint;
		
	// create arrays to store these textures.
	err = cudaMalloc3DArray( &_devKernelArrayRealPtr, &_channelDesc, arraySize );
	if (err != cudaSuccess)
		printf( "Error creating 3D array (%s)\n", cudaGetErrorString( err ) );
	
	// set up texture mapping. we can only create double-type texture maps, so
	// we put the real and imaginary components in separate texture maps.
	// create the structures for copying data to 3D arrays.
	cudaMemcpy3DParms copyParams = {0};
	copyParams.srcPtr = make_cudaPitchedPtr( (void *) pdevKernel, pKernelSize * sizeof( float/*grid*/ ), pKernelSize, pKernelSize );
	copyParams.dstArray = _devKernelArrayRealPtr;
	copyParams.extent = arraySize;
	copyParams.kind = cudaMemcpyDeviceToDevice;

	// create texture maps by copying real data from the kernel memory.
	err = cudaMemcpy3D( &copyParams );
	if (err != cudaSuccess)
		printf( "Error copying real data to cuda array (%s)\n", cudaGetErrorString( err ) );

} // createCUDAArrayForKernels

//
//	rearrangeKernel()
//
//	CJS: 14/03/2016
//
//	Rearrange the data in a kernel so that the real numbers are at the start and the imaginary numbers at the end.
//

void rearrangeKernel( cufftComplex/*grid*/ * pKernel, long int pSize )
{
	
	bool ok = true;
	cudaError_t err;

	// create temporary array for kernels.
	cufftComplex/*grid*/ * devTmp = NULL;
	ok = reserveGPUMemory( (void **) &devTmp, pSize * sizeof( cufftComplex/*grid*/ ), "creating temporary kernel memory on the device" );
	if (ok == true)
	{

		// copy kernel to temporary memory.
		err = cudaMemcpy( devTmp, pKernel, pSize * sizeof( cufftComplex/*grid*/ ), cudaMemcpyDeviceToDevice );
		if (err != cudaSuccess)
		{
			printf( "error copying kernel to temporary memory (%s)\n", cudaGetErrorString( err ) );
			ok = false;
		}
	
	}
	
	if (ok == true)
	{
	
		// determine suitable grid and block size.
		int threads = pSize;
		int blocks = 1;
		setThreadBlockSize1D( &threads, &blocks );

		// copy kernel from temporary memory back into kernel memory, but with the real numbers at the start and the imaginary numbers at the end.
		devRearrangeKernel<<< blocks, threads >>>(	/* pTarget = */ (float/*grid*/ *) pKernel,
								/* pSource = */ (float/*grid*/ *) devTmp,
								/* pElements = */ pSize );
		
	}
	
	// cleanup memory.
	cudaFree( (void *) devTmp );

} // rearrangeKernel

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
//	pNumFields - we may have to generate this many kernels if we're beam mosaicing. we need to take this figure into account when calculating the maximum kernel size.
//

bool generateKernel( int pW, int pA, bool pWProjection, bool pAProjection, int * phstKernelSize, int pOversample,
			float * phstPrimaryBeamMosaicing, float * phstPrimaryBeamAProjection, int pFieldID,
			int pMosaicIndex, griddegrid pGridDegrid, cufftComplex/*grid*/ ** pdevKernelPtr, int pNumFields )
{

	// we only use a maximum fraction of the available memory to create a workspace.
	const double MAX_MEM_TO_USE = 0.4;
	const int MAX_WORKSPACE_SIZE = 2048;
	
	bool ok = true;
	cudaError_t err;

	// for beam mosaicing we need to do some primary beam correction using the average (mosaic) primary beam. we only have to do this for degridding if we're not
	// using A-projection.
	bool doBeamCorrection = (phstPrimaryBeamMosaicing != NULL && (pGridDegrid == GRID || pAProjection == false));

	int numberOfWorkspacesRequired = 1;
	if (doBeamCorrection == true)
		numberOfWorkspacesRequired++;
	if (pWProjection == true)
		numberOfWorkspacesRequired++;
	if (pAProjection == true)
		numberOfWorkspacesRequired++;

	// find how much GPU memory is available.
	size_t freeMem = 0, totalMem = 0;
	err = cudaMemGetInfo( &freeMem, &totalMem );

	// calculate a pixel size from the free memory, and reduce the free memory to the next lowest power of 2.
	int maximumWorkspaceSize = sqrt( freeMem * MAX_MEM_TO_USE / (numberOfWorkspacesRequired * sizeof( cufftComplex/*grid*/ )) );
	int oversampledWorkspaceSize = 1;
	while (oversampledWorkspaceSize <= maximumWorkspaceSize)
		oversampledWorkspaceSize = oversampledWorkspaceSize * 2;
	oversampledWorkspaceSize = oversampledWorkspaceSize / 2;

	// set the workspace size and the maximum kernel size.
	int workspaceSize = oversampledWorkspaceSize / pOversample;
	if (workspaceSize > _hstUvPixels)
		workspaceSize = _hstUvPixels;
	if (workspaceSize > MAX_WORKSPACE_SIZE)
		workspaceSize = MAX_WORKSPACE_SIZE;
	oversampledWorkspaceSize = workspaceSize * pOversample;

	// display warning if the workspace size is too low.
	if (workspaceSize <= 125)
		printf( "\n! WARNING: The workspace size is set to %i x %i, which is too low. Reducing the oversampling parameter might help.\n\n", workspaceSize,
				workspaceSize );

	// if we are using beam mosaicing then we need to correct the image for the primary beam, and weight the image for mosaicing.
	cufftComplex/*grid*/ * devBeamCorrection = NULL;
	if (doBeamCorrection == true)
	{

		// create space for the primary beam on the device.
		float * devPrimaryBeam = NULL;
		reserveGPUMemory( (void **) &devPrimaryBeam, _hstBeamSize * _hstBeamSize * sizeof( float ),
					"declaring device memory for primary beam" );

		// copy primary beam to the device.
		moveHostToDevice( (void *) devPrimaryBeam, (void *) phstPrimaryBeamMosaicing, _hstBeamSize * _hstBeamSize * sizeof( float ),
					"copying primary beam to the device" );

		// set the primary beam during gridding and degridding. we tell this subroutine if we're using A-projection or not because in the absence of A-projection
		// we will need to correct for the primary beam function using same average beam that we use to weight the mosaic.
		setThreadBlockSize2D( _hstBeamSize, _hstBeamSize );
		devSetPrimaryBeamForGriddingAndDegridding<<< _gridSize2D, _blockSize2D >>>(	/* pImage = */ devPrimaryBeam,
												/* pSize = */ _hstBeamSize,
												/* pGridDegrid = */ pGridDegrid,
												/* pAProjection = */ pAProjection );

		// create the kernel and clear it.
		reserveGPUMemory( (void **) &devBeamCorrection, workspaceSize * workspaceSize * sizeof( cufftComplex/*grid*/ ),
					"declaring device memory for the beam-correction kernel" );
		zeroGPUMemory( (void *) devBeamCorrection, workspaceSize * workspaceSize * sizeof( cufftComplex/*grid*/ ), "clearing beam-correction kernel on the device" );

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

//if (pA == 0)
//{

//cudaDeviceSynchronize();
//float * tmpKernel = (float *) malloc( workspaceSize * workspaceSize * sizeof( cufftComplex/*grid*/ ) );
//cudaMemcpy( tmpKernel, devBeamCorrection, workspaceSize * workspaceSize * sizeof( cufftComplex/*grid*/ ), cudaMemcpyDeviceToHost );
//for ( long int i = 0; i < (long int) workspaceSize * (long int) workspaceSize; i++ )
//	tmpKernel[ i ] = tmpKernel[ i * 2 ];
//char kernelFilename[100];
//sprintf( kernelFilename, "beam-correction-%i-%i-%i.image", pFieldID, pA, (pGridDegrid == GRID ? 0 : 1) );
//_hstCasacoreInterface.WriteCasaImage( kernelFilename, workspaceSize, workspaceSize, _hstOutputRA,
//					_hstOutputDEC, _hstCellSize, tmpKernel, CONST_C / _hstAverageWavelength[ pMosaicIndex ], NULL );
//free( tmpKernel );

//}

	}

	// are we using W-projection ?
	cufftComplex/*grid*/ * devWKernel = NULL;
	if (pWProjection == true)
	{

		// create w-kernel.
		reserveGPUMemory( (void **) &devWKernel, workspaceSize * workspaceSize * sizeof( cufftComplex/*grid*/ ), "declaring device memory for w-kernel" );

		// convert the cell size from arcseconds to radians.
		double cellSizeRadians = sin( (_hstCellSize / 3600.0) * (PI / 180.0) );

		// generate the W-kernel on the GPU.
		setThreadBlockSize2D( workspaceSize, workspaceSize );
		devGenerateWKernel<<< _gridSize2D, _blockSize2D >>>(	/* pWKernel = */ devWKernel,
									/* pW = */ _hstWPlaneMean[ pMosaicIndex ][ pW ],
									/* pWorkspaceSize = */ workspaceSize,
									/* pCellSizeRadians = */ cellSizeRadians,
									/* pGridDegrid = */ pGridDegrid,
									/* pSize = */ _hstUvPixels );

	}

	// are we using A-projection ?
	cufftComplex/*grid*/ * devAKernel = NULL;
	if (pAProjection == true)
	{

		// reserve some memory for the A kernel and primary beam.
		float * devPrimaryBeam = NULL;
		reserveGPUMemory( (void **) &devAKernel, workspaceSize * workspaceSize * sizeof( cufftComplex/*grid*/ ), "declaring device memory for a-kernel" );
		reserveGPUMemory( (void **) &devPrimaryBeam, _hstBeamSize * _hstBeamSize * sizeof( float ), "declaring device memory for the primary beam" );

		// upload the primary beam to the device.
		moveHostToDevice( (void *) devPrimaryBeam, (void *) phstPrimaryBeamAProjection, _hstBeamSize * _hstBeamSize * sizeof( float ),
						"uploading primary beam to the device" );

		cudaDeviceSynchronize();

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

		cudaDeviceSynchronize();

		// free the primary beam.
		if (devPrimaryBeam != NULL)
			cudaFree( (void *) devPrimaryBeam );

//if (pA == 0)
//{
//
//cudaDeviceSynchronize();
//float * tmpKernel = (float *) malloc( workspaceSize * workspaceSize * sizeof( cufftComplex/*grid*/ ) );
//cudaMemcpy( tmpKernel, devAKernel, workspaceSize * workspaceSize * sizeof( cufftComplex/*grid*/ ), cudaMemcpyDeviceToHost );
//for ( long int i = 0; i < (long int) workspaceSize * (long int) workspaceSize; i++ )
//	tmpKernel[ i ] = tmpKernel[ i * 2 ];
//char kernelFilename[100];
//sprintf( kernelFilename, "a-kernel-%i-%i-%i.image", pFieldID, pA, (pGridDegrid == GRID ? 0 : 1) );
//_hstCasacoreInterface.WriteCasaImage( kernelFilename, workspaceSize, workspaceSize, _hstOutputRA,
//					_hstOutputDEC, _hstCellSize, tmpKernel, CONST_C / _hstAverageWavelength[ pMosaicIndex ], NULL );
//free( tmpKernel );
//
//}
	}
	
	// reserve some memory for the AA kernel and clear it.
	cufftComplex /*grid*/ * devAAKernel = NULL;
	reserveGPUMemory( (void **) &devAAKernel, workspaceSize * workspaceSize * sizeof( cufftComplex/*grid*/ ), "declaring device memory for the aa-kernel" );
	zeroGPUMemory( (void *) devAAKernel, workspaceSize * workspaceSize * sizeof( cufftComplex/*grid*/ ), "zeroing aa-kernel on the device" );

	err = cudaGetLastError();
	if (err != cudaSuccess)
		printf( "Unknown CUDA error in generateKernel() (%s)\n", cudaGetErrorString( err ) );

	// generate the AA-kernel on the GPU.
	setThreadBlockSize2D( _hstAAKernelSize, _hstAAKernelSize );
	devGenerateAAKernel<<< _gridSize2D, _blockSize2D >>>(	/* pAAKernel = */ devAAKernel,
								/* pKernelSize = */ _hstAAKernelSize,
								/* pWorkspaceSize = */ workspaceSize );
	err = cudaGetLastError();
	if (err != cudaSuccess)
		printf( "error generating AA kernel (%s)\n", cudaGetErrorString( err ) );
//{

//cudaDeviceSynchronize();
//float * tmpKernel = (float *) malloc( workspaceSize * workspaceSize * sizeof( cufftComplex/*grid*/ ) );
//cudaMemcpy( tmpKernel, devAAKernel, workspaceSize * workspaceSize * sizeof( cufftComplex/*grid*/ ), cudaMemcpyDeviceToHost );
//for ( long int i = 0; i < (long int) workspaceSize * (long int) workspaceSize; i++ )
//	tmpKernel[ i ] = tmpKernel[ i * 2 ];
//char kernelFilename[100];
//sprintf( kernelFilename, "aa-kernel-%i.image", pFieldID );
//_hstCasacoreInterface.WriteCasaImage( kernelFilename, workspaceSize, workspaceSize, _hstOutputRA,
//					_hstOutputDEC, _hstCellSize, tmpKernel, CONST_C / _hstAverageWavelength[ pMosaicIndex ], NULL );
//free( tmpKernel );

//}

	// we will need the AA-kernel to be in the image domain. FFT the AA-kernel.
	performFFT(	/* pdevGrid = */ &devAAKernel,
			/* pSize = */ workspaceSize,
			/* pFFTDirection = */ INVERSE,
			/* pFFTPlan = */ -1,
			/* pFFType = */ C2C );

	// create a kernel workspace to store the image-plane kernel.
	cufftComplex/*grid*/ * devImagePlaneKernel = NULL;
	reserveGPUMemory( (void **) &devImagePlaneKernel, workspaceSize * workspaceSize * sizeof( cufftComplex/*grid*/ ), "declaring device memory for image-plane kernel" );

//{

//cudaDeviceSynchronize();
//float * tmpKernel = (float *) malloc( workspaceSize * workspaceSize * sizeof( cufftComplex/*grid*/ ) );
//cudaMemcpy( tmpKernel, devBeamCorrection, workspaceSize * workspaceSize * sizeof( cufftComplex/*grid*/ ), cudaMemcpyDeviceToHost );
//for ( long int i = 0; i < (long int) workspaceSize * (long int) workspaceSize; i++ )
//	tmpKernel[ i ] = tmpKernel[ i * 2 ];
//char kernelFilename[100];
//sprintf( kernelFilename, "beam-correction-%i.image", pFieldID );
//_hstCasacoreInterface.WriteCasaImage( kernelFilename, workspaceSize, workspaceSize, _hstOutputRA,
//					_hstOutputDEC, _hstCellSize, tmpKernel, CONST_C / _hstAverageWavelength[ pMosaicIndex ], NULL );
//free( tmpKernel );

//}

	// we now work with the whole workspace.
	setThreadBlockSize2D( workspaceSize, workspaceSize );

	// we start off with the anti-aliasing kernel.
	cudaMemcpy( (void *) devImagePlaneKernel, (void *) devAAKernel, workspaceSize * workspaceSize * sizeof( cufftComplex/*grid*/ ), cudaMemcpyDeviceToDevice );

	// are we using W-projection ? Convolve with the kernel.
	if (pWProjection == true)
		devMultiplyImages<<< _gridSize2D, _blockSize2D >>>( devImagePlaneKernel, devWKernel, workspaceSize );

	// are we using A-projection ? Convolve with the kernel.
	if (pAProjection == true)
		devMultiplyImages<<< _gridSize2D, _blockSize2D >>>( devImagePlaneKernel, devAKernel, workspaceSize );

	// are we using beam correction ? Convolve with the kernel.
	if (doBeamCorrection == true)
		devMultiplyImages<<< _gridSize2D, _blockSize2D >>>( devImagePlaneKernel, devBeamCorrection, workspaceSize );

//{

//cudaDeviceSynchronize();
//float * tmpKernel = (float *) malloc( workspaceSize * workspaceSize * sizeof( cufftComplex/*grid*/ ) );
//cudaMemcpy( tmpKernel, devImagePlaneKernel, workspaceSize * workspaceSize * sizeof( cufftComplex/*grid*/ ), cudaMemcpyDeviceToHost );
//for ( long int i = 0; i < (long int) workspaceSize * (long int) workspaceSize; i++ )
//	tmpKernel[ i ] = tmpKernel[ i * 2 ];
//char kernelFilename[100];
//sprintf( kernelFilename, "combined-kernel-%i-%i-%i.image", pFieldID, pA, (pGridDegrid == GRID ? 0 : 1) );
//sprintf( kernelFilename, "combined-kernel-w%i-g%i.image", pW, (pGridDegrid == GRID ? 0 : 1) );
//sprintf( kernelFilename, "combined-kernel-%i.image", (pGridDegrid == GRID ? 0 : 1) );
//_hstCasacoreInterface.WriteCasaImage( kernelFilename, workspaceSize, workspaceSize, _hstOutputRA,
//					_hstOutputDEC, _hstCellSize, tmpKernel, CONST_C / _hstAverageWavelength[ pMosaicIndex ], NULL );
//free( tmpKernel );

//}

	// reserve some memory for the combined kernel, and clear it.
	cufftComplex/*grid*/ * devCombinedKernel = NULL;
	reserveGPUMemory( (void **) &devCombinedKernel, oversampledWorkspaceSize * oversampledWorkspaceSize * sizeof( cufftComplex/*grid*/ ),
				"declaring device memory for the combined kernel" );
	zeroGPUMemory( (void *) devCombinedKernel, oversampledWorkspaceSize * oversampledWorkspaceSize * sizeof( cufftComplex/*grid*/ ),
				"zeroing device memory for the combined kernel" );

	// copy the image-domain kernel into the centre of the combined kernel. i.e. we are padding our kernel by the oversampling factor.
	devCopyImage<<< _gridSize2D, _blockSize2D >>>(	/* pNewImage = */ devCombinedKernel,
							/* pOldImage = */ devImagePlaneKernel,
							/* pNewSize = */ oversampledWorkspaceSize,
							/* pOldSize = */ workspaceSize,
							/* pScale = */ 1.0,
							/* pThreadOffset = */ (oversampledWorkspaceSize - workspaceSize) / 2 );

	// clear the image-plane kernel.
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
	supportSize = findCutoffPixel(	/* pdevKernel = */ devCombinedKernel,
					/* pdevMaxValue = */ devMaxValue,
					/* pSize = */ oversampledWorkspaceSize,
					/* pCutoffFraction = */ 0.01,
//					/* pCutoffFraction = */ 0.0004,
					/* pFindType = */ FURTHEST );
	
	// free memory.
	if (devMaxValue != NULL)
		cudaFree( (void *) devMaxValue );

	// divide the support size by the oversampling factor, and round up.
	supportSize = (int) ceil( (double) supportSize / (double) pOversample );

//printf( "suggested support size is %i. ", supportSize );

	// ensure the support is at least 3.
	if (supportSize < 3)
		supportSize = 3;

//printf( "instead we will use %i\n", supportSize );

	// calculate the maximum support size, and make sure it's not larger than 200 for performance reasons.
	err = cudaMemGetInfo( &freeMem, &totalMem );
	freeMem = (int) ((double) freeMem * MAX_MEM_TO_USE / (double) (sizeof( cufftComplex/*grid*/ ) * pNumFields * pOversample * pOversample));
	int maximumSupportSize = (int) (sqrt( freeMem ) / 2.0);
	if (maximumSupportSize > 300)
		maximumSupportSize = 300;

	// if the required kernel size is larger than the maximum kernel size then display a warning.
	if (maximumSupportSize < supportSize)
		printf( "\n! WARNING: The maximum kernel support is set to %i x %i, which is too low (require %i x %i). Reduce the oversampling parameter !\n\n",
									maximumSupportSize, maximumSupportSize, supportSize, supportSize );

	// restrict support based upon the workspace size.
	if (supportSize > maximumSupportSize)
		supportSize = maximumSupportSize;

	// calculate kernel size.
	*phstKernelSize = (supportSize * 2) + 1;

//{

//cudaDeviceSynchronize();
//float * tmpKernel = (float *) malloc( oversampledWorkspaceSize * oversampledWorkspaceSize * sizeof( cufftComplex/*grid*/ ) );
//cudaMemcpy( tmpKernel, devCombinedKernel, oversampledWorkspaceSize * oversampledWorkspaceSize * sizeof( cufftComplex/*grid*/ ), cudaMemcpyDeviceToHost );
//for ( long int i = 0; i < (long int) oversampledWorkspaceSize * (long int) oversampledWorkspaceSize; i++ )
//	tmpKernel[ i ] = tmpKernel[ i * 2 ];
//char kernelFilename[100];
//if (pMosaicIndex == -1)
//	sprintf( kernelFilename, "kernelW2-aa-%i-%i.image", pW, (pGridDegrid == GRID ? 0 : 1) );
//else
//	sprintf( kernelFilename, "kernelW2-%i-%i-%i.image", pFieldID, pW, (pGridDegrid == GRID ? 0 : 1) );
//_hstCasacoreInterface.WriteCasaImage( kernelFilename, oversampledWorkspaceSize, oversampledWorkspaceSize, _hstOutputRA,
//					_hstOutputDEC, _hstCellSize, tmpKernel, CONST_C / _hstAverageWavelength[ pMosaicIndex ], NULL );
//free( tmpKernel );

//}

//if (pMosaicIndex > -1)
//{
//	supportSize = 40;
//	*phstKernelSize = (supportSize * 2) + 1;
//}

	// kernel data on the device.
	reserveGPUMemory( (void **) pdevKernelPtr, *phstKernelSize * *phstKernelSize * pOversample * pOversample * sizeof( cufftComplex/*grid*/ ),
				"declaring device memory for kernel" );

	// get a shortcut to the kernel.
	cufftComplex/*grid*/ * devKernel = *pdevKernelPtr;
		
	// copy constants to constant memory.
	err = cudaMemcpyToSymbol( _devSupport, &supportSize, sizeof( supportSize ) );
	if (err != cudaSuccess)
		printf( "error copying support size to device (%s)\n", cudaGetErrorString( err ) );
	err = cudaMemcpyToSymbol( _devKernelSize, phstKernelSize, sizeof( *phstKernelSize ) );
	if (err != cudaSuccess)
		printf( "error copying kernel size to device (%s)\n", cudaGetErrorString( err ) );
	err = cudaMemcpyToSymbol( _devAASupport, &_hstAASupport, sizeof( _hstAASupport ) );
	if (err != cudaSuccess)
		printf( "error copying AA support size to device (%s)\n", cudaGetErrorString( err ) );
	err = cudaMemcpyToSymbol( _devAAKernelSize, &_hstAAKernelSize, sizeof( _hstAAKernelSize ) );
	if (err != cudaSuccess)
		printf( "error copying AA kernel size to device (%s)\n", cudaGetErrorString( err ) );

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
	if (pFieldID >= 0 && pGridDegrid == GRID)
		devNormalise<<< blocks, threads>>>(	devKernel,
							(double) _hstGriddedVisibilitiesPerField[ pFieldID ] / (double) _griddedVisibilitiesForBeamMosaic,
							*phstKernelSize * *phstKernelSize * pOversample * pOversample );
	
	cudaDeviceSynchronize();

//140819
//	if (pFieldID == 0)
//	{
//		float * tmpKernel = (float *) malloc( *phstKernelSize * *phstKernelSize * sizeof( cufftComplex/*grid*/ ) );
//		for ( int oversampleI = 0; oversampleI < pOversample; oversampleI++ )
//			for ( int oversampleJ = 0; oversampleJ < pOversample; oversampleJ++ )
//			{
//				
//				// get the index of the oversampled kernels. no need to add the index of the w-kernel because
//				// we're putting them in separate arrays.
//				long int kernelIdx = ((long int)oversampleI * (long int)*phstKernelSize * (long int)*phstKernelSize) +
//							((long int)oversampleJ * (long int)*phstKernelSize * (long int)*phstKernelSize * (long int)pOversample);
//
//				cudaMemcpy( tmpKernel, &devKernel[ kernelIdx ], *phstKernelSize * *phstKernelSize * sizeof( cufftComplex/*grid*/ ), cudaMemcpyDeviceToHost );
//				for ( long int i = 0; i < *phstKernelSize * *phstKernelSize; i++ )
//					tmpKernel[ i ] = tmpKernel[ i * 2 ];
//				char kernelFilename[100];
//				sprintf( kernelFilename, "kernel-%li-%i-%i.image", kernelIdx, pW, (pGridDegrid == GRID ? 0 : 1) );
//				_hstCasacoreInterface.WriteCasaImage( kernelFilename, *phstKernelSize, *phstKernelSize, _hstOutputRA,
//						_hstOutputDEC, _hstCellSize, tmpKernel, CONST_C / _hstAverageWavelength[ pMosaicIndex ], NULL );
//
//			}
//		free( tmpKernel );
//	}
	
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
 
	// Open the parameter file and get all lines.
	FILE *fr = fopen( pParameterFile, "rt" );
	while (fgets( line, 1024, fr ) != NULL)
	{

		params[0] = '\0';
		sscanf( line, "%s %s", par, params );
		if (strcmp( par, MEASUREMENT_SET ) == 0)
			strcpy( _hstMeasurementSetPath, params );
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
		else if (strcmp( par, FIELD_ID ) == 0)
			strcpy( _hstFieldID, params );
		else if (strcmp( par, CACHE_LOCATION ) == 0)
			strcpy( _hstCacheLocation, params );
		else if (strcmp( par, FILE_ID ) == 0)
		{
			
			// the file ids are supplied as a comma-separated list of ints. any spaces on either side of the commas
			// are ignored.
			char * fileID;
			_hstMeasurementSets = 0;

			// count the number of file ids. strtok() expects a pointer to the string initially, and on subsequent call
			// expects NULL. we use a duplicate variable because strtok() changes strings, so can only be used once.
			char tmp[ 512 ];
			strcpy( tmp, params );
			while ((fileID = strtok( (_hstMeasurementSets > 0 ? NULL : tmp), " ," )) != NULL)
				_hstMeasurementSets = _hstMeasurementSets + 1;
			
			// only proceed if we found at least one file ID.
			if (_hstMeasurementSets > 0)
			{
				
				// clear any existing file IDs.
				if (_hstMosaicID != NULL)
					free( (void *) _hstMosaicID );

				// reserve memory for these file IDs.
				_hstMosaicID = (int *) malloc( _hstMeasurementSets * sizeof( int ) );
			
				// read file IDs.
				int i = 0;
				while ((fileID = strtok( (i > 0 ? NULL : params), " ," )) != NULL)
					_hstMosaicID[ i++ ] = atoi( fileID );
			
			}

		}
		else if (strcmp( par, SPW ) == 0)
			strcpy( _hstSpwRestriction, params );
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
		else if (strcmp( par, DATA_FIELD ) == 0)
			strcpy( _hstDataField, params );
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
		}

		// debugging options.
		else if (strcmp( line, SAVE_MOSAIC_DIRTY_IMAGE ) == 0)
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

void gridVisibilities(	cufftComplex/*grid*/ * pdevGrid,	// data area (device) holding the grid
			cufftComplex/*grid*/ * phstGrid,		// data area (host) holding the larger grid if we're gridding in batches.
			cufftComplex/*vis*/ * pdevVisibility,			// data area (device) holding the visibilities
			int pOversample,				// }
			int * phstKernelSize,				// }- kernel and gridding parameters
			int * phstSupportSize,				// }
			int * pdevKernelIndex,				// an array of kernel indexes assigned to each visibility
			bool pWProjection,				// true if w-projection is being used
			bool pAProjection,				// true if a-projection is being used
			int pWPlanes,					// the number of w-planes to use
			int pAPlanes,					// the number of a-planes to use
			VectorI * pdevGridPositions,			// a list of integer grid positions for each visibility
			float * pdevWeight,				// a list of weights for each visibility
			int * pVisibilitiesInKernelSet,		// an array holding the number of visibilities in each w-plane, for this batch of visibilities
			griddegrid pGridDegrid,				// GRID or DEGRID
			float * phstPrimaryBeamMosaicing,		// the primary beam for mosaicing
			float * phstPrimaryBeamAProjection,		// the primary beam for A-projection
			int pNumFields,				// the number of fields we have in our data.
			int pMosaicIndex,				// the index of the mosaic image currently being processed
			int pSize )					// the size of the image
{
	
	cudaError_t err;
		
	int firstVisibility = 0;

	err = cudaGetLastError();
	if (err != cudaSuccess)
		printf( "unknown CUDA error in gridVisibilities() [i] (%s)\n", cudaGetErrorString( err ) );

	// find the range of kernels sets in this data.
	int minKernelSet = (pWPlanes * pAPlanes);
	int maxKernelSet = -1;
	for ( int i = 0; i < (pWPlanes * pAPlanes); i++ )
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

	// are we gridding in batches ? if no host grid has been supplied then we HAVE to grid on the GPU.
	int numberOfBatches = _hstImageBatches, gpuRows = _hstGPURows;
	if (phstGrid == NULL)
	{
		numberOfBatches = 1;
		gpuRows = pSize;
	}

	if (numKernelSetsInData > 0)
	{

		// grid the visibilities one w-plane at a time.
		for ( int kernelSet = minKernelSet; kernelSet <= maxKernelSet; kernelSet++ )
		{

			// determine how many visibilities are in these kernel sets.
			int numVisibilities = pVisibilitiesInKernelSet[ kernelSet ];

			// calculate W- and A-plane.
			int wPlane = kernelSet / pAPlanes;
			int aPlane = kernelSet % pAPlanes;

			// initialise the 3D arrays to null.
			_devKernelArrayRealPtr = NULL;
			_devKernelArrayImagPtr = NULL;

			// generate the texture maps for these kernel sets.
			if (numVisibilities > 0)
			{

				// how many fields are there? If pNumFields is -1 then we only have one field, otherwise we are using beam mosaicing and may have lots.
				int numFields = (pNumFields == -1 ? 1 : pNumFields);

				// create workspace in which to build the kernel.
				cufftComplex/*grid*/ * devKernel = NULL;

				// create an array of kernel pointers for each field, and an array to store the sizes.
				cufftComplex/*grid*/ ** devFieldKernelPtr = (cufftComplex/*grid*/ **) malloc( numFields * sizeof( cufftComplex/*grid*/ * ) );
				int * hstFieldKernelSize = (int *) malloc( numFields * sizeof( int ) );

				cufftComplex/*grid*/ * devWorkspace = NULL;
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
							/* pNumFields = */ numFields );

					// create space for the kernel.
					reserveGPUMemory( (void **) &devFieldKernelPtr[ field ], hstFieldKernelSize[ field ] * hstFieldKernelSize[ field ] *
								pOversample * pOversample * sizeof( cufftComplex/*grid*/ ), "creating device memory for field kernel" );

					// copy the kernels for this field into the kernel area on the device.
					cudaMemcpy(	(void *) devFieldKernelPtr[ field ],
							(void *) devWorkspace,
							hstFieldKernelSize[ field ] * hstFieldKernelSize[ field ] * pOversample * pOversample *
									sizeof( cufftComplex/*grid*/ ),
							cudaMemcpyDeviceToDevice );

					// get rid of the workspace.
					cudaFree( (void *) devWorkspace );

				}

				cudaDeviceSynchronize();

				// set kernel size to the maximum of the field kernel sizes.
				for ( int field = 0; field < numFields; field++ )
					if (hstFieldKernelSize[ field ] > phstKernelSize[ kernelSet ] || field == 0)
						phstKernelSize[ kernelSet ] = hstFieldKernelSize[ field ];
				phstSupportSize[ kernelSet ] = (phstKernelSize[ kernelSet ] - 1) / 2;

				// create space for the kernels on the device, and clear this memory.
				reserveGPUMemory( (void **) &devKernel, phstKernelSize[ kernelSet ] * phstKernelSize[ kernelSet ] * kernelsPerSet *
							sizeof( cufftComplex/*grid*/ ), "creating device memory for the kernels" );
				zeroGPUMemory( (void *) devKernel, phstKernelSize[ kernelSet ] * phstKernelSize[ kernelSet ] * kernelsPerSet *
							sizeof( cufftComplex/*grid*/ ), "zeroing device memory for the kernels" );

				// copy the kernels for all fields into the kernel area on the device.
				for ( int field = 0; field < numFields; field++ )
				{
					setThreadBlockSize2D( hstFieldKernelSize[ field ], hstFieldKernelSize[ field ] );
					for ( int i = 0; i < pOversample * pOversample; i++ )					
						devCopyImage<<< _gridSize2D, _blockSize2D >>>(	/* pNewImage = */ &devKernel[	((field * pOversample * pOversample) + i) *
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

				// rearrange the kernels so that the real numbers are at the start and the imaginary numbers at the end.
				rearrangeKernel(	/* pKernel = */ devKernel,
							/* pSize = */ phstKernelSize[ kernelSet ] * phstKernelSize[ kernelSet ] * kernelsPerSet );

				// upload these kernels to CUDA arrays, where they will later be bound to texture maps.
				createCUDAArrayForKernels(	/* pdevKernel = */ devKernel,
								/* pKernelSize = */ phstKernelSize[ kernelSet ],
								/* pNumKernels = */ kernelsPerSet );

				// don't need device kernels any more as they are in the texture maps now.
				if (devKernel != NULL)
					cudaFree( (void *) devKernel );
			
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
				printf( "\n" );
	
				// copy constants to constant memory.
				err = cudaMemcpyToSymbol( _devSupport, &phstSupportSize[ kernelSet ], sizeof( phstSupportSize[ kernelSet ] ) );
				if (err != cudaSuccess)
					printf( "error copying support size to device (%s)\n", cudaGetErrorString( err ) );
				err = cudaMemcpyToSymbol( _devKernelSize, &phstKernelSize[ kernelSet ], sizeof( phstKernelSize[ kernelSet ] ) );
				if (err != cudaSuccess)
					printf( "error copying kernel size to device (%s)\n", cudaGetErrorString( err ) );

				// bind real array to texture map.
				err = cudaBindTextureToArray( _kernelTextureReal, _devKernelArrayRealPtr, _channelDesc );
				if (err != cudaSuccess)
					printf( "Error binding texture to real array (%s)\n", cudaGetErrorString( err ) );

				// bind imaginary array to texture map.
				err = cudaBindTextureToArray( _kernelTextureImag, _devKernelArrayImagPtr, _channelDesc );
				if (err != cudaSuccess)
					printf( "Error binding texture to imaginary array (%s)\n", cudaGetErrorString( err ) );

				// ensure we don't report old errors.
				err = cudaGetLastError();
				if (err != cudaSuccess)
					printf( "unknown CUDA error in gridVisibilities() [ii] (%s)\n", cudaGetErrorString( err ) );

				// we may need to do gridding a few rows at a time if the grid is too large is fit on the GPU.
				int startRow = 0;
				int endRow = pSize - 1;
				for ( int imageBatch = 0; imageBatch < numberOfBatches; imageBatch++ )
				{

					// if we have more than one batch then
					// 1. clear the grid on the GPU.
					// 2. work out which visibilities we need to upload. we need the visibilities for these rows, plus all the visibilities
					//    that are within one kernel size.
					if (numberOfBatches > 1)
					{

						// recalculate startRow and endRow.
						startRow = gpuRows * imageBatch;
						endRow = startRow + gpuRows - 1;
						if (endRow >= pSize)
							endRow = pSize - 1;

					}
					int batchRows = endRow - startRow + 1;

					// calculate which visibilities we need.
					int startRowForVis = startRow - phstSupportSize[ kernelSet ];
					int endRowForVis = endRow + phstSupportSize[ kernelSet ];
					if (startRowForVis < 0)
						startRowForVis = 0;
					if (endRowForVis >= pSize)
						endRowForVis = pSize - 1;

					// get the start and end visibility id for these rows. we can use a search to get this info.
					int startVisibility = firstVisibility;
					int endVisibility = firstVisibility + numVisibilities - 1;
					int numVisibilitiesInBatch = endVisibility + 1 - startVisibility;

					if (numberOfBatches > 1)
					{

						// declare some device memory for the first and last visibility ids.
						int * devVis = NULL;
						reserveGPUMemory( (void **) &devVis, sizeof( int ), "creating device memory for the visibility id" );

						// define the block/thread dimensions.
						int threads = numVisibilities;
						int blocks;
						setThreadBlockSize1D( &threads, &blocks );

						// do a search to find the start visibility.
						if (startRowForVis > 0)
						{
							startVisibility = numVisibilities;
							cudaMemcpy( devVis, &startVisibility, sizeof( int ), cudaMemcpyHostToDevice );
							devSearchForVis<<< blocks, threads >>>(	/* IN: pGridPositions = */ &pdevGridPositions[ firstVisibility ],
												/* IN: pRowID = */ startRowForVis,
												/* IN: pNumVisibilities = */ numVisibilities,
												/* IN: pFirstOrLast = */ FIRST,
												/* IN: pVisibilityID = */ devVis );
							cudaMemcpy( &startVisibility, devVis, sizeof( int ), cudaMemcpyDeviceToHost );
							startVisibility += firstVisibility;
						}

						// do a search to find the end visibility.
						if (endRowForVis < pSize - 1)
						{
							endVisibility = -1;
							cudaMemcpy( devVis, &endVisibility, sizeof( int ), cudaMemcpyHostToDevice );
							devSearchForVis<<< blocks, threads >>>(	/* IN: pGridPositions = */ &pdevGridPositions[ firstVisibility ],
												/* IN: pRowID = */ endRowForVis,
												/* IN: pNumVisibilities = */ numVisibilities,
												/* IN: pFirstOrLast = */ LAST,
												/* IN: pVisibilityID = */ devVis );
							cudaMemcpy( &endVisibility, devVis, sizeof( int ), cudaMemcpyDeviceToHost );
							endVisibility += firstVisibility;
						}

						// update the number of visibilities in this batch.
						numVisibilitiesInBatch = endVisibility + 1 - startVisibility;

						// free memory.
						if (devVis != NULL)
							cudaFree( (void *) devVis );

					}

					// only do gridding if we have something to grid.
					if (numVisibilitiesInBatch > 0)
					{

						// for degridding, if we have multiple batches then upload the section of image to the device.
						if (numberOfBatches > 1)
							moveHostToDevice( 	(void *) pdevGrid,
										(void *) &phstGrid[ (long int) imageBatch * (long int) gpuRows *
													(long int) pSize ],
										batchRows * pSize * sizeof( cufftComplex/*grid*/ ),
										"copying grid portion to device" );

						// define the block/thread dimensions.
						setThreadBlockSizeForGridding(	/* pThreadsX = */ phstKernelSize[ kernelSet ],
										/* pThreadsY = */ phstKernelSize[ kernelSet ],
										/* pItems = */ numVisibilitiesInBatch );
				
						// work out how much shared memory we need to store items-per-block visibilities.
						int sharedMemSize = _itemsPerBlock * (sizeof( cufftComplex ) + sizeof( VectorI ) + sizeof( int ) + sizeof( float ));

						// do the 2-d convolution loop on the device.
						devGridVisibilities<<< _gridSize2D, _blockSize2D, sharedMemSize >>>
								(	/* pGrid = */ pdevGrid,
									/* pVisibility = */ &pdevVisibility[ startVisibility ],
									/* pVisibilitiesPerBlock = */ _itemsPerBlock,
									/* pBlocksPerVisibility = */ _blocksPerItem,
									/* pGridPosition = */ &pdevGridPositions[ startVisibility ],
									/* pKernelIndex = */ (pdevKernelIndex != NULL ? &pdevKernelIndex[ startVisibility ] : NULL),
									/* pWeight = */ (pdevWeight != NULL ? &pdevWeight[ startVisibility ] : NULL),
									/* pNumVisibilities = */ numVisibilitiesInBatch,
									/* pGridDegrid = */ pGridDegrid,
									/* pNumKernels = */ kernelsPerSet,
									/* pYOffset = */ imageBatch * gpuRows,
									/* pNumberOfRows = */ batchRows,
									/* pSize = */ pSize,
									/* pComplex : 0 (N), or 1 (Y) = */ 1 );

						err = cudaGetLastError();
						if (err != cudaSuccess)
							printf( "error gridding visibilities on device (%s)\n", cudaGetErrorString( err ) );

						// for gridding, if we have more than one batch then download the section of image into host memory.
						if (numberOfBatches > 1 && pGridDegrid == GRID)
							moveDeviceToHost( (void *) &phstGrid[ (long int) imageBatch * (long int) gpuRows * (long int) pSize ],
										(void *) pdevGrid, batchRows * pSize * sizeof( cufftComplex/*grid*/ ),
										"copying grid portion to host" );

					}

				}
	
				cudaDeviceSynchronize();

				// don't need texture any more.
				cudaUnbindTexture( _kernelTextureReal );
				cudaUnbindTexture( _kernelTextureImag );

			}

			// free the 3D arrays.
			if (_devKernelArrayRealPtr != NULL)
			{
				cudaFreeArray( _devKernelArrayRealPtr );
				_devKernelArrayRealPtr = NULL;
			}
			if (_devKernelArrayImagPtr != NULL)
			{
				cudaFreeArray( _devKernelArrayImagPtr );
				_devKernelArrayImagPtr = NULL;
			}

			// use the next lot of visibilities.
			firstVisibility += numVisibilities;
		
		}

		// display the support sizes.
		bool first = true;
		printf( "                --> done. support size used in " );
		if (pGridDegrid == GRID)
			printf( "gridding = [" );
		else
			printf( "degridding = [" );
		for ( int i = 0; i < (pWPlanes * pAPlanes); i++ )
			if (pVisibilitiesInKernelSet[ i ] > 0)
			{
				if (first == false)
					printf( ", " );
				printf( "%i", phstSupportSize[ i ] );
				first = false;
			}
		printf( "]\n\n" );

	}
	
} // gridVisibilities

//
//	gridComponents()
//
//	CJS: 22/05/2020
//
//	Grids a list of clean components to the clean image.
//

void gridComponents(	float/*grid*/ * pdevGrid,			// data area (device) holding the grid
			float/*grid*/ * phstGrid,			// data area (host) holding the larger grid if we're gridding in batches.
			double * pdevComponentValue,			// data area (device) holding the visibilities
			int phstSupportSize,				// kernel and gridding parameters
			float/*grid*/ * pdevKernel,			// the kernel array.
			VectorI * pdevGridPositions,			// a list of integer grid positions for each visibility
			int pComponents,				// the number of components to grid.
			int pSize )					// the size of the image
{
	
	cudaError_t err;
		
	err = cudaGetLastError();
	if (err != cudaSuccess)
		printf( "unknown CUDA error in gridComponents() [i] (%s)\n", cudaGetErrorString( err ) );

	// are we gridding in batches ? if no host grid has been supplied then we HAVE to grid on the GPU.
	int numberOfBatches = _hstImageBatches;
	if (phstGrid == NULL)
		numberOfBatches = 1;

	// calculate kernel size.
	int hstKernelSize = (phstSupportSize * 2) + 1;

	// initialise the 3D arrays to null.
	_devKernelArrayRealPtr = NULL;

	// generate the texture maps for these kernel sets.
	if (pComponents > 0)
	{

		// upload these kernels to CUDA arrays, where they will later be bound to texture maps.
		createCUDAArrayForKernels(	/* pdevKernel = */ pdevKernel,
						/* pKernelSize = */ hstKernelSize,
						/* pNumKernels = */ 1 );
	
		// copy constants to constant memory.
		err = cudaMemcpyToSymbol( _devSupport, &phstSupportSize, sizeof( phstSupportSize ) );
		if (err != cudaSuccess)
			printf( "error copying support size to device (%s)\n", cudaGetErrorString( err ) );
		err = cudaMemcpyToSymbol( _devKernelSize, &hstKernelSize, sizeof( hstKernelSize ) );
		if (err != cudaSuccess)
			printf( "error copying kernel size to device (%s)\n", cudaGetErrorString( err ) );

		// bind real array to texture map.
		err = cudaBindTextureToArray( _kernelTextureReal, _devKernelArrayRealPtr, _channelDesc );
		if (err != cudaSuccess)
			printf( "Error binding texture to real array (%s)\n", cudaGetErrorString( err ) );

		// ensure we don't report old errors.
		err = cudaGetLastError();
		if (err != cudaSuccess)
			printf( "unknown CUDA error in gridComponents() [ii] (%s)\n", cudaGetErrorString( err ) );

		// we may need to do gridding a few rows at a time if the grid is too large is fit on the GPU.
		int startRow = 0;
		int endRow = pSize - 1;
		for ( int imageBatch = 0; imageBatch < numberOfBatches; imageBatch++ )
		{

			// if we have more than one batch then
			// 1. clear the grid on the GPU.
			// 2. work out which visibilities we need to upload. we need the visibilities for these rows, plus all the visibilities
			//    that are within one kernel size.
			if (numberOfBatches > 1)
			{

				// recalculate startRow and endRow.
				startRow = _hstGPURows * imageBatch;
				endRow = startRow + _hstGPURows - 1;
				if (endRow >= pSize)
					endRow = pSize - 1;

			}
			int batchRows = endRow - startRow + 1;

			// calculate which visibilities we need.
			int startRowForComponent = startRow - phstSupportSize;
			int endRowForComponent = endRow + phstSupportSize;
			if (startRowForComponent < 0)
				startRowForComponent = 0;
			if (endRowForComponent >= pSize)
				endRowForComponent = pSize - 1;

			// get the start and end visibility id for these rows. we can use a search to get this info.
			int startComponent = 0;
			int endComponent = pComponents - 1;
			int numComponentsInBatch = endComponent + 1 - startComponent;

			if (numberOfBatches > 1)
			{

				// declare some device memory for the first and last component ids.
				int * devComponent = NULL;
				reserveGPUMemory( (void **) &devComponent, sizeof( int ), "creating device memory for the visibility id" );

				// define the block/thread dimensions.
				int threads = pComponents;
				int blocks;
				setThreadBlockSize1D( &threads, &blocks );

				// do a search to find the start visibility.
				if (startRowForComponent > 0)
				{
					startComponent = pComponents;
					cudaMemcpy( devComponent, &startComponent, sizeof( int ), cudaMemcpyHostToDevice );
					devSearchForVis<<< blocks, threads >>>(	/* IN: pGridPositions = */ pdevGridPositions,
										/* IN: pRowID = */ startRowForComponent,
										/* IN: pNumVisibilities = */ pComponents,
										/* IN: pFirstOrLast = */ FIRST,
										/* IN: pVisibilityID = */ devComponent );
					cudaMemcpy( &startComponent, devComponent, sizeof( int ), cudaMemcpyDeviceToHost );
				}

				// do a search to find the end visibility.
				if (endRowForComponent < pSize - 1)
				{
					endComponent = -1;
					cudaMemcpy( devComponent, &endComponent, sizeof( int ), cudaMemcpyHostToDevice );
					devSearchForVis<<< blocks, threads >>>(	/* IN: pGridPositions = */ pdevGridPositions,
										/* IN: pRowID = */ endRowForComponent,
										/* IN: pNumVisibilities = */ pComponents,
										/* IN: pFirstOrLast = */ LAST,
										/* IN: pVisibilityID = */ devComponent );
					cudaMemcpy( &endComponent, devComponent, sizeof( int ), cudaMemcpyDeviceToHost );
				}

				// update the number of visibilities in this batch.
				numComponentsInBatch = endComponent + 1 - startComponent;

				// free memory.
				if (devComponent != NULL)
					cudaFree( (void *) devComponent );

			}

			// only do gridding if we have something to grid.
			if (numComponentsInBatch > 0)
			{

				// clear the grid.
				if (numberOfBatches > 1)
					moveHostToDevice(	(void *) pdevGrid,
								(void *) &phstGrid[ (long int) imageBatch * (long int) _hstGPURows * (long int) pSize ],
								batchRows * pSize * sizeof( float/*grid*/ ),
								"copying grid portion to device" );

				// define the block/thread dimensions.
				setThreadBlockSizeForGridding(	/* pThreadsX = */ hstKernelSize,
								/* pThreadsY = */ hstKernelSize,
								/* pItems = */ numComponentsInBatch );

				// work out how much shared memory we need to store items-per-block visibilities.
				int sharedMemSize = _itemsPerBlock * (sizeof( cufftComplex ) + sizeof( VectorI ) + sizeof( int ) + sizeof( float ));

				// do the 2-d convolution loop on the device.
				devGridVisibilities<<< _gridSize2D, _blockSize2D, sharedMemSize >>>
						(	/* pGrid = */ (cufftComplex/*grid*/ *) pdevGrid,
							/* pVisibility = */ (cufftComplex *) &pdevComponentValue[ startComponent ],
							/* pVisibilitiesPerBlock = */ _itemsPerBlock,
							/* pBlocksPerVisibility = */ _blocksPerItem,
							/* pGridPosition = */ &pdevGridPositions[ startComponent ],
							/* pKernelIndex = */ NULL,
							/* pWeight = */ NULL,
							/* pNumVisibilities = */ numComponentsInBatch,
							/* pGridDegrid = */ GRID,
							/* pNumKernels = */ 1,
							/* pYOffset = */ imageBatch * _hstGPURows,
							/* pNumberOfRows = */ _hstGPURows,
							/* pSize = */ pSize,
							/* pComplex : 0 (N), or 1 (Y) = */ 0 );

				err = cudaGetLastError();
				if (err != cudaSuccess)
					printf( "error gridding visibilities on device (%s)\n", cudaGetErrorString( err ) );

				// for gridding, if we have more than one batch then download the section of image into host memory.
				if (numberOfBatches > 1)
					moveDeviceToHost( (void *) &phstGrid[ (long int) imageBatch * (long int) _hstGPURows * (long int) pSize ],
								(void *) pdevGrid, batchRows * pSize * sizeof( float/*grid*/ ),
								"copying grid portion to host" );

			}

		}
	
		cudaDeviceSynchronize();

		// don't need texture any more.
		cudaUnbindTexture( _kernelTextureReal );

	}

	// free the 3D arrays.
	if (_devKernelArrayRealPtr != NULL)
	{
		cudaFreeArray( _devKernelArrayRealPtr );
		_devKernelArrayRealPtr = NULL;
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
	
	// create the deconvolution image, and clear it.
	cufftComplex/*grid*/ * devDeconvolutionImageGrid = NULL;
	reserveGPUMemory( (void **) &devDeconvolutionImageGrid, _hstPsfSize * _hstPsfSize * sizeof( cufftComplex/*grid*/ ),
				"declaring memory for deconvolution image" );
	zeroGPUMemory( (void *) devDeconvolutionImageGrid, _hstPsfSize * _hstPsfSize * sizeof( cufftComplex/*grid*/ ), "zeroing the grid on the device" );

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
	int tmphstVisibilitiesPerKernelSet = 1;
		
	// generate the deconvolution function by gridding a single visibility without w-projection
	// (mirror visibilities = false, use conjugate values = false).
	printf( "gridding visibilities for deconvolution function.....\n\n" );
	gridVisibilities(	/* pdevGrid = */ devDeconvolutionImageGrid,
				/* phstGrid = */ NULL,
				/* pdevVisibility = */ tmpdevVisibility,
				/* pOversample = */ 1,
				/* pKernelSize = */ &_hstAAKernelSize,
				/* pSupport = */ &_hstAASupport,
				/* pdevKernelIndex = */ tmpdevKernelIndex,
				/* pWProjection = */ false,
				/* pAProjection = */ false,
				/* pWPlanes = */ 1,
				/* pAPlanes = */ 1,
				/* pdevGridPositions = */ tmpdevGridPositions,
				/* pdevWeight = */ tmpdevWeight,
				/* pVisibilitiesInKernelSet = */ &tmphstVisibilitiesPerKernelSet,
				/* pGridDegrid = */ GRID,
				/* phstPrimaryBeamMosaicing = */ NULL,
				/* phstPrimaryBeamAProjection = */ NULL,
				/* pNumFields = */ -1,
				/* pMosaicIndex = */ -1,
				/* pSize = */ _hstPsfSize );
		
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

	// we either need to keep the deconvolution image on the host or the device. we don't need both.
	if (_hstImageBatches > 1)
	{
		if (_devDeconvolutionImage != NULL)
			cudaFree( (void *) _devDeconvolutionImage );
		_devDeconvolutionImage = NULL;
	}
	else
	{
		if (_hstDeconvolutionImage != NULL)
			free( (void *) _hstDeconvolutionImage );
		_hstDeconvolutionImage = NULL;
	}

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

void extractFromMosaic( float/*grid*/ * phstImageArray, float/*grid*/ * pdevMosaic, bool * phstMask, double * phstPhaseCentre, float * phstPrimaryBeamPattern )
{

	// create memory for mosaic, mask and beam on the device.
	bool * devMask = NULL;
	float/*grid*/ * devBeam = NULL;

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
	float/*grid*/ * devOutImage = NULL;
	reserveGPUMemory( (void **) &devOutImage, (long int) _hstUvPixels * (long int) _hstUvPixels * (long int) sizeof( float/*grid*/ ),
				"creating device memory for image extracted from mosaic" );

	// perform an image-plane reprojection of each image from the phase position of the field to the phase position of the mosaic.
	for ( int image = 0; image < _numMosaicImages; image++ )
	{

		// clear the grid.
		zeroGPUMemory( (void *) devOutImage, (long int) _hstUvPixels * (long int) _hstUvPixels * (long int) sizeof( float/*grid*/ ),
					"zeroing grid on the device" );

		// set out coordinate system RA and DEC.
		outCoordSystem.crVAL.x = phstPhaseCentre[ image * 2 ];
		outCoordSystem.crVAL.y = phstPhaseCentre[ (image * 2) + 1 ];

		// upload the primary beam to the device.
		cudaMemcpy( devBeam, &_hstPrimaryBeam[ (long int) image * (long int) _hstBeamSize * (long int) _hstBeamSize ], _hstBeamSize * _hstBeamSize *
								sizeof( float ), cudaMemcpyHostToDevice );

		// reproject this image in order to construct this part of the mosaic.
// cjs-mod		imagePlaneReprojection.ReprojectImage(	/* pdevInImage = */ pdevMosaic,
//							/* pdevOutImage = */ devOutImage,
//							/* pdevNormalisationPattern = */ NULL,
//							/* pdevPrimaryBeamPattern = */ NULL,
//							/* pInCoordinateSystem = */ inCoordSystem,
//							/* pOutCoordinateSystem = */ outCoordSystem,
//							/* pInSize = */ size,
//							/* pOutSize = */ size,
//							/* pdevInMask = */ devMask,
//							/* pdevBeamIn = */ NULL,
//							/* pdevBeamOut = */ devBeam,
//							/* pBeamScale = */ (float/*grid*/) _hstBeamSize / (float/*grid*/) _hstUvPixels,
//							/* pProjectionDirection = */ Reprojection::INPUT_TO_OUTPUT,
//							/* pAProjection = */ _hstAProjection,
//							/* pVerbose = */ false );

		// store the image on the host.
		moveDeviceToHost( (void *) &phstImageArray[ (long int) _hstUvPixels * (long int) _hstUvPixels * (long int) image ], (void *) devOutImage,
					_hstUvPixels * _hstUvPixels * sizeof( float/*grid*/ ), "copying mosaic component to host" );

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

void createMosaic( float/*grid*/ * phstMosaic, float/*grid*/ * phstImageArray, bool * phstMask, double * phstPhaseCentre, float ** phstPrimaryBeamPatternPtr )
{

	// create memory for mosaic, pixel weights, mask and primary beam on the device.
	float/*grid*/ * devMosaic = NULL;
	float * devNormalisationPattern = NULL;
	float * devPrimaryBeamPattern = NULL;
	bool * devMask = NULL;
	float * devBeam = NULL;
	reserveGPUMemory( (void **) &devMosaic, _hstUvPixels * _hstUvPixels * sizeof( float/*grid*/ ), "creating device memory for the mosaic" );
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
	zeroGPUMemory( (void *) devMosaic, _hstUvPixels * _hstUvPixels * sizeof( float/*grid*/ ), "zeroing mosaic on the device" );
	zeroGPUMemory( (void *) devNormalisationPattern, _hstBeamSize * _hstBeamSize * sizeof( float ), "zeroing normalisation pattern on the device" );
	if (phstPrimaryBeamPatternPtr != NULL)
		zeroGPUMemory( (void *) devPrimaryBeamPattern, _hstBeamSize * _hstBeamSize * sizeof( float ), "zeroing primary beam pattern on the device" );
	zeroGPUMemory( (void *) devMask, _hstUvPixels * _hstUvPixels * sizeof( bool ), "zeroing mask on the device" ); // set mask to false (zero).

	// create the device memory for the reprojection code.
	imagePlaneReprojection.CreateDeviceMemory( size );

	// create device memory for the input image.
	float/*grid*/ * devInImage = NULL;
	reserveGPUMemory( (void **) &devInImage, _hstUvPixels * _hstUvPixels * sizeof( double ), "creating device memory for image to be added to the mosaic" );

	// perform an image-plane reprojection of each image from the phase position of the field to the phase position of the mosaic.
	for ( int image = 0; image < _numMosaicImages; image++ )
	{

		// copy the image into a temporary work location on the device.
		cudaMemcpy( devInImage, &phstImageArray[ image * _hstUvPixels * _hstUvPixels ], _hstUvPixels * _hstUvPixels * sizeof( float/*grid*/ ),
				cudaMemcpyHostToDevice );

		// set in coordinate system RA and DEC.
		inCoordSystem.crVAL.x = phstPhaseCentre[ image * 2 ];
		inCoordSystem.crVAL.y = phstPhaseCentre[ (image * 2) + 1 ];

		// upload the primary beam to the device.
		if (image == 0 || _hstFileMosaic == true)
			cudaMemcpy( devBeam, &_hstPrimaryBeam[ image * _hstBeamSize * _hstBeamSize ], _hstBeamSize * _hstBeamSize * sizeof( float ),
					cudaMemcpyHostToDevice );

		// reproject this image in order to construct this part of the mosaic.
// cjs-mod		imagePlaneReprojection.ReprojectImage(	/* pdevInImage = */ devInImage,
//							/* pdevOutImage = */ devMosaic,
//							/* pdevNormalisationPattern = */ devNormalisationPattern,
//							/* pdevPrimaryBeamPattern = */ devPrimaryBeamPattern,
//							/* pInCoordinateSystem = */ inCoordSystem,
//							/* pOutCoordinateSystem = */ outCoordSystem,
//							/* pInSize = */ size,
//							/* pOutSize = */ size,
//							/* pdevInMask = */ NULL,
//							/* pdevBeamIn = */ devBeam,
//							/* pdevBeamOut = */ NULL,
//							/* pBeamScale = */ (double) _hstBeamSize / (double) _hstUvPixels,
//							/* pProjectionDirection = */ Reprojection::OUTPUT_TO_INPUT,
//							/* pAProjection = */ _hstAProjection,
//							/* pVerbose = */ false );

	}
	printf( "\n" );

	// free memory.
	if (devInImage != NULL)
		cudaFree( (void *) devInImage );

	// update the image from its weight.
// cjs-mod	imagePlaneReprojection.ReweightImage(	/* pdevOutImage = */ devMosaic,
//						/* pdevNormalisationPattern = */ devNormalisationPattern,
//						/* pdevPrimaryBeamPattern = */ devPrimaryBeamPattern,
//						/* pOutSize = */ size,
//						/* pdevOutMask = */ devMask );

	// store the image and the mask on the host.
	moveDeviceToHost( (void *) phstMosaic, (void *) devMosaic, _hstUvPixels * _hstUvPixels * sizeof( float/*grid*/ ), "copying mosaic image to host" );
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

void hogbomClean( int * pMinorCycle, bool * phstMask, double pHogbomLimit, float/*grid*/ * pdevCleanBeam, float/*grid*/ * pdevDirtyBeam,
			float/*grid*/ * pdevDirtyImage, float/*grid*/ * phstDirtyImage, VectorI * phstComponentListPos, double * phstComponentListValue,
			int * pComponentListItems )
{
	
	cudaError_t err;
		
	printf( "\n                minor cycles: " );
	fflush( stdout );
	
	double * devMaxValue;
	reserveGPUMemory( (void **) &devMaxValue, MAX_PIXEL_DATA_AREA_SIZE * sizeof( double ),
						"declaring device memory for psf max pixel value" );
	double * devMaxValueParallel;
	reserveGPUMemory( (void **) &devMaxValueParallel, _hstImageBatches * MAX_PIXEL_DATA_AREA_SIZE * sizeof( double ),
						"declaring device memory for psf max pixel value" );
		
	// reserve host memory for the maximum pixel value.
	double * tmpMaxValue = (double *) malloc( MAX_PIXEL_DATA_AREA_SIZE * sizeof( double ) );

	// keep a record of the minimum value. if it does up by a certain factor then we need to stop cleaning.
	double minimumValue = -1.0;
	
	cudaDeviceSynchronize();

	// pin the host memory.
	if (_hstImageBatches > 1)
	{
		cudaHostRegister( phstDirtyImage, (long int) _hstUvPixels * (long int) _hstUvPixels * (long int) sizeof( float ), 0 );
		cudaHostRegister( phstMask, (long int) _hstUvPixels * (long int) _hstUvPixels * (long int) sizeof( bool ), 0 );
	}

	err = cudaGetLastError();
	if (err != cudaSuccess)
		printf( "unknown CUDA error in hogbomClean() [i] (%s)\n", cudaGetErrorString( err ) );

	// create device memory for the mask. we only get the mask from the cache if we're using one.
	bool * devMask = NULL;
	if (phstMask != NULL && _hstImageBatches == 1)
	{
		reserveGPUMemory( (void **) &devMask, _hstUvPixels * _hstUvPixels * sizeof( bool ), "creating device memory for the mask" );
		cudaMemcpy( (void *) devMask, phstMask, _hstUvPixels * _hstUvPixels * sizeof( bool ), cudaMemcpyHostToDevice );
	}
	
	// loop over each minor cycle.
	bool firstMinorCycle = true;
	while (*pMinorCycle < _hstMinorCycles)
	{
		
		printf( "." );
		fflush( stdout );

		// are we cleaning on the device or the host ?
		if (_hstImageBatches == 1)
		{
		
			// get maximum pixel value.
			getMaxValue(	/* pdevImage = */ pdevDirtyImage,
					/* pdevMaxValue = */ devMaxValue,
					/* pWidth = */ _hstUvPixels,
					/* pHeight = */ _hstUvPixels,
					/* pdevMask = */ devMask );
	
			// get details back from the device.
			moveDeviceToHost( (void *) tmpMaxValue, (void *) devMaxValue, MAX_PIXEL_DATA_AREA_SIZE * sizeof( double ),
						"copying max pixel data to host" );

		}
		else
		{

			// only search for the maximum residual on the first minor cycle.
			if (firstMinorCycle == true)
			{

				const int ASYNC_STAGES = 2;
				const int NUM_STREAMS = 1; // ASYNC_STAGES + 1;

				// create the dirty image on the device.
				float/*grid*/ * devDirtyImage[ NUM_STREAMS ];
				for ( int i = 0; i < NUM_STREAMS; i++ )
					reserveGPUMemory( (void **) &devDirtyImage[ i ], _hstGPURows * _hstUvPixels * sizeof( double ),
							"reserving device memory for dirty image" );
				bool * devMaskAsync[ NUM_STREAMS ];
				for ( int i = 0; i < NUM_STREAMS; i++ )
					reserveGPUMemory( (void **) &devMask[ i ], _hstGPURows * _hstUvPixels * sizeof( bool ),
								"reserving device memory for image mask" );

				// create a new cuda stream.
				cudaStream_t cudaStream[ NUM_STREAMS ];
				for ( int i = 0; i < NUM_STREAMS; i++ )
					cudaStreamCreate( &cudaStream[ i ] );

				for ( int batch = 0; batch < _hstImageBatches + ASYNC_STAGES - 1; batch++ )
				{

					int writeBatch = batch;
					int processBatch = (NUM_STREAMS > 1 ? batch - 1 : batch);

					// calculate which cuda streams we're working with.
					int writeIndex = mod( writeBatch, NUM_STREAMS );
					int processIndex = mod( processBatch, NUM_STREAMS );

					err = cudaGetLastError();
					if (err != cudaSuccess)
						printf( "in hogbom clean (%s)\n", cudaGetErrorString( err ) );

					if (writeBatch >= 0 && writeBatch < _hstImageBatches)
					{

						// how many rows for writing ?
						int rows = _hstGPURows;
						if ((writeBatch + 1) * _hstGPURows > _hstUvPixels)
							rows = _hstUvPixels - (writeBatch * _hstGPURows);

						// upload a portion of the dirty image to the device.
						moveHostToDeviceAsync(	/* pToPtr = */ (void *) devDirtyImage[ writeIndex ],
									/* pFromPtr = */ (void *) &phstDirtyImage[ (long int) writeBatch * (long int) _hstGPURows *
															(long int) _hstUvPixels ],
									/* pSize = */ rows * _hstUvPixels * sizeof( float/*grid*/ ),
									/* pTask = */ "moving dirty image to device",
									/* pStream = */ cudaStream[ writeIndex ] );

						// upload the required part of the mask.
						moveHostToDeviceAsync(	(void *) devMaskAsync[ writeIndex ],
									(void *) &phstMask[ (long int) writeBatch * (long int) _hstGPURows *
													(long int) _hstUvPixels ],
									rows * _hstUvPixels * sizeof( bool ), "moving image mask to device",
									cudaStream[ writeIndex ] );

					}

					if (processBatch >= 0 && processBatch < _hstImageBatches)
					{

						// how many rows for reading ?
						int rows = _hstGPURows;
						if ((processBatch + 1) * _hstGPURows > _hstUvPixels)
							rows = _hstUvPixels - (processBatch * _hstGPURows);

						// get maximum pixel value.
						getMaxValue(	/* pdevImage = */ devDirtyImage[ processIndex ],
								/* pdevMaxValue = */ &devMaxValueParallel[ processBatch * MAX_PIXEL_DATA_AREA_SIZE ],
								/* pWidth = */ _hstUvPixels,
								/* pHeight = */ rows,
								/* pdevMask = */ devMaskAsync[ processIndex ],
								/* pStream = */ cudaStream[ processIndex ] );

					}

				}

				// destroy cuda stream.
				for ( int i = 0; i < NUM_STREAMS; i++ )
					cudaStreamDestroy( cudaStream[ i ] );

				// free memory.
				for ( int i = 0; i < NUM_STREAMS; i++ )
				{
					cudaFree( (void *) devDirtyImage[ i ] );
					cudaFree( (void *) devMaskAsync[ i ] );
				}

			}

			cudaDeviceSynchronize();

			// get details back from the device.
			double * tmpMaxValueParallel = (double *) malloc( _hstImageBatches * MAX_PIXEL_DATA_AREA_SIZE * sizeof( double ) );
			moveDeviceToHost( (void *) tmpMaxValueParallel, (void *) devMaxValueParallel,
						_hstImageBatches * MAX_PIXEL_DATA_AREA_SIZE * sizeof( double ), "copying max pixel data to host" );

			// get the maximum value from these batches.
			memcpy( tmpMaxValue, tmpMaxValueParallel, MAX_PIXEL_DATA_AREA_SIZE * sizeof( double ) );
			for ( int batch = 1; batch < _hstImageBatches; batch++ )
				if (tmpMaxValueParallel[ (batch * MAX_PIXEL_DATA_AREA_SIZE) + MAX_PIXEL_VALUE ] > tmpMaxValue[ MAX_PIXEL_VALUE ])
				{
					memcpy( tmpMaxValue, &tmpMaxValueParallel[ batch * MAX_PIXEL_DATA_AREA_SIZE ],
							MAX_PIXEL_DATA_AREA_SIZE * sizeof( double) );
					tmpMaxValue[ MAX_PIXEL_Y ] += (double) (batch * _hstGPURows);
				}

			// free memory.
			if (tmpMaxValueParallel != NULL)
				free( tmpMaxValueParallel );

			// update the maximum value back to the device.
			moveHostToDevice( (void *) devMaxValue, tmpMaxValue, MAX_PIXEL_DATA_AREA_SIZE * sizeof( double ), "copying max pixel data to device" );

			firstMinorCycle = false;

		}

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

		// are we subtracting/adding the beams in one batch, or many ?
		if (_hstImageBatches == 1)
		{
		
			// define the block/thread dimensions.
			setThreadBlockSize2D( _hstPsfSize, _hstPsfSize );
				
			// subtract dirty beams.
			devAddSubtractBeams<<< _gridSize2D, _blockSize2D >>>(	/* pImage = */ pdevDirtyImage,
										/* pBeam = */ pdevDirtyBeam,
										/* pMaxValue = */ devMaxValue,
										/* pWindowSize = */ _hstPsfSize / 2,
										/* pLoopGain = */ _hstLoopGain,
										/* pImageWidth = */ _hstUvPixels,
										/* pImageHeight = */ _hstUvPixels,
										/* pBeamSize = */ _hstPsfSize,
										/* pAddSubtract = */ SUBTRACT );
			err = cudaGetLastError();
			if (err != cudaSuccess)
				printf( "error subtracting beams (%s).\n", cudaGetErrorString( err ) );

		}
		else
		{

			const int ASYNC_STAGES = 3;
			const int NUM_STREAMS = 1; // ASYNC_STAGES + 1;

			// calculate first and last row that needs to be updated in the images.
			int firstRow = (int) (round( tmpMaxValue[ MAX_PIXEL_Y ] )) - (_hstPsfSize / 2);
			if (firstRow < 0)
				firstRow = 0;
			int lastRow = (int) (round( tmpMaxValue[ MAX_PIXEL_Y ] )) + (_hstPsfSize / 2);
			if (lastRow >= _hstUvPixels)
				lastRow = _hstUvPixels - 1;

			// calculate which sections we need.
			int firstSection = firstRow / _hstGPURows;
			int lastSection = lastRow / _hstGPURows;

			// create image portions on the device.
			float/*grid*/ * devImage[ NUM_STREAMS ];
			for ( int i = 0; i < NUM_STREAMS; i++ )
				reserveGPUMemory( (void **) &devImage[ i ], _hstGPURows * _hstUvPixels * sizeof( float/*grid*/ ),
							"reserving device memory for dirty/clean image" );
			bool * devMaskAsync[ NUM_STREAMS ];
			for ( int i = 0; i < NUM_STREAMS; i++ )
				reserveGPUMemory( (void **) &devMask[ i ], _hstGPURows * _hstUvPixels * sizeof( bool ),
							"reserving device memory for image mask" );

			// create a new cuda stream.
			cudaStream_t cudaStream[ NUM_STREAMS ];
			for ( int i = 0; i < NUM_STREAMS; i++ )
				cudaStreamCreate( &cudaStream[ i ] );

			for ( int batch = firstSection; batch <= lastSection + ASYNC_STAGES - 1; batch++ )
			{

				int writeBatch = batch;
				int processBatch = (NUM_STREAMS > 1 ? batch - 1 : batch);
				int readBatch = (NUM_STREAMS > 1 ? batch - 2 : batch);

				// calculate which image on the device we're reading and writing to.
				int writeIndex = mod( writeBatch, NUM_STREAMS );
				int processIndex = mod( processBatch, NUM_STREAMS );
				int readIndex = mod( readBatch, NUM_STREAMS );

				if (writeBatch >= firstSection && writeBatch <= lastSection)
				{

					// how many rows are we uploading ? ensure we don't upload more than we actually have.
					int rows = _hstGPURows;
					if ((writeBatch * _hstGPURows) + rows > _hstUvPixels)
						rows = _hstUvPixels - (writeBatch * _hstGPURows);

					// upload the required part of the dirty image.
					moveHostToDeviceAsync(	(void *) devImage[ writeIndex ],
								(void *) &phstDirtyImage[ (long int) writeBatch * (long int) _hstGPURows *
												(long int) _hstUvPixels ],
								rows * _hstUvPixels * sizeof( float/*grid*/ ), "moving dirty image to device",
								cudaStream[ writeIndex ] );

					// upload the required part of the mask.
					moveHostToDeviceAsync(	(void *) devMaskAsync[ writeIndex ],
								(void *) &phstMask[ (long int) writeBatch * (long int) _hstGPURows *
												(long int) _hstUvPixels ],
								rows * _hstUvPixels * sizeof( bool ), "moving image mask to device",
								cudaStream[ writeIndex ] );

				}

				if (processBatch >= firstSection && processBatch <= lastSection)
				{

					// update the Y coordinate of the data area to reflect the fact we've only uploaded part of our image.
					double yValue = tmpMaxValue[ MAX_PIXEL_Y ] - (double) (processBatch * _hstGPURows);
					moveHostToDeviceAsync( (void *) &devMaxValue[ MAX_PIXEL_Y ], (void *) &yValue, sizeof( double ),
								"moving image Y value to device", cudaStream[ processIndex ] );

					// how many rows are we uploading ? ensure we don't upload more than we actually have.
					int rows = _hstGPURows;
					if ((processBatch * _hstGPURows) + rows > _hstUvPixels)
						rows = _hstUvPixels - (processBatch * _hstGPURows);
		
					// define the block/thread dimensions.
					setThreadBlockSize2D( _hstPsfSize, _hstPsfSize );

					// subtract dirty beam.
					devAddSubtractBeams<<< _gridSize2D, _blockSize2D, 0, cudaStream[ processIndex ] >>>
							(	/* pImage = */ devImage[ processIndex ],
								/* pBeam = */ pdevDirtyBeam,
								/* pMaxValue = */ devMaxValue,
								/* pWindowSize = */ _hstPsfSize / 2,
								/* pLoopGain = */ _hstLoopGain,
								/* pImageWidth = */ _hstUvPixels,
								/* pImageHeight = */ rows,
								/* pBeamSize = */ _hstPsfSize,
								/* pAddSubtract = */ SUBTRACT );

					// find the largest residual that remains in this section of the image.
					getMaxValue(	/* pdevImage = */ devImage[ processIndex ],
							/* pdevMaxValue = */ &devMaxValueParallel[ processBatch * MAX_PIXEL_DATA_AREA_SIZE ],
							/* pWidth = */ _hstUvPixels,
							/* pHeight = */ rows,
							/* pdevMask = */ devMaskAsync[ processIndex ],
							/* pStream = */ cudaStream[ processIndex ] );

				}

				// download the updated dirty image portion into host memory.
				if (readBatch >= firstSection && readBatch <= lastSection)
				{

					// how many rows are we uploading ? ensure we don't upload more than we actually have.
					int rows = _hstGPURows;
					if ((readBatch * _hstGPURows) + rows > _hstUvPixels)
						rows = _hstUvPixels - (readBatch * _hstGPURows);

					moveDeviceToHostAsync(	(void *) &phstDirtyImage[ (long int) readBatch * (long int) _hstGPURows *
												(long int) _hstUvPixels ],
								(void *) devImage[ readIndex ],
								rows * _hstUvPixels * sizeof( float/*grid*/ ), "moving dirty image to host",
								cudaStream[ readIndex ] );

				}

			}

			// destroy cuda stream.
			for ( int i = 0; i < NUM_STREAMS; i++ )
				cudaStreamDestroy( cudaStream[ i ] );

			// free memory.
			for ( int i = 0; i < NUM_STREAMS; i++ )
			{
				cudaFree( (void *) devImage[ i ] );
				cudaFree( (void *) devMaskAsync[ i ] );
			}

			cudaDeviceSynchronize();

		}

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

	// unpin the host memory.
	if (_hstImageBatches > 1)
	{
		cudaHostUnregister( phstDirtyImage );
		cudaHostUnregister( phstMask );
	}
		
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

bool cottonSchwabClean( double * phstPhaseCentre, char * pFilenamePrefix, float/*grid*/ * pdevCleanBeam, float/*grid*/ * pdevDirtyBeam,
				float/*grid*/ ** phstDirtyImage, float/*grid*/ * phstDirtyImageCache, bool * phstMask,
				char * pCleanImageFilename, char * pResidualImageFilename )
{

	bool ok = true;
	cudaError_t err;

	// declare some device memory to hold the normalisation pattern.
	float * devNormalisationPattern = NULL;
	if (_hstNormalisationPattern != NULL)
	{

		reserveGPUMemory( (void **) &devNormalisationPattern, _hstBeamSize * _hstBeamSize * sizeof( float ), "declaring device memory for primary beam pattern" );

		// upload the primary beam pattern to the device.
		moveHostToDevice( (void *) devNormalisationPattern, (void *) _hstNormalisationPattern, _hstBeamSize * _hstBeamSize * sizeof( float ),
					"copying primary beam pattern to the host" );

	}

	// create memory for the primary beam pattern on the device.
	float * devPrimaryBeamPattern = NULL;
	if (_hstBeamMosaic == true)
	{

		reserveGPUMemory( (void **) &devPrimaryBeamPattern, _hstBeamSize * _hstBeamSize * sizeof( float ), "reserving device memory for primary beam pattern" );

		// copy primary beam pattern to the device.
		moveHostToDevice( (void *) devPrimaryBeamPattern, (void *) _hstPrimaryBeamPattern, _hstBeamSize * _hstBeamSize * sizeof( float ),
					"copying primary beam pattern to the device" );

	}

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

		float/*grid*/ * hstMosaic = NULL;

		// if we are mosaicing, create a mosaic from the dirty images.
		if (_hstFileMosaic == true)
		{

			// create memory for mosaic on the host.
			hstMosaic = (float/*grid*/ *) malloc( (long int) _hstUvPixels * (long int) _hstUvPixels * (long int) sizeof( float/*grid*/ ) );

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
			float/*grid*/ * devDirtyImage = NULL;
			if (_hstImageBatches == 1)
			{

				reserveGPUMemory( (void **) &devDirtyImage, _hstUvPixels * _hstUvPixels * sizeof( float/*grid*/ ),
							"reserving device memory for dirty image (cleaning)" );

				// get the dirty image, clean image and model images from the cache.
				if (hstMosaic != NULL)
					cudaMemcpy( (void *) devDirtyImage, hstMosaic, _hstUvPixels * _hstUvPixels * sizeof( float/*grid*/ ), cudaMemcpyHostToDevice );
				else
					cudaMemcpy( (void *) devDirtyImage, *phstDirtyImage, _hstUvPixels * _hstUvPixels * sizeof( float/*grid*/ ),
							cudaMemcpyHostToDevice );

			}

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

		// create the model image on the device.
		float/*grid*/ * devModelImage = NULL;
		reserveGPUMemory( (void **) &devModelImage, _hstGPURows * _hstUvPixels * sizeof( float/*grid*/ ), "creating memory for the model image" );

		// create the model image on the host.
		float/*grid*/ * hstModelImage = NULL;
		if (_hstImageBatches > 1)
			hstModelImage = (float/*grid*/ *) malloc( (long int) _hstUvPixels * (long int) _hstUvPixels * (long int) sizeof( float/*grid*/ ) );

		// upload the component values to the device.
		double * devComponentValue = NULL;
		reserveGPUMemory( (void **) &devComponentValue, numComponents * sizeof( double ), "reserving device memory for clean components" );
		moveHostToDevice( (void *) devComponentValue, hstComponentListValue, numComponents * sizeof( double ), "moving component list values to the device" );

		// upload the grid positions to the device.
		VectorI * devComponentPos = NULL;
		reserveGPUMemory( (void **) &devComponentPos, numComponents * sizeof( VectorI ), "reserving device memory for clean component positions" );
		moveHostToDevice( (void *) devComponentPos, hstComponentListPos, numComponents * sizeof( VectorI ), "moving component list positions to the device" );

		// upload a single pixel as a gridding kernel.
		float/*grid*/ * devKernel = NULL;
		float/*grid*/ kernel = 1.0;
		reserveGPUMemory( (void **) &devKernel, 1 * sizeof( float/*grid*/ ), "reserving device memory for the model image gridding kernel" );
		cudaMemcpy( devKernel, &kernel, sizeof( float/*grid*/ ), cudaMemcpyHostToDevice );

		// if we're not using gridding batches then clear the model image on the device.
		if (_hstImageBatches == 1)
			cudaMemset( devModelImage, 0, _hstUvPixels * _hstUvPixels * sizeof( float/*grid*/ ) );
		else
			memset( hstModelImage, 0, (long int) _hstUvPixels * (long int) _hstUvPixels * (long int) sizeof( float/*grid*/ ) );

		// grid the clean components to make a model image.
		gridComponents(	/* pdevGrid = */ devModelImage,
				/* phstGrid = */ hstModelImage,
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

		// now extract model images from this mosaic. we will extract them into the dirty image array since we don't need these any more (they will be
		// rebuilt following degridding).
		if (_hstFileMosaic == true)
			extractFromMosaic(	/* phstImageArray = */ phstDirtyImageCache,
						/* pdevMosaic = */ devModelImage,
						/* phstMask = */ phstMask,
						/* phstPhaseCentre = */ phstPhaseCentre,
						/* phstPrimaryBeamPattern = */ _hstPrimaryBeamPattern );

		// we can free either the host or device model image, depending on whether we're using batch gridding or not.
		// NOTE: If we're not using file mosaicing then the model image will be held in hstModelImage (batching) or devModelImage (no batching).
		if (_hstFileMosaic == true || _hstImageBatches > 1)
			cudaFree( (void *) devModelImage );
		if ((_hstFileMosaic == true || _hstImageBatches == 1) && hstModelImage != NULL)
			free( (void *) hstModelImage );

		// -------------------------------------------------------------------
		//
		// S T E P   3 :   C O N S T R U C T   N E W   D I R T Y   I M A G E S
		//
		// -------------------------------------------------------------------

		for ( int image = 0; image < _numMosaicImages; image++ )
		{

			printf( "\n                performing FFT on model image.....\n" );

			// is the FFT to be done on the host or the device ?
			if (_hstImageBatches == 1)
			{

				// if we are using file mosaicing then copy the model image from where it is temporarily stored in the dirty image cache.
				if (_hstFileMosaic == true)
				{
					reserveGPUMemory(	/* pMemPtr = */ (void **) &devModelImage,
								/* pSize = */ _hstUvPixels * _hstUvPixels * sizeof( float/*grid*/ ),
								/* pTask = */ "reserving device memory for model image" );
					cudaMemcpy(	/* dst = */ devModelImage,
							/* src = */ &phstDirtyImageCache[ _hstUvPixels * _hstUvPixels * image ],
							/* count = */ _hstUvPixels * _hstUvPixels * sizeof( float/*grid*/ ),
							/* kind = */ cudaMemcpyHostToDevice );
				}

				// divide the model image by the deconvolution image.
				setThreadBlockSize2D( _hstUvPixels, _hstUvPixels );
				devDivideImages<<< _gridSize2D, _blockSize2D >>>(	/* pOne = */ devModelImage,
											/* pTwo = */ _devDeconvolutionImage,
											/* pMask = */ NULL,
											/* pSizeOne = */ _hstUvPixels,
											/* pSizeTwo = */ _hstPsfSize );

				// FFT the model image into the UV domain.
				performFFT(	/* pdevGrid = */ (cufftComplex/*grid*/ **) &devModelImage,
						/* pSize = */ _hstUvPixels,
						/* pFFTDirection = */ FORWARD,
						/* pFFTPlan = */ -1,
						/* pFFTType = */ F2C );

			}
			else
			{

				// create the model image on the device ready for gridding.
				reserveGPUMemory( (void **) &devModelImage, _hstGPURows * _hstUvPixels * sizeof( cufftComplex/*grid*/ ),
							"creating memory for the model image" );

				// if we are using file mosaicing then copy the model image from where it is temporarily stored in the dirty image cache.
				if (_hstFileMosaic == true)
				{
					hstModelImage = (float/*grid*/ *) malloc( (long int) _hstUvPixels * (long int) _hstUvPixels *
											(long int) sizeof( float/*grid*/ ) );
					memcpy(	/* dst = */ hstModelImage,
						/* src = */ &phstDirtyImageCache[ (long int) _hstUvPixels * (long int) _hstUvPixels * (long int) image ],
						/* count = */ (long int) _hstUvPixels * (long int) _hstUvPixels * (long int) sizeof( float/*grid*/ ) );
				}

				// divide the model image by the deconvolution image.
				divideImage(	/* pImageOne = */ hstModelImage,
						/* pImageTwo = */ _hstDeconvolutionImage,
						/* pSizeOne = */ _hstUvPixels,
						/* pSizeTwo = */ _hstPsfSize );

				// perform the FFT of the model image on the host.
				performFFT_host(	/* phstGrid = */ (cufftComplex/*grid*/ **) &hstModelImage,
							/* pSize = */ _hstUvPixels,
							/* pFFTDirection = */ FORWARD,
							/* pFFTType = */ F2C );

			}

			printf( "\n" );

			// count total visibilities.
			long int totalVisibilities = 0;
			for ( int stageID = 0; stageID < _hstNumberOfStages[ image ]; stageID++ )
				totalVisibilities += _hstNumVisibilities[ image ][ stageID ];

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
							/* pWhatData = */ DATA_ALL );
				}

				// create some memory to store the residual visibilities.
				_hstResidualVisibility = (cufftComplex *) malloc( _hstNumVisibilities[ image ][ stageID ] * sizeof( cufftComplex ) );

				// calculate the number of batches.
				int numberOfBatches = 1;
				if (_hstNumVisibilities[ image ][ stageID ] > _hstPreferredVisibilityBatchSize)
					numberOfBatches = (_hstNumVisibilities[ image ][ stageID ] / _hstPreferredVisibilityBatchSize) + 1;

				// here is the start of the visibility batch loop.
				int hstCurrentVisibility = 0;
				int batch = 0;
				while (hstCurrentVisibility < _hstNumVisibilities[ image ][ stageID ])
				{

					// calculate the batch size.
					int hstVisibilityBatchSize = 0;
					{
						long int nextBatchSize = _hstNumVisibilities[ image ][ stageID ] - hstCurrentVisibility;
						if (nextBatchSize > _hstPreferredVisibilityBatchSize)
							nextBatchSize = _hstPreferredVisibilityBatchSize;
						hstVisibilityBatchSize = (int) nextBatchSize;
					}

					printf( "        degridding" );
					if (_hstNumberOfStages[ image ] > 1 || numberOfBatches > 1)
						printf( " " );
					else
						printf( " visibilities\n\n" );
					if (_hstNumberOfStages[ image ] > 1)
						printf( "host batch %i of %i", stageID + 1, _hstNumberOfStages[ image ] );
					if (_hstNumberOfStages[ image ] > 1 && numberOfBatches > 1)
						printf( ", " );
					if (numberOfBatches > 1)
						printf( "gpu batch %i of %i", batch + 1, numberOfBatches );
					if (_hstNumberOfStages[ image ] > 1 || numberOfBatches > 1)
					{
						int fractionDone = (int) round( (double) visibilitiesProcessed * 30.0 / (double) totalVisibilities );
						int fractionDoing = (int) round( (double) (visibilitiesProcessed + hstVisibilityBatchSize) * 30.0 /
												(double) totalVisibilities );
						printf( " [" );
						for ( int i = 0; i < fractionDone; i++ )
							printf( "*" );
						for ( int i = 0; i < (fractionDoing - fractionDone); i++ )
							printf( "+" );
						for ( int i = 0; i < (30 - fractionDoing); i++ )
							printf( "." );
						printf( "]\n\n" );
						visibilitiesProcessed += hstVisibilityBatchSize;
					}

					// upload the visibility batch size to the device.
					err = cudaMemcpyToSymbol( _devVisibilityBatchSize, &hstVisibilityBatchSize, sizeof( hstVisibilityBatchSize ) );
					if (err != cudaSuccess)
						printf( "error copying visibility batch size to device (%s)\n", cudaGetErrorString( err ) );

					// transfer the grid positions and kernel indexes from the host to the device.
					VectorI * devGridPosition = NULL;
					int * devKernelIndex = NULL;
					reserveGPUMemory( (void **) &devGridPosition, hstVisibilityBatchSize * sizeof( VectorI ),
								"reserving device memory for grid positions" );
					reserveGPUMemory( (void **) &devKernelIndex, hstVisibilityBatchSize * sizeof( int ),
								"reserving device memory for kernel indexes" );
					moveHostToDevice( (void *) devGridPosition, (void *) &_hstGridPosition[ hstCurrentVisibility ],
								hstVisibilityBatchSize * sizeof( VectorI ), "copying grid positions to the device" );
					moveHostToDevice( (void *) devKernelIndex, (void *) &_hstKernelIndex[ hstCurrentVisibility ],
								hstVisibilityBatchSize * sizeof( int ), "copying kernel indexes to the device" );

					// variables for device memory.
					int * devDensityMap = NULL;
					cufftComplex * devModelVisibilities = NULL;
					float * devWeight = NULL;
					cufftComplex * devOriginalVisibilities = NULL;

					// reserve device memory for the density map, model visibilities, weights and original visibilities.
					reserveGPUMemory( (void **) &devDensityMap, hstVisibilityBatchSize * sizeof( int ), "declaring device memory for density map" );
					reserveGPUMemory( (void **) &devModelVisibilities, hstVisibilityBatchSize * sizeof( cufftComplex ),
								"creating device memory for model visibilities" );
					if (_hstWeighting != NONE)
						reserveGPUMemory( (void **) &devWeight, hstVisibilityBatchSize * sizeof( float ), "creating device memory for weights" );
					reserveGPUMemory( (void **) &devOriginalVisibilities, hstVisibilityBatchSize * sizeof( cufftComplex ),
								"creating memory for original visibilities" );

					// upload density map, weights, and original visibilities to the device.
					moveHostToDevice( (void *) devDensityMap, (void *) &_hstDensityMap[ hstCurrentVisibility ],
								hstVisibilityBatchSize * sizeof( int ), "copying density map to the device" );
					if (_hstWeighting != NONE)
						moveHostToDevice( (void *) devWeight, (void *) &_hstWeight[ hstCurrentVisibility ],
									hstVisibilityBatchSize * sizeof( float ), "copying weights to the device" );
					moveHostToDevice( (void *) devOriginalVisibilities, (void *) &_hstVisibility[ hstCurrentVisibility ],
								hstVisibilityBatchSize * sizeof( cufftComplex ), "copying original visibilities to the device" );

					// set the model visibilities to zero.
					zeroGPUMemory( (void *) devModelVisibilities, hstVisibilityBatchSize * sizeof( cufftComplex ),
								"clearing the model visibilities on the device" );

					cudaDeviceSynchronize();

					// degridding with w-projection and oversampling.
					gridVisibilities(	/* pdevGrid = */ (cufftComplex/*grid*/ *) devModelImage,
								/* phstGrid = */ (cufftComplex/*grid*/ *) hstModelImage,
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
								/* pVisibilitiesInKernelSet = */
									&_hstVisibilitiesInKernelSet[ image ][ stageID ][ batch * _hstKernelSets ],
								/* pGridDegrid = */ DEGRID,
								/* phstPrimaryBeamMosaicing = */ _hstPrimaryBeamMosaicingPtr[ image ],
								/* phstPrimaryBeamAProjection = */ _hstPrimaryBeamAProjectionPtr[ image ],
								/* pNumFields = */ (_hstBeamMosaic == true ? _hstNumFieldsForBeamMosaic : -1),
								/* pMosaicIndex = */ image,
								/* pSize = */ _hstUvPixels );

					// apply density map, and subtract from the real visibilities:
					{

						// define the block/thread dimensions.
						int threads = hstVisibilityBatchSize;
						int blocks;
						setThreadBlockSize1D( &threads, &blocks );
	
						// multiply all the visibilities by the value of the density map at that position.
						devApplyDensityMap<<< blocks, threads >>>( devModelVisibilities, devDensityMap );

						// subtract the model visibilities from the real visibilities to get a new set of (dirty) visibilities.
						devSubtractVisibilities<<< blocks, threads >>>(	/* pOriginalVisibility = */ devOriginalVisibilities,
												/* pModelVisibility = */ devModelVisibilities );
//						cudaMemcpy( devModelVisibilities, devOriginalVisibilities, hstVisibilityBatchSize * sizeof( cufftComplex ), cudaMemcpyDeviceToDevice );

					}

					// download model visibilities to the host.
					moveDeviceToHost( (void *) &_hstResidualVisibility[ hstCurrentVisibility ], (void *) devModelVisibilities,
								hstVisibilityBatchSize * sizeof( cufftComplex ), "copying model visibilities to the host" );

					// free memory.
					if (devModelVisibilities != NULL)
						cudaFree( (void *) devModelVisibilities );
					if (devDensityMap != NULL)
						cudaFree( (void *) devDensityMap );
					if (devWeight != NULL)
						cudaFree( (void *) devWeight );
					if (devOriginalVisibilities != NULL)
						cudaFree( (void *) devOriginalVisibilities );
					if (devGridPosition != NULL)
						cudaFree( (void *) devGridPosition );
					if (devKernelIndex != NULL)
						cudaFree( (void *) devKernelIndex );

					// move to the next set of batch of data.
					hstCurrentVisibility = hstCurrentVisibility + hstVisibilityBatchSize;
					batch = batch + 1;

				}

				// uncache the data for this mosaic.
				if (_hstCacheData == true)
				{
					cacheData(	/* pFilenamePrefix = */ pFilenamePrefix,
							/* pMosaicID = */ image,
							/* pBatchID = */ stageID,
							/* pWhatData = */ DATA_RESIDUAL_VISIBILITIES );
					freeData(	/* pWhatData = */ DATA_ALL);
				}

			}

			// free model image.
			if (hstModelImage != NULL)
				free( (void *) hstModelImage );
			hstModelImage = NULL;
			if (devModelImage != NULL)
				cudaFree( (void *) devModelImage );
			devModelImage = NULL;

			// is the FFT to be done on the host or the device ?;
			cufftComplex/*grid*/ * hstDirtyImageGrid = NULL;
			if (_hstImageBatches > 1)
			{

				// create memory for the model and dirty image grids.
				hstDirtyImageGrid = (cufftComplex/*grid*/ *) malloc( (long int) _hstUvPixels * (long int) _hstUvPixels *
											(long int) sizeof( cufftComplex/*grid*/ ) );

				// clear the image grid.
				memset( (void *) hstDirtyImageGrid, 0, (long int) _hstUvPixels * (long int) _hstUvPixels * (long int) sizeof( cufftComplex/*grid*/ ) );

			}

			// create memory for the dirty image grid, and clear it.
			cufftComplex/*grid*/ * devDirtyImageGrid = NULL;
			reserveGPUMemory( (void **) &devDirtyImageGrid, _hstGPURows * _hstUvPixels * sizeof( cufftComplex/*grid*/ ),
						"reserving device memory for dirty image grid (cleaning)" );
			zeroGPUMemory( devDirtyImageGrid, _hstGPURows * _hstUvPixels * sizeof( cufftComplex/*grid*/ ),
						"zeroing the dirty image grid on the device" );

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
										DATA_RESIDUAL_VISIBILITIES );

				// calculate the number of batches.
				int numberOfBatches = 1;
				if (_hstNumVisibilities[ image ][ stageID ] > _hstPreferredVisibilityBatchSize)
					numberOfBatches = (_hstNumVisibilities[ image ][ stageID ] / _hstPreferredVisibilityBatchSize) + 1;

				// here is the start of the visibility batch loop.
				int hstCurrentVisibility = 0;
				int batch = 0;
				while (hstCurrentVisibility < _hstNumVisibilities[ image ][ stageID ])
				{

					// calculate the batch size.
					int hstVisibilityBatchSize = 0;
					{
						long int nextBatchSize = _hstNumVisibilities[ image ][ stageID ] - hstCurrentVisibility;
						if (nextBatchSize > _hstPreferredVisibilityBatchSize)
							nextBatchSize = _hstPreferredVisibilityBatchSize;
						hstVisibilityBatchSize = (int) nextBatchSize;
					}

					printf( "        gridding " );
					if (_hstNumberOfStages[ image ] == 1 && numberOfBatches == 1)
						printf( "visibilities\n\n" );
					if (_hstNumberOfStages[ image ] > 1)
						printf( "host batch %i of %i", stageID + 1, _hstNumberOfStages[ image ] );
					if (_hstNumberOfStages[ image ] > 1 && numberOfBatches > 1)
						printf( ", " );
					if (numberOfBatches > 1)
						printf( "gpu batch %i of %i", batch + 1, numberOfBatches );
					if (_hstNumberOfStages[ image ] > 1 || numberOfBatches > 1)
					{
						int fractionDone = (int) round( (double) visibilitiesProcessed * 30.0 / (double) totalVisibilities );
						int fractionDoing = (int) round( (double) (visibilitiesProcessed + hstVisibilityBatchSize) * 30.0 /
												(double) totalVisibilities );
						printf( " [" );
						for ( int i = 0; i < fractionDone; i++ )
							printf( "*" );
						for ( int i = 0; i < (fractionDoing - fractionDone); i++ )
							printf( "+" );
						for ( int i = 0; i < (30 - fractionDoing); i++ )
							printf( "." );
						printf( "]\n\n" );
						visibilitiesProcessed += hstVisibilityBatchSize;
					}

					// upload the visibility batch size to the device.
					err = cudaMemcpyToSymbol( _devVisibilityBatchSize, &hstVisibilityBatchSize, sizeof( hstVisibilityBatchSize ) );
					if (err != cudaSuccess)
						printf( "error copying visibility batch size to device (%s)\n", cudaGetErrorString( err ) );

					// transfer the grid positions and kernel indexes from the host to the device.
					VectorI * devGridPosition = NULL;
					int * devKernelIndex = NULL;
					reserveGPUMemory( (void **) &devGridPosition, hstVisibilityBatchSize * sizeof( VectorI ),
								"reserving device memory for grid positions" );
					reserveGPUMemory( (void **) &devKernelIndex, hstVisibilityBatchSize * sizeof( int ),
								"reserving device memory for kernel indexes" );
					moveHostToDevice( (void *) devGridPosition, (void *) &_hstGridPosition[ hstCurrentVisibility ],
								hstVisibilityBatchSize * sizeof( VectorI ), "copying grid positions to the device" );
					moveHostToDevice( (void *) devKernelIndex, (void *) &_hstKernelIndex[ hstCurrentVisibility ],
								hstVisibilityBatchSize * sizeof( int ), "copying kernel indexes to the device" );

					// variables for device memory.
					int * devDensityMap = NULL;
					cufftComplex * devModelVisibilities = NULL;
					float * devWeight = NULL;

					// reserve device memory for the density map, model visibilities, weights and original visibilities.
					reserveGPUMemory( (void **) &devDensityMap, hstVisibilityBatchSize * sizeof( int ), "declaring device memory for density map" );
					reserveGPUMemory( (void **) &devModelVisibilities, hstVisibilityBatchSize * sizeof( cufftComplex ),
								"creating device memory for model visibilities" );
					if (_hstWeighting != NONE)
						reserveGPUMemory( (void **) &devWeight, hstVisibilityBatchSize * sizeof( float ), "creating device memory for weights" );

					// upload density map, weights, and original visibilities to the device.
					moveHostToDevice( (void *) devDensityMap, (void *) &_hstDensityMap[ hstCurrentVisibility ],
								hstVisibilityBatchSize * sizeof( int ), "copying density map to the device" );
					if (_hstWeighting != NONE)
						moveHostToDevice( (void *) devWeight, (void *) &_hstWeight[ hstCurrentVisibility ],
									hstVisibilityBatchSize * sizeof( float ), "copying weights to the device" );
					moveHostToDevice( (void *) devModelVisibilities, (void *) &_hstResidualVisibility[ hstCurrentVisibility ],
								hstVisibilityBatchSize * sizeof( cufftComplex ), "copying original visibilities to the device" );

					cudaDeviceSynchronize();

					// grid the new set of dirty visibilities.
					gridVisibilities(	/* pdevGrid = */ devDirtyImageGrid,
								/* phstGrid = */ hstDirtyImageGrid,
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
								/* pVisibilitiesInKernelSet = */
									&_hstVisibilitiesInKernelSet[ image ][ stageID ][ batch * _hstKernelSets ],
								/* pGridDegrid = */ GRID,
								/* phstPrimaryBeamMosaicing = */ _hstPrimaryBeamMosaicingPtr[ image ],
								/* phstPrimaryBeamAProjection = */ _hstPrimaryBeamAProjectionPtr[ image ],
								/* pNumFields = */ (_hstBeamMosaic == true ? _hstNumFieldsForBeamMosaic : -1),
								/* pMosaicIndex = */ image,
								/* pSize = */ _hstUvPixels );

					// free memory.
					if (devModelVisibilities != NULL)
						cudaFree( (void *) devModelVisibilities );
					if (devDensityMap != NULL)
						cudaFree( (void *) devDensityMap );
					if (devWeight != NULL)
						cudaFree( (void *) devWeight );
					if (devGridPosition != NULL)
						cudaFree( (void *) devGridPosition );
					if (devKernelIndex != NULL)
						cudaFree( (void *) devKernelIndex );

					// move to the next set of batch of data.
					hstCurrentVisibility = hstCurrentVisibility + hstVisibilityBatchSize;
					batch = batch + 1;

				}

				// free the data.
				if (_hstCacheData == true)
					freeData( /* pWhatData = */ DATA_GRID_POSITIONS | DATA_KERNEL_INDEXES | DATA_DENSITIES | DATA_WEIGHTS |
										DATA_RESIDUAL_VISIBILITIES );

			}

			// free residual visibilities if they are still there.
			if (_hstResidualVisibility != NULL)
			{
				free( (void *) _hstResidualVisibility );
				_hstResidualVisibility = NULL;
			}

			double normalisation = 1.0;
			
			// normalise by the number of visibilities, but only if we're not beam mosaicing. If we're using beam mosaicing then
			// the normalisation will have been done using the normalisation pattern.
			if (_hstBeamMosaic == true)
				normalisation *= (double) _griddedVisibilitiesForBeamMosaic;
			else
				normalisation *= (double) _hstGriddedVisibilities[ image ];

			if (_hstWeighting != NONE)
				normalisation *= _hstAverageWeight[ image ];

			printf( "                performing FFT on gridded visibilities.....\n" );

			// is the FFT to be done on the host or the device ?
			if (_hstImageBatches == 1)
			{

				// FFT the dirty visibilities into the image domain.
				performFFT(	/* pdevGrid = */ &devDirtyImageGrid,
						/* pSize = */ _hstUvPixels,
						/* pFFTDirection = */ INVERSE,
						/* pFFTPlan = */ -1,
						/* pFFTType = */ C2F );

				// recast the dirty image as a float/*grid*/.
				float/*grid*/ * devDirtyImage = (float/*grid*/ *) devDirtyImageGrid;

				// define the block/thread dimensions.
				setThreadBlockSize2D( _hstUvPixels, _hstUvPixels );

				// divide the dirty image by the deconvolution image.
				devDivideImages<<< _gridSize2D, _blockSize2D >>>(	/* pOne = */ devDirtyImage,
											/* pTwo = */ _devDeconvolutionImage,
											/* pMask = */ NULL,
											/* pSizeOne = */ _hstUvPixels,
											/* pSizeTwo = */ _hstPsfSize );

				// divide the dirty image by the normalisation pattern (if we are beam mosaicing).
				if (devNormalisationPattern != NULL)
					devDivideImages<<< _gridSize2D, _blockSize2D >>>(	/* pOne = */ devDirtyImage,
												/* pTwo = */ devNormalisationPattern,
												/* pMask = */ NULL,
												/* pSizeOne = */ _hstUvPixels,
												/* pSizeTwo = */ _hstBeamSize );

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
				*phstDirtyImage = (float/*grid*/ *) malloc( _hstUvPixels * _hstUvPixels * sizeof( float/*grid*/ ) );

				// copy the residual image into the dirty image cache, or the dirty image if we're not mosaicing.
				if (_hstFileMosaic == true)
					moveDeviceToHost( (void *) &phstDirtyImageCache[ _hstUvPixels * _hstUvPixels * image ], (void *) devDirtyImage,
								_hstUvPixels * _hstUvPixels * sizeof( float/*grid*/ ), "copying residual image to host" );
				else
					moveDeviceToHost( (void *) *phstDirtyImage, (void *) devDirtyImage, _hstUvPixels * _hstUvPixels * sizeof( float/*grid*/ ),
								"copying residual image to host" );

			}
			else
			{

				// perform FFT on the host.
				performFFT_host(	/* phstGrid = */ (cufftComplex/*grid*/ **) &hstDirtyImageGrid,
							/* pSize = */ _hstUvPixels,
							/* pFFTDirection = */ INVERSE,
							/* pFFTType = */ C2F );

				// recast the dirty image as a float/*grid*/ array.
				*phstDirtyImage = (float/*grid*/ *) hstDirtyImageGrid;
				hstDirtyImageGrid = NULL;

				// normalise the image.
				for ( long int i = 0; i < (long int) _hstUvPixels * (long int) _hstUvPixels; i++ )
					(*phstDirtyImage)[ i ] /= normalisation;

				// divide the model image by the deconvolution image.
				divideImage(	/* pImageOne = */ *phstDirtyImage,
						/* pImageTwo = */ _hstDeconvolutionImage,
						/* pSizeOne = */ _hstUvPixels,
						/* pSizeTwo = */ _hstPsfSize );

				// for beam mosaicing, divide the dirty image by the normalisation pattern.
				if (_hstBeamMosaic == true)
					divideImage(	/* pImageOne = */ *phstDirtyImage,
							/* pImageTwo = */ _hstNormalisationPattern,
							/* pSizeOne = */ _hstUvPixels,
							/* pSizeTwo = */ _hstBeamSize );

			}

			printf( "\n" );

			// free memory.
			if (devDirtyImageGrid != NULL)
				cudaFree( (void *) devDirtyImageGrid );
			devDirtyImageGrid = NULL;

		}

		// increment major cycle.
		majorCycle++;

	}

	// free memory.
	if (devPrimaryBeamPattern != NULL)
		cudaFree( (void *) devPrimaryBeamPattern );
	if (devNormalisationPattern != NULL)
		cudaFree( (void *) devNormalisationPattern );

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
	float/*grid*/ * devCleanImage = NULL;
	reserveGPUMemory( (void **) &devCleanImage, _hstGPURows * _hstUvPixels * sizeof( float/*grid*/ ), "reserving device memory for the clean image" );

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
	float/*grid*/ * devKernel = NULL;
	reserveGPUMemory( (void **) &devKernel, cleanBeamSize * cleanBeamSize * sizeof( float/*grid*/ ),
				"reserving device memory for the clean component gridding kernel" );

	// cut out the centre portion of the kernel.
	for ( int i = 0; i < cleanBeamSize; i++ )
		cudaMemcpy(	&devKernel[ i * cleanBeamSize ],
				&pdevCleanBeam[ ((i + _hstPsfY - _hstCleanBeamSize) * _hstPsfSize) + _hstPsfX - _hstCleanBeamSize ],
				cleanBeamSize * sizeof( float/*grid*/ ),
				cudaMemcpyDeviceToDevice );

//memset( *phstDirtyImage, 0, (long int) _hstUvPixels * (long int) _hstUvPixels * (long int) sizeof( float/*grid*/ ) ); // cjs-mod

	// if we're not using gridding batches then copy the dirty image to the device so that we include our residuals.
	if (_hstImageBatches == 1)
		moveHostToDevice( (void *) devCleanImage, (void *) *phstDirtyImage, _hstUvPixels * _hstUvPixels * sizeof( float/*grid*/ ),
					"moving residual image to device" );

	// grid the clean components to make a clean image.
	gridComponents(	/* pdevGrid = */ devCleanImage,
			/* phstGrid = */ (_hstImageBatches > 1 ? *phstDirtyImage : NULL),
			/* pdevComponentValue = */ devComponentValue,
			/* phstSupportSize = */ _hstCleanBeamSize,
			/* pdevKernel = */ devKernel,
			/* pdevGridPositions = */ devComponentPos,
			/* pComponents = */ numComponents,
			/* pSize = */ _hstUvPixels );

	// if we're not batch gridding then create host memory for the clean image and get the image from the device.
	if (_hstImageBatches == 1)
		moveDeviceToHost( (void *) *phstDirtyImage, (void *) devCleanImage, _hstUvPixels * _hstUvPixels * sizeof( float/*grid*/ ),
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

double getErrorBetweenImageAndGaussianFit( float/*grid*/ * pdevImage, double * pdevError, int pSizeOfFittingRegion, double pX, double pY,
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

void generateDirtyBeam( cufftComplex/*grid*/ ** pdevDirtyBeam, cufftComplex/*grid*/ ** phstDirtyBeam, char * pFilename )
{

	printf( "        performing fft on psf grid.....\n" );

	// we either generate the dirty beam on the device, or if it's too big, then on the host.
	if (_hstImageBatches == 1)
	{
		
		// FFT the uv coverage to get the psf.
		performFFT(	/* pdevGrid = */ pdevDirtyBeam,
				/* pSize = */ _hstUvPixels,
				/* pFFTDirection = */ INVERSE,
				/* pFFTPlan = */ -1,
				/* pFFTType = */ C2F/*grid*/ );
		
		// define the block/thread dimensions.
		setThreadBlockSize2D( _hstUvPixels, _hstUvPixels );
	
		// divide the dirty beam by the deconvolution image.
		devDivideImages<<< _gridSize2D, _blockSize2D >>>(	/* pOne = */ (float/*grid*/ *) *pdevDirtyBeam,
									/* pTwo = */ _devDeconvolutionImage,
									/* pMask = */ NULL,
									/* pSizeOne = */ _hstUvPixels,
									/* pSizeTwo = */ _hstPsfSize );

		// chop out the central portion of the image.
		if (_hstPsfSize < _hstUvPixels)
		{

			float/*grid*/ * devtmpDirtyBeam = NULL;
			reserveGPUMemory( (void **) &devtmpDirtyBeam, _hstPsfSize * _hstPsfSize * sizeof( float/*grid*/ ), "reserving device memory for temporary psf" );

			// define the block/thread dimensions.
			setThreadBlockSize2D( _hstPsfSize, _hstPsfSize );

			// chop out the centre of the psf.
			devCopyImage<<< _gridSize2D, _blockSize2D >>>(	/* pNewImage = */ devtmpDirtyBeam,
									/* pOldImage = */ (float/*grid*/ *) *pdevDirtyBeam,
									/* pNewSize = */ _hstPsfSize,
									/* pOldSize = */ _hstUvPixels,
									/* pScale = */ 1.0,
									/* pThreadOffset = */ 0 );

			// reassign the dirty beam to the new memory area.
			cudaFree( (void *) *pdevDirtyBeam );
			*pdevDirtyBeam = (cufftComplex/*grid*/ *) devtmpDirtyBeam;

		}
			
	
		// get maximum pixel value.
		double * devMaxValue;
		reserveGPUMemory( (void **) &devMaxValue, MAX_PIXEL_DATA_AREA_SIZE * sizeof( double ), "declaring device memory for psf max pixel value" );
		
		// get the maximum complex value from this image.
		getMaxValue(	/* pdevImage = */ (float/*grid*/ *) *pdevDirtyBeam,
				/* pdevMaxValue = */ devMaxValue,
				/* pWidth = */ _hstPsfSize,
				/* pHeight = */ _hstPsfSize,
				/* pdevMask = */ NULL );
	
		cudaDeviceSynchronize();
	
		// set a suitable thread and block size.
		int threads = _hstPsfSize * _hstPsfSize;
		int blocks;
		setThreadBlockSize1D( &threads, &blocks );
		
		// normalise the psf so that the maximum value is 1.
		devNormalise<<< blocks, threads >>>( (float/*grid*/ *) *pdevDirtyBeam, devMaxValue, _hstPsfSize * _hstPsfSize );

		cudaDeviceSynchronize();

		// free memory.
		if (devMaxValue != NULL)
			cudaFree( devMaxValue );

		// create the dirty beam on the host, and copy to the host.
		(*phstDirtyBeam) = (cufftComplex/*grid*/ *) malloc( _hstPsfSize * _hstPsfSize * sizeof( float/*grid*/ ) );
		moveDeviceToHost( (void *) *phstDirtyBeam, (void *) *pdevDirtyBeam, _hstPsfSize * _hstPsfSize * sizeof( float/*grid*/ ),
					"copying dirty beam from device" );

	}
	else
	{

		// free the dirty beam on the device.
		cudaFree( (void *) *pdevDirtyBeam );
		*pdevDirtyBeam = NULL;

		// perform FFT on the host.
		performFFT_host(	/* phstGrid = */ (cufftComplex/*grid*/ **) phstDirtyBeam,
					/* pSize = */ _hstUvPixels,
					/* pFFTDirection = */ INVERSE,
					/* pFFTType = */ C2F );

		// re-cast the dirty beam from a complex to a double.
		float/*grid*/ * hstDirtyBeam = (float/*grid*/ *) *phstDirtyBeam;

		// divide each pixel by the deconvolution image.
		divideImage(	/* pImageOne = */ hstDirtyBeam,
				/* pImageTwo = */ _hstDeconvolutionImage,
				/* pSizeOne = */ _hstUvPixels,
				/* pSizeTwo = */ _hstPsfSize );

		// copy the central portion of the image to the start of the image. we are chopping away most of our psf so it will fit on the gpu.
		if (_hstPsfSize < _hstUvPixels)
			for ( int row = 0; row < _hstPsfSize; row++ )
				memmove(	/* destination = */ &hstDirtyBeam[ row * _hstPsfSize ],
						/* source = */ &hstDirtyBeam[ ((long int) (((_hstUvPixels - _hstPsfSize) / 2) + row) * (long int) _hstUvPixels) +
										 ((_hstUvPixels - _hstPsfSize) / 2) ],
						/* num = */ _hstPsfSize * sizeof( float/*grid*/ ) );

		// we don't bother reallocating the dirty beam to the smaller size - it will get released anyway as soon as this function finishes.

		// get the maximum complex value at the same time.
		double maxValue = 0;
		for ( long int index = 0; index < _hstPsfSize * _hstPsfSize; index++ )
			if (hstDirtyBeam[ index ] > maxValue)
				maxValue = hstDirtyBeam[ index ];

		// normalise the psf so that the maximum value is 1.
		if (maxValue > 0.0)
			for ( int index = 0; index < _hstPsfSize * _hstPsfSize; index++ )
				hstDirtyBeam[ index ] /= maxValue;

		// move the psf to the device.
		reserveGPUMemory( (void **) pdevDirtyBeam, _hstPsfSize * _hstPsfSize * sizeof( float/*grid*/ ), "reserving device memory for dirty beam" );
		moveHostToDevice( (void *) *pdevDirtyBeam, (void *) hstDirtyBeam, _hstPsfSize * _hstPsfSize * sizeof( float/*grid*/ ),
					"copying dirty beam to the device" );

	}

	printf( "\n" );
	
	// save the dirty beam.
	_hstCasacoreInterface.WriteCasaImage(	/* pFilename = */ pFilename,
						/* pWidth = */ _hstPsfSize,
						/* pHeight = */ _hstPsfSize,
						/* pRA = */ _hstOutputRA,
						/* pDec = */ _hstOutputDEC,
						/* pPixelSize = */ _hstCellSize,
						/* pImage = */ (float/*grid*/ *) *phstDirtyBeam,
						/* pFrequency = */ CONST_C / _hstAverageWavelength[ 0 ],
						/* pMask = */ NULL );

} // generateDirtyBeam

//
//	generateCleanBeam()
//
//	CJS: 05/11/2015
//
//	Generate the clean beam by fitting elliptical Gaussian to the dirty beam.
//

bool generateCleanBeam( float/*grid*/ * pdevCleanBeam, float/*grid*/ * pdevDirtyBeam, char * pFilename )
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
	
	cudaDeviceSynchronize();

	// create a new memory area to hold the maximum pixel value.
	double * devMaxValue;
	reserveGPUMemory( (void **) &devMaxValue, MAX_PIXEL_DATA_AREA_SIZE * sizeof( double ), "declaring device memory for kernel max pixel value" );

	// get the peak value from the kernel.
	getMaxValue(	/* pdevImage = */ pdevDirtyBeam,
			/* pdevMaxValue = */ devMaxValue,
			/* pWidth = */ _hstPsfSize,
			/* pHeight = */ _hstPsfSize,
			/* pdevMask = */ NULL );

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
	float/*grid*/ * hstCleanBeam = (float/*grid*/ *) malloc( (long int) _hstPsfSize * (long int) _hstPsfSize * sizeof( float/*grid*/ ) );

	// copy the clean beam to the host.
	ok = ok && moveDeviceToHost( (void *) hstCleanBeam, (void *) pdevCleanBeam, _hstPsfSize * _hstPsfSize * sizeof( float/*grid*/ ),
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
						currentVis = divideComplex( /* pOne = */ currentVis, /* pTwo = */ currentWeight );
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
//				else
//					_hstFieldIDArray[ newIndex ] = 0;

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
			currentVis = divideComplex( /* pOne = */ currentVis, /* pTwo = */ currentWeight );
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
	performFFT(	/* pdevGrid = */ (cufftComplex/*grid*/ **) &devBeam,
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

	// determine how much free memory is available.
//	size_t freeMem = 0, totalMem = 0;
//	cudaError_t err = cudaMemGetInfo( &freeMem, &totalMem );
//	if (err == cudaSuccess)
//		printf( "Memory free: %li MB from %li MB\n", freeMem / (1024 * 1024), totalMem / (1024 * 1024) );

	// copy the primary beam into a temporary work location.
	cudaMemcpy( pdevInBeam, phstPrimaryBeamIn, _hstBeamSize * _hstBeamSize * sizeof( float ), cudaMemcpyHostToDevice );

	// clear the output image.
	zeroGPUMemory( (void *) pdevOutBeam, _hstBeamSize * _hstBeamSize * sizeof( float ),
				"zeroing the reprojected output image on the device" );

	cudaDeviceSynchronize();

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
						/* pProjectionDirection = */ Reprojection::OUTPUT_TO_INPUT,
						/* pAProjection = */ false,
						/* pVerbose = */ pVerbose );

	cudaDeviceSynchronize();

	// store the beam on the host.
	cudaMemcpy( phstPrimaryBeamOut, pdevOutBeam, _hstBeamSize * _hstBeamSize * sizeof( float ), cudaMemcpyDeviceToHost );

	cudaDeviceSynchronize();

	// cut off any value less than 0.1% to zero.
	for ( int i = 0; i < _hstBeamSize * _hstBeamSize; i++ )
		if (abs( phstPrimaryBeamOut[ i ] ) < 0.001)
			phstPrimaryBeamOut[ i ] = 0.0;

	// save the reprojected beam.
//	char beamFilename[100];
//	sprintf( beamFilename, "beam-%i.image", pBeam );
//	_hstCasacoreInterface.WriteCasaImage( beamFilename, _hstBeamSize, _hstBeamSize, _hstOutputRA, _hstOutputDEC,
//						_hstCellSize * (double) _hstUvPixels / (double) _hstBeamSize, phstPrimaryBeamOut,
//						CONST_C / _hstAverageWavelength[ 0 ], NULL );

	// save the reprojected beam.
//	char beamFilename2[100];
//	sprintf( beamFilename2, "beam-in-%i.image", pBeam );
//	_hstCasacoreInterface.WriteCasaImage( beamFilename2, _hstBeamSize, _hstBeamSize, _hstOutputRA, _hstOutputDEC,
//						_hstBeamCellSize, phstPrimaryBeamIn, CONST_C / _hstAverageWavelength[ 0 ], NULL );

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

void setSpwAndChannelFlags( int pNumSpws, int * phstNumChannels, bool *** phstSpwChannelFlag )
{

	// create memory for the flags and set it all to true.
	*phstSpwChannelFlag = (bool **) malloc( pNumSpws * sizeof( bool * ) );
	for ( int spw = 0; spw < pNumSpws; spw++ )
	{
		(*phstSpwChannelFlag)[ spw ] = (bool *) malloc( phstNumChannels[ spw ] * sizeof( bool ) );
		for ( int channel = 0; channel < phstNumChannels[ spw ]; channel++ )
			if (_hstSpwRestriction[ 0 ] == '\0')
				(*phstSpwChannelFlag)[ spw ][ channel ] = false;
			else
				(*phstSpwChannelFlag)[ spw ][ channel ] = true;
	}

	// initialise an empty string.
	char singleSpw[ 1024 ];
	int posCharIn = 0, numCharOut = 0;

	// loop over all the characters in the spw array.
	while (numCharOut < strlen( _hstSpwRestriction ))
	{

		// check for a spw separator (i.e. comma).
		if (_hstSpwRestriction[ posCharIn ] == ',')
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
		else if (_hstSpwRestriction[ posCharIn ] != ' ')
		{
			singleSpw[ numCharOut ] = _hstSpwRestriction[ posCharIn ];
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

//printf( "right: %f, left: %f, top: %f, bottom: %f\n", right, left, top, bottom );

	// get the pixel in the middle of this region.
	double x = ((right + left) / 2.0);
	double y = ((top + bottom) / 2.0);

//printf( "centre: <%f,%f>\n", x, y );

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

//printf( "ra, dec: <%f,%f>\n\n", *pPhaseRA, *pPhaseDEC );

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

double * getPolarisationMultiplier( char * pMeasurementSetFilename, int * pNumPolarisations, int * pNumPolarisationConfigurations )
{

	// return value.	
	double * hstMultiplier = NULL;

	// get a list of polarisations.
	int * hstPolarisation = NULL;
	_hstCasacoreInterface.GetPolarisations(	/* pMeasurementSet = */ pMeasurementSetFilename,
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
				if (polarisationPtr[ i ] == RL_CONST && (whichStokes == STOKES_Q || whichStokes == STOKES_U))
					multiplierPtr[ i ] = 0.5;
				if (polarisationPtr[ i ] == LR_CONST && (whichStokes == STOKES_Q))
					multiplierPtr[ i ] = 0.5;
				if (polarisationPtr[ i ] == LR_CONST && (whichStokes == STOKES_U))
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

void calculateVisibilitiesPerKernelSet( int pNumVisibilities, VectorI * phstGridPosition, int * phstVisibilitiesInKernelSet )
{

	// calculate the number of visibilities per kernel set.
	if (_hstWProjection == true || _hstAProjection == true)
	{

		// calculate how many visibilities we have in each kernel set (A plane and W plane combination).
		int kernelSet = 0;
		int firstVisibility = 0;
		for ( int i = 0; i < pNumVisibilities; i++ )
			while (phstGridPosition[ i ].w > kernelSet)
			{
				phstVisibilitiesInKernelSet[ kernelSet ] = i - firstVisibility;
				kernelSet = kernelSet + 1;
				firstVisibility = i;
			}

		// update any remaining planes.
		phstVisibilitiesInKernelSet[ kernelSet ] = pNumVisibilities - firstVisibility;
		while (kernelSet < ((_hstAPlanes * _hstWPlanes) - 1))
		{
			kernelSet = kernelSet + 1;
			phstVisibilitiesInKernelSet[ kernelSet ] = 0;
		}

	}
	else
		phstVisibilitiesInKernelSet[ 0 ] = pNumVisibilities;

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

	int numRecords = _hstNumVisibilities[ pMosaicID ][ pStageID_one ] + _hstNumVisibilities[ pMosaicID ][ pStageID_two ];

	// build filename.
	char filenameOne[ 255 ], filenameTwo[ 255 ];
	if (_hstCacheLocation[0] != '\0')
	{
		sprintf( filenameOne, "%s%s-%02i-%i-cache.dat", _hstCacheLocation, pFilenamePrefix, pMosaicID, pStageID_one );
		sprintf( filenameTwo, "%s%s-%02i-%i-cache.dat", _hstCacheLocation, pFilenamePrefix, pMosaicID, pStageID_two );
	}
	else
	{
		sprintf( filenameOne, "%s-%02i-%i-cache.dat", pFilenamePrefix, pMosaicID, pStageID_one );
		sprintf( filenameTwo, "%s-%02i-%i-cache.dat", pFilenamePrefix, pMosaicID, pStageID_two );
	}

	// initialise the arrays to accommodate the records from stage one and two, and load the data from stage two.
	if (pLoadAllData == true)
	{

		// redimension arrays.
		if ((pWhatData & DATA_VISIBILITIES) == DATA_VISIBILITIES)
			_hstVisibility = (cufftComplex *) malloc( numRecords * sizeof( cufftComplex ) );
		if ((pWhatData & DATA_GRID_POSITIONS) == DATA_GRID_POSITIONS)
			_hstGridPosition = (VectorI *) malloc( numRecords * sizeof( VectorI ) );
		if ((pWhatData & DATA_KERNEL_INDEXES) == DATA_KERNEL_INDEXES)
			_hstKernelIndex = (int *) malloc( numRecords * sizeof( int ) );
		if ((pWhatData & DATA_DENSITIES) == DATA_DENSITIES)
			_hstDensityMap = (int *) malloc( numRecords * sizeof( int ) );
		if ((pWhatData & DATA_WEIGHTS) == DATA_WEIGHTS)
			_hstWeight = (float *) malloc( numRecords * sizeof( float ) );
		if ((pWhatData & DATA_RESIDUAL_VISIBILITIES) == DATA_RESIDUAL_VISIBILITIES)
			_hstResidualVisibility = (cufftComplex *) malloc( numRecords * sizeof( cufftComplex ) );

		// load data from stage two.
		FILE * frTwo = fopen( filenameTwo, "rb" );

		if ((pWhatData & DATA_VISIBILITIES) == DATA_VISIBILITIES)
			fread( (void *) _hstVisibility, sizeof( cufftComplex ), _hstNumVisibilities[ pMosaicID ][ pStageID_two ], frTwo );
		else
			fseek( frTwo, _hstNumVisibilities[ pMosaicID ][ pStageID_two ] * sizeof( cufftComplex ), SEEK_CUR );
		if ((pWhatData & DATA_GRID_POSITIONS) == DATA_GRID_POSITIONS)
			fread( (void *) _hstGridPosition, sizeof( VectorI ), _hstNumVisibilities[ pMosaicID ][ pStageID_two ], frTwo );
		else
			fseek( frTwo, _hstNumVisibilities[ pMosaicID ][ pStageID_two ] * sizeof( VectorI ), SEEK_CUR );
		if ((pWhatData & DATA_KERNEL_INDEXES) == DATA_KERNEL_INDEXES)
			fread( (void *) _hstKernelIndex, sizeof( int ), _hstNumVisibilities[ pMosaicID ][ pStageID_two ], frTwo );
		else
			fseek( frTwo, _hstNumVisibilities[ pMosaicID ][ pStageID_two ] * sizeof( int ), SEEK_CUR );
		if ((pWhatData & DATA_DENSITIES) == DATA_DENSITIES)
			fread( (void *) _hstDensityMap, sizeof( int ), _hstNumVisibilities[ pMosaicID ][ pStageID_two ], frTwo );
		else
			fseek( frTwo, _hstNumVisibilities[ pMosaicID ][ pStageID_two ] * sizeof( int ), SEEK_CUR );
		if ((pWhatData & DATA_WEIGHTS) == DATA_WEIGHTS)
			fread( (void *) _hstWeight, sizeof( float ), _hstNumVisibilities[ pMosaicID ][ pStageID_two ], frTwo );
		else
			fseek( frTwo, _hstNumVisibilities[ pMosaicID ][ pStageID_two ] * sizeof( float ), SEEK_CUR );
		if ((pWhatData & DATA_RESIDUAL_VISIBILITIES) == DATA_RESIDUAL_VISIBILITIES)
			fread( (void *) _hstResidualVisibility, sizeof( cufftComplex ), _hstNumVisibilities[ pMosaicID ][ pStageID_two ], frTwo );

		// close the file
		fclose( frTwo );

	}

	// redimension the arrays to accommodate the records from stage one.
	if (pLoadAllData == false)
	{
		if ((pWhatData & DATA_VISIBILITIES) == DATA_VISIBILITIES)
			_hstVisibility = (cufftComplex *) realloc( _hstVisibility, numRecords * sizeof( cufftComplex ) );
		if ((pWhatData & DATA_GRID_POSITIONS) == DATA_GRID_POSITIONS)
			_hstGridPosition = (VectorI *) realloc( _hstGridPosition, numRecords * sizeof( VectorI ) );
		if ((pWhatData & DATA_KERNEL_INDEXES) == DATA_KERNEL_INDEXES)
			_hstKernelIndex = (int *) realloc( _hstKernelIndex, numRecords * sizeof( int ) );
		if ((pWhatData & DATA_DENSITIES) == DATA_DENSITIES)
			_hstDensityMap = (int *) realloc( _hstDensityMap, numRecords * sizeof( int ) );
		if ((pWhatData & DATA_WEIGHTS) == DATA_WEIGHTS)
			_hstWeight = (float *) realloc( _hstWeight, numRecords * sizeof( float ) );
		if ((pWhatData & DATA_RESIDUAL_VISIBILITIES) == DATA_RESIDUAL_VISIBILITIES)
			_hstResidualVisibility = (cufftComplex *) realloc( _hstResidualVisibility, numRecords * sizeof( cufftComplex ) );
	}

	// load the records from stage one into the end of the arrays.
	FILE * frOne = fopen( filenameOne, "rb" );

	if ((pWhatData & DATA_VISIBILITIES) == DATA_VISIBILITIES)
		fread( (void *) &_hstVisibility[ _hstNumVisibilities[ pMosaicID ][ pStageID_two ] ], sizeof( cufftComplex ),
					_hstNumVisibilities[ pMosaicID ][ pStageID_one ], frOne );
	else
		fseek( frOne, _hstNumVisibilities[ pMosaicID ][ pStageID_one ] * sizeof( cufftComplex ), SEEK_CUR );
	if ((pWhatData & DATA_GRID_POSITIONS) == DATA_GRID_POSITIONS)
		fread( (void *) &_hstGridPosition[ _hstNumVisibilities[ pMosaicID ][ pStageID_two ] ], sizeof( VectorI ),
					_hstNumVisibilities[ pMosaicID ][ pStageID_one ], frOne );
	else
		fseek( frOne, _hstNumVisibilities[ pMosaicID ][ pStageID_one ] * sizeof( VectorI ), SEEK_CUR );
	if ((pWhatData & DATA_KERNEL_INDEXES) == DATA_KERNEL_INDEXES)
		fread( (void *) &_hstKernelIndex[ _hstNumVisibilities[ pMosaicID ][ pStageID_two ] ], sizeof( int ),
					_hstNumVisibilities[ pMosaicID ][ pStageID_one ], frOne );
	else
		fseek( frOne, _hstNumVisibilities[ pMosaicID ][ pStageID_one ] * sizeof( int ), SEEK_CUR );
	if ((pWhatData & DATA_DENSITIES) == DATA_DENSITIES)
		fread( (void *) &_hstDensityMap[ _hstNumVisibilities[ pMosaicID ][ pStageID_two ] ], sizeof( int ),
					_hstNumVisibilities[ pMosaicID ][ pStageID_one ], frOne );
	else
		fseek( frOne, _hstNumVisibilities[ pMosaicID ][ pStageID_one ] * sizeof( int ), SEEK_CUR );
	if ((pWhatData & DATA_WEIGHTS) == DATA_WEIGHTS)
		fread( (void *) &_hstWeight[ _hstNumVisibilities[ pMosaicID ][ pStageID_two ] ], sizeof( float ),
					_hstNumVisibilities[ pMosaicID ][ pStageID_one ], frOne );
	else
		fseek( frOne, _hstNumVisibilities[ pMosaicID ][ pStageID_one ] * sizeof( float ), SEEK_CUR );
	if ((pWhatData & DATA_RESIDUAL_VISIBILITIES) == DATA_RESIDUAL_VISIBILITIES)
		fread( (void *) &_hstResidualVisibility[ _hstNumVisibilities[ pMosaicID ][ pStageID_two ] ], sizeof( cufftComplex ),
					_hstNumVisibilities[ pMosaicID ][ pStageID_one ], frOne );

	// close the file.
	fclose( frOne );

	// update the number of stages and the number of visibilities.
	_hstNumberOfStages[ pMosaicID ] -= 1;
	_hstNumVisibilities[ pMosaicID ][ pStageID_one ] += _hstNumVisibilities[ pMosaicID ][ pStageID_two ];
	_hstNumVisibilities[ pMosaicID ] = (long int *) realloc( _hstNumVisibilities[ pMosaicID ], _hstNumberOfStages[ pMosaicID ] * sizeof( long int * ) );

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

} // mergeData

//
//	processMeasurementSet()
//
//	CJS: 05/07/2019
//
//	Loads the measurement sets into memory, and caches them so we don't have to store them all at the same time.
//

void processMeasurementSet( char * pFilenamePrefix, char * pMeasurementSetFilename, double * phstImagePhasePosition, int pFileIndex )
{
	
	// timings. only used for development and debugging. whenever the time is retrieved there is a preceeding call to cudaDeviceSynchronize(), which
	// ensures that the timings are accurate for each step. if these lines are commented out then the gpu and cpu code can run asynchronously, and
	// overall performance will improve.
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
	_hstCasacoreInterface.GetWavelengths(	/* pMeasurementSet = */ pMeasurementSetFilename,
						/* pNumSpws = */ &numSpws,
						/* pNumChannels = */ &hstNumChannels,
						/* pWavelength = */ &hstWavelength );

	// get the polarisation multiplier (and the number of polarisations) which describes how the polarisation products should be handled.
	int hstNumPolarisations = -1, hstNumPolarisationConfigurations = -1;
	double * hstMultiplier = getPolarisationMultiplier(	/* pMeasurementSetFilename = */ pMeasurementSetFilename,
								/* pNumPolarisations = */ &hstNumPolarisations,
								/* pNumPolarisationConfigurations = */ &hstNumPolarisationConfigurations );

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
	_hstCasacoreInterface.GetDataDesc(	/* pMeasurementSet = */ pMeasurementSetFilename,
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
				/* phstSpwChannelFlag = */ &hstSpwChannelFlag );

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
							/* IN: pFieldID = */ _hstFieldID,
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

	}

	// get the antennae from the file.
	bool * hstAntennaFlag = NULL;
	double * hstDishDiameter = NULL;
	int numberOfAntennae = _hstCasacoreInterface.GetAntennae(	/* pMeasurementSet = */ pMeasurementSetFilename,
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
	_hstMinWavelength = hstWavelength[ 0 ][ 0 ];
	_hstMaxWavelength = hstWavelength[ 0 ][ 0 ];
	
	// get the average wavelength.
	_hstAverageWavelength[ pFileIndex ] = 0;
	int hstValidChannels = 0;
	for ( int sample = 0; sample < hstNumSamples / 2; sample++ )
		for ( int channel = 0; channel < hstNumChannels[ hstDataDescSpw[ hstDataDescID[ sample ] ] ]; channel++ )
			if (hstSpwChannelFlag[ hstDataDescSpw[ hstDataDescID[ sample ] ] ][ channel ] == false)
			{
				double wavelength = hstWavelength[ hstDataDescSpw[ hstDataDescID[ sample ] ] ][ channel ];
				_hstAverageWavelength[ pFileIndex ] += wavelength;
				if (wavelength < _hstMinWavelength)
					_hstMinWavelength = wavelength;
				if (wavelength > _hstMaxWavelength)
					_hstMaxWavelength = wavelength;
				hstValidChannels++;
			}
	if (hstValidChannels > 0)
		_hstAverageWavelength[ pFileIndex ] /= (double) hstValidChannels;
	else
		_hstAverageWavelength[ pFileIndex ] = 1;

	// free data.
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

	// set primary beam parameters depending upon telescope.
	if (_hstDiskDiameterSupplied == false)
		switch (_hstTelescope)
		{
			case ALMA:
			case ALMA_7M:		{ _hstAiryDiskDiameter = 6.25; break; }
			case ALMA_12M:		{ _hstAiryDiskDiameter = 10.70; break; }
			case ASKAP:		{ _hstAiryDiskDiameter = 12.00; break; }
		}
	if (_hstDiskBlockageSupplied == false)
		switch (_hstTelescope)
		{
			case ALMA:
			case ALMA_7M:		{ _hstAiryDiskBlockage = 0.75; break; }
			case ALMA_12M:		{ _hstAiryDiskBlockage = 0.75; break; }
			case ASKAP:		{ _hstAiryDiskBlockage = 0.75; break; }
		}

	// generate primary beams.
	generatePrimaryBeamAiry(	/* phstPrimaryBeamIn = */ &hstPrimaryBeamIn,
					/* pWidth = */ _hstAiryDiskDiameter,
					/* pCutout = */ _hstAiryDiskBlockage,
					/* pWavelength = */ hstWavelengthForBeam );

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
	_hstCasacoreInterface.GetPhaseCentres(	/* pMeasurementSet = */ pMeasurementSetFilename,
						/* pNumFields = */ &hstNumFields,
						/* pPhaseCentre = */ &hstFieldPhaseFrom );

	// convert from radians to degrees.
	for ( int i = 0; i < hstNumFields * 2; i++ )
		hstFieldPhaseFrom[ i ] = hstFieldPhaseFrom[ i ] * 180.0 / PI;

	// ensure ra is in range 0 <= ra < 360.
	for ( int i = 0; i < hstNumFields * 2; i = i + 2 )
		if (hstFieldPhaseFrom[ i ] < 0.0)
			hstFieldPhaseFrom[ i ] += 360.0;

	// free the image primary beams if they exist.
	if (hstFieldPhaseTo != NULL)
	{
		free( (void *) hstFieldPhaseTo );
		hstFieldPhaseTo = NULL;
	}

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

	// if we're beam mosaicing then store the number of fields in the beam mosaic.
	if (_hstBeamMosaic == true)
		_hstNumFieldsForBeamMosaic = hstNumFields;

	//
	// what to do about phase rotation.....
	//
	// beam mosaicing:	(uv domain, used for multiple pointings) the data are phase rotated to the beam positions. we need this data for constructing the
	//			beams in the right place, but we need to phase rotate the data to the output phase centre.
	// field mosaicing:	(image domain, used for multiple pointings) the data are phase rotated to the beam positions. if the primary beam is much larger
	//			than the image area we are interested in then we need to phase rotate to an appropriate place.
	// file mosaicing:	(image domain, used for PAF data) the data are phase rotated to the primary beam location. if the primary beam is much larger than
	//			the image area we are interested in then we need to phase rotate to an appropriate place.
	// no mosaicing:	the data are phase rotated to the pointing position. we need to phase rotate to the output phase position.
	//

	if (_hstFileMosaic == true)
	{

		// we get the position of the ASKAP PAF beam, based upon the pointing position of the dish (for which we use the phase position).
		if (_hstTelescope == ASKAP)
			for ( int field = 0; field < hstNumFields; field++ )
				getASKAPBeamPosition(	/* pRA = */ &hstFieldPhaseFrom[ field * 2 ],
							/* pDEC = */ &hstFieldPhaseFrom[ (field * 2) + 1 ],
							/* pBeamIndex = */ _hstMosaicID[ pFileIndex ],
							/* pCentreRA = */ hstFieldPhaseFrom[ field * 2 ],
							/* pCentreDEC = */ hstFieldPhaseFrom[ (field * 2) + 1 ] );

		// get suitable phase positions for gridding.
		getSuitablePhasePositionForBeam(	/* pBeamIn = */ hstFieldPhaseFrom,
							/* pPhase = */ hstFieldPhaseTo,
							/* pNumBeams = */ hstNumFields );

	}

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
//const long int MEMORY_LIMIT = (long int) 2 * (long int) 1073741824;
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

	// for uniform or robust weighting we need to store the sum of weights in each cell.
	double * hstTotalWeightPerCell = NULL;
	if (_hstWeighting == ROBUST || _hstWeighting == UNIFORM)
	{
		hstTotalWeightPerCell = (double *) malloc( (long int) _hstUvPixels * (long int) _hstUvPixels * (long int) sizeof( double ) );
		memset( hstTotalWeightPerCell, 0, (long int) _hstUvPixels * (long int) _hstUvPixels * (long int) sizeof( double ) );
	}

	// we will need to work out the average weight in the gridded cells.
	_hstAverageWeight[ pFileIndex ] = 0.0;

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

		// create temporary arrays for the visibilities, flags, weights and field ids.
		cufftComplex * tmpVisibility = NULL;
		bool * tmpFlag = NULL;
		float * tmpWeight = NULL;
		int * tmpDataDescID = NULL;

		// load visibilities. we load ALL the visibilities to the host, and these are then processed on the device in batches.
		int numSamplesInStage = 0;
		_hstCasacoreInterface.GetVisibilities(	/* IN: pFilename = */ pMeasurementSetFilename,
							/* IN: pFieldID = */ _hstFieldID,
							/* OUT: pNumSamples = */ &numSamplesInStage,
							/* IN: pNumChannels = */ hstNumChannels,
							/* IN: pDataField = */ _hstDataField,
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

			cudaDeviceSynchronize();
		
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

		cudaDeviceSynchronize();

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

			cudaDeviceSynchronize();

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
			cudaDeviceSynchronize();

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

			cudaDeviceSynchronize();

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

			// we need to sort data into order of W plane, kernel index, U value and V value, and remove all the duplicates.
//			quickSortData(	/* pLeft = */ hstCurrentVisibility,
//					/* pRight = */ hstCurrentVisibility + hstVisibilityBatchSize - 1 );

			// compact the data so that items with a duplicate grid position are only gridded once.
			hstCurrentVisibility = compactData(	/* pTotalVisibilities = */ &_hstNumVisibilities[ pFileIndex ][ stageID ],
								/* pFirstVisibility = */ hstCurrentVisibility,
								/* pNumVisibilities = */ hstVisibilityBatchSize );
//			hstCurrentVisibility = hstCurrentVisibility + hstVisibilityBatchSize;

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
					hstTotalWeightPerCell[ (_hstGridPosition[ i ].v * _hstUvPixels) + _hstGridPosition[ i ].u ] +=
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

			_hstCacheData = _hstNumberOfStages[ pFileIndex ] > 1;

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

	// for uniform weighting update the weight using the new density map.
	if (_hstWeighting == UNIFORM)
		for ( int stageID = 0; stageID < _hstNumberOfStages[ pFileIndex ]; stageID++ )
		{

			// get the weights, densities and grid positions from the file for this stage.
			if (_hstCacheData == true)
				uncacheData(	/* pFilenamePrefix = */ pFilenamePrefix,
						/* pMosaicID = */ pFileIndex,
						/* pStageID = */ stageID,
						/* pWhatData = */ DATA_WEIGHTS | DATA_GRID_POSITIONS | DATA_DENSITIES );

			// divide each weight by the total weight in that cell, and add up the weights so we can make an average.
			for ( long int i = 0; i < _hstNumVisibilities[ pFileIndex ][ stageID ]; i++ )
				if (	_hstGridPosition[ i ].u >= 0 && _hstGridPosition[ i ].u < _hstUvPixels &&
					_hstGridPosition[ i ].v >= 0 && _hstGridPosition[ i ].v < _hstUvPixels)
				{
					_hstWeight[ i ] /= hstTotalWeightPerCell[ (_hstGridPosition[ i ].v * _hstUvPixels) + _hstGridPosition[ i ].u ];
					_hstAverageWeight[ pFileIndex ] += (double) _hstWeight[ i ] * (double) _hstDensityMap[ i ];
				}

			// re-cache the weights and free the densities and grid positions for this stage.
			if (_hstCacheData == true)
			{
				cacheData(	/* pFilenamePrefix = */ pFilenamePrefix,
						/* pMosaicID = */ pFileIndex,
						/* pStageID = */ stageID,
						/* pWhatData = */ DATA_WEIGHTS );
				freeData( /* pWhatData = */ DATA_DENSITIES | DATA_GRID_POSITIONS );
			}

		}

	// for robust weighting we need to calculate the average cell weighting, and then the parameter f^2.
	if (_hstWeighting == ROBUST)
	{

		// calculate the average cell weighting.
		double averageCell = 0.0;
		for ( int stageID = 0; stageID < _hstNumberOfStages[ pFileIndex ]; stageID++ )
		{

			// get the grid positions and densities.
			if (_hstCacheData == true)
				uncacheData(	/* pFilenamePrefix = */ pFilenamePrefix,
						/* pMosaicID = */ pFileIndex,
						/* pStageID = */ stageID,
						/* pWhatData = */ DATA_GRID_POSITIONS | DATA_DENSITIES );

			// calculate average cell weighting.
			for ( long int i = 0; i < _hstNumVisibilities[ pFileIndex ][ stageID ]; i++ )
				if (	_hstGridPosition[ i ].u >= 0 && _hstGridPosition[ i ].u < _hstUvPixels &&
					_hstGridPosition[ i ].v >= 0 && _hstGridPosition[ i ].v < _hstUvPixels)
					averageCell += hstTotalWeightPerCell[ (_hstGridPosition[ i ].v * _hstUvPixels) + _hstGridPosition[ i ].u ] *
										(double) _hstDensityMap[ i ];

			// free the data for the grid positions and densities.
			if (_hstCacheData == true)
				freeData( /* pWhatData = */ DATA_GRID_POSITIONS | DATA_DENSITIES );

		}

		// calculate average and f^2 parameter.
		if (_hstGriddedVisibilities[ pFileIndex ] > 0)
			averageCell /= (double) _hstGriddedVisibilities[ pFileIndex ];
		double fSquared = 0.0;
		if (averageCell != 0.0)
			fSquared = pow( 5 * pow( 10.0, -_hstRobust ), 2 ) / averageCell;

		for ( int stageID = 0; stageID < _hstNumberOfStages[ pFileIndex ]; stageID++ )
		{

			// get the grid positions, densities and weights.
			if (_hstCacheData == true)
				uncacheData(	/* pFilenamePrefix = */ pFilenamePrefix,
						/* pMosaicID = */ pFileIndex,
						/* pStageID = */ stageID,
						/* pWhatData = */ DATA_GRID_POSITIONS | DATA_WEIGHTS | DATA_DENSITIES );

			// update the weight of each visibility using the original weight, the sum of weights in the cell, and the f^2 parameter. also, add
			// up the weights so we can construct an average.
			for ( long int i = 0; i < _hstNumVisibilities[ pFileIndex ][ stageID ]; i++ )
				if (	_hstGridPosition[ i ].u >= 0 && _hstGridPosition[ i ].u < _hstUvPixels &&
					_hstGridPosition[ i ].v >= 0 && _hstGridPosition[ i ].v < _hstUvPixels)
				{
					_hstWeight[ i ] /= (1.0 + (hstTotalWeightPerCell[ (_hstGridPosition[ i ].v * _hstUvPixels) +
												_hstGridPosition[ i ].u ] * fSquared));
					_hstAverageWeight[ pFileIndex ] += (double) _hstWeight[ i ] * (double) _hstDensityMap[ i ];
				}

			// re-cache the weights and free the densities and grid positions for this stage.
			if (_hstCacheData == true)
			{
				cacheData(	/* pFilenamePrefix = */ pFilenamePrefix,
						/* pMosaicID = */ pFileIndex,
						/* pStageID = */ stageID,
						/* pWhatData = */ DATA_WEIGHTS );
				freeData( /* pWhatData = */ DATA_DENSITIES | DATA_GRID_POSITIONS );
			}

		}

	}

	// for all weighting, calculate the average weight.
	if (_hstWeighting != NONE)
		_hstAverageWeight[ pFileIndex ] /= (double) _hstGriddedVisibilities[ pFileIndex ];

	// how many primary beams are needed for this file ?
	int hstPrimaryBeamsForFile = 1;
	if (_hstAProjection == true || _hstBeamMosaic == true)
		hstPrimaryBeamsForFile = hstNumFields;

	// create some more primary beam space.
	if (_hstPrimaryBeam == NULL)
		_hstPrimaryBeam = (float *) malloc( (_hstNumPrimaryBeams + hstPrimaryBeamsForFile) * _hstBeamSize * _hstBeamSize * sizeof( float ) );
	else
		_hstPrimaryBeam = (float *) realloc( _hstPrimaryBeam, (_hstNumPrimaryBeams + hstPrimaryBeamsForFile) * _hstBeamSize * _hstBeamSize *
							sizeof( float ) );

	// create a reprojection object.
	Reprojection imagePlaneReprojection;

	// create two workspace primary beams on the device.
	float * devInBeam = NULL;
	float * devOutBeam = NULL;
	reserveGPUMemory( (void **) &devInBeam, _hstBeamSize * _hstBeamSize * sizeof( float ),
				"reserving memory for the input primary beam on the device" );
	reserveGPUMemory( (void **) &devOutBeam, _hstBeamSize * _hstBeamSize * sizeof( float ),
				"reserving memory for the output primary beam on the device" );

	// create the device memory needed by the reprojection code.
	Reprojection::rpVectI outSize = { /* x = */ _hstBeamSize, /* y = */ _hstBeamSize };
	imagePlaneReprojection.CreateDeviceMemory( outSize );

	// we need to do an image-plane reprojection of the beam to the common phase position, scaling the beams in the process so that they are
	// the same size as our images.
	for ( int beam = 0; beam < hstPrimaryBeamsForFile; beam++ )
	{
		printf( "Reprojecting beam %i to the new phase position\n", beam );
		imagePlaneReprojectPrimaryBeam(	/* pPrimaryBeamIn = */ hstPrimaryBeamIn,
						/* pPrimaryBeamOut = */ &_hstPrimaryBeam[ (_hstNumPrimaryBeams + beam) * _hstBeamSize * _hstBeamSize ],
						/* pBeam = */ beam,
						/* pInRA = */ hstFieldPhaseFrom[ beam * 2 ],
						/* pInDec = */ hstFieldPhaseFrom[ (beam * 2) + 1 ],
						/* pOutRA = */ hstFieldPhaseTo[ beam * 2 ],
						/* pOutDec = */ hstFieldPhaseTo[ (beam * 2) + 1 ],
						/* pdevInBeam = */ devInBeam,
						/* pdevOutBeam = */ devOutBeam,
						/* pBeamCellSize = */ _hstBeamCellSize,
						/* pImagePlaneReprojection = */ imagePlaneReprojection,
						/* pVerbose = */ true );
// save the reprojected beam.
//{
//char beamFilename[100];
//sprintf( beamFilename, "beam-mosaic-%i.image", beam );
//_hstCasacoreInterface.WriteCasaImage( beamFilename, _hstBeamSize, _hstBeamSize, hstFieldPhaseTo[ beam * 2 ], hstFieldPhaseTo[ (beam * 2) + 1 ],
//					_hstBeamCellSize, &_hstPrimaryBeam[ (_hstNumPrimaryBeams + beam) * _hstBeamSize * _hstBeamSize ],
//					CONST_C / _hstAverageWavelength[ beam ], NULL );
//}
	}

	// create some more space for the A-projection primary beams.
	if (_hstAProjection == true)
	{

		if (_hstPrimaryBeamAProjection == NULL)
			_hstPrimaryBeamAProjection = (float *) malloc( (_hstNumPrimaryBeams + hstPrimaryBeamsForFile) * _hstAPlanes * _hstBeamSize *
										_hstBeamSize * sizeof( float ) );
		else
			_hstPrimaryBeamAProjection = (float *) realloc( _hstPrimaryBeamAProjection, (_hstNumPrimaryBeams + hstPrimaryBeamsForFile) *
										_hstAPlanes * _hstBeamSize * _hstBeamSize * sizeof( float ) );

	}

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
								/* pPrimaryBeamOut = */ &_hstPrimaryBeamAProjection[ (((_hstNumPrimaryBeams + beam) *
															_hstAPlanes) + aPlane) *
															_hstBeamSize * _hstBeamSize ],
								/* pBeam = */ beam,
								/* pInRA = */ hstFieldPhaseFrom[ beam * 2 ],
								/* pInDec = */ hstFieldPhaseFrom[ (beam * 2) + 1 ],
								/* pOutRA = */ hstFieldPhaseTo[ beam * 2 ],
								/* pOutDec = */ hstFieldPhaseTo[ (beam * 2) + 1 ],
								/* pdevInBeam = */ devInBeam,
								/* pdevOutBeam = */ devOutBeam,
								/* pBeamCellSize = */ _hstBeamCellSize * (hstAPlaneWavelength[ aPlane ] /
														hstWavelengthForBeam),
								/* pImagePlaneReprojection = */ imagePlaneReprojection,
								/* pVerbose = */ false );

// save the reprojected beam.
//if (aPlane == 0)
//{
//char beamFilename[100];
//sprintf( beamFilename, "beam-aproj-%i-%i.image", beam, aPlane );
//_hstCasacoreInterface.WriteCasaImage( beamFilename, _hstUvPixels, _hstUvPixels, hstReprojectPhaseTo[ beam * 2 ], hstReprojectPhaseTo[ (beam * 2) + 1 ],
//					_hstCellSize, &_hstPrimaryBeamAProjection[ ((beam * _hstAPlanes) + aPlane) * _hstBeamSize * _hstBeamSize ],
//					CONST_C / _hstAverageWavelength[ beam ], NULL );
//}

		}

		printf( "\rReprojecting beams for %i a-planes.....100%%\n", _hstAPlanes );

	}

	// update the pointers to the primary beam arrays.
	_hstPrimaryBeamPtr[ pFileIndex ] = &_hstPrimaryBeam[ _hstNumPrimaryBeams * _hstBeamSize * _hstBeamSize ];
	if (_hstBeamMosaic == true)
		_hstPrimaryBeamMosaicingPtr[ pFileIndex ] = _hstPrimaryBeamPtr[ pFileIndex ];
	_hstPrimaryBeamAProjectionPtr[ pFileIndex ] = &_hstPrimaryBeamAProjection[ (_hstNumPrimaryBeams * _hstAPlanes) * _hstBeamSize * _hstBeamSize ];

	// update the number of primary beams with the extra ones we generated for this measurement set.
	_hstNumPrimaryBeams += hstPrimaryBeamsForFile;

	// free memory.
	if (devInBeam != NULL)
		cudaFree( (void *) devInBeam );
	if (devOutBeam != NULL)
		cudaFree( (void *) devOutBeam );
	if (hstTotalWeightPerCell != NULL)
		free( (void *) hstTotalWeightPerCell );
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

	// timings. only used for development and debugging. whenever the time is retrieved there is a preceeding call to cudaDeviceSynchronize(), which
	// ensures that the timings are accurate for each step. if these lines are commented out then the gpu and cpu code can run asynchronously, and
	// overall performance will improve.
	cudaError_t err;
	
	// read program arguments. we expect to see the program call (0), the input filename (1) and the output filename (2).
	if (pArgc != 2)
	{
		printf("Wrong number of arguments. I require the parameter filename.\n");
		return 1;
	}
	
	char * parameterFile = pArgv[ 1 ];
	
	// get the parameters from file.
	getParameters( parameterFile );

	// check path name is valid.
	int length = strlen( _hstCacheLocation );
	if (_hstCacheLocation[ length - 1 ] != '/' && length > 0)
	{
		_hstCacheLocation[ length ] = '/';
		_hstCacheLocation[ length + 1 ] = '\0';
	}
	
	// build output filenames.
	strcpy( outputCleanImageFilename, _hstOutputPrefix ); strcat( outputCleanImageFilename, CLEAN_IMAGE_EXTENSION ); strcat( outputCleanImageFilename, FILE_EXTENSION );
	strcpy( outputResidualImageFilename, _hstOutputPrefix ); strcat( outputResidualImageFilename, RESIDUAL_IMAGE_EXTENSION ); strcat( outputResidualImageFilename, FILE_EXTENSION );
	strcpy( outputDirtyBeamFilename, _hstOutputPrefix ); strcat( outputDirtyBeamFilename, DIRTY_BEAM_EXTENSION ); strcat( outputDirtyBeamFilename, FILE_EXTENSION );
	strcpy( outputCleanBeamFilename, _hstOutputPrefix ); strcat( outputCleanBeamFilename, CLEAN_BEAM_EXTENSION ); strcat( outputCleanBeamFilename, FILE_EXTENSION );
	strcpy( outputGriddedFilename, _hstOutputPrefix ); strcat( outputGriddedFilename, GRIDDED_EXTENSION ); strcat( outputGriddedFilename, FILE_EXTENSION );
	strcpy( outputDirtyImageFilename, _hstOutputPrefix ); strcat( outputDirtyImageFilename, DIRTY_IMAGE_EXTENSION ); strcat( outputDirtyImageFilename, FILE_EXTENSION );
	
	// get some properties from the device.
	cudaDeviceProp gpuProperties;
	cudaGetDeviceProperties( &gpuProperties, 0 );
	_maxThreadsPerBlock = gpuProperties.maxThreadsPerBlock;
	_warpSize = gpuProperties.warpSize;
	int * maxGridSize = gpuProperties.maxGridSize;
	_gpuMemory = (long int) gpuProperties.totalGlobalMem;

	printf( "\nGIMAGE\n" );
	printf( "======\n\n" );
	printf( "GPU properties:\n" );
	printf( "---------------\n\n" );
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
		printf( "polarisation: STOKES_I\n\n" );
	else if (_hstStokes == STOKES_Q)
		printf( "polarisation: STOKES_Q\n\n" );
	else if (_hstStokes == STOKES_U)
		printf( "polarisation: STOKES_U\n\n" );
	else if (_hstStokes == STOKES_V)
		printf( "polarisation: STOKES_V\n" );

	// set a flag to determine if we're using multiple file mosaic (i.e. PAFs).
	bool useMultiFiles = (_hstMeasurementSets > 0);
	if (_hstMeasurementSets == 0)
		_hstMeasurementSets = 1;

	// if we are using multiple measurement sets then we need to cache our data.
	if (_hstMeasurementSets > 1)
		_hstCacheData = true;

	// if we are constructing a large image then calculate how many rows we should process in each batch.
	_hstGPURows = (int) ceil( (double) _gpuMemory * 0.25 / ((double) sizeof( cufftComplex/*grid*/ ) * (double) _hstUvPixels) );
	if (_hstGPURows > _hstUvPixels)
		_hstGPURows = _hstUvPixels;
//if (_hstGPURows > 512) // cjs-mod
//	_hstGPURows = 512;
	_hstImageBatches = (int) ceil( (double) _hstUvPixels / (double) _hstGPURows );

	// calculate the size of the dirty beam. this must be the largest even-sized image that has a number of pixels up to the GPU batch size.
	if (_hstImageBatches > 1)
	{
		_hstPsfSize = (int) floor( sqrt( (double) (_hstGPURows * _hstUvPixels) ) );
		if (_hstPsfSize % 2 == 1)
			_hstPsfSize -= 1;
	}
	else
		_hstPsfSize = _hstUvPixels;
//_hstPsfSize = 2048; // cjs-mod

	// we don't want the psf to be larger than 2048 x 2048.
	if (_hstPsfSize > 2048)
		_hstPsfSize = 2048;

	printf( "gridding visibilities in %i batches, with %i rows per batch\n", _hstImageBatches, _hstGPURows );
	printf( "the size of the psf will be %i x %i pixels\n", _hstPsfSize, _hstPsfSize );
	printf( "the size of the primary beam will be %i x %i pixels\n\n", BEAM_SIZE, BEAM_SIZE );

	// turn on mosaicing if we are using multi files. we currently restrict this software to EITHER assembling a mosaic from the various files of a single
	// measurement set, OR assembling the images from multi files, with the same FOV and phase centre, into a single image. The latter is used for the multi-beams
	// of a PAF.
	_hstBeamMosaic = (_hstUseMosaicing == true && useMultiFiles == false);
	_hstFileMosaic = (_hstUseMosaicing == true && useMultiFiles == true);
	if (_hstUseMosaicing == false && useMultiFiles == true)
	{
		printf( "loading multiple measurement sets, so I am turning mosaicing on.\n\n" );
		_hstUseMosaicing = true;
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
	}
	printf( "\n" );
	if (_hstBeamMosaic == true)
	{
		printf( "mosaic: Y (multi field)\n" );
		printf( "        fields - %s\n", _hstFieldID );
		if (_hstBeamMosaic == true)
			printf( "            (these fields will be imaged using one uv plane)\n" );
		else
			printf( "            (these fields will be imaged separately)\n" );
	}
	else if (_hstFileMosaic == true)
	{
		printf( "mosaic: Y (multi file)\n" );
		printf( "        files -" );
		for ( int i = 0; i < _hstMeasurementSets; i++ )
			printf( " %i", _hstMosaicID[ i ] ) ;
		printf( "\n" );
	}
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
	_numMosaicImages = (_hstFileMosaic == true ? _hstMeasurementSets : 1);

	// create array to hold the batch info for each file.
	_hstNumVisibilities = (long int **) malloc( _numMosaicImages * sizeof( long int * ) );
	_hstNumberOfStages = (int *) malloc( _numMosaicImages * sizeof( int ) );

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

	// create some pointers to the primary beam arrays so we can index them for each mosaic image.
	_hstPrimaryBeamPtr = (float **) malloc( _hstMeasurementSets * sizeof( float * ) );
	_hstPrimaryBeamMosaicingPtr = (float **) malloc( _hstMeasurementSets * sizeof( float * ) );
	_hstPrimaryBeamAProjectionPtr = (float **) malloc( _hstMeasurementSets * sizeof( float * ) );
	for ( int i = 0; i < _hstMeasurementSets; i++ )
	{
		_hstPrimaryBeamPtr[ i ] = NULL;
		_hstPrimaryBeamMosaicingPtr[ i ] = NULL;
		_hstPrimaryBeamAProjectionPtr[ i ] = NULL;
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

	// build a list of filenames.
	char ** hstFilename = (char **) malloc( _hstMeasurementSets * sizeof( char * ) );
	for ( int i = 0; i < _hstMeasurementSets; i++ )
	{

		// create memory for the filename and copy from the global filename.
		hstFilename[ i ] = (char *) malloc( 128 * sizeof( char ) );
		strcpy( hstFilename[ i ], _hstMeasurementSetPath );

	}

	// if we're using multiple files then we need to replace the wildcard in the measurement set name with the correct file index.
	if (_hstFileMosaic == true)
	{
	
		// create memory for the filenames.
		for ( int i = 0; i < _hstMeasurementSets; i++ )
		{

			// build the filename by replacing the '**' characters in the measurement set by the index.
			bool replacementMade = false;
			for ( int j = 0; j < strlen( hstFilename[ i ] ) - 1; j++ )
				if (hstFilename[ i ][ j ] == '*' && hstFilename[ i ][ j + 1 ] == '*')
				{
					hstFilename[ i ][ j ] = ((_hstMosaicID[ i ] / 10) % 10) + '0';
					hstFilename[ i ][ j + 1 ] = (_hstMosaicID[ i ] % 10) + '0';
					replacementMade = true;
					break;
				}

			// build filename by replacing the '*' character in the measurement set by the index.
			if (replacementMade == false)
				for ( int j = 0; j < strlen( hstFilename[ i ] ); j++ )
					if (hstFilename[ i ][ j ] == '*')
					{
						hstFilename[ i ][ j ] = (_hstMosaicID[ i ] % 10) + '0';
						replacementMade = true;
						break;
					}

		}

	}

	// process the input measurement sets by loading the data and caching it.
	for ( int measurementSet = 0; measurementSet < _hstMeasurementSets; measurementSet++ )
		processMeasurementSet(	/* pFilenamePrefix = */ _hstOutputPrefix,
					/* pMeasurementSetFilename = */ hstFilename[ measurementSet ],
					/* phstImagePhasePosition = */ &hstImagePhaseCentre[ measurementSet * 2 ],
					/* pFileIndex = */ measurementSet );

	// free filenames
	if (hstFilename != NULL)
	{
		for ( int i = 0; i < _hstMeasurementSets; i++ )
			free( (void *) hstFilename[ i ] );
		free( (void *) hstFilename );
	}
	
	// generate image of convolution function.
	generateImageOfConvolutionFunction( outputDeconvolutionFilename );

	// for file mosaicing, the mosaic IDs are the file IDs. if we're not using file mosaicing then set the mosaic IDs to be the numbers 0,1,2,3,etc....
	if (_hstFileMosaic == false)
	{

		// clear any existing mosaic IDs.
		if (_hstMosaicID != NULL)
			free( (void *) _hstMosaicID );

		// declare N mosaic IDs, where N is the number of mosaic images.
		_hstMosaicID = (int *) malloc( _numMosaicImages * sizeof( int ) );

		// set these mosaic IDs.
		for ( int i = 0; i < _numMosaicImages; i++ )
			_hstMosaicID[ i ] = i;

	}

	// we need to count how many visibilities are being gridded with each image/beam.
	if (_hstBeamMosaic == true)
	{

		// find the minimum number of gridded visibilities. we will use this figure for our normalisation pattern. Our beams with higher numbers of gridded
		// visibilities will be corrected for in the gridding kernel.
		_griddedVisibilitiesForBeamMosaic = 0;
		for ( int i = 0; i < _hstNumPrimaryBeams; i++ )
			if (i == 0 || _hstGriddedVisibilitiesPerField[ i ] < _griddedVisibilitiesForBeamMosaic)
				_griddedVisibilitiesForBeamMosaic = _hstGriddedVisibilitiesPerField[ i ];

		// ensure the number of gridded visibilities is not zero.
		if (_griddedVisibilitiesForBeamMosaic == 0)
			_griddedVisibilitiesForBeamMosaic = 1;

	}

	// we need to make an image mask.
	bool * hstMask = (bool *) malloc( (long int) _hstUvPixels * (long int) _hstUvPixels * sizeof( bool ) );

	// reserve some memory for the number of visibilities per kernel set.
	_hstVisibilitiesInKernelSet = (int ***) malloc( _numMosaicImages * sizeof( int ** ) );
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
				hstMask[ index ] = (_hstPrimaryBeam[ (beamJ * _hstBeamSize) + beamI ] >= 0.2);
			}
		}

	}

	// if we are using beam mosaicing then we need to assemble a weighted image of the primary beam patterns, and build a mask from it.
	float * devNormalisationPattern = NULL;
	if (_hstBeamMosaic == true)
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

			for ( int beam = 0; beam < _hstNumPrimaryBeams; beam++ )
				_hstPrimaryBeamPattern[ index ] += pow( _hstPrimaryBeam[ ((long int) beam * (long int) _hstBeamSize * (long int) _hstBeamSize) +
											(long int) index ], 2 );
//				_hstPrimaryBeamPattern[ index ] += _hstPrimaryBeam[ ((long int) beam * (long int) _hstBeamSize * (long int) _hstBeamSize) +
//											(long int) index ];
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
			for ( int beam = 0; beam < _hstNumPrimaryBeams; beam++ )
			{
				float * beamPtr = &_hstPrimaryBeam[ (long int) beam * (long int) _hstBeamSize * (long int) _hstBeamSize ];
				_hstNormalisationPattern[ index ] += pow( beamPtr[ index ], 2 );
// cjs-mod				_hstNormalisationPattern[ index ] += pow( beamPtr[ index ], 1 );
			}

			// divide the normalisation pattern by the primary beam pattern in order than we smooth out the noise near the edges of the image.
			_hstNormalisationPattern[ index ] /= _hstPrimaryBeamPattern[ index ];

		}

		// if we're gridding on the device, then copy the normalisation pattern over.
		if (_hstImageBatches == 1)
		{
			reserveGPUMemory( (void **) &devNormalisationPattern, _hstBeamSize * _hstBeamSize * sizeof( float ),
						"reserving GPU memory for normalisation pattern" );
			moveHostToDevice( (void *) devNormalisationPattern, (void *) _hstNormalisationPattern, _hstBeamSize * _hstBeamSize * sizeof( float ),
						"copying normalisation pattern to the device" );
		}

	}

	// prepare an array to store the number of visibilities in each kernel set.
	for ( int image = 0; image < _numMosaicImages; image++ )
		_hstVisibilitiesInKernelSet[ image ] = (int **) malloc( _hstNumberOfStages[ image ] * sizeof( int * ) );

	// ----------------------------------------------------
	//
	// g r i d   v i s i b i l i t i e s
	//
	// ----------------------------------------------------

	cufftComplex/*grid*/ * devDirtyBeamGrid = NULL;
	if (_hstMinorCycles > 0)
	{

		// create memory for the psf on the device, and clear it.
		reserveGPUMemory( (void **) &devDirtyBeamGrid, _hstGPURows * _hstUvPixels * sizeof( cufftComplex/*grid*/ ), "declaring device memory for psf" );
		zeroGPUMemory( (void *) devDirtyBeamGrid, _hstGPURows * _hstUvPixels * sizeof( cufftComplex/*grid*/ ), "zeroing the dirty beam on the device" );

		// create the dirty beam grid, but only if we need to grid on the host.
		cufftComplex/*grid*/ * hstDirtyBeamGrid = NULL;
		if (_hstImageBatches > 1)
		{
			hstDirtyBeamGrid = (cufftComplex/*grid*/ *) malloc( (long int) _hstUvPixels * (long int) _hstUvPixels * sizeof( cufftComplex/*grid*/ ) );
			memset( hstDirtyBeamGrid, 0,  (long int) _hstUvPixels * (long int) _hstUvPixels * sizeof( cufftComplex/*grid*/ ) );
		}

		printf( "gridding visibilities for psf.....\n\n" );
		for ( int image = 0; image < _numMosaicImages; image++ )
		{

			// count the total number of visibilities.
			long int totalVisibilities = 0;
			for ( int stageID = 0; stageID < _hstNumberOfStages[ image ]; stageID++ )
				totalVisibilities += _hstNumVisibilities[ image ][ stageID ];

			// uncache the data for this mosaic image (if we have more than one field).
			long int visibilitiesProcessed = 0;
			for ( int stageID = 0; stageID < _hstNumberOfStages[ image ]; stageID++ )
			{

				// get the data from the file.
				if (_hstCacheData == true)
					uncacheData(	/* pFilenamePrefix = */ _hstOutputPrefix,
							/* pMosaicID = */ image,
							/* pStageID = */ stageID,
							/* pWhatData = */ DATA_GRID_POSITIONS | DATA_KERNEL_INDEXES | DATA_DENSITIES | DATA_WEIGHTS );
	
				// if the number of visibilities is greater than some maximum then we are going to set a smaller batch size, and load these
				// visibilities in batches.
				int hstVisibilityBatchSize = 0;
				{
					long int nextBatchSize = _hstNumVisibilities[ image ][ stageID ];
					if (nextBatchSize > _hstPreferredVisibilityBatchSize)
						nextBatchSize = _hstPreferredVisibilityBatchSize;
					hstVisibilityBatchSize = (int) nextBatchSize;
				}

				// calculate the number of batches.
				int numberOfBatches = 1;
				if (_hstNumVisibilities[ image ][ stageID ] > _hstPreferredVisibilityBatchSize)
					numberOfBatches = (_hstNumVisibilities[ image ][ stageID ] / hstVisibilityBatchSize) + 1;

				// reserve some memory for the number of visibilities per kernel set.
				_hstVisibilitiesInKernelSet[ image ][ stageID ] = (int *) malloc( _hstKernelSets * numberOfBatches * sizeof( int ) );

				// create space for the unity (psf) visibilities, the density map, and the weights on the device.
				cufftComplex * devBeamVisibility = NULL;
				int * devDensityMap = NULL;
				float * devWeight = NULL;
				VectorI * devGridPosition = NULL;
				int * devKernelIndex = NULL;
				reserveGPUMemory( (void **) &devBeamVisibility, hstVisibilityBatchSize * sizeof( cufftComplex ),
							"creating device memory for the psf visibilities" );
				reserveGPUMemory( (void **) &devDensityMap, hstVisibilityBatchSize * sizeof( int ), "declaring device memory for the density map" );
				reserveGPUMemory( (void **) &devGridPosition, hstVisibilityBatchSize * sizeof( VectorI ), "reserving device memory for grid positions" );
				reserveGPUMemory( (void **) &devKernelIndex, hstVisibilityBatchSize * sizeof( int ), "reserving device memory for kernel indexes" );
				if (_hstWeighting != NONE)
					reserveGPUMemory( (void **) &devWeight, hstVisibilityBatchSize * sizeof( float ), "declaring device memory for the weights" );

				// create some memory for storing the number of visibilities per W plane.
				int * hstVisibilitiesInWPlane = (int *) malloc( _hstWPlanes * sizeof( int ) );

				// keep looping until we have loaded and gridded all visibilities.
				int batch = 0;
				long int hstCurrentVisibility = 0;
				while (hstCurrentVisibility < _hstNumVisibilities[ image ][ stageID ])
				{
		
					// if the number of remaining samples is lower than the sample batch size, then reduce the sample batch size accordingly.
					if (_hstNumVisibilities[ image ][ stageID ] - hstCurrentVisibility < hstVisibilityBatchSize)
						hstVisibilityBatchSize = _hstNumVisibilities[ image ][ stageID ] - hstCurrentVisibility;

					if (_hstNumberOfStages[ image ] > 1 || numberOfBatches > 1)
						printf( "        gridding " );
					if (_hstNumberOfStages[ image ] > 1)
						printf( "host batch %i of %i", stageID + 1, _hstNumberOfStages[ image ] );
					if (_hstNumberOfStages[ image ] > 1 && numberOfBatches > 1)
						printf( ", " );
					if (numberOfBatches > 1)
						printf( "gpu batch %i of %i", batch + 1, numberOfBatches );
					if (_hstNumberOfStages[ image ] > 1 || numberOfBatches > 1)
					{
						int fractionDone = (int) round( (double) visibilitiesProcessed * 30.0 / (double) totalVisibilities );
						int fractionDoing = (int) round( (double) (visibilitiesProcessed + hstVisibilityBatchSize) * 30.0 /
													(double) totalVisibilities );
						printf( " [" );
						for ( int i = 0; i < fractionDone; i++ )
							printf( "*" );
						for ( int i = 0; i < (fractionDoing - fractionDone); i++ )
							printf( "+" );
						for ( int i = 0; i < (30 - fractionDoing); i++ )
							printf( "." );
						printf( "]\n\n" );
						visibilitiesProcessed += hstVisibilityBatchSize;
					}

					// upload the visibility batch size to the device.
					err = cudaMemcpyToSymbol( _devVisibilityBatchSize, &hstVisibilityBatchSize, sizeof( hstVisibilityBatchSize ) );
					if (err != cudaSuccess)
						printf( "error copying visibility batch size to device (%s)\n", cudaGetErrorString( err ) );

					int * hstVisibilitiesInKernelSet = &_hstVisibilitiesInKernelSet[ image ][ stageID ][ batch * _hstKernelSets ];

					// calculate the number of visibilities in each kernel set.
					calculateVisibilitiesPerKernelSet(	/* pNumVisibilities = */ hstVisibilityBatchSize,
										/* phstGridPosition = */ &_hstGridPosition[ hstCurrentVisibility ],
										/* phstVisibilitiesInKernelSet = */ hstVisibilitiesInKernelSet );

					// calculate the number of visibilities per kernel set.
//					if (_hstWProjection == true || _hstAProjection == true)
//					{

						// calculate how many visibilities we have in each kernel set (A plane and W plane combination).
//						int kernelSet = 0;
//						int firstVisibility = 0;
//						for ( int i = 0; i < hstVisibilityBatchSize; i++ )
//							while (_hstGridPosition[ hstCurrentVisibility + i ].w > kernelSet)
//							{
//								hstVisibilitiesInKernelSet[ kernelSet ] = i - firstVisibility;
//								kernelSet = kernelSet + 1;
//								firstVisibility = i;
//							}

						// update any remaining planes.
//						hstVisibilitiesInKernelSet[ kernelSet ] = hstVisibilityBatchSize - firstVisibility;
//						while (kernelSet < ((_hstAPlanes * _hstWPlanes) - 1))
//						{
//							kernelSet = kernelSet + 1;
//							hstVisibilitiesInKernelSet[ kernelSet ] = 0;
//						}

//					}
//					else
//						hstVisibilitiesInKernelSet[ 0 ] = hstVisibilityBatchSize;

					// calculate the number of visibilities per W plane.
					for ( int wPlane = 0, kernelSet = 0; wPlane < _hstWPlanes; wPlane++ )
					{
						hstVisibilitiesInWPlane[ wPlane ] = 0;
						for ( int aPlane = 0; aPlane < _hstAPlanes; aPlane++, kernelSet++ )
							hstVisibilitiesInWPlane[ wPlane ] += hstVisibilitiesInKernelSet[ kernelSet ];
					}

					// upload the grid positions, kernel indexes, and density map to the device.
					moveHostToDevice( (void *) devGridPosition, (void *) &_hstGridPosition[ hstCurrentVisibility ], hstVisibilityBatchSize *
								sizeof( VectorI ), "copying grid positions to the device" );
					moveHostToDevice( (void *) devKernelIndex, (void *) &_hstKernelIndex[ hstCurrentVisibility ], hstVisibilityBatchSize *
								sizeof( int ), "copying kernel indexes to the device" );
					moveHostToDevice( (void *) devDensityMap, (void *) &_hstDensityMap[ hstCurrentVisibility ], hstVisibilityBatchSize * sizeof( int ),
								"copying density map to the device" );

					// upload weights to the device.
					if (_hstWeighting != NONE)
						moveHostToDevice( (void *) devWeight, (void *) &_hstWeight[ hstCurrentVisibility ], hstVisibilityBatchSize *
									sizeof( float ), "copying weights to the device" );

					// set all the visibilities to (1, 0). these are the visibilities used for generating the dirty beam.
					int threads = hstVisibilityBatchSize;
					int blocks = 1;
					setThreadBlockSize1D( &threads, &blocks );
		
					// update the real part of each visibility to 1.
					devUpdateComplexArray<<< blocks, threads>>>( devBeamVisibility, hstVisibilityBatchSize, 1, 0 );
					err = cudaGetLastError();
					if (err != cudaSuccess)
						printf( "error building visibilities for psf (%s)\n", cudaGetErrorString( err ) );

					// apply the density map - multiply all the visibilities by the value of the density map at that position.
					devApplyDensityMap<<< blocks, threads >>>( devBeamVisibility, devDensityMap );
					err = cudaGetLastError();
					if (err != cudaSuccess)
						printf( "error applying the density map to the beam visibilities (%s)\n", cudaGetErrorString( err ) );

					// generate the uv coverage using the gridder.
					gridVisibilities(	/* pdevGrid = */ devDirtyBeamGrid,
								/* phstGrid = */ hstDirtyBeamGrid,
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
								/* pSize = */ _hstUvPixels );
		
					// get the next batch of data.
					hstCurrentVisibility += hstVisibilityBatchSize;
					batch = batch + 1;
		
				}

				// free memory.
				if (devBeamVisibility != NULL)
					cudaFree( (void *) devBeamVisibility );
				if (devDensityMap != NULL)
					cudaFree( (void *) devDensityMap );
				if (devWeight != NULL)
					cudaFree( (void *) devWeight );
				if (devGridPosition != NULL)
					cudaFree( (void *) devGridPosition );
				if (devKernelIndex != NULL)
					cudaFree( (void *) devKernelIndex );
				if (hstVisibilitiesInWPlane != NULL)
					free( (void *) hstVisibilitiesInWPlane );

				// free the data.
				if (_hstCacheData == true)
					freeData( /* pWhatData = */ DATA_GRID_POSITIONS | DATA_KERNEL_INDEXES | DATA_DENSITIES | DATA_WEIGHTS );

			}

		}

		// generate the dirty beam by FFTing the gridded data.
		generateDirtyBeam(	/* pdevDirtyBeam = */ &devDirtyBeamGrid,
					/* phstDirtyBeam = */ &hstDirtyBeamGrid,
					/* pFilename = */ outputDirtyBeamFilename );

		// free memory
		if (hstDirtyBeamGrid != NULL)
			free( (void *) hstDirtyBeamGrid );

	}

	// re-cast the dirty beam from complex to doubles/*grid*/.
	float/*grid*/ * devDirtyBeam = (float/*grid*/ *) devDirtyBeamGrid;
	devDirtyBeamGrid = NULL;

	// create the dirty image, and, for file mosaicing only, a cache of dirty images.
	float/*grid*/ * hstDirtyImage = NULL;
	float/*grid*/ * hstDirtyImageCache = NULL;
	if (_hstFileMosaic == true)
	{
		hstDirtyImageCache = (float/*grid*/ *) malloc( (long int) _hstUvPixels * (long int) _hstUvPixels * (long int) _numMosaicImages *
									(long int) sizeof( float/*grid*/ ) );
		memset( hstDirtyImageCache, 0, (long int) _hstUvPixels * (long int) _hstUvPixels * (long int) _numMosaicImages * (long int) sizeof( float/*grid*/ ) );
	}

	printf( "\ngridding visibilities for dirty image.....\n\n" );
	for ( int image = 0; image < _numMosaicImages; image++ )
	{

		if (_hstUseMosaicing == true)
		{

			printf( "\nProcessing mosaic field %i.....\n", image );
			printf( "--------------------------------\n" );

		}

		// declare device memory for the dirty image grid, and zero this memory.
		cufftComplex/*grid*/ * devDirtyImageGrid = NULL;
		reserveGPUMemory( (void **) &devDirtyImageGrid, _hstGPURows * _hstUvPixels * sizeof( cufftComplex/*grid*/ ), "declaring device memory for grid" );
		zeroGPUMemory( (void *) devDirtyImageGrid, _hstGPURows * _hstUvPixels * sizeof( cufftComplex/*grid*/ ), "zeroing the grid on the device" );

		// create a grid on the host. free it if it already exists.
		if (hstDirtyImage != NULL)
		{
			free( (void *) hstDirtyImage );
			hstDirtyImage = NULL;
		}
		cufftComplex/*grid*/ * hstDirtyImageGrid = NULL;
		hstDirtyImageGrid = (cufftComplex/*grid*/ *) malloc( (long int) _hstUvPixels * (long int) _hstUvPixels * (long int) sizeof( cufftComplex/*grid*/ ) );

		// count the total number of visibilities.
		long int totalVisibilities = 0;
		for ( int stageID = 0; stageID < _hstNumberOfStages[ image ]; stageID++ )
			totalVisibilities += _hstNumVisibilities[ image ][ stageID ];

		// uncache the data for this mosaic image (if we have more than one field).
		long int visibilitiesProcessed = 0;
		for ( int stageID = 0; stageID < _hstNumberOfStages[ image ]; stageID++ )
		{

			// get the data from the file.
			if (_hstCacheData == true)
				uncacheData(	/* pFilenamePrefix = */ _hstOutputPrefix,
						/* pMosaicID = */ image,
						/* pStageID = */ stageID,
						/* pWhatData = */ DATA_ALL );
	
			// if the number of visibilities is greater than 100 million then we are going to set a smaller batch size, and load these
			// visibilities in batches.
			int hstVisibilityBatchSize = 0;
			{
				long int nextBatchSize = _hstNumVisibilities[ image ][ stageID ];
				if (nextBatchSize > _hstPreferredVisibilityBatchSize)
					nextBatchSize = _hstPreferredVisibilityBatchSize;
				hstVisibilityBatchSize = (int) nextBatchSize;
			}

			// calculate the number of batches.
			int numberOfBatches = 1;
			if (_hstNumVisibilities[ image ][ stageID ] > _hstPreferredVisibilityBatchSize)
				numberOfBatches = (_hstNumVisibilities[ image ][ stageID ] / hstVisibilityBatchSize) + 1;

			// if the number of minor cycles is zero then we haven't generated a dirty beam, or calculated the number of visibilities per kernel set.
			// we need to do that here, starting with reserving some memory.
			if (_hstMinorCycles == 0)
				_hstVisibilitiesInKernelSet[ image ][ stageID ] = (int *) malloc( _hstKernelSets * numberOfBatches * sizeof( int ) );

			// create space for the visibilities, the unity (psf) visibilities, the density map, and the weights on the device.
			cufftComplex * devVisibility = NULL;
			int * devDensityMap = NULL;
			float * devWeight = NULL;
			VectorI * devGridPosition = NULL;
			int * devKernelIndex = NULL;
			reserveGPUMemory( (void **) &devVisibility, hstVisibilityBatchSize * sizeof( cufftComplex ), "creating device memory for the visibilities" );
			reserveGPUMemory( (void **) &devDensityMap, hstVisibilityBatchSize * sizeof( int ), "declaring device memory for the density map" );
			reserveGPUMemory( (void **) &devGridPosition, hstVisibilityBatchSize * sizeof( VectorI ), "declaring device memory for the grid positions" );
			reserveGPUMemory( (void **) &devKernelIndex, hstVisibilityBatchSize * sizeof( int ), "declaring device memory for the kernel indexes" );
			if (_hstWeighting != NONE)
				reserveGPUMemory( (void **) &devWeight, hstVisibilityBatchSize * sizeof( float ), "declaring device memory for the weights" );

			// keep looping until we have loaded and gridded all visibilities.
			int batch = 0;
			long int hstCurrentVisibility = 0;
			while (hstCurrentVisibility < _hstNumVisibilities[ image ][ stageID ])
			{
		
				// if the number of remaining samples is lower than the sample batch size, then reduce the sample batch size accordingly.
				if (_hstNumVisibilities[ image ][ stageID ] - hstCurrentVisibility < hstVisibilityBatchSize)
					hstVisibilityBatchSize = _hstNumVisibilities[ image ][ stageID ] - hstCurrentVisibility;

				if (_hstNumberOfStages[ image ] > 1 || numberOfBatches > 1)
					printf( "        gridding " );
				if (_hstNumberOfStages[ image ] > 1)
					printf( "host batch %i of %i", stageID + 1, _hstNumberOfStages[ image ] );
				if (_hstNumberOfStages[ image ] > 1 && numberOfBatches > 1)
					printf( ", " );
				if (numberOfBatches > 1)
					printf( "gpu batch %i of %i", batch + 1, numberOfBatches );
				if (_hstNumberOfStages[ image ] > 1 || numberOfBatches > 1)
				{
					int fractionDone = (int) round( (double) visibilitiesProcessed * 30.0 / (double) totalVisibilities );
					int fractionDoing = (int) round( (double) (visibilitiesProcessed + hstVisibilityBatchSize) * 30.0 /
												(double) totalVisibilities );
					printf( " [" );
					for ( int i = 0; i < fractionDone; i++ )
						printf( "*" );
					for ( int i = 0; i < (fractionDoing - fractionDone); i++ )
						printf( "+" );
					for ( int i = 0; i < (30 - fractionDoing); i++ )
						printf( "." );
					printf( "]\n\n" );
					visibilitiesProcessed += hstVisibilityBatchSize;
				}

				// upload the visibility batch size to the device.
				err = cudaMemcpyToSymbol( _devVisibilityBatchSize, &hstVisibilityBatchSize, sizeof( hstVisibilityBatchSize ) );
				if (err != cudaSuccess)
					printf( "error copying visibility batch size to device (%s)\n", cudaGetErrorString( err ) );

				int * hstVisibilitiesInKernelSet = &_hstVisibilitiesInKernelSet[ image ][ stageID ][ batch * _hstKernelSets ];

				// if the number of minor cycles is zero then we haven't generated a dirty beam, so we haven't counted the number of visibilities
				// per kernel set. we must do that here.
				if (_hstMinorCycles == 0)
					calculateVisibilitiesPerKernelSet(	/* pNumVisibilities = */ hstVisibilityBatchSize,
										/* phstGridPosition = */ &_hstGridPosition[ hstCurrentVisibility ],
										/* phstVisibilitiesInKernelSet = */ hstVisibilitiesInKernelSet );

				// upload the visibilities, grid positions, kernel indexes, and density map to the device.
				moveHostToDevice( (void *) devVisibility, (void *) &_hstVisibility[ hstCurrentVisibility ],
							hstVisibilityBatchSize * sizeof( cufftComplex ), "copying visibilities to the device" );
				moveHostToDevice( (void *) devGridPosition, (void *) &_hstGridPosition[ hstCurrentVisibility ], hstVisibilityBatchSize * sizeof( VectorI ),
							"copying grid positions to the device" );
				moveHostToDevice( (void *) devKernelIndex, (void *) &_hstKernelIndex[ hstCurrentVisibility ], hstVisibilityBatchSize * sizeof( int ),
							"copying kernel indexes to the device" );
				moveHostToDevice( (void *) devDensityMap, (void *) &_hstDensityMap[ hstCurrentVisibility ], hstVisibilityBatchSize * sizeof( int ),
							"copying density map to the device" );

				// upload weights to the device.
				if (_hstWeighting != NONE)
					moveHostToDevice( (void *) devWeight, (void *) &_hstWeight[ hstCurrentVisibility ], hstVisibilityBatchSize * sizeof( float ),
							"copying weights to the device" );

				err = cudaGetLastError();
				if (err != cudaSuccess)
					printf( "unknown CUDA error on line %d (%s)\n", __LINE__, cudaGetErrorString( err ) );
	
				// grid visibilities.
				gridVisibilities(	/* pdevGrid = */ devDirtyImageGrid,
							/* phstGrid = */ (_hstImageBatches > 1 ? hstDirtyImageGrid : NULL),
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
							/* phstPrimaryBeamMosaicing = */ _hstPrimaryBeamMosaicingPtr[ image ],
							/* phstPrimaryBeamAProjection = */ _hstPrimaryBeamAProjectionPtr[ image ],
							/* pNumFields = */ (_hstBeamMosaic == true ? _hstNumFieldsForBeamMosaic : -1),
							/* pMosaicIndex = */ image,
							/* pSize = */ _hstUvPixels );
		
				// get the next batch of data.
				hstCurrentVisibility += hstVisibilityBatchSize;
				batch = batch + 1;
		
			}

			// free memory.
			if (devVisibility != NULL)
				cudaFree( (void *) devVisibility );
			if (devDensityMap != NULL)
				cudaFree( (void *) devDensityMap );
			if (devWeight != NULL)
				cudaFree( (void *) devWeight );
			if (devGridPosition != NULL)
				cudaFree( (void *) devGridPosition );
			if (devKernelIndex != NULL)
				cudaFree( (void *) devKernelIndex );

			// free the data.
			if (_hstCacheData == true)
				freeData( /* pWhatData = */ DATA_ALL );

		}
	
		// copy the gridded data to the host, but only if we're not gridding in batches. If we're using batches then it will already be on the host.
		if (_hstImageBatches == 1)
			moveDeviceToHost( (void *) hstDirtyImageGrid, (void *) devDirtyImageGrid, _hstUvPixels * _hstUvPixels * sizeof( cufftComplex/*grid*/ ),
						"copying gridded data from device" );
	
		// save gridded data image, but only if we're not mosaicing in the image domain - we don't want to save multiple gridded file.
// we're no longer saving the grid because casacore would need to either make a copy of the grid (with floats rather than complex) or else
// rearrange the grid rendering it unsuitable for generating a dirty image.
//		if (_hstFileMosaic == false)
//		{
//			float * hstTemp = (float *) malloc( _hstUvPixels * _hstUvPixels * sizeof( float ) );
//			for ( long int i = 0; i < (long int) _hstUvPixels * (long int) _hstUvPixels; i++ )
//				hstTemp[ i ] = hstDirtyImageGrid[ i ].x;
//			_hstCasacoreInterface.WriteCasaImage(	/* pFilename = */ outputGriddedFilename,
//								/* pWidth = */ _hstUvPixels,
//								/* pHeight = */ _hstUvPixels,
//								/* pRA = */ 0.0,
//								/* pDec = */ 0.0,
//								/* pPixelSize = */ _hstCellSize,
//								/* pImage = */ hstTemp,
//								/* pFrequency = */ CONST_C / _hstAverageWavelength[ image ],
//								/* pMask = */ NULL );
//			free( (void *) hstTemp );
//		}

		// free the dirty image grid, but only if we're doing the fft on the device.
		if (_hstImageBatches == 1)
			free( (void *) hstDirtyImageGrid );

		double normalisation = 1.0;

		// we normalise the image by the number of gridded visibilities, but only if we're not using beam mosaicing. Beam mosaicing will do the normalisation
		// using the kernel.
		if (_hstBeamMosaic == false)
			normalisation *= (double) _hstGriddedVisibilities[ image ];
		else
			normalisation *= (double) _griddedVisibilitiesForBeamMosaic;

		if (_hstWeighting != NONE)
			normalisation *= _hstAverageWeight[ image ];

		printf( "\n        performing fft on dirty image grid.....\n" );

		// FFT the gridded data to get the dirty image. We can do this on the device or on the host, depending on how big it is.
		if (_hstImageBatches == 1)
		{
	
			// make dirty image on the device.
			performFFT(	/* pdevGrid = */ &devDirtyImageGrid,
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
				float/*grid*/ * devDirtyImageDbl = (float/*grid*/ *) devDirtyImageGrid;
				devNormalise<<< blocks, threads >>>( &devDirtyImageDbl[ i * MAX_THREADS ], normalisation, itemsThisStage );

			}
		
			// define the block/thread dimensions.
			setThreadBlockSize2D( _hstUvPixels, _hstUvPixels );

			// divide the dirty image by the deconvolution image.
			devDivideImages<<< _gridSize2D, _blockSize2D >>>(	/* pOne = */ (float/*grid*/ *) devDirtyImageGrid,
										/* pTwo = */ _devDeconvolutionImage,
										/* pMask = */ NULL,
										/* pSizeOne = */ _hstUvPixels,
										/* pSizeTwo = */ _hstPsfSize );

			// for beam mosaicing, divide the dirty image by the normalisation pattern.
			if (_hstBeamMosaic == true)
				devDivideImages<<< _gridSize2D, _blockSize2D >>>(	/* pOne = */ (float/*grid*/ *) devDirtyImageGrid,
											/* pTwo = */ devNormalisationPattern,
											/* pMask = */ NULL,
											/* pSizeOne = */ _hstUvPixels,
											/* pSizeTwo = */ _hstBeamSize );

			// copy the dirty image to the host.
			hstDirtyImage = (float/*grid*/ *) malloc( _hstUvPixels * _hstUvPixels * sizeof( float/*grid*/ ) );
			moveDeviceToHost( (void *) hstDirtyImage, (void *) devDirtyImageGrid, _hstUvPixels * _hstUvPixels * sizeof( float/*grid*/ ),
						"copying dirty image from device" );

		}
		else
		{

			// perform FFT on the host.
			performFFT_host(	/* phstGrid = */ (cufftComplex/*grid*/ **) &hstDirtyImageGrid,
						/* pSize = */ _hstUvPixels,
						/* pFFTDirection = */ INVERSE,
						/* pFFTType = */ C2F );

			// reassign the dirty image as a float/*grid*/.
			hstDirtyImage = (float/*grid*/ *) hstDirtyImageGrid;
			hstDirtyImageGrid = NULL;

			// normalise the image by the number of pixels.
			for ( long int i = 0; i < (long int) _hstUvPixels * (long int) _hstUvPixels; i++ )
				hstDirtyImage[ i ] /= normalisation;

			// do normalisation by dividing each pixel by the deconvolution image.
			divideImage(	/* pImageOne = */ hstDirtyImage,
					/* pImageTwo = */ _hstDeconvolutionImage,
					/* pSizeOne = */ _hstUvPixels,
					/* pSizeTwo = */ _hstPsfSize );

			// for beam mosaicing, divide the dirty image by the normalisation pattern.
			if (_hstBeamMosaic == true)
				divideImage(	/* pImageOne = */ hstDirtyImage,
						/* pImageTwo = */ _hstNormalisationPattern,
						/* pSizeOne = */ _hstUvPixels,
						/* pSizeTwo = */ _hstBeamSize );

		}

		printf( "\n" );

		// free memory.
		if (devDirtyImageGrid != NULL)
			cudaFree( (void *) devDirtyImageGrid );

		// if we're file mosaicing then copy the dirty image into the cache.
		if (_hstFileMosaic == true)
			memcpy( &hstDirtyImageCache[ (long int) _hstUvPixels * (long int) _hstUvPixels * (long int) image ],
				hstDirtyImage, (long int) _hstUvPixels * (long int) _hstUvPixels * (long int) sizeof( float/*grid*/ ) );

	}

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
		float/*grid*/ * devCleanBeam = NULL;
		reserveGPUMemory( (void **) &devCleanBeam, _hstPsfSize * _hstPsfSize * sizeof( float/*grid*/ ), "declaring device memory for clean beam" );

		err = cudaGetLastError();
		if (err != cudaSuccess)
			printf( "unknown CUDA error on line %d (%s)\n", __LINE__, cudaGetErrorString( err ) );
		
		// generate the clean beam (relies on the dirty beam already being within device memory).
		generateCleanBeam(	/* pdevCleanBeam = */ devCleanBeam,
					/* pdevDirtyBeam = */ devDirtyBeam,
					/* pFilename = */ outputCleanBeamFilename );

		cudaDeviceSynchronize();
		
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
		for ( int i = 0; i < _numMosaicImages; i++ )
			if (_hstVisibilitiesInKernelSet[ i ] != NULL)
			{
				for ( int stageID = 0; stageID < _hstNumberOfStages[ i ]; stageID++ )
					if (_hstVisibilitiesInKernelSet[ i ][ stageID ] != NULL)
						free( (void *) _hstVisibilitiesInKernelSet[ i ][ stageID ] );
				free( (void *) _hstVisibilitiesInKernelSet[ i ] );
			}
		free( (void *) _hstVisibilitiesInKernelSet );
	}
	if (_hstKernelSize != NULL)
		free( (void *) _hstKernelSize );
	if (_hstSupportSize != NULL)
		free( (void *) _hstSupportSize );
	if (_hstAverageWeight != NULL)
		free( (void *) _hstAverageWeight );
	if (_hstNumberOfStages != NULL)
		free( (void *) _hstNumberOfStages );
	if (_hstDeconvolutionImage != NULL)
		free( (void *) _hstDeconvolutionImage );
	if (hstImagePhaseCentre != NULL)
		free( (void *) hstImagePhaseCentre );
	if (_hstPrimaryBeam != NULL)
		free( (void *) _hstPrimaryBeam );
	if (_hstPrimaryBeamAProjection != NULL)
		free( (void *) _hstPrimaryBeamAProjection );
	if (_hstMosaicID != NULL)
		free( (void *) _hstMosaicID );
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
	if (hstDirtyImage != NULL)
		free( (void *) hstDirtyImage );
	if (hstMask != NULL)
		free( (void *) hstMask );
	if (_hstPrimaryBeamPtr != NULL)
		free( (void *) _hstPrimaryBeamPtr );
	if (_hstPrimaryBeamMosaicingPtr != NULL)
		free( (void *) _hstPrimaryBeamMosaicingPtr );
	if (_hstPrimaryBeamAProjectionPtr != NULL)
		free( (void *) _hstPrimaryBeamAProjectionPtr );

	return true;
	
} // main

