
// include the header file.
#include "cuKernelSet.h"

using namespace std;

//
//	N O N   C L A S S   M E M B E R S
//

//
//	spheroidalWaveFunction()
//
//	CJS: 09/04/2019
//
//	Returns the prolate spheroidal wave function at a distance pR from the centre.
//

__host__ __device__ static double spheroidalWaveFunction( double pR )
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
	if (abs( pR ) <= 1)
	{
						
		int part = 1;
		double radiusEnd = 1.0;
//		if (abs( pR ) < 0.75)
		if (pR < pow( 0.75, 2 ))
		{
			part = 0;
			radiusEnd = 0.75;
		}
							
//		double delRadiusSq = (pR * pR) - (radiusEnd * radiusEnd);
		double delRadiusSq = pR - (radiusEnd * radiusEnd);
							
		double top = 0;
		for ( int k = 0; k < NP; k++ )
			top = top + (dataP[ part ][ k ] * pow( delRadiusSq, k ));
			
		double bottom = 0;
		for ( int k = 0; k < NQ; k++ )
			bottom = bottom + (dataQ[ part ][ k ] * pow( delRadiusSq, k ));
					
		if (bottom != 0)
			val = top / bottom;
							
		// the gridding function is (1 - spheroidRadius^2) x gridsf
//		val = val * (1 - (pR * pR));
//		val = val * (1 - pR);

	}

	// return something.
	return val;

} // spheroidalWaveFunction

//
//	devBuildKernelCutoffHistogram()
//
//	CJS: 10/12/2021
//
//	Build a histogram of support sizes for the pixels above the threshold value.
//

__global__ void devBuildKernelCutoffHistogram
			(
			cufftComplex * pKernel,			// the complex kernel
			int pSize,					// the kernel image size
			double * pMaxValue,				// the maximum absolute value in the kernel
			double pCutoffFraction,			// the fraction of maximum value to search for
			int * pHistogram,				// the histogram (pre-zero'd)
			int pMaxSupport				// the maximum support size (i.e. size of the histogram)
			)
{
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;

	// ensure we are within the kernel image.
	if (i < pSize && j < pSize)
	{

		// set our search value to N% of the kernel maximum.
		double searchValue = pMaxValue[ MAX_PIXEL_VALUE ] * pCutoffFraction;

		// get pixel value.
		float pixelValue = cuCabsf( pKernel[ (j * pSize) + i ] );
		if (pixelValue >= searchValue)
		{

			// get the size of the kernel to this point.
			int supportX = abs( i - (pSize / 2) );
			int supportY = abs( j - (pSize / 2) );
			int support = supportX;
			if (supportY > supportX)
				support = supportY;
				
			// the histogram element (maxSupport + 1) is used for any support size that is larger than the maximum value.
			if (support > pMaxSupport)
				support = pMaxSupport + 1;
				
			// update the histogram for this support size.
			atomicAdd( &pHistogram[ support ], 1 );

		}
	
	}

} // devBuildKernelCutoffHistogram

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
		
		int kernelSupport = (pKernelSize - 1) / 2;
		int workspaceSupport = pWorkspaceSize / 2;

		// i and j are the coordinates within the workspace. get the equivalent coordinates within the aa-kernel.
		int imageI = i - workspaceSupport + kernelSupport;
		int imageJ = j - workspaceSupport + kernelSupport;
		
		// ensure we're within the bounds of the kernel.
		if (imageI >= 0 && imageI < pKernelSize && imageJ >= 0 && imageJ < pKernelSize)
		{

					
			// calculate the x-offset from the centre of the kernel.
			double x = (double) (imageI - kernelSupport);
			double y = (double) (imageJ - kernelSupport);
					
			// now, calculate the anti-aliasing kernel.
//			double val = spheroidalWaveFunction( x / (double) kernelSupport );
//			val *= spheroidalWaveFunction( y / (double) kernelSupport );
			double val = spheroidalWaveFunction( (pow( x, 2 ) + pow( y, 2 )) / pow( (double) kernelSupport, 2 ) );

			// update the appropriate pixel in the anti-aliasing kernel.
//			cufftComplex * aaKernelPtr = &pAAKernel[ (j * pWorkspaceSize) + i ];
			cufftComplex aaKernel = { .x = (float) val, .y = 0.0 };
//			aaKernelPtr->x = (float) val;
//			aaKernelPtr->y = 0;
			pAAKernel[ (j * pWorkspaceSize) + i ] = aaKernel;
		
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
		cufftDoubleComplex kernelValue = { .x = 0.0, .y = 0.0 };
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
			sincos( exponent, &(kernelValue.y), &(kernelValue.x) );
//			kernelValue.x = cos( 2.0 * PI * pW * (sqrt( 1.0 - rSquared ) - 1.0) ) * sqrt( 1.0 - rSquared );
//			kernelValue.y = (pGridDegrid == GRID ? -1 : +1) * sin( 2.0 * PI * pW * (sqrt( 1.0 - rSquared ) - 1.0) ) * sqrt( 1.0 - rSquared );

			// if we are gridding then we multiply the kernel by sqrt( 1.0 - l^2 - m^2 ). If degridding then this is a division.
			// NOTE: Neither Tim Cornwell's paper, nor ASKAPsoft, includes the bits below.
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
//		pWKernel[ (j * pWorkspaceSize) + i ] = kernelValue;
		pWKernel[ (j * pWorkspaceSize) + i ].x = (float) kernelValue.x;
		pWKernel[ (j * pWorkspaceSize) + i ].y = (float) kernelValue.y;

	}

} // devGenerateWKernel

//
//	devGenerateAKernel()
//
//	CJS: 08/11/2018
//
//	Generate the A-kernel in parallel.
//

__global__ void devGenerateAKernel
			(						//
			cufftComplex * pAKernel,			//
			cufftComplex * pPrimaryBeamAProjection,	//
			int pPrimaryBeamSize,				//
			int pWorkspaceSize				//
			)
{

	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;

	// ensure we're within the bounds of the kernel.
	if (i < pWorkspaceSize && j < pWorkspaceSize)
	{

		// calculate the required position within the primary beam.
		int iBeam = (int) floor( (double) i * (double) pPrimaryBeamSize / (double) pWorkspaceSize );
		int jBeam = (int) floor( (double) j * (double) pPrimaryBeamSize / (double) pWorkspaceSize );
		double fracI = ((double) i * (double) pPrimaryBeamSize / (double) pWorkspaceSize) - (double) iBeam;
		double fracJ = ((double) j * (double) pPrimaryBeamSize / (double) pWorkspaceSize) - (double) jBeam;

		// update kernel.
		pAKernel[ (j * pWorkspaceSize) + i ] = interpolateBeam(	/* pBeam = */ pPrimaryBeamAProjection,
										/* pBeamSize = */ pPrimaryBeamSize,
										/* pI = */ iBeam,
										/* pJ = */ jBeam,
										/* pFracI = */ fracI,
										/* pFracJ = */ fracJ );

	}

} // devGenerateAKernel

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

		// if we are gridding a mosaic then we use the primary beam pattern as the kernel. this is so that we add each field such that it is weighted by its
		// primary beam.
		// we then divide the final image by the sum of primary beams squared (beam1^2 + beam2^2 + ... ), which we hold in _hstNormalisationPattern, in order
		// to correct for our weighting function and also to remove the effect of the primary beam which will naturally be in our image.
		//
		// if we are gridding with a-projection then we use PB^2 because a-projection will divide by the beam pattern.
		//
		// if we are degridding then we simply want to reintroduce the primary beam, so we use: val = beam<i>.
		if (pGridDegrid == GRID && pAProjection == true)

			// update the beam.
			pImage[ (j * pSize) + i ] = pow( pImage[ (j * pSize) + i ], 2 );

	}

} // devSetPrimaryBeamForGriddingAndDegridding

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
		int imageI = ((i - pSupport) * pOversample) + workspaceSupport;
		int imageJ = ((j - pSupport) * pOversample) + workspaceSupport;
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
		cufftComplex value = { .x = 0.0, .y = 0.0 };
		if (imageI >= 0 && imageI < pWorkspaceSize && imageJ >= 0 && imageJ < pWorkspaceSize)
			value = pImage[ (imageJ * pWorkspaceSize) + imageI ];

		// update the kernel.
		pKernel[ (j * kernelSize) + i ] = value;
		
	}
	
} // devUpdateKernel

//
//	P U B L I C   C L A S S   M E M B E R S
//

//
//	KernelSet::KernelSet()
//
//	CJS: 23/02/2022
//
//	The constructor.
//

KernelSet::KernelSet()
{

	// create instances of casacore
	_casacoreInterface = CasacoreInterface::getInstance();

	// create instances of casacore
	_param = Parameters::getInstance();
	
	// initialise variables.
	oversample = 1;
	kernelSize = -1;
	supportSize = -1;
	kernel = NULL;
	visibilities = NULL;
	_numberOfStages = 0;
	_numberOfBatches = NULL;

} // KernelSet::KernelSet

KernelSet::KernelSet
			(
			int pOversample
			)
{

	// create instances of casacore
	_casacoreInterface = CasacoreInterface::getInstance();

	// create instances of casacore
	_param = Parameters::getInstance();
	
	// initialise variables.
	oversample = pOversample;
	kernelSize = -1;
	kernel = NULL;
	visibilities = NULL;
	_numberOfStages = 0;
	_numberOfBatches = NULL;

} // KernelSet::KernelSet

//
//	KernelSet::~KernelSet()
//
//	CJS: 23/02/2022
//
//	The destructor.
//

KernelSet::~KernelSet()
{

	if (kernel != NULL)
		free( (void *) kernel );
	if (visibilities != NULL)
	{
		for ( int stageID = 0; stageID < _numberOfStages; stageID++ )
			if (visibilities[ stageID ] != NULL)
				if (_numberOfBatches != NULL)
				{
					for ( int batch = 0; batch < _numberOfBatches[ stageID ]; batch++ )
						if (visibilities[ stageID ][ batch ] != NULL)
							free( (void *) visibilities[ stageID ][ batch ] );
					free( (void *) visibilities[ stageID ] );
				}
		free( (void *) visibilities );
	}
	if (_numberOfBatches != NULL)
		free( (void *) _numberOfBatches );

} // KernelSet::~KernelSet

//
//	createArrays()
//
//	CJS: 25/02/2022
//
//	Create the required arrays for storing the number of visibilities.
//

void KernelSet::createArrays( int pNumberOfStages, int * pNumberOfBatches, int pNumGPUs )
{

	_numberOfStages = pNumberOfStages;
	_numberOfBatches = (int *) malloc( _numberOfStages * sizeof( int ) );
	visibilities = (int ***) malloc( _numberOfStages * sizeof( int ** ) );
	memcpy( _numberOfBatches, pNumberOfBatches, _numberOfStages * sizeof( int ) );

	for ( int stageID = 0; stageID < _numberOfStages; stageID++ )
	{

		visibilities[ stageID ] = (int **) malloc( _numberOfBatches[ stageID ] * sizeof( int * ) );
		for ( int batchID = 0; batchID < _numberOfBatches[ stageID ]; batchID++ )
		{

			visibilities[ stageID ][ batchID ] = (int *) malloc( pNumGPUs * sizeof( int ) );
			for ( int gpu = 0; gpu < pNumGPUs; gpu++ )
				visibilities[ stageID ][ batchID ][ gpu ] = 0;

		}

	}

} // KernelSet::createArrays

//
//	findSupportForKernel()
//
//	CJS: 10/12/2021
//
//	Find an appropriate support size that encloses some fixed fraction of the pixels with values greater than the cutoff threshold.
//

int KernelSet::findSupportForKernel
			(
			cufftComplex * pdevKernel,			// the complex kernel.
			int pSize,					// the kernel size.
			double * pdevMaxValue,				// hold the maximum absolute value of the kernel.
			double pCutoffFraction,			// the fraction of the absolute value that we want to keep.
			int pMaxSupport				// the maximum oversampled support size.
			)
			
{

	// create memory to store the number of pixels with values in this range.
	int * devHistogram = NULL;
	reserveGPUMemory( (void **) &devHistogram, (pMaxSupport + 2) * sizeof( int ), "declaring device memory for kernel cutoff histogram", __LINE__ );
	zeroGPUMemory( (void *) devHistogram, (pMaxSupport + 2) * sizeof( int ), "clearing kernel cutoff histogram on the device", __LINE__ );
	
	// build a histogram of support sizes for the pixels above the threshold.
	setThreadBlockSize2D( pSize, pSize, _gridSize2D, _blockSize2D );
	devBuildKernelCutoffHistogram<<< _gridSize2D, _blockSize2D >>>(	/* pKernel = */ pdevKernel,
										/* pSize = */ pSize,
										/* pMaxValue = */ pdevMaxValue,
										/* pCutoffFraction = */ pCutoffFraction,
										/* pHistogram = */ devHistogram,
										/* pMaxSupport = */ pMaxSupport );
										
	// download histogram.
	int * hstHistogram = (int *) malloc( (pMaxSupport + 2) * sizeof( int ) );
	moveDeviceToHost( (void *) hstHistogram, (void *) devHistogram, (pMaxSupport + 2) * sizeof( int ), "moving kernel cutoff histogram to the host", __LINE__ );
	
	// free memory.
	if (devHistogram != NULL)
		cudaFree( (void *) devHistogram );
	
	// count the pixels in the histogram.
	int totalPixels = 0;
	for ( int i = 0; i <= pMaxSupport + 1; i++ )
		totalPixels += hstHistogram[ i ];
		
	if (totalPixels == 0)
		printf( "WARNING: found an empty kernel\n" );
		
	// find an appropriate support size so that 95% of the pixels are enclosed.
	int supportSize = 0;
	int summedPixels = 0;
	int threshold = (int) ceil( (double) totalPixels * 0.95 );
	for ( int i = 0; i <= pMaxSupport + 1; i++ )
	{
		summedPixels += hstHistogram[ i ];
		if (summedPixels >= threshold)
		{
			supportSize = i;
			break;
		}
	}
	
	// free memory.
	if (hstHistogram != NULL)
		free( (void *) hstHistogram );
		
	// return something.
	return supportSize;

} // KernelSet::findSupportForKernel

//
//	KernelSet::generateKernel
//
//	CJS: 24/03/2022
//
//	Generates a kernel set based upon a w-value, channel, and primary-beam pattern.
//

bool KernelSet::generateKernel
			(
			int pW,						// the W plane to generate kernels for (will be 0 if we're not using W projection).
			int pChannel,						// the A plane to generate kernels for (will be 0 if we're not using A projection).
										// * NOTE: pChannel is only used for debugging - it can be removed
			bool pWProjection,					// true if we're using W projection (DON'T use the global parameter).
			bool pAProjection,					// true if we're using A projection (DON'T use the global parameter).
			float * phstPrimaryBeamMosaicing,			// the primary beam for mosaicing (if needed)
			cufftComplex * phstPrimaryBeamAProjection,		// the primary beam for A projection (if needed)
			float * phstPrimaryBeamRatio,				// the ratio of the primary beam at the maximum wavelength to the primary beam at the
										//	a specific wavelength. this image is used for correcting the channels to the same flux.
			int pBeamSize,						// the size of the primary beam images.
			Data * phstData,					// the object holding the data for this mosaic component
			griddegrid pGridDegrid,				// either GRID or DEGRID
			bool * pKernelOverflow,				// we set this parameter to true if the kernel needs to be truncated due to size.
			double pAverageWeight,					// the average weight for kernel normalisation
			bool pUVPlaneMosaic					// true if we're making a UV-plane mosaic
			)
{

	// we only use a maximum fraction of the available memory to create a workspace.
	const int WORKSPACE_SIZE = 4096;
	
	bool ok = true;
	cudaError_t err;

	// for uv-plane mosaicing we need to do some primary beam correction using the average (mosaic) primary beam. we only have to do this for degridding if we're not
	// using A-projection.
	bool doMosaicing = (phstPrimaryBeamMosaicing != NULL && (pGridDegrid == GRID || (pGridDegrid == DEGRID && pAProjection == false)));
	
	// if we're degridding without mosaicing or a-projection then we need to correct the flux by multiplying by the ratio of the primary beam at the maximum wavelength
	// to the primary beam at a specific wavelength.
	bool doFluxCorrection = (doMosaicing == false && pAProjection == false && pGridDegrid == DEGRID && phstPrimaryBeamRatio != NULL);
	
	// are we doing A-projection?
	bool doAProjection = (pAProjection == true && phstPrimaryBeamAProjection != NULL);
	
	// are we doing W-projection?
	bool doWProjection = (pWProjection == true && phstData != NULL);

	int numberOfWorkspacesRequired = 1;
	if (doMosaicing == true)
		numberOfWorkspacesRequired++;
	if (doWProjection == true)
		numberOfWorkspacesRequired++;
	if (doAProjection == true)
		numberOfWorkspacesRequired++;
	if (doFluxCorrection == true)
		numberOfWorkspacesRequired++;

	// if we are using uv-plane mosaicing then we need to correct the image for the primary beam, and weight the image for mosaicing.
	cufftComplex * devBeamCorrection = NULL;
	if (doMosaicing == true)
	{

		// create space for the primary beam on the device, and copy it across.
		float * devPrimaryBeam = NULL;
		reserveGPUMemory( (void **) &devPrimaryBeam, pBeamSize * pBeamSize * sizeof( float ),
					"declaring device memory for primary beam", __LINE__ );
		moveHostToDevice( (void *) devPrimaryBeam, (void *) phstPrimaryBeamMosaicing, pBeamSize * pBeamSize * sizeof( float ),
					"copying primary beam to the device", __LINE__ );

		// set the primary beam during gridding and degridding. we tell this subroutine if we're using A-projection or not because in the absence of A-projection
		// we will need to correct for the primary beam function using same average beam that we use to weight the mosaic.
		setThreadBlockSize2D( pBeamSize, pBeamSize, _gridSize2D, _blockSize2D );
		devSetPrimaryBeamForGriddingAndDegridding<<< _gridSize2D, _blockSize2D >>>(	/* pImage = */ devPrimaryBeam,
												/* pSize = */ pBeamSize,
												/* pGridDegrid = */ pGridDegrid,
												/* pAProjection = */ pAProjection );

		// create the kernel and clear it.
		reserveGPUMemory( (void **) &devBeamCorrection, WORKSPACE_SIZE * WORKSPACE_SIZE * sizeof( cufftComplex ),
					"declaring device memory for the beam-correction kernel", __LINE__ );
		zeroGPUMemory( (void *) devBeamCorrection, WORKSPACE_SIZE * WORKSPACE_SIZE * sizeof( cufftComplex ), "clearing beam-correction kernel on the device", __LINE__ );

		// copy the kernel into the workspace.
		setThreadBlockSize2D( WORKSPACE_SIZE, WORKSPACE_SIZE, _gridSize2D, _blockSize2D );
		devScaleImage<<< _gridSize2D, _blockSize2D >>>(	/* pNewImage = */ devBeamCorrection,
									/* pOldImage = */ devPrimaryBeam,
									/* pNewSize = */ WORKSPACE_SIZE,
									/* pOldSize = */ pBeamSize,
									/* pScale = */ (double) pBeamSize / (double) WORKSPACE_SIZE );
		
		// free the primary beam.
		if (devPrimaryBeam != NULL)
			cudaFree( (void *) devPrimaryBeam );

//{

//float * tmpKernel = (float *) malloc( WORKSPACE_SIZE * WORKSPACE_SIZE * sizeof( cufftComplex ) );
//cudaMemcpy( tmpKernel, devBeamCorrection, WORKSPACE_SIZE * WORKSPACE_SIZE * sizeof( cufftComplex ), cudaMemcpyDeviceToHost );
//for ( long int i = 0; i < (long int) WORKSPACE_SIZE * (long int) WORKSPACE_SIZE; i++ )
//	tmpKernel[ i ] = tmpKernel[ i * 2 ];
//char kernelFilename[100];
//sprintf( kernelFilename, "beam-correction-%i-%i", pMosaicID, (pGridDegrid == GRID ? 0 : 1) );
//_hstCasacoreInterface.WriteCasaImage( kernelFilename, WORKSPACE_SIZE, WORKSPACE_SIZE, _hstOutputRA,
//					_hstOutputDEC, _param.CellSize * (double) _param.Oversample, tmpKernel, CONST_C / phstData->AverageWavelength, NULL );
//free( tmpKernel );

//}

	} // (doMosaicing == true)

	// are we using W-projection ?
	cufftComplex * devWKernel = NULL;
	if (doWProjection == true)
	{

		// create w-kernel.
		reserveGPUMemory( (void **) &devWKernel, WORKSPACE_SIZE * WORKSPACE_SIZE * sizeof( cufftComplex ), "declaring device memory for w-kernel", __LINE__ );

		// convert the cell size from arcseconds to radians, and take the sine to get the directional cosine.
		double cellSizeDirectionalCosine = sin( (_param->CellSize * (double) _param->Oversample / 3600.0) * (PI / 180.0) );

		// generate the W-kernel on the GPU.
		setThreadBlockSize2D( WORKSPACE_SIZE, WORKSPACE_SIZE, _gridSize2D, _blockSize2D );
		devGenerateWKernel<<< _gridSize2D, _blockSize2D >>>(	/* pWKernel = */ devWKernel,
									/* pW = */ phstData->WPlaneMean[ pW ],
									/* pWorkspaceSize = */ WORKSPACE_SIZE,
									/* pCellSizeDirectionalCosine = */ cellSizeDirectionalCosine,
									/* pGridDegrid = */ pGridDegrid,
									/* pSize = */ _param->ImageSize );

	} // (pWProjection == true)

	// are we using A-projection ?
	cufftComplex * devAKernel = NULL;
	if (doAProjection == true)
	{

		// reserve some memory for the A kernel and primary beam.
		cufftComplex * devPrimaryBeamAProjection = NULL;
		reserveGPUMemory( (void **) &devAKernel, WORKSPACE_SIZE * WORKSPACE_SIZE * sizeof( cufftComplex ), "declaring device memory for a-kernel", __LINE__ );
		reserveGPUMemory( (void **) &devPrimaryBeamAProjection, pBeamSize * pBeamSize * sizeof( cufftComplex ), "declaring device memory for the primary beam",
						__LINE__ );

		// upload the primary beam to the device.
		moveHostToDevice( (void *) devPrimaryBeamAProjection, (void *) phstPrimaryBeamAProjection, pBeamSize * pBeamSize * sizeof( cufftComplex ),
						"uploading primary beam to the device", __LINE__ );

		// generate the A kernel on the device. The A kernel should be the inverse of the primary beam (1 / pbeam) for gridding, and the primary beam for degridding.
		setThreadBlockSize2D( WORKSPACE_SIZE, WORKSPACE_SIZE, _gridSize2D, _blockSize2D );
		devGenerateAKernel<<< _gridSize2D, _blockSize2D >>>(	/* pAKernel = */ devAKernel,
									/* pPrimaryBeamAProjection = */ devPrimaryBeamAProjection,
									/* pPrimaryBeamSize = */ pBeamSize,
									/* pWorkspaceSize = */ WORKSPACE_SIZE );
		err = cudaGetLastError();
		if (err != cudaSuccess)
			printf( "error generating A kernel (%s)\n", cudaGetErrorString( err ) );

		// free the primary beam.
		if (devPrimaryBeamAProjection != NULL)
			cudaFree( (void *) devPrimaryBeamAProjection );

//if (pW == 0 && pChannel % 5 == 0)
//{

//float * tmpKernel = (float *) malloc( WORKSPACE_SIZE * WORKSPACE_SIZE * sizeof( cufftComplex ) );
//cudaMemcpy( tmpKernel, devAKernel, WORKSPACE_SIZE * WORKSPACE_SIZE * sizeof( cufftComplex ), cudaMemcpyDeviceToHost );
//for ( long int i = 0; i < (long int) WORKSPACE_SIZE * (long int) WORKSPACE_SIZE; i++ )
//	tmpKernel[ i ] = tmpKernel[ i * 2 ];
//char kernelFilename[100];
//sprintf( kernelFilename, "a-kernel-%i-%i-real", pChannel, (pGridDegrid == GRID ? 0 : 1) );
//_casacoreInterface->WriteCasaImage( kernelFilename, WORKSPACE_SIZE, WORKSPACE_SIZE, 0.0,
//					0.0, _param->CellSize * (double) _param->Oversample, tmpKernel, CONST_C / phstData->AverageWavelength, NULL, CasacoreInterface::J2000, 1 );
//cudaMemcpy( tmpKernel, devAKernel, WORKSPACE_SIZE * WORKSPACE_SIZE * sizeof( cufftComplex ), cudaMemcpyDeviceToHost );
//for ( long int i = 0; i < (long int) WORKSPACE_SIZE * (long int) WORKSPACE_SIZE; i++ )
//	tmpKernel[ i ] = tmpKernel[ (i * 2) + 1 ];
//sprintf( kernelFilename, "a-kernel-%i-%i-imag", pChannel, (pGridDegrid == GRID ? 0 : 1) );
//_casacoreInterface->WriteCasaImage( kernelFilename, WORKSPACE_SIZE, WORKSPACE_SIZE, 0.0,
//					0.0, _param.CellSize * (double) _param.Oversample, tmpKernel, CONST_C / phstData->AverageWavelength, NULL, CasacoreInterface::J2000, 1 );
//free( tmpKernel );

//}
	} // (pAProjection == true)
	
	// do we need to correct the flux for this channel by multiplying by the primary-beam ratio?
	cufftComplex * devFluxCorrection = NULL;
	if (doFluxCorrection == true)
	{

		// reserve some memory for the kernel and primary beam.
		cufftComplex * devPrimaryBeamRatio = NULL;
		reserveGPUMemory( (void **) &devFluxCorrection, WORKSPACE_SIZE * WORKSPACE_SIZE * sizeof( cufftComplex ),
						"declaring device memory for a-kernel", __LINE__ );
		reserveGPUMemory( (void **) &devPrimaryBeamRatio, pBeamSize * pBeamSize * sizeof( cufftComplex ),
						"declaring device memory for the primary beam ratio", __LINE__ );

		// and set the thread block size for processing images on the device.
		setThreadBlockSize2D( pBeamSize, pBeamSize, _gridSize2D, _blockSize2D );

		// upload the primary beam to the device.
		float * tmp = NULL;
		reserveGPUMemory( (void **) &tmp, pBeamSize * pBeamSize * sizeof( float ),
						"declaring temporary device memory for the primary beam ratio", __LINE__ );
		moveHostToDevice( (void *) tmp, (void *) phstPrimaryBeamRatio, pBeamSize * pBeamSize * sizeof( float ),
						"uploading primary beam to the device", __LINE__ );
		devConvertImage<<< _gridSize2D, _blockSize2D >>>(	/* pOut = */ devPrimaryBeamRatio,
									/* pIn = */ tmp,
									/* pSize = */ pBeamSize );
		if (tmp != NULL)
			cudaFree( (void *) tmp );

		// generate the kernel on the device.
		setThreadBlockSize2D( WORKSPACE_SIZE, WORKSPACE_SIZE, _gridSize2D, _blockSize2D );
		devScaleImage<<< _gridSize2D, _blockSize2D >>>(	/* pNewImage = */ devFluxCorrection,
									/* pOldImage = */ devPrimaryBeamRatio,
									/* pNewSize = */ WORKSPACE_SIZE,
									/* pOldSize = */ pBeamSize,
									/* pScale = */ (double) pBeamSize / (double) WORKSPACE_SIZE );
		err = cudaGetLastError();
		if (err != cudaSuccess)
			printf( "error generating flux-correction kernel (%s)\n", cudaGetErrorString( err ) );

		// free the primary beam.
		if (devPrimaryBeamRatio != NULL)
			cudaFree( (void *) devPrimaryBeamRatio );
//{

//float * tmpKernel = (float *) malloc( WORKSPACE_SIZE * WORKSPACE_SIZE * sizeof( cufftComplex ) );
//cudaMemcpy( tmpKernel, devFluxCorrection, WORKSPACE_SIZE * WORKSPACE_SIZE * sizeof( cufftComplex ), cudaMemcpyDeviceToHost );
//for ( long int i = 0; i < (long int) WORKSPACE_SIZE * (long int) WORKSPACE_SIZE; i++ )
//	tmpKernel[ i ] = tmpKernel[ i * 2 ];
//char kernelFilename[100];
//sprintf( kernelFilename, "flux-kernel-%i-real", pChannel );
//_casacoreInterface->WriteCasaImage( kernelFilename, WORKSPACE_SIZE, WORKSPACE_SIZE, 0.0,
//					0.0, _param->CellSize * (double) _param->Oversample, tmpKernel, CONST_C / phstData->AverageWavelength, NULL, CasacoreInterface::J2000, 1 );
//free( tmpKernel );

//}
	
	} // (doFluxCorrection == true)
	
	// reserve some memory for the AA kernel and clear it.
	cufftComplex * devAAKernel = NULL;
	
	reserveGPUMemory( (void **) &devAAKernel, WORKSPACE_SIZE * WORKSPACE_SIZE * sizeof( cufftComplex ), "declaring device memory for the aa-kernel", __LINE__ );
	zeroGPUMemory( (void *) devAAKernel, WORKSPACE_SIZE * WORKSPACE_SIZE * sizeof( cufftComplex ), "zeroing aa-kernel on the device", __LINE__ );

	err = cudaGetLastError();
	if (err != cudaSuccess)
		printf( "Unknown CUDA error in generateKernel() (%s)\n", cudaGetErrorString( err ) );

	// generate the AA-kernel on the GPU.
	int aaKernelSize = (int) ceil( sqrt( 2 ) * WORKSPACE_SIZE / (double) oversample );
	if (aaKernelSize % 2 == 0)
		aaKernelSize++;
	setThreadBlockSize2D( WORKSPACE_SIZE, WORKSPACE_SIZE, _gridSize2D, _blockSize2D );
	devGenerateAAKernel<<< _gridSize2D, _blockSize2D >>>(	/* pAAKernel = */ devAAKernel,
									/* pKernelSize = */ aaKernelSize,
									/* pWorkspaceSize = */ WORKSPACE_SIZE );
	err = cudaGetLastError();
	if (err != cudaSuccess)
		printf( "error generating AA kernel (%s)\n", cudaGetErrorString( err ) );
		
//{

//float * tmpKernel = (float *) malloc( WORKSPACE_SIZE * WORKSPACE_SIZE * sizeof( cufftComplex ) );
//cudaMemcpy( tmpKernel, devAAKernel, WORKSPACE_SIZE * WORKSPACE_SIZE * sizeof( cufftComplex ), cudaMemcpyDeviceToHost );
//for ( long int i = 0; i < (long int) WORKSPACE_SIZE * (long int) WORKSPACE_SIZE; i++ )
//	tmpKernel[ i ] = tmpKernel[ i * 2 ];
//char kernelFilename[100];
//sprintf( kernelFilename, "aa-kernel" );
//_hstCasacoreInterface.WriteCasaImage( kernelFilename, WORKSPACE_SIZE, WORKSPACE_SIZE, _hstOutputRA,
//					_hstOutputDEC, _param.CellSize * (double) _param.Oversample, tmpKernel, CONST_C / phstData->AverageWavelength, NULL );
//free( tmpKernel );

//}

	// create a kernel workspace to store the image-plane kernel.
	cufftComplex * devCombinedKernel = NULL;
	reserveGPUMemory( (void **) &devCombinedKernel, WORKSPACE_SIZE * WORKSPACE_SIZE * sizeof( cufftComplex ), "declaring device memory for image-plane kernel", __LINE__ );

//{

//float * tmpKernel = (float *) malloc( WORKSPACE_SIZE * WORKSPACE_SIZE * sizeof( cufftComplex ) );
//cudaMemcpy( tmpKernel, devBeamCorrection, WORKSPACE_SIZE * WORKSPACE_SIZE * sizeof( cufftComplex ), cudaMemcpyDeviceToHost );
//for ( long int i = 0; i < (long int) WORKSPACE_SIZE * (long int) WORKSPACE_SIZE; i++ )
//	tmpKernel[ i ] = tmpKernel[ i * 2 ];
//char kernelFilename[100];
//sprintf( kernelFilename, "beam-correction-%i", pMosaicID );
//_hstCasacoreInterface.WriteCasaImage( kernelFilename, WORKSPACE_SIZE, WORKSPACE_SIZE, _hstOutputRA,
//					_hstOutputDEC, _param.CellSize * (double) _param.Oversample, tmpKernel, CONST_C / phstData->AverageWavelength, NULL );
//free( tmpKernel );

//}

	// we start off with the anti-aliasing kernel.
	cudaMemcpy( (void *) devCombinedKernel, (void *) devAAKernel, WORKSPACE_SIZE * WORKSPACE_SIZE * sizeof( cufftComplex ), cudaMemcpyDeviceToDevice );

	// we now work with the whole workspace.
	setThreadBlockSize2D( WORKSPACE_SIZE, WORKSPACE_SIZE, _gridSize2D, _blockSize2D );

	// are we using W-projection ? Convolve with the kernel.
	if (doWProjection == true)
		devMultiplyArrays<<< _gridSize2D, _blockSize2D >>>(	/* pOne = */ devCombinedKernel,
									/* pTwo = */ devWKernel,
									/* pSize = */ WORKSPACE_SIZE,
									/* pConjugate = */ false );

	// are we using A-projection ? Convolve with the kernel.
	if (doAProjection == true)
		devMultiplyArrays<<< _gridSize2D, _blockSize2D >>>(	/* pOne = */ devCombinedKernel,
									/* pTwo = */ devAKernel,
									/* pSize = */ WORKSPACE_SIZE,
									/* pConjugate = */ false );

	// are we using beam correction ? Convolve with the kernel.
	if (doMosaicing == true)
		devMultiplyArrays<<< _gridSize2D, _blockSize2D >>>(	/* pOne = */ devCombinedKernel,
									/* pTwo = */ devBeamCorrection,
									/* pSize = */ WORKSPACE_SIZE,
									/* pConjugate = */ false );
									
	// are we doing flux correction ? Convolve with the kernel.
	if (doFluxCorrection == true)
		devMultiplyArrays<<< _gridSize2D, _blockSize2D >>>(	/* pOne = */ devCombinedKernel,
									/* pTwo = */ devFluxCorrection,
									/* pSize = */ WORKSPACE_SIZE,
									/* pConjugate = */ false );
	
	// free memory.
	if (devWKernel != NULL)
		cudaFree( (void *) devWKernel );
	if (devAKernel != NULL)
		cudaFree( (void *) devAKernel );
	if (devBeamCorrection != NULL)
		cudaFree( (void *) devBeamCorrection );
	if (devFluxCorrection != NULL)
		cudaFree( (void *) devFluxCorrection );
	if (devAAKernel != NULL)
		cudaFree( (void *) devAAKernel );

//if (pW == 0 && pChannel % 5 == 0)
//{

//float * tmpKernel = (float *) malloc( WORKSPACE_SIZE * WORKSPACE_SIZE * sizeof( cufftComplex ) );
//cudaMemcpy( tmpKernel, devCombinedKernel, WORKSPACE_SIZE * WORKSPACE_SIZE * sizeof( cufftComplex ), cudaMemcpyDeviceToHost );
//for ( long int i = 0; i < (long int) WORKSPACE_SIZE * (long int) WORKSPACE_SIZE; i++ )
//	tmpKernel[ i ] = tmpKernel[ i * 2 ];
//char kernelFilename[100];
//sprintf( kernelFilename, "combined-kernel-a%i-g%i", pChannel, (pGridDegrid == GRID ? 0 : 1) );
//_casacoreInterface->WriteCasaImage( kernelFilename, WORKSPACE_SIZE, WORKSPACE_SIZE, 0.0,
//					0.0, _param->CellSize * (double) oversample, tmpKernel, 1000.0, NULL, CasacoreInterface::J2000, 1 );
//free( tmpKernel );

//}

	// FFT the convolved kernel into the UV domain.
	performFFT(	/* pdevGrid = */ &devCombinedKernel,
			/* pSize = */ WORKSPACE_SIZE,
			/* pFFTDirection = */ FORWARD,
			/* pFFTPlan = */ -1,
			/* pFFTType = */ C2C,
			/* pResizeArray = */ false );

	// initialise support size.
	supportSize = 5;

	// create a new memory area to hold the maximum pixel value.
	double * devMaxValue;
	reserveGPUMemory( (void **) &devMaxValue, MAX_PIXEL_DATA_AREA_SIZE * sizeof( double ), "declaring device memory for kernel max pixel value", __LINE__ );

	// get the peak value from the kernel.
	getMaxValue(	/* pdevImage = */ devCombinedKernel,
			/* pdevMaxValue = */ devMaxValue,
			/* pWidth = */ WORKSPACE_SIZE,
			/* pHeight = */ WORKSPACE_SIZE,
			/* pIncludeComplexComponent = */ true,
			/* pMultiplyByConjugate = */ false,
			/* pdevMask = */ NULL );
						
	// build a histogram of support sizes for all the pixels above the threshold value. we use this histogram to determine an appropriate support size.
	supportSize = findSupportForKernel(	/* pdevKernel = */ devCombinedKernel,
						/* pSize = */ WORKSPACE_SIZE,
						/* pdevMaxValue = */ devMaxValue,
						/* pCutoffFraction = */ _param->KernelCutoffFraction,
						/* pMaxSupport = */ _param->KernelCutoffSupport * oversample );

	// free memory.
	if (devMaxValue != NULL)
		cudaFree( (void *) devMaxValue );

	// divide the support size by the oversampling factor, and round up.
	supportSize = (int) ceil( (double) supportSize / (double) oversample );

	// ensure the support is at least 5.
	if (supportSize < 5)
		supportSize = 5;

	// ensure support size is not larger than the maximum.
	if (supportSize > _param->KernelCutoffSupport)
	{
		supportSize = _param->KernelCutoffSupport;
		*pKernelOverflow = true;
	}

	// calculate kernel size.
	kernelSize = (supportSize * 2) + 1;

//if (doFluxCorrection == true)
//{

//float * tmpKernel = (float *) malloc( WORKSPACE_SIZE * WORKSPACE_SIZE * sizeof( cufftComplex ) );
//cudaMemcpy( tmpKernel, devCombinedKernel, WORKSPACE_SIZE * WORKSPACE_SIZE * sizeof( cufftComplex ), cudaMemcpyDeviceToHost );
//for ( long int i = 0; i < (long int) WORKSPACE_SIZE * (long int) WORKSPACE_SIZE; i++ )
//	tmpKernel[ i ] = sqrt( pow( tmpKernel[ i * 2 ], 2 ) + pow( tmpKernel[ (i * 2) + 1 ], 2 ) );
//char kernelFilename[100];
//sprintf( kernelFilename, "kernel-%i", pChannel );
//_casacoreInterface->WriteCasaImage( kernelFilename, WORKSPACE_SIZE, WORKSPACE_SIZE, 0.0,
//					0.0, _param->CellSize, tmpKernel, 1000.0, NULL, CasacoreInterface::J2000, 1 );
//free( tmpKernel );

//}

	// kernel data on the device.
	cufftComplex * devKernel = NULL;
	reserveGPUMemory( (void **) &devKernel, kernelSize * kernelSize * oversample * oversample * sizeof( cufftComplex ),
				"declaring device memory for kernel", __LINE__ );

	// define the block/thread dimensions.
	setThreadBlockSize2D( kernelSize, kernelSize, _gridSize2D, _blockSize2D );
					
	// calculate separate kernels for each (oversampled) intermediate grid position.
	for ( int oversampleI = 0; oversampleI < oversample; oversampleI++ )
		for ( int oversampleJ = 0; oversampleJ < oversample; oversampleJ++ )
				
			// copy the kernel from the anti-aliasing kernel into the actual kernel.
			devUpdateKernel<<< _gridSize2D, _blockSize2D >>>(	/* pKernel = */ &devKernel[ (long int) ((oversampleJ * oversample) + oversampleI) *
														(long int) (kernelSize * kernelSize) ],
										/* pImage = */ devCombinedKernel,
										/* pSupport = */ supportSize,
										/* pOversample = */ oversample,
										/* pOversampleI = */ oversampleI,
										/* pOversampleJ = */ oversampleJ,
										/* pWorkspaceSize = */ WORKSPACE_SIZE,
										/* pGridDegrid = */ pGridDegrid );
			
	// define the block/thread dimensions.
	int threads = kernelSize * kernelSize * oversample * oversample, blocks = 1;
	setThreadBlockSize1D( &threads, &blocks );
					
	// normalise the image following the FFT by dividing by the number of pixels.
	devNormalise<<< blocks, threads >>>( devKernel, (double) (WORKSPACE_SIZE * WORKSPACE_SIZE / (oversample * oversample)),
												kernelSize * kernelSize * oversample * oversample );
						
	// normalise the image again by dividing by the average weight.
	devNormalise<<< blocks, threads >>>( devKernel, pAverageWeight, kernelSize * kernelSize * oversample * oversample );

	// for UV mosaics, normalise the image following the FFT by dividing by the number of gridded visibilities. We actually do most of this normalisation in the image
	// domain:
	// The normalisation factor, N_i, will be different for each field, and we find the lowest N_i, which we call N_min and hold in _griddedVisibilitiesForUVPlaneMosaic,
	// and take it out as a factor. We correct for N_min in the image domain, and here we will correct for N_i / N_min.
	if (phstPrimaryBeamMosaicing != NULL && pGridDegrid == GRID && pUVPlaneMosaic == true && phstData != NULL)
		devNormalise<<< blocks, threads>>>(	devKernel,
							(double) phstData->GriddedVisibilities / (double) phstData->MinimumVisibilitiesInMosaic,
							kernelSize * kernelSize * oversample * oversample );


//	{
//		float * tmpKernel = (float *) malloc( kernelSize * kernelSize * sizeof( cufftComplex ) );
//		for ( int oversampleI = 0; oversampleI < oversample; oversampleI++ )
//			for ( int oversampleJ = 0; oversampleJ < pOvoversampleersample; oversampleJ++ )
//			{
//				
//				// get the index of the oversampled kernels. no need to add the index of the w-kernel because
//				// we're putting them in separate arrays.
//				long int kernelIdx = ((long int) oversampleI * (long int) kernelSize * (long int) kernelSize) +
//							((long int) oversampleJ * (long int) kernelSize * (long int) kernelSize * (long int) oversample);
//
//				cudaMemcpy( tmpKernel, &devKernel[ kernelIdx ], kernelSize * kernelSize * sizeof( cufftComplex ), cudaMemcpyDeviceToHost );
//				for ( long int i = 0; i < kernelSize * kernelSize; i++ )
//					tmpKernel[ i ] = tmpKernel[ i * 2 ];
//				char kernelFilename[100];
//				sprintf( kernelFilename, "kernel-%li-%i", kernelIdx, (pGridDegrid == GRID ? 0 : 1) );
//				_hstCasacoreInterface.WriteCasaImage( kernelFilename, kernelSize, kernelSize, _hstOutputRA,
//						_hstOutputDEC, _param.CellSize * (double) _param.Oversample, tmpKernel, CONST_C / phstData->AverageWavelength, NULL, CasacoreInterface::J2000, 1 );
//
//			}
//		free( tmpKernel );
//	}

	// cache the kernel.
	kernel = (cufftComplex *) malloc( kernelSize * kernelSize * oversample * oversample * sizeof( cufftComplex ) );
	moveDeviceToHost( (void *) kernel, (void *) devKernel, kernelSize * kernelSize * oversample * oversample * sizeof( cufftComplex ),
				"copying kernel from cache to device", __LINE__ );
	
	// free memory.
	if (devCombinedKernel != NULL)
		cudaFree( (void *) devCombinedKernel );
	if (devKernel != NULL)
		cudaFree( (void *) devKernel );
	
	// return success/failure.
	return ok;
	
} // KernelSet::generateKernel

//
//	P R I V A T E   C L A S S   M E M B E R S
//

