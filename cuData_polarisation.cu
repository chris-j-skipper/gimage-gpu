
// include the header file.
#include "cuData_polarisation.h"

using namespace std;

//
//	N O N   C L A S S   M E M B E R S
//

//
//	devCalculateGridPositions()
//
//	CJS: 11/11/2015
//
//	Calculate a grid position and a kernel index for each visibility.
//

__global__ void devCalculateGridPositions
			(
			VectorI * pGridPosition,			// a list of integer u,v,w grid coordinates - one per visibility (OUTPUT)
			int * pKernelIndex,				// a list of kernel indexes - one per visibility (OUTPUT)
			double pUvCellSize,				// }
			int pOversample,				// }- gridding parameters
			int pWPlanes,					// }
			int pPBChannels,				// }
			VectorD * pSample,				// a list of UVW coordinates for all the samples
			double ** pWavelength,				// a list of wavelengths for all the channels and spws
			int * pSpw,					// a list of spws for all the samples
			double * pWPlaneMax,				// a list of W limits for all the W planes
			int * pPBChannel,				// a list of A planes for each visibility
			int * pSampleID,				// the sample ID for each visibility
			int * pChannelID,				// the channel ID for each visibility
			int pSize,					// grid size in pixels
			int pVisibilityBatchSize,			// the number of visibilities to process
			int pNumSpws	)				// the number of spws
{
	
	// calculate visibility index and grid position index..
	long int visibilityIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	// ensure we haven't gone out of bounds.
	if (visibilityIndex < pVisibilityBatchSize)
	{

		// get the sample id and channel id.
		int sample = pSampleID[ visibilityIndex ];
		int channel = pChannelID[ visibilityIndex ];
		int spw = pSpw[ sample ];
		
		// declare grid position and kernel index.
		VectorI grid = { .u = 0, .v = 0, .w = 0 };
		int kernelIdx = 0;

		// only calculate a grid position if the spw is within the expected range. this visibility will already be flagged if this is not the case.
		if (spw >= 0 && spw < pNumSpws)
		{

			// get the sample UVW and the wavelength.
			VectorD uvw = pSample[ sample ];
			double wavelength = pWavelength[ spw ][ channel ];
		
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

			oversample.u = (int) round( exact.u * (double) pOversample );
			oversampleIndex.u = mod( oversample.u, pOversample );
			grid.u = intFloor( oversample.u, pOversample ) + (pSize / 2);
		
			oversample.v = (int) round( exact.v * (double) pOversample );
			oversampleIndex.v = mod( oversample.v, pOversample );
			grid.v = intFloor( oversample.v, pOversample ) + (pSize / 2);
				
			// calculate the kernel offset using the uOversample and vOversample.
			// no need to add the index of the w-plane. we will be gridding one w-plane at a time.
			kernelIdx = (oversampleIndex.u) + (oversampleIndex.v * pOversample);

			// calculate which w plane we are in.
			if (pWPlanes > 1)
				for ( int i = pWPlanes - 1; i >= 0; i-- )
					if (exact.w <= pWPlaneMax[ i ])
						grid.w = i;

			// replace the w-value with the kernel set index. This is calculated using (pbchan * num_w_planes) + w
			grid.w = (grid.w * pPBChannels) + pPBChannel[ visibilityIndex ];
//			grid.w = (pPBChannel[ visibilityIndex ] * pWPlanes) + grid.w; // cjs-mod

		}

		// update the arrays.
		pGridPosition[ visibilityIndex ] = grid;
		pKernelIndex[ visibilityIndex ] = kernelIdx;
	
	}
	
} // devCalculateGridPositions

//
//	devCalculateVisibility()
//
//	CJS: 31/10/2018
//
//	Calculate the visibility as either, for example of Stokes I, (LL + RR) / 2 or (XX + YY) / 2.
//

__global__ void devCalculateVisibility
			(
			cufftComplex * pVisibilityIn,			// contains all polarisation products
			cufftComplex * pVisibilityOut,		// the Stokes I, Q, U or V visibility
			int pNumPolarisations,				// the number of polarisation products
			double * pMultiplier,				// a list of constants for each polarisation product. i.e. for Stokes I we might have (0.5, 0, 0, 0.5).
			int * pPolarisationConfig,			// a list of polarisation configurations, indexed by sample ID
			int * pSampleID,
			long int pVisibilityBatchSize			// the number of visibilities to process
			)
{

	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	// ensure we are not out of bounds.
	if (index < pVisibilityBatchSize)
	{

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
			}
		}
		pVisibilityOut[ index ] = value;

	}

} // devCalculateVisibility

//
//	devCalculateFlag()
//
//	CJS: 31/10/2018
//
//	Calculate the visibility as either, for example of Stokes I, (LL + RR) / 2 or (XX + YY) / 2, and sets the flag if either of the polarisations are flagged.
//

__global__ void devCalculateFlag
			(
			bool * pFlagIn,				// flags for all polarisation products
			bool * pFlagOut,				// the combined flag
			int pNumPolarisations,				// the number of polarisation products
			double * pMultiplier,				// a list of constants for each polarisation product. i.e. for Stokes I we might have (0.5, 0, 0, 0.5).
			int * pPolarisationConfig,			// a list of polarisation configurations, indexed by sample ID
			int * pSampleID,
			long int pVisibilityBatchSize,		// the number of visibilities to process
			bool pFullStokes				// true if we're creating images for all Stokes products. in this case we need to check all the
									//	polarisation flags, not just those where the multiplier is > 0.
			)
{

	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	// ensure we are not out of bounds.
	if (index < pVisibilityBatchSize)
	{

		bool flag = false;

		// get sample ID and the polarisation config for this sample.
		int polarisationConfig = pPolarisationConfig[ pSampleID[ index ] ];

		// calculate flag.
		for ( int polarisation = 0; polarisation < pNumPolarisations; polarisation++ )
			if (pMultiplier[ (polarisationConfig * pNumPolarisations) + polarisation ] != 0.0 || pFullStokes == true)
				flag = flag || (pFlagIn[ (index * pNumPolarisations) + polarisation ] == true);
		pFlagOut[ index ] = flag;

	}

} // devCalculateFlag

//
//	devComputeMfsWeights()
//
//	CJS: 16/07/2021
//
//	Compute the MFS weights ((lamba_0 / lambda) - 1)^taylor_term
//

__global__ void devComputeMfsWeights( float * pMfsWeight, double ** pWavelength, int * pSpw, int * pSampleID, int * pChannelID, int pTaylorTerm,
					double pAverageWavelength, int pVisibilityBatchSize, int pNumSpws )
{
	
	// calculate visibility index and grid position index. we have twice as many grid positions as visibilities because
	// each visibility is gridded twice - once at B and the complex conjugate at -B.
	long int visibilityIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	// ensure we haven't gone out of bounds.
	if (visibilityIndex < pVisibilityBatchSize)
	{

		// get the sample id and channel id.
		int sample = pSampleID[ visibilityIndex ];
		int channel = pChannelID[ visibilityIndex ];
		int spw = pSpw[ sample ];

		// only proceed if the spw is within the expected range. this visibility will already be flagged if this is not the case.
		double mfsWeight = 0.0;
		if (spw >= 0 && spw < pNumSpws)
		{

			// get the wavelength.
			double wavelength = pWavelength[ spw ][ channel ];

			// calculate the MFS weight.
			mfsWeight = pow( (pAverageWavelength / wavelength) - 1.0, pTaylorTerm );

		}
		pMfsWeight[ visibilityIndex ] = (float) mfsWeight;

	}
	
} // devComputeMfsWeights

//
//	devCreateAdjustedMueller()
//
//	CJS: 02/12/2021
//
//	Adjust the Mueller matrix so that parts of the beam that are less than the 20% cut off instead go to infinity. This means that the inverse Mueller matrix
//	will drop to zero around the edges.
//
//	Here we apply the equation:
//
//		M' = M x [ |M| + B / |M| ]
//		         -----------------
//		                |M|
//
//	where B is calculated as B = (max|M|)^2 / 400.
//

//__global__ void devCreateAdjustedMueller
//			(
//			cufftDoubleComplex * pAdjustedMueller,	// the adjusted Mueller matrix
//			cufftDoubleComplex * pMueller,		// the Mueller matrix
//			int pSize,					// the image size
//			double pMaxValue				// the maximum value pixel from the Mueller matrix
//			)
//{
	
//	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
//	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	// check that we haven't gone outside the grid dimensions of the image (some threads will).
//	if ( i < pSize && j < pSize )
//	{
			
//		cufftDoubleComplex mueller = pMueller[ (j * pSize) + i ];
//		double absMueller = cuCabs( mueller );	
		
//		double scalar = 1.0;
//		if (absMueller != 0.0)
//			scalar = (1.0 + (pow( pMaxValue / absMueller, 2 ) / 400));
//		cufftDoubleComplex newValue = multComplex( mueller, scalar );
	
		// update the adjusted Mueller matrix.
//		pAdjustedMueller[ (j * pSize) + i ] = newValue;
	
//	}

//} // devCreateAdjustedMueller

//
//	devCreateAdjustedMueller()
//
//	CJS: 02/12/2021
//
//	Adjust a function so that its value goes to zero instead of infinity near the edges. The maximum value will be around 20x the minimum 
//	value in the input function.
//
//	Here we apply the equation:
//
//		M' =          1
//			------------------
//			1 + [  max|M|   ] ^ 4
//			    [ --------- ]
//			    [  12 |M|   ]
//

//__global__ void devCreateAdjustedMueller
//			(
//			cufftDoubleComplex * pAdjustedMueller,	// the adjusted Mueller matrix
//			cufftDoubleComplex * pMueller,		// the Mueller matrix
//			float * pPB,					// the primary beam
//			int pSize,					// the image size
//			float pMaxValue				// the maximum absolute pixel value from the primary beam
//			)
//{
	
//	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
//	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	// check that we haven't gone outside the grid dimensions of the image (some threads will).
//	if ( i < pSize && j < pSize )
//	{
			
//		cufftDoubleComplex mueller = pMueller[ (j * pSize) + i ];
//		cufftDoubleComplex newValue = multComplex( mueller, 1.0 / (1.0 + pow( pMaxValue / (pPB[ (j * pSize) + i ] * 20.0), 2 )) );
	
		// update the adjusted Mueller matrix.
//		pAdjustedMueller[ (j * pSize) + i ] = newValue;
	
//	}

//} // devCreateAdjustedMueller

//
//	devCreateAdjustedMueller()
//
//	CJS: 07/03/2022
//
//	Adjust a function so that its value goes to a constant instead of infinity near the edges.
//
//	Here we apply the equation:
//
//		M' =             M
//			--------------------
//			sqrt[ 1 + (0.2M)^2 ]
//

__global__ void devCreateAdjustedMueller
			(
			cufftDoubleComplex * pAdjustedMueller,	// the adjusted Mueller matrix
			cufftDoubleComplex * pMueller,		// the Mueller matrix
			int pSize					// the image size
			)
{
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	// check that we haven't gone outside the grid dimensions of the image (some threads will).
	if ( i < pSize && j < pSize )
	{
			
		cufftDoubleComplex mueller = pMueller[ (j * pSize) + i ];
	
		// update the adjusted Mueller matrix.
		pAdjustedMueller[ (j * pSize) + i ] = multComplex( mueller, 1.0 / sqrt( 1.0 + pow( 0.2 * cuCabs( mueller ), 2 ) ) );
	
	}

} // devCreateAdjustedMueller

//
//	devMergeComplexNumbers()
//
//	CJS: 27/09/2021
//
//	Merge arrays of real and imaginary components into an array of complex numbers.
//

__global__ void devMergeComplexNumbers
			(
			cufftComplex * pOut,				// output array.
			float * pIn,					// input array.
			int pSize,					// number of elements
			int pImaginaryPosition				// the start index of where the imaginary numbers are located.
			)
{

	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	// ensure we haven't gone out of bounds.
	if (index < pSize)
	{

		pOut[ index ].x = pIn[ index ];
		pOut[ index ].y = pIn[ pImaginaryPosition + index ];

	}

} // devMergeComplexNumbers

//
//	devMoveToStartOfImage()
//
//	CJS: 12/11/2019
//
//	Move the central part of an image to the start of the image. The initial size must be at least 2x the final size or else we could be overwriting pixels before we
//	move them.
//

__global__ void devMoveToStartOfImage
			(
			cufftComplex * pImage,				// the image to process
			int pInitialSize,				// the initial size of the image
			int pFinalSize					// the final size of the image. it must be at least 2x smaller so that the input and output
									//	regions don't overlap.
			)
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
//	devMultArrayComplexConjugate()
//
//	CJS: 27/09/2021
//
//	Multiply an array of complex numbers by their complex conjugates to create an array of real numbers.
//

__global__ void devMultArrayComplexConjugate
			(
			float * pOut,					// output array.
			cufftComplex * pIn,				// input array.
			int pSize					// number of elements
			)
{

	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	// ensure we haven't gone out of bounds.
	if (index < pSize)
	{
		cufftComplex in = pIn[ index ];
		pOut[ index ] = pow( in.x, 2 ) + pow( in.y, 2 );
	}

} // devMultArrayComplexConjugate

//
//	devPhaseCorrection()
//
//	CJS: 17/10/2018
//
//	Phase correct all the visibilities.
//

__global__ void devPhaseCorrection
			(
			cufftComplex * pVisibility,			// the visibilities to correct
			double * pPhase,				// a list of phase corrections indexed as [ sampleID ]
			double ** pWavelength,				// a list of wavelengths for each SPW - indexed as [ spwID ][ channelID ]
			int * pSpw,					// indexed as [ sampleID ]
			int * pSampleID,				// indexed as [ visibilityID ]
			int * pChannelID,				// indexed as [ visibilityID ]
			int pVisibilityBatchSize,			// the number of visibilities to process
			int pNumSpws					// # of SPWs
			)
{
	
	// calculate visibility index and grid position index. we have twice as many grid positions as visibilities because
	// each visibility is gridded twice - once at B and the complex conjugate at -B.
	long int visibilityIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	// ensure we haven't gone out of bounds.
	if (visibilityIndex < pVisibilityBatchSize)
	{

		// get the sample id and channel id.
		int sample = pSampleID[ visibilityIndex ];
		int channel = pChannelID[ visibilityIndex ];
		int spw = pSpw[ sample ];

		// only proceed if the spw is within the expected range. this visibility will already be flagged if this is not the case.
		cufftDoubleComplex newVis = { .x = 0, .y = 0 };
		if (spw >= 0 && spw < pNumSpws)
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
//	devSeparateComplexNumbers()
//
//	CJS: 27/09/2021
//
//	Separate an array of complex numbers into arrays of real and imaginary components.
//

__global__ void devSeparateComplexNumbers
			(
			float * pOut,					// output array.
			cufftComplex * pIn,				// input array.
			int pSize,					// number of elements
			int pImaginaryPosition				// the start index of where the imaginary numbers should be put
			)
{

	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	// ensure we haven't gone out of bounds.
	if (index < pSize)
	{

		pOut[ index ] = pIn[ index ].x;
		pOut[ pImaginaryPosition + index ] = pIn[ index ].y;

	}

} // devSeparateComplexNumbers

//
//	devTakeConjugateVisibility()
//
//	CJS: 18/10/2018
//
//	Take the conjugate values of a complex visibility data set, but only for the second half of the data.
//

__global__ void devTakeConjugateVisibility( cufftComplex * pVisibility, long int pCurrentVisibility, long int pNumVisibilities, int pVisibilityBatchSize )
{
	
	// calculate visibility index and grid position index. we have twice as many grid positions as visibilities because
	// each visibility is gridded twice - once at B and the complex conjugate at -B.
	int visibilityIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	// ensure we haven't gone out of bounds.
	if (visibilityIndex < pVisibilityBatchSize)
	{

		// ensure we are in the second half of the visibility data set.
		long int thisVisibility = pCurrentVisibility + (long int) visibilityIndex;
		if (thisVisibility >= (pNumVisibilities / 2))
			pVisibility[ visibilityIndex ].y *= -1;

	}

} // devTakeConjugateVisibility

//
//	P U B L I C   C L A S S   M E M B E R S
//

//
//	Data::Data()
//
//	CJS: 20/08/2021
//
//	The constructor.
//

Data::Data()
{

	// create instances of casacore
	_casacoreInterface = CasacoreInterface::getInstance();

	// create instances of casacore
	_param = Parameters::getInstance();

	_mosaicID = -1;
	_taylorTerms = 1;
	_wProjection = false;
	_aProjection = false;
	WPlanes = 1;
	PBChannels = 1;
	_cacheData = false;
	_stokes = STOKES_I;
	_stokesImages = 1;

	NextComponent = NULL;
	NumVisibilities = NULL;
	Batches = NULL;
	WPlaneMax = NULL;
	WPlaneMean = NULL;
	Visibility = NULL;
	ResidualVisibility = NULL;
	MfsWeight = NULL;
	GridPosition = NULL;
	KernelIndex = NULL;
	DensityMap = NULL;
	Weight = NULL;
	FieldID = NULL;
	Flag = NULL;
	SampleID = NULL;
	ChannelID = NULL;
	ComponentMask = NULL;
	MaximumWavelength = 1.0;

	// primary beams.
	AveragePrimaryBeamIn = NULL;
	PrimaryBeam = NULL;			// this primary beam is used for mosaicing, and also for setting the image mask. it is in the reference frame of the
						// 	output phase position.
	PrimaryBeamRatio = NULL;		// this primary beam is used to re-scale the flux so that every channel appears to have the same primary beam.
	PrimaryBeamInFrame = NULL;		// this primary beam is shown in the reference frame of each mosaic component (used for image-plane mosaicing).
	
	MuellerDeterminant = NULL;
	_muellerDeterminantIn = NULL;

	// Jones matrices
	JonesMatrix = NULL;
	JonesMatrixIn = NULL;
	
	// Mueller matrix
	MuellerMatrix = NULL;
	InverseMuellerMatrix = NULL;

	// initialise the flags to false.
	MuellerMatrixFlag = NULL;
	InverseMuellerMatrixFlag = NULL;

	// average weights.
	AverageWeight = NULL;
	GriddedVisibilities = 0;
	MinimumVisibilitiesInMosaic = 0;

	// phase positions of the data, and phase positions for imaging.
	PhaseFromRA = 0.0;
	PhaseFromDEC = 0.0;
	PhaseToRA = 0.0;
	PhaseToDEC = 0.0;

} // Data::Data

Data::Data
			(
			int pTaylorTerms,				// # Taylor terms
			int pMosaicID,
			bool pWProjection,
			bool pAProjection,
			int pWPlanes,
			int pPBChannels,
			bool pCacheData,				// true if data needs to be cached
			int pStokes,					// the Stokes image to construct
			int pStokesImages				// the number of Stokes images we need to make
			)
{

	// initialise default values.
	new (this) Data();

	// set up the object.
	Create( pTaylorTerms, pMosaicID, pWProjection, pAProjection, pWPlanes, pPBChannels, pCacheData, pStokes, pStokesImages );

} // Data::Data

//
//	Data::~Data()
//
//	CJS: 20/08/2021
//
//	The destructor.
//

Data::~Data()
{

	// free any data that is still not free, and delete the cache.
	FreeData( /* pWhatData = */ DATA_ALL );

	if (_cacheData == true)
		DeleteCache();

	// clean up memory.
	if (Visibility != NULL)
	{
		for ( int s = 0; s < _stokesImages; s++ )
			if (Visibility[ s ] != NULL)
			{
				for ( int t = 0; t < _taylorTerms; t++ )
					if (Visibility[ s ][ t ] != NULL)
						free( (void *) Visibility[ s ][ t ] );
				free( (void *) Visibility[ s ] );
			}
		free( (void *) Visibility );
	}
	if (GridPosition != NULL)
		free( (void *) GridPosition );
	if (KernelIndex != NULL)
		free( (void *) KernelIndex );
	if (DensityMap != NULL)
		free( (void *) DensityMap );
	if (Weight != NULL)
	{
		for ( int s = 0; s < _stokesImages; s++ )
			if (Weight[ s ] != NULL)
				free( (void *) Weight[ s ] );
		free( (void *) Weight );
	}
	if (ResidualVisibility != NULL)
	{
		for ( int s = 0; s < _stokesImages; s++ )
		{
			for ( int t = 0; t < _taylorTerms; t++ )
				if (ResidualVisibility[ s ][ t ] != NULL)
					free( (void *) ResidualVisibility[ s ][ t ] );
			free( (void *) ResidualVisibility[ s ] );
		}
		free( (void *) ResidualVisibility );
	}
	if (MfsWeight != NULL)
	{
		for ( int s = 0; s < _stokesImages; s++ )
			if (MfsWeight[ s ] != NULL)
			{
				for ( int t = 0; t < (_taylorTerms - 1) * 2; t++ )
					if (MfsWeight[ s ][ t ] != NULL)
						free( (void *) MfsWeight[ s ][ t ] );
				free( (void *) MfsWeight[ s ] );
			}
		free( (void *) MfsWeight );
	}
	if (WPlaneMean != NULL)
		free( (void *) WPlaneMean );
	if (WPlaneMax != NULL)
		free( (void *) WPlaneMax );
	if (Batches != NULL)
		free( (void *) Batches );
	if (NumVisibilities != NULL)
		free( (void *) NumVisibilities );
	if (AveragePrimaryBeamIn != NULL)
		free( (void *) AveragePrimaryBeamIn );
	if (PrimaryBeam != NULL)
		free( (void *) PrimaryBeam );
	if (PrimaryBeamRatio != NULL)
		free( (void *) PrimaryBeamRatio );
	if (PrimaryBeamInFrame != NULL)
		free( (void *) PrimaryBeamInFrame );
	if (FieldID != NULL)
		free( (void *) FieldID );
	if (Flag != NULL)
		free( (void *) Flag );
	if (SampleID != NULL)
		free( (void *) SampleID );
	if (ChannelID != NULL)
		free( (void *) ChannelID );
	if (MuellerMatrixFlag != NULL)
		free( (void *) MuellerMatrixFlag );
	if (InverseMuellerMatrixFlag != NULL)
		free( (void *) InverseMuellerMatrixFlag );
	if (ComponentMask != NULL)
		free( (void *) ComponentMask );
	if (AverageWeight != NULL)
		free( (void *) AverageWeight );
	if (MuellerDeterminant != NULL)
		free( (void *) MuellerDeterminant );
	if (_muellerDeterminantIn != NULL)
		free( (void *) _muellerDeterminantIn );

	// free the Jones and Mueller matrices if they haven't already been released.
	FreeJonesMatrices();
	FreeMuellerMatrices();
	
	if (JonesMatrixIn != NULL)
		free( (void *) JonesMatrixIn );

} // Data::~Data

//
//	BuildComponentMask()
//
//	CJS: 22/09/2021
//
//	Builds a image mask for a single mosaic component. This mask is only needed for image-plane mosaics, because it is in the reference frame of the mosaic component.
//

void Data::BuildComponentMask
			(
			float * pPrimaryBeamPattern,			// the primary beam pattern for the whole mosaic.
			double pCellSize,				// the cell size in arcseconds.
			double pOutputRA,				// the mosaic output RA position.
			double pOutputDEC,				// the mosaic output DEC position.
			int pBeamSize					// the size of primary beams in pixels.
			)
{

	// create a reprojection object.
	Reprojection imagePlaneReprojection;

	// create two workspace primary beams on the device.
	float * devInPattern = NULL;
	float * devOutPattern = NULL;
	reserveGPUMemory( (void **) &devInPattern, pBeamSize * pBeamSize * sizeof( float ), "reserving memory for the input primary beam pattern on the device", __LINE__ );
	reserveGPUMemory( (void **) &devOutPattern, pBeamSize * pBeamSize * sizeof( float ), "reserving memory for the output primary beam on the device", __LINE__ );

	// create space for the reprojected primary beam pattern.
	float * hstOutPattern = (float *) malloc( pBeamSize * pBeamSize * sizeof( float ) );
	memset( hstOutPattern, 0, pBeamSize * pBeamSize * sizeof( float ) );

	// create the device memory needed by the reprojection code.
	Reprojection::rpVectI outSize = { /* x = */ pBeamSize, /* y = */ pBeamSize };
	imagePlaneReprojection.CreateDeviceMemory( outSize );

	// firstly, reproject the primary beam pattern into the reference position of this mosaic component.
	reprojectImage(	/* phstImageIn = */ pPrimaryBeamPattern,
				/* phstImageOut = */ hstOutPattern,
				/* pImageInSize = */ pBeamSize,
				/* pImageOutSize = */ pBeamSize,
				/* pInputCellSize = */ pCellSize * (double) pBeamSize / (double) _param->ImageSize,
				/* pOutputCellSize = */ pCellSize * (double) pBeamSize / (double) _param->ImageSize,
				/* pInRA = */ pOutputRA,
				/* pInDec = */ pOutputDEC,
				/* pOutRA = */ PhaseFromRA,
				/* pOutDec = */ PhaseFromDEC,
				/* pdevInImage = */ devInPattern,
				/* pdevOutImage = */ devOutPattern,
				/* pImagePlaneReprojection = */ imagePlaneReprojection,
				/* pVerbose = */ false );

	// create a new mask, based upon the reprojected primary beam pattern.
	ComponentMask = (bool *) malloc( _param->ImageSize * _param->ImageSize * sizeof( bool ) );
	double scale = (double) pBeamSize / (double) _param->ImageSize;
	for ( int i = 0; i < _param->ImageSize; i++ )
		for ( int j = 0; j < _param->ImageSize; j++ )
		{
			int x = (int) floor( (double) i * scale );
			int y = (int) floor( (double) j * scale );
			ComponentMask[ (j * _param->ImageSize) + i ] = (hstOutPattern[ (y * pBeamSize) + x ] >= 0.2);
		}

	// free data.
	if (devInPattern != NULL)
		cudaFree( (void *) devInPattern );
	if (devOutPattern != NULL)
		cudaFree( (void *) devOutPattern );
	if (hstOutPattern != NULL)
		free( (void *) hstOutPattern );

} // BuildComponentMask

//
//	CacheData()
//
//	CJS: 25/03/2019
//
//	Store a whole set of visibilities, grid positions, kernel indexes, etc to disk, and free the memory. We use the offset parameter in the rare cases
//	where the data does not start at the beginning of the array. Most of the time the offset will be zero.
//

void Data::CacheData
			(
			int pBatchID,					// the CPU batch ID
			int pTaylorTerm,				// the Taylor term (or -1 for all Taylor terms). only used for residual visibilities.
			int pWhatData					// which data should we cache ?
			)
{

//printf( "%d (DATA): CACHING: pFilenamePrefix %s, pStageID %i\n", __LINE__, pMeasurementSetFilename, pBatchID );
	// build filename.
	char filename[ 255 ];

	// build the full filename.
	if (_param->CacheLocation[0] != '\0')
		sprintf( filename, "%s%s-%i-%i-cache.dat", _param->CacheLocation, _param->OutputPrefix, _mosaicID, pBatchID );
	else
		sprintf( filename, "%s-%i-%i-cache.dat", _param->OutputPrefix, _mosaicID, pBatchID );

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
	for ( int s = 0; s < _stokesImages; s++ )
		for ( int t = 0; t < _taylorTerms; t++ )
			if ((pWhatData & DATA_VISIBILITIES) == DATA_VISIBILITIES && (t == pTaylorTerm || pTaylorTerm == -1))
				fwrite( (void *) Visibility[ s ][ t ], sizeof( cufftComplex ), NumVisibilities[ pBatchID ], fr );
			else
				fseek( fr, NumVisibilities[ pBatchID ] * sizeof( cufftComplex ), SEEK_CUR );
	if ((pWhatData & DATA_GRID_POSITIONS) == DATA_GRID_POSITIONS)
		fwrite( (void *) GridPosition, sizeof( VectorI ), NumVisibilities[ pBatchID ], fr );
	else
		fseek( fr, NumVisibilities[ pBatchID ] * sizeof( VectorI ), SEEK_CUR );
	if ((pWhatData & DATA_KERNEL_INDEXES) == DATA_KERNEL_INDEXES)
		fwrite( (void *) KernelIndex, sizeof( int ), NumVisibilities[ pBatchID ], fr );
	else
		fseek( fr, NumVisibilities[ pBatchID ] * sizeof( int ), SEEK_CUR );
	if ((pWhatData & DATA_DENSITIES) == DATA_DENSITIES)
		fwrite( (void *) DensityMap, sizeof( int ), NumVisibilities[ pBatchID ], fr );
	else
		fseek( fr, NumVisibilities[ pBatchID ] * sizeof( int ), SEEK_CUR );
	if ((pWhatData & DATA_WEIGHTS) == DATA_WEIGHTS)
		for ( int s = 0; s < _stokesImages; s++ )
			fwrite( (void *) Weight[ s ], sizeof( float ), NumVisibilities[ pBatchID ], fr );
	else
		fseek( fr, NumVisibilities[ pBatchID ] * _stokesImages * sizeof( float ), SEEK_CUR );
	if (_param->Deconvolver == MFS)
		for ( int s = 0; s < _stokesImages; s++ )
			for ( int t = 0; t < (_taylorTerms - 1) * 2; t++ )
				if ((pWhatData & DATA_MFS_WEIGHTS) == DATA_MFS_WEIGHTS && (t == pTaylorTerm || pTaylorTerm == -1))
					fwrite( (void *) MfsWeight[ s ][ t ], sizeof( float ), NumVisibilities[ pBatchID ], fr );
				else
					fseek( fr, NumVisibilities[ pBatchID ] * sizeof( float ), SEEK_CUR );
	for ( int s = 0; s < _stokesImages; s++ )
		for ( int t = 0; t < _taylorTerms; t++ )
			if ((pWhatData & DATA_RESIDUAL_VISIBILITIES) == DATA_RESIDUAL_VISIBILITIES && (pTaylorTerm == t || pTaylorTerm == -1))
				fwrite( (void *) ResidualVisibility[ s ][ t ], sizeof( cufftComplex ), NumVisibilities[ pBatchID ], fr );
			else
				fseek( fr, NumVisibilities[ pBatchID ] * sizeof( cufftComplex ), SEEK_CUR );

	// close the file.
	fclose( fr );

	// free the memory. if there's an offset then we don't free the array - we only free the cells after the offset.
	FreeData( /* pWhatData = */ pWhatData );

} // CacheData

//
//	Data::ConvertJonesToMueller()
//
//	CJS: 16/02/2022
//
//	Takes the outer product of a Jones matrix to convert it into a Mueller matrix.
//

cufftDoubleComplex ** Data::ConvertJonesToMueller
			(
			cufftComplex ** phstJonesMatrix,		// the input Jones matrix
			int pImageSize					//
			)
{

	// and set the thread block size for processing images on the device.
	setThreadBlockSize2D( pImageSize, pImageSize, _gridSize2D, _blockSize2D );

	int items = pImageSize * pImageSize;
	int stages = items / MAX_THREADS;
	if (items % MAX_THREADS != 0)
		stages++;

	// copy Jones matrix to the device.
	cufftComplex ** devJonesMatrix = (cufftComplex **) malloc( 4 * sizeof( cufftComplex * ) );
	for ( int cell = 0; cell < 4; cell++ )
	{
		devJonesMatrix[ cell ] = NULL;
		if (phstJonesMatrix[ cell ] != NULL)
		{
			reserveGPUMemory( (void **) &devJonesMatrix[ cell ], pImageSize * pImageSize * sizeof( cufftComplex ),
							"reserving device memory the Jones matrix", __LINE__ );
			moveHostToDevice( (void *) devJonesMatrix[ cell ], (void *) phstJonesMatrix[ cell ], pImageSize * pImageSize *
							sizeof( cufftComplex ), "copying Jones matrix cell to the device", __LINE__ );
		}
	}

	// step 1:	copy the Jones matrix, and take the complex conjugate.
	cufftComplex ** devJonesConj = (cufftComplex **) malloc( 4 * sizeof( cufftComplex * ) );
	for ( int cell = 0; cell < 4; cell++ )
	{
		devJonesConj[ cell ] = NULL;
		if (devJonesMatrix[ cell ] != NULL)
		{
	
			reserveGPUMemory( (void **) &devJonesConj[ cell ], pImageSize * pImageSize * sizeof( cufftComplex ),
						"reserving device memory the conjugate Jones matrix", __LINE__ );
			cudaMemcpy( (void *) devJonesConj[ cell ], devJonesMatrix[ cell ], pImageSize * pImageSize * sizeof( cufftComplex ), cudaMemcpyDeviceToDevice );
		
			// calculate the conjugate of this image.
			devTakeConjugateImage<<< _gridSize2D, _blockSize2D >>>(	/* pImage = */ devJonesConj[ cell ],
											/* pSize = */ pImageSize );
		
		}
	}

	// step 1:	construct the 4x4 matrix, M1, that is the outer product of the Jones matrix with the conjugate of itself.
	//
	//			( XX*' ) = M . ( XX* ),	where	M1 = J (x) J*     (outer product)
	//			( XY*' )       ( XY* )
	//			( YX*' )       ( YX* )
	//			( YY*' )       ( YY* )
	//
	//		the primed variables, XX*', XY*', YX*', and YY*' are the observed signals. the unprimed variables are the true 'sky' signals, i.e. without
	//		the beam patterns or leakage. we want to solve for these unprimed values.
	//
	cufftDoubleComplex ** devM1 = outerProductImageMatrix(	/* pOne = */ devJonesMatrix,
									/* pTwo = */ devJonesConj,
									/* pMatrixSize1 = */ 2,
									/* pMatrixSize2 = */ 2,
									/* pImageSize = */ pImageSize );

	// free memory.
	if (devJonesMatrix != NULL)
	{
		for ( int cell = 0; cell < 4; cell++ )
			if (devJonesMatrix[ cell ] != NULL)
				cudaFree( (void *) devJonesMatrix[ cell ] );
		free( (void *) devJonesMatrix );
	}		
	if (devJonesConj != NULL)
	{
		for ( int cell = 0; cell < 4; cell++ )
			if (devJonesConj[ cell ] != NULL)
				cudaFree( (void *) devJonesConj[ cell ] );
		free( (void *) devJonesConj );
	}
	
	// step 2:	construct the 4x4 matrix, M2, that converts XX,XY,YX,YY into I',Q',U',V'.
	//
	//			( I' ) = 1/2 ( XX*' + YY*' ) = M2 . ( XX* )
	//			( Q' )       ( XX*' - YY*' )        ( XY* )
	//			( U' )       ( YX*' + XY*' )        ( YX* )
	//			( V' )       ( YX*' - XY*' )        ( YY* )
	//	
	// 		the summation we want is:
	//
	//			M2[0]  = M1[0] + M1[12]	M2[1]  = M1[1] + M1[13]	M2[2]  = M1[2] + M1[14]	M2[3]  = M1[3] + M1[15]
	//			M2[4]  = M1[0] - M1[12]	M2[5]  = M1[1] - M1[13]	M2[6]  = M1[2] - M1[14]	M2[7]  = M1[3] - M1[15]
	//			M2[8]  = M1[8] + M1[4]		M2[9]  = M1[9] + M1[5]		M2[10] = M1[10] + M1[6]	M2[11] = M1[11] + M1[7]
	//			M2[12] = M1[8] - M1[4]		M2[13] = M1[9] - M1[5]		M2[14] = M1[10] - M1[6]	M2[15] = M1[11] - M1[7]
	//

	cufftDoubleComplex ** devM2 = (cufftDoubleComplex **) malloc( 16 * sizeof( cufftDoubleComplex * ) );
	for ( int col = 0; col < 4; col++ )
	{
		for ( int row = 0; row < 4; row++ )
			devM2[ (row * 4) + col ] = NULL;
		if (devM1[ col ] != NULL || devM1[ col + 12 ] != NULL)
			for ( int row = 0; row < 2; row++ )
				reserveGPUMemory( (void **) &devM2[ (row * 4) + col ], pImageSize * pImageSize * sizeof( cufftDoubleComplex ),
														"reserving device memory for M2 matrix cell", __LINE__ );
		if (devM1[ col + 4 ] != NULL || devM1[ col + 8 ] != NULL)
			for ( int row = 2; row < 4; row++ )
				reserveGPUMemory( (void **) &devM2[ (row * 4) + col ], pImageSize * pImageSize * sizeof( cufftDoubleComplex ),
														"reserving device memory for M2 matrix cell", __LINE__ );
	}

	// copy an image into each cell.
	for ( int row = 0; row < 2; row++ )
		for ( int col = 0; col < 4; col++ )
		{
			if (devM2[ (row * 4) + col ] != NULL)
			{
				if (devM1[ col ] != NULL)
					cudaMemcpy( (void *) devM2[ (row * 4) + col ], (void *) devM1[ col ], pImageSize * pImageSize * sizeof( cufftDoubleComplex ),
							cudaMemcpyDeviceToDevice );
				else
					cudaMemset( (void *) devM2[ (row * 4) + col ], 0, pImageSize * pImageSize * sizeof( cufftDoubleComplex ) );
			}
			if (devM2[ (row * 4) + col + 8 ] != NULL)
			{
				if (devM1[ col + 8 ] != NULL)
					cudaMemcpy( (void *) devM2[ (row * 4) + col + 8 ], (void *) devM1[ col + 8 ],
										pImageSize * pImageSize * sizeof( cufftDoubleComplex ), cudaMemcpyDeviceToDevice );
				else
					cudaMemset( (void *) devM2[ (row * 4) + col + 8 ], 0, pImageSize * pImageSize * sizeof( cufftDoubleComplex ) );
			}
		}

	for ( int i = 0; i < stages; i++ )
	{

		// define the block/thread dimensions.
		int itemsThisStage = items - (i * MAX_THREADS);
		if (itemsThisStage > MAX_THREADS)
			itemsThisStage = MAX_THREADS;
		int threads = itemsThisStage;
		int blocks;
		setThreadBlockSize1D( &threads, &blocks );

		// add the second image to the first for each cell.
		for ( int col = 0; col < 4; col++ )		
		{
			if (devM1[ col + 12 ] != NULL && devM2[ col ] != NULL)
				devAddArrays<<< blocks, threads >>>(		/* pOne = */ &devM2[ col ][ /* CELL = */ i * MAX_THREADS ],
										/* pTwo = */ &devM1[ col + 12 ][ /* CELL = */ i * MAX_THREADS ],
										/* pSize = */ itemsThisStage );
			if (devM1[ col + 12 ] != NULL && devM2[ col + 4 ] != NULL)
				devSubtractArrays<<< blocks, threads >>>(	/* pOne = */ &devM2[ col + 4 ][ /* CELL = */ i * MAX_THREADS ],
										/* pTwo = */ &devM1[ col + 12 ][ /* CELL = */ i * MAX_THREADS ],
										/* pSize = */ itemsThisStage );
			if (devM1[ col + 4 ] != NULL && devM2[ col + 8 ] != NULL)
				devAddArrays<<< blocks, threads >>>(		/* pOne = */ &devM2[ col + 8 ][ /* CELL = */ i * MAX_THREADS ],
										/* pTwo = */ &devM1[ col + 4 ][ /* CELL = */ i * MAX_THREADS ],
										/* pSize = */ itemsThisStage );
			if (devM1[ col + 4 ] != NULL && devM2[ col + 12 ] != NULL)
				devSubtractArrays<<< blocks, threads >>>(	/* pOne = */ &devM2[ col + 12 ][ /* CELL = */ i * MAX_THREADS ],
										/* pTwo = */ &devM1[ col + 4 ][ /* CELL = */ i * MAX_THREADS ],
										/* pSize = */ itemsThisStage );
		}
								
	} // LOOP: i

	// divide the images by 2 to include the factor of 2 we find in Stokes I = (XX + YY) / 2.
	for ( int cell = 0; cell < 16; cell++ )
		if (devM2[ cell ] != NULL)
			devMultiplyImages<<< _gridSize2D, _blockSize2D >>>(	/* pOne = */ devM2[ cell ],
										/* pScalar = */ 0.5,
										/* pMask = */ NULL,
										/* pSizeOne = */ pImageSize );

	// free memory.
	if (devM1 != NULL)
	{
		for ( int cell = 0; cell < 16; cell++ )
			if (devM1[ cell ] != NULL)
				cudaFree( (void *) devM1[ cell ] );
		free( (void *) devM1 );
	}

	// step 3:	calculate M3 as the inverse of M2.
	//
	//			( XX* ) = M3 . ( I' )
	//			( XY* )        ( Q' )
	//			( YX* )        ( U' )
	//			( YY* )        ( V' )
	//
	cufftDoubleComplex ** devM3 = calculateInverseImageMatrix(	/* pdevMatrix = */ devM2,
									/* pMatrixSize = */ 4,
									/* pImageSize = */ pImageSize,
									/* pDivideByDeterminant = */ true );

	// free memory.
	if (devM2 != NULL)
	{
		for ( int cell = 0; cell < 16; cell++ )
			if (devM2[ cell ] != NULL)
				cudaFree( (void *) devM2[ cell ] );
		free( (void *) devM2 );
	}

	// step 3:	construct the 4x4 inverse Mueller matrix, that converts I',Q',U',V' into I,Q,U,V. this is the Mueller matrix, which provides the gridding
	//		kernels through Fourier transform of each matrix cell. the top row, for example, consists of the Stokes I beam pattern, and the leakage of Q,
	//		U, and V into I.
	//
	//			( I ) = 1/2 ( XX* + YY* ) = Mu^{-1} . ( I' )
	//			( Q )       ( XX* - YY* )             ( Q' )
	//			( U )       ( YX* + XY* )             ( U' )
	//			( V )       ( YX* - XY* )             ( V' )
	//
	// 		the summation we want is:
	//
	//			Mu[0]  = M2[0] + M2[12]	Mu[1]  = M2[1] + M2[13]	Mu[2]  = M2[2] + M2[14]	Mu[3]  = M2[3] + M2[15]
	//			Mu[4]  = M2[0] - M2[12]	Mu[5]  = M2[1] - M2[13]	Mu[6]  = M2[2] - M2[14]	Mu[7]  = M2[3] - M2[15]
	//			Mu[8]  = M2[8] + M2[4]		Mu[9]  = M2[9] + M2[5]		Mu[10] = M2[10] + M2[6]	Mu[11] = M2[11] + M2[7]
	//			Mu[12] = M2[8] - M2[4]		Mu[13] = M2[9] - M2[5]		Mu[14] = M2[10] - M2[6]	Mu[15] = M2[11] - M2[7]
	//
	cufftDoubleComplex ** devInverseMuellerMatrix = (cufftDoubleComplex **) malloc( 16 * sizeof( cufftDoubleComplex * ) );
	for ( int col = 0; col < 4; col++ )
	{
		for ( int row = 0; row < 4; row++ )
			devInverseMuellerMatrix[ (row * 4) + col ] = NULL;
		if (devM3[ col ] != NULL || devM3[ col + 12 ] != NULL)
			for ( int row = 0; row < 2; row++ )
				reserveGPUMemory( (void **) &devInverseMuellerMatrix[ (row * 4) + col ], pImageSize * pImageSize * sizeof( cufftDoubleComplex ),
													"reserving device memory for Mueller matrix cell", __LINE__ );
		if (devM3[ col + 4 ] != NULL || devM3[ col + 8 ] != NULL)
			for ( int row = 2; row < 4; row++ )
				reserveGPUMemory( (void **) &devInverseMuellerMatrix[ (row * 4) + col ], pImageSize * pImageSize * sizeof( cufftDoubleComplex ),
													"reserving device memory for Mueller matrix cell", __LINE__ );
	}

	// copy an image into each cell.
	for ( int row = 0; row < 2; row++ )
		for ( int col = 0; col < 4; col++ )
		{
			if (devInverseMuellerMatrix[ (row * 4) + col ] != NULL)
			{
				if (devM3[ col ] != NULL)
					cudaMemcpy( (void *) devInverseMuellerMatrix[ (row * 4) + col ], (void *) devM3[ col ],
							pImageSize * pImageSize * sizeof( cufftDoubleComplex ), cudaMemcpyDeviceToDevice );
				else
					cudaMemset( (void *) devInverseMuellerMatrix[ (row * 4) + col ], 0, pImageSize * pImageSize * sizeof( cufftDoubleComplex ) );
			}
			if (devInverseMuellerMatrix[ (row * 4) + col + 8 ] != NULL)
			{
				if (devM3[ col + 8 ] != NULL)
					cudaMemcpy( (void *) devInverseMuellerMatrix[ (row * 4) + col + 8 ], (void *) devM3[ col + 8 ],
							pImageSize * pImageSize * sizeof( cufftDoubleComplex ), cudaMemcpyDeviceToDevice );
				else
					cudaMemset( (void *) devInverseMuellerMatrix[ (row * 4) + col + 8 ], 0,
														pImageSize * pImageSize * sizeof( cufftDoubleComplex ) );
			}
		}

	for ( int i = 0; i < stages; i++ )
	{

		// define the block/thread dimensions.
		int itemsThisStage = items - (i * MAX_THREADS);
		if (itemsThisStage > MAX_THREADS)
			itemsThisStage = MAX_THREADS;
		int threads = itemsThisStage;
		int blocks;
		setThreadBlockSize1D( &threads, &blocks );

		// add the second image to the first for each cell.
		for ( int col = 0; col < 4; col++ )		
		{
			if (devM3[ col + 12 ] != NULL && devInverseMuellerMatrix[ col ] != NULL)
				devAddArrays<<< blocks, threads >>>(		/* pOne = */ &devInverseMuellerMatrix[ col ][ /* CELL = */ i * MAX_THREADS ],
										/* pTwo = */ &devM3[ col + 12 ][ /* CELL = */ i * MAX_THREADS ],
										/* pSize = */ itemsThisStage );
			if (devM3[ col + 12 ] != NULL && devInverseMuellerMatrix[ col + 4 ] != NULL)
				devSubtractArrays<<< blocks, threads >>>(	/* pOne = */ &devInverseMuellerMatrix[ col + 4 ][ /* CELL = */ i * MAX_THREADS ],
										/* pTwo = */ &devM3[ col + 12 ][ /* CELL = */ i * MAX_THREADS ],
										/* pSize = */ itemsThisStage );
			if (devM3[ col + 4 ] != NULL && devInverseMuellerMatrix[ col + 8 ] != NULL)
				devAddArrays<<< blocks, threads >>>(		/* pOne = */ &devInverseMuellerMatrix[ col + 8 ][ /* CELL = */ i * MAX_THREADS ],
										/* pTwo = */ &devM3[ col + 4 ][ /* CELL = */ i * MAX_THREADS ],
										/* pSize = */ itemsThisStage );
			if (devM3[ col + 4 ] != NULL && devInverseMuellerMatrix[ col + 12 ] != NULL)
				devSubtractArrays<<< blocks, threads >>>(	/* pOne = */ &devInverseMuellerMatrix[ col + 12 ][ /* CELL = */ i * MAX_THREADS ],
										/* pTwo = */ &devM3[ col + 4 ][ /* CELL = */ i * MAX_THREADS ],
										/* pSize = */ itemsThisStage );
		}
								
	} // LOOP: i

	// divide the images by 2 to include the factor of 2 we find in Stokes I = (XX + YY) / 2.
	for ( int cell = 0; cell < 16; cell++ )
		if (devInverseMuellerMatrix[ cell ] != NULL)
			devMultiplyImages<<< _gridSize2D, _blockSize2D >>>(	/* pOne = */ devInverseMuellerMatrix[ cell ],
										/* pScalar = */ 0.5,
										/* pMask = */ NULL,
										/* pSizeOne = */ pImageSize );

	// free memory.
	if (devM3 != NULL)
	{
		for ( int cell = 0; cell < 16; cell++ )
			if (devM3[ cell ] != NULL)
				cudaFree( (void *) devM3[ cell ] );
		free( (void *) devM3 );
	}

	// step 4:	calculate the Mueller matrix by taking the inverse, which we will use for degridding.
	//
	//			( I' ) = 1/2 ( XX* + YY* ) = Mu . ( I )
	//			( Q' )       ( XX* - YY* )        ( Q )
	//			( U' )       ( YX* + XY* )        ( U )
	//			( V' )       ( YX* - XY* )        ( V )
	//
	cufftDoubleComplex ** devMuellerMatrix = calculateInverseImageMatrix(	/* pdevMatrix = */ devInverseMuellerMatrix,
											/* pMatrixSize = */ 4,
											/* pImageSize = */ pImageSize,
											/* pDivideByDeterminant = */ true );

	// free memory.
	if (devInverseMuellerMatrix != NULL)
	{
		for ( int cell = 0; cell < 16; cell++ )
			if (devInverseMuellerMatrix[ cell ] != NULL)
				cudaFree( (void *) devInverseMuellerMatrix[ cell ] );
		free( (void *) devInverseMuellerMatrix );
	}
	
	// return something.
	return devMuellerMatrix;

} // Data::ConvertJonesToMueller

//
//	Data::CopyJonesMatrixIn()
//
//	CJS: 21/03/2022
//
//	Fill the input Jones matrix from another Jones matrix.
//

void Data::CopyJonesMatrixIn
			(
			cufftComplex ** pFromMatrix
			)
{

	for ( int cell = 0; cell < 4; cell++ )
	{
		if (JonesMatrixIn[ cell ] != NULL)
			free( (void *) JonesMatrixIn[ cell ] );
		if (pFromMatrix[ cell ] != NULL)
		{
			JonesMatrixIn[ cell ] = (cufftComplex *) malloc( _param->BeamInSize * _param->BeamInSize * sizeof( cufftComplex ) );
			memcpy( (void *) JonesMatrixIn[ cell ], (void *) pFromMatrix[ cell ], _param->BeamInSize * _param->BeamInSize * sizeof( cufftComplex ) );
		}
	}

} // Data::CopyJonesMatrixIn

//
//	Data::Create()
//
//	CJS: 01/03/2022
//
//	Set up the object.
//

void Data::Create
			(
			int pTaylorTerms,				// # Taylor terms
			int pMosaicID,
			bool pWProjection,
			bool pAProjection,
			int pWPlanes,
			int pPBChannels,
			bool pCacheData,				// true if data needs to be cached
			int pStokes,					// the Stokes image to construct
			int pStokesImages				// the number of Stokes images we need to make
			)
{

	_taylorTerms = pTaylorTerms;
	_mosaicID = pMosaicID;
	_wProjection = pWProjection;
	_aProjection = pAProjection;
	WPlanes = pWPlanes;
	PBChannels = pPBChannels;
	_cacheData = pCacheData;
	_stokes = pStokes;
	_stokesImages = pStokesImages;
	
	NumVisibilities = (long int *) malloc( 1 * sizeof( int ) );
	Stages = 0;
	Batches = (int *) malloc( 1 * sizeof( int ) );

	Visibility = (cufftComplex ***) malloc( _stokesImages * sizeof( cufftComplex ** ) );
	ResidualVisibility = (cufftComplex ***) malloc( _stokesImages * sizeof( cufftComplex **) );
	for ( int s = 0; s < _stokesImages; s++ )
	{
		Visibility[ /* STOKES = */ s ] = (cufftComplex **) malloc( _taylorTerms * sizeof( cufftComplex * ) );
		ResidualVisibility[ /* STOKES = */ s ] = (cufftComplex **) malloc( _taylorTerms * sizeof( cufftComplex * ) );
		for ( int t = 0; t < _taylorTerms; t++ )
		{
			Visibility[ /* STOKES = */ s ][ /* TAYLOR_TERM = */ t ] = NULL;
			ResidualVisibility[ /* STOKES = */ s ][ /* TAYLOR_TERM = */ t ] = NULL;
		}
	}
	if (_param->Deconvolver == MFS)
	{
		MfsWeight = (float ***) malloc( _stokesImages * sizeof( float ** ) );
		for ( int stokes = 0; stokes < _stokesImages; stokes++ )
		{
			MfsWeight[ stokes ] = (float **) malloc( (_taylorTerms - 1) * 2 * sizeof( float * ) );
			for ( int taylorTerm = 0; taylorTerm < (_taylorTerms - 1) * 2; taylorTerm++ )
				MfsWeight[ stokes ][ taylorTerm ] = NULL;
		}
	}

	Weight = (float **) malloc( _stokesImages * sizeof( float * ) );
	for ( int s = 0; s < _stokesImages; s++ )
		Weight[ /* STOKES = */ s ] = NULL;

	// average weights.
	AverageWeight = (double *) malloc( _stokesImages * sizeof( double ) );
	for ( int s = 0; s < _stokesImages; s++ )
		AverageWeight[ s ] = 0.0;
		
	// Jones matrices
	JonesMatrixIn = (cufftComplex **) malloc( 4 * sizeof( cufftComplex * ) );
	for ( int cell = 0; cell < 4; cell++ )
		JonesMatrixIn[ cell ] = NULL;

	// initialise the flags to false.
	MuellerMatrixFlag = (bool *) malloc( 16 * sizeof( bool ) );
	InverseMuellerMatrixFlag = (bool *) malloc( 16 * sizeof( bool ) );
	for ( int cell = 0; cell < 16; cell++ )
	{
		MuellerMatrixFlag[ cell ] = false;
		InverseMuellerMatrixFlag[ cell ] = false;
	}

} // Data::Create

//
//	DeleteCache()
//
//	CJS: 16/08/2021
//
//	Deletes the files in a cache.
//

void Data::DeleteCache()
{

	for ( int stageID = 0; stageID < Stages; stageID++ )
	{

		// build filename.
		char filename[ 255 ];
		if (_param->CacheLocation[0] != '\0')
			sprintf( filename, "%s%s-%i-%i-cache.dat", _param->CacheLocation, _param->OutputPrefix, _mosaicID, stageID );
		else
			sprintf( filename, "%s-%i-%i-cache.dat", _param->OutputPrefix, _mosaicID, stageID );

		// remove file.
		remove( filename );

	}

} // DeleteCache

//
//	FreeData()
//
//	CJS: 27/03/2019
//
//	Free the memory used to store the data for a mosaic image. If an offset is supplied then we don't free the array, we only free the cells after the offset.
//

void Data::FreeData( int pWhatData )
{

	// free the memory.
	if (Visibility != NULL)
		for ( int s = 0; s < _stokesImages; s++ )
			for ( int t = 0; t < _taylorTerms; t++ )
				if ((pWhatData & DATA_VISIBILITIES) == DATA_VISIBILITIES && Visibility[ s ][ t ] != NULL)
				{
					free( (void *) Visibility[ s ][ t ] );
					Visibility[ s ][ t ] = NULL;
				}
	if ((pWhatData & DATA_GRID_POSITIONS) == DATA_GRID_POSITIONS && GridPosition != NULL)
	{
		free( (void *) GridPosition );
		GridPosition = NULL;
	}
	if ((pWhatData & DATA_KERNEL_INDEXES) == DATA_KERNEL_INDEXES && KernelIndex != NULL)
	{
		free( (void *) KernelIndex );
		KernelIndex = NULL;
	}
	if ((pWhatData & DATA_DENSITIES) == DATA_DENSITIES && DensityMap != NULL)
	{
		free( (void *) DensityMap );
		DensityMap = NULL;
	}
	if (Weight != NULL)
		for ( int s = 0; s < _stokesImages; s++ )
			if ((pWhatData & DATA_WEIGHTS) == DATA_WEIGHTS && Weight[ s ] != NULL)
			{
				free( (void *) Weight[ s ] );
				Weight[ s ] = NULL;
			}
	if (_param->Deconvolver == MFS && MfsWeight != NULL)
		for ( int s = 0; s < _stokesImages; s++ )
			for ( int t = 0; t < (_taylorTerms - 1) * 2; t++ )
				if ((pWhatData & DATA_MFS_WEIGHTS) == DATA_MFS_WEIGHTS && MfsWeight[ s ][ t ] != NULL)
				{
					free( (void *) MfsWeight[ s ][ t ] );
					MfsWeight[ s ][ t ] = NULL;
				}
	if (ResidualVisibility != NULL)
		for ( int s = 0; s < _stokesImages; s++ )
			for ( int t = 0; t < _taylorTerms; t++ )
				if ((pWhatData & DATA_RESIDUAL_VISIBILITIES) == DATA_RESIDUAL_VISIBILITIES && ResidualVisibility[ s ][ t ] != NULL)
				{
					free( (void *) ResidualVisibility[ s ][ t ] );
					ResidualVisibility[ s ][ t ] = NULL;
				}

} // FreeData

//
//	Data::FreeJonesMatrices()
//
//	CJS: 24/02/2022
//
//	Release the Jones matrices.
//

void Data::FreeJonesMatrices()
{

	// free Jones matrices.
	if (JonesMatrix != NULL)
	{
		for ( int cell = 0; cell < 4; cell++ )
			if (JonesMatrix[ cell ] != NULL)
				free( (void *) JonesMatrix[ cell ] );
		free( (void *) JonesMatrix );
		JonesMatrix = NULL;
	}
	if (JonesMatrixIn != NULL)
		for ( int cell = 0; cell < 4; cell++ )
			if (JonesMatrixIn[ cell ] != NULL)
			{
				free( (void *) JonesMatrixIn[ cell ] );
				JonesMatrixIn[ cell ] = NULL;
			}

} // Data::FreeJonesMatrices

//
//	Data::FreeMuellerMatrices()
//
//	CJS: 24/02/2022
//
//	Release the Mueller matrices.
//

void Data::FreeMuellerMatrices()
{
	
	if (MuellerMatrix != NULL)
	{
		for ( int cell = 0; cell < 16; cell++ )
			if (MuellerMatrix[ cell ] != NULL)
				free( (void *) MuellerMatrix[ cell ] );
		free( (void *) MuellerMatrix );
		MuellerMatrix = NULL;
	}
	if (InverseMuellerMatrix != NULL)
	{
		for ( int cell = 0; cell < 16; cell++ )
			if (InverseMuellerMatrix[ cell ] != NULL)
				free( (void *) InverseMuellerMatrix[ cell ] );
		free( (void *) InverseMuellerMatrix );
		InverseMuellerMatrix = NULL;
	}

} // Data::FreeMuellerMatrices

//
//	Data::FreeOffAxisMuellerMatrices()
//
//	CJS: 24/02/2022
//
//	Release the Mueller matrix cells that are off-axis.
//

void Data::FreeOffAxisMuellerMatrices()
{

	for ( int i = 0; i < 4; i++ )
		for ( int j = 0; j < 4; j++ )
			if (i != j)
			{
				if (MuellerMatrix[ (j * 4) + i ] != NULL)
				{
					free( (void *) MuellerMatrix[ (j * 4) + i ] );
					MuellerMatrix[ (j * 4) + i ] = NULL;
				}
				MuellerMatrixFlag[ (j * 4) + i ] = false;
				if (InverseMuellerMatrix[ (j * 4) + i ] != NULL)
				{
					free( (void *) InverseMuellerMatrix[ (j * 4) + i ] );
					InverseMuellerMatrix[ (j * 4) + i ] = NULL;
				}
				InverseMuellerMatrixFlag[ (j * 4) + i ] = false;
				
			}

} // Data::FreeOffAxisMuellerMatrices

//
//	Data::FreeUnwantedMuellerMatrices()
//
//	CJS: 24/02/2022
//
//	Release the Mueller matrix cells that are not needed for this Stokes product.
//

void Data::FreeUnwantedMuellerMatrices( int pStokes )
{
			
	for ( int i = 0; i < 4; i++ )
		if (i != pStokes)
			for ( int j = 0; j < 4; j++ )
			{	
				if (MuellerMatrix[ (j * 4) + i ] != NULL)
				{
					free( (void *) MuellerMatrix[ (j * 4) + i ] );
					MuellerMatrix[ (j * 4) + i ] = NULL;
				}
				MuellerMatrixFlag[ (j * 4) + i ] = false;
				if (InverseMuellerMatrix[ (i * 4) + j ] != NULL)
				{
					free( (void *) InverseMuellerMatrix[ (i * 4) + j ] );
					InverseMuellerMatrix[ (i * 4) + j ] = NULL;
				}
				InverseMuellerMatrixFlag[ (i * 4) + j ] = false;
			}

} // Data::FreeUnwantedMuellerMatrices

//
//	GenerateAveragePrimaryBeam()
//
//	CJS: 21/03/2022
//
//	Processes the generated/loaded Jones matrices to get an average primary beam.
//
	
void Data::GenerateAveragePrimaryBeam
			(
			int pNumSpws,					// the number of spectral windows
			int * pNumChannels,				// the number of channels per spw
			bool ** pSpwChannelFlag,			// the flag for each spw and channel
			double ** pWavelength				// the wavelength of each spw and channel
			)
{
	
	// work out which channel is closest to the average wavelength.
	int averageChannel = 0;
	int cumulativeChannel = 0;
	double bestError = 0.0;
	for ( int spw = 0; spw < pNumSpws; spw++ )
		for ( int channel = 0; channel < pNumChannels[ spw ]; channel++, cumulativeChannel++ )
			if (pSpwChannelFlag[ spw ][ channel ] == false)
				if (abs( pWavelength[ spw ][ channel ] - AverageWavelength ) < bestError || (spw == 0 && channel == 0))
				{
					bestError = abs( pWavelength[ spw ][ channel ] - AverageWavelength );
					averageChannel = cumulativeChannel;
				}

	// construct the Jones matrix for the average channel.
	cufftComplex ** hstMedianJonesIn = (cufftComplex **) malloc( 4 * sizeof( cufftComplex * ) );
	for ( int cell = 0; cell < 4; cell++ )
		if (JonesMatrixIn[ cell ] != NULL)
		{
			hstMedianJonesIn[ cell ] = (cufftComplex *) malloc( _param->BeamInSize * _param->BeamInSize * sizeof( cufftComplex ) );
			memcpy( hstMedianJonesIn[ cell ], &JonesMatrixIn[ cell ][ averageChannel * _param->BeamInSize * _param->BeamInSize ],
					_param->BeamInSize * _param->BeamInSize * sizeof( cufftComplex ) );
		}
		else
			hstMedianJonesIn[ cell ] = NULL;
			
	// convert the Jones matrix for the average channel to a Mueller matrix.
	cufftDoubleComplex ** devMedianMuellerIn = ConvertJonesToMueller(	/* pJonesMatrix = */ hstMedianJonesIn,
										/* pImageSize = */ _param->BeamInSize );
														
	// free Jones matrix.
	if (hstMedianJonesIn != NULL)
	{
		for ( int cell = 0; cell < 4; cell++ )
			if (hstMedianJonesIn[ cell ] != NULL)
				free( (void *) hstMedianJonesIn[ cell ] );
		free( (void *) hstMedianJonesIn );
	}

	// get the primary beam pattern from the Mueller matrix.
	AveragePrimaryBeamIn = (float *) malloc( _param->BeamInSize * _param->BeamInSize * sizeof( float ) );
	memset( AveragePrimaryBeamIn, 0, _param->BeamInSize * _param->BeamInSize * sizeof( float ) );

	// calculate the determinant from the median Mueller matrix, and convert it to floats.
	cufftDoubleComplex * devDeterminant = determinantImageMatrix(	/* pdevMatrix = */ devMedianMuellerIn,
										/* pMatrixSize = */ 4,
										/* pImageSize = */ _param->BeamInSize );

	// define the block/thread dimensions.
	setThreadBlockSize2D( _param->BeamInSize, _param->BeamInSize, _gridSize2D, _blockSize2D );
	float * devTmp = NULL;
	reserveGPUMemory( (void **) &devTmp, _param->BeamInSize * _param->BeamInSize * sizeof( float ), "reserving device memory for temporary image", __LINE__ );
	devConvertImage<<< _gridSize2D, _blockSize2D >>>(	/* pOut = */ devTmp,
								/* pIn = */ devDeterminant,
								/* pSize = */ _param->BeamInSize );

	// store the determinant on the host. we will reproject it to the correct size later.
	_muellerDeterminantIn = (float *) malloc( _param->BeamInSize * _param->BeamInSize * sizeof( float ) );
	moveDeviceToHost( (void *) _muellerDeterminantIn, (void *) devTmp, _param->BeamInSize * _param->BeamInSize * sizeof( float ),
					"copying Mueller determinant to host", __LINE__ );

	// get the required primary beam from the Mueller matrix.
	int cell = 0;
	if (_param->Stokes == STOKES_I) cell = 0;
	if (_param->Stokes == STOKES_Q) cell = 5;
	if (_param->Stokes == STOKES_U) cell = 10;
	if (_param->Stokes == STOKES_V) cell = 15;
	if (devMedianMuellerIn[ cell ] != NULL)
		devConvertImage<<< _gridSize2D, _blockSize2D >>>(	/* pOut = */ devTmp,
									/* pIn = */ devMedianMuellerIn[ cell ],
									/* pSize = */ _param->BeamInSize );

	// copy primary beam to the host.
	moveDeviceToHost( (void *) AveragePrimaryBeamIn, (void *) devTmp, _param->BeamInSize * _param->BeamInSize * sizeof( float ),
															"copying primary beam to the host", __LINE__ );

	// free memory.
	if (devMedianMuellerIn != NULL)
	{
		for ( int cell = 0; cell < 16; cell++ )
			if (devMedianMuellerIn[ cell ] != NULL)
				cudaFree( (void *) devMedianMuellerIn[ cell ] );
		free( (void *) devMedianMuellerIn );
	}
	if (devDeterminant != NULL)
		cudaFree( (void *) devDeterminant );
	if (devTmp != NULL)
		cudaFree( (void *) devTmp );

	// normalise the primary beam to unity.
	double maxValue = 0.0;
	for ( int i = 0; i < _param->BeamInSize * _param->BeamInSize; i++ )
		if (AveragePrimaryBeamIn[ i ] > maxValue)
			maxValue = AveragePrimaryBeamIn[ i ];
	if (maxValue > 0.0)
		for ( int i = 0; i < _param->BeamInSize * _param->BeamInSize; i++ )
			AveragePrimaryBeamIn[ i ] /= maxValue;

	// save a generated primary beam, but only for the first mosaic component.
	if (_mosaicID == 0)
	{
		char beamFilename[ 100 ];
		sprintf( beamFilename, "%s%s", _param->OutputPrefix, Parameters::PRIMARY_BEAM_EXTENSION );
		_casacoreInterface->WriteCasaImage(	/* pFilename = */ beamFilename,
							/* pWidth = */ _param->BeamInSize,
							/* pHeight = */ _param->BeamInSize,
							/* pRA = */ _param->OutputRA,
							/* pDec = */ _param->OutputDEC,
							/* pPixelSize = */ _param->BeamInCellSize,
							/* pImage = */ AveragePrimaryBeamIn,
							/* pFrequency = */ CONST_C / AverageWavelength,
							/* pMask = */ NULL,
							/* pDirectionType = */ CasacoreInterface::J2000,
							/* pStokesImages = */ 1 );
	}

} // Data::GenerateAveragePrimaryBeam

//
//	Data::GenerateMuellerMatrix()
//
//	CJS: 16/02/2022
//
//	Generates the Mueller matrix and its inverse for all pb-correction channels.
//

void Data::GenerateMuellerMatrix
			(
			int pPBChannel,				// the channel
			int pImageSize,				// the size of the image held in each matrix cell
			float * pdevPrimaryBeam,			// the primary beam at the average wavelength.
			float pPBMaxValue				// the maximum absolute value of the primary beam
			)
{

	// and set the thread block size for processing images on the device.
	setThreadBlockSize2D( pImageSize, pImageSize, _gridSize2D, _blockSize2D );

	int items = pImageSize * pImageSize;
	int stages = items / MAX_THREADS;
	if (items % MAX_THREADS != 0)
		stages++;

	// step 1:	convert the Jones matrix to a Mueller matrix.
	cufftDoubleComplex ** devMuellerMatrix = ConvertJonesToMueller(	/* pJonesMatrix = */ JonesMatrix,
										/* pImageSize = */ pImageSize );

//if (pPBChannel == 0)
//for (int cell = 0; cell < 16; cell++)
//if (devM1[ cell ] != NULL)
//{

//cufftDoubleComplex * tmp = (cufftDoubleComplex *) malloc( pImageSize * pImageSize * sizeof( cufftDoubleComplex ) );
//cudaMemcpy( tmp, devM1[ cell ], pImageSize * pImageSize * sizeof( cufftDoubleComplex ), cudaMemcpyDeviceToHost );
//for ( long int i = 0; i < pImageSize * pImageSize; i++ )
//	((float *) tmp)[ i ] = (float) tmp[ i ].x;
//char filename[100];
//sprintf( filename, "m1-%i", cell );
//_casacoreInterface->WriteCasaImage( filename, pImageSize, pImageSize, 0.0, 0.0, 1.0, (float *) tmp, 1.0, NULL, CasacoreInterface::J2000, 1 );
//free( tmp );

//}

	// compute the adjusted inverse-Mueller matrix, which results in a matrix that falls away to zero near the edges.
	cufftDoubleComplex * devAdjustedInverseMueller = NULL;
	reserveGPUMemory( (void **) &devAdjustedInverseMueller, pImageSize * pImageSize * sizeof( cufftDoubleComplex ),
						"reserving device memory for adjusted Mueller", __LINE__ );
														
	// only adjust the diagonal elements of the matrix.
//	for ( int cell = 0; cell < 16; cell++ )
//		if ((cell == 0 || cell == 5 || cell == 10 || cell == 15) && devInverseMuellerMatrix[ cell ] != NULL)
//		{
			
			// adjust the inverse Mueller matrix so that each pixel is multiplied by:
			//
			//	       1
			//	------------------
			//	1 + [    Max     ]^2
			//	    [ ---------- ]
			//	    [ 12 x |Val| ]
			//
//			devCreateAdjustedMueller<<< _gridSize2D, _blockSize2D >>>(	/* pAdjustedMueller = */ devAdjustedInverseMueller,
//											/* pMueller = */ devInverseMuellerMatrix[ cell ],
//											/* pSize = */ pImageSize );

			// update the Mueller matrix with its adjusted version.
//			cudaMemcpy( devInverseMuellerMatrix[ cell ], devAdjustedInverseMueller, pImageSize * pImageSize * sizeof( cufftDoubleComplex ),
//																		cudaMemcpyDeviceToDevice );
											
//		}

	// free the adjusted mueller matrix.
	if (devAdjustedInverseMueller != NULL)
		cudaFree( (void *) devAdjustedInverseMueller );

	// step 5:	calculate the Mueller matrix by taking the inverse, which we will use for degridding.
	//
	//			( I' ) = 1/2 ( XX* + YY* ) = Mu . ( I )
	//			( Q' )       ( XX* - YY* )        ( Q )
	//			( U' )       ( YX* + XY* )        ( U )
	//			( V' )       ( YX* - XY* )        ( V )
	//
//	cufftDoubleComplex ** devMuellerMatrix = calculateInverseImageMatrix(	/* pdevMatrix = */ devInverseMuellerMatrix,
//											/* pMatrixSize = */ 4,
//											/* pImageSize = */ pImageSize,
//											/* pDivideByDeterminant = */ true );
											
	//cudaFree( (void *) devInverseMuellerMatrix );
	cufftDoubleComplex ** devInverseMuellerMatrix = calculateInverseImageMatrix(	/* pdevMatrix = */ devMuellerMatrix,
											/* pMatrixSize = */ 4,
											/* pImageSize = */ pImageSize,
											/* pDivideByDeterminant = */ false );
		
	// free the Mueller and inverse-Mueller matrices.
	FreeMuellerMatrices();

	// create the Mueller matrix cells.
	MuellerMatrix = (cufftComplex **) malloc( 16 * sizeof( cufftComplex * ) );
	for ( int cell = 0; cell < 16; cell++ )
		MuellerMatrix[ cell ] = NULL;
	InverseMuellerMatrix = (cufftComplex **) malloc( 16 * sizeof( cufftComplex * ) );
	for ( int cell = 0; cell < 16; cell++ )
		InverseMuellerMatrix[ cell ] = NULL;

	// copy the Mueller matrix and the inverse Mueller matrix to the host.
	cufftComplex * devTmp = NULL;
	reserveGPUMemory( (void **) &devTmp, pImageSize * pImageSize * sizeof( cufftComplex ), "reserving device memory for a temporary image", __LINE__ );
	for ( int cell = 0; cell < 16; cell++ )
	{
	
		// convert from double to single precision.
		if (devMuellerMatrix[ cell ] != NULL)
		{
			MuellerMatrix[ cell ] = (cufftComplex *) malloc( pImageSize * pImageSize * sizeof( cufftComplex ) );
			devConvertImage<<< _gridSize2D, _blockSize2D >>>(	/* pOut = */ devTmp,
										/* pIn = */ devMuellerMatrix[ cell ],
										/* pSize = */ pImageSize );
			moveDeviceToHost( (void *) MuellerMatrix[ cell ], (void *) devTmp,
						pImageSize * pImageSize * sizeof( cufftComplex ), "copying Mueller matrix cell to the host", __LINE__ );
			cudaFree( (void *) devMuellerMatrix[ cell ] );
		}
		MuellerMatrixFlag[ cell ] = (MuellerMatrix[ cell ] != NULL);
	
		if (devInverseMuellerMatrix[ cell ] != NULL)
		{
			InverseMuellerMatrix[ cell ] = (cufftComplex *) malloc( pImageSize * pImageSize * sizeof( cufftComplex ) );
			devConvertImage<<< _gridSize2D, _blockSize2D >>>(	/* pOut = */ devTmp,
										/* pIn = */ devInverseMuellerMatrix[ cell ],
										/* pSize = */ pImageSize );
			moveDeviceToHost( (void *) InverseMuellerMatrix[ cell ], (void *) devTmp,
						pImageSize * pImageSize * sizeof( cufftComplex ), "copying inverse Mueller matrix cell to the host", __LINE__ );
			cudaFree( (void *) devInverseMuellerMatrix[ cell ] );
		}
		InverseMuellerMatrixFlag[ cell ] = (InverseMuellerMatrix[ cell ] != NULL);
		
	}
	cudaFree( (void *) devTmp );
	free( (void *) devMuellerMatrix );
	free( (void *) devInverseMuellerMatrix );

} // GenerateMuellerMatrix

//
//	PerformUniformWeighting()
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

void Data::PerformUniformWeighting( double ** phstTotalWeightPerCell )
{

	AverageWeight = (double *) malloc( _param->NumStokesImages * sizeof( double ) );
	for ( int s = 0; s < _param->NumStokesImages; s++ )
		AverageWeight[ s ] = 0.0;
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
			AverageWeight[ s ] += phstTotalWeightPerCell[ s ][ i ];
		AverageWeight[ s ] /= (double) pow( (long int) _param->ImageSize, 2 );

		// normalise the gridded weights using the average weight.
		if (AverageWeight[ s ] > 0.0)
			for ( long int i = 0; i < (long int) _param->ImageSize * (long int) _param->ImageSize; i++ )
				phstTotalWeightPerCell[ s ][ i ] /= AverageWeight[ s ];

		// reset the average weight. we will compute it per visibility now.
		AverageWeight[ s ] = 0.0;
	
	} // LOOP: s

	//
	// update the weight in each cell using:
	//
	//	weight[ cell ] = weight[ cell ] / total_weight[ cell ]
	//
	griddedVisibilities += GriddedVisibilities;
	for ( int stageID = 0; stageID < Stages; stageID++ )
	{

		// get the weights, densities and grid positions from the file for this stage.
		if (_param->CacheData == true)
			UncacheData(	/* pStageID = */ stageID,
					/* pTaylorTerm = */ -1,
					/* pOffset = */ 0,
					/* pWhatData = */ DATA_WEIGHTS | DATA_GRID_POSITIONS | DATA_DENSITIES,
					/* pStokes = */ -1 );

		// divide each weight by the total weight in that cell, and add up the weights so we can make an average.
		for ( long int i = 0; i < NumVisibilities[ stageID ]; i++ )
		{
			VectorI grid = GridPosition[ i ];
			if (grid.u >= 0 && grid.u < _param->ImageSize && grid.v >= 0 && grid.v < _param->ImageSize)
				for ( int s = 0; s < _param->NumStokesImages; s++ )
				{			
					Weight[ s ][ i ] /= phstTotalWeightPerCell[ s ][ (grid.v * _param->ImageSize) + grid.u ];
					AverageWeight[ s ] += (double) Weight[ s ][ i ] * (double) DensityMap[ i ];
				}
		}

		// re-cache the weights and free the densities and grid positions for this stage.
		if (_param->CacheData == true)
		{
			CacheData(	/* pStageID = */ stageID,
					/* pTaylorTerm = */ -1,
					/* pWhatData = */ DATA_WEIGHTS );
			FreeData( /* pWhatData = */ DATA_DENSITIES | DATA_GRID_POSITIONS );
		}

	}
	
	// calculate average weights by dividing by the number of gridded visibilities.
	for ( int s = 0; s < _param->NumStokesImages; s++ )
		AverageWeight[ s ] /= (double) griddedVisibilities;

} // Data::PerformUniformWeighting

//
//	PerformRobustWeighting()
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

void Data::PerformRobustWeighting( double ** phstTotalWeightPerCell )
{

	AverageWeight = (double *) malloc( _param->NumStokesImages * sizeof( double ) );
	for ( int s = 0; s < _param->NumStokesImages; s++ )
		AverageWeight[ s ] = 0.0;
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

	griddedVisibilities += GriddedVisibilities;
	for ( int stageID = 0; stageID < Stages; stageID++ )
	{

		// get the grid positions, densities and weights.
		if (_param->CacheData == true)
			UncacheData(	/* pStageID = */ stageID,
					/* pTaylorTerm = */ -1,
					/* pOffset = */ 0,
					/* pWhatData = */ DATA_GRID_POSITIONS | DATA_WEIGHTS | DATA_DENSITIES,
					/* pStokes = */ -1 );

		// update the weight of each visibility using the original weight, the sum of weights in the cell, and the f^2 parameter. also, add
		// up the weights so we can construct an average.
		for ( long int i = 0; i < NumVisibilities[ stageID ]; i++ )
			if (	GridPosition[ i ].u >= 0 && GridPosition[ i ].u < _param->ImageSize &&
				GridPosition[ i ].v >= 0 && GridPosition[ i ].v < _param->ImageSize)
				for ( int s = 0; s < _param->NumStokesImages; s++ )
				{
					Weight[ s ][ i ] /= (1.0 + (phstTotalWeightPerCell[ s ][ (GridPosition[ i ].v * _param->ImageSize) + GridPosition[ i ].u ] * fSquared));
					AverageWeight[ s ] += (double) Weight[ s ][ i ] * (double) DensityMap[ i ];
				}

		// re-cache the weights and free the densities and grid positions for this stage.
		if (_param->CacheData == true)
		{
			CacheData(	/* pStageID = */ stageID,
					/* pTaylorTerm = */ -1,
					/* pWhatData = */ DATA_WEIGHTS );
			FreeData( /* pWhatData = */ DATA_DENSITIES | DATA_GRID_POSITIONS );
		}

	}
	
	// calculate average weights by dividing by the number of gridded visibilities.
	for ( int s = 0; s < _param->NumStokesImages; s++ )
		AverageWeight[ s ] /= (double) griddedVisibilities;

} // Data::PerformRobustWeighting

//
//	addMosaicComponent()
//
//	CJS: 20/08/2021
//
//	Adds a new mosaic component object.
//

void Data::addMosaicComponent( vector<Data> & pData )
{

	Data newComponent;
	pData.push_back( newComponent );
	pData[ pData.size() - 1 ].Create(	/* pTaylorTerms = */ _taylorTerms,
						/* pMosaicID = */ pData.size() - 1,
						/* pWProjection = */ _wProjection,
						/* pAProjection = */ _aProjection,
						/* pWPlanes = */ WPlanes,
						/* pPBChannels = */ PBChannels,
						/* pCacheData = */ _cacheData,
						/* pStokes = */ _stokes,
						/* pStokesImages = */ _stokesImages );
						
	// set up the pointers.
	if (pData.size() > 1)
		pData[ pData.size() - 2 ].NextComponent = &pData[ pData.size() - 1 ];

} // addMosaicComponent

//
//	getPolarisationMultiplier()
//
//	CJS: 07/04/2020
//
//	Gets an array of multiplier that describe how the polarisation products should be handled.
//
//		R = X + iY		X = (R + L) / 2
//		L = X - iY		Y = -i(R - L) / 2
//		-->
//		Stokes I = (RR* + LL*) / 2 = XX* + YY*
//		Stokes V = (RR* - LL*) / 2 = i(YX* - XY*)
//		Stokes Q = (RL* + LR*) / 2 = XX* - YY*
//		Stokes U = (RL* - LR*) / 2 = i(XY* + YX*)
//
//		XX* = (I + Q) / 2
//		XY* = -i(U - V) / 2
//		YX* = -i(U + V) / 2
//		YY* = (I - Q) / 2
//
//	This maths implies I shouldn't need the factor of 1/2 on the linear polarisations, but for some reason they do need to be there.
//

double ** Data::getPolarisationMultiplier( char * pMeasurementSetFilename, int * pNumPolarisations, int * pNumPolarisationConfigurations, char * pTableData )
{

	// return value.	
	double ** hstMultiplier = NULL;

	// get a list of polarisations.
	int * hstPolarisation = NULL;
	_casacoreInterface->GetPolarisations(	/* pMeasurementSet = */ (pTableData[ 0 ] == '\0' ? pMeasurementSetFilename : pTableData),
						/* pNumPolarisations = */ pNumPolarisations,
						/* pNumPolarisationConfigurations = */ pNumPolarisationConfigurations,
						/* pPolarisation = */ &hstPolarisation );

	// create a list of multipliers for constructing visibilities from the Stokes parameters.
	if (*pNumPolarisationConfigurations > 0 && *pNumPolarisations > 0)
	{
		hstMultiplier = (double **) malloc( _stokesImages * sizeof( double * ) );
		for ( int s = 0; s < _stokesImages; s++ )
		{
			hstMultiplier[ s ] = (double *) malloc( (*pNumPolarisationConfigurations) * (*pNumPolarisations) * sizeof( double ) );
			memset( hstMultiplier[ s ], 0, (*pNumPolarisationConfigurations) * (*pNumPolarisations) * sizeof( double ) );
		}
	}

	// if we have at least one polarisation product then we will check if we have the right one(s).
	if (*pNumPolarisations >= 1)
		for ( int config = 0; config < (*pNumPolarisationConfigurations); config++ )
		{

			const int UNDEF_CONST = 0, I_CONST = 1, Q_CONST = 2, U_CONST = 3, V_CONST = 4, RR_CONST = 5, RL_CONST = 6, LR_CONST = 7, LL_CONST = 8,
					XX_CONST = 9, XY_CONST = 10, YX_CONST = 11, YY_CONST = 12;

			// get pointer to these polarisations and multipliers.
			int * polarisationPtr = &hstPolarisation[ (config * (*pNumPolarisations)) ];

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
			
			for ( int stokesProduct = 0; stokesProduct < _stokesImages; stokesProduct++ )
			{
			
				double * multiplierPtr = &hstMultiplier[ stokesProduct ][ (config * (*pNumPolarisations)) ];

				// make sure we've got what we need.
				int whichStokes = _stokes;
				bool requestedImage = true;
				if (_aProjection == true && _param->LeakageCorrection == true)
				{
					whichStokes = stokesProduct;
					requestedImage = (whichStokes == _stokes);
				}
			
				if (whichStokes == STOKES_Q && (stokesQ == false) && (xx == false || yy == false) && (rl == false || lr == false))
					whichStokes = STOKES_NONE;
				if (whichStokes == STOKES_U && (stokesU == false) && (xy == false || yx == false) && (rl == false || lr == false))
					whichStokes = STOKES_NONE;
				if (whichStokes == STOKES_V && (stokesV == false) && (xy == false || yx == false) && (rr == false || ll == false))
					whichStokes = STOKES_NONE;
				if (whichStokes == STOKES_I && (stokesI == false) && (xx == false || yy == false) && (rr == false || ll == false))
					whichStokes = STOKES_NONE;

				// display a warning if we can't do the requested Stokes imaging.
				if (whichStokes == STOKES_NONE && _stokes != STOKES_NONE && requestedImage == true)
				{
	
					printf( "WARNING: Polarisation configuration %i does have the correct polarisation products to image Stokes ", config );
					switch (_stokes)
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
				
			} // LOOP: stokesProduct

	}

	// free the polarisations and multiplier.
	if (hstPolarisation != NULL)
		free( (void *) hstPolarisation );

	// return something.
	return hstMultiplier;

} // getPolarisationMultiplier

//
//	ProcessMeasurementSet()
//
//	CJS: 05/07/2019
//
//	Loads the measurement sets into memory, and caches them so we don't have to store them all at the same time.
//
//	Notes:
//		The TableData[] array holds the location of the tables, such as FIELD, ANTENNA, etc. If this parameter is empty then we use the measurement
//			set filename. The only times where we want to use a different location is when we are processing a MMS file, where all the tables will
//			be held in a different location.
//

void Data::ProcessMeasurementSet
			(
			int pFileIndex,
			double ** phstTotalWeightPerCell,
			vector<Data> & pData
			)
{

	// store the current number of mosaic components.
	int firstMosaicComponent = pData.size() - 1;

	// channels.
	int * hstNumChannels = NULL;
	int numSpws = 0;
	bool ** hstSpwChannelFlag = NULL;

	// get the total system memory.
	struct sysinfo memInfo;
	sysinfo( &memInfo );
	long int hstMaxMemory = memInfo.totalram / 2;
	
	printf( "\nProcessing measurement set: %s\n", _param->MeasurementSetPath[ pFileIndex ] );
	printf( "------------------------------------------------------------------------------\n\n" );

	// load channel wavelengths.
	double ** hstWavelength = NULL;
	_casacoreInterface->GetWavelengths(	/* pMeasurementSet = */ (_param->TableData[ pFileIndex ][ 0 ] == '\0' ? _param->MeasurementSetPath[ pFileIndex ] : _param->TableData[ pFileIndex ]),
						/* pNumSpws = */ &numSpws,
						/* pNumChannels = */ &hstNumChannels,
						/* pWavelength = */ &hstWavelength );

	// get the polarisation multiplier (and the number of polarisations) which describes how the polarisation products should be handled.
	int hstNumPolarisations = -1, hstNumPolarisationConfigurations = -1;
	double ** hstMultiplier = getPolarisationMultiplier(	/* pMeasurementSetFilename = */ _param->MeasurementSetPath[ pFileIndex ],
								/* pNumPolarisations = */ &hstNumPolarisations,
								/* pNumPolarisationConfigurations = */ &hstNumPolarisationConfigurations,
								/* pTableData = */ _param->TableData[ pFileIndex ] );

	// upload the polarisation multipliers to the device.
	double ** devMultiplier = (double **) malloc( _param->NumStokesImages * sizeof( double * ) );
	if (hstNumPolarisations > 0 && hstNumPolarisationConfigurations > 0)
		for ( int s = 0; s < _param->NumStokesImages; s++ )
		{
			reserveGPUMemory( (void **) &devMultiplier[ s ], hstNumPolarisationConfigurations * hstNumPolarisations * sizeof( double ),
						"declaring device memory for polarisation multipliers", __LINE__ );
			cudaMemcpy( devMultiplier[ s ], hstMultiplier[ s ], hstNumPolarisationConfigurations * hstNumPolarisations * sizeof( double ),
						cudaMemcpyHostToDevice );
		}

	// get the data description info, which is the polarisation configuration id and spectral window id.
	int numDataDescItems = -1;
	int * hstDataDescPolarisationConfig = NULL, * hstDataDescSpw = NULL;
	_casacoreInterface->GetDataDesc(	/* pMeasurementSet = */ (_param->TableData[ pFileIndex ][ 0 ] == '\0' ? _param->MeasurementSetPath[ pFileIndex ] : _param->TableData[ pFileIndex ]),
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
				/* phstSpwRestriction = */ _param->SpwRestriction[ pFileIndex ] );

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
		_casacoreInterface->GetSamples(	/* IN: pMeasurementSet = */ _param->MeasurementSetPath[ pFileIndex ],
							/* OUT: pNumSamples = */ &hstNumSamples,
							/* OUT: pSample = */ (double **) &hstSample,
							/* IN: pFieldID = */ _param->FieldID[ pFileIndex ],
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

	}

	// hold the number of fields found in each file, and the phase centre of each field, along with the phase centre of each image and the phase centre
	// of each beam.
	int hstNumFields = 0;
	double * hstFieldPhaseFrom = NULL;
	double * hstFieldPhaseTo = NULL;

	// load the phase centres for each field.
	_casacoreInterface->GetPhaseCentres(	/* pMeasurementSet = */ (_param->TableData[ pFileIndex ][ 0 ] == '\0' ? _param->MeasurementSetPath[ pFileIndex ] : _param->TableData[ pFileIndex ]),
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

	// turn on mosaicing, and disk caching, if we have multiple fields in this file.
	if (hstNumFields > 1)
	{
		_param->UseMosaicing = true;
		_param->UvPlaneMosaic = (_param->MosaicDomain == UV);
		_param->ImagePlaneMosaic = (_param->MosaicDomain == IMAGE);
		_param->CacheData = true;
	}

	// increase the number of mosaic components by the number of fields.
	for ( int i = 1; i < hstNumFields; i++ )
		addMosaicComponent( /* pData = */ pData );

	// get the antennae from the file.
	bool * hstAntennaFlag = NULL;
	double * hstDishDiameter = NULL;
	int numberOfAntennae = _casacoreInterface->GetAntennae(	/* pMeasurementSet = */ (_param->TableData[ pFileIndex ][ 0 ] == '\0' ? _param->MeasurementSetPath[ pFileIndex ] : 
																		_param->TableData[ pFileIndex ]),
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
	if (_param->Telescope == ALMA && (int) round( minimumDishDiameter ) == 7)
		_param->Telescope = ALMA_7M;
	if (_param->Telescope == ALMA && (int) round( minimumDishDiameter ) == 12)
		_param->Telescope = ALMA_12M;

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

	// store the total, minimum, and maximum wavelength, and the number of valid channels per spw.
	double * hstTmpTotal = (double *) malloc( numSpws * sizeof( double ) );
	double * hstTmpMax = (double *) malloc( numSpws * sizeof( double ) );
	double * hstTmpMin = (double *) malloc( numSpws * sizeof( double ) );
	long int * hstTmpValid = (long int *) malloc( numSpws * sizeof( long int ) );
	int * hstSamplesPerSpw = (int *) malloc( numSpws * sizeof( int ) );
	
	// store maximum wavelength.
	double maximumWavelength = 0.0;

	// calculate the total, minimum, and maximum wavelength, and the number of valid channels per spw.
	for ( int spw = 0; spw < numSpws; spw++ )
	{
		hstTmpTotal[ spw ] = 0.0;
		hstTmpMax[ spw ] = -1.0;
		hstTmpMin[ spw ] = -1.0;
		hstTmpValid[ spw ] = 0;
		hstSamplesPerSpw[ spw ] = 0;
		for ( int channel = 0; channel < hstNumChannels[ spw ]; channel++ )
		{
			
			// get total, max, and min for the data we're actually using.
			if (hstSpwChannelFlag[ spw ][ channel ] == false)
			{
				hstTmpTotal[ spw ] += CONST_C / hstWavelength[ spw ][ channel ];
				if (hstWavelength[ spw ][ channel ] < hstTmpMin[ spw ] || hstTmpMin[ spw ] == -1.0)
					hstTmpMin[ spw ] = hstWavelength[ spw ][ channel ];
				if (hstWavelength[ spw ][ channel ] > hstTmpMax[ spw ] || hstTmpMax[ spw ] == -1.0)
					hstTmpMax[ spw ] = hstWavelength[ spw ][ channel ];
				hstTmpValid[ spw ]++;
			}
			
			// get maximum wavelength in the measurement set.
			if (hstWavelength[ spw ][ channel ] > maximumWavelength || (spw == 0 && channel == 0))
				maximumWavelength = hstWavelength[ spw ][ channel ];
				
		}
	}

	// get the average wavelength by summing over all the data. we're not just summing over the spws and channels, we're summing over the actual visibility data.
	// we need to sum the frequencies, not the wavelengths, and then convert the average frequency into a wavelength.
	long int * hstValidChannels = (long int *) malloc( hstNumFields * sizeof( long int ) );
	double * hstMinWavelength = (double *) malloc( hstNumFields * sizeof( double ) );
	double * hstMaxWavelength = (double *) malloc( hstNumFields * sizeof( double ) );
	for ( int i = 0; i < hstNumFields; i++ )
	{
		pData[ firstMosaicComponent + i ].AverageWavelength = 0.0;
		hstValidChannels[ i ] = 0;
		hstMinWavelength[ i ] = -1.0;
		hstMaxWavelength[ i ] = -1.0;
	}
	for ( int sample = 0; sample < hstNumSamples / 2; sample++ )
	{
		int spw = hstDataDescSpw[ hstDataDescID[ sample ] ];
		int fieldID = hstSampleFieldID[ sample ];
		pData[ firstMosaicComponent + fieldID ].AverageWavelength += hstTmpTotal[ spw ];
		hstSamplesPerSpw[ spw ]++;
		if (hstTmpMin[ spw ] < hstMinWavelength[ fieldID ] || hstMinWavelength[ fieldID ] < 0.0)
			hstMinWavelength[ fieldID ] = hstTmpMin[ spw ];
		if (hstTmpMax[ spw ] > hstMaxWavelength[ fieldID ] || hstMaxWavelength[ fieldID ] < 0.0)
			hstMaxWavelength[ fieldID ] = hstTmpMax[ spw ];
		hstValidChannels[ fieldID ] += hstTmpValid[ spw ];
	}

	// calculate the average wavelength for each field by dividing by the number of channels, and converting from a frequency to a wavelength.
	for ( int field = 0; field < hstNumFields; field++ )
		if (pData[ firstMosaicComponent + field ].AverageWavelength > 0)
			pData[ firstMosaicComponent + field ].AverageWavelength = CONST_C * (double) hstValidChannels[ field ] / pData[ firstMosaicComponent + field ].AverageWavelength;
		else
			pData[ firstMosaicComponent + field ].AverageWavelength = 1.0;

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
	if (hstValidChannels != NULL)
		free( (void *) hstValidChannels );

	// calculate primary-beam channels for this measurement set.
	int ** hstWhichPBChannel = NULL;
	double * hstPBChannelWavelength = NULL;
	hstPBChannelWavelength = (double *) malloc( PBChannels * sizeof( double ) );
	calculatePBChannels(	/* OUT: phstWhichPBChannel = */ &hstWhichPBChannel,
				/* OUT: phstPBChannelWavelength = */ hstPBChannelWavelength,
				/* IN: phstWavelength = */ hstWavelength,
				/* IN: pNumSpws = */ numSpws,
				/* IN: phstNumChannels = */ hstNumChannels,
				/* IN: phstSpwChannelFlag = */ hstSpwChannelFlag );

	for ( int field = 0; field < hstNumFields; field++ )
		printf( "average wavelength for field %i is %6.4f mm (min: %6.4f mm, max %6.4f mm)\n", fieldIDMap[ field ],
				pData[ firstMosaicComponent + field ].AverageWavelength * 1000.0, hstMinWavelength[ field ] * 1000.0,
				hstMaxWavelength[ field ] * 1000.0 );
	printf( "        largest wavelength in MS is %6.4f mm\n\n", maximumWavelength * 1000.0 );
	
	bool loadedPrimaryBeam = (_param->BeamPattern[ pFileIndex ][0] != '\0');

	// if a beam filename was provided in the settings file then we load the beam here.
	if (loadedPrimaryBeam == true)
	{
	
		int totalChannels = 0;
		for ( int spw = 0; spw < numSpws; spw++ )
			totalChannels += hstNumChannels[ spw ];

		loadedPrimaryBeam = loadPrimaryBeam(	/* pBeamFilename = */ _param->BeamPattern[ pFileIndex ],
							/* phstPrimaryBeamIn = */ JonesMatrixIn,
							/* pSize = */ _param->BeamInSize,
							/* pNumChannels = */ totalChannels );
		
	}
	
	if (loadedPrimaryBeam == false)
	{
	
		// clear the primary beam if it exists.
		FreeJonesMatrices();
	
		if (_param->BeamType == AIRY)
		{

			// set primary beam parameters depending upon telescope.
			if (_param->DiskDiameterSupplied == false)
				switch (_param->Telescope)
				{
					case ALMA:
					case ALMA_7M:		{ _param->AiryDiskDiameter = 6.25; break; }
					case ALMA_12M:		{ _param->AiryDiskDiameter = 10.70; break; }
					case ASKAP:		{ _param->AiryDiskDiameter = 12.00; break; }
					case VLA:		{ _param->AiryDiskDiameter = 25.0; break; }
					case MEERKAT:		{ _param->AiryDiskDiameter = 13.5; break; }
					case EMERLIN:		{ _param->AiryDiskDiameter = minimumDishDiameter; break; }
				}
			if (_param->DiskBlockageSupplied == false)
				switch (_param->Telescope)
				{
					case ALMA:
					case ALMA_7M:		{ _param->AiryDiskBlockage = 0.75; break; }
					case ALMA_12M:		{ _param->AiryDiskBlockage = 0.75; break; }
					case ASKAP:		{ _param->AiryDiskBlockage = 0.75; break; }
					case VLA:		{ _param->AiryDiskBlockage = 0.75; break; }
					case MEERKAT:		{ _param->AiryDiskBlockage = 0.75; break; }
					case EMERLIN:		{ _param->AiryDiskBlockage = 0.75; break; }
				}

			// generate primary beams.
			generatePrimaryBeamAiry(	/* phstJonesMatrix = */ JonesMatrixIn,
							/* pWidth = */ _param->AiryDiskDiameter,
							/* pCutout = */ _param->AiryDiskBlockage,
							/* pNumSpws = */ numSpws,
							/* phstNumChannels = */ hstNumChannels,
							/* phstWavelength = */ hstWavelength );
							
		}
		else
			generatePrimaryBeamGaussian(	/* phstJonesMatrix = */ JonesMatrixIn,
							/* pWidth = */ minimumDishDiameter,
							/* pNumSpws = */ numSpws,
							/* phstNumChannels = */ hstNumChannels,
							/* phstWavelength = */ hstWavelength );

	}
	
	for ( int field = hstNumFields - 1; field >= 0; field-- )
	{
	
		// if this is not the first field then copy the Jones matrices
		if (field > 0)
			pData[ firstMosaicComponent + field ].CopyJonesMatrixIn( /* pFromMatrix = */ JonesMatrixIn );

		// generate the average primary beam for each field.
		pData[ firstMosaicComponent + field ].GenerateAveragePrimaryBeam(	/* pNumSpws = */ numSpws,
											/* pNumChannels = */ hstNumChannels,
											/* pSpwChannelFlag = */ hstSpwChannelFlag,
											/* pWavelength = */ hstWavelength );

		// reduce the number of channels if required by selecting the beam pattern from the centre of each channel range.
		pData[ firstMosaicComponent + field ].ReduceJonesMatrixChannels(	/* pNumSpws = */ numSpws,
											/* pNumChannels = */ hstNumChannels,
											/* pSpwChannelFlag = */ hstSpwChannelFlag,
											/* pWhichPBChannel = */ hstWhichPBChannel );
											
	}
		
	// update the maximum wavelength.
	for ( int field = 0; field < hstNumFields; field++ )
		pData[ firstMosaicComponent + field ].MaximumWavelength = maximumWavelength;
		
	//
	//
	// NOTE: hstBeamWidth should be calculated as the 20% width of a Gaussian beam for the maximum dish diameter and the average wavelength for each field.
	//		I will then be able to load/generate the primary beam AFTER loading the data
	//

	// if we are image-plane mosaicing then we need to measure the radius of the beam at the 1% level [in pixels].
	double hstBeamWidth = 0.0;
	if (_param->ImagePlaneMosaic == true)
		hstBeamWidth = getPrimaryBeamWidth(	/* phstBeam = */ pData[ firstMosaicComponent ].AveragePrimaryBeamIn,
							/* pBeamSize = */ _param->BeamInSize );

	// free data.
	if (hstSamplesPerSpw != NULL)
		free( (void *) hstSamplesPerSpw );

	// we get the position of the ASKAP PAF beam, based upon the pointing position of the dish (for which we use the phase position).
	if (_param->Telescope == ASKAP && _param->BeamID[ pFileIndex ] != -1)
	{
		double xOffset = 0.0, yOffset = 0.0;
		_casacoreInterface->GetASKAPBeamOffset(	/* pMeasurementSet = */ _param->MeasurementSetPath[ pFileIndex ],
								/* pBeamID = */ _param->BeamID[ pFileIndex ],
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

	// for image-plane mosaics, work out a suitable phase position for this mosaic component.
	if (_param->ImagePlaneMosaic == true)

		// get suitable phase positions for gridding.
		getSuitablePhasePositionForBeam(	/* pBeamIn = */ hstFieldPhaseFrom,
							/* pPhase = */ hstFieldPhaseTo,
							/* pNumBeams = */ hstNumFields,
							/* pBeamWidth = */ hstBeamWidth );

	// otherwise set the phase position of each field to the required output phase position.
	else
		for ( int field = 0; field < hstNumFields; field++ )
		{
			hstFieldPhaseTo[ field * 2 ] = _param->OutputRA;
			hstFieldPhaseTo[ (field * 2) + 1 ] = _param->OutputDEC;
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

	// calculate w-planes for all mosaic fields.
	for ( int field = 0; field < hstNumFields; field++ )
		pData[ firstMosaicComponent + field ].calculateWPlanes(	/* pFieldID = */ field,
										/* pCasaFieldID = */ fieldIDMap[ field ],
										/* pNumSamples = */ hstNumSamples,
										/* phstSample = */ hstSample,
										/* phstFieldID = */ hstSampleFieldID,
										/* pMinWavelength = */ hstMinWavelength[ field ] );
	printf( "\n" );

	// free the samples and field ids.
	if (hstSample != NULL)
		free( (void *) hstSample );
	if (hstSampleFieldID != NULL)
		free( (void *) hstSampleFieldID );

	// if we're image-plane mosaicing then copy the field phase position into the mosaic phase position so we can construct the mosaic at the end. we currently copy
	// the phase position of the first field since for file mosaicing all the fields in this file must be phase rotated to the same position.
	if (_param->ImagePlaneMosaic == true)
		for ( int field = 0; field < hstNumFields; field++ )
		{
			pData[ firstMosaicComponent + field ].ImagePlaneRA = hstFieldPhaseTo[ (field * 2) + 0 ];
			pData[ firstMosaicComponent + field ].ImagePlaneDEC = hstFieldPhaseTo[ (field * 2) + 1 ];
		}

	// ----------------------------------------------------
	//
	// l o a d   v i s i b i l i t i e s
	//
	// ----------------------------------------------------

	// count the number of antennae, and work out the number of baselines.
	int numberOfBaselines = (unflaggedAntennae * (unflaggedAntennae - 1)) / 2;

	// work out how much memory we need to load all the visibilities, flags, sample IDs and channel IDs.
	int memoryNeededPerVis = /* VIS = */ (sizeof( cufftComplex ) * _param->NumStokesImages) + /* FLAG = */ sizeof( bool ) + /* SAMPLE ID = */ sizeof( int ) +
					/* CHANNEL ID = */ sizeof( int ) + /* GRID POS = */ sizeof( VectorI ) + /* Kernel IDX = */ sizeof( int ) +
					/* DENSITY MAP = */ sizeof( int ) + /* WEIGHT = */ (sizeof( float ) * _param->NumStokesImages) +
					/* FIELD ID = */ sizeof( int ) +
					/* MFS WEIGHT = */ (_param->Deconvolver == MFS ? sizeof( float ) * _param->NumStokesImages * (_param->TaylorTerms - 1) * 2 : 0);

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
	_param->CacheData = _param->CacheData || (baselinesPerStage < numberOfBaselines);
	if (_param->CacheData == true)
		printf( "the data will be loaded in %i batch(es), and cached to disk\n\n", (int) ceil( (double) numberOfBaselines / (double) baselinesPerStage) );
	else
		printf( "the data will be loaded in one batch, and no disk caching will be used\n\n" );

	// clear the total weight per cell, but only if we're not making a UV mosaic. for uv mosaics this total will be added up over all mosaic components.
	if (_param->UvPlaneMosaic == false && (_param->Weighting == ROBUST || _param->Weighting == UNIFORM))
		for ( int s = 0; s < _param->NumStokesImages; s++ )
			memset( phstTotalWeightPerCell[ s ], 0, (long int) _param->ImageSize * (long int) _param->ImageSize * (long int) sizeof( double ) );

	printf( "Retrieving visibilities.....\n" );
	printf( "----------------------------\n\n" );

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

		printf( "\rloading...%3d%%, processing...%3d%%     currently: getting vis for antennae %i,%i to %i,%i",
				(int) ((double) sampleStageID * 100.0 / (double) hstNumSamples), (int) ((double) sampleStageID * 100.0 / (double) hstNumSamples),
				hstAntenna[ startAnt1 ], hstAntenna[ startAnt2 ], hstAntenna[ endAnt1 ], hstAntenna[ endAnt2 ] );
		fflush( stdout );

		// resize the array holding the number of batch records.
		for ( int mosaicID = firstMosaicComponent; mosaicID < firstMosaicComponent + hstNumFields; mosaicID++ )
		{
			pData[ mosaicID ].Stages = stageID + 1;
			pData[ mosaicID ].NumVisibilities = (long int *) realloc( pData[ mosaicID ].NumVisibilities,
											pData[ mosaicID ].Stages * sizeof( long int ) );
			pData[ mosaicID ].Batches = (int *) realloc( pData[ mosaicID ].Batches,
											pData[ mosaicID ].Stages * sizeof( int ) );
		}

		// ----------------------------------------------------
		//
		// f e t c h   v i s i b i l i t i e s   f r o m   m e a s u r e m e n t   s e t
		//
		// ----------------------------------------------------

		// create temporary arrays for the visibilities, flags, weights and field ids.
		cufftComplex * tmpVisibility = NULL;
		bool * tmpFlag = NULL;
		float * tmpWeight = NULL;
		int * tmpDataDescID = NULL;

		// load visibilities. we load ALL the visibilities to the host, and these are then processed on the device in batches.
		int numSamplesInStage = 0;
		_casacoreInterface->GetVisibilities(	/* IN: pFilename = */ _param->MeasurementSetPath[ pFileIndex ],
							/* IN: pFieldID = */ _param->FieldID[ pFileIndex ],
							/* OUT: pNumSamples = */ &numSamplesInStage,
							/* IN: pNumChannels = */ hstNumChannels,
							/* IN: pDataField = */ _param->DataField[ pFileIndex ],
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
		float ** hstSampleWeight = (float **) malloc( _param->NumStokesImages * sizeof( float * ) );
		for ( int stokes = 0; stokes < _param->NumStokesImages; stokes++ )
		{
			hstSampleWeight[ stokes ] = (float *) malloc( numSamplesInStage * 2 * sizeof( float ) );
			for ( int sample = 0; sample < numSamplesInStage; sample++ )
			{
				double weight = 0.0;
				for ( int polarisation = 0; polarisation < hstNumPolarisations; polarisation++ )
					weight += abs( hstMultiplier[ stokes ][ (hstPolarisationConfig[ sample ] * hstNumPolarisations) + polarisation ] ) *
									tmpWeight[ (sample * hstNumPolarisations) + polarisation ];
				hstSampleWeight[ stokes ][ sample ] = weight;
			}
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
		for ( int s = 0; s < _param->NumStokesImages; s++ )
			memcpy( &hstSampleWeight[ s ][ numSamplesInStage ], hstSampleWeight[ s ], numSamplesInStage * sizeof( float ) );
		memcpy( &hstSampleFieldID[ numSamplesInStage ], hstSampleFieldID, numSamplesInStage * sizeof( int ) );
		memcpy( &hstSpw[ numSamplesInStage ], hstSpw, numSamplesInStage * sizeof( int ) );
		memcpy( &hstPolarisationConfig[ numSamplesInStage ], hstPolarisationConfig, numSamplesInStage * sizeof( int ) );

		// we have now got twice as many samples in the stage.
		numSamplesInStage *= 2;

		// calculate how many visibilities we have to process here. this total will be reduced later once we have compacted our data.
		NumVisibilities[ stageID ] = 0;
		for ( int sample = 0; sample < numSamplesInStage; sample++ )
			NumVisibilities[ stageID ] += (long int) hstUnflaggedChannels[ hstSpw[ sample ] ];

		// declare some memory for the sample ID and channel ID for these visibilities.
		SampleID = (int *) malloc( NumVisibilities[ stageID ] * sizeof( int ) );
		ChannelID = (int *) malloc( NumVisibilities[ stageID ] * sizeof( int ) );

		// set the sample id and channel id for each visibility.
		long int visibilityID = 0;
		for ( int sample = 0; sample < numSamplesInStage / 2; sample++ )
			for ( int channel = 0; channel < hstNumChannels[ hstSpw[ sample ] ]; channel++ )
				if (hstSpwChannelFlag[ hstSpw[ sample ] ][ channel ] == false)
				{
					SampleID[ visibilityID ] = sample;
					ChannelID[ visibilityID ] = channel;
					visibilityID++;
				}

		// duplicate samples and channels, and then update the sample ID for the second half of the array.
		memcpy( &SampleID[ NumVisibilities[ stageID ] / 2 ], SampleID, NumVisibilities[ stageID ] * sizeof( int ) / 2 );
		memcpy( &ChannelID[ NumVisibilities[ stageID ] / 2 ], ChannelID, NumVisibilities[ stageID ] * sizeof( int ) / 2 );
		for ( long int visibility = NumVisibilities[ stageID ] / 2; visibility < NumVisibilities[ stageID ]; visibility++ )
			SampleID[ visibility ] += numSamplesInStage / 2;

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

		// calculate the visibilities and the flags on the GPU.
		calculateVisibilityAndFlag(	/* pStageID = */ stageID,
						/* pPreferredVisibilityBatchSize = */ _param->PREFERRED_VISIBILITY_BATCH_SIZE,
						/* pNumPolarisations = */ hstNumPolarisations,
						/* pNumSamplesInStage = */ numSamplesInStage,
						/* phstPolarisationConfig = */ hstPolarisationConfig,
						/* phstVisibilityIn = */ tmpVisibility,
						/* phstFlagIn = */ tmpFlag,
						/* pdevMultiplier = */ devMultiplier );
										
		// free workspace memory.
		if (tmpVisibility != NULL)
			free( (void *) tmpVisibility );
		if (tmpFlag != NULL)
			free( (void *) tmpFlag );
		if (hstPolarisationConfig != NULL)
			free( (void *) hstPolarisationConfig );

		// ----------------------------------------------------
		//
		// c a l c u l a t e   g r i d   p o s i t i o n s
		//
		// ----------------------------------------------------

		calculateGridPositions(	/* pStageID = */ stageID,
						/* pPreferredVisibilityBatchSize = */ _param->PREFERRED_VISIBILITY_BATCH_SIZE,
						/* pNumSpws = */ numSpws,
						/* pNumSamplesInStage = */ numSamplesInStage,
						/* pSampleStageID = */ sampleStageID,
						/* pTotalSamples = */ hstNumSamples,
						/* pOversample = */ _param->Oversample,
						/* pUvCellSize = */ _param->UvCellSize,
						/* phstWavelength = */ hstWavelength,
						/* phstNumChannels = */ hstNumChannels,
						/* phstPhase = */ hstPhase,
						/* phstWhichPBChannel = */ hstWhichPBChannel,
						/* phstSpw = */ hstSpw,
						/* phstSampleWeight = */ hstSampleWeight,
						/* phstSampleFieldID = */ hstSampleFieldID,
						/* phstSample = */ hstSample,
						/* pNumFields = */ hstNumFields,
						/* pNumGPUs = */ _param->NumGPUs,
						/* phstGPU = */ _param->GPU );

		// free data.
		if (Flag != NULL)
		{
			free( (void *) Flag );
			Flag = NULL;
		}
		if (SampleID != NULL)
		{
			free( (void *) SampleID );
			SampleID = NULL;
		}
		if (ChannelID != NULL)
		{
			free( (void *) ChannelID );
			ChannelID = NULL;
		}
		if (hstSpw != NULL)
			free( (void *) hstSpw );
		if (hstSample != NULL)
			free( (void *) hstSample );
		if (hstSampleWeight != NULL)
		{
			for ( int s = 0; s < _param->NumStokesImages; s++ )
				free( (void *) hstSampleWeight[ s ] );
			free( (void *) hstSampleWeight );
		}
		if (hstSampleFieldID != NULL)
			free( (void *) hstSampleFieldID );
		if (hstPhase != NULL)
			free( (void *) hstPhase );

		// if we've got data from multiple fields then we need to move it into separate mosaic components.
		separateFields(	/* pNumFields = */ hstNumFields,
					/* pStageID = */ stageID );

		// free data.
		if (FieldID != NULL)
		{
			free( (void *) FieldID );
			FieldID = NULL;
		}

		// sum the weights and gridded visibilities for each mosaic component.
		for ( int mosaicID = firstMosaicComponent; mosaicID < firstMosaicComponent + hstNumFields; mosaicID++ )
			pData[ mosaicID ].SumWeights(	/* phstTotalWeightPerCell = */ phstTotalWeightPerCell,
							/* pStageID = */ stageID );

		// save the visibility, grid position, kernel index, density map and weight.
		if (_param->CacheData == true)
			for ( int mosaicID = firstMosaicComponent; mosaicID < firstMosaicComponent + hstNumFields; mosaicID++ )
				pData[ mosaicID ].CacheData(	/* pBatchID = */ stageID,
								/* pTaylorTerm = */ -1,
								/* pWhatData = */ DATA_VISIBILITIES | DATA_GRID_POSITIONS | DATA_KERNEL_INDEXES | DATA_DENSITIES |
											DATA_WEIGHTS | DATA_MFS_WEIGHTS );

		// staging loop ends here.
		sampleStageID += numSamplesInStage;
		stageID++;

	} // LOOP: fetched all antennae data
	printf( "\rloading...100%%, processing...100%%     DONE                                                      \n\n" );

	// free memory.
	if (hstMultiplier != NULL)
	{
		for ( int s = 0; s < _param->NumStokesImages; s++ )
			free( (void *) hstMultiplier[ s ] );
		free( (void *) hstMultiplier );
	}
	if (hstDataDescPolarisationConfig != NULL)
		free( (void *) hstDataDescPolarisationConfig );
	if (hstDataDescSpw != NULL)
		free( (void *) hstDataDescSpw );
	if (hstDataDescFlag != NULL)
		free( (void *) hstDataDescFlag );
	if (hstWhichPBChannel != NULL)
	{
		for ( int spw = 0; spw < numSpws; spw++ )
			if (hstWhichPBChannel[ spw ] != NULL)
				free( (void *) hstWhichPBChannel[ spw ] );
		free( (void *) hstWhichPBChannel );
	}
	if (devMultiplier != NULL)
	{
		for ( int s = 0; s < _param->NumStokesImages; s++ )
			cudaFree( (void *) devMultiplier[ s ] );
		free( (void *) devMultiplier );
	}
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
	for ( int mosaicComponent = firstMosaicComponent; mosaicComponent < firstMosaicComponent + hstNumFields; mosaicComponent++ )
		pData[ mosaicComponent ].ShrinkCache(	/* pMaxMemory = */ hstMaxMemory );

	// see if we can turn off disk caching.
//	for ( int mosaicComponent = firstMosaicComponent; mosaicComponent < firstMosaicComponent + hstNumFields; mosaicComponent++ )
//		if (_param->CacheData == true && _param->MeasurementSets == 1 && pData[ mosaicComponent ].Stages == 1)
//		{
//			pData[ mosaicComponent ].UncacheData(	/* pBatchID = */ 0,
//								/* pTaylorTerm = */ -1,
//								/* pOffset = */ 0,
//								/* pWhatData = */ DATA_ALL );
//			_param->CacheData = false;
//		}

	// reproject the primary beam so we have it in the output reference frame for each mosaic component.
	for ( int field = 0; field < hstNumFields; field++ )
	{

		// for uniform weighting update the weight using the density map.
		if (_param->Weighting == UNIFORM && _param->UvPlaneMosaic == false)
			pData[ firstMosaicComponent + field ].PerformUniformWeighting( /* phstTotalWeightPerCell = */ phstTotalWeightPerCell );

		// for robust weighting we need to calculate the average cell weighting, and then the parameter f^2.
		if (_param->Weighting == ROBUST && _param->UvPlaneMosaic == false)
			pData[ firstMosaicComponent + field ].PerformRobustWeighting( /* phstTotalWeightPerCell = */ phstTotalWeightPerCell );

		// for natural weighting, we have already summed the weight and now need to calculate the average weight.
		if (_param->Weighting == NATURAL)
			for ( int s = 0; s < _param->NumStokesImages; s++ )
				pData[ firstMosaicComponent + field ].AverageWeight[ s ] /= (double) pData[ firstMosaicComponent + field ].GriddedVisibilities;

		// store the data phase position and the imaging phase position.
		pData[ firstMosaicComponent + field ].PhaseFromRA = hstFieldPhaseFrom[ field * 2 ];
		pData[ firstMosaicComponent + field ].PhaseFromDEC = hstFieldPhaseFrom[ (field * 2) + 1 ];
		pData[ firstMosaicComponent + field ].PhaseToRA = hstFieldPhaseTo[ field * 2 ];
		pData[ firstMosaicComponent + field ].PhaseToDEC = hstFieldPhaseTo[ (field * 2) + 1 ];

		// reproject the primary beam into the output phase position. for imaging mosaics we also need to reproject the beam to the mosaic component phase
		// position, and for A-projection we need to reproject the beam for each A-plane.
		pData[ firstMosaicComponent + field ].reprojectPrimaryBeams(	/* pBeamOutSize = */ _param->BeamSize,
										/* pBeamOutCellSize = */ _param->CellSize,
										/* pOutputRA = */ _param->OutputRA,
										/* pOutputDEC = */ _param->OutputDEC,
										/* pImagePlaneMosaic = */ _param->ImagePlaneMosaic,
										/* pMaxWavelength = */ maximumWavelength );

		// reproject the determinant of the Mueller matrix as well. this is needed for correcting the dirty images.			
		pData[ firstMosaicComponent + field ].ReprojectMuellerDeterminant(	/* pBeamOutSize = */ _param->BeamSize,
											/* pBeamOutCellSize = */ _param->CellSize,
											/* pToRA = */ _param->OutputRA,
											/* pToDEC = */ _param->OutputDEC );

{
	char filename[ 100 ];
	sprintf( filename, "mueller-determinant" );
	_casacoreInterface->WriteCasaImage(	/* pFilename = */ filename,
						/* pWidth = */ _param->BeamSize,
						/* pHeight = */ _param->BeamSize,
						/* pRA = */ _param->OutputRA,
						/* pDec = */ _param->OutputDEC,
						/* pPixelSize = */ _param->CellSize * (double) _param->ImageSize / (double) _param->BeamSize,
						/* pImage = */ MuellerDeterminant,
						/* pFrequency = */ CONST_C / AverageWavelength,
						/* pMask = */ NULL,
						/* pDirectionType = */ CasacoreInterface::J2000,
						/* pStokesImages = */ 1 );
}

		// work out how many GPU batches we'll need to process the visibilities for each stage.
		for ( int stageID = 0; stageID < pData[ firstMosaicComponent + field ].Stages; stageID++ )
		{
		
			// the number of visibilities in this stage cannot exceed the preferred visibility batch size.
			long int batchSize = pData[ firstMosaicComponent + field ].NumVisibilities[ stageID ];
			if (batchSize > _param->PREFERRED_VISIBILITY_BATCH_SIZE)
				batchSize = _param->PREFERRED_VISIBILITY_BATCH_SIZE;
				
			// create an array for the number of batches per stage.
			pData[ firstMosaicComponent + field ].Batches = (int *) malloc( pData[ firstMosaicComponent + field ].Stages * sizeof( int ) );
				
			// work out the number of batches.
			pData[ firstMosaicComponent + field ].Batches[ stageID ] =
						(int) ceil( (double) pData[ firstMosaicComponent + field ].NumVisibilities[ stageID ] /
								(double) (batchSize * _param->NumGPUs) );
			
		}

	} // LOOP: field

	// free memory.
	if (hstMinWavelength != NULL)
		free( (void *) hstMinWavelength );
	if (hstMaxWavelength != NULL)
		free( (void *) hstMaxWavelength );
	if (hstSpwChannelFlag != NULL)
	{
		for ( int spw = 0; spw < numSpws; spw++ )
			free( (void *) hstSpwChannelFlag[ spw ] );
		free( (void *) hstSpwChannelFlag );
	}
	if (hstFieldPhaseFrom != NULL)
		free( (void *) hstFieldPhaseFrom );
	if (hstFieldPhaseTo != NULL)
		free( (void *) hstFieldPhaseTo );
	if (hstNumChannels != NULL)
		free( (void *) hstNumChannels );
	if (hstUnflaggedChannels != NULL)
		free( (void *) hstUnflaggedChannels );
	if (hstPBChannelWavelength != NULL)
		free( (void *) hstPBChannelWavelength );
	if (_muellerDeterminantIn != NULL)
	{
		free( (void *) _muellerDeterminantIn );
		_muellerDeterminantIn = NULL;
	}

} // Data::ProcessMeasurementSet

//
//	Data::ReduceJonesMatrixChannels()
//
//	CJS: 21/03/2022
//
//	Add up the Jones matrices to reduce the number of channels.
//

void Data::ReduceJonesMatrixChannels
			(
			int pNumSpws,
			int * pNumChannels,
			bool ** pSpwChannelFlag,
			int ** pWhichPBChannel
			)
{

	// sum the primary beam over each pb channel, weighted by the wavelengths of the data we have.
	// we then reduce the number of Jones matrices - from the number of channels to the number of A-planes.
	for ( int cell = 0; cell < 4; cell++ )
		if (JonesMatrixIn[ cell ] != NULL)
		{
printf( "%d: cell %i not null\n", __LINE__, cell );
			// now, for each PB channel, we sum the Jones matrix over channel, to assemble a Jones matrix for each PB channel.
			int cumulativeChannel = 0;
			int * hstInputChannelsPerPBChannel = (int *) malloc( PBChannels * sizeof( int ) );
			memset( hstInputChannelsPerPBChannel, 0, PBChannels * sizeof( int ) );
			cufftComplex * tmpJonesMatrix = (cufftComplex *) malloc( PBChannels * _param->BeamInSize * _param->BeamInSize * sizeof( cufftComplex ) );
			memset( tmpJonesMatrix, 0, PBChannels * _param->BeamInSize * _param->BeamInSize * sizeof( cufftComplex ) );
			for ( int spw = 0; spw < pNumSpws; spw++ )
				for ( int channel = 0; channel < pNumChannels[ spw ]; channel++, cumulativeChannel++ )
					if (pSpwChannelFlag[ spw ][ channel ] == false)
					{
						for ( int i = 0; i < _param->BeamInSize * _param->BeamInSize; i++ )
						{
							tmpJonesMatrix[ (pWhichPBChannel[ spw ][ channel ] * _param->BeamInSize * _param->BeamInSize) + i ].x +=
											JonesMatrixIn[ cell ]
													[ (cumulativeChannel * _param->BeamInSize * _param->BeamInSize) + i ].x;
							tmpJonesMatrix[ (pWhichPBChannel[ spw ][ channel ] * _param->BeamInSize * _param->BeamInSize) + i ].y +=
											JonesMatrixIn[ cell ]
													[ (cumulativeChannel * _param->BeamInSize * _param->BeamInSize) + i ].y;
						}
						hstInputChannelsPerPBChannel[ pWhichPBChannel[ spw ][ channel ] ]++;
					}
					
			// normalise the Jones matrix to get the average beam for each A-plane.
			for ( int pbChannel = 0; pbChannel < PBChannels; pbChannel++ )
{
printf( "%d: %i channels for A-plane %i\n", __LINE__, hstInputChannelsPerPBChannel[ pbChannel ], pbChannel );
				if (hstInputChannelsPerPBChannel[ pbChannel ] != 0)
					for ( int i = 0, ptr = (pbChannel * _param->BeamInSize * _param->BeamInSize); i < _param->BeamInSize * _param->BeamInSize; i++, ptr++ )
					{
						tmpJonesMatrix[ ptr ].x /= (double) hstInputChannelsPerPBChannel[ pbChannel ];
						tmpJonesMatrix[ ptr ].y /= (double) hstInputChannelsPerPBChannel[ pbChannel ];
					}
}

				// reduce the size of the Jones matrix cell, and copy the temporary Jones matrix into the actual one.
			JonesMatrixIn[ cell ] = (cufftComplex *) realloc( JonesMatrixIn[ cell ],
												PBChannels * _param->BeamInSize * _param->BeamInSize * sizeof( cufftComplex ) );
			memcpy( JonesMatrixIn[ cell ], tmpJonesMatrix, PBChannels * _param->BeamInSize * _param->BeamInSize * sizeof( cufftComplex ) );
			
			// free data.
			if (hstInputChannelsPerPBChannel != NULL)
				free( (void *) hstInputChannelsPerPBChannel );
			if (tmpJonesMatrix != NULL)
				free( (void *) tmpJonesMatrix );
			
		} // (JonesMatrixIn[ cell ] != NULL)
else
printf( "%d: cell %i is null\n", __LINE__, cell );
			
	// LOOP: cell

} // Data::ReduceJonesMatrixChannels

//
//	ReprojectJonesMatrix()
//
//	CJS: 02/03/2022
//
//	Reprojects a single Jones matrix into the output frame.
//

void Data::ReprojectJonesMatrix( int pPBChannel, int pBeamOutSize, double pBeamOutCellSize )
{

	// create a reprojection object.
	Reprojection imagePlaneReprojection;

	// create two workspace primary beams on the device.
	float * devInBeam = NULL;
	float * devOutBeam = NULL;
	reserveGPUMemory( (void **) &devInBeam, _param->BeamInSize * _param->BeamInSize * sizeof( cufftComplex ), "reserving memory for the input primary beam on the device", __LINE__ );
	reserveGPUMemory( (void **) &devOutBeam, pBeamOutSize * pBeamOutSize * sizeof( cufftComplex ), "reserving memory for the output primary beam on the device", __LINE__ );

	// create the device memory needed by the reprojection code.
	Reprojection::rpVectI outSize = { /* x = */ pBeamOutSize, /* y = */ pBeamOutSize };
	imagePlaneReprojection.CreateDeviceMemory( outSize );
	
	// free any existing Jones matrix.
	if (JonesMatrix != NULL)
	{
		for ( int cell = 0; cell < 4; cell++ )
			free( (void *) JonesMatrix[ cell ] );
		free( (void *) JonesMatrix );
	}

	// reproject the Jones matrices for each channel. if we've generated beams then only the diagonal elements will be populated.
	JonesMatrix = (cufftComplex **) malloc( 4 * sizeof( cufftComplex * ) );
	for ( int cell = 0; cell < 4; cell++ )
	{
		JonesMatrix[ cell ] = NULL;
		if (JonesMatrixIn[ cell ] != NULL)
		{
	
			// create space for the images, and reproject the X, XY, YX, and Y components of the primary beam.
			JonesMatrix[ cell ] = (cufftComplex *) malloc( pBeamOutSize * pBeamOutSize * sizeof( cufftComplex ) );
			reprojectImage(	/* phstImageIn = */ &JonesMatrixIn[ cell ][ pPBChannel * _param->BeamInSize * _param->BeamInSize ],
						/* phstImageOut = */ JonesMatrix[ cell ],
						/* pImageInSize = */ _param->BeamInSize,
						/* pImageOutSize = */ pBeamOutSize,
						/* pInputCellSize = */ _param->BeamInCellSize,
						/* pOutputCellSize = */ pBeamOutCellSize * (double) _param->ImageSize / (double) pBeamOutSize,
						/* pInRA = */ PhaseFromRA,
						/* pInDec = */ PhaseFromDEC,
						/* pOutRA = */ PhaseToRA,
						/* pOutDec = */ PhaseToDEC,
						/* pdevInImage = */ devInBeam,
						/* pdevOutImage = */ devOutBeam,
						/* pImagePlaneReprojection = */ imagePlaneReprojection,
						/* pSquareBeam = */ false,
						/* pVerbose = */ false );
						
		}
	}

	// free memory.
	if (devInBeam != NULL)
		cudaFree( (void *) devInBeam );
	if (devOutBeam != NULL)
		cudaFree( (void *) devOutBeam );

} // Data::ReprojectJonesMatrix

//
//	ReprojectMuellerDeterminant()
//
//	CJS: 14/03/2022
//
//	Reprojects the determinant of the Mueller matrix to the required size and position.
//

//void Data::ReprojectMuellerDeterminant
//			(
//			float * phstMuellerDeterminantIn,
//			int pBeamOutSize,
//			double pBeamOutCellSize,			// the cell size of the image in arcseconds
//			double pToRA,					// mosaic/image output RA phase position
//			double pToDEC					// mosaic/image output DEC phase position
//			)
//{

	// create a reprojection object.
//	Reprojection imagePlaneReprojection;

	// create two workspace primary beams on the device.
//	float * devInBeam = NULL;
//	float * devOutBeam = NULL;
//	reserveGPUMemory( (void **) &devInBeam, BeamInSize * BeamInSize * sizeof( cufftComplex ), "reserving memory for the input primary beam on the device", __LINE__ );
//	reserveGPUMemory( (void **) &devOutBeam, pBeamOutSize * pBeamOutSize * sizeof( cufftComplex ), "reserving memory for the output primary beam on the device", __LINE__ );

	// create the device memory needed by the reprojection code.
//	Reprojection::rpVectI outSize = { /* x = */ pBeamOutSize, /* y = */ pBeamOutSize };
//	imagePlaneReprojection.CreateDeviceMemory( outSize );

	// we need to do an image-plane reprojection of the beam to the common phase position, scaling the beams in the process so that they are
	// the same size as our images.
//	MuellerDeterminant = (float *) malloc( pBeamOutSize * pBeamOutSize * sizeof( float ) );
//	memset( MuellerDeterminant, 0, pBeamOutSize * pBeamOutSize * sizeof( float ) );
//	reprojectImage(	/* phstImageIn = */ phstMuellerDeterminantIn,
//				/* phstImageOut = */ MuellerDeterminant,
//				/* pImageInSize = */ BeamInSize,
//				/* pImageOutSize = */ pBeamOutSize,
//				/* pInputCellSize = */ BeamInCellSize,
//				/* pOutputCellSize = */ pBeamOutCellSize * (double) _param->ImageSize / (double) pBeamOutSize,
//				/* pInRA = */ PhaseFromRA,
//				/* pInDec = */ PhaseFromDEC,
//				/* pOutRA = */ pToRA,
//				/* pOutDec = */ pToDEC,
//				/* pdevInImage = */ devInBeam,
//				/* pdevOutImage = */ devOutBeam,
//				/* pImagePlaneReprojection = */ imagePlaneReprojection,
//				/* pVerbose = */ true );

	// free memory.
//	if (devInBeam != NULL)
//		cudaFree( (void *) devInBeam );
//	if (devOutBeam != NULL)
//		cudaFree( (void *) devOutBeam );

//} // ReprojectMuellerDeterminant

//
//	ReprojectMuellerDeterminant()
//
//	CJS: 14/03/2022
//
//	Reprojects the determinant of the Mueller matrix to the required size and position.
//

void Data::ReprojectMuellerDeterminant
			(
			int pBeamOutSize,
			double pBeamOutCellSize,			// the cell size of the image in arcseconds
			double pToRA,					// mosaic/image output RA phase position
			double pToDEC					// mosaic/image output DEC phase position
			)
{

	if (_muellerDeterminantIn != NULL)
	{

		// create a reprojection object.
		Reprojection imagePlaneReprojection;

		// create two workspace primary beams on the device.
		float * devInBeam = NULL;
		float * devOutBeam = NULL;
		reserveGPUMemory( (void **) &devInBeam, _param->BeamInSize * _param->BeamInSize * sizeof( cufftComplex ), "reserving memory for the input primary beam on the device", __LINE__ );
		reserveGPUMemory( (void **) &devOutBeam, pBeamOutSize * pBeamOutSize * sizeof( cufftComplex ), "reserving memory for the output primary beam on the device", __LINE__ );

		// create the device memory needed by the reprojection code.
		Reprojection::rpVectI outSize = { /* x = */ pBeamOutSize, /* y = */ pBeamOutSize };
		imagePlaneReprojection.CreateDeviceMemory( outSize );

		// we need to do an image-plane reprojection of the beam to the common phase position, scaling the beams in the process so that they are
		// the same size as our images.
		MuellerDeterminant = (float *) malloc( pBeamOutSize * pBeamOutSize * sizeof( float ) );
		memset( MuellerDeterminant, 0, pBeamOutSize * pBeamOutSize * sizeof( float ) );
		reprojectImage(	/* phstImageIn = */ _muellerDeterminantIn,
					/* phstImageOut = */ MuellerDeterminant,
					/* pImageInSize = */ _param->BeamInSize,
					/* pImageOutSize = */ pBeamOutSize,
					/* pInputCellSize = */ _param->BeamInCellSize,
					/* pOutputCellSize = */ pBeamOutCellSize * (double) _param->ImageSize / (double) pBeamOutSize,
					/* pInRA = */ PhaseFromRA,
					/* pInDec = */ PhaseFromDEC,
					/* pOutRA = */ pToRA,
					/* pOutDec = */ pToDEC,
					/* pdevInImage = */ devInBeam,
					/* pdevOutImage = */ devOutBeam,
					/* pImagePlaneReprojection = */ imagePlaneReprojection,
					/* pVerbose = */ true );

		// free memory.
		if (devInBeam != NULL)
			cudaFree( (void *) devInBeam );
		if (devOutBeam != NULL)
			cudaFree( (void *) devOutBeam );

	}

} // ReprojectMuellerDeterminant

//
//	ReprojectPrimaryBeam()
//
//	CJS: 02/03/2022
//
//	Reprojects the primary beam to the required size and position.
//

float * Data::ReprojectPrimaryBeam
			(
			int pBeamOutSize,
			double pBeamOutCellSize,			// the cell size of the image in arcseconds
			double pToRA,					// mosaic/image output RA phase position
			double pToDEC,					// mosaic/image output DEC phase position
			double pToWavelength				// the wavelength we require the beam to have
			)
{

	// create a reprojection object.
	Reprojection imagePlaneReprojection;

	// create two workspace primary beams on the device.
	float * devInBeam = NULL;
	float * devOutBeam = NULL;
	reserveGPUMemory( (void **) &devInBeam, _param->BeamInSize * _param->BeamInSize * sizeof( cufftComplex ), "reserving memory for the input primary beam on the device", __LINE__ );
	reserveGPUMemory( (void **) &devOutBeam, pBeamOutSize * pBeamOutSize * sizeof( cufftComplex ), "reserving memory for the output primary beam on the device", __LINE__ );

	// create the device memory needed by the reprojection code.
	Reprojection::rpVectI outSize = { /* x = */ pBeamOutSize, /* y = */ pBeamOutSize };
	imagePlaneReprojection.CreateDeviceMemory( outSize );

	// we need to do an image-plane reprojection of the beam to the common phase position, scaling the beams in the process so that they are
	// the same size as our images.
	float * reprojectedBeam = (float *) malloc( pBeamOutSize * pBeamOutSize * sizeof( float ) );
	memset( reprojectedBeam, 0, pBeamOutSize * pBeamOutSize * sizeof( float ) );
	reprojectImage(	/* phstImageIn = */ AveragePrimaryBeamIn,
				/* phstImageOut = */ reprojectedBeam,
				/* pImageInSize = */ _param->BeamInSize,
				/* pImageOutSize = */ pBeamOutSize,
				/* pInputCellSize = */ _param->BeamInCellSize * pToWavelength / AverageWavelength,
				/* pOutputCellSize = */ pBeamOutCellSize * (double) _param->ImageSize / (double) pBeamOutSize,
				/* pInRA = */ PhaseFromRA,
				/* pInDec = */ PhaseFromDEC,
				/* pOutRA = */ pToRA,
				/* pOutDec = */ pToDEC,
				/* pdevInImage = */ devInBeam,
				/* pdevOutImage = */ devOutBeam,
				/* pImagePlaneReprojection = */ imagePlaneReprojection,
				/* pVerbose = */ true );

	// free memory.
	if (devInBeam != NULL)
		cudaFree( (void *) devInBeam );
	if (devOutBeam != NULL)
		cudaFree( (void *) devOutBeam );
		
	// return something.
	return reprojectedBeam;

} // ReprojectPrimaryBeam

//
//	ShrinkCache()
//
//	CJS: 20/08/2021
//
//	Attempt to shrink the cache by merging files together.
//

void Data::ShrinkCache( long int pMaxMemory )
{

//	for ( int stage = 0; stage < Stages; stage++ )
//	{

//		if (_hstCacheData == true)
//			UncacheData(	/* pStageID = */ stage,
//					/* pTaylorTerm = */ -1,
//					/* pOffset = */ 0,
//					/* pWhatData = */ DATA_GRID_POSITIONS );

//		for ( int i = 0; i < NumVisibilities[ stage ]; i++ ) // cjs-mod
//			if (_hstGridPosition[ mosaicComponent ][ i ].w < 0 || _hstGridPosition[ mosaicComponent ][ i ].w >= 128)
//				printf( "%d (DATA): invalid kernel set: mosaic component = %i, stage = %i, vis = %i, grid position = <%i, %i, %i>\n", __LINE__, mosaicComponent, stage, i, _hstGridPosition[ mosaicComponent ][ i ].u, _hstGridPosition[ mosaicComponent ][ i ].v, _hstGridPosition[ mosaicComponent ][ i ].w );

//		if (_hstCacheData == true)
//			FreeData( /* pWhatData = */ DATA_GRID_POSITIONS );

//	}

	// see if we can merge files together.
	if (Stages > 1)
	{

		int recordSize = (sizeof( cufftComplex ) * _taylorTerms) + sizeof( VectorI ) + sizeof( int ) + sizeof( int ) + sizeof( float );
		int whatData = DATA_VISIBILITIES | DATA_GRID_POSITIONS | DATA_KERNEL_INDEXES | DATA_DENSITIES | DATA_WEIGHTS;
		if (_param->Deconvolver == MFS)
		{
			whatData = whatData | DATA_MFS_WEIGHTS;
			recordSize += sizeof( float ) * (_taylorTerms - 1) * 2;
		}

		// calculate maximum number of visibilities to hold in ram.
		long int maxVis = pMaxMemory / (long int) recordSize;

		// see if each stage can be merged with the stage that came before it.
		int numberMerged = 1;
		for ( int stage = Stages - 1; stage >= 1; stage-- )
			if (NumVisibilities[ stage ] + NumVisibilities[ stage - 1 ] <= maxVis)
			{

				// this stage can be merged with the previous stage.
				mergeData(	/* pStageID_one = */ stage - 1,
						/* pStageID_two = */ stage,
						/* pLoadAllData = */ (numberMerged == 1),
						/* pWhatData = */ whatData );
				numberMerged++;

			}
			else if (numberMerged > 1)
			{

				printf( "merging %i data caches\n", numberMerged );

				// sort the data.
				quickSortData(	/* pLeft = */ 0,
						/* pRight = */ NumVisibilities[ stage ] - 1 );

				// compact the data.
				compactData(	/* pTotalVisibilities = */ &NumVisibilities[ stage ],
						/* pFirstVisibility = */ 0,
						/* pNumVisibilitiesToProcess = */ NumVisibilities[ stage ] );

//for ( int i = 0; i < NumVisibilities[ stage ]; i++ ) // cjs-mod
//	if (_hstGridPosition[ mosaicComponent ][ i ].w < 0 || _hstGridPosition[ mosaicComponent ][ i ].w >= 128)
//		printf( "%d (DATA): invalid kernel set: vis = %i, grid position = <%i, %i, %i>\n", __LINE__, i, _hstGridPosition[ mosaicComponent ][ i ].u,
//					_hstGridPosition[ mosaicComponent ][ i ].v, _hstGridPosition[ mosaicComponent ][ i ].w );

				// we have some data open which we need to save.
				CacheData(	/* pStageID = */ stage,
						/* pTaylorTerm = */ -1,
						/* pWhatData = */ whatData );
				numberMerged = 1;

			}
		if (numberMerged > 1)
		{

			printf( "merging %i data caches\n", numberMerged );

			// sort the data.
			quickSortData(	/* pLeft = */ 0,
					/* pRight = */ NumVisibilities[ /* STAGE = */ 0 ] - 1 );

			// compact the data.
			compactData(	/* pTotalVisibilities = */ &NumVisibilities[ /* STAGE = */ 0 ],
					/* pFirstVisibility = */ 0,
					/* pNumVisibilitiesToProcess = */ NumVisibilities[ /* STAGE = */ 0 ] );

//for ( int i = 0; i < NumVisibilities[ 0 ]; i++ ) // cjs-mod
//	if (_hstGridPosition[ mosaicComponent ][ i ].w < 0 || _hstGridPosition[ mosaicComponent ][ i ].w >= 128)
//		printf( "%d (DATA): invalid kernel set: vis = %i, grid position = <%i, %i, %i>\n", __LINE__, i, _hstGridPosition[ mosaicComponent ][ i ].u,
//					_hstGridPosition[ mosaicComponent ][ i ].v, _hstGridPosition[ mosaicComponent ][ i ].w );

			// we have some data open which we need to save.
			CacheData(	/* pStageID = */ 0,
					/* pTaylorTerm = */ -1,
					/* pWhatData = */ whatData );

		} // (numberMerged > 1)

	} // (Stages > 1)

//	for ( int stage = 0; stage < Stages; stage++ )
//	{

//		if (_hstCacheData == true)
//			UncacheData(	/* pStageID = */ stage,
//					/* pTaylorTerm = */ -1,
//					/* pOffset = */ 0,
//					/* pWhatData = */ DATA_GRID_POSITIONS );

//		for ( int i = 0; i < NumVisibilities[ stage ]; i++ ) // cjs-mod
//			if (_hstGridPosition[ mosaicComponent ][ i ].w < 0 || _hstGridPosition[ mosaicComponent ][ i ].w >= 128)
//				printf( "%d (DATA): invalid kernel set: mosaic component = %i, stage = %i, vis = %i, grid position = <%i, %i, %i>\n", __LINE__, mosaicComponent, stage, i, _hstGridPosition[ mosaicComponent ][ i ].u, _hstGridPosition[ mosaicComponent ][ i ].v, _hstGridPosition[ mosaicComponent ][ i ].w );

//		if (_hstCacheData == true)
//			FreeData( /* pWhatData = */ DATA_GRID_POSITIONS );

//	}

} // ShrinkCache

//
//	SumWeights()
//
//	CJS: 25/08/2021
//
//	Sum the gridded visibilities and the weights for this mosaic component.
//

void Data::SumWeights( double ** phstTotalWeightPerCell, int pStageID )
{

	// count the number of gridded visibilities (using the density map). We use this figure for normalising our
	// images. for beam mosaics we do this later on a per-field basis.
	for ( long int visibilityIndex = 0; visibilityIndex < NumVisibilities[ pStageID ]; visibilityIndex++ )
		if (	GridPosition[ visibilityIndex ].u >= 0 && GridPosition[ visibilityIndex ].u < _param->ImageSize &&
			GridPosition[ visibilityIndex ].v >= 0 && GridPosition[ visibilityIndex ].v < _param->ImageSize)
			GriddedVisibilities += DensityMap[ visibilityIndex ];

	// if we're using uniform or robust weighting then add up the total weight in each grid cell (if we're using weighting).
	if (_param->Weighting == UNIFORM || _param->Weighting == ROBUST)
		for ( long int i = 0; i < NumVisibilities[ pStageID ]; i++ )
			if (	GridPosition[ i ].u >= 0 && GridPosition[ i ].u < _param->ImageSize &&
				GridPosition[ i ].v >= 0 && GridPosition[ i ].v < _param->ImageSize)
				for ( int s = 0; s < _stokesImages; s++ )
					phstTotalWeightPerCell[ s ][ (GridPosition[ i ].v * _param->ImageSize) + GridPosition[ i ].u ] +=
														((double) Weight[ s ][ i ] * (double) DensityMap[ i ]);

	// if we're using natural weighting then add up the total weight in the whole grid.
	if (_param->Weighting == NATURAL)
		for ( long int i = 0; i < NumVisibilities[ pStageID ]; i++ )
			if (	GridPosition[ i ].u >= 0 && GridPosition[ i ].u < _param->ImageSize &&
				GridPosition[ i ].v >= 0 && GridPosition[ i ].v < _param->ImageSize)
				for ( int s = 0; s < _stokesImages; s++ )
					AverageWeight[ s ] += ((double) Weight[ s ][ i ] * (double) DensityMap[ i ]);

} // SumWeights

//
//	UncacheData()
//
//	CJS: 25/03/2019
//
//	Retrieve a whole set of visibilities, grid positions, kernel indexes, etc from disk. if an offset is supplied then the data should be loaded
//	into the arrays at this position, so we need to expand rather than initialise the arrays.
//

void Data::UncacheData( int pStageID, int pTaylorTerm, long int pOffset, int pWhatData, int pStokes )
{
//printf( "%d (DATA): UNCACHING: pFilenamePrefix %s, pStageID %i, pOffset %li\n", __LINE__, _param->OutputPrefix, pStageID, pOffset );
	// build filename.
	char filename[ 255 ];

	// build the full filename.
	if (_param->CacheLocation[0] != '\0')
		sprintf( filename, "%s%s-%i-%i-cache.dat", _param->CacheLocation, _param->OutputPrefix, _mosaicID, pStageID );
	else
		sprintf( filename, "%s-%i-%i-cache.dat", _param->OutputPrefix, _mosaicID, pStageID );

	// open the file for reading.
	FILE * fr = fopen( filename, "rb" );

	// create the required memory, and read the data.
	for ( int stokes = 0; stokes < _stokesImages; stokes++ )
		for ( int t = 0; t < _taylorTerms; t++ )
			if ((pWhatData & DATA_VISIBILITIES) == DATA_VISIBILITIES && (stokes == pStokes || pStokes == -1) && (pTaylorTerm == t || pTaylorTerm == -1))
			{
				if (pOffset == 0)
					Visibility[ stokes ][ t ] = (cufftComplex *) malloc( NumVisibilities[ pStageID ] * sizeof( cufftComplex ) );
				else
					Visibility[ stokes ][ t ] = (cufftComplex *) realloc( Visibility[ stokes ][ t ],
													(pOffset + NumVisibilities[ pStageID ]) * sizeof( cufftComplex ) );
				fread( (void *) &Visibility[ stokes ][ t ][ pOffset ], sizeof( cufftComplex ), NumVisibilities[ pStageID ], fr );
			}
			else
				fseek( fr, NumVisibilities[ pStageID ] * sizeof( cufftComplex ), SEEK_CUR );

	if ((pWhatData & DATA_GRID_POSITIONS) == DATA_GRID_POSITIONS)
	{
		if (pOffset == 0)
			GridPosition = (VectorI *) malloc( NumVisibilities[ pStageID ] * sizeof( VectorI ) );
		else
			GridPosition = (VectorI *) realloc( GridPosition, (pOffset + NumVisibilities[ pStageID ]) * sizeof( VectorI ) );
		fread( (void *) &GridPosition[ pOffset ], sizeof( VectorI ), NumVisibilities[ pStageID ], fr );
	}
	else
		fseek( fr, NumVisibilities[ pStageID ] * sizeof( VectorI ), SEEK_CUR );

	if ((pWhatData & DATA_KERNEL_INDEXES) == DATA_KERNEL_INDEXES)
	{
		if (pOffset == 0)
			KernelIndex = (int *) malloc( NumVisibilities[ pStageID ] * sizeof( int ) );
		else
			KernelIndex = (int *) realloc( KernelIndex, (pOffset + NumVisibilities[ pStageID ]) * sizeof( int ) );
		fread( (void *) &KernelIndex[ pOffset ], sizeof( int ), NumVisibilities[ pStageID ], fr );
	}
	else
		fseek( fr, NumVisibilities[ pStageID ] * sizeof( int ), SEEK_CUR );

	if ((pWhatData & DATA_DENSITIES) == DATA_DENSITIES)
	{
		if (pOffset == 0)
			DensityMap = (int *) malloc( NumVisibilities[ pStageID ] * sizeof( int ) );
		else
			DensityMap = (int *) realloc( DensityMap, (pOffset + NumVisibilities[ pStageID ]) * sizeof( int ) );
		fread( (void *) &DensityMap[ pOffset ], sizeof( int ), NumVisibilities[ pStageID ], fr );
	}
	else
		fseek( fr, NumVisibilities[ pStageID ] * sizeof( int ), SEEK_CUR );

	for ( int stokes = 0; stokes < _stokesImages; stokes++ )
		if ((pWhatData & DATA_WEIGHTS) == DATA_WEIGHTS && (stokes == pStokes || pStokes == -1))
		{
			if (pOffset == 0)
				Weight[ stokes ] = (float *) malloc( NumVisibilities[ pStageID ] * sizeof( float ) );
			else
				Weight[ stokes ] = (float *) realloc( Weight[ stokes ], (pOffset + NumVisibilities[ pStageID ]) * sizeof( float ) );
			fread( (void *) &Weight[ stokes ][ pOffset ], sizeof( float ), NumVisibilities[ pStageID ], fr );
		}
		else
			fseek( fr, NumVisibilities[ pStageID ] * sizeof( float ), SEEK_CUR );

	if (_param->Deconvolver == MFS)
		for ( int stokes = 0; stokes < _stokesImages; stokes++ )
			for ( int t = 0; t < (_taylorTerms - 1) * 2; t++ )
				if ((pWhatData & DATA_MFS_WEIGHTS) == DATA_MFS_WEIGHTS && (stokes == pStokes || pStokes == -1) &&
					(pTaylorTerm == t || pTaylorTerm == -1))
				{
					if (pOffset == 0)
						MfsWeight[ stokes ][ t ] = (float *) malloc( NumVisibilities[ pStageID ] * sizeof( float ) );
					else
						MfsWeight[ stokes ][ t ] = (float *) realloc( MfsWeight[ stokes ][ t ],
														(pOffset + NumVisibilities[ pStageID ]) * sizeof( float ) );
					fread( (void *) &MfsWeight[ t ][ pOffset ], sizeof( float ), NumVisibilities[ pStageID ], fr );
				}
				else
					fseek( fr, NumVisibilities[ pStageID ] * sizeof( float ), SEEK_CUR );

	for ( int stokes = 0; stokes < _stokesImages; stokes++ )
		for ( int t = 0; t < _taylorTerms; t++ )
			if ((pWhatData & DATA_RESIDUAL_VISIBILITIES) == DATA_RESIDUAL_VISIBILITIES && (stokes == pStokes || pStokes == -1) &&
					(pTaylorTerm == t || pTaylorTerm == -1))
			{
				if (pOffset == 0)
					ResidualVisibility[ stokes ][ t ] = (cufftComplex *) malloc( NumVisibilities[ pStageID ] * sizeof( cufftComplex ) );
				else
					ResidualVisibility[ stokes ][ t ] = (cufftComplex *) realloc( ResidualVisibility[ stokes ][ t ],
														(pOffset + NumVisibilities[ pStageID ]) * sizeof( cufftComplex ) );
				fread( (void *) &ResidualVisibility[ stokes ][ t ][ pOffset ], sizeof( cufftComplex ), NumVisibilities[ pStageID ], fr );
			}


	// close the file.
	fclose( fr );

} // UncacheData

//
//	P R I V A T E   C L A S S   M E M B E R S
//

//
//	calculateInverseImageMatrix()
//
//	CJS: 01/11/2021
//
//	Calculate the inverse of an image matrix.
//

cufftDoubleComplex ** Data::calculateInverseImageMatrix
			(
			cufftDoubleComplex ** pdevMatrix,		// an array of image pointers that describe an NxN matrix
			int pMatrixSize,				// the matrix size, i.e. NxN
			int pImageSize,					// the size of the image held in each matrix cell
			bool pDivideByDeterminant
			)
{
	
	cufftDoubleComplex ** devReturnMatrix = (cufftDoubleComplex **) malloc( pMatrixSize * pMatrixSize * sizeof( cufftDoubleComplex * ) );
	
	// construct a reduced matrix for each matrix cell.
	cufftDoubleComplex ** devReducedMatrix = (cufftDoubleComplex **) malloc( (pMatrixSize - 1) * (pMatrixSize - 1) * sizeof( cufftDoubleComplex * ) );
		
	// and set the thread block size for processing images on the device.
	setThreadBlockSize2D( pImageSize, pImageSize, _gridSize2D, _blockSize2D );
	
	// loop over the matrix. to calculate the inverse we construct a reduced matrix for each cell, and calculate the determinant.
	for ( int row = 0; row < pMatrixSize; row++ )
		for ( int col = 0; col < pMatrixSize; col++ )
		{
		
			// construct the reduced matrix from all the matrix cells that are not in this row or column.
			for ( int destRow = 0; destRow < pMatrixSize - 1; destRow++ )
				for ( int destCol = 0; destCol < pMatrixSize - 1; destCol++ )
				{
					int sourceRow = (destRow < row ? destRow : destRow + 1 );
					int sourceCol = (destCol < col ? destCol : destCol + 1 );
					devReducedMatrix[ (destRow * (pMatrixSize - 1)) + destCol ] = pdevMatrix[ (sourceRow * pMatrixSize) + sourceCol ];
				}
				
			// calculate the determinant of the reduced matrix.
			devReturnMatrix[ (row * pMatrixSize) + col ] = determinantImageMatrix(	/* pMatrix = */ devReducedMatrix,
													/* pMatrixSize = */ pMatrixSize - 1,
													/* pImageSize = */ pImageSize );
			
			// for odd cells, multiply the image by -1.
			if (((row + col) % 2 == 1) && devReturnMatrix[ (row * pMatrixSize) + col ] != NULL)
				devMultiplyImages<<< _gridSize2D, _blockSize2D >>>(	/* pOne = */ devReturnMatrix[ (row * pMatrixSize) + col ],
											/* pScalar = */ -1.0,
											/* pMask = */ NULL,
											/* pSizeOne = */ pImageSize );
			
		
		} // LOOP: col
	// LOOP: row
			
	// free memory.
	if (devReducedMatrix != NULL)
		free( (void *) devReducedMatrix );
	
	// calculate the adjoint.
	for ( int row = 0; row < pMatrixSize - 1; row++ )
		for ( int col = row + 1; col < pMatrixSize; col++ )
		{
			cufftDoubleComplex * devTmp = devReturnMatrix[ (row * pMatrixSize) + col ];
			devReturnMatrix[ (row * pMatrixSize) + col ] = devReturnMatrix[ (col * pMatrixSize) + row ];
			devReturnMatrix[ (col * pMatrixSize) + row ] = devTmp;
		}
		
	if (pDivideByDeterminant == true)
	{
		
		// calculate the determinant of the whole matrix.
		cufftDoubleComplex * devDeterminant = determinantImageMatrix(	/* pMatrix = */ pdevMatrix,
											/* pMatrixSize = */ pMatrixSize,
											/* pImageSize = */ pImageSize );
										
		// divide each cell by the determinant.
		if (devDeterminant != NULL)
			for ( int row = 0; row < pMatrixSize; row++ )
				for ( int cell = 0; cell < pMatrixSize; cell++ )
					if (devReturnMatrix[ (row * pMatrixSize) + cell ] != NULL)
						devDivideImages<<< _gridSize2D, _blockSize2D >>>(	/* pOne = */ devReturnMatrix[ (row * pMatrixSize) + cell ],
													/* pTwo = */ devDeterminant,
													/* pMask = */ NULL,
													/* pSizeOne = */ pImageSize,
													/* pSizeTwo = */ pImageSize );
											
		// free memory.
		if (devDeterminant != NULL)
			cudaFree( (void *) devDeterminant );
		
	}
	
	// return something.
	return devReturnMatrix;

} // calculateInverseImageMatrix

//
//	calculateGridPositions()
//
//	CJS: 25/08/2021
//
//	Calculate the grid positions for a set of visibilities and UVW coordinates.
//

void Data::calculateGridPositions
			(
			int pStageID,					//
			long int pPreferredVisibilityBatchSize,	//
			int pNumSpws,					//
			int pNumSamplesInStage,			//
			int pSampleStageID,				//
			int pTotalSamples,				//
			int pOversample,				//
			double pUvCellSize,				//
			double ** phstWavelength,			//
			int * phstNumChannels,				//
			double * phstPhase,				//
			int ** phstWhichPBChannel,			//
			int * phstSpw,					//
			float ** phstSampleWeight,			//
			int * phstSampleFieldID,			//
			VectorD * phstSample,				//
			int pNumFields,				//
			int pNumGPUs,					//
			int * phstGPU					//
			)
{

	// if the number of visibilities is greater than the maximum number then we are going to set a smaller batch size, and load these visibilities in batches.
	int hstVisibilityBatchSize = 0;
	{
		long int nextBatchSize = NumVisibilities[ pStageID ];
		if (nextBatchSize > pPreferredVisibilityBatchSize)
			nextBatchSize = pPreferredVisibilityBatchSize;
		hstVisibilityBatchSize = (int) nextBatchSize;
	}

	// create space on the devices.
	VectorD ** devSample = (VectorD **) malloc( pNumGPUs * sizeof( VectorD * ) );
	double *** hstdevWavelength = (double ***) malloc( pNumGPUs * sizeof( double ** ) );
	double *** devWavelength = (double ***) malloc( pNumGPUs * sizeof( double ** ) );
	int ** devSpw = (int **) malloc( pNumGPUs * sizeof( int * ) );
	double ** devWPlaneMax = (double **) malloc( pNumGPUs * sizeof( double * ) );
	double ** devPhase = (double **) malloc( pNumGPUs * sizeof( double * ) );
	cufftComplex *** devVisibility = (cufftComplex ***) malloc( _stokesImages * sizeof( cufftComplex ** ) );
	for ( int s = 0; s < _stokesImages; s++ )
		devVisibility[ s ] = (cufftComplex **) malloc( pNumGPUs * sizeof( cufftComplex * ) );
	int ** devPBChannel = (int **) malloc( pNumGPUs * sizeof( int * ) );
	int ** devChannelID = (int **) malloc( pNumGPUs * sizeof( int * ) );
	int ** devSampleID = (int **) malloc( pNumGPUs * sizeof( int * ) );
	int * hstPBChannel = (int *) malloc( hstVisibilityBatchSize * pNumGPUs * sizeof( int ) );

	for ( int gpu = 0; gpu < pNumGPUs; gpu++ )
	{

		cudaSetDevice( phstGPU[ gpu ] );

		reserveGPUMemory( (void **) &devSample[ gpu ], pNumSamplesInStage * sizeof( VectorD ), "creating device memory for the samples", __LINE__ );
		hstdevWavelength[ gpu ] = (double **) malloc( pNumSpws * sizeof( double * ) );
		for ( int spw = 0; spw < pNumSpws; spw++ )
			reserveGPUMemory( (void **) &hstdevWavelength[ gpu ][ spw ], phstNumChannels[ spw ] * sizeof( double ), "creating device memory for spw", __LINE__ );

		reserveGPUMemory( (void **) &devWavelength[ gpu ], pNumSpws * sizeof( double * ), "creating device memory for the wavelengths", __LINE__ );
		reserveGPUMemory( (void **) &devSpw[ gpu ], pNumSamplesInStage * sizeof( int ), "creating device memory for the spw ids", __LINE__ );
		reserveGPUMemory( (void **) &devWPlaneMax[ gpu ], WPlanes * sizeof( double ), "creating device memory for the w-plane limits", __LINE__ );
		reserveGPUMemory( (void **) &devPhase[ gpu ], pNumSamplesInStage * sizeof( double ), "creating device memory for the phases", __LINE__ );
		for ( int s = 0; s < _stokesImages; s++ )
			reserveGPUMemory( (void **) &devVisibility[ s ][ gpu ], hstVisibilityBatchSize * sizeof( cufftComplex ), "creating device memory for the visibilities",
						__LINE__ );
		reserveGPUMemory( (void **) &devPBChannel[ gpu ], hstVisibilityBatchSize * sizeof( int ), "reserving device memory for the pb-correction channels", __LINE__ );
		reserveGPUMemory( (void **) &devSampleID[ gpu ], hstVisibilityBatchSize * sizeof( int ), "creating device memory for the sample ID", __LINE__ );
		reserveGPUMemory( (void **) &devChannelID[ gpu ], hstVisibilityBatchSize * sizeof( int ), "creating device memory for the channel ID", __LINE__ );

		// upload the samples to the device.
		moveHostToDevice( (void *) devSample[ gpu ], (void *) phstSample, pNumSamplesInStage * sizeof( VectorD ), "copying samples to the device", __LINE__ );

		// upload the wavelengths to the device.
		for ( int spw = 0; spw < pNumSpws; spw++ )
			moveHostToDevice( (void *) hstdevWavelength[ gpu ][ spw ], (void *) phstWavelength[ spw ], phstNumChannels[ spw ] * sizeof( double ),
						"copying wavelengths to device", __LINE__ );

		// upload the spw ids, wavelength pointers, w-plane limits and phases to the device.
		moveHostToDevice( (void *) devSpw[ gpu ], (void *) phstSpw, pNumSamplesInStage * sizeof( int ), "copying spw ids to the device", __LINE__ );
		moveHostToDevice( (void *) devWavelength[ gpu ], (void *) hstdevWavelength[ gpu ], pNumSpws * sizeof( double * ), "copying wavelength pointers to the device",
						__LINE__ );
		moveHostToDevice( (void *) devWPlaneMax[ gpu ], (void *) WPlaneMax, WPlanes * sizeof( double ),
						"copying w-plane limits to the device", __LINE__ );
		moveHostToDevice( (void *) devPhase[ gpu ], (void *) phstPhase, pNumSamplesInStage * sizeof( double ), "copying phases to device", __LINE__ );

	}

	// save the original number of visibilities.
	long int originalVisibilities = NumVisibilities[ pStageID ];

	// create arrays for the grid positions, kernel indexes, density maps, weights, field IDs, visibilities, and spectral beam weight.
	GridPosition = (VectorI *) malloc( NumVisibilities[ pStageID ] * sizeof( VectorI ) );
	KernelIndex = (int *) malloc( NumVisibilities[ pStageID ] * sizeof( int ) );
	DensityMap = (int *) malloc( NumVisibilities[ pStageID ] * sizeof( int ) );
	if (pNumFields > 1)
		FieldID = (int *) malloc( NumVisibilities[ pStageID ] * sizeof( int ) );
	for ( int s = 0; s < _stokesImages; s++ )
	{
		for ( int t = 1; t < _taylorTerms; t++ )
			Visibility[ s ][ t ] = (cufftComplex *) malloc( NumVisibilities[ pStageID ] * sizeof( cufftComplex ) );
		Weight[ s ] = (float *) malloc( NumVisibilities[ pStageID ] * sizeof( float ) );
		if (_param->Deconvolver == MFS)
			for ( int t = 0; t < (_taylorTerms - 1) * 2; t++ )
				MfsWeight[ s ][ t ] = (float *) malloc( NumVisibilities[ pStageID ] * sizeof( float ) );
	}

	// store the number of visibilities we will process on each GPU per batch.
	int * hstVisibilitiesPerGPU = (int *) malloc( pNumGPUs * sizeof( int ) );

	// keep looping until we have calculated the grid positions for all the UVWs.
	long int hstCurrentVisibility = 0, uncompactedVisibilityID = 0;
	while (hstCurrentVisibility < NumVisibilities[ pStageID ])
	{
		
		printf( "\rloading...%3d%%, processing...%3d%%     currently: processing (calculating grid positions)",
			(int) ((double) (pSampleStageID + pNumSamplesInStage) * 100.0 / (double) pTotalSamples),
			(int) ((double) (pSampleStageID + SampleID[ hstCurrentVisibility ]) * 100.0 / (double) pTotalSamples) );
		fflush( stdout );

		// calculate how many visibilities we are processing on each GPU.
		long int visibilitiesAllGPUs = 0;
		for ( int gpu = 0; gpu < pNumGPUs; gpu++ )
		{
			long int remainingVisibilities = NumVisibilities[ pStageID ] - hstCurrentVisibility - (gpu * hstVisibilityBatchSize);
			if (remainingVisibilities >= hstVisibilityBatchSize)
				hstVisibilitiesPerGPU[ gpu ] = hstVisibilityBatchSize;
			if (remainingVisibilities >= 0 && remainingVisibilities < hstVisibilityBatchSize)
				hstVisibilitiesPerGPU[ gpu ] = remainingVisibilities;
			if (remainingVisibilities < 0)
				hstVisibilitiesPerGPU[ gpu ] = 0;
			visibilitiesAllGPUs += hstVisibilitiesPerGPU[ gpu ];
		}

		// update the weights, and field ids of each visibility, and initialise density map to 1.
		for ( long int i = 0; i < visibilitiesAllGPUs; i++ )
		{

			int sampleID = SampleID[ hstCurrentVisibility + i ];
			int channelID = ChannelID[ hstCurrentVisibility + i ];
			int spw = phstSpw[ sampleID ];

			for ( int s = 0; s < _stokesImages; s++ )
				Weight[ s ][ hstCurrentVisibility + i ] = phstSampleWeight[ s ][ sampleID ];
			if (pNumFields > 1)
				FieldID[ hstCurrentVisibility + i ] = phstSampleFieldID[ sampleID ];
			DensityMap[ hstCurrentVisibility + i ] = 1;
			
			// set the pb-correction channels.	
			hstPBChannel[ i ] = phstWhichPBChannel[ spw ][ channelID ];
				
		}

		for ( int gpu = 0; gpu < pNumGPUs; gpu++ )
			if (hstVisibilitiesPerGPU[ gpu ] > 0)
			{

				cudaSetDevice( phstGPU[ gpu ] );

				// upload the visibilities, sample IDs and channel IDs to the device.
				for ( int s = 0; s < _stokesImages; s++ )
					moveHostToDevice( (void *) devVisibility[ s ][ gpu ],
								(void *) &Visibility[ s ][ /* TAYLOR_TERM = */ 0 ][ hstCurrentVisibility + (gpu * hstVisibilityBatchSize) ],
								hstVisibilitiesPerGPU[ gpu ] * sizeof( cufftComplex ), "copying visibilities to device", __LINE__ );
				moveHostToDevice( (void *) devSampleID[ gpu ], (void *) &SampleID[ hstCurrentVisibility + (gpu * hstVisibilityBatchSize) ],
							hstVisibilitiesPerGPU[ gpu ] * sizeof( int ), "copying sample ID to device", __LINE__ );
				moveHostToDevice( (void *) devChannelID[ gpu ], (void *) &ChannelID[ hstCurrentVisibility + (gpu * hstVisibilityBatchSize) ],
							hstVisibilitiesPerGPU[ gpu ] * sizeof( int ), "copying channel ID to device", __LINE__ );

				// set a suitable thread and block size.
				int threads = hstVisibilitiesPerGPU[ gpu ];
				int blocks = 1;
				setThreadBlockSize1D( &threads, &blocks );

				// mirror the visibilities by taking the conjugate values for all the second half of the data set.
				for ( int s = 0; s < _stokesImages; s++ )
					devTakeConjugateVisibility<<< blocks, threads >>>(	/* IN/OUT: pVisibility = */ devVisibility[ s ][ gpu ],
												/* IN: pCurrentVisibility = */ uncompactedVisibilityID +
																	(gpu * hstVisibilityBatchSize),
												/* IN: pNumVisibilities = */ originalVisibilities,
												/* IN: pVisibilityBatchSize = */ hstVisibilitiesPerGPU[ gpu ] );
				cudaError_t err = cudaGetLastError();
				if (err != cudaSuccess)
					printf( "error taking conjugate values (%s)\n", cudaGetErrorString( err ) );

				// do phase correction on the visibilities.
				for ( int s = 0; s < _stokesImages; s++ )
					devPhaseCorrection<<< blocks, threads >>>(	/* IN/OUT: pVisibility = */ devVisibility[ s ][ gpu ],
											/* IN: pPhase = */ devPhase[ gpu ],
											/* IN: pWavelength = */ devWavelength[ gpu ],
											/* IN: pSpw = */ devSpw[ gpu ],
											/* IN: pSampleID = */ devSampleID[ gpu ],
											/* IN: pChannelID = */ devChannelID[ gpu ],
											/* IN: pVisibilityBatchSize = */ hstVisibilitiesPerGPU[ gpu ],
											/* IN: pNumSpws = */ pNumSpws );
				err = cudaGetLastError();
				if (err != cudaSuccess)
					printf( "error doing phase correction (%s)\n", cudaGetErrorString( err ) );

			}

		// download visibilities from the device.
		for ( int gpu = 0; gpu < pNumGPUs; gpu++ )
			if (hstVisibilitiesPerGPU[ gpu ] > 0)
			{
				cudaSetDevice( phstGPU[ gpu ] );
				for ( int s = 0; s < _stokesImages; s++ )
					moveDeviceToHost(	(void *) &Visibility[ s ][ /* TAYLOR_TERM = */ 0 ][ hstCurrentVisibility + (gpu * hstVisibilityBatchSize) ],
								(void *) devVisibility[ s ][ gpu ],
								hstVisibilitiesPerGPU[ gpu ] * sizeof( cufftComplex ), "copying visibilities from the device", __LINE__ );
			}

		// compute the MFS weights ((lambda_0 / lambda) - 1)^taylor_term, and the visibilities for t > 0.
		if (_param->Deconvolver == MFS)
		{

			// reserve GPU memory
			float ** devMfsWeight = (float **) malloc( pNumGPUs * sizeof( float * ) );
			cufftComplex ** devMfsVisibility = (cufftComplex **) malloc( pNumGPUs * sizeof( cufftComplex * ) );
			for ( int gpu = 0; gpu < pNumGPUs; gpu++ )
				if (hstVisibilitiesPerGPU[ gpu ] > 0)
				{
					cudaSetDevice( phstGPU[ gpu ] );
					reserveGPUMemory( (void **) &devMfsWeight[ gpu ], hstVisibilitiesPerGPU[ gpu ] * sizeof( float ),
								"reserving device memory for the Mfs weights", __LINE__ );
					reserveGPUMemory( (void **) &devMfsVisibility[ gpu ], hstVisibilitiesPerGPU[ gpu ] * sizeof( cufftComplex ),
								"reserving GPU memory for Mfs visibilities", __LINE__ );
				}
				else
				{
					devMfsWeight[ gpu ] = NULL;
					devMfsVisibility[ gpu ] = NULL;
				}

			// compute the mfs weights.
			for ( int t = 1; t <= (_taylorTerms - 1) * 2; t++ )
			{

				for ( int gpu = 0; gpu < pNumGPUs; gpu++ )
					if (hstVisibilitiesPerGPU[ gpu ] > 0)
					{

						cudaSetDevice( phstGPU[ gpu ] );

						// set a suitable thread and block size.
						int threads = hstVisibilitiesPerGPU[ gpu ];
						int blocks = 1;
						setThreadBlockSize1D( &threads, &blocks );

						// compute mfs weights on the GPU.
						devComputeMfsWeights<<< blocks, threads >>>(	/* OUT: pMfsWeight = */ devMfsWeight[ gpu ],
												/* IN: pWavelength = */ devWavelength[ gpu ],
												/* IN: pSpw = */ devSpw[ gpu ],
												/* IN: pSampleID = */ devSampleID[ gpu ],
												/* IN: pChannelID = */ devChannelID[ gpu ],
												/* IN: pTaylorTerm = */ t,
												/* IN: pAverageWavelength = */ AverageWavelength,
												/* IN: pVisibilityBatchSize = */ hstVisibilitiesPerGPU[ gpu ],
												/* IN: pNumSpws = */ pNumSpws );

					} // LOOP: gpu

				// download mfs weights from the device.
				for ( int gpu = 0; gpu < pNumGPUs; gpu++ )
					if (hstVisibilitiesPerGPU[ gpu ] > 0)
					{
					
						cudaSetDevice( phstGPU[ gpu ] );
						moveDeviceToHost( (void *) &MfsWeight[ /* STOKES = */ 0 ][ /* TAYLOR_TERM = */ t - 1 ]
															[ hstCurrentVisibility + (gpu * hstVisibilityBatchSize) ],
									(void *) devMfsWeight[ gpu ],
									hstVisibilitiesPerGPU[ gpu ] * sizeof( float ), "copying mfs weights from the device", __LINE__ );
									
						// copy these Mfs weights to the other Stokes products. they are all initially the same, but once we have compacted our
						// visibilities then they will be different.
						if (_stokesImages > 1)
							for ( int s = 1; s < _stokesImages; s++ )
								memcpy( (void *) &MfsWeight[ s ][ /* TAYLOR_TERM = */ t - 1 ]
														[ hstCurrentVisibility + (gpu * hstVisibilityBatchSize) ],
									(void *) &MfsWeight[ /* STOKES = */ 0 ][ /* TAYLOR_TERM = */ t - 1 ]
														[ hstCurrentVisibility + (gpu * hstVisibilityBatchSize) ],
									hstVisibilitiesPerGPU[ gpu ] * sizeof( float ) );
						
					}

				if (t < _taylorTerms)
					for ( int stokes = 0; stokes < _stokesImages; stokes++ )
					{

						for ( int gpu = 0; gpu < pNumGPUs; gpu++ )
							if (hstVisibilitiesPerGPU[ gpu ] > 0)
							{

								cudaSetDevice( phstGPU[ gpu ] );

								// copy the visibilities.
								cudaMemcpy( (void *) devMfsVisibility[ gpu ], (void *) devVisibility[ stokes ][ gpu ],
										hstVisibilitiesPerGPU[ gpu ] * sizeof( cufftComplex ), cudaMemcpyDeviceToDevice );

								// set a suitable thread and block size.
								int threads = hstVisibilitiesPerGPU[ gpu ];
								int blocks = 1;
								setThreadBlockSize1D( &threads, &blocks );

								// multiply the visibilities by the Mfs weights.
								devMultiplyArrays<<< blocks, threads >>>(	/* pOne = */ devMfsVisibility[ gpu ],
														/* pTwo = */ devMfsWeight[ gpu ],
														/* pSize =*/ hstVisibilitiesPerGPU[ gpu ] );

							}

						// copy visibilities to the host.
						for ( int gpu = 0; gpu < pNumGPUs; gpu++ )
							if (hstVisibilitiesPerGPU[ gpu ] > 0)
							{
								cudaSetDevice( phstGPU[ gpu ] );
								moveDeviceToHost(	(void *) &Visibility[ stokes ][ t ]
													[ hstCurrentVisibility + (gpu * hstVisibilityBatchSize) ],
											(void *) devMfsVisibility[ gpu ],
											hstVisibilitiesPerGPU[ gpu ] * sizeof( cufftComplex ),
											"copying Mfs visibilities to the host", __LINE__ );
							}
						
					} // LOOP: stokes

				// (t < _taylorTerms)

			} // LOOP: t

			// free memory.
			if (devMfsWeight != NULL)
			{
				for ( int gpu = 0; gpu < pNumGPUs; gpu++ )
					if (devMfsWeight[ gpu ] != NULL)
					{
						cudaSetDevice( phstGPU[ gpu ] );
						cudaFree( (void *) devMfsWeight[ gpu ] );
					}
				free( (void *) devMfsWeight );
			}
			if (devMfsVisibility != NULL)
			{
				for ( int gpu = 0; gpu < pNumGPUs; gpu++ )
					if (devMfsVisibility[ gpu ] != NULL)
					{
						cudaSetDevice( phstGPU[ gpu ] );
						cudaFree( (void *) devMfsVisibility[ gpu ] );
					}
				free( (void *) devMfsVisibility );
			}

		} // (_param->Deconvolver == MFS)

		// copy the pb-correction channels to the device.
		for ( int gpu = 0; gpu < pNumGPUs; gpu++ )
			if (hstVisibilitiesPerGPU[ gpu ] > 0)
			{
				cudaSetDevice( phstGPU[ gpu ] );
				moveHostToDevice( (void *) devPBChannel[ gpu ], (void *) &hstPBChannel[ gpu * hstVisibilityBatchSize ],
							hstVisibilitiesPerGPU[ gpu ] * sizeof( int ), "copying a-planes to the device", __LINE__ );
			}

		// create memory for grid positions, and kernel indexes.
		VectorI ** devGridPosition = (VectorI **) malloc( pNumGPUs * sizeof( VectorI * ) );
		int ** devKernelIndex = (int **) malloc( pNumGPUs * sizeof( int * ) );
		for ( int gpu = 0; gpu < pNumGPUs; gpu++ )
			if (hstVisibilitiesPerGPU[ gpu ] > 0)
			{

				cudaSetDevice( phstGPU[ gpu ] );

				reserveGPUMemory( (void **) &devGridPosition[ gpu ], hstVisibilitiesPerGPU[ gpu ] * sizeof( VectorI ),
								"reserving device memory for grid positions", __LINE__ );
				reserveGPUMemory( (void **) &devKernelIndex[ gpu ], hstVisibilitiesPerGPU[ gpu ] * sizeof( int ),
								"reserving device memory for kernel indexes", __LINE__ );

				// set a suitable thread and block size.
				int threads = hstVisibilitiesPerGPU[ gpu ];
				int blocks = 1;
				setThreadBlockSize1D( &threads, &blocks );

				// calculate the grid positions and kernel indexes.
				devCalculateGridPositions<<< blocks, threads >>>(	/* OUT: pGridPosition = */ devGridPosition[ gpu ],
											/* OUT: pKernelIndex = */ devKernelIndex[ gpu ],
											/* IN: pUvCellSize = */ pUvCellSize,
											/* IN: pOversample = */ pOversample,
											/* IN: pWPlanes = */ WPlanes,
											/* IN: pPBChannels = */ PBChannels,
											/* IN: pSample = */ devSample[ gpu ],
											/* IN: pWavelength = */ devWavelength[ gpu ],
											/* IN: pSpw = */ devSpw[ gpu ],
											/* IN: pWPlaneMax = */ devWPlaneMax[ gpu ],
											/* IN: pPBChannel = */ devPBChannel[ gpu ],
											/* IN: pSampleID = */ devSampleID[ gpu ],
											/* IN: pChannelID = */ devChannelID[ gpu ],
											/* IN: pSize = */ _param->ImageSize,
											/* IN: pVisibilityBatchSize = */ hstVisibilitiesPerGPU[ gpu ],
											/* IN: pNumSpws = */ pNumSpws );
				cudaError_t err = cudaGetLastError();
				if (err != cudaSuccess)
					printf( "error calculating grid positions (%s)\n", cudaGetErrorString( err ) );

			}
			else
			{
				devGridPosition[ gpu ] = NULL;
				devKernelIndex[ gpu ] = NULL;
			}

		// download these grid positions and kernel indexes to host memory.
		for ( int gpu = 0; gpu < pNumGPUs; gpu++ )
			if (hstVisibilitiesPerGPU[ gpu ] > 0)
			{
				cudaSetDevice( phstGPU[ gpu ] );
				moveDeviceToHost( (void *) &GridPosition[ hstCurrentVisibility + (gpu * hstVisibilityBatchSize) ], (void *) devGridPosition[ gpu ],
							hstVisibilitiesPerGPU[ gpu ] * sizeof( VectorI ), "copying grid positions to host", __LINE__ );
				moveDeviceToHost( (void *) &KernelIndex[ hstCurrentVisibility + (gpu * hstVisibilityBatchSize) ], (void *) devKernelIndex[ gpu ],
							hstVisibilitiesPerGPU[ gpu ] * sizeof( int ), "copying kernel indexes to host", __LINE__ );
			}

		// free memory.
		if (devGridPosition != NULL)
		{
			for ( int gpu = 0; gpu < pNumGPUs; gpu++ )
				if (devGridPosition[ gpu ] != NULL)
				{
					cudaSetDevice( phstGPU[ gpu ] );
					cudaFree( (void *) devGridPosition[ gpu ] );
				}
			free( (void *) devGridPosition );
		}
		if (devKernelIndex != NULL)
		{
			for ( int gpu = 0; gpu < pNumGPUs; gpu++ )
				if (devKernelIndex[ gpu ] != NULL)
				{
					cudaSetDevice( phstGPU[ gpu ] );
					cudaFree( (void *) devKernelIndex[ gpu ] );
				}
			free( (void *) devKernelIndex );
		}

//for ( int i = 0; i < NumVisibilities[ pStageID ]; i++ ) // cjs-mod
//	if (_hstGridPosition[ *pMosaicComponent ][ i ].w < 0 || _hstGridPosition[ *pMosaicComponent ][ i ].w >= 128)
//		printf( "%d (DATA): invalid kernel set: vis = %i, grid position = <%i, %i, %i>\n", __LINE__, i, _hstGridPosition[ *pMosaicComponent ][ i ].u,
//				_hstGridPosition[ *pMosaicComponent ][ i ].v, _hstGridPosition[ *pMosaicComponent ][ i ].w );

		// compact the data so that items with a duplicate grid position are only gridded once.
		hstCurrentVisibility = compactData(	/* pTotalVisibilities = */ &NumVisibilities[ pStageID ],
							/* pFirstVisibility = */ hstCurrentVisibility,
							/* pNumVisibilitiesToProcess = */ visibilitiesAllGPUs );
//		hstCurrentVisibility = hstCurrentVisibility + visibilitiesAllGPUs;

//for ( int i = 0; i < NumVisibilities[ pStageID ]; i++ ) // cjs-mod
//	if (_hstGridPosition[ *pMosaicComponent ][ i ].w < 0 || _hstGridPosition[ *pMosaicComponent ][ i ].w >= 128)
//		printf( "%d (DATA): invalid kernel set: vis = %i, grid position = <%i, %i, %i>\n", __LINE__, i, _hstGridPosition[ *pMosaicComponent ][ i ].u,
//				_hstGridPosition[ *pMosaicComponent ][ i ].v, _hstGridPosition[ *pMosaicComponent ][ i ].w );

		// we track the visibility ID so that we can take the complex conjugate of the second half of the data set.
		uncompactedVisibilityID += visibilitiesAllGPUs;

	} // LOOP: (hstCurrentVisibility < _hstNumVisibilities in stage)

	printf( "\rloading...%3d%%, processing...%3d%%     currently: processing (sorting)                           ",
		(int) ((double)(pSampleStageID + pNumSamplesInStage) * 100.0 / (double) pTotalSamples),
		(int) ((double)(pSampleStageID + pNumSamplesInStage) * 100.0 / (double) pTotalSamples) );
	fflush( stdout );

	// we need to sort data into order of W plane, kernel index, U value and V value.
	quickSortData(	/* pLeft = */ 0,
			/* pRight = */ NumVisibilities[ pStageID ] - 1 );

//for ( int i = 0; i < NumVisibilities[ pStageID ]; i++ ) // cjs-mod
//	if (_hstGridPosition[ *pMosaicComponent ][ i ].w < 0 || _hstGridPosition[ *pMosaicComponent ][ i ].w >= 128)
//		printf( "%d (DATA): invalid kernel set: vis = %i, grid position = <%i, %i, %i>\n", __LINE__, i, _hstGridPosition[ *pMosaicComponent ][ i ].u,
//				_hstGridPosition[ *pMosaicComponent ][ i ].v, _hstGridPosition[ *pMosaicComponent ][ i ].w );

	printf( "\rloading...%3d%%, processing...%3d%%     currently: processing (compacting)                        ",
		(int) ((double)( pSampleStageID + pNumSamplesInStage) * 100.0 / (double) pTotalSamples),
		(int) ((double)( pSampleStageID + pNumSamplesInStage) * 100.0 / (double) pTotalSamples) );
	fflush( stdout );

	// compact the data again so that items with a duplicate grid position are only gridded once.
	compactData(	/* pTotalVisibilities = */ &NumVisibilities[ pStageID ],
			/* pFirstVisibility = */ 0,
			/* pNumVisibilitiesToProcess = */ NumVisibilities[ pStageID ] );

//for ( int i = 0; i < NumVisibilities[ stageID ]; i++ ) // cjs-mod
//	if (_hstGridPosition[ *pMosaicComponent ][ i ].w < 0 || _hstGridPosition[ *pMosaicComponent ][ i ].w >= 128)
//		printf( "%d (DATA): invalid kernel set: vis = %i, grid position = <%i, %i, %i>\n", __LINE__, i, _hstGridPosition[ *pMosaicComponent ][ i ].u,
//				_hstGridPosition[ *pMosaicComponent ][ i ].v, _hstGridPosition[ *pMosaicComponent ][ i ].w );

	// free memory.
	for ( int gpu = 0; gpu < pNumGPUs; gpu++ )
	{
		cudaSetDevice( phstGPU[ gpu ] );
		cudaFree( (void *) devSpw[ gpu ] );
		cudaFree( (void *) devWPlaneMax[ gpu ] );
		cudaFree( (void *) devPhase[ gpu ] );
		for ( int s = 0; s < _stokesImages; s++ )
			cudaFree( (void *) devVisibility[ s ][ gpu ] );
		cudaFree( (void *) devSampleID[ gpu ] );
		cudaFree( (void *) devChannelID[ gpu ] );
		cudaFree( (void *) devPBChannel[ gpu ] );
		cudaFree( (void *) devSample[ gpu ] );
		cudaFree( (void *) devWavelength[ gpu ] );
		for ( int spw = 0; spw < pNumSpws; spw++ )
			cudaFree( (void *) hstdevWavelength[ gpu ][ spw ] );
		free( (void *) hstdevWavelength[ gpu ] );
	}
	free( (void *) devSpw );
	free( (void *) devWPlaneMax );
	free( (void *) devPhase );
	for ( int s = 0; s < _stokesImages; s++ )
		free( (void *) devVisibility[ s ] );
	free( (void *) devVisibility );
	free( (void *) devSampleID );
	free( (void *) devChannelID );
	free( (void *) devPBChannel );
	free( (void *) hstPBChannel );
	free( (void *) devSample );
	free( (void *) devWavelength );
	free( (void *) hstdevWavelength );
	cudaSetDevice( phstGPU[ 0 ] );

} // Data::calculateGridPositions

//
//	calculatePBChannels()
//
//	CJS: 03/04/2020
//
//	Calculate which channels/spws are in which A plane.
//

void Data::calculatePBChannels
			(
			int *** phstWhichPBChannel,			// maps spw and channel to the PB channel
			double * phstPBChannelWavelength,		// holds the wavelength of each PB channel
			double ** phstWavelength,			// holds the wavelength of each spw/channel
			int pNumSpws,
			int * phstNumChannels,
			bool ** phstSpwChannelFlag
			)
{

	printf( "setting %i pb-correction channel(s) for %i SPW(s):\n", _param->PBChannels, pNumSpws );

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

	// sort the array of wavelengths in reverse order, so that we mirror casa in having the channels in frequency order.
	for ( int wavelength1 = 0; wavelength1 < numWavelengths - 1; wavelength1++ )
		for ( int wavelength2 = wavelength1 + 1; wavelength2 < numWavelengths; wavelength2++ )
			if ( hstWavelength[ wavelength2 ] > hstWavelength[ wavelength1 ] )
			{
				double tmp = hstWavelength[ wavelength1 ];
				hstWavelength[ wavelength1 ] = hstWavelength[ wavelength2 ];
				hstWavelength[ wavelength2 ] = tmp;
			}

	// calculate the wavelengths.
	for ( int i = 0; i < _param->PBChannels; i++ )
		phstPBChannelWavelength[ i ] = hstWavelength[ ((2 * i) + 1) * numWavelengths / (2 * _param->PBChannels) ];

	// free data.
	if (hstWavelength != NULL)
		free( (void *) hstWavelength );

	// update each spw and channel with the appropriate A-plane.
	(*phstWhichPBChannel) = (int **) malloc( pNumSpws * sizeof( int * ) );
	int ** hstWhichPBChannel = (*phstWhichPBChannel);
	for ( int spw = 0; spw < pNumSpws; spw++ )
	{

		// create array for this spw, and find the appropriate A-plane.
		hstWhichPBChannel[ spw ] = (int *) malloc( phstNumChannels[ spw ] * sizeof( int ) );
		for ( int channel = 0; channel < phstNumChannels[ spw ]; channel++ )
		{

			// find the closest A-plane.
			double bestError = 0.0;
			for ( int pbChannel = 0; pbChannel < _param->PBChannels; pbChannel++ )
				if (abs( phstWavelength[ spw ][ channel ] - phstPBChannelWavelength[ pbChannel ] ) < bestError || pbChannel == 0)
				{
					hstWhichPBChannel[ spw ][ channel ] = pbChannel;
					bestError = abs( phstWavelength[ spw ][ channel ] - phstPBChannelWavelength[ pbChannel ] );
				}

		} // LOOP: channel
		
	} // LOOP: spw

} // Data::calculatePBChannels

//
//	calculateVisibilityAndFlag()
//
//	CJS: 25/08/2021
//
//	Calculates a set of visibilities and flags from the raw data read in from the measurement set.
//

void Data::calculateVisibilityAndFlag
			(
			int pStageID,					//
			long int pPreferredVisibilityBatchSize,	//
			int pNumPolarisations,				//
			int pNumSamplesInStage,			//
			int * phstPolarisationConfig,			//
			cufftComplex * phstVisibilityIn,		//
			bool * phstFlagIn,				//
			double ** pdevMultiplier			//
			)
{

	cufftComplex * devVisibilityIn = NULL;
	cufftComplex * devVisibilityOut = NULL;
	bool * devFlagIn = NULL, * devFlagOut = NULL;
	int * devPolarisationConfig = NULL;
	int * devSampleID = NULL;

	// if the number of visibilities is greater than the maximum number then we are going to set a smaller batch size, and load these
	// visibilities in batches.
	int hstVisibilityBatchSize = 0;
	{
		long int nextBatchSize = NumVisibilities[ pStageID ] / 2;
		if (nextBatchSize > pPreferredVisibilityBatchSize)
			nextBatchSize = pPreferredVisibilityBatchSize;
		hstVisibilityBatchSize = (int) nextBatchSize;
	}

	reserveGPUMemory( (void **) &devVisibilityIn, pNumPolarisations * hstVisibilityBatchSize * sizeof( cufftComplex ),
				"reserving device memory for loading visibilities", __LINE__ );
	reserveGPUMemory( (void **) &devVisibilityOut, hstVisibilityBatchSize * sizeof( cufftComplex ), "reserving device memory for the processed visibilities", __LINE__ );
	reserveGPUMemory( (void **) &devFlagIn, pNumPolarisations * hstVisibilityBatchSize * sizeof( bool ), "reserving device memory for loading flags", __LINE__ );
	reserveGPUMemory( (void **) &devFlagOut, hstVisibilityBatchSize * sizeof( bool ), "reserving device memory for the processed flags", __LINE__ );
	reserveGPUMemory( (void **) &devPolarisationConfig, pNumSamplesInStage * sizeof( int ), "reserving device memory for the polarisation config id", __LINE__ );
	reserveGPUMemory( (void **) &devSampleID, hstVisibilityBatchSize * sizeof( int ), "creating device memory for the sample ID", __LINE__ );

	// upload the polarisation config ids to the device.
	cudaError_t err = cudaMemcpy( devPolarisationConfig, phstPolarisationConfig, pNumSamplesInStage * sizeof( int ), cudaMemcpyHostToDevice );

	// create the visibility and flag arrays to store the data.
	for ( int s = 0; s < _stokesImages; s++ )
		Visibility[ s ][ /* TaylorTerm = */ 0 ] = (cufftComplex *) malloc( NumVisibilities[ pStageID ] * sizeof( cufftComplex ) );
	Flag = (bool *) malloc( NumVisibilities[ pStageID ] * sizeof( bool ) );

	// keep looping until we have calculated all the visibilities and flags.
	long int hstCurrentVisibility = 0;
	while (hstCurrentVisibility < (NumVisibilities[ pStageID ] / 2))
	{

		// if the number of remaining visibilities is lower than the visibility batch size, then reduce the visibility batch size accordingly.
		if ((NumVisibilities[ pStageID ] / 2) - hstCurrentVisibility < hstVisibilityBatchSize)
			hstVisibilityBatchSize = (NumVisibilities[ pStageID ] / 2) - hstCurrentVisibility;

		// upload the visibilities to the device.
		moveHostToDevice( (void *) devVisibilityIn, (void *) &phstVisibilityIn[ pNumPolarisations * hstCurrentVisibility ],
					pNumPolarisations * hstVisibilityBatchSize * sizeof( cufftComplex ), "copying loaded visibilities to device", __LINE__ );

		// upload the flags to the device.
		moveHostToDevice( (void *) devFlagIn, (void *) &phstFlagIn[ pNumPolarisations * hstCurrentVisibility ],
					pNumPolarisations * hstVisibilityBatchSize * sizeof( bool ), "copying loaded flags to device", __LINE__ );

		// upload the sample IDs to the device.
		moveHostToDevice( (void *) devSampleID, (void *) &SampleID[ hstCurrentVisibility ], hstVisibilityBatchSize * sizeof( int ),
					"copying sample ID to device", __LINE__ );

		int threads = hstVisibilityBatchSize;
		int blocks = 1;
		setThreadBlockSize1D( &threads, &blocks );
		
		// calculate visibilities for each of the required Stokes products.
		for ( int stokes = 0; stokes < _stokesImages; stokes++ )
		{

			// calculate the visibilities and their flags.
			devCalculateVisibility<<< blocks, threads >>>(	/* IN: pVisibilityIn = */ devVisibilityIn,
										/* OUT: pVisibilityOut = */ devVisibilityOut,
										/* IN: pNumPolarisations = */ pNumPolarisations,
										/* IN: pMultiplier = */ pdevMultiplier[ stokes ],
										/* IN: pPolarisationConfig = */ devPolarisationConfig,
										/* IN: pSampleID = */ devSampleID,
										/* IN: pVisibilityBatchSize = */ hstVisibilityBatchSize );

			// copy visibilities and flags out.
			moveDeviceToHost( (void *) &Visibility[ stokes ][ /* TAYLOR_TERM = */ 0 ][ hstCurrentVisibility ], (void *) devVisibilityOut,
						hstVisibilityBatchSize * sizeof( cufftComplex ), "copying calculated visibility from device", __LINE__ );
					
		}

		// calculate the visibilities and their flags.
		devCalculateFlag<<< blocks, threads >>>(	/* IN: pFlagIn = */ devFlagIn,
								/* OUT: pFlagOut = */ devFlagOut,
								/* IN: pNumPolarisations = */ pNumPolarisations,
								/* IN: pMultiplier = */ pdevMultiplier[ /* STOKES = */ 0 ],	// just use the first element of multiplier
								/* IN: pPolarisationConfig = */ devPolarisationConfig,
								/* IN: pSampleID = */ devSampleID,
								/* IN: pVisibilityBatchSize = */ hstVisibilityBatchSize,
								/* IN: pFullStokes = */ (_stokesImages > 1) );

		// copy flags out.		
		moveDeviceToHost( (void *) &Flag[ hstCurrentVisibility ], (void *) devFlagOut,
					hstVisibilityBatchSize * sizeof( bool ), "copying calculated flags from device", __LINE__ );
	
		// get the next batch of data.
		hstCurrentVisibility = hstCurrentVisibility + hstVisibilityBatchSize;

	}

	// duplicate visibilities. the conjugates will be calculated on the GPU.
	for ( int s = 0; s < _stokesImages; s++ )
		memcpy( &Visibility[ s ][ /* TaylorTerm = */ 0 ][ NumVisibilities[ pStageID ] / 2 ], Visibility[ s ][ /* TaylorTerm = */ 0 ],
			(NumVisibilities[ pStageID ] / 2) * sizeof( cufftComplex ) );
	memcpy( &Flag[ NumVisibilities[ pStageID ] / 2 ], Flag, (NumVisibilities[ pStageID ] / 2) * sizeof( bool ) );

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

} // Data::calculateVisibilityAndFlag

//
//	calculateWPlanes()
//
//	CJS: 07/08/2015
//
//	Calculate some values such as support, kernel size and w-cell size. We need to work out the maximum baseline length first. The support is given
//	by the square root of the number of uv cells that fit into the maximum baseline, multiplied by 1.5. The w-cell size is given by the maximum
//	baseline length divided by the number of w-planes, multiplied by two.
//

void Data::calculateWPlanes( int pFieldID, int pCasaFieldID, int pNumSamples, VectorD * phstSample, int * phstFieldID, double pMinWavelength )
{

	// calculate the maximum absolute w value [in metres].
	double maxW = 0.0;
	for ( int i = 0; i < pNumSamples; i++ )
		if (abs( phstSample[ i ].w ) > maxW && phstFieldID[ i ] == pFieldID)
			maxW = abs( phstSample[ i ].w );

	// calculate the maximum and minimum w [in lambda].
	maxW = maxW / pMinWavelength;
	double minW = -maxW;

	// create some memory for the mean and maximum W.
	WPlaneMean = (double *) malloc( WPlanes * sizeof( double ) );
	WPlaneMax = (double *) malloc( WPlanes * sizeof( double ) );

	if (WPlanes > 1)
	{

		double B = maxW / pow( (double) WPlanes / 2.0, 1.5 );
		for ( int i = (WPlanes / 2) - 1; i < WPlanes; i++ )
			WPlaneMax[ i ] = B * pow( (double) (i - ((WPlanes / 2) - 1)), 1.5 );
		for ( int i = 0; i < (WPlanes / 2) - 1; i++ )
			WPlaneMax[ i ] = -WPlaneMax[ WPlanes - 2 - i ];

		// set the mean and maximum w values for this plane.
		for ( int i = 0; i < WPlanes; i++ )
			if (i == 0)
				WPlaneMean[ i ] = ((WPlaneMax[ i ] - minW) / 2.0) + minW;
			else
				WPlaneMean[ i ] = ((WPlaneMax[ i ] - WPlaneMax[ i - 1 ]) / 2.0) + WPlaneMax[ i - 1 ];
			
		printf( "mean values of w-planes for field %i: [", pCasaFieldID );
		for ( int i = 0; i < WPlanes; i++ )
		{
			if (i > 0)
				printf( ", " );
			printf( "%5.4f", WPlaneMean[ i ] );
		}
		printf( "] lambda\n" );

	}
	else if (WPlanes == 1)
	{
		WPlaneMean[ 0 ] = 0;
		WPlaneMax[ 0 ] = maxW;
	}
	
} // Data::calculateWPlanes

//
//	compactData()
//
//	CJS: 15/10/2018
//
//	Compacts the visibility data so that items with a duplicate kernel index and grid position are only gridded once.
//

long int Data::compactData( long int * pTotalVisibilities, long int pFirstVisibility, long int pNumVisibilitiesToProcess )
{

	// store the current visibility and weight in double format. the arrays are single precision, so we must store them separately.
	cufftDoubleComplex ** currentVis = (cufftDoubleComplex **) malloc( _stokesImages * sizeof( cufftDoubleComplex * ) );
	double * currentWeight = (double *) malloc( _stokesImages * sizeof( double ) );
	double ** currentMfsWeight = NULL;
	if (_param->Deconvolver == MFS)
		currentMfsWeight = (double **) malloc( _stokesImages * sizeof( double * ) );
	for ( int s = 0; s < _stokesImages; s++ )
	{
		currentVis[ s ] = (cufftDoubleComplex *) malloc( _taylorTerms * sizeof( cufftDoubleComplex ) );
		if (_param->Deconvolver == MFS)
			currentMfsWeight[ s ] = (double *) malloc( (_taylorTerms - 1) * 2 * sizeof( double ) );
	}

	long int toIndex = pFirstVisibility - 1;
	int lastKernelIndex = -1, lastFieldID = -1;
	int compactedVisibilities = 0;
	VectorI lastPos = { -1, -1, -1 };
	
	// compact the array by looping through and adding up visibilities with the same kernel index and grid position.
	for ( long int fromIndex = pFirstVisibility; fromIndex < pFirstVisibility + pNumVisibilitiesToProcess; fromIndex++ )
	{

		// ensure this visibility isn't flagged.
		bool flagged = false;
		if (Flag != NULL)
			flagged = Flag[ fromIndex ];
		if (flagged == false)
		{

			// determine if this visibility can be merged with another visibility. this will be the case if the grid position, kernel index and field id
			// are identical.
			bool mergeVisibilities = false;
			if (GridPosition[ fromIndex ].u == lastPos.u)
				if (GridPosition[ fromIndex ].v == lastPos.v)
					if (GridPosition[ fromIndex ].w == lastPos.w)
						if (KernelIndex[ fromIndex ] == lastKernelIndex)
						{
							if (FieldID != NULL)
								mergeVisibilities = (FieldID[ fromIndex ] == lastFieldID);
							else
								mergeVisibilities = true;
						}

			// this visibility has a different position or kernel from the last, so move to a new array element.
			if (mergeVisibilities == false)
			{

				// store the last visibility and weight, if needed.
				if (lastKernelIndex >= 0)
					for ( int s = 0; s < _stokesImages; s++ )
					{
					
						if (_param->Weighting != NONE)
						{
							currentWeight[ s ] /= (double) DensityMap[ toIndex ];
							Weight[ s ][ toIndex ] = (float) currentWeight[ s ];
							if (currentWeight[ s ] != 0.0)
							{
								for ( int t = 0; t < _taylorTerms; t++ )
									currentVis[ s ][ t ] = divideComplex(	/* pOne = */ currentVis[ s ][ t ],
														/* pTwo = */ currentWeight[ s ] );
								if (_param->Deconvolver == MFS)
									for ( int t = 0; t < (_taylorTerms - 1) * 2; t++ )
										currentMfsWeight[ s ][ t ] /= currentWeight[ s ];
							}
							else
							{
								for ( int t = 0; t < _taylorTerms; t++ )
								{
									currentVis[ s ][ t ].x = 0.0;
									currentVis[ s ][ t ].y = 0.0;
								}
								if (_param->Deconvolver == MFS)
									for ( int t = 0; t < (_taylorTerms - 1) * 2; t++ )
										currentMfsWeight[ s ][ t ] = 0.0;
							}
						}
						for ( int t = 0; t < _taylorTerms; t++ )
						{
							Visibility[ s ][ t ][ toIndex ].x = (float) currentVis[ s ][ t ].x;
							Visibility[ s ][ t ][ toIndex ].y = (float) currentVis[ s ][ t ].y;
						}
						if (_param->Deconvolver == MFS)
							for ( int t = 0; t < (_taylorTerms - 1) * 2; t++ )
								MfsWeight[ s ][ t ][ toIndex ] = (float) currentMfsWeight[ s ][ t ];
							
					} // LOOP: s

				// increment the index that we should write to, and the count of compacted visibilities.
				toIndex = toIndex + 1;
				compactedVisibilities = compactedVisibilities + 1;

				// copy the data into the new array elements.
				KernelIndex[ toIndex ] = KernelIndex[ fromIndex ];
				GridPosition[ toIndex ] = GridPosition[ fromIndex ];

				for ( int s = 0; s < _stokesImages; s++ )
				{
					for ( int t = 0; t < _taylorTerms; t++ )
					{
						currentVis[ s ][ t ].x = (double) Visibility[ s ][ t ][ fromIndex ].x;
						currentVis[ s ][ t ].y = (double) Visibility[ s ][ t ][ fromIndex ].y;
					}
					if (_param->Deconvolver == MFS)
						for ( int t = 0; t < (_taylorTerms - 1) * 2; t++ )
							currentMfsWeight[ s ][ t ] = (double) MfsWeight[ s ][ t ][ fromIndex ];
					if (_param->Weighting != NONE)
					{
						for ( int t = 0; t < _taylorTerms; t++ )
							currentVis[ s ][ t ] = multComplex(	/* pOne = */ currentVis[ s ][ t ],
												/* pTwo = */ (double) Weight[ s ][ fromIndex ] );
						if (_param->Deconvolver == MFS)
							for ( int t = 0; t < (_taylorTerms - 1) * 2; t++ )
								currentMfsWeight[ s ][ t ] = currentMfsWeight[ s ][ t ] * (double) Weight[ s ][ fromIndex ];
						currentWeight[ s ] = ((double) Weight[ s ][ fromIndex ] * (double) DensityMap[ fromIndex ]);
					}
				}
				DensityMap[ toIndex ] = DensityMap[ fromIndex ];
				if (Flag != NULL)
					Flag[ toIndex ] = false;
				if (FieldID != NULL)
					FieldID[ toIndex ] = FieldID[ fromIndex ];

				// store the current kernel indexes, field ID and grid positions
				lastKernelIndex = KernelIndex[ fromIndex ];
				lastPos = GridPosition[ fromIndex ];
				if (FieldID != NULL)
					lastFieldID = FieldID[ fromIndex ];

			} // (mergeVisibilities == false)
			
			// or alternatively add this visibility to the one we're currently accumulating.
			else if (mergeVisibilities == true)
			{
				for ( int s = 0; s < _stokesImages; s++ )
				{
					double weight = 1.0;
					if (_param->Weighting != NONE)
					{
						weight = (double) Weight[ s ][ fromIndex ];
						currentWeight[ s ] += weight * (double) DensityMap[ fromIndex ];
					}
					for ( int t = 0; t < _taylorTerms; t++ )
					{
						currentVis[ s ][ t ].x += (double) Visibility[ s ][ t ][ fromIndex ].x * weight;
						currentVis[ s ][ t ].y += (double) Visibility[ s ][ t ][ fromIndex ].y * weight;
					}
					if (_param->Deconvolver == MFS)
						for ( int t = 0; t < (_taylorTerms - 1) * 2; t++ )
							currentMfsWeight[ s ][ t ] += (double) MfsWeight[ s ][ t ][ fromIndex ] * weight;
				}

				// add up the density map at this position.
				DensityMap[ toIndex ] += DensityMap[ fromIndex ];

			} // (mergeVisibilities == true)

		} // (flagged == false)

	} // LOOP: fromIndex

	// store the last visibility and weight, if needed.
	if (lastKernelIndex >= 0)
		for ( int s = 0; s < _stokesImages; s++ )
		{
			if (_param->Weighting != NONE)
			{
				currentWeight[ s ] /= (double) DensityMap[ toIndex ];
				Weight[ s ][ toIndex ] = (float) currentWeight[ s ];
				if (currentWeight[ s ] != 0.0)
				{
					for ( int t = 0; t < _taylorTerms; t++ )
						currentVis[ s ][ t ] = divideComplex( /* pOne = */ currentVis[ s ][ t ], /* pTwo = */ currentWeight[ s ] );
					if (_param->Deconvolver == MFS)
						for ( int t = 0; t < (_taylorTerms - 1) * 2; t++ )
							currentMfsWeight[ s ][ t ] /= currentWeight[ s ];
				}
				else
				{
					for ( int t = 0; t < _taylorTerms; t++ )
					{
						currentVis[ s ][ t ].x = 0.0;
						currentVis[ s ][ t ].y = 0.0;
					}
					if (_param->Deconvolver == MFS)
						for ( int t = 0; t < (_taylorTerms - 1) * 2; t++ )
							currentMfsWeight[ s ][ t ] = 0.0;
				}
			}
			for ( int t = 0; t < _taylorTerms; t++ )
			{
				Visibility[ s ][ t ][ toIndex ].x = (float) currentVis[ s ][ t ].x;
				Visibility[ s ][ t ][ toIndex ].y = (float) currentVis[ s ][ t ].y;
			}
			if (_param->Deconvolver == MFS)
				for ( int t = 0; t < (_taylorTerms - 1) * 2; t++ )
					MfsWeight[ s ][ t ][ toIndex ] = (float) currentMfsWeight[ s ][ t ];
		} // LOOP: s

	// have we shrunk the array ?
	if (compactedVisibilities < pNumVisibilitiesToProcess)
	{

		// move visibilities, flags, sample IDs, and channel IDs to earlier in the memory.
		long int nextBatchIndex = pFirstVisibility + pNumVisibilitiesToProcess;
		if (nextBatchIndex < *pTotalVisibilities)
		{
			long int moveFrom = nextBatchIndex;
			long int moveTo = toIndex + 1;
			long int moveNumber = *pTotalVisibilities - moveFrom;
			for ( int s = 0; s < _stokesImages; s++ )
				for ( int t = 0; t < _taylorTerms; t++ )
					memmove( (void *) &Visibility[ s ][ t ][ moveTo ], (void *) &Visibility[ s ][ t ][ moveFrom ],
							moveNumber * (long int) sizeof( cufftComplex ) );

			if (Flag != NULL)
				memmove( (void *) &Flag[ moveTo ], (void *) &Flag[ moveFrom ], moveNumber * (long int) sizeof( bool ) );
			if (SampleID != NULL)
				memmove( (void *) &SampleID[ moveTo ], (void *) &SampleID[ moveFrom ], moveNumber * (long int) sizeof( int ) );
			if (ChannelID != NULL)
				memmove( (void *) &ChannelID[ moveTo ], (void *) &ChannelID[ moveFrom ], moveNumber * (long int) sizeof( int ) );
		}

		// update the total number of visibilities.
		*pTotalVisibilities -= (pNumVisibilitiesToProcess - compactedVisibilities);

		// compact the arrays.
		GridPosition = (VectorI *) realloc( GridPosition, *pTotalVisibilities * sizeof( VectorI ) );
		KernelIndex = (int *) realloc( KernelIndex, *pTotalVisibilities * sizeof( int ) );
		DensityMap = (int *) realloc( DensityMap, *pTotalVisibilities * sizeof( int ) );
		if (Flag != NULL)
			Flag = (bool *) realloc( Flag, *pTotalVisibilities * sizeof( bool ) );
		if (SampleID != NULL)
			SampleID = (int *) realloc( SampleID, *pTotalVisibilities * sizeof( int ) );
		if (ChannelID != NULL)
			ChannelID = (int *) realloc( ChannelID, *pTotalVisibilities * sizeof( int ) );
		if (FieldID != NULL)
			FieldID = (int *) realloc( FieldID, *pTotalVisibilities * sizeof( int ) );
		for ( int s = 0; s < _stokesImages; s++ )
		{
			for ( int t = 0; t < _taylorTerms; t++ )
				Visibility[ s ][ t ] = (cufftComplex *) realloc( Visibility[ s ][ t ], *pTotalVisibilities * sizeof( cufftComplex ) );
			if (_param->Weighting != NONE)
				Weight[ s ] = (float *) realloc( Weight[ s ], *pTotalVisibilities * sizeof( float ) );
			if (_param->Deconvolver == MFS)
				for ( int t = 0; t < (_taylorTerms - 1) * 2; t++ )
					MfsWeight[ s ][ t ] = (float *) realloc( MfsWeight[ s ][ t ], *pTotalVisibilities * sizeof( float ) );
		}

	}

	// free memory.
	if (currentVis != NULL)
	{
		for ( int s = 0; s < _stokesImages; s++ )
			free( (void *) currentVis[ s ] );
		free( (void *) currentVis );
	}
	if (currentMfsWeight != NULL)
	{
		for ( int s = 0; s < _stokesImages; s++ )
			free( (void *) currentMfsWeight[ s ] );
		free( (void *) currentMfsWeight );
	}
	if (currentWeight != NULL)
		free( (void *) currentWeight );

	// return the index of the next batch of data.
	return toIndex + 1;

} // Data::compactData

//
//	compactFieldIDs()
//
//	CJS: 30/11/2018
//
//	We may only be using a few fields amongst the many in the MS, so we want to renumber them as 0,1,2,3,etc, instead of e.g. 3,7,9,11,23,etc.
//

void Data::compactFieldIDs( double ** pPhaseCentrePtr, double ** pPhaseCentreImagePtr, int * pNumFields, int * pFieldID, int ** pFieldIDMap, int pNumSamples )
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

} // Data::compactFieldIDs

//
//	determinantImageMatrix()
//
//	CJS: 01/11/2021
//
//	Find the determinant on a NxN matrix of complex images.
//

cufftDoubleComplex * Data::determinantImageMatrix
			(
			cufftDoubleComplex ** pdevMatrix,		// an array of image pointers that describe an NxN matrix
			int pMatrixSize,				// the matrix size, i.e. NxN
			int pImageSize					// the size of the image held in each matrix cell
			)
{

	cufftDoubleComplex * devReturnImage = NULL;
	
	// create a new image.
	reserveGPUMemory( (void **) &devReturnImage, pImageSize * pImageSize * sizeof( cufftDoubleComplex ), "declaring device memory for matrix determinant", __LINE__ );

	// if there's only one matrix element then we return it by copying the image.
	if (pMatrixSize == 1)
	{
		if (*pdevMatrix != NULL)
			cudaMemcpy( (void *) devReturnImage, (void *) *pdevMatrix, pImageSize * pImageSize * sizeof( cufftDoubleComplex ), cudaMemcpyDeviceToDevice );
		else
		{
			cudaFree( (void *) devReturnImage );
			devReturnImage = NULL;
		}
	}
	else
	{
	
		// otherwise, we move along the top row, multiplying each cell by the determinant of the remaining matrix.
		cufftDoubleComplex ** devReducedMatrix = (cufftDoubleComplex **) malloc( (pMatrixSize - 1) * (pMatrixSize - 1) * sizeof( cufftDoubleComplex * ) );
		
		// clear the return image.
		zeroGPUMemory( (void *) devReturnImage, pImageSize * pImageSize * sizeof( cufftDoubleComplex ), "zeroing device memory for the determinant", __LINE__ );
		
		// and set the thread block size for processing images on the device.
		setThreadBlockSize2D( pImageSize, pImageSize, _gridSize2D, _blockSize2D );

		int items = pImageSize * pImageSize;
		int stages = items / MAX_THREADS;
		if (items % MAX_THREADS != 0)
			stages++;
		
		bool allNull = true;
		for ( int cell = 0; cell < pMatrixSize; cell++ )
			if (pdevMatrix[ cell ] != NULL)
			{
			
				// construct the reduced matrix from all the matrix cells that are not in row 0, or column 'cell'.
				for ( int destRow = 0; destRow < pMatrixSize - 1; destRow++ )
					for ( int destCol = 0; destCol < pMatrixSize - 1; destCol++ )
					{
						int sourceRow = destRow + 1;
						int sourceCol = (destCol < cell ? destCol : destCol + 1 );
						devReducedMatrix[ (destRow * (pMatrixSize - 1)) + destCol ] = pdevMatrix[ (sourceRow * pMatrixSize) + sourceCol ];
					}
					
				// calculate the determinant of the reduced matrix.
				cufftDoubleComplex * devDeterminant = determinantImageMatrix(	/* pMatrix = */ devReducedMatrix,
													/* pMatrixSize = */ pMatrixSize - 1,
													/* pImageSize = */ pImageSize );
				if (devDeterminant != NULL)
				{
				
					allNull = false;
											
					// multiply this image by the image in matrix cell [cell, 0].
					devMultiplyImages<<< _gridSize2D, _blockSize2D >>>(	/* pOne = */ devDeterminant,
												/* pTwo = */ pdevMatrix[ cell ],
												/* pMask = */ NULL,
												/* pSizeOne = */ pImageSize,
												/* pSizeTwo = */ pImageSize );
					
					// for odd cells, multiply the image by -1.
					if (cell % 2 == 1)
						devMultiplyImages<<< _gridSize2D, _blockSize2D >>>(	/* pOne = */ devDeterminant,
													/* pScalar = */ -1.0,
													/* pMask = */ NULL,
													/* pSizeOne = */ pImageSize );
					
					// add this part of the determinant to the return value.
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
						devAddArrays<<< blocks, threads >>>(	/* pOne = */ &devReturnImage[ /* CELL = */ i * MAX_THREADS ],
											/* pTwo = */ &devDeterminant[ /* CELL = */ i * MAX_THREADS ],
											/* pSize = */ itemsThisStage );
											
					} // LOOP: i
					
					// free memory.
					if (devDeterminant != NULL)
						cudaFree( (void *) devDeterminant );
					
				}
			
			} // LOOP: cell
			
		// set the return image to null if we couldn't find anything to multiply.
		if (allNull == true)
		{
			cudaFree( (void *) devReturnImage );
			devReturnImage = NULL;
		}
			
		// free memory.
		if (devReducedMatrix != NULL)
			free( (void *) devReducedMatrix );
	
	} // (pMatrixSize > 1)
	
	// return an image.
	return devReturnImage;

} // determinantImageMatrix

//
//	doPhaseCorrectionSamples()
//
//	CJS: 12/08/2015
//
//	Phase correct the samples using the PhaseCorrection class.
//

void Data::doPhaseCorrectionSamples( PhaseCorrection * pPhaseCorrection, int pNumSamples, double * pPhaseCentreIn, double * pPhaseCentreOut, VectorD * pSample, 
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
	
} // Data::doPhaseCorrectionSamples

//
//	generatePrimaryBeamAiry()
//
//	CJS: 01/011/2019
//
//	Generates primary beams for each channel from an Airy disk with a blockage at the centre.
//

void Data::generatePrimaryBeamAiry
	(
	cufftComplex ** phstJonesMatrix,		// the primary beam array in which to place the generated beams. this is the full Jones matrix, and we
							//		only generate images in cells 0, and 3. cells 1 and 2 remain empty.
	double pWidth,					// the width of the dish
	double pCutout,				// the cutout representing the focus box
	int pNumSpws,					// the number of spws
	int * phstNumChannels,				// the number of channels per spw
	double ** phstWavelength			// the wavelength
	)
{

	const double PIXELS_PER_METRE = 8.0;

	printf( "generating primary beam using an Airy disk from a uniformly illuminated %4.2f m aperture dish with a %4.2f m blockage.....\n\n", pWidth, pCutout );
	
	// count the number of channels.
	int totalChannels = 0;
	for ( int spw = 0; spw < pNumSpws; spw++ )
		totalChannels += phstNumChannels[ spw ];

	// simulate an airy disk primary beam. we use 6x the required beam size to generate the beam, but then chop it down by a factor of 6 later.
	int workspaceSupport = Parameters::BEAM_SIZE * 3;
	int workspaceSize = (workspaceSupport * 2); // in pixels
	
	// the workspace size is 6x larger than our beam size, so we set the beam size here.
	_param->BeamInSize = workspaceSize / 6;

	// create primary beams on the host. we only set the diagonal array elements because these give us all the non-leakage beam patterns for I, Q, U and V.
	for ( int i = 0; i < 4; i += 3 )
	{
		phstJonesMatrix[ /* CELL = */ i ] = (cufftComplex *) malloc( totalChannels * _param->BeamInSize * _param->BeamInSize * sizeof( cufftComplex ) );
		memset( (void *) phstJonesMatrix[ /* CELL = */ i ], 0, totalChannels * _param->BeamInSize * _param->BeamInSize * sizeof( cufftComplex ) );
	}

	// create beam on device.
	cufftComplex * devBeam = NULL;
	reserveGPUMemory( (void **) &devBeam, workspaceSize * workspaceSize * sizeof( cufftComplex ), "reserving device memory for primary beam", __LINE__ );
	
	// create workspace beam on host.
	cufftComplex * hstBeam = (cufftComplex *) malloc( workspaceSize * workspaceSize * sizeof( cufftComplex ) );
	memset( (void *) hstBeam, 0, workspaceSize * workspaceSize * sizeof( cufftComplex ) );

	// calculate the image-plane pixel size of the primary beam, based upon spw 0, channel 0.
	double uvPixelSize = 1.0 / PIXELS_PER_METRE; 								// in metres
	double uvPixelSizeInLambda = uvPixelSize / phstWavelength[ /* SPW = */ 0 ][ /* CHANNEL = */ 0 ];	// in units of lambda
	double imFieldOfView = (180.0 * 3600.0 / PI) * (1.0 / uvPixelSizeInLambda);				// in arcsec
	
	// set the beam cell size and frequency.
	_param->BeamInCellSize = imFieldOfView / workspaceSize;
	
	int cumulativeChannel = 0;
	for ( int spw = 0; spw < pNumSpws; spw++ )
		for ( int channel = 0; channel < phstNumChannels[ spw ]; channel++, cumulativeChannel++ )
		{

			// calculate the max and min radius using the width and blockage. we divide by two to get the radius, and square (so we don't need to keep taking
			// square roots).
			// we scale the radii by the ratio of wavelengths to that of spw 0, channel 0, in order that shorter wavelengths create larger apertures.
			double maxRadius = pow( pWidth * PIXELS_PER_METRE * phstWavelength[ /* SPW = */ 0 ][ /* CHANNEL = */ 0 ] /
									(2.0 * phstWavelength[ spw ][ channel ]), 2 );
			double minRadius = pow( pCutout * PIXELS_PER_METRE * phstWavelength[ /* SPW = */ 0 ][ /* CHANNEL = */ 0 ] /
									(2.0 * phstWavelength[ spw ][ channel ]), 2 );

			long int beamPtr = 0;
			for ( int j = 0; j < workspaceSize; j++ )
				for ( int i = 0; i < workspaceSize; i++, beamPtr++ )
				{
					double r = (double) (pow( i - workspaceSupport, 2 ) + pow( j - workspaceSupport, 2 ));
					hstBeam[ beamPtr ].x = ( r >= minRadius && r <= maxRadius ? 1.0 : 0.0 );
				}

			// copy beam to device.
			moveHostToDevice( (void *) devBeam, (void *) hstBeam, workspaceSize * workspaceSize * sizeof( cufftComplex ),
						"copying primary beam to device", __LINE__ );

			// FFT the primary beam into the uv domain.
			performFFT(	/* pdevGrid = */ (cufftComplex **) &devBeam,
					/* pSize = */ workspaceSize,
					/* pFFTDirection = */ FORWARD,
					/* pFFTPlan = */ -1,
					/* pFFTType = */ C2C,
					/* pResizeArray = */ false );

			// get the maximum value from the beam. create a new memory area to hold the maximum pixel value.
			double * devMaxValue;
			reserveGPUMemory( (void **) &devMaxValue, MAX_PIXEL_DATA_AREA_SIZE * sizeof( double ), "declaring device memory for kernel max pixel value", 
																					__LINE__ );

			// get the peak value from the kernel.
			getMaxValue(	/* pdevImage = */ devBeam,
					/* pdevMaxValue = */ devMaxValue,
					/* pWidth = */ workspaceSize,
					/* pHeight = */ workspaceSize,
					/* pIncludeComplexComponent = */ true,
					/* pMultiplyByConjugate = */ true,
					/* pdevMask = */ NULL );

			// get the maximum value, and its position.
			double maxValue = 0.0;
			cudaMemcpy( &maxValue, &devMaxValue[ MAX_PIXEL_VALUE ], sizeof( double ), cudaMemcpyDeviceToHost );

			// free the max value memory area.
			if (devMaxValue != NULL)
				cudaFree( (void *) devMaxValue );

			// define the block/thread dimensions.
			int threads = workspaceSize * workspaceSize;
			int blocks;
			setThreadBlockSize1D( &threads, &blocks );

			// normalise the image
			devNormalise<<< blocks, threads >>>(	/* pArray = */ devBeam,
								/* pConstant = */ sqrt( maxValue ),
								/* pItems = */ workspaceSize * workspaceSize );

			// reduce the size by a factor of 6.
			threads = _param->BeamInSize * _param->BeamInSize;
			setThreadBlockSize1D( &threads, &blocks );

			// move the centre of the image into the first part of the image.
			devMoveToStartOfImage<<< blocks, threads >>>(	/* pImage = */ devBeam,
									/* pInitialSize = */ workspaceSize,
									/* pFinalSize = */ _param->BeamInSize );

			// copy primary beam back to host.
			cudaMemcpy( (void *) &(phstJonesMatrix[ /* CELL = */ 0 ][ cumulativeChannel * _param->BeamInSize * _param->BeamInSize ]), (void *) devBeam,
									_param->BeamInSize * _param->BeamInSize * sizeof( cufftComplex ), cudaMemcpyDeviceToHost );
	
		} // LOOP: spw & channel
							
	// copy primary beam from cell 0 to cell 3.
	memcpy( (void *) phstJonesMatrix[ /* CELL = */ 3 ], (void *) phstJonesMatrix[ /* CELL = */ 0 ],
							totalChannels * _param->BeamInSize * _param->BeamInSize * sizeof( cufftComplex ) );

	// free memory.
	if (devBeam != NULL)
		cudaFree( (void *) devBeam );
	if (hstBeam != NULL)
		free( (void *) hstBeam );

} // Data::generatePrimaryBeamAiry

//
//	generatePrimaryBeamGaussian()
//
//	CJS: 05/01/2022
//
//	Generates primary beams for each channel from a Gaussian.
//

void Data::generatePrimaryBeamGaussian
	(
	cufftComplex ** phstJonesMatrix,		// the primary beam array in which to place the generated beams. this is the full Jones matrix, and we
							//		only generate images in cells 0, and 3. cells 1 and 2 remain empty.
	double pWidth,					// the width of the dish
	int pNumSpws,					// the number of spws
	int * phstNumChannels,				// the number of channels per spw
	double ** phstWavelength			// the wavelength
	)
{

	printf( "generating primary beam using a Gaussian based upon a %4.2f m aperture dish.....\n", pWidth );
	
	// count the number of channels.
	int totalChannels = 0;
	for ( int spw = 0; spw < pNumSpws; spw++ )
		totalChannels += phstNumChannels[ spw ];

	_param->BeamInSize = Parameters::BEAM_SIZE; // in pixels

	// create primary beams on the host. we only set the diagonal array elements because these give us all the non-leakage beam patterns for I, Q, U and V.
	for ( int i = 0; i < 4; i += 3 )
	{
		phstJonesMatrix[ /* CELL = */ i ] = (cufftComplex *) malloc( totalChannels * _param->BeamInSize * _param->BeamInSize * sizeof( cufftComplex ) );
		memset( (void *) phstJonesMatrix[ /* CELL = */ i ], 0, totalChannels * _param->BeamInSize * _param->BeamInSize * sizeof( cufftComplex ) );
	}

	// create beam on device.
	cufftComplex * devBeam = NULL;
	reserveGPUMemory( (void **) &devBeam, _param->BeamInSize * _param->BeamInSize * sizeof( cufftComplex ), "reserving device memory for primary beam", __LINE__ );
	
	// calculate the image-plane FWHM of the beam (in arcseconds), and the beam cell size in pixels. the whole image should be 4x the beam width of spw 0, channel 0.
	// the fwhm of an airy disk is roughly 1.028 lambda/D, but this is the power pattern; the fwhm of the voltage pattern is 1.028 x sqrt(2) lambda/D = 1.454 lambda/D.
	double maxLambda = phstWavelength[ /* SPW = */ 0 ][ /* CHANNEL = */ 0 ];
	double fwhm = (180.0 * 3600.0 / PI) * 1.454 * maxLambda / pWidth;	// in arcseconds	// 1.15 // 1.028
	_param->BeamInCellSize = fwhm * 4.0 / (double) _param->BeamInSize;			// in arcseconds
	
	printf( "        max wavelength %f m, fwhm %f arcseconds, beam cell size %f arcseconds\n\n", maxLambda, fwhm, _param->BeamInCellSize );

	int cumulativeChannel = 0;
	for ( int spw = 0; spw < pNumSpws; spw++ )
		for ( int channel = 0; channel < phstNumChannels[ spw ]; channel++, cumulativeChannel++ )
		{
		
			// calculate the FWHM, and radius of the Gaussian, both in pixels, for this channel.
			// the Gaussian has form: exp( -(x / radius)^2 ), and the FWHM is given by  2.radius.sqrt( ln2 )
			fwhm = (180.0 * 3600.0 / PI) * 1.454 * phstWavelength[ spw ][ channel ] / (pWidth * _param->BeamInCellSize);	// 1.15 // 1.028
			double radius = (fwhm / 2.0) / sqrt( log( 2 ) );	// in C++ log(X) gives the natural log.

			// define the block/thread dimensions.
			setThreadBlockSize2D( _param->BeamInSize, _param->BeamInSize, _gridSize2D, _blockSize2D );

			// construct the primary beam on the device.
			devMakeBeam<<< _gridSize2D, _blockSize2D >>>(	/* pBeam = */ devBeam,
									/* pAngle = */ 0.0,
									/* pR1 = */ radius,
									/* pR2 = */ radius,
									/* pX = */ (double) (_param->BeamInSize / 2),
									/* pY = */ (double) (_param->BeamInSize / 2),
									/* pSize = */ _param->BeamInSize );

			int items = _param->BeamInSize * _param->BeamInSize;
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
									
				// take the square root of each array element.
//				devSquareRoot<<< _gridSize2D, _blockSize2D >>>(	/* pArray = */ &devBeam[ /* CELL = */ i * MAX_THREADS ],
//											/* pSize = */ itemsThisStage );
							
			}

			// copy primary beam back to host.
			cudaMemcpy( (void *) &(phstJonesMatrix[ /* CELL = */ 0 ][ cumulativeChannel * _param->BeamInSize * _param->BeamInSize ]), (void *) devBeam,
									_param->BeamInSize * _param->BeamInSize * sizeof( cufftComplex ), cudaMemcpyDeviceToHost );
	
		} // LOOP: spw & channel
							
	// copy primary beam from cell 0 to cell 3.
	memcpy( (void *) phstJonesMatrix[ /* CELL = */ 3 ], (void *) phstJonesMatrix[ /* CELL = */ 0 ],
							totalChannels * _param->BeamInSize * _param->BeamInSize * sizeof( cufftComplex ) );

	// free memory.
	if (devBeam != NULL)
		cudaFree( (void *) devBeam );

} // Data::generatePrimaryBeamGaussian

//
//	getASKAPBeamPosition()
//
//	CJS: 20/12/2019
//
//	Calculates the position of each ASKAP beam relative to the phase centre.
//

void Data::getASKAPBeamPosition( double * pRA, double * pDEC, double pXOffset, double pYOffset, double pCentreRA, double pCentreDEC )
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

//printf( "getASKAPBeamPosition: RA-in, DEC-in <%8.6f, %8.6f>, beamOffset <%8.6f, %8.6f>, RA-out, DEC-out <%8.6f, %8.6f>\n\n", (pCentreRA - 360.0) * PI / 180.0,
//		pCentreDEC * PI / 180.0, pXOffset, pYOffset, *pRA * PI / 180.0, *pDEC * PI / 180.0 );

} // Data::getASKAPBeamPosition

//
//	getPrimaryBeamWidth()
//
//	CJS: 04/02/2020
//
//	Calculate the width of the primary beam in pixels, at the 1% level.
//

double Data::getPrimaryBeamWidth( float * phstBeam, int pBeamSize )
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

} // Data::getPrimaryBeamWidth

//
//	getSuitablePhasePositionForBeam()
//
//	CJS: 03/02/2020
//
//	Works out a suitable phase position for gridding based upon a primary beam at one position and a required image at another position.
//

void Data::getSuitablePhasePositionForBeam
			(
			double * pBeamIn,				// the generated/loaded primary beam
			double * pPhase,				// an array of phase positions for each field in this file
			int pNumBeams,					// the number of fields in the file
			double pBeamWidth				// the width of the primary beam in pixels
			)
{

	Reprojection imagePlaneReprojection;

	// set up input pixel vector and CD matrix.
	Reprojection::rpVectD tmpPixelIn = { /* x = */ _param->BeamSize / 2, /* y = */ _param->BeamSize / 2, /*z = */ 0 };
	Reprojection::rpMatr2x2 tmpCDIn = { /* a11 = */ -sin( rad( _param->BeamInCellSize / 3600.0 ) ), /* a12 = */ 0.0, /* a21 = */ 0.0, /* a22 = */ sin( rad( _param->BeamInCellSize / 3600.0 ) ) };

	// set up output pixel vector and CD matrix.
	Reprojection::rpVectD tmpPixelOut = { /* x = */ _param->ImageSize / 2, /* y = */ _param->ImageSize / 2, /*z = */ 0 };
	Reprojection::rpMatr2x2 tmpCDOut = { /* a11 = */ -sin( rad( _param->CellSize / 3600.0 ) ), /* a12 = */ 0.0, /* a21 = */ 0.0, /* a22 = */ sin( rad( _param->CellSize / 3600.0 ) ) };

	// build input and output size.
	Reprojection::rpVectI inSize = { /* x = */ _param->BeamSize, /* y = */ _param->BeamSize };
	Reprojection::rpVectI outSize = { /* x = */ _param->ImageSize, /* y = */ _param->ImageSize };

	// build in coordinate system.
	Reprojection::rpCoordSys inCoordSystem;
	inCoordSystem.crPIX = tmpPixelIn;
	inCoordSystem.cd = tmpCDIn;
	inCoordSystem.epoch = Reprojection::EPOCH_J2000;

	// build out coordinate system.
	Reprojection::rpCoordSys outCoordSystem;
	outCoordSystem.crVAL.x = _param->OutputRA;
	outCoordSystem.crVAL.y = _param->OutputDEC;
	outCoordSystem.crPIX = tmpPixelOut;
	outCoordSystem.cd = tmpCDOut;
	outCoordSystem.epoch = Reprojection::EPOCH_J2000;

	const int TOP_LEFT = 0;
	const int TOP_RIGHT = 2;
	const int BOTTOM_LEFT = 4;
	const int BOTTOM_RIGHT = 6;
	const int X = 0;
	const int Y = 1;

	double top = 0.0, bottom = (double) (_param->ImageSize - 1), left = (double) (_param->ImageSize - 1), right = 0.0;

	// loop over all the beams.
	for ( int beam = 0; beam < pNumBeams; beam++ )
	{

		inCoordSystem.crVAL.x = pBeamIn[ beam * 2 ];
		inCoordSystem.crVAL.y = pBeamIn[ (beam * 2) + 1 ];

		// build a list of pixels to be reprojected.
		double * pixel = (double *) malloc( 8 * sizeof( double ) );
		pixel[ TOP_LEFT + X ] = (double) ((_param->BeamSize / 2) - pBeamWidth); // top-left
		pixel[ TOP_LEFT + Y ] = (double) ((_param->BeamSize / 2) + pBeamWidth);
		pixel[ TOP_RIGHT + X ] = (double) ((_param->BeamSize / 2) + pBeamWidth); // top-right
		pixel[ TOP_RIGHT + Y ] = (double) ((_param->BeamSize / 2) + pBeamWidth);
		pixel[ BOTTOM_LEFT + X ] = (double) ((_param->BeamSize / 2) - pBeamWidth); // bottom-left
		pixel[ BOTTOM_LEFT + Y ] = (double) ((_param->BeamSize / 2) - pBeamWidth);
		pixel[ BOTTOM_RIGHT + X ] = (double) ((_param->BeamSize / 2) + pBeamWidth); // bottom-right
		pixel[ BOTTOM_RIGHT + Y ] = (double) ((_param->BeamSize / 2) - pBeamWidth);

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
	if (top >= (_param->ImageSize - 1))
		top = (double) (_param->ImageSize - 1);
	if (right >= (_param->ImageSize - 1))
		right = (double) (_param->ImageSize - 1);
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

} // Data::getSuitablePhasePositionForBeam

//
//	loadPrimaryBeam()
//
//	CJS: 28/09/2021
//
//	Loads the complex primary beam from a csv file.
//

bool Data::loadPrimaryBeam
			(
			char * pBeamFilename,				//
			cufftComplex ** phstPrimaryBeamIn,		//
			int pSize,					//
			int pNumChannels				//
			)
{ 

	char * fileData = NULL;

	printf( "loading primary beams: %s\n\n", pBeamFilename );
	printf( "        beam size %i x %i pixels, beam cell size %6.2f arcsec/pixel\n", pSize, pSize, _param->BeamInCellSize );
	printf( "        expected %i channels, and %ix%i image size\n", pNumChannels, pSize, pSize );
	
	bool * stokesFound = (bool *) malloc( 4 * sizeof( bool ) );
	for ( int s = 0; s < 4; s++ )
		stokesFound[ s ] = false;
	
	// load each of the X, XY, YX, and YX beams.
	for ( int beam = 0; beam < 4; beam++ )
	{
	
		fileData = NULL;
	
		if (beam == 0)
			printf( "        loading XX " );
		else if (beam == 1)
			printf( ", XY " );
		else if (beam == 2)
			printf( ", YX " );
		else if (beam == 3)
			printf( ", YY " );
		fflush( stdout );
		
		char filename[ 100 ];
		strcpy( filename, pBeamFilename );
		for ( int i = 0; i < strlen( filename ) - 1; i++ )
			if (filename[ i ] == '*' && filename[ i + 1 ] == '*')
			{
				if (beam == 0)		{ filename[ i ] = 'X'; filename[ i + 1 ] = 'X'; }
				else if (beam == 1)	{ filename[ i ] = 'X'; filename[ i + 1 ] = 'Y'; }
				else if (beam == 2)	{ filename[ i ] = 'Y'; filename[ i + 1 ] = 'X'; }
				else if (beam == 3)	{ filename[ i ] = 'Y'; filename[ i + 1 ] = 'Y'; }
			}
			
		long int fileSize = 0;

		// open the file.
		FILE * beamFile = fopen( filename, "rt" );
		if (beamFile == NULL)
		{
			printf( "(missing)" );
			fflush( stdout );
			continue;
		}
		
		printf( "(found)" );
		fflush( stdout );
		stokesFound[ beam ] = true;

		// create some memory for the beam, and set it to zero.
		phstPrimaryBeamIn[ beam ] = (cufftComplex *) malloc( pNumChannels * pSize * pSize * sizeof( cufftComplex ) );
		memset( phstPrimaryBeamIn[ beam ], 0.0, pNumChannels * pSize * pSize * sizeof( cufftComplex ) );
		
		fseek( beamFile, 0L, SEEK_END );
		fileSize = ftell( beamFile );

		// rewind the file.
		rewind( beamFile );
		
		long int bytesProcessed = 0;
		int channel = 0, j = 0, i = 0;
		int bytesInArray = 0;
		while (fileSize > 0)
		{
		
			long int batchSize = 750 * 1024 * 1024; // 500 MB
			if (fileSize < (batchSize - bytesInArray))
				batchSize = fileSize + bytesInArray;

			// create some memory to hold the contents of this file.
			if (fileData == NULL)
				fileData = (char *) malloc( batchSize );

			// get the contents of the file.
			fgets( fileData + bytesInArray, batchSize - bytesInArray, beamFile );
			
			// decrement the file size by the number of bytes we've read.
			fileSize = fileSize - (batchSize - bytesInArray);

			// replace all parentheses, the complex character j, and a zero ascii (eof marker), with spaces.
			for ( long int cell = 0; cell < batchSize; cell++ )
				if (fileData[ cell ] == '(' || fileData[ cell ] == ')' || fileData[ cell ] == 'j' || fileData[ cell ] == 0)
					fileData[ cell ] = ' ';

			// replace all instances of '+-' with ' -'.
			for ( long int cell = 0; cell < batchSize - 1; cell++ )
				if (fileData[ cell ] == '+' && fileData[ cell + 1 ] == '-')
					fileData[ cell ] = ' ';

			// loop over the expected number of pixels.
			char * current = fileData, * next;
			while (channel < pNumChannels && i < pSize && j < pSize)
			{
			
				// if we are near the end of the data we have (within the last MB) then break
				if (current - fileData > batchSize - (2 * 1024 * 1024) && fileSize > 0)
					break;

				phstPrimaryBeamIn[ beam ][ (channel * pSize * pSize) + (j * pSize) + i ].x = strtod( current, &next );
				if (current == next)
				{
					printf( "failed to read real numeric value: fileSize remaining %li, batchSize %li, bytesProcessed (total) %li, bytesProcessed (this batch) %li, channel %i, i %i, j %i\n", fileSize, batchSize, bytesProcessed + current - fileData, current - fileData, channel, i, j );
					printf( "        >" );
					for ( int c = 0; c < 100; c++ )
						printf( "%c", *(current + c) );
					printf( "\n        >" );
					for ( int c = 0; c < 100; c++ )
						printf( "%i:%d,", c, *(current + c) );
					printf( "\n" );
					abort();
				}
				current = next;
				phstPrimaryBeamIn[ beam ][ (channel * pSize * pSize) + (j * pSize) + i ].y = strtod( current, &next );
				if (current == next)
				{
					printf( "failed to read imaginary numeric value: fileSize remaining %li, batchSize %li, bytesProcessed (total) %li, bytesProcessed (this batch) %li, channel %i, i %i, j %i\n", fileSize, batchSize, bytesProcessed + current - fileData, current - fileData, channel, i, j );
					printf( "        >" );
					for ( int c = 0; c < 100; c++ )
						printf( "%c", *(current + c) );
					printf( "\n        >" );
					for ( int c = 0; c < 100; c++ )
						printf( "%i:%d,", c, *(current + c) );
					printf( "\n" );
					abort();
				}
				current = next;
				
				// increment i, j and channel.
				i = i + 1;
				if (i == pSize)
				{
					j = j + 1;
					i = 0;
					if (j == pSize)
					{
						channel = channel + 1;
						j = 0;
					}
				}
			
			}
			
			// if we still have data to fetch then get it. move the section of memory we haven't processed yet to the start of the array.
			if (fileSize > 0)
			{
				bytesInArray = batchSize - (current - fileData) - 1;		// the number of bytes that have been read but not processed. we
												// subtract one because the last character in the array is always
												//	ASCII 0 - an end of data marker.
				memmove( fileData, current, bytesInArray );
				bytesProcessed += (current - fileData);
			}

		} // (fileSize > 0)

		// free the memory.
		free( fileData );

		// close the file
		fclose( beamFile );
	
	} // LOOP: beam
	printf( "\n\n" );
	
	// turn off leakage correction if we didn't find XY or YX.
	if (_param->AProjection == true && _param->LeakageCorrection == true && (stokesFound[ 1 ] == false || stokesFound[ 2 ] == false))
		_param->LeakageCorrection = false;
		
	// display an error if we didn't find XX or YY.
	bool success = true;
	if (stokesFound[ 0 ] == false || stokesFound[ 3 ] == false)
	{
		printf( "WARNING: the primary beam is missing either XX or YY. We will generate a beam instead\n\n" );
		success = false;
	}
	
	// return something.
	return success;

} // Data::loadPrimaryBeam

//
//	mergeData()
//
//	CJS: 22/07/2020
//
//	Merge two data caches together.
//

void Data::mergeData( int pStageID_one, int pStageID_two, bool pLoadAllData, int pWhatData )
{

	// load the data from stage two into the start of the array.
	if (pLoadAllData == true)
		UncacheData(	/* pBatchID = */ pStageID_two,
				/* pTaylorTerm = */ -1,
				/* pOffset = */ 0,
				/* pWhatData = */ pWhatData,
				/* pStokes = */ -1 );

	// load the records from stage one into the end of the arrays.
	UncacheData(	/* pBatchID = */ pStageID_one,
			/* pTaylorTerm = */ -1,
			/* pOffset = */ NumVisibilities[ pStageID_two ],
			/* pWhatData = */ pWhatData,
			/* pStokes = */ -1 );

	// update the number of visibilities.
	NumVisibilities[ pStageID_one ] += NumVisibilities[ pStageID_two ];

	// if this is not the last stage then rename the subsequent files.
	if (pStageID_two < Stages - 1)
		for ( int stageID = pStageID_two; stageID < Stages - 1; stageID++ )
		{

			// build filename.
			char filenameOld[ 255 ], filenameNew[ 255 ];
			if (_param->CacheLocation[0] != '\0')
			{
				sprintf( filenameNew, "%s%s-%i-%i-cache.dat", _param->CacheLocation, _param->OutputPrefix, _mosaicID, stageID );
				sprintf( filenameOld, "%s%s-%i-%i-cache.dat", _param->CacheLocation, _param->OutputPrefix, _mosaicID, stageID + 1 );
			}
			else
			{
				sprintf( filenameNew, "%s-%i-%i-cache.dat", _param->OutputPrefix, _mosaicID, stageID );
				sprintf( filenameOld, "%s-%i-%i-cache.dat", _param->OutputPrefix, _mosaicID, stageID + 1 );
			}

			// rename file.
			rename( filenameOld, filenameNew );

			// update number of visibilities.
			NumVisibilities[ stageID ] = NumVisibilities[ stageID + 1 ];

		}
	else
	{

		// build filename.
		char filename[ 255 ];
		if (_param->CacheLocation[0] != '\0')
			sprintf( filename, "%s%s-%i-%i-cache.dat", _param->CacheLocation, _param->OutputPrefix, _mosaicID, pStageID_two );
		else
			sprintf( filename, "%s-%i-%i-cache.dat", _param->OutputPrefix, _mosaicID, pStageID_two );

		// remove file.
		remove( filename );

	}

	// update the number of stages and reallocate the array..
	Stages -= 1;
	NumVisibilities = (long int *) realloc( NumVisibilities, Stages * sizeof( long int * ) );

} // Data::mergeData

//
//	outerProductImageMatrix()
//
//	CJS: 01/11/2021
//
//	Construct the outer product on an NxN matrix.
//

cufftDoubleComplex ** Data::outerProductImageMatrix
			(
			cufftComplex ** pOne,				// complex image 1
			cufftComplex ** pTwo,				// complex image 2
			int pMatrixSize1,				// the matrix size for matrix 1, i.e. NxN
			int pMatrixSize2,				// the matrix size for matrix 2, i.e. NxN
			int pImageSize					// the size of the image held in each matrix cell
			)
{

	cufftDoubleComplex ** devReturnMatrix = (cufftDoubleComplex **) malloc( pMatrixSize1 * pMatrixSize2 * pMatrixSize1 * pMatrixSize2 * sizeof( cufftDoubleComplex * ) );
		
	// and set the thread block size for processing images on the device.
	setThreadBlockSize2D( pImageSize, pImageSize, _gridSize2D, _blockSize2D );
	
	// populate each cell of the matrix.
	for ( int iOne = 0; iOne < pMatrixSize1; iOne++ )
		for ( int iTwo = 0; iTwo < pMatrixSize2; iTwo++ )
			for ( int jOne = 0; jOne < pMatrixSize1; jOne++ )
				for ( int jTwo = 0; jTwo < pMatrixSize2; jTwo++ )
				{				
					int cell = (((jOne * pMatrixSize2) + jTwo) * pMatrixSize1 * pMatrixSize2) + (iOne * pMatrixSize2) + iTwo;
					if (pOne[ (jOne * pMatrixSize1) + iOne ] != NULL && pTwo[ (jTwo * pMatrixSize2) + iTwo ] != NULL)
					{
					
						reserveGPUMemory( (void **) &devReturnMatrix[ cell ], pImageSize * pImageSize * sizeof( cufftDoubleComplex ),
									"reserving device memory for the outer product matrix", __LINE__ );
									
						// copy across from image 1, converting from single to double precision.
						devMoveImages<<< _gridSize2D, _blockSize2D >>>(	/* pOne = */ devReturnMatrix[ cell ],
													/* pTwo = */ pOne[ (jOne * pMatrixSize1) + iOne ],
													/* pSize = */ pImageSize );
								
						// multiply by image 2.
						devMultiplyImages<<< _gridSize2D, _blockSize2D >>>(	/* pOne = */ devReturnMatrix[ cell ],
													/* pTwo = */ pTwo[ (jTwo * pMatrixSize2) + iTwo ],
													/* pMask = */ NULL,
													/* pSizeOne = */ pImageSize,
													/* pSizeTwo = */ pImageSize );
						
						
					}
					else
						devReturnMatrix[ cell ] = NULL;
						
				}

	// return something.
	return devReturnMatrix;

} // outerProductImageMatrix

//
//	parseChannelRange()
//
//	CJS: 26/11/2019
//
//	Parses a range of channels and works out which channels should be included.
//

void Data::parseChannelRange( char * pChannelRange, int pNumChannels, bool * phstSpwChannelFlag )
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

} // Data::parseChannelRange

//
//	parseSpwSpecifier()
//
//	CJS: 26/11/2019
//
//	Parses a single SPW specifier and works out which channels should be included.
//

void Data::parseSpwSpecifier( char * pSpwSpecifier, int pNumSpws, int * phstNumChannels, bool ** phstSpwChannelFlag )
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
			parseChannelRange(	/* pChannelRange = */ channelRange,
						/* pNumChannels = */ phstNumChannels[ spw ],
						/* phstSpwChannelFlag = */ phstSpwChannelFlag[ spw ] );

		}

	}

} // Data::parseSpwSpecifier

//
//	quickSortData()
//
//	CJS: 15/10/2018
//
//	Sort a list of kernel indexes and grid positions, and swap the corresponding visibilities (if required).
//

void Data::quickSortData( long int pLeft, long int pRight )
{

	if (pLeft <= pRight)
	{
	
		long int i = pLeft, j = pRight;

		VectorI pivot = GridPosition[ (pLeft + pRight) / 2 ];
		int pivotKernel = KernelIndex[ (pLeft + pRight) / 2 ];
		int pivotFieldID = -1;
		if (FieldID != NULL)
			pivotFieldID = FieldID[ (pLeft + pRight) / 2 ];

		// partition, and sort by field ID (if we're preserving it), W plane, A plane, V position, U position, and kernel index.
		while (i <= j)
		{

			while (true)
			{
				bool ok = false;
				if (FieldID != NULL)
				{
					if (FieldID[ i ] > pivotFieldID)
						break;
					if (FieldID[ i ] < pivotFieldID)
						ok = true;
				}
				if (ok == false)
				{

					if (GridPosition[ i ].w > pivot.w)
						break;
					if (GridPosition[ i ].w == pivot.w)
					{

						if (GridPosition[ i ].v > pivot.v)
							break;
						if (GridPosition[ i ].v == pivot.v)
						{

							if (GridPosition[ i ].u > pivot.u)
								break;
							if (GridPosition[ i ].u == pivot.u)
							{

								if (KernelIndex[ i ] > pivotKernel)
									break;
								if (KernelIndex[ i ] == pivotKernel)
									break;
							}
						}
					}
				}
				i = i + 1;
			}

			while (true)
			{
				bool ok = false;
				if (FieldID != NULL)
				{

					if (FieldID[ j ] < pivotFieldID)
						break;
					if (FieldID[ j ] > pivotFieldID)
						ok = true;
				}
				if (ok == false)
				{

					if (GridPosition[ j ].w < pivot.w)
						break;
					if (GridPosition[ j ].w == pivot.w)
					{

						if (GridPosition[ j ].v < pivot.v)
							break;
						if (GridPosition[ j ].v == pivot.v)
						{

							if (GridPosition[ j ].u < pivot.u)
								break;
							if (GridPosition[ j ].u == pivot.u)
							{

								if (KernelIndex[ j ] < pivotKernel)
									break;
								if (KernelIndex[ j ] == pivotKernel)
									break;
							}
						}
					}
				}
				j = j - 1;
			}

			if (i <= j)
			{

				// swap the grid positions, kernel indexes, visibilities, flags, field IDs, densities, weights, and spectral beam weights.
				swap( GridPosition[ i ], GridPosition[ j ] );
				swap( KernelIndex[ i ], KernelIndex[ j ] );
				for ( int s = 0; s < _stokesImages; s++ )
					for ( int t = 0; t < _taylorTerms; t++ )
						swap( Visibility[ s ][ t ][ i ], Visibility[ s ][ t ][ j ] );
				if (Flag != NULL)
					swap( Flag[ i ], Flag[ j ] );
				if (FieldID != NULL)
					swap( FieldID[ i ], FieldID[ j ] );
				if (_param->Weighting != NONE)
					for ( int s = 0; s < _stokesImages; s++ )
						swap( Weight[ s ][ i ], Weight[ s ][ j ] );
				if (_param->Deconvolver == MFS)
					for ( int s = 0; s < _stokesImages; s++ )
						for ( int t = 0; t < (_taylorTerms - 1) * 2; t++ )
							swap( MfsWeight[ s ][ t ][ i ], MfsWeight[ s ][ t ][ j ] );
				swap( DensityMap[ i ], DensityMap[ j ] );

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
	
} // Data::quickSortData

//
//	quickSortFieldIDs()
//
//	CJS: 19/11/2018
//
//	Sort a list of field IDs.
//

void Data::quickSortFieldIDs( int * pFieldID, int pLeft, int pRight )
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
	
} // Data::quickSortFieldIDs

//
//	reprojectPrimaryBeams()
//
//	CJS: 31/08/2021
//
//	Reprojects the primary beam into the output frame, the mosaic-component frame (if required), and also for A-projection (if required).
//

void Data::reprojectPrimaryBeams
			(
			int pBeamOutSize,
			double pBeamOutCellSize,			// the cell size of the image in arcseconds
			double pOutputRA,				// mosaic/image output RA phase position
			double pOutputDEC,				// mosaic/image output DEC phase position
			bool pImagePlaneMosaic,			// true if this is an image-plane mosaic
			double pMaxWavelength				// the maximum wavelength in this measurement set
			)
{
								
	// reproject primary beam at average wavelength to correct position.		
	PrimaryBeam = ReprojectPrimaryBeam(	/* pBeamOutSize = */ pBeamOutSize,
						/* pBeamOutCellSize = */ pBeamOutCellSize,
						/* pToRA = */ pOutputRA,
						/* pToDEC = */ pOutputDEC,
						/* pToWavelength = */ AverageWavelength );
							
	// reproject primary beam at maximum wavelength to correct position.		
	float * tmpPrimaryBeamMaxWavelength = ReprojectPrimaryBeam(	/* pBeamOutSize = */ pBeamOutSize,
									/* pBeamOutCellSize = */ pBeamOutCellSize,
									/* pToRA = */ pOutputRA,
									/* pToDEC = */ pOutputDEC,
									/* pToWavelength = */ pMaxWavelength );

	// construct the primary beam ratio.
	PrimaryBeamRatio = (float *) malloc( pBeamOutSize * pBeamOutSize * sizeof( float ) );
			
	// divide the primary beam by the primary beam at the maximum wavelength, because it the ratio that we need to use.
	for ( int i = 0; i < pBeamOutSize * pBeamOutSize; i++ )
	{
		if (tmpPrimaryBeamMaxWavelength[ i ] != 0.0)
			PrimaryBeamRatio[ i ] = PrimaryBeam[ i ] / tmpPrimaryBeamMaxWavelength[ i ];
		else
			PrimaryBeamRatio[ i ] = 1.0;
		PrimaryBeamRatio[ i ] = pow( pow( 0.2, 4 ) + pow( PrimaryBeamRatio[ i ], 4 ), 0.25 );
	}
			
	if (tmpPrimaryBeamMaxWavelength != NULL)
		free( (void *) tmpPrimaryBeamMaxWavelength );
						
	// if we've doing an image-plane mosaic then reproject the primary beam to the phase position in the frame of each mosaic component.
	if (pImagePlaneMosaic == true)		
		PrimaryBeamInFrame = ReprojectPrimaryBeam(	/* pBeamOutSize = */ pBeamOutSize,
								/* pBeamOutCellSize = */ pBeamOutCellSize,
								/* pToRA = */ PhaseToRA,
								/* pToDEC = */ PhaseToDEC,
								/* pToWavelength = */ AverageWavelength );

} // reprojectPrimaryBeams

//
//	rotateX(), rotateY()
//
//	CJS: 13/04/2021
//
//	Rotate a vector about the x and y axes respectively.
//

VectorD Data::rotateX( VectorD pIn, double pAngle )
{

	VectorD out = { .u = pIn.u, .v = (pIn.v * cos( pAngle )) + (pIn.w * sin( pAngle )), .w = (pIn.w * cos( pAngle )) - (pIn.v * sin( pAngle )) };

	// return something.
	return out;

} // Data::rotateX

VectorD Data::rotateY( VectorD pIn, double pAngle )
{

	VectorD out = { .u = (pIn.u * cos( pAngle )) + (pIn.w * sin( pAngle )), .v = pIn.v, .w = (pIn.w * cos( pAngle )) - (pIn.u * sin( pAngle )) };

	// return something.
	return out;

} // Data::rotateY

//
//	setSpwAndChannelFlags()
//
//	CJS: 26/11/2019
//
//	Parses the SPW parameter and sets a flag to determine which spws and channels should be included in the image.
//

void Data::setSpwAndChannelFlags( int pNumSpws, int * phstNumChannels, bool *** phstSpwChannelFlag, char * phstSpwRestriction )
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
			parseSpwSpecifier(	/* pSpwSpecifier = */ singleSpw,
						/* pNumSpws = */ pNumSpws,
						/* phstNumChannels = */ phstNumChannels,
						/* phstSpwChannelFlag = */ *phstSpwChannelFlag );

		}

} // Data::setSpwAndChannelFlags

//
//	reprojectImage()
//
//	CJS: 30/11/2018
//
//	Reproject an image to a different phase position.
//

void Data::reprojectImage
			(
			cufftComplex * phstImageIn,			// input image
			cufftComplex * phstImageOut,			// output image
			int pImageInSize,				// input image size in pixels
			int pImageOutSize,				// output image size in pixels
			double pInputCellSize,				// input image cell size in arcseconds
			double pOutputCellSize,			// output image cell size in arcseconds
			double pInRA,					// input RA
			double pInDec,					// input DEC
			double pOutRA,					// output RA
			double pOutDec,				// output DEC
			float * pdevInImage,				// input image on the device
			float * pdevOutImage,				// output image on the device
			Reprojection & pImagePlaneReprojection,	// the reprojection object
			bool pSquareBeam,				// true if we should multiply the beam by its complex conjugate
			bool pVerbose					// display some diagnostics
			)
{

	// set up input pixel vector and CD matrix.
	Reprojection::rpVectD tmpPixelIn = { /* x = */ pImageInSize / 2, /* y = */ pImageInSize / 2, /*z = */ 0 };
	Reprojection::rpMatr2x2 tmpCDIn = {	/* a11 = */ -sin( rad( pInputCellSize / 3600.0 ) ),
						/* a12 = */ 0.0,
						/* a21 = */ 0.0,
						/* a22 = */ sin( rad( pInputCellSize / 3600.0 ) ) };

	// set up output pixel vector and CD matrix.
	Reprojection::rpVectD tmpPixelOut = { /* x = */ pImageOutSize / 2, /* y = */ pImageOutSize / 2, /*z = */ 0 };
	Reprojection::rpMatr2x2 tmpCDOut = {	/* a11 = */ -sin( rad( (pOutputCellSize / 3600.0) ) ),
						/* a12 = */ 0.0,
						/* a21 = */ 0.0,
						/* a22 = */ sin( rad( (pOutputCellSize / 3600.0) ) ) };

	// build input and output size.
	Reprojection::rpVectI inSize = { /* x = */ pImageInSize, /* y = */ pImageInSize };
	Reprojection::rpVectI outSize = { /* x = */ pImageOutSize, /* y = */ pImageOutSize };

	// build beam size.
	Reprojection::rpVectI beamSize = { /* x = */ pImageOutSize, /* y = */ pImageOutSize };

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

	cufftComplex * devWorkspace = NULL;
	reserveGPUMemory( (void **) &devWorkspace, pImageInSize * pImageInSize * sizeof( cufftComplex ), "reserving GPU memory for beam workspace", __LINE__ );

	// copy the image into a temporary work location.
	cudaMemcpy( devWorkspace, phstImageIn, pImageInSize * pImageInSize * sizeof( cufftComplex ), cudaMemcpyHostToDevice );
	
	int items = pImageInSize * pImageInSize;
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

		// rearrange the workspace data so that we have all the real numbers first and the imaginary numbers second.
		devSeparateComplexNumbers<<< blocks, threads >>>(	/* pOut = */ &pdevInImage[ /* CELL = */ i * MAX_THREADS ],
									/* pIn = */ &devWorkspace[ /* CELL = */ i * MAX_THREADS ],
									/* pSize = */ itemsThisStage,
									/* pImaginaryPosition = */ pImageInSize * pImageInSize );

	} // LOOP: i

	// free data.
	if (devWorkspace != NULL)
	{
		cudaFree( (void *) devWorkspace );
		devWorkspace = NULL;
	}

	// clear the output image.
	zeroGPUMemory( (void *) pdevOutImage, pImageOutSize * pImageOutSize * sizeof( cufftComplex ), "zeroing the reprojected output image on the device", __LINE__ );

	// reproject this image.
	pImagePlaneReprojection.ReprojectImage(	/* pdevInImage = */ pdevInImage,
							/* pdevOutImage = */ pdevOutImage,
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

	pImagePlaneReprojection.ReprojectImage(	/* pdevInImage = */ pdevInImage + (pImageInSize * pImageInSize),
							/* pdevOutImage = */ pdevOutImage + (pImageOutSize * pImageOutSize),
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

	reserveGPUMemory( (void **) &devWorkspace, pImageOutSize * pImageOutSize * sizeof( cufftComplex ), "reserving GPU memory for beam workspace", __LINE__ );

	items = pImageOutSize * pImageOutSize;
	stages = items / MAX_THREADS;
	if (items % MAX_THREADS != 0)
		stages++;

	int threads, blocks;
	for ( int i = 0; i < stages; i++ )
	{

		// define the block/thread dimensions.
		int itemsThisStage = items - (i * MAX_THREADS);
		if (itemsThisStage > MAX_THREADS)
			itemsThisStage = MAX_THREADS;
		threads = itemsThisStage;
		setThreadBlockSize1D( &threads, &blocks );

		// rearrange the beams so that we have complex numbers, instead of separate real and imaginary numbers.
		devMergeComplexNumbers<<< blocks, threads >>>(	/* pOut = */ &devWorkspace[ /* CELL = */ i * MAX_THREADS ],
									/* pIn = */ &pdevOutImage[ /* CELL = */ i * MAX_THREADS ],
									/* pSize = */ itemsThisStage,
									/* pImaginaryPosition = */ pImageOutSize * pImageOutSize );

	}

	// should the beam be squared? i.e. multiplied by its own complex conjugate?
	if (pSquareBeam == true)
		for ( int i = 0; i < stages; i++ )
		{

			// define the block/thread dimensions.
			int itemsThisStage = items - (i * MAX_THREADS);
			if (itemsThisStage > MAX_THREADS)
				itemsThisStage = MAX_THREADS;
			threads = itemsThisStage;
			setThreadBlockSize1D( &threads, &blocks );

			// rearrange the beams so that we have complex numbers, instead of separate real and imaginary numbers.
			devMultArrayComplexConjugate<<< blocks, threads >>>(	/* pOut = */ &pdevOutImage[ /* CELL = */ i * MAX_THREADS ],
										/* pIn = */ &devWorkspace[ /* CELL = */ i * MAX_THREADS ],
										/* pSize = */ itemsThisStage );

		}
	else
		cudaMemcpy( pdevOutImage, devWorkspace, pImageOutSize * pImageOutSize * sizeof( cufftComplex ), cudaMemcpyDeviceToDevice );

	// free data.
	if (devWorkspace != NULL)
		cudaFree( (void *) devWorkspace );

	// store the beam on the host.
	if (pSquareBeam == true)
		cudaMemcpy( phstImageOut, pdevOutImage, pImageOutSize * pImageOutSize * sizeof( float ), cudaMemcpyDeviceToHost );
	else
		cudaMemcpy( phstImageOut, pdevOutImage, pImageOutSize * pImageOutSize * sizeof( cufftComplex ), cudaMemcpyDeviceToHost );

	// cut off any value less than 0.1% to zero.
//	if (pSquareBeam == true)
//	{
//		float * tmp = (float *) phstImageOut;
//		for ( int i = 0; i < pImageSize * pImageSize; i++ )
//			if (abs( tmp[ i ] ) < 0.001)
//				tmp[ i ] = 0.0;
//	}
//	else
//		for ( int i = 0; i < pImageSize * pImageSize; i++ )
//			if (abs( phstImageOut[ i ].x ) < 0.001)
//				phstImageOut[ i ].x = 0.0;

	// save the reprojected beam.
//	char beamFilename[100];
//	sprintf( beamFilename, "beam-%i-%f-%f", pBeam, pOutRA, pOutDec );
//	_casacoreInterface.WriteCasaImage( beamFilename, pImageSize, pImageSize, pOutRA, pOutDec,
//						pOutputCellSize * (double) _param->ImageSize / (double) pImageSize, phstImageOut,
//						CONST_C / _hstAverageWavelength[ 0 ], NULL );

	// save the reprojected beam.
//	char beamFilename2[100];
//	sprintf( beamFilename2, "beam-in-%i", pBeam );
//	_casacoreInterface.WriteCasaImage( beamFilename2, pImageSize, pImageSize, 0.0, 0.0,
//						_hstBeamCellSize, phstImageIn, CONST_C / _hstAverageWavelength[ 0 ], NULL );

} // reprojectImage

void Data::reprojectImage
			(
			float * phstImageIn,				// input image
			float * phstImageOut,				// output image
			int pImageInSize,				// input image size in pixels
			int pImageOutSize,				// output image size in pixels
			double pInputCellSize,				// input image cell size in arcseconds
			double pOutputCellSize,			// output image cell size in arcseconds
			double pInRA,					// input RA
			double pInDec,					// input DEC
			double pOutRA,					// output RA
			double pOutDec,				// output DEC
			float * pdevInImage,				// input image on the device
			float * pdevOutImage,				// output image on the device
			Reprojection & pImagePlaneReprojection,		// the reprojection object
			bool pVerbose					// display some diagnostics
			)
{

	// set up input pixel vector and CD matrix.
	Reprojection::rpVectD tmpPixelIn = { /* x = */ pImageInSize / 2, /* y = */ pImageInSize / 2, /*z = */ 0 };
	Reprojection::rpMatr2x2 tmpCDIn = {	/* a11 = */ -sin( rad( pInputCellSize / 3600.0 ) ),
						/* a12 = */ 0.0,
						/* a21 = */ 0.0,
						/* a22 = */ sin( rad( pInputCellSize / 3600.0 ) ) };

	// set up output pixel vector and CD matrix.
	Reprojection::rpVectD tmpPixelOut = { /* x = */ pImageOutSize / 2, /* y = */ pImageOutSize / 2, /*z = */ 0 };
	Reprojection::rpMatr2x2 tmpCDOut = {	/* a11 = */ -sin( rad( (pOutputCellSize / 3600.0) ) ),
						/* a12 = */ 0.0,
						/* a21 = */ 0.0,
						/* a22 = */ sin( rad( (pOutputCellSize / 3600.0) ) ) };

	// build input and output size.
	Reprojection::rpVectI inSize = { /* x = */ pImageInSize, /* y = */ pImageInSize };
	Reprojection::rpVectI outSize = { /* x = */ pImageOutSize, /* y = */ pImageOutSize };

	// build beam size.
	Reprojection::rpVectI beamSize = { /* x = */ pImageOutSize, /* y = */ pImageOutSize };

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

	// copy the image into a temporary work location.
	cudaMemcpy( pdevInImage, phstImageIn, pImageInSize * pImageInSize * sizeof( float ), cudaMemcpyHostToDevice );
	zeroGPUMemory( (void *) pdevOutImage, pImageOutSize * pImageOutSize * sizeof( float ), "zeroing the reprojected output image on the device", __LINE__ );

	// reproject this image.
	pImagePlaneReprojection.ReprojectImage(	/* pdevInImage = */ pdevInImage,
							/* pdevOutImage = */ pdevOutImage,
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
	cudaMemcpy( phstImageOut, pdevOutImage, pImageOutSize * pImageOutSize * sizeof( float ), cudaMemcpyDeviceToHost );

} // reprojectImage

//
//	separateFields()
//
//	CJS: 25/08/2021
//
//	Separate the fields in these data into separate mosaic components.
//

void Data::separateFields( int pNumFields, int pStageID )
{

	// if we've got data from multiple fields then we need to move it into separate mosaic components.
	for ( int field = pNumFields - 1; field > 0; field-- )
	{

		// get pointer to this field.
		Data * fieldData = this;
		for ( int i = 0; i < field; i++ )
			fieldData = fieldData->NextComponent;

		// loop through the data, counting the number of records with this field, and noting the start point.
		int startID = -1, records = 0;
		for ( int visibilityIndex = 0; visibilityIndex < NumVisibilities[ pStageID ]; visibilityIndex++ )
			if (FieldID[ visibilityIndex ] == field)
			{
				if (startID == -1)
					startID = visibilityIndex;
				records++;
			}

		fieldData->NumVisibilities[ pStageID ] = records;
		if (records > 0)
		{

			// create the arrays for the new field.
			for ( int stokes = 0; stokes < _stokesImages; stokes++ )
			{
				for ( int taylorTerm = 0; taylorTerm < _taylorTerms; taylorTerm++ )
					fieldData->Visibility[ stokes ][ taylorTerm ] = (cufftComplex *) malloc( fieldData->NumVisibilities[ pStageID ] *
															sizeof( cufftComplex ) );
				fieldData->Weight[ stokes ] = (float *) malloc( fieldData->NumVisibilities[ pStageID ] * sizeof( float ) );
				if (_param->Deconvolver == MFS)
					for ( int taylorTerm = 0; taylorTerm < (_taylorTerms - 1) * 2; taylorTerm++ )
						fieldData->MfsWeight[ stokes ][ taylorTerm ] = (float *) malloc( fieldData->NumVisibilities[ pStageID ] * sizeof( float ) );
			}
			fieldData->GridPosition = (VectorI *) malloc( fieldData->NumVisibilities[ pStageID ] * sizeof( VectorI ) );
			fieldData->KernelIndex = (int *) malloc( fieldData->NumVisibilities[ pStageID ] * sizeof( int ) );
			fieldData->DensityMap = (int *) malloc( fieldData->NumVisibilities[ pStageID ] * sizeof( int ) );

			// move the data from the old field.
			for ( int stokes = 0; stokes < _stokesImages; stokes++ )
			{
				for ( int taylorTerm = 0; taylorTerm < _taylorTerms; taylorTerm++ )
					memcpy( fieldData->Visibility[ stokes ][ taylorTerm ], &Visibility[ stokes ][ taylorTerm ][ startID ],
							records * sizeof( cufftComplex ) );
				memcpy( fieldData->Weight[ stokes ], &Weight[ stokes ][ startID ], records * sizeof( float ) );
				if (_param->Deconvolver == MFS)
					for ( int taylorTerm = 0; taylorTerm < (_taylorTerms - 1) * 2; taylorTerm++ )
						memcpy( fieldData->MfsWeight[ stokes ][ taylorTerm ], &MfsWeight[ stokes ][ taylorTerm ][ startID ],
																		records * sizeof( float ) );
			}
			memcpy( fieldData->GridPosition, &GridPosition[ startID ], records * sizeof( VectorI ) );
			memcpy( fieldData->KernelIndex, &KernelIndex[ startID ], records * sizeof( int ) );
			memcpy( fieldData->DensityMap, &DensityMap[ startID ], records * sizeof( int ) );

			// shrink the size of the old array.
			NumVisibilities[ pStageID ] -= records;
			for ( int stokes = 0; stokes < _stokesImages; stokes++ )
			{
				for ( int taylorTerm = 0; taylorTerm < _taylorTerms; taylorTerm++ )
					Visibility[ stokes ][ taylorTerm ] = (cufftComplex *) realloc( Visibility[ stokes ][ taylorTerm ],
														NumVisibilities[ pStageID ] * sizeof( cufftComplex ) );
				Weight[ stokes ] = (float *) realloc( Weight[ stokes ], NumVisibilities[ pStageID ] * sizeof( float ) );
				if (_param->Deconvolver == MFS)
					for ( int taylorTerm = 0; taylorTerm < (_taylorTerms - 1) * 2; taylorTerm++ )
						MfsWeight[ stokes ][ taylorTerm ] = (float *) realloc( MfsWeight[ stokes ][ taylorTerm ],
															NumVisibilities[ pStageID ] * sizeof( float ) );
			}
			GridPosition = (VectorI *) realloc( GridPosition, NumVisibilities[ pStageID ] * sizeof( VectorI ) );
			KernelIndex = (int *) realloc( KernelIndex, NumVisibilities[ pStageID ] * sizeof( int ) );
			DensityMap = (int *) realloc( DensityMap, NumVisibilities[ pStageID ] * sizeof( int ) );

		} // (records > 0)

	} // LOOP :field

} // separateFields

