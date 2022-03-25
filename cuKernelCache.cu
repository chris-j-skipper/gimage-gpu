
// include the header file.
#include "cuKernelCache.h"

using namespace std;

//
//	N O N   C L A S S   M E M B E R S
//

//
//	P U B L I C   C L A S S   M E M B E R S
//

//
//	KernelCache::KernelCache()
//
//	CJS: 23/02/2022
//
//	The constructor.
//

KernelCache::KernelCache()
{

	// create instances of casacore
	_casacoreInterface = CasacoreInterface::getInstance();

	// create instances of casacore
	_param = Parameters::getInstance();

	pbChannels = 1;
	wPlanes = 1;
	wProjection = false;
	aProjection = false;
	_useMosaicing = false;
	_uvPlaneMosaic = false;
	_data = NULL;
	_beamSize = 128;
	stokesProducts = 1;
	_stokes = STOKES_NONE;
	stokesFlag = NULL;
	oversample = 1;
	gridDegrid = GRID;
	
	_devPrimaryBeamMaxWavelength = NULL;
	_hstPrimaryBeamMosaicing = NULL;

} // KernelCache::KernelCache

//
//	KernelCache::~KernelCache()
//
//	CJS: 23/02/2022
//
//	The destructor.
//

KernelCache::~KernelCache()
{

	if (_devPrimaryBeamMaxWavelength != NULL)
		cudaFree( (void *) _devPrimaryBeamMaxWavelength );
	if (_hstPrimaryBeamMosaicing != NULL)
		free( (void *) _hstPrimaryBeamMosaicing );
	if (stokesFlag != NULL)
		free( (void *) stokesFlag );

} // KernelCache::~KernelCache

//
//	KernelCache::operator()
//
//	CJS: 28/02/2022
//
//	Overload the () operator can access kernel sets by indexing the stokes parameter, channel and w-plane. The first
//	function is the mutator (i.e. updates the array) and the second instance is the accessor (reads from the array).
//

KernelSet & KernelCache::operator()( int pPBChannel, int pStokes, int pWPlane )
{
	
	return kernelSet[ pPBChannel ][ pStokes ][ pWPlane ];

} // KernelCache::operator()

const KernelSet & KernelCache::operator()( int pPBChannel, int pStokes, int pWPlane ) const
{

	return kernelSet[ pPBChannel ][ pStokes ][ pWPlane ];

} // KernelCache::operator()

//
//	CountVisibilities()
//
//	CJS: 22/06/2020
//
//	Calculate the number of visibilities in each stage, batch, and GPU.
//

void KernelCache::CountVisibilities( Data * pData, int pMaxBatchSize, int pNumGPUs )
{

	// store the data object, and count the visibilities.
	_data = pData;
	CountVisibilities( 	/* pMaxBatchSize = */ pMaxBatchSize,
				/* pNumGPUs = */ pNumGPUs );

} // KernelCache::CountVisibilities

void KernelCache::CountVisibilities( int pMaxBatchSize, int pNumGPUs )
{

	// create required arrays for all the kernel sets.
	for ( int stokes = 0; stokes < stokesProducts; stokes++ )
		for ( int pbChannel = 0; pbChannel < pbChannels; pbChannel++ )
			for ( int wPlane = 0; wPlane < wPlanes; wPlane++ )
				kernelSet[ pbChannel ][ stokes ][ wPlane ].createArrays(	/* pNumberOfStages = */ _data->Stages,
												/* pNumberOfBatches = */ _data->Batches,
												/* pNumGPUs = */ pNumGPUs );

	// store how many visibilities we're giving to each GPU, and default to zero.
	int * hstGPUVisibilities = (int *) malloc( pNumGPUs * sizeof( int ) );

	// loop over each stage.
	for ( int stageID = 0; stageID < _data->Stages; stageID++ )
	{

		int batchID = 0;
		for ( int gpu = 0; gpu < pNumGPUs; gpu++ )
			hstGPUVisibilities[ gpu ] = 0;

		// initialise the kernel set to that of the first visibility.
		int thisKernelSet = -1;
		if (_data->NumVisibilities[ stageID ] > 0)
			thisKernelSet = _data->GridPosition[ /* VIS = */ 0 ].w;

		int currentGPU = 0;
		bool gpusFull = false;

		// get the w-plane and channel.
		int wPlane = thisKernelSet / _data->PBChannels;
		int pbChannel = thisKernelSet % _data->PBChannels;
		if (pbChannels == 1)
			pbChannel = 0;

		for ( int visibility = 0; visibility < _data->NumVisibilities[ stageID ]; visibility++ )
		{

			// if we reach a new kernel set then we need to switch GPUs.
			if (_data->GridPosition[ visibility ].w > thisKernelSet)
			{

				// find a GPU that still has space on it.
				currentGPU++;
				if (currentGPU == pNumGPUs)
					currentGPU = 0;
				while (hstGPUVisibilities[ currentGPU ] == pMaxBatchSize)
				{
					currentGPU++;
					if (currentGPU == pNumGPUs)
						currentGPU = 0;
				}
				thisKernelSet = _data->GridPosition[ visibility ].w;
		
				// get the w-plane and channel.
				wPlane = thisKernelSet / _data->PBChannels;
				pbChannel = thisKernelSet % _data->PBChannels;
				if (pbChannels == 1)
					pbChannel = 0;

			}

			// increment the visibility count.
			kernelSet[ pbChannel ][ /* STOKES = */ 0 ][ wPlane ].visibilities[ stageID ][ batchID ][ currentGPU ]++;
			hstGPUVisibilities[ currentGPU ]++;

			// have the number of visibilities for this gpu reached the batch size? if so, find a gpu with some space.
			if (hstGPUVisibilities[ currentGPU ] == pMaxBatchSize)
			{
				gpusFull = true;
				int nextGPU = currentGPU + 1;
				if (nextGPU == pNumGPUs)
					nextGPU = 0;
				while (nextGPU != currentGPU)
				{
					if (hstGPUVisibilities[ nextGPU ] < pMaxBatchSize)
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

			// if the number of visibilities for all gpus have reached the batch size then we need to use the next batch.
			if (gpusFull == true && visibility < _data->NumVisibilities[ stageID ] - 1)
			{

				// use the next batch.
				batchID++;

				// each batch starts with gpu 0.
				currentGPU = 0;

				// reset count of visibilities per gpu.
				for ( int gpu = 0; gpu < pNumGPUs; gpu++ )
					hstGPUVisibilities[ gpu ] = 0;
				gpusFull = false;

			}

		} // LOOP: visibility

	} // LOOP: stageID

	// copy the number of visibilities to all the other Stokes products.
	if (stokesProducts > 1)
		for ( int stokes = 1; stokes < stokesProducts; stokes++ )
			for ( int pbChannel = 0; pbChannel < pbChannels; pbChannel++ )
				for ( int wPlane = 0; wPlane < wPlanes; wPlane++ )
				{
					KernelSet & fromKernelSet = kernelSet[ pbChannel ][ /* STOKES = */ 0 ][ wPlane ];
					KernelSet & toKernelSet = kernelSet[ pbChannel ][ stokes ][ wPlane ];
					for ( int stageID = 0; stageID < _data->Stages; stageID++ )
						for ( int batchID = 0; batchID < _data->Batches[ stageID ]; batchID++ )
							for ( int gpu = 0; gpu < pNumGPUs; gpu++ )
								toKernelSet.visibilities[ stageID ][ batchID ][ gpu ] =
														fromKernelSet.visibilities[ stageID ][ batchID ][ gpu ];
				}

	// free memory.
	if (hstGPUVisibilities != NULL)
		free( (void *) hstGPUVisibilities );

} // KernelCache::CountVisibilities

//
//	KernelCache::Create()
//
//	CJS: 24/02/2022
//
//	Create a new kernel cache.
//

void KernelCache::Create
			(
			int pPBChannels,
			int pWPlanes,
			bool pWProjection,
			bool pAProjection,
			bool pUseMosaicing,
			bool pUVPlaneMosaic,
			Data * pData,
			int pBeamSize,
			int pStokesProducts,
			int pStokes,
			int pOversample,
			griddegrid pGridDegrid,
			float * phstPrimaryBeamAtMaxWavelength,
			float * phstPrimaryBeamMosaicing
			)
{

	// initialise variables.
	pbChannels = pPBChannels;
	wPlanes = pWPlanes;
	wProjection = pWProjection;
	aProjection = pAProjection;
	_useMosaicing = pUseMosaicing;
	_uvPlaneMosaic = pUVPlaneMosaic;
	_data = pData;
	_beamSize = pBeamSize;
	stokesProducts = pStokesProducts;
	_stokes = pStokes;
	stokesFlag = (bool *) malloc( stokesProducts * sizeof( bool ) );
	for ( int s = 0; s < stokesProducts; s++ )
		stokesFlag[ s ] = false;
	oversample = pOversample;
	gridDegrid = pGridDegrid;

	// construct kernel cache.
	KernelSet newKernelSet( /* pOversample = */ oversample );
	for ( int pbChannel = 0; pbChannel < pbChannels; pbChannel++ )
	{
		vector<vector<KernelSet> > tmpChannelVec;
		for ( int stokes = 0; stokes < stokesProducts; stokes++ )
		{
			vector<KernelSet> tmpStokesVec;
			for ( int wPlane = 0; wPlane < wPlanes; wPlane++ )
				tmpStokesVec.push_back( newKernelSet );
			tmpChannelVec.push_back( tmpStokesVec );
		}
		kernelSet.push_back( tmpChannelVec );
	}

	// we will need the primary beam at the maximum wavelength.
	if (aProjection == false && _useMosaicing == false && gridDegrid == DEGRID && phstPrimaryBeamAtMaxWavelength != NULL)
	{
		reserveGPUMemory( (void **) &_devPrimaryBeamMaxWavelength, _beamSize * _beamSize * sizeof( float ),
					"reserving device memory for the primary beam at the max. wavelength", __LINE__ );
		moveHostToDevice( (void *) _devPrimaryBeamMaxWavelength, (void *) phstPrimaryBeamAtMaxWavelength,
					_beamSize * _beamSize * sizeof( float ), "copying primary beam at the max. wavelength to the device", __LINE__ );
	}

	// we will need the primary beam if we're mosiacing.
	if (_useMosaicing == true && phstPrimaryBeamMosaicing != NULL)
	{
		_hstPrimaryBeamMosaicing = (float *) malloc( _beamSize * _beamSize * sizeof( float ) );
		memcpy( (void *) _hstPrimaryBeamMosaicing, (void *) phstPrimaryBeamMosaicing, _beamSize * _beamSize * sizeof( float ) );
	}

} // KernelCache::Create

//
//	KernelCache::GenerateKernelCache()
//
//	CJS: 21/02/2022
//
//	Generates a cache of kernels for gridding and degridding.
//

void KernelCache::GenerateKernelCache( int pPBChannel )
{	

	double total = (double) (pbChannels * stokesProducts);

	for ( int stokes = 0; stokes < stokesProducts; stokes++ )
	{

//		double frac = (double) ((stokes * pbChannels) + pPBChannel + 1) * 100.0 / total;
		double frac = (double) ((pPBChannel * stokesProducts) + stokes + 1) * 100.0 / total;
		printf( "\r        generating kernel cache.....%3d%%", (int) ceil( frac ) );
		fflush( stdout );

		bool processThisStokes = true;

		// we should ignore this Stokes product if we're using A-projection and the required Mueller matrix cell is missing.
		if (aProjection == true && gridDegrid == GRID && _data != NULL)
			if (_data->InverseMuellerMatrixFlag[ (_stokes * 4) + stokes ] == false)
				processThisStokes = false;
		if (aProjection == true && gridDegrid == DEGRID && _data != NULL)
			if (_data->MuellerMatrixFlag[ (stokes * 4) + _stokes ] == false)
				processThisStokes = false;

		if (processThisStokes == true)
		{

			// flag this Stokes product as having a kernel cache.
			stokesFlag[ stokes ] = true;

			// get average weight for kernel normalisation.
			double averageWeight = 1.0;
			if (_param->Weighting != NONE && _data != NULL)
				if (_data->AverageWeight != NULL)
					averageWeight = _data->AverageWeight[ stokes ];

			// get the primary beams for A-projection.
			cufftComplex * primaryBeamAProjection = NULL;
			if (aProjection == true && gridDegrid == GRID && _data != NULL)
				primaryBeamAProjection = _data->InverseMuellerMatrix[ (_stokes * 4) + stokes ];
			if (aProjection == true && gridDegrid == DEGRID && _data != NULL)
				primaryBeamAProjection = _data->MuellerMatrix[ (stokes * 4) + _stokes ];

//if (pPBChannel % 5 == 0)
//{

//float * tmpKernel = (float *) malloc( _beamSize * _beamSize * sizeof( cufftComplex ) );
//memcpy( tmpKernel, primaryBeamAProjection, _beamSize * _beamSize * sizeof( cufftComplex ) );
//for ( int i = 0; i < _beamSize * _beamSize; i++ )
//	tmpKernel[ i ] = tmpKernel[ i * 2 ];
//char kernelFilename[100];
//sprintf( kernelFilename, "a-kernel-%i-%i-real", pPBChannel, (gridDegrid == GRID ? 0 : 1) );
//_casacoreInterface->WriteCasaImage( kernelFilename, _beamSize, _beamSize, 0.0,
//					0.0, _param->CellSize, tmpKernel, CONST_C / _data->AverageWavelength, NULL, CasacoreInterface::J2000, 1 );
//memcpy( tmpKernel, primaryBeamAProjection, _beamSize * _beamSize * sizeof( cufftComplex ) );
//for ( int i = 0; i < _beamSize * _beamSize; i++ )
//	tmpKernel[ i ] = tmpKernel[ (i * 2) + 1 ];
//sprintf( kernelFilename, "a-kernel-%i-%i-imag", pPBChannel, (gridDegrid == GRID ? 0 : 1) );
//_casacoreInterface->WriteCasaImage( kernelFilename, _beamSize, _beamSize, 0.0,
//					0.0, _param->CellSize, tmpKernel, CONST_C / _data->AverageWavelength, NULL, CasacoreInterface::J2000, 1 );
//free( tmpKernel );

//}

			// get the primary beam ratio if needed.
			float * hstPrimaryBeamRatio = NULL;
			if (aProjection == false && _useMosaicing == false && gridDegrid == DEGRID && _data != NULL)
			{

				// convert the Jones matrix to a Mueller matrix.
				cufftDoubleComplex ** devMueller = _data->ConvertJonesToMueller(	/* pJonesMatrix = */ _data->JonesMatrix,
													/* pImageSize = */ _beamSize );
							
				float * devPrimaryBeamRatio = NULL;
				reserveGPUMemory( (void **) &devPrimaryBeamRatio, _beamSize * _beamSize * sizeof( float ),
							"reserving device memory for the primary beam ratio", __LINE__ );

				// define the block/thread dimensions.
				setThreadBlockSize2D( _beamSize, _beamSize, _gridSize2D, _blockSize2D );
															
				// get the required primary beam from the Mueller matrix.
				int cell = 0;
				if (_stokes == STOKES_I) cell = 0;
				if (_stokes == STOKES_Q) cell = 5;
				if (_stokes == STOKES_U) cell = 10;
				if (_stokes == STOKES_V) cell = 15;
				if (devMueller[ cell ] != NULL)
					devConvertImage<<< _gridSize2D, _blockSize2D >>>(	/* pOut = */ devPrimaryBeamRatio,
												/* pIn = */ devMueller[ cell ],
												/* pSize = */ _beamSize );

				// divide the primary beam by the primary beam at the maximum wavelength.
				devDivideImages<<< _gridSize2D, _blockSize2D >>>(	/* pOne = */ devPrimaryBeamRatio,
											/* pTwo = */ _devPrimaryBeamMaxWavelength,
											/* pMask = */ NULL,
											/* pSizeOne = */ _beamSize,
											/* pSizeTwo = */ _beamSize );

				hstPrimaryBeamRatio = (float *) malloc( _beamSize * _beamSize * sizeof( float ) );
				moveDeviceToHost( (void *) hstPrimaryBeamRatio, (void *) devPrimaryBeamRatio, _beamSize * _beamSize * sizeof( float ),
							"coping primary-beam ratio to the device", __LINE__ );
							
				// ensure any large values level off at a value of 5.
				for ( int index = 0; index < _beamSize * _beamSize; index++ )
					hstPrimaryBeamRatio[ index ] = 1.0 / pow( pow( 1.0 / 5.0, 4 ) + pow( 1.0 / hstPrimaryBeamRatio[ index ], 4 ), 0.25 );
							
//if (pPBChannel == 200)
//{
//	free( (void *) _data->PrimaryBeamRatio );
//	_data->PrimaryBeamRatio = (float *) malloc( _beamSize * _beamSize * sizeof( float ) );
//	memcpy( _data->PrimaryBeamRatio, hstPrimaryBeamRatio, _beamSize * _beamSize * sizeof( float ) );
//}

				// free memory.
				if (devPrimaryBeamRatio != NULL)
					cudaFree( (void *) devPrimaryBeamRatio );
				if (devMueller != NULL)
				{
					for ( int cell = 0; cell < 16; cell++ )
						if (devMueller[ cell ] != NULL)
							cudaFree( (void *) devMueller[ cell ] );
					free( (void *) devMueller );
				}

			} // build primary-beam ratio

			for ( int wPlane = 0; wPlane < wPlanes; wPlane++ )
			{

				bool kernelOverflow = false;

				// generate kernel.
				kernelSet[ pPBChannel ][ stokes ][ wPlane ].generateKernel(	/* pW = */ wPlane,
												/* pChannel = */ pPBChannel,
												/* pWProjection = */ wProjection,
												/* pAProjection = */ aProjection,
												/* phstPrimaryBeamMosaicing = */ _hstPrimaryBeamMosaicing,
												/* phstPrimaryBeamAProjection = */ primaryBeamAProjection,
												/* phstPrimaryBeamRatio = */ hstPrimaryBeamRatio,
												/* pBeamSize = */ _beamSize,
												/* phstData = */ _data,
												/* pGridDegrid = */ gridDegrid,
												/* pKernelOverflow = */ &kernelOverflow,
												/* pAverageWeight = */ (gridDegrid == GRID ? averageWeight : 1.0),
												/* pUVPlaneMosaic = */ _uvPlaneMosaic );

			} // LOOP: wPlane

			// free memory
			if (hstPrimaryBeamRatio != NULL)
				free( (void *) hstPrimaryBeamRatio );

		} // (processThisStokes == true)

	} // LOOP: stokes

} // KernelCache::GenerateKernelCache

//
//	P R I V A T E   C L A S S   M E M B E R S
//

