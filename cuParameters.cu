
// include the header file.
#include "cuParameters.h"

using namespace std;

//Parameters Parameters::_thisInstance;

//
//	C O N S T A N T S
//

	const char Parameters::MEASUREMENT_SET[] = "measurement-set:";
	const char Parameters::FIELD_ID[] = "field-id:";
	const char Parameters::BEAM_ID[] = "beam-id:";
	const char Parameters::SPW[] = "spw:";
	const char Parameters::DATA_FIELD[] = "data-field:";
	const char Parameters::TABLE_DATA[] = "table-data:";
	const char Parameters::OUTPUT_PREFIX[] = "output-prefix:";
	const char Parameters::CELL_SIZE[] = "cell-size:";
	const char Parameters::PIXELS_UV[] = "pixels-uv:";
	const char Parameters::W_PLANES[] = "w-planes:";
	const char Parameters::A_PROJECTION[] = "a-projection:";
	const char Parameters::LEAKAGE_CORRECTION[] = "leakage-correction:";
	const char Parameters::OVERSAMPLE[] = "oversample:";
	const char Parameters::KERNEL_CUTOFF_FRACTION[] = "kernel-cutoff-fraction:";
	const char Parameters::KERNEL_CUTOFF_SUPPORT[] = "kernel-cutoff-support:";
	const char Parameters::DECONVOLVER[] = "deconvolver:";
	const char Parameters::MINOR_CYCLES[] = "minor-cycles:";
	const char Parameters::LOOP_GAIN[] = "loop-gain:";
	const char Parameters::CYCLEFACTOR[] = "cyclefactor:";
	const char Parameters::THRESHOLD[] = "threshold:";
	const char Parameters::OUTPUT_RA[] = "output-ra:";
	const char Parameters::OUTPUT_DEC[] = "output-dec:";
	const char Parameters::WEIGHTING[] = "weighting:";
	const char Parameters::ROBUST_PARAMETER[] = "robust:";
	const char Parameters::PB_CHANNELS[] = "pb-channels:";
	const char Parameters::MOSAIC_DOMAIN[] = "mosaic-domain:";
	const char Parameters::AIRY_DISK_DIAMETER[] = "airy-disk-diameter:";
	const char Parameters::AIRY_DISK_BLOCKAGE[] = "airy-disk-blockage:";
	const char Parameters::BEAM_PATTERN[] = "beam-pattern:";
	const char Parameters::BEAM_SIZE_PIXELS[] = "beam-size:";
	const char Parameters::BEAM_CELL_SIZE[] = "beam-cell-size:";
	const char Parameters::BEAM_TYPE[] = "beam-type:";
	const char Parameters::STOKES[] = "stokes:";
	const char Parameters::TELESCOPE[] = "telescope:";
	const char Parameters::CACHE_LOCATION[] = "cache-location:";
	const char Parameters::GPU_LIST[] = "gpu:";
	
	// filename extensions
	const char Parameters::DIRTY_BEAM_EXTENSION[] = "-dirty-beam";
	const char Parameters::CLEAN_BEAM_EXTENSION[] = "-clean-beam";
	const char Parameters::GRIDDED_EXTENSION[] = "-gridded";
	const char Parameters::DIRTY_IMAGE_EXTENSION[] = "-dirty-image";
	const char Parameters::CLEAN_IMAGE_EXTENSION[] = "-clean-image";
	const char Parameters::RESIDUAL_IMAGE_EXTENSION[] = "-residual-image";
	const char Parameters::DECONVOLUTION_EXTENSION[] = "-deconvolution";
	const char Parameters::PRIMARY_BEAM_EXTENSION[] = "-primary-beam";
	const char Parameters::PRIMARY_BEAM_PATTERN_EXTENSION[] = "-primary-beam-pattern";
	const char Parameters::ALPHA_EXTENSION[] = "-alpha";

//
//	P U B L I C   C L A S S   M E M B E R S
//

//
//	Parameters::Parameters()
//
//	CJS: 18/03/2022
//
//	The constructor.
//

Parameters::Parameters()
{

	Telescope = UNKNOWN_TELESCOPE;
		
	MeasurementSets = 0;			// the number of measurement sets found in the parameter file.
	ImageSize = 0;
	Oversample = 8;
	CellSize = 0;				// the angular size of each output pixel
	UvCellSize = 0;			// in units of lambda
	OutputRA = 0.0;
	OutputDEC = 0.0;
	WProjection = false;
	AProjection = false;
	LeakageCorrection = false;
	WPlanes = 1;
	PBChannels = 1;
	Weighting = NATURAL;
	Robust = 0.0;
	AiryDiskDiameter = 25.0;		// the diameter of the Airy disk.
	AiryDiskBlockage = 0.0;		// the width of the blockage at the centre of the Airy disk.
	DiskDiameterSupplied = false;
	DiskBlockageSupplied = false;
	CacheData = false;			// we set this flag to true if we need to cache and uncache our data.
	TaylorTerms = 1;
	
	// gpu parameters.
	NumGPUs = 0;
	GPU = NULL;
	GPUParam[ 0 ] = '\0';

	// data parameters.
	strcpy( OutputPrefix, "output" );
	CacheLocation[ 0 ] = '\0';
	FieldID = NULL;
	DataField = NULL;
	SpwRestriction = NULL;
	MeasurementSetPath = NULL;
	TableData = NULL;

	// primary beams.
	BeamSize = -1;
	BeamInSize = -1;
	BeamInCellSize = -1.0;
	BeamPattern = NULL;			// the file or files with the primary beam patterns.
	BeamType = AIRY;			// the primary beam type - can be AIRY, GAUSSIAN, or FROMFILE if it's being loaded.
	BeamID = NULL;				// the ASKAP PAF beam ID for each measurement set.
	
	// dirty beams.
	PsfSize = 0;

	// kernel parameters.
	KernelCutoffFraction = 0.01;
	KernelCutoffSupport = 250;
	
	// hogbom parameters.
	Deconvolver = HOGBOM;
	MinorCycles = 10;
	LoopGain = 0.1;
	CycleFactor = 1.5;
	Threshold = 0.0;
	
	// stokes parameters.
	Stokes = STOKES_I;
	NumStokesImages = 1;

	// mosaic?
	UseMosaicing = false;
	ImagePlaneMosaic = false;
	UvPlaneMosaic = false;
	MosaicDomain = IMAGE;

} // Parameters::Parameters

//
//	Parameters::~Parameters()
//
//	CJS: 18/03/2022
//
//	The destructor.
//

Parameters::~Parameters()
{

	// free data.
	if (FieldID != NULL)
	{
		for ( int i = 0; i < MeasurementSets; i++ )
			if (FieldID[ i ] != NULL)
				free( (void *) FieldID[ i ] );
		free( (void *) FieldID );
	}
	if (DataField != NULL)
	{
		for ( int i = 0; i < MeasurementSets; i++ )
			if (DataField[ i ] != NULL)
				free( (void *) DataField[ i ] );
		free( (void *) DataField );
	}
	if (SpwRestriction != NULL)
	{
		for ( int i = 0; i < MeasurementSets; i++ )
			if (SpwRestriction[ i ] != NULL)
				free( (void *) SpwRestriction[ i ] );
		free( (void *) SpwRestriction );
	}
	if (MeasurementSetPath != NULL)
	{
		for ( int i = 0; i < MeasurementSets; i++ )
			if (MeasurementSetPath[ i ] != NULL)
				free( (void *) MeasurementSetPath[ i ] );
		free( (void *) MeasurementSetPath );
	}
	if (TableData != NULL)
	{
		for ( int i = 0; i < MeasurementSets; i++ )
			if (TableData[ i ] != NULL)
				free( (void *) TableData[ i ] );
		free( (void *) TableData );
	}
	if (BeamID != NULL)
		free( (void *) BeamID );
	if (BeamPattern != NULL)
	{
		for ( int i = 0; i < MeasurementSets; i++ )
			if (BeamPattern[ i ] != NULL)
				free( (void *) BeamPattern[ i ] );
		free( (void *) BeamPattern );
	}
	if (GPU != NULL)
		free( (void *) GPU );

} // Parameters::~Parameters

Parameters * Parameters::_thisInstance = NULL;

//
//	Parameters::getInstance()
//
//	CJS: 18/03/2022
//
//	Get the instance of the class so that only one instance will exist.
//

Parameters * Parameters::getInstance()
{

	// return a pointer to this object.
//	return &_thisInstance;

	if (_thisInstance == NULL)
		_thisInstance = new Parameters();
	return _thisInstance;

} // Parameters::getInstance

//
//	GetParameters()
//
//	CJS: 07/08/2015
//
//	Load the following parameters from the parameter file gridder-params: uv cell size, uv grid size, # w-planes, oversample.
//

void Parameters::GetParameters( char * pParameterFile )
{

	char params[1024], line[2048], par[1024];

	// initialise arrays.
	MeasurementSetPath = (char **) malloc( sizeof( char * ) );
	MeasurementSetPath[ 0 ] = NULL;
	BeamID = (int *) malloc( sizeof( int ) );
	BeamID[ 0 ] = -1;
	FieldID = (char **) malloc( sizeof( char * ) );
	FieldID[ 0 ] = NULL;
	SpwRestriction = (char **) malloc( sizeof( char * ) );
	SpwRestriction[ 0 ] = NULL;
	DataField = (char **) malloc( sizeof( char * ) );
	DataField[ 0 ] = NULL;
	TableData = (char **) malloc( sizeof( char * ) );
	TableData[ 0 ] = NULL;
	BeamPattern = (char **) malloc( sizeof( char * ) );
	BeamPattern[ 0 ] = NULL;

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
			MeasurementSets++;
			MeasurementSetPath = (char **) realloc( MeasurementSetPath, MeasurementSets * sizeof( char * ) );
			MeasurementSetPath[ MeasurementSets - 1 ] = (char *) malloc( 1024 * sizeof( char ) );
			strcpy( MeasurementSetPath[ MeasurementSets - 1 ], params );

			// the mosaic, or beam, id. initialised to -1 and will be updated if we specify a wildcard.
			BeamID = (int *) realloc( BeamID, MeasurementSets * sizeof( int ) );
			BeamID[ MeasurementSets - 1 ] = -1;

			// field id.
			FieldID = (char **) realloc( FieldID, MeasurementSets * sizeof( char * ) );
			FieldID[ MeasurementSets - 1 ] = (char *) malloc( 1024 * sizeof( char ) );
			strcpy( FieldID[ MeasurementSets - 1 ], "\0" );

			// spw.
			SpwRestriction = (char **) realloc( SpwRestriction, MeasurementSets * sizeof( char * ) );
			SpwRestriction[ MeasurementSets - 1 ] = (char *) malloc( 1024 * sizeof( char ) );
			strcpy( SpwRestriction[ MeasurementSets - 1 ], "\0" );

			// data field.
			DataField = (char **) realloc( DataField, MeasurementSets * sizeof( char * ) );
			DataField[ MeasurementSets - 1 ] = (char *) malloc( 1024 * sizeof( char ) );
			strcpy( DataField[ MeasurementSets - 1 ], "CORRECTED_DATA" );

			// table data.
			TableData = (char **) realloc( TableData, MeasurementSets * sizeof( char * ) );
			TableData[ MeasurementSets - 1 ] = (char *) malloc( 1024 * sizeof( char ) );
			strcpy( TableData[ MeasurementSets - 1 ], "\0" );

			// beam pattern.
			BeamPattern = (char **) realloc( BeamPattern, MeasurementSets * sizeof( char * ) );
			BeamPattern[ MeasurementSets - 1 ] = (char *) malloc( 1024 * sizeof( char ) );
			strcpy( BeamPattern[ MeasurementSets - 1 ], "\0" );

			minMeasurementSet = MeasurementSets - 1;
			maxMeasurementSet = MeasurementSets - 1;

//			strcpy( MeasurementSetPath, params );
		}
		else if (strcmp( par, FIELD_ID ) == 0 && MeasurementSets > 0)
		{

			// ensure we've read a path name already.
			if (minMeasurementSet >= 0 && maxMeasurementSet >= 0)
				for ( int i = minMeasurementSet; i <= maxMeasurementSet; i++ )
					strcpy( FieldID[ i ], params );

		}
		else if (strcmp( par, SPW ) == 0 && MeasurementSets > 0)
		{

			// ensure we've read a path name already.
			if (minMeasurementSet >= 0 && maxMeasurementSet >= 0)
				for ( int i = minMeasurementSet; i <= maxMeasurementSet; i++ )
					strcpy( SpwRestriction[ i ], params );

		}
		else if (strcmp( par, DATA_FIELD ) == 0 && MeasurementSets > 0)
		{

			// ensure we've read a path name already.
			if (minMeasurementSet >= 0 && maxMeasurementSet >= 0)
				for ( int i = minMeasurementSet; i <= maxMeasurementSet; i++ )
					strcpy( DataField[ i ], params );

		}
		else if (strcmp( par, TABLE_DATA ) == 0 && MeasurementSets > 0)
		{

			// ensure we've read a path name already.
			if (minMeasurementSet >= 0 && maxMeasurementSet >= 0)
				for ( int i = minMeasurementSet; i <= maxMeasurementSet; i++ )
					strcpy( TableData[ i ], params );

		}
		else if (strcmp( par, BEAM_PATTERN ) == 0)
		{

			// ensure we've read a path name already.
			if (minMeasurementSet >= 0 && maxMeasurementSet >= 0)
				for ( int i = minMeasurementSet; i <= maxMeasurementSet; i++ )
					strcpy( BeamPattern[ i ], params );

		}
		else if (strcmp( par, BEAM_ID ) == 0)
		{

			// ensure we've read a path name already.
			if (minMeasurementSet >= 0 && maxMeasurementSet >= 0)
				for ( int i = minMeasurementSet; i <= maxMeasurementSet; i++ )
					BeamID[ i ] = atoi( params );

		}
		else if (strcmp( par, OUTPUT_PREFIX ) == 0)
			strcpy( OutputPrefix, params );
		else if (strcmp( par, CELL_SIZE ) == 0)
			CellSize = atof( params );
		else if (strcmp( par, PIXELS_UV ) == 0)
			ImageSize = atoi( params );
		else if (strcmp( par, W_PLANES ) == 0)
		{
			WPlanes = atoi( params );
			WProjection = (WPlanes > 0);
			if (WProjection == false)
				WPlanes = 1;
		}
		else if (strcmp( par, A_PROJECTION ) == 0)
		{
			if (strcmp( params, "Y" ) == 0 || strcmp( params, "y" ) == 0 || strcmp( params, "YES" ) == 0 || strcmp( params, "yes" ) == 0)
				AProjection = true;
		}
		else if (strcmp( par, LEAKAGE_CORRECTION ) == 0)
		{
			if (strcmp( params, "Y" ) == 0 || strcmp( params, "y" ) == 0 || strcmp( params, "YES" ) == 0 || strcmp( params, "yes" ) == 0)
				LeakageCorrection = true;
		}
		else if (strcmp( par, OVERSAMPLE ) == 0)
			Oversample = atof( params );
		else if (strcmp( par, KERNEL_CUTOFF_FRACTION ) == 0)
			KernelCutoffFraction = atof( params );
		else if (strcmp( par, KERNEL_CUTOFF_SUPPORT ) == 0)
			KernelCutoffSupport = atoi( params );
		else if (strcmp( par, CACHE_LOCATION ) == 0)
			strcpy( CacheLocation, params );
		else if (strcmp( par, DECONVOLVER ) == 0)
		{
			if (strcmp( params, "HOGBOM" ) == 0 || strcmp( params, "hogbom" ) == 0)
				Deconvolver = HOGBOM;
			if (strcmp( params, "MFS" ) == 0 || strcmp( params, "mfs" ) == 0)
			{
				Deconvolver = MFS;
				TaylorTerms = 2;
			}
		}
		else if (strcmp( par, MINOR_CYCLES ) == 0)
			MinorCycles = atoi( params );
		else if (strcmp( par, LOOP_GAIN ) == 0)
			LoopGain = atof( params );
		else if (strcmp( par, CYCLEFACTOR ) == 0)
			CycleFactor = atof( params );
		else if (strcmp( par, THRESHOLD ) == 0)
			Threshold = atof( params );
		else if (strcmp( par, OUTPUT_RA ) == 0)
			OutputRA = atof( params );
		else if (strcmp( par, OUTPUT_DEC ) == 0)
			OutputDEC = atof( params );
		else if (strcmp( par, WEIGHTING ) == 0)
		{
			if (strcmp( params, "NATURAL" ) == 0)
				Weighting = NATURAL;
			if (strcmp( params, "UNIFORM" ) == 0)
				Weighting = UNIFORM;
			if (strcmp( params, "ROBUST" ) == 0)
				Weighting = ROBUST;
		}
		else if (strcmp( par, ROBUST_PARAMETER ) == 0)
			Robust = atof( params );
		else if (strcmp( par, PB_CHANNELS ) == 0)
			PBChannels = atoi( params );
		else if (strcmp( par, MOSAIC_DOMAIN ) == 0)
			MosaicDomain = (strcmp( params, "IMAGE" ) == 0 || strcmp( params, "image" ) == 0 ? IMAGE : UV);
		else if (strcmp( par, AIRY_DISK_DIAMETER ) == 0)
		{
			AiryDiskDiameter = atof( params );
			DiskDiameterSupplied = true;
		}
		else if (strcmp( par, AIRY_DISK_BLOCKAGE ) == 0)
		{
			AiryDiskBlockage = atof( params );
			DiskBlockageSupplied = true;
		}
		else if (strcmp( par, BEAM_SIZE_PIXELS ) == 0)
			BeamInSize = atoi( params );
		else if (strcmp( par, BEAM_CELL_SIZE ) == 0)
			BeamInCellSize = atof( params );
		else if (strcmp( par, BEAM_TYPE ) == 0)
		{
			if (strcmp( params, "AIRY" ) == 0)
				BeamType = AIRY;
			else if (strcmp( params, "GAUSSIAN" ) == 0)
				BeamType = GAUSSIAN;
			else if (strcmp( params, "FROMFILE" ) == 0)
				BeamType = FROMFILE;
		}
		else if (strcmp( par, STOKES ) == 0)
		{
			if (strcmp( params, "I" ) == 0 || strcmp( params, "i" ) == 0)
				Stokes = STOKES_I;
			else if (strcmp( params, "Q" ) == 0 || strcmp( params, "q" ) == 0)
				Stokes = STOKES_Q;
			else if (strcmp( params, "U" ) == 0 || strcmp( params, "u" ) == 0)
				Stokes = STOKES_U;
			else if (strcmp( params, "V" ) == 0 || strcmp( params, "v" ) == 0)
				Stokes = STOKES_V;
		}
		else if (strcmp( par, TELESCOPE ) == 0)
		{
			if (strcmp( params, "ASKAP" ) == 0)
				Telescope = ASKAP;
			else if (strcmp( params, "ALMA" ) == 0)
				Telescope = ALMA;
			else if (strcmp( params, "EMERLIN" ) == 0)
				Telescope = EMERLIN;
			else if (strcmp( params, "VLA" ) == 0)
				Telescope = VLA;
			else if (strcmp( params, "MEERKAT" ) == 0)
				Telescope = MEERKAT;
		}
		else if (strcmp( par, GPU_LIST ) == 0)
			strcpy( GPUParam, params );
            
	}
	fclose( fr );
	
} // Parameters::GetParameters

//
//	P R I V A T E   C L A S S   M E M B E R S
//
