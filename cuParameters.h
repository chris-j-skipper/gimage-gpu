#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <complex>

// include types.
#ifndef Included_Types
#define Included_Types
#include "cuTypes.h"
#endif

using namespace std;

//
//	NON CLASS MEMBERS
//

//
//	Parameters
//
//	CJS: 23/02/2022
//
//	Holds the parameters for the clean
//

class Parameters
{
	
	public:
		
		// default constructor
		Parameters();
   
		// destructor
		~Parameters();

		// get instance.
		static Parameters * getInstance();

//
//		ENUMERATED TYPES
//

//
//		STRUCTURES
//

//
//		CONSTANTS
//

		// the size of the primary beam and deconvolution image.
		static const int BEAM_SIZE = 256;
		static const int PREFERRED_VISIBILITY_BATCH_SIZE = 4000000;
	
		// filename extensions
		static const char DIRTY_BEAM_EXTENSION[];
		static const char CLEAN_BEAM_EXTENSION[];
		static const char GRIDDED_EXTENSION[];
		static const char DIRTY_IMAGE_EXTENSION[];
		static const char CLEAN_IMAGE_EXTENSION[];
		static const char RESIDUAL_IMAGE_EXTENSION[];
		static const char DECONVOLUTION_EXTENSION[];
		static const char PRIMARY_BEAM_EXTENSION[];
		static const char PRIMARY_BEAM_PATTERN_EXTENSION[];
		static const char ALPHA_EXTENSION[];

//
//		GLOBAL VARIABLES
//

		// telescope.
		telescope Telescope;

		int MeasurementSets;			// the number of measurement sets found in the parameter file.
		int ImageSize;
		int Oversample;
		double CellSize;			// the angular size of each output pixel
		double UvCellSize;			// in units of lambda
		double OutputRA;
		double OutputDEC;
		bool WProjection;
		bool AProjection;
		bool LeakageCorrection;
		int WPlanes;
		int PBChannels;
		weighting Weighting;
		double Robust;
		double AiryDiskDiameter;		// the diameter of the Airy disk.
		double AiryDiskBlockage;		// the width of the blockage at the centre of the Airy disk.
		bool DiskDiameterSupplied;
		bool DiskBlockageSupplied;
		bool CacheData;			// we set this flag to true if we need to cache and uncache our data.
		int TaylorTerms;
		
		// gpu parameters.
		int NumGPUs;
		int * GPU;
		char GPUParam[1024];

		// data parameters.
		char OutputPrefix[1024];
		char CacheLocation[1024];
		char ** FieldID;
		char ** DataField;
		char ** SpwRestriction;
		char ** MeasurementSetPath;
		char ** TableData;

		// primary beams.
		int BeamSize;
		int BeamInSize;
		double BeamInCellSize;
		char ** BeamPattern;			// the file or files with the primary beam patterns.
		bool BeamStokes;			// true if the beam patterns are the Stokes beam patterns I, Q, U and V, or false if they're XX, XY, YX, YY.
		beamtype BeamType;			// the primary beam type - can be AIRY, GAUSSIAN, or FROMFILE if it's being loaded.
		int * BeamID;				// the ASKAP PAF beam ID for each measurement set.
		
		// dirty beams
		int PsfSize;

		// kernel parameters.
		double KernelCutoffFraction;
		int KernelCutoffSupport;
	
		// hogbom parameters.
		deconvolver Deconvolver;
		int MinorCycles;
		double LoopGain;
		double CycleFactor;
		double Threshold;

		// stokes parameters.
		//
		// XX = I + Q	I = (XX + YY) / 2	RR = R + V	I = (LL + RR) / 2
		// YY = I - Q	Q = (XX - YY) / 2	LL = I - V	V = (RR - LL) / 2
		// XY = U + iV	U = (XY + YX) / 2	RL = Q + iU	Q = (RL + LR) / 2
		// YX = U - iV	V = i(YX - XY) / 2	LR = Q - iU	U = i(LR - RL) / 2
		//
		// if we are doing both A-projection and cleaning then we need to process ALL Stokes images. otherwise, we just process the one the user asks for.
		int Stokes;
		int NumStokesImages;

		// mosaic?
		bool UseMosaicing;
		bool ImagePlaneMosaic;
		bool UvPlaneMosaic;
		mosaicdomain MosaicDomain;

//
//		FUNCTIONS
//

		void GetParameters( char * pParameterFile );

	private:

		// the static instance.
		static Parameters * _thisInstance;

//
//		CONSTANTS
//

		// the input parameters from file gridder-params.
		static const char MEASUREMENT_SET[];
		static const char FIELD_ID[];
		static const char BEAM_ID[];
		static const char SPW[];
		static const char DATA_FIELD[];
		static const char TABLE_DATA[];
		static const char OUTPUT_PREFIX[];
		static const char CELL_SIZE[];
		static const char PIXELS_UV[];
		static const char W_PLANES[];
		static const char A_PROJECTION[];
		static const char LEAKAGE_CORRECTION[];
		static const char OVERSAMPLE[];
		static const char KERNEL_CUTOFF_FRACTION[];
		static const char KERNEL_CUTOFF_SUPPORT[];
		static const char DECONVOLVER[];
		static const char MINOR_CYCLES[];
		static const char LOOP_GAIN[];
		static const char CYCLEFACTOR[];
		static const char THRESHOLD[];
		static const char OUTPUT_RA[];
		static const char OUTPUT_DEC[];
		static const char WEIGHTING[];
		static const char ROBUST_PARAMETER[] ;
		static const char PB_CHANNELS[];
		static const char MOSAIC_DOMAIN[];
		static const char AIRY_DISK_DIAMETER[];
		static const char AIRY_DISK_BLOCKAGE[];
		static const char BEAM_PATTERN[];
		static const char BEAM_STOKES[];
		static const char BEAM_SIZE_PIXELS[];
		static const char BEAM_CELL_SIZE[];
		static const char BEAM_TYPE[];
		static const char STOKES[];
		static const char TELESCOPE[];
		static const char CACHE_LOCATION[];
		static const char GPU_LIST[];

//
//		GLOBAL VARIABLES
//

//
//		FUNCTIONS
//

}; // Parameters

