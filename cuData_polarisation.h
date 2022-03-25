#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <complex>
#include "cuda.h"
#include "cufft.h"
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

// the parameters class.
#ifndef Included_Parameters
#define Included_Parameters
#include "cuParameters.h"
#endif

//
//	Data
//
//	CJS: 20/08/2021
//
//	Holds the data (visibilities, grid positions, etc) for a single mosaic component, and functions for manipulating this data.
//

class Data
{
	
	public:
		
		// default constructor
		Data();
		Data( int pTaylorTerms, int pMosaicID, bool pWProjection, bool pAProjection, int pWPlanes, int pPBChannels,
			bool pCacheData, int pStokes, int pStokesImages );
   
		// destructor3703
		~Data();

//
//	ENUMERATED TYPES
//

//
//	STRUCTURES
//

//
//	CONSTANTS
//

//
//	GLOBAL VARIABLES
//
	
		Data * NextComponent;

		// variables that are retained, and cached if necessary.
		cufftComplex *** Visibility;
		VectorI * GridPosition;
		int * KernelIndex;
		int * DensityMap;
		float ** Weight;
		cufftComplex *** ResidualVisibility;
		float *** MfsWeight;

		// other variables that are kept.
		double * WPlaneMean;
		double * WPlaneMax;
		int PBChannels;
		int WPlanes;

		// primary beams.
		float * AveragePrimaryBeamIn;		// this primary beam is based upon the average frequency in the file, and is centred in the image. it can be reprojected
							// to any position and frequency required.
		float * PrimaryBeam;			// this primary beam is used for mosaicing, and also for setting the image mask. it is in the reference frame of
							//	the output phase position.
		float * PrimaryBeamRatio;		// this primary beam is used for rescaling the dirty image to a primary beam that is common to all channels. we use
							//	the maximum wavelength for this purpose.
		float * PrimaryBeamInFrame;		// this primary beam is shown in the reference frame of each mosaic component (used for image-plane mosaicing).
		
		float * MuellerDeterminant;

		// mosaic component mask (only for image-plane mosaics).
		bool * ComponentMask;

		// variables that are used while importing data. these might eventually be made private.
		bool * Flag;
		int * SampleID;
		int * ChannelID;
		int * FieldID;

		double AverageWavelength;
		double MaximumWavelength;
		long int GriddedVisibilities;
		long int MinimumVisibilitiesInMosaic;
		int Stages;
		int * Batches;
		long int * NumVisibilities;
		double * AverageWeight;

		// hold the required phase centre of each mosaic component. these variables is only used for image-plane mosaics.
		double ImagePlaneRA;
		double ImagePlaneDEC;

		// hold the phase positions of the data, and the phase positions we want to image at.
		double PhaseFromRA;
		double PhaseFromDEC;
		double PhaseToRA;
		double PhaseToDEC;
		
		// Jones matrices.
		cufftComplex ** JonesMatrixIn;
		cufftComplex ** JonesMatrix;

		// mueller matrix and inverse matrix.
		cufftComplex ** MuellerMatrix;
		cufftComplex ** InverseMuellerMatrix;
		
		// flags to record whether or not cells of the Mueller and inverse-Mueller matrices are populated or null. these flags are used during gridding to skip
		// certain Stokes products.
		bool * MuellerMatrixFlag;
		bool * InverseMuellerMatrixFlag;

//
//	FUNCTIONS
//

		void AdjustPrimaryBeamForAProjection( int pBeamSize, bool * pMask, bool pImagePlaneMosaic );
		void BuildComponentMask( float * pPrimaryBeamPattern, double pCellSize, double pOutputRA, double pOutputDEC, int pBeamSize );
		void CacheData( int pBatchID, int pTaylorTerm, int pWhatData );
		cufftDoubleComplex ** ConvertJonesToMueller( cufftComplex ** phstJonesMatrix, int pImageSize );
		void CopyJonesMatrixIn( cufftComplex ** pFromMatrix );
		void Create( int pTaylorTerms, int pMosaicID, bool pWProjection, bool pAProjection, int pWPlanes, int pPBChannels,
				bool pCacheData, int pStokes, int pStokesImages );
		void DeleteCache();
		void FreeData( int pWhatData );
		void FreeJonesMatrices();
		void FreeMuellerMatrices();
		void FreeOffAxisMuellerMatrices();
		void FreeUnwantedMuellerMatrices( int pStokes );
		void GenerateAveragePrimaryBeam( int pNumSpws, int * pNumChannels, bool ** pSpwChannelFlag, double ** pWavelength );
		void GenerateMuellerMatrix( int pPBChannel, int pImageSize, float * pdevPrimaryBeam, float pPBMaxValue );
		void PerformUniformWeighting( double ** phstTotalWeightPerCell );
		void PerformRobustWeighting( double ** phstTotalWeightPerCell );
		void ProcessMeasurementSet( int pFileIndex, double ** phstTotalWeightPerCell, vector<Data> & pData );
		void ReduceJonesMatrixChannels( int pNumSpws, int * pNumChannels, bool ** pSpwChannelFlag, int ** pWhichPBChannel );
		void ReprojectJonesMatrix( int pPBChannel, int pBeamOutSize, double pBeamOutCellSize );
		void ReprojectMuellerDeterminant( int pBeamOutSize, double pBeamOutCellSize, double pToRA, double pToDEC );
		float * ReprojectPrimaryBeam( int pBeamOutSize, double pBeamOutCellSize, double pToRA, double pToDEC, double pToWavelength );
		void ShrinkCache( long int pMaxMemory );
		void SumWeights( double ** phstTotalWeightPerCell, int pStageID );
		void UncacheData( int pBatchID, int pTaylorTerm, long int pOffset, int pWhatData, int pStokes );

	private:

//
//	CONSTANTS
//


//
//	GLOBAL VARIABLES
//

		// casacore interface.
		CasacoreInterface * _casacoreInterface;

		// parameters.
		Parameters * _param;

		bool _wProjection;
		bool _aProjection;
		bool _cacheData;
		int _mosaicID;
		int _stokes;
		int _stokesImages;
		int _taylorTerms;
		
		// the mueller determinant in the input frame.
		float * _muellerDeterminantIn;

		// kernel calls.
		dim3 _gridSize2D;
		dim3 _blockSize2D;

//
//	FUNCTIONS

		void addMosaicComponent( std::vector<Data> & pData );
		cufftDoubleComplex ** calculateInverseImageMatrix( cufftDoubleComplex ** pdevMatrix, int pMatrixSize, int pImageSize, bool pDivideByDeterminant );
		void calculateGridPositions( int pStageID, long int pPreferredVisibilityBatchSize, int pNumSpws, int pNumSamplesInStage, int pSampleStageID, int pTotalSamples,
						int pOversample, double pUvCellSize, double ** phstWavelength, int * phstNumChannels, double * phstPhase,
						int ** phstWhichPBChannel, int * phstSpw, float ** phstSampleWeight, int * phstSampleFieldID, VectorD * phstSample,
						int pNumFields, int pNumGPUs, int * phstGPU );
		void calculatePBChannels( int *** phstWhichPBChannel, double * phstPBChannelWavelength, double ** phstWavelength, int pNumSpws, int * phstNumChannels, bool ** phstSpwChannelFlag );
		void calculateVisibilityAndFlag( int pStageID, long int pPreferredVisibilityBatchSize, int pNumPolarisations, int pNumSamplesInStage,
							int * phstPolarisationConfig, cufftComplex * phstVisibilityIn, bool * phstFlagIn, double ** pdevMultiplier );
		void calculateWPlanes( int pFieldID, int pCasaFieldID, int pNumSamples, VectorD * phstSample, int * phstFieldID, double pMinWavelength );
		long int compactData( long int * pTotalVisibilities, long int pFirstVisibility, long int pNumVisibilitiesToProcess );
		void compactFieldIDs( double ** pPhaseCentrePtr, double ** pPhaseCentreImagePtr, int * pNumFields, int * pFieldID, int ** pFieldIDMap, int pNumSamples );
		cufftDoubleComplex * determinantImageMatrix( cufftDoubleComplex ** pdevMatrix, int pMatrixSize, int pImageSize );
		void doPhaseCorrectionSamples( PhaseCorrection * pPhaseCorrection, int pNumSamples, double * pPhaseCentreIn, double * pPhaseCentreOut, VectorD * pSample, 
					int * pFieldID, double * pPhase );
		void generatePrimaryBeamAiry( cufftComplex ** phstJonesMatrix, double pWidth, double pCutout, int pNumSpws, int * phstNumChannels, double ** phstWavelength );
		void generatePrimaryBeamGaussian( cufftComplex ** phstJonesMatrix, double pWidth, int pNumSpws, int * phstNumChannels, double ** phstWavelength );
		void getASKAPBeamPosition( double * pRA, double * pDEC, double pXOffset, double pYOffset, double pCentreRA, double pCentreDEC );
		double getPrimaryBeamWidth( float * phstBeam, int pBeamSize );
		void getSuitablePhasePositionForBeam( double * pBeamIn, double * pPhase, int pNumBeams, double pBeamWidth );
		double ** getPolarisationMultiplier( char * pMeasurementSetFilename, int * pNumPolarisations, int * pNumPolarisationConfigurations, char * pTableData );
		bool loadPrimaryBeam( char * pBeamFilename, cufftComplex ** phstPrimaryBeamIn, int pSize, int pNumChannels );
		void mergeData( int pStageID_one, int pStageID_two, bool pLoadAllData, int pWhatData );
		cufftDoubleComplex ** outerProductImageMatrix( cufftComplex ** pOne, cufftComplex ** pTwo, int pMatrixSize1, int pMatrixSize2, int pImageSize );
		void parseChannelRange( char * pChannelRange, int pNumChannels, bool * phstSpwChannelFlag );
		void parseSpwSpecifier( char * pSpwSpecifier, int pNumSpws, int * phstNumChannels, bool ** phstSpwChannelFlag );
		void quickSortData( long int pLeft, long int pRight );
		void quickSortFieldIDs( int * pFieldID, int pLeft, int pRight );
		void reprojectPrimaryBeams( int pBeamOutSize, double pBeamOutCellSize, double pOutputRA, double pOutputDEC, bool pImagePlaneMosaic, double pMaxWavelength );
		void separateFields( int pNumFields, int pStageID );
		void setSpwAndChannelFlags( int pNumSpws, int * phstNumChannels, bool *** phstSpwChannelFlag, char * phstSpwRestriction );
		void reprojectImage( cufftComplex * phstImageIn, cufftComplex * phstImageOut, int pImageInSize, int pImageOutSize, double pInputCellSize,
					double pOutputCellSize, double pInRA, double pInDec, double pOutRA, double pOutDec, float * pdevInImage, float * pdevOutImage,
					Reprojection & pImagePlaneReprojection, bool pSquareBeam, bool pVerbose );
		void reprojectImage( float * phstImageIn, float * phstImageOut, int pImageInSize, int pImageOutSize, double pInputCellSize, double pOutputCellSize,
					double pInRA, double pInDec, double pOutRA, double pOutDec, float * pdevInImage, float * pdevOutImage,
					Reprojection & pImagePlaneReprojection, bool pVerbose );
		VectorD rotateX( VectorD pIn, double pAngle );
		VectorD rotateY( VectorD pIn, double pAngle );

}; // Data

