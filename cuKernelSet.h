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

// include functions.
#include "cuFunctions.h"

// my casacore interface.
#ifndef Included_Casacore
#define Included_Casacore
#include "cppCasacoreInterface.h"
#endif

// data class.
#ifndef Included_Data
#define Included_Data
#include "cuData_polarisation.h"
#endif

// the parameters class.
#ifndef Included_Parameters
#define Included_Parameters
#include "cuParameters.h"
#endif

//
//	NON CLASS MEMBERS
//

__host__ __device__ static double spheroidalWaveFunction( double pR );
__global__ void devBuildKernelCutoffHistogram( cufftComplex * pKernel, int pSize, double * pMaxValue, double pCutoffFraction, int * pHistogram, int pMaxSupport );
__global__ void devGenerateAAKernel( cufftComplex * pAAKernel, int pKernelSize, int pWorkspaceSize );
__global__ void devGenerateWKernel( cufftComplex * pWKernel, double pW, int pWorkspaceSize, double pCellSizeDirectionalCosine, griddegrid pGridDegrid, int pSize );
__global__ void devGenerateAKernel( cufftComplex * pAKernel, cufftComplex * pPrimaryBeamAProjection, int pPrimaryBeamSupport, int pWorkspaceSize );
__global__ void devSetPrimaryBeamForGriddingAndDegridding( float * pImage, int pSize, griddegrid pGridDegrid, bool pAProjection );
__global__ void devUpdateKernel( cufftComplex * pKernel, cufftComplex * pImage, int pSupport, int pOversample, int pOversampleI, int pOversampleJ,
					int pWorkspaceSize, griddegrid pGridDegrid );

//
//	KernelSet
//
//	CJS: 23/02/2022
//
//	Holds the data for a single kernel set. A set of kernels means all the oversampled kernels for a single w-plane, channel, and mosaic component.
//

class KernelSet
{
	
	public:
		
		// default constructor
		KernelSet();
		KernelSet( int pOversample );
   
		// destructor
		~KernelSet();

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

		int oversample;
		int kernelSize;
		int supportSize;
		cufftComplex * kernel;
		int *** visibilities;

//
//	FUNCTIONS
//

		void createArrays( int pNumberOfStages, int * pNumberOfBatches, int pNumGPUs );
		int findSupportForKernel( cufftComplex * pdevKernel, int pSize, double * pdevMaxValue, double pCutoffFraction, int pMaxSupport );
		bool generateKernel( int pW, int pChannel, bool pWProjection, bool pAProjection, float * phstPrimaryBeamMosaicing, cufftComplex * phstPrimaryBeamAProjection, float * phstPrimaryBeamRatio, int pBeamSize, Data * phstData, griddegrid pGridDegrid, bool * pKernelOverflow, double pAverageWeight, bool pUVPlaneMosaic );

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
		
		int _numberOfStages;
		int * _numberOfBatches;

		// kernel calls.
		dim3 _gridSize2D;
		dim3 _blockSize2D;

//
//	FUNCTIONS
//

}; // KernelSet

