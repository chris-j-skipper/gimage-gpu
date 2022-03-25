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

// the kernel set class.
#ifndef Included_KernelSet
#define Included_KernelSet
#include "cuKernelSet.h"
#endif

// the parameters class.
#ifndef Included_Parameters
#define Included_Parameters
#include "cuParameters.h"
#endif

using namespace std;

//
//	NON CLASS MEMBERS
//

//
//	KernelSet
//
//	CJS: 23/02/2022
//
//	Holds the data for a single kernel set. A set of kernels means all the oversampled kernels for a single w-plane, channel, and mosaic component.
//

class KernelCache
{
	
	public:
		
		// default constructor
		KernelCache();
   
		// destructor
		~KernelCache();

		// overload the () operator.
		KernelSet & operator()( int pPBChannel, int pStokes, int pWPlane );			// to change data members.
		const KernelSet & operator()( int pPBChannel, int pStokes, int pWPlane ) const;	// to read data members.

//
//		ENUMERATED TYPES
//

//
//		STRUCTURES
//

//
//		CONSTANTS
//

//
//		GLOBAL VARIABLES
//

		vector<vector<vector<KernelSet> > > kernelSet;
		int pbChannels;
		int wPlanes;
		bool wProjection;
		bool aProjection;
		int oversample;
		griddegrid gridDegrid;
		int stokesProducts;
		bool * stokesFlag;

//
//		FUNCTIONS
//

		void CountVisibilities( Data * pData, int pMaxBatchSize, int pNumGPUs );
		void CountVisibilities( int pBatchSize, int pNumGPUs );
		void Create( int pPBChannels, int pWPlanes, bool pWProjection, bool pAProjection, bool pUseMosaicing, bool pUVPlaneMosaic, Data * pData, int pBeamSize, int pStokesProducts, int pStokes, int pOversample, griddegrid pGridDegrid, float * phstPrimaryBeamAtMaxWavelength, float * phstPrimaryBeamMosaicing );
		void GenerateKernelCache( int pPBChannel );

	private:

//
//		CONSTANTS
//


//
//		GLOBAL VARIABLES
//

		// casacore interface.
		CasacoreInterface * _casacoreInterface;

		// parameters.
		Parameters * _param;

		bool _useMosaicing;
		bool _uvPlaneMosaic;
		Data * _data;
		int _beamSize;
		int _oversampledBeamSize;
		double _cellSize;
		int _stokes;
		
		// we hold the primary beam at the maximum wavelength in memory because we'll need it for each channel.
		float * _devPrimaryBeamMaxWavelength;
		
		// we also need a primary beam for mosaicing.
		float * _hstPrimaryBeamMosaicing;

		// kernel calls.
		dim3 _gridSize2D;
		dim3 _blockSize2D;

//
//		FUNCTIONS
//

}; // KernelCache

