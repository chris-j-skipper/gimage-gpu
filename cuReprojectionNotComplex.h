#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <complex>
#include "cuda.h"
#include "cufft.h"

//
//	Reprojection
//
//	CJS: 14/11/2018
//
//	Performs image-plane reprojection on a GPU.
//

class Reprojection
{
	
	public:
		
		// default constructor
		Reprojection();
   
		// destructor
		~Reprojection();

//
//	ENUMERATED TYPES
//

		typedef enum
		{
			EPOCH_J2000,
			EPOCH_B1950,
			EPOCH_GALACTIC
		} Epoch;

		typedef enum
		{
			OUTPUT_TO_INPUT,
			INPUT_TO_OUTPUT
		} ProjectionDirection;

//
//	STRUCTURES
//

		// vector with doubles. can be used either as 2 or 3 element vector.
		struct rpVectD
		{
			double x;
			double y;
			double z;
		};
		typedef struct rpVectD rpVectD;

		// vector with integers.
		struct rpVectI
		{
			int x;
			int y;
		};
		typedef struct rpVectI rpVectI;

		// 2x2 matrix.
		struct rpMatr2x2
		{
			double a11;
			double a12;
			double a21;
			double a22;
		};
		typedef struct rpMatr2x2 rpMatr2x2;

		// 3x3 matrix.
		struct rpMatr3x3
		{
			double a11;
			double a12;
			double a13;
			double a21;
			double a22;
			double a23;
			double a31;
			double a32;
			double a33;
		};
		typedef struct rpMatr3x3 rpMatr3x3;

		// stores a coordinate system - reference pixel coordinates and world coordinates, and transformation matrix CD.
		struct rpCoordSys
		{

			// the reference pixel.
			rpVectD crVAL;
			rpVectD crPIX;

			// the linear conversion between pixel coordinates and RA and DEC offsets.
			rpMatr2x2 cd;
			rpMatr2x2 inv_cd;

			// the rotation matrices to convert between coordinates near the origin (RA 0, DEC 0) to the required RA and DEC.
			rpMatr3x3 toWorld;
			rpMatr3x3 toPixel;

			// an epoch enumerated type. either J2000, B1950 or GALACTIC.
			Epoch epoch;

		};
		typedef struct rpCoordSys rpCoordSys;

//
//	CONSTANTS
//

		// define a maximum number of interpolation points for each output pixel. this is the maximum along each axis, so the actual
		// number of interpolation points is n x n per pixel.
		static const int MAX_INTERPOLATION_POINTS = 10;

		// default pixel value - usually black (0).
		static const double DEFAULT_PIXEL_VALUE = 0.0;

//
//	GENERAL FUNCTIONS
//

		__host__ __device__ static double minD( double pA, double pB );
		__host__ __device__ static int minI( int pA, int pB );
		__host__ __device__ static double maxD( double pA, double pB );
		__host__ __device__ static int maxI( int pA, int pB );
		__host__ __device__ static double interpolateValue( rpVectD pPosition, double pBLValue, double pBRValue, double pTLValue, double pTRValue );
		__host__ __device__ static bool correctOrientation( rpVectD pTopLeft, rpVectD pTopRight, rpVectD pBottomLeft, rpVectD pBottomRight );

//
//	TRIG FUNCTIONS
//

		__host__ __device__ static double rad( double pIn );
		__host__ __device__ static double deg( double pIn );
		__host__ __device__ static void angleRange( double * pValue, double pCentre, double pMax );

//
//	MATRIX FUNCTIONS
//

		__host__ __device__ static rpMatr2x2 calculateInverseMatrix( rpMatr2x2 pMatrix );
		__host__ __device__ static rpMatr3x3 transpose( rpMatr3x3 pOldMatrix );
		__host__ __device__ static rpVectD multMatrixVector( rpMatr3x3 pMatrix, rpVectD pVector );
		__host__ __device__ static rpMatr3x3 multMatrix( rpMatr3x3 pMatrix1, rpMatr3x3 pMatrix2 );

//
//	FUNCTIONS
//

		void CreateDeviceMemory( rpVectI pOutSize );
		void GetCoordinates( double pX, double pY, rpCoordSys pCoordinateSystem, rpVectI pSize, double * pPhaseRA, double * pPhaseDEC );
		void ReprojectImage( float * pdevInImage, float * pdevOutImage, float * pdevNormalisationPattern, float * pdevPrimaryBeamPattern,
					rpCoordSys pInCoordinateSystem, rpCoordSys pOutCoordinateSystem, rpVectI pInSize, rpVectI pOutSize, bool * pdevInMask,
					float * pdevBeamIn, float * pdevBeamOut, rpVectI pBeamSize, ProjectionDirection pProjectionDirection,
					bool pAProjection, bool pVerbose );
		void ReprojectPixel( double * pPixel, int pNumPixels, rpCoordSys pInCoordinateSystem, rpCoordSys pOutCoordinateSystem, rpVectI pInSize, rpVectI pOutSize );
		void ReweightImage( float * pdevOutImage, float * pdevNormalisationPattern, float * pdevPrimaryBeamPattern, rpVectI pOutSize, bool * pdevOutMask,
					rpVectI pBeamSize );

	private:

//
//	CONSTANTS
//

		static const double PI = 3.14159265359;

		// coordinates of the Galactic coordinate system north pole in the J2000 coordinate system.
		static const double NP_RA_GAL_IN_J2000 = 192.859496;
		static const double NP_DEC_GAL_IN_J2000 = 27.128353;
		static const double NP_RA_OFFSET_GAL_IN_J2000 = 302.932069;
	
		// coordinates of the J2000 coordinate system north pole in the galactic coordinate system.
		static const double NP_RA_J2000_IN_GAL = 122.932000;
		static const double NP_DEC_J2000_IN_GAL = 27.128431;
		static const double NP_RA_OFFSET_J2000_IN_GAL = 12.860114;
	
		// coordinates of the Galactic coordinate system north pole in the B1950 coordinate system.
		static const double NP_RA_GAL_IN_B1950 = 192.250000;
		static const double NP_DEC_GAL_IN_B1950 = 27.400000;
		static const double NP_RA_OFFSET_GAL_IN_B1950 = 303.000000;
	
		// coordinates of the B1950 coordinate system north pole in the galactic coordinate system.
		static const double NP_RA_B1950_IN_GAL = 123.000000;
		static const double NP_DEC_B1950_IN_GAL = 27.400000;
		static const double NP_RA_OFFSET_B1950_IN_GAL = 12.250000;
	
		// coordinates of the J2000 coordinate system north pole in the B1950 coordinate system.
		static const double NP_RA_J2000_IN_B1950 = 359.686210;
		static const double NP_DEC_J2000_IN_B1950 = 89.721785;
		static const double NP_RA_OFFSET_J2000_IN_B1950 = 0.327475;
	
		// coordinates of the B1950 coordinate system north pole in the J2000 coordinate system.
		static const double NP_RA_B1950_IN_J2000 = 180.315843;
		static const double NP_DEC_B1950_IN_J2000 = 89.72174782;
		static const double NP_RA_OFFSET_B1950_IN_J2000 = 179.697628;

//
//	GLOBAL VARIABLES
//

		// input image and coordinate system.
		rpVectI _inSize;
		rpCoordSys _inCoordinateSystem;

		// output image and coordinate system.
		rpVectI _outSize;
		rpCoordSys _outCoordinateSystem;

		// rotation matrix for epoch conversion. this is built once at the start.
		rpMatr3x3 _epochConversion;

		// the maximum threads per block.
		int _maxThreadsPerBlock;

		// stores the output to input translation map and the map-valid mask.
		double * _devProjectionMap;
		bool * _devMapValid;

//
//	GENERAL FUNCTIONS
//

		void toUppercase( char * pChar );
		void setThreadBlockSize2D( int pThreadsX, int pThreadsY, dim3 & pGridSize2D, dim3 & pBlockSize2D );

//
//	TRIG FUNCTIONS
//

		double arctan( double pValueTop, double pValueBottom );

//
//	EULER ROTATION FUNCTIONS
//

		rpMatr3x3 rotateX( double pAngle );
		rpMatr3x3 rotateY( double pAngle );
		rpMatr3x3 rotateZ( double pAngle );
		void calculateRotationMatrix( rpCoordSys * pCoordinateSystem, bool pEpochConversion );

//
//	EPOCH CONVERSION FUNCTIONS
//

		rpMatr3x3 epochConversionMatrix( double pNP_RA, double pNP_DEC, double pNP_RA_OFFSET );
		rpMatr3x3 doEpochConversion( rpCoordSys pFrom, rpCoordSys pTo );
		Epoch getEpoch( char * pEpoch );

//
//	REPROJECTION AND REGRIDDING FUNCTIONS
//

		void reprojection( float * pdevInImage, float * pdevOutImage, float * pdevNormalisationPattern, float * pdevPrimaryBeamPattern,
					bool * pdevInMask, float * pdevBeamIn, float * pdevBeamOut, rpVectI pBeamSize, ProjectionDirection pProjectionDirection,
					bool pAProjection );

}; // Reprojection
