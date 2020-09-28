#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <ctype.h>

// include types.
#ifndef Included_Types
#define Included_Types
#include "../gridding/cppTypes.h"
#endif

//
//	STRUCTURES
//

// latitude and logitude coordinates.
struct PolarCoords
{
	double longitude;
	double latitude;
	char epoch[20];
};
typedef struct PolarCoords PolarCoords;

// 3-element vector.
struct VectorXYZ
{
	double x;
	double y;
	double z;
};
typedef struct VectorXYZ VectorXYZ;
	
// 3x3 matrix.
//struct Matrix3x3
//{
//	double a11;
//	double a12;
//	double a13;
//	double a21;
//	double a22;
//	double a23;
//	double a31;
//	double a32;
//	double a33;
//};
//typedef struct Matrix3x3 Matrix3x3;

class PhaseCorrection
{
	
	public:

//
//	GLOBAL VARIABLES
//
	
		// input and output uvw vectors.
		VectorXYZ uvwIn;
		VectorXYZ uvwOut;
		
		double phase;
			
		// input and output phase centres.
		PolarCoords inCoords;
		PolarCoords outCoords;
		
		// are we doing full uv reprojection, or just calculating the phase change?
		bool uvProjection;
		
//
//	PUBLIC FUNCTIONS
//
		
		void init();
		void rotate();
		
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

		// the full rotation matrix from the input to the output coordinate system. this matrix would need
		// to be calculated once and then stored so that it can be applied to many baseline vectors.
		Matrix3x3 _uvwRotation;

		// rotation matrix for epoch conversion.
		Matrix3x3 _epochConversion;

//
//	PRIVATE FUNCTIONS
//
		
		void toUppercase( char * pChar );
		double rad( double pIn );
		VectorXYZ multMatrixVector( Matrix3x3 pMatrix, VectorXYZ pVector );
		Matrix3x3 multMatrix( Matrix3x3 pMatrix1, Matrix3x3 pMatrix2 );
		Matrix3x3 transpose( Matrix3x3 pOldMatrix );
		Matrix3x3 inverse2x2( Matrix3x3 pOldMatrix );
		Matrix3x3 rotateX( double pAngle );
		Matrix3x3 rotateY( double pAngle );
		Matrix3x3 rotateZ( double pAngle );
		Matrix3x3 convertXYZtoUVW( PolarCoords pCoords );
		Matrix3x3 convertUVWtoXYZ( PolarCoords pCoords );
		Matrix3x3 epochConversionMatrix( double pNP_RA, double pNP_DEC, double pNP_RA_OFFSET );
		Matrix3x3 doEpochConversion( PolarCoords pIn, PolarCoords pOut );
		double getPathLengthDifference( VectorXYZ pUVW );
		Matrix3x3 reprojectUV();
	
}; // PhaseCorrection
