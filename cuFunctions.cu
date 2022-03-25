#include "cuFunctions.h"

//
//	HOST & DEVICE FUNCTIONS
//

//
//	conjugate()
//
//	CJS: 24/09/2021
//
//	Calculate the complex conjugate of a complex number.
//

__host__ __device__ cufftComplex conjugate( cufftComplex pIn )
{

	cufftComplex pOut = { .x = pIn.x, .y = -pIn.y };

	// return something.
	return pOut;

} // conjugate

//
//	divideComplex()
//
//	CJS: 28/09/2015
//
//	Divide one complex number by another.
//

__host__ __device__ cufftDoubleComplex divideComplex( cufftDoubleComplex pOne, double pTwo )
{
	
	cufftDoubleComplex newValue = { .x = 0.0, .y = 0.0 };
	
	if (pTwo != 0.0)
	{
		newValue.x = (pOne.x / pTwo);
		newValue.y = (pOne.y / pTwo);
	}
	
	// return something.
	return newValue;
	
} // divideComplex

__host__ __device__ cufftComplex divideComplex( cufftComplex pOne, float pTwo )
{
	
	cufftComplex newValue = { .x = 0.0, .y = 0.0 };
	
	if (pTwo != 0.0)
	{
		newValue.x = (pOne.x / pTwo);
		newValue.y = (pOne.y / pTwo);
	}
	
	// return something.
	return newValue;
	
} // divideComplex

__host__ __device__ cufftComplex divideComplex( cufftComplex pOne, cufftComplex pTwo )
{
	
	cufftComplex newValue = { .x = 0.0, .y = 0.0 };
	
	// to avoid the numbers we are dividing being too large or too small we divide the numerator and denominator by a common factor.
	double multiplier = 1.0;
	if (pOne.x != 0.0 || pOne.y != 0.0 || pTwo.x != 0.0 || pTwo.y != 0.0)
	{
		while ((abs( pOne.x ) < multiplier || pOne.x == 0.0) && (abs( pOne.y ) < multiplier || pOne.y == 0.0) &&
			(abs( pTwo.x ) < multiplier || pTwo.x == 0.0) && (abs( pTwo.y ) < multiplier || pTwo.y == 0.0))
			multiplier /= 10.0;
		while ((abs( pOne.x ) > multiplier || pOne.x == 0.0) && (abs( pOne.y ) > multiplier || pOne.y == 0.0) &&
			(abs( pTwo.x ) > multiplier || pTwo.x == 0.0) && (abs( pTwo.y ) > multiplier || pTwo.y == 0.0))
			multiplier *= 10.0;
		pOne = divideComplex( /* pOne = */ pOne, /* pTwo = */ multiplier );
		pTwo = divideComplex( /* pOne = */ pTwo, /* pTwo = */ multiplier );
	}
	
	double denominator = pow( pTwo.x, 2 ) + pow( pTwo.y, 2 );
	if (denominator != 0.0)
	{
		newValue.x = ((pOne.x * pTwo.x) + (pOne.y * pTwo.y)) / denominator;
		newValue.y = ((pOne.y * pTwo.x) - (pOne.x * pTwo.y)) / denominator;
	}
	
	// return something.
	return newValue;
	
} // divideComplex

__host__ __device__ cufftDoubleComplex divideComplex( cufftDoubleComplex pOne, cufftDoubleComplex pTwo )
{
	
	cufftDoubleComplex newValue = { .x = 0.0, .y = 0.0 };
	
	// to avoid the numbers we are dividing being too large or too small we divide the numerator and denominator by a common factor.
	double multiplier = 1.0;
	if (pOne.x != 0.0 || pOne.y != 0.0 || pTwo.x != 0.0 || pTwo.y != 0.0)
	{
		while ((abs( pOne.x ) < multiplier || pOne.x == 0.0) && (abs( pOne.y ) < multiplier || pOne.y == 0.0) &&
			(abs( pTwo.x ) < multiplier || pTwo.x == 0.0) && (abs( pTwo.y ) < multiplier || pTwo.y == 0.0))
			multiplier /= 10.0;
		while ((abs( pOne.x ) > multiplier || pOne.x == 0.0) && (abs( pOne.y ) > multiplier || pOne.y == 0.0) &&
			(abs( pTwo.x ) > multiplier || pTwo.x == 0.0) && (abs( pTwo.y ) > multiplier || pTwo.y == 0.0))
			multiplier *= 10.0;
		pOne = divideComplex( /* pOne = */ pOne, /* pTwo = */ multiplier );
		pTwo = divideComplex( /* pOne = */ pTwo, /* pTwo = */ multiplier );
	}
	
	double denominator = pow( pTwo.x, 2 ) + pow( pTwo.y, 2 );
	if (denominator != 0.0)
	{
		newValue.x = ((pOne.x * pTwo.x) + (pOne.y * pTwo.y)) / denominator;
		newValue.y = ((pOne.y * pTwo.x) - (pOne.x * pTwo.y)) / denominator;
	}
	
	// return something.
	return newValue;
	
} // divideComplex

__host__ __device__ cufftDoubleComplex divideComplex( cufftDoubleComplex pOne, cufftComplex pTwo )
{
	
	cufftDoubleComplex newValue = { .x = 0.0, .y = 0.0 };
	
	// to avoid the numbers we are dividing being too large or too small we divide the numerator and denominator by a common factor.
	double multiplier = 1.0;
	if (pOne.x != 0.0 || pOne.y != 0.0 || pTwo.x != 0.0 || pTwo.y != 0.0)
	{
		while ((abs( pOne.x ) < multiplier || pOne.x == 0.0) && (abs( pOne.y ) < multiplier || pOne.y == 0.0) &&
			(abs( pTwo.x ) < multiplier || pTwo.x == 0.0) && (abs( pTwo.y ) < multiplier || pTwo.y == 0.0))
			multiplier /= 10.0;
		while ((abs( pOne.x ) > multiplier || pOne.x == 0.0) && (abs( pOne.y ) > multiplier || pOne.y == 0.0) &&
			(abs( pTwo.x ) > multiplier || pTwo.x == 0.0) && (abs( pTwo.y ) > multiplier || pTwo.y == 0.0))
			multiplier *= 10.0;
		pOne = divideComplex( /* pOne = */ pOne, /* pTwo = */ multiplier );
		pTwo = divideComplex( /* pOne = */ pTwo, /* pTwo = */ multiplier );
	}
	
	double denominator = pow( pTwo.x, 2 ) + pow( (double) pTwo.y, 2 );
	if (denominator != 0.0)
	{
		newValue.x = ((pOne.x * (double) pTwo.x) + (pOne.y * (double) pTwo.y)) / denominator;
		newValue.y = ((pOne.y * (double) pTwo.x) - (pOne.x * (double) pTwo.y)) / denominator;
	}
	
	// return something.
	return newValue;
	
} // divideComplex

__host__ __device__ cufftDoubleComplex divideComplex( double pOne, cufftDoubleComplex pTwo )
{
	
	cufftDoubleComplex newValue = { .x = 0.0, .y = 0.0 };
	
	// to avoid the numbers we are dividing being too large or too small we divide the numerator and denominator by a common factor.
	double multiplier = 1.0;
	if (pOne != 0.0 || pTwo.x != 0.0 || pTwo.y != 0.0)
	{
		while ((abs( pOne ) < multiplier || pOne == 0.0) &&
			(abs( pTwo.x ) < multiplier || pTwo.x == 0.0) && (abs( pTwo.y ) < multiplier || pTwo.y == 0.0))
			multiplier /= 10.0;
		while ((abs( pOne ) > multiplier || pOne == 0.0) &&
			(abs( pTwo.x ) > multiplier || pTwo.x == 0.0) && (abs( pTwo.y ) > multiplier || pTwo.y == 0.0))
			multiplier *= 10.0;
		pOne = pOne / multiplier;
		pTwo = divideComplex( /* pOne = */ pTwo, /* pTwo = */ multiplier );
	}
	
	double denominator = pow( pTwo.x, 2 ) + pow( pTwo.y, 2 );
	if (denominator != 0.0)
	{
		newValue.x = (pOne * pTwo.x) / denominator;
		newValue.y = -(pOne * pTwo.y) / denominator;
	}
	
	// return something.
	return newValue;
	
} // divideComplex

//
//	gaussian2D()
//
//	CJS: 05/11/2015
//
//	Create an elliptical 2D Gaussian at position (pX, pY), with long and short axes pR1 and pR2, rotated at pAngle.
//

__host__ __device__ double gaussian2D( double pNormalisation, double pX, double pY, double pAngle, double pR1, double pR2 )
{
	
	// calculate the distance along the long and short axes.
	double rOne = ((pY * cos( pAngle )) + (pX * sin( pAngle )));
	double rTwo = ((pX * cos( pAngle )) - (pY * sin( pAngle )));
	
	// we want the axis-one distance as a multiple of the Gaussian width in this direction, and then squared.
	if (pR1 != 0)
		rOne = pow( rOne / pR1, 2 );
	else
		
		// the Gaussian has no width along axis one. we use a flag of -1 to indicate that we need to return a zero value.
		if (rOne != 0)
			rOne = -1;
	
	// we want the axis-two distance as a multiple of the Gaussian width in this direction, and then squared.
	if (pR2 != 0)
		rTwo = pow( rTwo / pR2, 2 );
	else
		
		// the Gaussian has no width along axis two. we use a flag of -1 to indicate that we need to return a zero value.
		if (rTwo != 0)
			rTwo = -1;
	
	// calculate the return value.
	double returnValue = pNormalisation;
	if (rOne >= 0)
		returnValue = returnValue * exp( -rOne );
	if (rTwo >= 0)
		returnValue = returnValue * exp( -rTwo );
	
	// if either of the distances is < 1 then this means our Gaussian has no width in this direction, and our return
	// value should be 0.
	if (rOne < 0 || rTwo < 0)
		returnValue = 0;
		
	// return something.
	return returnValue;
	
} // gaussian2D

//
//	intFloor()
//
//	CJS: 25/08/2021
//
//	Returns the floor of a division, rounded down to the nearest integer.
//

__host__ __device__ int intFloor( int pValue1, int pValue2 )
{

	// return something.
	return (int) floor( (double) pValue1 / (double) pValue2 );

} // intFloor

//
//	interpolateBeam()
//
//	CJS: 02/02/2022
//
//	Bi-linear interpolation of the primary beam.
//

__host__ __device__ double interpolateBeam( float * pBeam, int pBeamSize, int pI, int pJ, double pFracI, double pFracJ )
{

	double beam = 0.0;

	if (pI >= 0 && pI < pBeamSize - 1 && pJ >= 0 && pJ < pBeamSize - 1)
	{
		double beamTL = pBeam[ ((pJ + 1) * pBeamSize) + pI ];
		double beamTR = pBeam[ ((pJ + 1) * pBeamSize) + pI + 1 ];
		double beamBL = pBeam[ (pJ * pBeamSize) + pI ];
		double beamBR = pBeam[ (pJ * pBeamSize) + pI + 1 ];
		double beamTop = ((beamTR - beamTL) * pFracI) + beamTL;
		double beamBottom = ((beamBR - beamBL) * pFracI) + beamBL;
		beam = ((beamTop - beamBottom) * pFracJ) + beamBottom;
	}
	
	// return something.
	return beam;

} // interpolateBeam

__host__ __device__ cufftComplex interpolateBeam( cufftComplex * pBeam, int pBeamSize, int pI, int pJ, double pFracI, double pFracJ )
{

	cufftComplex beam = { .x = 0.0, .y = 0.0 };

	if (pI >= 0 && pI < pBeamSize - 1 && pJ >= 0 && pJ < pBeamSize - 1)
	{
		cufftComplex beamTL = pBeam[ ((pJ + 1) * pBeamSize) + pI ];
		cufftComplex beamTR = pBeam[ ((pJ + 1) * pBeamSize) + pI + 1 ];
		cufftComplex beamBL = pBeam[ (pJ * pBeamSize) + pI ];
		cufftComplex beamBR = pBeam[ (pJ * pBeamSize) + pI + 1 ];
		cufftComplex beamTop = {	.x = ((beamTR.x - beamTL.x) * pFracI) + beamTL.x,
						.y = ((beamTR.y - beamTL.y) * pFracI) + beamTL.y };
		cufftComplex beamBottom = {	.x = ((beamBR.x - beamBL.x) * pFracI) + beamBL.x,
						.y = ((beamBR.y - beamBL.y) * pFracI) + beamBL.y };
		beam.x = ((beamTop.x - beamBottom.x) * pFracJ) + beamBottom.x;
		beam.y = ((beamTop.y - beamBottom.y) * pFracJ) + beamBottom.y;
	}
	
	// return something.
	return beam;

} // interpolateBeam

//
//	mod()
//
//	CJS: 25/08/2021
//
//	Find the modulus of two numbers.
//

__host__ __device__ int mod( int pValue1, int pValue2 )
{

	int value = intFloor( /* pValue1 = */ pValue1, /* pValue2 = */ pValue2 );

	// return something.
	return (pValue1 - (value * pValue2));

} // mod

//
//	multComplex()
//
//	CJS: 28/09/2015
//
//	Multiply two complex numbers.
//

__host__ __device__ cufftDoubleComplex multComplex( cufftDoubleComplex pOne, double pTwo )
{
	
	cufftDoubleComplex newValue;
	
	newValue.x = (pOne.x * pTwo);
	newValue.y = (pOne.y * pTwo);
	
	// return something.
	return newValue;
	
} // MultComplex

__host__ __device__ cufftComplex multComplex( cufftComplex pOne, float pTwo )
{
	
	cufftComplex newValue;
	
	newValue.x = (pOne.x * pTwo);
	newValue.y = (pOne.y * pTwo);
	
	// return something.
	return newValue;
	
} // MultComplex

__host__ __device__ cufftComplex multComplex( cufftComplex pOne, cufftComplex pTwo )
{
	
	cufftComplex newValue;
	
	newValue.x = (pOne.x * pTwo.x) - (pOne.y * pTwo.y);
	newValue.y = (pOne.x * pTwo.y) + (pOne.y * pTwo.x);
	
	// return something.
	return newValue;
	
} // MultComplex

__host__ __device__ cufftDoubleComplex multComplex( cufftDoubleComplex pOne, cufftComplex pTwo )
{
	
	cufftDoubleComplex newValue;
	
	newValue.x = (pOne.x * (double) pTwo.x) - (pOne.y * (double) pTwo.y);
	newValue.y = (pOne.x * (double) pTwo.y) + (pOne.y * (double) pTwo.x);
	
	// return something.
	return newValue;
	
} // MultComplex

__host__ __device__ cufftDoubleComplex multComplex( cufftDoubleComplex pOne, cufftDoubleComplex pTwo )
{
	
	cufftDoubleComplex newValue;
	
	newValue.x = (pOne.x * (double) pTwo.x) - (pOne.y * (double) pTwo.y);
	newValue.y = (pOne.x * (double) pTwo.y) + (pOne.y * (double) pTwo.x);
	
	// return something.
	return newValue;
	
} // MultComplex

__host__ __device__ cufftDoubleComplex multComplex( cufftComplex pOne, cufftDoubleComplex pTwo )
{
	
	cufftDoubleComplex newValue;
	
	newValue.x = ((double) pOne.x * pTwo.x) - ((double) pOne.y * pTwo.y);
	newValue.y = ((double) pOne.x * pTwo.y) + ((double) pOne.y * pTwo.x);
	
	// return something.
	return newValue;
	
} // MultComplex

//
//	swap()
//
//	CJS: 21/06/2021
//
//	Swap two values.
//

__host__ __device__ void swap( VectorI & pOne, VectorI & pTwo )
{

	VectorI tmp = pOne;
	pOne = pTwo;
	pTwo = tmp;

} // swap

__host__ __device__ void swap( int & pOne, int & pTwo )
{

	int tmp = pOne;
	pOne = pTwo;
	pTwo = tmp;

} // swap

__host__ __device__ void swap( cufftComplex & pOne, cufftComplex & pTwo )
{

	cufftComplex tmp = pOne;
	pOne = pTwo;
	pTwo = tmp;

} // swap

__host__ __device__ void swap( bool & pOne, bool & pTwo )
{

	bool tmp = pOne;
	pOne = pTwo;
	pTwo = tmp;

} // swap

__host__ __device__ void swap( float & pOne, float & pTwo )
{

	float tmp = pOne;
	pOne = pTwo;
	pTwo = tmp;

} // swap

//
//	KERNEL FUNCTIONS
//

//
//	devAddArrays()
//
//	CJS: 22/06/2020
//
//	Adds two arrays using non-atomic additions.
//

__global__ void devAddArrays( cufftComplex * pOne, cufftComplex * pTwo, int pSize )
{
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	// check that we are within the array bounds.
	if (i < pSize)
	{
//cufftComplex tmp = pOne[ i ];
		cufftComplex addition = pTwo[ i ];
		pOne[ i ].x += addition.x;
		pOne[ i ].y += addition.y;
//if (i == 36575)
//	printf( "devAddArrays: pOne <%6.4e,%6.4e>, pTwo <%6.4e,%6.4e>, pOne' <%6.4e,%6.4e>\n", tmp.x, tmp.y, pTwo[ i ].x, pTwo[ i ].y, pOne[ i ].x, pOne[ i ].y );
	}

} // devAddArrays

__global__ void devAddArrays( cufftDoubleComplex * pOne, cufftDoubleComplex * pTwo, int pSize )
{
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	// check that we are within the array bounds.
	if (i < pSize)
	{
//cufftDoubleComplex tmp = pOne[ i ];
		cufftDoubleComplex addition = pTwo[ i ];
		pOne[ i ].x += addition.x;
		pOne[ i ].y += addition.y;
//if (i == 36575)
//	printf( "devAddArrays: pOne <%6.4e,%6.4e>, pTwo <%6.4e,%6.4e>, pOne' <%6.4e,%6.4e>\n", tmp.x, tmp.y, pTwo[ i ].x, pTwo[ i ].y, pOne[ i ].x, pOne[ i ].y );
	}

} // devAddArrays

__global__ void devAddArrays( float * pOne, float * pTwo, int pSize )
{
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	// check that we are within the array bounds.
	if (i < pSize)
		pOne[ i ] += pTwo[ i ];

} // devAddArrays

//
//	devApplyMask()
//
//	CJS: 24/11/2021
//
//	If any pixels are masked, set them to zero.
//

__global__ void devApplyMask
			(
			cufftComplex * pImage,				// the image to process
			bool * pMask,					// the pixel mask
			int pSize					// image size
			)
{
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	// check that we haven't gone outside the grid dimensions of the image (some threads will).
	if ( i < pSize && j < pSize )
		if (pMask[ (j * pSize) + i ] == false)
		{
			pImage[ (j * pSize) + i ].x = 0.0;
			pImage[ (j * pSize) + i ].y = 0.0;
		}
	
} // devUpperThreshold

//
//	devConvertImages()
//
//	CJS: 14/02/2022
//
//	Convert all the cells of an image from one format to another.
//

__global__ void devConvertImage
			(
			cufftComplex * pOut,
			cufftDoubleComplex * pIn,
			int pSize
			)
{
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	// check that we haven't gone outside the grid dimensions of image one (some threads will).
	if ( i < pSize && j < pSize )
	{
		pOut[ (j * pSize) + i ].x = (float) pIn[ (j * pSize) + i ].x;
		pOut[ (j * pSize) + i ].y = (float) pIn[ (j * pSize) + i ].y;
	}

} // devConvertImage

__global__ void devConvertImage
			(
			cufftComplex * pOut,
			float * pIn,
			int pSize
			)
{
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	// check that we haven't gone outside the grid dimensions of image one (some threads will).
	if ( i < pSize && j < pSize )
	{
		pOut[ (j * pSize) + i ].x = pIn[ (j * pSize) + i ];
		pOut[ (j * pSize) + i ].y = 0.0;
	}

} // devConvertImage

__global__ void devConvertImage
			(
			float * pOut,
			cufftDoubleComplex * pIn,
			int pSize
			)
{
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	// check that we haven't gone outside the grid dimensions of image one (some threads will).
	if ( i < pSize && j < pSize )
		pOut[ (j * pSize) + i ] = pIn[ (j * pSize) + i ].x;

} // devConvertImage

//
//	devDivideImages()
//
//	CJS: 18/01/2016
//
//	Divide one image by another, possibly of a different size.
//

__global__ void devDivideImages( float * pOne, float * pTwo, bool * pMask, int pSizeOne, int pSizeTwo )
{
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	// check that we haven't gone outside the grid dimensions of image one (some threads will).
	if ( i < pSizeOne && j < pSizeOne )
	{

		// ensure this pixel isn't masked.
		bool masked = false;
		if (pMask != NULL)
			masked = (pMask[ (j * pSizeOne) + i ] == false);
		if (masked == false)
		{

			// calculate position in image two.			
			int iTwo = (int) ((double) i * (double) pSizeTwo / (double) pSizeOne);
			int jTwo = (int) ((double) j * (double) pSizeTwo / (double) pSizeOne);
			
			// divide images.
			float two = pTwo[ (jTwo * pSizeTwo) + iTwo ];
			if (two != 0.0)
				pOne[ (j * pSizeOne) + i ] /= two;
			else
				pOne[ (j * pSizeOne) + i ] = 0.0;

		}
		else
			pOne[ (j * pSizeOne) + i ] = 0.0;
		
	}
	
} // devDivideImages

__global__ void devDivideImages( cufftComplex * pOne, cufftComplex * pTwo, bool * pMask, int pSizeOne, int pSizeTwo )
{
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	// check that we haven't gone outside the grid dimensions of image one (some threads will).
	if ( i < pSizeOne && j < pSizeOne )
	{

		// ensure this pixel isn't masked.
		bool masked = false;
		if (pMask != NULL)
			masked = (pMask[ (j * pSizeOne) + i ] == false);
		if (masked == false)
		{
	
			// calculate position in image two.
			int iTwo = (int) ((double) i * (double) pSizeTwo / (double) pSizeOne);
			int jTwo = (int) ((double) j * (double) pSizeTwo / (double) pSizeOne);
		
			// divide images.
			pOne[ (j * pSizeOne) + i ] = divideComplex( /* pOne = */ pOne[ (j * pSizeOne) + i ], /* pTwo = */ pTwo[ (jTwo * pSizeTwo) + iTwo ] );

		}
		else
		{
			pOne[ (j * pSizeOne) + i ].x = 0.0;
			pOne[ (j * pSizeOne) + i ].y = 0.0;
		}
		
	}
	
} // devDivideImages

__global__ void devDivideImages( cufftDoubleComplex * pOne, float * pTwo, bool * pMask, int pSizeOne, int pSizeTwo )
{
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	// check that we haven't gone outside the grid dimensions of image one (some threads will).
	if ( i < pSizeOne && j < pSizeOne )
	{

		// ensure this pixel isn't masked.
		bool masked = false;
		if (pMask != NULL)
			masked = (pMask[ (j * pSizeOne) + i ] == false);
		if (masked == false)
		{
	
			// calculate position in image two.
			int iTwo = (int) ((double) i * (double) pSizeTwo / (double) pSizeOne);
			int jTwo = (int) ((double) j * (double) pSizeTwo / (double) pSizeOne);
		
			// divide images.
			double two = (double) pTwo[ (jTwo * pSizeTwo) + iTwo ];
			if (two != 0.0)
			{
				pOne[ (j * pSizeOne) + i ].x /= two;
				pOne[ (j * pSizeOne) + i ].y /= two;
			}
			else
			{
				pOne[ (j * pSizeOne) + i ].x = 0.0;
				pOne[ (j * pSizeOne) + i ].y = 0.0;
			}

		}
		else
		{
			pOne[ (j * pSizeOne) + i ].x = 0.0;
			pOne[ (j * pSizeOne) + i ].y = 0.0;
		}
		
	}
	
} // devDivideImages

__global__ void devDivideImages( cufftComplex * pOne, float * pTwo, bool * pMask, int pSizeOne, int pSizeTwo )
{
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	// check that we haven't gone outside the grid dimensions of image one (some threads will).
	if ( i < pSizeOne && j < pSizeOne )
	{

		// ensure this pixel isn't masked.
		bool masked = false;
		if (pMask != NULL)
			masked = (pMask[ (j * pSizeOne) + i ] == false);
		if (masked == false)
		{
	
			// calculate position in image two.
			int iTwo = (int) ((double) i * (double) pSizeTwo / (double) pSizeOne);
			int jTwo = (int) ((double) j * (double) pSizeTwo / (double) pSizeOne);
		
			// divide images.
			float two = pTwo[ (jTwo * pSizeTwo) + iTwo ];
			if (two != 0.0)
			{
				pOne[ (j * pSizeOne) + i ].x /= two;
				pOne[ (j * pSizeOne) + i ].y /= two;
			}
			else
			{
				pOne[ (j * pSizeOne) + i ].x = 0.0;
				pOne[ (j * pSizeOne) + i ].y = 0.0;
			}

		}
		else
		{
			pOne[ (j * pSizeOne) + i ].x = 0.0;
			pOne[ (j * pSizeOne) + i ].y = 0.0;
		}
		
	}
	
} // devDivideImages

__global__ void devDivideImages( cufftDoubleComplex * pOne, cufftDoubleComplex * pTwo, bool * pMask, int pSizeOne, int pSizeTwo )
{
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	// check that we haven't gone outside the grid dimensions of image one (some threads will).
	if ( i < pSizeOne && j < pSizeOne )
	{

		// ensure this pixel isn't masked.
		bool masked = false;
		if (pMask != NULL)
			masked = (pMask[ (j * pSizeOne) + i ] == false);
		if (masked == false)
		{
	
			// calculate position in image two.
			int iTwo = (int) ((double) i * (double) pSizeTwo / (double) pSizeOne);
			int jTwo = (int) ((double) j * (double) pSizeTwo / (double) pSizeOne);
		
			// divide images.
			pOne[ (j * pSizeOne) + i ] = divideComplex( /* pOne = */ pOne[ (j * pSizeOne) + i ], /* pTwo = */ pTwo[ (jTwo * pSizeTwo) + iTwo ] );

		}
		else
		{
			pOne[ (j * pSizeOne) + i ].x = 0.0;
			pOne[ (j * pSizeOne) + i ].y = 0.0;
		}
		
	}
	
} // devDivideImages

__global__ void devDivideImages( float * pOne, cufftDoubleComplex * pTwo, bool * pMask, int pSizeOne, int pSizeTwo )
{
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	// check that we haven't gone outside the grid dimensions of image one (some threads will).
	if ( i < pSizeOne && j < pSizeOne )
	{

		// ensure this pixel isn't masked.
		bool masked = false;
		if (pMask != NULL)
			masked = (pMask[ (j * pSizeOne) + i ] == false);
		if (masked == false)
		{
	
			// calculate position in image two.
			int iTwo = (int) ((double) i * (double) pSizeTwo / (double) pSizeOne);
			int jTwo = (int) ((double) j * (double) pSizeTwo / (double) pSizeOne);
		
			// divide images.
			pOne[ (j * pSizeOne) + i ] /= pTwo[ (jTwo * pSizeTwo) + iTwo ].x;

		}
		else
		{
			pOne[ (j * pSizeOne) + i ] = 0.0;
			pOne[ (j * pSizeOne) + i ] = 0.0;
		}
		
	}
	
} // devDivideImages

__global__ void devDivideImages( cufftDoubleComplex * pOne, cufftComplex * pTwo, bool * pMask, int pSizeOne, int pSizeTwo )
{
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	// check that we haven't gone outside the grid dimensions of image one (some threads will).
	if ( i < pSizeOne && j < pSizeOne )
	{

		// ensure this pixel isn't masked.
		bool masked = false;
		if (pMask != NULL)
			masked = (pMask[ (j * pSizeOne) + i ] == false);
		if (masked == false)
		{
	
			// calculate position in image two.
			int iTwo = (int) ((double) i * (double) pSizeTwo / (double) pSizeOne);
			int jTwo = (int) ((double) j * (double) pSizeTwo / (double) pSizeOne);
		
			// divide images.
			pOne[ (j * pSizeOne) + i ] = divideComplex( /* pOne = */ pOne[ (j * pSizeOne) + i ], /* pTwo = */ pTwo[ (jTwo * pSizeTwo) + iTwo ] );

		}
		else
		{
			pOne[ (j * pSizeOne) + i ].x = 0.0;
			pOne[ (j * pSizeOne) + i ].y = 0.0;
		}
		
	}
	
} // devDivideImages

//
//	devFFTShift()
//
//	CJS: 24/09/2015
//
//	Perform an FFT shift on the data following the FFT operation.
//

__global__ void devFFTShift
			(
			cufftComplex * pDestination,			// the output image
			cufftComplex * pSource,			// the input image
			fftdirection pFFTDirection,			// FORWARD or INVERSE
			int pSize					// the size of the image
			)
{
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	int pivotOne = (int) floor( (double) pSize / 2.0 );
	int pivotTwo = (int) ceil( (double) pSize / 2.0 );
	int iFrom, jFrom;
	
	// if we are doing an inverse FFT shift then swap the pivots.
	if (pFFTDirection == INVERSE)
	{
		int tmp = pivotOne;
		pivotOne = pivotTwo;
		pivotTwo = tmp;
	}
	
	// calculate which pixel we should take our value from.
	if (i < pivotTwo)
		iFrom = i + pivotOne;
	else
		iFrom = i - pivotTwo;
	if (j < pivotTwo)
		jFrom = j + pivotOne;
	else
		jFrom = j - pivotTwo;
	
	// if we are within the bounds of the array, do FFT shift.
	if (i >= 0 && i < pSize && j >= 0 && j < pSize)
		memcpy( &pDestination[ (j * pSize) + i ], &pSource[ (jFrom * pSize) + iFrom ], sizeof( cufftComplex ) );
	
} // devFFTShift

__global__ void devFFTShift
			(
			double * pDestination,				// the output image
			cufftComplex * pSource,			// the input image
			fftdirection pFFTDirection,			// FORWARD or INVERSE
			int pSize					// the size of the image
			)
{
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	int pivotOne = (int) floor( (double) pSize / 2.0 );
	int pivotTwo = (int) ceil( (double) pSize / 2.0 );
	int iFrom, jFrom;
	
	// if we are doing an inverse FFT shift then swap the pivots.
	if (pFFTDirection == INVERSE)
	{
		int tmp = pivotOne;
		pivotOne = pivotTwo;
		pivotTwo = tmp;
	}
	
	// calculate which pixel we should take our value from.
	if (i < pivotTwo)
		iFrom = i + pivotOne;
	else
		iFrom = i - pivotTwo;
	if (j < pivotTwo)
		jFrom = j + pivotOne;
	else
		jFrom = j - pivotTwo;
	
	// if we are within the bounds of the array, do FFT shift.
	if (i >= 0 && i < pSize && j >= 0 && j < pSize)
		pDestination[ (j * pSize) + i ] = pSource[ (jFrom * pSize) + iFrom ].x;
	
} // devFFTShift

__global__ void devFFTShift
			(
			float * pDestination,				// the output image
			cufftComplex * pSource,			// the input image
			fftdirection pFFTDirection,			// FORWARD or INVERSE
			int pSize					// the size of the image
			)
{
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	int pivotOne = (int) floor( (double) pSize / 2.0 );
	int pivotTwo = (int) ceil( (double) pSize / 2.0 );
	int iFrom, jFrom;
	
	// if we are doing an inverse FFT shift then swap the pivots.
	if (pFFTDirection == INVERSE)
	{
		int tmp = pivotOne;
		pivotOne = pivotTwo;
		pivotTwo = tmp;
	}
	
	// calculate which pixel we should take our value from.
	if (i < pivotTwo)
		iFrom = i + pivotOne;
	else
		iFrom = i - pivotTwo;
	if (j < pivotTwo)
		jFrom = j + pivotOne;
	else
		jFrom = j - pivotTwo;
	
	// if we are within the bounds of the array, do FFT shift.
	if (i >= 0 && i < pSize && j >= 0 && j < pSize)
		pDestination[ (j * pSize) + i ] = pSource[ (jFrom * pSize) + iFrom ].x;
	
} // devFFTShift


__global__ void devFFTShift
			(
			cufftComplex * pDestination,			// the output image
			float * pSource,				// the input image
			fftdirection pFFTDirection,			// FORWARD or INVERSE
			int pSize					// the size of the image
			)
{
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	int pivotOne = (int) floor( (double) pSize / 2.0 );
	int pivotTwo = (int) ceil( (double) pSize / 2.0 );
	int iFrom, jFrom;
	
	// if we are doing an inverse FFT shift then swap the pivots.
	if (pFFTDirection == INVERSE)
	{
		int tmp = pivotOne;
		pivotOne = pivotTwo;
		pivotTwo = tmp;
	}
	
	// calculate which pixel we should take our value from.
	if (i < pivotTwo)
		iFrom = i + pivotOne;
	else
		iFrom = i - pivotTwo;
	if (j < pivotTwo)
		jFrom = j + pivotOne;
	else
		jFrom = j - pivotTwo;
	
	// if we are within the bounds of the array, do FFT shift.
	if (i >= 0 && i < pSize && j >= 0 && j < pSize)
	{
		pDestination[ (j * pSize) + i ].x = pSource[ (jFrom * pSize) + iFrom ];
		pDestination[ (j * pSize) + i ].y = 0.0;
	}
	
} // devFFTShift

//
//	devFindCutoffPixelParallel()
//
//	CJS: 03/04/2019
//
//	Finds the furthest pixel from the centre of the kernel that is at least 1% of the maximum kernel value.
//

__global__ void devFindCutoffPixelParallel
			(
			cufftComplex * pKernel,			// the image to process
			int pSize,					// the size of the image
			double * pMaxValue,				// a pointer to the peak pixel value
			int pCellsPerThread,				// how many cells should we process with each CUDA thread?
			int * pTmpResults,				// we store the support calculated for each block in this array. they will be processed separately
			double pCutoffFraction,			// the fraction of the peak pixel value we should search for
			findpixel pFindType				// can be FURTHEST or CLOSEST, depending on whether we need the furthest pixel from the centre
									//	with value > X, or the closest pixel with a value < X.
			)
{
	
	// the dynamic memory area stores arrays of visibilities, samples and frequencies.
	extern __shared__ int shrSupport[];
	
	// get the starting cell index.
	long int cell = ((blockIdx.x * blockDim.x) + threadIdx.x) * (long int) pCellsPerThread;
	
	int bestSupport = -1;
	for ( long int index = cell; index < cell + pCellsPerThread; index++ )
	{
		
		// ensure we are within bounds.
		if (index < (long int) (pSize * pSize))
		{

			// set our search value to N% of the kernel maximum.
			double searchValue = pMaxValue[ MAX_PIXEL_VALUE ] * pCutoffFraction;

			// get coordinates and pixel value.
			int i = index % pSize;
			int j = index / pSize;
			float pixelValue = cuCabsf( pKernel[ index ] );
			if ((pixelValue >= searchValue && pCutoffFraction > -1) || (pixelValue < 0.0 && pCutoffFraction == -1))
			{

				// get the size of the kernel to this point.
				int supportX = abs( i - (pSize / 2) );
				int supportY = abs( j - (pSize / 2) );
				int support = supportX;
				if (supportY > supportX)
					support = supportY;

				// update the maximum value.
				if (((support > bestSupport) && pFindType == FURTHEST) || ((support < bestSupport) && pFindType == CLOSEST) || bestSupport == -1)
					bestSupport = support;

			}
		
		}
		
	} // LOOP: index
		
	// update maximum values.
	shrSupport[ threadIdx.x ] = bestSupport;
	
	__syncthreads();
	
	// now, get the maximum/minimum value from the shared array.
	if (threadIdx.x == 0)
	{
	
		int bestSupport = -1;
	
		for ( int i = 0; i < blockDim.x; i++ )
		{
		
			int support = shrSupport[ i ];
			if (support == -1 && pFindType == CLOSEST)
				support = pSize / 2;
			if (support == -1 && pFindType == FURTHEST)
				support = 0;
			
			// is this value greater than the previous greatest?
			if ((support > bestSupport && pFindType == FURTHEST) || (support < bestSupport && pFindType == CLOSEST) || i == 0)
				bestSupport = support;
			
		}
		
		// update global memory with these values.
		pTmpResults[ blockIdx.x ] = bestSupport;

	}

} // devFindCutoffPixelParallel

__global__ void devFindCutoffPixelParallel
			(
			float * pKernel,				// the image to process
			int pSize,					// the size of the image
			double * pMaxValue,				// a pointer to the peak pixel value
			int pCellsPerThread,				// how many cells should we process with each CUDA thread?
			int * pTmpResults,				// we store the support calculated for each block in this array. they will be processed separately
			double pCutoffFraction,			// the fraction of the peak pixel value we should search for
			findpixel pFindType				// can be FURTHEST or CLOSEST, depending on whether we need the furthest pixel from the centre
									//	with value > X, or the closest pixel with a value < X.
			)
{
	
	// the dynamic memory area stores arrays of visibilities, samples and frequencies.
	extern __shared__ int shrSupport[];
	
	// get the starting cell index.
	long int cell = ((blockIdx.x * blockDim.x) + threadIdx.x) * (long int) pCellsPerThread;
	
	int bestSupport = -1;
	for ( long int index = cell; index < cell + pCellsPerThread; index++ )
	{
		
		// ensure we are within bounds.
		if (index < (long int) (pSize * pSize))
		{

			// set our search value to N% of the kernel maximum.
			double searchValue = pMaxValue[ MAX_PIXEL_VALUE ] * pCutoffFraction;

			// get coordinates and pixel value.
			int i = index % pSize;
			int j = index / pSize;
			float pixelValue = pKernel[ index ];
			if ((pixelValue >= searchValue && pCutoffFraction > -1) || (pixelValue < 0.0 && pCutoffFraction == -1))
			{

				// get the size of the kernel to this point.
				int supportX = abs( i - (pSize / 2) );
				int supportY = abs( j - (pSize / 2) );
				int support = supportX;
				if (supportY > supportX)
					support = supportY;

				// update the maximum value.
				if (((support > bestSupport) && pFindType == FURTHEST) || ((support < bestSupport) && pFindType == CLOSEST) || bestSupport == -1)
					bestSupport = support;

			}
		
		}
		
	}
		
	// update maximum values.
	shrSupport[ threadIdx.x ] = bestSupport;
	
	__syncthreads();
	
	// now, get the maximum/minimum value from the shared array.
	if (threadIdx.x == 0)
	{
	
		int bestSupport = -1;
	
		for ( int i = 0; i < blockDim.x; i++ )
		{
		
			int support = shrSupport[ i ];
			if (support == -1 && pFindType == CLOSEST)
				support = pSize / 2;
			if (support == -1 && pFindType == FURTHEST)
				support = 0;
			
			// is this value greater than the previous greatest?
			if ((support > bestSupport && pFindType == FURTHEST) || (support < bestSupport && pFindType == CLOSEST) || bestSupport == -1)
				bestSupport = support;
			
		}
		
		// update global memory with these values.
		pTmpResults[ blockIdx.x ] = bestSupport;

	}

} // devFindCutoffPixelParallel

//
//	devFindCutoffPixel()
//
//	CJS: 02/04/2019
//
//	Finds the furthest pixel from the centre of the kernel that is at least X% of the maximum kernel value, or the closest pixel to the centre that is less
//	than X% of the maximum.
//

__global__ void devFindCutoffPixel
			(
			int * pTmpResults,				// the array of support sizes, one item for each block that was processed
			int * pSupport,				// the final support size to return
			int pElements,					// the number of elements in the array
			findpixel pFindType				// can be FURTHEST or CLOSEST, depending on whether we need the furthest pixel from the centre
									//	with value > X, or the closest pixel with a value < X.
			)
{
	
	int bestSupport = -1;
	
	// get maximum value.
	for ( int i = 0; i < pElements; i++ )
	{
			
		// get the support.
		int support = pTmpResults[ i ];
			
		// is this value greater than the previous greatest?
		if (((support > bestSupport) && pFindType == FURTHEST) || ((support < bestSupport) && pFindType == CLOSEST) || i == 0)
			bestSupport = support;
			
	}

	// update maximum support.
	*pSupport = bestSupport;

} // devFindCutoffPixel

//
//	devGetMaxValue()
//
//	CJS: 06/11/2015
//
//	Get the maximum value from a 1d array, and store this number along with the x and y coordinates.
//

__global__ void devGetMaxValue
			(
			double * pArray,				// the array to search
			double * pMaxValue,				// the data area to store details of the maximum value
			bool pUseAbsolute,				// true if we should take the absolute value of each pixel, rather than include negative numbers
			int pElements					// the number of elements
			)
{
	
	double maxValue = 0;
	double maxI = 0;
	double maxJ = 0;
	double maxValueReal = 0;
	double maxValueImag = 0;
	
	// get maximum value.
	for ( int i = 0; i < pElements; i++ )
	{
			
		// get the value.
		double value = pArray[ MAX_PIXEL_VALUE ];
			
		// is this value greater than the previous greatest?
		if ((value > maxValue && pUseAbsolute == false) || (abs( value ) > abs( maxValue ) && pUseAbsolute == true))
		{
			maxValue = value;
			maxI = pArray[ MAX_PIXEL_X ];
			maxJ = pArray[ MAX_PIXEL_Y ];
			maxValueReal = pArray[ MAX_PIXEL_REAL ];
			maxValueImag = pArray[ MAX_PIXEL_IMAG ];
		}
		pArray = pArray + MAX_PIXEL_DATA_AREA_SIZE;
			
	}

	// update maximum values.
	pMaxValue[ MAX_PIXEL_VALUE ] = maxValue;
	pMaxValue[ MAX_PIXEL_X ] = maxI;
	pMaxValue[ MAX_PIXEL_Y ] = maxJ;
	pMaxValue[ MAX_PIXEL_REAL ] = maxValueReal;
	pMaxValue[ MAX_PIXEL_IMAG ] = maxValueImag;
	
} // devGetMaxValue

//
//	devGetMaxValueParallel()
//
//	CJS: 06/11/2015
//
//	Get the maximum complex number from an 2-d array, and store this number along with the x and y coordinates.
//
//	Uses the absolute value if pIncludeComplexComponent == true, else only uses the real value.
//

__global__ void devGetMaxValueParallel
			(
			cufftComplex * pArray,				// the array to search
			int pWidth,					// the array width
			int pHeight,					// the array height
			int pCellsPerThread,				// the number of array cells we should search per thread
			double * pBlockMax,				// an array that stores the maximum values per block
			bool pIncludeComplexComponent,		// true if we should use the magnitude of the complex number, false if we just use the real value
			bool pMultiplyByConjugate,			// true if each pixel should be multiplied by its own complex conjugate
			bool * pMask					// an optional mask that restricts which pixels we search
			)
{
	
	// the dynamic memory area stores the maximum value found for each thread.
	extern __shared__ double shrMaxValue[];
	
	double maxValue = 0;
	double maxI = 0;
	double maxJ = 0;
	double maxValueReal = 0;
	double maxValueImag = 0;
	
	// get the starting cell index.
	int cell = ((blockIdx.x * blockDim.x) + threadIdx.x) * pCellsPerThread;
	
	for ( int i = cell; i < cell + pCellsPerThread; i++ )
	{
		
		// ensure we are within bounds.
		if (i < pWidth * pHeight)
		{
		
			float value = 0;
			if (pMultiplyByConjugate == false && pIncludeComplexComponent == true)
				value = cuCabsf( pArray[ i ] );
			else if (pMultiplyByConjugate == true && pIncludeComplexComponent == true)
			{
				cufftComplex v1 = pArray[ i ];
				cufftComplex v2 = conjugate( v1 );
				value = cuCabsf( multComplex( v1, v2 ) );
			}
			else if (pMultiplyByConjugate == true && pIncludeComplexComponent == false)
			{
				cufftComplex v1 = pArray[ i ];
				cufftComplex v2 = conjugate( v1 );
				value = multComplex( v1, v2 ).x;
			}
			else if (pMultiplyByConjugate == false && pIncludeComplexComponent == false)
				value = pArray[ i ].x;

			// has this cell been masked? we can include it if there is no mask provided, or if the mask is TRUE (i.e. cell is good).
			bool includeCell = (pMask == NULL);
			if (pMask != NULL)
				includeCell = (pMask[ i ] == true);
			
			// is this value greater than the previous greatest?
			if (value > maxValue && includeCell == true)
			{
				maxValue = value;
				maxI = (double) (i % pWidth);
				maxJ = i / pWidth;
				maxValueReal = (double) pArray[ i ].x;
				maxValueImag = (double) pArray[ i ].y;
			}
		
		}
		
	}
		
	// update maximum values.
	shrMaxValue[ (threadIdx.x * MAX_PIXEL_DATA_AREA_SIZE) + MAX_PIXEL_VALUE ] = maxValue;
	shrMaxValue[ (threadIdx.x * MAX_PIXEL_DATA_AREA_SIZE) + MAX_PIXEL_X ] = maxI;
	shrMaxValue[ (threadIdx.x * MAX_PIXEL_DATA_AREA_SIZE) + MAX_PIXEL_Y ] = maxJ;
	shrMaxValue[ (threadIdx.x * MAX_PIXEL_DATA_AREA_SIZE) + MAX_PIXEL_REAL ] = maxValueReal;
	shrMaxValue[ (threadIdx.x * MAX_PIXEL_DATA_AREA_SIZE) + MAX_PIXEL_IMAG ] = maxValueImag;
	
	__syncthreads();
	
	// now, get the maximum value from the shared array.
	if (threadIdx.x == 0)
	{
	
		double maxValue = 0;
		double maxI = 0;
		double maxJ = 0;
		double maxValueReal = 0;
		double maxValueImag = 0;
	
		for ( int i = 0; i < blockDim.x; i++ )
		{
		
			double value = shrMaxValue[ i * MAX_PIXEL_DATA_AREA_SIZE ];
			
			// is this value greater than the previous greatest?
			if (value > maxValue)
			{
				maxValue = value;
				maxI = shrMaxValue[ (i * MAX_PIXEL_DATA_AREA_SIZE) + 1 ];
				maxJ = shrMaxValue[ (i * MAX_PIXEL_DATA_AREA_SIZE) + 2 ];
				maxValueReal = shrMaxValue[ (i * MAX_PIXEL_DATA_AREA_SIZE) + 3 ];
				maxValueImag = shrMaxValue[ (i * MAX_PIXEL_DATA_AREA_SIZE) + 4 ];
			}
			
		}
		
		// update global memory with these values.
		pBlockMax[ (blockIdx.x * MAX_PIXEL_DATA_AREA_SIZE) + MAX_PIXEL_VALUE ] = maxValue;
		pBlockMax[ (blockIdx.x * MAX_PIXEL_DATA_AREA_SIZE) + MAX_PIXEL_X ] = maxI;
		pBlockMax[ (blockIdx.x * MAX_PIXEL_DATA_AREA_SIZE) + MAX_PIXEL_Y ] = maxJ;
		pBlockMax[ (blockIdx.x * MAX_PIXEL_DATA_AREA_SIZE) + MAX_PIXEL_REAL ] = maxValueReal;
		pBlockMax[ (blockIdx.x * MAX_PIXEL_DATA_AREA_SIZE) + MAX_PIXEL_IMAG ] = maxValueImag;
	
	}
	
} // devGetMaxValueParallel

__global__ void devGetMaxValueParallel
			(
			float * pArray,				// the array to search
			int pWidth,					// the array width
			int pHeight,					// the array height
			int pCellsPerThread,				// the number of array cells we should search per thread
			double * pBlockMax,				// an array that stores the maximum values per block
			bool pUseAbsolute,				// true if we should take the absolute value of each pixel, rather than include negative numbers
			bool * pMask					// an optional mask that restricts which pixels we search
			)
{
	
	// the dynamic memory area stores arrays of visibilities, samples and frequencies.
	extern __shared__ double shrMaxValue[];
	
	double maxValue = 0;
	double maxI = 0;
	double maxJ = 0;
	
	// get the starting cell index.
	int cell = ((blockIdx.x * blockDim.x) + threadIdx.x) * pCellsPerThread;
	
	for ( int i = cell; i < cell + pCellsPerThread; i++ )
	{
		
		// ensure we are within bounds.
		if (i < pWidth * pHeight)
		{
		
			float value = pArray[ i ];

			// has this cell been masked? we can include it if there is no mask provided, or if the mask is TRUE (i.e. cell is good).
			bool includeCell = (pMask == NULL);
			if (pMask != NULL)
				includeCell = (pMask[ i ] == true);
			
			// is this value greater than the previous greatest?
			if (((value > maxValue && pUseAbsolute == false) || (abs( value ) > abs( maxValue ) && pUseAbsolute == true)) && includeCell == true)
			{
				maxValue = (double) value;
				maxI = (double) (i % pWidth);
				maxJ = i / pWidth;
			}
		
		}
		
	}
		
	// update maximum values.
	shrMaxValue[ (threadIdx.x * MAX_PIXEL_DATA_AREA_SIZE) + MAX_PIXEL_VALUE ] = maxValue;
	shrMaxValue[ (threadIdx.x * MAX_PIXEL_DATA_AREA_SIZE) + MAX_PIXEL_X ] = maxI;
	shrMaxValue[ (threadIdx.x * MAX_PIXEL_DATA_AREA_SIZE) + MAX_PIXEL_Y ] = maxJ;
	shrMaxValue[ (threadIdx.x * MAX_PIXEL_DATA_AREA_SIZE) + MAX_PIXEL_REAL ] = maxValue;
	shrMaxValue[ (threadIdx.x * MAX_PIXEL_DATA_AREA_SIZE) + MAX_PIXEL_IMAG ] = 0;
	
	__syncthreads();
	
	// now, get the maximum value from the shared array.
	if (threadIdx.x == 0)
	{
	
		double maxValue = 0;
		double maxI = 0;
		double maxJ = 0;
		double maxValueReal = 0;
		double maxValueImag = 0;
	
		for ( int i = 0; i < blockDim.x; i++ )
		{
		
			double value = shrMaxValue[ (i * MAX_PIXEL_DATA_AREA_SIZE) + MAX_PIXEL_VALUE ];
			
			// is this value greater than the previous greatest?
			if ((value > maxValue && pUseAbsolute == false) || (abs( value ) > abs( maxValue ) && pUseAbsolute == true))
			{
				maxValue = value;
				maxI = shrMaxValue[ (i * MAX_PIXEL_DATA_AREA_SIZE) + MAX_PIXEL_X ];
				maxJ = shrMaxValue[ (i * MAX_PIXEL_DATA_AREA_SIZE) + MAX_PIXEL_Y ];
				maxValueReal = shrMaxValue[ (i * MAX_PIXEL_DATA_AREA_SIZE) + MAX_PIXEL_REAL ];
				maxValueImag = shrMaxValue[ (i * MAX_PIXEL_DATA_AREA_SIZE) + MAX_PIXEL_IMAG ];
			}
			
		}
		
		// update global memory with these values.
		pBlockMax[ (blockIdx.x * MAX_PIXEL_DATA_AREA_SIZE) + MAX_PIXEL_VALUE ] = maxValue;
		pBlockMax[ (blockIdx.x * MAX_PIXEL_DATA_AREA_SIZE) + MAX_PIXEL_X ] = maxI;
		pBlockMax[ (blockIdx.x * MAX_PIXEL_DATA_AREA_SIZE) + MAX_PIXEL_Y ] = maxJ;
		pBlockMax[ (blockIdx.x * MAX_PIXEL_DATA_AREA_SIZE) + MAX_PIXEL_REAL ] = maxValueReal;
		pBlockMax[ (blockIdx.x * MAX_PIXEL_DATA_AREA_SIZE) + MAX_PIXEL_IMAG ] = maxValueImag;
	
	}
	
} // devGetMaxValueParallel

//
//	devGetPrimaryBeam()
//
//	CJS: 16/02/2022
//
//	Extracts the primary beam from a 4x4 Mueller matrix (for XX, XY, YX, and YY).
//

__global__ void devGetPrimaryBeam
			(
			float * pPrimaryBeam,				// the output primary beam.
			cufftDoubleComplex ** pMueller,		// the 4x4 Mueller matrix.
			int pImageSize,				// the image size
			int pStokes					// the Stokes image type (I, Q, U, or V)
			)
{
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;

	// ensure we are within the image.
	if (i < pImageSize && j < pImageSize)
	{
	
		int index = (j * pImageSize) + i;
		float returnValue = 0.0;
	
		if (pStokes == STOKES_I && pMueller[ /* CELL */ 0 ] != NULL && pMueller[ /* CELL */ 15 ] != NULL)
			returnValue = (pMueller[ /* CELL */ 0 ][ index ].x + pMueller[ /* CELL */ 15 ][ index ].x) / 2.0;
			
		if (pStokes == STOKES_Q && pMueller[ /* CELL */ 0 ] != NULL && pMueller[ /* CELL */ 15 ] != NULL)
			returnValue = (pMueller[ /* CELL */ 0 ][ index ].x - pMueller[ /* CELL */ 15 ][ index ].x) / 2.0;
			
		if (pStokes == STOKES_U && pMueller[ /* CELL */ 5 ] != NULL && pMueller[ /* CELL */ 10 ] != NULL)
			returnValue = -(pMueller[ /* CELL */ 5 ][ index ].y + pMueller[ /* CELL */ 10 ][ index ].y) / 2.0;
			
		if (pStokes == STOKES_V && pMueller[ /* CELL */ 5 ] != NULL && pMueller[ /* CELL */ 10 ] != NULL)
			returnValue = -(pMueller[ /* CELL */ 10 ][ index ].y - pMueller[ /* CELL */ 5 ][ index ].y) / 2.0;
		
		// return the value for this pixel.
		pPrimaryBeam[ index ] = returnValue;
	
	}

} // devGetPrimaryBeam

//
//	devMakeBeam()
//
//	CJS: 06/11/2015
//
//	Constructs a Gaussian beam from a set of parameters.
//

__global__ void devMakeBeam
			(
			float * pBeam,					// the output clean beam
			double pAngle,					// the rotation angle of the psf
			double pR1,					// the length of axis 1
			double pR2,					// the length of axis 2
			double pX,					// }- centre of Gaussian
			double pY,					// }
			int pSize					// the size of the clean beam
			)
{
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	// check that we haven't gone outside the grid dimensions (some threads will).
	if ( i < pSize && j < pSize )
		
		// populate Gaussian image.
		pBeam[ (j * pSize) + i ] = (float) gaussian2D(	/* pNormalisation = */ 1.0,
									/* pX = */ (double) i - pX,
									/* pY = */ (double) j - pY,
									/* pAngle = */ pAngle,
									/* pR1 = */ pR1,
									/* pR2 = */ pR2 );
	
} // devMakeBeam

__global__ void devMakeBeam
			(
			cufftComplex * pBeam,				// the output clean beam
			double pAngle,					// the rotation angle of the psf
			double pR1,					// the length of axis 1
			double pR2,					// the length of axis 2
			double pX,					// }- centre of Gaussian
			double pY,					// }
			int pSize					// the size of the clean beam
			)
{
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	// check that we haven't gone outside the grid dimensions (some threads will).
	if ( i < pSize && j < pSize )
		
		// populate Gaussian image.
		pBeam[ (j * pSize) + i ].x = (float) gaussian2D(	/* pNormalisation = */ 1.0,
									/* pX = */ (double) i - pX,
									/* pY = */ (double) j - pY,
									/* pAngle = */ pAngle,
									/* pR1 = */ pR1,
									/* pR2 = */ pR2 );
		pBeam[ (j * pSize) + i ].y = 0.0;
	
} // devMakeBeam

//
//	devMoveImages()
//
//	CJS: 23/11/2021
//
//	Move one image into another, changing the type.
//

__global__ void devMoveImages
			(
			cufftDoubleComplex * pOne,
			cufftComplex * pTwo,
			int pSize
			)
{
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	// check that we haven't gone outside the grid dimensions of image one (some threads will).
	if ( i < pSize && j < pSize )
	{
		pOne[ (j * pSize) + i ].x = (double) pTwo[ (j * pSize) + i ].x;
		pOne[ (j * pSize) + i ].y = (double) pTwo[ (j * pSize) + i ].y;
	}

} // devMoveImages

//
//	devMultiplyArrays()
//
//	CJS: 18/01/2016
//
//	Multiply two arrays together. Some of these functions are 1D, and some are 2D.
//

__global__ void devMultiplyArrays( cufftComplex * pOne, int * pTwo, int pSize )
{

	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	// ensure we are within the array batch limits.
	if (index < pSize)
	{

		// get the 2nd value at this grid position.
		double two = (double) pTwo[ index ];

		// multiply the 1st value by the 2nd value.
		pOne[ index ].x *= two;
		pOne[ index ].y *= two;

	}

} // devMultiplyArrays

__global__ void devMultiplyArrays( cufftComplex * pOne, float * pTwo, int pSize )
{

	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	// ensure we are within the array batch limits.
	if (index < pSize)
	{

		// get the 2nd value at this grid position.
		double two = (double) pTwo[ index ];

		// multiply the 1st value by the 2nd value.
		pOne[ index ].x *= two;
		pOne[ index ].y *= two;

	}

} // devMultiplyArrays

__global__ void devMultiplyArrays( cufftComplex * pOne, cufftComplex * pTwo, int pSize, bool pConjugate )
{
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	// check that we haven't gone outside the grid dimensions (some threads will).
	if ( i < pSize && j < pSize )
		
		// multiply images together.
		if (pConjugate == false)
			pOne[ (j * pSize) + i ] = multComplex( /* pOne = */ pOne[ (j * pSize) + i ], /* pTwo = */ pTwo[ (j * pSize) + i ] );
		else
			pOne[ (j * pSize) + i ] = multComplex( /* pOne = */ pOne[ (j * pSize) + i ], /* pTwo = */ conjugate( pTwo[ (j * pSize) + i ] ) );
	
} // devMultiplyArrays

__global__ void devMultiplyArrays( float * pOne, float * pTwo, int pSize )
{
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	// check that we haven't gone outside the grid dimensions (some threads will).
	if ( i < pSize && j < pSize )
	{
		
		// multiply images together.
		pOne[ (j * pSize) + i ] *= pTwo[ (j * pSize) + i ];
		
	}
	
} // devMultiplyArrays

__global__ void devMultiplyArrays( float * pOne, float * pTwo, bool * pMask, int pSizeOne, int pSizeTwo )
{
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	// check that we haven't gone outside the grid dimensions of image one (some threads will).
	if ( i < pSizeOne && j < pSizeOne )
	{

		// ensure this pixel isn't masked.
		bool masked = false;
		if (pMask != NULL)
			masked = (pMask[ (j * pSizeOne) + i ] == false);
		if (masked == false)
		{

			// calculate position in image two.
			int iTwo = (int) ((double) i * (double) pSizeTwo / (double) pSizeOne);
			int jTwo = (int) ((double) j * (double) pSizeTwo / (double) pSizeOne);
		
			// multiply
			pOne[ (j * pSizeOne) + i ] *= pTwo[ (jTwo * pSizeTwo) + iTwo ];

		}
		else
			pOne[ (j * pSizeOne) + i ] = 0.0;
		
	}
	
} // devMultiplyArrays

//
//	devMultiplyImages()
//
//	CJS: 06/08/2021
//
//	Multiply one image by another, possibly of a different size.
//

__global__ void devMultiplyImages
			(
			float * pOne,					// image one
			float * pTwo,					// image two
			bool * pMask,					// image mask
			int pSizeOne,					// }- image sizes (we scale image two to be the same size as image one before multiplying)
			int pSizeTwo					// }
			)
{
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	// check that we haven't gone outside the grid dimensions of image one (some threads will).
	if ( i < pSizeOne && j < pSizeOne )
	{

		// ensure this pixel isn't masked.
		bool masked = false;
		if (pMask != NULL)
			masked = (pMask[ (j * pSizeOne) + i ] == false);
		if (masked == false)
		{

			// calculate position in image two.
			int iTwo = (int) ((double) i * (double) pSizeTwo / (double) pSizeOne);
			int jTwo = (int) ((double) j * (double) pSizeTwo / (double) pSizeOne);
		
			// multiply images.
			pOne[ (j * pSizeOne) + i ] *= pTwo[ (jTwo * pSizeTwo) + iTwo ];

		}
		else
			pOne[ (j * pSizeOne) + i ] = 0.0;
		
	}
	
} // devMultiplyImages

__global__ void devMultiplyImages
			(
			cufftComplex * pOne,				// image one
			cufftComplex * pTwo,				// image two
			bool * pMask,					// image mask
			int pSizeOne,					// }- image sizes (we scale image two to be the same size as image one before multiplying)
			int pSizeTwo					// }
			)
{
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	// check that we haven't gone outside the grid dimensions of image one (some threads will).
	if ( i < pSizeOne && j < pSizeOne )
	{

		// ensure this pixel isn't masked.
		bool masked = false;
		if (pMask != NULL)
			masked = (pMask[ (j * pSizeOne) + i ] == false);
		if (masked == false)
		{

			// calculate position in image two.
			int iTwo = (int) ((double) i * (double) pSizeTwo / (double) pSizeOne);
			int jTwo = (int) ((double) j * (double) pSizeTwo / (double) pSizeOne);
		
			// multiply images.
			pOne[ (j * pSizeOne) + i ] = multComplex( /* pOne = */ pOne[ (j * pSizeOne) + i ], /* pTwo = */ pTwo[ (jTwo * pSizeTwo) + iTwo ] );

		}
		else
		{
			pOne[ (j * pSizeOne) + i ].x = 0.0;
			pOne[ (j * pSizeOne) + i ].y = 0.0;
		}
		
	}
	
} // devMultiplyImages

__global__ void devMultiplyImages
			(
			cufftComplex * pOne,				// image one
			float * pTwo,				// image two
			bool * pMask,					// image mask
			int pSizeOne,					// }- image sizes (we scale image two to be the same size as image one before multiplying)
			int pSizeTwo					// }
			)
{
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	// check that we haven't gone outside the grid dimensions of image one (some threads will).
	if ( i < pSizeOne && j < pSizeOne )
	{

		// ensure this pixel isn't masked.
		bool masked = false;
		if (pMask != NULL)
			masked = (pMask[ (j * pSizeOne) + i ] == false);
		if (masked == false)
		{

			// calculate position in image two.
			int iTwo = (int) ((double) i * (double) pSizeTwo / (double) pSizeOne);
			int jTwo = (int) ((double) j * (double) pSizeTwo / (double) pSizeOne);
		
			// multiply images.
			float scalar = pTwo[ (jTwo * pSizeTwo) + iTwo ];
			pOne[ (j * pSizeOne) + i ].x *= scalar;
			pOne[ (j * pSizeOne) + i ].y *= scalar;

		}
		else
		{
			pOne[ (j * pSizeOne) + i ].x = 0.0;
			pOne[ (j * pSizeOne) + i ].y = 0.0;
		}
		
	}
	
} // devMultiplyImages

__global__ void devMultiplyImages
			(
			cufftDoubleComplex * pOne,			// image one
			cufftComplex * pTwo,				// image two
			bool * pMask,					// image mask
			int pSizeOne,					// }- image sizes (we scale image two to be the same size as image one before multiplying)
			int pSizeTwo					// }
			)
{
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	// check that we haven't gone outside the grid dimensions of image one (some threads will).
	if ( i < pSizeOne && j < pSizeOne )
	{

		// ensure this pixel isn't masked.
		bool masked = false;
		if (pMask != NULL)
			masked = (pMask[ (j * pSizeOne) + i ] == false);
		if (masked == false)
		{

			// calculate position in image two.
			int iTwo = (int) ((double) i * (double) pSizeTwo / (double) pSizeOne);
			int jTwo = (int) ((double) j * (double) pSizeTwo / (double) pSizeOne);
		
			// multiply images.
			pOne[ (j * pSizeOne) + i ] = multComplex( /* pOne = */ pOne[ (j * pSizeOne) + i ], /* pTwo = */ pTwo[ (jTwo * pSizeTwo) + iTwo ] );

		}
		else
		{
			pOne[ (j * pSizeOne) + i ].x = 0.0;
			pOne[ (j * pSizeOne) + i ].y = 0.0;
		}
		
	}
	
} // devMultiplyImages

__global__ void devMultiplyImages
			(
			cufftDoubleComplex * pOne,			// image one
			cufftDoubleComplex * pTwo,				// image two
			bool * pMask,					// image mask
			int pSizeOne,					// }- image sizes (we scale image two to be the same size as image one before multiplying)
			int pSizeTwo					// }
			)
{
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	// check that we haven't gone outside the grid dimensions of image one (some threads will).
	if ( i < pSizeOne && j < pSizeOne )
	{

		// ensure this pixel isn't masked.
		bool masked = false;
		if (pMask != NULL)
			masked = (pMask[ (j * pSizeOne) + i ] == false);
		if (masked == false)
		{

			// calculate position in image two.
			int iTwo = (int) ((double) i * (double) pSizeTwo / (double) pSizeOne);
			int jTwo = (int) ((double) j * (double) pSizeTwo / (double) pSizeOne);
		
			// multiply images.
			pOne[ (j * pSizeOne) + i ] = multComplex( /* pOne = */ pOne[ (j * pSizeOne) + i ], /* pTwo = */ pTwo[ (jTwo * pSizeTwo) + iTwo ] );

		}
		else
		{
			pOne[ (j * pSizeOne) + i ].x = 0.0;
			pOne[ (j * pSizeOne) + i ].y = 0.0;
		}
		
	}
	
} // devMultiplyImages

__global__ void devMultiplyImages
			(
			cufftComplex * pOne,				// image one
			float pScalar,					// constant value
			bool * pMask,					// image mask
			int pSizeOne					// }- image size (we scale image two to be the same size as image one before multiplying)
			)
{
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	// check that we haven't gone outside the grid dimensions of image one (some threads will).
	if ( i < pSizeOne && j < pSizeOne )
	{

		// ensure this pixel isn't masked.
		bool masked = false;
		if (pMask != NULL)
			masked = (pMask[ (j * pSizeOne) + i ] == false);
		if (masked == false)
		{
		
			// multiply images.
			pOne[ (j * pSizeOne) + i ].x *= pScalar;
			pOne[ (j * pSizeOne) + i ].y *= pScalar;

		}
		else
		{
			pOne[ (j * pSizeOne) + i ].x = 0.0;
			pOne[ (j * pSizeOne) + i ].y = 0.0;
		}
		
	}
	
} // devMultiplyImages

__global__ void devMultiplyImages
			(
			cufftDoubleComplex * pOne,			// image one
			float pScalar,					// constant value
			bool * pMask,					// image mask
			int pSizeOne					// }- image size (we scale image two to be the same size as image one before multiplying)
			)
{
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	// check that we haven't gone outside the grid dimensions of image one (some threads will).
	if ( i < pSizeOne && j < pSizeOne )
	{

		// ensure this pixel isn't masked.
		bool masked = false;
		if (pMask != NULL)
			masked = (pMask[ (j * pSizeOne) + i ] == false);
		if (masked == false)
		{
		
			// multiply images.
			pOne[ (j * pSizeOne) + i ].x *= pScalar;
			pOne[ (j * pSizeOne) + i ].y *= pScalar;

		}
		else
		{
			pOne[ (j * pSizeOne) + i ].x = 0.0;
			pOne[ (j * pSizeOne) + i ].y = 0.0;
		}
		
	}
	
} // devMultiplyImages

//
//	devNormalise()
//
//	CJS: 06/11/2015
//
//	Divide an array of complex numbers by a constant.
//

__global__ void devNormalise
			(
			cufftComplex * pArray,				// array to process
			double pConstant,				// the normalisation constant
			int pItems					// number of items
			)
{
	
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	// update the value, but only if we are within the bounds of the array.
	if (index < pItems)
	{
		pArray[ index ].x /= pConstant;
		pArray[ index ].y /= pConstant;
	}
	
} // devNormalise

__global__ void devNormalise
			(
			double * pArray,				// array to process
			double * pConstant,				// a pointer to the normalisation constant
			int pItems					// number of items
			)
{
	
	// store the constant in shared memory.
	__shared__ double constant;
	if (threadIdx.x == 0)
		constant = *pConstant;
	
	__syncthreads();
	
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	// update the value, but only if we are within the bounds of the array.
	if (index < pItems)
		pArray[ index ] /= constant;
	
} // devNormalise

__global__ void devNormalise
			(
			double * pArray,				// array to process
			double pConstant,				// normalisation constant
			int pItems					// number of items
			)
{
	
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	// update the value, but only if we are within the bounds of the array.
	if (index < pItems)
		pArray[ index ] /= pConstant;
	
} // devNormalise

__global__ void devNormalise
			(
			float * pArray,				// array to process
			double pConstant,				// normalisation constant
			int pItems					// number of items
			)
{
	
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	// update the value, but only if we are within the bounds of the array.
	if (index < pItems)
		pArray[ index ] /= pConstant;
	
} // devNormalise

__global__ void devNormalise
			(
			float * pArray,				// array to process
			double * pConstant,				// pointer to the normalisation constant
			int pItems					// number of items
			)
{
	
	// store the constant in shared memory.
	__shared__ double constant;
	if (threadIdx.x == 0)
		constant = *pConstant;
	
	__syncthreads();
	
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	// update the value, but only if we are within the bounds of the array.
	if (index < pItems)
		pArray[ index ] /= constant;
	
} // devNormalise

//
//	devResizeImage()
//
//	CJS: 29/03/2019
//
//	Copy one image into another image.
//

__global__ void devResizeImage( cufftComplex * pNewImage, cufftComplex * pOldImage, int pNewSize, int pOldSize )
{

	// calculate support.
	int oldSupport = pOldSize / 2;
	int newSupport = pNewSize / 2;

	// the thread indexes correspond to pixels in the new image, but we add an offset because we may only be interested in updating a small portion of the new image.
	int iNew = (blockIdx.x * blockDim.x) + threadIdx.x;
	int jNew = (blockIdx.y * blockDim.y) + threadIdx.y;

	// ensure we're within the bounds of the new image. we check against the old and new image size because the number of threads will always be the smallest image size.
	if (iNew >= 0 && iNew < pOldSize && iNew < pNewSize && jNew >= 0 && jNew < pOldSize && jNew < pNewSize)
	{

		int iOld = -1, jOld = -1;
		if (pNewSize > pOldSize)
		{
			iOld = iNew;
			jOld = jNew;
			iNew += newSupport - oldSupport;
			jNew += newSupport - oldSupport;
		}
		else
		{

			// calculate old pixel position.
			iOld = iNew - newSupport + oldSupport;
			jOld = jNew - newSupport + oldSupport;
		
		}

		// copy pixel from old image to new image.
		if (iOld >= 0 && iOld < pOldSize && jOld >= 0 && jOld < pOldSize )
			pNewImage[ (jNew * pNewSize) + iNew ] = pOldImage[ (jOld * pOldSize) + iOld ];

	}

} // devResizeImage

__global__ void devResizeImage( float * pNewImage, float * pOldImage, int pNewSize, int pOldSize )
{

	// calculate support.
	int oldSupport = pOldSize / 2;
	int newSupport = pNewSize / 2;

	// the thread indexes correspond to pixels in the new image, but we add an offset because we may only be interested in updating a small portion of the new image.
	int iNew = (blockIdx.x * blockDim.x) + threadIdx.x;
	int jNew = (blockIdx.y * blockDim.y) + threadIdx.y;

	// ensure we're within the bounds of the new image. we check against the old and new image size because the number of threads will always be the smallest image size.
	if (iNew >= 0 && iNew < pOldSize && iNew < pNewSize && jNew >= 0 && jNew < pOldSize && jNew < pNewSize)
	{

		if (pNewSize > pOldSize)
		{
			iNew += (pNewSize - pOldSize) / 2;
			jNew += (pNewSize - pOldSize) / 2;
		}

		// calculate old pixel position.
		int iOld = iNew - newSupport + oldSupport;
		int jOld = jNew - newSupport + oldSupport;

		// copy pixel from old image to new image.
		if (iOld >= 0 && iOld < pOldSize && jOld >= 0 && jOld < pOldSize )
			pNewImage[ (jNew * pNewSize) + iNew ] = pOldImage[ (jOld * pOldSize) + iOld ];

	}

} // devResizeImage

//
//	devReverseXDirection()
//
//	CJS: 24/06/2021
//
//	Reverse the x-direction of the image.
//

__global__ void devReverseXDirection( cufftComplex * pGrid, int pSize )
{
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	// check that we haven't gone outside the grid dimensions (some threads will).
	if ( i < (pSize / 2) && j < pSize )
	{
	
		// calculate cell indexes for swapping.
		long int indexOne = (j * pSize) + i;
		long int indexTwo = (j * pSize) + (pSize - i - 1);
		
		// reverse the x-axis.
		swap( pGrid[ indexOne ], pGrid[ indexTwo ] );
		
	}
	
} // devReverseXDirection

__global__ void devReverseXDirection( float * pGrid, int pSize )
{
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	// check that we haven't gone outside the grid dimensions (some threads will).
	if (i < (pSize / 2) && j < pSize)
	{
	
		// calculate cell indexes for swapping.
		long int indexOne = (j * pSize) + i;
		long int indexTwo = (j * pSize) + (pSize - i - 1);
		
		// reverse the x-axis.
		swap( pGrid[ indexOne ], pGrid[ indexTwo ] );
		
	}
	
} // devReverseXDirection

//
//	devReverseYDirection()
//
//	CJS: 04/10/2018
//
//	Reverse the y-direction of the image.
//

__global__ void devReverseYDirection( cufftComplex * pGrid, int pSize )
{
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	// check that we haven't gone outside the grid dimensions (some threads will).
	if ( i < pSize && j < (pSize / 2) )
	{
	
		// calculate cell indexes for swapping.
		long int indexOne = (j * pSize) + i;
		long int indexTwo = ((pSize - j - 1) * pSize) + i;
		
		// reverse the x-axis.
		swap( pGrid[ indexOne ], pGrid[ indexTwo ] );
		
	}
	
} // devReverseYDirection

__global__ void devReverseYDirection( float * pGrid, int pSize )
{
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	// check that we haven't gone outside the grid dimensions (some threads will).
	if (i < pSize && j < (pSize / 2))
	{
	
		// calculate cell indexes for swapping.
		long int indexOne = (j * pSize) + i;
		long int indexTwo = ((pSize - j - 1) * pSize) + i;
		
		// reverse the x-axis.
		swap( pGrid[ indexOne ], pGrid[ indexTwo ] );
		
	}
	
} // devReverseYDirection

//
//	devScaleImage()
//
//	CJS: 29/03/2019
//
//	Copy one image into another, differently size, image.
//
//	Preconditions:	The number of threads corresponds to the size of the new image.
//

__global__ void devScaleImage( cufftComplex * pNewImage, float * pOldImage, int pNewSize, int pOldSize, double pScale )
{

	// calculate support.
	int oldSupport = pOldSize / 2;
	int newSupport = pNewSize / 2;

	// the thread indexes correspond to pixels in the new image, but we add an offset because we may only be interested in updating a small portion of the new image.
	int iNew = (blockIdx.x * blockDim.x) + threadIdx.x;
	int jNew = (blockIdx.y * blockDim.y) + threadIdx.y;

	// ensure we're within the bounds of the new image.
	if (iNew >= 0 && iNew < pNewSize && jNew >= 0 && jNew < pNewSize)
	{

		// calculate old pixel position.
		int iOld = (int) ((double) (iNew - newSupport) * pScale) + oldSupport;
		int jOld = (int) ((double) (jNew - newSupport) * pScale) + oldSupport;

		// copy pixel from old image to new image.
		if (iOld >= 0 && iOld < pOldSize && jOld >= 0 && jOld < pOldSize )
		{
			pNewImage[ (jNew * pNewSize) + iNew ].x = pOldImage[ (jOld * pOldSize) + iOld ];
			pNewImage[ (jNew * pNewSize) + iNew ].y = 0.0;
		}

	}

} // devScaleImage

__global__ void devScaleImage( cufftComplex * pNewImage, cufftComplex * pOldImage, int pNewSize, int pOldSize, double pScale )
{

	// calculate support.
	int oldSupport = pOldSize / 2;
	int newSupport = pNewSize / 2;

	// the thread indexes correspond to pixels in the new image, but we add an offset because we may only be interested in updating a small portion of the new image.
	int iNew = (blockIdx.x * blockDim.x) + threadIdx.x;
	int jNew = (blockIdx.y * blockDim.y) + threadIdx.y;

	// ensure we're within the bounds of the new image.
	if (iNew >= 0 && iNew < pNewSize && jNew >= 0 && jNew < pNewSize)
	{

		// calculate old pixel position.
		int iOld = (int) ((double) (iNew - newSupport) * pScale) + oldSupport;
		int jOld = (int) ((double) (jNew - newSupport) * pScale) + oldSupport;

		// copy pixel from old image to new image.
		if (iOld >= 0 && iOld < pOldSize && jOld >= 0 && jOld < pOldSize )
		{
			pNewImage[ (jNew * pNewSize) + iNew ].x = pOldImage[ (jOld * pOldSize) + iOld ].x;
			pNewImage[ (jNew * pNewSize) + iNew ].y = pOldImage[ (jOld * pOldSize) + iOld ].y;
		}

	}

} // devScaleImage

//
//	devSquareRoot()
//
//	CJS: 06/01/2022
//
//	Take the square root of each complex array element.
//

__global__ void devSquareRoot( cufftComplex * pArray, int pSize )
{

	int i = (blockIdx.x * blockDim.x) + threadIdx.x;

	// ensure we're within the bounds of the kernel.
	if (i >= 0 && i < pSize)
	{
		cufftComplex tmp = pArray[ i ];
		pArray[ i ].x = sqrt( (tmp.x + sqrt( pow( tmp.x, 2 ) + pow( tmp.y, 2 ) )) / 2 );
		pArray[ i ].y = sqrt( (-tmp.x + sqrt( pow( tmp.x, 2 ) + pow( tmp.y, 2 ) )) / 2 );
	}

} // devSubtractArrays

//
//	devSubtractArrays()
//
//	CJS: 15/08/2018
//
//	Subtract one array from another
//

__global__ void devSubtractArrays( cufftComplex * pOne, cufftComplex * pTwo, int pSize )
{

	int i = (blockIdx.x * blockDim.x) + threadIdx.x;

	// ensure we're within the bounds of the kernel.
	if (i >= 0 && i < pSize)
	{
		pOne[ i ].x -= pTwo[ i ].x;
		pOne[ i ].y -= pTwo[ i ].y;
	}

} // devSubtractArrays

__global__ void devSubtractArrays( cufftDoubleComplex * pOne, cufftDoubleComplex * pTwo, int pSize )
{

	int i = (blockIdx.x * blockDim.x) + threadIdx.x;

	// ensure we're within the bounds of the kernel.
	if (i >= 0 && i < pSize)
	{
		pOne[ i ].x -= pTwo[ i ].x;
		pOne[ i ].y -= pTwo[ i ].y;
	}

} // devSubtractArrays

//
//	devTakeConjugateImage()
//
//	CJS: 01/11/2021
//
//	Takes the conjugate for each item in a list of complex numbers.
//

__global__ void devTakeConjugateImage
			(
			cufftComplex * pImage,				// an image consisting of complex numbers
			int pSize					// the size of each image axis
			)
{
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	// check that we haven't gone outside the grid dimensions of the image (some threads will).
	if ( i < pSize && j < pSize )
		pImage[ (j * pSize) + i ].y *= -1.0;

} // devTakeConjugateImage

//
//	devUpdateComplexArray()
//
//	CJS: 23/11/2015
//
//	Update the elements of a complex array.
//

__global__ void devUpdateComplexArray( cufftComplex * pArray, int pElements, float pReal, float pImaginary )
{
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	// check that we are within the array bounds.
	if (i < pElements)
	{
		pArray[ i ].x = pReal;
		pArray[ i ].y = pImaginary;
	}
	
} // devUpdateComplexArray

//
//	devUpperThreshold()
//
//	CJS: 24/11/2021
//
//	If any pixels of an image are above a threshold value, set them to zero.
//

__global__ void devUpperThreshold
			(
			cufftComplex * pImage,				// the image to process
			float pThreshold,				// the maximum value
			int pSize					// image size
			)
{
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	// check that we haven't gone outside the grid dimensions of the image (some threads will).
	if ( i < pSize && j < pSize )
		if (cuCabsf( pImage[ (j * pSize) + i ] ) > pThreshold)
		{
			pImage[ (j * pSize) + i ].x = 0.0;
			pImage[ (j * pSize) + i ].y = 0.0;
		}
	
} // devUpperThreshold

//
//	HOST FUNCTIONS
//

//
//	deg()
//
//	CJS: 08/04/2019
//
//	Convert radians to degrees
//

double deg( double pIn )
{
	
	return ( pIn * 180.0 / PI );

} // deg

//
//	finaliseFFT()
//
//	CJS: 15/03/2016
//
//	Delete an FFT plan.
//

void finaliseFFT( cufftHandle pFFTPlan )
{

	// destroy the FFT plan.
	cufftDestroy( pFFTPlan );
	pFFTPlan = -1;
	
} // finaliseFFT

//
//	findCutoffPixel()
//
//	CJS: 03/04/2019
//
//	Find the furthest pixel from the centre of the kernel which is >= pCutoffFraction % of the peak kernel value. this value is used as the support size
//	when the kernel is cropped.
//

int findCutoffPixel( cufftComplex * pdevKernel, double * pdevMaxValue, int pSize, double pCutoffFraction, findpixel pFindType )
{
	
	const int PIXELS_PER_THREAD = 32;
	
	bool ok = true;
	int support = 0;
	cudaError_t err;

	// create memory to hold the support on the device.
	int * devSupport = NULL;
	reserveGPUMemory( (void **) &devSupport, sizeof( int ), "reserving device memory for the cutoff support", __LINE__ );
	zeroGPUMemory( (void *) devSupport, sizeof( int ), "zeroing device memory for the cutoff support", __LINE__ );
		
	// find a suitable thread/block size for finding the cutoff pixel. each thread block will find the largest support needed for
	// N pixels (held in the above constant), and write the result to a shared memory array. a final loop will then
	// get the max from the array.
	int threads = (pSize * pSize / PIXELS_PER_THREAD) + 1, blocks = 1;
	setThreadBlockSize1D( &threads, &blocks );
		
	// declare global memory for writing the result of each block.
	int * devTmpResults = NULL;
	ok = reserveGPUMemory( (void **) &devTmpResults, blocks * sizeof( int ), "declaring device memory for max support", __LINE__ );
	if (ok == true)
	{
		
		// get maximum support in parallel.
		devFindCutoffPixelParallel<<< blocks, threads, threads * sizeof( int ) >>>(	/* pKernel = */ pdevKernel,
												/* pSize = */ pSize,
												/* pMaxValue = */ pdevMaxValue,
												/* pCellsPerThread = */ PIXELS_PER_THREAD,
												/* pTmpResults = */ devTmpResults,
												/* pCutoffFraction = */ pCutoffFraction,
												/* pFindType = */ pFindType );
		err = cudaGetLastError();
		if (err != cudaSuccess)
		{
			printf( "error finding max support (parallel) (%s)\n", cudaGetErrorString( err ) );
			ok = false;
		}
		
	}
	
	if (ok == true)
	{
			
		// get support from the block list.
		devFindCutoffPixel<<< 1, 1 >>>(	/* pTmpResults = */ devTmpResults,
							/* pSupport = */ devSupport,
							/* pElements = */ blocks,
							/* pFindType = */ pFindType );
		err = cudaGetLastError();
		if (err != cudaSuccess)
		{
			printf( "error getting final support value (%s)\n", cudaGetErrorString( err ) );
			ok = false;
		}
			
	}

	// get the support from the storage area.
	moveDeviceToHost( (void *) &support, (void *) devSupport, sizeof( int ), "copying support from the host", __LINE__ );
		
	// free memory.
	if (devTmpResults != NULL)
		cudaFree( (void *) devTmpResults );
	if (devSupport != NULL)
		cudaFree( (void *) devSupport );
	
	// return the support
	return support;

} // findCutoffPixel

int findCutoffPixel( float * pdevKernel, double * pdevMaxValue, int pSize, double pCutoffFraction, findpixel pFindType )
{
	
	const int PIXELS_PER_THREAD = 32;
	
	int support = 0;
	cudaError_t err;

	// create memory to hold the support on the device.
	int * devSupport = NULL;
	reserveGPUMemory( (void **) &devSupport, sizeof( int ), "reserving device memory for the cutoff support", __LINE__ );
	zeroGPUMemory( (void *) devSupport, sizeof( int ), "zeroing device memory for the cutoff support", __LINE__ );
		
	// find a suitable thread/block size for finding the cutoff pixel. each thread block will find the largest support needed for
	// N pixels (held in the above constant), and write the result to a shared memory array. a final loop will then
	// get the max from the array.
	int threads = (pSize * pSize / PIXELS_PER_THREAD) + 1, blocks = 1;
	setThreadBlockSize1D( &threads, &blocks );
		
	// declare global memory for writing the result of each block.
	int * devTmpResults = NULL;
	reserveGPUMemory( (void **) &devTmpResults, blocks * sizeof( int ), "declaring device memory for max support", __LINE__ );
		
	// get maximum support in parallel.
	devFindCutoffPixelParallel<<< blocks, threads, threads * sizeof( int ) >>>(	/* pKernel = */ pdevKernel,
											/* pSize = */ pSize,
											/* pMaxValue = */ pdevMaxValue,
											/* pCellsPerThread = */ PIXELS_PER_THREAD,
											/* pTmpResults = */ devTmpResults,
											/* pCutoffFraction = */ pCutoffFraction,
											/* pFindType = */ pFindType );
	err = cudaGetLastError();
	if (err != cudaSuccess)
		printf( "error finding max support (parallel) (%s)\n", cudaGetErrorString( err ) );
		
	// get support from the block list.
	devFindCutoffPixel<<< 1, 1 >>>(	/* pTmpResults = */ devTmpResults,
						/* pSupport = */ devSupport,
						/* pElements = */ blocks,
						/* pFindType = */ pFindType );
	err = cudaGetLastError();
	if (err != cudaSuccess)
		printf( "error getting final support value (%s)\n", cudaGetErrorString( err ) );

	// get the support from the storage area.
	moveDeviceToHost( (void *) &support, (void *) devSupport, sizeof( int ), "copying support from the host", __LINE__ );
		
	// free memory.
	if (devTmpResults != NULL)
		cudaFree( (void *) devTmpResults );
	if (devSupport != NULL)
		cudaFree( (void *) devSupport );
	
	// return the support
	return support;

} // findCutoffPixel

//
//	getMaxValue()
//
//	CJS: 23/11/2015
//
//	Gets the maximum complex value from a 2D image array, and writes it (along with the x and y coordinates)
//	to a 3 double area in global memory.
//

bool getMaxValue( cufftComplex * pdevImage, double * pdevMaxValue, int pWidth, int pHeight, bool pIncludeComplexComponent, bool pMultiplyByConjugate, bool * pdevMask )
{
	
	const int PIXELS_PER_THREAD = 32;
	
	bool ok = true;
	cudaError_t err;
		
	// find a suitable thread/block size for finding the maximum pixel value. each thread block will find the max
	// over N pixels (held in the above constant), and write the result to a shared memory array. a final loop will then
	// get the max from the array.
	int threads = (pWidth * pHeight / PIXELS_PER_THREAD) + 1, blocks = 1;
	setThreadBlockSize1D( &threads, &blocks );
		
	// declare global memory for writing the result of each block.
	double * devTmpResults = NULL;
	ok = reserveGPUMemory( (void **) &devTmpResults, MAX_PIXEL_DATA_AREA_SIZE * blocks * sizeof( double ), "declaring device memory for psf max block value", __LINE__ );
		
	if (ok == true)
	{
		
		// get maximum pixel value.
		devGetMaxValueParallel<<< blocks, threads, MAX_PIXEL_DATA_AREA_SIZE * threads * sizeof( double ) >>>
					(	/* pArray = */ pdevImage,
						/* pWidth = */ pWidth,
						/* pHeight = */ pHeight,
						/* pCellsPerThread = */ PIXELS_PER_THREAD,
						/* pBlockMax = */ devTmpResults,
						/* pIncludeComplexComponent = */ pIncludeComplexComponent,
						/* pMultiplyByConjugate = */ pMultiplyByConjugate,
						/* pMask = */ pdevMask );
		err = cudaGetLastError();
		if (err != cudaSuccess)
		{
			printf( "error getting psf max pixel value (%s)\n", cudaGetErrorString( err ) );
			ok = false;
		}
		
	}
	
	if (ok == true)
	{
			
		// get maximum pixel value from the block list.
		devGetMaxValue<<< 1, 1 >>>(	/* pArray = */ devTmpResults,
						/* pMaxValue = */ pdevMaxValue,
						/* pUseAbsolute = */ false,
						/* pElements = */ blocks );
		err = cudaGetLastError();
		if (err != cudaSuccess)
		{
			printf( "error getting final max value (%s)\n", cudaGetErrorString( err ) );
			ok = false;
		}
			
	}
		
	// free memory.
	if (devTmpResults != NULL)
		cudaFree( (void *) devTmpResults );
	
	// return success flag.
	return ok;
	
} // getMaxValue

bool getMaxValue( float * pdevImage, double * pdevMaxValue, int pWidth, int pHeight, bool pUseAbsolute, bool * pdevMask )
{
	
	const int PIXELS_PER_THREAD = 32;
	
	bool ok = true;
	cudaError_t err;
		
	// find a suitable thread/block size for finding the maximum pixel value. each thread block will find the max
	// over N pixels (held in the above constant), and write the result to a shared memory array. a final loop will then
	// get the max from the array.
	int threads = (pWidth * pHeight / PIXELS_PER_THREAD) + 1, blocks = 1;
	setThreadBlockSize1D( &threads, &blocks );
		
	// declare global memory for writing the result of each block.
	double * devTmpResults = NULL;
	ok = reserveGPUMemory( (void **) &devTmpResults, MAX_PIXEL_DATA_AREA_SIZE * blocks * sizeof( double ), "declaring device memory for psf max block value", __LINE__ );
		
	if (ok == true)
	{
		
		// get maximum pixel value.
		devGetMaxValueParallel<<< blocks, threads, MAX_PIXEL_DATA_AREA_SIZE * threads * sizeof( double ) >>>
					(	/* pArray = */ pdevImage,
						/* pWidth = */ pWidth,
						/* pHeight = */ pHeight,
						/* pCellsPerThread = */ PIXELS_PER_THREAD,
						/* pBlockMax = */ devTmpResults,
						/* pUseAbsolute = */ pUseAbsolute,
						/* pMask = */ pdevMask );
		err = cudaGetLastError();
		if (err != cudaSuccess)
		{
			printf( "error getting psf max pixel value (%s)\n", cudaGetErrorString( err ) );
			ok = false;
		}
		
	}
	
	if (ok == true)
	{
			
		// get maximum pixel value from the block list.
		devGetMaxValue<<< 1, 1 >>>(	/* pArray = */ devTmpResults,
						/* pMaxValue = */ pdevMaxValue,
						/* pUseAbsolute = */ pUseAbsolute,
						/* pElements = */ blocks );
		err = cudaGetLastError();
		if (err != cudaSuccess)
		{
			printf( "error getting final max value (%s)\n", cudaGetErrorString( err ) );
			ok = false;
		}
			
	}
		
	// free memory.
	if (devTmpResults != NULL)
		cudaFree( (void *) devTmpResults );
	
	// return success flag.
	return ok;
	
} // getMaxValue

//
//	initialiseFFT()
//
//	CJS: 15/03/2016
//
//	Create an FFT plan.
//

cufftHandle initialiseFFT( int pSize )
{

	// generate the FFT plan.
	cufftHandle fftPlan;
	cufftPlan2d( &fftPlan, pSize, pSize, CUFFT_C2C );
	
	// return the plan.
	return fftPlan;
	
} // initialiseFFT

//
//	moveDeviceToHost()
//
//	CJS: 29/04/2020
//
//	Copy some memory from the device to the host, and display an error if it failed.
//

bool moveDeviceToHost( void * pToPtr, void * pFromPtr, long int pSize, const char * pTask, int pLineNumber )
{

	cudaError_t err = cudaMemcpy( pToPtr, pFromPtr, pSize, cudaMemcpyDeviceToHost );
	if (err != cudaSuccess)
	{
		printf( "Error %s (%i: %s)\n", pTask, pLineNumber, cudaGetErrorString( err ) );
		exit( 1 );
	}

	// return something.
	return (err == cudaSuccess);

} // moveDeviceToHost

//
//	moveHostToDevice()
//
//	CJS: 29/04/2020
//
//	Copy some memory from the host to the device, and display an error if it failed.
//

bool moveHostToDevice( void * pToPtr, void * pFromPtr, long int pSize, const char * pTask, int pLineNumber )
{

	cudaError_t err = cudaMemcpy( pToPtr, pFromPtr, pSize, cudaMemcpyHostToDevice );
	if (err != cudaSuccess)
	{
		printf( "Error %s (%i: %s)\n", pTask, pLineNumber, cudaGetErrorString( err ) );
		exit( 1 );
	}

	// return something.
	return (err == cudaSuccess);

} // moveHostToDevice

//
//	performFFT()
//
//	CJS: 11/08/2015
//
//	Make a dirty image by inverse FFTing the gridded visibilites.
//

bool performFFT( cufftComplex ** pdevGrid, int pSize, fftdirection pFFTDirection, cufftHandle pFFTPlan, ffttype pFFTType, bool pResizeArray )
{
	
	bool ok = true;
	cudaError_t err;
	dim3 gridSize2D( 1, 1 );
	dim3 blockSize2D( 1, 1 );

	// reserve some memory to hold a temporary image, which allows us to do an FFT shift.
	cufftComplex * devTmpImage = NULL;
	reserveGPUMemory( (void **) &devTmpImage, sizeof( cufftComplex ) * pSize * pSize, "reserving device memory for the FFT temporary image", __LINE__ );

	// reverse the y-direction because images produced from inverse FFTs will be upside-down, so we need our image to also be upside-down before
	// switching to the uv domain.
	if (pFFTDirection == FORWARD)
	{
		setThreadBlockSize2D( pSize, pSize / 2, gridSize2D, blockSize2D );
		if (pFFTType == F2F || pFFTType == F2C)
			devReverseYDirection<<< gridSize2D, blockSize2D >>>(	/* pGrid = */ (float *) *pdevGrid,
										/* pSize = */ pSize );
		else
			devReverseYDirection<<< gridSize2D, blockSize2D >>>(	/* pGrid = */ *pdevGrid,
										/* pSize = */ pSize );
		err = cudaGetLastError();
		if (err != cudaSuccess)
		{
			printf( "error swapping image y-coordinates (%s)\n", cudaGetErrorString( err ) );
			ok = false;
		}
	}

	// do forward FFT shift.
	setThreadBlockSize2D( pSize, pSize, gridSize2D, blockSize2D );
	if (pFFTType == F2F || pFFTType == F2C)
		devFFTShift<<< gridSize2D, blockSize2D >>>(	/* pDestination = */ devTmpImage,
								/* pSource = */ (float *) *pdevGrid,
								/* pFFTDirection = */ FORWARD,
								/* pSize = */ pSize );
	else
		devFFTShift<<< gridSize2D, blockSize2D >>>(	/* pDestination = */ devTmpImage,
								/* pSource = */ *pdevGrid,
								/* pFFTDirection = */ FORWARD,
								/* pSize = */ pSize );

	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		printf( "error doing FFT shift prior to FFT (%s)\n", cudaGetErrorString( err ) );
		ok = false;
	}

	// if we're doing a 'from real' transform then free the grid and recreate it.
	if (pFFTType == F2C || pFFTType == F2F)
	{
		cudaFree( (void *) *pdevGrid );
		reserveGPUMemory( (void **) pdevGrid, pSize * pSize * sizeof( cufftComplex ), "creating device memory for enlarged grid following FFT", __LINE__ );
	}

	// move image from temporary memory.
	cudaMemcpy( (void *) *pdevGrid, (void *) devTmpImage, pSize * pSize * sizeof( cufftComplex ), cudaMemcpyDeviceToDevice );

	// free memory.
	if (devTmpImage != NULL)
		cudaFree( (void *) devTmpImage );

	// create the plan if it doesn't exist.
	bool destroyPlan = false;
	if (pFFTPlan == -1)
	{
		pFFTPlan = initialiseFFT( pSize );
		destroyPlan = true;
	}

	// execute the fft.
	cufftExecC2C( pFFTPlan, *pdevGrid, *pdevGrid, (pFFTDirection == INVERSE ? CUFFT_INVERSE : CUFFT_FORWARD) );

	// destroy the plan if we need to.
	if (destroyPlan == true)
		finaliseFFT( pFFTPlan );

	// reserve some memory to hold a temporary image, which allows us to do an FFT shift.
	reserveGPUMemory( (void **) &devTmpImage, pSize * pSize * sizeof( cufftComplex ), "reserving device memory for the FFT temporary image", __LINE__ );

	// move image to temporary memory.
	cudaMemcpy( (void *) devTmpImage, (void *) *pdevGrid, pSize * pSize * sizeof( cufftComplex ), cudaMemcpyDeviceToDevice );

	// if we are doing a 'to real' FFT, and we want to resize the data area, then destroy the original data area and redimension.
	if ((pFFTType == C2F || pFFTType == F2F) && (pResizeArray == true || pFFTType == F2F))
	{
		cudaFree( (void *) *pdevGrid );
		reserveGPUMemory( (void **) pdevGrid, sizeof( float ) * pSize * pSize, "reserving device memory for smaller FFT image", __LINE__ );	
	}

	// do inverse FFT shift.
	setThreadBlockSize2D( pSize, pSize, gridSize2D, blockSize2D );
	if (pFFTType == C2F || pFFTType == F2F)
		devFFTShift<<< gridSize2D, blockSize2D >>>(	/* pDestination = */ (float *) *pdevGrid,
								/* pSource = */ devTmpImage,
								/* pFFTDirection = */ INVERSE,
								/* pSize = */ pSize );
	else
		devFFTShift<<< gridSize2D, blockSize2D >>>(	/* pDestination = */ *pdevGrid,
								/* pSource = */ devTmpImage,
								/* pFFTDirection = */ INVERSE,
								/* pSize = */ pSize );

	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		printf( "error doing FFT shift following FFT (%s)\n", cudaGetErrorString( err ) );
		ok = false;
	}

	// free memory.
	if (devTmpImage != NULL)
		cudaFree( (void *) devTmpImage );

	// reverse the y-direction because images produces here from inverse FFTs will be naturally upside-down.
	if (pFFTDirection == INVERSE)
	{
		setThreadBlockSize2D( pSize, pSize / 2, gridSize2D, blockSize2D );
		if (pFFTType == C2F || pFFTType == F2F)
			devReverseYDirection<<< gridSize2D, blockSize2D >>>(	/* pGrid = */ (float *) *pdevGrid,
										/* pSize = */ pSize );
		else
			devReverseYDirection<<< gridSize2D, blockSize2D >>>(	/* pGrid = */ *pdevGrid,
										/* pSize = */ pSize );
		err = cudaGetLastError();
		if (err != cudaSuccess)
		{
			printf( "error swapping image y-coordinates (%s)\n", cudaGetErrorString( err ) );
			ok = false;
		}
	}
		
	// return success/failure.
	return ok;
	
} // performFFT

//
//	rad()
//
//	CJS: 01/04/2019
//
//	Convert degrees to radians
//

double rad( double pIn )
{
	
	return ( pIn * PI / 180.0 );

} // rad

//
//	reserveGPUMemory()
//
//	CJS: 15/10/2019
//
//	Reserve some memory on the GPU, and display an error if this fails.
//

bool reserveGPUMemory( void ** pMemPtr, long int pSize, const char * pTask, int pLineNumber )
{

	cudaError_t err = cudaMalloc( pMemPtr, pSize );
	if (err != cudaSuccess)
	{
		printf( "Error %s (%i: %s)\n", pTask, pLineNumber, cudaGetErrorString( err ) );
		exit( 1 );
	}

	// return something.
	return (err == cudaSuccess);

} // reserveGPUMemory

//
//	setThreadBlockSize1D()
//
//	CJS:	06/11/2015
//
//	Determine a suitable thread and block size for the current GPU.
//	The number of threads must be less than the maximum number allowed by the current GPU.
//

void setThreadBlockSize1D( int * pThreads, int * pBlocks )
{

	// get some properties from the device.
	int device = 0;
	cudaDeviceProp gpuProperties;
	cudaGetDevice( &device );
	cudaGetDeviceProperties( &gpuProperties, device );
	int maxThreadsPerBlock = gpuProperties.maxThreadsPerBlock;
	
	// store the total number of threads.
	int totalThreads = *pThreads;
	
	*pBlocks = 1;
	if ( *pThreads > maxThreadsPerBlock )
	{
		*pThreads = maxThreadsPerBlock;
		*pBlocks = (totalThreads / maxThreadsPerBlock);
		if (totalThreads % maxThreadsPerBlock != 0)
			(*pBlocks)++;
	}
	
} // setThreadBlockSize1D

//
//	setThreadBlockSize2D()
//
//	CJS:	10/11/2015
//
//	Determine a suitable thread and block size for the current GPU.
//	The number of threads must be less than the maximum number allowed by the current GPU.
//
//	This subroutine is used when we have a single, large XxY grid (i.e. when we are processing an image).
//

void setThreadBlockSize2D( int pThreadsX, int pThreadsY, dim3 & pGridSize2D, dim3 & pBlockSize2D )
{

	// get some properties from the device.
	int device = 0;
	cudaDeviceProp gpuProperties;
	cudaGetDevice( &device );
	cudaGetDeviceProperties( &gpuProperties, device );
	int maxThreadsPerBlock = gpuProperties.maxThreadsPerBlock;
	
	// store the total number of X and Y threads.
	int totalThreadsX = pThreadsX;
	int totalThreadsY = pThreadsY;
	
	pBlockSize2D.x = pThreadsX;
	pBlockSize2D.y = pThreadsY;
	pGridSize2D.x = 1;
	pGridSize2D.y = 1;
	
	// do we have too many threads?
	while ( (pBlockSize2D.x * pBlockSize2D.y) > maxThreadsPerBlock )
	{
		
		// increment the number of Y blocks.
		pGridSize2D.y = pGridSize2D.y + 1;
		pBlockSize2D.y = (int) ceil( (double) totalThreadsY / (double) pGridSize2D.y );
		
		// if this doesn't help, increment the number of X blocks. if we have multiple iterations of this loop then
		// we will be incrementing Y, X, Y, X, Y, X, Y, .... etc.
		if ( (pBlockSize2D.x * pBlockSize2D.y) > maxThreadsPerBlock )
		{
			pGridSize2D.x = pGridSize2D.x + 1;
			pBlockSize2D.x = (int) ceil( (double) totalThreadsX / (double) pGridSize2D.x );
		}
		
	}
	
} // setThreadBlockSize2D

//
//	zeroGPUMemory()
//
//	CJS: 15/10/2019
//
//	Zero some memory on the GPU, and display an error if this fails.
//

bool zeroGPUMemory( void * pMemPtr, long int pSize, const char * pTask, int pLineNumber )
{

	cudaError_t err = cudaMemset( pMemPtr, 0, pSize );
	if (err != cudaSuccess)
	{
		printf( "Error %s (%i: %s)\n", pTask, pLineNumber, cudaGetErrorString( err ) );
		exit( 1 );
	}

	// return something.
	return (err == cudaSuccess);

} // zeroGPUMemory
