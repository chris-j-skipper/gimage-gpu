
// include the header file.
#include "cuReprojectionNotComplex.h"

using namespace std;

//
//	N O N   C L A S S   M E M B E R S
//

//
//	atomicAddDouble()
//
//	CJS: 09/10/2018
//
//	Fudges an atomic add for doubles, which CUDA cannot do by itself.
//

__device__ double atomicAddDoubleReprojection( double * pAddress, double pVal )
{

	unsigned long long int * address_as_ull = (unsigned long long int *)pAddress;
	unsigned long long int old = *address_as_ull, assumed;

	do
	{
		assumed = old;
		old = atomicCAS( address_as_ull, assumed, __double_as_longlong( pVal + __longlong_as_double( assumed ) ) );
	} while (assumed != old);

	return __longlong_as_double( old );

} // atomicAddDoubleReprojection

//
//	applyMap()
//
//	CJS: 25/11/2015
//
//	Convert an output/input coordinate to an input/output coordinate using the supplied interpolated grid.
//

__host__ __device__ Reprojection::rpVectD applyMap( Reprojection::rpVectI pOld, double * pMap, bool * pMapValid, Reprojection::rpVectI pOldSize,
								bool * pPixelValid, Reprojection::rpVectI pNewSize )
{
	
	Reprojection::rpVectD newCoordinates = { .x = 0, .y = 0, .z = 0 };
	
	// ensure that the supplied old image coordinates are within the bounds of the image (with an additional
	// pixel added to either end. e.g. -1 --> _oldSize instead of 0 --> _oldSize - 1).
	if (pOld.x >= -1 && pOld.x <= (double)pOldSize.x && pOld.y >= -1 && pOld.y <= (double)pOldSize.y)
	{
		
		// convert this coordinate to grid positions.
		int gridIndex = (((pOld.y + 1) * pOldSize.x) + pOld.x + 1);
	
		// get the new pixel coordinates from this positions, checking at the same time that the map is valid.
		if (pMapValid[ gridIndex ] == false)
			*pPixelValid = false;
		newCoordinates.x = pMap[ gridIndex * 2 ];
		newCoordinates.y = pMap[ (gridIndex * 2) + 1 ];
		 
	}
	else
		*pPixelValid = false;
	
	// return the input pixel coordinates.
	return newCoordinates;
	
} // applyMap

//
//	pixelToWorld()
//
//	CJS:	08/07/15
//
//	Convert pixel coordinates into world coordinates using the supplied coordinate system. CASA uses WCSLIB to do this,
//	which does a complete spherical transformation of the coordinates. However, here we compromise and just use the
//	FITS matrix transformation (CD) to do a linear conversion from pixel to intermediate coordinates, and then multiply
//	the first coordinate by cos(dec) in order to convert from an angular size to degrees of RA (i.e. implement sinusoidal
//	projection).
//
//	The rotation matrix attached to the coordinate system will convert the coordinates from the origin (chosen to be
//	RA 0, DEC 0) to the relevant RA and DEC position of the reference pixel. Epoch conversion will be done if required.
//
//	Note that the routine returns the new position in cartesian coordinates (using directional cosines). This is because
//	the world to pixel routine needs cartesian coordinates.
//
//	The returned 'wrap around' flag warns if this pixel is outside the range -180 to 180 in RA, or -90 to 90 in DEC.
//

__host__ __device__ Reprojection::rpVectD pixelToWorld( Reprojection::rpVectD pPixelPosition, Reprojection::rpCoordSys pCoordinateSystem, bool * pWrapAround )
{
	
	// subtract reference pixel from position.
	Reprojection::rpVectD pixelOffset;

	// get the pixel offset from the reference pixel (usually the centre pixel).
	pixelOffset.x = pPixelPosition.x - pCoordinateSystem.crPIX.x;
	pixelOffset.y = pPixelPosition.y - pCoordinateSystem.crPIX.y;

	// apply coordinate system CD transformation matrix to convert the pixel offset into cartesian coordinates.
	Reprojection::rpVectD intermediatePosition;
	intermediatePosition.x = (pCoordinateSystem.cd.a11 * pixelOffset.x) + (pCoordinateSystem.cd.a12 * pixelOffset.y);
	intermediatePosition.y = (pCoordinateSystem.cd.a21 * pixelOffset.x) + (pCoordinateSystem.cd.a22 * pixelOffset.y);

	// if l^2 + m^2 > 1, then no conversion exists.
	if (pow( intermediatePosition.x, 2 ) + pow( intermediatePosition.y, 2 ) > 1)
		*pWrapAround = true;
	
	// get x, y and z cartesian coordinates. the x axis is in the pointing direction, y is to the east, and z is up.
	Reprojection::rpVectD cartesianOffset;
	cartesianOffset.y = intermediatePosition.x;
	cartesianOffset.z = intermediatePosition.y;
	cartesianOffset.x = sqrt( 1.0 - pow( intermediatePosition.x, 2 ) - pow( intermediatePosition.y, 2 ) );
	
	// rotate these cartesian coordinates by the reference pixel's ra and dec so that they are relative to ra 0, dec 0.
	cartesianOffset = Reprojection::multMatrixVector( pCoordinateSystem.toWorld, cartesianOffset );
	
	// return the world position.
	return cartesianOffset;

} // pixelToWorld

//
//	worldToPixel()
//
//	CJS:	08/07/15
//
//	Convert world coordinates into pixel coordinates using the supplied coordinate system. Now we must use
//	the inverse transformation matrix, which was calculated earlier from CD.
//

__host__ __device__ Reprojection::rpVectD worldToPixel( Reprojection::rpVectD pWorldPosition, Reprojection::rpCoordSys pCoordinateSystem, bool * pWrapAround )
{
	
	// rotate the vector to bring it from its world position near RA and DEC to the origin, which we choose
	// to be RA 0, DEC 0.
	Reprojection::rpVectD cartesianOffset = Reprojection::multMatrixVector( pCoordinateSystem.toPixel, pWorldPosition );
	
	// we now need to convert back into polar coordinates.
	Reprojection::rpVectD intermediatePosition = { .x = cartesianOffset.y, .y = cartesianOffset.z };
	
	// ensure right ascention is within the required range (-180 to 180 degrees).
	Reprojection::angleRange( &intermediatePosition.x, 0, 180 );
	
	// ensure declination is within the required range (-90 to 90 degrees).
	Reprojection::angleRange( &intermediatePosition.y, 0, 90 );
	
	// if longitude is less than -180 deg, or more than 180 deg, then no conversion exists.
	if (intermediatePosition.x < -180 || intermediatePosition.x > 180)
		*pWrapAround = true;
	
	// apply coordinate system inverse-CD transformation matrix.
	Reprojection::rpVectD pixelOffset;
	pixelOffset.x = (pCoordinateSystem.inv_cd.a11 * intermediatePosition.x) + (pCoordinateSystem.inv_cd.a12 * intermediatePosition.y);
	pixelOffset.y = (pCoordinateSystem.inv_cd.a21 * intermediatePosition.x) + (pCoordinateSystem.inv_cd.a22 * intermediatePosition.y);
	
	// add reference pixel coordinates.
	Reprojection::rpVectD pixelPosition;
	pixelPosition.x = pixelOffset.x + pCoordinateSystem.crPIX.x;
	pixelPosition.y = pixelOffset.y + pCoordinateSystem.crPIX.y;
	
	// return the world position.
	return pixelPosition;

} // worldToPixel

//
//	devBuildMap()
//
//	CJS: 28/11/2018
//
//	Generates a map that translates out-pixel coordinates to in-pixel coordinates.
//

__global__ void devBuildMap( double * pMap, bool * pMapValid, Reprojection::rpVectI pMapSize,
				Reprojection::rpCoordSys pOldCoordinateSystem, Reprojection::rpCoordSys pNewCoordinateSystem )
{
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;

	// ensure we're within the image boundaries.
	if (i < pMapSize.x && j < pMapSize.y)
	{
			
		Reprojection::rpVectD oldPixelCoordinate = { .x = (double) i - 1, .y = (double) j - 1 };
					
		// convert these pixel coordinates to world coordinates (cartesian).
		bool wrapAround = false;
		Reprojection::rpVectD worldCoordinate = pixelToWorld( oldPixelCoordinate, pOldCoordinateSystem, &wrapAround );
			
		// convert the world coordinates back into input pixel coordinates.
		Reprojection::rpVectD newPixelCoordinate = worldToPixel( worldCoordinate, pNewCoordinateSystem, &wrapAround );
			
		// store the in pixel coordinate for future use. this mapping is only valid if the output image hasn't
		// wrapped around to the other side of the celestial sphere. once we get to + or - 180 degrees from the
		// reference pixel we stop mapping.
		pMapValid[ (j * pMapSize.x) + i ] = (wrapAround == false);
		pMap[ ((j * pMapSize.x) + i) * 2 ] = newPixelCoordinate.x;
		pMap[ (((j * pMapSize.x) + i) * 2) + 1 ] = newPixelCoordinate.y;

	}

} // devBuildMap

//
//	devSquareRoot()
//
//	CJS: 12/12/2019
//
//	Take the square root of each pixel in an image.
//

__global__ void devSquareRoot( double * pImage, Reprojection::rpVectI pMapSize )
{
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;

	// ensure we're within the image boundaries.
	if (i < pMapSize.x && j < pMapSize.y)
	{

		double value = pImage[ (j * pMapSize.x) + i ];
		if (value >= 0)
			pImage[ (j * pMapSize.x) + i ] = sqrt( value );

	}

} // devSquareRoot

//
//	devGetMaxPixel()
//
//	CJS: 12/12/2019
//
//	Gets the maximum pixel from an image.
//

__global__ void devGetMaxPixel( double * pImage, double * pMaxValue, Reprojection::rpVectI pMapSize )
{

	double maxValue = 0.0;
	for ( int i = 0; i < pMapSize.x * pMapSize.y; i++ )
	{
		double value = pImage[ i ];
		if (value > maxValue)
			maxValue = value;
	}

	// update result.
	*pMaxValue = maxValue;

} // devGetMaxPixel

//
//	devNormalise()
//
//	CJS: 12/12/2019
//
//	Normalise the pixels by dividing by a constant value.
//

__global__ void devNormalise( double * pImage, double * pNormalisation, Reprojection::rpVectI pMapSize )
{
	
	// store the constant in shared memory.
	__shared__ double normalisation;
	if (threadIdx.x == 0)
		normalisation = *pNormalisation;
	
	__syncthreads();
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;

	// ensure we're within the image boundaries.
	if (i < pMapSize.x && j < pMapSize.y && normalisation != 0.0)
		pImage[ (j * pMapSize.x) + i ] /= normalisation;

} // devNormalise

//
//	devSetMask()
//
//	CJS: 12/12/2019
//
//	Sets a mask according to the values of the pixels in an image.
//

__global__ void devSetMask( double * pImage, bool * pMask, Reprojection::rpVectI pImageSize )
{
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;

	// ensure we're within the image boundaries.
	if (i < pImageSize.x && j < pImageSize.y)
		pMask[ (j * pImageSize.x) + i ] = (pImage[ (j * pImageSize.x) + i ] >= 0.2);

} // devSetMask

//
//	devReprojectionInToOut()
//
//	CJS: 17/12/2019
//
//	Performs regridding and reprojection between the input and output images.
//

__global__ void devReprojectionInToOut( double * pInImage, double * pOutImage, double * pProjectionMap, bool * pMapValid,
						double * pNormalisationPattern, double * pPrimaryBeamPattern, Reprojection::rpVectI pMapSize, Reprojection::rpVectI pInSize,
						Reprojection::rpVectI pOutSize, bool * pInMask, double * pBeamIn, double * pBeamOut,
						bool pAProjection )
{
	
	const int POS_BL = 0;
	const int POS_BR = 1;
	const int POS_TL = 2;
	const int POS_TR = 3;
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;

	// ensure we're within the boundaries of the image.
	if (i < pInSize.x && j < pInSize.y)
	{
			
		// convert each of these four pixels to the input coordinate system.
		bool pixelValid = true;
					
		// get the position of the interpolation point in the input image.
		Reprojection::rpVectI inPixel = {	.x = i,
							.y = j };
											
		// convert these input image pixel coordinates into output image pixel coordinates
		// by interpolating the map. we pass in the pixelValid boolean again, but at this
		// point we already know that it is valid.
		Reprojection::rpVectD outPixelInterpolationPoint = applyMap(	inPixel,
										pProjectionMap,
										pMapValid,
										pMapSize,
										&pixelValid,
										pInSize );
					
		// calculate the four pixel coordinates surrounding this interpolation point.
		Reprojection::rpVectI pixel[4];
		pixel[POS_BL].x = (int) floor( outPixelInterpolationPoint.x );
		pixel[POS_BL].y = (int) floor( outPixelInterpolationPoint.y );
		pixel[POS_BR].x = pixel[POS_BL].x + 1; pixel[POS_BR].y = pixel[POS_BL].y;
		pixel[POS_TL].x = pixel[POS_BL].x; pixel[POS_TL].y = pixel[POS_BL].y + 1;
		pixel[POS_TR].x = pixel[POS_BL].x + 1; pixel[POS_TR].y = pixel[POS_BL].y + 1;

		// ensure all pixels are within the extent of the input image.
		bool withinRange = true;
		for ( int m = 0; m < 4; m++ )
		{
			withinRange = withinRange && (pixel[m].x >= 0) && (pixel[m].x < pOutSize.x);
			withinRange = withinRange && (pixel[m].y >= 0) && (pixel[m].y < pOutSize.y);
		}
		if (withinRange == true)
		{

			// calculate memory location of these pixels within the output image.
			int location[4];
			for ( int m = 0; m < 4; m++ )
				location[m] = (pixel[m].y * pOutSize.x) + pixel[m].x;

			// if we have an input mask then check if this pixel is masked.
			bool masked = false;
			if (pInMask != NULL)
				masked = (pInMask[ (j * pInSize.x) + i ] == false);

			// only update values if we're not masked.
			if (masked == false)
			{

				// store the value of the input image pixel.
				double value = pInImage[ (j * pInSize.x) + i ];

				// get the fraction of the output pixels that should be updated with this value.
				double blFraction = Reprojection::interpolateValue( outPixelInterpolationPoint, 1, 0, 0, 0 );
				double brFraction = Reprojection::interpolateValue( outPixelInterpolationPoint, 0, 1, 0, 0 );
				double tlFraction = Reprojection::interpolateValue( outPixelInterpolationPoint, 0, 0, 1, 0 );
				double trFraction = Reprojection::interpolateValue( outPixelInterpolationPoint, 0, 0, 0, 1 );

				// if the output image has an associated primary beam then we need to multiply our value by this beam. we don't need to do this if
				// we're using A-Projection because A-projection will handle this function.
				if (pBeamOut != NULL && pAProjection == false)
				{
					double beamValue = Reprojection::interpolateValue(	outPixelInterpolationPoint,
												pBeamOut[ location[POS_BL] ],
												pBeamOut[ location[POS_BR] ],
												pBeamOut[ location[POS_TL] ],
												pBeamOut[ location[POS_TR] ] );
					value *= beamValue;
				}

				// only add up the primary beam if a primary beam was supplied.
				if (pBeamIn != NULL && (pNormalisationPattern != NULL || pPrimaryBeamPattern != NULL || pAProjection == true))
				{

					// get the primary beam for these interpolation points.
					double beamIn = pBeamIn[ (j * pInSize.x) + i ];

					// weight the image using the primary beam. we don't need to do this if we're not using A-projection because the beam correction
					// and the beam weighting will cancel each other out.
					if (pAProjection == true)
						value *= beamIn;

					// the normalisation pattern is the sum of the primary beams at this pixel, and will be used later to normalise this pixel.
					if (pNormalisationPattern != NULL)
					{
						//value *= beamIn; // pow( beamIn, 2 );
						atomicAddDoubleReprojection( &pNormalisationPattern[ location[POS_BL] ], pow( beamIn, 1 ) * blFraction );
						atomicAddDoubleReprojection( &pNormalisationPattern[ location[POS_BR] ], pow( beamIn, 1 ) * brFraction );
						atomicAddDoubleReprojection( &pNormalisationPattern[ location[POS_TL] ], pow( beamIn, 1 ) * tlFraction );
						atomicAddDoubleReprojection( &pNormalisationPattern[ location[POS_TR] ], pow( beamIn, 1 ) * trFraction );
					}

					// the primary beam pattern is the sum of the primary beams squared at this pixel, and will be used later to suppress the noise
					// at the edges of the mosaic.
					if (pPrimaryBeamPattern != NULL)
					{
						atomicAddDoubleReprojection( &pPrimaryBeamPattern[ location[POS_BL] ], pow( beamIn, 2 ) * blFraction );
						atomicAddDoubleReprojection( &pPrimaryBeamPattern[ location[POS_BR] ], pow( beamIn, 2 ) * brFraction );
						atomicAddDoubleReprojection( &pPrimaryBeamPattern[ location[POS_TL] ], pow( beamIn, 2 ) * tlFraction );
						atomicAddDoubleReprojection( &pPrimaryBeamPattern[ location[POS_TR] ], pow( beamIn, 2 ) * trFraction );
					}

				}

				// update the pixel value.
				atomicAddDoubleReprojection( &pOutImage[ location[POS_BL] ], value * blFraction );
				atomicAddDoubleReprojection( &pOutImage[ location[POS_BR] ], value * brFraction );
				atomicAddDoubleReprojection( &pOutImage[ location[POS_TL] ], value * tlFraction );
				atomicAddDoubleReprojection( &pOutImage[ location[POS_TR] ], value * trFraction );

			}

		}

	}

} // devReprojectionInToOut

__global__ void devReprojectionInToOut( float * pInImage, float * pOutImage, double * pProjectionMap, bool * pMapValid,
						double * pNormalisationPattern, double * pPrimaryBeamPattern, Reprojection::rpVectI pMapSize, Reprojection::rpVectI pInSize,
						Reprojection::rpVectI pOutSize, bool * pInMask, double * pBeamIn, double * pBeamOut,
						bool pAProjection )
{
	
	const int POS_BL = 0;
	const int POS_BR = 1;
	const int POS_TL = 2;
	const int POS_TR = 3;
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;

	// ensure we're within the boundaries of the image.
	if (i < pInSize.x && j < pInSize.y)
	{
			
		// convert each of these four pixels to the input coordinate system.
		bool pixelValid = true;
					
		// get the position of the interpolation point in the input image.
		Reprojection::rpVectI inPixel = {	.x = i,
							.y = j };
											
		// convert these input image pixel coordinates into output image pixel coordinates
		// by interpolating the map. we pass in the pixelValid boolean again, but at this
		// point we already know that it is valid.
		Reprojection::rpVectD outPixelInterpolationPoint = applyMap(	inPixel,
										pProjectionMap,
										pMapValid,
										pMapSize,
										&pixelValid,
										pInSize );
					
		// calculate the four pixel coordinates surrounding this interpolation point.
		Reprojection::rpVectI pixel[4];
		pixel[POS_BL].x = (int) floor( outPixelInterpolationPoint.x );
		pixel[POS_BL].y = (int) floor( outPixelInterpolationPoint.y );
		pixel[POS_BR].x = pixel[POS_BL].x + 1; pixel[POS_BR].y = pixel[POS_BL].y;
		pixel[POS_TL].x = pixel[POS_BL].x; pixel[POS_TL].y = pixel[POS_BL].y + 1;
		pixel[POS_TR].x = pixel[POS_BL].x + 1; pixel[POS_TR].y = pixel[POS_BL].y + 1;

		// ensure all pixels are within the extent of the input image.
		bool withinRange = true;
		for ( int m = 0; m < 4; m++ )
		{
			withinRange = withinRange && (pixel[m].x >= 0) && (pixel[m].x < pOutSize.x);
			withinRange = withinRange && (pixel[m].y >= 0) && (pixel[m].y < pOutSize.y);
		}
		if (withinRange == true)
		{

			// calculate memory location of these pixels within the output image.
			int location[4];
			for ( int m = 0; m < 4; m++ )
				location[m] = (pixel[m].y * pOutSize.x) + pixel[m].x;

			// if we have an input mask then check if this pixel is masked.
			bool masked = false;
			if (pInMask != NULL)
				masked = (pInMask[ (j * pInSize.x) + i ] == false);

			// only update values if we're not masked.
			if (masked == false)
			{

				// store the value of the input image pixel.
				double value = (double) pInImage[ (j * pInSize.x) + i ];

				// get the fraction of the output pixels that should be updated with this value.
				double blFraction = Reprojection::interpolateValue( outPixelInterpolationPoint, 1, 0, 0, 0 );
				double brFraction = Reprojection::interpolateValue( outPixelInterpolationPoint, 0, 1, 0, 0 );
				double tlFraction = Reprojection::interpolateValue( outPixelInterpolationPoint, 0, 0, 1, 0 );
				double trFraction = Reprojection::interpolateValue( outPixelInterpolationPoint, 0, 0, 0, 1 );

				// if the output image has an associated primary beam then we need to multiply our value by this beam. we don't need to do this if
				// we're using A-Projection because A-projection will handle this function.
				if (pBeamOut != NULL && pAProjection == false)
				{
					double beamValue = Reprojection::interpolateValue(	outPixelInterpolationPoint,
												(double) pBeamOut[ location[POS_BL] ],
												(double) pBeamOut[ location[POS_BR] ],
												(double) pBeamOut[ location[POS_TL] ],
												(double) pBeamOut[ location[POS_TR] ] );
					value *= beamValue;
				}

				// only add up the primary beam if a primary beam was supplied.
				if (pBeamIn != NULL && (pNormalisationPattern != NULL || pPrimaryBeamPattern != NULL || pAProjection == true))
				{

					// get the primary beam for these interpolation points.
					double beamIn = pBeamIn[ (j * pInSize.x) + i ];

					// weight the image using the primary beam. we don't need to do this if we're not using A-projection because the beam correction
					// and the beam weighting will cancel each other out.
					if (pAProjection == true)
						value *= beamIn;

					// the normalisation pattern is the sum of the primary beams at this pixel, and will be used later to normalise this pixel.
					if (pNormalisationPattern != NULL)
					{
						//value *= beamIn; // pow( beamIn, 2 );
						atomicAddDoubleReprojection( &pNormalisationPattern[ location[POS_BL] ], pow( beamIn, 1 ) * blFraction );
						atomicAddDoubleReprojection( &pNormalisationPattern[ location[POS_BR] ], pow( beamIn, 1 ) * brFraction );
						atomicAddDoubleReprojection( &pNormalisationPattern[ location[POS_TL] ], pow( beamIn, 1 ) * tlFraction );
						atomicAddDoubleReprojection( &pNormalisationPattern[ location[POS_TR] ], pow( beamIn, 1 ) * trFraction );
					}

					// the primary beam pattern is the sum of the primary beams squared at this pixel, and will be used later to suppress the noise
					// at the edges of the mosaic.
					if (pPrimaryBeamPattern != NULL)
					{
						atomicAddDoubleReprojection( &pPrimaryBeamPattern[ location[POS_BL] ], pow( beamIn, 2 ) * blFraction );
						atomicAddDoubleReprojection( &pPrimaryBeamPattern[ location[POS_BR] ], pow( beamIn, 2 ) * brFraction );
						atomicAddDoubleReprojection( &pPrimaryBeamPattern[ location[POS_TL] ], pow( beamIn, 2 ) * tlFraction );
						atomicAddDoubleReprojection( &pPrimaryBeamPattern[ location[POS_TR] ], pow( beamIn, 2 ) * trFraction );
					}

				}

				// update the pixel value.
				atomicAdd( &pOutImage[ location[POS_BL] ], (float) (value * blFraction) );
				atomicAdd( &pOutImage[ location[POS_BR] ], (float) (value * brFraction) );
				atomicAdd( &pOutImage[ location[POS_TL] ], (float) (value * tlFraction) );
				atomicAdd( &pOutImage[ location[POS_TR] ], (float) (value * trFraction) );

			}

		}

	}

} // devReprojectionInToOut

//
//	devReprojectionOutToIn()
//
//	CJS: 28/11/2018
//
//	Performs regridding and reprojection between the input and output images.
//	We need to handle the case where the output image pixels are much larger than the input image pixels (we need
//	to sum over many pixels), and also when the output image pixels are much smaller than the input image pixels
//	(we need to interpolate between input image pixels).
//
//	This routine works by comparing the size of the input and output image pixels, and choosing a number of
//	interpolation points for each output pixel. For example, overlaying the input and output images in world
//	coordinates may give:
//
//		+--------+--------+--------+--------+
//		|        |        |        |        |	+---+
//		|        |#   #   #   #   #|  #     |	|   |	= input image pixels
//		|        |        |                 |	+---+
//		+--------+#-------+--------+--#-----+
//		|        |     X====X====X |        |	+===+
//		|        |#    I  |      I |  #     |	I   I	= output image pixel
//		|        |     X  | X    X |        |	+===+
//		+--------+#----I--+------I-+--#-----+
//		|        |     X====X====X |        |	# # #	  region centred on the output image pixel, that extends on
//		|        |#       |        |  #     |	#   #	= all four sides to the surrounding output image pixels. this
//		|        |        |        |        |	# # #	  is the region we sum over.
//		+--------+#---#---#---#---#+--#-----+
//		|        |        |        |        |	  X	= interpolation point. the centre point has weight 1, the ones
//		|        |        |        |        |					along side it have weight 0.5, and the
//		|        |        |        |        |					ones on the diagonals have weight 0.25.
//		+--------+--------+--------+--------+
//
//	The program uses bilinear interpolation to calculate the value of the input grid at each interpolation point. These
//	values are then summed using a weighting that depends upon the position of the interpolation point relative to the output
//	pixel (the centre of the output pixel has weight 1, and this drops to 0 as we near the adjacent output pixels). If the
//	output pixel is small compared to the input pixels then we use a small number of interpolation points (one would do the
//	job, but we use a minimum of 3x3). If the output pixel is large compared to the input pixels then we use many
//	interpolation points (enough to ensure that at least one interpolation point is found within each fully-enclosed input
//	pixel).
//

__global__ void devReprojectionOutToIn( double * pInImage, double * pOutImage, double * pProjectionMap, bool * pMapValid,
						double * pNormalisationPattern, double * pPrimaryBeamPattern, Reprojection::rpVectI pMapSize, Reprojection::rpVectI pInSize,
						Reprojection::rpVectI pOutSize, bool * pInMask, double * pBeamIn, double * pBeamOut,
						bool pAProjection )
{
	
	const int POS_BL = 0;
	const int POS_BR = 1;
	const int POS_TL = 2;
	const int POS_TR = 3;
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;

	// ensure we're within the boundaries of the image.
	if (i < pOutSize.x && j < pOutSize.y)
	{

		double pixelValue = 0.0; // default the pixel value to zero.
			
		// convert each of these four pixels to the input coordinate system.
		bool pixelValid = true;

		// get the position of the interpolation point in the output image.
		Reprojection::rpVectI outPixel = {	.x = i,
							.y = j };
											
		// convert these output image pixel coordinates into input image pixel coordinates
		// by interpolating the map. we pass in the pixelValid boolean again, but at this
		// point we already know that it is valid.
		Reprojection::rpVectD inPixelInterpolationPoint = applyMap(	outPixel,
										pProjectionMap,
										pMapValid,
										pMapSize,
										&pixelValid,
										pOutSize );

		// calculate the four pixel coordinates surrounding this interpolation point.
		Reprojection::rpVectI pixel[4];
		pixel[POS_BL].x = (int) floor( inPixelInterpolationPoint.x );
		pixel[POS_BL].y = (int) floor( inPixelInterpolationPoint.y );
		pixel[POS_BR].x = pixel[POS_BL].x + 1; pixel[POS_BR].y = pixel[POS_BL].y;
		pixel[POS_TL].x = pixel[POS_BL].x; pixel[POS_TL].y = pixel[POS_BL].y + 1;
		pixel[POS_TR].x = pixel[POS_BL].x + 1; pixel[POS_TR].y = pixel[POS_BL].y + 1;

		// ensure the pixel is within the extent of the input image.
		Reprojection::rpVectI nearestPixel = {	.x = (int) round( inPixelInterpolationPoint.x ),
							.y = (int) round( inPixelInterpolationPoint.y ) };

		// ensure all pixels are within the extent of the input image.
		bool withinRange = true;
		for ( int m = 0; m < 4; m++ )
		{
			withinRange = withinRange && (pixel[m].x >= 0) && (pixel[m].x < pInSize.x);
			withinRange = withinRange && (pixel[m].y >= 0) && (pixel[m].y < pInSize.y);
		}
		if (withinRange == true)
		{

			// calculate memory location of this pixel within the input image.
			int location[4];
			for ( int m = 0; m < 4; m++ )
				location[m] = (pixel[m].y * pInSize.x) + pixel[m].x;

			// if we have an input mask then check if any of these pixels are masked.
			bool masked = false;
			if (pInMask != NULL)
				for ( int m = 0; m < 4; m++ )
					masked = masked || (pInMask[ location[m] ] == false);

			// only add up values if we're not masked.
			if (masked == false)
			{

				// get an bilinearly interpolated value from the input pixel image.
				double value = Reprojection::interpolateValue(	inPixelInterpolationPoint,
										pInImage[ location[POS_BL] ],
										pInImage[ location[POS_BR] ],
										pInImage[ location[POS_TL] ],
										pInImage[ location[POS_TR] ] );

				// only add up the primary beam if a primary beam was supplied.
				if (pBeamIn != NULL && (pNormalisationPattern != NULL || pPrimaryBeamPattern != NULL || pAProjection == true))
				{

					// add up the primary beam for these interpolation points.
					double beamIn = Reprojection::interpolateValue(	inPixelInterpolationPoint,
												pBeamIn[ location[POS_BL] ],
												pBeamIn[ location[POS_BR] ],
												pBeamIn[ location[POS_TL] ],
												pBeamIn[ location[POS_TR] ] );

					// if we should be applying the primary beam then we multiply the pixel value by the beam value.
					if (pAProjection == true)
						value *= beamIn;

					// get the primary beam for this pixel position. the input primary beam is the correct beam for the input image, but
					// its phase position and size correspond to those of the output image.
// cjs-mod					double beamIn = pBeamIn[ (j * pOutSize.x) + i ]; // cjs-mod

					// the normalisation pattern is the sum of the primary beams at this pixel, and will be used later to normalise this pixel.
					if (pNormalisationPattern != NULL)
					{
// cjs-mod						value *= beamIn; // pow( beamIn, 2 );
						pNormalisationPattern[ (j * pOutSize.x) + i ] += pow( beamIn, 1 );
					}

					// the primary beam pattern is the sum of the primary beams squared at this pixel, and will be used later to suppress the noise
					// at the edges of the mosaic.
					if (pPrimaryBeamPattern != NULL)
						pPrimaryBeamPattern[ (j * pOutSize.x) + i ] += pow( beamIn, 2 );

				}

				// if the output image has an associated primary beam then we need to multiply our value by this beam. we don't need to do this if we're
				// using A-projection because A-projection will handle this function.
				if (pBeamOut != NULL && pAProjection == false)
					value *= pBeamOut[ (j * pOutSize.x) + i ];

				pixelValue = value;

			}

		}	

		// update the pixel value.
		pOutImage[ (j * pOutSize.x) + i ] += pixelValue;

	}

} // devReprojectionOutToIn

__global__ void devReprojectionOutToIn( float * pInImage, float * pOutImage, double * pProjectionMap, bool * pMapValid,
						double * pNormalisationPattern, double * pPrimaryBeamPattern, Reprojection::rpVectI pMapSize, Reprojection::rpVectI pInSize,
						Reprojection::rpVectI pOutSize, bool * pInMask, double * pBeamIn, double * pBeamOut,
						bool pAProjection )
{
	
	const int POS_BL = 0;
	const int POS_BR = 1;
	const int POS_TL = 2;
	const int POS_TR = 3;
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;

	// ensure we're within the boundaries of the image.
	if (i < pOutSize.x && j < pOutSize.y)
	{

		float pixelValue = 0.0; // default the pixel value to zero.
			
		// convert each of these four pixels to the input coordinate system.
		bool pixelValid = true;

		// get the position of the interpolation point in the output image.
		Reprojection::rpVectI outPixel = {	.x = i,
							.y = j };
											
		// convert these output image pixel coordinates into input image pixel coordinates
		// by interpolating the map. we pass in the pixelValid boolean again, but at this
		// point we already know that it is valid.
		Reprojection::rpVectD inPixelInterpolationPoint = applyMap(	outPixel,
										pProjectionMap,
										pMapValid,
										pMapSize,
										&pixelValid,
										pOutSize );

		// calculate the four pixel coordinates surrounding this interpolation point.
		Reprojection::rpVectI pixel[4];
		pixel[POS_BL].x = (int) floor( inPixelInterpolationPoint.x );
		pixel[POS_BL].y = (int) floor( inPixelInterpolationPoint.y );
		pixel[POS_BR].x = pixel[POS_BL].x + 1; pixel[POS_BR].y = pixel[POS_BL].y;
		pixel[POS_TL].x = pixel[POS_BL].x; pixel[POS_TL].y = pixel[POS_BL].y + 1;
		pixel[POS_TR].x = pixel[POS_BL].x + 1; pixel[POS_TR].y = pixel[POS_BL].y + 1;

		// ensure the pixel is within the extent of the input image.
		Reprojection::rpVectI nearestPixel = {	.x = (int) round( inPixelInterpolationPoint.x ),
							.y = (int) round( inPixelInterpolationPoint.y ) };

		// ensure all pixels are within the extent of the input image.
		bool withinRange = true;
		for ( int m = 0; m < 4; m++ )
		{
			withinRange = withinRange && (pixel[m].x >= 0) && (pixel[m].x < pInSize.x);
			withinRange = withinRange && (pixel[m].y >= 0) && (pixel[m].y < pInSize.y);
		}
		if (withinRange == true)
		{

			// calculate memory location of this pixel within the input image.
			int location[4];
			for ( int m = 0; m < 4; m++ )
				location[m] = (pixel[m].y * pInSize.x) + pixel[m].x;

			// if we have an input mask then check if any of these pixels are masked.
			bool masked = false;
			if (pInMask != NULL)
				for ( int m = 0; m < 4; m++ )
					masked = masked || (pInMask[ location[m] ] == false);

			// only add up values if we're not masked.
			if (masked == false)
			{

				// get an bilinearly interpolated value from the input pixel image.
				double value = Reprojection::interpolateValue(	inPixelInterpolationPoint,
										(double) pInImage[ location[POS_BL] ],
										(double) pInImage[ location[POS_BR] ],
										(double) pInImage[ location[POS_TL] ],
										(double) pInImage[ location[POS_TR] ] );

				// only add up the primary beam if a primary beam was supplied.
				if (pBeamIn != NULL && (pNormalisationPattern != NULL || pPrimaryBeamPattern != NULL || pAProjection == true))
				{

					// add up the primary beam for these interpolation points.
					double beamIn = Reprojection::interpolateValue(	inPixelInterpolationPoint,
												pBeamIn[ location[POS_BL] ],
												pBeamIn[ location[POS_BR] ],
												pBeamIn[ location[POS_TL] ],
												pBeamIn[ location[POS_TR] ] );

					// if we should be applying the primary beam then we multiply the pixel value by the beam value.
					if (pAProjection == true)
						value *= beamIn;

					// get the primary beam for this pixel position. the input primary beam is the correct beam for the input image, but
					// its phase position and size correspond to those of the output image.
// cjs-mod					double beamIn = pBeamIn[ (j * pOutSize.x) + i ]; // cjs-mod

					// the normalisation pattern is the sum of the primary beams at this pixel, and will be used later to normalise this pixel.
					if (pNormalisationPattern != NULL)
					{
// cjs-mod						value *= beamIn; // pow( beamIn, 2 );
						pNormalisationPattern[ (j * pOutSize.x) + i ] += pow( beamIn, 1 );
					}

					// the primary beam pattern is the sum of the primary beams squared at this pixel, and will be used later to suppress the noise
					// at the edges of the mosaic.
					if (pPrimaryBeamPattern != NULL)
						pPrimaryBeamPattern[ (j * pOutSize.x) + i ] += pow( beamIn, 2 );

				}

				// if the output image has an associated primary beam then we need to multiply our value by this beam. we don't need to do this if we're
				// using A-projection because A-projection will handle this function.
				if (pBeamOut != NULL && pAProjection == false)
					value *= (double) pBeamOut[ (j * pOutSize.x) + i ];

				pixelValue = (float) value;

			}

		}	

		// update the pixel value.
		pOutImage[ (j * pOutSize.x) + i ] += pixelValue;

	}

} // devReprojectionOutToIn

//
//	devReweight()
//
//	CJS: 05/12/2018
//
//	Divide each pixel by its normalisation pattern. We are constructing an (weighted) average of our mosaic.
//

__global__ void devReweight( double * pOutImage, double * pNormalisationPattern, double * pPrimaryBeamPattern, Reprojection::rpVectI pOutSize)
{
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;

	// ensure we're within the boundaries of the image.
	if (i < pOutSize.x && j < pOutSize.y)
	{

		// apply the normalisation pattern to the image. this step corrects for the weighting applied to each pixel depending on its position in the primary beam.
		if (pNormalisationPattern != NULL)
		{

			double normalisationPattern = pNormalisationPattern[ (j * pOutSize.x) + i ];
			if (normalisationPattern != 0.0)
				pOutImage[ (j * pOutSize.x) + i ] /= normalisationPattern;
			else
				pOutImage[ (j * pOutSize.x) + i ] = 0.0;

		}

		// apply the primary beam pattern to the image. this step suppresses the noise near the edge of the mosaic.
		if (pPrimaryBeamPattern != NULL)
		{

			double primaryBeamPattern = pPrimaryBeamPattern[ (j * pOutSize.x) + i ];

			pOutImage[ (j * pOutSize.x) + i ] *= primaryBeamPattern;

		}

	}

} // devReweight

//
//	P U B L I C   C L A S S   M E M B E R S
//

//
//	Reprojection::Reprojection()
//
//	CJS: 14/11/2018
//
//	The constructor.
//

Reprojection::Reprojection()
{

	// initialise arrays to null.
	_devProjectionMap = NULL;
	_devMapValid = NULL;
	
} // Reprojection::Reprojection

//
//	Reprojection::~Reprojection()
//
//	CJS: 14/11/2018
//
//	The destructor.
//

Reprojection::~Reprojection()
{

	// free device memory.
	if (_devProjectionMap != NULL)
		cudaFree( _devProjectionMap );
	if (_devMapValid != NULL)
		cudaFree( _devMapValid );
	
} // Reprojection::~Reprojection

//
//	Reprojection::CreateDeviceMemory()
//
//	CJS: 10/04/2019
//
//	Sets up the device arrays required for reprojection. These will be destroyed in the destructor.
//

void Reprojection::CreateDeviceMemory( rpVectI pOutSize )
{

	cudaError_t err;

	// work out a suitable grid size.
	rpVectI mapSize;
	mapSize.x = (int) ceil( (double) (pOutSize.x + 2) );
	mapSize.y = (int) ceil( (double) (pOutSize.y + 2) );

	// map between output and input images.
	err = cudaMalloc( (void **) &_devProjectionMap, mapSize.x * mapSize.y * sizeof( double ) * 2 );
	if (err != cudaSuccess)
		printf( "error creating device memory for output/input pixel map (%s)\n", cudaGetErrorString( err ) );

	// map valid flag.
	err = cudaMalloc( (void **) &_devMapValid, mapSize.x * mapSize.y * sizeof( bool ) );
	if (err != cudaSuccess)
		printf( "error creating device memory for the output/input pixel map validity flag (%s)\n", cudaGetErrorString( err ) );

} // Reprojection::CreateDeviceMemory

//
//	Rprojection:: GetCoordinates()
//
//	CJS: 04/02/2020
//
//	Gets the RA and DEC for a single pixel.
//

void Reprojection::GetCoordinates( double pX, double pY, rpCoordSys pCoordinateSystem, rpVectI pSize, double * pPhaseRA, double * pPhaseDEC )
{

	// calculate the rotation matrices to convert from pixel coordinates (centred at ra 0, dec 0) to world coordinates at the required ra, dec.
	calculateRotationMatrix( &pCoordinateSystem, true );
			
	Reprojection::rpVectD oldPixelCoordinate = { .x = pX, .y = pY };
					
	// convert these pixel coordinates to world coordinates (cartesian).
	bool wrapAround = false;
	Reprojection::rpVectD worldCoordinate = pixelToWorld( oldPixelCoordinate, pCoordinateSystem, &wrapAround );

	// update the RA and DEC with the calculate values.
	*pPhaseRA = deg( atan2( worldCoordinate.y, worldCoordinate.x ) );
	*pPhaseDEC = deg( asin( worldCoordinate.z ) );
	if (*pPhaseRA < 0)
		*pPhaseRA += 360.0;

} // Reprojection::GetCoordinates

//
//	Rprojection:: ReprojectImage()
//
//	CJS: 14/11/2018
//
//	Do an image-plane reprojection.
//

void Reprojection::ReprojectImage( double * pdevInImage, double * pdevOutImage, double * pdevNormalisationPattern, double * pdevPrimaryBeamPattern,
					rpCoordSys pInCoordinateSystem, rpCoordSys pOutCoordinateSystem, rpVectI pInSize, rpVectI pOutSize, bool * pdevInMask,
					double * pdevBeamIn, double * pdevBeamOut, ProjectionDirection pProjectionDirection,
					bool pAProjection, bool pVerbose )
{
	
	// get some properties from the device.
	cudaDeviceProp gpuProperties;
	cudaGetDeviceProperties( &gpuProperties, 0 );
	_maxThreadsPerBlock = gpuProperties.maxThreadsPerBlock;

	// set up epoch conversion matrix.
	_epochConversion.a11 = 1;
	_epochConversion.a22 = 1;
	_epochConversion.a33 = 1;

	// store parameters.
	_inCoordinateSystem = pInCoordinateSystem;
	_outCoordinateSystem = pOutCoordinateSystem;
	_inSize = pInSize;
	_outSize = pOutSize;

	// calculate the rotation matrices to convert from pixel coordinates (centred at ra 0, dec 0) to world coordinates at the required ra, dec.
	// we don't want to do this every time because that would be slow and unnecessary. we only do epoch conversion for the input coordinate system
	// conversion, because we have decided that everything in this program will use the output coordinate system epoch.
	calculateRotationMatrix( &_outCoordinateSystem, false );

	// calculate the inverse of the output CD transformation matrix. this inverse will be needed
	// when we transform from world to pixel coordinates.
	_outCoordinateSystem.inv_cd = calculateInverseMatrix( _outCoordinateSystem.cd );

//	printf( "image-plane reprojection:\n" );
	if (pVerbose == true)
	{
		printf( "        from <%f°,%f°> to <%f°,%f°>, ", _inCoordinateSystem.crVAL.x, _inCoordinateSystem.crVAL.y, _outCoordinateSystem.crVAL.x, _outCoordinateSystem.crVAL.y );
		printf( "size <%i,%i> to <%i,%i>\n", _inSize.x, _inSize.y, _outSize.x, _outSize.y );
	}

	// prepare the epoch conversion matrix. we convert from the input epoch to the output epoch because we have decided that all the world
	// coordinates in this program will use the output epoch.
	_epochConversion = doEpochConversion( _inCoordinateSystem, _outCoordinateSystem );

	// calculate the rotation matrices to convert from pixel coordinates (centred at ra 0, dec 0) to world coordinates at the required ra, dec.
	// we don't want to do this every time because that would be slow and unnecessary. we only do epoch conversion for the input coordinate system
	// conversion, because we have decided that everything in this program will use the output coordinate system epoch.
	calculateRotationMatrix( &_inCoordinateSystem, true );

	// calculate the inverse of the input CD transformation matrix. this inverse will be needed
	// when we transform from world to pixel coordinates.
	_inCoordinateSystem.inv_cd = calculateInverseMatrix( _inCoordinateSystem.cd );

	// do the reprojection and regridding.
	reprojection( pdevInImage, pdevOutImage, pdevNormalisationPattern, pdevPrimaryBeamPattern, pdevInMask, pdevBeamIn, pdevBeamOut, pProjectionDirection, pAProjection );

} // Reprojection::ReprojectImage

void Reprojection::ReprojectImage( float * pdevInImage, float * pdevOutImage, double * pdevNormalisationPattern, double * pdevPrimaryBeamPattern,
					rpCoordSys pInCoordinateSystem, rpCoordSys pOutCoordinateSystem, rpVectI pInSize, rpVectI pOutSize, bool * pdevInMask,
					double * pdevBeamIn, double * pdevBeamOut, ProjectionDirection pProjectionDirection,
					bool pAProjection, bool pVerbose )
{
	
	// get some properties from the device.
	cudaDeviceProp gpuProperties;
	cudaGetDeviceProperties( &gpuProperties, 0 );
	_maxThreadsPerBlock = gpuProperties.maxThreadsPerBlock;

	// set up epoch conversion matrix.
	_epochConversion.a11 = 1;
	_epochConversion.a22 = 1;
	_epochConversion.a33 = 1;

	// store parameters.
	_inCoordinateSystem = pInCoordinateSystem;
	_outCoordinateSystem = pOutCoordinateSystem;
	_inSize = pInSize;
	_outSize = pOutSize;

	// calculate the rotation matrices to convert from pixel coordinates (centred at ra 0, dec 0) to world coordinates at the required ra, dec.
	// we don't want to do this every time because that would be slow and unnecessary. we only do epoch conversion for the input coordinate system
	// conversion, because we have decided that everything in this program will use the output coordinate system epoch.
	calculateRotationMatrix( &_outCoordinateSystem, false );

	// calculate the inverse of the output CD transformation matrix. this inverse will be needed
	// when we transform from world to pixel coordinates.
	_outCoordinateSystem.inv_cd = calculateInverseMatrix( _outCoordinateSystem.cd );

//	printf( "image-plane reprojection:\n" );
	if (pVerbose == true)
	{
		printf( "        from <%f°,%f°> to <%f°,%f°>, ", _inCoordinateSystem.crVAL.x, _inCoordinateSystem.crVAL.y, _outCoordinateSystem.crVAL.x, _outCoordinateSystem.crVAL.y );
		printf( "size <%i,%i> to <%i,%i>\n", _inSize.x, _inSize.y, _outSize.x, _outSize.y );
	}

	// prepare the epoch conversion matrix. we convert from the input epoch to the output epoch because we have decided that all the world
	// coordinates in this program will use the output epoch.
	_epochConversion = doEpochConversion( _inCoordinateSystem, _outCoordinateSystem );

	// calculate the rotation matrices to convert from pixel coordinates (centred at ra 0, dec 0) to world coordinates at the required ra, dec.
	// we don't want to do this every time because that would be slow and unnecessary. we only do epoch conversion for the input coordinate system
	// conversion, because we have decided that everything in this program will use the output coordinate system epoch.
	calculateRotationMatrix( &_inCoordinateSystem, true );

	// calculate the inverse of the input CD transformation matrix. this inverse will be needed
	// when we transform from world to pixel coordinates.
	_inCoordinateSystem.inv_cd = calculateInverseMatrix( _inCoordinateSystem.cd );

	// do the reprojection and regridding.
	reprojection( pdevInImage, pdevOutImage, pdevNormalisationPattern, pdevPrimaryBeamPattern, pdevInMask, pdevBeamIn, pdevBeamOut, pProjectionDirection, pAProjection );

} // Reprojection::ReprojectImage

//
//	Rprojection:: ReprojectPixel()
//
//	CJS: 04/02/2020
//
//	Reproject a single pixel from one image to another.
//

void Reprojection::ReprojectPixel( double * pPixel, int pNumPixels, rpCoordSys pInCoordinateSystem, rpCoordSys pOutCoordinateSystem, rpVectI pInSize, rpVectI pOutSize )
{

	// set up epoch conversion matrix.
	_epochConversion.a11 = 1;
	_epochConversion.a22 = 1;
	_epochConversion.a33 = 1;

	// store parameters.
	_inCoordinateSystem = pInCoordinateSystem;
	_outCoordinateSystem = pOutCoordinateSystem;
	_inSize = pInSize;
	_outSize = pOutSize;

	// calculate the rotation matrices to convert from pixel coordinates (centred at ra 0, dec 0) to world coordinates at the required ra, dec.
	// we don't want to do this every time because that would be slow and unnecessary. we only do epoch conversion for the input coordinate system
	// conversion, because we have decided that everything in this program will use the output coordinate system epoch.
	calculateRotationMatrix( &pOutCoordinateSystem, false );

	// calculate the inverse of the output CD transformation matrix. this inverse will be needed
	// when we transform from world to pixel coordinates.
	pOutCoordinateSystem.inv_cd = calculateInverseMatrix( pOutCoordinateSystem.cd );

	// prepare the epoch conversion matrix. we convert from the input epoch to the output epoch because we have decided that all the world
	// coordinates in this program will use the output epoch.
	_epochConversion = doEpochConversion( pInCoordinateSystem, pOutCoordinateSystem );

	// calculate the rotation matrices to convert from pixel coordinates (centred at ra 0, dec 0) to world coordinates at the required ra, dec.
	// we don't want to do this every time because that would be slow and unnecessary. we only do epoch conversion for the input coordinate system
	// conversion, because we have decided that everything in this program will use the output coordinate system epoch.
	calculateRotationMatrix( &pInCoordinateSystem, true );

	// calculate the inverse of the input CD transformation matrix. this inverse will be needed
	// when we transform from world to pixel coordinates.
	pInCoordinateSystem.inv_cd = calculateInverseMatrix( pInCoordinateSystem.cd );

	// loop over each pixel.
	for ( int i = 0; i < pNumPixels; i++ )
	{
			
		Reprojection::rpVectD oldPixelCoordinate = { .x = pPixel[ i * 2 ], .y = pPixel[ (i * 2) + 1 ] };
					
		// convert these pixel coordinates to world coordinates (cartesian).
		bool wrapAround = false;
		Reprojection::rpVectD worldCoordinate = pixelToWorld( oldPixelCoordinate, pInCoordinateSystem, &wrapAround );
			
		// convert the world coordinates back into input pixel coordinates.
		Reprojection::rpVectD newPixelCoordinate = worldToPixel( worldCoordinate, pOutCoordinateSystem, &wrapAround );

		// update the pixel array with the output pixel coordinates.
		pPixel[ i * 2 ] = newPixelCoordinate.x;
		pPixel[ (i * 2) + 1 ] = newPixelCoordinate.y;

	}

} // Reprojection::ReprojectPixel

//
//	Reprojection::ReweightImage()
//
//	CJS: 05/12/2018
//
//	Calculate the pixel value by dividing each pixel by its weight.
//

void Reprojection::ReweightImage( double * pdevOutImage, double * pdevNormalisationPattern, double * pdevPrimaryBeamPattern, rpVectI pOutSize, bool * pdevOutMask )
{

	cudaError_t err;

	// work out size of kernel call.
	dim3 gridSize2D( 1, 1 );
	dim3 blockSize2D( 1, 1 );

	// if we have been supplied with a primary beam pattern then normalise the image so that the primary been pattern has a maximum of 1, and update the mask if one is
	// provided.
	if (pdevPrimaryBeamPattern != NULL)
	{

		// take the square root of each of the pixels (since the primary beams are added in quadrature).
		setThreadBlockSize2D( _outSize.x, _outSize.y, gridSize2D, blockSize2D );
		devSquareRoot<<< gridSize2D, blockSize2D >>>(	/* pImage = */ pdevPrimaryBeamPattern,
								/* pMapSize = */ _outSize );

		// create memory to hold the maximum pixel value.
		double * devMaxValue;
		err = cudaMalloc( (void **) &devMaxValue, sizeof( double ) );
		if (err != cudaSuccess)
			printf( "error creating device memory for the maximum value of the primary beam pattern (%s)\n", cudaGetErrorString( err ) );
		else
		{

			// set the maximum value to zero.
			cudaMemset( (void *) devMaxValue, 0, sizeof( double ) );

			// get the maximum pixel value.
			devGetMaxPixel<<< 1, 1 >>>(	/* pImage = */ pdevPrimaryBeamPattern,
							/* pMaxValue = */ devMaxValue,
							/* pMapSize = */ _outSize );

			// normalise the primary beam pattern using the maximum value.
			setThreadBlockSize2D( _outSize.x, _outSize.y, gridSize2D, blockSize2D );
			devNormalise<<< gridSize2D, blockSize2D >>>(	/* pImage = */ pdevPrimaryBeamPattern,
									/* pNormalisation = */ devMaxValue,
									/* pMapSize = */ _outSize );

		}

		// free memory.
		if (devMaxValue != NULL)
			cudaFree( (void *) devMaxValue );

		// set the mask (if one is provided).
		if (pdevOutMask != NULL)
		{

			setThreadBlockSize2D( _outSize.x, _outSize.y, gridSize2D, blockSize2D );
			devSetMask<<< gridSize2D, blockSize2D >>>(	/* pImage = */ pdevPrimaryBeamPattern,
									/* pMask = */ pdevOutMask,
									/* pMapSize = */ _outSize );

		}

	}

	// ensure weight has been calculated.
	if (pdevNormalisationPattern != NULL)
	{

		// divide each pixel by its normalisation pattern.
		setThreadBlockSize2D( pOutSize.x, pOutSize.y, gridSize2D, blockSize2D );
		devReweight<<< gridSize2D, blockSize2D >>>(	/* pOutImage = */ pdevOutImage,
								/* pNormalisationPattern = */ pdevNormalisationPattern,
								/* pPrimaryBeamPattern = */ pdevPrimaryBeamPattern,
								/* pOutSize = */ pOutSize );
		err = cudaGetLastError();
		if (err != cudaSuccess)
			printf( "error reweighting mosaic image on the device (%s)\n", cudaGetErrorString( err ) );

	}

} // Reprojection::ReweightImage

//
//	G E N E R A L   F U N C T I O N S
//

//
//	Reprojection::minD(), minI()
//
//	CJS: 08/07/2015
//
//	Find the minimum of two values. Double and Int types implemented.
//

__host__ __device__ double Reprojection::minD( double pA, double pB )
{
	
	return (pA < pB) ? pA : pB;

} // Reprojection::minD

__host__ __device__ int Reprojection::minI( int pA, int pB )
{
	
	return (pA < pB) ? pA : pB;

} // Reprojection::minI

//
//	Reprojection::maxD(), maxI()
//
//	CJS: 08/07/2015
//
//	Find the maximum of two values. Double and Int types implemented.
//

__host__ __device__ double Reprojection::maxD( double pA, double pB )
{
	
	return (pA > pB) ? pA : pB;

} // Reprojection::maxD

__host__ __device__ int Reprojection::maxI( int pA, int pB )
{
	
	return (pA > pB) ? pA : pB;

} // Reprojection::maxI

//
//	Reprojection::toUppercase()
//
//	CJS: 03/08/2015
//
//	Convert a string to uppercase.
//

void Reprojection::toUppercase( char * pChar )
{
	
	for ( char * ptr = pChar; *ptr; ptr++ )
		*ptr = toupper( *ptr );

} // Reprojection::toUppercase

//
//	Reprojection::correctOrientation()
//
//	CJS: 27/11/2015
//
//	Check if a square of points has the correct orientation. If some of the points have crossed the boundary
//	where an image is stitched together on the opposite side of the celestial sphere then the square will not
//	have the correct orientation, and bilinear interpolation will be unpredicatable.
//

__host__ __device__ bool Reprojection::correctOrientation( rpVectD pTopLeft, rpVectD pTopRight, rpVectD pBottomLeft, rpVectD pBottomRight )
{
	
	bool squareValid = true;
						
	// first, get some vectors between our grid points.
	rpVectD topLeftToTopRight = { .x = pTopRight.x - pTopLeft.x, .y = pTopRight.y - pTopLeft.y, .z = pTopRight.z - pTopLeft.z };
	rpVectD topLeftToBottomRight = { .x = pBottomRight.x - pTopLeft.x, .y = pBottomRight.y - pTopLeft.y, .z = pBottomRight.z - pTopLeft.z };
	rpVectD topLeftToBottomLeft = { .x = pBottomLeft.x - pTopLeft.x, .y = pBottomLeft.y - pTopLeft.y, .z = pBottomLeft.z - pTopLeft.z };
		
	// take the cross products of TL->BR x TL->TR, and TL->BL x TL->BR. the z-components should both be positive if
	// our square is pointing in the right direction.
	double z1 = (topLeftToBottomRight.x * topLeftToTopRight.y) - (topLeftToBottomRight.y * topLeftToTopRight.x);
	double z2 = (topLeftToBottomLeft.x * topLeftToBottomRight.y) - (topLeftToBottomLeft.y * topLeftToBottomRight.x);
			
	// if either of these z-components is negative then this pixel is invalid.
	if (z1 <= 0 || z2 <= 0)
		squareValid = false;
	
	// return flag.
	return squareValid;

} // Reprojection::correctOrientation

//
//	P R I V A T E   C L A S S   M E M B E R S
//

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

void Reprojection::setThreadBlockSize2D( int pThreadsX, int pThreadsY, dim3 & pGridSize2D, dim3 & pBlockSize2D )
{
	
	// store the total number of X and Y threads.
	int totalThreadsX = pThreadsX;
	int totalThreadsY = pThreadsY;
	
	pBlockSize2D.x = pThreadsX;
	pBlockSize2D.y = pThreadsY;
	pGridSize2D.x = 1;
	pGridSize2D.y = 1;
	
	// do we have too many threads?
	while ( (pBlockSize2D.x * pBlockSize2D.y) > _maxThreadsPerBlock )
	{
		
		// increment the number of Y blocks.
		pGridSize2D.y = pGridSize2D.y + 1;
		pBlockSize2D.y = (int) ceil( (double)totalThreadsY / (double)pGridSize2D.y );
		
		// if this doesn't help, increment the number of X blocks. if we have multiple iterations of this loop then
		// we will be incrementing Y, X, Y, X, Y, X, Y, .... etc.
		if ( (pBlockSize2D.x * pBlockSize2D.y) > _maxThreadsPerBlock )
		{
			pGridSize2D.x = pGridSize2D.x + 1;
			pBlockSize2D.x = (int) ceil( (double) totalThreadsX / (double) pGridSize2D.x );
		}
		
	}
	
} // setThreadBlockSize2D

//
//	T R I G   F U N C T I O N S
//

//
//	Reprojection::rad(), deg()
//
//	CJS:	10/07/2015
//
//	convert between degrees and radians.
//

__host__ __device__ double Reprojection::rad( double pIn )
{
	
	return ( pIn * PI / 180.0 );

} // Reprojection::rad

__host__ __device__ double Reprojection::deg( double pIn )
{
	
	return ( pIn * 180.0 / PI );

} // Reprojection::deg

//
//	Reprojection::arctan()
//
//	CJS:	10/07/2015
//
//	calculate the arctangent of top / bottom, and convert to degrees.
//

double Reprojection::arctan( double pValueTop, double pValueBottom )
{
	
	// calculate arctangent of top / bottom.
	double result;
	if (pValueBottom != 0)
		result = deg( atan( pValueTop / pValueBottom ) );
	else
		result = (pValueTop >= 0) ? 90 : -90;
	
	// we should have a result between -90 and +90 deg. if the denominator is negative then add 180 to be within
	// range 90 to 270 deg.
	if (pValueBottom < 0)
		result = result + 180;
	
	return result;

} // Reprojection::arctan

//
//	Reprojection::angleRange()
//
//	CJS:	10/07/2015
//
//	Ensure that an angle is within a given range by adding +/-n.360.
//

__host__ __device__ void Reprojection::angleRange( double * pValue, double pCentre, double pMax )
{
	
	while (*pValue < (pCentre - pMax))
		*pValue = *pValue + (2 * pMax);
	while (*pValue > (pCentre + pMax))
		*pValue = *pValue - (2 * pMax);

} // Reprojection::angleRange

//
//	M A T R I X   F U N C T I O N S
//

//
//	Reprojection::calculateInverseMatrix()
//
//	CJS:	08/07/2015
//
//	calculate the inverse of a 2x2 matrix.
//

__host__ __device__ Reprojection::rpMatr2x2 Reprojection::calculateInverseMatrix( rpMatr2x2 pMatrix )
{
	
	// calculate the determinant.
	double determinant = (pMatrix.a11 * pMatrix.a22) - (pMatrix.a12 * pMatrix.a21);
		
	// create an empty matrix.
	rpMatr2x2 inverse = { .a11 = 0, .a12 = 0, .a21 = 0, .a22 = 0 };
	
	// the matrix is supposed to be invertible, but we should check anyway.
	if (determinant != 0)
	{
		inverse.a11 = pMatrix.a22 / determinant;
		inverse.a12 = -pMatrix.a12 / determinant;
		inverse.a21 = -pMatrix.a21 / determinant;
		inverse.a22 = pMatrix.a11 / determinant;
	}
	
	// return the inverse matrix.
	return inverse;

} // Reprojection::calculateInverseMatrix

//
//	Reprojection::transpose()
//
//	CJS: 28/07/2015
//
//	Construct the transpose of a 3x3 matrix.
//

__host__ __device__ Reprojection::rpMatr3x3 Reprojection::transpose( rpMatr3x3 pOldMatrix )
{
	
	rpMatr3x3 newMatrix = pOldMatrix;
	
	// copy transposed cells.
	newMatrix.a12 = pOldMatrix.a21;
	newMatrix.a13 = pOldMatrix.a31;
	newMatrix.a21 = pOldMatrix.a12;
	newMatrix.a23 = pOldMatrix.a32;
	newMatrix.a31 = pOldMatrix.a13;
	newMatrix.a32 = pOldMatrix.a23;
	
	// return something.
	return newMatrix;

} // Reprojection::transpose

//
//	Reprojection::multMatrixVector()
//
//	CJS: 27/07/2015
//
//	Multiply a matrix by a vector.
//

__host__ __device__ Reprojection::rpVectD Reprojection::multMatrixVector( rpMatr3x3 pMatrix, rpVectD pVector )
{
	
	rpVectD newVector;
	
	// multiply 3x3 matrix with 3x1 vector.
	newVector.x = (pMatrix.a11 * pVector.x) + (pMatrix.a12 * pVector.y) + (pMatrix.a13 * pVector.z);
	newVector.y = (pMatrix.a21 * pVector.x) + (pMatrix.a22 * pVector.y) + (pMatrix.a23 * pVector.z);
	newVector.z = (pMatrix.a31 * pVector.x) + (pMatrix.a32 * pVector.y) + (pMatrix.a33 * pVector.z);
	
	// return something.
	return newVector;

} // Reprojection::multMatrixVector

//
//	Reprojection::multMatrix
//
//	CJS: 27/07/2015
//
//	Multiply two 3x3 matrices together.
//

__host__ __device__ Reprojection::rpMatr3x3 Reprojection::multMatrix( rpMatr3x3 pMatrix1, rpMatr3x3 pMatrix2 )
{
	
	rpMatr3x3 newMatrix;
	
	// row 1.
	newMatrix.a11 = (pMatrix1.a11 * pMatrix2.a11) + (pMatrix1.a12 * pMatrix2.a21) + (pMatrix1.a13 * pMatrix2.a31);
	newMatrix.a12 = (pMatrix1.a11 * pMatrix2.a12) + (pMatrix1.a12 * pMatrix2.a22) + (pMatrix1.a13 * pMatrix2.a32);
	newMatrix.a13 = (pMatrix1.a11 * pMatrix2.a13) + (pMatrix1.a12 * pMatrix2.a23) + (pMatrix1.a13 * pMatrix2.a33);
	
	// row 2.
	newMatrix.a21 = (pMatrix1.a21 * pMatrix2.a11) + (pMatrix1.a22 * pMatrix2.a21) + (pMatrix1.a23 * pMatrix2.a31);
	newMatrix.a22 = (pMatrix1.a21 * pMatrix2.a12) + (pMatrix1.a22 * pMatrix2.a22) + (pMatrix1.a23 * pMatrix2.a32);
	newMatrix.a23 = (pMatrix1.a21 * pMatrix2.a13) + (pMatrix1.a22 * pMatrix2.a23) + (pMatrix1.a23 * pMatrix2.a33);
	
	// row 3.
	newMatrix.a31 = (pMatrix1.a31 * pMatrix2.a11) + (pMatrix1.a32 * pMatrix2.a21) + (pMatrix1.a33 * pMatrix2.a31);
	newMatrix.a32 = (pMatrix1.a31 * pMatrix2.a12) + (pMatrix1.a32 * pMatrix2.a22) + (pMatrix1.a33 * pMatrix2.a32);
	newMatrix.a33 = (pMatrix1.a31 * pMatrix2.a13) + (pMatrix1.a32 * pMatrix2.a23) + (pMatrix1.a33 * pMatrix2.a33);
	
	// return something.
	return newMatrix;

} // Reprojection::multMatrix

//
//	E U L E R   R O T A T I O N   F U N C T I O N S
//

//
//	Reprojection::rotateX
//
//	CJS: 24/07/2015
//
//	Construct a 3x3 matrix to rotate a vector about the X-axis.
//

Reprojection::rpMatr3x3 Reprojection::rotateX( double pAngle )
{
	
	rpMatr3x3 rotationMatrix;
	
	// row 1.
	rotationMatrix.a11 = 1;
	rotationMatrix.a12 = 0;
	rotationMatrix.a13 = 0;
	
	// row 2.
	rotationMatrix.a21 = 0;
	rotationMatrix.a22 = cos( rad( pAngle ) );
	rotationMatrix.a23 = -sin( rad( pAngle ) );
	
	// row 3.
	rotationMatrix.a31 = 0;
	rotationMatrix.a32 = sin( rad( pAngle ) );
	rotationMatrix.a33 = cos( rad( pAngle ) );
	
	// return something.
	return rotationMatrix;

} // Reprojection::rotateX

//
//	Reprojection::rotateY
//
//	CJS: 24/07/2015
//
//	Construct a 3x3 matrix to rotate a vector about the Y-axis.
//

Reprojection::rpMatr3x3 Reprojection::rotateY( double pAngle )
{
	
	rpMatr3x3 rotationMatrix;
	
	// row 1.
	rotationMatrix.a11 = cos( rad( pAngle ) );
	rotationMatrix.a12 = 0;
	rotationMatrix.a13 = -sin( rad( pAngle ) );
	
	// row 2.
	rotationMatrix.a21 = 0;
	rotationMatrix.a22 = 1;
	rotationMatrix.a23 = 0;
	
	// row 3.
	rotationMatrix.a31 = sin( rad( pAngle ) );
	rotationMatrix.a32 = 0;
	rotationMatrix.a33 = cos( rad( pAngle ) );
	
	// return something.
	return rotationMatrix;

} // Reprojection::rotateY

//
//	Reprojection::rotateZ
//
//	CJS: 24/07/2015
//
//	Construct a 3x3 matrix to rotate a vector about the Z-axis.
//

Reprojection::rpMatr3x3 Reprojection::rotateZ( double pAngle )
{
	
	rpMatr3x3 rotationMatrix;
	
	// row 1.
	rotationMatrix.a11 = cos( rad( pAngle ) );
	rotationMatrix.a12 = -sin( rad( pAngle ) );
	rotationMatrix.a13 = 0;
	
	// row 2.
	rotationMatrix.a21 = sin( rad( pAngle ) );
	rotationMatrix.a22 = cos( rad( pAngle ) );
	rotationMatrix.a23 = 0;
	
	// row 3.
	rotationMatrix.a31 = 0;
	rotationMatrix.a32 = 0;
	rotationMatrix.a33 = 1;
	
	// return something.
	return rotationMatrix;

} // Reprojection::rotateZ

//
//	Reprojection::calculateRotationMatrix()
//
//	CJS: 03/08/2015
//
//	Calculate a rotation matrix that moves from pixel coordinates (i.e. centred on ra 0, dec 0) to the required ra and dec. The
//	inverse matrix is also calculated.
//

void Reprojection::calculateRotationMatrix( rpCoordSys * pCoordinateSystem, bool pEpochConversion )
{
	
	// rotate about y-axis to bring to the correct latitude.
	pCoordinateSystem->toWorld = rotateY( pCoordinateSystem->crVAL.y );
		
	// rotate about z-axis to bring to the correct longitude.
	pCoordinateSystem->toWorld = multMatrix( rotateZ( pCoordinateSystem->crVAL.x ), pCoordinateSystem->toWorld );
		
	// do epoch conversion if required. if input and output are in the same epoch, then the epoch conversion matrix will be
	// the identity matrix.
	if (pEpochConversion == true)
		pCoordinateSystem->toWorld = multMatrix( _epochConversion, pCoordinateSystem->toWorld );
	
	// calculate the inverse as well. we'll need it to convert from world to pixel coordinates.
	pCoordinateSystem->toPixel = transpose( pCoordinateSystem->toWorld );

} // Reprojection::calculateRotationMatrix

//
//	E P O C H   C O N V E R S I O N   F U N C T I O N S
//

//
//	Reprojection::epochConversionMatrix()
//
//	CJS: 03/08/2015
//
//	Constructs a rotation matrix that converts coordinates from one epoch to another. This is done using a longitude rotation, a latitude rotation, and
//	another longitude rotation. The three rotation angles are specified as constants at the top of this program, and can be easily found using an online tool
//	such as NED (https://ned.ipac.caltech.edu/forms/calculator.html). Using NED, simply convert a position at RA 0, DEC 90 from one epoch to another and the three
//	rotation angles are given as the output coordinates (RA, DEC, PA).
//

Reprojection::rpMatr3x3 Reprojection::epochConversionMatrix( double pNP_RA, double pNP_DEC, double pNP_RA_OFFSET )
{
	
	// rotate about the Z-axis by RA to bring the output north pole to RA zero.
	rpMatr3x3 rotationMatrix = rotateZ( -pNP_RA );
	
	// rotate about the Y-axis by DEC to bring the output north pole to DEC 90.
	rotationMatrix = multMatrix( rotateY( 90 - pNP_DEC ), rotationMatrix );
	
	// rotate about the Z-axis by Position Angle (PA) to bring the output epoch origin to RA zero.
	rotationMatrix = multMatrix( rotateZ( pNP_RA_OFFSET ), rotationMatrix );
	
	// return something.
	return rotationMatrix;

} // Reprojection::epochConversionMatrix

//
//	Reprojection::doEpochConversion()
//
//	CJS: 03/08/2015
//
//	Construct a matrix that does epoch conversion between two positions. We simply compare the from and to
//	epoch, and then construct a suitable rotation matrix.
//

Reprojection::rpMatr3x3 Reprojection::doEpochConversion( rpCoordSys pFrom, rpCoordSys pTo )
{
	
	// default to no epoch conversion.
	rpMatr3x3 epochConversion = { .a11 = 1, .a12 = 0, .a13 = 0, .a21 = 0, .a22 = 1, .a23 = 0, .a31 = 0, .a32 = 0, .a33 = 1 };
	
	// J2000 to/from galactic.
	if (pFrom.epoch == EPOCH_J2000 && pTo.epoch == EPOCH_GALACTIC)
		epochConversion = epochConversionMatrix( NP_RA_GAL_IN_J2000, NP_DEC_GAL_IN_J2000, NP_RA_OFFSET_GAL_IN_J2000 );
	if (pFrom.epoch == EPOCH_GALACTIC && pTo.epoch == EPOCH_J2000)
		epochConversion = epochConversionMatrix( NP_RA_J2000_IN_GAL, NP_DEC_J2000_IN_GAL, NP_RA_OFFSET_J2000_IN_GAL );
	
	// B1950 to/from galactic.
	if (pFrom.epoch == EPOCH_B1950 && pTo.epoch == EPOCH_GALACTIC)
		epochConversion = epochConversionMatrix( NP_RA_GAL_IN_B1950, NP_DEC_GAL_IN_B1950, NP_RA_OFFSET_GAL_IN_B1950 );
	if (pFrom.epoch == EPOCH_GALACTIC && pTo.epoch == EPOCH_B1950)
		epochConversion = epochConversionMatrix( NP_RA_B1950_IN_GAL, NP_DEC_B1950_IN_GAL, NP_RA_OFFSET_B1950_IN_GAL );
	
	// B1950 to/from J2000.
	if (pFrom.epoch == EPOCH_B1950 && pTo.epoch == EPOCH_J2000)
		epochConversion = epochConversionMatrix( NP_RA_J2000_IN_B1950, NP_DEC_J2000_IN_B1950, NP_RA_OFFSET_J2000_IN_B1950 );
	if (pFrom.epoch == EPOCH_J2000 && pTo.epoch == EPOCH_B1950)
		epochConversion = epochConversionMatrix( NP_RA_B1950_IN_J2000, NP_DEC_B1950_IN_J2000, NP_RA_OFFSET_B1950_IN_J2000 );
	
	// return something.
	return epochConversion;

} // Reprojection::doEpochConversion

//
//	Reprojection::getEpoch()
//
//	CJS: 03/08/2015
//
//	Determine whether the epoch is J2000, B1950 or GALACTIC. Returns an enumerated type.
//

Reprojection::Epoch Reprojection::getEpoch( char * pEpoch )
{
	
	const char J2000[20] = "J2000";
	const char B1950[20] = "B1950";
	const char GALACTIC[20] = "GALACTIC";
	
	// default to J2000.
	Epoch thisEpoch = EPOCH_J2000;
	
	toUppercase( pEpoch );
	if ( strcmp( pEpoch, J2000 ) == 0 )
		thisEpoch = EPOCH_J2000;
	else if ( strcmp( pEpoch, B1950 ) == 0 )
		thisEpoch = EPOCH_B1950;
	else if ( strcmp( pEpoch, GALACTIC ) == 0 )
		thisEpoch = EPOCH_GALACTIC;
	
	// return something.
	return thisEpoch;

} // Reprojection:getEpoch

//
//	R E P R O J E C T I O N   A N D   R E G R I D D I N G   F U N C T I O N S
//

//
//	Reprojection::interpolateValue()
//
//	CJS:	08/07/15
//
//	use 'pPosition' to do bilinear interpolation between 4 data points.
//

__host__ __device__ double Reprojection::interpolateValue( rpVectD pPosition, double pBLValue, double pBRValue, double pTLValue, double pTRValue )
{
	
	// subtract the integer part of the position. we don't need this here.
	rpVectI integerPart = { .x = (int) floor( pPosition.x ), .y = (int) floor( pPosition.y ) };
	rpVectD fraction = { .x = pPosition.x - (double)integerPart.x, .y = pPosition.y - (double)integerPart.y };
		
	// interpolate top and bottom in the x-direction.
	double valueTop = ((pTRValue - pTLValue) * fraction.x) + pTLValue;
	double valueBottom = ((pBRValue - pBLValue) * fraction.x) + pBLValue;
		
	// interpolate in y-direction.
	return ((valueTop - valueBottom) * fraction.y) + valueBottom;

} // Reprojection::interpolateValue

//
//	Reprojection::reprojection()
//
//	CJS: 07/07/2015
//
//	Performs regridding and reprojection between the input and output images.
//	We need to handle the case where the output image pixels are much larger than the input image pixels (we need
//	to sum over many pixels), and also when the output image pixels are much smaller than the input image pixels
//	(we need to interpolate between input image pixels).
//
//	This routine works by comparing the size of the input and output image pixels, and choosing a number of
//	interpolation points for each output pixel. For example, overlaying the input and output images in world
//	coordinates may give:
//
//		+--------+--------+--------+--------+
//		|        |        |        |        |	+---+
//		|        |#   #   #   #   #|  #     |	|   |	= input image pixels
//		|        |        |                 |	+---+
//		+--------+#-------+--------+--#-----+
//		|        |     X====X====X |        |	+===+
//		|        |#    I  |      I |  #     |	I   I	= output image pixel
//		|        |     X  | X    X |        |	+===+
//		+--------+#----I--+------I-+--#-----+
//		|        |     X====X====X |        |	# # #	  region centred on the output image pixel, that extends on
//		|        |#       |        |  #     |	#   #	= all four sides to the surrounding output image pixels. this
//		|        |        |        |        |	# # #	  is the region we sum over.
//		+--------+#---#---#---#---#+--#-----+
//		|        |        |        |        |	  X	= interpolation point. the centre point has weight 1, the ones
//		|        |        |        |        |					along side it have weight 0.5, and the
//		|        |        |        |        |					ones on the diagonals have weight 0.25.
//		+--------+--------+--------+--------+
//
//	The program uses bilinear interpolation to calculate the value of the input grid at each interpolation point. These
//	values are then summed using a weighting that depends upon the position of the interpolation point relative to the output
//	pixel (the centre of the output pixel has weight 1, and this drops to 0 as we near the adjacent output pixels). If the
//	output pixel is small compared to the input pixels then we use a small number of interpolation points (one would do the
//	job, but we use a minimum of 3x3). If the output pixel is large compared to the input pixels then we use many
//	interpolation points (enough to ensure that at least one interpolation point is found within each fully-enclosed input
//	pixel).
//

void Reprojection::reprojection( double * pdevInImage, double * pdevOutImage, double * pdevNormalisationPattern, double * pdevPrimaryBeamPattern,
					bool * pdevInMask, double * pdevBeamIn, double * pdevBeamOut, ProjectionDirection pProjectionDirection,
					bool pAProjection )
{

	cudaError_t err;
	
	// need to build a grid that gives the pixel coordinates in the input image of each pixel in the output image.
	// the grid is only populated for every N pixels in the x and y directions, where N is the decimate parameter.
	// for pixels not found in this table we interpolate between those that are.

	// work out a suitable grid size.
	rpVectI mapSize;
	if (pProjectionDirection == OUTPUT_TO_INPUT)
	{
		mapSize.x = _outSize.x + 2;
		mapSize.y = _outSize.y + 2;
	}
	else
	{
		mapSize.x = _inSize.x + 2;
		mapSize.y = _inSize.y + 2;
	}

	// work out size of kernel call.
	dim3 gridSize2D( 1, 1 );
	dim3 blockSize2D( 1, 1 );

	cudaDeviceSynchronize();

	// catch any unreported errors.
	err = cudaGetLastError();
	if (err != cudaSuccess)
		printf( "captured unreported error (%s)\n", cudaGetErrorString( err ) );

	// generate map between output and input pixel coordinates.
	setThreadBlockSize2D( mapSize.x, mapSize.y, gridSize2D, blockSize2D );
	devBuildMap<<< gridSize2D, blockSize2D >>>(	/* pMap = */ _devProjectionMap,
							/* pMapValid = */ _devMapValid,
							/* pMapSize = */ mapSize,
							/* pOldCoordinateSystem = */ (pProjectionDirection == OUTPUT_TO_INPUT ? _outCoordinateSystem : _inCoordinateSystem),
							/* pNewCoordinateSystem = */ (pProjectionDirection == OUTPUT_TO_INPUT ? _inCoordinateSystem : _outCoordinateSystem) );
	cudaDeviceSynchronize();
	err = cudaGetLastError();
	if (err != cudaSuccess)
		printf( "error building output/input pixel map (%s)\n", cudaGetErrorString( err ) );

	// we either reproject from the output image to the input image, or from the input image to the output image.
	if (pProjectionDirection == OUTPUT_TO_INPUT)
	{

		// update the output image by reading values from the input image.
		setThreadBlockSize2D( _outSize.x, _outSize.y, gridSize2D, blockSize2D );
		devReprojectionOutToIn<<< gridSize2D, blockSize2D >>>( pdevInImage, pdevOutImage, _devProjectionMap, _devMapValid, pdevNormalisationPattern, pdevPrimaryBeamPattern,
									mapSize, _inSize, _outSize, pdevInMask, pdevBeamIn, pdevBeamOut, pAProjection );

	}
	else
	{

		// update the output image by reading values from the input image.
		setThreadBlockSize2D( _inSize.x, _inSize.y, gridSize2D, blockSize2D );
		devReprojectionInToOut<<< gridSize2D, blockSize2D >>>( pdevInImage, pdevOutImage, _devProjectionMap, _devMapValid, pdevNormalisationPattern, pdevPrimaryBeamPattern,
									mapSize, _inSize, _outSize, pdevInMask, pdevBeamIn, pdevBeamOut, pAProjection );

	}

	cudaDeviceSynchronize();
	err = cudaGetLastError();
	if (err != cudaSuccess)
		printf( "error creating reprojected image on the device (%s)\n", cudaGetErrorString( err ) );

} // Reprojection::reprojection

void Reprojection::reprojection( float * pdevInImage, float * pdevOutImage, double * pdevNormalisationPattern, double * pdevPrimaryBeamPattern,
					bool * pdevInMask, double * pdevBeamIn, double * pdevBeamOut, ProjectionDirection pProjectionDirection,
					bool pAProjection )
{

	cudaError_t err;
	
	// need to build a grid that gives the pixel coordinates in the input image of each pixel in the output image.
	// the grid is only populated for every N pixels in the x and y directions, where N is the decimate parameter.
	// for pixels not found in this table we interpolate between those that are.

	// work out a suitable grid size.
	rpVectI mapSize;
	if (pProjectionDirection == OUTPUT_TO_INPUT)
	{
		mapSize.x = _outSize.x + 2;
		mapSize.y = _outSize.y + 2;
	}
	else
	{
		mapSize.x = _inSize.x + 2;
		mapSize.y = _inSize.y + 2;
	}

	// work out size of kernel call.
	dim3 gridSize2D( 1, 1 );
	dim3 blockSize2D( 1, 1 );

	cudaDeviceSynchronize();

	// catch any unreported errors.
	err = cudaGetLastError();
	if (err != cudaSuccess)
		printf( "captured unreported error (%s)\n", cudaGetErrorString( err ) );

	// generate map between output and input pixel coordinates.
	setThreadBlockSize2D( mapSize.x, mapSize.y, gridSize2D, blockSize2D );
	devBuildMap<<< gridSize2D, blockSize2D >>>(	/* pMap = */ _devProjectionMap,
							/* pMapValid = */ _devMapValid,
							/* pMapSize = */ mapSize,
							/* pOldCoordinateSystem = */ (pProjectionDirection == OUTPUT_TO_INPUT ? _outCoordinateSystem : _inCoordinateSystem),
							/* pNewCoordinateSystem = */ (pProjectionDirection == OUTPUT_TO_INPUT ? _inCoordinateSystem : _outCoordinateSystem) );
	cudaDeviceSynchronize();
	err = cudaGetLastError();
	if (err != cudaSuccess)
		printf( "error building output/input pixel map (%s)\n", cudaGetErrorString( err ) );

	// we either reproject from the output image to the input image, or from the input image to the output image.
	if (pProjectionDirection == OUTPUT_TO_INPUT)
	{

		// update the output image by reading values from the input image.
		setThreadBlockSize2D( _outSize.x, _outSize.y, gridSize2D, blockSize2D );
		devReprojectionOutToIn<<< gridSize2D, blockSize2D >>>( pdevInImage, pdevOutImage, _devProjectionMap, _devMapValid, pdevNormalisationPattern,
									pdevPrimaryBeamPattern, mapSize, _inSize, _outSize, pdevInMask, pdevBeamIn, pdevBeamOut,
									pAProjection );

	}
	else
	{

		// update the output image by reading values from the input image.
		setThreadBlockSize2D( _inSize.x, _inSize.y, gridSize2D, blockSize2D );
		devReprojectionInToOut<<< gridSize2D, blockSize2D >>>( pdevInImage, pdevOutImage, _devProjectionMap, _devMapValid, pdevNormalisationPattern,
									pdevPrimaryBeamPattern, mapSize, _inSize, _outSize, pdevInMask, pdevBeamIn, pdevBeamOut,
									pAProjection );

	}

	cudaDeviceSynchronize();
	err = cudaGetLastError();
	if (err != cudaSuccess)
		printf( "error creating reprojected image on the device (%s)\n", cudaGetErrorString( err ) );

} // Reprojection::reprojection
