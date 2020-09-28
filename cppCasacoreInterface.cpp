// include the header file.
#include "cppCasacoreInterface.h"

// include casacore libraries.
#include <casacore/images/Images/ImageInterface.h>
#include <casacore/images/Images/ImageUtilities.h>
#include <casacore/lattices/Lattices/TiledShape.h>
#include <casacore/coordinates/Coordinates/CoordinateSystem.h>
#include <casacore/casa/Arrays/ArrayIO.h>
#include <casacore/casa/Arrays/IPosition.h>
#include <casacore/casa/Arrays/Cube.h>
#include <casacore/coordinates/Coordinates/Coordinate.h>
#include <casacore/coordinates/Coordinates/DirectionCoordinate.h>
#include <casacore/coordinates/Coordinates/SpectralCoordinate.h>
#include <casacore/coordinates/Coordinates/Projection.h>
#include <casacore/measures/Measures/MFrequency.h>
#include <casacore/tables/TaQL/TableParse.h>
#include <casacore/tables/Tables/RefRows.h>
#include <casacore/tables/Tables/ScalarColumn.h>

#include <time.h>

using namespace std;
using namespace casacore;

CasacoreInterface CasacoreInterface::_thisInstance;

//
//	P R I V A T E   C O N S T A N T S
//

const double CasacoreInterface::CONST_C = 299792458.0;
const double CasacoreInterface::PI = 3.141592654;

//
//	P U B L I C   C L A S S   M E M B E R S
//

//
//	CasacoreInterface::CasacoreInterface()
//
//	CJS: 08/03/2017
//
//	The constructor.
//

CasacoreInterface::CasacoreInterface()
{
	
	// create a new image utilities object.
	//_imageUtilities = new ImageUtilities();
	
} // CasacoreInterface::CasacoreInterface

//
//	CasacoreInterface::~CasacoreInterface()
//
//	CJS: 08/03/2017
//
//	The destructor.
//

CasacoreInterface::~CasacoreInterface()
{
	
	// delete the image utilities object.
	//delete _imageUtilities;
	
} // CasacoreInterface::~CasacoreInterface

//
//	CasacoreInterface::getInstance()
//
//	CJS: 20/02/2018
//
//	Get the instance of the class so that only one instance will exist.
//

CasacoreInterface * CasacoreInterface::getInstance()
{

	// return a pointer to this object.
	return &_thisInstance;

} // CasacoreInterface::getInstance

//
//	CasacoreInterface::LoadBitmap()
//
//	CJS: 08/03/2017
//
//	public interface to load bitmap files, with various overloads.
//

bool CasacoreInterface::LoadBitmap( const char * pFilename, complex<double> ** pImageData )
{
	
	int width = 0, height = 0;
	double imageScale = 0;
	return loadBitmap( pFilename, pImageData, &width, &height, &imageScale );
	
} // CasacoreInterface::LoadBitmap

bool CasacoreInterface::LoadBitmap( const char * pFilename, complex<double> ** pImageData, int * pWidth,
					int * pHeight )
{
	
	double imageScale = 0;
	return loadBitmap( pFilename, pImageData, pWidth, pHeight, &imageScale );
	
} // CasacoreInterface::LoadBitmap

bool CasacoreInterface::LoadBitmap( const char * pFilename, complex<double> ** pImageData, double * pImageScale )
{
	
	int width = 0, height = 0;
	return loadBitmap( pFilename, pImageData, &width, &height, pImageScale );
	
} // CasacoreInterface::LoadBitmap

bool CasacoreInterface::LoadBitmap( const char * pFilename, complex<double> ** pImageData, int * pWidth, int * pHeight,
					double * pImageScale )
{
	
	return loadBitmap( pFilename, pImageData, pWidth, pHeight, pImageScale );
	
} // CasacoreInterface::LoadBitmap

//
//	CasacoreInterface::SaveBitmap()
//
//	CJS: 13/03/2017
//
//	public interface to save bitmap files, with various overloads.
//

bool CasacoreInterface::SaveBitmap( const char * pFilename, complex<double> * pImageData, int pWidth, int pHeight )
{
	
	double imageScale = 0;
	return saveBitmap( pFilename, pImageData, pWidth, pHeight, &imageScale );
	
} // CasacoreInterface::SaveBitmap

bool CasacoreInterface::SaveBitmap( const char * pFilename, complex<double> * pImageData, int pWidth, int pHeight,
					double * pImageScale )
{
	
	return saveBitmap( pFilename, pImageData, pWidth, pHeight, pImageScale );
	
} // CasacoreInterface::SaveBitmap

//
//	CasacoreInterface::WriteCasaImage()
//
//	CJS: 08/03/2017
//
//	public interface to writing casa images.
//

void CasacoreInterface::WriteCasaImage( const char * pFilename, int pWidth, int pHeight, double pRA, double pDec,
						double pPixelSize, complex<double> * pImage, double pFrequency, bool * pMask )
{
	
	writeCasaImage( pFilename, pWidth, pHeight, pRA, pDec, pPixelSize, (double *) pImage, pFrequency, pMask, COMPLEX );
				
} // CasacoreInterface::WriteCasaImage

void CasacoreInterface::WriteCasaImage( const char * pFilename, int pWidth, int pHeight, double pRA, double pDec,
						double pPixelSize, double * pImage, double pFrequency, bool * pMask )
{
	
	writeCasaImage( pFilename, pWidth, pHeight, pRA, pDec, pPixelSize, pImage, pFrequency, pMask, DOUBLE );
				
} // CasacoreInterface::WriteCasaImage

void CasacoreInterface::WriteCasaImage( const char * pFilename, int pWidth, int pHeight, double pRA, double pDec,
						double pPixelSize, float * pImage, double pFrequency, bool * pMask )
{
	
	writeCasaImage( pFilename, pWidth, pHeight, pRA, pDec, pPixelSize, pImage, pFrequency, pMask, FLOAT );
				
} // CasacoreInterface::WriteCasaImage

void CasacoreInterface::WriteCasaImage( const char * pFilename, int pWidth, int pHeight, double pRA, double pDec,
						double pPixelSize, complex<float> * pImage, double pFrequency, bool * pMask )
{
	
	writeCasaImage( pFilename, pWidth, pHeight, pRA, pDec, pPixelSize, pImage, pFrequency, pMask, COMPLEX_FLOAT );
				
} // CasacoreInterface::WriteCasaImage

//
//	CasacoreInterface::GetAntennae()
//
//	CJS: 16/04/2020
//
//	Get a list of antennae.
//

int CasacoreInterface::GetAntennae( const char * pMeasurementSet, double ** pDishDiameter, bool ** pFlagged )
{

	return getAntennae( pMeasurementSet, pDishDiameter, pFlagged );

} // CasacoreInterface::GetAntennae

//
//	CasacoreInterface::NumberOfAntennae()
//
//	CJS: 21/03/2017
//
//	public interface for counting the number of antennae.
//

int CasacoreInterface::NumberOfAntennae( const char * pFilename )
{
	
	return numberOfAntennae( pFilename );
				
} // CasacoreInterface::NumberOfAntennae
						
//
//	CasacoreInterface::GetWavelengths()
//
//	CJS: 10/03/2017
//
//	get a list of channel wavelengths from the measurement set.
//

bool CasacoreInterface::GetWavelengths( const char * pMeasurementSet, int * pNumSpws, int ** pNumChannels, double *** pWavelength )
{
	
	return getWavelengths( pMeasurementSet, pNumSpws, pNumChannels, pWavelength );
	
} // CasacoreInterface::GetWavelengths
						
//
//	CasacoreInterface::GetDataDesc()
//
//	CJS: 27/04/2020
//
//	get all the data from the data description table.
//

void CasacoreInterface::GetDataDesc( const char * pMeasurementSet, int * pNumDataDescItems, int ** pPolarisationConfig, int ** pSpw )
{

	getDataDesc( pMeasurementSet, pNumDataDescItems, pPolarisationConfig, pSpw );

} // CasacoreInterface::GetDataDesc

//
//	CasacoreInterface::GetPolarisations()
//
//	CJS: 18/02/2020
//
//	get a list of polarisations from the measurement set.
//

bool CasacoreInterface::GetPolarisations( const char * pMeasurementSet, int * pNumPolarisations, int * pNumPolarisationConfigurations, int ** pPolarisation )
{

	return getPolarisations( pMeasurementSet, pNumPolarisations, pNumPolarisationConfigurations, pPolarisation );

} // CasacoreInterface::GetPolarisations
						
//
//	CasacoreInterface::GetSamples()
//
//	CJS: 10/03/2017
//
//	get a list of samples from the measurement set.
//

bool CasacoreInterface::GetSamples( const char * pMeasurementSet, int * pNumSamples, double ** pSample,  char * pFieldID, bool * pDataDescFlag, int pDataDescItems,
					int ** pFieldIDArray, int ** pDataDescID, int ** pAntenna1, int ** pAntenna2 )
{
	
	return getSamples( pMeasurementSet, pNumSamples, pSample, pFieldID, pDataDescFlag, pDataDescItems, pFieldIDArray, pDataDescID, pAntenna1, pAntenna2 );
	
} // CasacoreInterface::GetSamples

//
//	CasacoreInterface::GetVisibilities()
//
//	CJS: 25/07/2016
//
//	Load the visibilities from the measurement set (we should have one visibility for each sample/channel combination).
//

void CasacoreInterface::GetVisibilities( const char * pFilename, char * pFieldID, int * pNumSamples, int * pNumChannels, char * pDataField,
						complex<float> ** pUniVisibility, bool ** pFlag, double ** pSample, float ** pWeight,
						int ** pFieldIDArray, int ** pDataDescID, int pNumPolarisations,
						int pStartAnt1, int pStartAnt2, int pEndAnt1, int pEndAnt2, int pNumberOfAntennae, double pCurrentSample,
						double pTotalSamples, int pNumSpws, int * pDataDescSPW, bool * pDataDescFlag, int pNumDataDesc,
						bool ** pSpwChannelFlag )
{
	
	getVisibilities( pFilename, pFieldID, pNumSamples, pNumChannels, pDataField, pUniVisibility, pFlag, pSample, pWeight,
				pFieldIDArray, pDataDescID, pNumPolarisations, pStartAnt1, pStartAnt2, pEndAnt1, pEndAnt2, pNumberOfAntennae, pCurrentSample, pTotalSamples,
				pNumSpws, pDataDescSPW, pDataDescFlag, pNumDataDesc, pSpwChannelFlag );
	
} // CasacoreInterface::GetVisibilities

//
//	CasacoreInterface::GetPhaseCentres()
//
//	CJS: 26/09/2018
//
//	Get a list of phase centres for each field.
//

bool CasacoreInterface::GetPhaseCentres( const char * pMeasurementSet, int * pNumFields, double ** pPhaseCentre )
{

	return getPhaseCentres( pMeasurementSet, pNumFields, pPhaseCentre );

} // CasacoreInterface::GetPhaseCentres

//
//	P R I V A T E   C L A S S   M E M B E R S
//

//
//	CasacoreInterface::loadBitmap()
//
//	CJS: 07/07/2015
//
//	load a bitmap file, and return the image size and a boolean indicating success.
//	the image must be 8-bit greyscale.
//

bool CasacoreInterface::loadBitmap( const char * pFilename, complex<double> ** pImageData, int * pWidth, int * pHeight,
					double * pImageScale )
{
	
	bool ok = true;
	unsigned char * fileInfo = NULL, * fileHeader = NULL, * tmpData = NULL;
	
	// open the bitmap file.
	FILE * inputFile = fopen( pFilename, "r" );
	if (inputFile == NULL)
	{
		printf("Could not open file \"%s\".\n", pFilename);
		ok = false;
	}
	else
	{
		
		// reserve memory for the start of the file header, and read it from the file. we only
		// read the first 18 bytes, because these contain information about how large the header is. once we
		// know this we can read the rest of the header.
		fileInfo = (unsigned char *) malloc( 18 );
		size_t num_read = fread( fileInfo, sizeof( unsigned char ), 18, inputFile );
				
		// ensure we've read the correct number of bytes.
		if (num_read != 18)
		{
			printf( "Error: read only %lu values from the file header.\n", num_read );
			ok = false;
		}

		// make sure this is a bitmap file by checking that the first two bytes are ASCII codes 'B' and 'M'.
		if (ok == true)
			if ((fileInfo[BIT_CONST] != 'B') || (fileInfo[MAP_CONST] != 'M'))
			{
				printf( "Error: this is not a bitmap file.\n" );
				ok = false;
			}
			
		// get the size of the file header (i.e. a pointer to the start of the actual image).
		int fileHeaderSize = 0;
		if (ok == true)
			memcpy( &fileHeaderSize, &fileInfo[FILE_HEADER_SIZE], 4 );
			
		// get the size of the bitmap info header (the bitmap info header is followed by the colour table,
		// so we need to know the offset in order to read the colours).
		int bitmapInfoHeaderSize = 0;
		if (ok == true)
			memcpy( &bitmapInfoHeaderSize, &fileInfo[BITMAP_INFO_HEADER], 4 );
		
		// need to add 14 because the bitmap info header size does not include the first 14 bytes of the file (which
		// technically are part of the file header but not the bitmap header; we lump everything in together so that
		// all of our offsets are from the start of the file - less confusing this way).
		bitmapInfoHeaderSize = bitmapInfoHeaderSize + 14;
			
		// get the rest of the file header now we know how big it is. we already have the first 18 bytes,
		// which should be copied to the start of the new memory area.
		if (ok == true)
		{
			fileHeader = (unsigned char *) malloc( fileHeaderSize );
			memcpy( fileHeader, fileInfo, 18 );
			num_read = fread( &fileHeader[18], sizeof( unsigned char ), fileHeaderSize - 18, inputFile );
			if (num_read != (fileHeaderSize - 18))
			{
				printf( "Error: read only %lu values from the file header.\n", num_read + 18 );
				ok = false;
			}
		}
		
		// get the input image flux scale. this value may be stored in the reserved part of the bitmap file header
		// (0x06 -> 0x09), and will not be supplied if the input image has been saved using something like GIMP or
		// Photoshop. if it is zero, then we assume a scale of 1 Jy/PIXEL. this value gets re-scaled along with our
		// image, and is then written back to the output file.
		if (ok == true)
			memcpy( pImageScale, &fileHeader[RESERVED], 4 );
		
		if (*pImageScale == 0)
			*pImageScale = 1;
			
		// ensure we have an 8-bit image.
		if (ok == true)
		{
			short bitCount;
			memcpy( &bitCount, &fileHeader[BIT_COUNT], 2 );
			if (bitCount != 8)
			{
				printf( "Error: expecting an 8-bit greyscale image. This one is %hi bit.\n", bitCount );
				ok = false;
			}
		}
			
		// ensure the image in not compressed.
		if (ok == true)
		{
			int compressionMethod;
			memcpy( &compressionMethod, &fileHeader[COMPRESSION_TYPE], 4 );
			if (compressionMethod != 0)
			{
				printf( "Error: can only handle uncompressed bitmaps." );
				ok = false;
			}
		}
			
		if (ok == true)
		{
			
			// get the width and height of the image.
			memcpy( pWidth, &fileHeader[IMAGE_WIDTH], 4 );
			memcpy( pHeight, &fileHeader[IMAGE_HEIGHT], 4 );
		
			// ensure width and height are greater than zero.
			if (*pWidth <= 0 || *pHeight <= 0)
			{
				printf( "Error: invalid image size (%i x %i).\n", *pWidth, *pHeight );
				ok = false;
			}
			
		}
		
		if (ok == true)
		{
			
			// ensure the number of colours used is 256.
			int coloursUsed = 0;
			memcpy( &coloursUsed, &fileHeader[COLOURS_USED], 4 );
			if (coloursUsed != 256)
			{
				printf( "ERROR: Can only handle 256 colours in pallette.\n" );
				ok = false;
			}
			
		}
		
		// get the number of significant colours used. this value can (theoretically) be less than COLOURS_USED
		// if an image is only using (e.g.) 37 shades rather than all 256. in practice, this is never implemented, and
		// this value will either be 0 (= all colours) or will match COLOURS_USED. however, only SIGNIFICANT_COLOURS are
		// written to the pallette, so we have to handle this parameter just in case.
		int significantColours = 0;
		if (ok == true)
			memcpy( &significantColours, &fileHeader[SIGNIFICANT_COLOURS], 4 );
		
		// if significant colours = 0, then they are ALL significant so set to 256.
		if (significantColours == 0)
			significantColours = 256;
			
		unsigned int colour[256];
		if (ok == true)
		{
				
			// load colour table from bmp.
			for ( unsigned int i = 0; i < significantColours; ++i )
			{
				
				memcpy( &colour[i], &fileHeader[bitmapInfoHeaderSize + (i * 4)], 4 );
				
				// convert pallette colour to greyscale, using 0.2990, 0.5870, 0.1140 RGB weighting. add 0.5
				// to round to nearest integer (since C only rounds down).
				unsigned char red = colour[i] >> 16;
				unsigned char green = (colour[i] >> 8) - (red << 8);
				unsigned char blue = colour[i] - (red << 16) - (green << 8);
				colour[i] = (unsigned int) ((((double)red * 0.2990) + ((double)green * 0.5870) +
								((double)blue * 0.1140)) + 0.5);
				
			}
				
			// reserve some memory for the image, and read it from the file.
			tmpData = (unsigned char *) malloc( *pWidth * *pHeight );
			*pImageData = (complex<double> *) malloc( *pWidth * *pHeight * sizeof( complex<double> ) );
			num_read = fread( tmpData, sizeof( unsigned char ), *pWidth * *pHeight, inputFile );
				
			// ensure we've read the correct number of bytes.
			if (num_read != *pWidth * *pHeight)
			{
				printf( "Error: read only %lu values from the image.\n", num_read );
				ok = false;
			}
				
		}
			
		if (ok == true)
		{
				
			// update image values using the values from the colour table.
			complex<double> * complexData = *pImageData;
			for ( int i = 0; i < *pWidth * *pHeight; i++ )
				complexData[i] = complex<double>( (double)colour[tmpData[i]] * *pImageScale, 0 );
				
		}
		
		// close file.
		fclose( inputFile );
	
	}
	
	// tidy up memory.
	if (fileInfo != NULL)
		free( (void *) fileInfo );
	if (fileHeader != NULL)
		free( (void *) fileHeader );
	if (tmpData != NULL)
		free( (void *) tmpData );
	
	// return success flag.
	return ok;
	
} // CasacoreInterface::loadBitmap

//
//	CasacoreInterface::saveBitmap()
//
//	CJS: 10/08/2015
//
//	Save a bitmap file.
//

bool CasacoreInterface::saveBitmap( const char * pFilename, complex<double> * pImageData, int pWidth, int pHeight,
					double * pImageScale )
{
	
	unsigned char * image = NULL;
	
	const int HEADER_SIZE = 1078;
	
	// allocate and build the header.
	unsigned char * fileHeader = (unsigned char *) malloc( HEADER_SIZE );
	memset( fileHeader, 0, HEADER_SIZE );

	// file header.
	fileHeader[BIT_CONST] = 'B'; fileHeader[MAP_CONST] = 'M';					// bfType
	int size = (pWidth * pHeight) + HEADER_SIZE; memcpy( &fileHeader[IMAGE_SIZE], &size, 4 );	// bfSize
	int offBits = HEADER_SIZE; memcpy( &fileHeader[FILE_HEADER_SIZE], &offBits, 4 );		// bfOffBits

	// image header.
	size = 40; memcpy( &fileHeader[BITMAP_INFO_HEADER], &size, 4 );					// biSize
	memcpy( &fileHeader[IMAGE_WIDTH], &pWidth, 4 );							// biWidth
	memcpy( &fileHeader[IMAGE_HEIGHT], &pHeight, 4 );						// biHeight
	short planes = 1; memcpy( &fileHeader[COLOUR_PLANES], &planes, 2 );				// biPlanes
	short bitCount = 8; memcpy( &fileHeader[BIT_COUNT], &bitCount, 2 );				// biBitCount
	int coloursUsed = 256; memcpy( &fileHeader[COLOURS_USED], &coloursUsed, 4 );			// biClrUsed

	// colour table.
	for (unsigned int i = 0; i < 256; ++i)
	{
		unsigned int colour = (i << 16) + (i << 8) + i;
		memcpy( &fileHeader[54 + (i * 4)], &colour, 4 );
	}
	
	bool ok = true;

	// open file.
	FILE * outputFile = fopen( pFilename, "w" );
	if (outputFile == NULL)
	{
		printf( "Could not open file \"%s\".\n", pFilename );
		ok = false;
	}
	else
	{

		// write the file header.
		size_t num_written = fwrite( fileHeader, 1, 1078, outputFile );
		if (num_written != 1078)
		{
			printf( "Error: cannot write to file.\n" );
			ok = false;
		}
		
		// find the maximum and minimum pixel values.
		double min = real( pImageData[ 0 ] );
		double max = real( pImageData[ 0 ] );
		for ( int i = 1; i < pWidth * pHeight; i++ )
		{
			if (real( pImageData[ i ] ) < min)
				min = real( pImageData[ i ] );
			if (real( pImageData[ i ] ) > max)
				max = real( pImageData[ i ] );
		}
		
		printf("min: %f, max: %f\n", min, max );

		// add 1% allowance to max - we don't want saturation.
		max = ((max - min) * 1.1) + min;

min = 0; max = 1;
		
		// construct the image.
		image = (unsigned char *) malloc( pWidth * pHeight * sizeof( unsigned char ) );
		for ( int i = 0; i < pWidth * pHeight; i++ )
			image[i] = (unsigned char)( (real( pImageData[ i ] ) - min) * ((double)256 / (max - min)) );
		
		// write the data.
		if (ok == true)
		{
			
			size_t num_written = fwrite( image, 1, pWidth * pHeight, outputFile );
			if (num_written != (pWidth * pHeight))
			{
				printf( "Error: cannot write to file.\n" );
				ok = false;
			}
			
		}

		// close file.
		fclose( outputFile );
		
	}

	// cleanup memory.
	free( (void *) fileHeader );
	if (image != NULL)
		free( image );
	
	// return success flag.
	return ok;
	
} // CasacoreInterface::saveBitmap

void CasacoreInterface::ReadCasaImage( const char * pFilename )
{

	ImageUtilities * imageUtilities = new ImageUtilities();

	ImageInterface<Float> * imageInterface;
	imageUtilities->openImage(	imageInterface,
					pFilename );

	// create a new coordinate system.
	CoordinateSystem inCoordinateSystem = imageInterface->coordinates();

	printf( "nCoordinates = %i\n", inCoordinateSystem.nCoordinates() );
	printf( "nPixelAxes = %i\n", inCoordinateSystem.nPixelAxes() );
	printf( "nWorldAxes = %i\n", inCoordinateSystem.nWorldAxes() );
	printf( "imageInterface.ndim() = %i\n", imageInterface->ndim() );

	int rowLength = imageInterface->shape()(0);
	IPosition rowShape( imageInterface->ndim() );

	printf( "shape.size() = %i\n", rowShape.size() );

	DirectionCoordinate inDirectionCoordinate = inCoordinateSystem.directionCoordinate();
	SpectralCoordinate inSpectralCoordinate = inCoordinateSystem.spectralCoordinate();
	StokesCoordinate inStokesCoordinate = inCoordinateSystem.stokesCoordinate();
	
} // CasacoreInterface::ReadCasaImage

//
//	CasacoreInterface::writeCasaImage()
//
//	CJS: 08/03/2017
//
//	write an image to a casa image.
//
		
void CasacoreInterface::writeCasaImage( const char * pFilename, int pWidth, int pHeight, double pRA,
						double pDec, double pPixelSize, void * pImage,
						double pFrequency, bool * pMask, ffttype pFFTType )
{

	// image size.
	IPosition imageSize = IPosition( 4, pWidth, pHeight, 1, 1 );

	// create a tiled shape.
	TiledShape outShape( imageSize );

	// the transformation matrix is diagonal so that the image is aligned with ra and dec (casa doesn't
	// like rotated coordinate systems).
	casacore::Matrix<Double> dirTransform( 2, 2, 2 );
	dirTransform( 0, 0 ) = 1;
	dirTransform( 0, 1 ) = 0;
	dirTransform( 1, 0 ) = 0;
	dirTransform( 1, 1 ) = 1;

	// create a new direction coordinate.
	DirectionCoordinate outDirectionCoordinate(	MDirection::J2000, 
							Projection( Projection::SIN ),
							pRA * PI / (double) 180.0, pDec * PI / (double) 180.0,
							-pPixelSize * PI / ((double) 180.0 * (double) 3600.0),
							pPixelSize * PI / ((double) 180.0 * (double) 3600.0),
							dirTransform,
							pWidth / 2, pHeight / 2 );

	// to change the axis units to degrees:
	casacore::Vector<String> outUnits( 2 ); outUnits = "deg";
	outDirectionCoordinate.setWorldAxisUnits( outUnits );

	casacore::Vector<int> whichStokes( 4 );
	whichStokes( 0 ) = Stokes::I; whichStokes( 1 ) = Stokes::Q; whichStokes( 2 ) = Stokes::U; whichStokes( 3 ) = Stokes::V;
	StokesCoordinate outStokesCoordinate( whichStokes );

	// create a new spectral coordinate.
	SpectralCoordinate outSpectralCoordinate( /* type = */ MFrequency::REST, /* f0 = */ pFrequency, /* inc = */ 1, /* refPix = */ 0, /* restFrequency = */ 0.0 );

	// create a new coordinate system.
	CoordinateSystem outCoordinateSystem;

	// add the direction and spectral coordinates.
	outCoordinateSystem.addCoordinate( outDirectionCoordinate );
	outCoordinateSystem.addCoordinate( outStokesCoordinate );
	outCoordinateSystem.addCoordinate( outSpectralCoordinate );

	//printf( "Num Pixel Axes: %i\n", outCoordinateSystem.nPixelAxes() );
	int coordinate = -1; int& coordinateRef = coordinate;
	int axis = -1; int& axisRef = axis;
	outCoordinateSystem.findPixelAxis( coordinateRef, axisRef, 0 );
	outCoordinateSystem.findPixelAxis( coordinateRef, axisRef, 1 );
	outCoordinateSystem.findPixelAxis( coordinateRef, axisRef, 2 );

	// create memory for the image pixels and the mask. we pass in the float pointer, and use SHARE so that CASA will use our data, and not make a copy of it (COPY),
	// or release the data when it's finished with it (TAKE_OVER).
	Array<Float> outPixels( imageSize, (float *) pImage, SHARE );
	Array<Bool> outMask( imageSize );

	// copy the image into the arrays.
	long int ptr = 0;
	for ( int j = 0; j < pHeight; j++ )
		for ( int i = 0; i < pWidth; i++, ptr++ )
		{
			
			IPosition arrayPos( 4, i, j, 0, 0 );
//			if (pFFTType == COMPLEX)
//				outPixels( arrayPos ) = (float) real( ((complex<double> *) pImage)[ ptr ] );
//			else if (pFFTType == DOUBLE)
//				outPixels( arrayPos ) = (float) (((double *) pImage)[ ptr ]);
//			else if (pFFTType == FLOAT)
//				outPixels( arrayPos ) = (float) (((float *) pImage)[ ptr ]);
//			else if (pFFTType == COMPLEX_FLOAT)
//				outPixels( arrayPos ) = (float) real( ((complex<float> *) pImage)[ ptr ] );
			if (pMask == NULL)
				outMask( arrayPos ) = true;
			else
				outMask( arrayPos ) = (pMask[ ptr ]);
			
		}

	// create an IO object?
	LogIO outIO;

	ImageUtilities * imageUtilities = new ImageUtilities();

	// write the image to a casa image directory (AIPSPP).
	imageUtilities->writeImage(	outShape,
					outCoordinateSystem,
					pFilename,
					outPixels,
					outIO,
					outMask );

	delete imageUtilities;
						
} // CasacoreInterface::writeCasaImage

//
//	CasacoreInterface::getAntennae()
//
//	CJS: 16/04/2020
//
//	Get all the antennae from the file, including the flagged ones.
//

int CasacoreInterface::getAntennae( const char * pMeasurementSet, double ** pDishDiameter, bool ** pFlagged )
{

	char sqlCommand[ 1000 ];
	
	// construct the SQL command.
	sprintf( sqlCommand,	"SELECT		FLAG_ROW, DISH_DIAMETER "
				"FROM		%s/ANTENNA", pMeasurementSet );

	// issue SQL command to get the antennae.
	Table tblAntennae = tableCommand( sqlCommand );

	int count = tblAntennae.nrow();

	ScalarColumn<Bool> colFlags( tblAntennae, "FLAG_ROW" );
	ScalarColumn<Double> colDishDiameter( tblAntennae, "DISH_DIAMETER" );
		
	if (count > 0)
	{

		IPosition shape( 1, count );

		// reserve memory for flags, and fetch them.
		(*pFlagged) = (bool *) malloc( sizeof( bool ) * count );
		Vector<Bool> arrayFlags( shape, *pFlagged, SHARE );
		colFlags.getColumn( arrayFlags, false );

		// reserve memory for dish diameters, and fetch them.
		(*pDishDiameter) = (double *) malloc( sizeof( double ) * count );
		Vector<Double> arrayDishDiameter( shape, *pDishDiameter, SHARE );
		colDishDiameter.getColumn( arrayDishDiameter, false );
		
	}
	
	// return the number of antennae.
	return count;

} // CasacoreInterface::getAntennae

//
//	CasacoreInterface::numberOfAntennae()
//
//	CJS: 21/03/2017
//
//	count the number of antennae.
//

int CasacoreInterface::numberOfAntennae( const char * pMeasurementSet )
{
	
	int count = 0;
	char sqlCommand[ 1000 ];
	
	// construct the SQL command.
	sprintf( sqlCommand,	"SELECT		DISH_DIAMETER "
				"FROM		%s/ANTENNA", pMeasurementSet );
	
	// issue SQL command to get the antennae.
	Table tblAntennae = tableCommand( sqlCommand );
	ScalarColumn<double> dishDiameter( tblAntennae, "DISH_DIAMETER" );
	Vector<double> dishDiameterVector;
		
	// get the whole column, and the shape of this column.
	dishDiameter.getColumn( dishDiameterVector, true );
	IPosition dishDiameterShape = dishDiameterVector.shape();
	
	// ensure that we have a 1-D, and at least one row.
	if (dishDiameterShape.nelements() == 1)
		count = (int)dishDiameterShape[ 0, 0 ];
	
	// return the number of antennae.
	return count;
				
} // CasacoreInterface::numberOfAntennae
						
//
//	CasacoreInterface::getWavelengths()
//
//	CJS: 10/03/2017
//
//	get a list of channel wavelengths from the measurement set.
//

bool CasacoreInterface::getWavelengths( const char * pMeasurementSet, int * pNumSpws, int ** pNumChannels, double *** pWavelength )
{
	
	char sqlCommand[ 1000 ];
	
	// construct the SQL command.
	sprintf( sqlCommand,	"SELECT		CHAN_FREQ, NUM_CHAN "
				"FROM		%s/SPECTRAL_WINDOW ", pMeasurementSet );
	
	// issue SQL command to get the channel frequencies.
	Table tblChannels = tableCommand( sqlCommand );

	// count the spws.
	*pNumSpws = tblChannels.nrow();

	// get the number of channels per SPW.
	if (*pNumSpws > 0)
	{

		// create memory for the number of channels, and retrieve the data.
		*pNumChannels = (int *) malloc( *pNumSpws * sizeof( int ) );
		
		ScalarColumn<int> numChan( tblChannels, "NUM_CHAN" );
		IPosition shape( 1, *pNumSpws );
		Vector<int> vectorNumChan( shape, *pNumChannels, SHARE );
		numChan.getColumn( vectorNumChan, false );

	}

	if (*pNumSpws > 0)
	{
	
		ArrayColumn<double> chanFreq( tblChannels, "CHAN_FREQ" );

		// reserve memory for wavelengths.
		(*pWavelength) = (double **) malloc( (*pNumSpws) * sizeof( double * ) );
		for ( int spw = 0; spw < *pNumSpws; spw++ )
		{

			// get the whole column, and the shape of this column.
			Array<double> chanFreqArray;
			chanFreq.get( spw, chanFreqArray, false );

			(*pWavelength)[ spw ] = (double *) malloc( (*pNumChannels)[ spw ] * sizeof( double ) );

			// loop through the channel array, getting the frequencies.
			for ( int i = 0; i < (*pNumChannels)[ spw ]; i++ )
			{

				// create a pointer to the frequency cell, and get this frequency.
				IPosition freqCell( 1, i );
				(*pWavelength)[ spw ][ i ] = (CONST_C / (double) chanFreqArray( freqCell ));
		
			}

		}
	
	}

	// return success or fail.
	return (*pNumSpws > 0);
	
} // CasacoreInterface::getWavelengths
						
//
//	CasacoreInterface::getDataDesc()
//
//	CJS: 27/04/2020
//
//	get all the data from the data description table.
//

void CasacoreInterface::getDataDesc( const char * pMeasurementSet, int * pNumDataDescItems, int ** pPolarisationConfig, int ** pSpw )
{

	// initialise count.
	*pNumDataDescItems = 0;
	
	//float * sample = NULL;
	char sqlCommand[ 1000 ];
	
	// construct the SQL command.
	sprintf( sqlCommand, 	"SELECT		POLARIZATION_ID, SPECTRAL_WINDOW_ID "
				"FROM		%s/DATA_DESCRIPTION", pMeasurementSet );

	// issue SQL command to get data descriptions.
	Table tblDataDescription = tableCommand( sqlCommand );
	ScalarColumn<int> colPolarisationConfig( tblDataDescription, "POLARIZATION_ID" );
	ScalarColumn<int> colSpw( tblDataDescription, "SPECTRAL_WINDOW_ID" );
	*pNumDataDescItems = tblDataDescription.nrow();
		
	if (*pNumDataDescItems > 0)
	{

		// reserve memory for polarisation configs and spws.
		(*pPolarisationConfig) = (int *) malloc( *pNumDataDescItems * sizeof( int ) );
		(*pSpw) = (int *) malloc( *pNumDataDescItems * sizeof( int ) );

		// create the arrays.
		IPosition shape( 1, *pNumDataDescItems );
		Vector<int> vectorPolarisationConfig( shape, *pPolarisationConfig, SHARE );
		Vector<int> vectorSpw( shape, *pSpw, SHARE );

		// get the data.
		colPolarisationConfig.getColumn( vectorPolarisationConfig, false );
		colSpw.getColumn( vectorSpw, false );
		
	}

} // CasacoreInterface::getDataDesc
						
//
//	CasacoreInterface::getPolarisations()
//
//	CJS: 18/02/2020
//
//	get a list of polarisations from the measurement set
//

bool CasacoreInterface::getPolarisations( const char * pMeasurementSet, int * pNumPolarisations, int * pNumPolarisationConfigurations, int ** pPolarisation )
{

	// initialise counts.
	*pNumPolarisations = 0;
	*pNumPolarisationConfigurations = 0;
	
	//float * sample = NULL;
	char sqlCommand[ 1000 ];
	
	// construct the SQL command.
	sprintf( sqlCommand, 	"SELECT		CORR_TYPE "
				"FROM		%s/POLARIZATION", pMeasurementSet );

	// issue SQL command to get polarisations.
	Table tblPolarisations = tableCommand( sqlCommand );
	ArrayColumn<int> colPolarisations( tblPolarisations, "CORR_TYPE" );
	Array<int> arrayPolarisations;
		
	// get the whole column, and the shape of this column.
	colPolarisations.getColumn( arrayPolarisations, false );
	IPosition polarisationShape = arrayPolarisations.shape();
	
	// ensure that we have a single row with a 1-D arrays, and count the number elements in that array.
	if (polarisationShape.nelements() == 2)
	{
		*pNumPolarisations = (int) polarisationShape[ 0, 0 ];
		*pNumPolarisationConfigurations = (int) polarisationShape[ 0, 1 ];
	}

	if (*pNumPolarisations > 0 && *pNumPolarisationConfigurations > 0)
	{

		// reserve memory for polarisations.
		(*pPolarisation) = (int *) malloc( *pNumPolarisationConfigurations * *pNumPolarisations * sizeof( int ) );

		// copy the polarisation data into the required array.
		memcpy( *pPolarisation, arrayPolarisations.data(), *pNumPolarisationConfigurations * *pNumPolarisations * sizeof( int ) );
		
	}
	
	// return success or fail.
	return (*pNumPolarisations > 0 && *pNumPolarisationConfigurations > 0);
	
} // CasacoreInterface::getPolarisations
						
//
//	CasacoreInterface::getSamples()
//
//	CJS: 10/03/2017
//
//	get a list of samples from the measurement set.
//

bool CasacoreInterface::getSamples( const char * pMeasurementSet, int * pNumSamples, double ** pSample, char * pFieldID, bool * pDataDescFlag, int pDataDescItems,
					int ** pFieldIDArray, int ** pDataDescID, int ** pAntenna1, int ** pAntenna2 )
{
	
	//float * sample = NULL;
	char sqlCommand[ 1000 ];
	
	// construct the SQL command.
	sprintf( sqlCommand, 	"SELECT		UVW, FIELD_ID, DATA_DESC_ID, ANTENNA1, ANTENNA2 "
				"FROM		%s "
				"WHERE		(ANTENNA1 <> ANTENNA2) "
				"AND		(FLAG_ROW = FALSE)", pMeasurementSet );
	if (pFieldID[0] != '\0')
		sprintf( sqlCommand + strlen( sqlCommand ), " "
				"AND		(FIELD_ID IN (%s)) ", pFieldID );

	// restrict by data description id if we need to.
	if (pDataDescFlag != NULL)
	{

		sprintf( sqlCommand + strlen( sqlCommand ), " "
				"AND		(DATA_DESC_ID IN (" );

		bool commaNeeded = false;
		for ( int dataDesc = 0; dataDesc < pDataDescItems; dataDesc++ )
			if (pDataDescFlag[ dataDesc ] == false)
			{
				if (commaNeeded == true)
					sprintf( sqlCommand + strlen( sqlCommand ), ", " );
				sprintf( sqlCommand + strlen( sqlCommand ), "%i", dataDesc );
				commaNeeded = true;
			}
		if (commaNeeded == false)
			sprintf( sqlCommand + strlen( sqlCommand ), "-1" );

		sprintf( sqlCommand + strlen( sqlCommand ), "))" );

	}

	// issue SQL command to get samples.
	Table tblSamples = tableCommand( sqlCommand );
	*pNumSamples = tblSamples.nrow();
		
	if (*pNumSamples > 0)
	{

		// copy the sample data into the required array.
		ArrayColumn<double> colSamples( tblSamples, "UVW" );
		(*pSample) = (double *) malloc( 3 * sizeof( double ) * *pNumSamples );
		IPosition shape( 2, 3, *pNumSamples );
		Array<double> arraySamples( shape, *pSample, SHARE );
		colSamples.getColumn( arraySamples, false );

	}

	if (*pNumSamples > 0)
	{

		IPosition shape( 1, *pNumSamples );

		// copy the field ID data into the required array.
		ScalarColumn<Int> colFieldID( tblSamples, "FIELD_ID" );
		(*pFieldIDArray) = (int *) malloc( *pNumSamples * sizeof( int ) );
		Vector<Int> vectorFieldID( shape, *pFieldIDArray, SHARE );
		colFieldID.getColumn( vectorFieldID, false );

		// copy the data desc id data into the required array.
		ScalarColumn<Int> colDataDescID( tblSamples, "DATA_DESC_ID" );
		(*pDataDescID) = (int *) malloc( *pNumSamples * sizeof( int ) );
		Vector<Int> vectorDataDescID( shape, *pDataDescID, SHARE );
		colDataDescID.getColumn( vectorDataDescID, false );

		// copy the antennae columns into the required arrays.
		ScalarColumn<Int> colAntenna1( tblSamples, "ANTENNA1" );
		ScalarColumn<Int> colAntenna2( tblSamples, "ANTENNA2" );
		(*pAntenna1) = (int *) malloc( *pNumSamples * sizeof( int ) );
		(*pAntenna2) = (int *) malloc( *pNumSamples * sizeof( int ) );
		Vector<Int> vectorAntenna1( shape, *pAntenna1, SHARE );
		Vector<Int> vectorAntenna2( shape, *pAntenna2, SHARE );
		colAntenna1.getColumn( vectorAntenna1, true );
		colAntenna2.getColumn( vectorAntenna2, true );
		
	}
	
	// return success or fail.
	return (*pNumSamples > 0);
	
} // CasacoreInterface::getSamples

//
//	CasacoreInterface::getVisibilities()
//
//	CJS: 25/07/2016
//
//	Load the visibilities from the measurement set (we should have one visibility for each sample/channel combination).
//

void CasacoreInterface::getVisibilities( const char * pFilename, char * pFieldID, int * pNumSamples, int * pNumChannels, char * pDataField,
						complex<float> ** pVisibility, bool ** pFlag, double ** pSample, float ** pWeight,
						int ** pFieldIDArray, int ** pDataDescID, int pNumPolarisations, int pStartAnt1, int pStartAnt2,
						int pEndAnt1, int pEndAnt2, int pNumberOfAntennae, double pCurrentSample, double pTotalSamples,
						int pNumSpws, int * pDataDescSPW, bool * pDataDescFlag, int pNumDataDesc, bool ** pSpwChannelFlag )
{

	// initialise the number of samples in this batch to zero.
	*pNumSamples = 0;

	// ensure the data pointers start off as null.
	*pSample = NULL;
	*pFlag = NULL;
	*pWeight = NULL;
	*pFieldIDArray = NULL;
	*pDataDescID = NULL;
	*pVisibility = NULL;

	// initialise the memory offsets to zero.
	int sampleOffset = 0;
	long int flagOffset = 0;
	int weightOffset = 0;
	int fieldIDOffset = 0;
	int dataDescOffset = 0;
	long int visibilityOffset = 0;

	// if we've been given the SPWs from the data description table then we need to look through the channel list to see if we need to retrieve our data in batches.
	// we do this because we can only retrieve samples with the SAME number of channels in each query.
	int currentSpw = 0;
	while (currentSpw < pNumSpws)
	{

		// issue SQL command to get visibilities.
		char sqlCommand[ 1000 ];
		sprintf(	sqlCommand,	"SELECT		%s, FLAG, UVW, WEIGHT, FIELD_ID, DATA_DESC_ID "
						"FROM		%s "
						"WHERE		(ANTENNA1 <> ANTENNA2) "
						"AND		(FLAG_ROW = FALSE) "
						"AND		((ANTENNA1 = %i AND ANTENNA2 >= %i AND ANTENNA2 <= %i)",
									pDataField, pFilename,
									pStartAnt1, pStartAnt2, (pStartAnt1 == pEndAnt1 ? pEndAnt2 : pNumberOfAntennae - 1) );
		if (pEndAnt1 > pStartAnt1 + 1)
			sprintf( sqlCommand + strlen( sqlCommand ),
						" OR		 (ANTENNA1 > %i AND ANTENNA1 < %i)",
								pStartAnt1, pEndAnt1 );
		if (pEndAnt1 > pStartAnt1)
			sprintf( sqlCommand + strlen( sqlCommand ),
						" OR		 (ANTENNA1 = %i AND ANTENNA2 >= %i AND ANTENNA2 <= %i)",
								pEndAnt1, pEndAnt1 + 1, pEndAnt2 );
		sprintf( sqlCommand + strlen( sqlCommand ),    ")" );
		if (pFieldID[0] != '\0')
			sprintf( sqlCommand + strlen( sqlCommand ),
						" AND		(FIELD_ID IN (%s))", pFieldID );

		// look for all the spws with this number of channels.
		bool itemFound = false;
		while (itemFound == false && currentSpw < pNumSpws)
		{

			char whereCommand[ 1000 ];
			sprintf( whereCommand,	" AND		(DATA_DESC_ID IN (" );

			// search for all the data descriptions with this spw.
			for ( int dataDesc = 0; dataDesc < pNumDataDesc; dataDesc++ )
				if (pDataDescSPW[ dataDesc ] == currentSpw && pDataDescFlag[ dataDesc ] == false)
				{
					if (itemFound == true)
						sprintf( whereCommand + strlen( whereCommand ), ",%i", dataDesc );
					else
						sprintf( whereCommand + strlen( whereCommand ), "%i", dataDesc );
					itemFound = true;
				}

			sprintf( whereCommand + strlen( whereCommand ),
								"))" );

			// update the sql statement with the new where statement.
			if (itemFound == true)
				strcat( sqlCommand, whereCommand );
			else
				currentSpw++;

		}

		// only execute sql statement if we managed to find some data description ids to search for.
		if (currentSpw < pNumSpws)
		{

			Table tblVisibilities = tableCommand( sqlCommand );

			// process samples.
			int numSamples = tblVisibilities.nrow();
			*pNumSamples += numSamples;

			// declare slices for getting the channels that we are interested in.
			Vector<Vector<Slice> > slices( 2 );
			int numChannels = 0;
			{

				// count the number of separate channel ranges we have got.
				int channelRanges = 0;
				bool inValid = false;
				for ( int channel = 0; channel < pNumChannels[ currentSpw ]; channel++ )
				{
					if (pSpwChannelFlag[ currentSpw ][ channel ] == false && inValid == false)
						channelRanges++;
					if (pSpwChannelFlag[ currentSpw ][ channel ] == false)
						numChannels++;
					inValid = (pSpwChannelFlag[ currentSpw ][ channel ] == false);
				}
				slices[ 1 ].resize( channelRanges );		// update the number of separate channel ranges we have got.

				inValid = false;
				int startChannel = -1;
				int slice = -1;
				for ( int channel = 0; channel < pNumChannels[ currentSpw ]; channel++ )
				{

					if (pSpwChannelFlag[ currentSpw ][ channel ] == false && inValid == false)
					{
						startChannel = channel;
						slice++;
					}

					if (pSpwChannelFlag[ currentSpw ][ channel ] == true && inValid == true)
						slices[ 1 ][ slice ] = Slice( startChannel, channel - startChannel );
					inValid = (pSpwChannelFlag[ currentSpw ][ channel ] == false);

				}
				if (inValid == true)
					slices[ 1 ][ slice ] = Slice( startChannel, pNumChannels[ currentSpw ] - startChannel );

			}

			if (numSamples > 0)
			{

				ArrayColumn<double> colSamples( tblVisibilities, "UVW" );

				// create memory for samples and retrieve them.
				if (*pSample == NULL)
					(*pSample) = (double *) malloc( *pNumSamples * 3 * sizeof( double ) );
				else
					(*pSample) = (double *) realloc( *pSample, *pNumSamples * 3 * sizeof( double ) );

				IPosition shape( 2, 3, numSamples );
				Array<double> arraySamples( shape, &(*pSample)[ sampleOffset * 3 ], SHARE );
				colSamples.getColumn( arraySamples, false );

			}

			// process visibility flags.
			if (numSamples > 0)
			{

				ArrayColumn<Bool> colFlags( tblVisibilities, "FLAG" );

				// create memory for the flags, and retrieve them.
				if (*pFlag == NULL)
					(*pFlag) = (bool *) malloc( (long int) *pNumSamples * (long int) numChannels * (long int) pNumPolarisations *
										(long int) sizeof( bool ) );
				else
					(*pFlag) = (bool *) realloc( *pFlag, (flagOffset + ((long int) numSamples * (long int) numChannels *
									(long int) pNumPolarisations)) * (long int) sizeof( bool ) );

				IPosition shape( 3, pNumPolarisations, numChannels, numSamples );
				Array<Bool> arrayFlags( shape, &(*pFlag)[ flagOffset ], SHARE );
				colFlags.getColumn( slices, arrayFlags, false );

			}

			// process weights.
			if (numSamples > 0)
			{

				ArrayColumn<Float> colWeights( tblVisibilities, "WEIGHT" );

				// create memory for the weights, and retrieve them.
				if (*pWeight == NULL)
					(*pWeight) = (float *) malloc( *pNumSamples * pNumPolarisations * sizeof( float ) );
				else
					(*pWeight) = (float *) realloc( *pWeight, *pNumSamples * pNumPolarisations * sizeof( float ) );

				IPosition shape( 2, pNumPolarisations, numSamples );
				Array<Float> arrayWeights( shape, &(*pWeight)[ weightOffset ], SHARE );
				colWeights.getColumn( arrayWeights, false );

			}

			// process field ids
			if (numSamples > 0)
			{

				ScalarColumn<Int> colFieldID( tblVisibilities, "FIELD_ID" );

				// create memory for the field ids, and retrieve them.
				if (*pFieldIDArray == NULL)
					(*pFieldIDArray) = (int *) malloc( *pNumSamples * sizeof( int ) );
				else
					(*pFieldIDArray) = (int *) realloc( *pFieldIDArray, *pNumSamples * sizeof( int ) );

				IPosition shape( 1, numSamples );
				Vector<Int> vectorFieldID( shape, &(*pFieldIDArray)[ fieldIDOffset ], SHARE );
				colFieldID.getColumn( vectorFieldID, false );

			}

			// process data desc id
			if (numSamples > 0)
			{

				ScalarColumn<Int> colDataDescID( tblVisibilities, "DATA_DESC_ID" );

				// create memory for the data desc id, and retrieve them.
				if (*pDataDescID == NULL)
					(*pDataDescID) = (int *) malloc( *pNumSamples * sizeof( int ) );
				else
					(*pDataDescID) = (int *) realloc( *pDataDescID, *pNumSamples * sizeof( int ) );

				IPosition shape( 1, numSamples );
				Vector<Int> vectorDataDescID( shape, &(*pDataDescID)[ dataDescOffset ], SHARE ) ;
				colDataDescID.getColumn( vectorDataDescID, false );

			}

			// process visibilities. we do this inside a block so that objects will go out of scope and release memory.
			if (numSamples > 0)
			{

				// create memory for the visibilities.
				if (*pVisibility == NULL)
					(*pVisibility) = (complex<float> *) malloc( (long int) numSamples * (long int) numChannels * (long int) pNumPolarisations *
											sizeof( complex<float> ) );
				else
					(*pVisibility) = (complex<float> *) realloc( *pVisibility, (visibilityOffset + ((long int) numSamples * (long int) numChannels *
											(long int) pNumPolarisations)) * sizeof( complex<float> ) );

				ArrayColumn<Complex> colVisibilities( tblVisibilities, pDataField );
				IPosition shape( 3, pNumPolarisations, numChannels, numSamples );
				Array<Complex> arrayVisibilities( shape, &(*pVisibility)[ visibilityOffset ], SHARE );
				colVisibilities.getColumn( slices, arrayVisibilities, false );

			}

			printf( "\rloading visibilities.....%3d%%", (int) ((double) (pCurrentSample + sampleOffset + numSamples) * 100.0 /
										(double) pTotalSamples) );
			fflush( stdout );

			// update the offsets.
			sampleOffset += numSamples;
			flagOffset += (long int) numSamples * (long int) numChannels * (long int) pNumPolarisations;
			weightOffset += numSamples * pNumPolarisations;
			fieldIDOffset += numSamples;
			dataDescOffset += numSamples;
			visibilityOffset += (long int) numSamples * (long int) numChannels * (long int) pNumPolarisations;
			currentSpw++;

		}

	}

} // CasacoreInterface::getVisibilities

//
//	CasacoreInterface::getPhaseCentres()
//
//	CJS: 26/09/2018
//
//	Get a list of phase centres for each field.
//

bool CasacoreInterface::getPhaseCentres( const char * pMeasurementSet, int * pNumFields, double ** pPhaseCentre )
{
	
	//float * wavelength = NULL;
	char sqlCommand[ 1000 ];
	
	// construct the SQL command.
	sprintf( sqlCommand,	"SELECT		PHASE_DIR "
				"FROM		%s/FIELD", pMeasurementSet );
	
	// issue SQL command to get the fields.
	Table tblFields = tableCommand( sqlCommand );
	ArrayColumn<double> colPhaseCentre( tblFields, "PHASE_DIR" );
	Array<double> phaseCentreArray;
		
	// get the whole column, and the shape of this column.
	colPhaseCentre.getColumn( phaseCentreArray, true );
	IPosition phaseCentreShape = phaseCentreArray.shape();

	if (phaseCentreShape.nelements() == 3)
		*pNumFields = (int)phaseCentreShape[ 0, 2 ];

	if (*pNumFields > 0)
	{

		// reserve some memory for the phase centres.
		*pPhaseCentre = (double *) malloc( *pNumFields * 2 * sizeof( double ) );

		if (phaseCentreShape[ 0, 0 ] == 2)
		{

			// loop over the fields, getting the ra and dec for each field.
			for ( int field = 0; field < *pNumFields; field++ )
			{

				// there are two data items in the array - these should be RA and dec.
				IPosition uCellRA( 3, 0, 0, field );
				IPosition uCellDec( 3, 1, 0, field );
				double tmpRA = phaseCentreArray( uCellRA );
				double tmpDec = phaseCentreArray( uCellDec );

				// update the phase centre array.
				(*pPhaseCentre)[ field * 2 ] = tmpRA;
				(*pPhaseCentre)[ (field * 2) + 1 ] = tmpDec;

			}

		}

	}

} // CasacoreInterface::getPhaseCentres
