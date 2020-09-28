#include <stdio.h>
#include <stdlib.h>
#include <complex>

//
//	CasacoreInterface
//
//	CJS: 08/03/2017
//
//	Provides an interface between the GPU/CPU gridder and casacore.
//

class CasacoreInterface
{
	
	public:
		
		// default constructor
		CasacoreInterface();
   
		// destructor
		~CasacoreInterface();

		// get instance.
		static CasacoreInterface * getInstance();
	
		// load bitmap
		bool LoadBitmap( const char * pFilename, std::complex<double> ** pImageData );
		bool LoadBitmap( const char * pFilename, std::complex<double> ** pImageData, int * pWidth, int * pHeight );
		bool LoadBitmap( const char * pFilename, std::complex<double> ** pImageData, double * pImageScale );
		bool LoadBitmap( const char * pFilename, std::complex<double> ** pImageData, int * pWidth, int * pHeight, double * pImageScale );
						
		// save bitmap
		bool SaveBitmap( const char * pFilename, std::complex<double> * pImageData, int pWidth, int pHeight );
		bool SaveBitmap( const char * pFilename, std::complex<double> * pImageData, int pWidth, int pHeight, double * pImageScale );

		// read from a casa image
		void ReadCasaImage( const char * pFilename );
		
		// write a casa image
		void WriteCasaImage( const char * pFilename, int pWidth, int pHeight, double pRA, double pDec,
					double pPixelSize, std::complex<double> * pImage, double pFrequency, bool * pMask );
		void WriteCasaImage( const char * pFilename, int pWidth, int pHeight, double pRA, double pDec,
					double pPixelSize, double * pImage, double pFrequency, bool * pMask );
		void WriteCasaImage( const char * pFilename, int pWidth, int pHeight, double pRA, double pDec,
					double pPixelSize, float * pImage, double pFrequency, bool * pMask );
		void WriteCasaImage( const char * pFilename, int pWidth, int pHeight, double pRA, double pDec,
					double pPixelSize, std::complex<float> * pImage, double pFrequency, bool * pMask );
						
		// count the number of antennae in the measurement set.
		int NumberOfAntennae( const char * pMeasurementSet );

		// get a list of antennae.
		int GetAntennae( const char * pMeasurementSet, double ** pDishDiameter, bool ** pFlagged );
						
		// get a list of channel frequencies from the measurement set
		bool GetWavelengths( const char * pMeasurementSet, int * pNumSpws, int ** pNumChannels, double *** pWavelength );

		// get all the data from the data description table.
		void GetDataDesc( const char * pMeasurementSet, int * pNumDataDescItems, int ** pPolarisationConfig, int ** pSpw );

		// get a list of polarisations from the measurement set.
		bool GetPolarisations( const char * pMeasurementSet, int * pNumPolarisations, int * pNumPolarisationConfigurations, int ** pPolarisation );
						
		// get a list of samples from the measurement set
		bool GetSamples( const char * pMeasurementSet, int * pNumSamples, double ** pSample, char * pFieldID, bool * pDataDescFlag, int pDataDescItems,
				 int ** pFieldIDArray, int ** pDataDescID, int ** pAntenna1, int ** pAntenna2 );
					
		// load a batch of visibilities from the file
		void GetVisibilities( const char * pFilename, char * pFieldID, int * pNumSamples, int * pNumChannels, char * pDataField,
					std::complex<float> ** pVisibility, bool ** pFlag, double ** pSample, float ** pWeight,
					int ** pFieldIDArray, int ** pDataDescID, int pNumPolarisations, int pStartAnt1, int pStartAnt2,
					int pEndAnt1, int pEndAnt2, int pNumberOfAntennae, double pCurrentSample, double pTotalSamples,
					int pNumSpws, int * pDataDescSPW, bool * pDataDescFlag, int pNumDataDesc, bool ** pSpwChannelFlag );

		// load the phase centres for each field.
		bool GetPhaseCentres( const char * pMeasurementSet, int * pNumFields, double ** pPhaseCentre );
	
	private:

		// the static instance.
		static CasacoreInterface _thisInstance;

		static const double CONST_C;
		static const double PI;

		// bitmap file header positions.
		static const int BIT_CONST = 0x00;
		static const int MAP_CONST = 0x01;
		static const int IMAGE_SIZE = 0x02;
		static const int RESERVED = 0x06;
		static const int FILE_HEADER_SIZE = 0x0A;
		static const int BITMAP_INFO_HEADER = 0x0E;
		static const int IMAGE_WIDTH = 0x12;
		static const int IMAGE_HEIGHT = 0x16;
		static const int COLOUR_PLANES = 0x1A;
		static const int BIT_COUNT = 0x1C;
		static const int COMPRESSION_TYPE = 0x1E;
		static const int COLOURS_USED = 0x2E;
		static const int SIGNIFICANT_COLOURS = 0x32;

		// enumerated types.
		enum ffttype { COMPLEX, DOUBLE, FLOAT, COMPLEX_FLOAT };
		
		// load the bitmap.
		bool loadBitmap( const char * pFilename, std::complex<double> ** pImageData, int * pWidth, int * pHeight,
					double * pImageScale );
						
		// save the bitmap.
		bool saveBitmap( const char * pFilename, std::complex<double> * pImageData, int pWidth, int pHeight,
					double * pImageScale );
		
		// write a casa image.
		void writeCasaImage( const char * pFilename, int pWidth, int pHeight, double pRA, double pDec,
					double pPixelSize, void * pImage, double pFrequency, bool * pMask, ffttype pFFTType );

		// get all the antennae from the file, including the flagged ones.
		int getAntennae( const char * pMeasurementSet, double ** pDishDiameter, bool ** pFlagged );
						
		// count the number of antennae in the measurement set.
		int numberOfAntennae( const char * pMeasurementSet );
						
		// get a list of channel frequencies from the measurement set.
		bool getWavelengths( const char * pMeasurementSet, int * pNumSpws, int ** pNumChannels, double *** pWavelength );

		// get all the data from the data description table.
		void getDataDesc( const char * pMeasurementSet, int * pNumDataDescItems, int ** pPolarisationConfig, int ** pSpw );

		// get a list of polarisations from the measurement set.
		bool getPolarisations( const char * pMeasurementSet, int * pNumPolarisations, int * pNumPolarisationConfigurations, int ** pPolarisation );
						
		// get a list of samples from the measurement set.
		bool getSamples( const char * pMeasurementSet, int * pNumSamples, double ** pSample, char * pFieldID, bool * pDataDescFlag, int pDataDescItems,
					int ** pFieldIDArray, int ** pDataDescID, int ** pAntenna1, int ** pAntenna2 );
					
		// load a batch of visibilities from the file.
		void getVisibilities( const char * pFilename, char * pFieldID, int * pNumSamples, int * pNumChannels, char * pDataField,
					std::complex<float> ** pVisibility, bool ** pFlag, double ** pSample, float ** pWeight,
					int ** pFieldIDArray, int ** pDataDescID, int pNumPolarisations, int pStartAnt1, int pStartAnt2,
					int pEndAnt1, int pEndAnt2, int pNumberOfAntennae, double pCurrentSample, double pTotalSamples,
					int pNumSpws, int * pDataDescSPW, bool * pDataDescFlag, int pNumDataDesc, bool ** pSpwChannelFlag );

		// load the phase centres for each field.
		bool getPhaseCentres( const char * pMeasurementSet, int * pNumFields, double ** pPhaseCentre );
    
}; // CasacoreInterface
