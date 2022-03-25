

//
//	ENUMERATED TYPES
//

enum fftdirection { FORWARD, INVERSE };
enum griddegrid { GRID, DEGRID };
enum weighting { NONE, NATURAL, UNIFORM, ROBUST };
enum deconvolver { HOGBOM, MFS };
enum findpixel { CLOSEST, FURTHEST };
//enum stokes { STOKES_I, STOKES_Q, STOKES_U, STOKES_V, STOKES_NONE, STOKES_ALL };
enum telescope { UNKNOWN_TELESCOPE, ASKAP, EMERLIN, ALMA, ALMA_7M, ALMA_12M, VLA, MEERKAT };
enum ffttype { C2C, F2C, C2F, F2F };
enum masktype { MASK_MAX, MASK_MIN };
enum mosaicdomain { IMAGE, UV };
enum addsubtract { ADD, SUBTRACT };
enum visibilitytype { RESIDUAL, OBSERVED };
enum imagetype { DIRTY, PSF };
enum beamtype { AIRY, GAUSSIAN, FROMFILE };

//
//	STRUCTURES
//

// 3x3 matrix with floats.
struct Matrix3x3
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
typedef struct Matrix3x3 Matrix3x3;

// vector with double. Can be used either as a 2 or 3 element vector.
struct VectorD
{
	double u;
	double v;
	double w;
};
typedef struct VectorD VectorD;

// vector with integers. Can be used either as a 2 or 3 element vector.
struct VectorI
{
	int u;
	int v;
	int w;
};
typedef struct VectorI VectorI;

//
//	CONSTANTS
//

// speed of light.
const double CONST_C = 299792458.0;
const double PI = 3.141592654;

const int MAX_THREADS = 33554432;		// maximum number of total threads per cuda call (32768 x 1024). We can actually have 65535 x 1024, but we set
						//	the limit lower.

// the type of data to cache and uncache.
const int DATA_VISIBILITIES = 0x01;
const int DATA_GRID_POSITIONS = 0x02;
const int DATA_KERNEL_INDEXES = 0x04;
const int DATA_DENSITIES = 0x08;
const int DATA_WEIGHTS = 0x10;
const int DATA_MFS_WEIGHTS = 0x20;
const int DATA_RESIDUAL_VISIBILITIES = 0x40;
const int DATA_ALL = DATA_VISIBILITIES | DATA_GRID_POSITIONS | DATA_KERNEL_INDEXES | DATA_DENSITIES | DATA_WEIGHTS | DATA_MFS_WEIGHTS;

// stokes parameters.
const int STOKES_I = 0;
const int STOKES_Q = 1;
const int STOKES_U = 2;
const int STOKES_V = 3;
const int STOKES_NONE = 4;
const int STOKES_ALL = 5;

// the size of the data area required by the routine to find the maximum pixel value.
const int MAX_PIXEL_DATA_AREA_SIZE = 5;
const int MAX_PIXEL_VALUE = 0;
const int MAX_PIXEL_X = 1;
const int MAX_PIXEL_Y = 2;
const int MAX_PIXEL_REAL = 3;
const int MAX_PIXEL_IMAG = 4;
