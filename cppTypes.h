//
//	STRUCTURES
//

// vector with floats. Can be used either as a 2 or 3 element vector.
struct VectorF
{
	double u;
	double v;
	double w;
};
typedef struct VectorF VectorF;

// 2-d vector with integers. used to store grid positions.
struct GridUV_I
{
	int u;
	int v;
};
typedef struct GridUV_I GridUV_I;

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
