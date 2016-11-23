/**
 * Field Theory Processing Library
 * Copyright Thomas Winyard 2016
 */

#ifndef FTPL_TYPES
#define FTPL_TYPES

#include <math.h>
#include <stdarg.h>

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;

namespace FTPL{
typedef struct color_float { float red, blue, green;} color_float ; 
typedef struct color_uchar { unsigned char red, blue, green;} color_uchar ;
enum slice_plane {X,Y,Z};

/* FOR NOW JUST A GENERAL INT_V and FLOAT_V aka general vectors as target/local space values spaces */

class int2;
class int3;
class float2;

class int_v;
class float_v;

class float_v {
    public:
	int dim;
	float* f;
        float_v(int d) {
	    this->dim = d;
	    this->f = new float[d];
	    for(int i = 0; i < d; i++){
		this->f[i] = 0.0f;
	    }
        };
        float_v(int d, ...) {
	    this->dim = d;
	    this->f = new float[d];
	    va_list ap;
	    va_start(ap, d);
	    for(int i=0; i<d; i++){
	    	this->f[i] = va_arg(ap,float);
	    }
        };

        float length() const {
	    float sum = 0.0;
	    for(int i=0; i<this->dim; i++){sum += (this->f[i])*(this->f[i]);}
            return sqrt((float)(sum));
        };
        float_v normalize() {
            float l = this->length();
	    float_v n(this->dim);
	    for(int i=0; i<this->dim; i++){n.f[i] = this->f[i] / l;}
            return n;
        };
        float distance(float_v other) const ;
        float dot(float_v other) const ;
        bool operator==(float_v other) const;
        template <class T>
        float4 operator+(T v) const {
            float_v n(this->dim);
	    for(int i=0; i<this->dim; i++){n.f[i] = this->f[i] + v;}
            return n;
        };
        template <class T>
        float_v operator-(T v) const {
            float_v n(this->dim);
	    for(int i=0; i<this->dim; i++){n.f[i] = this->f[i] - v;}
            return n;
        };
        template <class T>
        float4 operator*(T v) const {
            float_v n(this->dim);
	    for(int i=0; i<this->dim; i++){n.f[i] = this->f[i] * v;}
            return n;
        };
        template <class T>
        float_v operator/(T v) const {
            float_v n(this->dim);
	    for(int i=0; i<this->dim; i++){n.f[i] = this->f[i] / v;}
            return n;
        };
        ~float_v(){delete f};
};

//SkyrmionTargetSpace operators
template <> inline
float_v float_v::operator+(float_v other) const {
    float_v v(this->dim);
    for(int i=0; i<this->dim; i++){v.f[i] = this->f[i] + other.f[i];}
    return v;
}
template <> inline
float_v float_v::operator-(float_v other) const {
    float_v v(this->dim);
    for(int i=0; i<this->dim; i++){v.f[i] = this->f[i] - other.f[i];}
    return v;
}
template <> inline
float_v float_v::operator*(float_v other) const {
    float_v v;
    for(int i=0; i<this->dim; i++){v.f[i] = this->f[i] * other.f[i];}
    return v;
}

inline
float_v operator+(float scalar, float_v other) {
    return other.operator+(scalar);
}
inline
float_v operator-(float scalar, float_v other) {
    float_v v;
    for(int i=0; i<this->dim; i++){v.f[i] = scalar - other.f[i];}
    return v;
}
inline
float_v operator*(float scalar, float_v other) {
    return other.operator*(scalar);
}

// int_v

class int_v {
    public:

	int dim;
	int* x;
        int_v(int d) {
	    this->dim = d;
	    this->x = new int[d];
	    for(int i = 0; i < d; i++){
		this->x[i] = 0;
	    }
        };
        int_v(int d, ...) {
	    this->dim = d;
	    this->x = new int[d];
	    va_list ap;
	    va_start(ap, d);
	    for(int i=0; i<d; i++){
	    	this->x[i] = va_arg(ap,int);
	    }
        };

        float length() const {
	    float sum = 0.0;
	    for(int i=0; i<this->dim; i++){sum += (float)((this->x[i])*(this->x[i]));}
            return sqrt((float)(sum));
        };
        float_v normalize() {
            float l = this->length();
	    float_v n(this->dim);
	    for(int i=0; i<this->dim; i++){n.f[i] = this->x[i] / l;}
            return n;
        };
        float distance(float_v other) const ;
        float dot(float_v other) const ;
        float distance(int_v other) const ;
        float dot(int_v other) const ;

        bool operator==(int_v other) const;
        bool operator==(float_v other) const;
        float_v operator+(float v) const;
        float_v operator+(float_v v) const;
        int_v operator+(int_v v) const;
        float_v operator-(float v) const;
        float_v operator-(float_v v) const;
        int_v operator-(int_v v) const;
        float_v operator*(float v) const;
        float_v operator*(float_v v) const;
        int_v operator*(int_v v) const;
        float_v toFloat() const {
            return float_v(x,y);
	};
        ~int_v(){delete x};
}

/*class Region {
    public:
        int3 offset;
        int3 size;
        Region(int x_size, int y_size) {
            offset = int3(0,0,0);
            size = int3(x_size, y_size,0);
        };
        Region(int x_offset, int y_offset, int x_size, int y_size) {
            offset = int3(x_offset, y_offset,0);
            size = int3(x_size, y_size, 0);
        };
        Region(int x_size, int y_size, int z_size) {
            offset = int3(0,0,0);
            size = int3(x_size, y_size, z_size);
        };
        Region(int x_offset, int y_offset, int z_offset, int x_size, int y_size, int z_size) {
            offset = int3(x_offset, y_offset, z_offset);
            size = int3(x_size, y_size, z_size);
        };
};*/

// int2
inline
float_v int_v::operator+(float other) const {
    float_v v(this->dim);
    for(int i=0; i<this->dim; i++){v.f[i] = this->x[i] + other;}
    return v;
}

inline
float_v int_v::operator+(float_v other) const {
    float_v v(this->dim);
    for(int i=0; i<this->dim; i++){v.f[i] = this->x[i] + other.f[i];}
    return v;
}

inline
int_v int_v::operator+(int_v other) const {
    int2 v(this->dim);
    for(int i=0; i<this->dim; i++){v.x[i] = this->x[i] + other.x[i];}
    return v;
}
inline
float_v int_v::operator-(float other) const {
    float_v v(this->dim);
    for(int i=0; i<this->dim; i++){v.f[i] = this->x[i] - other;}
    return v;
}
inline
float_v int_v::operator-(float_v other) const {
    float_v v(this->dim);
    for(int i=0; i<this->dim; i++){v.f[i] = this->x[i] - other.f[i];}
    return v;
}
inline
int_v int_v::operator-(int_v other) const {
    int_v v(this->dim);
    for(int i=0; i<this->dim; i++){v.x[i] = this->x[i] - other.x[i];}
    return v;
}
inline
float_v int_v::operator*(float other) const {
    float_v v(this->dim);
    for(int i=0; i<this->dim; i++){v.f[i] = this->x[i] * other;}
    return v;
}
inline
float_v int_v::operator*(float_v other) const {
    float_v v(this->dim);
    for(int i=0; i<this->dim; i++){v.f[i] = this->x[i] * other.f[i];}
    return v;
}
inline
int_v int_v::operator*(int_v other) const {
    int_v v(this->dim);
    for(int i=0; i<this->dim; i++){v.x[i] = this->x[i] + other.x[i];}
    return v;
}

inline
float2 operator+(float scalar, int_v other) {
    return other.operator+(scalar);
}
inline
float_v operator-(float scalar, int_v other) {
    float_v v(other.dim);
    for(int i=0; i<this->dim; i++){v.f[i] = scalar - other.x[i];}
    return v;
}
inline
float_v operator*(float scalar, int_v other) {
    return other.operator*(scalar);
}

}; // end FTPL namespace
#endif
