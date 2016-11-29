/**
 * Field Theory Processing Library
 * Copyright Thomas Winyard 2016
 */

#ifndef FTPL_DERIVATIVES
#define FTPL_DERIVATIVES

#include "Types.hpp"
#include "Exceptions.hpp"
#include <typeinfo>
#include <iostream>

namespace FTPL{

template <class S, class T, class V>
void derivative_1storder(const S * field, T * derivative, const V * pos, const V * dir, float spacing) {
	derivative = (field(pos+dir) - field(pos-dir))/(2.0*spacing);
}

template <class S, class T, class V>
void derivative_2ndorder(const S * field, T * derivative, const V * pos, const V * dir, float spacing) {
	derivative = (-1.0*field(pos+2*dir) + 8.0*field(pos+dir) - 8.0*field(pos-dir) + field(pos-2*dir))/(12.0*spacing);
}

template <class S, class T, class V>
void doublederivative_diag_1storder(const S * field, T * derivative, const V * pos, const V * dir, float spacing) {
	derivative = (field(pos+dir) - 2.0*field(pos)  + field(pos-dir))/(spacing*spacing);
}

template <class S, class T, class V>
void doublederivative_offdiag_1storder(const S * field, T * derivative, const V * pos, const V * dir1, const V * dir2, float spacing1, float spacing2) {
	derivative = (field(pos+dir1+dir2) - field(pos+dir1-dir2)  - field(pos-dir1+dir2) + field(pos-dir1-dir2))/(4.0*spacing1*spacing2);
}

template <class S, class T, class V>
void doublederivative_diag_2ndorder(const S * field, T * derivative, const V * pos, const V * dir, float spacing) {
	derivative = (-field(pos+2*dir) + 16.0*field(pos+dir) - 30.0*field(pos) + 16.0*field(pos-dir) - field(pos-2*dir))/(12.0*spacing*spacing);
}

template <class S, class T, class V>
void doublederivative_offdiag_2ndorder(const S * field, T * derivative, const V * pos, const V * dir, float spacing) {
	derivative = (field(pos+2*dir1+2*dir2) - 8.0*field(pos+dir1+2*dir2) + 8.0*field(pos-dir1+2*dir2) - field(pos-2*dir1+2*dir2) - 8.0*field(pos+2*dir1+dir2) +64.0*field(pos+dir1+dir2) -64.0*field(pos-dir1+dir2) + 8.0* field(pos-2*dir1+dir2) + 8.0*field(pos+2*dir1-dir2) - 64.0*field(pos+dir1-dir2)+64.0*field(pos-dir1-dir2) - 8.0*field(pos-2*dir1-dir2) - field(pos+2*dir1-2*dir2) + 8.0*field(pos+dir1-2*dir2) - 8.0*field(pos-dir1-2*dir2) + field(pos-2*dir1-2*dir2))/(144.0*spacing1*spacing2);
}

}; // end FTPL namespace
#endif
