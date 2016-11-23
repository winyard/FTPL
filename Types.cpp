#include "Types.hpp"
namespace FTPL {

// float_v
float float_v::distance(float_v other) const {
    float sum = 0.0;
    for(int i=0; i<this->dim; i++){sum += (this->f[i]-other.f[i])*(this->f[i]-other.f[i]);}
    return sqrt(sum);
}
float float_v::dot(float_v other) const {
    float sum = 0.0;
    for(int i=0; i<this->dim; i++){sum += f[i]*other.f[i];}
    return sum;
}

};
