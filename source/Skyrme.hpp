/**
 * Field Theory Processing Library
 * Copyright Thomas Winyard 2016
 */

#ifndef FTPL_H_
#define FTPL_H_

#define _USE_MATH_DEFINES // windows crap

#include <cmath>

#include "Exceptions.hpp"
#include "Types.hpp"
#include "IntensityTransformations.hpp"
#ifdef USE_GTK
#include <gtk/gtk.h>
#include <gdk-pixbuf/gdk-pixbuf.h>
#endif
#include <fstream>
#include <typeinfo>
#include <string>
#include <stdlib.h>
#include <map>
#include <iostream>
#include <fstream>
using namespace std;

namespace FTPL {

template <class T>
class BabySkyrmeField : public BaseFieldTheory {
    public:
        BabySkyrmeField(const char * filepath);
        BabySkyrmeField(unsigned int width, unsigned int height);
        template <class U>
        BabySkyrmeField(BabySkyrmeField<U> * otherBabySkyrmeField, IntensityTransformation IT = IntensityTransformation(DEFAULT));//not sure I need this one!
        T get(int i) const;
        T get(int x, int y) const;
        T get(int2 pos) const;
        T * get(Region r) const;
		float2 getSpacing() const;
		void setSpacing(float2 spacing);
        void set(int x, int y, T value);
        void set(int2, T value);
        void set(int i, T v);
        void set(Region region, T value);
        int3 getSize() const;
        void save(const char * filename); // need to alter imagetype the file type or something
#ifdef USE_GTK
        void pixbufToData(GtkImage * image);
#endif
	//The maths
	float getDerivative(int direction, int component, int2 pos);
	float getDerivative(int direction, int component, int x, int j);
        template <class U>
        BabySkyrmeField<T> & operator=(const BabySkyrmeField<U> &otherBabySkyrmeField);
        bool inBounds(int x, int y) const;
        bool inBounds(int i) const;
        int getTotalSize() const;
        BabySkyrmeField<T> crop(Region r) const;
	private:
	float2 spacing;
	float mu, mpi;
};

template <class T>
class SkyrmeField : public Field<T> {
    public:
        SkyrmeField(const char * filename); // for reading raw files
        SkyrmeField(int width, int height, int depth);
        SkyrmeField(int3 size);
        template <class U>
        SkyrmeField(SkyrmeField<U> * otherSkyrmeField, IntensityTransformation IT = IntensityTransformation(DEFAULT));
        T get(int x, int y, int z) const;
        T get(int3 pos) const;
        T get(int i) const;
        T * get(Region r) const;
		float3 getSpacing() const;
		void setSpacing(float3 spacing);
        void set(int x, int y, int z, T value);
        void set(int3 pos, T v);
        void set(int i, T v);
        void set(Region region, T value);
        int getDepth() const;
        int3 getSize() const;
        void save(const char * filename);
        void saveSlice(int slice, slice_plane direction, const char * filepath, const char * imageType);
        template <class U>
        SkyrmeField<T> & operator=(const SkyrmeField<U> &otherSkyrmeField);
        bool inBounds(int x, int y, int z) const;
        bool inBounds(int3 pos) const;
        bool inBounds(int i) const;
        int getTotalSize() const;
		Visualization * display();
		Visualization * display(float level, float window);
		Visualization * display(int slice, slice_plane direction);
		Visualization * display(int slice, slice_plane direction, float level, float window);
		Visualization * displayMIP();
		Visualization * displayMIP(float level, float window);
		Visualization * displayMIP(slice_plane direction);
		Visualization * displayMIP(slice_plane direction, float level, float window);
		SkyrmeField<T> * crop(Region r) const;
    private:
        int depth;
	float Fpi, epi, mpi;
};

inline double round( double d ) {
    return floor( d + 0.5 );
}

template <class T>
void Field<T>::setDefaultLevelWindow() {
    this->defaultLevel = 0.5;
    this->defaultWindow = 1.0;
}


template <class T>
uchar levelWindow(T value, float level, float window) {
    float result;
    if(value < level-window*0.5f) {
        result = 0.0f;
    } else if(value > level+window*0.5f) {
        result = 1.0f;
    } else {
        result = (float)(value-(level-window*0.5f)) / window;
    }
    result = round(result*255);
    return result;
}

/* --- Spesialized toT functions --- */

void toT(bool * r, uchar * p) ;
void toT(uchar * r, uchar * p) ;
void toT(char * r, uchar * p) ;
void toT(ushort * r, uchar * p) ;
void toT(short * r, uchar * p) ;
void toT(uint * r, uchar * p) ;
void toT(int * r, uchar * p) ;
void toT(float * r, uchar * p) ;
void toT(color_uchar * r, uchar * p) ;
void toT(color_float * r, uchar * p) ;
void toT(float2 * r, uchar * p) ;
void toT(float3 * r, uchar * p) ;

#ifdef USE_GTK
template <class T>
void BabySkyrmeField<T>::pixbufToData(GtkImage * image) {
	gdk_threads_enter ();
    GdkPixbuf * pixBuf = gtk_image_get_pixbuf((GtkImage *) image);
    for(int i = 0; i < this->width*this->height; i++) {
        guchar * pixels = gdk_pixbuf_get_pixels(pixBuf);
        unsigned char * c = (unsigned char *)((pixels + i * gdk_pixbuf_get_n_channels(pixBuf)));
        toT(&this->data[i], c);
    }
    gdk_threads_leave();
}
#endif
int validateSlice(int slice, slice_plane direction, int3 size);

template <class T>
T maximum(T a, T b, bool * change) {
    *change = a < b;
    return a > b ? a : b;
}

template <>
inline float2 maximum<float2>(float2 a, float2 b, bool * change) {
    float2 c;
    c.x = a.x > b.x ? a.x : b.x;
    c.y = a.y > b.y ? a.y : b.y;
    *change = a.x < b.x || a.y < b.y;
    return c;
}

template <>
inline float3 maximum<float3>(float3 a, float3 b, bool * change) {
    float3 c;
    c.f[0] = a.f[0] > b.f[0] ? a.f[0] : b.f[0];
    c.f[1] = a.f[1] > b.f[1] ? a.f[1] : b.f[1];
    c.f[2] = a.f[2] > b.f[2] ? a.f[2] : b.f[2];
    *change = a.f[0] < b.f[0] || a.f[1] < b.f[1] || a.f[2] < b.f[2];
    return c;
}

template <>
inline float4 maximum<float4>(float4 a, float4 b, bool * change) {
    float4 c;
    c.f[0] = a.f[0] > b.f[0] ? a.f[0] : b.f[0];
    c.f[1] = a.f[1] > b.f[1] ? a.f[1] : b.f[1];
    c.f[2] = a.f[2] > b.f[2] ? a.f[2] : b.f[2];
    c.f[3] = a.f[3] > b.f[3] ? a.f[3] : b.f[3];
    *change = a.f[0] < b.f[0] || a.f[1] < b.f[1] || a.f[2] < b.f[2] || a.f[3] < b.f[3];
    return c;
}

template <>
inline color_uchar maximum<color_uchar>(color_uchar a, color_uchar b, bool * change) {
    color_uchar c;
    c.red = a.red > b.red ? a.red : b.red;
    c.green = a.green > b.green ? a.green : b.green;
    c.blue = a.blue > b.blue ? a.blue : b.blue;
    *change = a.red < b.red || a.green < b.green || a.blue < b.blue;
    return c;
}

/* --- Constructors & destructors --- */
template <class T>
Field<T>::Field() {
    T * d = NULL;
    this->isVectorType = true;
    this->setDefaultLevelWindow();
#ifdef USE_GTK
    Init();
#endif
}

template <class T>
BabySkyrmeField<T>::BabySkyrmeField(const char * filename) {
ifstream openfile(filename);
int a,b;
double d, e, g;
    this->isVolume = false;
// 1st line read in size axbxc and spacing dx,dy,dz
openfile >> a >> b >> d >> e ;
    this->width = a;
    this->height = b;
    this->spacing = float2(d,e);
    this->data = new T[a*b];
// 2nd line read in parameters
openfile >> d >> g;
    this->mu = d;
    this->mpi = g;
// read in data
while(!openfile.eof())
{
    openfile >> a >> b >> d >> e >> g ;
    if(!this->inBounds(a,b))
        throw OutOfBoundsException(a, b, this->width, this->height, __LINE__, __FILE__);
    this->data[a+b*this->width] = float3(d,e,g);
}
cout << "Baby Skyrme Field of dimensions [" << this->width<<"," << this->height << "] succesfully read from file " << filename << "\n";
}

template <class T> // set all addedparameters to be equivalent after a modification!
template <class U>
BabySkyrmeField<T>::BabySkyrmeField(BabySkyrmeField<U> * otherBabySkyrmeField, IntensityTransformation it) {
    this->isVolume = false;
    this->width = otherBabySkyrmeField->getWidth();
    this->height = otherBabySkyrmeField->getHeight();
    this->data = new T[this->height*this->width];
    this->spacing = otherBabySkyrmeField->getSpacing();
    it.transform(otherBabySkyrmeField->getData(), this->data, this->getTotalSize());
}

template <class T> 
template <class U>
BabySkyrmeField<T>& BabySkyrmeField<T>::operator=(const BabySkyrmeField<U> &otherBabySkyrmeField) {
    if(this->width != otherBabySkyrmeField.getWidth() || this->height != otherBabySkyrmeField.getHeight())
        throw ConversionException("Baby Skyrme field size mismatch in assignment", __LINE__, __FILE__);
    
    IntensityTransformation it;
    it.transform(otherBabySkyrmeField->getData(), this->data, this->getTotalSize());

    return *this;
}


template <class T> 
template <class U>
SkyrmeField<T>::SkyrmeField(SkyrmeField<U> * otherSkyrmeField, IntensityTransformation it) {
    this->width = otherSkyrmeField->getWidth();
    this->isVolume = true;
    this->height = otherSkyrmeField->getHeight();
    this->depth = otherSkyrmeField->getDepth();
    this->spacing = otherSkyrmeField->getSpacing();
    this->data = new T[this->height*this->width*this->depth];
    it.transform(otherSkyrmeField->getData(), this->data, this->getTotalSize());
}

template <class T> 
template <class U>
SkyrmeField<T>& SkyrmeField<T>::operator=(const SkyrmeField<U> &otherSkyrmeField) {
    if(this->width != otherSkyrmeField.getWidth() || 
        this->height != otherSkyrmeField.getHeight() ||
        this->depth != otherSkyrmeField.getDepth())
        throw ConversionException("Skyrme field size mismatch in assignment", __LINE__, __FILE__);
    
    IntensityTransformation it;
    it.transform(otherSkyrmeField->getData(), this->data, this->getTotalSize());

    return *this;
}


template <class T>
SkyrmeField<T>::SkyrmeField(const char * filename) {
ifstream openfile(filename);
int a,b,c;
double d, e, g, k;
    this->isVolume = true;
// 1st line read in size axbxc and spacing dx,dy,dz
openfile >> a >> b >> c >> d >> e >> g;
    this->width = a;
    this->height = b;
    this->depth = c;
    this->spacing = float3(c,d,e);
    this->data = new T[a*b*c];
// 2nd line read in parameters
openfile >> d >> e >> g;
    this->Fpi = d;
    this->epi = e;
    this->mpi = g;
// read in data
while(!openfile.eof())
{
    openfile >> a >> b >> c >> d >> e >> g >> k;
    if(!this->inBounds(a,b,c))
        throw OutOfBoundsException(a, b, c, this->width, this->height, this->depth, __LINE__, __FILE__);
    this->data[a+b*this->width+c*this->width*this->height] = float4(d,e,g,k);
}
cout << "Skyrme Field of dimensions [" << this->width<<"," << this->height << "," << this->depth << "] succesfully read from file " << filename << "\n";
}

template <class T>
SkyrmeField<T>::SkyrmeField(int width, int height, int depth) {
    this->data = new T[width*height*depth];
    this->width = width;
    this->height = height;
    this->depth = depth;
    this->isVolume = true;
    this->spacing = float3(1.0f,1.0f,1.0f);
}

template <class T>
SkyrmeField<T>::SkyrmeField(int3 size) {
    this->data = new T[size.x*size.y*size.z];
    this->width = size.x;
    this->height = size.y;
    this->depth = size.z;
    this->isVolume = true;
    this->spacing = float3(1.0f,1.0f,1.0f);
}


template <class T>
BabySkyrmeField<T>::BabySkyrmeField(unsigned int width, unsigned int height) {
    this->data = new T[width*height];
    this->width = width;
    this->height = height;
    this->isVolume = false;
    this->spacing = float3(1.0f,1.0f,1.0f); // SHOULD BE A FLOAT2 !?!?!?! as is the dx/dy no?
}

template <class T>
BabySkyrmeField<T>::BabySkyrmeField(int2 size) {
    this->data = new T[size.x*size.y];
    this->width = size.x;
    this->height = size.y;
    this->isVolume = false;
    this->spacing = float3(1.0f,1.0f,1.0f); // SHOULD BE A FLOAT2 !?!?!?! as is the dx/dy no?
}


template <class T>
Field<T>::~Field() {
	delete[] this->data;
}

//Save functions
void saveBabySkyrmeField(const char * filename);
template <class T>
void BabySkyrmeField<T>::save(const char * filename) {
ofstream savefile(filename);
// 1st line save size axb and spacing dx,dy
savefile << this->width << " " << this->height << "\n";
// 2nd line read save parameters
savefile << this->mu << " " << this->mpi << "\n";
// save data
for(int i = 0; i < this->width ; i++){
for(int j = 0; j < this->height ; j++){
    float3 outputbuffer = this->get(i, j);
    savefile << i << " " << j << " " << outputbuffer.f[0] << " " << outputbuffer.f[1] << " " << outputbuffer.f[2] << "\n";
}}
cout << "Baby Skyrme Field output to file " << filename << "\n";
}

template <class T>
void SkyrmeField<T>::save(const char * filename) {
ofstream savefile(filename);
// 1st line save size axbxc and spacing dx,dy,dz
savefile << this->width << " " << this->height << " " << this->depth << " " << this->spacing[0] << " " << this->spacing[1] << " " << this->spacing[2] << "\n";
// 2nd line read save parameters
savefile << this->Fpi << " " << this->epi << " " << this->mpi << "\n";
// save data
for(int i = 0; i < this->width ; i++){
for(int j = 0; j < this->height ; j++){
for(int k = 0; k < this->depth ; k++){
    float4 outputbuffer = this->get(i, j, k);
    savefile << i << " " << j << " " << k << " " << outputbuffer.f[0] << " " << outputbuffer.f[1] << " " << outputbuffer.f[2] << " " << outputbuffer.f[3] << "\n";
}}}
cout << "Skyrme Field output to file " << filename << "\n";
}


Visualization * displayVisualization(BaseField * d, float level, float window);
template <class T>
Visualization * Field<T>::display() {
    return displayVisualization(this, defaultLevel, defaultWindow);
}

template <class T>
Visualization * Field<T>::display(float level, float window) {
    return displayVisualization(this, level, window);
}

template <class T>
Visualization * SkyrmeField<T>::display() {
    return displayVisualization(this, this->defaultLevel, this->defaultWindow);
}
template <class T>
Visualization * SkyrmeField<T>::display(float level, float window) {
    return displayVisualization(this, level, window);
}
Visualization * displayVolumeVisualization(BaseField * d, int slice, slice_plane direction, float level, float window);
template <class T>
Visualization * SkyrmeField<T>::display(int slice, slice_plane direction) {
    return displayVolumeVisualization(this, slice, direction, this->defaultLevel, this->defaultWindow);
}

template <class T>
Visualization * SkyrmeField<T>::display(int slice, slice_plane direction, float level, float window) {
    return displayVolumeVisualization(this, slice, direction, level, window);
}
Visualization * displayMIPVisualization(BaseField * d, slice_plane direction, float level, float window);
template <class T>
Visualization * SkyrmeField<T>::displayMIP() {
    return displayMIPVisualization(this, X, this->defaultLevel, this->defaultWindow);
}
template <class T>
Visualization * SkyrmeField<T>::displayMIP(float level, float window) {
    return displayMIPVisualization(this, X, level, window);
}
template <class T>
Visualization * SkyrmeField<T>::displayMIP(slice_plane direction, float level, float window) {
    return displayMIPVisualization(this, direction, level, window);
}
template <class T>
Visualization * SkyrmeField<T>::displayMIP(slice_plane direction) {
    return displayMIPVisualization(this, direction, this->defaultLevel, this->defaultWindow);
}

#ifdef USE_GTK
struct _saveData {
	GtkWidget * fs;
	Visualization * viz;
};
#endif

template <class T>
std::string Field<T>::getAttribute(std::string str) {
    return this->attributes[str];
}

template <class T>
void Field<T>::setAttribute(std::string key, std::string value) {
    this->attributes[key] = value;
}

template <class T>
int Field<T>::getWidth() const {
    return this->width;
}

template <class T>
int Field<T>::getHeight() const {
    return this->height;
}

template <class T>
int SkyrmeField<T>::getDepth() const {
    return this->depth;
}

template <class T>
int3 BabySkyrmeField<T>::getSize() const {
    int3 size;
    size.x = this->width;
    size.y = this->height;
    size.z = 0;
    return size;
}
template <class T>
int3 SkyrmeField<T>::getSize() const {
    int3 size;
    size.x = this->width;
    size.y = this->height;
    size.z = this->depth;
    return size;
}

template <class T>
void Field<T>::setData(T * data) {
    this->data = data;
}

template <class T>
const T * Field<T>::getData() {
    return this->data;
}

template <class T>
bool BabySkyrmeField<T>::inBounds(int i) const {
    return i >= 0 && i < this->getTotalSize();
}

template <class T>
bool SkyrmeField<T>::inBounds(int i) const {
    return i >= 0 && i < this->getTotalSize();
}

template <class T>
bool BabySkyrmeField<T>::inBounds(int x, int y) const {
    return x >= 0 && x < this->width && y >= 0 && y < this->height;
}

template <class T>
bool SkyrmeField<T>::inBounds(int x, int y, int z) const {
    return x >= 0 && x < this->width 
        && y >= 0 && y < this->height 
        && z >= 0 && z < this->depth;
}

template <class T>
bool SkyrmeField<T>::inBounds(int3 pos) const {
    return pos.x >= 0 && pos.x < this->width 
        && pos.y >= 0 && pos.y < this->height 
        && pos.z >= 0 && pos.z < this->depth;
}


template <class T>
int BabySkyrmeField<T>::getTotalSize() const {
    return this->width*this->height;
}

template <class T>
int SkyrmeField<T>::getTotalSize() const {
    return this->width*this->height*this->depth;
}

template <class T>
void BabySkyrmeField<T>::set(int x, int y, T value) {
    if(!this->inBounds(x,y))
        throw OutOfBoundsException(x, y, this->width, this->height, __LINE__, __FILE__);
    this->data[x+y*this->width] = value;
}

template <class T>
void BabySkyrmeField<T>::set(int2 pos, T value) {
    this->set(pos.x, pos.y, value);
}

template <class T>
void BabySkyrmeField<T>::set(int i, T value) {
    if(!this->inBounds(i))
        throw OutOfBoundsException(i, this->width*this->height, __LINE__, __FILE__);
    this->data[i] = value;
}

template <class T>
void SkyrmeField<T>::set(int i, T value) {
    if(!this->inBounds(i))
        throw OutOfBoundsException(i, this->width*this->height*this->depth, __LINE__, __FILE__);
    this->data[i] = value;
}

template <class T>
T BabySkyrmeField<T>::get(int x, int y) const {
    if(!this->inBounds(x,y))
        throw OutOfBoundsException(x, y, this->width, this->height, __LINE__, __FILE__);
    return this->data[x+y*this->width];
}

template <class T>
T BabySkyrmeField<T>::get(int2 pos) const {
    return this->get(pos.x, pos.y);
}

template <class T>
T BabySkyrmeField<T>::get(int i) const {
    if(!this->inBounds(i))
        throw OutOfBoundsException(i, this->width*this->height, __LINE__, __FILE__);
    return this->data[i];
}

template <class T>
T * BabySkyrmeField<T>::get(Region r) const {
    T * res = new T[r.size.x*r.size.y];
    int counter = 0;
    for(int y = r.offset.y; y < r.size.y; y++) {
    for(int x = r.offset.x; x < r.size.x; x++) {
        res[counter] = this->get(x,y);
    }}
    return res;
}

template <class T>
void BabySkyrmeField<T>::set(Region r, T value) {
    for(int y = r.offset.y; y < r.size.y; y++) {
    for(int x = r.offset.x; x < r.size.x; x++) {
        this->set(x,y,value);
    }}
}

template <class T>
BabySkyrmeField<T> BabySkyrmeField<T>::crop(Region r) const {
    BabySkyrmeField<T> * res = new BabySkyrmeField<T>(r.size);
    res->setData(this->get(r));
}

template <class T>
T SkyrmeField<T>::get(int i) const {
    if(!this->inBounds(i))
        throw OutOfBoundsException(i, this->width*this->height*this->depth, __LINE__, __FILE__);
    return this->data[i];
}

template <class T>
void SkyrmeField<T>::set(int x, int y, int z, T value) {
    if(!this->inBounds(x,y,z))
        throw OutOfBoundsException(x, y, z, this->width, this->height, this->depth, __LINE__, __FILE__);
    this->data[x+y*this->width+z*this->width*this->height] = value;
}

template <class T>
void SkyrmeField<T>::set(int3 pos, T value) {
    this->set(pos.x, pos.y, pos.z, value);
}

template <class T>
T SkyrmeField<T>::get(int x, int y, int z) const {
    if(!this->inBounds(x,y,z))
        throw OutOfBoundsException(x, y, z, this->width, this->height, this->depth, __LINE__, __FILE__);
    return this->data[x+y*this->width+z*this->width*this->height];
}

template <class T>
T SkyrmeField<T>::get(int3 pos) const {
    return this->get(pos.x, pos.y, pos.z);
}

template <class T>
T * SkyrmeField<T>::get(Region r) const {
    T * res = new T[r.size.x*r.size.y*r.size.z];
    int counter = 0;
    for(int z = r.offset.z; z < r.offset.z+r.size.z; z++) {
    for(int y = r.offset.y; y < r.offset.y+r.size.y; y++) {
    for(int x = r.offset.x; x < r.offset.x+r.size.x; x++) {
        res[counter] = this->get(x,y,z);
        counter++;
    }}}
    return res;
}

template <class T>
void SkyrmeField<T>::set(Region r, T value) {
    for(int z = r.offset.z; z < r.offset.z+r.size.z; z++) {
    for(int y = r.offset.y; y < r.offset.y+r.size.y; y++) {
    for(int x = r.offset.x; x < r.offset.x+r.size.x; x++) {
        this->set(x,y,z,value);
    }}}
}

template <class T>
SkyrmeField<T> * SkyrmeField<T>::crop(Region r) const {
    SkyrmeField<T> * res = new SkyrmeField<T>(r.size);
    res->setData(this->get(r));
    return res;
}

template <class T>
void Field<T>::fill(T value) {
    for(int i = 0; i < getTotalSize(); i++)
        data[i] = value;
}

template <class T>
float3 SkyrmeField<T>::getSpacing() const {
	return this->spacing;
}

template <class T>
float2 BabySkyrmeField<T>::getSpacing() const {
	return this->spacing;
}

template <class T>
void SkyrmeField<T>::setSpacing(float3 spacing) {
	this->spacing = spacing;
}

template <class T>
void BabySkyrmeField<T>::setSpacing(float2 spacing) {
	this->spacing = spacing;
}

template <class T>
float * Field<T>::getFloatData() const {
    float * floatData = new float[this->getTotalSize()];
#pragma omp parallel for
    for(int i = 0; i < this->getTotalSize(); i++) {
        floatData[i] = (float)toSingleValue(this->data[i]);
    }
    return floatData;
}

template <class T>
float Field<T>::getFloatData(int3 pos) const {
    return toSingleValue(this->data[pos.x+pos.y*this->width+pos.z*this->width*this->height]);
}

template <class T>
float3 Field<T>::getVectorFloatData(int3 pos) const {
    return toVectorData(this->data[pos.x+pos.y*this->width+pos.z*this->width*this->height]);
}

template <class T>
float3 * Field<T>::getVectorFloatData() const {
    float3 * floatData = new float3[this->getTotalSize()];
#pragma omp parallel for
    for(int i = 0; i < this->getTotalSize(); i++) {
        floatData[i] = toVectorData(this->data[i]);
    }
    return floatData;
}

template <class T>
float3 Field<T>::getDerivative(int direction, int x, int y){
	if(direction == 0){
		return (-data[(x+2)+y*this->width] + 8.0*data[(x+1)+y*this->width] - 8.0*data[(x-1)+y*this->width] + data[(x-2)+y*this->width])/(12.0*spacing[0]);
	}
	else
	{
		return (-data[x+(y+2)*this->width] + 8.0*data[x+(y+1)*this->width] - 8.0*data[x+(y-1)*this->width] + data[x+(y-2)*this->width])/(12.0*spacing[1]);
	}
}

template <class T>
float3 Field<T>::getDerivative(int direction, int2 pos){
	return getDerivative(direction, pos.x, pos.y);
}

template <class T>
float4 Field<T>::getDerivative(int direction, int x, int y, int z){
	if(direction == 0){
		return (-data[(x+2)+y*this->width+z*this->width*this->height] + 8.0*data[(x+1)+y*this->width+z*this->width*this->height] - 8.0*data[(x-1)+y*this->width+z*this->width*this->height] + data[(x-2)+y*this->width+z*this->width*this->height])/(12.0*spacing[0]);
	}
	else if(direction == 1)
	{
		return (-data[x+(y+2)*this->width+z*this->width*this->height] + 8.0*data[x+(y+1)*this->width+z*this->width*this->height] - 8.0*data[x+(y-1)*this->width+z*this->width*this->height] + data[x+(y-2)*this->width+z*this->width*this->height])/(12.0*spacing[1]);
	}
	else
	{
		return (-data[x+y*this->width+(z+2)*this->width*this->height] + 8.0*data[x+y*this->width+(z+1)*this->width*this->height] - 8.0*data[x+y*this->width+(z-1)*this->width*this->height] + data[x+y*this->width+(z-2)*this->width*this->height])/(12.0*spacing[2]);
	}
}

template <class T>
float4 Field<T>::getDerivative(int direction, int3 pos){
	return getDerivative(direction, pos.x, pos.y, pos.z);
}

template <class T>
float4 SkyrmeField::getDerivative
/// ETC> ETC>



}

 // End FTPL namespace
#endif /* FTPL_H_ */
