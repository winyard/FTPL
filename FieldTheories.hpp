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

class BaseField {
    public:

};
/***********************/
/*       FIELDS        */
/***********************/
// TOO ADD -> taking regions etc.
template <class T>
class Field {
    public:
	int dim;
        Field(int d, ...);
	Field(int d, vector<int> size);
	Field(int d,T value, vector<int> size);
        ~Field();
        const T * getData();
        const T getData(...);
        const T getData(vector<int> pos);
        vector<int> getSize();
        int getTotalSize();
        void setData(T * datain);
	void setData(T, ...);
	void setData(T, vector<int> pos);
        void fill(T value);
    protected:
        // T * data;
	//int * size;
        vector<T> data;
	vector<int>  size;
	

};

/* --- Constructors & Destructors --- */
template <class T>
Field<T>::Field(int d, ...): dim(d) {
    dim = d;
    size = new int[d];
    va_list ap;
    va_start(ap, d);
    for(int i=0; i<d; i++){
    	this->size[i] = va_arg(ap,int);
    }
    this->data = new T[this->getTotalSize()];
    this->clearField;
}
Eigen::ArrayXXi

template <class T>
Field<T>::Field(int d, vector<int> sizein): dim(d), size(sizein)  {
    data.resize(getTotalSize());
    clearField;
}

template <class T>
Field<T>::Field(int d,T value, Eigen::VectorXd<int> size) {
    this->dim = d;
    this->size = new int[d];
    for(int i=0; i<d; i++){
    	this->size[i] = size(i+1);
    }
    this->data = new T[this->getTotalSize()];
    this->fill(value);
}

template <class T>
Field<T>::Field(int d,T value, ...) {
    this->dim = d;
    this->size = new int[d];
    va_list ap;
    va_start(ap, d);
    for(int i=0; i<d; i++){
    	this->size[i] = va_arg(ap,int);
    }
    this->data = new T[this->getTotalSize()];
    this->fill(value);
}

template <class T>
Field<T>::~Field() {
	delete[] this->data;
	delete[] this->size;
}

/* --- Fetch & Set --- */

template <class T>
const T * Field<T>::getData() {
    return this->data;
}

template <class T>
const T * Field<T>::getData(...) {
    va_list ap;
    va_start(ap, this->dim);
    int point=0;
    for(int i=0; i<this->dim; i++){
    	int localpoint = va_arg(ap,int);
	for(int j=0; j<i; j++){
		localpoint *= size[j];
	}
    	point += localpoint;	
    }
    return this->data[point];
}

template <class T>
const T * Field<T>::getData(Eigen::VectorXd<int> pos) {
    int point=0;
    for(int i=0; i<this->dim; i++){
    	int localpoint = pos[i];
	for(int j=0; j<i; j++){
		localpoint *= size[j];
	}
    	point += localpoint;	
    }
    return this->data[point];
}


template <class T>
void Field<T>::setData(T * datain){
    this->data = datain;
}

template <class T>
void Field<T>::setData(T value, ...){
    va_list ap;
    va_start(ap, this->dim);
    int point=0;
    for(int i=0; i<this->dim; i++){
    	int localpoint = va_arg(ap,int);
	for(int j=0; j<i; j++){
		localpoint *= size[j];
	}
    	point += localpoint;	
    }
    this->data[point] = value;
}
        
template <class T>
void Field<T>::setData(T value, Eigen::VectorXd<int> pos){
    int point=0;
    for(int i=0; i<this->dim; i++){
    	int localpoint = pos[i];
	for(int j=0; j<i; j++){
		localpoint *= size[j];
	}
    	point += localpoint;	
    }
    this->data[point] = value;
}

template <class T>
void Field<T>::fill(T value){
    for(int i=0; i<this->getTotalSize(); i++){
    this->data[i] = value;
    }
}

template <class T>
int * Field<T>::getSize(){
    return this->size;
}

template <class T>
int Field<T>::getTotalSize(){
    int total = size[0];
    for(int i=1; i<this->dim; i++){
	total *= size[i];
    }
    return total;
}

/***********************/
/*   Field Theories    */
/***********************/

class BaseFieldTheory {
   public:
	int dim;
	BaseFieldTheory(); // NOTE - the derived class must set the dimensions before allowing the constrcutor to be called!
	~BaseFieldTheory(); // Deletes all data including fields etc. so only "new's" that have allocated data (which you shouldnt be doing) need to be deleted in derived destructor
	save(); // will save and load all fields and parameters.
	load();
	plot(); // will plot fields or energies on a flat spatial grid
	spaceTransformation(); // tranform the physical space using spaitial rotations and scalings etc.
	fieldTransformation(); // transform a target space by some class T (normally some matrix or something)
	setBoundaryType(); // set the type of boundary in each direction 0-fixed(size = bdw), 1-dirichlet (dx = 0), 2-periodic 
	void updateEnergy(); // cycles through with correct boundary condtions and calculates the energy at each point from the derived classes energy functional
	void gradientFlow(int iterations, T functionto be run after each iteration); // cycle through with correct boundary condtions and updates each point based on the derived classes energy gradient calculation
	virtual float calculateEnergy(...); // need to be overriden by derived class with correct pointwise energy calculation
	template <class T>
	virtual T single_derivative(Field<T> * f, int wrt, ...);
	template <class T>
	virtual T single_derivative(Field<T> * f, int wrt, Eigen::VectorXd<int> pos);
	template <class T>
	virtual T double_derivative(Field<T> * f, int wrt1, int wrt2, ...);
	template <class T>
	virtual T double_derivative(Field<T> * f, int wrt1, int wrt2, Eigen::VectorXd<int> pos);
   protected:
	template <class T>
	Field * createField(T type);
	Field ** fields; // pointer towards the fields of the theory so they can be interacted with (but need to know how many fields we are dealing with!)
	int No_fields;
	int * bdw; // number of boundary points that are never updated or contribute to energies etc. in each direction
	int * boundarytype; // indicates the type of boundary in each direction 0-fixed(size = bdw), 1-dirichlet (dx = 0), 2-periodic 
	float * energydensity;
	float energy;
	float * spacing;
	int * size;
};

/* --- Constructors & Destructors --- */
BaseFieldTheory::BaseFieldTheory(int d, ...){
	this->dim = d;
	this->size = new int[d];
	va_list ap;
    	va_start(ap, d);
   	for(int i=0; i<d; i++){
    	size[i] = va_arg(ap,int);
    	}
	BaseFieldTheory();
};



BaseFieldTheory::BaseFieldTheory(int d, int * size){
	this->dim = d;
	this->size = new int[d];
   	for(int i=0; i<d; i++){
    	size[i] = size(i+1);
    	}
	BaseFieldTheory();
};

BaseFieldTheory::BaseFieldTheory(){
	// this should be run once dim and size are set!
	this->No_fields = 0;
	this->energydensity = new float[this->getTotalSize()];
	this->energy = -1.0;
	// set deault spacing of dx = 1	
	this->spacing = new float[this->dim];
	for(int i =0; i<this->dim; i++){
	spacing[i] = 1.0;
	}
	//set default boundary conditions (standard 2 layer static boundary)
	this->bdw = new int[2*this->dim];
	this->boundarytype = new int[2*this->dim];
	for(int i =0; i<this->dim; i++){
	this->bdw[2*i] = 2; // set the +ve boundary
	this->bdw[2*i+1] = 2; // set the -ve boundary
	this->boundarytype[2*i] = 2; // set the +ve boundary
	this->boundarytype[2*i+1] = 2; // set the -ve boundary
	};
	
};

template <class T>
Field<T> * BaseFieldTheory::createField(T type){
		
	No_fields += 1;
	




};



BaseFieldTheory::~BaseFieldTheory(){
	delete[] size;
	delete[] energydensity;
	delete[] spacing;
};

/* --- Derivatives --- */

template <class T>
T BaseFieldTheory::single_derivative(Field<T> * f, int wrt, Eigen::VectorXd<int> pos) {
	Eigen::VectorXd<int> dir(this->dim);
	for(int i = 1; i <= this->dim; i++){ if(i == wrt){dir(i) = 1;}else{dir(i) = 0;}};
	return (-1.0*f.getData(pos+2*dir) + 8.0*f.getData(pos+dir) - 8.0*f.getData(pos-dir) + f.getData(pos-2*dir))/(12.0*this->spacing[wrt]);
}

template <class T>
T BaseFieldTheory::single_derivative(Field<T> * f, int wrt, ...) {
	Eigen::VectorXd<int> pos(this->dim);
   	va_list ap;
    	va_start(ap, this->dim);
   	for(int i=1; i<=d; i++){
    	pos[i] = va_arg(ap,int);
    	}
	return this->single_derivative(f, wrt, pos);
}



template <class T>
T BaseFieldTheory::double_derivative(Field<T> * f, int wrt1, int wrt2, Eigen::VectorXd<int> pos) {
	if(wr1 == wrt2)
	{
	Eigen::VectorXd<int> dir(this->dim);
	for(int i = 1; i <= this->dim; i++){ if(i == wrt){dir(i) = 1;}else{dir(i) = 0;}};
	return (-1.0*f.getData(pos+2*dir) + 16.0*f.getData(pos+dir) - 30.0*f.getData(pos) + 16.0*f.getData(pos-dir) - f.getData(pos-2*dir))/(12.0*this->spacing[wrt1]*this->spacing[wrt1]);
	}
	else
	{
	Eigen::VectorXd<int> dir1(this->dim);
	Eigen::VectorXd<int> dir2(this->dim);
	for(int i = 1; i <= this->dim; i++){ 
		if(i == wrt1){dir1(i) = 1;}else{dir1(i) = 0;}
		if(i == wrt2){dir2(i) = 1;}else{dir2(i) = 0;}
	};
	return (f.getData(pos+2*dir1+2*dir2) - 8.0*f.getData(pos+dir1+2*dir2) + 8.0*f.getData(pos-dir1+2*dir2) - f.getData(pos-2*dir1+2*dir2) - 8.0*f.getData(pos+2*dir1+dir2) +64.0*f.getData(pos+dir1+dir2) -64.0*f.getData(pos-dir1+dir2) + 8.0* f.getData(pos-2*dir1+dir2) + 8.0*f.getData(pos+2*dir1-dir2) - 64.0*f.getData(pos+dir1-dir2)+64.0*f.getData(pos-dir1-dir2) - 8.0*f.getData(pos-2*dir1-dir2) - f.getData(pos+2*dir1-2*dir2) + 8.0*f.getData(pos+dir1-2*dir2) - 8.0*f.getData(pos-dir1-2*dir2) + f.getData(pos-2*dir1-2*dir2))/(144.0*this->spacing[wrt1]*this->spacing[wrt2]);

	}
}

template <class T>
T BaseFieldTheory::double_derivative(Field<T> * f, int wrt1, int wrt2, ...) {
	Eigen::VectorXd<int> pos(this->dim);
   	va_list ap;
    	va_start(ap, this->dim);
   	for(int i=1; i<=d; i++){
    	pos[i] = va_arg(ap,int);
    	}
	return this->double_derivative(f, wrt1, wrt2, pos);
}

//energy stuff (some is virtual and needs to be overwritten but if not will spit out an error message)

void BaseFieldTheory::updateEnergy(){
	float sum = 0.0;
	for(int i = 0; i < this->size[0]; i++){
	for(int j = 0; j < this->size[1]; j++){
		float buffer = this->calculateEnergy(i,j);
		this->energyDensity[i+j*size[0]]=buffer;
		sum += buffer;
	}}
	this->energy = sum*spacing[0]*spacing[1];
};

virtual float BaseFieldTheory::calculateEnergy(...)
{
cout << "ERROR! - either the incorrect number of parameters was entered into calculateEnergy or the calculateEnergy has not been set in the derived Field Theory class!\n";
return -1.0;
}



}

 // End FTPL namespace
#endif /* FTPL_H_ */
