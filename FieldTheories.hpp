/**
 * Field Theory Processing Library
 * Copyright Thomas Winyard 2016
 */

#ifndef FTPL_H_
#define FTPL_H_

#define _USE_MATH_DEFINES // windows crap

#include <cmath>

#include "Exceptions.hpp"
#include <fstream>
#include <typeinfo>
#include <string>
#include <stdlib.h>
#include <map>
#include <iostream>
#include <vector>
#include <memory>
#include <Eigen/Dense>


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
		Field(int d, vector<int> size);
        ~Field();
		T operator()(...);
        const vector<T> getData();
        const T getData(vector<int> pos);
        vector<int> getSize();
        int getSize(int com);
        int getTotalSize();
        void setData(vector<T> datain);
		void setData(T, vector<int> pos);
        void fill(T value);
		void save(ofstream output);
		void load(ifstream input);
		void normalise();
        void update();
        void resize(vector<int> sizein);
    protected:
        vector<T> data;
		vector<int>  size;
        vector<T> buffer;
};

/* --- Constructors & Destructors --- */

template <class T>
Field<T>::Field(int d, vector<int> sizein): dim(d), size(sizein)  {
    data.resize(getTotalSize());
    buffer.resize(getTotalSize());
}

/* --- Fetch & Set --- */

template <class T>
const vector<T> Field<T>::getData() {
    return data;
}


template <class T>
T Field<T>::operator()(...){ // can be optimised better!
    va_list ap;
    va_start(ap, dim);
    int point=0;
    int localpoint;
    for(int i=0; i<dim; i++){
        localpoint = va_arg(ap,int);
        for(int j=0; j<i; j++){
            localpoint *= size[j];
        }
        point += localpoint;
    }
    return data[point];
}

template <class T>
const T Field<T>::getData(vector<int> pos) {// can be optimised better! - similar to above method
    int point=0;
    for(int i=0; i<dim; i++){
    	int localpoint = pos[i];
	for(int j=0; j<i; j++){
		localpoint *= size[j];
	}
    	point += localpoint;	
    }
    return data[point];
}


template <class T>
void Field<T>::setData(vector<T> datain){
    data = datain;
}

template <class T>
void Field<T>::setData(T value, vector<int> pos){
    int point=0;
    for(int i=0; i<dim; i++){
    	int localpoint = pos[i];
	for(int j=0; j<i; j++){
		localpoint *= size[j];
	}
    	point += localpoint;	
    }
    data[point] = value;
}

template <class T>
void Field<T>::fill(T value){
    for(int i=0; i<getTotalSize(); i++){
    data[i] = value;
    }
}

template <class T>
vector<int> Field<T>::getSize(){
    return size;
}

template <class T>
int Field<T>::getSize(int com){
	return size[com];
}

template <class T>
int Field<T>::getTotalSize(){
    int total = size[0];
    for(int i=1; i<dim; i++){
	total *= size[i];
    }
    return total;
}

template <class T>
void Field<T>::load(ifstream input){ // may need to be altered for various spaces! - not sure!
	for(int i=0; i < getTotalSize(); i++){
		input >> data[i];
	}
}

template <class T>
void Field<T>::save(ofstream output){ // may need to be altered for various spaces! - not sure!
	for(int i=0; i < getTotalSize(); i++){
		output << data[i] << "\n";
	}
}

template <class T>
void Field<T>::normalise(){
	for(int i=0; i < getTotalSize(); i++){
		data[i] = data[i].normalise();
	}
}

template <class T>
void Field<T>::update() { // moves the buffer values to the data
    data = buffer;
}


template <class T>
void Field<T>::resize(vector<int> sizein) { // resize the fields for different accuracy
    if(size.size() == sizein.size()) {
        size = sizein;
        data.resize(getTotalSize());
        buffer.resize(getTotalSize());
    }
    else{
        cout << "Tried to resize field with incorrect size of dimension vector!\n";
    }
}

/***********************/
/*    Target Space     */
/***********************/

class TargetSpace {
    public:
        TargetSpace();
        ~TargetSpace();
        template<class T>
        inline Field<T> * field(int i);
        template<class T>
        Field<T> * addField(int dim, vector<int> size, string target);
        void resize(vector<int> sizein);
        void load_fields(ifstream loadfile);
        void save_fields(ofstream savefile);
    private:
        int no_fields;
        // Add aditional Field types as they are created here!
        std::vector<Field<Eigen::VectorXd>> fields1;
        std::vector<Field<double>> fields2;
        std::vector<Field<int>> fields3;
        std::vector<Field<Eigen::ArrayXXf>> fields4;
};


TargetSpace::TargetSpace(){
    no_fields = 0;
}

    template<class T>
    Field<T> * TargetSpace::field(int i){
        if(i <= fields1.size()-1){
            return  fields1[i];
        }
        else if(i <= fields1.size() + fields2.size() - 1){
            return fields2[i-fields1.size()];
        }
        else if(i <= fields1.size() + fields2.size() +fields3.size() -1){
            return fields3[i-fields1.size()-fields2.size()];
        }
        else{
            return fields4[i-fields1.size()-fields2.size()-fields3.size() - 1];
        }
    }

void TargetSpace::resize(vector<int> sizein){
    for(int i = 0; i < no_fields; i++){
        field(i)->resize(sizein);
    }
}

Field<T> * TargetSpace::addField(int dim, vector<int> size, string target){
    if(target == "Eigen::VectorXd"){fields1.push_back(Field<Eigen::VectorXd>(dim, size));return fields1[fields1.size()-1];}
    else if(target == "double"){fields2.push_back(Field<double>(dim, size));return fields2[fields2.size()-1];}
    else if(target == "int"){fields3.push_back(Field<int>(dim, size));return fields3[fields3.size()-1];}
    else if(target == "Eigen::ArrayXXd"){fields4.push_back(Field<Eigen::ArrayXXf>(dim, size));return fields4[fields4.size()-1];}
    else{cout << "The container TargetSpace does not contain a vector for the type " << target
              << " please create one and edit the class to loop through it correctly!\n";}
    no_fields = no_fields + 1;
}




    void TargetSpace::save_fields(ofstream savefile){
        for(int i = 0; i < no_fields; i++){
            field(i).save(savefile);
        }
    }

    void TargetSpace::load_fields(ifstream loadfile){
        for(int i = 0; i < no_fields; i++){
            field(i).load(loadfile);
        }
    }
/***********************/
/*   Field Theories    */
/***********************/

class BaseFieldTheory {
   public:
	int dim;
	BaseFieldTheory(int d, vector<int> size);
	~BaseFieldTheory(); // Deletes all data including fields etc. so only "new's" that have allocated data (which you shouldnt be doing) need to be deleted in derived destructor
	void save(const char * savepath); // will save and load all fields and parameters.
	void load(const char * loadpath);
	void plot(const char * plotpath); // will plot fields or energies on a flat spatial grid
	void spaceTransformation(); // tranform the physical space using spaitial rotations and scalings etc.
    template <class T>
	void fieldTransformation(T tranformation); // transform a target space by some class T (normally some matrix or something)
	void setBoundaryType(vector<int> boundaryin); // set the type of boundary in each direction 0-fixed(size = bdw), 1-dirichlet (dx = 0), 2-periodic
	void updateEnergy(); // cycles through with correct boundary condtions and calculates the energy at each point from the derived classes energy functional
	void gradientFlow(int iterations, void (*correct_func)() ); // cycle through with correct boundary condtions and updates each point based on the derived classes energy gradient calculation
    virtual void calculateGradientFlow(...);
    virtual double calculateEnergy(...); // need to be overriden by derived class with correct pointwise energy calculation
    bool inBoundary(...);
    int getTotalSize();
	template <class T>
	T single_derivative(Field<T> * f, int wrt, ...);
	template <class T>
	T single_derivative(Field<T> * f, int wrt, vector<int> pos);
	template <class T>
	T double_derivative(Field<T> * f, int wrt1, int wrt2, ...);
	template <class T>
	T double_derivative(Field<T> * f, int wrt1, int wrt2, vector<int> pos);
   protected:
	template <class T>
	Field<T> * createField(T type);
	//vector<unique_ptr<Field>> fields;
    TargetSpace fields;
	vector<int> bdw; // number of boundary points that are never updated or contribute to energies etc. in each direction
	vector<int> boundarytype; // indicates the type of boundary in each direction 0-fixed(size = bdw), 1-dirichlet (dx = 0), 2-periodic
	vector<double> energydensity;
	double energy;
	vector<double> spacing;
	vector<int> size;
};

/* --- Constructors & Destructors --- */
BaseFieldTheory::BaseFieldTheory(int d, vector<int> sizein): dim(d), size(sizein){
        energydensity.resize(getTotalSize());
        energy = -1.0;
        // set deault spacing of dx = 1
        spacing.resize(dim);
        for(int i =0; i<dim; i++){
            spacing[i] = 1.0;
        }
        //set default boundary conditions (standard 2 layer static boundary)
        bdw.resize(2*dim);
        boundarytype.resize(2*dim);
        for(int i =0; i<dim; i++){
            bdw[2*i] = 2; // set the +ve boundary
            bdw[2*i+1] = 2; // set the -ve boundary
            boundarytype[2*i] = 0; // set the +ve boundary
            boundarytype[2*i+1] = 0; // set the -ve boundary
        };
};

template <class T>
Field<T> * BaseFieldTheory::createField(T type){
    return fields.addField(dim, size, type);
};

int BaseFieldTheory::getTotalSize(){
    int total = size[0];
    for(int i=1; i<dim; i++){
        total *= size[i];
    }
    return total;
}

/* --- Derivatives --- */

template <class T>
T BaseFieldTheory::single_derivative(Field<T> * f, int wrt, Eigen::VectorXd<int> pos) {
    Eigen::VectorXd dir(dim);
	for(int i = 0; i < dim; i++){ if(i == wrt){dir[i] = 1;}else{dir[i] = 0;}};
	return (-1.0*f.getData(pos+2*dir) + 8.0*f.getData(pos+dir) - 8.0*f.getData(pos-dir) + f.getData(pos-2*dir))/(12.0*spacing[wrt]);
}

template <class T>
T BaseFieldTheory::single_derivative(const Field<T> f, int wrt, ...) {
    Eigen::VectorXd<int> pos(dim);
   	va_list ap;
    	va_start(ap, dim);
   	for(int i=0; i<dim; i++){
    	pos[i] = va_arg(ap,int);
    	}
	return single_derivative(f, wrt, pos);
}



template <class T>
T BaseFieldTheory::double_derivative(const Field<T> f, int wrt1, int wrt2, Eigen::VectorXd<int> pos) {
	if(wr1 == wrt2)
	{
        Eigen::VectorXd<int> dir(dim);
	    for(int i = 0; i <= dim; i++){ if(i == wrt){dir[i] = 1;}else{dir[i] = 0;}};
	    return (-1.0*f.getData(pos+2*dir) + 16.0*f.getData(pos+dir) - 30.0*f.getData(pos) + 16.0*f.getData(pos-dir) - f.getData(pos-2*dir))/(12.0*this->spacing[wrt1]*this->spacing[wrt1]);
	}
	else
	{
	    vector<int> dir1(dim);
	    vector<int> dir2(dim);
	    for(int i = 0; i < dim; i++){
		    if(i == wrt1){dir1[i] = 1;}else{dir1[i] = 0;}
		    if(i == wrt2){dir2[i] = 1;}else{dir2[i] = 0;}
	    };
	    return (f.getData(pos+2*dir1+2*dir2) - 8.0*f.getData(pos+dir1+2*dir2) + 8.0*f.getData(pos-dir1+2*dir2) - f.getData(pos-2*dir1+2*dir2) - 8.0*f.getData(pos+2*dir1+dir2) +64.0*f.getData(pos+dir1+dir2) -64.0*f.getData(pos-dir1+dir2) + 8.0* f.getData(pos-2*dir1+dir2) + 8.0*f.getData(pos+2*dir1-dir2) - 64.0*f.getData(pos+dir1-dir2)+64.0*f.getData(pos-dir1-dir2) - 8.0*f.getData(pos-2*dir1-dir2) - f.getData(pos+2*dir1-2*dir2) + 8.0*f.getData(pos+dir1-2*dir2) - 8.0*f.getData(pos-dir1-2*dir2) + f.getData(pos-2*dir1-2*dir2))/(144.0*this->spacing[wrt1]*this->spacing[wrt2]);

	}
}

template <class T>
T BaseFieldTheory::double_derivative(const Field<T> f, int wrt1, int wrt2, ...) {
    Eigen::VectorXd<int> pos(dim);
   	va_list ap;
    	va_start(ap, dim);
   	for(int i=0; i<dim; i++){
    	pos[i] = va_arg(ap,int);
    	}
	return double_derivative(f, wrt1, wrt2, pos);
}

//energy stuff (some is virtual and needs to be overwritten but if not will spit out an error message)

void BaseFieldTheory::updateEnergy(){ // only currently for 2-dim's!
	double sum = 0.0;
	for(int i = bdw; i < size[0]-bdw; i++){
	for(int j = bdw; j < size[1]-bdw; j++){
    if(inBoundary) {
        double buffer = calculateEnergy(i, j);
        energyDensity[i + j * size[0]] = buffer;
        sum += buffer;
    }}}
	energy = sum*spacing[0]*spacing[1];
};

virtual float BaseFieldTheory::calculateEnergy(...)
{
cout << "ERROR! - either the incorrect number of parameters was entered into calculateEnergy or the calculateEnergy has not been set in the derived Field Theory class!\n";
return -1.0;
}

void gradientFlow(int iterations, T functiontoberunaftereachiteration = NULL, int often=0){ // needs to be updated to gradient flow the entire field then update the field!
    for(int no = 0; no < iterationsl no++){
    for(int i = bdw; i < size[0]-bdw; i++){
    for(int j = bdw; j < size[1]-bdw; j++){
    if(inBoundary) {
        for(int f = 0; f < fields.size(); f++){
            fields[f](i,j) =  gradientflowcalculation(field_name,i,j);
        }
    }}}
    if(function != NULL && no%often == 0){
        field.function();
    }
    }
}


//Time Dependent versions of fields and fields theories
template <class T>
class timeDependentField: public Field<T>{
	public:
		//constructors etc. also
		T * getTimeDerivative();
		T getTimeDerivative(...);
		T getTimeDerivative(vector<int> pos);
		void setTimeDerivative();
		void setTimeDerivative(...);
		void setTimeDerivative(vector<int> pos);
		fillTimeDerivative(T value);
	private:
		T * timeDerivative;
};




class timeDependentFieldTheory: public BaseFieldTheory {
	public:
		//loads of extra stuff for calculating time derivatives etc. and on initilisation call the timeDependentFields rather than the standard fields!

	private:
};




}

 // End FTPL namespace
#endif /* FTPL_H_ */
