/**
 * Field Theory Processing Library
 * Copyright Thomas Winyard 2016
 */

#include <cmath>

#include "Exceptions.hpp"
#include <fstream>
#include <typeinfo>
#include <string>
#include <stdlib.h>
#include <map>
#include <iostream>
#include <vector>
#include <omp.h>
#include <memory>
#include <Eigen/Dense>
#include <stdarg.h>


using namespace std;

namespace FTPL {
/***********************/
/*       FIELDS        */
/***********************/
// TOO ADD -> taking regions etc.
template <class T>
class Field {
    public:
        vector<T> data;
        vector<T> buffer;
	    int dim;
        inline const T  __attribute__((always_inline)) getData(vector<int> pos);
        inline const T  __attribute__((always_inline)) getData(Eigen::VectorXi pos);
        inline void  __attribute__((always_inline)) setBuffer(T value, vector<int> pos);
		Field(int d, vector<int> size, bool isdynamic);
        ~Field(){};
		inline T operator()(...); // TO BE WRITTEN
        inline const vector<T> getData();
        vector<int> getSize();
        int getSize(int com);
        inline int getTotalSize();
        inline void setData(vector<T> datain);
		inline void setData(T value, vector<int> pos);
        inline void setData(T value, Eigen::VectorXi pos);
        void fill(T value);
		void save_field(ofstream& output);
		void load_field(ifstream& input);
		void normalise();
        void update_field();
        void update_gradient(double dt);
        void resize(vector<int> sizein);
        void progressTime(double time_step);
    protected:
		vector<int>  size;
        vector<T> dt;
        bool dynamic;
};

/* --- Constructors & Destructors --- */

template <class T>
Field<T>::Field(int d, vector<int> sizein, bool isdynamic): dim(d), size(sizein), dynamic(isdynamic)  {
    data.resize(getTotalSize());
    buffer.resize(getTotalSize());
    if(dynamic){
        dt.resize(getTotalSize());
    }
}

/* --- Fetch & Set --- */

template <class T>
inline const vector<T> Field<T>::getData() {
    return data;
}


template <class T>
inline T __attribute__((always_inline))  Field<T>::operator()(...){ // can be optimised better!
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
inline const __attribute__((always_inline))  T Field<T>::getData(vector<int> pos) {// can be optimised better! - similar to above method
    int point = pos[0];
    int multiplier = 1;
    for(int i=1; i<dim; i++){
        multiplier *= size[i-1];
    	point += pos[i]*multiplier;
    }
    return data[point];
}

    template <class T>
inline const __attribute__((always_inline))  T Field<T>::getData(Eigen::VectorXi pos) {// can be optimised better! - similar to above method
        int point = pos[0];
        int multiplier = 1;
        for(int i=1; i<dim; i++){
            multiplier *= size[i-1];
            point += pos[i]*multiplier;
        }
        return data[point];
    }



    template <class T>
inline void Field<T>::setData(vector<T> datain){
    data = datain;
}

template <class T>
inline void __attribute__((always_inline))  Field<T>::setData(T value, vector<int> pos){
    int point = pos[0];
    int multiplier = 1;
    for(int i=1; i<dim; i++){
        multiplier *= size[i-1];
        point += pos[i]*multiplier;
    }
    data[point] = value;
}

    template <class T>
inline void __attribute__((always_inline))  Field<T>::setBuffer(T value, vector<int> pos){
        int point = pos[0];
        int multiplier = 1;
        for(int i=1; i<dim; i++){
            multiplier *= size[i-1];
            point += pos[i]*multiplier;
        }
        buffer[point] = value;
    }

    template <class T>
inline void __attribute__((always_inline))  Field<T>::setData(T value, Eigen::VectorXi pos){
        int point = pos[0];
        int multiplier = 1;
        for(int i=1; i<dim; i++){
            multiplier *= size[i-1];
            point += pos[i]*multiplier;
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
inline int Field<T>::getTotalSize(){
    int total = size[0];
    for(int i=1; i<dim; i++){
	total *= size[i];
    }
    return total;
}

template <class T>
void Field<T>::load_field(ifstream& input){ // may need to be altered for various spaces! - not sure!
	for(int i=0; i < getTotalSize(); i++){
		    //input >> data[i];
        }
    if(dynamic){
        for(int i=0; i < getTotalSize(); i++){
            //input >> dt[i];
        }
    }
}

template <class T>
void Field<T>::save_field(ofstream& output){ // may need to be altered for various spaces! - not sure!
	for(int i=0; i < getTotalSize(); i++){
		output << data[i] << "\n";
	}
    if(dynamic){
        for(int i=0; i < getTotalSize(); i++){
            output << dt[i] << "\n";
        }
    }
}

template <class T>
void Field<T>::normalise(){
	for(int i=0; i < getTotalSize(); i++){
		data[i].normalize();
	}
    if(dynamic){
        for(int i=0; i < getTotalSize(); i++){
            dt[i] = dt[i] - (data[i].dot(dt[i]))*data[i];
        }
    }
}

template <class T>
void Field<T>::update_field() { // moves the buffer values to the data
    data = buffer;
}

    template <class T>
    void Field<T>::update_gradient(double dt) { // moves the buffer values to the data
        for(int i = 0; i < getTotalSize(); i++){
            data[i] += dt*buffer[i];
        }
    }


template <class T>
void Field<T>::resize(vector<int> sizein) { // resize the fields for different accuracy
    if(size.size() == sizein.size()) {
        size = sizein;
        data.resize(getTotalSize());
        buffer.resize(getTotalSize());
        if(dynamic){
            dt.resize(getTotalSize());
        }
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
        ~TargetSpace(){};
        template<class T>
        inline Field<T> * field(int i);
        Field<Eigen::VectorXd> * addField(int dim, vector<int> size, Field<Eigen::VectorXd> * target, bool isDynamic);
        Field<double> * addField(int dim, vector<int> size, Field<double> * target, bool isDynamic);
        Field<int> * addField(int dim, vector<int> size, Field<int> * target, bool isDynamic);
        Field<Eigen::MatrixXd> * addField(int dim, vector<int> size, Field<Eigen::MatrixXd> * target, bool isDynamic);
        void resize(vector<int> sizein);
        void load_fields(ifstream& loadfile);
        void save_fields(ofstream& savefile);
        void update_fields();
        void update_gradients(double dt);
    private:
        int no_fields;
        // Add aditional Field types as they are created here!
        std::vector<Field<Eigen::VectorXd>> fields1;
        std::vector<Field<double>> fields2;
        std::vector<Field<int>> fields3;
        std::vector<Field<Eigen::MatrixXd>> fields4;
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
    for(int i = 0; i < fields1.size(); i++){
        fields1[i].resize(sizein);
    }
    for(int i = 0; i < fields2.size(); i++){
        fields2[i].resize(sizein);
    }
    for(int i = 0; i < fields3.size(); i++){
        fields3[i].resize(sizein);
    }
    for(int i = 0; i < fields4.size(); i++){
        fields4[i].resize(sizein);
    }
}

Field<Eigen::VectorXd> * TargetSpace::addField(int dim, vector<int> size, Field<Eigen::VectorXd> * target, bool isDynamic) {
    fields1.push_back(Field < Eigen::VectorXd > (dim, size, isDynamic));
    return &fields1[fields1.size() - 1];
    no_fields = no_fields + 1;
}

Field<double> * TargetSpace::addField(int dim, vector<int> size, Field<double> * target, bool isDynamic) {
    fields2.push_back(Field < double > (dim, size, isDynamic));
    return &fields2[fields2.size() - 1];
    no_fields = no_fields + 1;
}

Field<int> * TargetSpace::addField(int dim, vector<int> size, Field<int> * target, bool isDynamic) {
    fields3.push_back(Field < int > (dim, size, isDynamic));
    return &fields3[fields3.size() - 1];
    no_fields = no_fields + 1;
}

Field<Eigen::MatrixXd> * TargetSpace::addField(int dim, vector<int> size, Field<Eigen::MatrixXd> * target, bool isDynamic) {
    fields4.push_back(Field < Eigen::MatrixXd > (dim, size, isDynamic));
    return &fields4[fields4.size() - 1];
    no_fields = no_fields + 1;
}



    void TargetSpace::save_fields(ofstream& savefile){
        for(int i = 0; i < fields1.size(); i++){
            fields1[i].save_field(savefile);
        }
        for(int i = 0; i < fields2.size(); i++){
            fields2[i].save_field(savefile);
        }
        for(int i = 0; i < fields3.size(); i++){
            fields3[i].save_field(savefile);
        }
        for(int i = 0; i < fields4.size(); i++){
            fields4[i].save_field(savefile);
        }
    }

    void TargetSpace::load_fields(ifstream& loadfile){
        for(int i = 0; i < fields1.size(); i++){
            fields1[i].load_field(loadfile);
        }
        for(int i = 0; i < fields2.size(); i++){
            fields2[i].load_field(loadfile);
        }
        for(int i = 0; i < fields3.size(); i++){
            fields3[i].load_field(loadfile);
        }
        for(int i = 0; i < fields4.size(); i++){
            fields4[i].load_field(loadfile);
        }
    }

    void TargetSpace::update_fields(){
        for(int i = 0; i < fields1.size(); i++){
            fields1[i].update_field();
        }
        for(int i = 0; i < fields2.size(); i++){
            fields2[i].update_field();
        }
        for(int i = 0; i < fields3.size(); i++){
            fields3[i].update_field();
        }
        for(int i = 0; i < fields4.size(); i++){
            fields4[i].update_field();
        }
    }

    void TargetSpace::update_gradients(double dt){
        for(int i = 0; i < fields1.size(); i++){
            fields1[i].update_gradient(dt);
        }
        for(int i = 0; i < fields2.size(); i++){
            fields2[i].update_gradient(dt);
        }
        for(int i = 0; i < fields3.size(); i++){
            fields3[i].update_gradient(dt);
        }
        for(int i = 0; i < fields4.size(); i++){
            fields4[i].update_gradient(dt);
        }
    }
/***********************/
/*   Field Theories    */
/***********************/

class BaseFieldTheory {
   public:
	int dim;
	BaseFieldTheory(int d, vector<int> size); // TO BE WRITTEN
	~BaseFieldTheory(){};
    inline vector<int> next(vector<int> current);
    inline vector<int>  __attribute__((always_inline)) convert(int in);
    inline virtual void calculateGradientFlow(vector<int> pos); // TO BE WRITTEN
    inline virtual double calculateEnergy(vector<int> pos);// TO BE WRITTEN
    void RK4(); // TO BE WRITTEN
	void save(const char * savepath); // TO BE WRITTEN
	void load(const char * loadpath); // TO BE WRITTEN
	void plot(const char * plotpath); // TO BE WRITTEN
	void spaceTransformation(); // TO BE WRITTEN
    template <class T>
	void fieldTransformation(T tranformation); // TO BE WRITTEN
	void setBoundaryType(vector<int> boundaryin); // TO BE WRITTEN
	void updateEnergy(); // TO BE WRITTEN
    inline virtual void gradientFlow(int iterations); // TO BE WRITTEN
    void setTimeInterval(double dt_in);
    double getEnergy(){return energy;};
    inline bool inBoundary(vector<int> pos); // TO BE WRITTEN
    inline  void setSpacing(vector<double> spacein);
    vector<double> getSpacing();
    int getTotalSize();
	template <class T>
	inline T  __attribute__((always_inline)) single_derivative(Field<T> * f, int wrt, ...);
	template <class T>
	inline T  __attribute__((always_inline)) single_derivative(Field<T> * f, int wrt, vector<int> posin);
	template <class T>
    inline T  __attribute__((always_inline)) single_derivative(Field<T> * f, int wrt, Eigen::VectorXi pos);
    template <class T>
	inline T  __attribute__((always_inline)) double_derivative(Field<T> * f, int wrt1, int wrt2, ...);
	template <class T>
	inline T  __attribute__((always_inline)) double_derivative(Field<T> * f, int wrt1, int wrt2, vector<int> posin);
    template <class T>
    inline T  __attribute__((always_inline)) double_derivative(Field<T> * f, int wrt1, int wrt2, Eigen::VectorXi pos);
   protected:
	template <class T>
	Field<T> * createField(Field<T> * target, bool isDynamic);
	//vector<unique_ptr<Field>> fields;
    TargetSpace fields;
	vector<int> bdw; // number of boundary points that are never updated or contribute to energies etc. in each direction
	vector<int> boundarytype; // indicates the type of boundary in each direction 0-fixed(size = bdw), 1-dirichlet (dx = 0), 2-periodic
	vector<double> energydensity;
	double energy;
	vector<double> spacing;
	vector<int> size;
    double dt;
};

    inline bool BaseFieldTheory::inBoundary(vector<int> pos){
        bool value = true;
        for(int i = 0; i < dim; i++){
            if(pos[i] < bdw[2*i] || pos[i] >= size[i]-bdw[2*i+1]){
                value = false;
            }
        }
        return value;
    }

    void BaseFieldTheory::setTimeInterval(double dt_in) {
        dt = dt_in;
    }

    void BaseFieldTheory::setSpacing(vector<double> spacein){
        spacing = spacein;
    }

    vector<double> BaseFieldTheory::getSpacing(){
        return spacing;
    }

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
Field<T> * BaseFieldTheory::createField(Field<T> * target, bool isDynamic){
    return fields.addField(dim, size, target, isDynamic);
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
inline T  __attribute__((always_inline)) BaseFieldTheory::single_derivative(Field<T> * f, int wrt, vector<int> posin) {
    Eigen::VectorXi pos(dim);
    for(int i = 0; i < posin.size(); i++){
        pos[i] = posin[i];
    }
    return single_derivative(f,wrt,pos);
}

template <class T>
inline T  __attribute__((always_inline)) BaseFieldTheory::single_derivative(Field<T> * f, int wrt, Eigen::VectorXi pos) {
    Eigen::VectorXi dir(dim);
    for(int i = 0; i < dim; i++){ if(i == wrt){dir[i] = 1;}else{dir[i] = 0;}};
    return (-1.0*f->getData(pos+2*dir) + 8.0*f->getData(pos+dir) - 8.0*f->getData(pos-dir) + f->getData(pos-2*dir))/(12.0*spacing[wrt]);
}

template <class T>
inline T  __attribute__((always_inline)) BaseFieldTheory::single_derivative(Field<T> * f, int wrt, ...) {
    Eigen::VectorXi pos(dim);
   	va_list ap;
    	va_start(ap, dim);
   	for(int i=0; i<dim; i++){
    	pos[i] = va_arg(ap,int);
    	}
	return single_derivative(f, wrt, pos);
}

template <class T>
inline T  __attribute__((always_inline)) BaseFieldTheory::double_derivative(Field<T> * f, int wrt1, int wrt2, vector<int> posin) {
    Eigen::VectorXi pos(posin.size());
    for(int i = 0; i < posin.size(); i++){
        pos[i] = posin[i];
    }
    return double_derivative(f,wrt1,wrt2,pos);
}

template <class T>
inline T  __attribute__((always_inline)) BaseFieldTheory::double_derivative(Field<T> * f, int wrt1, int wrt2, Eigen::VectorXi pos) {
	if(wrt1 == wrt2)
	{
        Eigen::VectorXi dir(dim);
	    for(int i = 0; i < dim; i++){ if(i == wrt1){dir[i] = 1;}else{dir[i] = 0;}};
        return (-1.0*f->getData(pos+2*dir) + 16.0*f->getData(pos+dir) - 30.0*f->getData(pos) + 16.0*f->getData(pos-dir)
                - f->getData(pos-2*dir))/(12.0*spacing[wrt1]*spacing[wrt1]);
	}
	else
	{
        Eigen::VectorXi dir1(dim);
        Eigen::VectorXi dir2(dim);
	    for(int i = 0; i < dim; i++){
		    if(i == wrt1){dir1[i] = 1;}else{dir1[i] = 0;}
		    if(i == wrt2){dir2[i] = 1;}else{dir2[i] = 0;}
	    };
        return (f->getData(pos+2*dir1+2*dir2) - 8.0*f->getData(pos+dir1+2*dir2) + 8.0*f->getData(pos-dir1+2*dir2)
                - f->getData(pos-2*dir1+2*dir2) - 8.0*f->getData(pos+2*dir1+dir2) +64.0*f->getData(pos+dir1+dir2)
                -64.0*f->getData(pos-dir1+dir2) + 8.0*f->getData(pos-2*dir1+dir2) + 8.0*f->getData(pos+2*dir1-dir2)
                - 64.0*f->getData(pos+dir1-dir2)+64.0*f->getData(pos-dir1-dir2) - 8.0*f->getData(pos-2*dir1-dir2)
                - f->getData(pos+2*dir1-2*dir2) + 8.0*f->getData(pos+dir1-2*dir2) - 8.0*f->getData(pos-dir1-2*dir2)
                + f->getData(pos-2*dir1-2*dir2))/(144.0*spacing[wrt1]*spacing[wrt2]);

	}
}

template <class T>
inline  __attribute__((always_inline)) T BaseFieldTheory::double_derivative(Field<T> * f, int wrt1, int wrt2, ...) {
    Eigen::VectorXi pos(dim);
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
        #pragma omp parallel for reduction(+:sum)
        for (int i = 0; i < getTotalSize(); i++) {
            vector<int> pos = convert(i);
            if (inBoundary(pos)) {
                double buffer = calculateEnergy(pos);
                energydensity[i] = buffer;
                sum += buffer;
            }
        }
	energy = sum*spacing[0]*spacing[1];
};

inline double BaseFieldTheory::calculateEnergy(vector<int> pos)
{
cout << "ERROR! - either the incorrect number of parameters was entered into calculateEnergy or the calculateEnergy has not been set in the derived Field Theory class!\n";
return -1.0;
}

inline void BaseFieldTheory::gradientFlow(int iterations){ // needs to be updated to gradient flow the entire field then update the field!
    int no = 0;
    #pragma omp parallel
    while(no < iterations){
        //generalise and optimise!!

            {
            #pragma omp for nowait
                for (int i = 0; i < getTotalSize(); i++) {
                    calculateGradientFlow(convert(i));
                }
            }
        //cout << omp_get_thread_num() << " : " << no << "\n";
        #pragma omp barrier
        #pragma omp master
        {
            fields.update_gradients(dt);
            no += 1;
        }
    #pragma omp barrier
    }
}

inline vector<int> BaseFieldTheory::next(vector<int> current){
    for(int i = 0; i < dim; i++){
        if(current[i]<size[i]-1){
            for(int j = 0; j < i; j++){
                current[j] = 0;
            }
                current[i] += 1;
                return current;
        }
    }
}

inline vector<int> __attribute__((always_inline)) BaseFieldTheory::convert(int in){
    vector<int> value(dim);
    value[0] = in;
    for(int i = 0; i < dim-1; i++) {
        value[i+1] = 0;
        if (value[i] >= size[i]){
            value[i+1] = value[i]/size[i];
            value[i] = value[i]%size[i];
        }
    }
    return value;
}

inline void BaseFieldTheory::calculateGradientFlow(vector<int> pos){
    cout << "ERROR! you havent written a calculateGradientFlow function!\n";
}

void BaseFieldTheory::save(const char * savepath){
    cout << "You havent yet written a save function!\n";
}


void BaseFieldTheory::load(const char * loadpath){
    cout << "You havent yet written a load function!\n";
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
		void fillTimeDerivative(T value);
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
