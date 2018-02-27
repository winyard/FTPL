/**
 * Field Theory Processing Library
 * Copyright Thomas Winyard 2016
 */


#ifndef FIELD_THEORIES_H
#define FIELD_THEORIES_H

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

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
#include <chrono>
#include <random>
#include <mpi.h>
#include "../source_plot/plots.hpp"

class Timer
{
public:
    Timer() : beg_(clock_::now()) {}
    void reset() { beg_ = clock_::now(); }
    double elapsed() const {
        return std::chrono::duration_cast<second_>
                (clock_::now() - beg_).count(); }

private:
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1> > second_;
    std::chrono::time_point<clock_> beg_;
};

void printMPIDetails(){
    MPI_Barrier(MPI_COMM_WORLD);
    if(MPI::COMM_WORLD.Get_rank() == 0){
        std::cout << "\n----------------------------------------------------------------\n";
        std::cout << "Requested " << MPI::COMM_WORLD.Get_size() << " MPI processes, let's hear from them:\n\n";
    }
    MPI_Barrier(MPI_COMM_WORLD);
    #pragma omp parallel
    {
        #pragma omp master
        std::cout << "Node " << MPI::COMM_WORLD.Get_rank() << " - I have " << omp_get_thread_num() << " threads\n";
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if(MPI::COMM_WORLD.Get_rank() == 0){
        std::cout << "----------------------------------------------------------------\n\n";
    }
    MPI_Barrier(MPI_COMM_WORLD);
}


using namespace std;

namespace FTPL {
    class BaseFieldTheory;
/***********************/
/*       FIELDS        */
/***********************/
// TOO ADD -> taking regions etc.
template <class T>
class Field {
    public:
        vector<T> data;
        vector<T> buffer;
        vector<T> buffer_dt;
        vector<T> dt;
        vector<T> k0_result;
        vector<T> k1_result;
        vector<vector<T>> single_derivatives;
        vector<vector<vector<T>>> double_derivatives;
        T min;
        T max;
        bool dynamic;
	    int dim;
        CUDA_HOSTDEV inline const T  __attribute__((always_inline)) getData( const vector<int> pos);
        CUDA_HOSTDEV inline void  __attribute__((always_inline)) setBuffer(const T value, vector<int> &pos);
        CUDA_HOSTDEV inline int __attribute__((always_inline)) checkindex(int index);
        CUDA_HOSTDEV Field(int d, vector<int> size, bool isdynamic, bool isnormalised);
        CUDA_HOSTDEV ~Field(){};
        CUDA_HOSTDEV inline T operator()(...); // TO BE WRITTEN
        CUDA_HOSTDEV inline const vector<T> getData();
        CUDA_HOSTDEV vector<int> getSize();
        CUDA_HOSTDEV int getSize(int com);
        CUDA_HOSTDEV inline int __attribute__((always_inline)) getTotalSize();
        CUDA_HOSTDEV inline void  __attribute__((always_inline)) setData(vector<T> datain);
        CUDA_HOSTDEV inline void  __attribute__((always_inline)) setData(T value, vector<int> pos);
        CUDA_HOSTDEV void fill(T value);
        CUDA_HOSTDEV void fill_dt(T value);
        CUDA_HOSTDEV void save_field(ofstream& output);
        CUDA_HOSTDEV void load_field(ifstream& input);
        CUDA_HOSTDEV void normalise();
        CUDA_HOSTDEV void update_field();
        CUDA_HOSTDEV void update_gradient(double dt);
        CUDA_HOSTDEV inline void updateRK4(int k);
        CUDA_HOSTDEV inline void updateRK4(int k, int i);
        CUDA_HOSTDEV Field<T>* resize(vector<int> sizein);
        CUDA_HOSTDEV void progressTime(double time_step);
        CUDA_HOSTDEV void update_derivatives(vector<double> spacing);
        CUDA_HOSTDEV inline void alter_point(int i, T value, vector<double> spacing, bool doubleDerivative);
        CUDA_HOSTDEV void setboundarytype(vector<int> boundaryin, double boundaryconstantin = 1.0);
        CUDA_HOSTDEV double getBoundaryConstant();
        CUDA_HOSTDEV inline T __attribute__((always_inline)) rotate(int point, double winding);
        //CUDA_HOSTDEV inline T morph(double winding, T object);
        CUDA_HOSTDEV inline T morph(double winding, Eigen::Vector2d object);
        CUDA_HOSTDEV inline T morph(double winding, double object);
        bool normalised;
        vector<int> boundarytype; // indicates the type of boundary in each direction 0-fixed(size = bdw), 1-dirichlet (dx = 0), 2-periodic
        double boundaryconstant;
        int boundarymorphtype;
    protected:
		vector<int>  size;
        vector<vector<T>> k_sum;
};
    template<class T>//due to awkward types being passed about
    T Field<T>::morph(double winding, double object) {
        cout << "THIS IS THE WRONG TYPE FOR A 2d ROTATION ETC!!!\n";
        return object;
    }

    /*template<class T>//due to awkward types being passed about
    T Field<T>::morph(double winding, T object) {
        cout << "THIS IS THE WRONG TYPE FOR A 2d ROTATION ETC!!!\n";
        return object;
    }*/

    template<class T>//due to awkward types being passed about
    T Field<T>::morph(double winding, Eigen::Vector2d object) {
        //cout << "morphed as 2d correctly\n";
        if(boundarymorphtype == 1){
            Eigen::Vector2d test2;
            test2[0] = 0.0;
            test2[1] = winding;
            return object + test2;
        }
        else{
            Eigen::Matrix2d rot;
            rot << cos(winding), -sin(winding), sin(winding), cos(winding);
            return rot*object;
        }
    }

    template<class T>
    T Field<T>::rotate(int point, double winding){
            return morph(winding,data[point]);
    }

    template<class T>
    double Field<T>::getBoundaryConstant(){
        return boundaryconstant;
    }

    template<class T>
    void Field<T>::setboundarytype(vector<int> boundaryin, double boundaryconstantin){
        if(boundaryin.size() == dim*2){boundarytype = boundaryin;}
        else{cout << "boundary input is of wrong size, should be a vector<int> of size " << dim*2 << " 2 for each dimension\n";}
        boundaryconstant = boundaryconstantin;
    }

template<class T>
    void Field<T>::alter_point(int i, T value, vector<double> spacing, bool doubleDerivative){
        buffer[i] = data[i];
        data[i] = value;
        T dif = data[i]-buffer[i];
        //now correct the derivatives:
        double mult1 = 1.0;
        for(int wrt1 = 0; wrt1<dim; wrt1++){
            T change = dif/(12.0*spacing[wrt1]);
            single_derivatives[wrt1][i-2*mult1] += -change;
            single_derivatives[wrt1][i-1*mult1] += 8.0*change;
            single_derivatives[wrt1][i+1*mult1] += -8.0*change;
            single_derivatives[wrt1][i+2*mult1] += change;
            if(doubleDerivative) {
                change = dif / (12.0 * spacing[wrt1] * spacing[wrt1]);
                double_derivatives[wrt1][wrt1][i - 2 * mult1] += -change;
                double_derivatives[wrt1][wrt1][i - mult1] += +16.0 * change;
                double_derivatives[wrt1][wrt1][i] += -30.0 * change;
                double_derivatives[wrt1][wrt1][i + mult1] += +16.0 * change;
                double_derivatives[wrt1][wrt1][i + 2 * mult1] += -change;
                double mult2 = 1.0;
                for (int wrt2 = 0; wrt2 < wrt1; wrt2++) {
                    change = dif / (144.0 * spacing[wrt1] * spacing[wrt2]);
                    double_derivatives[wrt2][wrt1][i - 2 * mult2 - 2 * mult1] += change;
                    double_derivatives[wrt2][wrt1][i - mult2 - 2 * mult1] += -8.0 * change;
                    double_derivatives[wrt2][wrt1][i + mult2 - 2 * mult1] += 8.0 * change;
                    double_derivatives[wrt2][wrt1][i + 2 * mult2 - 2 * mult1] += -change;
                    double_derivatives[wrt2][wrt1][i - 2 * mult2 - mult1] += -8.0 * change;
                    double_derivatives[wrt2][wrt1][i - mult2 - mult1] += +64.0 * change;
                    double_derivatives[wrt2][wrt1][i + mult2 - mult1] += -64.0 * change;
                    double_derivatives[wrt2][wrt1][i + 2 * mult2 - mult1] += +8.0 * change;
                    double_derivatives[wrt2][wrt1][i - 2 * mult2 + mult1] += +8.0 * change;
                    double_derivatives[wrt2][wrt1][i - mult2 + mult1] += -64.0 * change;
                    double_derivatives[wrt2][wrt1][i + mult2 + mult1] += +64.0 * change;
                    double_derivatives[wrt2][wrt1][i + 2 * mult2 + mult1] += -8.0 * change;
                    double_derivatives[wrt2][wrt1][i - 2 * mult2 + 2 * mult1] += -change;
                    double_derivatives[wrt2][wrt1][i - mult2 + 2 * mult1] += 8.0 * change;
                    double_derivatives[wrt2][wrt1][i + mult2 + 2 * mult1] += -8.0 * change;
                    double_derivatives[wrt2][wrt1][i + 2 * mult2 + 2 * mult1] += change;
                    mult2 *= size[wrt2];
                }
            }
            mult1 *= size[wrt1];
        }
    }

/* --- Constructors & Destructors --- */

template <class T>
Field<T>::Field(int d, vector<int> sizein, bool isdynamic, bool isnormalised): dim(d), size(sizein), dynamic(isdynamic), normalised(isnormalised)  {
    data.resize(getTotalSize());
    buffer.resize(getTotalSize());
    if(dynamic){
        dt.resize(getTotalSize());
        k_sum.resize(2);
        k_sum[0].resize(getTotalSize());
        k_sum[1].resize(getTotalSize());
        k0_result.resize(getTotalSize());
        k1_result.resize(getTotalSize());
        buffer_dt.resize(getTotalSize());
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
inline const __attribute__((always_inline))  T Field<T>::getData( const vector<int> pos) {// can be optimised better! - similar to above method
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
    inline void Field<T>::setData(T value, vector<int> pos){
        int point = pos[0];
        int multiplier = 1;
        for(int i = 1 ; i < dim; i++){
            multiplier *= size[i-1];
            point += pos[i]*multiplier;
        }
        data[point] = value;
    }

    template <class T>
inline void __attribute__((always_inline))  Field<T>::setBuffer(const T value, vector<int> &pos){
        int point = pos[0];
        int multiplier = 1;
        for(int i=1; i<dim; i++){
            multiplier *= size[i-1];
            point += pos[i]*multiplier;
        }
        buffer[point] = value;
    }

template <class T>
void Field<T>::fill(T value){
    for(int i=0; i<getTotalSize(); i++){
    data[i] = value;
    }
}

template <class T>
void Field<T>::fill_dt(T value){
    for(int i=0; i<getTotalSize(); i++){
        dt[i] = value;
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
    //cout << "ive been asked for my size " << total << "\n";
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
    if(normalised) {
        for (int i = 0; i < getTotalSize(); i++) {
            data[i].normalize();
        }
        if (dynamic) {
            for (int i = 0; i < getTotalSize(); i++) {
                dt[i] = dt[i] - (data[i].dot(dt[i])) * data[i];
            }
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

    template<class T>
    void Field<T>::updateRK4(int k, int i){
            if(k == 0) {

                data[i] = buffer[i] + 0.5 * k0_result[i];
                dt[i] = buffer_dt[i] + 0.5 * k1_result[i];
                k_sum[0][i] = k0_result[i];
                k_sum[1][i] = k1_result[i];
            }else if(k == 1){
                data[i] = buffer[i] + 0.5 * k0_result[i];
                dt[i] = buffer_dt[i] + 0.5 * k1_result[i];
                k_sum[0][i] += 2.0*k0_result[i];
                k_sum[1][i] += 2.0*k1_result[i];
            } else if(k == 2){
                data[i] =  buffer[i] + k0_result[i];
                dt[i] = buffer_dt[i] + k1_result[i];
                k_sum[0][i] += 2.0*k0_result[i];
                k_sum[1][i] += 2.0*k1_result[i];
            } else{
                data[i] =  buffer[i] + (k_sum[0][i] + k0_result[i])/6.0;
                dt[i] = buffer_dt[i] + (k_sum[1][i] + k1_result[i])/6.0;
            }
        }

        template<class T>
        void Field<T>::updateRK4(int k){
            if(k <3) {
                data = k0_result;
                dt = k1_result;
            }else {
                data = k_sum[0];
                dt = k_sum[1];
            }
    }



template <class T>
Field<T>* Field<T>::resize(vector<int> sizein) { // resize the fields for different accuracy
    if(size.size() == sizein.size()) {
        size = sizein;
        data.resize(getTotalSize());
        buffer.resize(getTotalSize());
        if(dynamic){
            dt.resize(getTotalSize());
            buffer_dt.resize(getTotalSize());
            k_sum[0].resize(getTotalSize());
            k_sum[1].resize(getTotalSize());
            k0_result.resize(getTotalSize());
            k1_result.resize(getTotalSize());
        }
    }
    else{
        cout << "Tried to resize field with incorrect size of dimension vector!\n";
    }
    return this;
}

/***********************/
/*    Target Space     */
/***********************/

class TargetSpace {
    public:
        CUDA_HOSTDEV TargetSpace();
        CUDA_HOSTDEV ~TargetSpace(){};
        CUDA_HOSTDEV Field<Eigen::VectorXd> * addField(int dim, vector<int> size, Field<Eigen::VectorXd> * target, bool isDynamic, bool isNormalised);
        CUDA_HOSTDEV Field<double> * addField(int dim, vector<int> size, Field<double> * target, bool isDynamic, bool isNormalised);
        CUDA_HOSTDEV Field<int> * addField(int dim, vector<int> size, Field<int> * target, bool isDynamic, bool isNormalised);
        CUDA_HOSTDEV Field<Eigen::MatrixXd> * addField(int dim, vector<int> size, Field<Eigen::MatrixXd> * target, bool isDynamic, bool isNormalised);
        CUDA_HOSTDEV void MPICommDomain(int partner,int point,vector<int> dataDomain,vector<int> derivativeDomain,int tag,bool send,int doubleDerivatives = 2, bool sendDynamic = false);
        CUDA_HOSTDEV void resize(vector<int> sizein);
        CUDA_HOSTDEV void load_fields(ifstream& loadfile, int totalSize);
        CUDA_HOSTDEV void save_fields(ofstream& savefile, int totalSize);
        CUDA_HOSTDEV void update_fields();
        CUDA_HOSTDEV void update_gradients(double dt);
        CUDA_HOSTDEV void update_RK4(int k, int pos);
        CUDA_HOSTDEV void normalise();
        CUDA_HOSTDEV void moveToBuffer();
        CUDA_HOSTDEV void cutKinetic(int pos);
        CUDA_HOSTDEV int no_fields;
        CUDA_HOSTDEV void storeDerivatives(BaseFieldTheory * theory, bool doubleDerivative);
        CUDA_HOSTDEV void randomise(int i, int field, vector<double> spacing, double dt, bool doubleDerivative = false);
        CUDA_HOSTDEV void derandomise(int i, int field, vector<double> spacing, bool doubleDerivative = false);
        CUDA_HOSTDEV void resetPointers();
        CUDA_HOSTDEV void CUDA_uploadFields();
    private:
        // Add aditional Field types as they are created here!
        std::vector<Field<Eigen::VectorXd>*> fields1;
        std::vector<Field<Eigen::VectorXd>**> fields1_pointers;
        std::vector<Field<double>*> fields2;
        std::vector<Field<double>**> fields2_pointers;
        std::vector<Field<int>*> fields3;
        std::vector<Field<int>**> fields3_pointers;
        std::vector<Field<Eigen::MatrixXd>*> fields4;
        std::vector<Field<Eigen::MatrixXd>**> fields4_pointers;
};

    class BaseFieldTheory {
    public:
        int dim;
        CUDA_HOSTDEV  BaseFieldTheory(int d, vector<int> size, bool isDynamic); // TO BE WRITTEN
        CUDA_HOSTDEV ~BaseFieldTheory(){};
        CUDA_HOSTDEV inline vector<int> next(vector<int> current);
        CUDA_HOSTDEV inline vector<int>  __attribute__((always_inline)) convert(int in);
        CUDA_HOSTDEV inline virtual void calculateGradientFlow(int pos); // TO BE WRITTEN
        CUDA_HOSTDEV inline virtual double calculateEnergy(int pos);// TO BE WRITTEN
        CUDA_HOSTDEV inline virtual double calculateCharge(int pos);
        CUDA_HOSTDEV inline virtual void __attribute__((always_inline)) RK4calc(int i);
        CUDA_HOSTDEV inline virtual double __attribute__((always_inline)) metric(int i, int j, vector<double> pos = {0});
        CUDA_HOSTDEV inline void MPICommEnergy(int partner,int pos,vector<int> domain,int tag,bool send);
        CUDA_HOSTDEV inline int periodicindex(int index);
        CUDA_HOSTDEV void RK4(int iterations, bool cutEnergy, int often); // TO BE WRITTEN
        CUDA_HOSTDEV void save(const char * savepath); // TO BE WRITTEN
        CUDA_HOSTDEV void load(const char * loadpath, bool message = true); // TO BE WRITTEN
        CUDA_HOSTDEV void plot(const char * plotpath); // TO BE WRITTEN
        CUDA_HOSTDEV void spaceTransformation(); // TO BE WRITTEN
        template <class T>
        CUDA_HOSTDEV void fieldTransformation(T tranformation); // TO BE WRITTEN
        CUDA_HOSTDEV void setAllBoundaryType(vector<int> boundaryin); // TO BE WRITTEN
        CUDA_HOSTDEV void setbdw(vector<int> bdwin); // TO BE WRITTEN
        CUDA_HOSTDEV void setStandardMetric(string type);
        CUDA_HOSTDEV void updateEnergy(); // TO BE WRITTEN
        CUDA_HOSTDEV void updateCharge();
        CUDA_HOSTDEV void gradientFlow(int iterations, int often); // TO BE WRITTEN
        CUDA_HOSTDEV void setTimeInterval(double dt_in);
        CUDA_HOSTDEV double getEnergy(){return energy;};
        CUDA_HOSTDEV inline bool inBoundary(vector<int> pos);
        CUDA_HOSTDEV inline bool inBoundary(int pos);
        CUDA_HOSTDEV inline virtual __attribute__((always_inline)) vector<double> calculateDynamicEnergy(int pos);
        CUDA_HOSTDEV inline  void setSpacing(vector<double> spacein);
        CUDA_HOSTDEV void addParameter( double * parameter_in, string name);
        CUDA_HOSTDEV vector<double> getSpacing();
        CUDA_HOSTDEV int getTotalSize();
        CUDA_HOSTDEV double getTotalSpacing();
        CUDA_HOSTDEV void plotEnergy();
        CUDA_HOSTDEV void plotCharge();
        CUDA_HOSTDEV void printParameters();
        CUDA_HOSTDEV void setMetricType(string type);
        CUDA_HOSTDEV bool annealing(int iterations, int often, int often_cut, bool output = true);
        CUDA_HOSTDEV void getDomain(vector<int> dimensions, vector<int> * target, vector<int> * targetBoundaries);
        template <class T>
        CUDA_HOSTDEV inline T single_time_derivative(Field<T> * f, int wrt, int &point) __attribute__((always_inline))  ;
        template <class T>
        CUDA_HOSTDEV inline T single_derivative(Field<T> * f, int wrt, int &point) __attribute__((always_inline))  ;
        template <class T>
        CUDA_HOSTDEV inline T double_derivative(Field<T> * f, int wrt1, int wrt2, int &point) __attribute__((always_inline)) ;
        CUDA_HOSTDEV double getCharge(){return charge;};
        CUDA_HOSTDEV void MPIFunction(bool (BaseFieldTheory::*localFunction)(int, int, int, bool), int dimSplit, int loops, int localIterations, int * parameters, int sendDerivative = 2, bool sendDynamic = false, vector<int> domainSize = {});
        CUDA_HOSTDEV void MPIAnnealing(int iterations, int localIterations, int localPoints, int often, int often_cut);
        CUDA_HOSTDEV inline void addOne(vector<int>& pos, vector<int> limits, int component = 0);
        CUDA_HOSTDEV int * calculateSplitting(int dimension, int noNodes);
        CUDA_HOSTDEV inline void sendRecvBoundaries(MPI_Comm * COMM_GRID,vector<int> * dataDomain,vector<int> * derivedDomain);
        CUDA_HOSTDEV void virtual gradientCorrections(int loop);
        CUDA_HOSTDEV void virtual correctTheory(int loop);
    protected:
        int metric_type = 0;
        template <class T>
        CUDA_HOSTDEV Field<T> * createField(Field<T> * target, bool isDynamic, bool isNormalised = false);
        //vector<unique_ptr<Field>> fields;
        TargetSpace fields;
        vector<int> bdw; // number of boundary points that are never updated or contribute to energies etc. in each direction
        vector<double> energydensity;
        double energy, potential, kinetic;
        vector<double> spacing;
        vector<int> size;
        vector<double> chargedensity;
        double charge;
        double dt;
        bool dynamic;
        bool curved = false;
        bool setDerivatives = false;
        vector<double *> parameters;
        vector<string> parameterNames;
        vector<int> boundarytype;
        bool  normalise_me;
    };

    void BaseFieldTheory::setAllBoundaryType(vector<int> boundaryin){
        cout << "WARNING!!! - Boundary Conditions are still under construction!\n";
        boundarytype = boundaryin;
} // TO BE WRITTEN

    void BaseFieldTheory::setbdw(vector<int> bdwin){
        cout << "WARNING!!! - alternat bdw are still under construction!\n";
        bdw = bdwin;
        cout << "set\n";
    } // TO BE WRITTEN

    void BaseFieldTheory::plot(const char *plotpath) {
        cout << "ERROR!!! - Plot function not yet written, please correct in code!\n";
    }

    void BaseFieldTheory::sendRecvBoundaries(MPI_Comm *COMM_GRID, vector<int> *dataDomain,
                                             vector<int> *derivedDomain) {
        cout << "ERROR!! - sendRecvBOundaries function not yet written, please correct code!\n";
    }

    void BaseFieldTheory::getDomain(vector<int> dimensions, vector<int> * target, vector<int> * targetBoundaries) {
        vector<int> pos(dim);
        int totalSize = 1;
        int point;
        for (int j = 0; j < dim; j++) { pos[j] = 0; totalSize *= dimensions[j];}
        for (int i = 0; i < totalSize; i++) {
            //porgress the pos value using the limiter of domain
            if (i > 0) {
                addOne(pos, dimensions);
            }
            //calculate the corresponding 1-dim value (bear in mind the different limits!)
            point = pos[0];
            int multiplier = 1;
            for (int i = 1; i < dim; i++) {
                multiplier *= size[i - 1];
                point += pos[i] * multiplier;
            }
            //add the points to the corresponding vectors
            vector<int> temp_size = size;
            size = dimensions; // temporarily set the size of the theory to get the correct boundary values
            if (inBoundary(pos)) {
                target->push_back(point);
            }else{targetBoundaries->push_back(point);}
            size = temp_size;
        }
    }

    void BaseFieldTheory::MPIAnnealing(int iterations, int localIterations, int localPoints, int often, int often_cut) {
        vector<int> domainSize;
        for(int i = 0; i < dim; i++){
            domainSize.push_back(4+localPoints);
        }
        //resize the data to the recieved sizes
        if(MPI::COMM_WORLD.Get_rank()!=0){
            if(domainSize.size() != dim){cout << "ERROR!! you have supplied a domainSize with the incorrect dimension into MPIFunction!\n";}
            fields.resize(domainSize);
            size = domainSize;
        }
        fields.storeDerivatives(this,false);
        setDerivatives = true;
        updateEnergy();
        if(MPI::COMM_WORLD.Get_rank()!=0) {
            omp_set_num_threads(1);
        }
        double oldEnergy = energy;
        int check = 0;
        for(int no = 0; no < iterations; no++) {
            int parameters[2];
            MPIFunction(&BaseFieldTheory::annealing, dim, often, localIterations, parameters, 0, false, domainSize);
            //sum up the energydensity
            if(MPI::COMM_WORLD.Get_rank()==0) {
                //fields.storeDerivatives(this,false);
                updateEnergy();
                updateCharge();
                cout << no << "(" << (int) 100*no/iterations<<"%) energy = " << energy << " dt = " << dt << " charge = " << charge << "\n";
                if (oldEnergy == energy) { check++; }
                else {
                    oldEnergy = energy;
                    check = 0;
                }
                if (check == often_cut) {
                    check = 0;
                    dt *= 0.9;
                    //fields.storeDerivatives(this,false);
                }
            }
            MPI_Bcast(&dt,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
        }
    }

    void BaseFieldTheory::addOne(vector<int>& pos, vector<int> limits, int component){
        if(pos[component] == limits[component]-1){
            pos[component] = 0;
            addOne(pos, limits, component + 1);
        }else {
            pos[component] += 1;
        }
    }

    void BaseFieldTheory::MPICommEnergy(int partner,int pos,vector<int> domain,int tag,bool send){
        for(int i = 0; i < domain.size(); i++) {
            if(send) {
                MPI_Send(&energydensity[pos+domain[i]],1,MPI_DOUBLE,partner,tag,MPI_COMM_WORLD);
            }else{
                MPI_Recv(&energydensity[pos+domain[i]],1,MPI_DOUBLE,partner,tag,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            }
        }
    }

    void TargetSpace::MPICommDomain(int partner,int point,vector<int> dataDomain,vector<int> derivativeDomain,int tag,bool send,int sendDerivatives, bool sendDynamic){
        for(int no = 0; no < fields1.size(); no++){
            for(int i = 0; i < dataDomain.size(); i++) {
                if (send) {
                    MPI_Send(fields1[no]->data[point + dataDomain[i]].data(),
                             fields1[no]->data[point + dataDomain[i]].size(), MPI_DOUBLE, partner, tag+i, MPI_COMM_WORLD);
                    if(sendDynamic){
                        MPI_Send(fields1[no]->dt[point + dataDomain[i]].data(),
                                 fields1[no]->dt[point + dataDomain[i]].size(), MPI_DOUBLE, partner, tag+100+i, MPI_COMM_WORLD);
                    }
                } else {
                    MPI_Recv(fields1[no]->data[point + dataDomain[i]].data(),
                             fields1[no]->data[point + dataDomain[i]].size(), MPI_DOUBLE, partner, tag+i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    if(sendDynamic){
                        MPI_Recv(fields1[no]->dt[point + dataDomain[i]].data(),
                                 fields1[no]->dt[point + dataDomain[i]].size(), MPI_DOUBLE, partner, tag+100+i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    }
                }
            }
            if(sendDerivatives > 0) {
                for (int i = 0; i < derivativeDomain.size(); i++) {
                    if (send) {
                        for (int wrt1 = 0; wrt1 < fields1[no]->dim; wrt1++) {
                            MPI_Send(fields1[no]->single_derivatives[wrt1][point + derivativeDomain[i]].data(),
                                     fields1[no]->single_derivatives[wrt1][point + derivativeDomain[i]].size(),
                                     MPI_DOUBLE, partner, i * (tag + 10) + wrt1,
                                     MPI_COMM_WORLD);
                            if (sendDerivatives == 2) {
                                for (int wrt2 = 0; wrt2 <= wrt1; wrt2++) {
                                    MPI_Send(
                                            fields1[no]->double_derivatives[wrt2][wrt1][point +
                                                                                        derivativeDomain[i]].data(),
                                            fields1[no]->double_derivatives[wrt2][wrt1][point +
                                                                                        derivativeDomain[i]].size(),
                                            MPI_DOUBLE, partner, i * (tag + 10) + fields1[no]->dim + wrt1 + wrt2,
                                            MPI_COMM_WORLD);
                                }
                            }
                        }
                    } else {
                        for (int wrt1 = 0; wrt1 < fields1[no]->dim; wrt1++) {
                            MPI_Recv(fields1[no]->single_derivatives[wrt1][point + derivativeDomain[i]].data(),
                                     fields1[no]->single_derivatives[wrt1][point + derivativeDomain[i]].size(),
                                     MPI_DOUBLE, partner, i * (tag + 10) + wrt1,
                                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                            if (sendDerivatives == 2) {
                                for (int wrt2 = 0; wrt2 <= wrt1; wrt2++) {
                                    MPI_Recv(
                                            fields1[no]->double_derivatives[wrt2][wrt1][point +
                                                                                        derivativeDomain[i]].data(),
                                            fields1[no]->double_derivatives[wrt2][wrt1][point +
                                                                                        derivativeDomain[i]].size(),
                                            MPI_DOUBLE, partner, i * (tag + 10) + fields1[no]->dim + wrt1 + wrt2,
                                            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                                }
                            }
                        }
                    }
                }
            }
        }

        for(int no = 0; no < fields2.size(); no++){
            for(int i = 0; i < dataDomain.size(); i++) {
                if(send) {
                    MPI_Send(&fields2[no]->data[point + dataDomain[i]],
                             1, MPI_DOUBLE,partner,tag,MPI_COMM_WORLD);
                    if(sendDynamic){
                        MPI_Send(&fields2[no]->dt[point + dataDomain[i]],
                                 1, MPI_DOUBLE,partner,tag,MPI_COMM_WORLD);
                    }
                }else{
                    MPI_Recv(&fields2[no]->data[point + dataDomain[i]],
                             1, MPI_DOUBLE,partner,tag,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                    if(sendDynamic){
                        MPI_Recv(&fields2[no]->dt[point + dataDomain[i]],
                                 1, MPI_DOUBLE,partner,tag,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                    }
                }
            }
            if(sendDerivatives > 0) {
                for (int i = 0; i < derivativeDomain.size(); i++) {
                    if (send) {
                        for (int wrt1 = 0; wrt1 < fields2[no]->dim; wrt1++) {
                            MPI_Send(&fields2[no]->single_derivatives[wrt1][point + derivativeDomain[i]],
                                     1, MPI_DOUBLE, partner, tag,
                                     MPI_COMM_WORLD);
                            if (sendDerivatives == 2) {
                                for (int wrt2 = 0; wrt2 <= wrt1; wrt2++) {
                                    MPI_Send(&fields2[no]->double_derivatives[wrt2][wrt1][point + derivativeDomain[i]],
                                             1, MPI_DOUBLE, partner, tag,
                                             MPI_COMM_WORLD);
                                }
                            }
                        }
                    } else {
                        for (int wrt1 = 0; wrt1 < fields2[no]->dim; wrt1++) {
                            MPI_Recv(&fields2[no]->single_derivatives[wrt1][point + derivativeDomain[i]],
                                     1, MPI_DOUBLE, partner, tag,
                                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                            if (sendDerivatives == 2) {
                                for (int wrt2 = 0; wrt2 <= wrt1; wrt2++) {
                                    MPI_Recv(&fields2[no]->double_derivatives[wrt2][wrt1][point + derivativeDomain[i]],
                                             1, MPI_DOUBLE, partner, tag,
                                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                                }
                            }
                        }
                    }
                }
            }
        }
        for(int no = 0; no < fields3.size(); no++){
            for(int i = 0; i < dataDomain.size(); i++) {
                if(send) {
                    MPI_Send(&fields3[no]->data[point + dataDomain[i]],
                             1, MPI_INT,partner,tag,MPI_COMM_WORLD);
                    if(sendDynamic){
                        MPI_Send(&fields3[no]->dt[point + dataDomain[i]],
                                 1, MPI_INT,partner,tag,MPI_COMM_WORLD);
                    }
                }else{
                    MPI_Recv(&fields3[no]->data[point + dataDomain[i]],
                             1, MPI_INT,partner,tag,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                    if(sendDynamic){
                        MPI_Recv(&fields3[no]->dt[point + dataDomain[i]],
                                 1, MPI_INT,partner,tag,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                    }
                }
            }
            if(sendDerivatives > 0) {
                for (int i = 0; i < derivativeDomain.size(); i++) {
                    if (send) {
                        for (int wrt1 = 0; wrt1 < fields3[no]->dim; wrt1++) {
                            MPI_Send(&fields3[no]->single_derivatives[wrt1][point + derivativeDomain[i]],
                                     1, MPI_INT, partner, tag,
                                     MPI_COMM_WORLD);
                            if (sendDerivatives == 2) {
                                for (int wrt2 = 0; wrt2 <= wrt1; wrt2++) {
                                    MPI_Send(&fields3[no]->double_derivatives[wrt2][wrt1][point + derivativeDomain[i]],
                                             1, MPI_INT, partner, tag,
                                             MPI_COMM_WORLD);
                                }
                            }
                        }
                    } else {
                        for (int wrt1 = 0; wrt1 < fields3[no]->dim; wrt1++) {
                            MPI_Recv(&fields3[no]->single_derivatives[wrt1][point + derivativeDomain[i]],
                                     1, MPI_INT, partner, tag,
                                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                            if (sendDerivatives == 2) {
                                for (int wrt2 = 0; wrt2 <= wrt1; wrt2++) {
                                    MPI_Recv(&fields3[no]->double_derivatives[wrt2][wrt1][point + derivativeDomain[i]],
                                             1, MPI_INT, partner, tag,
                                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                                }
                            }
                        }
                    }
                }
            }
        }
        for(int no = 0; no < fields4.size(); no++){
            for(int i = 0; i < dataDomain.size(); i++) {
                if(send) {
                    MPI_Send(fields4[no]->data[point + dataDomain[i]].data(),
                             fields4[no]->data[point + dataDomain[i]].cols()*fields4[no]->data[point + dataDomain[i]].rows(), MPI_DOUBLE,partner,tag,MPI_COMM_WORLD);
                    if(sendDynamic){
                        MPI_Send(fields4[no]->dt[point + dataDomain[i]].data(),
                                 fields4[no]->dt[point + dataDomain[i]].cols()*fields4[no]->dt[point + dataDomain[i]].rows(), MPI_DOUBLE,partner,tag,MPI_COMM_WORLD);
                    }
                }else{
                    MPI_Recv(fields4[no]->data[point + dataDomain[i]].data(),
                             fields4[no]->data[point + dataDomain[i]].cols()*fields4[no]->data[point + dataDomain[i]].rows(), MPI_DOUBLE,partner,tag,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                    if(sendDynamic){
                        MPI_Recv(fields4[no]->dt[point + dataDomain[i]].data(),
                                 fields4[no]->dt[point + dataDomain[i]].cols()*fields4[no]->dt[point + dataDomain[i]].rows(), MPI_DOUBLE,partner,tag,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                    }
                }
            }
            if(sendDerivatives > 0) {
                for (int i = 0; i < derivativeDomain.size(); i++) {
                    if (send) {
                        for (int wrt1 = 0; wrt1 < fields4[no]->dim; wrt1++) {
                            MPI_Send(fields4[no]->single_derivatives[wrt1][point + derivativeDomain[i]].data(),
                                     fields4[no]->single_derivatives[wrt1][point + derivativeDomain[i]].cols() *
                                     fields4[no]->single_derivatives[wrt1][point + derivativeDomain[i]].rows(),
                                     MPI_DOUBLE, partner, tag,
                                     MPI_COMM_WORLD);
                            if (sendDerivatives == 2) {
                                for (int wrt2 = 0; wrt2 <= wrt1; wrt2++) {
                                    MPI_Send(
                                            fields4[no]->double_derivatives[wrt2][wrt1][point +
                                                                                        derivativeDomain[i]].data(),
                                            fields4[no]->double_derivatives[wrt2][wrt1][point +
                                                                                        derivativeDomain[i]].cols() *
                                            fields4[no]->double_derivatives[wrt2][wrt1][point +
                                                                                        derivativeDomain[i]].rows(),
                                            MPI_DOUBLE, partner, tag,
                                            MPI_COMM_WORLD);
                                }
                            }
                        }
                    } else {
                        for (int wrt1 = 0; wrt1 < fields4[no]->dim; wrt1++) {
                            MPI_Recv(fields4[no]->single_derivatives[wrt1][point + derivativeDomain[i]].data(),
                                     fields4[no]->single_derivatives[wrt1][point + derivativeDomain[i]].rows() *
                                     fields4[no]->single_derivatives[wrt1][point + derivativeDomain[i]].cols(),
                                     MPI_DOUBLE, partner, tag,
                                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                            if (sendDerivatives == 2) {
                                for (int wrt2 = 0; wrt2 <= wrt1; wrt2++) {
                                    MPI_Recv(
                                            fields4[no]->double_derivatives[wrt2][wrt1][point +
                                                                                        derivativeDomain[i]].data(),
                                            fields4[no]->double_derivatives[wrt2][wrt1][point +
                                                                                        derivativeDomain[i]].cols() *
                                            fields4[no]->double_derivatives[wrt2][wrt1][point +
                                                                                        derivativeDomain[i]].rows(),
                                            MPI_DOUBLE, partner, tag,
                                            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                                }
                            }
                        }
                    }
                }
            }
        }

    }

    int * BaseFieldTheory::calculateSplitting(int dimension, int noNodes) {
        int MPIdim[dimension];
        int minimalSquare = (int) floor(exp((1.0 / dimension) * log(noNodes)));
        int leftover = noNodes - pow(minimalSquare, dimension);
        for (int i = 0; i < dim; i++) {
            MPIdim[i] = minimalSquare;
        }
        MPIdim[0] += (int) floor(exp((1.0 / (dimension - 1)) * log(leftover)));
        cout << "the selected topology is ";
        for (int i = 0; i < dim; i++) {
            if (i != 0) { cout << "x"; }
            cout << MPIdim[i];
        }
        cout << "\n";
        cout << "there are " << noNodes - pow(minimalSquare, dimension - 1) - MPIdim[0]
             << " unused nodes out of a total of " << noNodes << " which is "
             << 100.0 * (noNodes - pow(minimalSquare, dimension - 1) - MPIdim[0]) / noNodes << "%\n";
        if (noNodes - pow(minimalSquare, dimension - 1) - MPIdim[0] != 0) {
            cout
                    << "would suggest exiting and rerunning code with correct number of nodes! - but hey, I'm not your mother\n";
        }

        return MPIdim;
    }

void BaseFieldTheory::MPIFunction(bool (BaseFieldTheory::*localFunction)(int, int, int, bool), int dimSplit, int loops, int localIterations, int * parameters, int sendDerivatives, bool sendDynamic, vector<int> domainSize) {
        int globalId = MPI::COMM_WORLD.Get_rank();
        int totalNo = MPI::COMM_WORLD.Get_size();

        if(domainSize.empty()) {
            //Split function equally across the grid
            //calculate the grid splitting
            int wrapAround[dim];

            int* MPIdim = calculateSplitting(dim, totalNo - 1);

            //setting the boundary types
            for (int i = 0; i < dim; i++) {
                if (boundarytype[2 * i] == 1) {
                    cout << "Warning Periodic boundary conditions have not been comprehensively tested for MPI\n";
                    wrapAround[i] = 1;
                } else { wrapAround[i] = 0; }
            }

            int totalGridNodes = 1;
            for (int i = 0; i < dim; i++) {
                totalGridNodes *= MPIdim[i];
            }

            //calculate the size of the domains
            vector<int> standardCellSize(dim);
            vector<int> finalCellSize(dim);
            for(int i = 0; i < dim; i++){
                standardCellSize[i] = floor((size[i] - bdw[2*i] - bdw[2*i+1])/MPIdim[i]);
                finalCellSize[i] = (size[i] - bdw[2*i] - bdw[2*i+1]) - standardCellSize[i]*(MPIdim[i]-1);
            }

            MPI_Comm COMM_NEW;
            MPI_Comm_split(MPI_COMM_WORLD, globalId == 0, 0, &COMM_NEW);
            if(globalId == 0){
                // recieve the topology YAY!
                int positions[totalGridNodes];
                bool final[totalGridNodes];
                for(int i = 0; i < totalGridNodes; i++) {
                    int localPos[dim];
                    MPI_Recv(&localPos, dim, MPI_INT, i+1, 9, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    int point = localPos[0]*standardCellSize[0];
                    int multiplier = 1;
                    for (int i = 1; i < dim; i++) {
                        multiplier *= size[i - 1];
                        point += localPos[i]*standardCellSize[i] * multiplier;
                    }
                    positions[i] = point;
                    if(localPos[0]>=MPIdim[1]){
                        final[i] = true;
                    }else{
                        final[i]=false;
                    }
                }

                // calculate the domain vectors (for both standard and final cells)
                    vector<int> standardDataDomain;
                    vector<int> standardDerivedDomain;
                    vector<int> finalDataDomain;
                    vector<int> finalDerivedDomain;
                    getDomain(standardCellSize,&standardDataDomain,&standardDerivedDomain);
                    getDomain(finalCellSize,&finalDataDomain,&finalDerivedDomain);

                    //calculate the starting points for each node

                    //split the grid and send the various data to the slaves:
                    for(int no = 0; no < totalGridNodes; no++){
                        if(final[no]){ fields.MPICommDomain(no, positions[no], finalDerivedDomain, finalDerivedDomain, 0, true, sendDerivatives, sendDynamic);}
                        else{ fields.MPICommDomain(no, positions[no], standardDerivedDomain, standardDerivedDomain, 0, true, sendDerivatives, sendDynamic);}
                    }
                    double oldEnergy = 100000000.0;
                    int cutCount = 0;
                    double cutFactor = 0.95;
                    const char * tempFile = "MPItemp_field";
                    const char * plotFile = "MPIplot";
                    for(int loop = 0; loop < loops; loop++) {
                        //recieve data for outputs
                        energy = 0;
                        /* reduce energy values with master node contributing 0 */
                        //parameters are 0 - oftenPrint, 1 - cutKinetic/cutCount, 2- outputoften, 3 plot?
                        if(loop%parameters[0]==0) {
                            cout << loop << ":" << " energy = " << energy << "\n";
                        }
                        if(parameters[1] > 0){
                            if(oldEnergy <= energy){
                                cutCount++;
                                if(cutCount%parameters[1] == 0){dt *= cutFactor*dt;cutCount = 0;}
                            }
                            /* Broadcast dt */
                            oldEnergy = energy;
                        }
                        if(loop%parameters[2]==0){
                            for(int no = 0; no < totalGridNodes; no++){
                                // Pull in the fields and then save them to the supplied temp file
                                if(final[no]){ fields.MPICommDomain(no, positions[no], finalDerivedDomain, finalDerivedDomain, 0, false, sendDerivatives, sendDynamic);}
                                else{ fields.MPICommDomain(no, positions[no], standardDerivedDomain, standardDerivedDomain, 0, false, sendDerivatives, sendDynamic);}
                            }
                            save(tempFile);
                            if(parameters[3] == 1){
                                updateEnergy();
                                updateCharge();
                                plot(plotFile);
                            }
                        }

                    }

            }else{
                MPI_Comm COMM_GRID;
                MPI_Cart_create(COMM_NEW, dim, MPIdim, wrapAround, 1, &COMM_GRID);

                //communicate where you are in the grid to the master node
                int mypos[dim];/* get the position from the virtual topology */
                MPI_Send(&mypos,dim,MPI_INT, 0, 9, MPI_COMM_WORLD);
                MPI_Barrier(COMM_NEW);

                vector<int> derivativeDomain;
                vector<int> dataDomain;

                //now recieve your initial data
                fields.MPICommDomain(0,0,derivativeDomain,derivativeDomain,0,false,sendDerivatives,sendDynamic);

                //run main loop
                for(int loop = 0; loop < loops; loop++){
                    (this->*localFunction)(localIterations,1,1,false);
                    // parameters - 4 oftenBoundary
                    if(loop%parameters[4] == 0) {
                        sendRecvBoundaries(&COMM_GRID, &dataDomain, &derivativeDomain);
                    }
                }
            }

        }
        else{
            //create the transfer Domains for the fields and derived quantities (a vector containing all the values that are to be transfered between master and slave nodes)
            vector<int> dataDomain;
            vector<int> derivedDomain;
            int totalSize = domainSize[0];
            int corrector;
            for(int i = 1; i < domainSize.size(); i++){
                totalSize *= domainSize[i];
            }
            int point=0;
            vector<int> pos(dim);
            for(int j = 0; j < dim; j++){pos[j]=0;}
            for(int i = 0; i < totalSize; i++){
                    //porgress the pos value using the limiter of domain
                    if(i > 0){
                        addOne(pos,domainSize);
                    }
                    //calculate the corresponding 1-dim value (bear in mind the different limits!
                    point = pos[0];
                    int multiplier = 1;
                    for(int i=1; i<dim; i++){
                        multiplier *= size[i-1];
                        point += pos[i]*multiplier;
                    }
                    //add the points to the corresponding vectors
                    vector<int> temp_size = size;
                    size = domainSize;
                    if(inBoundary(pos)){
                        dataDomain.push_back(point);
                    }
                    size = temp_size;
                    derivedDomain.push_back(point);
                    corrector = point;
                }

            if(globalId == 0){
                std::random_device rd;
                std::mt19937 mt(rd());
                std::uniform_int_distribution<int> p_rand(0, getTotalSize()-corrector);
                int points[totalNo];
                int loop = 0;
                    #pragma omp parallel
                    while(loop < loops) {
                        #pragma omp for nowait
                        for (int no = 1; no < totalNo; no++) {
                            int updated;
                            MPI_Recv(&updated, 1, MPI_INT, no, loop, MPI_COMM_WORLD,
                                     MPI_STATUS_IGNORE);//recieve if the point has been updated
                            if (updated == 1) {
                                fields.MPICommDomain(no, points[no], dataDomain, derivedDomain, loop, false, sendDerivatives, sendDynamic);
                                MPICommEnergy(no, points[no], derivedDomain, loop, false);
                            }
                            int pos = p_rand(mt);// correct to right random no. generator
                            points[no] = pos;
                            fields.MPICommDomain(no, pos, derivedDomain, derivedDomain, loop, true, sendDerivatives, sendDynamic);
                            MPICommEnergy(no, pos, derivedDomain, loop, true);
                        }
                        #pragma omp barrier
                        #pragma omp master
                        {
                            loop++;
                        }
                    }
            }else{
                int updated = 0;
                for(int loop = 0; loop < loops; loop++){
                    MPI_Send(&updated,1,MPI_INT,0,loop,MPI_COMM_WORLD);
                    if(updated == 1){
                            fields.MPICommDomain(0,0,dataDomain,derivedDomain,loop,true,sendDerivatives,sendDynamic);
                            MPICommEnergy(0,0,derivedDomain,loop,true);
                    }
                    fields.MPICommDomain(0,0,derivedDomain,derivedDomain,loop,false,sendDerivatives, sendDynamic);
                    MPICommEnergy(0,0,derivedDomain,loop,false);
                    updated = (this->*localFunction)(localIterations, 1, 1, false);
                }
            }
        }


    }

    void BaseFieldTheory::printParameters(){
        for(int i = 0; i < parameters.size(); i++){
            cout << parameterNames[i] << ": " << *parameters[i] << " ";
        }
        cout << "\n";
    }

    void TargetSpace::randomise(int i, int field, vector<double> spacing, double dt, bool doubleDerivative){
        std::random_device rd;
        std::mt19937 mt(rd());
        if(field <= fields1.size()-1){
            Eigen::VectorXd value = fields1[field]->data[i];
            std::uniform_real_distribution<double> dist(-dt, dt);
            for(int j = 0; j < value.size(); j++){
                value[j] += dist(mt);
            }
            if(fields1[field]->normalised){
                value.normalize();
            }
            fields1[field]->alter_point(i,value,spacing,doubleDerivative);
        }
        else if(field <= fields1.size() + fields2.size() - 1){
            std::uniform_real_distribution<double> dist(fields2[field - fields1.size()]->min, fields2[field - fields1.size()]->max);
            double value = dist(mt);
            fields2[field - fields1.size()]->alter_point(i,value,spacing,doubleDerivative);
        }
        else if(field<= fields1.size() + fields2.size() +fields3.size() -1){
            std::uniform_int_distribution<int> dist(fields3[field - fields1.size() - fields2.size()]->min, fields3[field - fields1.size() - fields2.size()]->max);
            int value = dist(mt);
            fields3[field - fields1.size() - fields2.size()]->alter_point(i,value,spacing,doubleDerivative);
        }
        else{
            Eigen::MatrixXd value = fields4[field - fields1.size() - fields2.size()  - fields3.size()]->data[i];
            for(int j = 0; j < value.rows(); j++){
                for(int k = 0; k < value.cols(); k++){
                    std::uniform_real_distribution<double> dist(fields4[field - fields1.size() - fields2.size()  - fields3.size()]->min(j,k), fields4[field - fields1.size() - fields2.size()  - fields3.size()]->max(j,k));
                    value(j,k) = dist(mt);
            }}
            if(fields4[field - fields1.size() - fields2.size()  - fields3.size()]->normalised){
                value.normalize();
            }
            fields4[field - fields1.size() - fields2.size()  - fields3.size()]->alter_point(i,value,spacing,doubleDerivative);
        }
    }

    void TargetSpace::derandomise(int i, int field, vector<double> spacing,bool doubleDerivative){
        if(field <= fields1.size()-1){
                fields1[field]->alter_point(i,fields1[field]->buffer[i],spacing,doubleDerivative);
        }
        else if(field <= fields1.size() + fields2.size() - 1){
                int point = field - fields1.size();
                fields2[point]->alter_point(i,fields2[point]->buffer[i],spacing,doubleDerivative);
        }
        else if(field <= fields1.size() + fields2.size() +fields3.size() -1){
            int point = field - fields1.size() - fields2.size();
                fields3[point]->alter_point(i,fields3[point]->buffer[i],spacing,doubleDerivative);
        }
        else{
                int point = field - fields1.size() - fields2.size() - fields3.size();
                fields4[point]->alter_point(i,fields4[point]->buffer[i],spacing,doubleDerivative);
        }
    }



    void TargetSpace::storeDerivatives(BaseFieldTheory * theory,bool doubleDerivative){
        for(int i = 0; i < fields1.size(); i++){
            fields1[i]->single_derivatives.resize(theory->dim);
            if(doubleDerivative) {
                fields1[i]->double_derivatives.resize(theory->dim);
            }
            for(int j = 0; j < theory->dim ; j++) {
                fields1[i]->single_derivatives[j].resize(theory->getTotalSize());
                if(doubleDerivative) {
                    fields1[i]->double_derivatives[j].resize(theory->dim);
                    for (int k = j; k < theory->dim; k++) {
                        fields1[i]->double_derivatives[j][k].resize(theory->getTotalSize());
                    }
                }
            }
            for(int j = 0; j < theory->getTotalSize() ; j++) {
            if (theory->inBoundary(j)) {
                for (int wrt1 = 0; wrt1 < theory->dim; wrt1++) {
                    fields1[i]->single_derivatives[wrt1][j] = theory->single_derivative(fields1[i], wrt1, j);
                    if(doubleDerivative) {
                        for (int wrt2 = wrt1; wrt2 < theory->dim; wrt2++) {
                            fields1[i]->double_derivatives[wrt1][wrt2][j] = theory->double_derivative(fields1[i], wrt1,
                                                                                                      wrt2, j);
                        }
                    }
                }
            }else{
                for (int wrt1 = 0; wrt1 < theory->dim; wrt1++) {
                    fields1[i]->single_derivatives[wrt1][j] = Eigen::VectorXd::Zero(fields1[i]->data[0].size());
                    if(doubleDerivative) {
                        for (int wrt2 = wrt1; wrt2 < theory->dim; wrt2++) {
                            fields1[i]->double_derivatives[wrt1][wrt2][j] = Eigen::VectorXd::Zero(
                                    fields1[i]->data[0].size());
                        }
                    }
                }
            }
            }
        }
        for(int i = 0; i < fields2.size(); i++){
            fields2[i]->single_derivatives.resize(theory->dim);
            fields2[i]->double_derivatives.resize(theory->dim);
            for(int j = 0; j < theory->dim ; j++) {
                fields2[i]->single_derivatives[j].resize(theory->getTotalSize());
                fields2[i]->double_derivatives[j].resize(theory->dim);
                for(int k = j; k < theory->dim; k++ ) {
                    fields2[i]->double_derivatives[j][k].resize(theory->getTotalSize());
                }
            }
            for(int j = 0; j < theory->getTotalSize() ; j++) {
            if (theory->inBoundary(j)) {
                for(int wrt1 = 0; wrt1 < theory->dim ; wrt1++) {
                    fields2[i]->single_derivatives[wrt1][j]=theory->single_derivative(fields2[i],wrt1,j);
                    for(int wrt2 = wrt1; wrt2 < theory->dim ; wrt2++) {
                        fields2[i]->double_derivatives[wrt1][wrt2][j]=theory->double_derivative(fields2[i],wrt1,wrt2,j);
                    }
                }
            }else{
                for (int wrt1 = 0; wrt1 < theory->dim; wrt1++) {
                    fields2[i]->single_derivatives[wrt1][j] = 0.0;
                    for (int wrt2 = wrt1; wrt2 < theory->dim; wrt2++) {
                        fields2[i]->double_derivatives[wrt1][wrt2][j] = 0.0;
                    }
                }
            }
            }
        }
        for(int i = 0; i < fields3.size(); i++){
            fields3[i]->single_derivatives.resize(theory->dim);
            fields3[i]->double_derivatives.resize(theory->dim);
            for(int j = 0; j < theory->dim ; j++) {
                fields3[i]->single_derivatives[j].resize(theory->getTotalSize());
                fields3[i]->double_derivatives[j].resize(theory->dim);
                for(int k = j; k < theory->dim; k++ ) {
                    fields3[i]->double_derivatives[j][k].resize(theory->getTotalSize());
                }
            }
            for(int j = 0; j < theory->getTotalSize() ; j++) {
            if (theory->inBoundary(j)) {
                for(int wrt1 = 0; wrt1 < theory->dim ; wrt1++) {
                    fields3[i]->single_derivatives[wrt1][j]=theory->single_derivative(fields3[i],wrt1,j);
                    for(int wrt2 = wrt1; wrt2 < theory->dim ; wrt2++) {
                        fields3[i]->double_derivatives[wrt1][wrt2][j]=theory->double_derivative(fields3[i],wrt1,wrt2,j);
                    }
                }
            }else{
                for (int wrt1 = 0; wrt1 < theory->dim; wrt1++) {
                    fields3[i]->single_derivatives[wrt1][j] = 0;
                    for (int wrt2 = wrt1; wrt2 < theory->dim; wrt2++) {
                        fields3[i]->double_derivatives[wrt1][wrt2][j] = 0;
                    }
                }
            }
            }
        }
        for(int i = 0; i < fields4.size(); i++){
            fields4[i]->single_derivatives.resize(theory->dim);
            fields4[i]->double_derivatives.resize(theory->dim);
            for(int j = 0; j < theory->dim ; j++) {
                fields4[i]->single_derivatives[j].resize(theory->getTotalSize());
                fields4[i]->double_derivatives[j].resize(theory->dim);
                for(int k = j; k < theory->dim; k++ ) {
                    fields4[i]->double_derivatives[j][k].resize(theory->getTotalSize());
                }
            }
            for(int j = 0; j < theory->getTotalSize() ; j++) {
            if (theory->inBoundary(j)) {
                for(int wrt1 = 0; wrt1 < theory->dim ; wrt1++) {
                    fields4[i]->single_derivatives[wrt1][j]=theory->single_derivative(fields4[i],wrt1,j);
                    for(int wrt2 = wrt1; wrt2 < theory->dim ; wrt2++) {
                        fields4[i]->double_derivatives[wrt1][wrt2][j]=theory->double_derivative(fields4[i],wrt1,wrt2,j);
                    }
                }
            }else{
                for (int wrt1 = 0; wrt1 < theory->dim; wrt1++) {
                    fields4[i]->single_derivatives[wrt1][j] = Eigen::MatrixXd::Zero(fields4[i]->data[0].rows(),fields4[i]->data[0].cols());
                    for (int wrt2 = wrt1; wrt2 < theory->dim; wrt2++) {
                        fields4[i]->double_derivatives[wrt1][wrt2][j] = Eigen::MatrixXd::Zero(fields4[i]->data[0].rows(),fields4[i]->data[0].cols());
                    }
                }
            }
            }
        }
    }

    void TargetSpace::moveToBuffer(){
        for(int i = 0; i < fields1.size(); i++){
            fields1[i]->buffer = fields1[i]->data;
            if(fields1[i]->dynamic){fields1[i]->buffer_dt = fields1[i]->dt;}
        }
        for(int i = 0; i < fields2.size(); i++){
            fields2[i]->buffer = fields2[i]->data;
            if(fields2[i]->dynamic){fields2[i]->buffer_dt = fields2[i]->dt;}
        }
        for(int i = 0; i < fields3.size(); i++){
            fields3[i]->buffer = fields3[i]->data;
            if(fields3[i]->dynamic){fields3[i]->buffer_dt = fields3[i]->dt;}
        }
        for(int i = 0; i < fields4.size(); i++){
            fields4[i]->buffer = fields4[i]->data;
            if(fields4[i]->dynamic){fields4[i]->buffer_dt = fields4[i]->dt;}
        }

    }

    void TargetSpace::resetPointers(){
        for(int i = 0; i < fields1.size(); i++){
            fields1_pointers[i] = &fields1[i];
        }
        for(int i = 0; i < fields2.size(); i++){
            fields2_pointers[i] = &fields2[i];
        }
        for(int i = 0; i < fields3.size(); i++){
            fields3_pointers[i] = &fields3[i];
        }
        for(int i = 0; i < fields4.size(); i++){
            fields4_pointers[i] = &fields4[i];
        }
    }

void TargetSpace::cutKinetic(int pos){
    for(int i = 0; i < fields1.size(); i++){
        fields1[i]->dt[pos] = Eigen::VectorXd::Zero(fields1[i]->dt[pos].size());
    }
    for(int i = 0; i < fields2.size(); i++){
        fields2[i]->dt[pos] = 0.0;
    }
    for(int i = 0; i < fields3.size(); i++){
        fields3[i]->dt[pos] = 0;
    }
    for(int i = 0; i < fields4.size(); i++){

        fields4[i]->dt[pos] = Eigen::MatrixXd::Zero(fields4[i]->dt[pos].rows(), fields4[i]->dt[pos].cols());;
    }
}

void TargetSpace::normalise(){
    for(int i = 0; i < fields1.size(); i++){
        fields1[i]->normalise();
    }
    for(int i = 0; i < fields4.size(); i++){
        cout << "WARNING!!!!! currently normalisation is not correct for matricies, please correct this where this cout statement is in FieldTheories.hpp\n";
        cout << "It appears to be working for the annealong function or maybe a mistake was made???? - I would start by comparing the two function anyway :) - good luck future tom! \n";
        fields1[i]->normalise();
    }
}

TargetSpace::TargetSpace(){
    no_fields = 0;
}

void TargetSpace::resize(vector<int> sizein){
    for(int i = 0; i < fields1.size(); i++){
        fields1[i] = fields1[i]->resize(sizein);
    }
    for(int i = 0; i < fields2.size(); i++){
        fields2[i] = fields2[i]->resize(sizein);
    }
    for(int i = 0; i < fields3.size(); i++){
        fields3[i] = fields3[i]->resize(sizein);
    }
    for(int i = 0; i < fields4.size(); i++){
        fields4[i] = fields4[i]->resize(sizein);
    }
    resetPointers();
}

Field<Eigen::VectorXd> * TargetSpace::addField(int dim, vector<int> size, Field<Eigen::VectorXd> * target, bool isDynamic, bool isNormalised) {
    fields1.push_back(new Field < Eigen::VectorXd > (dim, size, isDynamic,isNormalised));
    fields1_pointers.push_back(&target);
    no_fields = no_fields + 1;
    return fields1[fields1.size() - 1];
}

Field<double> * TargetSpace::addField(int dim, vector<int> size, Field<double> * target, bool isDynamic, bool isNormalised) {
    if(isNormalised){cout << "Warning! isNormalised is set true for double field, this makes no sense so setting to false!\n";}
    fields2.push_back(new Field < double > (dim, size, isDynamic,false));
    fields2_pointers.push_back(&target);
    no_fields = no_fields + 1;
    return fields2[fields2.size() - 1];
}

Field<int> * TargetSpace::addField(int dim, vector<int> size, Field<int> * target, bool isDynamic, bool isNormalised) {
    if(isNormalised){cout << "Warning! isNormalised is set true for int field, this makes no sense so setting to false!\n";}
    fields3.push_back(new Field < int > (dim, size, isDynamic,false));
    fields3_pointers.push_back(&target);
    no_fields = no_fields + 1;
    return fields3[fields3.size() - 1];
}

Field<Eigen::MatrixXd> * TargetSpace::addField(int dim, vector<int> size, Field<Eigen::MatrixXd> * target, bool isDynamic, bool isNormalised) {
    fields4.push_back(new Field < Eigen::MatrixXd > (dim, size, isDynamic,isNormalised));
    fields4_pointers.push_back(&target);
    no_fields = no_fields + 1;
    return fields4[fields4.size() - 1];
}



    void TargetSpace::save_fields(ofstream& savefile, int totalSize){
        for(int i = 0; i < fields1.size(); i++){
            savefile << fields1[i]->data[0].size();
            savefile << "\n";
            for(int j =0; j < totalSize; j++) {
                for(int k = 0; k < fields1[i]->data[j].size(); k++) {
                    savefile << fields1[i]->data[j](k) << " ";
                }
                savefile << "\n";
            }
        }
        for(int i = 0; i < fields2.size(); i++){
            for(int j =0; j < totalSize; j++) {
                savefile << fields2[i]->data[j] << "\n";
            }
        }
        for(int i = 0; i < fields3.size(); i++){
            for(int j =0; j < totalSize; j++) {
                savefile << fields3[i]->data[j] << "\n";
            }
        }
        for(int i = 0; i < fields4.size(); i++){
            savefile << fields4[i]->data[0].rows();
            savefile << fields4[i]->data[0].cols();
            savefile << "\n";
            for(int j =0; j < totalSize; j++) {
                for(int k = 0; k < fields4[i]->data[j].rows(); k++) {
                for(int h = 0; h < fields4[i]->data[j].cols(); h++){
                    savefile << fields4[i]->data[j](k,h) << " ";
                }}
                savefile << "\n";
            }
        }
    }

    void TargetSpace::load_fields(ifstream& loadfile, int totalSize){
        for(int i = 0; i < fields1.size(); i++){
            int target_dim;
            loadfile >> target_dim;
            for(int j =0; j < totalSize; j++){
                fields1[i]->data[j].resize(target_dim);
                for(int k = 0; k < target_dim; k++){
                   loadfile >> fields1[i]->data[j](k);
                }
            }
        }
        for(int i = 0; i < fields2.size(); i++){
            for(int j =0; j < totalSize; j++){
                loadfile >> fields2[i]->data[j];
            }
        }
        for(int i = 0; i < fields3.size(); i++){
            for(int j =0; j < totalSize; j++){
                loadfile >> fields3[i]->data[j];
            }
        }
        for(int i = 0; i < fields4.size(); i++){
            int target_rows,target_cols;
            loadfile >> target_rows;
            loadfile >> target_cols;
            for(int j =0; j < totalSize; j++) {
                fields4[i]->data[j].resize(target_rows,target_cols);
                for(int k = 0; k < fields4[i]->data[j].rows(); k++){
                for(int h = 0; h < fields4[i]->data[j].cols(); h++){
                    loadfile >> fields4[i]->data[j](k,h);
                }}
            }
        }
    }

    void TargetSpace::update_fields(){
        for(int i = 0; i < fields1.size(); i++){
            fields1[i]->update_field();
        }
        for(int i = 0; i < fields2.size(); i++){
            fields2[i]->update_field();
        }
        for(int i = 0; i < fields3.size(); i++){
            fields3[i]->update_field();
        }
        for(int i = 0; i < fields4.size(); i++){
            fields4[i]->update_field();
        }
    }

    void TargetSpace::update_gradients(double dt){
        for(int i = 0; i < fields1.size(); i++){
            fields1[i]->update_gradient(dt);
        }
        for(int i = 0; i < fields2.size(); i++){
            fields2[i]->update_gradient(dt);
        }
        for(int i = 0; i < fields3.size(); i++){
            fields3[i]->update_gradient(dt);
        }
        for(int i = 0; i < fields4.size(); i++){
            fields4[i]->update_gradient(dt);
        }
    }

    void TargetSpace::update_RK4(int k, int pos){
        for(int i = 0; i < fields1.size(); i++){
            if(fields1[i]->dynamic){fields1[i]->updateRK4(k, pos);}
        }
        for(int i = 0; i < fields2.size(); i++){
            if(fields2[i]->dynamic){fields2[i]->updateRK4(k, pos);}
        }
        for(int i = 0; i < fields3.size(); i++){
            if(fields3[i]->dynamic){fields3[i]->updateRK4(k, pos);}
        }
        for(int i = 0; i < fields4.size(); i++){
            if(fields4[i]->dynamic){fields4[i]->updateRK4(k, pos);}
        }
    }
/***********************/
/*   Field Theories    */
/***********************/

    double BaseFieldTheory::metric(int i, int j, vector<double> pos){
        Eigen::MatrixXd g;
        double gtt;
       /* switch(metric_type) {
            case 0:
                if (i == j) { return 1.0; }
                else { return 0.0; };
            case 1: return 0.0;
            case 2: return 0.0;
            case 3: return 0.0;
            case 4:
                double r = 0.0;
                for (int i = 0; i < dim; i++) {
                    r += pos[i] * pos[i];
                }
                r = sqrt(r);
                r = 4.0 / (1.0 - r);
                if (i == j) {return r;} else {return 0.0;};
            case 5:
                double r = 0.0;
                for (int i = 0; i < dim; i++) {
                    r += pos[i] * pos[i];
                }
                r = sqrt(r);
                r = 4.0 / (1.0 - r);
                if (i == j) { return r; } else { return 0.0; };
        }*/
        return 0.0;
    }

    void BaseFieldTheory::setMetricType(string type){
        if(type == "flat"){metric_type = 0; curved = false;}
        else if(type == "polar"){metric_type = 1; curved = true;}
        else if(type == "spherical"){metric_type = 2; curved = true;}
        else if(type == "cylindrical"){metric_type = 3; curved = true;}
        else if(type == "poincare"){metric_type = 4; curved = true;}
        else if(type == "ADS"){metric_type = 5; curved = true;}
        else{cout << "WARNING!! " << type << " is not a recognised metric type, please specific another or add the details to ""setMetricType"" and ""Metric"" functions!\n";
        }
    }

    void BaseFieldTheory::addParameter(double * parameter_in, string name){
        parameters.push_back(parameter_in);
        parameterNames.push_back(name);
    }

    int BaseFieldTheory::periodicindex(int index){
        if(index < 0){return index + getTotalSize();}
        else if(index >= getTotalSize()){return index - getTotalSize();}
        else{return index;}
    }

bool BaseFieldTheory::annealing(int iterations, int often, int often_cut, bool output){
    bool altered = false;
    if(dynamic && output){
        cout << "Warning! You have run annealing on a dynamic theory, this will kill the kinetic componenet!\n";
        for(int i = 0; i < getTotalSize(); i++){
            fields.cutKinetic(i);
        }
    }
    double check_energy = -1.0;
    int cut_no = 0;
    int seperator = 2*getTotalSize()/size[dim-1];
    if(!setDerivatives) {
        fields.storeDerivatives(this,false);
        setDerivatives = true;
        updateEnergy();
    }
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<int> f_rand(0, fields.no_fields-1);
    int no_total = 0;
    int no = 0;
    #pragma omp parallel
    {
        int sep = getTotalSize()/omp_get_num_threads();
        double newEnergy;
        double oldEnergy;
        double newEnergyDensity[1+4*dim+16*(dim-1)*(dim-1)];
        int store = 0;
        int area_start = omp_get_thread_num()*sep;
        int area_end = omp_get_thread_num()*sep + sep - 1;
        int area_length = area_end - area_start;
        std::uniform_int_distribution<int> p_rand_left(area_start,area_start + area_length/2);
        std::uniform_int_distribution<int> p_rand_right(area_start + area_length/2,area_end);
        while (no_total<iterations) {
            int pos = -1;
            if(no_total%2 == 0) {
                while (!inBoundary(pos)) {
                    pos = p_rand_left(mt);// correct to right random no. generator
                }
            }else{
                while (!inBoundary(pos)) {
                    pos = p_rand_right(mt);// correct to right random no. generator
                }
            }
            int field = f_rand(mt); // correct for int random no. generator
            fields.randomise(pos, field, spacing, dt);
            newEnergy = calculateEnergy(pos);
            oldEnergy = energydensity[pos];
            int store = 0;
            newEnergyDensity[store] = newEnergy;
            double mult1 = 1.0;
            for (int i = 0; i < dim; i++) {
                store++;
                int point = pos + mult1;
                if (inBoundary(point)||!output) {
                    newEnergyDensity[store] = calculateEnergy(point);
                    newEnergy += newEnergyDensity[store];
                    oldEnergy += energydensity[point];
                }
                store++;
                point = pos - mult1;
                if (inBoundary(point)||!output) {
                    newEnergyDensity[store] = calculateEnergy(point);
                    newEnergy += newEnergyDensity[store];
                    oldEnergy += energydensity[point];
                }
                store++;
                point = pos + 2*mult1;
                if (inBoundary(point)||!output) {
                    newEnergyDensity[store] = calculateEnergy(point);
                    newEnergy += newEnergyDensity[store];
                    oldEnergy += energydensity[point];
                }
                store++;
                point = pos - 2*mult1;
                if (inBoundary(point)||!output) {
                    newEnergyDensity[store] = calculateEnergy(point);
                    newEnergy += newEnergyDensity[store];
                    oldEnergy += energydensity[point];
                }
                double mult2 = 1.0;
                for (int j = 0; j < i; j++) {
                    store++;
                    point = pos + mult1 + mult2;
                    if (inBoundary(point)||!output) {
                        newEnergyDensity[store] = calculateEnergy(point);
                        newEnergy += newEnergyDensity[store];
                        oldEnergy += energydensity[point];
                    }
                    store++;
                    point = pos + 2*mult1 + mult2;
                    if (inBoundary(point)||!output) {
                        newEnergyDensity[store] = calculateEnergy(point);
                        newEnergy += newEnergyDensity[store];
                        oldEnergy += energydensity[point];
                    }
                    store++;
                    point = pos + mult1 + 2*mult2;
                    if (inBoundary(point)||!output) {
                        newEnergyDensity[store] = calculateEnergy(point);
                        newEnergy += newEnergyDensity[store];
                        oldEnergy += energydensity[point];
                    }
                    store++;
                    point = pos + 2*mult1 + 2*mult2;
                    if (inBoundary(point)||!output) {
                        newEnergyDensity[store] = calculateEnergy(point);
                        newEnergy += newEnergyDensity[store];
                        oldEnergy += energydensity[point];
                    }

                    store++;
                    point = pos - mult1 - mult2;
                    if (inBoundary(point)||!output) {
                        newEnergyDensity[store] = calculateEnergy(point);
                        newEnergy += newEnergyDensity[store];
                        oldEnergy += energydensity[point];
                    }
                    store++;
                    point = pos - 2*mult1 - mult2;
                    if (inBoundary(point)||!output) {
                        newEnergyDensity[store] = calculateEnergy(point);
                        newEnergy += newEnergyDensity[store];
                        oldEnergy += energydensity[point];
                    }
                    store++;
                    point = pos - mult1 - 2*mult2;
                    if (inBoundary(point)||!output) {
                        newEnergyDensity[store] = calculateEnergy(point);
                        newEnergy += newEnergyDensity[store];
                        oldEnergy += energydensity[point];
                    }
                    store++;
                    point = pos - 2*mult1 - 2*mult2;
                    if (inBoundary(point)||!output) {
                        newEnergyDensity[store] = calculateEnergy(point);
                        newEnergy += newEnergyDensity[store];
                        oldEnergy += energydensity[point];
                    }
                    if(i != j) {
                        store++;
                        point = pos + mult1 - mult2;
                        if (inBoundary(point)||!output) {
                            newEnergyDensity[store] = calculateEnergy(point);
                            newEnergy += newEnergyDensity[store];
                            oldEnergy += energydensity[point];
                        }
                        store++;
                        point = pos + 2*mult1 - mult2;
                        if (inBoundary(point)||!output) {
                            newEnergyDensity[store] = calculateEnergy(point);
                            newEnergy += newEnergyDensity[store];
                            oldEnergy += energydensity[point];
                        }
                        store++;
                        point = pos + mult1 - 2*mult2;
                        if (inBoundary(point)||!output) {
                            newEnergyDensity[store] = calculateEnergy(point);
                            newEnergy += newEnergyDensity[store];
                            oldEnergy += energydensity[point];
                        }
                        store++;
                        point = pos + 2*mult1 - 2*mult2;
                        if (inBoundary(point)||!output) {
                            newEnergyDensity[store] = calculateEnergy(point);
                            newEnergy += newEnergyDensity[store];
                            oldEnergy += energydensity[point];
                        }



                        store++;
                        point = pos - mult1 + mult2;
                        if (inBoundary(point)||!output) {
                            newEnergyDensity[store] = calculateEnergy(point);
                            newEnergy += newEnergyDensity[store];
                            oldEnergy += energydensity[point];
                        }
                        store++;
                        point = pos - 2*mult1 + mult2;
                        if (inBoundary(point)||!output) {
                            newEnergyDensity[store] = calculateEnergy(point);
                            newEnergy += newEnergyDensity[store];
                            oldEnergy += energydensity[point];
                        }
                        store++;
                        point = pos - mult1 + 2*mult2;
                        if (inBoundary(point)||!output) {
                            newEnergyDensity[store] = calculateEnergy(point);
                            newEnergy += newEnergyDensity[store];
                            oldEnergy += energydensity[point];
                        }
                        store++;
                        point = pos - 2*mult1 + 2*mult2;
                        if (inBoundary(point)||!output) {
                            newEnergyDensity[store] = calculateEnergy(point);
                            newEnergy += newEnergyDensity[store];
                            oldEnergy += energydensity[point];
                        }
                    }
                    mult2 *= size[j];
                }
            mult1 *= size[i];
            }
            if(store >= 1+4*dim+16*(dim-1)*(dim-1)){cout << "ERRROR! - missmmatch of sizes for newEnergyDensity array to value of point variable!!!\n";}
            if (newEnergy > oldEnergy) { // add some heat term! :) - as well as some fall off
                fields.derandomise(pos, field, spacing);
            } else {
                altered = true;
                store = 0;
                energydensity[pos] = newEnergyDensity[store];
                double mult1 = 1.0;
                for (int i = 0; i < dim; i++) {
                    store++;
                    int point = pos + mult1;
                    if (inBoundary(point)||!output) {
                        energydensity[point] = newEnergyDensity[store];
                    }
                    store++;
                    point = pos - mult1;
                    if (inBoundary(point)||!output) {
                        energydensity[point] = newEnergyDensity[store];
                    }
                    store++;
                    point = pos + 2*mult1;
                    if (inBoundary(point)||!output) {
                        energydensity[point] = newEnergyDensity[store];
                    }
                    store++;
                    point = pos - 2*mult1;
                    if (inBoundary(point)||!output) {
                        energydensity[point] = newEnergyDensity[store];
                    }
                    double mult2 = 1.0;
                    for (int j = 0; j < i; j++) {
                        store++;
                        point = pos + mult1 + mult2;
                        if (inBoundary(point)||!output) {
                            energydensity[point] = newEnergyDensity[store];
                        }
                        store++;
                        point = pos + 2*mult1 + mult2;
                        if (inBoundary(point)||!output) {
                            energydensity[point] = newEnergyDensity[store];
                        }
                        store++;
                        point = pos + mult1 + 2*mult2;
                        if (inBoundary(point)||!output) {
                            energydensity[point] = newEnergyDensity[store];
                        }
                        store++;
                        point = pos + 2*mult1 + 2*mult2;
                        if (inBoundary(point)||!output) {
                            energydensity[point] = newEnergyDensity[store];
                        }

                        store++;
                        point = pos - mult1 - mult2;
                        if (inBoundary(point)||!output) {
                            energydensity[point] = newEnergyDensity[store];
                        }
                        store++;
                        point = pos - 2*mult1 - mult2;
                        if (inBoundary(point)||!output) {
                            energydensity[point] = newEnergyDensity[store];
                        }
                        store++;
                        point = pos - mult1 - 2*mult2;
                        if (inBoundary(point)||!output) {
                            energydensity[point] = newEnergyDensity[store];
                        }
                        store++;
                        point = pos - 2*mult1 - 2*mult2;
                        if (inBoundary(point)||!output) {
                            energydensity[point] = newEnergyDensity[store];
                        }
                        if(i != j) {
                            store++;
                            point = pos + mult1 - mult2;
                            if (inBoundary(point)||!output) {
                                energydensity[point] = newEnergyDensity[store];
                            }
                            store++;
                            point = pos + 2*mult1 - mult2;
                            if (inBoundary(point)||!output) {
                                energydensity[point] = newEnergyDensity[store];
                            }
                            store++;
                            point = pos + mult1 - 2*mult2;
                            if (inBoundary(point)||!output) {
                                energydensity[point] = newEnergyDensity[store];
                            }
                            store++;
                            point = pos + 2*mult1 - 2*mult2;
                            if (inBoundary(point)||!output) {
                                energydensity[point] = newEnergyDensity[store];
                            }



                            store++;
                            point = pos - mult1 + mult2;
                            if (inBoundary(point)||!output) {
                                energydensity[point] = newEnergyDensity[store];
                            }
                            store++;
                            point = pos - 2*mult1 + mult2;
                            if (inBoundary(point)||!output) {
                                energydensity[point] = newEnergyDensity[store];
                            }
                            store++;
                            point = pos - mult1 + 2*mult2;
                            if (inBoundary(point)||!output) {
                                energydensity[point] = newEnergyDensity[store];
                            }
                            store++;
                            point = pos - 2*mult1 + 2*mult2;
                            if (inBoundary(point)||!output) {
                                energydensity[point] = newEnergyDensity[store];
                            }
                        }
                        mult2 *= size[j];
                    }
                    mult1 *= size[i];
                }

                }

            if (no % often == 0) {
                #pragma omp barrier
                #pragma omp master
                {
                    no_total++;
                    no=0;
                    if(output) {
                        updateEnergy();
                        updateCharge();
                        cout << no_total << ": energy = " << energy << " : dt = " << dt << " charge = " << charge
                             << "\n";
                        save("temp_field");

                        if (energy == check_energy) {
                            cut_no++;
                            if (cut_no % often_cut == 0) {
                                dt = 0.9 * dt;
                                cut_no = 0;
                            }
                        } else {
                            cut_no = 0;
                            check_energy = energy;
                        }
                    }
                }
                #pragma omp barrier
                }
                #pragma omp barrier
                #pragma omp master
                {
                no++;
                }
        }
    }
    if(output) {
        setDerivatives = false;
    }
    return altered;
}

inline vector<double> calculateDynamicEnergy(int pos){
    cout << " the calculate Dynamic Energy function has not been set in the Derived class!\n";
}


    void BaseFieldTheory::RK4(int iterations, bool cutEnergy, int often){
        double time = 0.0;
        double newenergy = 0.0;
        double sum;
        #pragma omp parallel
        for(int no = 0; no < iterations; no++){
            time += dt;
            #pragma omp master
            {
                for (int i = 0; i < fields.no_fields; i++) {
                    fields.moveToBuffer();
                }
            }
            for(int k = 0; k < 4; k++) {
                #pragma omp for nowait
                for (int i = 0; i < getTotalSize(); i++) {
                    RK4calc(i);
                }
                #pragma omp barrier
                #pragma omp for nowait
                for (int i = 0; i < getTotalSize(); i++) {
                        fields.update_RK4(k,i);
                }
            }
            #pragma omp master
            {
                    fields.normalise();
            }
            if(cutEnergy){
                sum = 0.0;
                #pragma omp barrier
                #pragma omp for nowait reduction(+:sum)
                for (int i = 0; i < getTotalSize(); i++) {
                    if (inBoundary(i)) {
                        double buffer = calculateEnergy(i); // change all references to calculatePotential
                        sum += buffer;
                    }
                }
                #pragma omp barrier
                #pragma omp master
                {
                    newenergy = sum * getTotalSpacing();
                }
                #pragma omp barrier
                    if (newenergy >= potential + 0.0000001) {
                        #pragma omp for nowait
                        for (int i = 0; i < getTotalSize(); i++) {
                            fields.cutKinetic(i);
                        }
                    }
                    #pragma omp barrier
                    #pragma omp master
                    {
                        potential = newenergy;
                    }
            }
            #pragma omp barrier
            #pragma omp master
            if(no%often == 0){
                updateEnergy();
                cout << "time " << time << " : Energy = " << energy << " Kinetic = " << kinetic << " Potential = " << potential <<"\n";
            }
        }
    }

    inline bool BaseFieldTheory::inBoundary(vector<int> pos){
        bool value = true;
        for(int i = 0; i < dim; i++){
            if(pos[i] < bdw[2*i] || pos[i] >= size[i]-bdw[2*i+1]){
                value = false;
            }
        }
        return value;
    }

    inline bool BaseFieldTheory::inBoundary(int pos){
        return inBoundary(convert(pos));
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
BaseFieldTheory::BaseFieldTheory(int d, vector<int> sizein, bool isDynamic): dim(d), size(sizein){
        dynamic = isDynamic;
        energydensity.resize(getTotalSize());
        chargedensity.resize(getTotalSize());
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
Field<T> * BaseFieldTheory::createField(Field<T> * target, bool isDynamic, bool isNormalised){
    dynamic = isDynamic;
    return fields.addField(dim, size, target, isDynamic, isNormalised);
};

int BaseFieldTheory::getTotalSize(){
    int total = size[0];
    for(int i=1; i<dim; i++){
        total *= size[i];
    }
    return total;
}

    double BaseFieldTheory::getTotalSpacing(){
        double total = spacing[0];
        for(int i=1; i<dim; i++){
            total *= spacing[i];
        }
        return total;
    }

/* --- Derivatives --- */
template <class T>
inline T  BaseFieldTheory::single_time_derivative(Field<T> * f, int wrt, int &point) {

    int mult=1;
    for(int i=1; i<=wrt; i++){
        mult *= size[i-1];
    }
    return (-1.0*f->dt[point+2*mult] + 8.0*f->dt[point+mult] - 8.0*f->dt[point-mult] + f->dt[point-2*mult])/(12.0*spacing[wrt]);
    }


template <class T>
inline T  BaseFieldTheory::single_derivative(Field<T> * f, int wrt, int &point) {
    if(setDerivatives) {
        return f->single_derivatives[wrt][point];
    }else {
        int mult = 1;
        for (int i = 1; i <= wrt; i++) {
            mult *= size[i - 1];
        }
        int dpos = convert(point)[wrt];
        if((boundarytype[2*wrt] == 0 && boundarytype[2*wrt+1] == 0) || (dpos+2 < size[wrt] && dpos-2 >= 0) ) {
            return (-1.0 * f->data[point + 2 * mult] + 8.0 * f->data[point + mult] - 8.0 * f->data[point - mult] +
                    f->data[point - 2 * mult]) / (12.0 * spacing[wrt]);
        }else if(f->boundarytype[2*wrt] == 1) { //periodic conditions
                if(dpos - 2 <0){
                    if(dpos - 1 <0){
                        return (-1.0 * f->data[point + 2 * mult] + 8.0 * f->data[point + mult] - 8.0 * f->data[point + (size[wrt]-1)*mult] +
                                f->data[point + (size[wrt]-2)*mult]) / (12.0 * spacing[wrt]);
                    }else{
                        return (-1.0 * f->data[point + 2 * mult] + 8.0 * f->data[point + mult] - 8.0 * f->data[point - mult] +
                                f->data[point + (size[wrt]-2)*mult]) / (12.0 * spacing[wrt]);
                    }
                }
                else{
                    if(dpos + 1 >= size[wrt]){
                        return (-1.0 * f->data[point - (size[wrt]-2)*mult] + 8.0 * f->data[point - (size[wrt]-1)*mult] - 8.0 * f->data[point - mult] +
                                f->data[point - 2 * mult]) / (12.0 * spacing[wrt]);
                    }else{
                        return (-1.0 * f->data[point - (size[wrt]-2)*mult] + 8.0 * f->data[point + mult] - 8.0 * f->data[point - mult] +
                                f->data[point - 2 * mult]) / (12.0 * spacing[wrt]);
                    }
                }
            }
            else if(f->boundarytype[2*wrt] == 4){ //periodic and transform in 2d conditions
                int wind_dir;
                double windpos;
                if(wrt == 0){windpos = convert(point)[1];wind_dir = 1;}
                else{windpos = convert(point)[0];wind_dir = 0;}
            double winding;
            if(f->boundarymorphtype==1){winding = f->getBoundaryConstant()*2.0*M_PI/(spacing[wind_dir]*(size[wind_dir]));}
            else {winding = f->getBoundaryConstant()*2.0*M_PI*windpos/(size[wind_dir]);}
                if(dpos - 2 <0){
                    if(dpos - 1 <0){
                        return (-1.0 * f->data[point + 2 * mult] + 8.0 * f->data[point + mult] - 8.0 * f->rotate(point + (size[wrt]-1)*mult,-winding) +
                            f->rotate(point + (size[wrt]-2)*mult,-winding)) / (12.0 * spacing[wrt]);
                    }else{
                        return (-1.0 * f->data[point + 2 * mult] + 8.0 * f->data[point + mult] - 8.0 * f->data[point - mult] +
                            f->rotate(point + (size[wrt]-2)*mult,-winding)) / (12.0 * spacing[wrt]);
                    }
                }
                else{
                    if(dpos + 1 >= size[wrt]){
                        return (-1.0 * f->rotate(point - (size[wrt]-2)*mult,winding) + 8.0 * f->rotate(point - (size[wrt]-1)*mult,winding) - 8.0 * f->data[point - mult] +
                            f->data[point - 2 * mult]) / (12.0 * spacing[wrt]);
                    }else{
                        return (-1.0 * f->rotate(point - (size[wrt]-2)*mult,winding) + 8.0 * f->data[point + mult] - 8.0 * f->data[point - mult] +
                            f->data[point - 2 * mult]) / (12.0 * spacing[wrt]);
                    }
                }

            }
            else{return 0.0*f->data[point];}
        }
    }
    /*for(int i = 0; i < dim; i++){ if(i == wrt){dir[i] = 1;}else{dir[i] = 0;}};
    return (-1.0*f->getData(pos+2*dir) + 8.0*f->getData(pos+dir) - 8.0*f->getData(pos-dir) + f->getData(pos-2*dir))/(12.0*spacing[wrt]);*/

template <class T>
inline T  BaseFieldTheory::double_derivative(Field<T> * f, int wrt1, int wrt2, int &point) {
    if(setDerivatives){
        if(wrt1 > wrt2) {
            return f->double_derivatives[wrt2][wrt1][point];
        }else{
            return f->double_derivatives[wrt1][wrt2][point];
        }
    }else {
        if (wrt1 == wrt2) {
            int mult = 1;
            for (int i = 1; i <= wrt1; i++) {
                mult *= size[i - 1];
            }
            int dpos = convert(point)[wrt1];
            if((boundarytype[2*wrt1] == 0 && boundarytype[2*wrt1+1] == 0) || (dpos+2 < size[wrt1] && dpos-2 >= 0) ) {
                return (-1.0 * f->data[point + 2 * mult] + 16.0 * f->data[point + mult] - 30.0 * f->data[point] +
                        16.0 * f->data[point - mult]
                        - f->data[point - 2 * mult]) / (12.0 * spacing[wrt1] * spacing[wrt1]);
            }else if(f->boundarytype[2*wrt1] == 1){ //periodic conditions
                    if(dpos - 2 <0){
                        if(dpos - 1 <0){
                            return (-1.0 * f->data[point + 2 * mult] + 16.0 * f->data[point + mult] - 30.0 * f->data[point] +
                                    16.0 * f->data[point + (size[wrt1]-1)*mult] - f->data[point + (size[wrt1]-2)*mult]) / (12.0 * spacing[wrt1] * spacing[wrt1]);
                        }else{
                            return (-1.0 * f->data[point + 2 * mult] + 16.0 * f->data[point + mult] - 30.0 * f->data[point] +
                                    16.0 * f->data[point - mult] - f->data[point + (size[wrt1]-2)*mult]) / (12.0 * spacing[wrt1] * spacing[wrt1]);
                        }
                    }
                    else{
                        if(dpos + 1 >= size[wrt1]){
                            return (-1.0 * f->data[point - (size[wrt1]-2)*mult] + 16.0 * f->data[point - (size[wrt1]-1)*mult] - 30.0 * f->data[point] +
                                    16.0 * f->data[point - mult] - f->data[point - 2 * mult]) / (12.0 * spacing[wrt1] * spacing[wrt1]);
                        }else{
                            return (-1.0 * f->data[point - (size[wrt1]-2)*mult] + 16.0 * f->data[point + mult] - 30.0 * f->data[point] +
                                    16.0 * f->data[point - mult] - f->data[point - 2 * mult]) / (12.0 * spacing[wrt1] * spacing[wrt1]);
                        }
                    }
                }
            else if(f->boundarytype[2*wrt1] == 4){ //periodic conditions with winding in 2d
                int wind_dir;
                double windpos;
                if(wrt1 == 0){windpos = convert(point)[1];wind_dir = 1;}
                else{windpos = 0.0;}//convert(point)[0];wind_dir = 0;}
                double winding;
                if(f->boundarymorphtype==1){winding = f->getBoundaryConstant()*2.0*M_PI/(spacing[wind_dir]*(size[wind_dir]));}
                else {winding = f->getBoundaryConstant()*2.0*M_PI*windpos/(size[wind_dir]);}
                if(dpos - 2 <0){
                    if(dpos - 1 <0){
                        return (-1.0 * f->data[point + 2 * mult] + 16.0 * f->data[point + mult] - 30.0 * f->data[point] +
                                16.0 * f->rotate(point + (size[wrt1]-1)*mult,-winding) - f->rotate(point + (size[wrt1]-2)*mult,-winding)) / (12.0 * spacing[wrt1] * spacing[wrt1]);
                    }else{
                        return (-1.0 * f->data[point + 2 * mult] + 16.0 * f->data[point + mult] - 30.0 * f->data[point] +
                                16.0 * f->data[point - mult] - f->rotate(point + (size[wrt1]-2)*mult,-winding)) / (12.0 * spacing[wrt1] * spacing[wrt1]);
                    }
                }
                else{
                    if(dpos + 1 >= size[wrt1]){
                        return (-1.0 * f->rotate(point - (size[wrt1]-2)*mult,winding) + 16.0 * f->rotate(point - (size[wrt1]-1)*mult,winding) - 30.0 * f->data[point] +
                                16.0 * f->data[point - mult] - f->data[point - 2 * mult]) / (12.0 * spacing[wrt1] * spacing[wrt1]);
                    }else{
                        return (-1.0 * f->rotate(point - (size[wrt1]-2)*mult,winding) + 16.0 * f->data[point + mult] - 30.0 * f->data[point] +
                                16.0 * f->data[point - mult] - f->data[point - 2 * mult]) / (12.0 * spacing[wrt1] * spacing[wrt1]);
                    }
                }
            }
                else{return 0.0*f->data[point];}

            /*for(int i = 0; i < dim; i++){ if(i == wrt1){dir[i] = 1;}else{dir[i] = 0;}};
            return (-1.0*f->getData(pos+2*dir) + 16.0*f->getData(pos+dir) - 30.0*f->getData(pos) + 16.0*f->getData(pos-dir)
                    - f->getData(pos-2*dir))/(12.0*spacing[wrt1]*spacing[wrt1]);*/
        } else {
            int mult1 = 1;
            int mult2 = 1;
            if (wrt1 > wrt2) {
                for (int i = 1; i <= wrt1; i++) {
                    mult1 *= size[i - 1];
                    if (i == wrt2) { mult2 = mult1; }
                }
            } else {
                for (int i = 1; i <= wrt2; i++) {
                    mult2 *= size[i - 1];
                    if (i == wrt1) { mult1 = mult2; }
                }
            }
            int dpos1 = convert(point)[wrt1];
            int dpos2 = convert(point)[wrt2];

            int doub = mult1 + mult2;
            int alt = mult1 - mult2;
            if((boundarytype[2*wrt1] == 0 && boundarytype[2*wrt1+1] == 0 && boundarytype[2*wrt2]==0 && boundarytype[2*wrt2+1]==0) || (dpos1+2 < size[wrt1] && dpos1-2 >= 0 && dpos2+2 < size[wrt2] && dpos2-2 >= 0) )
            {
                return (f->data[point + 2 * doub] - 8.0 * f->data[point + mult1 + 2 * mult2] +
                        8.0 * f->data[point - mult1 + 2 * mult2]
                        - f->data[point - 2 * alt] - 8.0 * f->data[point + 2 * mult1 + mult2] +
                        64.0 * f->data[point + doub]
                        - 64.0 * f->data[point - alt] + 8.0 * f->data[point - 2 * mult1 + mult2] +
                        8.0 * f->data[point + 2 * mult1 - mult2]
                        - 64.0 * f->data[point + alt] + 64.0 * f->data[point - doub] -
                        8.0 * f->data[point - 2 * mult1 - mult2]
                        - f->data[point + 2 * alt] + 8.0 * f->data[point + mult1 - 2 * mult2] -
                        8.0 * f->data[point - mult1 - 2 * mult2]
                        + f->data[point - 2 * doub]) / (144.0 * spacing[wrt1] * spacing[wrt2]);
            }
            else if(f->boundarytype[2 * wrt1] != 4 && f->boundarytype[2 * wrt2] != 4){
                double mult1m1, mult1m2, mult1p1, mult1p2, mult2m1, mult2m2, mult2p1, mult2p2;
                if ((f->boundarytype[2 * wrt1] == 1 && dpos1 - 2 < 0) ||
                    (f->boundarytype[2 * wrt1 + 1] == 1 && dpos1 + 2 >= size[wrt1])) { //periodic conditions for wrt1
                    if (dpos1 - 1 < 0) {
                        mult1m1 = (size[wrt1] - 1) * mult1;
                        mult1m2 = (size[wrt1] - 2) * mult1;
                        mult1p1 = mult1;
                        mult1p2 = 2 * mult1;
                    } else if (dpos1 - 2 < 0) {
                        mult1m1 = -mult1;
                        mult1m2 = (size[wrt1] - 2) * mult1;
                        mult1p1 = mult1;
                        mult1p2 = 2 * mult1;
                    } else if (dpos1 + 1 >= size[wrt1]) {
                        mult1m1 = -mult1;
                        mult1m2 = -2 * mult1;
                        mult1p1 = -(size[wrt1] - 1) * mult1;
                        mult1p2 = -(size[wrt1] - 2) * mult1;
                    } else if (dpos1 + 2 >= size[wrt1]) {
                        mult1m1 = -mult1;
                        mult1m2 = -2 * mult1;
                        mult1p1 = mult1;
                        mult1p2 = -(size[wrt1] - 2) * mult1;
                    }
                } else {
                    mult1m1 = -mult1;
                    mult1m2 = -2 * mult1;
                    mult1p1 = mult1;
                    mult1p2 = 2 * mult1;
                }
                if ((f->boundarytype[2 * wrt2] == 1 && dpos2 - 2 < 0) ||
                    (f->boundarytype[2 * wrt2 + 1] == 1 && dpos2 + 2 >= size[wrt2])) { //periodic conditions for wrt1
                    if (dpos2 - 1 < 0) {
                        mult2m1 = (size[wrt2] - 1) * mult2;
                        mult2m2 = (size[wrt2] - 2) * mult2;
                        mult2p1 = mult2;
                        mult2p2 = 2 * mult2;
                    } else if (dpos2 - 2 < 0) {
                        mult2m1 = -mult2;
                        mult2m2 = (size[wrt2] - 2) * mult2;
                        mult2p1 = mult2;
                        mult2p2 = 2 * mult2;
                    } else if (dpos2 + 1 >= size[wrt2]) {
                        mult2m1 = -mult2;
                        mult2m2 = -2 * mult2;
                        mult2p1 = -(size[wrt2] - 1) * mult2;
                        mult2p2 = -(size[wrt2] - 2) * mult2;
                    } else if (dpos2 + 2 >= size[wrt2]) {
                        mult2m1 = -mult2;
                        mult2m2 = -2 * mult2;
                        mult2p1 = mult2;
                        mult2p2 = -(size[wrt2] - 2) * mult2;
                    }
                } else {
                    mult2m1 = -mult2;
                    mult2m2 = -2 * mult2;
                    mult2p1 = mult2;
                    mult2p2 = 2 * mult2;
                }
                return (f->data[point + mult1p2 + mult2p2] - 8.0 * f->data[point + mult1p1 + mult2p2] +
                        8.0 * f->data[point + mult1m1 + mult2p2]
                        - f->data[point + mult1m2 + mult2p2] - 8.0 * f->data[point + mult1p2 + mult2p1] +
                        64.0 * f->data[point + mult1p1 + mult2p1]
                        - 64.0 * f->data[point + mult1m1 + mult2p1] + 8.0 * f->data[point + mult1m2 + mult2p1] +
                        8.0 * f->data[point + mult1p2 + mult2m1]
                        - 64.0 * f->data[point + mult1p1 + mult2m1] + 64.0 * f->data[point + mult1m1 + mult2m1] -
                        8.0 * f->data[point + mult1m2 + mult2m1]
                        - f->data[point + mult1p2 + mult2m2] + 8.0 * f->data[point + mult1p1 + mult2m2] -
                        8.0 * f->data[point + mult1m1 + mult2m2]
                        + f->data[point + mult1m2 + mult2m2]) / (144.0 * spacing[wrt1] * spacing[wrt2]);

            } else{


                double mult1m1, mult1m2, mult1p1, mult1p2, mult2m1, mult2m2, mult2p1, mult2p2;
                if(f->boundarytype[2 * wrt2] == 4){
                    int buffer = wrt1;
                    wrt1 = wrt2;
                    wrt2 = buffer;
                    buffer = mult1;
                    mult1 = mult2;
                    mult2 = buffer;
                    buffer = dpos1;
                    dpos1 = dpos2;
                    dpos2 = buffer;
                }
                int wind_dir = wrt2;
                double windpos = dpos2;
                double winding,windingbump;
                if(f->boundarymorphtype==1){winding = f->getBoundaryConstant()*2.0*M_PI/(spacing[wind_dir]*(size[wind_dir]));windingbump = 0.0;}
                else {winding = f->getBoundaryConstant()*2.0*M_PI*windpos/(size[wind_dir]); windingbump = f->getBoundaryConstant()*2.0*M_PI/(size[wind_dir]);}
                if ((f->boundarytype[2 * wrt2] == 1 && dpos2 - 2 < 0) ||
                    (f->boundarytype[2 * wrt2 + 1] == 1 && dpos2 + 2 >= size[wrt2])) { //periodic conditions for wrt2
                    if (dpos2 - 1 < 0) {
                        mult2m1 = (size[wrt2] - 1) * mult2;
                        mult2m2 = (size[wrt2] - 2) * mult2;
                        mult2p1 = mult2;
                        mult2p2 = 2 * mult2;
                    } else if (dpos2 - 2 < 0) {
                        mult2m1 = -mult2;
                        mult2m2 = (size[wrt2] - 2) * mult2;
                        mult2p1 = mult2;
                        mult2p2 = 2 * mult2;
                    } else if (dpos2 + 1 >= size[wrt2]) {
                        mult2m1 = -mult2;
                        mult2m2 = -2 * mult2;
                        mult2p1 = -(size[wrt2] - 1) * mult2;
                        mult2p2 = -(size[wrt2] - 2) * mult2;
                    } else if (dpos2 + 2 >= size[wrt2]) {
                        mult2m1 = -mult2;
                        mult2m2 = -2 * mult2;
                        mult2p1 = mult2;
                        mult2p2 = -(size[wrt2] - 2) * mult2;
                    }
                }else {
                    mult2m1 = -mult2;
                    mult2m2 = -2 * mult2;
                    mult2p1 = mult2;
                    mult2p2 = 2 * mult2;
                }


                if (dpos1 - 2 < 0 || dpos1 + 2 >= size[wrt1]) { //periodic conditions for wrt1
                    if (dpos1 - 1 < 0) {
                        mult1m1 = (size[wrt1] - 1) * mult1;
                        mult1m2 = (size[wrt1] - 2) * mult1;
                        mult1p1 = mult1;
                        mult1p2 = 2 * mult1;

                        return (f->data[point + mult1p2 + mult2p2] - 8.0 * f->data[point + mult1p1 + mult2p2] +
                                8.0 * f->rotate(point + mult1m1 + mult2p2,-winding-2*windingbump)
                                - f->rotate(point + mult1m2 + mult2p2,-winding-2*windingbump) - 8.0 * f->data[point + mult1p2 + mult2p1] +
                                64.0 * f->data[point + mult1p1 + mult2p1]
                                - 64.0 * f->rotate(point + mult1m1 + mult2p1,-winding-windingbump) + 8.0 * f->rotate(point + mult1m2 + mult2p1,-winding-windingbump) +
                                8.0 * f->data[point + mult1p2 + mult2m1]
                                - 64.0 * f->data[point + mult1p1 + mult2m1] + 64.0 * f->rotate(point + mult1m1 + mult2m1,-winding+windingbump) -
                                8.0 * f->rotate(point + mult1m2 + mult2m1,-winding+windingbump)
                                - f->data[point + mult1p2 + mult2m2] + 8.0 * f->data[point + mult1p1 + mult2m2] -
                                8.0 * f->rotate(point + mult1m1 + mult2m2,-winding+2*windingbump)
                                + f->rotate(point + mult1m2 + mult2m2,-winding+2*windingbump)) / (144.0 * spacing[wrt1] * spacing[wrt2]);

                    } else if (dpos1 - 2 < 0) {
                        mult1m1 = -mult1;
                        mult1m2 = (size[wrt1] - 2) * mult1;
                        mult1p1 = mult1;
                        mult1p2 = 2 * mult1;

                        return (f->data[point + mult1p2 + mult2p2] - 8.0 * f->data[point + mult1p1 + mult2p2] +
                                8.0 * f->data[point + mult1m1 + mult2p2]
                                - f->rotate(point + mult1m2 + mult2p2,-winding-2*windingbump) - 8.0 * f->data[point + mult1p2 + mult2p1] +
                                64.0 * f->data[point + mult1p1 + mult2p1]
                                - 64.0 * f->data[point + mult1m1 + mult2p1] + 8.0 * f->rotate(point + mult1m2 + mult2p1,-winding-windingbump) +
                                8.0 * f->data[point + mult1p2 + mult2m1]
                                - 64.0 * f->data[point + mult1p1 + mult2m1] + 64.0 * f->data[point + mult1m1 + mult2m1] -
                                8.0 * f->rotate(point + mult1m2 + mult2m1,-winding+windingbump)
                                - f->data[point + mult1p2 + mult2m2] + 8.0 * f->data[point + mult1p1 + mult2m2] -
                                8.0 * f->data[point + mult1m1 + mult2m2]
                                + f->rotate(point + mult1m2 + mult2m2,-winding+2*windingbump)) / (144.0 * spacing[wrt1] * spacing[wrt2]);

                    } else if (dpos1 + 1 >= size[wrt1]) {
                        mult1m1 = -mult1;
                        mult1m2 = -2 * mult1;
                        mult1p1 = -(size[wrt1] - 1) * mult1;
                        mult1p2 = -(size[wrt1] - 2) * mult1;

                        return (f->rotate(point + mult1p2 + mult2p2,winding+2*windingbump) - 8.0 * f->rotate(point + mult1p1 + mult2p2,winding+2*windingbump) +
                                8.0 * f->data[point + mult1m1 + mult2p2]
                                - f->data[point + mult1m2 + mult2p2] - 8.0 * f->rotate(point + mult1p2 + mult2p1,winding+windingbump) +
                                64.0 * f->rotate(point + mult1p1 + mult2p1,winding+windingbump)
                                - 64.0 * f->data[point + mult1m1 + mult2p1] + 8.0 * f->data[point + mult1m2 + mult2p1] +
                                8.0 * f->rotate(point + mult1p2 + mult2m1,winding-windingbump)
                                - 64.0 * f->rotate(point + mult1p1 + mult2m1,winding-windingbump) + 64.0 * f->data[point + mult1m1 + mult2m1] -
                                8.0 * f->data[point + mult1m2 + mult2m1]
                                - f->rotate(point + mult1p2 + mult2m2,winding-2*windingbump) + 8.0 * f->rotate(point + mult1p1 + mult2m2,winding-2*windingbump) -
                                8.0 * f->data[point + mult1m1 + mult2m2]
                                + f->data[point + mult1m2 + mult2m2]) / (144.0 * spacing[wrt1] * spacing[wrt2]);

                    } else if (dpos1 + 2 >= size[wrt1]) {
                        mult1m1 = -mult1;
                        mult1m2 = -2 * mult1;
                        mult1p1 = mult1;
                        mult1p2 = -(size[wrt1] - 2) * mult1;

                        return (f->rotate(point + mult1p2 + mult2p2,winding+2*windingbump) - 8.0 * f->data[point + mult1p1 + mult2p2] +
                                8.0 * f->data[point + mult1m1 + mult2p2]
                                - f->data[point + mult1m2 + mult2p2] - 8.0 * f->rotate(point + mult1p2 + mult2p1,winding+windingbump) +
                                64.0 * f->data[point + mult1p1 + mult2p1]
                                - 64.0 * f->data[point + mult1m1 + mult2p1] + 8.0 * f->data[point + mult1m2 + mult2p1] +
                                8.0 * f->rotate(point + mult1p2 + mult2m1,winding-windingbump)
                                - 64.0 * f->data[point + mult1p1 + mult2m1] + 64.0 * f->data[point + mult1m1 + mult2m1] -
                                8.0 * f->data[point + mult1m2 + mult2m1]
                                - f->rotate(point + mult1p2 + mult2m2,winding-2*windingbump) + 8.0 * f->data[point + mult1p1 + mult2m2] -
                                8.0 * f->data[point + mult1m1 + mult2m2]
                                + f->data[point + mult1m2 + mult2m2]) / (144.0 * spacing[wrt1] * spacing[wrt2]);

                    }
                }else {
                    mult1m1 = -mult1;
                    mult1m2 = -2 * mult1;
                    mult1p1 = mult1;
                    mult1p2 = 2 * mult1;

                    return (f->data[point + mult1p2 + mult2p2] - 8.0 * f->data[point + mult1p1 + mult2p2] +
                            8.0 * f->data[point + mult1m1 + mult2p2]
                            - f->data[point + mult1m2 + mult2p2] - 8.0 * f->data[point + mult1p2 + mult2p1] +
                            64.0 * f->data[point + mult1p1 + mult2p1]
                            - 64.0 * f->data[point + mult1m1 + mult2p1] + 8.0 * f->data[point + mult1m2 + mult2p1] +
                            8.0 * f->data[point + mult1p2 + mult2m1]
                            - 64.0 * f->data[point + mult1p1 + mult2m1] + 64.0 * f->data[point + mult1m1 + mult2m1] -
                            8.0 * f->data[point + mult1m2 + mult2m1]
                            - f->data[point + mult1p2 + mult2m2] + 8.0 * f->data[point + mult1p1 + mult2m2] -
                            8.0 * f->data[point + mult1m1 + mult2m2]
                            + f->data[point + mult1m2 + mult2m2]) / (144.0 * spacing[wrt1] * spacing[wrt2]);

                }





            }
            }

            // old function (much slower!)
            /*return (f->getData(pos+2*dir1+2*dir2) - 8.0*f->getData(pos+dir1+2*dir2) + 8.0*f->getData(pos-dir1+2*dir2)
                    - f->getData(pos-2*dir1+2*dir2) - 8.0*f->getData(pos+2*dir1+dir2) +64.0*f->getData(pos+dir1+dir2)
                    -64.0*f->getData(pos-dir1+dir2) + 8.0*f->getData(pos-2*dir1+dir2) + 8.0*f->getData(pos+2*dir1-dir2)
                    - 64.0*f->getData(pos+dir1-dir2)+ 64.0*f->getData(pos-dir1-dir2) - 8.0*f->getData(pos-2*dir1-dir2)
                    - f->getData(pos+2*dir1-2*dir2) + 8.0*f->getData(pos+dir1-2*dir2) - 8.0*f->getData(pos-dir1-2*dir2)
                    + f->getData(pos-2*dir1-2*dir2))/(144.0*spacing[wrt1]*spacing[wrt2]);*/

        }
    }

void BaseFieldTheory::updateEnergy() {
    if (dynamic) {
        double sumpot = 0.0;
        double sumkin = 0.0;
    #pragma omp parallel for reduction(+:sumpot,sumkin)
        for (int i = 0; i < getTotalSize(); i++) {
            if (inBoundary(i)) {
                vector<double> buffer = calculateDynamicEnergy(i);
                energydensity[i] = buffer[0] + buffer[1];
                sumpot += buffer[0];
                sumkin += buffer[1];
            }
        }
        potential = sumpot * getTotalSpacing();
        kinetic = sumkin * getTotalSpacing();
        energy = potential + kinetic;
    } else {
        double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < getTotalSize(); i++) {
        if (inBoundary(i)) {
            double buffer = calculateEnergy(i);
            energydensity[i] = buffer;
            sum += buffer;
        }
    }
    energy = sum * getTotalSpacing();
    potential = energy;
    kinetic = 0.0;
    }
};

void BaseFieldTheory::updateCharge(){
        double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
        for(int i = 0; i < getTotalSize(); i++) {
            if (inBoundary(i)) {
                double buffer = calculateCharge(i);
                chargedensity[i] = buffer;
                sum += buffer;
            }
        }
        charge = sum * getTotalSpacing();

}

inline double BaseFieldTheory::calculateEnergy(int pos)
{
cout << "ERROR! - either the incorrect number of parameters was entered into calculateEnergy or the calculateEnergy has not been set in the derived Field Theory class!\n";
return -1.0;
}

inline double BaseFieldTheory::calculateCharge(int pos)
{
    cout << "ERROR! - either the incorrect number of parameters was entered into calculateCharge or the calculateCharge has not been set in the derived Field Theory class!\n";
    return -1.0;
}

inline vector<double> BaseFieldTheory::calculateDynamicEnergy(int pos){
    cout << "Error! - the calculateDynamicEnergy virtual function has not been overwritten in the derived Field Theory! Please correct!! \n";
    return {-1.0,-1.0};
}

inline void BaseFieldTheory::RK4calc(int i){
    cout << "Error! - the RK4calc virtual function has not been overwritten in the derived Field Theory! Please correct!! \n";
}


    void BaseFieldTheory::gradientFlow(int iterations, int often){
        if(!normalise_me){cout << "WARNING! - all normalisation is currently removed for his theory, alter normalsie_me value on initialisation to change!\n";}
    if(dynamic){
        cout << "Warning! You have run gradient flow on a dynamic theory, this will kill the kinetic componenet!\n";
        for(int i = 0; i < getTotalSize(); i++){
            fields.cutKinetic(i);
        }
    }
    int no = 0;
    double sum = 0.0;
    #pragma omp parallel
    while(no < iterations){
            #pragma omp for nowait
                for (int i = 0; i < getTotalSize(); i++) {
                    calculateGradientFlow(i);
                }
        #pragma omp barrier
        #pragma omp master
        {
            fields.update_gradients(dt);
            if(normalise_me){fields.normalise();}
            no += 1;
            sum = 0.0;
        }
        #pragma omp barrier
        #pragma omp for nowait reduction(+:sum)
        for (int i = 0; i < getTotalSize(); i++) {
            if (inBoundary(i)) {
                double buffer = calculateEnergy(i);
                sum += buffer;
            }
        }
        #pragma omp barrier
        #pragma omp master
        {
            double newenergy = sum * getTotalSpacing();
            if(newenergy >= energy+0.00000000001){dt = dt*0.5; cout << "CUTTING! due to - " << newenergy << " " << energy << "\n";}
            energy = newenergy;
            if (no % often == 0) {
                cout << "Energy is " << energy << "\n";
                dt = dt * 1.2;
            }
            correctTheory(no);
        }
    }
}

    void BaseFieldTheory::correctTheory(int loop){
        gradientCorrections(loop);

    }

    void BaseFieldTheory::gradientCorrections(int loop){
        cout << "[ERROR!] - gradient corrections function should be overitten or write a flow for it here!\n";

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

inline void BaseFieldTheory::calculateGradientFlow(int pos){
    cout << "ERROR! you havent written a calculateGradientFlow function!\n";
}

void BaseFieldTheory::save(const char * savepath){
    ofstream outf(savepath);
    //first output the derived class name to ensure loading the correct theory type!
    //first output dimensions of the field theory!
    for(int i = 0; i < dim; i++) {
        outf << size[i] << " " ;
    }
    outf << "\n";
    //second output the spacing of the field theory
    for(int i = 0; i < dim; i++) {
        outf << spacing[i] << " " ;
    }
    outf << "\n";
    //third output the parameters of the field theory
    for(int i = 0; i < parameters.size(); i++) {
        outf << *parameters[i] << " " ;
    }
    outf << "\n";
    outf << "field_start:\n";
    //fourth output the fields (this will determine autmatically if dynamic and need to output the time derivative also)
    fields.save_fields(outf, getTotalSize());
    //close the file
    outf.close();
}


void BaseFieldTheory::load(const char * loadpath, bool message){
    if(message){
    cout << "loading from file " << loadpath << "\n";
    }
    ifstream infile(loadpath);
    //first load the derived class name to ensure loading the correct theory type!
    //first load dimensions of the field theory!
    for(int i = 0; i < dim; i++){
        infile >> size[i];
    }
    if(message) {
        cout << "dim: ";
        for (int i = 0; i < dim; i++) {
            cout << size[i];
            if (i < dim - 1) { cout << "x"; }
        }
        cout << "\n";
    }
    // and resize the theory to the loaded dimensions
        fields.resize(size);
        energydensity.resize(getTotalSize());
        chargedensity.resize(getTotalSize());
    //second load the spacing of the field theory
    for(int i = 0; i < dim; i++){
        infile >> spacing[i];
    }
    if(message) {
        cout << "spacing: ";
        for (int i = 0; i < dim; i++) {
            cout << spacing[i];
            if (i < dim - 1) { cout << "x"; }
        }
        cout << "\n";
    }
    //third load the parameters of the field theory
    for(int i = 0; i < parameters.size(); i++){
        infile >> *parameters[i];
    }
    if(message) {
        printParameters();
    }
    string check;
    infile >> check;
    if(check == "field_start:") {
        //fourth load the fields (this will determine autmatically if dynamic and need to output the time derivative also)
        fields.load_fields(infile, getTotalSize());
        //close the file
        if(message) {
            cout << "loading completed succesfully!\n";
        }
    }else {cout << "ERROR! incorrect number of parameters in file, have you loaded a matching field theory or altered the class since saving!?\n";cout << "exiting load function!\n";}
    infile.close();
}

void BaseFieldTheory::plotEnergy() {
    if(dim == 2){
        plot2D(size, energydensity);
    }
    if(dim == 3){
        double isovalue = -1.0;
        for(int i = 0; i < getTotalSize(); i++) {
            if(energydensity[i] > isovalue){
                isovalue = energydensity[i];
            }
        }
        plot3D(size, energydensity , isovalue/2.0);
    }

}

    void BaseFieldTheory::plotCharge() {
        if(dim == 2){
            plot2D(size, chargedensity);
        }
        if(dim == 3){
            double isovalue = -1.0;
            for(int i = 0; i < getTotalSize(); i++) {
                if(chargedensity[i] > isovalue){
                    isovalue = chargedensity[i];
                }
            }
            plot3D(size, chargedensity , isovalue/2.0);
        }

    }

}

#endif
 // End FTPL namespace
