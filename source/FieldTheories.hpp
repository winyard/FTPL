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
#include <chrono>
#include <random>
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
        inline const T  __attribute__((always_inline)) getData( const vector<int> pos);
        inline void  __attribute__((always_inline)) setBuffer(const T value, vector<int> &pos);
		Field(int d, vector<int> size, bool isdynamic);
        ~Field(){};
		inline T operator()(...); // TO BE WRITTEN
        inline const vector<T> getData();
        vector<int> getSize();
        int getSize(int com);
        inline int __attribute__((always_inline)) getTotalSize();
        inline void  __attribute__((always_inline)) setData(vector<T> datain);
		inline void  __attribute__((always_inline)) setData(T value, vector<int> pos);
        void fill(T value);
        void fill_dt(T value);
		void save_field(ofstream& output);
		void load_field(ifstream& input);
		void normalise();
        void update_field();
        void update_gradient(double dt);
        inline void updateRK4(int k);
        inline void updateRK4(int k, int i);
        void resize(vector<int> sizein);
        void progressTime(double time_step);
        void update_derivatives(vector<double> spacing);
        void alter_point(int i, T value, vector<double> spacing);
    protected:
		vector<int>  size;
        vector<vector<T>> k_sum;
};

template<class T>
    void Field<T>::alter_point(int i, T value, vector<double> spacing){
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
            change = dif/(12.0*spacing[wrt1]*spacing[wrt1]);
            double_derivatives[wrt1][wrt1][i-2*mult1] += -change;
            double_derivatives[wrt1][wrt1][i-mult1] += +16.0*change;
            double_derivatives[wrt1][wrt1][i] += -30.0*change;
            double_derivatives[wrt1][wrt1][i+mult1] += +16.0*change;
            double_derivatives[wrt1][wrt1][i+2*mult1] += -change;
            double mult2 = 1.0;
            for(int wrt2=0; wrt2<wrt1; wrt2++) {
                change = dif/(144.0*spacing[wrt1]*spacing[wrt2]);
                double_derivatives[wrt2][wrt1][i-2*mult2-2*mult1] += change;
                double_derivatives[wrt2][wrt1][i-mult2-2*mult1] += -8.0*change;
                double_derivatives[wrt2][wrt1][i+mult2-2*mult1] += 8.0*change;
                double_derivatives[wrt2][wrt1][i+2*mult2-2*mult1] += -change;
                double_derivatives[wrt2][wrt1][i-2*mult2-mult1] += -8.0*change;
                double_derivatives[wrt2][wrt1][i-mult2-mult1] += +64.0*change;
                double_derivatives[wrt2][wrt1][i+mult2-mult1] += -64.0*change;
                double_derivatives[wrt2][wrt1][i+2*mult2-mult1] += +8.0*change;
                double_derivatives[wrt2][wrt1][i-2*mult2+mult1] += +8.0*change;
                double_derivatives[wrt2][wrt1][i-mult2+mult1] += -64.0*change;
                double_derivatives[wrt2][wrt1][i+mult2+mult1] += +64.0*change;
                double_derivatives[wrt2][wrt1][i+2*mult2+mult1] += -8.0*change;
                double_derivatives[wrt2][wrt1][i-2*mult2+2*mult1] += -change;
                double_derivatives[wrt2][wrt1][i-mult2+2*mult1] += 8.0*change;
                double_derivatives[wrt2][wrt1][i+mult2+2*mult1] += -8.0*change;
                double_derivatives[wrt2][wrt1][i+2*mult2+2*mult1] += change;
                mult2 *= size[wrt2];
            }
            mult1 *= size[wrt1];
        }
    }

/* --- Constructors & Destructors --- */

template <class T>
Field<T>::Field(int d, vector<int> sizein, bool isdynamic): dim(d), size(sizein), dynamic(isdynamic)  {
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
void Field<T>::resize(vector<int> sizein) { // resize the fields for different accuracy
    if(size.size() == sizein.size()) {
        size = sizein;
        data.resize(getTotalSize());
        buffer.resize(getTotalSize());
        if(dynamic){
            dt.resize(getTotalSize());
            buffer_dt.resize(getTotalSize());
            k_sum[0].resize(getTotalSize());
            k_sum[1].resize(getTotalSize());
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
        void load_fields(ifstream& loadfile, int totalSize);
        void save_fields(ofstream& savefile, int totalSize);
        void update_fields();
        void update_gradients(double dt);
        void update_RK4(int k, int pos);
        void normalise();
        void moveToBuffer();
        void cutKinetic(int pos);
        int no_fields;
        void storeDerivatives(BaseFieldTheory * theory);
        void randomise(int i, int field, vector<double> spacing, bool normalise);
        void derandomise(int i, int field, vector<double> spacing);
    private:
        // Add aditional Field types as they are created here!
        std::vector<Field<Eigen::VectorXd>> fields1;
        std::vector<Field<double>> fields2;
        std::vector<Field<int>> fields3;
        std::vector<Field<Eigen::MatrixXd>> fields4;
};

    class BaseFieldTheory {
    public:
        int dim;
        BaseFieldTheory(int d, vector<int> size, bool isDynamic); // TO BE WRITTEN
        ~BaseFieldTheory(){};
        inline vector<int> next(vector<int> current);
        inline vector<int>  __attribute__((always_inline)) convert(int in);
        inline virtual void calculateGradientFlow(int pos); // TO BE WRITTEN
        inline virtual double calculateEnergy(int pos);// TO BE WRITTEN
        inline virtual void __attribute__((always_inline)) RK4calc(int i);
        inline virtual double __attribute__((always_inline)) metric(int i, int j, vector<double> pos = {0});
        void RK4(int iterations, bool normalise, bool cutEnergy, int often); // TO BE WRITTEN
        void save(const char * savepath); // TO BE WRITTEN
        void load(const char * loadpath); // TO BE WRITTEN
        void plot(const char * plotpath); // TO BE WRITTEN
        void spaceTransformation(); // TO BE WRITTEN
        template <class T>
        void fieldTransformation(T tranformation); // TO BE WRITTEN
        void setBoundaryType(vector<int> boundaryin); // TO BE WRITTEN
        void setStandardMetric(string type);
        void updateEnergy(); // TO BE WRITTEN
        void gradientFlow(int iterations, int often, bool normalise); // TO BE WRITTEN
        void setTimeInterval(double dt_in);
        double getEnergy(){return energy;};
        inline bool inBoundary(vector<int> pos);
        inline bool inBoundary(int pos);
        inline virtual __attribute__((always_inline)) vector<double> calculateDynamicEnergy(int pos);
        inline  void setSpacing(vector<double> spacein);
        void addParameter( double * parameter_in);
        vector<double> getSpacing();
        int getTotalSize();
        double getTotalSpacing();
        void plotEnergy();
        void setMetricType(string type);
        void annealing(int iterations, int often, bool normalise = false);
        template <class T>
        inline T single_time_derivative(Field<T> * f, int wrt, int &point) __attribute__((always_inline))  ;
        template <class T>
        inline T single_derivative(Field<T> * f, int wrt, int &point) __attribute__((always_inline))  ;
        template <class T>
        inline T double_derivative(Field<T> * f, int wrt1, int wrt2, int &point) __attribute__((always_inline)) ;
    protected:
        int metric_type = 0;
        template <class T>
        Field<T> * createField(Field<T> * target, bool isDynamic);
        //vector<unique_ptr<Field>> fields;
        TargetSpace fields;
        vector<int> bdw; // number of boundary points that are never updated or contribute to energies etc. in each direction
        vector<int> boundarytype; // indicates the type of boundary in each direction 0-fixed(size = bdw), 1-dirichlet (dx = 0), 2-periodic
        vector<double> energydensity;
        double energy, potential, kinetic;
        vector<double> spacing;
        vector<int> size;
        double dt;
        bool dynamic;
        bool curved = false;
        bool setDerivatives = false;
        vector<double *> parameters;
    };

    void TargetSpace::randomise(int i, int field, vector<double> spacing, bool normalise){
        std::random_device rd;
        std::mt19937 mt(rd());
        if(field <= fields1.size()-1){
            Eigen::VectorXd value = fields1[field].data[i];
            for(int j = 0; j < value.size(); j++){
                std::uniform_real_distribution<double> dist(fields1[field].min[j], fields1[field].max[j]);
                value[j] = fields1[field].data[i][j] + dist(mt);
            }
            if(normalise){
                value.normalize();
            }
            fields1[field].alter_point(i,value,spacing);
        }
        else if(field <= fields1.size() + fields2.size() - 1){
            std::uniform_real_distribution<double> dist(fields2[field - fields1.size()].min, fields2[field - fields1.size()].max);
            double value = dist(mt);
            fields2[field - fields1.size()].alter_point(i,value,spacing);
        }
        else if(field<= fields1.size() + fields2.size() +fields3.size() -1){
            std::uniform_int_distribution<int> dist(fields3[field - fields1.size() - fields2.size()].min, fields3[field - fields1.size() - fields2.size()].max);
            int value = dist(mt);
            fields3[field - fields1.size() - fields2.size()].alter_point(i,value,spacing);
        }
        else{
            Eigen::MatrixXd value = fields4[field - fields1.size() - fields2.size()  - fields3.size()].data[i];
            for(int j = 0; j < value.rows(); j++){
                for(int k = 0; k < value.cols(); k++){
                    std::uniform_real_distribution<double> dist(fields4[field - fields1.size() - fields2.size()  - fields3.size()].min(j,k), fields4[field - fields1.size() - fields2.size()  - fields3.size()].max(j,k));
                    value(j,k) = dist(mt);
            }}
            if(normalise){
                value.normalize();
            }
            fields4[field - fields1.size() - fields2.size()  - fields3.size()].alter_point(i,value,spacing);
        }
    }

    void TargetSpace::derandomise(int i, int field, vector<double> spacing){
        if(field <= fields1.size()-1){
                fields1[field].alter_point(i,fields1[field].buffer[i],spacing);
        }
        else if(field <= fields1.size() + fields2.size() - 1){
                int point = field - fields1.size();
                fields2[point].alter_point(i,fields2[point].buffer[i],spacing);
        }
        else if(field <= fields1.size() + fields2.size() +fields3.size() -1){
            int point = field - fields1.size() - fields2.size();
                fields3[point].alter_point(i,fields3[point].buffer[i],spacing);
        }
        else{
                int point = field - fields1.size() - fields2.size() - fields3.size();
                fields4[point].alter_point(i,fields4[point].buffer[i],spacing);
        }
    }



    void TargetSpace::storeDerivatives(BaseFieldTheory * theory){
        for(int i = 0; i < fields1.size(); i++){
            fields1[i].single_derivatives.resize(theory->dim);
            fields1[i].double_derivatives.resize(theory->dim);
            for(int j = 0; j < theory->dim ; j++) {
                fields1[i].single_derivatives[j].resize(theory->getTotalSize());
                fields1[i].double_derivatives[j].resize(theory->dim);
                for(int k = j; k < theory->dim; k++ ) {
                    fields1[i].double_derivatives[j][k].resize(theory->getTotalSize());
                }
            }
            for(int j = 0; j < theory->getTotalSize() ; j++) {
            if (theory->inBoundary(j)) {
                for (int wrt1 = 0; wrt1 < theory->dim; wrt1++) {
                    fields1[i].single_derivatives[wrt1][j] = theory->single_derivative(&fields1[i], wrt1, j);
                    for (int wrt2 = wrt1; wrt2 < theory->dim; wrt2++) {
                        fields1[i].double_derivatives[wrt1][wrt2][j] = theory->double_derivative(&fields1[i], wrt1,
                                                                                                 wrt2, j);
                    }
                }
            }
            }
        }
        for(int i = 0; i < fields2.size(); i++){
            fields2[i].single_derivatives.resize(theory->dim);
            fields2[i].double_derivatives.resize(theory->dim);
            for(int j = 0; j < theory->dim ; j++) {
                fields2[i].single_derivatives[j].resize(theory->getTotalSize());
                fields2[i].double_derivatives[j].resize(theory->dim);
                for(int k = j; k < theory->dim; k++ ) {
                    fields2[i].double_derivatives[j][k].resize(theory->getTotalSize());
                }
            }
            for(int j = 0; j < theory->getTotalSize() ; j++) {
            if (theory->inBoundary(j)) {
                for(int wrt1 = 0; wrt1 < theory->dim ; wrt1++) {
                    fields2[i].single_derivatives[wrt1][j]=theory->single_derivative(&fields2[i],wrt1,j);
                    for(int wrt2 = wrt1; wrt2 < theory->dim ; wrt2++) {
                        fields2[i].double_derivatives[wrt1][wrt2][j]=theory->double_derivative(&fields2[i],wrt1,wrt2,j);
                    }
                }
            }}
        }
        for(int i = 0; i < fields3.size(); i++){
            fields3[i].single_derivatives.resize(theory->dim);
            fields3[i].double_derivatives.resize(theory->dim);
            for(int j = 0; j < theory->dim ; j++) {
                fields3[i].single_derivatives[j].resize(theory->getTotalSize());
                fields3[i].double_derivatives[j].resize(theory->dim);
                for(int k = j; k < theory->dim; k++ ) {
                    fields3[i].double_derivatives[j][k].resize(theory->getTotalSize());
                }
            }
            for(int j = 0; j < theory->getTotalSize() ; j++) {
            if (theory->inBoundary(j)) {
                for(int wrt1 = 0; wrt1 < theory->dim ; wrt1++) {
                    fields3[i].single_derivatives[wrt1][j]=theory->single_derivative(&fields3[i],wrt1,j);
                    for(int wrt2 = wrt1; wrt2 < theory->dim ; wrt2++) {
                        fields3[i].double_derivatives[wrt1][wrt2][j]=theory->double_derivative(&fields3[i],wrt1,wrt2,j);
                    }
                }
            }}
        }
        for(int i = 0; i < fields4.size(); i++){
            fields4[i].single_derivatives.resize(theory->dim);
            fields4[i].double_derivatives.resize(theory->dim);
            for(int j = 0; j < theory->dim ; j++) {
                fields4[i].single_derivatives[j].resize(theory->getTotalSize());
                fields4[i].double_derivatives[j].resize(theory->dim);
                for(int k = j; k < theory->dim; k++ ) {
                    fields4[i].double_derivatives[j][k].resize(theory->getTotalSize());
                }
            }
            for(int j = 0; j < theory->getTotalSize() ; j++) {
            if (theory->inBoundary(j)) {
                for(int wrt1 = 0; wrt1 < theory->dim ; wrt1++) {
                    fields4[i].single_derivatives[wrt1][j]=theory->single_derivative(&fields4[i],wrt1,j);
                    for(int wrt2 = wrt1; wrt2 < theory->dim ; wrt2++) {
                        fields4[i].double_derivatives[wrt1][wrt2][j]=theory->double_derivative(&fields4[i],wrt1,wrt2,j);
                    }
                }
            }}
        }
    }

    void TargetSpace::moveToBuffer(){
        for(int i = 0; i < fields1.size(); i++){
            fields1[i].buffer = fields1[i].data;
            if(fields1[i].dynamic){fields1[i].buffer_dt = fields1[i].dt;}
        }
        for(int i = 0; i < fields2.size(); i++){
            fields2[i].buffer = fields2[i].data;
            if(fields2[i].dynamic){fields2[i].buffer_dt = fields2[i].dt;}
        }
        for(int i = 0; i < fields3.size(); i++){
            fields3[i].buffer = fields3[i].data;
            if(fields3[i].dynamic){fields3[i].buffer_dt = fields3[i].dt;}
        }
        for(int i = 0; i < fields4.size(); i++){
            fields4[i].buffer = fields4[i].data;
            if(fields4[i].dynamic){fields4[i].buffer_dt = fields4[i].dt;}
        }

    }

void TargetSpace::cutKinetic(int pos){
    for(int i = 0; i < fields1.size(); i++){
        fields1[i].dt[pos] = Eigen::VectorXd::Zero(fields1[i].dt[pos].size());
    }
    for(int i = 0; i < fields2.size(); i++){
        fields2[i].dt[pos] = 0.0;
    }
    for(int i = 0; i < fields3.size(); i++){
        fields3[i].dt[pos] = 0;
    }
    for(int i = 0; i < fields4.size(); i++){

        fields4[i].dt[pos] = Eigen::MatrixXd::Zero(fields4[i].dt[pos].rows(), fields4[i].dt[pos].cols());;
    }
}

void TargetSpace::normalise(){
    for(int i = 0; i < fields1.size(); i++){
        fields1[i].normalise();
    }
    for(int i = 0; i < fields4.size(); i++){
        fields1[i].normalise();
    }
}

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
    no_fields = no_fields + 1;
    return &fields1[fields1.size() - 1];
}

Field<double> * TargetSpace::addField(int dim, vector<int> size, Field<double> * target, bool isDynamic) {
    fields2.push_back(Field < double > (dim, size, isDynamic));
    no_fields = no_fields + 1;
    return &fields2[fields2.size() - 1];
}

Field<int> * TargetSpace::addField(int dim, vector<int> size, Field<int> * target, bool isDynamic) {
    fields3.push_back(Field < int > (dim, size, isDynamic));
    no_fields = no_fields + 1;
    return &fields3[fields3.size() - 1];
}

Field<Eigen::MatrixXd> * TargetSpace::addField(int dim, vector<int> size, Field<Eigen::MatrixXd> * target, bool isDynamic) {
    fields4.push_back(Field < Eigen::MatrixXd > (dim, size, isDynamic));
    no_fields = no_fields + 1;
    return &fields4[fields4.size() - 1];
}



    void TargetSpace::save_fields(ofstream& savefile, int totalSize){
        for(int i = 0; i < fields1.size(); i++){
            for(int j =0; j < totalSize; j++) {
                for(int k = 0; k < fields1[i].data[j].size(); k++) {
                    savefile << fields1[i].data[j](k) << " ";
                }
                savefile << "\n";
            }
        }
        for(int i = 0; i < fields2.size(); i++){
            for(int j =0; j < totalSize; j++) {
                savefile << fields2[i].data[j] << "\n";
            }
        }
        for(int i = 0; i < fields3.size(); i++){
            for(int j =0; j < totalSize; j++) {
                savefile << fields3[i].data[j] << "\n";
            }
        }
        for(int i = 0; i < fields4.size(); i++){
            for(int j =0; j < totalSize; j++) {
                for(int k = 0; k < fields4[i].data[j].rows(); k++) {
                for(int h = 0; h < fields4[i].data[j].cols(); h++){
                    savefile << fields4[i].data[j](k,h) << " ";
                }}
                savefile << "\n";
            }
        }
    }

    void TargetSpace::load_fields(ifstream& loadfile, int totalSize){
        for(int i = 0; i < fields1.size(); i++){
            for(int j =0; j < totalSize; j++){
                for(int k = 0; k < fields1[i].data[j].size(); k++){
                   loadfile >> fields1[i].data[j](k);
                }
            }
        }
        for(int i = 0; i < fields2.size(); i++){
            for(int j =0; j < totalSize; j++){
                loadfile >> fields2[i].data[j];
            }
        }
        for(int i = 0; i < fields3.size(); i++){
            for(int j =0; j < totalSize; j++){
                loadfile >> fields3[i].data[j];
            }
        }
        for(int i = 0; i < fields4.size(); i++){
            for(int j =0; j < totalSize; j++) {
                for(int k = 0; k < fields4[i].data[j].rows(); k++){
                for(int h = 0; h < fields4[i].data[j].cols(); h++){
                    loadfile >> fields4[i].data[j](k,h);
                }}
            }
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

    void TargetSpace::update_RK4(int k, int pos){
        for(int i = 0; i < fields1.size(); i++){
            if(fields1[i].dynamic){fields1[i].updateRK4(k, pos);}
        }
        for(int i = 0; i < fields2.size(); i++){
            if(fields2[i].dynamic){fields2[i].updateRK4(k, pos);}
        }
        for(int i = 0; i < fields3.size(); i++){
            if(fields3[i].dynamic){fields3[i].updateRK4(k, pos);}
        }
        for(int i = 0; i < fields4.size(); i++){
            if(fields4[i].dynamic){fields4[i].updateRK4(k, pos);}
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

    void BaseFieldTheory::addParameter(double * parameter_in){
        parameters.push_back(parameter_in);
    }

void BaseFieldTheory::annealing(int iterations, int often, bool normalise){
    if(dynamic){
        cout << "Warning! You have run annealing on a dynamic theory, this will kill the kinetic componenet!\n";
        for(int i = 0; i < getTotalSize(); i++){
            fields.cutKinetic(i);
        }
    }
    updateEnergy();
    int seperator = 2*getTotalSize()/size[dim-1];
    fields.storeDerivatives(this);
    setDerivatives = true;
    updateEnergy();
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<int> f_rand(0, fields.no_fields-1);
    #pragma omp parallel
    {
        int sep = getTotalSize()/omp_get_num_threads();
        double newEnergy;
        double oldEnergy;
        double newEnergyDensity[1+2*dim+4*dim*dim];
        int store = 0;
        std::uniform_int_distribution<int> p_rand(omp_get_thread_num()*sep,omp_get_thread_num()*sep + sep - 1 );
        for (int no = 0; no < iterations; no++) {
            int pos = -1;
            while (!inBoundary(pos)) {
                pos = p_rand(mt);// correct to right random no. generator
            }
            int field = f_rand(mt); // correct for int random no. generator
            fields.randomise(pos, field, spacing, normalise);
            newEnergy = calculateEnergy(pos);
            oldEnergy = energydensity[pos];
            int store = 0;
            newEnergyDensity[store] = calculateEnergy(pos);
            double mult1 = 1.0;
            for (int i = 0; i < dim; i++) {
                store++;
                int point = pos + mult1;
                if (inBoundary(point)) {
                    newEnergyDensity[store] = calculateEnergy(point);
                    newEnergy += newEnergyDensity[store];
                    oldEnergy += energydensity[point];
                }
                store++;
                point = pos - mult1;
                if (inBoundary(point)) {
                    newEnergyDensity[store] = calculateEnergy(point);
                    newEnergy += newEnergyDensity[store];
                    oldEnergy += energydensity[point];
                }
                double mult2 = 1.0;
                for (int j = 0; j <= i; j++) {
                    store++;
                    point = pos + mult1 + mult2;
                    if (inBoundary(point)) {
                        newEnergyDensity[store] = calculateEnergy(point);
                        newEnergy += newEnergyDensity[store];
                        oldEnergy += energydensity[point];
                    }
                    store++;
                    point = pos - mult1 - mult2;
                    if (inBoundary(point)) {
                        newEnergyDensity[store] = calculateEnergy(point);
                        newEnergy += newEnergyDensity[store];
                        oldEnergy += energydensity[point];
                    }
                    if(i != j) {
                        store++;
                        point = pos + mult1 - mult2;
                        if (inBoundary(point)) {
                            newEnergyDensity[store] = calculateEnergy(point);
                            newEnergy += newEnergyDensity[store];
                            oldEnergy += energydensity[point];
                        }
                        store++;
                        point = pos - mult1 + mult2;
                        if (inBoundary(point)) {
                            newEnergyDensity[store] = calculateEnergy(point);
                            newEnergy += newEnergyDensity[store];
                            oldEnergy += energydensity[point];
                        }
                    }
                    mult2 *= size[j];
                }
            mult1 *= size[i];
            }
            if (newEnergy > oldEnergy) { // add some heat term! :) - as well as some fall off
                fields.derandomise(pos, field, spacing);
            } else {
                store = 0;
                energydensity[pos] = newEnergyDensity[store];
                double mult1 = 1.0;
                for (int i = 0; i < dim; i++) {
                    store++;
                    int point = pos + mult1;
                    if (inBoundary(point)) {
                        energydensity[point] = newEnergyDensity[store];
                    }
                    store++;
                    point = pos - mult1;
                    if (inBoundary(point)) {
                        energydensity[point] = newEnergyDensity[store];
                    }
                    double mult2 = 1.0;
                    for (int j = 0; j <= i; j++) {
                        store++;
                        point = pos + mult1 + mult2;
                        if (inBoundary(point)) {
                            energydensity[point] = newEnergyDensity[store];
                        }
                        store++;
                        point = pos - mult1 - mult2;
                        if (inBoundary(point)) {
                            energydensity[point] = newEnergyDensity[store];
                        }
                        if(i != j) {
                            store++;
                            point = pos + mult1 - mult2;
                            if (inBoundary(point)) {
                                energydensity[point] = newEnergyDensity[store];
                            }
                            store++;
                            point = pos - mult1 + mult2;
                            if (inBoundary(point)) {
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
                    updateEnergy();
                    cout << no << ": the energy so far is " << energy << "\n";
                }
                #pragma omp barrier
                }
        }
    }
    setDerivatives = false;
}

inline vector<double> calculateDynamicEnergy(int pos){
    cout << " the calculate Dynamic Energy function has not been set in the Derived class!\n";
}


    void BaseFieldTheory::RK4(int iterations, bool normalise, bool cutEnergy, int often){
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
                if (normalise) {
                    fields.normalise();
                }
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
    dynamic = isDynamic;
    return fields.addField(dim, size, target, isDynamic);
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

        return (-1.0 * f->data[point + 2 * mult] + 8.0 * f->data[point + mult] - 8.0 * f->data[point - mult] +
                f->data[point - 2 * mult]) / (12.0 * spacing[wrt]);
    }
    /*for(int i = 0; i < dim; i++){ if(i == wrt){dir[i] = 1;}else{dir[i] = 0;}};
    return (-1.0*f->getData(pos+2*dir) + 8.0*f->getData(pos+dir) - 8.0*f->getData(pos-dir) + f->getData(pos-2*dir))/(12.0*spacing[wrt]);*/
}

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

            return (-1.0 * f->data[point + 2 * mult] + 16.0 * f->data[point + mult] - 30.0 * f->data[point] +
                    16.0 * f->data[point - mult]
                    - f->data[point - 2 * mult]) / (12.0 * spacing[wrt1] * spacing[wrt1]);

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


            int doub = mult1 + mult2;
            int alt = mult1 - mult2;

            return (f->data[point + 2 * doub] - 8.0 * f->data[point + mult1 + 2 * mult2] +
                    8.0 * f->data[point - mult1 + 2 * mult2]
                    - f->data[point - 2 * alt] - 8.0 * f->data[point + 2 * mult1 + mult2] + 64.0 * f->data[point + doub]
                    - 64.0 * f->data[point - alt] + 8.0 * f->data[point - 2 * mult1 + mult2] +
                    8.0 * f->data[point + 2 * mult1 - mult2]
                    - 64.0 * f->data[point + alt] + 64.0 * f->data[point - doub] -
                    8.0 * f->data[point - 2 * mult1 - mult2]
                    - f->data[point + 2 * alt] + 8.0 * f->data[point + mult1 - 2 * mult2] -
                    8.0 * f->data[point - mult1 - 2 * mult2]
                    + f->data[point - 2 * doub]) / (144.0 * spacing[wrt1] * spacing[wrt2]);

            // old function (much slower!)
            /*return (f->getData(pos+2*dir1+2*dir2) - 8.0*f->getData(pos+dir1+2*dir2) + 8.0*f->getData(pos-dir1+2*dir2)
                    - f->getData(pos-2*dir1+2*dir2) - 8.0*f->getData(pos+2*dir1+dir2) +64.0*f->getData(pos+dir1+dir2)
                    -64.0*f->getData(pos-dir1+dir2) + 8.0*f->getData(pos-2*dir1+dir2) + 8.0*f->getData(pos+2*dir1-dir2)
                    - 64.0*f->getData(pos+dir1-dir2)+ 64.0*f->getData(pos-dir1-dir2) - 8.0*f->getData(pos-2*dir1-dir2)
                    - f->getData(pos+2*dir1-2*dir2) + 8.0*f->getData(pos+dir1-2*dir2) - 8.0*f->getData(pos-dir1-2*dir2)
                    + f->getData(pos-2*dir1-2*dir2))/(144.0*spacing[wrt1]*spacing[wrt2]);*/

        }
    }
}

void BaseFieldTheory::updateEnergy() { // only currently for 2-dim's!
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

inline double BaseFieldTheory::calculateEnergy(int pos)
{
cout << "ERROR! - either the incorrect number of parameters was entered into calculateEnergy or the calculateEnergy has not been set in the derived Field Theory class!\n";
return -1.0;
}

inline vector<double> BaseFieldTheory::calculateDynamicEnergy(int pos){
    cout << "Error! - the calculateDynamicEnergy virtual function has not been overwritten in the derived Field Theory! Please correct!! \n";
    return {-1.0,-1.0};
}

inline void BaseFieldTheory::RK4calc(int i){
    cout << "Error! - the RK4calc virtual function has not been overwritten in the derived Field Theory! Please correct!! \n";
}


    void BaseFieldTheory::gradientFlow(int iterations, int often, bool normalise){
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
            if(normalise){fields.normalise();}
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
            if(newenergy >= energy+0.0000001){dt = dt*0.5; cout << "CUTTING! due to - " << newenergy << " " << energy << "\n";}
            energy = newenergy;
            if (no % often == 0) {
                cout << "Energy is " << energy <<"\n";
            }
        }
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

inline void BaseFieldTheory::calculateGradientFlow(int pos){
    cout << "ERROR! you havent written a calculateGradientFlow function!\n";
}

void BaseFieldTheory::save(const char * savepath){
    cout << "saving to file " <<savepath << "\n";
    ofstream outf(savepath);
    //first output the derived class name to ensure loading the correct theory type!
    //first output dimensions of the field theory!
    for(int i = 0; i < dim; i++) {
        outf << size[i] << " " ;
    }
    cout << "\n";
    //second output the parameters of the field theory
    for(int i = 0; i < parameters.size(); i++) {
        outf << parameters[i] << " " ;
    }
    cout << "\n";
    //third output the fields (this will determine autmatically if dynamic and need to output the time derivative also)
    fields.save_fields(outf, getTotalSize());
    //close the file
    outf.close();
    cout << "Saving complete!\n";
}


void BaseFieldTheory::load(const char * loadpath){
    cout << "loading from file " << loadpath << "\n";
    ifstream infile(loadpath);
    //first load the derived class name to ensure loading the correct theory type!
    //first load dimensions of the field theory!
    for(int i = 0; i < dim; i++){
        infile >> size[i];
    }
    // and resize the theory to the loaded dimensions
        fields.resize(size);
    //second load the parameters of the field theory
    for(int i = 0; i < parameters.size(); i++){
        infile >> *parameters[i];
    }
    //third load the fields (this will determine autmatically if dynamic and need to output the time derivative also)
    fields.load_fields(infile, getTotalSize());
    //close the file
    infile.close();
    cout << "loading completed!\n";
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

}
 // End FTPL namespace
