/**
 * Field Theory Processing Library
 * Copyright Thomas Winyard 2016
 */

#ifndef BABY_SKYRME_H
#define BABY_SKYRME_H

#include <cmath>
#include "FieldTheories.hpp"
#include <Eigen/Dense>

using namespace std;

namespace FTPL {

class BabySkyrmeModel : public BaseFieldTheory {
    public:
	//place your fields here (likely to need to acces them publicly)
		Field<Eigen::VectorXd> * f; // using Xd (general size) is dangerous as sizes are often derived! ALways define a dimension ahead of time!
    //maths (higher up functions run slightly faster these are the important ones!)
		CUDA_HOSTDEV inline virtual void __attribute__((always_inline)) calculateGradientFlow(int pos) final;
		CUDA_HOSTDEV inline virtual void __attribute__((always_inline)) RK4calc(int pos) final;
		CUDA_HOSTDEV inline virtual double __attribute__((always_inline)) calculateEnergy(int pos) final;
		CUDA_HOSTDEV inline virtual __attribute__((always_inline)) vector<double> calculateDynamicEnergy(int pos) final;
	//Other Useful functions
		CUDA_HOSTDEV void initialCondition(int B, double x_in, double y_in, double phi);
		CUDA_HOSTDEV double initial(double r);
	//required functions
		CUDA_HOSTDEV BabySkyrmeModel(const char * filepath, bool isDynamic = false);
		CUDA_HOSTDEV BabySkyrmeModel(int width, int height, bool isDynamic = false);
		CUDA_HOSTDEV ~BabySkyrmeModel(){};
		CUDA_HOSTDEV void setParameters(double mu_in, double mpi_in);
		CUDA_HOSTDEV double calculateCharge(int pos);
		CUDA_HOSTDEV double getCharge(){return charge;};
		CUDA_HOSTDEV void updateCharge();
	private:
	// parameters
		double mu, mpi;
		double charge;
		vector<double> chargedensity;
};

inline void BabySkyrmeModel::RK4calc(int pos){
	if(inBoundary(pos)) {
		Eigen::Vector3d fx = single_derivative(f, 0, pos);
		Eigen::Vector3d fy = single_derivative(f, 1, pos);
		Eigen::Vector3d fxx = double_derivative(f, 0, 0, pos);
		Eigen::Vector3d fyy = double_derivative(f, 1, 1, pos);
		Eigen::Vector3d fxy = double_derivative(f, 0, 1, pos);
		Eigen::Vector3d ftx = single_time_derivative(f, 0, pos);//need to add dt
		Eigen::Vector3d fty = single_time_derivative(f, 1, pos);//need to add dt
		Eigen::Vector3d f0 = f->data[pos];
		Eigen::Vector3d ft = f->dt[pos];

//tryadding .noalias to A could be faster
		//Eigen::Matrix3d A = Eigen::Matrix3d::Identity()*(1.0 + mu*mu*(fx.squaredNorm() + fy.squaredNorm()))-mu*mu*(fx*fx.transpose()+fy*fy.transpose());
		Eigen::Matrix3d A = Eigen::Matrix3d::Identity()*(1.0 + mu*mu*(fx.squaredNorm() + fy.squaredNorm()))-mu*mu*(fx*fx.transpose()+fy*fy.transpose());

		/*Eigen::Vector3d b = fxx + fyy + mu * mu * (ft * (fxx.dot(ft) + fyy.dot(ft)) + 2.0 * ftx * ft.dot(fx) +
												   2.0 * fty * ft.dot(fy) - (fxx + fyy) * ft.squaredNorm() -
												   fx * ft.dot(ftx) - fy * ft.dot(fty) -
												   ft * (fx.dot(ftx) + fy.dot(fty))
												   - fxx * fx.squaredNorm() - fyy * fy.squaredNorm() -
												   2.0*fxy * fx.dot(fy) - fx * (fxx + fyy).dot(fx) -
												   fy * (fxx + fyy).dot(fy) -
												   fx * (fxx.dot(fx) + fxy.dot(fy)) -
												   fy * (fxy.dot(fx) + fyy.dot(fy))
												   + (fxx + fyy) * (fx.squaredNorm() + fy.squaredNorm()) +
												   2.0 * fx * (fxy.dot(fy) + fxx.dot(fx)) +
												   2.0 * fy * (fyy.dot(fy) + fxy.dot(fx)));*/
		Eigen::Vector3d b = fxx + fyy + mu * mu * (ft * (fxx.dot(ft) + fyy.dot(ft)) + 2.0 * ftx * ft.dot(fx)
					+ 2.0 * fty * ft.dot(fy) - (fxx + fyy) * ft.squaredNorm() - fx * ft.dot(ftx) - fy * ft.dot(fty)
					- ft * (fx.dot(ftx) + fy.dot(fty)) - 2.0*fxy * fx.dot(fy)
					- fx * fyy.dot(fx) - fy * fxx.dot(fy) - fx * fxy.dot(fy) - fy * fxy.dot(fx)
					+ fxx*fy.squaredNorm() + fyy*fx.squaredNorm() + 2.0 * fx*fxy.dot(fy) + 2.0 * fy * fxy.dot(fx));

			b[2] += mpi * mpi;
			double lagrange = -0.5 * b.dot(f0) - 0.5 * ft.squaredNorm() * (1.0 + mu*mu*(fx.squaredNorm() + fy.squaredNorm()));
			b += 2.0 * lagrange * f0;
			//f0 = A.colPivHouseholderQr().solve(b);//try some different solvers!
			f0 = A.ldlt().solve(b);
			//f0 = A.jacobiSvd().solve(b);
			f->k0_result[pos] = dt * ft;
			f->k1_result[pos] = dt * f0;
			//if(f0[0] > 0.01){cout << "found result is " << f0 << " given as " << f->k0_result[pos] << "\n and the other " << f->k1_result[pos] << "\n";}
	}
	else{
		f->k0_result[pos] = Eigen::Vector3d::Zero();
		f->k1_result[pos] = Eigen::Vector3d::Zero();
	}
}

void BabySkyrmeModel::setParameters(double mu_in, double mpi_in){
    mu = mu_in;
    mpi = mpi_in;
}

BabySkyrmeModel::BabySkyrmeModel(int width, int height, bool isDynamic): BaseFieldTheory(2, {width,height}, isDynamic) {
	//vector<int> sizein = {width, height};
	//BaseFieldTheory(2,sizein);
	f = createField(f, isDynamic);
    Eigen::Vector3d minimum(-0.01,-0.01,-0.01);
    Eigen::Vector3d maximum(0.01,0.01,0.01);
    f->min = minimum;
    f->max = maximum;
    addParameter(&mu); // need to add any parameters that you want to be saved/loaded when using the .save/.load function (always add them in the same order!)
    addParameter(&mpi);
	chargedensity.resize(getTotalSize()); // need to resize charge density as this is not included in the BaseFieldTheory
};


BabySkyrmeModel::BabySkyrmeModel(const char * filename, bool isDynamic): BaseFieldTheory(2, {2,2}, isDynamic){
    // mearly place holders so the fields can be initialised
	f = createField(f, isDynamic);
    addParameter(&mu);
    addParameter(&mpi);
    load(filename);
	chargedensity.resize(getTotalSize());
};

//maths!
double BabySkyrmeModel::calculateEnergy(int pos){
	Eigen::Vector3d fx = single_derivative(f, 0, pos);
	Eigen::Vector3d fy = single_derivative(f, 1, pos);

    return 0.5*(fx.squaredNorm() + fy.squaredNorm() + mu*mu*(fx.cross(fy).squaredNorm())) + mpi*mpi*(1.0 - f->data[pos][2]);
};

vector<double> BabySkyrmeModel::calculateDynamicEnergy(int pos){
    Eigen::Vector3d fx = single_derivative(f, 0, pos);
    Eigen::Vector3d fy = single_derivative(f, 1, pos);
	Eigen::Vector3d ft = f->dt[pos];
    vector<double> result(2);

    result[0] = 0.5*(fx.squaredNorm() + fy.squaredNorm() + mu*mu*((fx.cross(fy)).squaredNorm())) + mpi*mpi*(1.0 - f->data[pos][2]);
	result[1] = 0.5*ft.squaredNorm() + 0.5*mu*mu*(ft.cross(fx).squaredNorm() + ft.cross(fy).squaredNorm() );
    return result;
};

double BabySkyrmeModel::calculateCharge(int pos){
	Eigen::Vector3d fx = single_derivative(f, 0, pos);
	Eigen::Vector3d fy = single_derivative(f, 1, pos);
	return (1.0/(4.0*M_PI))*( (f->data[pos]).dot(fx.cross(fy)) );
};

void BabySkyrmeModel::updateCharge(){
	double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
	for(int i = 0; i < getTotalSize(); i++) {
        if (inBoundary(i)) {
            double buffer = calculateCharge(i);
            chargedensity[i] = buffer;
            sum += buffer;
        }
	}
	charge = sum*spacing[0]*spacing[1];

};

inline void BabySkyrmeModel::calculateGradientFlow(int pos){
        if(inBoundary(pos)) {
            Eigen::Vector3d fx = single_derivative(f, 0, pos);
            Eigen::Vector3d fy = single_derivative(f, 1, pos);
            Eigen::Vector3d fxx = double_derivative(f, 0, 0, pos);
            Eigen::Vector3d fyy = double_derivative(f, 1, 1, pos);
            Eigen::Vector3d fxy = double_derivative(f, 0, 1, pos);
            Eigen::Vector3d f0 = f->data[pos];

            Eigen::Vector3d gradient = fxx + fyy + mu * mu * (fxx*fy.squaredNorm() + fyy*fx.squaredNorm() + fx*(fxy.dot(fy) - fyy.dot(fx)) + fy*(fxy.dot(fx) - fxx.dot(fy)) - 2.0*fxy*fx.dot(fy) );
            gradient[2] += mpi*mpi;
            double lagrange = -gradient.dot(f0);
            gradient += lagrange*f0;
            f->buffer[pos] = gradient;
        } else{
            Eigen::Vector3d zero(0,0,0);
            f->buffer[pos] =zero;
        }
    }

void BabySkyrmeModel::initialCondition(int B, double x_in, double y_in, double phi){
	if(dynamic) {
		Eigen::Vector3d value(0,0,0);
		f->fill_dt(value);
	}
	double xmax = size[0]*spacing[0]/2.0;
	double ymax = size[1]*spacing[1]/2.0;
	for(int i = bdw[0]; i < size[0]-bdw[1]; i++){
	for(int j = bdw[2]; j < size[1]-bdw[3]; j++){
		double x = i*spacing[0]-xmax;
		double y = j*spacing[1]-ymax;
		double r = sqrt((x-x_in)*(x-x_in) + (y-y_in)*(y-y_in));
		double theta = atan2(y_in-y,x_in-x) - phi;
		Eigen::Vector3d value(sin(initial(r))*cos(B*theta), sin(initial(r))*sin(B*theta), cos(initial(r)));
		f->setData(value, {i,j});
	}}
}

	double BabySkyrmeModel::initial(double r)
	{
		double a;
		double initialradius = 5.0;
		if(r > initialradius)
		{
			a = 0;
		}
		else
		{
			a=M_PI*(1.0 - r/initialradius);
		}
		return (a);
	}

}

#endif