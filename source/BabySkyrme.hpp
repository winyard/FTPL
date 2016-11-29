/**
 * Field Theory Processing Library
 * Copyright Thomas Winyard 2016
 */

#include <cmath>
#include "FieldTheories.hpp"
#include <Eigen/Dense>

using namespace std;

namespace FTPL {

class BabySkyrmeModel : public BaseFieldTheory {
    public:
		Field<Eigen::VectorXd> * f;
    //maths (higher up functions run slightly faster)
        inline virtual void __attribute__((always_inline)) calculateGradientFlow(int pos) final;
		inline virtual void __attribute__((always_inline)) RK4calc(int k, int pos) final;
	//required functions
        BabySkyrmeModel(const char * filepath);
        BabySkyrmeModel(int width, int height);
		~BabySkyrmeModel(){};
 //       BabySkyrmeModel(BabySkyrmeModel * otherBabySkyrmeModel, Transformation T = IntensityTransformation(DEFAULT));//Allows you to act with a transformation (rotation, isorotation etc.)
        void save(const char * filename);
	//The maths
		inline virtual double __attribute__((always_inline)) calculateEnergy(int pos) final;
		double calculateCharge(int pos);
        double getCharge(){return charge;};
		void updateCharge();
		void setCharge(int i, int j, double value);
		void initialCondition(int B, double x_in, double y_in, double phi);
		double initial(double r);
        void setParameters(double mu_in, double mpi_in);
        void calculateAandb(vector<int> pos);
	private:
	// parameters and fields
		double mu, mpi;
		double charge;
		vector<double> chargedensity;
};

inline RK4calc(int k, int pos){
	if(inBoundary(pos)) {
		Eigen::Vector3d fx = single_derivative(f, 0, pos);
		Eigen::Vector3d fy = single_derivative(f, 1, pos);
		Eigen::Vector3d fxx = double_derivative(f, 0, 0, pos);
		Eigen::Vector3d fyy = double_derivative(f, 1, 1, pos);
		Eigen::Vector3d fxy = double_derivative(f, 0, 1, pos);
		Eigen::Vector3d f0 = f->data[pos];


	}
}

void BabySkyrmeModel::setParameters(double mu_in, double mpi_in){
    mu = mu_in;
    mpi = mpi_in;
}

BabySkyrmeModel::BabySkyrmeModel(int width, int height): BaseFieldTheory(2, {width,height}) {
	//vector<int> sizein = {width, height};
	//BaseFieldTheory(2,sizein);
	f = createField(f, false);
	chargedensity.resize(getTotalSize());
};


BabySkyrmeModel::BabySkyrmeModel(const char * filename): BaseFieldTheory(dim,{2,2}){
    // mearly place holders so the fields can be initialised
	f = createField(f, false);

    load(filename);
	chargedensity.resize(getTotalSize());
};

//maths!
double BabySkyrmeModel::calculateEnergy(int pos){
	Eigen::Vector3d fx = single_derivative(f, 0, pos);
	Eigen::Vector3d fy = single_derivative(f, 1, pos);
    return 0.5*(fx.squaredNorm() + fy.squaredNorm() + mu*mu*(fx.cross(fy).squaredNorm())) + mpi*mpi*(1.0 - f->data[pos][2]);
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

void BabySkyrmeModel::setCharge(int i, int j, double value){
	chargedensity[i + spacing[0]*j] = value;
}

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
	double xmax = size[0]*spacing[0]/2.0;
	double ymax = size[1]*spacing[1]/2.0;
	for(int i = bdw[0]; i < size[0]-bdw[1]; i++){
	for(int j = bdw[2]; j < size[1]-bdw[3]; j++){
		double x = i*spacing[0]-xmax;
		double y = j*spacing[1]-ymax;
		double r = sqrt((x-x_in)*(x-x_in) + (y-y_in)*(y-y_in));
		double theta = atan2(y_in-y,x_in-x) - phi;
		Eigen::Vector2i pos(i,j);
		Eigen::Vector3d value(sin(initial(r))*cos(B*theta), sin(initial(r))*sin(B*theta), cos(initial(r)));
		f->setData(value, pos);
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
