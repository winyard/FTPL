/**
 * Field Theory Processing Library
 * Copyright Thomas Winyard 2016
 */

#include <cmath>
#include "FieldTheories.hpp"
#include <Eigen/Dense>
#include <chrono>

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

class BabySkyrmeModel : public BaseFieldTheory {
    public:
		Field<Eigen::VectorXd> * f;
    //maths (higher up functions run slightly faster)
        inline virtual void calculateGradientFlow(vector<int> pos) final;
	//required functions
        BabySkyrmeModel(const char * filepath);
        BabySkyrmeModel(int width, int height);
		~BabySkyrmeModel(){};
 //       BabySkyrmeModel(BabySkyrmeModel * otherBabySkyrmeModel, Transformation T = IntensityTransformation(DEFAULT));//Allows you to act with a transformation (rotation, isorotation etc.)
        void save(const char * filename);
	//The maths
		inline virtual double calculateEnergy(vector<int> pos) final;
		double calculateCharge(int i,int j);
		double calculateCharge(vector<int> pos);
        double getCharge(){return charge;};
		void updateCharge();
		void setCharge(int i, int j, double value);
		void initialCondition(int B, double x_in, double y_in, double phi);
        virtual void gradientFlow(int iterations, int often) final;
		double initial(double r);

        void setParameters(double mu_in, double mpi_in);
	private:
	// parameters and fields
		double mu, mpi;
		double charge;
		vector<double> chargedensity;
};

    void BabySkyrmeModel::gradientFlow(int iterations, int often) { // needs to be updated to gradient flow the entire field then update the field!
        for(int no = 0; no < iterations; no++) {
            //Timer tmr;
            BaseFieldTheory::gradientFlow(10);
            //cout << "Gradient Flow took " << tmr.elapsed() << "\n";
            //tmr.reset();
            f->normalise();
            //cout << "Normalisation took " << tmr.elapsed() << "\n";
            //tmr.reset();
            if (no % often == 0) {
                double storeEnergy = energy;
                updateEnergy();
                if(energy >= storeEnergy){dt = 0.5*dt;}//else{dt = 1.01*dt;};
                cout << "Energy is " << energy <<"\n";
            }
            //cout << "Energy took " << tmr.elapsed() << "\n";
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
double BabySkyrmeModel::calculateEnergy(vector<int> pos){
	Eigen::Vector3d fx = single_derivative(f, 0, pos);
	Eigen::Vector3d fy = single_derivative(f, 1, pos);
    Eigen::Vector3d vac(0,0,1);
    return 0.5*(fx.squaredNorm() + fy.squaredNorm() + mu*mu*(fx.cross(fy).squaredNorm())) + mpi*mpi*(1.0 - f->getData(pos)[2]);
};

double BabySkyrmeModel::calculateCharge(vector<int> pos){
	Eigen::Vector3d fx = single_derivative(f, 0, pos);
	Eigen::Vector3d fy = single_derivative(f, 1, pos);
	return (1.0/(4.0*M_PI))*( (f->getData(pos)).dot(fx.cross(fy)) );
};

double BabySkyrmeModel::calculateCharge(int i, int j){
vector<int> inpos = {i,j};
return calculateCharge(inpos);

};

void BabySkyrmeModel::updateCharge(){
	double sum = 0.0;
	for(int i = bdw[0]; i < size[0]-bdw[1]; i++){
	for(int j = bdw[2]; j < size[1]-bdw[3]; j++){
		double buffer = calculateCharge(i,j);
		setCharge(i,j,buffer);
		sum += buffer;
	}}
	charge = sum*spacing[0]*spacing[1];

};

void BabySkyrmeModel::setCharge(int i, int j, double value){
	chargedensity[i + spacing[0]*j] = value;
}

inline void BabySkyrmeModel::calculateGradientFlow(vector<int> pos){
        if(inBoundary(pos)) {
            Eigen::Vector3d fx = single_derivative(f, 0, pos);
            Eigen::Vector3d fy = single_derivative(f, 1, pos);
            Eigen::Vector3d fxx = double_derivative(f, 0, 0, pos);
            Eigen::Vector3d fyy = double_derivative(f, 1, 1, pos);
            Eigen::Vector3d fxy = double_derivative(f, 0, 1, pos);
            Eigen::Vector3d f0 = f->getData(pos);
            Eigen::Vector3d vac(0, 0, 1);

            Eigen::Vector3d gradient = fxx + fyy + mu * mu * (fxx*fy.squaredNorm() + fyy*fx.squaredNorm()
                                         +fx*fxy.dot(fy) + fy*fxy.dot(fx) - 2.0*fxy*fx.dot(fy) - fx*fyy.dot(fx) - fy*fxx.dot(fy));
            gradient[2] += mpi*mpi;
            double lagrange = -gradient.dot(f0);
            gradient += lagrange*f0;
            f->setBuffer(gradient, pos);
        } else{
            Eigen::Vector3d zero(0,0,0);
            f->setBuffer(zero, pos);
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
