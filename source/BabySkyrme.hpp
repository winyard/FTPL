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
		inline virtual void __attribute__((always_inline)) calculateGradientFlow(int pos) final;
		inline virtual void __attribute__((always_inline)) RK4calc(int pos) final;
		inline virtual double __attribute__((always_inline)) calculateEnergy(int pos) final;
		inline virtual __attribute__((always_inline)) vector<double> calculateDynamicEnergy(int pos) final;
	//Other Useful functions
		void initialCondition(int B, double x_in, double y_in, double phi);
        void addSoliton(int B, double x_in, double y_in, double phi);
		double initial(double r);
	//required functions
		BabySkyrmeModel(const char * filepath, bool isDynamic = false);
		BabySkyrmeModel(int width, int height, bool isDynamic = false);
		~BabySkyrmeModel(){};
		void setParameters(double mu_in, double mpi_in);
		double calculateCharge(int pos);
		double getCharge(){return charge;};
        void virtual correctMetric(int loop);
        Eigen::Matrix2d g; // metric
	private:
	// parameters
		double mu, mpi;
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
	f = createField(f, isDynamic, true);
    Eigen::Vector3d minimum(-0.01,-0.01,-0.01);
    Eigen::Vector3d maximum(0.01,0.01,0.01);
    f->min = minimum;
    f->max = maximum;
    addParameter(&mu, "mu"); // need to add any parameters that you want to be saved/loaded when using the .save/.load function (always add them in the same order!)
    addParameter(&mpi, "mpi");
    normalise_me = true;
    Eigen::MatrixXd met(2,2); // metric
    g(0,0) = 1.0;
    g(1,1) = 4.0;
    g(0,1) = sqrt(3.0);
    g(1,0) = sqrt(3.0);
};


BabySkyrmeModel::BabySkyrmeModel(const char * filename, bool isDynamic): BaseFieldTheory(2, {2,2}, isDynamic){
    // mearly place holders so the fields can be initialised
	f = createField(f, isDynamic, true);
    addParameter(&mu, "mu");
    addParameter(&mpi, "mpi");
    load(filename);
    normalise_me = true;
    g(0,0) = 1.0;
    g(1,1) = 1.0;
    g(0,1) = 0.0;
    g(1,0) = 0.0;
};

//maths!
double BabySkyrmeModel::calculateEnergy(int pos){
	Eigen::Vector3d fx = single_derivative(f, 0, pos);
	Eigen::Vector3d fy = single_derivative(f, 1, pos);
    //return 0.5*(fx.squaredNorm() + fy.squaredNorm() + mu*mu*(fx.cross(fy).squaredNorm())) + mpi*mpi*(1.0 - f->data[pos][2]);
    //return 0.5*(g[0][0]*fx.squaredNorm() + g[1][1]*fy.squaredNorm() + 2.0*g[0][1]*fx.dot(fy) + mu*mu*(g[0][0]*g[1][1] - pow(g[0][1],2))*(fx.cross(fy).squaredNorm())) + mpi*mpi*(1.0 - f->data[pos][2]);
    //return 0.5*(g(0,0)*fx.squaredNorm() + g(1,1)*fy.squaredNorm() + 2.0*g(0,1)*fx.dot(fy) + mu*mu*(fx.cross(fy).squaredNorm())) + mpi*mpi*(1.0 - f->data[pos][2]);
        int N = 3;
        double r = pow(sqrt(pow(f->data[pos][0],2) + pow(f->data[pos][1],2)),N);
        double theta = N*atan2(f->data[pos][1],f->data[pos][0]);

    return 0.5*(g(0,0)*fx.squaredNorm() + g(1,1)*fy.squaredNorm() + 2.0*g(0,1)*fx.dot(fy) + mu*mu*(fx.cross(fy).squaredNorm()))
           + mpi*mpi*(1.0 - f->data[pos][2])
           + 2.0*mpi*mpi*pow(f->data[pos][1],2)*(1.0 - f->data[pos][2])
           + 2.5*pow(f->data[pos][0],4)
           + 3.0*(1+pow(r,2)-2.0*r*cos(theta))*(1.0 - f->data[pos][2]) ;
};

    void BabySkyrmeModel::correctMetric(int loop){
        if(loop%10 == 0) {
            double A = 0.0;
            double B = 0.0;
            double C = 0.0;
            for (int i = 0; i < getTotalSize(); i++) {
                if (inBoundary(i)) {
                    Eigen::Vector3d fx = single_derivative(f, 0, i);
                    Eigen::Vector3d fy = single_derivative(f, 1, i);

                    A += fx.squaredNorm() * getTotalSpacing();
                    B += fy.squaredNorm() * getTotalSpacing();
                    C += fx.dot(fy) * getTotalSpacing();

                }
            }

            if (C > 0.000001) {
                g(0, 0) = sqrt((1 + sqrt(1 + 4 * B * A / pow(C, 2))) / (2.0 * A / B));
                g(1, 1) = sqrt((1 + sqrt(1 + 4 * B * A / pow(C, 2))) / (2.0 * B / A));
                g(0, 1) = sqrt(g(0, 0) * g(1, 1) - 1.0);
                g(1, 0) = g(0, 1);

            } else {
                g(0, 0) = B / sqrt(A * B);
                g(1, 1) = A / sqrt(A * B);
                g(0, 1) = sqrt(g(0, 0) * g(1, 1) - 1.0 + 0.000000000001);
                g(1, 0) = g(0, 1);

            }
        }

    }

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


inline void BabySkyrmeModel::calculateGradientFlow(int pos){
        if(inBoundary(pos)) {
            Eigen::Vector3d fx = single_derivative(f, 0, pos);
            Eigen::Vector3d fy = single_derivative(f, 1, pos);
            Eigen::Vector3d fxx = double_derivative(f, 0, 0, pos);
            Eigen::Vector3d fyy = double_derivative(f, 1, 1, pos);
            Eigen::Vector3d fxy = double_derivative(f, 0, 1, pos);
            Eigen::Vector3d f0 = f->data[pos];

            /*Eigen::Vector3d gradient = fxx + fyy + mu * mu * (fxx*fy.squaredNorm() + fyy*fx.squaredNorm() + fx*(fxy.dot(fy) - fyy.dot(fx)) + fy*(fxy.dot(fx) - fxx.dot(fy)) - 2.0*fxy*fx.dot(fy) );
            gradient[2] += mpi*mpi;
            double lagrange = -gradient.dot(f0);
            gradient += lagrange*f0;
            f->buffer[pos] = gradient;*/

            /*Eigen::Vector3d gradient = g[0][0]*fxx + g[1][1]*fyy + 2.0*g[0][1]*fxy
                                       + mu * mu * (g[0][0]*g[1][1] - pow(g[0][1],2)) (fxx*fy.squaredNorm() + fyy*fx.squaredNorm() + fx*(fxy.dot(fy) - fyy.dot(fx)) + fy*(fxy.dot(fx) - fxx.dot(fy)) - 2.0*fxy*fx.dot(fy) );
            gradient[2] += mpi*mpi;
            double lagrange = -gradient.dot(f0);
            gradient += lagrange*f0;
            f->buffer[pos] = gradient;*/

            Eigen::Vector3d gradient = g(0,0)*fxx + g(1,1)*fyy + 2.0*g(0,1)*fxy
                                       + mu * mu * (fxx*fy.squaredNorm() + fyy*fx.squaredNorm() + fx*(fxy.dot(fy) - fyy.dot(fx)) + fy*(fxy.dot(fx) - fxx.dot(fy)) - 2.0*fxy*fx.dot(fy) );
            gradient[2] += mpi*mpi;

            gradient[2] += mpi*mpi*2.0*pow(f->data[pos][1],2);
            gradient[1] += -mpi*mpi*4.0*f->data[pos][1]*(1.0 - f->data[pos][2]);
            gradient[0] += -10.0*pow(f->data[pos][0],3);

               int N = 3;
               double r = sqrt( pow(f->data[pos][0],2) + pow(f->data[pos][1],2)   );
               double theta = atan2( f->data[pos][1], f->data[pos][0]);

            gradient[0] += -3.0*( 2.0*N*f->data[pos][0]*pow(r,2*N-2)  -   2.0*N*f->data[pos][0]*pow(r,N-2)*cos(theta) - N*2.0*pow(r,N-1)*f->data[pos][1]*sin(N*theta) )*(1.0 - f->data[pos][2]);
            gradient[1] += -3.0*( 2.0*N*f->data[pos][1]*pow(r,2*N-2)  -   2.0*N*f->data[pos][1]*pow(r,N-2)*cos(theta) + N*2.0*pow(r,N-1)*f->data[pos][0]*sin(N*theta) )*(1.0 - f->data[pos][2]);
            gradient[2] += 3.0*(1+pow(r,2*N)-2.0*pow(r,N)*cos(N*theta));


            double lagrange = -gradient.dot(f0);
            gradient += lagrange*f0;
            f->buffer[pos] = gradient;

        } else{
            Eigen::Vector3d zero(0,0,0);
            f->buffer[pos] =zero;
        }
    }

void BabySkyrmeModel::addSoliton(int B, double x_in, double y_in, double phi){
	if(dynamic) {
		Eigen::Vector3d value(0,0,0);
		f->fill_dt(value);
	}
    vector<int> pos(dim);
	double xmax = size[0]*spacing[0]/2.0;
	double ymax = size[1]*spacing[1]/2.0;
	for(int i = bdw[0]; i < size[0]-bdw[1]; i++){
	for(int j = bdw[2]; j < size[1]-bdw[3]; j++){
        pos[0] = i;
        pos[1] = j;
        {
        double x = i * spacing[0] - xmax;
        double y = j * spacing[1] - ymax;
        double r = sqrt((x - x_in) * (x - x_in) + (y - y_in) * (y - y_in));
        double theta = atan2(y_in - y, x_in - x) - phi;
        Eigen::Vector3d value(sin(initial(r)) * cos(B * theta), sin(initial(r)) * sin(B * theta), cos(initial(r)));
        int point = pos[0]+pos[1]*size[0];
        Eigen::Vector3d f0 = f->data[point];
        if (value[2] < -0.99999) {
            f->setData(value, {i, j});
        } else if (f0[2] < -0.99999) {
            f->setData(f0, {i, j});
        } else {
            double W1[2], W2[2], W3[2];
            W1[0] = value[0] / (1.0 + value[2]);
            W1[1] = value[1] / (1.0 + value[2]);

            W2[0] = f0[0] / (1.0 + f0[2]);
            W2[1] = f0[1] / (1.0 + f0[2]);

            W3[0] = W1[0] + W2[0];
            W3[1] = W1[1] + W2[1];

            value[2] = (1.0 - W3[0]*W3[0] - W3[1]*W3[1]) / (1.0 + pow(W3[0], 2) + pow(W3[1], 2));
            value[0] = 2.0 * W3[0] / (1.0 + pow(W3[0], 2) + pow(W3[1], 2));
            value[1] = 2.0 * W3[1] / (1.0 + pow(W3[0], 2) + pow(W3[1], 2));

            f->setData(value, {i, j});
        }

    }
	}}
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
                {
                    double x = i * spacing[0] - xmax;
                    double y = j * spacing[1] - ymax;
                    double r = sqrt((x - x_in) * (x - x_in) + (y - y_in) * (y - y_in));
                    double theta = atan2(y_in - y, x_in - x) - phi;
                    Eigen::Vector3d value(sin(initial(r)) * cos(B * theta), sin(initial(r)) * sin(B * theta), cos(initial(r)));
                    f->setData(value, {i, j});
                }
            }}
    }

	double BabySkyrmeModel::initial(double r)
	{
		double a;
		double initialradius = 2.0;
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