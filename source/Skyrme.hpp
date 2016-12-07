/**
 * Field Theory Processing Library
 * Copyright Thomas Winyard 2016
 */

#include <cmath>
#include "FieldTheories.hpp"
#include <Eigen/Dense>
#include "RationalMaps.hpp"

using namespace std;

namespace FTPL {

    class SkyrmeModel : public BaseFieldTheory {
    public:
        //place your fields here (likely to need to acces them publicly)
        Field<Eigen::VectorXd> * f;
        //maths (higher up functions run slightly faster these are the important ones!)
        inline virtual void __attribute__((always_inline)) calculateGradientFlow(int pos) final;
        inline virtual void __attribute__((always_inline)) RK4calc(int pos) final;
        inline virtual double __attribute__((always_inline)) calculateEnergy(int pos) final;
        inline virtual __attribute__((always_inline)) vector<double> calculateDynamicEnergy(int pos) final;
        //Other Useful functions
        void initialCondition(int B, double x_in, double y_in, double z_in, double phi);
        double initial(double r);
        //required functions
        SkyrmeModel(const char * filepath, bool isDynamic = false);
        SkyrmeModel(int width, int height, int depth, bool isDynamic = false);
        ~SkyrmeModel(){};
        void setParameters(double Fpi_in, double epi_in, double mpi);
        double calculateCharge(int pos);
        double getCharge(){return charge;};
        void updateCharge();
    private:
        // parameters
        double Fpi, epi, mpi;
        double charge;
        vector<double> chargedensity;
    };

    inline void SkyrmeModel::RK4calc(int pos){
        if(inBoundary(pos)) {
            Eigen::Vector4d fx = single_derivative(f, 0, pos);
            Eigen::Vector4d fy = single_derivative(f, 1, pos);
            Eigen::Vector4d fz = single_derivative(f, 2, pos);
            Eigen::Vector4d fxx = double_derivative(f, 0, 0, pos);
            Eigen::Vector4d fyy = double_derivative(f, 1, 1, pos);
            Eigen::Vector4d fzz = double_derivative(f, 2, 2, pos);
            Eigen::Vector4d fxy = double_derivative(f, 0, 1, pos);
            Eigen::Vector4d fxz = double_derivative(f, 0, 2, pos);
            Eigen::Vector4d fyz = double_derivative(f, 1, 2, pos);
            Eigen::Vector4d ftx = single_time_derivative(f, 0, pos);
            Eigen::Vector4d fty = single_time_derivative(f, 1, pos);
            Eigen::Vector4d ftz = single_time_derivative(f, 2, pos);
            Eigen::Vector4d f0 = f->data[pos];
            Eigen::Vector4d ft = f->dt[pos];

//tryadding .noalias to A could be faster
            //Eigen::Matrix3d A = Eigen::Matrix3d::Identity()*(1.0 + mu*mu*(fx.squaredNorm() + fy.squaredNorm()))-mu*mu*(fx*fx.transpose()+fy*fy.transpose());
            Eigen::Matrix4d A = Eigen::Matrix4d::Identity()*(Fpi*Fpi/4.0 + (1.0/(epi*epi))*( fx.squaredNorm() + fy.squaredNorm() + fz.squaredNorm() ))-(1.0/(epi*epi))*( fx*fx.transpose()+fy*fy.transpose()+fz*fz.transpose() );

           /* Eigen::Vector4d b = (Fpi*Fpi/4.0)*(fxx + fyy + fzz) + (1.0/(epi*epi))*(ft * (fxx.dot(ft) + fyy.dot(ft) + fzz.dot(ft)) + 2.0 * ftx * ft.dot(fx) +
                                                       2.0 * fty * ft.dot(fy) + 2.0*ftz * ft.dot(fz) - (fxx + fyy + fzz) * ft.squaredNorm() -
                                                       fx * ft.dot(ftx) - fy * ft.dot(fty) - fz*ft.dot(ftz)
                                                       - ft * (fx.dot(ftx) + fy.dot(fty) + fz.dot(ftz) )
                                                       - fxx * fx.squaredNorm() - fyy * fy.squaredNorm() - fzz * fz.squaredNorm()
                                                       - 2.0*fxy * fx.dot(fy) - 2.0*fxz * fx.dot(fz) - 2.0*fyz * fy.dot(fz)
                                                       - fx * (fxx + fyy + fzz).dot(fx) - fy * (fxx + fyy + fzz).dot(fy) - fz * (fxx + fyy + fzz).dot(fz)
                                                       - fx * (fxx.dot(fx) + fxy.dot(fy) + fxz.dot(fz)) - fy * (fxy.dot(fx) + fyy.dot(fy) + fyz.dot(fz))
                                                       - fz * (fxz.dot(fx) + fyz.dot(fy) + fzz.dot(fz))
                                                       + (fxx + fyy + fzz) * (fx.squaredNorm() + fy.squaredNorm() + fz.squaredNorm()) +
                                                       2.0 * fx * (fxy.dot(fy) + fxx.dot(fx) + fxz.dot(fz)) +
                                                       2.0 * fy * (fyy.dot(fy) + fxy.dot(fx) + fyz.dot(fz))
                                                       + 2.0 * fz * (fzz.dot(fy) + fxz.dot(fx) + fyz.dot(fy)) );*/

            Eigen::Vector4d b = (Fpi*Fpi/4.0)*( fxx + fyy + fzz) + (1.0/(epi*epi))* (ft * (fxx.dot(ft) + fyy.dot(ft) + fzz.dot(ft)) + 2.0 * ftx * ft.dot(fx)
                                   + 2.0 * fty * ft.dot(fy) + 2.0*ftz*ft.dot(fz) - (fxx + fyy + fzz) * ft.squaredNorm() - fx * ft.dot(ftx) - fy * ft.dot(fty) - fz * ft.dot(ftz)
                                   - ft * (fx.dot(ftx) + fy.dot(fty) + fz.dot(ftz)) - 2.0*fxy * fx.dot(fy) - 2.0*fxz * fx.dot(fz) - 2.0*fyz * fy.dot(fz)
                                   - fx * fyy.dot(fx) - fy * fxx.dot(fy) - fx*fzz.dot(fx) - fy*fzz.dot(fy) - fx * fxy.dot(fy) - fy * fxy.dot(fx) - fx*fxz.dot(fz) - fy*fyz.dot(fz)
                                         - fz * fyy.dot(fz) - fz * fxx.dot(fz) - fz * fyz.dot(fy) - fz * fxz.dot(fx)
                                   + fxx*fy.squaredNorm() + fyy*fx.squaredNorm() + fxx*fz.squaredNorm() + fyy*fz.squaredNorm() + fzz*fy.squaredNorm() + fzz*fx.squaredNorm() + 2.0 * fx*fxy.dot(fy) + 2.0 * fy * fxy.dot(fx)
                                            + 2.0 * fx*fxz.dot(fz) + 2.0 * fy * fyz.dot(fz) + 2.0 * fz*fxz.dot(fx) + 2.0 * fz * fyz.dot(fy));




           /* Eigen::Vector4d b = (Fpi*Fpi/4.0)*(fxx + fyy + fzz) + (1.0/(epi*epi))*(ft * (fxx.dot(ft) + fyy.dot(ft) + fzz.dot(ft)) + 2.0 * ftx * ft.dot(fx)
                                                       + 2.0 * fty * ft.dot(fy) + 2.0*ftz*ft.dot(fz) - (fxx + fyy + fzz) * ft.squaredNorm() - fx * ft.dot(ftx) - fy * ft.dot(fty)
                                                       - fz*ft.dot(ftz) - ft * (fx.dot(ftx) + fy.dot(fty) + fz.dot(ftz)) - 2.0*fxy * fx.dot(fy)
                                                       - 2.0*fxz*fx.dot(fz) - 2.0*fyz*fy.dot(fz) - fx * (fyy.dot(fx) + fzz.dot(fx)) - fy *(fxx.dot(fy)+fzz.dot(fy))
                                                       - fz*(fxx.dot(fz)+fyy.dot(fz)) + fxx*(fy.squaredNorm() + fz.squaredNorm()) + fyy*(fx.squaredNorm() + fz.squaredNorm())
                                                       + fzz*(fx.squaredNorm() + fy.squaredNorm()) + fx*(fxy.dot(fy) + fxz.dot(fz)) + fy*(fxy.dot(fx) + fyz.dot(fz))
                                                        + fz*(fxz.dot(fx) + fyz.dot(fy)));*/

            b[0] += mpi * mpi;
            double lagrange = -0.5 * b.dot(f0) - (Fpi*Fpi/8.0) * ft.squaredNorm() - ft.squaredNorm()*(1.0/(2.0*epi*epi))*(fx.squaredNorm() + fy.squaredNorm() + fz.squaredNorm());
            b += 2.0 * lagrange * f0;
            //f0 = A.colPivHouseholderQr().solve(b);//try some different solvers!
            f0 = A.ldlt().solve(b);
            //f0 = A.jacobiSvd().solve(b);
            f->k0_result[pos] = dt * ft;
            f->k1_result[pos] = dt * f0;
            //if(f0[0] > 0.01){cout << "found result is " << f0 << " given as " << f->k0_result[pos] << "\n and the other " << f->k1_result[pos] << "\n";}
        }
        else{
            f->k0_result[pos] = Eigen::Vector4d::Zero();
            f->k1_result[pos] = Eigen::Vector4d::Zero();
        }
    }

    void SkyrmeModel::setParameters(double Fpi_in, double epi_in, double mpi_in){
        Fpi = Fpi_in;
        epi = epi_in;
        mpi = mpi_in;
    }

    SkyrmeModel::SkyrmeModel(int width, int height, int depth, bool isDynamic): BaseFieldTheory(3, {width,height,depth}, isDynamic) {
        //vector<int> sizein = {width, height};
        //BaseFieldTheory(2,sizein);
        f = createField(f, isDynamic);
        Eigen::Vector4d minimum(-0.01,-0.01,-0.01,-0.01);
        Eigen::Vector4d maximum(0.01,0.01,0.01,0.01);
        f->min = minimum;
        f->max = maximum;
        chargedensity.resize(getTotalSize());
    };


    SkyrmeModel::SkyrmeModel(const char * filename, bool isDynamic): BaseFieldTheory(3, {10,10,10}, isDynamic){
        // mearly place holders so the fields can be initialised
        f = createField(f, isDynamic);
        Eigen::Vector4d minimum(-0.01,-0.01,-0.01,-0.01);
        Eigen::Vector4d maximum(0.01,0.01,0.01,0.01);
        f->min = minimum;
        f->max = maximum;
        load(filename);
        chargedensity.resize(getTotalSize());
    };

     double SkyrmeModel::calculateEnergy(int pos){
        Eigen::Vector4d fx = single_derivative(f, 0, pos);
        Eigen::Vector4d fy = single_derivative(f, 1, pos);
        Eigen::Vector4d fz = single_derivative(f, 2, pos);

        return ((1.0)/(12.0*M_PI*M_PI))*((Fpi*Fpi/8.0)*(fx.squaredNorm()+fy.squaredNorm()+fz.squaredNorm())
               + (1.0/(2.0*epi*epi))*(fx.squaredNorm()*(fy.squaredNorm()+fz.squaredNorm()) + fy.squaredNorm()*fz.squaredNorm()
               - fx.dot(fy)*fx.dot(fy) - fx.dot(fz)*fx.dot(fz) - fy.dot(fz)*fy.dot(fz) ) + mpi*mpi*(1.0 - f->data[pos][0]));
    };

    vector<double> SkyrmeModel::calculateDynamicEnergy(int pos){
        Eigen::Vector4d fx = single_derivative(f, 0, pos);
        Eigen::Vector4d fy = single_derivative(f, 1, pos);
        Eigen::Vector4d fz = single_derivative(f, 2, pos);
        Eigen::Vector4d ft = f->dt[pos];
        vector<double> result(2);

        result[0] = ((1.0)/(12.0*M_PI*M_PI))*((Fpi*Fpi/8.0)*(fx.squaredNorm()+fy.squaredNorm()+fz.squaredNorm())
                    + (1.0/(2.0*epi*epi))*(fx.squaredNorm()*(fy.squaredNorm()+fz.squaredNorm()) + fy.squaredNorm()*fz.squaredNorm()
                    - fx.dot(fy)*fx.dot(fy) - fx.dot(fz)*fx.dot(fz) - fy.dot(fz)*fy.dot(fz) ) + mpi*mpi*(1.0 - f->data[pos][0]));
        result[1] = ((1.0)/(12.0*M_PI*M_PI))*((Fpi*Fpi/8.0)*ft.squaredNorm() + 0.5*(1.0/(epi*epi))*((fx.squaredNorm()+fy.squaredNorm()+fz.squaredNorm())*ft.squaredNorm() - fx.dot(ft)*fx.dot(ft)- fy.dot(ft)*fy.dot(ft)- fz.dot(ft)*fz.dot(ft)));
        return result;
    };

    double SkyrmeModel::calculateCharge(int pos){
        Eigen::Vector4d fx = single_derivative(f, 0, pos);
        Eigen::Vector4d fy = single_derivative(f, 1, pos);
        Eigen::Vector4d fz = single_derivative(f, 2, pos);
        return 0.2;//(1.0/(2.0*M_PI*M_PI))*( levicivita*df*df*df*f(f->data[pos]).dot(fx));
    };

    void SkyrmeModel::updateCharge(){
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

    inline void SkyrmeModel::calculateGradientFlow(int pos){
        if(inBoundary(pos)) {
            Eigen::Vector4d fx = single_derivative(f, 0, pos);
            Eigen::Vector4d fy = single_derivative(f, 1, pos);
            Eigen::Vector4d fz = single_derivative(f, 2, pos);
            Eigen::Vector4d fxx = double_derivative(f, 0, 0, pos);
            Eigen::Vector4d fyy = double_derivative(f, 1, 1, pos);
            Eigen::Vector4d fzz = double_derivative(f, 2, 2, pos);
            Eigen::Vector4d fxy = double_derivative(f, 0, 1, pos);
            Eigen::Vector4d fxz = double_derivative(f, 0, 2, pos);
            Eigen::Vector4d fyz = double_derivative(f, 1, 2, pos);
            Eigen::Vector4d f0 = f->data[pos];

            Eigen::Vector4d gradient = (Fpi*Fpi/4.0)*(fxx + fyy + fzz) + (1.0/epi) * (fxx*(fy.squaredNorm()+fz.squaredNorm()) + fyy*(fx.squaredNorm()+fz.squaredNorm()) + fzz*(fx.squaredNorm()+fy.squaredNorm())
                                       + fx*(fxy.dot(fy) + fxz.dot(fz) - fyy.dot(fx) - fzz.dot(fx)) + fy*(fxy.dot(fx) + fyz.dot(fz) - fxx.dot(fy) - fzz.dot(fy))
                                       + fz*(fxz.dot(fx) + fyz.dot(fy) - fxx.dot(fz) - fyy.dot(fz)) - 2.0*(fxy*fx.dot(fy) + fxz*fx.dot(fz) + fyz*fy.dot(fz)  ) );
            gradient[0] += mpi*mpi;
            double lagrange = -gradient.dot(f0);
            gradient += lagrange*f0;
            f->buffer[pos] = gradient;
        } else{
            Eigen::Vector4d zero(0,0,0,0);
            f->buffer[pos] =zero;
        }
    }

    void SkyrmeModel::initialCondition(int B, double x_in, double y_in, double z_in, double phi){
        if(dynamic) {
            Eigen::Vector4d value(0,0,0,0);
            f->fill_dt(value);
        }
        double xmax = size[0]*spacing[0]/2.0;
        double ymax = size[1]*spacing[1]/2.0;
        double zmax = size[2]*spacing[2]/2.0;
        for(int i = bdw[0]; i < size[0]-bdw[1]; i++){
            for(int j = bdw[2]; j < size[1]-bdw[3]; j++){
                for(int k = bdw[4]; k < size[2]-bdw[5]; k++){
                double x = i*spacing[0]-xmax;
                double y = j*spacing[1]-ymax;
                double z = k*spacing[2]-zmax;
                double r = sqrt((x-x_in)*(x-x_in) + (y-y_in)*(y-y_in) + (z-z_in)*(z-z_in));
                    if(r<0.000001)
                    {
                        Eigen::Vector4d value(-1.0,0.0,0.0,0.0);
                        f->setData(value, {i,j,k});
                    }else {
                        double theta = atan2(y_in - y, x_in - x) - phi;
                        double thi = acos((z - z_in) / r);
                        vector<double> rational = FTPL::rationalMap(thi,theta,B);
                        double mod = sqrt(rational[0] * rational[0] + rational[1] * rational[1]);
                        double constant = 1.0 / (1.0 + mod * mod);

                        double res0 = cos(initial(r));
                        double res1 = sin(initial(r)) * constant * 2.0 * rational[0];
                        double res2 = sin(initial(r)) * constant * 2.0 * rational[1];
                        double res3 = sin(initial(r)) * constant * (1.0 - mod * mod);
                        Eigen::Vector4d value(res0, res1, res2, res3);
                        f->setData(value, {i, j, k});
                    }
            }}}
    }

    double SkyrmeModel::initial(double r)
    {
        double a;
        double initialradius = 1.6;
        if(r > initialradius)
        {
            a = 0;
        }
        else
        {
            a = M_PI*(1.0-r/initialradius);
        }
        return (a);
    }

}