/**
 * Field Theory Processing Library
 * Copyright Thomas Winyard 2016
 */


#ifndef FTPL_VORTICIES_HPP
#define FTPL_VORTICIES_HPP

#include <cmath>
#include "FieldTheories.hpp"
#include <Eigen/Dense>

using namespace std;

namespace FTPL {

    class VortexModel : public BaseFieldTheory {
    public:
        //place your fields here (likely to need to acces them publicly)
        Field<Eigen::VectorXd> * phi1;
        Field<Eigen::VectorXd> * phi2;
        Field<Eigen::VectorXd> * A;
        //maths (higher up functions run slightly faster these are the important ones!)
        inline virtual void __attribute__((always_inline)) calculateGradientFlow(int pos) final;
        inline virtual void __attribute__((always_inline)) RK4calc(int pos) final;
        inline virtual double __attribute__((always_inline)) calculateEnergy(int pos) final;
        inline virtual __attribute__((always_inline)) vector<double> calculateDynamicEnergy(int pos) final;
        //Other Useful functions
        void initialCondition(int B, double x_in, double y_in, double phi);
        void addSoliton(int B, double x_in, double y_in, double phi);
        double initial(double r);
        double inta(double r);
        //required functions
        VortexModel(const char * filepath, bool isDynamic = false);
        VortexModel(int width, int height, bool isDynamic = false);
        ~VortexModel(){};
        void setParameters(double lambda1_in, double lambda2_in, double m1_in, double m2_in, double k0_in);
        void setAnisotropy(double lambda1x_in, double lambda1y_in, double lambda2x_in, double lambda2y_in);
        double calculateCharge(int pos);
        double getCharge(){return charge;};
        inline double phi1sq(int pos);
        inline double phi2sq(int pos);
        void makePeriodic();
        void gradientCorrections(int loop);
        void createFixedPoints(int i1, int j1, int i2, int j2);
        void setCouplings(int e1in, int e2in);
        void plotPhi1();
        void plotPhi2();
    private:
        // parameters
        double lambda1, lambda2, m1 , m2, k0;
        double lambda1x, lambda1y, lambda2x, lambda2y;
        int fixedpoints = 0;
        int fixed1i, fixed1j, fixed2i, fixed2j;
        double e1 = 1.0;
        double e2 = 1.0;
    };

    void VortexModel::makePeriodic(){
        setAllBoundaryType({1,1,1,1});
        phi1->setboundarytype({4,4,1,1});
        phi2->setboundarytype({4,4,1,1});
        A->setboundarytype({4,4,1,1});
        phi1->boundarymorphtype = 2;
        phi2->boundarymorphtype = 2;
        A->boundarymorphtype = 1;
        bdw = {0,0,0,0};
        for(int i=0; i < getTotalSize() ; i++){
            //phi1->data[i] = phi1->rotate(i,-phi1->getBoundaryConstant()*2.0*M_PI*convert(i)[0]*convert(i)[1]/(getTotalSize()));
            //phi2->data[i] = phi2->rotate(i,-phi1->getBoundaryConstant()*2.0*M_PI*convert(i)[0]*convert(i)[1]/(getTotalSize()));
            //A->data[i] = A->rotate(i,-phi1->getBoundaryConstant()*2.0*M_PI*convert(i)[0]/(spacing[0]*getTotalSize()));
        }
    }

    inline void VortexModel::RK4calc(int pos){
        cout << "ERROR - No time dependent Vortex equations are yet set up!\n";
    }

    void VortexModel::createFixedPoints(int i1, int j1, int i2, int j2){
        fixedpoints = 1;
        fixed1i = i1;
        fixed1j = j1;
        fixed2i = i2;
        fixed2j = j2;
    }

    void VortexModel::setCouplings(int e1in, int e2in){
        e1 = e1in;
        e2 = e2in;
    }

    void VortexModel::gradientCorrections(int loop){
        if(fixedpoints == 1){
            Eigen::Vector2d value(0,0);
            phi1->setData(value,{fixed1i,fixed1j});
            phi1->setData(value,{fixed2i,fixed2j});
            phi2->setData(value,{fixed1i,fixed1j});
            phi2->setData(value,{fixed2i,fixed2j});
        }
    }

    void VortexModel::setParameters(double lambda1_in, double lambda2_in, double m1_in, double m2_in, double k0_in){
        lambda1 = lambda1_in;
        lambda2 = lambda2_in;
        m1 = m1_in;
        m2 = m2_in;
        k0 = k0_in;
    }

    void VortexModel::setAnisotropy(double lambda1x_in, double lambda1y_in, double lambda2x_in, double lambda2y_in){
        lambda1x = lambda1x_in;
        lambda1y = lambda1y_in;
        lambda2x = lambda2x_in;
        lambda2y = lambda2y_in;
    }

    VortexModel::VortexModel(int width, int height, bool isDynamic): BaseFieldTheory(2, {width,height}, isDynamic) {
        //vector<int> sizein = {width, height};
        //BaseFieldTheory(2,sizein);
        phi1 = createField(phi1, isDynamic, true);
        phi2 = createField(phi2, isDynamic, true);
        A = createField(A, isDynamic, true);
        Eigen::Vector3d minimum(-0.01,-0.01,-0.01);
        Eigen::Vector3d maximum(0.01,0.01,0.01);
        phi1->min = minimum;
        phi2->min = minimum;
        A->min = minimum;
        phi1->max = maximum;
        phi2->max = maximum;
        A->max = maximum;
        addParameter(&lambda1, "lambda1"); // need to add any parameters that you want to be saved/loaded when using the .save/.load function (always add them in the same order!)
        addParameter(&lambda2, "lambda2");
        addParameter(&m1, "m1");
        addParameter(&m2, "m2");
        addParameter(&k0, "k0");
        addParameter(&lambda1x, "lambda1x");
        addParameter(&lambda1y, "lambda1y");
        addParameter(&lambda2x, "lambda2x");
        addParameter(&lambda2y, "lambda2y");
        normalise_me = false;
        phi1->boundaryconstant = 0.0;
        phi2->boundaryconstant = 0.0;
        A->boundaryconstant = 0.0;
        phi1->boundarytype = {0,0,0,0};
        phi2->boundarytype={0,0,0,0};
        A->boundarytype={0,0,0,0};
        setAllBoundaryType({0,0,0,0});
        bdw = {2,2,2,2};

    };


    VortexModel::VortexModel(const char * filename, bool isDynamic): BaseFieldTheory(2, {2,2}, isDynamic){
        // mearly place holders so the fields can be initialised
        phi1 = createField(phi1, isDynamic, true);
        phi2 = createField(phi2, isDynamic, true);
        A = createField(A, isDynamic, true);
        addParameter(&lambda1, "lambda1"); // need to add any parameters that you want to be saved/loaded when using the .save/.load function (always add them in the same order!)
        addParameter(&lambda2, "lambda2");
        addParameter(&m1, "m1");
        addParameter(&m2, "m2");
        addParameter(&k0, "k0");
        addParameter(&lambda1x, "lambda1x");
        addParameter(&lambda1y, "lambda1y");
        addParameter(&lambda2x, "lambda2x");
        addParameter(&lambda2y, "lambda2y");
        load(filename);
        normalise_me = false;
        phi1->boundaryconstant = 0.0;
        phi2->boundaryconstant = 0.0;
        A->boundaryconstant = 0.0;
        phi1->boundarytype = {0,0,0,0};
        phi2->boundarytype={0,0,0,0};
        A->boundarytype={0,0,0,0};
        setAllBoundaryType({0,0,0,0});
        bdw = {2,2,2,2};
    };

//maths!
    double VortexModel::phi1sq(int pos)
    {
        return phi1->data[pos].squaredNorm();
    }
    double VortexModel::phi2sq(int pos)
    {
        return phi2->data[pos].squaredNorm();
    }
    double VortexModel::calculateEnergy(int pos){
        Eigen::Vector2d phi1x = single_derivative(phi1, 0, pos);
        Eigen::Vector2d phi1y = single_derivative(phi1, 1, pos);
        Eigen::Vector2d phi2x = single_derivative(phi2, 0, pos);
        Eigen::Vector2d phi2y = single_derivative(phi2, 1, pos);
        Eigen::Vector2d Ax = single_derivative(A, 0, pos);
        Eigen::Vector2d Ay = single_derivative(A, 1, pos);

        double sum = pow(lambda1x,2)*(phi1x.squaredNorm()) + pow(lambda1y,2)*(phi1y.squaredNorm())
                     + pow(e1,2)*(pow(lambda1x,2)*pow(A->data[pos][0],2) + pow(lambda1y,2)*pow(A->data[pos][1],2))*phi1sq(pos)
                     + 2.0*e1*pow(lambda1x,2)*A->data[pos][0]*(phi1->data[pos][1]*phi1x[0]-phi1->data[pos][0]*phi1x[1])
                     + 2.0*e1*pow(lambda1y,2)*A->data[pos][1]*(phi1->data[pos][1]*phi1y[0]-phi1->data[pos][0]*phi1y[1]);

        sum +=       pow(lambda2x,2)*(phi2x.squaredNorm()) + pow(lambda2y,2)*(phi2y.squaredNorm())
                     + pow(e2,2)*(pow(lambda2x,2)*pow(A->data[pos][0],2) + pow(lambda2y,2)*pow(A->data[pos][1],2))*phi2sq(pos)
                     + 2.0*e2*pow(lambda2x,2)*A->data[pos][0]*(phi2->data[pos][1]*phi2x[0]-phi2->data[pos][0]*phi2x[1])
                     + 2.0*e2*pow(lambda2y,2)*A->data[pos][1]*(phi2->data[pos][1]*phi2y[0]-phi2->data[pos][0]*phi2y[1]);
        sum += pow( Ax[1]-Ay[0] ,2);
        sum += 0.25*lambda1*pow(m1*m1 - phi1sq(pos),2);
        sum += 0.25*lambda2*pow(m2*m2 - phi2sq(pos),2);
        sum += k0*(1.0 - 0.5*phi1->data[pos].dot(phi2->data[pos]) );
        return 0.5*sum;
    };

    vector<double> VortexModel::calculateDynamicEnergy(int pos){
        cout << "ERROR! - DYnamical energy is not defined of this static model!\n";
        vector<double> result(2);
        return result;
    };

    double VortexModel::calculateCharge(int pos){
        Eigen::Vector2d Ax = single_derivative(A, 0, pos);
        Eigen::Vector2d Ay = single_derivative(A, 1, pos);
        return (1.0/(2.0*M_PI))*( Ax[1]-Ay[0] );
    };


    inline void VortexModel::calculateGradientFlow(int pos){
        if(inBoundary(pos)) {
            Eigen::Vector2d phi1x = single_derivative(phi1, 0, pos);
            Eigen::Vector2d phi1y = single_derivative(phi1, 1, pos);
            Eigen::Vector2d phi1xx = double_derivative(phi1, 0, 0, pos);
            Eigen::Vector2d phi1yy = double_derivative(phi1, 1, 1, pos);
            Eigen::Vector2d phi2x = single_derivative(phi2, 0, pos);
            Eigen::Vector2d phi2y = single_derivative(phi2, 1, pos);
            Eigen::Vector2d phi2xx = double_derivative(phi2, 0, 0, pos);
            Eigen::Vector2d phi2yy = double_derivative(phi2, 1, 1, pos);
            Eigen::Vector2d Ax = single_derivative(A, 0, pos);
            Eigen::Vector2d Ay = single_derivative(A, 1, pos);
            Eigen::Vector2d Axx = double_derivative(A, 0, 0, pos);
            Eigen::Vector2d Ayy = double_derivative(A, 1, 1, pos);
            Eigen::Vector2d Axy = double_derivative(A, 0, 1, pos);

            Eigen::Vector2d phi1gradient = pow(lambda1x,2)*phi1xx + pow(lambda1y,2)*phi1yy - pow(e1,2)*(pow(lambda1x,2)*pow(A->data[pos][0],2) + pow(lambda1y,2)*pow(A->data[pos][1],2))*(phi1->data[pos]);

            phi1gradient[0] = phi1gradient[0] + e1*pow(lambda1x,2)*(A->data[pos][0]*(phi1x[1]) + Ax[0]*phi1->data[pos][1] + A->data[pos][0]*phi1x[1]);
            phi1gradient[1] = phi1gradient[1] - e1*pow(lambda1x,2)*(A->data[pos][0]*(phi1x[0]) + Ax[0]*phi1->data[pos][0] + A->data[pos][0]*phi1x[0]);

            phi1gradient[0] = phi1gradient[0] + e1*pow(lambda1y,2)*(A->data[pos][1]*(phi1y[1]) + Ay[1]*phi1->data[pos][1] + A->data[pos][1]*phi1y[1]);
            phi1gradient[1] = phi1gradient[1] - e1*pow(lambda1y,2)*(A->data[pos][1]*(phi1y[0]) + Ay[1]*phi1->data[pos][0] + A->data[pos][1]*phi1y[0]);

            phi1gradient = phi1gradient + 0.5*lambda1*phi1->data[pos]*(m1*m1 - (phi1sq(pos)));

            phi1gradient = phi1gradient + k0*0.25*phi2->data[pos];

            Eigen::Vector2d phi2gradient = pow(lambda2x,2)*phi2xx + pow(lambda2y,2)*phi2yy - pow(e2,2)*(pow(lambda2x,2)*pow(A->data[pos][0],2) + pow(lambda2y,2)*pow(A->data[pos][1],2))*(phi2->data[pos]  );

            phi2gradient[0] = phi2gradient[0] + e2*pow(lambda2x,2)*(A->data[pos][0]*(phi2x[1]) + Ax[0]*phi2->data[pos][1] + A->data[pos][0]*phi2x[1]);
            phi2gradient[1] = phi2gradient[1] - e2*pow(lambda2x,2)*(A->data[pos][0]*(phi2x[0]) + Ax[0]*phi2->data[pos][0] + A->data[pos][0]*phi2x[0]);
            phi2gradient[0] = phi2gradient[0] + e2*pow(lambda2y,2)*(A->data[pos][1]*(phi2y[1]) + Ay[1]*phi2->data[pos][1] + A->data[pos][1]*phi2y[1]);
            phi2gradient[1] = phi2gradient[1] - e2*pow(lambda2y,2)*(A->data[pos][1]*(phi2y[0]) + Ay[1]*phi2->data[pos][0] + A->data[pos][1]*phi2y[0]);

            phi2gradient = phi2gradient + 0.5*lambda2*phi2->data[pos]*(m2*m2 - (phi2sq(pos)));
            phi2gradient = phi2gradient + k0*0.25*phi1->data[pos];

            Eigen::Vector2d Agradient;

            Agradient[0] = -( Axy[1] - Ayy[0]) - pow(e1,2)*pow(lambda1x,2)*A->data[pos][0]*phi1sq(pos) - pow(e2,2)*pow(lambda2x,2)*A->data[pos][0]*phi2sq(pos);
            Agradient[1] = ( Axx[1] - Axy[0] ) - pow(e1,2)*pow(lambda1y,2)*A->data[pos][1]*phi1sq(pos) - pow(e2,2)*pow(lambda2y,2)*A->data[pos][1]*phi2sq(pos);

            Agradient[0] = Agradient[0] - e1*pow(lambda1x,2)*(phi1->data[pos][1]*phi1x[0]-phi1->data[pos][0]*phi1x[1]);
            Agradient[0] = Agradient[0] - e2*pow(lambda2x,2)*(phi2->data[pos][1]*phi2x[0]-phi2->data[pos][0]*phi2x[1]);
            Agradient[1] = Agradient[1] - e1*pow(lambda1y,2)*(phi1->data[pos][1]*phi1y[0]-phi1->data[pos][0]*phi1y[1]);
            Agradient[1] = Agradient[1] - e2*pow(lambda2y,2)*(phi2->data[pos][1]*phi2y[0]-phi2->data[pos][0]*phi2y[1]);

            phi1->buffer[pos] = phi1gradient;
            phi2->buffer[pos] = phi2gradient;
            A->buffer[pos] = Agradient;
        } else{
            Eigen::Vector2d zero(0,0);
            phi1->buffer[pos] = zero;
            phi2->buffer[pos] = zero;
            A->buffer[pos] = zero;
        }
    }

    void VortexModel::addSoliton(int B, double x_in, double y_in, double phi){
        if(dynamic) {
            Eigen::Vector2d value(0,0);
            phi1->fill_dt(value);
            phi2->fill_dt(value);
            A->fill_dt(value);
        }
        vector<int> pos(dim);
        double xmax = size[0]*spacing[0]/2.0;
        double ymax = size[1]*spacing[1]/2.0;
        for(int i = 0; i < size[0]; i++){
            for(int j = 0; j < size[1]; j++){
                pos[0] = i;
                pos[1] = j;
                {
                    double x = i * spacing[0] - xmax;
                    double y = j * spacing[1] - ymax;
                    double r = sqrt((x - x_in) * (x - x_in) + (y - y_in) * (y - y_in));
                    double theta = atan2(y - y_in, x - x_in) - phi;
                    int point = pos[0]+pos[1]*size[0];

                    Eigen::Vector2d value1(cos(B*theta)*phi1->data[point][0]-sin(B*theta)*phi1->data[point][1],sin(B*theta)*phi1->data[point][0]+cos(B*theta)*phi1->data[point][1]);
                    Eigen::Vector2d value2(cos((e2/e1)*B*theta)*phi2->data[point][0]-sin((e2/e1)*B*theta)*phi2->data[point][1],sin((e2/e1)*B*theta)*phi2->data[point][0]+cos((e2/e1)*B*theta)*phi2->data[point][1]);

                    phi1->rotate(point,B*theta);
                    phi2->rotate(point,(e2/e1)*B*theta);

                    phi1->data[point] = value1*initial(r);
                    phi2->data[point] = value2*initial(r);

                    Eigen::Vector2d valueA(-B*inta(r)*sin(theta), B*inta(r)*cos(theta));
                    A->data[point] = (A->data[point] + valueA );

                }
            }}
        phi1->boundaryconstant += B;
        phi2->boundaryconstant += B;
        A->boundaryconstant += B;
    }

    void VortexModel::initialCondition(int B, double x_in, double y_in, double phi){
        if(dynamic) {
            Eigen::Vector2d value(0,0);
            phi1->fill_dt(value);
            phi2->fill_dt(value);
            A->fill_dt(value);
        }
        double xmax = size[0]*spacing[0]/2.0;
        double ymax = size[1]*spacing[1]/2.0;
        for(int i = 0; i < size[0]; i++){
            for(int j = 0; j < size[1]; j++){
                //if(!inBoundary({i,j})){
                    double x = i * spacing[0] - xmax;
                    double y = j * spacing[1] - ymax;
                    double r = sqrt((x - x_in) * (x - x_in) + (y - y_in) * (y - y_in));
                    double theta = atan2(y - y_in, x - x_in) - phi;
                    Eigen::Vector2d valuep1(m1*initial(r)*cos(B*theta) , m1*initial(r)*sin(B*theta));
                    Eigen::Vector2d valuep2(m2*initial(r)*cos((e2/e1)*B*theta), m2*initial(r)*sin((e2/e1)*B*theta));
                    Eigen::Vector2d valueA(-B*inta(r)*sin(theta), B*inta(r)*cos(theta));
                    phi1->setData(valuep1, {i, j});
                    phi2->setData(valuep2, {i, j});
                    A->setData(valueA, {i, j});//}
            }}
        phi1->boundaryconstant = B;
        phi2->boundaryconstant = B;
        A->boundaryconstant = B;
    }

    double VortexModel::initial(double r)
    {
        double a;
        double initialradius = 3.0;
        if(r > initialradius)
        {
            a = 1;
        }
        else
        {
            //a=m1 - m1*(1.0 - r/initialradius);
            a = 1-1*exp(-5.0*(r*r));
        }
        return (a);
    }


    double VortexModel::inta(double r)
    {
        if(r > 3.0)
        {
            return 1.0/r;
        }
        else
        {
            return exp(-5.0*(r*r));
        }
    }

}

#endif //FTPL_VORTICIES_HPP
