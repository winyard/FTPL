//
// Created by tom on 2018-01-31.
//

#ifndef FTPL_EXTENDEDHOPFIONS_HPP
#define FTPL_EXTENDEDHOPFIONS_HPP

#include <cmath>
#include "FieldTheories.hpp"
#include <Eigen/Dense>

using namespace std;

namespace FTPL {

    class ExtendedHopfionModel : public BaseFieldTheory {
    public:
        //place your fields here (likely to need to access them publicly)
        Field<Eigen::VectorXd> * phi1;
        Field<Eigen::VectorXd> * phi2;
        Field<Eigen::VectorXd> * A;

        //maths (higher up functions run slightly faster these are the important ones!)
        inline virtual void __attribute__((always_inline)) calculateGradientFlow(int pos) final;
        inline virtual void __attribute__((always_inline)) RK4calc(int pos) final;
        inline virtual double __attribute__((always_inline)) calculateEnergy(int pos) final;
        inline virtual __attribute__((always_inline)) vector<double> calculateDynamicEnergy(int pos) final;
        //Other Useful functions
        void calculateParameters();
        void initialCondition(int B, double x_in, double y_in, double phi);
        void addSoliton(int B, double x_in, double y_in, double phi);
        void setAnisotropy(vector<vector<Eigen::MatrixXd>> Qin);
        double initial(double r);
        double inta(double r);
        //required functions
        ExtendedHopfionModel(const char * filepath, bool isDynamic = false);
        ExtendedHopfionModel(int width, int height, bool isDynamic = false);
        ~ExtendedHopfionModel(){};
        void setParameters(double lambda1_in, double lambda2_in, double m1_in, double m2_in, double k0_in);
        double calculateCharge(int pos);
        double getCharge(){return charge;};
        inline double phi1sq(int pos);
        inline double phi2sq(int pos);
        void makePeriodic();
        void virtual correctTheory(int loop);
        void setAnisotropy(vector<vector<Eigen::Matrix2d>> Qin);
        void plotPhi1sq();
        void plotPhi2sq();
        void plotAx();
        void plotAy();
        void outputSpace();
        void storeFields();
    private:
        // parameters
        double lambda1, lambda2, m1 , m2, k0, e;
        int NoFields = 2;

    };

    void ExtendedHopfionModel::storeFields(){

        ofstream output("/home/tom/FTPLdata.gp");

        for(int i=0; i<size[0]; i++) {
            for (int j = 0; j < size[1]; j++) {
                for (int k = 0; k < size[2]; k++) {
                double x = i * spacing[0];
                double y = j * spacing[1];
                vector<int> p(2);
                p[0] = i;
                p[1] = j;
                int point = p[0];
                int multiplier = 1;
                for (int a = 1; a < dim; a++) {
                    multiplier *= size[a - 1];
                    point += p[a] * multiplier;
                }
                output << x << " " << y << " " << x << " " << y << " " << energydensity[point] << " "
                       << chargedensity[point] << "\n";
            }
            output << "\n";
        }
        output << "\n";
        }
        cout << "Fields output to file!\n";
    }

    inline void ExtendedHopfionModel::RK4calc(int pos){
        cout << "ERROR - No time dependent Vortex equations are yet set up!\n";
    }

    void ExtendedHopfionModel::setParameters(double lambda1_in, double lambda2_in, double m1_in, double m2_in, double k0_in){
        lambda1 = lambda1_in;
        lambda2 = lambda2_in;
        m1 = m1_in;
        m2 = m2_in;
        k0 = k0_in;
        e = 1;
    }

    ExtendedHopfionModel::ExtendedHopfionModel(int width, int height, int length, bool isDynamic): BaseFieldTheory(3, {width,height,length}, isDynamic) {
        //vector<int> sizein = {width, height};
        //BaseFieldTheory(2,sizein);
        phi1 = createField(phi1, isDynamic, true);
        phi2 = createField(phi2, isDynamic, true);
        A = createField(A, isDynamic, true);
        Eigen::Vector2d minimum2d(-0.01,-0.01);
        Eigen::Vector2d maximum2d(0.01,0.01);
        Eigen::Vector3d minimum3d(-0.01,-0.01,-0.01);
        Eigen::Vector3d maximum3d(0.01,0.01,0.01);
        phi1->min = minimum2d;
        phi2->min = minimum2d;
        A->min = minimum3d;
        phi1->max = maximum2d;
        phi2->max = maximum2d;
        A->max = maximum3d;
        addParameter(&lambda1, "lambda1"); // need to add any parameters that you want to be saved/loaded when using the .save/.load function (always add them in the same order!)
        addParameter(&lambda2, "lambda2");
        addParameter(&m1, "m1");
        addParameter(&m2, "m2");
        addParameter(&k0, "k0");
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


    ExtendedHopfionModel::ExtendedHopfionModel(const char * filename, bool isDynamic): BaseFieldTheory(3, {2,2,2}, isDynamic){
        // mearly place holders so the fields can be initialised
        phi1 = createField(phi1, isDynamic, true);
        phi2 = createField(phi2, isDynamic, true);
        A = createField(A, isDynamic, true);
        addParameter(&lambda1, "lambda1"); // need to add any parameters that you want to be saved/loaded when using the .save/.load function (always add them in the same order!)
        addParameter(&lambda2, "lambda2");
        addParameter(&m1, "m1");
        addParameter(&m2, "m2");
        addParameter(&k0, "k0");
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
    double ExtendedHopfionModel::phi1sq(int pos)
    {
        return phi1->data[pos].squaredNorm();
    }
    double ExtendedHopfionModel::phi2sq(int pos)
    {
        return phi2->data[pos].squaredNorm();
    }
    double ExtendedHopfionModel::calculateEnergy(int pos){
        Eigen::Vector2d phi1x = single_derivative(phi1, 0, pos);
        Eigen::Vector2d phi1y = single_derivative(phi1, 1, pos);
        Eigen::Vector2d phi1z = single_derivative(phi1, 2, pos);
        Eigen::Vector2d phi2x = single_derivative(phi2, 0, pos);
        Eigen::Vector2d phi2y = single_derivative(phi2, 1, pos);
        Eigen::Vector2d phi2z = single_derivative(phi2, 2, pos);
        Eigen::Vector2d Ax = single_derivative(A, 0, pos);
        Eigen::Vector2d Ay = single_derivative(A, 1, pos);
        Eigen::Vector2d Az = single_derivative(A, 2, pos);

        vector<vector<Eigen::Vector2d>> d(2, vector<Eigen::Vector2d>(2));
        vector<Eigen::Vector2d> p(2);

        d[0][0] = phi1x;
        d[0][1] = phi1y;
        d[0][2] = phi1z;
        d[1][0] = phi2x;
        d[1][1] = phi2y;
        d[1][2] = phi2z;

        p[0] = phi1->data[pos];
        p[1] = phi2->data[pos];

        double sum = 0.0;

        for(int alpha = 0; alpha < NoFields; alpha++) {
            int beta = alpha;
                for (int j = 0; j < dim; j++) {
                    int k = j;
                        double D = d[alpha][j].dot(d[beta][k]) + e*e*A->data[pos](j)*A->data[pos](k)*(p[alpha].dot(p[beta]))
                                   -e*A->data[pos](j)*(p[alpha][0]*d[beta][k][1] - p[alpha][1]*d[beta][k][0])
                                   -e*A->data[pos](k)*(p[beta][0]*d[alpha][j][1] - p[beta][1]*d[alpha][j][0]);
                        if(alpha != beta){D *= 2.0;}
                        sum += D;
                }
        }

        sum += pow( Ax[1]-Ay[0] ,2);
        sum += pow( Ay[2]-Az[1] ,2);
        sum += pow( Az[0]-Ax[2] ,2);

        sum += lambda1*pow(phi1sq(pos) + phi2sq(pos) - 1.0,2);
        return 0.5*sum;
    };

    vector<double> ExtendedHopfionModel::calculateDynamicEnergy(int pos){
        cout << "ERROR! - DYnamical energy is not defined of this static model!\n";
        vector<double> result(2);
        return result;
    };

    double ExtendedHopfionModel::calculateCharge(int pos){
        Eigen::Vector2d Ax = single_derivative(A, 0, pos);
        Eigen::Vector2d Ay = single_derivative(A, 1, pos);
        return (1.0/(2.0*M_PI))*( Ax[1]-Ay[0] );
    };


    inline void ExtendedHopfionModel::calculateGradientFlow(int pos){
        if(inBoundary(pos)) {
            Eigen::Vector2d phi1x = single_derivative(phi1, 0, pos);
            Eigen::Vector2d phi1y = single_derivative(phi1, 1, pos);
            Eigen::Vector2d phi1z = single_derivative(phi1, 2, pos);
            Eigen::Vector2d phi1xx = double_derivative(phi1, 0, 0, pos);
            Eigen::Vector2d phi1yy = double_derivative(phi1, 1, 1, pos);
            Eigen::Vector2d phi1zz = double_derivative(phi1, 2, 2, pos);
            Eigen::Vector2d phi1xy = double_derivative(phi1, 0, 1, pos);
            Eigen::Vector2d phi1xz = double_derivative(phi1, 0, 2, pos);
            Eigen::Vector2d phi1yz = double_derivative(phi1, 1, 2, pos);
            Eigen::Vector2d phi2x = single_derivative(phi2, 0, pos);
            Eigen::Vector2d phi2y = single_derivative(phi2, 1, pos);
            Eigen::Vector2d phi2z = single_derivative(phi2, 2, pos);
            Eigen::Vector2d phi2xx = double_derivative(phi2, 0, 0, pos);
            Eigen::Vector2d phi2yy = double_derivative(phi2, 1, 1, pos);
            Eigen::Vector2d phi2zz = double_derivative(phi2, 2, 2, pos);
            Eigen::Vector2d phi2xy = double_derivative(phi2, 0, 1, pos);
            Eigen::Vector2d phi2xz = double_derivative(phi2, 0, 2, pos);
            Eigen::Vector2d phi2yz = double_derivative(phi2, 1, 2, pos);
            Eigen::Vector2d Ax = single_derivative(A, 0, pos);
            Eigen::Vector2d Ay = single_derivative(A, 1, pos);
            Eigen::Vector2d Az = single_derivative(A, 2, pos);
            Eigen::Vector2d Axx = double_derivative(A, 0, 0, pos);
            Eigen::Vector2d Ayy = double_derivative(A, 1, 1, pos);
            Eigen::Vector2d Azz = double_derivative(A, 2, 2, pos);
            Eigen::Vector2d Axy = double_derivative(A, 0, 1, pos);
            Eigen::Vector2d Axz = double_derivative(A, 0, 2, pos);
            Eigen::Vector2d Ayz = double_derivative(A, 1, 2, pos);

            Eigen::Vector2d phi1gradient(0,0);
            Eigen::Vector2d phi2gradient(0,0);

            vector<Eigen::Vector2d> D(NoFields);
            vector<Eigen::Vector2d> Dsum(NoFields);
            Eigen::Vector2d DAsum;
            Eigen::Vector2d DA;
            vector<vector<vector<Eigen::Vector2d>>> dd(NoFields);
            vector<vector<Eigen::Vector2d>> d(NoFields, vector<Eigen::Vector2d>(3));
            vector<Eigen::Vector3d> dA(3);
            vector<Eigen::Vector2d> p(NoFields);

            for(int i = 0; i < NoFields; i++){
                dd[i].resize(3);
                for(int j = 0; j < 3; j++){
                    dd[i][j].resize(3);
                }
            }

            dd[0][0][0] = phi1xx;
            dd[0][1][1] = phi1yy;
            dd[0][2][2] = phi1zz;
            dd[0][0][1] = phi1xy;
            dd[0][1][0] = phi1xy;
            dd[0][0][2] = phi1xz;
            dd[0][2][0] = phi1xz;
            dd[0][1][2] = phi1yz;
            dd[0][2][1] = phi1yz;

            dd[1][0][0] = phi2xx;
            dd[1][1][1] = phi2yy;
            dd[1][2][2] = phi2zz;
            dd[1][0][1] = phi2xy;
            dd[1][1][0] = phi2xy;
            dd[1][0][2] = phi2xz;
            dd[1][2][0] = phi2xz;
            dd[1][1][2] = phi2yz;
            dd[1][2][1] = phi2yz;

            d[0][0] = phi1x;
            d[0][1] = phi1y;
            d[0][2] = phi1z;
            d[1][0] = phi2x;
            d[1][1] = phi2y;
            d[1][2] = phi2z;

            dA[0] = Ax;
            dA[1] = Ay;
            dA[2] = Az;

            p[0] = phi1->data[pos];
            p[1] = phi2->data[pos];

            for(int alpha = 0; alpha < NoFields; alpha++){
                DAsum.setZero();
                DA.setZero();
                Dsum[alpha].setZero();
            }


            for(int alpha = 0; alpha < NoFields; alpha++) {
                for (int beta = 0; beta <= alpha; beta++) {
                    for (int j = 0; j < dim; j++) {
                        for (int k = 0; k <dim; k++) {
                            DA.setZero();

                            D[alpha].setZero();
                            D[beta].setZero();

                            /*double D = d[alpha][j].dot(d[beta][k]) + e*e*A->data[pos](j)*A->data[pos](k)*(p[alpha].dot(p[beta]))
                                       -e*A->data[pos](j)*(p[alpha][0]*d[beta][k][1] - p[alpha][1]*d[beta][k][0])
                                       -e*A->data[pos](k)*(p[beta][0]*d[alpha][j][1] - p[beta][1]*d[alpha][j][0]);
                            if(alpha != beta){D *= 2.0;}*/

                            D[alpha] += dd[beta][j][k] - e * e * A->data[pos](j) * A->data[pos](k) * p[beta];
                            D[alpha](0) += e * A->data[pos](j) * (d[beta][k](1));
                            D[alpha](1) += -e * A->data[pos](j) * (d[beta][k][0]);

                            D[alpha](0) += e * dA[j](k) * (p[beta][1]) + e * A->data[pos](k) * (d[beta][j][1]);
                            D[alpha](1) += -e * dA[j](k) * (p[beta][0]) - e * A->data[pos](k) * (d[beta][j][0]);

                            DA[j] += (-e * e * A->data[pos](k) * p[alpha].dot(p[beta])
                                                              + e * (p[alpha][0] * d[beta][k][1] - p[alpha][1] * d[beta][k][0]));

                            //if (alpha != beta || j != k) {
                            /*D[beta] += dd[alpha][k][j] - e * e * A->data[pos](j) * A->data[pos](k) * p[alpha];
                            D[beta](0) += e * A->data[pos](k) * (d[alpha][j](1));
                            D[beta](1) += -e * A->data[pos](k) * (d[alpha][j][0]);

                            D[beta](0) += e * dA[k][j] * (p[alpha][1]) + e * A->data[pos](j) * (d[alpha][k][1]);
                            D[beta](1) += -e * dA[k][j] * (p[alpha][0]) - e * A->data[pos](j) * (d[alpha][k][0]);

                            DA[k] += (-e * e * A->data[pos](j) * p[alpha].dot(p[beta]) + e *
                                                                                                                 (p[beta][0] * d[alpha][j][1] - p[beta][1] * d[alpha][j][0]));*/
                            D[alpha] *= 2;
                            DA[j] *= 2;

                            if(alpha != beta) {
                                DA *= 2;
                                D[beta] *= 2;
                                D[alpha] *= 2;
                                Dsum[beta] += 0.5 * D[beta];
                            }

                            DAsum += 0.5 * DA;

                            Dsum[alpha] += 0.5 * D[alpha];
                        }
                    }
                }
            }

            phi1gradient = Dsum[0] + 0.5*lambda1*phi1->data[pos]*(m1*m1 - (phi1sq(pos)));
            //phi1gradient = phi1gradient + k0*0.25*phi2->data[pos];

            phi2gradient = Dsum[1] + 0.5*lambda2*phi2->data[pos]*(m2*m2 - (phi2sq(pos)));
            //phi2gradient = phi2gradient + k0*0.25*phi1->data[pos];

            Eigen::Vector2d Agradient;
            Agradient = DAsum;

            Agradient[0] += -( Axy[1] - Ayy[0]);
            Agradient[1] += ( Axx[1] - Axy[0] );

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

    void ExtendedHopfionModel::addSoliton(int B, double x_in, double y_in, double phi){
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
                    double theta = atan2(y_in - y, x_in - x) - phi;
                    int point = pos[0]+pos[1]*size[0];

                    phi1->rotate(point,B*theta);
                    phi2->rotate(point,B*theta);

                    phi1->data[point] = 0.5*(initial(r) + phi1->data[point].norm())*phi1->data[point]/phi1->data[point].norm();
                    phi2->data[point] = 0.5*(initial(r) + phi2->data[point].norm())*phi2->data[point]/phi2->data[point].norm();

                    Eigen::Vector2d valueA(-B*inta(r)*sin(theta), B*inta(r)*cos(theta));
                    A->data[point] = 0.5*(A->data[point] + valueA );

                }
            }}
        phi1->boundaryconstant += B;
        phi2->boundaryconstant += B;
        A->boundaryconstant += B;
    }

    void ExtendedHopfionModel::plotPhi1sq(){
        vector<double> plotdata;
        plotdata.resize(getTotalSize());
        for(int i = 0; i < getTotalSize(); i++){
            plotdata[i] = phi1->data[i].dot(phi1->data[i]);
        }
        plot2D(size, plotdata);
    }

    void ExtendedHopfionModel::plotPhi2sq(){
        vector<double> plotdata;
        plotdata.resize(getTotalSize());
        for(int i = 0; i < getTotalSize(); i++){
            plotdata[i] = phi2->data[i].dot(phi2->data[i]);
        }
        plot2D(size, plotdata);
    }

    void ExtendedHopfionModel::plotAx(){
        vector<double> plotdata;
        plotdata.resize(getTotalSize());
        for(int i = 0; i < getTotalSize(); i++){
            plotdata[i] = A->data[i](0);
        }
        plot2D(size, plotdata);
    }

    void ExtendedHopfionModel::plotAy(){
        vector<double> plotdata;
        plotdata.resize(getTotalSize());
        for(int i = 0; i < getTotalSize(); i++){
            plotdata[i] = A->data[i](1);
        }
        plot2D(size, plotdata);
    }

    void ExtendedHopfionModel::initialCondition(int B, double x_in, double y_in, double phi){
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
                //if(x>0 && y < x && y > -x){theta = 4.0*atan2(y - y_in, x - x_in) - 4.0*atan2(-x + x_in, x - x_in);}
                //else{theta = 0.0;}
                /*if(y<0 && x < -y && x > y){theta = atan22.0*atan2(-x + x_in, x - x_in) - phi + (2.0*atan2(y - y_in, x - x_in) - 2.0*atan2(x - x_in, x - x_in));}
                if(y<0 && x <= y){theta = 2.0*atan2(-x + x_in, x - x_in) - phi;}
                if(y<0 && x >= -y){theta = 2.0*atan2(x - x_in, x - x_in) - phi;}
                if(y>=0 && x >= y){theta = 2.0*atan2(x - x_in, x - x_in) - phi;}
                if(y>=0 && x <= -y){theta = 2.0*atan2(-x + x_in, x - x_in) - phi;}*/
                Eigen::Vector2d valuep1(initial(r)*cos(B*theta) , initial(r)*sin(B*theta));
                Eigen::Vector2d valuep2(initial(r)*cos(B*theta), initial(r)*sin(B*theta));
                Eigen::Vector2d valueA(-B*inta(r)*sin(theta), B*inta(r)*cos(theta));
                phi1->setData(valuep1, {i, j});
                phi2->setData(valuep2, {i, j});
                A->setData(valueA, {i, j});//}
            }}
        phi1->boundaryconstant = B;
        phi2->boundaryconstant = B;
        A->boundaryconstant = B;
    }

    double ExtendedHopfionModel::initial(double r)
    {
        double a;
        double initialradius = 1.8;
        if(r > initialradius)
        {
            a = m1;
        }
        else
        {
            //a=m1 - m1*(1.0 - r/initialradius);
            a = m1-m1*exp(-20.0*(r*r));
        }
        return (a);
    }


    double ExtendedHopfionModel::inta(double r)
    {
        if(r > 1.8)
        {
            return 1.0/r;
        }
        else
        {
            return exp(-20.0*(r*r));
        }
    }

}


#endif //FTPL_EXTENDEDHOPFIONS_HPP
