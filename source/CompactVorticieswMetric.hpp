//
// Created by tom on 2017-11-08.
//

#ifndef FTPL_COMPACTVORTICIESWMETRIC_HPP
#define FTPL_COMPACTVORTICIESWMETRIC_HPP

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
        void setAnisotropy(double lambda1x_in, double lambda1y_in, double lambda1xy_in, double lambda2x_in, double lambda2y_in, double lambda2xy_in);
        double calculateCharge(int pos);
        double getCharge(){return charge;};
        inline double phi1sq(int pos);
        inline double phi2sq(int pos);
        void makePeriodic();
        Eigen::Matrix2d g;// constant metric
        Eigen::Matrix2d lambda1a;
        Eigen::Matrix2d lambda2a;
        void virtual minimiseMap(int loop);
    private:
        // parameters
        double lambda1, lambda2, m1 , m2, k0;
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

    void VortexModel::setParameters(double lambda1_in, double lambda2_in, double m1_in, double m2_in, double k0_in){
        lambda1 = lambda1_in;
        lambda2 = lambda2_in;
        m1 = m1_in;
        m2 = m2_in;
        k0 = k0_in;
    }

    void VortexModel::CalculateParameters(){
        for(int i = 0; i < NoFields; i++)
        {
            for(int j = 0; j < NoFields ; j++) {
                S[i][j]=M*Q[i][j]*M.transpose();
            }
        }
        /*detL = L[0][0];
        double sumL = 1.0;
        for(int i = 0; i <dim; i++) {
            detL = detL * L[i][i];
        }
        if(dim > 1){
            for(int i = 0; i <dim; i++) { for(int j = 0; j <dim; j++) {
                    if(i!=j) {
                        sumL = sumL * L[i][j];
                    }
            }}
            detL = detL - sumL;
        }*/
    }

    void VortexModel::minimiseMap(int loop){
        if(loop%10 == 0) {
            double Asum = 0.0;
            double Bsum = 0.0;
            double Csum = 0.0;
            for (int i = 0; i < getTotalSize(); i++) {
                if (inBoundary(i)) {
                    int pos = i;
                    Eigen::Vector2d phi1x = single_derivative(phi1, 0, i);
                    Eigen::Vector2d phi1y = single_derivative(phi1, 1, i);
                    Eigen::Vector2d phi2x = single_derivative(phi2, 0, i);
                    Eigen::Vector2d phi2y = single_derivative(phi2, 1, i);

                    d[0][0] = phi1x;
                    d[0][1] = phi1y;
                    d[1][0] = phi2x;
                    d[1][1] = phi2y;

                    p[0] = phi1->data[i];
                    p[1] = phi2->data[i];

                    for(int alpha = 0; alpha < NoFields; alpha++) {
                        for (int beta = 0; beta < NoFields; beta++) {
                            for (int j = 0; j < NoFields; j++) {
                                for (int k = 0; k < NoFields; k++) {
                                    D[alpha][beta][i][k] = d[alpha][j].dot(d[beta][k]) + e*e*A[j]*A[k]*(p[alpha].dot(p[alpha]))
                                            +e*A[j]*(p[alpha][0]*d[beta][k][1] - p[alpha][1]*d[beta][k][0])
                                             +e*A[k]*(p[beta][0]*d[alpha][j][1] - p[beta][1]*d[alpha][j][0]);


                                }
                            }
                        }
                    }

                    phi

                }
            }

            if (Csum > 0.000001) {
               9 g(0, 0) = sqrt((1 + sqrt(1 + 4 * Bsum * Asum / pow(Csum, 2))) / (2.0 * Asum / Bsum));
                g(1, 1) = sqrt((1 + sqrt(1 + 4 * Bsum * Asum / pow(Csum, 2))) / (2.0 * Bsum / Asum));
                g(0, 1) = sqrt(g(0, 0) * g(1, 1) - 1.0);
                g(1, 0) = g(0, 1);

            } else {
                g(0, 0) = Bsum / sqrt(Asum * Bsum);
                g(1, 1) = Asum / sqrt(Asum * Bsum);
                g(0, 1) = sqrt(g(0, 0) * g(1, 1) - 1.0 + 0.000000000001);
                g(1, 0) = g(0, 1);

            }
        }

    }

    void VortexModel::setAnisotropy(double lambda1x_in, double lambda1y_in, double lambda1xy_in, double lambda2x_in, double lambda2y_in, double lambda2xy_in){
        lambda1a(0,0) = lambda1x_in;
        lambda1a(1,1) = lambda1y_in;
        lambda1a(0,1) = lambda1xy_in;
        lambda1a(1,0) = lambda1a(0,1);
        lambda2a(0,0) = lambda2x_in;
        lambda2a(1,1) = lambda2y_in;
        lambda2a(0,1) = lambda2xy_in;
        lambda2a(1,0) = lambda2a(0,1);
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
        addParameter(&lambda1a(0,0), "lambda1x");
        addParameter(&lambda1a(1,1), "lambda1y");
        addParameter(&lambda1a(0,1), "lambda1xy");
        addParameter(&lambda2a(0,0), "lambda2x");
        addParameter(&lambda2a(1,1), "lambda2y");
        addParameter(&lambda2a(0,1), "lambda2xy");
        normalise_me = false;
        phi1->boundaryconstant = 0.0;
        phi2->boundaryconstant = 0.0;
        A->boundaryconstant = 0.0;
        phi1->boundarytype = {0,0,0,0};
        phi2->boundarytype={0,0,0,0};
        A->boundarytype={0,0,0,0};
        setAllBoundaryType({0,0,0,0});
        bdw = {2,2,2,2};
        g(0,0) = 1.0;
        g(1,1) = 1.0;
        g(0,1) = 0.0;
        g(1,0) = 0.0;

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
        addParameter(&lambda1a(0,0), "lambda1x");
        addParameter(&lambda1a(1,1), "lambda1y");
        addParameter(&lambda1a(0,1), "lambda1xy");
        addParameter(&lambda2a(0,0), "lambda2x");
        addParameter(&lambda2a(1,1), "lambda2y");
        addParameter(&lambda2a(0,1), "lambda2xy");
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

        double lambda1x = lambda1a(0,0);
        double lambda1y = lambda1a(1,1);
        double lambda2x = lambda2a(0,0);
        double lambda2y = lambda2a(1,1);

        double coef1 = pow(lambda1a(0,0),2) + pow(lambda1a(0,1),2);
        double coef2 = pow(lambda1a(1,1),2) + pow(lambda1a(0,1),2);
        double coef3 = lambda1a(0,1)*(lambda1a(0,0) + lambda1a(1,1));

        double sum = ( g(0,0)*coef1 + g(0,1)*coef3 )*( (phi1x.squaredNorm()) + pow(A->data[pos][0],2)*phi1sq(pos)
                    + 2.0*A->data[pos][0]*(phi1->data[pos][1]*phi1x[0]-phi1->data[pos][0]*phi1x[1]))

                    +( g(1,1)*coef2 + g(0,1)*coef3 )*( (phi1y.squaredNorm()) + pow(A->data[pos][1],2)*phi1sq(pos)
                    + 2.0*A->data[pos][1]*(phi1->data[pos][1]*phi1y[0]-phi1->data[pos][0]*phi1y[1]))

                    +( g(0,0)*coef3 + g(1,1)*coef3 + g(0,1)*(coef1+coef2) )*( (phi1x.dot(phi1y)) + A->data[pos][0]*A->data[pos][1]*phi1sq(pos)
                    + A->data[pos][0]*(phi1->data[pos][1]*phi1y[0]-phi1->data[pos][0]*phi1y[1])  + A->data[pos][1]*(phi1->data[pos][1]*phi1x[0]-phi1->data[pos][0]*phi1x[1]) );

        coef1 = pow(lambda2a(0,0),2) + pow(lambda2a(0,1),2);
        coef2 = pow(lambda2a(1,1),2) + pow(lambda2a(0,1),2);
        coef3 = lambda2a(0,1)*(lambda2a(0,0) + lambda2a(1,1));

        sum += ( g(0,0)*coef1 + g(0,1)*coef3 )*( (phi2x.squaredNorm()) + pow(A->data[pos][0],2)*phi2sq(pos)
                     + 2.0*A->data[pos][0]*(phi2->data[pos][1]*phi2x[0]-phi2->data[pos][0]*phi2x[1]))

                     +( g(1,1)*coef2 + g(0,1)*coef3 )*( (phi2y.squaredNorm()) + pow(A->data[pos][1],2)*phi2sq(pos)
                     + 2.0*A->data[pos][1]*(phi2->data[pos][1]*phi2y[0]-phi2->data[pos][0]*phi2y[1]))

                     +( g(0,0)*coef3 + g(1,1)*coef3 + g(0,1)*(coef1+coef2) )*( (phi2x.dot(phi2y)) + A->data[pos][0]*A->data[pos][1]*phi2sq(pos)
                     + A->data[pos][0]*(phi2->data[pos][1]*phi2y[0]-phi2->data[pos][0]*phi2y[1]) + A->data[pos][1]*(phi2->data[pos][1]*phi2x[0]-phi2->data[pos][0]*phi2x[1])   );

        d[0][0] = phi1x;
        d[0][1] = phi1y;
        d[1][0] = phi2x;
        d[1][1] = phi2y;

        p[0] = phi1->data[i];
        p[1] = phi2->data[i];

        double sum = 0.0;

        for(int alpha = 0; alpha < NoFields; alpha++) {
            for (int beta = 0; beta < NoFields; beta++) {
                for (int j = 0; j < dim; j++) {
                    for (int k = 0; k < dim; k++) {
                        D[alpha][beta][j][k] = d[alpha][j].dot(d[beta][k]) + e*e*A[j]*A[k]*(p[alpha].dot(p[alpha]))
                                               +e*A[j]*(p[alpha][0]*d[beta][k][1] - p[alpha][1]*d[beta][k][0])
                                               +e*A[k]*(p[beta][0]*d[alpha][j][1] - p[beta][1]*d[alpha][j][0]);
                        sum += P[alpha][beta][j][k]*D[alpha][beta][j][k];


                    }
                }
            }
        }


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
            Eigen::Vector2d phi1xy = double_derivative(phi1, 0, 1, pos);
            Eigen::Vector2d phi2x = single_derivative(phi2, 0, pos);
            Eigen::Vector2d phi2y = single_derivative(phi2, 1, pos);
            Eigen::Vector2d phi2xx = double_derivative(phi2, 0, 0, pos);
            Eigen::Vector2d phi2yy = double_derivative(phi2, 1, 1, pos);
            Eigen::Vector2d phi2xy = double_derivative(phi2, 0, 1, pos);
            Eigen::Vector2d Ax = single_derivative(A, 0, pos);
            Eigen::Vector2d Ay = single_derivative(A, 1, pos);
            Eigen::Vector2d Axx = double_derivative(A, 0, 0, pos);
            Eigen::Vector2d Ayy = double_derivative(A, 1, 1, pos);
            Eigen::Vector2d Axy = double_derivative(A, 0, 1, pos);




            double lambda1x = lambda1a(0,0);
            double lambda1y = lambda1a(1,1);
            double lambda2x = lambda2a(0,0);
            double lambda2y = lambda2a(1,1);

            /*Eigen::Vector2d Agradient;

            Agradient[0] = -( Axy[1] - Ayy[0]) - pow(lambda1x,2)*A->data[pos][0]*phi1sq(pos) - pow(lambda2x,2)*A->data[pos][0]*phi2sq(pos);
            Agradient[1] = ( Axx[1] - Axy[0] ) - pow(lambda1y,2)*A->data[pos][1]*phi1sq(pos) - pow(lambda2y,2)*A->data[pos][1]*phi2sq(pos);

            Agradient[0] = Agradient[0] - pow(lambda1x,2)*(phi1->data[pos][1]*phi1x[0]-phi1->data[pos][0]*phi1x[1]);
            Agradient[0] = Agradient[0] - pow(lambda2x,2)*(phi2->data[pos][1]*phi2x[0]-phi2->data[pos][0]*phi2x[1]);
            Agradient[1] = Agradient[1] - pow(lambda1y,2)*(phi1->data[pos][1]*phi1y[0]-phi1->data[pos][0]*phi1y[1]);
            Agradient[1] = Agradient[1] - pow(lambda2y,2)*(phi2->data[pos][1]*phi2y[0]-phi2->data[pos][0]*phi2y[1]);*/

            Eigen::Vector2d phi1gradient(0,0);
            Eigen::Vector2d phi2gradient(0,0);


            for(int alpha = 0; alpha < NoFields; alpha++) {
                for (int beta = 0; beta < NoFields; beta++) {
                    for (int j = 0; j < dim; j++) {
                        for (int k = 0; k < dim; k++) {
                            D[alpha][beta][j][k] = d[alpha][j].dot(d[beta][k]) + e*e*A[j]*A[k]*(p[alpha].dot(p[alpha]))
                                                   +e*A[j]*(p[alpha][0]*d[beta][k][1] - p[alpha][1]*d[beta][k][0])
                                                   +e*A[k]*(p[beta][0]*d[alpha][j][1] - p[beta][1]*d[alpha][j][0]);

                            D = dd[beta][k][j]) - e*e*A[j]*A[k]*(p[beta]);
                            D[0] = -e*A[j]*(d[beta][k][1]);
                            D[1] = -e*A[j]*(-d[beta][k][0]);
                            D[0] = +e*A
                                                   +e*A[k]*(p[beta][0]*d[alpha][j][1] - p[beta][1]*d[alpha][j][0]);



                            sum += P[alpha][beta][j][k]*D[alpha][beta][j][k];


                        }
                    }
                }
            }

            phi1gradient1 += P[0][0][0][0]*(phi1xx + e*e*A



            double coef1 = pow(lambda1a(0,0),2) + pow(lambda1a(0,1),2);
            double coef2 = pow(lambda1a(1,1),2) + pow(lambda1a(0,1),2);
            double coef3 = lambda1a(0,1)*(lambda1a(0,0) + lambda1a(1,1));

            double x1c = ( g(0,0)*coef1 + g(0,1)*coef3 );
            double y1c = ( g(1,1)*coef2 + g(0,1)*coef3 );
            double xy1c = ( g(0,0)*coef3 + g(1,1)*coef3 + g(0,1)*(coef1+coef2) );

            Eigen::Vector2d phi1gradient = x1c*( phi1xx - pow(A->data[pos][0],2)*phi1->data[pos]  );

            phi1gradient[0] += x1c*( A->data[pos][0]*(phi1x[1]) + Ax[0]*phi1->data[pos][1] + A->data[pos][0]*phi1x[1]);
            phi1gradient[1] += -x1c*(A->data[pos][0]*(phi1x[0]) + Ax[0]*phi1->data[pos][0] + A->data[pos][0]*phi1x[0]);

            phi1gradient += y1c*( phi1yy - pow(A->data[pos][1],2)*phi1->data[pos]  );

            phi1gradient[0] += y1c*(A->data[pos][1]*(phi1y[1]) + Ay[1]*phi1->data[pos][1] + A->data[pos][1]*phi1y[1]);
            phi1gradient[1] += -y1c*(A->data[pos][1]*(phi1y[0]) + Ay[1]*phi1->data[pos][0] + A->data[pos][1]*phi1y[0]);

            phi1gradient += xy1c*( phi1xy - A->data[pos][0]*A->data[pos][1]*phi1->data[pos] );

            phi1gradient[0] += 0.5*xy1c*(A->data[pos][0]*(phi1y[1]) + Ay[0]*phi1->data[pos][1] + A->data[pos][0]*phi1y[1]);
            phi1gradient[0] += 0.5*xy1c*(A->data[pos][1]*(phi1x[1]) + Ax[1]*phi1->data[pos][1] + A->data[pos][1]*phi1x[1]);
            phi1gradient[1] += -0.5*xy1c*(A->data[pos][0]*(phi1y[0]) + Ay[0]*phi1->data[pos][0] + A->data[pos][0]*phi1y[0]);
            phi1gradient[1] += -0.5*xy1c*(A->data[pos][1]*(phi1x[0]) + Ax[1]*phi1->data[pos][0] + A->data[pos][1]*phi1x[0]);

            phi1gradient = phi1gradient + 0.5*lambda1*phi1->data[pos]*(m1*m1 - (phi1sq(pos)));

            phi1gradient = phi1gradient + k0*0.25*phi2->data[pos];

            coef1 = pow(lambda2a(0,0),2) + pow(lambda2a(0,1),2);
            coef2 = pow(lambda2a(1,1),2) + pow(lambda2a(0,1),2);
            coef3 = lambda2a(0,1)*(lambda2a(0,0) + lambda2a(1,1));

            double x2c = ( g(0,0)*coef1 + g(0,1)*coef3 );
            double y2c = ( g(1,1)*coef2 + g(0,1)*coef3 );
            double xy2c = ( g(0,0)*coef3 + g(1,1)*coef3 + g(0,1)*(coef1+coef2) );

            Eigen::Vector2d phi2gradient = x2c*( phi2xx - pow(A->data[pos][0],2)*phi2->data[pos]  );

            phi2gradient[0] += x2c*( A->data[pos][0]*(phi2x[1]) + Ax[0]*phi2->data[pos][1] + A->data[pos][0]*phi2x[1]);
            phi2gradient[1] += -x2c*(A->data[pos][0]*(phi2x[0]) + Ax[0]*phi2->data[pos][0] + A->data[pos][0]*phi2x[0]);

            phi2gradient += y2c*( phi2yy - pow(A->data[pos][1],2)*phi2->data[pos]  );

            phi2gradient[0] += y2c*(A->data[pos][1]*(phi2y[1]) + Ay[1]*phi2->data[pos][1] + A->data[pos][1]*phi2y[1]);
            phi2gradient[1] += -y2c*(A->data[pos][1]*(phi2y[0]) + Ay[1]*phi2->data[pos][0] + A->data[pos][1]*phi2y[0]);

            phi2gradient += xy2c*( phi2xy - (A->data[pos][0])*(A->data[pos][1])*phi2->data[pos] );

            phi2gradient[0] += 0.5*xy2c*(A->data[pos][0]*(phi2y[1]) + Ay[0]*phi2->data[pos][1] + A->data[pos][0]*phi2y[1]);
            phi2gradient[0] += 0.5*xy2c*(A->data[pos][1]*(phi2x[1]) + Ax[1]*phi2->data[pos][1] + A->data[pos][1]*phi2x[1]);
            phi2gradient[1] += -0.5*xy2c*(A->data[pos][0]*(phi2y[0]) + Ay[0]*phi2->data[pos][0] + A->data[pos][0]*phi2y[0]);
            phi2gradient[1] += -0.5*xy2c*(A->data[pos][1]*(phi2x[0]) + Ax[1]*phi2->data[pos][0] + A->data[pos][1]*phi2x[0]);

            phi2gradient = phi2gradient + 0.5*lambda2*phi2->data[pos]*(m2*m2 - (phi2sq(pos)));
            phi2gradient = phi2gradient + k0*0.25*phi1->data[pos];

            Eigen::Vector2d Agradient;

            Agradient[0] = -( Axy[1] - Ayy[0]) - x1c*A->data[pos][0]*phi1sq(pos) - x2c*A->data[pos][0]*phi2sq(pos);
            Agradient[1] = ( Axx[1] - Axy[0] ) - y1c*A->data[pos][1]*phi1sq(pos) - y2c*A->data[pos][1]*phi2sq(pos);

            Agradient[0] = Agradient[0] - x1c*(phi1->data[pos][1]*phi1x[0]-phi1->data[pos][0]*phi1x[1]);
            Agradient[0] = Agradient[0] - x2c*(phi2->data[pos][1]*phi2x[0]-phi2->data[pos][0]*phi2x[1]);
            Agradient[1] = Agradient[1] - y1c*(phi1->data[pos][1]*phi1y[0]-phi1->data[pos][0]*phi1y[1]);
            Agradient[1] = Agradient[1] - y2c*(phi2->data[pos][1]*phi2y[0]-phi2->data[pos][0]*phi2y[1]);

            Agradient[0] += -1.0*xy1c*0.5*A->data[pos][1]*phi1sq(pos);
            Agradient[1] += -1.0*xy1c*0.5*A->data[pos][0]*phi1sq(pos);

            Agradient[0] += -1.0*xy2c*0.5*A->data[pos][1]*phi2sq(pos);
            Agradient[1] += -1.0*xy2c*0.5*A->data[pos][0]*phi2sq(pos);

            Agradient[0] = Agradient[0] - 0.5*xy1c*(phi1->data[pos][1]*phi1y[0]-phi1->data[pos][0]*phi1y[1]);
            Agradient[1] = Agradient[1] - 0.5*xy1c*(phi1->data[pos][1]*phi1x[0]-phi1->data[pos][0]*phi1x[1]);

            Agradient[0] = Agradient[0] - 0.5*xy2c*(phi2->data[pos][1]*phi2y[0]-phi2->data[pos][0]*phi2y[1]);
            Agradient[1] = Agradient[1] - 0.5*xy2c*(phi2->data[pos][1]*phi2x[0]-phi2->data[pos][0]*phi2x[1]);

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

    double VortexModel::initial(double r)
    {
        double a;
        double initialradius = 3.0;
        if(r > initialradius)
        {
            a = m1;
        }
        else
        {
            //a=m1 - m1*(1.0 - r/initialradius);
            a = m1-m1*exp(-5.0*(r*r));
        }
        return (a);
    }


    double VortexModel::inta(double r)
    {
        if(r > 4.0)
        {
            return 1.0/r;
        }
        else
        {
            return exp(-5.0*(r*r));
        }
    }

}


#endif //FTPL_COMPACTVORTICIESWMETRIC_HPP
