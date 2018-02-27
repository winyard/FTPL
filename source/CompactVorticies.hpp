//
// Created by tom on 2017-11-20.
//

#ifndef FTPL_COMPACTVORTICIES_HPP
#define FTPL_COMPACTVORTICIES_HPP

#include <cmath>
#include "FieldTheories.hpp"
#include <Eigen/Dense>

using namespace std;

namespace FTPL {

    class VortexModel : public BaseFieldTheory {
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
        VortexModel(const char * filepath, bool isDynamic = false);
        VortexModel(int width, int height, bool isDynamic = false);
        ~VortexModel(){};
        void setParameters(double lambda1_in, double lambda2_in, double m1_in, double m2_in, double k0_in);
        double calculateCharge(int pos);
        double getCharge(){return charge;};
        inline double phi1sq(int pos);
        inline double phi2sq(int pos);
        void makePeriodic();
        Eigen::Matrix2d lambda1a;
        Eigen::Matrix2d lambda2a;
        void minimiseMap(int loop);
        void virtual correctTheory(int loop);
        void setAnisotropy(vector<vector<Eigen::Matrix2d>> Qin);
        void plotPhi1sq();
        void plotPhi2sq();
        void plotAx();
        void plotAy();
        void outputSpace();
        void perturbL();
        void storeFields();
    private:
        // parameters
        double lambda1, lambda2, m1 , m2, k0, e;
        Eigen::Matrix2d L; // Linear Tori Map
        vector<vector<Eigen::Matrix2d>> Qt;
        vector<vector<Eigen::Matrix2d>> St;
        double detL;
        int NoFields = 2;

    };

    void VortexModel::storeFields(){

        ofstream output("/home/tom/FTPLdata.gp");

        double xmax = size[0]*spacing[0]/2.0;
        double ymax = size[1]*spacing[1]/2.0;

        for(int i=0; i<size[0]; i++){
        for(int j = 0;j<size[1];j++){
            double x = i*spacing[0];
            double y = j*spacing[1];
            Eigen::Vector2d v1(L(0,1), L(0,0));
            Eigen::Vector2d v2(L(1,1), L(1,0));
            double xout = x*v2[0] + y*v1[0];
            double yout = x*v2[1] + y*v1[1];
            vector<int> p(2);
            p[0] = i;
            p[1] = j;
            int point = p[0];
            int multiplier = 1;
            for(int a=1; a<dim; a++){
                multiplier *= size[a-1];
                point += p[a]*multiplier;
            }
            output << x << " " << y << " " << xout << " " << yout << " " << energydensity[point] << " " << chargedensity[point] << "\n";
        }
            output << "\n";
        }
        cout << "Fields output to file!\n";
    }

    void VortexModel::perturbL(){
        L << 0.467, 0.964, 0.964, 0.7564;
        double detLin = L.determinant();
        if(detLin<0){detLin = -sqrt(-detLin);}
        else {detLin = sqrt(detLin);}
        L(0,0) = L(0,0)/detLin;
        L(1,0) = L(1,0)/detLin;
        L(0,1) = L(0,1)/detLin;
        L(1,1) = L(1,1)/detLin;
    }

    void VortexModel::outputSpace(){
        cout << "vectors of the space are \n";
        cout << "|" << L(0,0) << "| |" << L(0,1) <<"|\n";
        cout << "|" << L(1,0) << "| |" << L(1,1) <<"|\n";
        Eigen::Vector2d v1(L(0,0),L(1,0));
        Eigen::Vector2d v2(L(1,0),L(1,1));
        cout << "det L = " << L.determinant()<<"\n";
        cout << "with angle = " << acos(v1.dot(v2)/(sqrt(v1.dot(v1)*v2.dot(v2))))/M_PI << " pi\n";
        cout << "St11:\n";
        cout << St[0][0] <<"\n";
        cout << "St22:\n";
        cout << St[1][1] << "\n";
        cout << "St12:\n";
        cout << St[0][1] << "\n";
        cout << "St21:\n";
        cout << St[1][0] << "\n";
    }

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

    void VortexModel::correctTheory(int loop) {
        if(loop%100 == 0) {
            minimiseMap(loop);
            calculateParameters();
        }
    }

    void VortexModel::setParameters(double lambda1_in, double lambda2_in, double m1_in, double m2_in, double k0_in){
        lambda1 = lambda1_in;
        lambda2 = lambda2_in;
        m1 = m1_in;
        m2 = m2_in;
        k0 = k0_in;
        e = 1;
    }

    void VortexModel::calculateParameters(){
        Eigen::Matrix2d M = L.inverse();
        for(int alpha = 0; alpha < NoFields; alpha++)
        {
            for(int beta = 0; beta < NoFields ; beta++) {
                St[alpha][beta]=M*Qt[alpha][beta]*M.transpose();
            }
        }
        detL = L.determinant();
    }

    void VortexModel::minimiseMap(int loop){
        //calculate all the sums of the differentials
        vector<vector<Eigen::Matrix2d>> sum(NoFields, vector<Eigen::Matrix2d>(NoFields));
        sum[0][0](0,0) = 0.0;
        sum[0][0](0,1) = 0.0;
        sum[0][0](1,0) = 0.0;
        sum[0][0](1,1) = 0.0;

        sum[0][1](0,0) = 0.0;
        sum[0][1](0,1) = 0.0;
        sum[0][1](1,0) = 0.0;
        sum[0][1](1,1) = 0.0;

        sum[1][0](0,0) = 0.0;
        sum[1][0](0,1) = 0.0;
        sum[1][0](1,0) = 0.0;
        sum[1][0](1,1) = 0.0;

        sum[1][1](0,0) = 0.0;
        sum[1][1](0,1) = 0.0;
        sum[1][1](1,0) = 0.0;
        sum[1][1](1,1) = 0.0;

        for(int pos = 0; pos < getTotalSize(); pos++){
            if(inBoundary(pos)){
                Eigen::Vector2d phi1x = single_derivative(phi1, 0, pos);
                Eigen::Vector2d phi1y = single_derivative(phi1, 1, pos);
                Eigen::Vector2d phi2x = single_derivative(phi2, 0, pos);
                Eigen::Vector2d phi2y = single_derivative(phi2, 1, pos);

                vector<vector<Eigen::Vector2d>> d(2, vector<Eigen::Vector2d>(2));
                vector<Eigen::Vector2d> p(2);

                d[0][0] = phi1x;
                d[0][1] = phi1y;
                d[1][0] = phi2x;
                d[1][1] = phi2y;

                p[0] = phi1->data[pos];
                p[1] = phi2->data[pos];

                for(int alpha = 0; alpha < NoFields; alpha++) {
                    for (int beta = 0; beta <= alpha; beta++) {
                        for (int j = 0; j < dim; j++) {
                            for (int k = 0; k <dim; k++) {
                                double D = d[alpha][j].dot(d[beta][k]) + e*e*A->data[pos](j)*A->data[pos](k)*(p[alpha].dot(p[beta]))
                                           -e*A->data[pos](j)*(p[alpha][0]*d[beta][k][1] - p[alpha][1]*d[beta][k][0])
                                           -e*A->data[pos](k)*(p[beta][0]*d[alpha][j][1] - p[beta][1]*d[alpha][j][0]);

                                sum[alpha][beta](j,k) += 0.5*D;
                                if(alpha != beta){sum[beta][alpha](j,k) += 0.5*D;}

                            }
                        }
                    }
                }

            }
        }
        //calculate P as a 4x4 vector (0,0) -> 0, (1,0) -> 1, (0,1) -> 2, (1,1) -> 3
        Eigen::Matrix4d P;
        P.setZero();
        /*for(int alpha = 0; alpha < 2; alpha++) {
        for(int beta = 0; beta < 2; beta++) {
            P(0, 0) += St[alpha][beta](0, 0) * sum[alpha][beta](0,0);
            P(0, 1) += St[alpha][beta](0, 0) * sum[alpha][beta](1,0);
            P(0, 2) += St[alpha][beta](0, 0) * sum[alpha][beta](0,1);
            P(0, 3) += St[alpha][beta](0, 0) * sum[alpha][beta](1,1);

            P(1, 0) += St[alpha][beta](1, 0) * sum[alpha][beta](0,0);
            P(1, 1) += St[alpha][beta](1, 0) * sum[alpha][beta](1,0);
            P(1, 2) += St[alpha][beta](1, 0) * sum[alpha][beta](0,1);
            P(1, 3) += St[alpha][beta](1, 0) * sum[alpha][beta](1,1);

            P(2, 0) += St[alpha][beta](0, 1) * sum[alpha][beta](0,0);
            P(2, 1) += St[alpha][beta](0, 1) * sum[alpha][beta](1,0);
            P(2, 2) += St[alpha][beta](0, 1) * sum[alpha][beta](0,1);
            P(2, 3) += St[alpha][beta](0, 1) * sum[alpha][beta](1,1);

            P(3, 0) += St[alpha][beta](1, 1) * sum[alpha][beta](0,0);
            P(3, 1) += St[alpha][beta](1, 1) * sum[alpha][beta](1,0);
            P(3, 2) += St[alpha][beta](1, 1) * sum[alpha][beta](0,1);
            P(3, 3) += St[alpha][beta](1, 1) * sum[alpha][beta](1,1);
        }}*/

        for(int alpha = 0; alpha < 2; alpha++) {
            for(int beta = 0; beta < 2; beta++) {
                P(0, 0) += St[alpha][beta](0, 0) * sum[alpha][beta](0,0);
                P(0, 1) += St[alpha][beta](0, 1) * sum[alpha][beta](0,0);
                P(0, 2) += St[alpha][beta](0, 0) * sum[alpha][beta](0,1);
                P(0, 3) += St[alpha][beta](0, 1) * sum[alpha][beta](0,1);

                P(1, 0) += St[alpha][beta](1, 0) * sum[alpha][beta](0,0);
                P(1, 1) += St[alpha][beta](1, 1) * sum[alpha][beta](0,0);
                P(1, 2) += St[alpha][beta](1, 0) * sum[alpha][beta](0,1);
                P(1, 3) += St[alpha][beta](1, 1) * sum[alpha][beta](0,1);

                P(2, 0) += St[alpha][beta](0, 0) * sum[alpha][beta](1,0);
                P(2, 1) += St[alpha][beta](0, 1) * sum[alpha][beta](1,0);
                P(2, 2) += St[alpha][beta](0, 0) * sum[alpha][beta](1,1);
                P(2, 3) += St[alpha][beta](0, 1) * sum[alpha][beta](1,1);

                P(3, 0) += St[alpha][beta](1, 0) * sum[alpha][beta](1,0);
                P(3, 1) += St[alpha][beta](1, 1) * sum[alpha][beta](1,0);
                P(3, 2) += St[alpha][beta](1, 0) * sum[alpha][beta](1,1);
                P(3, 3) += St[alpha][beta](1, 1) * sum[alpha][beta](1,1);
            }}

        //calculate JP
        Eigen::Matrix4d J;
        J.setZero();
        J(0,3) = 1.0;
        J(1,2) = -1.0;
        J(2,1) = -1.0;
        J(3,0) = 1.0;
        Eigen::Matrix4d JP;
        JP = J*P;
        //calculate JP's spectrum and find eigenvector with lowest eigenvalue

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> es(JP);
        int lowest = -10;
        double lowestvalue = 999999;
        for(int i = 0; i<4;i++){

            Eigen::Matrix2d S;
            double eigenv;
            if(es.eigenvalues()(i)<0){eigenv = -sqrt(-es.eigenvalues()(i));}
            else {eigenv = sqrt(es.eigenvalues()(i));}
            eigenv = 1;
            S(0,0) = es.eigenvectors().col(i)(0)/eigenv;
            S(1,0) = es.eigenvectors().col(i)(1)/eigenv;
            S(0,1) = es.eigenvectors().col(i)(2)/eigenv;
            S(1,1) = es.eigenvectors().col(i)(3)/eigenv;
            Eigen::Matrix2d Lcheck;
            Lcheck = S.inverse();

            Eigen::Vector4d X;
            X(0) = es.eigenvectors().col(i)(0);
            X(1) = es.eigenvectors().col(i)(1);
            X(2) = es.eigenvectors().col(i)(2);
            X(3) = es.eigenvectors().col(i)(3);

            //cout << i <<" "<<X.dot(J*X)<<" "<<Lcheck.determinant()<<"\n";

            if(es.eigenvalues()(i)< lowestvalue ) {
                //if(X.dot(J*X) > 0.000001){
                    lowest = i;
                    lowestvalue = es.eigenvalues()(i);
                //}
            }
        }
        /*cout << "JP's spectrum is:\n";
        cout << es.eigenvectors()<<"\n";
        cout << "with eigenvalues:\n";
        cout << es.eigenvalues()<<"\n";
        cout << "found the lowest eigenvalued eigenvector of JP's spectrum: value = " << es.eigenvalues()(lowest)<< ", eigenvector = " << es.eigenvectors().col(lowest) << "\n";*/

        //check x dot J x not less than zero

        //rescale X
        Eigen::Vector4d X;
        double eigenv;
        if(es.eigenvalues()(lowest)<0){eigenv = -sqrt(-es.eigenvalues()(lowest));}
        else {eigenv = sqrt(es.eigenvalues()(lowest));}
        X(0) = es.eigenvectors().col(lowest)(0);
        X(1) = es.eigenvectors().col(lowest)(1);
        X(2) = es.eigenvectors().col(lowest)(2);
        X(3) = es.eigenvectors().col(lowest)(3);

        //eigenv = 1;
        Eigen::Matrix2d S;
        S(0,0) = X(0)/eigenv;
        S(1,0) = X(1)/eigenv;
        S(0,1) = X(2)/eigenv;
        S(1,1) = X(3)/eigenv;

        double detS = S.determinant();
        if(detS<0){detS = -sqrt(-detS);}
        else {detS = sqrt(detS);}
        S(0,0) = S(0,0)/detS;
        S(1,0) = S(1,0)/detS;
        S(0,1) = S(0,1)/detS;
        S(1,1) = S(1,1)/detS;

        //cout << "S is now :\n"<<S<<"\n"<<"with det S = " << S.determinant()<<"\n";

        // store the new parameters
        Eigen::Matrix2d M;
        M = S*L.inverse();
        L = M.inverse();
        //cout << "*" << lowest <<"* "<<X.dot(J*X)<<" "<<L.determinant()<<"\n";
    }

    void VortexModel::setAnisotropy(vector<vector<Eigen::Matrix2d>> Qin){
        cout << "setting anisotropy\n";
        Qt.resize(NoFields);
        St.resize(NoFields);
        for(int i = 0; i < NoFields; i++){
            Qt[i].resize(NoFields);
            St[i].resize(NoFields);
            for(int j =0; j < NoFields; j++){
                Qt[i][j](0,0) = Qin[i][j](0,0);
                Qt[i][j](1,0) = Qin[i][j](1,0);
                Qt[i][j](0,1) = Qin[i][j](0,1);
                Qt[i][j](1,1) = Qin[i][j](1,1);
            }
        }
        calculateParameters();
        cout << "please insert a check that Q is valid (symmetric in alpha,beta and i,j)\n";
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
        L.setIdentity();
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
        Qt.resize(2);
        St.resize(2);
        Qt[0].resize(2);Qt[1].resize(2);
        St[0].resize(2);St[1].resize(2);

    };


    VortexModel::VortexModel(const char * filename, bool isDynamic): BaseFieldTheory(2, {2,2}, isDynamic){
        // mearly place holders so the fields can be initialised
        phi1 = createField(phi1, isDynamic, true);
        phi2 = createField(phi2, isDynamic, true);
        A = createField(A, isDynamic, true);
        L.setIdentity();
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
        Qt.resize(2);
        St.resize(2);
        Qt[0].resize(2);Qt[1].resize(2);
        St[0].resize(2);St[1].resize(2);
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

        vector<vector<Eigen::Vector2d>> d(2, vector<Eigen::Vector2d>(2));
        vector<Eigen::Vector2d> p(2);

        d[0][0] = phi1x;
        d[0][1] = phi1y;
        d[1][0] = phi2x;
        d[1][1] = phi2y;

        p[0] = phi1->data[pos];
        p[1] = phi2->data[pos];

        double sum = 0.0;

        for(int alpha = 0; alpha < NoFields; alpha++) {
            for (int beta = 0; beta <= alpha; beta++) {
                for (int j = 0; j < dim; j++) {
                    for (int k = 0; k <dim; k++) {
                        double D = d[alpha][j].dot(d[beta][k]) + e*e*A->data[pos](j)*A->data[pos](k)*(p[alpha].dot(p[beta]))
                                               -e*A->data[pos](j)*(p[alpha][0]*d[beta][k][1] - p[alpha][1]*d[beta][k][0])
                                               -e*A->data[pos](k)*(p[beta][0]*d[alpha][j][1] - p[beta][1]*d[alpha][j][0]);
                        if(alpha != beta){D *= 2.0;}
                        sum += St[alpha][beta](j,k)*D;

                    }
                }
            }
        }
        /*sum = 0.0;
        for(int alpha = 0; alpha < NoFields; alpha++) {
            for (int j = 0; j < dim; j++) {
                   sum += d[alpha][j].dot(d[alpha][j]) + e*e*A->data[pos](j)*A->data[pos](j)*(p[alpha].dot(p[alpha]))
                         -2.0*e*A->data[pos](j)*(p[alpha][0]*d[alpha][j][1] - p[alpha][1]*d[alpha][j][0]);
            }
        }*/


        sum += pow( Ax[1]-Ay[0] ,2);

        sum += 0.25*lambda1*pow(m1*m1 - phi1sq(pos),2);
        sum += 0.25*lambda2*pow(m2*m2 - phi2sq(pos),2);
        sum += k0*(1.0-0.5*phi1->data[pos].dot(phi2->data[pos]));
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

            Eigen::Vector2d phi1gradient(0,0);
            Eigen::Vector2d phi2gradient(0,0);

            vector<Eigen::Vector2d> D(NoFields);
            vector<Eigen::Vector2d> Dsum(NoFields);
            Eigen::Vector2d DAsum;
            Eigen::Vector2d DA;
            vector<vector<vector<Eigen::Vector2d>>> dd(NoFields);
            vector<vector<Eigen::Vector2d>> d(NoFields, vector<Eigen::Vector2d>(2));
            vector<Eigen::Vector2d> dA(2);
            vector<Eigen::Vector2d> p(NoFields);

            for(int i = 0; i < NoFields; i++){
                dd[i].resize(2);
                for(int j = 0; j < 2; j++){
                    dd[i][j].resize(2);
                }
            }

            dd[0][0][0] = phi1xx;
            dd[0][1][1] = phi1yy;
            dd[0][0][1] = phi1xy;
            dd[0][1][0] = phi1xy;

            dd[1][0][0] = phi2xx;
            dd[1][1][1] = phi2yy;
            dd[1][0][1] = phi2xy;
            dd[1][1][0] = phi2xy;

            d[0][0] = phi1x;
            d[0][1] = phi1y;
            d[1][0] = phi2x;
            d[1][1] = phi2y;

            dA[0] = Ax;
            dA[1] = Ay;

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

                                DA[j] += St[alpha][beta](j, k) * (-e * e * A->data[pos](k) * p[alpha].dot(p[beta])
                                                           + e * (p[alpha][0] * d[beta][k][1] - p[alpha][1] * d[beta][k][0]));

                                //if (alpha != beta || j != k) {
                                    D[beta] += dd[alpha][k][j] - e * e * A->data[pos](j) * A->data[pos](k) * p[alpha];
                                    D[beta](0) += e * A->data[pos](k) * (d[alpha][j](1));
                                    D[beta](1) += -e * A->data[pos](k) * (d[alpha][j][0]);

                                    D[beta](0) += e * dA[k][j] * (p[alpha][1]) + e * A->data[pos](j) * (d[alpha][k][1]);
                                    D[beta](1) += -e * dA[k][j] * (p[alpha][0]) - e * A->data[pos](j) * (d[alpha][k][0]);

                                    DA[k] += St[alpha][beta](j, k) * (-e * e * A->data[pos](j) * p[alpha].dot(p[beta]) + e *
                                             (p[beta][0] * d[alpha][j][1] - p[beta][1] * d[alpha][j][0]));
                                /*}
                                else{
                                    D[alpha] *= 2;
                                    DA *= 2;
                                }*/

                                if(alpha != beta) {
                                    DA *= 2;
                                    D[beta] *= 2;
                                    D[alpha] *= 2;
                                    Dsum[beta] += 0.5 * St[alpha][beta](j, k) * D[beta];
                                }

                                DAsum += 0.5 * DA;

                                Dsum[alpha] += 0.5 * St[alpha][beta](j, k) * D[alpha];
                        }
                    }
                }
            }

            phi1gradient = Dsum[0] + 0.5*lambda1*phi1->data[pos]*(m1*m1 - (phi1sq(pos)));
            phi1gradient = phi1gradient + k0*0.25*phi2->data[pos];

            phi2gradient = Dsum[1] + 0.5*lambda2*phi2->data[pos]*(m2*m2 - (phi2sq(pos)));
            phi2gradient = phi2gradient + k0*0.25*phi1->data[pos];

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

    void VortexModel::plotPhi1sq(){
        vector<double> plotdata;
        plotdata.resize(getTotalSize());
        for(int i = 0; i < getTotalSize(); i++){
            plotdata[i] = phi1->data[i].dot(phi1->data[i]);
        }
        plot2D(size, plotdata);
    }

    void VortexModel::plotPhi2sq(){
        vector<double> plotdata;
        plotdata.resize(getTotalSize());
        for(int i = 0; i < getTotalSize(); i++){
            plotdata[i] = phi2->data[i].dot(phi2->data[i]);
        }
        plot2D(size, plotdata);
    }

    void VortexModel::plotAx(){
        vector<double> plotdata;
        plotdata.resize(getTotalSize());
        for(int i = 0; i < getTotalSize(); i++){
            plotdata[i] = A->data[i](0);
        }
        plot2D(size, plotdata);
    }

    void VortexModel::plotAy(){
        vector<double> plotdata;
        plotdata.resize(getTotalSize());
        for(int i = 0; i < getTotalSize(); i++){
            plotdata[i] = A->data[i](1);
        }
        plot2D(size, plotdata);
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

    double VortexModel::initial(double r)
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


    double VortexModel::inta(double r)
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


#endif //FTPL_COMPACTVORTICIES_HPP
