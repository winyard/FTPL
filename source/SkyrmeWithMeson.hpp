/**
 * Field Theory Processing Library
 * Copyright Thomas Winyard 2016
 */

#ifndef SKYRME_VECTOR_H
#define SKYRME_VECTOR_H

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

#include "Skyrme.hpp"

using namespace std;

namespace FTPL {

    class SkyrmeModelwithMeson : public SkyrmeModel {
    public:
        bool withVector = false;
        CUDA_HOSTDEV  SkyrmeModelwithMeson(int width, int height, int depth, bool isDynamic): SkyrmeModel(width,height,depth,isDynamic){};
        CUDA_HOSTDEV  double RhoEnergy(int field_no, int i, int j);
        CUDA_HOSTDEV  inline virtual double __attribute__((always_inline)) calculateEnergy(int pos) final;
        CUDA_HOSTDEV  void addVectorMeson();
        CUDA_HOSTDEV  virtual void initialCondition(int B, double x_in, double y_in, double z_in, double phi);
        CUDA_HOSTDEV  double vectorprofile(double r);
    private:
        vector<vector<Field<Eigen::VectorXd>*>> mesons;
        double c1 = 0.141;
        double c2 = 0.198;
        double c3 = 0.153;
        double c4 = 0.050;
        double c5 = 0.038;
        double c6 = 0.078;
        double c7 = 0.049;
        double mpi = 0.0;
        double mV =1.0/sqrt(2);
    };

    double SkyrmeModelwithMeson::vectorprofile(double r){
        if(r<0.5){return r;}
        else if(r<1.0){return 1.0-r;}
        else {return 0.0;}
    }

    void SkyrmeModelwithMeson::initialCondition(int B, double x_in, double y_in, double z_in, double phi){
        SkyrmeModel::initialCondition(B,x_in,y_in,z_in,phi);
        if(withVector){
            Eigen::Vector3d zero(0,0,0);
            mesons[0][0]->fill(zero);
            mesons[0][1]->fill(zero);
            mesons[0][2]->fill(zero);
            double xmax = size[0]*spacing[0]/2.0;
            double ymax = size[1]*spacing[1]/2.0;
            double zmax = size[2]*spacing[2]/2.0;
            for(int i = bdw[0]; i < size[0]-bdw[1]; i++){
                for(int j = bdw[2]; j < size[1]-bdw[3]; j++){
                    for(int k = bdw[4]; k < size[2]-bdw[5]; k++) {
                        double x = i * spacing[0] - xmax - x_in;
                        double y = j * spacing[1] - ymax - y_in;
                        double z = k * spacing[2] - zmax - z_in;
                        double r = sqrt(x * x + y * y + z * z);
                        Eigen::Vector3d r_hat(x / r, y / r, z / r);
                        Eigen::Vector3d result1(0.0,-r_hat[2],r_hat[1]);
                        Eigen::Vector3d result2(r_hat[2],0.0,-r_hat[0]);
                        Eigen::Vector3d result3(-r_hat[1],r_hat[0],0.0);
                        mesons[0][0]->setData(vectorprofile(r)*result1 , {i,j,k});
                        mesons[0][1]->setData(vectorprofile(r)*result2 , {i,j,k});
                        mesons[0][2]->setData(vectorprofile(r)*result3 , {i,j,k});
                    }}}

        }
    }

    void SkyrmeModelwithMeson::addVectorMeson(){
        withVector = true;
        mesons.resize(1);
        mesons[0].resize(3);
        mesons[0][0] = createField(mesons[0][0], false);
        mesons[0][1] = createField(mesons[0][1], false);
        mesons[0][2] = createField(mesons[0][2], false);
        Eigen::Vector3d minimum(-0.01,-0.01,-0.01);
        Eigen::Vector3d maximum(0.01,0.01,0.01);
        mesons[0][0]->min = minimum;
        mesons[0][0]->max = maximum;
        mesons[0][1]->min = minimum;
        mesons[0][1]->max = maximum;
        mesons[0][2]->min = minimum;
        mesons[0][2]->max = maximum;
        Eigen::Vector3d zero(0,0,0);
        mesons[0][0]->fill(zero);
        mesons[0][1]->fill(zero);
        mesons[0][2]->fill(zero);
    }

    double SkyrmeModelwithMeson::calculateEnergy(int pos){
        Eigen::Vector4d fx = single_derivative(f, 0, pos);
        Eigen::Vector4d fy = single_derivative(f, 1, pos);
        Eigen::Vector4d fz = single_derivative(f, 2, pos);

        double B = 0.0;

        for(int i = 0; i < 4; i++){
            for(int j = 0; j < 4; j++){
                for(int k = 0; k < 4; k++){
                    for(int l = 0; l < 4; l++){
                        B += levi(i+1,j+1,k+1,l+1)*fx[i]*fy[j]*fz[k]*f->data[pos][l];
                    }
                }
            }
        }

        double energy = c1*(fx.squaredNorm()+fy.squaredNorm()+fz.squaredNorm())
             + c2*(fx.squaredNorm()*(fy.squaredNorm()+fz.squaredNorm()) + fy.squaredNorm()*fz.squaredNorm()
             - fx.dot(fy)*fx.dot(fy) - fx.dot(fz)*fx.dot(fz) - fy.dot(fz)*fy.dot(fz) ) + mpi*mpi*(1.0 - f->data[pos][0]);

        energy -= 1.0*B;

        if(withVector){

            vector<vector<Eigen::Vector3d>> dV;
            dV.resize(3);
            for(int i=0;i<3;i++){
                dV[i].resize(3);
                for(int j=0;j<3;j++){
                    dV[i][j] = single_derivative(mesons[0][j],i,pos);
                }}
            vector<Eigen::Vector3d> H;
            Eigen::Vector3d pi;
            vector<Eigen::Vector3d> dpi;
            dpi.resize(3);
            H.resize(3);
            for(int i=0;i<3;i++){
                pi[i] = f->data[pos][i+1];
                dpi[0][i] = fx[i+1];
                dpi[1][i] = fy[i+1];
                dpi[2][i] = fz[i+1];
            }

            H[0] = dpi[0].cross(pi) + f->data[pos][0]*dpi[0] - fx[0]*pi;
            H[1] = dpi[1].cross(pi) + f->data[pos][0]*dpi[1] - fy[0]*pi;
            H[2] = dpi[2].cross(pi) + f->data[pos][0]*dpi[2] - fz[0]*pi;

            vector<Eigen::Vector3d> V(3);

            for(int i=0;i<3;i++){
                V[i] = mesons[0][i]->data[pos];
            }

            for(int i = 0; i < 3; i++){
            for(int j = 0; j < 3; j++){
            if(i != j){
                energy += 0.5*(dV[i][j].squaredNorm() - dV[i][j].dot(dV[j][i])); //1st term
                energy += 8.0*c3*dV[i][j].dot(V[j].cross(V[i])); // 3rd term (simplify)
                energy += 8.0*c4*(V[i].cross(V[j])).squaredNorm(); // 4th term
            }else{
                energy += 0.5*mV*mV*V[i].squaredNorm(); // mass term (2nd term)
            }}}

            for(int i = 0; i < 3; i++){
            for(int j = 0; j < 3; j++){
            if(i != j){
                energy += 16.0*c5*( (H[i].cross(V[j])).squaredNorm() - (H[i].cross(V[j])).dot(H[j].cross(V[i])) ); // 1st term (simplify, with other terms)
                energy += 2.0*H[i].cross(H[j]).dot( dV[i][j] - dV[j][i] );//2nd term
                energy += 8.0*c7*H[i].cross(H[j]).dot(V[i].cross(V[j]));//3rd term
                energy += 4.0*c6*H[i].cross(H[j]).dot( H[i].cross(V[j]) - H[j].cross(V[j]) );//4th term
                energy += H[i].cross(V[j]).dot( dV[i][j] - dV[j][i] );//5th term
                energy -= 4.0*c3*V[i].cross(V[j]).dot( H[i].cross(V[j]) - H[j].cross(V[i]) );//6th term
            }
            }}
        }
        return energy/(2.0*M_PI*M_PI);
    }

}

#endif