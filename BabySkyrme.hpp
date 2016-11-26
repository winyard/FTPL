/**
 * Field Theory Processing Library
 * Copyright Thomas Winyard 2016
 */

#ifndef FTPL_H_
#define FTPL_H_

#include <cmath>
#include "FieldTheories.hpp"

using namespace std;

namespace FTPL {

class BabySkyrmeModel : public BaseFieldTheory {
    public:
	//required functions
        BabySkyrmeModel(const char * filepath);
        BabySkyrmeModel(unsigned int width, unsigned int height): f(width,height);
        BabySkyrmeModel(BabySkyrmeField * otherBabySkyrmeField, Transformation T = IntensityTransformation(DEFAULT));//Allows you to act with a transformation (rotation, isorotation etc.)
        void save(const char * filename);
	//The maths
	float calculateEnergy(int i,int j);
	float calculateEnergy(Eigen::VectorXd<int> pos);
	float calculateCharge(int i,int j);
	float calculateCharge(Eigen::VectorXd<int> pos);
	void updateEnergy();
	void updateCharge();
	private:
	// parameters and fields
	float2 spacing;
	float mu, mpi;
	Field<Eigen::VectorXd> * f;
};


BabySkyrmeModel::BabySkyrmeModel(unsigned int width, unsigned int height): {
	vector<int> sizein = (width, height);
	BaseFieldTheory(2,sizein);
	f = createField(Eigen::VectorXd);
};


BabySkyrmeModel::BabySkyrmeModel(const char * filename){
    // mearly place holders so the fields can be initialised
    dim = 2;
    size = (2,2);
    f = createField(Eigen::VectorXd);
    load(filename);
};

//maths!
float BabySkyrmeModel::calculateEnergy(Eigen::VectorXd<int> pos){
	Eigen::VectorXd dx = single_derivative(f, 1, pos);
	Eigen::VectorXd dy = single_derivative(f, 2, pos);
	return 0.5*(dx.squarednorm() + dy.squarednorm() + mu*mu*(dx.cross(dy)).squarednorm()) + mpi*mpi*(1.0 - f.getValue(pos)[3]);
};

float BabySkyrmeModel::calculateEnergy(int i, int j){
Eigen::VectorXd<int> inpos = (i,j);
return this->calculateEnergy(inpos)

};

float BabySkyrmeModel::calculateCharge(Eigen::VectorXd<int> pos){
	Eigen::VectorXd dx = this->single_derivative(f, 1, pos);
	Eigen::VectorXd dy = this->single_derivative(f, 2, pos);
	return (1.0/(4.0*M_PI))*( (f.getValue(pos)).dot(dx.cross(dy)) );
};

float BabySkyrmeModel::calculateCharge(int i, int j){
Eigen::VectorXd<int> inpos = (i,j);
return this->calculateCharge(inpos)

};

void BabySkyrmeModel::updateEnergy(){
	float sum = 0.0;
	for(int i = 0; i < this->size[0]; i++){
	for(int j = 0; j < this->size[1]; j++){
		float buffer = this->calculateEnergy(i,j);
		this->setEnergy(i,j,buffer):
		sum += buffer;
	}}
	this->energy = sum*spacing[0]*spacing[1];

};

	void updateCharge();

}

 // End FTPL namespace
#endif /* FTPL_H_ */
