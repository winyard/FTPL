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
	Field<Eigen::VectorXd<double,3>> f;
};


BabySkyrmeModel::BabySkyrmeModel(unsigned int width, unsigned int height): f(width,height){
	vector<int> sizein = (width, height);
	BaseFieldTheory(2,sizein);
	f = createField(field<Eigen::VectorXd<float,3>>(2,(Eigen::VectorXd)[0,0,1],sizein);
};


BabySkyrmeModel::BabySkyrmeModel(const char * filename){

};

template <class T>
BabySkyrmeModel::BabySkyrmeModel(const char * filename) {
ifstream openfile(filename);
int a,b;
double d, e, g;
// 1st line read in size axbxc and spacing dx,dy,dz
    openfile >> a >> b >> d >> e ;
    Eigen::VectorXd<int> sizein(2) = (a, b);
    Eigen::VectorXd<int> spacingin(2) = (d, e);
    BaseFieldTheory(2,sizein,spacingin);
    this->f = field<Eigen::VectorXd<float,3>>(2,(Eigen::VectorXd)[0,0,1],sizein)	
// 2nd line read in parameters
    openfile >> d >> g;
    this->mu = d;
    this->mpi = g;
// read in data
while(!openfile.eof())
{
    openfile >> a >> b >> d >> e >> g ;
    this->data[a+b*sizein(2)] = (d,e,g);
}
cout << "Baby Skyrme Field of dimensions [" << sizein(0) <<"," << sizein(1) << "] succesfully read from file " << filename << "\n";
}

//Save functions
void BabySkyrmeModel::save(const char * filename) {
ofstream savefile(filename);
// 1st line save size axb and spacing dx,dy
savefile << this->size[0] << " " << this->size[1] << "\n";
// 2nd line read save parameters
savefile << this->mu << " " << this->mpi << "\n";
// save data
for(int i = 0; i < this->size[0] ; i++){
for(int j = 0; j < this->size[1] ; j++){
     outputbuffer = this->f.get(i, j);
    savefile << i << " " << j << " " << outputbuffer[1] << " " << outputbuffer[2] << " " << outputbuffer[3] << "\n";
}}
cout << "Baby Skyrme Field output to file " << filename << "\n";
}

//maths!
float BabySkyrmeModel::calculateEnergy(Eigen::VectorXd<int> pos){
	Eigen::VectorXd dx = this->single_derivative(f, 1, pos);
	Eigen::VectorXd dy = this->single_derivative(f, 2, pos);
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
