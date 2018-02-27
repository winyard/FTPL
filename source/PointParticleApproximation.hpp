#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <unistd.h>
#include <stdlib.h>

using namespace std;

const int nopos = 3;

const int fix1stphase = 1;

const double max_temp = 10.0;
const int max_loops = 1000000000;
const double tempconst = 0.9999999;
const double T_min = pow(10,-6);

const double max_walk = 0.1;//ceil(nx/20);

const int often = 1000000;//50000000;
const int calcoften = 10;
//constants from the energies(alter)
/*const double A = -31.4;
const double B = -0.92;
const double r_e = 0.55;
const double a_exp = log(2);///0.55;*/

const double m = 0.2;
const double s = 0.5;
const double R = 10.0;

const double A = -32.15;
const double B = 0.76;
const double r_e= 0.73;
const double a_exp = 1.13;

double pos[nopos][3];
double inttemp[nopos], gravtemp;
double inten[nopos][nopos], graven[nopos];

//////////////////////
/*START OF FUNCTIONS*/
//////////////////////

double E11(double r)
{
    return (1.0/s)*log(R/r) + (m/s)*cyl_bessel_k(0, r);
}

double E22(double r)
{
    return s*log(R/r) + (s/m)*cyl_bessel_k(0, r);
}

double E12(double r)
{
    return -log(R/r) + cyl_bessel_k(0, r);
}

double int_pot(double dist, double com1, double com2)
{
    if(com1 != com2){return E12(dist);}
    else
    {
        if(com1 == 1){return E11(dist);}
        else{return E22(dist);}
    }

    double distance(double x1, double y1, double x2, double y2)
    {
        return sqrt(pow(x1-x2,2) + pow(y1-y2,2));
    }


/*Calculate the energy of current configuration*/
    double calcenergy()
    {
        int i,j;
        double energy;
        energy = 0.0;
//calculate interaction energies
        for(i=0;i<nopos;i++)
        {
            for(j=0;j<i;j++)
            {
                if(i != j)
                {
                    double dist = distance(pos[i][0],pos[i][1],pos[j][0],pos[j][1]);

                    inten[i][j] = int_pot(dist,pos[i][2],pos[j][2]);
                    inten[j][i] = inten[i][j];
                    energy = energy + inten[i][j];
                }
            }
        }

        return energy;
    }

    double updateenergy(int n, double initialenergy)
    {
        int i,j;
        double energy, entemp;
        energy = initialenergy;
//update interaction energies
        for(i=0;i<nopos;i++)
        {
            if(n != i)
            {
                double dist = distance(pos[n][0],pos[n][1],pos[i][0],pos[i][1]);

                inttemp[i] = int_pot(dist,pos[n][2],pos[i][2]);

                energy = energy - inten[n][i] + inttemp[i];

            }
        }
        return energy;
    }

    int configurationvalid()
    {
        int i,j,subcheck;
        subcheck = 1;
        for(i = 0;i<nopos;i++)
        {
            //is it within the disc
            if(pow(pos[i][0],2) + pow(pos[i][1],2) >= R)
            {
                subcheck = 0;
            }
            //check they are not overlapping
            for(j = 0;j<nopos;j++)
            {
                if(pos[i][0] == pos[j][0] && pos[i][1] == pos[j][1] && i != j)
                {
                    subcheck = 0;
                }
            }
        }
        return subcheck;
    }

//find random initialconditons and return energy;
    double initialconditions()
    {
        int n,check;
        double energy;
        check = 0;
// loop untill initial conditions are valid
        while(check == 0)
        {
            // set positions (0,1) and rotation (2) randomly
            for(n = 0; n<nopos; n++)
            {
                pos[n][0] = R;
                pos[n][1] = R;
                while(pow(pos[n][0],2) + pow(pos[n][1],2) > R)
                {
                    pos[n][0] = R*(((double)rand() / (double)(RAND_MAX))*2.0 - 1.0);
                    pos[n][1] = R*(((double)rand() / (double)(RAND_MAX))*2.0 - 1.0);
                }
            }
            //check they are within the grid
            if(configurationvalid() == 1)
            {
                energy = calcenergy();
                if(energy == energy)
                {
                    check = 1;
                }
            }
        }
// calculate the initial energy of the configuration
        return energy;
    }

    void output(int n, double T, double energy)
    {
        int i;
        cout << T << " - " << energy << "\n";
        for(i = 0;i<nopos;i++)
        {
            cout << "[" << pos[i][0] << "," << pos[i][1] << "," << pos[i][2] << " ](" << sqrt(pow(pos[i][0],2) + pow(pos[i][1],2)) << "," << atan2(pos[i][1],pos[i][0]) << ")";
            cout << "\n";
        }
        cout << "Energy = " << energy << "\n";

        ofstream outftemp("output_temp");
        outftemp << n <<"," << T << " -";
        for(i = 0;i<nopos;i++)
        {
            outftemp << "[" << pos[i][0] << "," << pos[i][1] << "," << pos[i][2] << " ](" << sqrt(pow(pos[i][0],2) + pow(pos[i][1],2)) << "," << atan2(pos[i][1],pos[i][0]) << ")";
        }
        outftemp << " - " << energy << "\n";
        outftemp.close();
    }



/*MAIN FUNCTION*/
    int main_approximation( int nopos )
    {
        int n,i,k,update;
        double j;
        double energy,oldenergy,T;

        srand (time(NULL));

//set initial conditions
        oldenergy = initialconditions();

//set parameters
        T = max_temp;

/*BEGIN MAIN LOOP!*/
        n = -1;
        while(T > T_min)
        {
            n += 1;
//for(n = 0;n<=max_loops;n++)
//{
//output result everynow and then
            if(n%often == 0)
            {
                output(n,T,oldenergy);
            }
            if(n%calcoften == 0)
            {
                energy = calcenergy();
                if(abs(oldenergy - energy)> 10.1)
                {
                    cout << "AGH! the error is too high!\n";
                    cout << energy << " , " << oldenergy << "\n";
                }
                oldenergy = energy;
            }

//select valid neighbouring configuration
            update = 0;
            while(update == 0)
            {
                k = rand()%nopos;
                i = rand()%2;
                j=0;
                while(j==0)
                {
                    j = max_walk*(((double)rand() / (double)(RAND_MAX))*2.0 - 1.0);
                }
                pos[k][i] = pos[k][i] + j;

                if(configurationvalid() == 1)
                {
                    update = 1;
                }
                else
                {
                    pos[k][i] = pos[k][i] - j;
                }
            }
//compare newconfiguration with old and calc update probability
            update = 0;
            energy = updateenergy(k,oldenergy);
            if(energy > oldenergy || energy != energy)
            {
                double p = exp(-(energy - oldenergy)/T); //probability
                double random = ((double)rand() / (double)(RAND_MAX));
                if(p >= random)
                {
                    update = 1;
                }
            }
            else
            {
                update = 1;
            }

            if(k==0 && i==2 && fix1stphase == 1)
            {
                update = 0;
            }

//update new configuration
            if(update == 1)
            {
                oldenergy = energy;
                for(i = 0;i<nopos;i++)
                {
                    if(i != k)
                    {
                        inten[i][k] = inttemp[i];
                        inten[k][i] = inttemp[i];
                    }
                }

            }
            else
            {
                pos[k][i] = pos[k][i] - j;
            }

// reduce temperature
            T *= tempconst;
        }

//OUTPUT OPTIMAL POSITIONS
        cout << "\n";
        cout << "\n";
        cout << "FINAL POSITIONS ARE:\n";
        output(n,T,oldenergy);

        ofstream outf("approx");
        for(i=0;i<nopos;i++)
        {
            double r = sqrt(pow(pos[i][0],2) + pow(pos[i][1],2));
            outf << pos[i][0] << " " << pos[i][1] << " " << pos[i][2] << "\n";
        }
        outf.close();

    }



