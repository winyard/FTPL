/**
 * Field Theory Processing Library
 * Copyright Thomas Winyard 2016
 */

using namespace std;

namespace FTPL {

    double rationalI(int B) {
        switch (B) {
            case 1:
                return 1.0;
            case 2:
                return M_PI + (8.0 / 3.0);
            case 3:
                return 13.58;
            case 4:
                return 20.65;
            case 5:
                return 35.75;
            case 6:
                return 50.76;
            case 7:
                return 60.87;
            case 8:
                return 85.63;
            case 9:
                return 112.83;
            default:
                cout << "ERROR!! Requested rational map I value for charge " << B << "is unknown, please set\n";
                return 0;
        }
    }

    vector<double> rationalMap(double thi, double theta, int B)
    {
        double reala,realb,ima,imb;
        double a5r = 3.07;
        double b5r = 3.94;
        double a6r = 0.16;
        double b7r = 7.0/sqrt(5);
        double a9r = -1.98;
        switch (B)
        {
            case 1:
                reala = tan(thi/2.0)*cos(theta);
                ima = tan(thi/2.0)*sin(theta);
                realb = 1.0;
                imb = 0.0;
                break;
            case 2:
                reala = pow(tan(thi/2.0),2)*cos(2.0*theta);
                ima = pow(tan(thi/2.0),2)*sin(2.0*theta);
                realb = 1.0;
                imb = 0.0;
                break;
            case 3:
                reala = pow(tan(thi/2.0),3)*cos(3.0*theta) + sqrt(3)*tan(thi/2.0)*sin(theta);
                ima = pow(tan(thi/2.0),3)*sin(3.0*theta) - sqrt(3)*tan(thi/2.0)*cos(theta);
                realb = -sqrt(3)*pow(tan(thi/2.0),2)*sin(2.0*theta) - 1.0;
                imb = sqrt(3)*pow(tan(thi/2.0),2)*cos(2.0*theta);
                break;
            case 4:
                reala = pow(tan(thi/2.0),4)*cos(4.0*theta) - 2.0*sqrt(3)*pow(tan(thi/2.0),2)*sin(2.0*theta) + 1.0;
                ima = pow(tan(thi/2.0),4)*sin(4.0*theta) + 2.0*sqrt(3)*pow(tan(thi/2.0),2)*cos(2.0*theta);
                realb = pow(tan(thi/2.0),4)*cos(4.0*theta) + 2.0*sqrt(3)*pow(tan(thi/2.0),2)*sin(2.0*theta) + 1.0;
                imb = pow(tan(thi/2.0),4)*sin(4.0*theta) - 2.0*sqrt(3)*pow(tan(thi/2.0),2)*cos(2.0*theta);
                break;
            case 5:
                reala = pow(tan(thi/2.0),5)*cos(5.0*theta) + b5r*pow(tan(thi/2.0),3)*cos(3.0*theta) + a5r*tan(thi/2.0)*cos(theta);
                ima = pow(tan(thi/2.0),5)*sin(5.0*theta) + b5r*pow(tan(thi/2.0),3)*sin(3.0*theta) + a5r*tan(thi/2.0)*sin(theta);
                realb = a5r*pow(tan(thi/2.0),4)*cos(4.0*theta) - b5r*pow(tan(thi/2.0),2)*cos(2.0*theta) + 1.0;
                imb = a5r*pow(tan(thi/2.0),4)*sin(4.0*theta) - b5r*pow(tan(thi/2.0),2)*sin(2.0*theta);
                break;
            case 6:
                if(thi == 0 && theta == 0)
                {
                    thi = 0.001;
                }
                reala = pow(tan(thi/2.0),4)*cos(4.0*theta);
                ima = pow(tan(thi/2.0),4)*sin(4.0*theta) + a6r;
                realb = -a6r*pow(tan(thi/2.0),6)*sin(6.0*theta) + pow(tan(thi/2.0),2)*cos(2.0*theta);
                imb = a6r*pow(tan(thi/2.0),6)*cos(6.0*theta) + pow(tan(thi/2.0),2)*sin(2.0*theta);
                break;
            case 7:
                if(thi == 0 && theta == 0)
                {
                    thi = 0.001;
                }
                reala = b7r*pow(tan(thi/2.0),6)*cos(6.0*theta) - 7.0*pow(tan(thi/2.0),4)*cos(4.0*theta) - b7r*pow(tan(thi/2.0),2)*cos(2.0*theta) - 1.0;
                ima = b7r*pow(tan(thi/2.0),6)*sin(6.0*theta) - 7.0*pow(tan(thi/2.0),4)*sin(4.0*theta) - b7r*pow(tan(thi/2.0),2)*sin(2.0*theta);
                realb = pow(tan(thi/2.0),7)*cos(7.0*theta) + b7r*pow(tan(thi/2.0),5)*cos(5.0*theta) + 7.0*pow(tan(thi/2.0),3)*cos(3.0*theta) - b7r*tan(thi/2.0)*cos(theta);
                imb = pow(tan(thi/2.0),7)*sin(7.0*theta) + b7r*pow(tan(thi/2.0),5)*sin(5.0*theta) + 7.0*pow(tan(thi/2.0),3)*sin(3.0*theta) - b7r*tan(thi/2.0)*sin(theta);
                break;
            case 8:
                reala = pow(tan(thi/2.0),6)*cos(6.0*theta) - 0.14;
                ima = pow(tan(thi/2.0),6)*sin(6.0*theta);
                realb = 0.14*pow(tan(thi/2.0),8)*cos(8.0*theta) + pow(tan(thi/2.0),2)*cos(2.0*theta);
                imb = 0.14*pow(tan(thi/2.0),8)*sin(8.0*theta) + pow(tan(thi/2.0),2)*sin(2.0*theta);
                if(realb == 0 && imb == 0)
                {
                    realb = 0.14*pow(tan((thi+0.00001)/2.0),8)*cos(8.0*theta) + pow(tan((thi+0.00001)/2.0),2)*cos(2.0*theta);
                    imb = 0.14*pow(tan((thi+0.00001)/2.0),8)*sin(8.0*theta) + pow(tan((thi+0.00001)/2.0),2)*sin(2.0*theta);
                }
                break;
            case 9:
                reala = -5.0*sqrt(3)*pow(tan(thi/2.0),6)*sin(6.0*theta) - 9.0*pow(tan(thi/2.0),4)*cos(4.0*theta) - 3.0*sqrt(3)*pow(tan(thi/2.0),2)*sin(2.0*theta) + 1.0 + a9r*(pow(tan(thi/2.0),8)*cos(8.0*theta) + sqrt(3)*pow(tan(thi/2.0),6)*sin(6.0*theta) - pow(tan(thi/2.0),4)*cos(4.0*theta) - sqrt(3)*pow(tan(thi/2.0),2)*sin(2.0*theta));
                ima = 5.0*sqrt(3)*pow(tan(thi/2.0),6)*cos(6.0*theta) - 9.0*pow(tan(thi/2.0),4)*sin(4.0*theta) + 3.0*sqrt(3)*pow(tan(thi/2.0),2)*cos(2.0*theta) + a9r*(pow(tan(thi/2.0),8)*sin(8.0*theta) - sqrt(3)*pow(tan(thi/2.0),6)*cos(6.0*theta) - pow(tan(thi/2.0),4)*sin(4.0*theta) + sqrt(3)*pow(tan(thi/2.0),2)*cos(2.0*theta));
                realb = -pow(tan(thi/2.0),9)*cos(9.0*theta) + 3.0*sqrt(3)*pow(tan(thi/2.0),7)*sin(7.0*theta) + 9.0*pow(tan(thi/2.0),5)*cos(5.0*theta) + 5.0*sqrt(3)*pow(tan(thi/2.0),3)*sin(3.0*theta) + a9r*(sqrt(3)*pow(tan(thi/2.0),7)*sin(7.0*theta) + pow(tan(thi/2.0),5)*cos(5.0*theta) - sqrt(3)*pow(tan(thi/2.0),3)*sin(3.0*theta) - tan(thi/2.0)*cos(theta)  );
                imb = -pow(tan(thi/2.0),9)*sin(9.0*theta) - 3.0*sqrt(3)*pow(tan(thi/2.0),7)*cos(7.0*theta) + 9.0*pow(tan(thi/2.0),5)*sin(5.0*theta) - 5.0*sqrt(3)*pow(tan(thi/2.0),3)*cos(3.0*theta) + a9r*(-sqrt(3)*pow(tan(thi/2.0),7)*cos(7.0*theta) + pow(tan(thi/2.0),5)*sin(5.0*theta) + sqrt(3)*pow(tan(thi/2.0),3)*cos(3.0*theta) - tan(thi/2.0)*sin(theta));
                break;
            case 32: {
                double a321r = -0.1;
                double a321i = -33.3;
                double a322r = 1992.1;
                double a322i = 906.8;
                double a323r = 312.0;
                double a323i = 20441.1;
                double a324r = 24183.5;
                double a324i = 8826.7;
                double a325r = -8633.2;
                double a325i = -4867.6;
                double a326r = 383.3;
                double a326i = -559.6;
                reala = pow(tan(thi / 2.0), 2) * (a321r * cos(2.0 * theta) - a321i * sin(2.0 * theta)) +
                        pow(tan(thi / 2.0), 7) * (a322r * cos(7.0 * theta) - a322i * sin(7.0 * theta)) +
                        pow(tan(thi / 2.0), 12) * (a323r * cos(12.0 * theta) - a323i * sin(12.0 * theta)) +
                        pow(tan(thi / 2.0), 17) * (a324r * cos(17.0 * theta) - a324i * sin(17.0 * theta)) +
                        pow(tan(thi / 2.0), 22) * (a325r * cos(22.0 * theta) - a325i * sin(22.0 * theta)) +
                        pow(tan(thi / 2.0), 27) * (a326r * cos(27.0 * theta) - a326i * sin(27.0 * theta)) +
                        pow(tan(thi / 2.0), 32) * cos(32.0 * theta);
                ima = pow(tan(thi / 2.0), 2) * (a321r * sin(2.0 * theta) + a321i * cos(2.0 * theta)) +
                      pow(tan(thi / 2.0), 7) * (a322r * sin(7.0 * theta) + a322i * cos(7.0 * theta)) +
                      pow(tan(thi / 2.0), 12) * (a323r * sin(12.0 * theta) + a323i * cos(12.0 * theta)) +
                      pow(tan(thi / 2.0), 17) * (a324r * sin(17.0 * theta) + a324i * cos(17.0 * theta)) +
                      pow(tan(thi / 2.0), 22) * (a325r * sin(22.0 * theta) + a325i * cos(22.0 * theta)) +
                      pow(tan(thi / 2.0), 27) * (a326r * sin(27.0 * theta) + a326i * cos(27.0 * theta)) +
                      pow(tan(thi / 2.0), 32) * sin(32.0 * theta);
                realb = 1.0 + pow(tan(thi / 2.0), 5) * (a326r * cos(5.0 * theta) - a326i * sin(5.0 * theta)) +
                        pow(tan(thi / 2.0), 10) * (a325r * cos(10.0 * theta) - a325i * sin(10.0 * theta)) +
                        pow(tan(thi / 2.0), 15) * (a324r * cos(15.0 * theta) - a324i * sin(15.0 * theta)) +
                        pow(tan(thi / 2.0), 20) * (a323r * cos(20.0 * theta) - a323i * sin(20.0 * theta)) +
                        pow(tan(thi / 2.0), 25) * (a322r * cos(25.0 * theta) - a322i * sin(25.0 * theta)) +
                        pow(tan(thi / 2.0), 30) * (a321r * cos(30.0 * theta) - a321i * sin(30.0 * theta));
                imb = pow(tan(thi / 2.0), 5) * (a326r * sin(5.0 * theta) + a326i * cos(5.0 * theta)) +
                      pow(tan(thi / 2.0), 10) * (a325r * sin(10.0 * theta) + a325i * cos(10.0 * theta)) +
                      pow(tan(thi / 2.0), 15) * (a324r * sin(15.0 * theta) + a324i * cos(15.0 * theta)) +
                      pow(tan(thi / 2.0), 20) * (a323r * sin(20.0 * theta) + a323i * cos(20.0 * theta)) +
                      pow(tan(thi / 2.0), 25) * (a322r * sin(25.0 * theta) + a322i * cos(25.0 * theta)) +
                      pow(tan(thi / 2.0), 30) * (a321r * sin(30.0 * theta) + a321i * cos(30.0 * theta));
            }
                break;
            default:
                cout << "ERROR!! Requested rational map for charge " << B << "is unknown, please add the conversion function\n";
                return {0,0};
        }
        if(realb==0 & imb==0)
        {
            cout << "WARNING! - rational map realb/imb = 0, currently corrected to realb=0.00001 \n";
            cout << reala << " " << ima << "\n";
        }
        double real = (reala*realb + ima*imb)/(realb*realb + imb*imb);
        double imaginary = (-reala*imb + ima*realb)/(realb*realb + imb*imb);

        return {real,imaginary};

    }

}