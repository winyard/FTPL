//#include "FieldTheories.hpp"

#include "Skyrme.hpp"

int main() {

    Eigen::initParallel();
    cout << "creating\n";
    FTPL::SkyrmeModel my_model(45, 45, 45, false);
    cout << "created!\n";
    Eigen::Vector4d input(1, 0, 0, 0);
    my_model.f->fill(input);

    double space = 0.2;
    my_model.setSpacing({space, space, space});
    my_model.setParameters(8.0,0.5,0.0);

    cout << "do initial conditions\n";
    my_model.initialCondition(1, 0, 0, 0, 0);
    cout << "now clac eneergy\n";
    my_model.updateEnergy();
    cout << "Energy = " << my_model.getEnergy() << "\n";


    my_model.setTimeInterval(0.1*space*space*space);
    cout << "Running Gradient Flow:\n";
    Timer tmr;
    my_model.gradientFlow(100000, 10, true);
    cout << "1000 loops finished in " <<tmr.elapsed() << " sec\n";

   /* Eigen::initParallel();
    cout << "creating\n";
    FTPL::BabySkyrmeModel my_model(200, 200, true);
    cout << "created!\n";
    Eigen::Vector3d input(0, 0, 1);
    my_model.f->fill(input);

    double space = 0.2;
    my_model.setSpacing({space, space});
    my_model.setParameters(1.0,sqrt(0.1));

    my_model.setTimeInterval(0.5*space);
cout << "do initial conditions\n";
    my_model.initialCondition(1, 0, 0, 0);
    cout << "now clac eneergy\n";
    my_model.updateEnergy();
    cout << "Energy = " << my_model.getEnergy() << "\n";
    my_model.updateCharge();
    cout << "Charge = " << my_model.getCharge() << "\n";


    cout << "Time to test the RK4 method!\n";
    Timer tmr;
    my_model.RK4(1000,true,true,10);
    cout << "1000 loops finished in " <<tmr.elapsed() << " sec\n";
    my_model.setTimeInterval(0.5*space*space*space);
    my_model.gradientFlow(100, 10, true);
    cout << "ALL DONE!!\n";*/


    /*cout << "Running Gradient Flow:\n";


    Timer tmr;
    my_model.gradientFlow(100, 10, true);
    cout << "1000 loops finished in " <<tmr.elapsed() << " sec\n";*/
}

